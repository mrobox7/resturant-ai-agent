"""HTTP layer. Problem-agnostic — the request/response shapes are the only
things worth touching here, and only if the problem needs different fields."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app import backend_api
from app.agents.graph import graph, render_graph
from app.agents.state import AgentState
from app.config import configure_logging
from app.errors import BackendValidationError

app = FastAPI(title="Agent")
configure_logging(app)


class RunRequest(BaseModel):
    query: str
    customer_id: int  # attached by the client after /session — never model-derived


class RunResponse(BaseModel):
    final_answer: str
    status: str
    trace: list[str] = []


class SessionRequest(BaseModel):
    phone: str | None = None
    email: str | None = None
    name: str | None = None  # only required if no existing customer matches


class SessionResponse(BaseModel):
    customer_id: int
    name: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/graph")
def get_graph() -> dict:
    """Mermaid source for the compiled graph, handy for showing the flow."""
    return {"mermaid": render_graph()}


@app.post("/session", response_model=SessionResponse)
def start_session(payload: SessionRequest) -> SessionResponse:
    """Resolves a customer by phone/email, creating one if none matches.

    This is the one and only place customer identity is established — a
    plain deterministic backend call, no LLM, no graph involved. Every
    /agent/run call afterward carries the resulting customer_id; the model
    is never asked to produce or verify one.
    """
    if not payload.phone and not payload.email:
        raise HTTPException(status_code=422, detail="phone or email is required")

    try:
        customer = backend_api.lookup_customer(phone=payload.phone, email=payload.email)
        if customer is None:
            if not payload.name:
                raise HTTPException(
                    status_code=404,
                    detail="no matching customer — provide a name to create one",
                )
            customer = backend_api.create_customer(
                {"name": payload.name, "phone": payload.phone, "email": payload.email, "preferences": {}}
            )
    except BackendValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.detail) from exc

    return SessionResponse(customer_id=customer["id"], name=customer["name"])


@app.post("/agent/run", response_model=RunResponse)
def run(payload: RunRequest) -> RunResponse:
    # thread_id = customer_id: the graph's checkpointer keeps this customer's
    # state (pending confirmations, loaded preferences, trace history)
    # across calls, so only this turn's delta needs to be sent — not a fresh
    # AgentState every time.
    config = {"configurable": {"thread_id": str(payload.customer_id)}}
    prior_trace_len = len(graph.get_state(config).values.get("trace", []))

    try:
        # invoke() returns the state as a plain dict; validating it back into
        # AgentState keeps the response typed and fails loudly if a node ever
        # writes a field the schema does not allow.
        result = AgentState.model_validate(
            graph.invoke(
                {"current_query": payload.query, "customer_id": payload.customer_id},
                config=config,
            )
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return RunResponse(
        final_answer=result.final_answer,
        status=result.status,
        # trace is append-only across the whole checkpointed session; only
        # this turn's new lines belong in one response, not the full history.
        trace=result.trace[prior_trace_len:],
    )
