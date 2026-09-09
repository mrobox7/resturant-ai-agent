"""HTTP layer. Problem-agnostic — the request/response shapes are the only
things worth touching here, and only if the problem needs different fields."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.agents.graph import graph, render_graph
from app.agents.state import AgentState
from app.config import configure_logging

app = FastAPI(title="Agent")
configure_logging(app)


class RunRequest(BaseModel):
    query: str


class RunResponse(BaseModel):
    final_answer: str
    status: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/graph")
def get_graph() -> dict:
    """Mermaid source for the compiled graph, handy for showing the flow."""
    return {"mermaid": render_graph()}


@app.post("/agent/run", response_model=RunResponse)
def run(payload: RunRequest) -> RunResponse:
    try:
        # invoke() returns the state as a plain dict; validating it back into
        # AgentState keeps the response typed and fails loudly if a node ever
        # writes a field the schema does not allow.
        result = AgentState.model_validate(graph.invoke(AgentState(current_query=payload.query)))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return RunResponse(final_answer=result.final_answer, status=result.status)
