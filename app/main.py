"""FastAPI entrypoint exposing the agent as an HTTP endpoint."""
import logging
from fastapi import FastAPI, HTTPException

from app.agent import run_agent
from app.models import AgentRequest, TaskResponse

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="AI Agent Starter Kit")


@app.post("/agent/run", response_model=TaskResponse)
def run(payload: AgentRequest):
    try:
        return run_agent(payload.request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
def health():
    return {"status": "ok"}
