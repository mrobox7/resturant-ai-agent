from pydantic import BaseModel
from typing import List


class AgentRequest(BaseModel):
    request: str


class TaskResponse(BaseModel):
    """The structured, validated shape every agent run must return.
    Swap this out for whatever schema the interview task needs."""
    answer: str
    tools_used: List[str] = []
