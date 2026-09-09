"""The state passed between nodes.

Add fields the problem needs; nodes return partial dicts of these keys.
"""
import operator
from typing import Annotated, Literal

from pydantic import BaseModel, Field

Status = Literal["pending", "needs_clarification", "escalated", "done"]


class AgentState(BaseModel):
    # operator.add makes this channel append-only: two nodes writing messages
    # concatenate instead of the later one clobbering the earlier. Every other
    # field is last-write-wins.
    messages: Annotated[list[dict], operator.add] = Field(default_factory=list)

    current_query: str = Field(default="")
    intent: str | None = Field(default=None)
    tool_results: list[dict] = Field(default_factory=list)
    needs_confirmation: bool = Field(default=False)
    status: Status = Field(default="pending")
    final_answer: str = Field(default="")
