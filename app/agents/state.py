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

    # Attached by code from the request, never asked of the model — the
    # identity boundary. See app/main.py: RunRequest.customer_id. Matches
    # the backend's own type (an integer id), not a string.
    customer_id: int = Field(default=0)

    # A booking/order the agent has assembled and is waiting on the user to
    # confirm before it calls the mutating backend endpoint. Cleared once
    # confirmation_gate_node acts on it (accepted or declined).
    pending_action: dict | None = Field(default=None)

    # Set unconditionally by confirmation_gate_node on every single
    # invocation ("continue" | "respond" | "end") — routing-only, never
    # trusted from a prior turn's checkpoint, so the conditional edge after
    # that node never reads a stale value.
    route_after_confirmation: str | None = Field(default=None)

    # Loaded once per thread (checkpointed session), not re-fetched every
    # turn: the customer's preferences and a few recent episode summaries,
    # kept here as structured data the code can look up directly rather than
    # text stuffed into `messages` on every turn — that's what would bloat
    # the context window over a long session.
    customer_preferences: dict = Field(default_factory=dict)
    recent_episodes: list[str] = Field(default_factory=list)
    preferences_loaded: bool = Field(default=False)

    # One human-readable line per node, appended (never replaced) as the
    # graph runs — the UI's "chain of thought" view over what actually
    # happened, not the model's hidden reasoning.
    trace: Annotated[list[str], operator.add] = Field(default_factory=list)
