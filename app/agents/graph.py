"""Wires the nodes into a StateGraph. Topology is problem-agnostic; edit nodes.py."""
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END

from app.agents.state import AgentState
from app.agents.nodes import (
    guardrail_node,
    confirmation_gate_node,
    intent_gate_node,
    context_gate_node,
    tool_exec_node,
    respond_node,
)


def _route_after_guardrail(state: AgentState) -> str:
    """Skip the rest of the graph if the guardrail already rejected the input."""
    return "escalated" if state.status == "escalated" else "continue"


def _route_after_confirmation(state: AgentState) -> str:
    """confirmation_gate_node sets this explicitly on every path — see its
    docstring for why nothing here ever falls back to a stale value."""
    return state.route_after_confirmation or "continue"


def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("guardrail", guardrail_node)
    workflow.add_node("confirmation_gate", confirmation_gate_node)
    workflow.add_node("intent_gate", intent_gate_node)
    workflow.add_node("context_gate", context_gate_node)
    workflow.add_node("tool_exec", tool_exec_node)
    workflow.add_node("respond", respond_node)

    workflow.set_entry_point("guardrail")
    # The path map is explicit so the branch targets are known statically —
    # without it LangGraph cannot resolve the destinations and draw_mermaid()
    # renders an incomplete graph.
    workflow.add_conditional_edges(
        "guardrail",
        _route_after_guardrail,
        {"continue": "confirmation_gate", "escalated": END},
    )
    workflow.add_conditional_edges(
        "confirmation_gate",
        _route_after_confirmation,
        {"continue": "intent_gate", "respond": "respond", "end": END},
    )
    workflow.add_edge("intent_gate", "context_gate")
    workflow.add_edge("context_gate", "tool_exec")
    workflow.add_edge("tool_exec", "respond")
    workflow.add_edge("respond", END)

    return workflow


# In-process only — state does not survive a restart. Sufficient for
# "maintain state through the workflow" within a session; swapping in
# langgraph's sqlite checkpointer is a one-line change if surviving a
# restart is ever required (see CLAUDE.md: SQLite is the stack's lightweight
# persistence default).
graph = build_graph().compile(checkpointer=MemorySaver())


def render_graph() -> str:
    """Mermaid source for the compiled graph, for the /graph endpoint."""
    return graph.get_graph().draw_mermaid()
