"""Graph-level tests: wiring and routing, not node behaviour."""
import json
from unittest.mock import MagicMock, patch

import pytest

from app.agents import nodes
from app.agents.graph import graph, render_graph
from app.agents.state import AgentState

NODE_NAMES = {"guardrail", "confirmation_gate", "intent_gate", "context_gate", "tool_exec", "respond"}


@pytest.fixture
def llm():
    with patch.object(nodes, "chat_completion") as mock:
        yield mock


@pytest.fixture
def config():
    """A distinct thread per test — the compiled graph now requires a
    checkpointer-backed thread_id on every invoke()."""
    return {"configurable": {"thread_id": "test-thread"}}


def test_graph_compiles():
    assert graph is not None


def test_every_node_is_wired_in():
    assert NODE_NAMES <= set(graph.get_graph().nodes)


def test_render_graph_emits_mermaid_covering_the_whole_flow():
    mermaid = render_graph()

    assert mermaid.startswith("---")
    # Regression guard: a conditional edge without an explicit path map
    # compiles fine but renders a graph that stops at the branch.
    for name in NODE_NAMES:
        assert name in mermaid


def test_rejected_input_short_circuits_without_calling_the_model(llm, config):
    """The guardrail exists to spend nothing on input that cannot be served."""
    result = graph.invoke(AgentState(current_query=""), config=config)

    assert result["status"] == "escalated"
    llm.assert_not_called()


def test_accepted_input_runs_through_to_an_answer(llm, reply, config):
    # An out-of-vocabulary classification clamps to "unknown", which falls
    # to the echo placeholder in tool_exec — zero extra LLM calls, so this
    # stays a clean two-call trace (intent_gate, respond) end to end.
    llm.side_effect = [reply("unknown"), reply(json.dumps({"answer": "done"}))]

    result = graph.invoke(AgentState(current_query="a real question"), config=config)

    assert result["status"] == "done"
    assert result["final_answer"] == "done"
    assert result["intent"] == "unknown"
    assert result["tool_results"]


def test_result_validates_back_into_the_state_model(llm, reply, config):
    """main.py relies on this: invoke() returns a dict, not an AgentState."""
    llm.side_effect = [reply("unknown"), reply(json.dumps({"answer": "done"}))]

    result = graph.invoke(AgentState(current_query="a real question"), config=config)

    assert isinstance(result, dict)
    assert AgentState.model_validate(result).final_answer == "done"


def test_pending_action_persists_and_executes_across_two_invokes(config):
    """The concrete proof checkpointing works: a booking staged in turn 1 is
    executed in turn 2 from the same thread, with no re-extraction — the
    second invoke only sends this turn's query, not the whole prior state."""
    fake_action = {"tool": "echo", "args": {"payload": "staged"}}
    graph.invoke(
        {"current_query": "irrelevant", "pending_action": fake_action, "needs_confirmation": True},
        config=config,
    )

    with patch.object(nodes, "chat_completion") as llm:
        llm.return_value = MagicMock()
        llm.return_value.choices[0].message.content = json.dumps({"answer": "booked"})
        result = graph.invoke({"current_query": "yes"}, config=config)

    assert result["status"] == "done"
    assert result["pending_action"] is None
    assert result["tool_results"][-1]["tool"] == "echo"
