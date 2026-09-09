"""Graph-level tests: wiring and routing, not node behaviour."""
import json
from unittest.mock import patch

import pytest

from app.agents import nodes
from app.agents.graph import graph, render_graph
from app.agents.state import AgentState

NODE_NAMES = {"guardrail", "intent_gate", "tool_exec", "respond"}


@pytest.fixture
def llm():
    with patch.object(nodes, "chat_completion") as mock:
        yield mock


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


def test_rejected_input_short_circuits_without_calling_the_model(llm):
    """The guardrail exists to spend nothing on input that cannot be served."""
    result = graph.invoke(AgentState(current_query=""))

    assert result["status"] == "escalated"
    llm.assert_not_called()


def test_accepted_input_runs_through_to_an_answer(llm, reply):
    llm.side_effect = [reply("support"), reply(json.dumps({"answer": "done"}))]

    result = graph.invoke(AgentState(current_query="a real question"))

    assert result["status"] == "done"
    assert result["final_answer"] == "done"
    assert result["intent"] == "support"
    assert result["tool_results"]


def test_result_validates_back_into_the_state_model(llm, reply):
    """main.py relies on this: invoke() returns a dict, not an AgentState."""
    llm.side_effect = [reply("support"), reply(json.dumps({"answer": "done"}))]

    result = graph.invoke(AgentState(current_query="a real question"))

    assert isinstance(result, dict)
    assert AgentState.model_validate(result).final_answer == "done"
