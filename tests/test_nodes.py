import json
from unittest.mock import patch

import pytest

from app.agents import nodes
from app.agents.state import AgentState


@pytest.fixture
def llm():
    """Patches chat_completion where nodes.py looks it up."""
    with patch.object(nodes, "chat_completion") as mock:
        yield mock


def answer(text: str) -> str:
    return json.dumps({"answer": text})


# --- guardrail: deterministic, so no mocking is needed ------------------


def test_guardrail_rejects_empty_query():
    result = nodes.guardrail_node(AgentState(current_query="   "))

    assert result["status"] == "escalated"


def test_guardrail_rejects_oversized_query():
    oversized = "x" * (nodes.MAX_QUERY_CHARS + 1)

    result = nodes.guardrail_node(AgentState(current_query=oversized))

    assert result["status"] == "escalated"


def test_guardrail_passes_valid_query_through_unchanged():
    result = nodes.guardrail_node(AgentState(current_query="a real question"))

    # An empty dict means "I changed nothing" — the node's way of continuing.
    assert result == {}


# --- intent gate -------------------------------------------------------


def test_intent_gate_asks_the_model_when_no_cheap_rule_matches(llm, reply):
    llm.return_value = reply("  Billing  ")

    result = nodes.intent_gate_node(AgentState(current_query="where is my invoice"))

    assert result["intent"] == "billing"
    assert llm.call_count == 1


def test_intent_gate_tolerates_an_empty_completion(llm, reply):
    llm.return_value = reply(None)

    result = nodes.intent_gate_node(AgentState(current_query="?"))

    assert result["intent"] == "unknown"


# --- tool execution ----------------------------------------------------


def test_tool_exec_records_tool_output():
    state = AgentState(current_query="payload", intent="anything")

    result = nodes.tool_exec_node(state)

    assert len(result["tool_results"]) == 1
    assert result["tool_results"][0]["tool"] == "echo"


def test_tool_exec_captures_failure_instead_of_raising():
    state = AgentState(current_query="payload", intent="anything")

    with patch.dict(nodes.TOOL_REGISTRY, {"echo": _boom}):
        result = nodes.tool_exec_node(state)

    assert "error" in result["tool_results"][0]["output"]


def _boom(_payload):
    raise RuntimeError("backend down")


# --- respond -----------------------------------------------------------


def test_respond_returns_the_parsed_answer(llm, reply):
    llm.return_value = reply(answer("42"))

    result = nodes.respond_node(AgentState(current_query="q"))

    assert result == {"final_answer": "42", "status": "done"}
    assert llm.call_count == 1


def test_respond_repairs_one_malformed_reply(llm, reply):
    llm.side_effect = [reply("not json at all"), reply(answer("42"))]

    result = nodes.respond_node(AgentState(current_query="q"))

    assert result == {"final_answer": "42", "status": "done"}
    assert llm.call_count == 2


def test_respond_escalates_rather_than_repairing_twice(llm, reply):
    llm.side_effect = [reply("nope"), reply("still nope")]

    result = nodes.respond_node(AgentState(current_query="q"))

    assert result == {"status": "escalated"}
    assert llm.call_count == 2


def test_respond_rejects_json_missing_the_answer_key(llm, reply):
    llm.side_effect = [reply('{"wrong_key": 1}'), reply(answer("42"))]

    result = nodes.respond_node(AgentState(current_query="q"))

    assert result["final_answer"] == "42"
