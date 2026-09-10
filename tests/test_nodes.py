import json
from unittest.mock import MagicMock, patch

import pytest

from app.agents import nodes
from app.agents.state import AgentState


def tool_call_reply(name: str, arguments: dict) -> MagicMock:
    """A stand-in for a chat completion response carrying one tool call."""
    response = MagicMock()
    call = MagicMock()
    call.function.name = name
    call.function.arguments = json.dumps(arguments)
    response.choices[0].message.tool_calls = [call]
    return response


def no_tool_call_reply() -> MagicMock:
    response = MagicMock()
    response.choices[0].message.tool_calls = None
    return response


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

    # No status/final_answer means "I changed nothing that stops the graph" —
    # only the trace records that the check ran.
    assert "status" not in result
    assert "final_answer" not in result
    assert result["trace"] == ["Guardrail: query accepted"]


# --- intent gate -------------------------------------------------------


def test_intent_gate_asks_the_model_when_no_cheap_rule_matches(llm, reply):
    llm.return_value = reply("  Availability  ")

    result = nodes.intent_gate_node(AgentState(current_query="is there room tomorrow"))

    assert result["intent"] == "availability"
    assert llm.call_count == 1


def test_intent_gate_tolerates_an_empty_completion(llm, reply):
    llm.return_value = reply(None)

    result = nodes.intent_gate_node(AgentState(current_query="?"))

    assert result["intent"] == "unknown"


def test_intent_gate_clamps_an_out_of_vocabulary_reply_to_unknown(llm, reply):
    """A model reply outside the known intent set must never be trusted
    through to tool_exec's dispatch — that's a silent misroute, not a
    graceful fallback."""
    llm.return_value = reply("billing_inquiry")

    result = nodes.intent_gate_node(AgentState(current_query="???"))

    assert result["intent"] == "unknown"


# --- confirmation gate: deterministic, no LLM mocking needed -----------


def _pending(tool: str = "echo") -> dict:
    return {"tool": tool, "args": {"payload": "staged"}}


def test_confirmation_gate_passes_through_when_nothing_pending():
    result = nodes.confirmation_gate_node(AgentState(current_query="hi"))

    assert result == {"route_after_confirmation": "continue"}


def test_confirmation_gate_executes_on_affirmative_reply():
    state = AgentState(current_query="yes please", pending_action=_pending(), needs_confirmation=True)

    with patch.dict(nodes.TOOL_REGISTRY, {"echo": (tool := MagicMock(return_value={"ok": True}))}):
        result = nodes.confirmation_gate_node(state)

    tool.assert_called_once_with(payload="staged")
    assert result["route_after_confirmation"] == "respond"
    assert result["pending_action"] is None
    assert result["needs_confirmation"] is False


def test_confirmation_gate_declines_without_calling_the_tool():
    state = AgentState(current_query="no, cancel that", pending_action=_pending(), needs_confirmation=True)

    with patch.dict(nodes.TOOL_REGISTRY, {"echo": (tool := MagicMock())}):
        result = nodes.confirmation_gate_node(state)

    tool.assert_not_called()
    assert result["route_after_confirmation"] == "end"
    assert result["status"] == "done"
    assert result["pending_action"] is None


def test_confirmation_gate_asks_again_on_an_inconclusive_reply():
    state = AgentState(current_query="what time is it", pending_action=_pending(), needs_confirmation=True)

    with patch.dict(nodes.TOOL_REGISTRY, {"echo": (tool := MagicMock())}):
        result = nodes.confirmation_gate_node(state)

    tool.assert_not_called()
    assert result["route_after_confirmation"] == "end"
    assert result["status"] == "needs_clarification"
    assert "pending_action" not in result  # left untouched, still there next turn


def test_confirmation_gate_does_not_mistake_a_substring_for_negation():
    """Regression: naive `"no" in query` matches inside "November"/"know" —
    tokenized matching must not."""
    state = AgentState(
        current_query="November 10th at 7 works for me",
        pending_action=_pending(),
        needs_confirmation=True,
    )

    with patch.dict(nodes.TOOL_REGISTRY, {"echo": (tool := MagicMock())}):
        result = nodes.confirmation_gate_node(state)

    tool.assert_not_called()
    assert result["route_after_confirmation"] == "end"
    assert result["status"] == "needs_clarification"  # inconclusive, not declined


def test_confirmation_gate_appends_an_episode_after_a_successful_booking():
    state = AgentState(
        current_query="yes",
        pending_action=_pending("create_reservation"),
        needs_confirmation=True,
        customer_id=1,
    )
    fake_reservation = {
        "table_id": 3, "slot_datetime": "2026-09-10T19:00:00", "party_size": 4,
    }
    with (
        patch.dict(nodes.TOOL_REGISTRY, {"create_reservation": MagicMock(return_value=fake_reservation)}),
        patch.object(nodes.memory, "append_episode") as append_episode,
    ):
        nodes.confirmation_gate_node(state)

    append_episode.assert_called_once()
    assert append_episode.call_args.args[0] == 1


# --- context gate --------------------------------------------------------


def test_context_gate_is_a_noop_for_non_booking_intents():
    with (
        patch.dict(nodes.TOOL_REGISTRY, {"get_user_preferences": (prefs := MagicMock())}),
        patch.object(nodes.memory, "read_episodes") as read_episodes,
    ):
        result = nodes.context_gate_node(AgentState(current_query="what's on the menu", intent="menu"))

    prefs.assert_not_called()
    read_episodes.assert_not_called()
    assert result == {}


def test_context_gate_is_a_noop_once_already_loaded():
    state = AgentState(current_query="book a table", intent="booking", preferences_loaded=True)

    with patch.dict(nodes.TOOL_REGISTRY, {"get_user_preferences": (prefs := MagicMock())}):
        result = nodes.context_gate_node(state)

    prefs.assert_not_called()
    assert result == {}


def test_context_gate_loads_preferences_and_episodes_for_booking_intent():
    state = AgentState(current_query="book a table for 4", intent="booking", customer_id=1)

    with (
        patch.dict(
            nodes.TOOL_REGISTRY,
            {"get_user_preferences": MagicMock(return_value={"name": "Priya", "preferences": {"seating": "outdoor"}})},
        ),
        patch.object(nodes.memory, "read_episodes", return_value=["booked before"]),
    ):
        result = nodes.context_gate_node(state)

    assert result["customer_preferences"] == {"seating": "outdoor"}
    assert result["preferences_loaded"] is True
    assert result["recent_episodes"] == ["booked before"]


def test_context_gate_leaves_preferences_unloaded_on_failure_so_it_retries():
    state = AgentState(current_query="book a table", intent="booking", customer_id=1)

    with patch.dict(nodes.TOOL_REGISTRY, {"get_user_preferences": MagicMock(side_effect=RuntimeError("backend down"))}):
        result = nodes.context_gate_node(state)

    assert "preferences_loaded" not in result  # stays False, retried next relevant turn


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


def _boom(**_kwargs):
    raise RuntimeError("backend down")


def test_tool_exec_calls_get_menu_with_model_extracted_filters(llm):
    llm.return_value = tool_call_reply("get_menu", {"category": "starter"})
    state = AgentState(current_query="what starters do you have?", intent="menu")

    with patch.dict(nodes.TOOL_REGISTRY, {"get_menu": (get_menu := MagicMock(return_value=[]))}):
        result = nodes.tool_exec_node(state)

    get_menu.assert_called_once_with(category="starter")
    assert result["tool_results"][0]["tool"] == "get_menu"


def test_tool_exec_calls_get_menu_with_no_filters_when_model_declines(llm):
    llm.return_value = no_tool_call_reply()
    state = AgentState(current_query="what's on the menu?", intent="menu")

    with patch.dict(nodes.TOOL_REGISTRY, {"get_menu": (get_menu := MagicMock(return_value=[]))}):
        nodes.tool_exec_node(state)

    get_menu.assert_called_once_with()


def test_tool_exec_uses_get_menu_only_for_menu_intent():
    state = AgentState(current_query="payload", intent="other")

    result = nodes.tool_exec_node(state)

    assert result["tool_results"][0]["tool"] == "echo"


# --- booking preparation: table_id is resolved, never asked of the user --


ONE_AVAILABLE_TABLE = {
    "slot_datetime": "2026-09-10T19:00:00",
    "party_size": 4,
    "available_tables": [{"table_id": 7, "table_number": 7, "capacity": 4, "location": "outdoor"}],
}


def test_prepare_booking_resolves_table_id_via_availability(llm):
    llm.return_value = tool_call_reply(
        "create_reservation", {"slot_datetime": "2026-09-10T19:00:00", "party_size": 4}
    )
    state = AgentState(current_query="book a table for 4 at 7pm", intent="booking", customer_id=1)

    with patch.dict(nodes.TOOL_REGISTRY, {"get_availability": MagicMock(return_value=ONE_AVAILABLE_TABLE)}):
        result = nodes.tool_exec_node(state)

    assert result["pending_action"]["args"]["table_id"] == 7
    assert result["pending_action"]["args"]["customer_id"] == 1
    assert result["needs_confirmation"] is True


def test_prepare_booking_asks_for_missing_fields_without_calling_availability(llm):
    llm.return_value = tool_call_reply("create_reservation", {"party_size": 4})  # no slot_datetime
    state = AgentState(current_query="book a table for 4", intent="booking", customer_id=1)

    with patch.dict(nodes.TOOL_REGISTRY, {"get_availability": (avail := MagicMock())}):
        result = nodes.tool_exec_node(state)

    avail.assert_not_called()
    assert result["tool_results"][0]["output"]["status"] == "missing_fields"
    assert "pending_action" not in result


def test_prepare_booking_uses_seating_preference_to_disambiguate(llm):
    llm.return_value = tool_call_reply(
        "create_reservation", {"slot_datetime": "2026-09-10T19:00:00", "party_size": 2}
    )
    state = AgentState(
        current_query="book a table for 2 at 7pm",
        intent="booking",
        customer_id=1,
        customer_preferences={"seating": "outdoor"},
    )
    two_tables = {
        **ONE_AVAILABLE_TABLE,
        "available_tables": [
            {"table_id": 1, "table_number": 1, "capacity": 2, "location": "indoor"},
            {"table_id": 6, "table_number": 6, "capacity": 2, "location": "outdoor"},
        ],
    }

    with patch.dict(nodes.TOOL_REGISTRY, {"get_availability": MagicMock(return_value=two_tables)}):
        result = nodes.tool_exec_node(state)

    assert result["pending_action"]["args"]["table_id"] == 6  # the outdoor one


def test_prepare_booking_auto_picks_the_lowest_numbered_table_when_still_tied(llm):
    """No preference to break the tie and no location stated: any of the
    remaining tables is as good as another, so booking proceeds rather than
    blocking on a distinction the customer never asked about."""
    llm.return_value = tool_call_reply(
        "create_reservation", {"slot_datetime": "2026-09-10T19:00:00", "party_size": 2}
    )
    state = AgentState(current_query="book a table for 2 at 7pm", intent="booking", customer_id=1)
    two_tables = {
        **ONE_AVAILABLE_TABLE,
        "available_tables": [
            {"table_id": 6, "table_number": 6, "capacity": 2, "location": "outdoor"},
            {"table_id": 1, "table_number": 1, "capacity": 2, "location": "indoor"},
        ],
    }

    with patch.dict(nodes.TOOL_REGISTRY, {"get_availability": MagicMock(return_value=two_tables)}):
        result = nodes.tool_exec_node(state)

    assert result["pending_action"]["args"]["table_id"] == 1  # table_number 1, the lowest
    assert result["needs_confirmation"] is True


def test_prepare_booking_reports_no_availability(llm):
    llm.return_value = tool_call_reply(
        "create_reservation", {"slot_datetime": "2026-09-10T19:00:00", "party_size": 2}
    )
    state = AgentState(current_query="book a table for 2 at 7pm", intent="booking", customer_id=1)
    no_tables = {**ONE_AVAILABLE_TABLE, "available_tables": []}

    with patch.dict(nodes.TOOL_REGISTRY, {"get_availability": MagicMock(return_value=no_tables)}):
        result = nodes.tool_exec_node(state)

    assert result["tool_results"][0]["output"]["status"] == "no_table_resolved"
    assert "pending_action" not in result


# --- respond -----------------------------------------------------------


def test_respond_returns_the_parsed_answer(llm, reply):
    llm.return_value = reply(answer("42"))

    result = nodes.respond_node(AgentState(current_query="q"))

    assert result["final_answer"] == "42"
    assert result["status"] == "done"
    assert llm.call_count == 1


def test_respond_repairs_one_malformed_reply(llm, reply):
    llm.side_effect = [reply("not json at all"), reply(answer("42"))]

    result = nodes.respond_node(AgentState(current_query="q"))

    assert result["final_answer"] == "42"
    assert result["status"] == "done"
    assert llm.call_count == 2


def test_respond_escalates_rather_than_repairing_twice(llm, reply):
    llm.side_effect = [reply("nope"), reply("still nope")]

    result = nodes.respond_node(AgentState(current_query="q"))

    assert result["status"] == "escalated"
    assert "final_answer" not in result
    assert llm.call_count == 2


def test_respond_rejects_json_missing_the_answer_key(llm, reply):
    llm.side_effect = [reply('{"wrong_key": 1}'), reply(answer("42"))]

    result = nodes.respond_node(AgentState(current_query="q"))

    assert result["final_answer"] == "42"
