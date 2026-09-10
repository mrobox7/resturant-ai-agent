import itertools
import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app import backend_api
from app.agents import nodes
from app.main import app

client = TestClient(app)

# Each test gets its own thread_id (the graph's in-memory checkpointer is a
# module-level singleton shared across the whole test session) so that one
# test's pending confirmation or trace history can never leak into another.
_customer_ids = itertools.count(9001)


@pytest.fixture
def customer_id() -> int:
    return next(_customer_ids)


@pytest.fixture
def llm():
    with patch.object(nodes, "chat_completion") as mock:
        yield mock


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_graph_endpoint_returns_mermaid():
    body = client.get("/graph").json()

    assert body["mermaid"].startswith("---")


def test_run_returns_the_final_answer(llm, reply, customer_id):
    llm.side_effect = [reply("unknown"), reply(json.dumps({"answer": "hello"}))]

    response = client.post("/agent/run", json={"query": "a real question", "customer_id": customer_id})

    assert response.status_code == 200
    body = response.json()
    assert body["final_answer"] == "hello"
    assert body["status"] == "done"
    assert body["trace"]  # a non-empty chain-of-thought trail


def test_run_reports_a_guardrail_rejection_rather_than_failing(llm, customer_id):
    response = client.post("/agent/run", json={"query": "", "customer_id": customer_id})

    assert response.status_code == 200
    assert response.json()["status"] == "escalated"
    llm.assert_not_called()


def test_run_rejects_a_malformed_request_body():
    assert client.post("/agent/run", json={}).status_code == 422


def test_run_response_trace_covers_only_the_new_turn(llm, reply, customer_id):
    """The checkpointer accumulates a full-session trace; each response
    should carry only what happened on *this* turn, not the whole history —
    proven by length, since the per-node trace lines are identical
    boilerplate turn to turn and can't be told apart by content alone."""
    llm.side_effect = [reply("unknown"), reply(json.dumps({"answer": "first"}))]
    first = client.post("/agent/run", json={"query": "hello", "customer_id": customer_id}).json()

    llm.side_effect = [reply("unknown"), reply(json.dumps({"answer": "second"}))]
    second = client.post("/agent/run", json={"query": "hello again", "customer_id": customer_id}).json()

    assert len(first["trace"]) == len(second["trace"])
    # Same shape both turns, from a single-turn graph run — never doubling as
    # the checkpointed session grows, which is what a slicing bug would show.
    assert len(second["trace"]) == 4


# --- /session ------------------------------------------------------------


def test_session_requires_phone_or_email():
    assert client.post("/session", json={}).status_code == 422


def test_session_returns_existing_customer():
    with patch.object(backend_api, "lookup_customer", return_value={"id": 1, "name": "Priya Sharma"}):
        response = client.post("/session", json={"email": "priya.sharma@example.com"})

    assert response.status_code == 200
    assert response.json() == {"customer_id": 1, "name": "Priya Sharma"}


def test_session_creates_a_customer_when_none_matches():
    with (
        patch.object(backend_api, "lookup_customer", return_value=None),
        patch.object(backend_api, "create_customer", return_value={"id": 42, "name": "New Customer"}) as create,
    ):
        response = client.post(
            "/session", json={"email": "new@example.com", "name": "New Customer"}
        )

    assert response.status_code == 200
    assert response.json() == {"customer_id": 42, "name": "New Customer"}
    create.assert_called_once()


def test_session_404s_when_no_match_and_no_name_to_create_one():
    with patch.object(backend_api, "lookup_customer", return_value=None):
        response = client.post("/session", json={"email": "nobody@example.com"})

    assert response.status_code == 404


# --- identity boundary -----------------------------------------------------


def test_no_tool_schema_ever_exposes_customer_id_to_the_model():
    """Structural test: customer_id is code-attached, never something the
    model is asked to produce — see CLAUDE.md's identity-boundary principle."""
    from app.tools import TOOL_SCHEMAS

    for schema in TOOL_SCHEMAS:
        properties = schema["function"]["parameters"].get("properties", {})
        assert "customer_id" not in properties, schema["function"]["name"]
