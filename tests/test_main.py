import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.agents import nodes
from app.main import app

client = TestClient(app)


@pytest.fixture
def llm():
    with patch.object(nodes, "chat_completion") as mock:
        yield mock


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_graph_endpoint_returns_mermaid():
    body = client.get("/graph").json()

    assert body["mermaid"].startswith("---")


def test_run_returns_the_final_answer(llm, reply):
    llm.side_effect = [reply("support"), reply(json.dumps({"answer": "hello"}))]

    response = client.post("/agent/run", json={"query": "a real question"})

    assert response.status_code == 200
    assert response.json() == {"final_answer": "hello", "status": "done"}


def test_run_reports_a_guardrail_rejection_rather_than_failing(llm):
    response = client.post("/agent/run", json={"query": ""})

    assert response.status_code == 200
    assert response.json()["status"] == "escalated"
    llm.assert_not_called()


def test_run_rejects_a_malformed_request_body():
    assert client.post("/agent/run", json={}).status_code == 422
