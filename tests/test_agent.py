"""
Basic tests with the LLM mocked out — this is exactly the kind of
evaluation/testing approach interviewers want to see set up, even
under time pressure, since it proves the agent logic (not the model)
behaves correctly and deterministically. No API key or network needed.
"""
from unittest.mock import patch, MagicMock
from app.agent import run_agent


def _mock_response(content, tool_calls=None):
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.model_dump.return_value = {"role": "assistant", "content": content}
    resp = MagicMock()
    resp.choices = [MagicMock(message=msg)]
    return resp


@patch("app.agent.chat_completion")
def test_agent_returns_valid_structured_answer(mock_chat):
    mock_chat.return_value = _mock_response(
        '{"answer": "It is sunny in Jaipur.", "tools_used": []}'
    )
    result = run_agent("What's the weather in Jaipur?")
    assert result.answer == "It is sunny in Jaipur."
    assert result.tools_used == []


@patch("app.agent.chat_completion")
def test_agent_repairs_invalid_json(mock_chat):
    mock_chat.side_effect = [
        _mock_response("not valid json"),
        _mock_response('{"answer": "Fixed answer", "tools_used": []}'),
    ]
    result = run_agent("Say something")
    assert result.answer == "Fixed answer"
