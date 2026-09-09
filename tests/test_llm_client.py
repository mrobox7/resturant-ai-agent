import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app import llm_client
from app.errors import RateLimitError, TransientLLMError

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def provider():
    with patch.object(llm_client, "_get_client") as get_client:
        client = MagicMock()
        get_client.return_value = client
        yield client.chat.completions.create


@pytest.fixture
def no_sleep():
    with patch.object(llm_client.time, "sleep") as sleep:
        yield sleep


def test_import_needs_no_api_key():
    result = subprocess.run(
        [sys.executable, "-c", "import app.llm_client; print(app.llm_client._client)"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
        env={"PATH": "/usr/bin:/bin"},  # no *_API_KEY of any kind
    )
    assert result.returncode == 0, result.stderr
    # ...and it must not have built a client just by being imported.
    assert result.stdout.strip() == "None"


def test_transient_failure_is_retried_then_succeeds(
    provider, no_sleep, connection_error, reply
):
    provider.side_effect = [connection_error, reply("recovered")]

    response = llm_client.chat_completion([{"role": "user", "content": "hi"}])

    assert response.choices[0].message.content == "recovered"
    assert no_sleep.call_count == 1


def test_transient_failure_backs_off_exponentially_then_raises(
    provider, no_sleep, connection_error
):
    provider.side_effect = connection_error

    with pytest.raises(TransientLLMError):
        llm_client.chat_completion([{"role": "user", "content": "hi"}])

    # Three attempts: two waits between them, doubling, and no wait after the
    # final failure.
    assert [call.args[0] for call in no_sleep.call_args_list] == [2, 4]
    assert provider.call_count == 3


def test_rate_limit_sleeps_exactly_what_the_provider_asked(
    provider, no_sleep, rate_limited, reply
):
    provider.side_effect = [rate_limited(7.5), reply("ok")]

    llm_client.chat_completion([{"role": "user", "content": "hi"}])

    # 7.5 from the Retry-After header, not 2**attempt from the backoff formula.
    no_sleep.assert_called_once_with(7.5)


def test_rate_limit_surfaces_retry_after_when_retries_run_out(
    provider, no_sleep, rate_limited
):
    provider.side_effect = rate_limited(3.0)

    with pytest.raises(RateLimitError) as excinfo:
        llm_client.chat_completion([{"role": "user", "content": "hi"}])

    assert excinfo.value.retry_after == 3.0


def test_client_error_is_not_retried(provider, no_sleep, status_error):
    """A 400 is a bad request; retrying only reproduces it."""
    provider.side_effect = status_error(400)

    with pytest.raises(Exception) as excinfo:
        llm_client.chat_completion([{"role": "user", "content": "hi"}])

    assert not isinstance(excinfo.value, TransientLLMError)
    assert provider.call_count == 1
    no_sleep.assert_not_called()


def test_server_error_is_retried(provider, no_sleep, status_error):
    provider.side_effect = status_error(503)

    with pytest.raises(TransientLLMError):
        llm_client.chat_completion([{"role": "user", "content": "hi"}])

    assert provider.call_count == 3


def test_tools_are_only_sent_when_present(provider, reply):
    provider.return_value = reply("ok")

    llm_client.chat_completion([{"role": "user", "content": "hi"}])
    assert "tools" not in provider.call_args.kwargs

    schema = [{"type": "function", "function": {"name": "echo"}}]
    llm_client.chat_completion([{"role": "user", "content": "hi"}], tools=schema)
    assert provider.call_args.kwargs["tools"] == schema
