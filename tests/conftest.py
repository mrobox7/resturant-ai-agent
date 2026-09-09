from unittest.mock import MagicMock

import httpx
import pytest
from openai import APIConnectionError, APIStatusError
from openai import RateLimitError as ProviderRateLimitError


def llm_reply(content: str) -> MagicMock:
    """A stand-in for a chat completion response carrying `content`."""
    response = MagicMock()
    response.choices[0].message.content = content
    return response


@pytest.fixture
def reply():
    return llm_reply


@pytest.fixture
def request_obj() -> httpx.Request:
    return httpx.Request("POST", "http://provider.test/v1/chat/completions")


@pytest.fixture
def rate_limited(request_obj):
    """A provider 429 that asks for a specific wait."""

    def _make(retry_after: float) -> ProviderRateLimitError:
        response = httpx.Response(
            429, headers={"retry-after": str(retry_after)}, request=request_obj
        )
        return ProviderRateLimitError("rate limited", response=response, body=None)

    return _make


@pytest.fixture
def connection_error(request_obj) -> APIConnectionError:
    return APIConnectionError(request=request_obj)


@pytest.fixture
def status_error(request_obj):
    def _make(code: int) -> APIStatusError:
        return APIStatusError(
            f"http {code}", response=httpx.Response(code, request=request_obj), body=None
        )

    return _make
