"""Thin wrapper around Groq's chat completions endpoint.

Groq speaks the OpenAI chat-completions protocol, which is why the `openai`
package is the client here — it is a protocol client, not a commitment to
OpenAI. Should Groq be down or rate-limiting mid-task, any other provider
speaking the same protocol is reachable by changing llm_base_url,
llm_model and llm_api_key in .env, with no change here.
"""
import logging
import time

from openai import OpenAI
from openai import APIConnectionError, APIStatusError, APITimeoutError
from openai import RateLimitError as ProviderRateLimitError

from app.config.settings import settings
from app.errors import RateLimitError, TransientLLMError

logger = logging.getLogger(__name__)

_client: OpenAI | None = None

# Used when the provider rate-limits us without saying for how long.
DEFAULT_RETRY_AFTER_SECONDS = 5.0


def _get_client() -> OpenAI:
    """Builds the client on first use, not at import time.

    Constructing it at import would make this module — and everything that
    imports it — unimportable without credentials, including tests that mock
    chat_completion out entirely. Deferring it means a key is only needed
    when a call actually happens.
    """
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=settings.llm_api_key.get_secret_value(),
            base_url=settings.llm_base_url,
        )
    return _client


def _retry_after_from(exc: Exception) -> float:
    """Reads the provider's Retry-After header, falling back to a fixed wait."""
    response = getattr(exc, "response", None)
    header = getattr(response, "headers", {}) or {}
    try:
        return float(header.get("retry-after", DEFAULT_RETRY_AFTER_SECONDS))
    except (TypeError, ValueError):
        return DEFAULT_RETRY_AFTER_SECONDS


def chat_completion(messages: list, tools: list | None = None, **kwargs):
    """Calls the model, retrying the failures that are worth retrying.

    Transient failures (connection dropped, timeout, 5xx) back off
    exponentially. Rate limits sleep for exactly as long as the provider
    asked — backoff maths on a 429 either hammers the endpoint too early or
    idles long past the window. Everything else is a bug in the request and
    is raised immediately rather than retried.
    """
    for attempt in range(1, settings.max_retries + 1):
        last_attempt = attempt == settings.max_retries
        try:
            params = {"model": settings.llm_model, "messages": messages, **kwargs}
            if tools:
                params["tools"] = tools
            return _get_client().chat.completions.create(**params)

        except ProviderRateLimitError as exc:
            retry_after = _retry_after_from(exc)
            if last_attempt:
                raise RateLimitError(retry_after, "rate limited; retries exhausted") from exc
            logger.warning("Rate limited — sleeping %.1fs as instructed", retry_after)
            time.sleep(retry_after)

        except (APIConnectionError, APITimeoutError) as exc:
            if last_attempt:
                raise TransientLLMError(f"connection failed after {attempt} attempts") from exc
            _backoff(attempt, exc)

        except APIStatusError as exc:
            # 4xx other than 429 means the request itself is wrong; retrying
            # it just reproduces the same error.
            if exc.status_code < 500:
                raise
            if last_attempt:
                raise TransientLLMError(f"server error after {attempt} attempts") from exc
            _backoff(attempt, exc)

    raise TransientLLMError(f"exhausted {settings.max_retries} attempts")


def _backoff(attempt: int, exc: Exception) -> None:
    wait = 2**attempt
    logger.warning(
        "LLM call failed (attempt %d/%d): %s — retrying in %ds",
        attempt, settings.max_retries, exc, wait,
    )
    time.sleep(wait)
