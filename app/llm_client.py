"""
Thin wrapper around an OpenAI-compatible chat completions endpoint.
Works unmodified with OpenAI, Groq, and OpenRouter — just change
llm_base_url / llm_model / llm_api_key in .env.
"""
import time
import logging
from openai import OpenAI
from app.config import settings

logger = logging.getLogger(__name__)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """Construct the OpenAI client lazily, on first real use.

    Building it at import time means importing this module (e.g. to run
    tests that mock chat_completion entirely) fails without a real API key
    present. Deferring construction keeps the module importable/testable
    with zero credentials, and only requires a key when a call actually
    happens.
    """
    global _client
    if _client is None:
        _client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    return _client


def chat_completion(messages: list, tools: list | None = None, **kwargs):
    """Call the LLM with simple exponential-backoff retries on failure."""
    last_err = None
    for attempt in range(1, settings.max_retries + 1):
        try:
            return _get_client().chat.completions.create(
                model=settings.llm_model,
                messages=messages,
                tools=tools,
                **kwargs,
            )
        except Exception as exc:  # broad on purpose: network/rate-limit/provider errors
            last_err = exc
            wait = 2 ** attempt
            logger.warning(
                "LLM call failed (attempt %d/%d): %s — retrying in %ds",
                attempt, settings.max_retries, exc, wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"LLM call failed after {settings.max_retries} attempts") from last_err
