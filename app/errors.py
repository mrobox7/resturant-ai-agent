class TransientLLMError(Exception):
    """Transient failure: timeout, 5xx. Retry with exponential backoff."""

    pass


class RateLimitError(Exception):
    """HTTP 429. Retry after exact sleep duration, not exponential backoff."""

    def __init__(self, retry_after: float, message: str = "Rate limited"):
        super().__init__(message)
        self.retry_after = retry_after


class MalformedResponseError(Exception):
    """Bad JSON/invalid structured response from LLM. Retry with repair prompt."""

    pass


class InvalidInputError(Exception):
    """Deterministic failure given the input (e.g. over length limit).
    Never retry — must be caught by validation BEFORE any LLM call."""

    pass
