"""Configure observability with logfire."""
import logging

import logfire
from fastapi import FastAPI


def configure_logging(app: FastAPI | None = None) -> None:
    """Always trace to the console; also export to Logfire Cloud if LOGFIRE_TOKEN is set.

    Console output means logs are inspectable locally with no Logfire account
    at all — useful for an interview/dev box. `send_to_logfire="if-token-present"`
    is what makes the cloud export opt-in on top of that, not instead of it.
    """
    logfire.configure(
        send_to_logfire="if-token-present",
        console=logfire.ConsoleOptions(min_log_level="info"),
    )
    # Bridges stdlib `logging` (already used throughout app/) into logfire, so
    # every existing logger.info/warning/error call becomes a logfire log too
    # without rewriting call sites one by one.
    logging.getLogger().addHandler(logfire.LogfireLoggingHandler())
    if app:
        logfire.instrument_fastapi(app)
