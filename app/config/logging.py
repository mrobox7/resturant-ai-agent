"""Configure observability with logfire."""
import os
import logfire
from fastapi import FastAPI


def configure_logging(app: FastAPI | None = None) -> None:
    """Initialize logfire if LOGFIRE_TOKEN is set. No-op if not."""
    if os.getenv("LOGFIRE_TOKEN"):
        logfire.configure()
        if app:
            logfire.instrument_fastapi(app)
