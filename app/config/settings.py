from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed config, read once from the environment or .env at startup.

    Validated and coerced in one place at startup rather than scattered
    os.getenv() calls, so a malformed value fails immediately with a real
    error message instead of surfacing as a TypeError mid-request.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Anything else in the environment is somebody else's config, not a
        # mistake. The default is to forbid it, which turns one stray variable
        # into a crash at import.
        extra="ignore",
    )

    # SecretStr so the key cannot be printed by accident. Pydantic renders it
    # as '**********' in reprs, logs and — the reason it is here — validation
    # error messages, which quote the offending value back at you.
    llm_api_key: SecretStr = SecretStr("")
    llm_base_url: str = "https://api.groq.com/openai/v1"
    # Providers retire models without warning; if a call 404s with
    # model_not_found, ask the endpoint what it serves:
    #   uv run python -c "from app.llm_client import _get_client; \
    #     print([m.id for m in _get_client().models.list().data])"
    llm_model: str = "openai/gpt-oss-20b"

    max_retries: int = 3


settings = Settings()
