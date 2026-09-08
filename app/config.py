from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, validated config — reads .env once at startup instead of
    scattering os.getenv() calls (with manual type coercion and silent
    string defaults) across the codebase."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Works unmodified with any OpenAI-compatible provider: OpenAI, Groq,
    # OpenRouter — just change these three in .env.
    llm_api_key: str = ""
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "llama-3.3-70b-versatile"

    max_tool_iterations: int = 5
    max_retries: int = 3


settings = Settings()
