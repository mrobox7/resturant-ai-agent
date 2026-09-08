# AI Agent Starter Kit

A minimal, working example of an **agentic workflow**: tool/function calling,
structured output validated with Pydantic, retries with backoff, typed
config via `pydantic-settings`, a FastAPI endpoint, Docker (built with
`uv`), and tests with the LLM mocked out.

It's deliberately generic so you can gut the domain logic (`app/tools.py`,
`app/models.py`, the system prompt in `app/agent.py`) and rebuild it around
whatever the actual interview prompt asks for, while keeping the same
plumbing: agent loop → tool execution → structured/validated final answer.

## Why this shape

Mapped directly to the evaluation criteria in the guidelines you were given:

- **Agent design** → `agent.py` runs a bounded tool-calling loop, not an
  unbounded `while True`.
- **Structured outputs** → the final answer is a Pydantic model
  (`TaskResponse`), and a malformed reply triggers one "repair" call instead
  of crashing.
- **Reliability** → `llm_client.py` retries transient failures with
  exponential backoff, and the OpenAI client is constructed lazily (on
  first real use, not at import time) so the module stays importable and
  testable with zero credentials present.
- **Typed config** → `app/config.py` uses `pydantic-settings` instead of
  scattered `os.getenv()` calls — env vars are validated and type-coerced
  once, at startup, with real error messages if something's malformed.
- **Clean, modular code** → LLM calls, tool definitions, agent logic, and
  the API layer are separate files, not one script.
- **Testing/evaluation** → `tests/test_agent.py` mocks the LLM so tests run
  deterministically and don't burn API calls.
- **Production readiness talking points** → Docker + docker-compose, built
  with `uv` so the container installs the exact locked dependency versions
  from `uv.lock`, not "whatever pip resolves today."

## Setup

Requires Python >= 3.14 and [`uv`](https://docs.astral.sh/uv/).

1. `cp .env.example .env` and fill in ONE provider's key (Groq and
   OpenRouter both have free tiers — see the interview guidelines).
2. Local run:
   ```
   uv sync
   uv run uvicorn app.main:app --reload
   ```
3. Docker run:
   ```
   docker compose up --build
   ```
4. Test it:
   ```
   curl -X POST localhost:8000/agent/run \
     -H "Content-Type: application/json" \
     -d '{"request": "What is the weather in Jaipur, and what is 12 * 7?"}'
   ```
5. Run tests: `uv run pytest -v`

## Adapting this during the interview

- New tool → add a function + JSON schema entry in `tools.py`, register it in
  `TOOL_REGISTRY`. The agent loop doesn't change.
- New output shape → edit `TaskResponse` in `models.py`.
- New config value → add a typed field to `Settings` in `config.py`.
- New task → rewrite `SYSTEM_PROMPT` in `agent.py`.
- If they want a *different* framework (LangChain/LangGraph/CrewAI), the
  concepts here (loop, tool schema, validated output, retries) transfer
  directly — you're just swapping who owns the loop.

## Known trade-offs (good to say out loud if asked)

- `max_tool_iterations` is a blunt safeguard against infinite loops; a real
  system would also track cost/token budget per request.
- The "repair" step only retries once — production would log the failure
  and probably fall back to a safe default rather than trusting a second
  free-form JSON blob.
- No conversation persistence/state store here — each call is stateless.
  Add a session store (Redis/DB) if the task needs multi-turn memory.
- `calculate()` uses a restricted `eval` for demo speed; in production use a
  real expression parser (e.g. `asteval`), not `eval` at all.
- `Dockerfile` uses `--no-dev` on `uv sync` so `pytest` never ships in the
  runtime image — dev tooling and prod dependencies are deliberately split.
