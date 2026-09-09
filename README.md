# Agent scaffold

A small LangGraph agent behind a FastAPI endpoint, with typed config, a
retry policy, and tests that run without an API key.

The nodes are placeholders. They demonstrate the shape of the thing — a
guardrail that rejects bad input before spending a call, an intent step, a
tool step, a structured answer with one repair attempt — and are meant to be
replaced with whatever the actual task needs.

## Setup

Needs Python 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env      # fill in one provider's key
```

```bash
make dev      # API on :8000
make ui       # chat UI on :8501
make test     # test suite, no API key needed
make graph    # print the graph as mermaid
```

Or `docker compose up --build`.

## The provider

Groq, via the `openai` package — Groq speaks the OpenAI chat-completions
protocol, so that package is a protocol client here rather than a commitment
to OpenAI. Nothing in the code names a provider; `llm_base_url`, `llm_model`
and `llm_api_key` are the whole of it, which is the escape hatch if a free
tier starts rate-limiting at the wrong moment.

Model names go stale. On a 404 saying `model_not_found`, ask the endpoint
what it serves:

```bash
uv run python -c "from app.llm_client import _get_client; \
  print([m.id for m in _get_client().models.list().data])"
```

## Layout

```
app/
  config/settings.py   typed env config
  config/logging.py    logfire, skipped when no token is set
  errors.py            the failure kinds, split by how each is handled
  llm_client.py        chat_completion + retry policy
  models.py            structured-output schemas
  agents/state.py      the state passed between nodes
  agents/nodes.py      the four steps          <- rewrite per task
  agents/graph.py      how they are wired
  tools.py             tool functions + schemas <- rewrite per task
  backend_api.py       everything that touches a real system
  main.py              HTTP layer
ui/streamlit_app.py    chat UI
```

## The node contract

A node takes the state and returns **only the fields it changed**:

```python
def guardrail_node(state: AgentState) -> dict:
    if bad(state.current_query):
        return {"status": "escalated"}
    return {}                      # changed nothing, carry on
```

Returning the whole state works but quietly overwrites anything a
concurrent node wrote, so partial dicts are the convention. `messages` is
declared `Annotated[list[dict], operator.add]`, which makes that one channel
append instead of overwrite.

Routing lives in `graph.py`, not inside the nodes, so the flow stays
readable as a diagram (`make graph`).

## Failures

Retrying is not one behaviour, so `errors.py` splits it four ways:

| | |
|---|---|
| `InvalidInputError` | Deterministic given the input. Retrying reproduces it, so it is caught before any call is made. |
| `TransientLLMError` | Dropped connection, timeout, 5xx. Retried with exponential backoff. |
| `RateLimitError` | A 429 carries a `Retry-After`. Sleeps for exactly that; backoff maths would either return too early or idle well past the window. |
| `MalformedResponseError` | The reply parsed as text but not as the requested shape. Retried once with a repair prompt, then escalated rather than trusting a third attempt. |

A 4xx that isn't a 429 is a bad request and is raised immediately.

## Extending it

**A tool** is a function plus a schema entry in `tools.py`. Anything reaching
a real system goes through `backend_api.py`, which is also where
authorization belongs — not in `tools.py`, and never in a prompt, since
anything the model can see it can be talked out of.

**A step** is a function in `nodes.py` plus a line in `graph.py`.

**Structured output** is `models.py`. The two-model split there is worth
keeping: fields the model fills are all optional, so an omission is a null to
handle rather than a validation crash, and fields like `user_id` and
`timestamp` are attached by code and never asked for — a model that can write
its own caller identity is an identity an attacker can set from the prompt.

## Tests

`uv run pytest`. The LLM is mocked throughout, so the suite is deterministic
and costs nothing. `chat_completion` is patched where the module under test
looks it up.

The client is built on first call rather than at import, which is what lets
the whole suite import `app.llm_client` with no credentials present — there
is a test asserting exactly that.

## Known gaps

- No persistence. Each request is independent; multi-turn needs a
  checkpointer or an external store.
- No token or cost budget. `max_retries` bounds attempts, not spend.
- The retry policy has no jitter, so simultaneous clients back off in step.
- `logfire` is wired but only reports if `LOGFIRE_TOKEN` is set.
