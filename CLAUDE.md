# Shuru AI Software Engineer — interview practice conventions

This file is read automatically at the start of any session in this
project. Apply everything below to whatever problem is described, without
being asked to re-derive it. This is a decision framework, not a fixed
template — write only the files a given problem actually needs.

## What this interview is actually grading

Per the interview guidelines: the coding round asks for a working AI agent
or agentic workflow — prompting/structured outputs, tool/function calling
and state handling, API/file/DB/service integrations, and validation/
retries/error handling/testing. Evaluation criteria, explicitly: problem
solving and agent design; clean, modular Python code; LLM integration and
reliability; testing/evaluation approach; ability to explain trade-offs
and production readiness. Every convention below exists to serve one of
these five directly — if a choice doesn't visibly serve one of them, it's
not worth the time in a live round.

Required setup per the guidelines: Python + IDE, Docker/Docker Compose,
Git + a package manager, and your own LLM API key (no key is provided) —
Groq, OpenRouter free models, or Google AI Studio are the named free
options. Test this setup before the interview starts, not during it.

## Stack defaults

- Python >=3.14, dependency management via `uv` only. `pyproject.toml` +
  `uv.lock`. Never `requirements.txt` or bare `pip`.
- FastAPI for any HTTP layer.
- Agent runtime: **LangGraph** (`StateGraph`, conditional edges,
  checkpointing). Do not use LangChain's chains, agent constructors, or
  integrations — write LLM calls directly against the `openai` SDK in
  OpenAI-compatible mode (works unmodified with OpenAI, Groq, OpenRouter via
  `base_url`). LangChain is only justified if a problem needs a specific
  named integration that would be slower to hand-write than to import.
- Config: `pydantic-settings`, not `os.getenv` / `python-dotenv` directly.
  Reads `.env`. Never commit `.env` — only `.env.example` with placeholders.
- UI when needed: Streamlit, kept under ~40 lines.
- Observability: `logfire` for general app/FastAPI tracing (one line to
  configure). Langfuse **Cloud hobby tier** only if a problem specifically
  needs LLM-call-level tracing — never self-hosted Langfuse in an
  interview setting (multi-container stack, not worth the setup risk).
- Guardrails: no dedicated library (NeMo Guardrails, Guardrails AI, Llama
  Guard). Build the two-tier gate described below instead — it's faster to
  write than to configure any of those, and it's the same pattern already
  used for intent classification. Name those libraries only if asked what
  a real production system would use.
- Docker: keep `Dockerfile` + `docker-compose.yml` ready (this is
  explicitly required per the interview guidelines), built with `uv sync
  --frozen --no-dev`. Do not rebuild the image on every iteration during
  development — iterate with `uv run uvicorn --reload`, run
  `docker compose up` once near the end to prove it works.
- No S3, no managed cloud DB. If a problem needs persistence, use SQLite
  (stdlib `sqlite3`) or an in-memory store. If it needs file storage, use
  local disk (`/tmp`).

## Recurring architectural principles

Apply these regardless of problem domain — they are the actual thing being
evaluated, not incidental detail.

1. **Deterministic gates before expensive/risky calls.** Any "should I do
   X" decision that can be answered by a plain code check (a required
   field is present, a keyword matches, a value crosses a threshold)
   is answered in code, not asked of the LLM as a judgment call. The LLM
   is only invoked for genuine ambiguity a rule can't resolve.
2. **Instruction/data separation.** Model output is always a request, a
   draft, or an extraction — never executable code, a raw DB query, or a
   source of truth for identity/authorization. The model proposes; code
   disposes.
3. **Identity and authorization live outside the model's reach.**
   `user_id`, session identity, and permission checks are never fields an
   LLM is asked to extract or produce. They are attached by code from the
   authenticated session/request context. Any field that describes *who
   is asking* (not what's being asked) never enters a prompt or an
   extraction schema.
4. **Two-schema split for any extraction task.** One schema for what the
   LLM fills in (nullable, genuinely uncertain, lives in the source text)
   and a separate schema for what the system already knows with certainty
   (identity, timestamps) — assembled together by code, never by the model.
5. **Full PII never crosses into LLM context.** Any tool that fetches a
   record with sensitive fields (address, phone, payment info) returns a
   redacted view to the agent; the full record stays backend-side, used
   only when a specific action needs it directly.
6. **Capped retries, typed fallback, always.** Nothing retries
   indefinitely. Every retry loop (transient failure, malformed output,
   unresolved ambiguity) has an explicit cap and an explicit fallback
   (escalate, reroute, flag for review) — never a bare loop or a silent
   crash.
7. **Failure bucket discipline** — three kinds, three handlings:
   - *Transient* (timeout, 5xx) → retry same request, exponential backoff,
     capped.
   - *Deterministic given input* (token limit exceeded, malformed request
     shape) → never retried — must be caught by validation **before** the
     call happens.
   - *Stochastic-recoverable* (malformed JSON reply) → retry-worthy
     because resampling can genuinely fix it, but with a *repair prompt*
     (tell the model what was wrong), not a blind resend. Capped
     separately from transient retries.
   - 429 specifically: honor the provider's `Retry-After` value exactly.
     Never apply your own backoff formula to it — cooldown policy is
     provider-specific.
8. **Tool calling never talks to a database or filesystem directly.**
   Tools call a backend/API layer; that layer owns real authorization
   checks and real queries (parameterized, never string-interpolated).
   The flexibility of what a tool's input schema can express is fine —
   the risk was always in what's allowed to be *returned*, and that's
   enforced one layer below the model, same as any other client.
9. **Chained/multi-tool calls need no special logic.** A tool-calling
   loop that checks "does the model still see a gap an available tool can
   fill" and repeats until it doesn't is sufficient — there is no separate
   "chaining" mechanism to design.
10. **Chunking/RAG:** semantic or sentence-boundary chunking over
    fixed-size splitting. Cross-chunk dependency is a real,
    not-fully-solvable problem — mitigate with overlapping windows /
    neighbor-chunk retrieval, backstop with faithfulness evals, don't
    claim any single technique guarantees it away.

## Node contract (LangGraph)

Every node function in `nodes.py`:
- has a one-line docstring naming its single responsibility
- reads only the `AgentState` fields it actually needs
- returns only a **partial dict** of the fields it changes — never the
  whole state (LangGraph merges this via the state's own reducers, e.g.
  `Annotated[list, operator.add]` for `messages`)
- calls an LLM only if the node's job genuinely requires reasoning;
  deterministic checks (guardrail rules, gate fast-paths) make zero calls

Standard node set, adapted per problem — not all are needed every time:
`guardrail_node` (deterministic decline check) → `intent_gate_node`
(rule-first, cheap-model fallback) → `agent_node` (main model, tools
bound) ⇄ `tools_node` (no model, executes registered tools, loops back to
`agent_node`) → `respond_node` (parses/validates final structured output,
one repair attempt on failure).

## Model selection per node

- Guardrail / deterministic gate checks: no model.
- Ambiguous intent classification (gate's fallback tier): cheapest
  adequate model (e.g. Llama 3.1 8B on Groq).
- Agent node (tool-calling, multi-step reasoning): the strongest available
  model for the task — don't economize here.
- Tools node: no model, pure code execution.
- Respond node's repair attempt: same model as the agent node (already
  has full context) — reuse, don't introduce a third model tier without a
  clear reason.

## Testing conventions

- Deterministic logic (gates, rules, escalation checks) gets pure-function
  tests — no mocking.
- LLM-touching logic gets `chat_completion` mocked, never a real call in
  a test.
- One test per failure bucket for anything with retry logic, including an
  explicit assertion that the mock was never called for the
  pre-validated-rejection case.
- Prefer a structural test over a happy-path-only test where identity
  boundaries matter (e.g. assert `user_id` never appears in an extraction
  schema or in what's sent to the LLM).

## What NOT to do in this setting

- Don't wire real cloud infra (S3, managed DB, self-hosted observability
  stack) — every one of these has a lightweight stand-in listed above.
- Don't spend early time perfecting one path before getting a real
  (non-mocked) LLM round-trip working — real tool-calling has failure
  modes mocks never surface.
- Don't build comprehensive test coverage — a few tests chosen for what
  they prove beats many chosen for coverage percentage.
- Don't silently retry a deterministic-given-input failure, and don't
  silently loop past a retry cap without an explicit fallback.

## Narration reminder

"Explain trade-offs" is a scored criterion, not small talk. For every
non-obvious choice made under this file's conventions, be ready to state
it in one sentence: why a gate is deterministic instead of an LLM call,
why a retry is capped, why identity never enters a prompt. If a choice
here can't be justified in one sentence, that's worth noticing before the
interviewer asks.