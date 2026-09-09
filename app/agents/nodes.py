"""The four agent steps.

This is one of two files meant to be rewritten per problem (the other is
tools.py). Every node here is a placeholder that demonstrates the contract,
not real logic:

  - takes an AgentState
  - returns ONLY a partial dict of the fields it changed, never the whole state
  - has a single responsibility
"""
import json
import logging

from app.agents.state import AgentState
from app.errors import InvalidInputError, MalformedResponseError
from app.llm_client import chat_completion
from app.tools import TOOL_REGISTRY

logger = logging.getLogger(__name__)

MAX_QUERY_CHARS = 4000


def _validate(query: str) -> None:
    """Deterministic checks. Raises InvalidInputError, which is never retried."""
    if not query.strip():
        raise InvalidInputError("query is empty")
    if len(query) > MAX_QUERY_CHARS:
        raise InvalidInputError(f"query exceeds {MAX_QUERY_CHARS} characters")


def guardrail_node(state: AgentState) -> dict:
    """Rejects input that is invalid on its face, before any LLM call is made."""
    try:
        _validate(state.current_query)
    except InvalidInputError as exc:
        logger.info("Guardrail rejected the query: %s", exc)
        return {"status": "escalated", "final_answer": f"Rejected: {exc}"}
    return {}


def intent_gate_node(state: AgentState) -> dict:
    """Labels the query with an intent, cheaply first and via the LLM only if needed."""
    # Tier 1: a deterministic check that costs nothing. Replace the mapping
    # with whatever the problem's intents actually are; anything not matched
    # here falls through to the model.
    keyword_intents: dict[str, tuple[str, ...]] = {}

    query = state.current_query.lower()
    for intent, keywords in keyword_intents.items():
        if any(keyword in query for keyword in keywords):
            return {"intent": intent}

    # Tier 2: ask the model, but only for the cases tier 1 could not settle.
    response = chat_completion(
        [
            {
                "role": "system",
                "content": "Reply with a single lowercase word naming the user's intent.",
            },
            {"role": "user", "content": state.current_query},
        ]
    )
    content = response.choices[0].message.content or ""
    return {"intent": content.strip().lower() or "unknown"}


def tool_exec_node(state: AgentState) -> dict:
    """Runs whatever tools the intent calls for and records their output."""
    # Placeholder dispatch: echoes the query through the one registered tool.
    # Replace with real selection — either driven by state.intent or by the
    # model's tool_calls, depending on what the problem needs.
    tool_name = "echo"
    tool = TOOL_REGISTRY[tool_name]

    try:
        output = tool(state.current_query)
    except Exception as exc:  # a failing tool is data for the model, not a crash
        logger.warning("Tool %s failed: %s", tool_name, exc)
        output = {"error": str(exc)}

    return {"tool_results": [{"tool": tool_name, "output": output}]}


def respond_node(state: AgentState) -> dict:
    """Produces the final answer, repairing one malformed reply before giving up."""
    messages = [
        {
            "role": "system",
            "content": (
                "Answer the user using the tool results provided. "
                'Reply with ONLY a JSON object: {"answer": "<your answer>"}'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Query: {state.current_query}\n"
                f"Tool results: {json.dumps(state.tool_results)}"
            ),
        },
    ]

    try:
        return {"final_answer": _answer_from(chat_completion(messages)), "status": "done"}
    except MalformedResponseError as exc:
        logger.warning("Malformed reply (%s) — asking the model to repair it", exc)

    repair = messages + [
        {
            "role": "user",
            "content": "That was not valid JSON. Reply with only the JSON object.",
        }
    ]
    try:
        return {"final_answer": _answer_from(chat_completion(repair)), "status": "done"}
    except MalformedResponseError as exc:
        # One repair attempt is the cap. Past that, escalate rather than
        # trust a third free-form blob.
        logger.error("Repair attempt also failed: %s", exc)
        return {"status": "escalated"}


def _answer_from(response) -> str:
    """Pulls the answer field out of a JSON reply, or flags it as malformed."""
    raw = response.choices[0].message.content or ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MalformedResponseError(f"reply was not JSON: {raw[:200]}") from exc

    if not isinstance(parsed, dict) or "answer" not in parsed:
        raise MalformedResponseError(f"reply missing 'answer' key: {raw[:200]}")
    return str(parsed["answer"])
