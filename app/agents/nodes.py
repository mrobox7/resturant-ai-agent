"""The agent's graph steps.

This is one of two files meant to be rewritten per problem (the other is
app/tools/). Every node:

  - takes an AgentState
  - returns ONLY a partial dict of the fields it changed, never the whole state
  - has a single responsibility
"""
import json
import logging

import logfire

from app import memory
from app.agents.state import AgentState
from app.errors import InvalidInputError, MalformedResponseError
from app.llm_client import chat_completion
from app.tools import TOOL_REGISTRY, TOOL_SCHEMAS

logger = logging.getLogger(__name__)

MAX_QUERY_CHARS = 4000

AFFIRMATIVE_PHRASES = frozenset({"yes", "yeah", "yep", "yup", "confirm", "correct", "sure", "go ahead", "book it", "do it", "please do"})
NEGATIVE_PHRASES = frozenset({"no", "nope", "nah", "cancel", "nevermind", "never mind", "don't", "stop", "not now"})

KNOWN_INTENTS = frozenset({"menu", "availability", "booking", "view_booking", "order", "view_order"})


def _matches_any(query: str, phrases: frozenset[str]) -> bool:
    """Tokenized containment, not substring containment — "not interested"
    must never match "no" just because the letters appear inside "not"."""
    words = set(query.split())
    return any(phrase in words if " " not in phrase else phrase in query for phrase in phrases)


def _schema_for(name: str) -> list[dict]:
    return [s for s in TOOL_SCHEMAS if s["function"]["name"] == name]


MENU_TOOL_SCHEMA = _schema_for("get_menu")
AVAILABILITY_TOOL_SCHEMA = _schema_for("get_availability")
CREATE_RESERVATION_TOOL_SCHEMA = _schema_for("create_reservation")
VIEW_RESERVATIONS_TOOL_SCHEMA = _schema_for("view_reservations")
CREATE_ORDER_TOOL_SCHEMA = _schema_for("create_order")
VIEW_ORDERS_TOOL_SCHEMA = _schema_for("view_orders")


def _validate(query: str) -> None:
    """Deterministic checks. Raises InvalidInputError, which is never retried."""
    if not query.strip():
        raise InvalidInputError("query is empty")
    if len(query) > MAX_QUERY_CHARS:
        raise InvalidInputError(f"query exceeds {MAX_QUERY_CHARS} characters")


def guardrail_node(state: AgentState) -> dict:
    """Rejects input that is invalid on its face, before any LLM call is made."""
    with logfire.span("guardrail_node", query=state.current_query):
        try:
            _validate(state.current_query)
        except InvalidInputError as exc:
            logger.info("Guardrail rejected the query: %s", exc)
            return {
                "status": "escalated",
                "final_answer": f"Rejected: {exc}",
                "trace": [f"Guardrail: rejected — {exc}"],
            }
        return {"trace": ["Guardrail: query accepted"]}


def confirmation_gate_node(state: AgentState) -> dict:
    """Resolves a pending yes/no on a staged booking/order deterministically —
    a keyword check settles the common case, no LLM call needed for it.

    Always sets `route_after_confirmation` explicitly, on every path, so the
    conditional edge after this node never reads a stale value left over
    from a previous turn's checkpoint.
    """
    with logfire.span("confirmation_gate_node", pending=bool(state.needs_confirmation)):
        if not state.needs_confirmation or not state.pending_action:
            return {"route_after_confirmation": "continue"}

        query = state.current_query.lower().strip()
        if _matches_any(query, AFFIRMATIVE_PHRASES):
            return _execute_pending(state)
        if _matches_any(query, NEGATIVE_PHRASES):
            return {
                "pending_action": None,
                "needs_confirmation": False,
                "final_answer": "No problem — cancelled. Let me know if you'd like to try something else.",
                "status": "done",
                "route_after_confirmation": "end",
                "trace": ["Confirmation gate: user declined, cancelled the pending action"],
            }
        return {
            "final_answer": "Sorry, should I go ahead with that — yes or no?",
            "status": "needs_clarification",
            "route_after_confirmation": "end",
            "trace": ["Confirmation gate: reply wasn't a clear yes/no, asked again"],
        }


def _execute_pending(state: AgentState) -> dict:
    """Runs the staged tool call for real, now that the user confirmed."""
    action = state.pending_action
    result = _exec_tool(action["tool"], action["args"])
    output = result["tool_results"][0]["output"]
    if "error" not in output and action["tool"] in ("create_reservation", "create_order"):
        memory.append_episode(state.customer_id, _episode_summary(action["tool"], output))
    return {
        **result,
        "pending_action": None,
        "needs_confirmation": False,
        "route_after_confirmation": "respond",
    }


def _episode_summary(tool: str, output: dict) -> str:
    if tool == "create_reservation":
        return (
            f"Booked table {output['table_id']} for {output['party_size']} "
            f"at {output['slot_datetime']}"
        )
    return f"Ordered {output.get('quantity', 1)}x menu item #{output['menu_item_id']} (reservation {output['reservation_id']})"


def intent_gate_node(state: AgentState) -> dict:
    """Labels the query with an intent, cheaply first and via the LLM only if needed."""
    with logfire.span("intent_gate_node", query=state.current_query):
        # Tier 1: a deterministic check that costs nothing. Ordered so the
        # least ambiguous signals win first: "view_*" phrases (multi-word,
        # unlikely to false-positive) before their action counterparts, and
        # "order" (checked before "booking") since a bare word like
        # "reservation" shows up in both booking AND order-against-a-
        # -reservation phrasing ("order X for reservation 2") — "reservation"
        # alone is deliberately NOT a booking-intent keyword for that reason;
        # anything that ambiguous falls through to tier 2's constrained LLM
        # classification instead of being guessed here.
        keyword_intents: dict[str, tuple[str, ...]] = {
            "menu": (
                "menu", "dish", "dishes", "eat", "food", "vegetarian", "vegan",
                "starter", "main course", "dessert", "gluten", "nut", "allerg",
                "price", "cost", "how much",
            ),
            "view_booking": (
                "my booking", "my reservation", "my table", "check my reservation",
                "reservation status", "existing booking",
            ),
            "view_order": (
                "my order", "what did i order", "order history", "my food order",
            ),
            "availability": (
                "available", "availability", "free table", "open slot", "any tables",
            ),
            "order": (
                "order", "i'd like to have", "i want to order", "get me", "i'll have",
            ),
            "booking": ("book", "reserve", "table for"),
        }

        query = state.current_query.lower()
        for intent, keywords in keyword_intents.items():
            if any(keyword in query for keyword in keywords):
                return {"intent": intent, "trace": [f"Intent gate: '{intent}' (keyword match)"]}

        # Tier 2: ask the model, but only for the cases tier 1 could not settle.
        # The reply must be one of the known intents or it's not actionable —
        # an out-of-vocabulary reply is a deterministic-given-input failure,
        # caught here by validation rather than trusted through to tool_exec's
        # dispatch, where it would silently fall into the echo placeholder
        # with no signal that classification actually failed.
        response = chat_completion(
            [
                {
                    "role": "system",
                    "content": (
                        "Classify the user's message as exactly one of: "
                        + ", ".join(sorted(KNOWN_INTENTS))
                        + ". Reply with only that word, lowercase. If none fit, reply 'unknown'."
                    ),
                },
                {"role": "user", "content": state.current_query},
            ]
        )
        content = response.choices[0].message.content or ""
        candidate = content.strip().lower()
        intent = candidate if candidate in KNOWN_INTENTS else "unknown"
        return {"intent": intent, "trace": [f"Intent gate: '{intent}' (model classified)"]}


def context_gate_node(state: AgentState) -> dict:
    """Forces get_user_preferences + episodic-memory recall before a booking
    or order proceeds — never left to the model's discretion to call or skip.
    No-ops for every other intent, and runs at most once per thread: once
    `preferences_loaded` is true it stays true for the rest of the session.
    """
    with logfire.span("context_gate_node", intent=state.intent):
        if state.intent not in ("booking", "order") or state.preferences_loaded:
            return {}

        try:
            prefs = TOOL_REGISTRY["get_user_preferences"](customer_id=state.customer_id)
        except Exception as exc:
            # Left un-loaded on failure — a transient backend blip shouldn't
            # permanently strand the session with empty preferences; the
            # next booking/order turn retries this same gate.
            logger.warning("get_user_preferences failed: %s", exc)
            return {"trace": [f"Context gate: preferences fetch failed — {exc}"]}

        episodes = memory.read_episodes(state.customer_id)
        return {
            "customer_preferences": prefs.get("preferences", {}),
            "preferences_loaded": True,
            "recent_episodes": episodes,
            "trace": [f"Context gate: loaded preferences + {len(episodes)} past episode(s)"],
        }


def tool_exec_node(state: AgentState) -> dict:
    """Runs whatever tools the intent calls for and records their output."""
    with logfire.span("tool_exec_node", intent=state.intent):
        if state.intent == "menu":
            return _exec_tool("get_menu", _extract_args(
                state.current_query, MENU_TOOL_SCHEMA,
                "The user is asking about a restaurant's menu. Call get_menu "
                "with whatever filters their question implies. Call it with "
                "no arguments if they want the whole menu.",
            ))
        if state.intent == "availability":
            return _exec_tool("get_availability", _extract_args(
                state.current_query, AVAILABILITY_TOOL_SCHEMA,
                "The user wants to know what's free. Call get_availability "
                "with the date/time and party size their message implies.",
            ))
        if state.intent == "view_booking":
            args = _extract_args(
                state.current_query, VIEW_RESERVATIONS_TOOL_SCHEMA,
                "The user wants to see their booking(s). Call view_reservations "
                "with a reservation_id or status filter if they named one, "
                "otherwise call it with no arguments.",
            )
            args["customer_id"] = state.customer_id
            return _exec_tool("view_reservations", args)
        if state.intent == "view_order":
            args = _extract_args(
                state.current_query, VIEW_ORDERS_TOOL_SCHEMA,
                "The user wants to see their order(s). Call view_orders with "
                "a reservation_id if they named one, otherwise no arguments.",
            )
            args["customer_id"] = state.customer_id
            return _exec_tool("view_orders", args)
        if state.intent == "booking":
            return _prepare_booking(state)
        if state.intent == "order":
            return _prepare_order(state)
        # Placeholder dispatch for everything else. Replace as more
        # general-purpose intents/tools are added.
        return _exec_tool("echo", {"payload": state.current_query})


def _extract_args(query: str, schema: list[dict], instruction: str) -> dict:
    """One LLM function-calling round to turn free text into a tool's arguments.

    Genuine extraction tasks ("a table for 4 at 7pm", "vegetarian starters
    under 300") have no deterministic parse, so this is the one place per
    intent that calls the model — via real function-calling rather than a
    hand-rolled prompt to parse, so the model either produces valid
    arguments or doesn't; there's no free-form reply to interpret.
    """
    response = chat_completion(
        [
            {"role": "system", "content": instruction},
            {"role": "user", "content": query},
        ],
        tools=schema,
    )
    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        return {}
    try:
        parsed = json.loads(tool_calls[0].function.arguments or "{}")
    except json.JSONDecodeError:
        return {}
    # Some providers explicitly fill unused optional properties with null
    # rather than omitting them — schemas below allow that (Groq's own
    # tool-call validator rejects null against a plain "integer"/"string"
    # type otherwise), so an explicit null here means "not provided," same
    # as the key being absent, everywhere downstream that checks presence.
    return {k: v for k, v in parsed.items() if v is not None}


def _prepare_booking(state: AgentState) -> dict:
    """Extracts booking details, resolves a concrete table_id from
    availability, and stages the booking for confirmation — the mutating
    create_reservation call only happens once the user confirms (see
    confirmation_gate_node). The user is never asked for a table_id
    directly; they wouldn't have one — it's resolved the same way
    create_order resolves a menu item name and a reservation id."""
    instruction = (
        "The user wants to book a table. Call create_reservation with the "
        "slot_datetime, party_size, and location their message implies."
    )
    if state.customer_preferences:
        instruction += f" Known customer preferences: {state.customer_preferences}."

    args = _extract_args(state.current_query, CREATE_RESERVATION_TOOL_SCHEMA, instruction)
    missing = [f for f in ("slot_datetime", "party_size") if f not in args]
    if missing:
        output = {"status": "missing_fields", "missing": missing}
        trace = f"Tool exec: booking incomplete, missing {missing}"
        return {"tool_results": [{"tool": "create_reservation", "output": output}], "trace": [trace]}

    special_requests = args.pop("special_requests", None)
    seating_preference = state.customer_preferences.get("seating")
    try:
        table_id = _resolve_table_id(
            **args, preferred_location=seating_preference if isinstance(seating_preference, str) else None
        )
    except ValueError as exc:
        output = {"status": "no_table_resolved", "detail": str(exc)}
        trace = f"Tool exec: booking — {exc}"
        return {"tool_results": [{"tool": "create_reservation", "output": output}], "trace": [trace]}

    booking_args = {
        "customer_id": state.customer_id,
        "table_id": table_id,
        "slot_datetime": args["slot_datetime"],
        "party_size": args["party_size"],
        "special_requests": special_requests,
    }
    summary = {"status": "awaiting_confirmation", **{k: v for k, v in booking_args.items() if k != "customer_id"}}
    return {
        "pending_action": {"tool": "create_reservation", "args": booking_args},
        "needs_confirmation": True,
        "tool_results": [{"tool": "create_reservation", "output": summary}],
        "trace": [f"Tool exec: booking staged for confirmation — {summary}"],
    }


def _resolve_table_id(
    *,
    slot_datetime: str,
    party_size: int,
    location: str | None = None,
    preferred_location: str | None = None,
) -> int:
    """Resolves soft booking criteria to one concrete table_id via a real
    get_availability call — never guessed, never asked of the user. When
    the user didn't state a location and several tables are free, the
    customer's stored seating preference (if any) breaks the tie; if several
    are still free after that, any one of them is as good as another to the
    customer, so the lowest table_number is picked rather than blocking the
    booking on a distinction nobody asked about — the only real failure is
    zero tables free."""
    availability = TOOL_REGISTRY["get_availability"](
        slot_datetime=slot_datetime, party_size=party_size, location=location
    )
    tables = availability["available_tables"]
    if not tables:
        raise ValueError(f"no tables free for {party_size} at {slot_datetime}")

    if len(tables) > 1 and not location and preferred_location:
        narrowed = [t for t in tables if t["location"] == preferred_location]
        if narrowed:
            tables = narrowed

    return min(tables, key=lambda t: t["table_number"])["table_id"]


def _prepare_order(state: AgentState) -> dict:
    """Extracts order details and stages them for confirmation. Resolving the
    dish name to a real menu_item_id and, if omitted, which reservation to
    attach it to both happen inside create_order itself once confirmed —
    not duplicated here, so there's exactly one place that logic lives."""
    instruction = (
        "The user wants to order food. Call create_order with the dish name "
        "and quantity their message implies."
    )
    if state.customer_preferences:
        instruction += f" Known customer preferences: {state.customer_preferences}."

    args = _extract_args(state.current_query, CREATE_ORDER_TOOL_SCHEMA, instruction)
    if "menu_item_name" not in args:
        output = {"status": "missing_fields", "missing": ["menu_item_name"]}
        trace = "Tool exec: order incomplete, missing menu_item_name"
        return {"tool_results": [{"tool": "create_order", "output": output}], "trace": [trace]}

    args["customer_id"] = state.customer_id
    summary = {
        "status": "awaiting_confirmation",
        "menu_item_name": args["menu_item_name"],
        "quantity": args.get("quantity", 1),
        "reservation_id": args.get("reservation_id"),
    }
    return {
        "pending_action": {"tool": "create_order", "args": args},
        "needs_confirmation": True,
        "tool_results": [{"tool": "create_order", "output": summary}],
        "trace": [f"Tool exec: order staged for confirmation — {summary}"],
    }


def _exec_tool(tool_name: str, kwargs: dict) -> dict:
    tool = TOOL_REGISTRY[tool_name]
    try:
        output = tool(**kwargs)
    except Exception as exc:  # a failing tool is data for the model, not a crash
        logger.warning("Tool %s failed: %s", tool_name, exc)
        output = {"error": str(exc)}
        trace_line = f"Tool exec: {tool_name}({kwargs}) failed — {exc}"
    else:
        trace_line = f"Tool exec: {tool_name}({kwargs}) -> {output}"

    return {"tool_results": [{"tool": tool_name, "output": output}], "trace": [trace_line]}


def respond_node(state: AgentState) -> dict:
    """Produces the final answer, repairing one malformed reply before giving up."""
    messages = [
        {
            "role": "system",
            "content": (
                "Answer the user using the tool results provided. If a tool "
                "result has status 'missing_fields', ask the user for that "
                "information. If it has status 'awaiting_confirmation', "
                "summarize it clearly and ask the user to confirm with yes/no. "
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

    with logfire.span("respond_node"):
        try:
            answer = _answer_from(chat_completion(messages))
            return {
                "final_answer": answer,
                "status": "done",
                "trace": ["Respond: answered"],
            }
        except MalformedResponseError as exc:
            logger.warning("Malformed reply (%s) — asking the model to repair it", exc)

        repair = messages + [
            {
                "role": "user",
                "content": "That was not valid JSON. Reply with only the JSON object.",
            }
        ]
        try:
            answer = _answer_from(chat_completion(repair))
            return {
                "final_answer": answer,
                "status": "done",
                "trace": ["Respond: malformed reply, repaired on retry"],
            }
        except MalformedResponseError as exc:
            # One repair attempt is the cap. Past that, escalate rather than
            # trust a third free-form blob.
            logger.error("Repair attempt also failed: %s", exc)
            return {"status": "escalated", "trace": ["Respond: repair failed, escalated"]}


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
