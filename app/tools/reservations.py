"""Booking tools. customer_id is never part of what the model extracts —
it's a required keyword the caller (tool_exec_node) attaches from
AgentState.customer_id after extraction, same as every other identity field
in this codebase.
"""
from app import backend_api
from app.tools.schemas import ReservationCreateInput, ReservationOut, ViewReservationsQuery


def create_reservation(*, customer_id: int, **kwargs) -> dict:
    """Book a table. The backend validates the slot is actually still free."""
    fields = ReservationCreateInput(**kwargs)  # deterministic validation, before any network call
    payload = {"customer_id": customer_id, **fields.model_dump(mode="json")}
    raw = backend_api.create_reservation(payload)
    # mode="json": datetimes as ISO strings, not Python objects — tool_results
    # is fed to json.dumps() in respond_node, which can't serialize a datetime.
    return ReservationOut(**raw).model_dump(mode="json")


def view_reservations(*, customer_id: int, **kwargs) -> list[dict]:
    """Look up a customer's bookings — one specific reservation, or their whole history."""
    query = ViewReservationsQuery(**kwargs)
    if query.reservation_id is not None:
        raw = backend_api.get_reservation(query.reservation_id)
        return [ReservationOut(**raw).model_dump(mode="json")]
    raw_items = backend_api.list_customer_reservations(customer_id, status=query.status)
    return [ReservationOut(**item).model_dump(mode="json") for item in raw_items]


CREATE_RESERVATION_SCHEMA = {
    "type": "function",
    "function": {
        "name": "create_reservation",
        "description": (
            "Book a table for the customer at a given date/time and party size. "
            "The specific table is resolved automatically from availability — "
            "never ask the user for a table_id, they won't have one."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "slot_datetime": {
                    "type": "string",
                    "description": "ISO 8601 date/time of the booking, e.g. '2026-09-10T19:00:00'.",
                },
                "party_size": {
                    "type": "integer",
                    "description": "Number of people in the party.",
                },
                # [type, "null"]: optional properties must tolerate an
                # explicit null — some providers fill unused optional slots
                # that way rather than omitting them, and reject the call
                # otherwise. _extract_args strips nulls back out.
                "location": {
                    "type": ["string", "null"],
                    "enum": ["indoor", "outdoor"],
                    "description": "Seating preference, if the user stated one.",
                },
                "special_requests": {
                    "type": ["string", "null"],
                    "description": "Any special requests, e.g. a birthday, a window seat.",
                },
            },
            "required": ["slot_datetime", "party_size"],
        },
    },
}

VIEW_RESERVATIONS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "view_reservations",
        "description": "Look up the customer's existing bookings, or one specific booking by id.",
        "parameters": {
            "type": "object",
            "properties": {
                "reservation_id": {
                    "type": ["integer", "null"],
                    "description": "A specific reservation to look up, if the user named one.",
                },
                "status": {
                    "type": ["string", "null"],
                    "enum": ["confirmed", "cancelled", "completed"],
                    "description": "Restrict to bookings in this status.",
                },
            },
            "required": [],
        },
    },
}
