from app import backend_api
from app.tools.schemas import AvailabilityQuery, AvailableTable

LOCATIONS = ["indoor", "outdoor"]


def get_availability(**kwargs) -> dict:
    """List tables free at a given slot, optionally filtered by location."""
    query = AvailabilityQuery(**kwargs)  # deterministic validation, before any network call
    raw = backend_api.get_availability(query.model_dump(mode="json", exclude_none=True))
    return {
        "slot_datetime": raw["slot_datetime"],
        "party_size": raw["party_size"],
        "available_tables": [AvailableTable(**t).model_dump() for t in raw["available_tables"]],
    }


GET_AVAILABILITY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_availability",
        "description": (
            "Check which tables are free at a given date/time for a given party size. "
            "Use this before booking, or when the user asks if a slot is open."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "slot_datetime": {
                    "type": "string",
                    "description": (
                        "The date and time to check, ISO 8601 (e.g. '2026-09-10T19:00:00'). "
                        "Slots are 30-minute increments from 12:00 to 22:30."
                    ),
                },
                "party_size": {
                    "type": "integer",
                    "description": "Number of people in the party.",
                },
                "location": {
                    # [type, "null"]: optional properties must tolerate an
                    # explicit null — some providers fill unused optional
                    # slots that way rather than omitting them, and reject
                    # the call otherwise. _extract_args strips nulls back out.
                    "type": ["string", "null"],
                    "enum": LOCATIONS,
                    "description": "Restrict to indoor or outdoor seating.",
                },
            },
            "required": ["slot_datetime", "party_size"],
        },
    },
}
