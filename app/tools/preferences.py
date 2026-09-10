"""Preferences lookup. Unlike every other tool here, the model never chooses
to call this one — context_gate_node calls it deterministically whenever the
intent is booking or ordering (see app/agents/nodes.py). It's still
registered as a tool (not just a bare backend_api call) because it validates
its output and enforces the PII redaction below, the same contract every
other tool honors.
"""
from app import backend_api
from app.tools.schemas import CustomerPreferencesOut


def get_user_preferences(*, customer_id: int) -> dict:
    """Fetch the customer's stored preferences. Phone and email never leave
    this function — only name and preferences are fit to reach agent state."""
    raw = backend_api.get_customer(customer_id)
    return CustomerPreferencesOut(**raw).model_dump()


GET_USER_PREFERENCES_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_user_preferences",
        "description": "Fetch the customer's stored dining preferences (seating, dietary, etc).",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}
