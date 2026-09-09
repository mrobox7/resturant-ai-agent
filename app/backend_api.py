"""The boundary between the agent and whatever actually holds the data.

Every call the tools make to a database, HTTP service, or filesystem belongs
here. Authorization belongs here too — never in tools.py, and never in a
prompt, because anything the model can see it can also be talked out of.
Callers pass the identity down; this layer decides what that identity may do.
"""


def fetch(payload: str, requesting_user_id: str | None = None) -> dict:
    """Stub for a real backend read. Replace with the problem's data source."""
    # A real implementation authorizes first and raises/returns empty on denial:
    #     if not _may_read(requesting_user_id, payload):
    #         raise PermissionError(...)
    return {"payload": payload, "requested_by": requesting_user_id}
