from app import backend_api


def echo(payload: str) -> dict:
    """Placeholder tool: hands its input to the backend and returns the result."""
    return backend_api.fetch(payload)


ECHO_SCHEMA = {
    "type": "function",
    "function": {
        "name": "echo",
        "description": "Placeholder tool. Replace with the problem's real tools.",
        "parameters": {
            "type": "object",
            "properties": {
                "payload": {"type": "string", "description": "Text to pass through"},
            },
            "required": ["payload"],
        },
    },
}
