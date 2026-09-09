"""Tool implementations and the schemas advertised to the model.

This is the second of the two files meant to be rewritten per problem. A tool
is a plain function plus an entry in TOOL_SCHEMAS describing it to the model;
anything that touches a real system goes through backend_api.py so that
authorization stays outside the LLM's reach.
"""
from app import backend_api


def echo(payload: str) -> dict:
    """Placeholder tool: hands its input to the backend and returns the result."""
    return backend_api.fetch(payload)


TOOL_REGISTRY = {
    "echo": echo,
}

TOOL_SCHEMAS = [
    {
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
    },
]
