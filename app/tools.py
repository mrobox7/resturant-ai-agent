"""
Example tools the agent can call. Each tool is:
  - a plain Python function that does the work
  - a JSON schema entry (OpenAI tool-calling format) describing it to the LLM

Swap these for whatever the interview prompt actually needs (a real API call,
a DB query, a file read...) — the calling pattern in agent.py doesn't change.
"""
import math


def get_weather(city: str) -> dict:
    """Mock weather lookup — replace with a real API call in production."""
    fake_data = {
        "jaipur": {"condition": "sunny", "temp_c": 34},
        "bengaluru": {"condition": "cloudy", "temp_c": 24},
        "mumbai": {"condition": "humid", "temp_c": 30},
    }
    return fake_data.get(city.lower(), {"condition": "unknown", "temp_c": None})


def calculate(expression: str) -> float:
    """Evaluate a basic arithmetic expression from a restricted character set.

    NOTE: real eval() is unsafe for production — this is scoped down for
    demo purposes. Use a proper expression parser (e.g. asteval) for real use.
    """
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        raise ValueError("Expression contains disallowed characters")
    return eval(expression, {"__builtins__": {}}, {"math": math})


TOOL_REGISTRY = {
    "get_weather": get_weather,
    "calculate": calculate,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a basic arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]
