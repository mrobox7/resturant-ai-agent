"""
Core agent loop: send the conversation to the LLM, execute any tool calls it
requests, feed the results back, and repeat until the model returns a final
answer — then validate that answer against a Pydantic schema before
returning it (with one repair attempt if it's malformed).
"""
import json
import logging
from pydantic import ValidationError

from app.llm_client import chat_completion
from app.tools import TOOL_REGISTRY, TOOL_SCHEMAS
from app.models import TaskResponse
from app.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a task-completion agent. Use the available tools when
they help answer the user's request. When you have everything you need, reply
with ONLY a JSON object matching this schema and nothing else:
{"answer": "<final answer text>", "tools_used": ["<tool names you called>"]}
"""


def run_agent(user_request: str) -> TaskResponse:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_request},
    ]
    tools_used = []

    for _ in range(settings.max_tool_iterations):
        response = chat_completion(messages, tools=TOOL_SCHEMAS)
        choice = response.choices[0].message

        if choice.tool_calls:
            messages.append(choice.model_dump(exclude_none=True))
            for call in choice.tool_calls:
                fn_name = call.function.name
                args = json.loads(call.function.arguments or "{}")
                logger.info("Calling tool %s with %s", fn_name, args)
                try:
                    result = TOOL_REGISTRY[fn_name](**args)
                    tools_used.append(fn_name)
                except Exception as exc:
                    result = {"error": str(exc)}
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result),
                })
            continue

        # No more tool calls — expect the final structured answer
        return _parse_final_answer(choice.content, tools_used)

    raise RuntimeError("Agent exceeded max tool iterations without a final answer")


def _parse_final_answer(raw: str, tools_used: list) -> TaskResponse:
    """Parse and validate the model's final JSON reply, with one repair retry."""
    try:
        data = json.loads(raw)
        data.setdefault("tools_used", tools_used)
        return TaskResponse(**data)
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning("Final answer failed validation (%s) — asking model to fix it", exc)
        repair = chat_completion([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Fix this so it is valid JSON matching the schema: {raw}"},
        ])
        data = json.loads(repair.choices[0].message.content)
        data.setdefault("tools_used", tools_used)
        return TaskResponse(**data)
