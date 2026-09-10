"""Tool implementations and the schemas advertised to the model.

This is the second of the two files/packages meant to be rewritten per
problem (the other is app/agents/nodes.py). Each module here is a plain
function plus an entry in TOOL_SCHEMAS describing it to the model; anything
that touches a real system goes through backend_api.py so that authorization
stays outside the LLM's reach.
"""
from app.tools.registry import TOOL_REGISTRY, TOOL_SCHEMAS

__all__ = ["TOOL_REGISTRY", "TOOL_SCHEMAS"]
