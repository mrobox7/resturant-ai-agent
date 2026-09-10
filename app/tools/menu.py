from app import backend_api
from app.tools.schemas import MenuItem, MenuQuery

# The backend's actual category/tag vocabulary (confirmed against the live
# GET /menu response). Listed as `enum` in the tool schema below so the model
# extracts one of these exact strings instead of a plausible-sounding guess
# like "starters" for "starter" — a free-text field here is silently wrong,
# not an error, since a mismatched filter just returns zero results.
CATEGORIES = ["starter", "main", "dessert"]
TAGS = ["vegetarian", "vegan", "contains-nuts", "contains-dairy"]


def get_menu(**kwargs) -> list[dict]:
    """Fetch menu items from the restaurant backend, filtered by the given criteria."""
    query = MenuQuery(**kwargs)  # deterministic validation, before any network call
    raw_items = backend_api.get_menu(query.model_dump(exclude_none=True))
    return [MenuItem(**item).model_dump() for item in raw_items]


GET_MENU_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_menu",
        "description": (
            "List menu items at the restaurant, optionally filtered by category, "
            "dietary tags, price, or availability. Use this to answer questions "
            "about what's on the menu, prices, or dietary options."
        ),
        "parameters": {
            "type": "object",
            # Every property here is optional, and every type is a
            # [type, "null"] union: some providers (Groq's tool-call
            # validator, observed live) fill an unused optional property
            # with an explicit null rather than omitting it, and reject the
            # call outright if the schema's type doesn't allow that.
            # _extract_args strips nulls back out before use.
            "properties": {
                "category": {
                    "type": ["string", "null"],
                    "enum": CATEGORIES,
                    "description": "Restrict to one category.",
                },
                "tags_any": {
                    "type": ["array", "null"],
                    "items": {"type": "string", "enum": TAGS},
                    "description": "Include items matching at least one of these tags.",
                },
                "tags_all": {
                    "type": ["array", "null"],
                    "items": {"type": "string", "enum": TAGS},
                    "description": "Include only items matching all of these tags.",
                },
                "exclude_tags": {
                    "type": ["array", "null"],
                    "items": {"type": "string", "enum": TAGS},
                    "description": "Exclude items matching any of these tags (e.g. an allergy).",
                },
                "max_price": {
                    "type": ["number", "null"],
                    "description": "Only include items priced at or below this amount.",
                },
                "available_only": {
                    "type": ["boolean", "null"],
                    "description": "If true (default), only include items currently available.",
                },
            },
            "required": [],
        },
    },
}
