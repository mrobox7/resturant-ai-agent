"""Ordering tools. customer_id is attached by the caller, never extracted by
the model — same identity boundary as app/tools/reservations.py.
"""
from app import backend_api
from app.tools.menu import get_menu
from app.tools.reservations import view_reservations
from app.tools.schemas import OrderCreateInput, OrderOut, ViewOrdersQuery


def _resolve_menu_item_id(name: str) -> int:
    """The model names a dish by text; the backend needs its numeric id.
    Resolved against a real GET /menu call, not guessed."""
    items = get_menu()
    name_lower = name.strip().lower()
    for item in items:
        if item["name"].lower() == name_lower:
            return item["id"]
    for item in items:
        if name_lower in item["name"].lower():
            return item["id"]
    raise ValueError(f"'{name}' is not on the menu")


def _resolve_reservation_id(customer_id: int) -> int:
    """No reservation named: fine if there's exactly one confirmed booking to attach to."""
    reservations = view_reservations(customer_id=customer_id, status="confirmed")
    if not reservations:
        raise ValueError("no confirmed reservation found — book a table first")
    if len(reservations) > 1:
        ids = ", ".join(str(r["id"]) for r in reservations)
        raise ValueError(f"multiple confirmed reservations ({ids}) — specify which one")
    return reservations[0]["id"]


def create_order(*, customer_id: int, **kwargs) -> dict:
    """Add a menu item to a reservation's order."""
    fields = OrderCreateInput(**kwargs)  # deterministic validation, before any network call
    reservation_id = fields.reservation_id or _resolve_reservation_id(customer_id)
    menu_item_id = _resolve_menu_item_id(fields.menu_item_name)
    raw = backend_api.create_order(
        reservation_id, {"menu_item_id": menu_item_id, "quantity": fields.quantity}
    )
    # mode="json": datetimes as ISO strings, not Python objects — tool_results
    # is fed to json.dumps() in respond_node, which can't serialize a datetime.
    return OrderOut(**raw).model_dump(mode="json")


def view_orders(*, customer_id: int, **kwargs) -> list[dict]:
    """Look up a customer's orders — for one reservation, or across all of them."""
    query = ViewOrdersQuery(**kwargs)
    if query.reservation_id is not None:
        raw_items = backend_api.list_reservation_orders(query.reservation_id)
    else:
        raw_items = backend_api.list_customer_orders(customer_id)
    return [OrderOut(**item).model_dump(mode="json") for item in raw_items]


CREATE_ORDER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "create_order",
        "description": (
            "Add a menu item to the customer's order. Name the dish as it "
            "appears on the menu; the reservation it's attached to defaults "
            "to the customer's one confirmed booking if not specified."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "menu_item_name": {
                    "type": "string",
                    "description": "The dish name, as it appears on the menu.",
                },
                # [type, "null"]: optional properties must tolerate an
                # explicit null — some providers fill unused optional slots
                # that way rather than omitting them, and reject the call
                # otherwise. _extract_args strips nulls back out.
                "quantity": {
                    "type": ["integer", "null"],
                    "description": "How many of this item. Defaults to 1.",
                },
                "reservation_id": {
                    "type": ["integer", "null"],
                    "description": "Which reservation to attach this order to, if the customer has more than one active booking.",
                },
            },
            "required": ["menu_item_name"],
        },
    },
}

VIEW_ORDERS_SCHEMA = {
    "type": "function",
    "function": {
        "name": "view_orders",
        "description": "Look up the customer's order history, or the orders on one specific reservation.",
        "parameters": {
            "type": "object",
            "properties": {
                "reservation_id": {
                    "type": ["integer", "null"],
                    "description": "Restrict to orders on this reservation only.",
                },
            },
            "required": [],
        },
    },
}
