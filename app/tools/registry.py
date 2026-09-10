"""Assembles every tool module into the registry the agent binds to."""
from app.tools.availability import GET_AVAILABILITY_SCHEMA, get_availability
from app.tools.echo import ECHO_SCHEMA, echo
from app.tools.menu import GET_MENU_SCHEMA, get_menu
from app.tools.orders import (
    CREATE_ORDER_SCHEMA,
    VIEW_ORDERS_SCHEMA,
    create_order,
    view_orders,
)
from app.tools.preferences import GET_USER_PREFERENCES_SCHEMA, get_user_preferences
from app.tools.reservations import (
    CREATE_RESERVATION_SCHEMA,
    VIEW_RESERVATIONS_SCHEMA,
    create_reservation,
    view_reservations,
)

TOOL_REGISTRY = {
    "echo": echo,
    "get_menu": get_menu,
    "get_availability": get_availability,
    "create_reservation": create_reservation,
    "view_reservations": view_reservations,
    "create_order": create_order,
    "view_orders": view_orders,
    "get_user_preferences": get_user_preferences,
}

TOOL_SCHEMAS = [
    ECHO_SCHEMA,
    GET_MENU_SCHEMA,
    GET_AVAILABILITY_SCHEMA,
    CREATE_RESERVATION_SCHEMA,
    VIEW_RESERVATIONS_SCHEMA,
    CREATE_ORDER_SCHEMA,
    VIEW_ORDERS_SCHEMA,
    GET_USER_PREFERENCES_SCHEMA,
]
