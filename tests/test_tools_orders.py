from unittest.mock import patch

import pytest

from app.tools import orders

MENU = [
    {"id": 1, "name": "Paneer Tikka", "category": "starter", "price": 350.0, "tags": [], "available": True},
    {"id": 5, "name": "Dal Makhani", "category": "main", "price": 400.0, "tags": [], "available": True},
]

ORDER_OUT = {"id": 1, "reservation_id": 9, "menu_item_id": 1, "quantity": 2, "created_at": "2026-09-09T06:54:23.630791"}

CONFIRMED_RESERVATION = {
    "id": 9, "customer_id": 1, "table_id": 3, "slot_datetime": "2026-09-10T19:00:00",
    "party_size": 4, "special_requests": None, "status": "confirmed",
    "created_at": "2026-09-09T06:54:23.630791",
}


def test_create_order_resolves_dish_name_case_insensitively():
    with (
        patch.object(orders, "get_menu", return_value=MENU),
        patch.object(orders.backend_api, "create_order", return_value=ORDER_OUT) as backend,
    ):
        orders.create_order(customer_id=1, menu_item_name="paneer tikka", quantity=2, reservation_id=9)

    backend.assert_called_once_with(9, {"menu_item_id": 1, "quantity": 2})


def test_create_order_rejects_a_dish_not_on_the_menu():
    with patch.object(orders, "get_menu", return_value=MENU):
        with pytest.raises(ValueError, match="not on the menu"):
            orders.create_order(customer_id=1, menu_item_name="Pizza", reservation_id=9)


def test_create_order_auto_resolves_the_one_confirmed_reservation():
    with (
        patch.object(orders, "get_menu", return_value=MENU),
        patch.object(orders, "view_reservations", return_value=[CONFIRMED_RESERVATION]),
        patch.object(orders.backend_api, "create_order", return_value=ORDER_OUT) as backend,
    ):
        orders.create_order(customer_id=1, menu_item_name="Dal Makhani")

    backend.assert_called_once_with(9, {"menu_item_id": 5, "quantity": 1})


def test_create_order_asks_when_reservation_is_ambiguous():
    with (
        patch.object(orders, "get_menu", return_value=MENU),
        patch.object(orders, "view_reservations", return_value=[CONFIRMED_RESERVATION, {**CONFIRMED_RESERVATION, "id": 10}]),
    ):
        with pytest.raises(ValueError, match="multiple confirmed reservations"):
            orders.create_order(customer_id=1, menu_item_name="Dal Makhani")
