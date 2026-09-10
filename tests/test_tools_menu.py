from unittest.mock import patch

import pytest
from pydantic import ValidationError

from app.tools import menu


@pytest.fixture
def backend():
    with patch.object(menu.backend_api, "get_menu") as backend:
        yield backend


# --- input validation: deterministic, no network call ----------------------


def test_get_menu_rejects_invalid_max_price_before_calling_backend(backend):
    with pytest.raises(ValidationError):
        menu.get_menu(max_price=-5)

    backend.assert_not_called()


# --- happy path --------------------------------------------------------


def test_get_menu_validates_and_returns_backend_items(backend):
    backend.return_value = [
        {
            "id": 1,
            "name": "Paneer Tikka",
            "category": "starters",
            "price": 8.5,
            "tags": ["vegetarian"],
            "available": True,
        }
    ]

    result = menu.get_menu(category="starters", available_only=True)

    assert result == [
        {
            "id": 1,
            "name": "Paneer Tikka",
            "category": "starters",
            "price": 8.5,
            "tags": ["vegetarian"],
            "available": True,
            "description": None,
        }
    ]
    backend.assert_called_once_with({"category": "starters", "available_only": True})


def test_get_menu_raises_on_malformed_backend_item(backend):
    backend.return_value = [{"id": "not-an-int", "name": "x"}]

    with pytest.raises(ValidationError):
        menu.get_menu()
