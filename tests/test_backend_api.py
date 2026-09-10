from unittest.mock import MagicMock, patch

import pytest
import requests

from app import backend_api
from app.errors import BackendValidationError


@pytest.fixture
def get():
    with patch.object(backend_api.requests, "get") as get:
        yield get


@pytest.fixture
def no_sleep():
    with patch.object(backend_api.time, "sleep") as sleep:
        yield sleep


def _response(status_code: int, json_body=None) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = json_body if json_body is not None else {}
    response.text = ""
    if status_code >= 500:
        response.raise_for_status.side_effect = requests.HTTPError(f"http {status_code}")
    else:
        response.raise_for_status.side_effect = None
    return response


def test_get_menu_returns_parsed_json_on_success(get, no_sleep):
    get.return_value = _response(200, [{"id": 1, "name": "Soup"}])

    result = backend_api.get_menu({"category": "starters"})

    assert result == [{"id": 1, "name": "Soup"}]
    get.assert_called_once()
    no_sleep.assert_not_called()


def test_get_menu_retries_transient_connection_error_then_succeeds(get, no_sleep):
    get.side_effect = [requests.ConnectionError("down"), _response(200, [])]

    result = backend_api.get_menu({})

    assert result == []
    assert get.call_count == 2
    no_sleep.assert_called_once_with(1)


def test_get_menu_retries_server_error_then_raises_after_cap(get, no_sleep):
    get.return_value = _response(500)

    with pytest.raises(requests.HTTPError):
        backend_api.get_menu({})

    from app.config import settings

    assert get.call_count == settings.max_retries + 1


def test_get_menu_client_error_is_not_retried(get, no_sleep):
    get.return_value = _response(400, {"detail": "max_price must be positive"})

    with pytest.raises(BackendValidationError, match="max_price must be positive"):
        backend_api.get_menu({"max_price": -1})

    get.assert_called_once()
    no_sleep.assert_not_called()


@pytest.fixture
def post():
    with patch.object(backend_api.requests, "post") as post:
        yield post


def test_create_reservation_returns_parsed_json_on_success(post, no_sleep):
    post.return_value = _response(201, {"id": 5, "status": "confirmed"})

    result = backend_api.create_reservation({"customer_id": 1, "table_id": 3})

    assert result == {"id": 5, "status": "confirmed"}
    post.assert_called_once()


def test_create_reservation_surfaces_business_rule_failure(post, no_sleep):
    post.return_value = _response(409, {"detail": "slot no longer available"})

    with pytest.raises(BackendValidationError, match="slot no longer available"):
        backend_api.create_reservation({"customer_id": 1, "table_id": 3})

    post.assert_called_once()
    no_sleep.assert_not_called()


def test_lookup_customer_returns_none_on_404(get, no_sleep):
    get.return_value = _response(404, {"detail": "Customer not found"})

    result = backend_api.lookup_customer(email="nobody@example.com")

    assert result is None
    no_sleep.assert_not_called()


def test_lookup_customer_returns_match(get, no_sleep):
    get.return_value = _response(200, {"id": 1, "name": "Priya Sharma"})

    result = backend_api.lookup_customer(email="priya.sharma@example.com")

    assert result == {"id": 1, "name": "Priya Sharma"}
