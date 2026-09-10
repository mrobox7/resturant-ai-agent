"""The boundary between the agent and whatever actually holds the data.

Every call the tools make to a database, HTTP service, or filesystem belongs
here. Authorization belongs here too — never in tools.py, and never in a
prompt, because anything the model can see it can also be talked out of.
Callers pass the identity down; this layer decides what that identity may do.
"""
import logging
import time

import requests

from app.config import settings
from app.errors import BackendValidationError

logger = logging.getLogger(__name__)


def fetch(payload: str, requesting_user_id: str | None = None) -> dict:
    """Stub for a real backend read. Replace with the problem's data source."""
    # A real implementation authorizes first and raises/returns empty on denial:
    #     if not _may_read(requesting_user_id, payload):
    #         raise PermissionError(...)
    return {"payload": payload, "requested_by": requesting_user_id}


def _request(
    method: str,
    path: str,
    *,
    params: dict | None = None,
    json: dict | None = None,
    not_found_ok: bool = False,
):
    """Shared retry/backoff policy for every call to the restaurant backend.

    Transient failures (connect error, timeout, 5xx) are retried with
    exponential backoff up to settings.max_retries. A 4xx is a deterministic
    failure given the input and is never retried; the backend's own `detail`
    message (e.g. "slot no longer available", "menu item not found") is
    surfaced in the raised error instead of a generic HTTP status string,
    since that's exactly what the agent needs to relay to the user.
    """
    url = f"{settings.backend_api_base_url}{path}"
    call = getattr(requests, method)

    for attempt in range(settings.max_retries + 1):
        try:
            response = call(url, params=params, json=json, timeout=settings.backend_api_timeout)
        except (requests.ConnectionError, requests.Timeout) as exc:
            if attempt == settings.max_retries:
                raise
            logger.warning(
                "%s %s transient failure (attempt %d): %s", method.upper(), path, attempt, exc
            )
            time.sleep(2**attempt)
            continue

        if response.status_code >= 500:
            if attempt == settings.max_retries:
                response.raise_for_status()
            logger.warning(
                "%s %s got %d (attempt %d), retrying",
                method.upper(), path, response.status_code, attempt,
            )
            time.sleep(2**attempt)
            continue

        if response.status_code == 404 and not_found_ok:
            return None

        if response.status_code >= 400:
            try:
                parsed = response.json()
            except ValueError:
                parsed = None
            detail = parsed.get("detail") if isinstance(parsed, dict) else None
            raise BackendValidationError(str(detail or response.text or response.status_code))

        return response.json() if response.content else {}


def get_menu(params: dict) -> list[dict]:
    """GET /menu — a public read, no identity involved."""
    return _request("get", "/menu", params=params)


def get_availability(params: dict) -> dict:
    """GET /availability — a public read, no identity involved."""
    return _request("get", "/availability", params=params)


def create_reservation(payload: dict) -> dict:
    """POST /reservations. The backend validates the slot is actually free."""
    return _request("post", "/reservations", json=payload)


def get_reservation(reservation_id: int) -> dict:
    """GET /reservations/{id}."""
    return _request("get", f"/reservations/{reservation_id}")


def list_customer_reservations(customer_id: int, status: str | None = None) -> list[dict]:
    """GET /customers/{id}/reservations."""
    params = {"status": status} if status else {}
    return _request("get", f"/customers/{customer_id}/reservations", params=params)


def create_order(reservation_id: int, payload: dict) -> dict:
    """POST /reservations/{id}/orders. The backend validates the menu item exists."""
    return _request("post", f"/reservations/{reservation_id}/orders", json=payload)


def list_reservation_orders(reservation_id: int) -> list[dict]:
    """GET /reservations/{id}/orders."""
    return _request("get", f"/reservations/{reservation_id}/orders")


def list_customer_orders(customer_id: int) -> list[dict]:
    """GET /customers/{id}/orders — across all of the customer's reservations."""
    return _request("get", f"/customers/{customer_id}/orders")


def lookup_customer(phone: str | None = None, email: str | None = None) -> dict | None:
    """GET /customers/lookup. Returns None on a 404 — "not found" is an expected outcome here, not a failure."""
    params = {k: v for k, v in {"phone": phone, "email": email}.items() if v}
    return _request("get", "/customers/lookup", params=params, not_found_ok=True)


def create_customer(payload: dict) -> dict:
    """POST /customers."""
    return _request("post", "/customers", json=payload)


def get_customer(customer_id: int) -> dict:
    """GET /customers/{id}. Includes the full record — redaction is the tool layer's job, not this one's."""
    return _request("get", f"/customers/{customer_id}")
