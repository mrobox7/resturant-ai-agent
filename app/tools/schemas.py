"""Pydantic schemas for every tool: what the model may ask for, and what
comes back. Validating both ends means a malformed filter never reaches the
network, and a malformed backend response never reaches the model unnoticed.

Each *Input model is what the model extracts — it deliberately excludes
customer_id, which is always attached by code after extraction, never asked
of the model. Each *Out model mirrors the backend's own response shape
(confirmed against its live OpenAPI schema), so a shape drift there surfaces
as a validation error here rather than a silent mismatch.
"""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Location = Literal["indoor", "outdoor"]
ReservationStatus = Literal["confirmed", "cancelled", "completed"]


# --- menu --------------------------------------------------------------


class MenuQuery(BaseModel):
    """What the model may filter the menu by. Mirrors GET /menu's query params."""

    category: str | None = None
    tags_any: list[str] | None = None
    tags_all: list[str] | None = None
    exclude_tags: list[str] | None = None
    max_price: float | None = Field(default=None, gt=0)
    available_only: bool = True


class MenuItem(BaseModel):
    """One menu item, as returned by GET /menu."""

    id: int
    name: str
    category: str
    price: float
    tags: list[str] = Field(default_factory=list)
    available: bool
    description: str | None = None


# --- availability --------------------------------------------------------


class AvailabilityQuery(BaseModel):
    """Mirrors GET /availability's query params."""

    slot_datetime: datetime
    party_size: int = Field(gt=0)
    location: Location | None = None


class AvailableTable(BaseModel):
    """One free table, as returned inside GET /availability's response."""

    table_id: int
    table_number: int
    capacity: int
    location: str


# --- reservations --------------------------------------------------------


class ReservationCreateInput(BaseModel):
    """What the model extracts for a booking. customer_id is attached by
    code afterward — it is never a field the model is asked to produce."""

    table_id: int
    slot_datetime: datetime
    party_size: int = Field(gt=0)
    special_requests: str | None = None


class ReservationOut(BaseModel):
    """Mirrors ReservationOut from the backend's OpenAPI schema."""

    id: int
    customer_id: int
    table_id: int
    slot_datetime: datetime
    party_size: int
    special_requests: str | None = None
    status: ReservationStatus
    created_at: datetime


class ViewReservationsQuery(BaseModel):
    """What the model extracts to look up existing bookings. customer_id is
    attached by code — a specific reservation_id, if given, overrides it."""

    reservation_id: int | None = None
    status: ReservationStatus | None = None


# --- orders ----------------------------------------------------------------


class OrderCreateInput(BaseModel):
    """What the model extracts to place an order. The model names the dish
    by its menu text, not a numeric id — resolving the name to menu_item_id
    is the tool's job, done against a real GET /menu call, not a guess.
    reservation_id is optional: if omitted, the tool resolves it deterministically
    when the customer has exactly one confirmed reservation."""

    menu_item_name: str
    quantity: int = Field(default=1, gt=0)
    reservation_id: int | None = None


class OrderOut(BaseModel):
    """Mirrors OrderOut from the backend's OpenAPI schema."""

    id: int
    reservation_id: int
    menu_item_id: int
    quantity: int
    created_at: datetime


class ViewOrdersQuery(BaseModel):
    """What the model extracts to look up existing orders. customer_id is
    attached by code — a specific reservation_id, if given, scopes the
    lookup to that reservation instead of the customer's whole history."""

    reservation_id: int | None = None


# --- customer preferences ---------------------------------------------


class CustomerPreferencesOut(BaseModel):
    """The redacted view of GET /customers/{id} that may reach the agent's
    state/LLM context — phone and email never leave backend_api.py."""

    name: str
    preferences: dict = Field(default_factory=dict)
