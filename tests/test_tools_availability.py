from unittest.mock import patch

import pytest
from pydantic import ValidationError

from app.tools import availability


def test_get_availability_rejects_non_positive_party_size_before_backend_call():
    with patch.object(availability.backend_api, "get_availability") as backend:
        with pytest.raises(ValidationError):
            availability.get_availability(slot_datetime="2026-09-10T19:00:00", party_size=0)

    backend.assert_not_called()


def test_get_availability_validates_and_returns_tables():
    raw = {
        "slot_datetime": "2026-09-10T19:00:00",
        "party_size": 4,
        "available_tables": [
            {"table_id": 3, "table_number": 3, "capacity": 4, "location": "indoor"},
        ],
    }
    with patch.object(availability.backend_api, "get_availability", return_value=raw):
        result = availability.get_availability(slot_datetime="2026-09-10T19:00:00", party_size=4)

    assert result["available_tables"] == [
        {"table_id": 3, "table_number": 3, "capacity": 4, "location": "indoor"}
    ]
