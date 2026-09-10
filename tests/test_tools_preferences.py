from unittest.mock import patch

from app.tools import preferences


def test_get_user_preferences_redacts_phone_and_email():
    """The concrete PII-redaction test: even if the backend record carries
    phone/email, they must never survive into what reaches agent state."""
    full_record = {
        "id": 1,
        "name": "Priya Sharma",
        "phone": "+91-9876543210",
        "email": "priya.sharma@example.com",
        "preferences": {"seating": "outdoor", "dietary": ["vegetarian"]},
        "created_at": "2026-09-09T06:54:23.630791",
    }
    with patch.object(preferences.backend_api, "get_customer", return_value=full_record):
        result = preferences.get_user_preferences(customer_id=1)

    assert result == {"name": "Priya Sharma", "preferences": {"seating": "outdoor", "dietary": ["vegetarian"]}}
    assert "phone" not in result
    assert "email" not in result
