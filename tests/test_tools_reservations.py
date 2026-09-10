from unittest.mock import patch

from app.tools import reservations

RESERVATION_OUT = {
    "id": 5,
    "customer_id": 1,
    "table_id": 3,
    "slot_datetime": "2026-09-10T19:00:00",
    "party_size": 4,
    "special_requests": None,
    "status": "confirmed",
    "created_at": "2026-09-09T06:54:23.630791",
}


def test_create_reservation_attaches_customer_id_from_the_caller_not_the_model():
    with patch.object(reservations.backend_api, "create_reservation", return_value=RESERVATION_OUT) as backend:
        reservations.create_reservation(
            customer_id=1, table_id=3, slot_datetime="2026-09-10T19:00:00", party_size=4
        )

    sent_payload = backend.call_args.args[0]
    assert sent_payload["customer_id"] == 1


def test_view_reservations_fetches_one_by_id():
    with patch.object(reservations.backend_api, "get_reservation", return_value=RESERVATION_OUT) as get_one:
        result = reservations.view_reservations(customer_id=1, reservation_id=5)

    get_one.assert_called_once_with(5)
    assert result == [RESERVATION_OUT]


def test_view_reservations_falls_back_to_customer_history():
    with patch.object(
        reservations.backend_api, "list_customer_reservations", return_value=[RESERVATION_OUT]
    ) as list_all:
        result = reservations.view_reservations(customer_id=1)

    list_all.assert_called_once_with(1, status=None)
    assert result == [RESERVATION_OUT]
