from unittest.mock import patch

from app import memory


def test_read_episodes_returns_empty_list_for_unknown_customer(tmp_path):
    with patch.object(memory, "MEMORY_DIR", tmp_path):
        assert memory.read_episodes("no-such-customer") == []


def test_append_then_read_roundtrips(tmp_path):
    with patch.object(memory, "MEMORY_DIR", tmp_path):
        memory.append_episode("123", "Booked table 4 for 2 at 2026-09-10T19:00:00")
        memory.append_episode("123", "Ordered 2x menu item #7 (reservation 9)")

        episodes = memory.read_episodes("123")

    assert len(episodes) == 2
    assert "Booked table 4" in episodes[0]
    assert "Ordered 2x" in episodes[1]


def test_read_episodes_respects_limit(tmp_path):
    with patch.object(memory, "MEMORY_DIR", tmp_path):
        for i in range(10):
            memory.append_episode("123", f"episode {i}")

        episodes = memory.read_episodes("123", limit=3)

    assert len(episodes) == 3
    assert "episode 9" in episodes[-1]  # most recent, last


def test_append_episode_swallows_a_filesystem_failure(tmp_path):
    """A memory-write failure is a side effect going wrong, not a reason to
    fail the booking/order that already succeeded — see app/memory.py."""
    with patch.object(memory, "MEMORY_DIR", tmp_path / "unwritable"):
        with patch("pathlib.Path.mkdir", side_effect=OSError("disk full")):
            memory.append_episode("123", "should not raise")  # no exception
