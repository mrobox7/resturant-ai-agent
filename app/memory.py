"""Episodic memory: a short, human-readable log per customer, appended
whenever a booking or order completes. Read back at the start of a new
session (see context_gate_node in app/agents/nodes.py) so the agent has
light continuity across sessions without that history living in the hot
prompt path — only a handful of summary lines are ever loaded, into
AgentState.recent_episodes, not the full file.
"""
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

MEMORY_DIR = Path(__file__).resolve().parent.parent / "memory" / "episodes"


def _episode_file(customer_id: str) -> Path:
    return MEMORY_DIR / f"{customer_id}.md"


def append_episode(customer_id: str, summary: str) -> None:
    """Append one dated line. Never raises — a memory-write failure is a
    side effect going wrong, not a reason to fail the booking/order that
    already succeeded against the real backend."""
    try:
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        with _episode_file(customer_id).open("a", encoding="utf-8") as f:
            f.write(f"- {timestamp}: {summary}\n")
    except OSError as exc:
        logger.warning("Failed to write episodic memory for %s: %s", customer_id, exc)


def read_episodes(customer_id: str, limit: int = 5) -> list[str]:
    """Return up to the `limit` most recent episode lines, oldest first."""
    path = _episode_file(customer_id)
    if not path.exists():
        return []
    try:
        lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except OSError as exc:
        logger.warning("Failed to read episodic memory for %s: %s", customer_id, exc)
        return []
    return lines[-limit:]
