"""NAS Kill-Switch — persistent panic state

Single source of truth for "all NAS trading is halted". Survives Flask
restarts via a sentinel file at backtest_data/nas_kill.flag. When the
flag is present:
  * /api/nas/panic-status reports `killed: true`
  * Each variant's _check_guardrails returns ('System killed', False)
  * Flask startup re-applies enabled=False to all 8 NAS *_DEFAULTS
  * Re-enabling requires hitting /api/nas/resume (or deleting the file)

The intent is single-shot resilience: even a VPS reboot mid-trip will
NOT silently resume trading. To resume, the user (or Claude after the
trip) explicitly clears the flag.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

KILL_FLAG_PATH = Path(__file__).resolve().parents[1] / "backtest_data" / "nas_kill.flag"


def is_killed() -> bool:
    """True when the NAS panic kill is active."""
    return KILL_FLAG_PATH.exists()


def set_killed(reason: str = "manual panic", source: str = "api") -> dict:
    """Create the kill flag. Idempotent — re-arming just updates the timestamp."""
    KILL_FLAG_PATH.parent.mkdir(parents=True, exist_ok=True)
    info = {
        "killed_at": datetime.now().isoformat(timespec="seconds"),
        "reason": reason,
        "source": source,
    }
    try:
        KILL_FLAG_PATH.write_text(json.dumps(info, indent=2))
        logger.warning(f"[NAS-KILL] Flag SET — reason={reason!r} source={source!r}")
    except Exception as e:
        logger.error(f"[NAS-KILL] Failed to write kill flag: {e}")
        raise
    return info


def clear_killed() -> bool:
    """Remove the kill flag. Returns True if a flag was actually cleared."""
    if not KILL_FLAG_PATH.exists():
        return False
    try:
        KILL_FLAG_PATH.unlink()
        logger.warning("[NAS-KILL] Flag CLEARED — NAS is allowed to trade again")
        return True
    except Exception as e:
        logger.error(f"[NAS-KILL] Failed to clear kill flag: {e}")
        raise


def get_kill_info() -> Optional[dict]:
    """Return the kill flag content (timestamp + reason) or None if not killed."""
    if not KILL_FLAG_PATH.exists():
        return None
    try:
        return json.loads(KILL_FLAG_PATH.read_text())
    except Exception as e:
        logger.error(f"[NAS-KILL] Flag exists but unreadable: {e}")
        return {"killed_at": "unknown", "reason": "flag-file-unreadable", "source": "unknown"}
