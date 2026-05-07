"""Pull a snapshot of `market_data.db` from VPS → laptop for offline dev.

The VPS at 94.136.185.54 is canonical for `market_data.db`. The laptop is
dev-only. Run this script when you want a fresh copy of the VPS DB on the
laptop (for example, to backtest new strategy logic locally before pushing
to VPS).

Safe to run any time. Uses scp over SSH key auth. Refuses to overwrite a
laptop DB that's *newer* than the VPS DB (which would mean someone violated
the no-laptop-writes rule — manual intervention needed).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCAL_DB = ROOT / "backtest_data" / "market_data.db"
VPS_USER = "arun"
VPS_HOST = "94.136.185.54"
VPS_DB = "/home/arun/quantifyd/backtest_data/market_data.db"


def _vps_db_size() -> int:
    """Return VPS DB byte size, or -1 on failure."""
    try:
        out = subprocess.check_output(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
             f"{VPS_USER}@{VPS_HOST}", f"stat -c %s {VPS_DB}"],
            stderr=subprocess.STDOUT, text=True, timeout=30,
        )
        return int(out.strip())
    except Exception as e:
        print(f"[pull] could not stat VPS DB: {e}")
        return -1


def main() -> int:
    LOCAL_DB.parent.mkdir(parents=True, exist_ok=True)

    vps_size = _vps_db_size()
    if vps_size < 0:
        print("[pull] aborting — could not contact VPS. Check SSH key access.")
        return 2

    if LOCAL_DB.exists():
        local_size = LOCAL_DB.stat().st_size
        print(f"[pull] local DB exists: {local_size:,} bytes")
        print(f"[pull] VPS DB:           {vps_size:,} bytes")
        if local_size > vps_size:
            print(
                "[pull] WARNING — local DB is LARGER than VPS DB. This means\n"
                "       the laptop has data the VPS doesn't, which violates\n"
                "       the VPS-canonical rule (set 2026-05-07).\n"
                "       Investigate before overwriting. Aborting.\n"
                "       To override: rm -f the local DB first, then re-run."
            )
            return 3
    else:
        print(f"[pull] no local DB yet — fetching {vps_size:,} bytes from VPS")

    if os.getenv("DRY_RUN"):
        print("[pull] DRY_RUN — would scp now. Skipping.")
        return 0

    cmd = ["scp", "-C", "-o", "ConnectTimeout=30",
           f"{VPS_USER}@{VPS_HOST}:{VPS_DB}",
           str(LOCAL_DB) + ".tmp"]
    print(f"[pull] running: {' '.join(cmd)}")
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"[pull] scp failed with rc={rc}")
        return rc

    tmp = Path(str(LOCAL_DB) + ".tmp")
    if not tmp.exists():
        print("[pull] scp completed but .tmp file missing — odd")
        return 4

    # Atomic-ish swap (Windows-safe — replace removes existing first)
    if LOCAL_DB.exists():
        LOCAL_DB.unlink()
    tmp.replace(LOCAL_DB)
    print(f"[pull] success — {LOCAL_DB.stat().st_size:,} bytes at {LOCAL_DB}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
