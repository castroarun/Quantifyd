#!/usr/bin/env python3
"""Dump per-system NAS MTM snapshots -> static/nas_mtm.json.

Runs every minute during market hours via cron, OUT-OF-PROCESS so the
NAS dashboard can read the curves WITHOUT requiring a gunicorn restart
to register the /api/nas/mtm route. Same shape as that endpoint, so the
React frontend can read either source.

Reads only — never writes to NAS DBs.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Frontend-key -> path-to-db (mirrors _NAS_MTM_KEY_MAP + nas_eod_report SYSTEMS).
DBS = {
    'nas':           'nas_trading.db',
    'nas-atm':       'nas_atm_trading.db',
    'nas-atm2':      'nas_atm2_trading.db',
    'nas-atm4':      'nas_atm4_trading.db',
    'nas-916-otm':   'nas_916_otm_trading.db',
    'nas-916-atm':   'nas_916_atm_trading.db',
    'nas-916-atm2':  'nas_916_atm2_trading.db',
    'nas-916-atm4':  'nas_916_atm4_trading.db',
}

OUT = ROOT / 'static' / 'nas_mtm.json'


def _today_curve(db_path: Path) -> list:
    if not db_path.exists():
        return []
    try:
        c = sqlite3.connect(str(db_path), timeout=5)
        c.row_factory = sqlite3.Row
        try:
            # nas_mtm_snapshots is created lazily by services.nas_mtm — until
            # the first snapshot, the table doesn't exist.
            tbl = c.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='nas_mtm_snapshots'"
            ).fetchone()
            if not tbl:
                return []
            today = date.today().isoformat()
            rows = c.execute(
                "SELECT ts, day_pnl FROM nas_mtm_snapshots "
                "WHERE snap_date=? ORDER BY id", (today,)
            ).fetchall()
            return [[r['ts'], r['day_pnl']] for r in rows]
        finally:
            c.close()
    except Exception:
        return []


def main() -> int:
    systems = {}
    for key, fname in DBS.items():
        pts = _today_curve(ROOT / 'backtest_data' / fname)
        systems[key] = {
            'points': pts,
            'last': pts[-1][1] if pts else 0.0,
            'n': len(pts),
        }
    payload = {
        'generated_at': __import__('datetime').datetime.now().isoformat(
            timespec='seconds'),
        'systems': systems,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(payload, separators=(',', ':')))
    tmp.replace(OUT)  # atomic on POSIX
    return 0


if __name__ == '__main__':
    sys.exit(main())
