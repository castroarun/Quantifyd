#!/usr/bin/env python3
"""Dump per-system NAS MTM snapshots + today's trade events -> static/nas_mtm.json.

Runs every minute during market hours via cron, OUT-OF-PROCESS so the
NAS dashboard reads live curves WITHOUT a gunicorn restart.

The dumper RECOMPUTES `realized` at each snapshot's timestamp using the
same per-leg formula the dashboard uses (sum (entry-exit)*qty across
closed legs with exit_time<=ts). This keeps historical snapshots in
sync with the card's Day P&L — and reconciles even when individual
SL_HIT closes occur inside an open strangle (which nas_trades.net_pnl
alone doesn't capture).

Events: pulled from the per-system orders table (nas_orders /
nas_atm_orders) and classified into entry / adjust / sl_hit / exit for
chart annotations.

Reads only.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DBS = [
    ('nas',           'nas_trading.db',         'nas_orders'),
    ('nas-atm',       'nas_atm_trading.db',     'nas_atm_orders'),
    ('nas-atm2',      'nas_atm2_trading.db',    'nas_atm_orders'),
    ('nas-atm4',      'nas_atm4_trading.db',    'nas_atm_orders'),
    ('nas-916-otm',   'nas_916_otm_trading.db', 'nas_orders'),
    ('nas-916-atm',   'nas_916_atm_trading.db', 'nas_atm_orders'),
    ('nas-916-atm2',  'nas_916_atm2_trading.db','nas_atm_orders'),
    ('nas-916-atm4',  'nas_916_atm4_trading.db','nas_atm_orders'),
]

OUT = ROOT / 'static' / 'nas_mtm.json'


def _classify(signal_type: str | None, tx: str | None,
              exit_reason: str | None = None) -> tuple[str, str]:
    """Return (event_type, label) for chart annotation."""
    s = (signal_type or '').upper()
    r = (exit_reason or '').upper()
    if 'SL_HIT' in s or 'SL_HIT' in r:
        return ('sl_hit', 'SL hit')
    if 'EOD' in s or 'TIME_EXIT' in s or 'EOD' in r or 'TIME' in r:
        return ('exit', 'Exit')
    if 'ROLL' in s or 'ADJ' in s or 'ROLL' in r or 'BOUNDARY' in r:
        return ('adjust', 'Adjust')
    if (tx or '').upper() == 'SELL':
        return ('entry', 'Entry')
    return ('exit', 'Close')


def _table_exists(c, name: str) -> bool:
    return c.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,)).fetchone() is not None


def _system_payload(db_path: Path, orders_table: str) -> dict:
    today = date.today().isoformat()
    if not db_path.exists():
        return {'points': [], 'last': 0.0, 'n': 0, 'events': []}
    try:
        c = sqlite3.connect(str(db_path), timeout=5)
        c.row_factory = sqlite3.Row
        try:
            # 1) today's closed legs — fold into a sorted exit-time series
            legs: list[tuple[str, float]] = []
            for p in c.execute(
                "SELECT exit_time, entry_price, exit_price, qty, "
                "transaction_type FROM nas_positions "
                "WHERE status='CLOSED' AND created_at >= ?", (today,)):
                xt = p['exit_time']
                ep = p['entry_price']; xp = p['exit_price']; q = p['qty'] or 0
                if not xt or ep is None or xp is None or not q:
                    continue
                side = (p['transaction_type'] or 'SELL').upper()
                leg_pnl = (ep - xp) * q if side == 'SELL' else (xp - ep) * q
                legs.append((str(xt), float(leg_pnl)))
            legs.sort(key=lambda x: x[0])

            def realized_at(ts: str) -> float:
                tot = 0.0
                for xt, pl in legs:
                    if xt <= ts:
                        tot += pl
                    else:
                        break
                return tot

            # 2) today's snapshots — recompute day_pnl = realized@ts + stored_unrealized
            points: list[list] = []
            if _table_exists(c, 'nas_mtm_snapshots'):
                for r in c.execute(
                    "SELECT ts, unrealized FROM nas_mtm_snapshots "
                    "WHERE snap_date=? ORDER BY id", (today,)):
                    u = float(r['unrealized'] or 0)
                    dp = realized_at(str(r['ts'])) + u
                    points.append([r['ts'], round(dp, 2)])

            # 3) events from orders + closed-leg exit reasons
            events = []
            if _table_exists(c, orders_table):
                for r in c.execute(
                    f"SELECT created_at, signal_type, transaction_type, price, "
                    f"tradingsymbol FROM {orders_table} "
                    f"WHERE date(created_at) = date('now','localtime') "
                    f"ORDER BY id"):
                    et, lab = _classify(r['signal_type'], r['transaction_type'])
                    events.append({
                        'ts': r['created_at'], 'type': et, 'label': lab,
                        'sig': r['signal_type'], 'sym': r['tradingsymbol'],
                        'tx': r['transaction_type'], 'price': r['price'],
                    })
            # add SL_HIT exits from positions if not present in orders
            for p in c.execute(
                "SELECT exit_time, exit_reason, tradingsymbol FROM nas_positions "
                "WHERE status='CLOSED' AND created_at >= ? AND exit_reason='SL_HIT'",
                (today,)):
                if not p['exit_time']:
                    continue
                ts = str(p['exit_time'])
                if not any(e['ts'] == ts and e['type'] == 'sl_hit' for e in events):
                    events.append({'ts': ts, 'type': 'sl_hit', 'label': 'SL hit',
                                   'sig': 'SL_HIT', 'sym': p['tradingsymbol'],
                                   'tx': None, 'price': None})
            events.sort(key=lambda e: e['ts'])

            last = points[-1][1] if points else 0.0
            return {'points': points, 'last': last, 'n': len(points),
                    'events': events}
        finally:
            c.close()
    except Exception:
        return {'points': [], 'last': 0.0, 'n': 0, 'events': []}


def main() -> int:
    systems = {}
    for key, fname, orders_tbl in DBS:
        systems[key] = _system_payload(ROOT / 'backtest_data' / fname,
                                       orders_tbl)
    payload = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'systems': systems,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(payload, separators=(',', ':')))
    tmp.replace(OUT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
