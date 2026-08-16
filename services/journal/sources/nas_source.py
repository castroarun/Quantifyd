"""
Read completed NAS-family strangle trades into the journal.

Covers ALL live/paper NAS books, not just the original OTM one:
  NIFTY  — NAS OTM / ATM / ATM2 / ATM4  and the 09:16 one-shot variants
  SENSEX — ATM / ATM2 / ATM4

Each database holds one system; the table is `nas_trades` (OTM books) or
`nas_atm_trades` (ATM books). Every row is a strangle round-trip with both
legs aggregated, so one journal row = one strangle cycle.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any, Dict, List

from config import DATA_DIR

from ..metrics import hold_minutes

logger = logging.getLogger(__name__)

# (strategy label, db filename, table, venue)
NAS_BOOKS = [
    ('NAS-OTM',        'nas_trading.db',            'nas_trades',     'NIFTY'),
    ('NAS-ATM',        'nas_atm_trading.db',        'nas_atm_trades', 'NIFTY'),
    ('NAS-ATM2',       'nas_atm2_trading.db',       'nas_atm_trades', 'NIFTY'),
    ('NAS-ATM4',       'nas_atm4_trading.db',       'nas_atm_trades', 'NIFTY'),
    ('NAS-916-OTM',    'nas_916_otm_trading.db',    'nas_trades',     'NIFTY'),
    ('NAS-916-ATM',    'nas_916_atm_trading.db',    'nas_atm_trades', 'NIFTY'),
    ('NAS-916-ATM2',   'nas_916_atm2_trading.db',   'nas_atm_trades', 'NIFTY'),
    ('NAS-916-ATM4',   'nas_916_atm4_trading.db',   'nas_atm_trades', 'NIFTY'),
    ('SENSEX-ATM',     'sensex_atm_trading.db',     'nas_atm_trades', 'SENSEX'),
    ('SENSEX-ATM2',    'sensex_atm2_trading.db',    'nas_atm_trades', 'SENSEX'),
    ('SENSEX-ATM4',    'sensex_atm4_trading.db',    'nas_atm_trades', 'SENSEX'),
]

LOT_SIZE = {'NIFTY': 65, 'SENSEX': 20}


def _num(v, default=0.0) -> float:
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _rows_from(db_file: str, table: str) -> List[sqlite3.Row]:
    path = DATA_DIR / db_file
    if not path.exists():
        return []
    try:
        c = sqlite3.connect(str(path))
        c.row_factory = sqlite3.Row
        try:
            rows = c.execute(
                f'SELECT * FROM {table} WHERE exit_time IS NOT NULL '
                f'ORDER BY entry_time'
            ).fetchall()
        except sqlite3.OperationalError:
            rows = []
        c.close()
        return rows
    except Exception as e:
        logger.warning('%s read failed: %s', db_file, e)
        return []


def fetch_closed_trades() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for strategy, db_file, table, venue in NAS_BOOKS:
        rows = _rows_from(db_file, table)
        lot = LOT_SIZE.get(venue, 65)
        for r in rows:
            d = dict(r)
            lots = int(_num(d.get('lots'), 0) or 0)
            qty = lots * lot

            entry_prem = _num(d.get('call_entry_premium')) + _num(d.get('put_entry_premium'))
            exit_prem = _num(d.get('call_exit_premium')) + _num(d.get('put_exit_premium'))
            gross = _num(d.get('gross_pnl'))
            net = _num(d.get('net_pnl'), gross)

            entry_time = d.get('entry_time')
            exit_time = d.get('exit_time')

            # Short strangle: premium collected at entry, paid back at exit.
            # Risk proxy = premium collected (used only for R-multiple).
            risk = entry_prem * qty if qty else 0.0
            r_mult = (net / risk) if risk else None

            strikes = []
            if d.get('call_strike'):
                strikes.append(f"{int(_num(d['call_strike']))}CE")
            if d.get('put_strike'):
                strikes.append(f"{int(_num(d['put_strike']))}PE")
            instrument = f"{venue} {'/'.join(strikes)}" if strikes else venue

            out.append({
                'source_db': db_file.replace('.db', ''),
                'source_table': table,
                'source_id': str(d.get('id')),
                'strategy': strategy,
                'instrument': instrument,
                'instrument_type': 'OPTIONS',
                'direction': 'SHORT',
                'qty': qty,
                'entry_price': round(entry_prem, 2),
                'entry_time': entry_time,
                'exit_price': round(exit_prem, 2),
                'exit_time': exit_time,
                'exit_reason': d.get('exit_reason'),
                'pnl_gross': round(gross, 2),
                'pnl_charges': round(gross - net, 2),
                'pnl_net': round(net, 2),
                'r_multiple': round(r_mult, 3) if r_mult is not None else None,
                'initial_risk': round(risk, 2) if risk else None,
                'hold_minutes': hold_minutes(entry_time, exit_time),
                'mode': 'LIVE',
                'grade': None,
                'mistake_flag': 0,
            })
    logger.info('[journal.nas] %d closed trades across %d books',
                len(out), len(NAS_BOOKS))
    return out
