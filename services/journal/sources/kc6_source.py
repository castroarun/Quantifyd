"""
Read completed KC6 trades from backtest_data/kc6_trading.db.
KC6 is a swing/CNC mean-reversion system — charges treated as CNC.
"""

from __future__ import annotations
import sqlite3
import logging
from typing import List, Dict, Any

from config import DATA_DIR
from ..charges import compute_charges
from ..metrics import gross_pnl

logger = logging.getLogger(__name__)

KC6_DB = DATA_DIR / 'kc6_trading.db'


def fetch_closed_trades() -> List[Dict[str, Any]]:
    if not KC6_DB.exists():
        return []
    out: List[Dict[str, Any]] = []
    try:
        c = sqlite3.connect(str(KC6_DB))
        c.row_factory = sqlite3.Row
        rows = c.execute('SELECT * FROM kc6_trades ORDER BY entry_date').fetchall()
        c.close()
    except Exception as e:
        logger.warning('kc6_trading.db read failed: %s', e)
        return []

    for r in rows:
        d = dict(r)
        direction = 'LONG'
        qty = d.get('qty') or 0
        entry_price = d.get('entry_price') or 0
        exit_price = d.get('exit_price') or 0
        gross = d.get('pnl_abs') if d.get('pnl_abs') is not None else gross_pnl(direction, qty, entry_price, exit_price)
        charges = compute_charges('EQUITY', direction, qty, entry_price, exit_price, product='CNC')
        net = (gross or 0) - charges.total
        # KC6 hold in days; convert to minutes (approx 6.25h * 60 = 375 min/day)
        hold_days = d.get('hold_days') or 0
        hold_min = int(hold_days) * 375 if hold_days else None

        out.append({
            'source_db': 'kc6_trading',
            'source_table': 'kc6_trades',
            'source_id': d['id'],
            'strategy': 'KC6',
            'instrument': d.get('symbol') or '',
            'instrument_type': 'EQUITY',
            'direction': direction,
            'qty': qty,
            'entry_price': entry_price,
            'entry_time': d.get('entry_date'),  # date only — fine for sort
            'exit_price': exit_price,
            'exit_time': d.get('exit_date'),
            'exit_reason': d.get('exit_reason'),
            'pnl_gross': round(float(gross or 0), 2),
            'pnl_charges': round(charges.total, 2),
            'pnl_net': round(float(net), 2),
            'r_multiple': None,
            'initial_risk': None,
            'hold_minutes': hold_min,
            'mode': 'LIVE',
        })
    return out
