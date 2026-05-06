"""
Read closed ORB cash-equity positions from backtest_data/orb_trading.db
and project them into journal_trades rows.
"""

from __future__ import annotations
import sqlite3
import logging
from pathlib import Path
from typing import List, Dict, Any

from config import DATA_DIR
from ..charges import compute_charges
from ..metrics import hold_minutes, gross_pnl, initial_risk, r_multiple

logger = logging.getLogger(__name__)

ORB_DB = DATA_DIR / 'orb_trading.db'


def fetch_closed_trades() -> List[Dict[str, Any]]:
    """Return list of journal-shaped trade dicts from orb_positions WHERE
    status='CLOSED'.
    """
    if not ORB_DB.exists():
        return []
    out: List[Dict[str, Any]] = []
    try:
        c = sqlite3.connect(str(ORB_DB))
        c.row_factory = sqlite3.Row
        try:
            rows = c.execute(
                """
                SELECT * FROM orb_positions
                WHERE status='CLOSED' AND exit_time IS NOT NULL
                ORDER BY entry_time
                """
            ).fetchall()
        except sqlite3.OperationalError as e:
            logger.warning('orb_positions read failed: %s', e)
            c.close()
            return []
        c.close()
    except Exception as e:
        logger.warning('orb_trading.db open failed: %s', e)
        return []

    for r in rows:
        d = dict(r)
        direction = d.get('direction') or 'LONG'
        qty = d.get('qty') or 0
        entry_price = d.get('entry_price') or 0.0
        exit_price = d.get('exit_price')
        if exit_price is None:
            continue
        # Gross P&L: prefer the value the engine recorded; recompute as fallback.
        pnl_pts = d.get('pnl_pts')
        gross = d.get('pnl_inr')
        if gross is None:
            gross = gross_pnl(direction, qty, entry_price, exit_price)
        # Charges
        charges = compute_charges('EQUITY', direction, qty, entry_price, exit_price, product='MIS')
        net = (gross or 0) - charges.total
        # R-multiple — ORB SL is at OR edge, recorded in sl_price
        risk = initial_risk(direction, qty, entry_price, d.get('sl_price'))
        r = r_multiple(net, risk)
        hold = hold_minutes(d.get('entry_time'), d.get('exit_time'))

        out.append({
            'source_db': 'orb_trading',
            'source_table': 'orb_positions',
            'source_id': d['id'],
            'strategy': 'ORB-CASH',
            'instrument': d.get('instrument') or '',
            'instrument_type': 'EQUITY',
            'direction': direction,
            'qty': qty,
            'entry_price': entry_price,
            'entry_time': d.get('entry_time'),
            'exit_price': exit_price,
            'exit_time': d.get('exit_time'),
            'exit_reason': d.get('exit_reason'),
            'pnl_gross': round(float(gross or 0), 2),
            'pnl_charges': round(charges.total, 2),
            'pnl_net': round(float(net), 2),
            'r_multiple': r,
            'initial_risk': round(risk, 2) if risk is not None else None,
            'hold_minutes': hold,
            'mode': 'LIVE',
        })
    return out
