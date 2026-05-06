"""
Read completed NAS strangle trades from backtest_data/nas_trading.db.
Each row is a strangle round-trip (CE + PE legs aggregated).
"""

from __future__ import annotations
import sqlite3
import logging
from typing import List, Dict, Any

from config import DATA_DIR
from ..metrics import hold_minutes

logger = logging.getLogger(__name__)

NAS_DB = DATA_DIR / 'nas_trading.db'


def fetch_closed_trades() -> List[Dict[str, Any]]:
    if not NAS_DB.exists():
        return []
    out: List[Dict[str, Any]] = []
    try:
        c = sqlite3.connect(str(NAS_DB))
        c.row_factory = sqlite3.Row
        try:
            rows = c.execute('SELECT * FROM nas_trades ORDER BY entry_time').fetchall()
        except sqlite3.OperationalError:
            c.close()
            return []
        c.close()
    except Exception as e:
        logger.warning('nas_trading.db read failed: %s', e)
        return []

    for r in rows:
        d = dict(r)
        # NAS is a short-strangle on options — direction = SHORT
        direction = 'SHORT'
        # Aggregate qty in lots (we don't have per-share precision here)
        lots = d.get('lots') or 0
        # Treat NAS as a single row; use total premium values for entry/exit
        ce_entry = d.get('call_entry_premium') or 0
        pe_entry = d.get('put_entry_premium') or 0
        ce_exit = d.get('call_exit_premium') or 0
        pe_exit = d.get('put_exit_premium') or 0
        entry_price = ce_entry + pe_entry
        exit_price = ce_exit + pe_exit
        gross = d.get('gross_pnl') or 0
        net = d.get('net_pnl') or gross
        charges = (gross - net) if (net is not None and gross is not None) else 0

        out.append({
            'source_db': 'nas_trading',
            'source_table': 'nas_trades',
            'source_id': d['id'],
            'strategy': 'NAS',
            'instrument': f'NIFTY {d.get("call_strike", "")}/{d.get("put_strike", "")}',
            'instrument_type': 'OPTION',
            'direction': direction,
            'qty': lots,
            'entry_price': round(entry_price, 2),
            'entry_time': d.get('entry_time'),
            'exit_price': round(exit_price, 2),
            'exit_time': d.get('exit_time'),
            'exit_reason': d.get('exit_reason'),
            'pnl_gross': round(float(gross), 2),
            'pnl_charges': round(float(charges), 2),
            'pnl_net': round(float(net), 2),
            'r_multiple': None,
            'initial_risk': None,
            'hold_minutes': hold_minutes(d.get('entry_time'), d.get('exit_time')),
            'mode': 'PAPER',  # NAS variants are all paper as of 2026-05
        })
    return out
