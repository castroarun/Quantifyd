"""
Read completed strangle trades from backtest_data/strangle_trading.db.
ORB-Index strangles — projected as one row per round-trip.
"""

from __future__ import annotations
import sqlite3
import logging
from typing import List, Dict, Any

from config import DATA_DIR
from ..metrics import hold_minutes

logger = logging.getLogger(__name__)

STRANGLE_DB = DATA_DIR / 'strangle_trading.db'


def fetch_closed_trades() -> List[Dict[str, Any]]:
    if not STRANGLE_DB.exists():
        return []
    out: List[Dict[str, Any]] = []
    try:
        c = sqlite3.connect(str(STRANGLE_DB))
        c.row_factory = sqlite3.Row
        try:
            rows = c.execute('SELECT * FROM strangle_trades ORDER BY entry_ts').fetchall()
        except sqlite3.OperationalError:
            c.close()
            return []
        c.close()
    except Exception as e:
        logger.warning('strangle_trading.db read failed: %s', e)
        return []

    for r in rows:
        d = dict(r)
        direction = (d.get('direction') or 'LONG').upper()
        ce_entry = d.get('ce_entry') or 0
        pe_entry = d.get('pe_entry') or 0
        ce_exit = d.get('ce_exit') or 0
        pe_exit = d.get('pe_exit') or 0
        gross = d.get('gross_pnl') or 0
        net = d.get('net_pnl') or 0
        costs = d.get('costs') or max(0, gross - net)
        out.append({
            'source_db': 'strangle_trading',
            'source_table': 'strangle_trades',
            'source_id': d['id'],
            'strategy': f'ORB-INDEX-{(d.get("variant_id") or "").upper()}',
            'instrument': f'NIFTY {d.get("ce_strike", "")}/{d.get("pe_strike", "")}',
            'instrument_type': 'OPTION',
            'direction': direction,
            'qty': 1,
            'entry_price': round(ce_entry + pe_entry, 2),
            'entry_time': d.get('entry_ts'),
            'exit_price': round(ce_exit + pe_exit, 2),
            'exit_time': d.get('exit_ts'),
            'exit_reason': d.get('exit_reason'),
            'pnl_gross': round(float(gross), 2),
            'pnl_charges': round(float(costs), 2),
            'pnl_net': round(float(net), 2),
            'r_multiple': None,
            'initial_risk': None,
            'hold_minutes': d.get('hold_minutes') or hold_minutes(d.get('entry_ts'), d.get('exit_ts')),
            'mode': 'PAPER',
        })
    return out
