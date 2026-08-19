"""Book liveness — a READ-ONLY projection over each book's own trade record.

Answers, per system, the question the pages cannot: is this thing actually
trading? Returns last trade date, days idle, trades in the last 30 days,
cumulative net, win rate and a short cumulative series for a sparkline.

Reads only. It never imports an engine, never writes, and holds no state, so it
cannot affect live or paper trading (standing rule, .claude/CLAUDE.md).
Every source is wrapped: a broken or missing store yields nulls, never an error.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Blueprint, jsonify

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
BD = ROOT / 'backtest_data'

book_liveness_bp = Blueprint('book_liveness', __name__, url_prefix='/api/books')

# (db file, table, date columns to try, pnl columns to try, optional WHERE)
SQLITE_SOURCES: Dict[str, List[tuple]] = {
    'orb-cash':        [('orb_trading.db', 'orb_positions', ('exit_time', 'entry_time'), ('pnl_inr',), "status='CLOSED'")],
    'n500m':           [('n500m_trading.db', 'n500m_positions', ('exit_time', 'entry_time'), ('pnl_inr',), "status!='OPEN'")],
    'i75wr':           [('intraday_75wr.db', 'i75_positions', ('exit_time', 'entry_time'), ('pnl_inr',), "status='CLOSED'")],
    'pairs':           [('pair_trading.db', 'pair_trades', ('exit_date', 'entry_date'), ('pnl', 'net_pnl', 'pnl_inr'), None)],
    'mst':             [('mst_trading.db', 'mst_positions', ('exit_time', 'entry_time'), ('pnl_inr',), "status='CLOSED'")],
    'breakout-paper':  [('breakout_paper.db', 'bp_closed', ('exit_date',), ('net_pnl',), None)],
    'momentum-3l':     [('momentum_paper.db', 'mp_closed', ('exit_date',), ('net_pnl',), None)],
    'ha-paper':        [('ha_paper.db', 'hap_fills', ('ts',), ('pnl',), "pnl IS NOT NULL")],
    'fnoms-paper':     [('fnoms_paper.db', 'fms_fills', ('ts',), ('pnl',), "pnl IS NOT NULL")],
    'ohol-paper':      [('ohol_paper.db', 'ohp_fills', ('ts',), ('pnl',), "pnl IS NOT NULL")],
    'orb-paper':       [('orb_paper.db', 'obp_fills', ('ts',), ('pnl',), "pnl IS NOT NULL")],
    'kc6':             [('kc6_trading.db', 'kc6_trades', ('exit_date',), ('pnl_abs',), None)],
    'orb-index':       [('strangle_trading.db', 'strangle_trades', ('exit_date', 'entry_date'), ('net_pnl', 'pnl', 'pnl_inr'), None)],
    # NAS family — one row per strangle cycle, aggregated across every book DB
    'nas-nifty': [
        ('nas_trading.db', 'nas_trades', ('exit_time',), ('net_pnl', 'gross_pnl'), 'exit_time IS NOT NULL'),
        ('nas_atm_trading.db', 'nas_atm_trades', ('exit_time',), ('net_pnl', 'gross_pnl'), 'exit_time IS NOT NULL'),
        ('nas_atm2_trading.db', 'nas_atm_trades', ('exit_time',), ('net_pnl', 'gross_pnl'), 'exit_time IS NOT NULL'),
        ('nas_atm4_trading.db', 'nas_atm_trades', ('exit_time',), ('net_pnl', 'gross_pnl'), 'exit_time IS NOT NULL'),
        ('nas_916_otm_trading.db', 'nas_trades', ('exit_time',), ('net_pnl', 'gross_pnl'), 'exit_time IS NOT NULL'),
        ('nas_916_atm_trading.db', 'nas_atm_trades', ('exit_time',), ('net_pnl', 'gross_pnl'), 'exit_time IS NOT NULL'),
        ('nas_916_atm2_trading.db', 'nas_atm_trades', ('exit_time',), ('net_pnl', 'gross_pnl'), 'exit_time IS NOT NULL'),
        ('nas_916_atm4_trading.db', 'nas_atm_trades', ('exit_time',), ('net_pnl', 'gross_pnl'), 'exit_time IS NOT NULL'),
    ],
    'nas-sensex': [
        ('sensex_atm_trading.db', 'nas_atm_trades', ('exit_time',), ('net_pnl', 'gross_pnl'), 'exit_time IS NOT NULL'),
        ('sensex_atm2_trading.db', 'nas_atm_trades', ('exit_time',), ('net_pnl', 'gross_pnl'), 'exit_time IS NOT NULL'),
        ('sensex_atm4_trading.db', 'nas_atm_trades', ('exit_time',), ('net_pnl', 'gross_pnl'), 'exit_time IS NOT NULL'),
    ],
}

# JSON-state books: (file, extractor name)
JSON_SOURCES = {
    'nwv': 'nwv_trade_paper.json',
}


def _cols(conn: sqlite3.Connection, table: str) -> List[str]:
    return [r[1] for r in conn.execute(f'PRAGMA table_info({table})')]


def _pick(available: List[str], wanted) -> Optional[str]:
    for w in wanted:
        if w in available:
            return w
    return None


def _read_sqlite(spec: tuple) -> List[tuple]:
    """-> [(date_str, pnl_float)] for one (db, table) source."""
    dbfile, table, datecols, pnlcols, where = spec
    path = BD / dbfile
    if not path.exists():
        return []
    out: List[tuple] = []
    try:
        conn = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
        try:
            available = _cols(conn, table)
            if not available:
                return []
            dcol = _pick(available, datecols)
            pcol = _pick(available, pnlcols)
            if not dcol:
                return []
            sel = f'SELECT substr({dcol},1,10) d, {pcol if pcol else "NULL"} p FROM {table}'
            clauses = [f'{dcol} IS NOT NULL']
            if where:
                clauses.append(where)
            sel += ' WHERE ' + ' AND '.join(clauses) + f' ORDER BY {dcol}'
            for d, p in conn.execute(sel):
                if d:
                    out.append((d, float(p) if p is not None else None))
        finally:
            conn.close()
    except Exception as e:
        logger.debug('[liveness] %s/%s read failed: %s', dbfile, table, e)
    return out


def _read_nwv() -> List[tuple]:
    path = BD / JSON_SOURCES['nwv']
    if not path.exists():
        return []
    try:
        hist = json.loads(path.read_text()).get('history') or []
        out = []
        for h in hist:
            if h.get('reason') == 'SKIP_IGNORE':
                continue          # a skipped week is not a trade
            out.append((str(h.get('week'))[:10], float(h.get('net_rs') or 0)))
        return sorted(out)
    except Exception as e:
        logger.debug('[liveness] nwv read failed: %s', e)
        return []


def _summarise(rows: List[tuple], today: datetime) -> Dict[str, Any]:
    rows = sorted([r for r in rows if r[0]], key=lambda r: r[0])
    if not rows:
        return {'trades': 0, 'last_trade': None, 'days_idle': None, 'trades_30d': 0,
                'net_total': None, 'net_30d': None, 'win_rate': None, 'series': []}
    cut = (today - timedelta(days=30)).strftime('%Y-%m-%d')
    last = rows[-1][0]
    try:
        days_idle = (today.date() - datetime.strptime(last, '%Y-%m-%d').date()).days
    except Exception:
        days_idle = None
    priced = [r for r in rows if r[1] is not None]
    wins = sum(1 for _, p in priced if p > 0)
    recent = [r for r in rows if r[0] >= cut]
    cum, series = 0.0, []
    for d, p in priced[-60:]:
        cum += p
        series.append({'d': d, 'c': round(cum, 1)})
    return {
        'trades': len(rows),
        'last_trade': last,
        'days_idle': days_idle,
        'trades_30d': len(recent),
        'net_total': round(sum(p for _, p in priced), 1) if priced else None,
        'net_30d': round(sum(p for d, p in priced if d >= cut), 1) if priced else None,
        'win_rate': round(100.0 * wins / len(priced), 1) if priced else None,
        'series': series,
    }


def compute_liveness() -> Dict[str, Any]:
    today = datetime.now()
    books: Dict[str, Any] = {}
    for book_id, specs in SQLITE_SOURCES.items():
        rows: List[tuple] = []
        for spec in specs:
            rows.extend(_read_sqlite(spec))
        books[book_id] = _summarise(rows, today)
    books['nwv'] = _summarise(_read_nwv(), today)
    return {'generated_at': today.isoformat(timespec='seconds'), 'books': books}


@book_liveness_bp.route('/liveness', methods=['GET'])
def api_book_liveness():
    try:
        return jsonify(compute_liveness())
    except Exception as e:
        logger.error('[liveness] failed: %s', e, exc_info=True)
        return jsonify({'error': str(e), 'books': {}}), 500
