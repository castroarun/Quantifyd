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
import re
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

# Books that keep positions and closed trades in a JSON state file rather than SQLite.
# Both write the same trade record: symbol/qty/buy/sell/entry_date/exit_date/net_pnl.
JSON_BOOKS: Dict[str, str] = {
    'oa-real':   'oa_real_state.json',
    'ipo-paper': 'ipo_paper_state.json',
}


def _read_json_book(book_id: str):
    """-> (closed rows [(date, pnl)], open_count, last_entry_date)."""
    fname = JSON_BOOKS.get(book_id)
    if not fname:
        return [], 0, None
    path = BD / fname
    if not path.exists():
        return [], 0, None
    try:
        st = json.loads(path.read_text(encoding='utf-8'))
    except Exception as e:
        logger.debug('[liveness] %s unreadable: %s', fname, e)
        return [], 0, None
    rows = []
    for tr in (st.get('trades') or []):
        d = (tr.get('exit_date') or tr.get('entry_date') or '')[:10]
        if not d:
            continue
        try:
            rows.append((d, float(tr.get('net_pnl') or 0)))
        except (TypeError, ValueError):
            continue
    pos = st.get('positions') or []
    last_entry = max((p.get('entry_date') or '')[:10] for p in pos) if pos else None
    return rows, len(pos), (last_entry or None)


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


def _age(day: Optional[str], today: datetime) -> Optional[int]:
    """Days between a YYYY-MM-DD string and today; None if unparseable."""
    if not day:
        return None
    try:
        return (today.date() - datetime.strptime(day[:10], '%Y-%m-%d').date()).days
    except Exception:
        return None


# Books that hold positions between trades. (db, table, entry-date col, open-WHERE)
# A None WHERE means the table only ever holds open rows — closed positions are
# moved out to a *_closed / *_fills store by the engine that owns them.
OPEN_SOURCES: Dict[str, tuple] = {
    'orb-cash':       ('orb_trading.db', 'orb_positions', 'entry_time', "status='OPEN'"),
    'n500m':          ('n500m_trading.db', 'n500m_positions', 'entry_time', "status='OPEN'"),
    'i75wr':          ('intraday_75wr.db', 'i75_positions', 'entry_time', "status='OPEN'"),
    'pairs':          ('pair_trading.db', 'pair_positions', 'entry_date', "status='OPEN'"),
    'mst':            ('mst_trading.db', 'mst_positions', 'entry_time', "status='OPEN'"),
    'kc6':            ('kc6_trading.db', 'kc6_positions', 'entry_date', "status='OPEN'"),
    'orb-index':      ('strangle_trading.db', 'strangle_positions', 'entry_date', "status='OPEN'"),
    'breakout-paper': ('breakout_paper.db', 'bp_positions', 'entry_date', None),
    'momentum-3l':    ('momentum_paper.db', 'mp_positions', 'entry_date', None),
    'ha-paper':       ('ha_paper.db', 'hap_positions', 'entry_time', None),
    'fnoms-paper':    ('fnoms_paper.db', 'fms_positions', 'entry_date', None),
    'ohol-paper':     ('ohol_paper.db', 'ohp_positions', 'entry_time', None),
    'orb-paper':      ('orb_paper.db', 'obp_positions', 'entry_time', None),
}


def _read_open(book_id: str) -> tuple:
    """-> (open_count, last_entry_date) for one book; (0, None) if unknown."""
    spec = OPEN_SOURCES.get(book_id)
    if not spec:
        return 0, None
    dbfile, table, dcol, where = spec
    path = BD / dbfile
    if not path.exists():
        return 0, None
    try:
        conn = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
        try:
            available = _cols(conn, table)
            if dcol not in available:
                return 0, None
            sql = f'SELECT COUNT(*), MAX(substr({dcol},1,10)) FROM {table}'
            if where and 'status' in available:
                sql += f' WHERE {where}'
            n, last = conn.execute(sql).fetchone()
            return int(n or 0), last
        finally:
            conn.close()
    except Exception as e:
        logger.debug('[liveness] open-position read failed for %s: %s', book_id, e)
        return 0, None


def _summarise(rows: List[tuple], today: datetime,
               open_n: int = 0, last_entry: Optional[str] = None) -> Dict[str, Any]:
    rows = sorted([r for r in rows if r[0]], key=lambda r: r[0])
    if not rows:
        # Nothing closed yet is not the same as nothing happening — a book with
        # an open position is working, it just has no exit to report.
        return {'trades': 0, 'last_trade': None, 'days_idle': None, 'trades_30d': 0,
                'net_total': None, 'net_30d': None, 'win_rate': None, 'series': [],
                'open_positions': open_n, 'last_entry': last_entry,
                'days_since_activity': _age(last_entry, today)}
    cut = (today - timedelta(days=30)).strftime('%Y-%m-%d')
    last = rows[-1][0]
    days_idle = _age(last, today)
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
        'open_positions': open_n,
        'last_entry': last_entry,
        'days_since_activity': _age(max([d for d in (last, last_entry) if d]), today),
    }


def compute_liveness() -> Dict[str, Any]:
    today = datetime.now()
    books: Dict[str, Any] = {}
    for book_id, specs in SQLITE_SOURCES.items():
        rows: List[tuple] = []
        for spec in specs:
            rows.extend(_read_sqlite(spec))
        books[book_id] = _summarise(rows, today, *_read_open(book_id))
    books['nwv'] = _summarise(_read_nwv(), today)
    for book_id in JSON_BOOKS:
        rows, n_open, last_entry = _read_json_book(book_id)
        books[book_id] = _summarise(rows, today, n_open, last_entry)
    return {'generated_at': today.isoformat(timespec='seconds'), 'books': books}


_ISO_DAY = re.compile(r'^\d{4}-\d{2}-\d{2}$')


def _chain(points):
    """Time-weighted cumulative return from [{d, nav, capital?}].

    Each day's return is measured against the previous day's NAV with that day's capital
    change backed out, then chained. A deposit therefore moves the line by zero, which is
    the whole point of comparing a funded book to an index.
    """
    out, cum = [], 1.0
    prev_nav, prev_cap = None, None
    for pt in points:
        raw = pt.get('d')
        if not isinstance(raw, str) or not _ISO_DAY.match(raw[:10]):
            continue                       # no usable date: never let it become the base
        d = raw[:10]
        try:
            nav = float(pt.get('nav') or 0)
        except (TypeError, ValueError):
            continue
        if nav <= 0:
            continue
        cap = pt.get('capital')
        cap = float(cap) if cap not in (None, '') else None
        if prev_nav:
            flow = (cap - prev_cap) if (cap is not None and prev_cap is not None) else 0.0
            cum *= (1.0 + (((nav - flow) / prev_nav) - 1.0))
        out.append({'d': d, 'r': round((cum - 1.0) * 100, 4), 'nav': round(nav)})
        prev_nav = nav
        prev_cap = cap if cap is not None else prev_cap
    return out


# The Momentum Portfolio, in the order the tabs sit in.
PORTFOLIO_BOOKS = [
    ('momentum-3l', 'True North', '#0F6E56'),
    ('oa-real',     'Open Alpha', '#7C3AED'),
    ('ipo-paper',   'IPO Base',   '#C2410C'),
]


def _raw_points(book_id: str):
    """-> [{d, nav, capital}] for one book, capital filled in where the book omits it.

    True North keeps its curve in SQLite; the other two keep it in JSON state. IPO Base
    records nav without capital, so its current capital stands for every day it has -
    which is right while capital has only been set once, and is what makes its first day
    read as funding rather than as a 100% gain.
    """
    if book_id == 'momentum-3l':
        db = BD / 'momentum_paper.db'
        if not db.exists():
            return []
        try:
            conn = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
            try:
                rows = conn.execute(
                    'SELECT d, nav, COALESCE(capital,0) FROM mp_nav '
                    'WHERE nav > 0 ORDER BY d').fetchall()
            finally:
                conn.close()
        except Exception as e:
            logger.debug('[curve] mp_nav unreadable: %s', e)
            return []
        return _fill_capital([{'d': str(d)[:10], 'nav': float(n),
                               'capital': (float(c) or None)} for d, n, c in rows])

    fname = JSON_BOOKS.get(book_id)
    if not fname or not (BD / fname).exists():
        return []
    try:
        st = json.loads((BD / fname).read_text(encoding='utf-8'))
    except Exception as e:
        logger.debug('[curve] %s unreadable: %s', fname, e)
        return []
    out = []
    for pt in (st.get('navcurve') or st.get('nav') or []):
        d = pt.get('d')
        if not isinstance(d, str) or not _ISO_DAY.match(d[:10]):
            continue
        cap = pt.get('capital')
        try:
            out.append({'d': d[:10], 'nav': float(pt.get('nav') or 0),
                        'capital': float(cap) if cap not in (None, '') else None})
        except (TypeError, ValueError):
            continue
    return _fill_capital(out)


def _fill_capital(points):
    """Fill the gaps in a book's capital column WITHOUT inventing any.

    Forward from the last known value (a day with no flow), backward from the first known
    value for the days before it, and the day's own NAV only if the book never records
    capital at all - which makes it enter at par and contribute zero return on the day it
    is funded.
    """
    known = [p['capital'] for p in points if p['capital'] is not None]
    first_known = known[0] if known else None
    last = None
    for p in points:
        if p['capital'] is not None:
            last = p['capital']
        elif last is not None:
            p['capital'] = last                     # no flow that day
        elif first_known is not None:
            p['capital'] = first_known              # before the first recorded figure
        else:
            p['capital'] = p['nav']                 # never recorded: enter at par ...
            last = p['capital']                     # ... once, then carry it forward
    return points


def _portfolio_points():
    """The three books summed per day, each carried forward over days it did not report.

    A book contributes nothing before its first point - nav 0 AND capital 0 - so the day
    it is funded appears as a flow, not as performance.
    """
    per = {bid: {p['d']: p for p in _raw_points(bid)} for bid, _l, _c in PORTFOLIO_BOOKS}
    days = sorted({d for m in per.values() for d in m})
    if not days:
        return []
    last = {bid: None for bid in per}
    out = []
    for d in days:
        nav = cap = 0.0
        for bid, m in per.items():
            if d in m:
                last[bid] = m[d]
            cur = last[bid]
            if cur is None:
                continue                       # not started: contributes nothing
            nav += cur['nav']
            cap += cur['capital'] or 0.0
        if nav > 0:
            out.append({'d': d, 'nav': nav, 'capital': cap or None})
    return out


def _as_index(curve):
    """A book's time-weighted curve as a base-100 series the chart can rebase by division."""
    return [{'d': pt['d'], 'c': round(100.0 * (1 + pt['r'] / 100.0), 6)} for pt in curve]


def _book_curve(book_id: str):
    """-> (inception, [{d, r, nav}]) for any book the portfolio can chart."""
    if book_id == 'momentum-3l':
        from services import momentum_paper as mp
        return (mp._get('inception') or '')[:10], mp.book_curve()
    fname = JSON_BOOKS.get(book_id)
    if not fname:
        return None, []
    path = BD / fname
    if not path.exists():
        return None, []
    try:
        st = json.loads(path.read_text(encoding='utf-8'))
    except Exception as e:
        logger.debug('[curve] %s unreadable: %s', fname, e)
        return None, []
    pts = st.get('navcurve') or st.get('nav') or []
    inception = (st.get('inception') or st.get('started') or
                 (pts[0].get('d') if pts else None))
    return (str(inception)[:10] if inception else None), _chain(pts)


@book_liveness_bp.route('/<book_id>/benchmarks', methods=['GET'])
def api_book_benchmarks(book_id):
    """This book's time-weighted curve plus the comparison indices.

    The index series comes from momentum_paper.benchmark_series(), which caches for an
    hour against daily bars that only change after the close.
    """
    try:
        extra = {}
        if book_id == 'portfolio':
            pts = _portfolio_points()
            curve = _chain(pts)
            inception = pts[0]['d'] if pts else None
            # each book beside the combined line, so one chart shows who is carrying it
            for bid, label, color in PORTFOLIO_BOOKS:
                _inc, bc = _book_curve(bid)
                if len(bc) >= 2:
                    extra[bid] = {'label': label, 'points': _as_index(bc),
                                  'color': color, 'on': False}
        else:
            inception, curve = _book_curve(book_id)
        series = {}
        if curve:
            try:
                from services import momentum_paper as mp
                series = mp.benchmark_series(curve[0]['d']) or {}
            except Exception as e:                      # a missing index must not 500 the page
                logger.warning('[curve] benchmark series failed for %s: %s', book_id, e)
        series.update(extra)
        return jsonify({'inception': inception, 'book': curve, 'series': series})
    except Exception as e:
        logger.error('[curve] failed for %s: %s', book_id, e, exc_info=True)
        return jsonify({'error': str(e), 'book': [], 'series': {}}), 500


@book_liveness_bp.route('/liveness', methods=['GET'])
def api_book_liveness():
    try:
        return jsonify(compute_liveness())
    except Exception as e:
        logger.error('[liveness] failed: %s', e, exc_info=True)
        return jsonify({'error': str(e), 'books': {}}), 500
