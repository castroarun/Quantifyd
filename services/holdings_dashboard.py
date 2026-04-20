"""
Holdings dashboard — data layer for the new /app/holdings page.

Caches (all under backtest_data/):
  - holdings_meta.db       per-symbol 52w hi/lo + 5d/20d moves, refreshed 06:30
  - holdings_events.db     upcoming corporate actions from NSE, refreshed 07:00
  - holdings_snapshots.db  post-close daily snapshot (16:00) for history view

Module entry points:
  - refresh_holdings_meta()         -> int
  - refresh_corporate_actions()     -> int
  - capture_daily_snapshot()        -> str iso date
  - get_digest()                    -> dict (live)  /api/holdings/digest
  - get_snapshot(date=None)         -> dict          /api/holdings/snapshot
  - list_snapshots()                -> list[dict]    /api/holdings/snapshots
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import urllib.request
from datetime import date, datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

META_DB = 'backtest_data/holdings_meta.db'
EVENTS_DB = 'backtest_data/holdings_events.db'
SNAPSHOTS_DB = 'backtest_data/holdings_snapshots.db'


META_SCHEMA = """
CREATE TABLE IF NOT EXISTS holdings_meta (
    tradingsymbol TEXT PRIMARY KEY,
    exchange TEXT,
    instrument_token INTEGER,
    last_close REAL, last_close_date TEXT,
    week52_high REAL, week52_high_date TEXT,
    week52_low REAL, week52_low_date TEXT,
    all_time_high REAL, all_time_high_date TEXT,
    change_5d_pct REAL, change_5d_abs REAL, change_5d_from_price REAL,
    change_20d_pct REAL, change_20d_abs REAL, change_20d_from_price REAL,
    refreshed_at TEXT
);
"""

EVENTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS holdings_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tradingsymbol TEXT NOT NULL,
    event_date TEXT NOT NULL,
    event_type TEXT NOT NULL,
    purpose TEXT, detail TEXT, record_date TEXT,
    fetched_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_holdings_events_sym_date
    ON holdings_events(tradingsymbol, event_date);
"""

SNAPSHOTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS holdings_snapshots (
    snap_date TEXT PRIMARY KEY,
    generated_at TEXT NOT NULL,
    summary_json TEXT NOT NULL,
    movers_today_json TEXT NOT NULL,
    extremes_json TEXT NOT NULL,
    holdings_json TEXT NOT NULL
);
"""

_lock = threading.Lock()


def _conn(path: str, schema: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    c = sqlite3.connect(path)
    c.row_factory = sqlite3.Row
    c.executescript(schema)
    return c


# ──── Kite source ────────────────────────────────────────────────

def _kite_holdings() -> list[dict]:
    from services.kite_service import get_kite
    kite = get_kite()
    raw = kite.holdings() or []
    out = []
    for h in raw:
        qty = (h.get('quantity') or 0) + (h.get('t1_quantity') or 0)
        if qty <= 0:
            continue
        out.append({
            'tradingsymbol': h.get('tradingsymbol'),
            'exchange': h.get('exchange', 'NSE'),
            'instrument_token': h.get('instrument_token'),
            'qty': qty,
            'avg_price': h.get('average_price'),
            'last_price': h.get('last_price'),
        })
    return out


def _ltps(symbols: list[str]) -> dict:
    from services.kite_service import get_kite
    kite = get_kite()
    try:
        q = kite.quote([f'NSE:{s}' for s in symbols])
    except Exception as e:
        logger.warning(f'[Holdings] quote fetch failed: {e}')
        return {}
    return {k.split(':', 1)[-1]: v for k, v in q.items()}


# ──── Meta refresh (nightly) ─────────────────────────────────────

def refresh_holdings_meta() -> int:
    from services.kite_service import get_kite
    import time as _t
    kite = get_kite()
    holdings = _kite_holdings()
    if not holdings:
        logger.info('[Holdings] no holdings, meta skipped')
        return 0

    today = date.today()
    from_dt = datetime.combine(today - timedelta(days=420), datetime.min.time())
    to_dt = datetime.combine(today, datetime.min.time())
    written = 0
    with _lock:
        c = _conn(META_DB, META_SCHEMA)
        try:
            for h in holdings:
                tok = h['instrument_token']
                sym = h['tradingsymbol']
                if not tok:
                    continue
                try:
                    candles = kite.historical_data(tok, from_dt, to_dt, 'day')
                except Exception as e:
                    logger.warning(f'[Holdings] daily fetch {sym}: {e}')
                    continue
                completed = []
                for x in candles or []:
                    d_ = x['date'].date() if hasattr(x['date'], 'date') else x['date']
                    if d_ < today:
                        completed.append(x)
                if not completed:
                    continue
                last = completed[-1]
                window = completed[-252:] if len(completed) >= 252 else completed
                hi = max(window, key=lambda x: x['high'])
                lo = min(window, key=lambda x: x['low'])
                ath = max(completed, key=lambda x: x['high'])

                def _d(x):
                    v = x['date']
                    return (v.date() if hasattr(v, 'date') else v).isoformat()

                def _chg(n):
                    if len(completed) < n + 1:
                        return None, None, None
                    prior = completed[-1 - n]['close']
                    if not prior:
                        return None, None, None
                    ab = last['close'] - prior
                    return round(ab / prior * 100, 3), round(ab, 2), round(prior, 2)

                c5p, c5a, c5f = _chg(5)
                c20p, c20a, c20f = _chg(20)
                c.execute(
                    '''INSERT INTO holdings_meta(
                        tradingsymbol, exchange, instrument_token,
                        last_close, last_close_date,
                        week52_high, week52_high_date,
                        week52_low, week52_low_date,
                        all_time_high, all_time_high_date,
                        change_5d_pct, change_5d_abs, change_5d_from_price,
                        change_20d_pct, change_20d_abs, change_20d_from_price,
                        refreshed_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(tradingsymbol) DO UPDATE SET
                        last_close=excluded.last_close,
                        last_close_date=excluded.last_close_date,
                        week52_high=excluded.week52_high,
                        week52_high_date=excluded.week52_high_date,
                        week52_low=excluded.week52_low,
                        week52_low_date=excluded.week52_low_date,
                        all_time_high=excluded.all_time_high,
                        all_time_high_date=excluded.all_time_high_date,
                        change_5d_pct=excluded.change_5d_pct,
                        change_5d_abs=excluded.change_5d_abs,
                        change_5d_from_price=excluded.change_5d_from_price,
                        change_20d_pct=excluded.change_20d_pct,
                        change_20d_abs=excluded.change_20d_abs,
                        change_20d_from_price=excluded.change_20d_from_price,
                        refreshed_at=excluded.refreshed_at
                    ''',
                    (sym, h['exchange'], tok,
                     round(last['close'], 2), _d(last),
                     round(hi['high'], 2), _d(hi),
                     round(lo['low'], 2), _d(lo),
                     round(ath['high'], 2), _d(ath),
                     c5p, c5a, c5f, c20p, c20a, c20f,
                     datetime.now().isoformat()))
                written += 1
                _t.sleep(0.35)
            c.commit()
        finally:
            c.close()
    logger.info(f'[Holdings] meta refresh: {written} symbols')
    return written


# ──── Corporate actions (nightly) ────────────────────────────────

def _nse_fetch(url: str) -> list:
    import http.cookiejar
    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    hdrs = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.nseindia.com/',
    }
    try:
        opener.open(urllib.request.Request('https://www.nseindia.com/', headers=hdrs),
                    timeout=20).read()
    except Exception as e:
        logger.warning(f'[Holdings] NSE prime failed: {e}')
    with opener.open(urllib.request.Request(url, headers=hdrs), timeout=25) as r:
        data = r.read()
    parsed = json.loads(data.decode('utf-8', errors='replace') or '{}')
    if isinstance(parsed, list):
        return parsed
    return parsed.get('data') or parsed.get('result') or []


def _parse_event(purpose: str) -> tuple[str, str]:
    if not purpose:
        return 'meeting', ''
    p = purpose.upper()
    if 'DIVIDEND' in p:
        import re
        m = re.search(r'Rs\.?\s*([\d.]+)', purpose, re.I)
        amt = m.group(1) if m else None
        return 'dividend', (f'Rs {amt}/share' if amt else purpose)
    if 'BONUS' in p:
        return 'bonus', purpose
    if 'SPLIT' in p or 'SUB-DIVISION' in p or 'SUB DIVISION' in p:
        return 'split', purpose
    if 'BUY' in p and 'BACK' in p:
        return 'buyback', purpose
    if 'RESULT' in p or 'AUDIT' in p or 'BOARD MEETING' in p:
        return 'results', purpose
    return 'meeting', purpose


def _norm_date(d: str) -> str:
    if not d:
        return ''
    for fmt in ('%d-%b-%Y', '%d-%m-%Y', '%Y-%m-%d'):
        try:
            return datetime.strptime(d, fmt).date().isoformat()
        except ValueError:
            pass
    return d


def refresh_corporate_actions(days_ahead: int = 45) -> int:
    holdings = _kite_holdings()
    symbols = {h['tradingsymbol'] for h in holdings}
    if not symbols:
        logger.info('[Holdings] no holdings, events skipped')
        return 0
    today = date.today()
    from_d = today.strftime('%d-%m-%Y')
    to_d = (today + timedelta(days=days_ahead)).strftime('%d-%m-%Y')
    url = (
        'https://www.nseindia.com/api/corporates-corporateActions?'
        f'index=equities&from_date={from_d}&to_date={to_d}'
    )
    try:
        rows = _nse_fetch(url)
    except Exception as e:
        logger.warning(f'[Holdings] NSE fetch failed: {e}')
        return 0
    written = 0
    with _lock:
        c = _conn(EVENTS_DB, EVENTS_SCHEMA)
        try:
            c.execute('DELETE FROM holdings_events WHERE event_date >= ?',
                      (today.isoformat(),))
            for r in rows:
                sym = r.get('symbol')
                if not sym or sym not in symbols:
                    continue
                purpose = r.get('subject') or r.get('purpose') or ''
                evt, detail = _parse_event(purpose)
                edate = _norm_date(r.get('exDate') or r.get('ex_date')
                                    or r.get('bcEndDate') or r.get('bcStartDate') or '')
                if not edate:
                    continue
                rdate = _norm_date(r.get('recDate') or '')
                c.execute(
                    '''INSERT INTO holdings_events(
                        tradingsymbol, event_date, event_type, purpose,
                        detail, record_date, fetched_at
                    ) VALUES(?,?,?,?,?,?,?)''',
                    (sym, edate, evt, purpose, detail, rdate,
                     datetime.now().isoformat()))
                written += 1
            c.commit()
        finally:
            c.close()
    logger.info(f'[Holdings] events refresh: {written}')
    return written


# ──── Digest helpers ─────────────────────────────────────────────

def _load_meta(symbols: list[str]) -> dict:
    with _lock:
        c = _conn(META_DB, META_SCHEMA)
        try:
            if not symbols:
                return {}
            ph = ','.join(['?'] * len(symbols))
            rows = c.execute(
                f'SELECT * FROM holdings_meta WHERE tradingsymbol IN ({ph})',
                symbols).fetchall()
            return {r['tradingsymbol']: dict(r) for r in rows}
        finally:
            c.close()


def _load_events(symbols: list[str], days_ahead: int = 30) -> list:
    with _lock:
        c = _conn(EVENTS_DB, EVENTS_SCHEMA)
        try:
            if not symbols:
                return []
            ph = ','.join(['?'] * len(symbols))
            today_s = date.today().isoformat()
            to_s = (date.today() + timedelta(days=days_ahead)).isoformat()
            rows = c.execute(
                f'''SELECT * FROM holdings_events
                    WHERE tradingsymbol IN ({ph})
                    AND event_date BETWEEN ? AND ?
                    ORDER BY event_date ASC, tradingsymbol ASC''',
                symbols + [today_s, to_s]).fetchall()
            return [dict(r) for r in rows]
        finally:
            c.close()


def _tag_hi(r: dict) -> Optional[str]:
    pa = r.get('pct_from_ath')
    p52 = r.get('pct_from_52h')
    if pa is not None and pa >= -0.5:
        return 'at_ath'
    if p52 is not None and p52 >= -1:
        return 'at_52h'
    if p52 is not None and p52 >= -3:
        return 'near_52h'
    return None


def _tag_lo(r: dict) -> Optional[str]:
    p52 = r.get('pct_from_52l')
    if p52 is not None and p52 <= 1:
        return 'at_52l'
    if p52 is not None and p52 <= 3:
        return 'near_52l'
    return None


def _enrich(holdings, ltps, meta) -> list:
    out = []
    for h in holdings:
        sym = h['tradingsymbol']
        qty = h['qty']
        avg = h['avg_price'] or 0
        q = ltps.get(sym, {})
        ltp = q.get('last_price') or h.get('last_price') or 0
        ohlc = q.get('ohlc', {}) or {}
        prev_close = ohlc.get('close') or 0
        day_pct = ((ltp - prev_close) / prev_close * 100) if prev_close else 0
        day_abs = (ltp - prev_close) * qty if prev_close else 0
        invested = avg * qty
        current = ltp * qty
        total_pnl = current - invested
        total_pct = (total_pnl / invested * 100) if invested else 0
        m = meta.get(sym, {})
        w52h = m.get('week52_high')
        w52l = m.get('week52_low')
        ath = m.get('all_time_high')
        p52h = ((ltp - w52h) / w52h * 100) if w52h else None
        p52l = ((ltp - w52l) / w52l * 100) if w52l else None
        pa = ((ltp - ath) / ath * 100) if ath else None
        out.append({
            'tradingsymbol': sym,
            'qty': qty,
            'avg_price': round(avg, 2),
            'ltp': round(ltp, 2),
            'prev_close': round(prev_close, 2) if prev_close else None,
            'day_pct': round(day_pct, 3),
            'day_pnl_inr': round(day_abs, 2),
            'invested': round(invested, 2),
            'current': round(current, 2),
            'total_pnl_inr': round(total_pnl, 2),
            'total_pnl_pct': round(total_pct, 2),
            'week52_high': w52h, 'week52_high_date': m.get('week52_high_date'),
            'week52_low': w52l, 'week52_low_date': m.get('week52_low_date'),
            'all_time_high': ath,
            'pct_from_52h': round(p52h, 2) if p52h is not None else None,
            'pct_from_52l': round(p52l, 2) if p52l is not None else None,
            'pct_from_ath': round(pa, 2) if pa is not None else None,
            'change_5d_pct': m.get('change_5d_pct'),
            'change_20d_pct': m.get('change_20d_pct'),
        })
    return out


def _movers(records, key: str):
    ranked = [r for r in records if r.get(key) is not None]
    gainers = sorted([r for r in ranked if r[key] > 0],
                     key=lambda r: r[key], reverse=True)[:5]
    losers = sorted([r for r in ranked if r[key] < 0],
                    key=lambda r: r[key])[:5]
    return {'gainers': gainers, 'losers': losers}


def _extremes(records):
    hi, lo = [], []
    for r in records:
        th = _tag_hi(r)
        if th:
            hi.append({**r, 'tag': th})
        tl = _tag_lo(r)
        if tl:
            lo.append({**r, 'tag': tl})
    hi.sort(key=lambda r: r['pct_from_52h'] or 0, reverse=True)
    lo.sort(key=lambda r: r['pct_from_52l'] or 0)
    return {'high': hi, 'low': lo}


def _summary(records):
    invested = sum(r['invested'] for r in records)
    current = sum(r['current'] for r in records)
    day_pnl = sum(r['day_pnl_inr'] for r in records)
    total_pnl = current - invested
    return {
        'count': len(records),
        'invested': round(invested, 2),
        'current': round(current, 2),
        'day_pnl': round(day_pnl, 2),
        'day_pct': round(day_pnl / current * 100, 3) if current else 0,
        'total_pnl': round(total_pnl, 2),
        'total_pct': round(total_pnl / invested * 100, 2) if invested else 0,
    }


def get_digest() -> dict:
    holdings = _kite_holdings()
    if not holdings:
        return {
            'summary': _summary([]),
            'holdings': [],
            'movers_today': {'gainers': [], 'losers': []},
            'movers_weekly': {'gainers': [], 'losers': []},
            'extremes': {'high': [], 'low': []},
            'events': [], 'next_event': None,
        }
    symbols = [h['tradingsymbol'] for h in holdings]
    ltps = _ltps(symbols)
    meta = _load_meta(symbols)
    records = _enrich(holdings, ltps, meta)
    events = _load_events(symbols, 30)
    return {
        'summary': _summary(records),
        'holdings': records,
        'movers_today': _movers(records, 'day_pct'),
        'movers_weekly': _movers(records, 'change_5d_pct'),
        'extremes': _extremes(records),
        'events': events,
        'next_event': events[0] if events else None,
    }


# ──── Snapshots (daily history) ──────────────────────────────────

def capture_daily_snapshot(snap_date: Optional[date] = None) -> str:
    d = snap_date or date.today()
    dkey = d.isoformat()
    holdings = _kite_holdings()
    if not holdings:
        logger.info(f'[Holdings] snapshot {dkey} skipped — no holdings')
        return dkey
    symbols = [h['tradingsymbol'] for h in holdings]
    ltps = _ltps(symbols)
    meta = _load_meta(symbols)
    records = _enrich(holdings, ltps, meta)
    summary = _summary(records)
    movers = _movers(records, 'day_pct')
    extremes = _extremes(records)
    with _lock:
        c = _conn(SNAPSHOTS_DB, SNAPSHOTS_SCHEMA)
        try:
            c.execute('DELETE FROM holdings_snapshots WHERE snap_date=?', (dkey,))
            c.execute(
                '''INSERT INTO holdings_snapshots(
                    snap_date, generated_at, summary_json,
                    movers_today_json, extremes_json, holdings_json
                ) VALUES(?,?,?,?,?,?)''',
                (dkey, datetime.now().isoformat(),
                 json.dumps(summary), json.dumps(movers),
                 json.dumps(extremes), json.dumps(records)))
            c.commit()
        finally:
            c.close()
    logger.info(f'[Holdings] snapshot stored {dkey}')
    return dkey


def get_snapshot(snap_date: Optional[str] = None) -> Optional[dict]:
    with _lock:
        c = _conn(SNAPSHOTS_DB, SNAPSHOTS_SCHEMA)
        try:
            if snap_date:
                row = c.execute(
                    'SELECT * FROM holdings_snapshots WHERE snap_date=?',
                    (snap_date,)).fetchone()
            else:
                row = c.execute(
                    'SELECT * FROM holdings_snapshots ORDER BY snap_date DESC LIMIT 1'
                ).fetchone()
            if not row:
                return None
            return {
                'snap_date': row['snap_date'],
                'generated_at': row['generated_at'],
                'summary': json.loads(row['summary_json']),
                'movers_today': json.loads(row['movers_today_json']),
                'extremes': json.loads(row['extremes_json']),
                'holdings': json.loads(row['holdings_json']),
            }
        finally:
            c.close()


def list_snapshots(limit: int = 120) -> list[dict]:
    with _lock:
        c = _conn(SNAPSHOTS_DB, SNAPSHOTS_SCHEMA)
        try:
            rows = c.execute(
                'SELECT snap_date, generated_at, summary_json FROM holdings_snapshots '
                'ORDER BY snap_date DESC LIMIT ?', (limit,)).fetchall()
            out = []
            for r in rows:
                s = json.loads(r['summary_json'])
                out.append({
                    'snap_date': r['snap_date'],
                    'generated_at': r['generated_at'],
                    'day_pnl': s.get('day_pnl'),
                    'day_pct': s.get('day_pct'),
                    'total_pnl': s.get('total_pnl'),
                    'current': s.get('current'),
                    'count': s.get('count'),
                })
            return out
        finally:
            c.close()


if __name__ == '__main__':
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'digest'
    if cmd == 'meta':
        print(f'meta: {refresh_holdings_meta()} symbols')
    elif cmd == 'events':
        print(f'events: {refresh_corporate_actions()}')
    elif cmd == 'snapshot':
        print(f'snapshot: {capture_daily_snapshot()}')
    else:
        print(json.dumps(get_digest(), indent=2, default=str)[:2000])
