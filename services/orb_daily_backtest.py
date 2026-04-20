"""
ORB daily backtest — re-run today's live-equivalent signals post-close.

Scans the ORB universe from Kite 5-min + daily candles, applies live-engine
filters (CPR wide, gap, RSI), and records each stock's outcome for the day:
  - TAKEN (passed filters) with entry/exit/pnl
  - BLOCKED by a filter + what the outcome would have been
  - NO_BREAKOUT / NO_DATA / ERROR

Stores rows in backtest_data/orb_backtest.db and exposes them to the
Performance page via /api/orb/backtest.

Meant to be invoked once/day by APScheduler (cron 15:45 IST) or CLI.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

UNIVERSE = [
    'ADANIENT', 'TATASTEEL', 'BEL', 'VEDL', 'BPCL', 'M&M', 'BAJFINANCE',
    'TRENT', 'HAL', 'IRCTC', 'GRASIM', 'GODREJPROP', 'RELIANCE', 'AXISBANK',
    'APOLLOHOSP',
]

# Filter thresholds — keep in sync with config.ORB_DEFAULTS.
# Live engine uses SL at the opposite OR boundary (not a fixed %) and a
# 1.5R target. If neither hits, the position exits at EOD squareoff (15:20).
CAPITAL_PER_TRADE = 20_000
R_MULTIPLE = 1.5
GAP_LONG_BLOCK_PCT = 1.0
RSI_LONG_MIN = 60
RSI_SHORT_MAX = 40
CPR_WIDTH_THRESHOLD = 0.5
OR_MINUTES = 15

DB_PATH_DEFAULT = 'backtest_data/orb_backtest.db'


SCHEMA = """
CREATE TABLE IF NOT EXISTS orb_backtest_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date TEXT NOT NULL,
    generated_at TEXT NOT NULL,
    universe_size INTEGER,
    trades_taken INTEGER,
    signals_blocked INTEGER,
    net_pnl_inr REAL,
    notes TEXT,
    UNIQUE(run_date)
);

CREATE TABLE IF NOT EXISTS orb_backtest_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date TEXT NOT NULL,
    instrument TEXT NOT NULL,
    direction TEXT,                  -- LONG / SHORT / NULL (no signal)
    signal_type TEXT NOT NULL,       -- TAKEN / BLOCKED / NO_BREAKOUT / SKIP_WIDE_CPR / NO_DATA / ERROR
    block_reason TEXT,
    entry_time TEXT,
    entry_price REAL,
    exit_time TEXT,
    exit_price REAL,
    exit_reason TEXT,
    pnl_pct REAL,
    pnl_inr REAL,
    or_high REAL,
    or_low REAL,
    gap_pct REAL,
    cpr_width_pct REAL,
    rsi_15m REAL,
    FOREIGN KEY(run_date) REFERENCES orb_backtest_runs(run_date)
);

CREATE INDEX IF NOT EXISTS idx_orb_backtest_signals_run_date
    ON orb_backtest_signals(run_date);
"""


_db_lock = threading.Lock()


def _get_conn(db_path: str = DB_PATH_DEFAULT) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(db_path)) or '.', exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


# ───────── Helpers ────────────────────────────────────────────────

def _naive(dt):
    return dt.replace(tzinfo=None) if getattr(dt, 'tzinfo', None) else dt


def _compute_rsi(closes, period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    gains = losses = 0.0
    for i in range(1, period + 1):
        ch = closes[i] - closes[i - 1]
        if ch > 0:
            gains += ch
        else:
            losses -= ch
    if losses == 0:
        return 100.0
    rs = (gains / period) / (losses / period)
    rsi = 100 - 100 / (1 + rs)
    for i in range(period + 1, len(closes)):
        ch = closes[i] - closes[i - 1]
        g = max(ch, 0)
        l_ = max(-ch, 0)
        gains = (gains * (period - 1) + g)
        losses = (losses * (period - 1) + l_)
        if losses == 0:
            rsi = 100.0
        else:
            rs = (gains / period) / (losses / period)
            rsi = 100 - 100 / (1 + rs)
    return rsi


def _fifteen_min_closes(candles_5m):
    out = []
    bucket = []
    cur_start = None
    for c in candles_5m:
        t = _naive(c['date'])
        b = t.replace(minute=(t.minute // 15) * 15, second=0, microsecond=0)
        if cur_start is None or b != cur_start:
            if bucket:
                out.append(bucket[-1]['close'])
            bucket = [c]
            cur_start = b
        else:
            bucket.append(c)
    if bucket:
        out.append(bucket[-1]['close'])
    return out


@dataclass
class Signal:
    instrument: str
    direction: Optional[str] = None
    signal_type: str = 'NO_DATA'
    block_reason: Optional[str] = None
    entry_time: Optional[str] = None
    entry_price: Optional[float] = None
    exit_time: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl_pct: Optional[float] = None
    pnl_inr: Optional[float] = None
    or_high: Optional[float] = None
    or_low: Optional[float] = None
    gap_pct: Optional[float] = None
    cpr_width_pct: Optional[float] = None
    rsi_15m: Optional[float] = None


def _simulate_trade(candles, idx, entry, direction, or_high, or_low):
    """SL at the opposite OR boundary, target = entry ± R_MULTIPLE × risk
    (matches live engine's sl_type='or_opposite', target_type='r_multiple').
    If neither hits, exit at EOD squareoff (15:20)."""
    if direction == 'LONG':
        sl = or_low
        risk = entry - sl
        tgt = entry + risk * R_MULTIPLE
    else:
        sl = or_high
        risk = sl - entry
        tgt = entry - risk * R_MULTIPLE
    sl_pct = -risk / entry * 100 if entry else 0
    tgt_pct = risk * R_MULTIPLE / entry * 100 if entry else 0
    eod_time = datetime.strptime('15:20', '%H:%M').time()
    for j in range(idx + 1, len(candles)):
        c = candles[j]
        t = _naive(c['date'])
        if t.time() >= eod_time:
            pnl_pct = ((c['close'] - entry) / entry * 100
                       if direction == 'LONG' else (entry - c['close']) / entry * 100)
            return t, c['close'], 'EOD', pnl_pct
        if direction == 'LONG':
            if c['low'] <= sl:
                return t, sl, 'SL', sl_pct
            if c['high'] >= tgt:
                return t, tgt, 'TGT', tgt_pct
        else:
            if c['high'] >= sl:
                return t, sl, 'SL', sl_pct
            if c['low'] <= tgt:
                return t, tgt, 'TGT', tgt_pct
    last = candles[-1]
    t = _naive(last['date'])
    pnl_pct = ((last['close'] - entry) / entry * 100
               if direction == 'LONG' else (entry - last['close']) / entry * 100)
    return t, last['close'], 'OPEN', pnl_pct


# ───────── Main entry ─────────────────────────────────────────────

def run_backtest(run_date: Optional[date] = None,
                 db_path: str = DB_PATH_DEFAULT) -> dict:
    """Run the ORB backtest for `run_date` (defaults to today), persist
    results, and return a summary dict."""
    from services.kite_service import get_kite
    kite = get_kite()

    run_date = run_date or date.today()
    session_start = datetime.combine(run_date, datetime.min.time()).replace(
        hour=9, minute=15)
    or_end = session_start + timedelta(minutes=OR_MINUTES)
    now = datetime.now()

    insts = kite.instruments('NSE')
    tok_map = {i['tradingsymbol']: i['instrument_token']
               for i in insts if i['tradingsymbol'] in UNIVERSE}

    signals: list[Signal] = []

    for sym in UNIVERSE:
        token = tok_map.get(sym)
        if not token:
            signals.append(Signal(instrument=sym, signal_type='NO_DATA',
                                   block_reason='missing token'))
            continue

        try:
            candles = kite.historical_data(token, session_start, now, '5minute')
        except Exception as e:
            signals.append(Signal(instrument=sym, signal_type='ERROR',
                                   block_reason=f'5m fetch: {e}'))
            continue

        if not candles:
            signals.append(Signal(instrument=sym, signal_type='NO_DATA'))
            continue

        today_open = candles[0]['open']

        # Prev trading day daily (strictly before run_date)
        try:
            daily = kite.historical_data(
                token,
                datetime.combine(run_date - timedelta(days=15), datetime.min.time()),
                datetime.combine(run_date, datetime.min.time()),
                'day')
        except Exception as e:
            signals.append(Signal(instrument=sym, signal_type='ERROR',
                                   block_reason=f'daily fetch: {e}'))
            continue

        prev = None
        for d in reversed(daily or []):
            d_date = d['date'].date() if hasattr(d['date'], 'date') else d['date']
            if d_date < run_date:
                prev = d
                break
        if not prev:
            signals.append(Signal(instrument=sym, signal_type='NO_DATA',
                                   block_reason='no prev day'))
            continue

        ph, pl, pc = prev['high'], prev['low'], prev['close']
        pivot = (ph + pl + pc) / 3
        bc = (ph + pl) / 2
        tc = 2 * pivot - bc
        cpr_w = abs(tc - bc) / pivot * 100 if pivot > 0 else 0
        gap_pct = (today_open - pc) / pc * 100 if pc else 0
        is_wide = cpr_w > CPR_WIDTH_THRESHOLD

        or_candles = [c for c in candles if _naive(c['date']) < or_end]
        if len(or_candles) < 3:
            signals.append(Signal(instrument=sym, signal_type='NO_DATA',
                                   block_reason=f'only {len(or_candles)} OR candles',
                                   gap_pct=round(gap_pct, 3),
                                   cpr_width_pct=round(cpr_w, 4)))
            continue

        or_high = max(c['high'] for c in or_candles)
        or_low = min(c['low'] for c in or_candles)
        rsi_val = _compute_rsi(_fifteen_min_closes(candles), period=14)

        base = dict(or_high=round(or_high, 2), or_low=round(or_low, 2),
                    gap_pct=round(gap_pct, 3), cpr_width_pct=round(cpr_w, 4),
                    rsi_15m=round(rsi_val, 2) if rsi_val is not None else None)

        if is_wide:
            signals.append(Signal(instrument=sym, signal_type='SKIP_WIDE_CPR',
                                   block_reason=f'CPR width {cpr_w:.2f}%',
                                   **base))
            continue

        # Walk for breakouts
        long_break = short_break = None
        close_time = datetime.strptime('15:00', '%H:%M').time()
        for i in range(1, len(candles)):
            prev_c, cur = candles[i - 1], candles[i]
            ct = _naive(cur['date'])
            if ct < or_end:
                continue
            if ct.time() >= close_time:
                break
            if not long_break and cur['close'] > or_high and prev_c['close'] <= or_high:
                long_break = (ct, cur['close'], i)
            if not short_break and cur['close'] < or_low and prev_c['close'] >= or_low:
                short_break = (ct, cur['close'], i)
            if long_break and short_break:
                break

        # Record each direction independently
        recorded_any = False
        for direction, br in [('LONG', long_break), ('SHORT', short_break)]:
            if not br:
                continue
            recorded_any = True
            t0, entry, idx = br
            block_reason = None
            if direction == 'LONG':
                if gap_pct > GAP_LONG_BLOCK_PCT:
                    block_reason = f'gap {gap_pct:.2f}%>1%'
                elif rsi_val is None or rsi_val < RSI_LONG_MIN:
                    block_reason = f'RSI {rsi_val if rsi_val is None else f"{rsi_val:.1f}"}<60'
            else:
                if rsi_val is None or rsi_val > RSI_SHORT_MAX:
                    block_reason = f'RSI {rsi_val if rsi_val is None else f"{rsi_val:.1f}"}>40'

            exit_t, exit_px, exit_reason, pnl_pct = _simulate_trade(
                candles, idx, entry, direction, or_high, or_low)
            qty = max(1, int(CAPITAL_PER_TRADE / entry))
            pnl_inr = round((exit_px - entry) * qty if direction == 'LONG'
                             else (entry - exit_px) * qty, 2)

            signals.append(Signal(
                instrument=sym,
                direction=direction,
                signal_type='BLOCKED' if block_reason else 'TAKEN',
                block_reason=block_reason,
                entry_time=t0.strftime('%H:%M'),
                entry_price=round(entry, 2),
                exit_time=exit_t.strftime('%H:%M'),
                exit_price=round(exit_px, 2),
                exit_reason=exit_reason,
                pnl_pct=round(pnl_pct, 3),
                pnl_inr=pnl_inr,
                **base,
            ))

        if not recorded_any:
            signals.append(Signal(instrument=sym, signal_type='NO_BREAKOUT', **base))

    # Persist
    taken = [s for s in signals if s.signal_type == 'TAKEN']
    blocked = [s for s in signals if s.signal_type == 'BLOCKED']
    net_pnl = round(sum((s.pnl_inr or 0) for s in taken), 2)

    run_date_str = run_date.isoformat()
    with _db_lock:
        conn = _get_conn(db_path)
        try:
            # Replace any prior run for this date
            conn.execute('DELETE FROM orb_backtest_signals WHERE run_date=?',
                          (run_date_str,))
            conn.execute('DELETE FROM orb_backtest_runs WHERE run_date=?',
                          (run_date_str,))
            conn.execute(
                'INSERT INTO orb_backtest_runs(run_date, generated_at, universe_size, '
                'trades_taken, signals_blocked, net_pnl_inr, notes) '
                'VALUES(?,?,?,?,?,?,?)',
                (run_date_str, datetime.now().isoformat(), len(UNIVERSE),
                 len(taken), len(blocked), net_pnl, None))
            for s in signals:
                d = asdict(s)
                d['run_date'] = run_date_str
                cols = ','.join(d.keys())
                placeholders = ','.join(['?'] * len(d))
                conn.execute(
                    f'INSERT INTO orb_backtest_signals({cols}) VALUES({placeholders})',
                    list(d.values()))
            conn.commit()
        finally:
            conn.close()

    logger.info(
        f'[ORB-BT] {run_date_str}: taken={len(taken)} blocked={len(blocked)} '
        f'net={net_pnl}'
    )
    return {
        'run_date': run_date_str,
        'trades_taken': len(taken),
        'signals_blocked': len(blocked),
        'net_pnl_inr': net_pnl,
        'signals': [asdict(s) for s in signals],
    }


def get_backtest_run(run_date: Optional[str] = None,
                      db_path: str = DB_PATH_DEFAULT) -> Optional[dict]:
    """Return a stored run. If run_date is None, returns the most recent."""
    with _db_lock:
        conn = _get_conn(db_path)
        try:
            if run_date:
                run = conn.execute(
                    'SELECT * FROM orb_backtest_runs WHERE run_date=?',
                    (run_date,)).fetchone()
            else:
                run = conn.execute(
                    'SELECT * FROM orb_backtest_runs ORDER BY run_date DESC LIMIT 1'
                ).fetchone()
            if not run:
                return None
            run = dict(run)
            signals = conn.execute(
                'SELECT * FROM orb_backtest_signals WHERE run_date=? ORDER BY '
                'signal_type, instrument', (run['run_date'],)).fetchall()
            run['signals'] = [dict(s) for s in signals]
            return run
        finally:
            conn.close()


def list_backtest_runs(limit: int = 30,
                        db_path: str = DB_PATH_DEFAULT) -> list[dict]:
    """Return summaries of the most recent backtest runs, newest first."""
    with _db_lock:
        conn = _get_conn(db_path)
        try:
            rows = conn.execute(
                'SELECT * FROM orb_backtest_runs ORDER BY run_date DESC LIMIT ?',
                (limit,)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


if __name__ == '__main__':
    # CLI: python -m services.orb_daily_backtest [YYYY-MM-DD]
    import sys
    d = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else None
    out = run_backtest(run_date=d)
    print(f"taken={out['trades_taken']} blocked={out['signals_blocked']} net=Rs{out['net_pnl_inr']:+.0f}")
