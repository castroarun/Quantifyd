"""
BlueSky ATH-Breakout PAPER book (research/142, G5 soak) — Rs 10L, EOD, VPS cron.

Adopted taxable spec (decoded + optimized, see /app/backtest/bluesky-ath-breakout-research142):
  universe : NSE dailies in market_data.db (>=260 rows), 20d-median traded value >= Rs 5cr,
             ETFs excluded, NO mcap floor
  setup    : prev close within 20% of the all-time-high CLOSE and below it;
             IBD-RS percentile (2xr63+r126+r189+r252, ranked over eligibles, as of t-1) >= 70
  signal   : today's close ABOVE the prior ATH-close  ->  buy-stop for TOMORROW at the pivot
  entry    : next day: fill = open if open >= pivot else (pivot if high >= pivot else MISS)
  exits    : close <= buy*0.92 (stop) or close < SMA20 (trail), booked at that close
  book     : 8 slots, 18.75% of NAV per position, cash-constrained, RS-desc selection,
             25bps/side cost; NIFTYBEES < SMA200 gate blocks NEW signals
Paper only — no orders are placed anywhere.

Run nightly by cron after 15:35 IST. `--dry` computes and prints without writing.
State: backtest_data/bluesky_paper_state.json (lockfile + atomic replace).
UI feed: static/app/bluesky_paper.json (served at /app/bluesky_paper.json, no restart).
"""
import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DB = ROOT / 'backtest_data' / 'market_data.db'
STATE = ROOT / 'backtest_data' / 'bluesky_paper_state.json'
LOCK = ROOT / 'backtest_data' / 'bluesky_paper_state.lock'
UI_JSON = ROOT / 'static' / 'app' / 'bluesky_paper.json'

CAPITAL = 1_000_000
SLOTS = 8
SIZE_PCT = 0.1875
STOP = 0.08
TRAIL_SMA = 20
RS_MIN = 70.0
TV_FLOOR = 5e7
GATE_SMA = 200
COST = 0.0025
ETF_RE = re.compile(r'(BEES|ETF|LIQUID|GILT|SENSEX|NIF[A-Z]*50)')
DRY = '--dry' in sys.argv


def ist_now():
    return datetime.utcnow() + pd.Timedelta(hours=5, minutes=30)


def load_state():
    if STATE.exists():
        return json.load(open(STATE))
    return dict(capital=CAPITAL, cash=float(CAPITAL), positions=[], pending=[],
                nav=[], trades=[], missed=[], started=str(date.today()), last_run=None)


def save_state(st):
    tmp = STATE.with_suffix('.json.tmp')
    json.dump(st, open(tmp, 'w'), indent=1, default=str)
    os.replace(tmp, STATE)


def acquire_lock():
    for _ in range(30):
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            time.sleep(2)
    return False


def load_wide():
    conn = sqlite3.connect(str(DB))
    syms = [r[0] for r in conn.execute(
        "select symbol from (select symbol, count(*) n, max(date) mx from market_data_unified "
        "where timeframe='day' group by symbol) where n >= 260 and mx >= date('now','-14 day')")]
    closes, opens, highs, vols = {}, {}, {}, {}
    for s in syms:
        df = pd.read_sql_query(
            "select date, open, high, close, volume from market_data_unified "
            "where symbol=? and timeframe='day' and date >= date('now','-450 day') "
            "order by date", conn, params=(s,))
        df['date'] = pd.to_datetime(df['date'].str[:10])
        df = df.drop_duplicates('date').set_index('date').sort_index()
        if len(df) < 260:
            continue
        closes[s], opens[s], highs[s], vols[s] = df['close'], df['open'], df['high'], df['volume']
    # full-history ATH-close needs ALL history, not 450d — fetch separately (cheap: max(close) before window)
    ath_base = {}
    for s in closes:
        r = conn.execute("select max(close) from market_data_unified where symbol=? "
                         "and timeframe='day' and date < date('now','-450 day')", (s,)).fetchone()
        ath_base[s] = r[0] if r and r[0] else 0.0
    conn.close()
    wide = {k: pd.DataFrame(v) for k, v in
            dict(close=closes, open=opens, high=highs, vol=vols).items()}
    return wide, pd.Series(ath_base)


def fetch_today(wide, today):
    """If today's bar is missing, merge live OHLC from Kite (read-only; canonical DB untouched)."""
    close = wide['close']
    if today in close.index and close.loc[today].notna().sum() > len(close.columns) * 0.5:
        return wide, 'db'
    from kiteconnect import KiteConnect
    from config import KITE_API_KEY
    tok = json.load(open(ROOT / 'backtest_data' / 'access_token.json'))['access_token']
    kite = KiteConnect(api_key=KITE_API_KEY)
    kite.set_access_token(tok)
    syms = list(close.columns)
    rows = {}
    for i in range(0, len(syms), 400):
        batch = ['NSE:' + s for s in syms[i:i + 400]]
        try:
            q = kite.ohlc(batch)
        except Exception as e:
            print('ohlc batch failed:', e)
            continue
        for k, v in q.items():
            s = k.split(':', 1)[1]
            o = v.get('ohlc', {})
            if o.get('close') is not None and o.get('open'):
                rows[s] = (o.get('open'), o.get('high'), o.get('low'), v.get('last_price'))
        time.sleep(0.4)
    for name, idx in (('open', 0), ('high', 1)):
        newrow = {s: rows.get(s, (np.nan,) * 4)[idx] for s in syms}
        wide[name].loc[today] = pd.Series(newrow)
    # after close, last_price == today's close
    wide['close'].loc[today] = pd.Series({s: rows.get(s, (np.nan,) * 4)[3] for s in syms})
    wide['vol'].loc[today] = np.nan   # volume unknown from ohlc; TV uses trailing median, ok
    for k in wide:
        wide[k] = wide[k].sort_index()
    return wide, f'kite_ohlc({len(rows)})'


def main():
    now = ist_now()
    if not DRY and not (now.hour, now.minute) >= (15, 35):
        print(f'{now} — before 15:35 IST, aborting (use --dry to preview)')
        return
    if not DRY and not acquire_lock():
        print('could not acquire state lock — another run in progress?')
        return
    try:
        st = load_state()
        wide, ath_base = load_wide()
        today = pd.Timestamp(str(now.date()))
        wide, src = fetch_today(wide, today)
        close, open_, high, vol = wide['close'], wide['open'], wide['high'], wide['vol']
        if today not in close.index:
            print('no bar for today — holiday? nothing to do')
            return
        etf = [c for c in close.columns if ETF_RE.search(c)]

        tv = (close * vol).rolling(20).median()
        eligible = (tv.shift(1).loc[today] >= TV_FLOOR)
        eligible[etf] = False
        # ATH-close as of yesterday = window cummax folded with the pre-window ATH base
        athc_prev = close.shift(1).cummax().clip(lower=ath_base, axis=1)
        sma_t = close.rolling(TRAIL_SMA).mean()
        nb = close['NIFTYBEES'] if 'NIFTYBEES' in close.columns else None
        weak = bool(nb.loc[:today].iloc[-1] < nb.rolling(GATE_SMA).mean().loc[:today].iloc[-1]) if nb is not None else False

        # RS as of t-1
        c1 = close.shift(1)
        score = 2 * (c1 / c1.shift(63) - 1) + (c1 / c1.shift(126) - 1) \
            + (c1 / c1.shift(189) - 1) + (c1 / c1.shift(252) - 1)
        rs_row = (score.loc[today].where(eligible).rank(pct=True) * 100)

        log = []
        # ---- exits at today's close ----
        kept = []
        for p in st['positions']:
            s = p['symbol']
            cl = float(close.loc[today].get(s, np.nan))
            if np.isnan(cl):
                kept.append(p); continue
            reason = None
            if cl <= p['buy'] * (1 - STOP):
                reason = 'stop_8pct'
            elif str(p['entry_date']) != str(today.date()) and cl < float(sma_t.loc[today, s]):
                reason = 'trail_sma20'
            if reason:
                st['cash'] += p['qty'] * cl * (1 - COST)
                tr = dict(symbol=s, entry_date=p['entry_date'], exit_date=str(today.date()),
                          buy=p['buy'], sell=round(cl, 2), qty=p['qty'],
                          ret_pct=round((cl / p['buy'] - 1) * 100, 2), reason=reason)
                st['trades'].append(tr)
                log.append(f"EXIT {s} {reason} @{cl:.2f} ({tr['ret_pct']:+.1f}%)")
            else:
                kept.append(p)
        st['positions'] = kept

        # ---- entries from yesterday's pending (buy-stop at pivot semantics) ----
        nav_prev = st['cash'] + sum(p['qty'] * float(close.loc[today].get(p['symbol'], p['buy']))
                                    for p in st['positions'])
        for pen in sorted(st['pending'], key=lambda x: -x.get('rs', 0)):
            s = pen['symbol']
            o = float(open_.loc[today].get(s, np.nan))
            h = float(high.loc[today].get(s, np.nan))
            piv = pen['pivot']
            if np.isnan(o) or np.isnan(h):
                st['missed'].append(dict(**pen, why='no_data')); continue
            fill = o if o >= piv else (piv if h >= piv else None)
            if fill is None:
                st['missed'].append(dict(**pen, why='never_reached_pivot'))
                log.append(f"MISS {s} pivot {piv} not reached (high {h})")
                continue
            if len(st['positions']) >= SLOTS:
                st['missed'].append(dict(**pen, why='book_full')); continue
            qty = int(SIZE_PCT * nav_prev / fill)
            if qty < 1 or st['cash'] < qty * fill * (1 + COST):
                st['missed'].append(dict(**pen, why='cash_short')); continue
            st['cash'] -= qty * fill * (1 + COST)
            st['positions'].append(dict(symbol=s, qty=qty, buy=round(fill, 2),
                                        entry_date=str(today.date()), pivot=piv,
                                        signal_date=pen['signal_date']))
            log.append(f"ENTRY {s} x{qty} @{fill:.2f} (pivot {piv})")
        st['pending'] = []

        # ---- scan today's signals for tomorrow ----
        if not weak:
            prev_c = c1.loc[today]
            piv_row = athc_prev.loc[today]
            cl_row = close.loc[today]
            cand = close.columns[(prev_c < piv_row) & (prev_c >= 0.8 * piv_row)
                                 & eligible.fillna(False) & (rs_row >= RS_MIN)
                                 & (cl_row > piv_row)]
            for s in cand:
                st['pending'].append(dict(symbol=s, pivot=round(float(piv_row[s]), 2),
                                          rs=round(float(rs_row[s]), 1),
                                          signal_date=str(today.date())))
            log.append(f"SCAN {len(cand)} new signals (gate OK)")
        else:
            log.append('SCAN skipped — gate weak (NIFTYBEES < SMA200)')

        nav = st['cash'] + sum(p['qty'] * float(close.loc[today].get(p['symbol'], p['buy']))
                               for p in st['positions'])
        st['nav'].append(dict(date=str(today.date()), nav=round(nav, 0)))
        st['last_run'] = str(now)
        st['gate_weak'] = weak
        st['data_source'] = src

        navs = pd.Series({r['date']: r['nav'] for r in st['nav']}).astype(float)
        dd = float((navs / navs.cummax() - 1).min() * 100) if len(navs) > 1 else 0.0
        wins = [t for t in st['trades'] if t['ret_pct'] > 0]
        ui = dict(updated=str(now), nav=round(nav, 0), capital=st['capital'],
                  ret_pct=round((nav / st['capital'] - 1) * 100, 2), max_dd_pct=round(dd, 2),
                  gate_weak=weak, positions=st['positions'], pending=st['pending'],
                  trades=st['trades'][-60:], n_trades=len(st['trades']),
                  win_pct=round(100 * len(wins) / len(st['trades']), 1) if st['trades'] else None,
                  nav_curve=st['nav'][-500:], missed_tail=st['missed'][-20:],
                  spec='trail-20 taxable pick; no mcap floor; gate 200DMA; 25bps; Rs 10L paper',
                  study='/app/backtest/bluesky-ath-breakout-research142', log=log)
        print(f"{now} NAV Rs {nav:,.0f} ({(nav/st['capital']-1)*100:+.2f}%) "
              f"pos {len(st['positions'])}/{SLOTS} pending {len(st['pending'])} "
              f"gate {'WEAK' if weak else 'ok'} src {src}")
        for line in log:
            print(' ', line)
        if not DRY:
            save_state(st)
            UI_JSON.parent.mkdir(parents=True, exist_ok=True)
            json.dump(ui, open(UI_JSON, 'w'), indent=1, default=str)
    finally:
        if not DRY and LOCK.exists():
            LOCK.unlink()


if __name__ == '__main__':
    main()
