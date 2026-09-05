"""research/152 — Multi-Year Breakout engine.

EXTENSION of research/142's decoded bananapatterns engine
(`research/142_bananapatterns_replication/scripts/bluesky_replay.py`), not a rewrite:
the common mechanics (IBD-weighted RS percentile, Rs 5cr liquidity floor, ETF exclusion,
buy-stop-at-the-pivot fill, close-based stop + SMA trail, seed-ensemble slot allocation)
are imported / reproduced verbatim, and `simulate_ext` is asserted equal to
`bluesky_replay.simulate` when its extra mechanics are switched off (see --selftest).

What is NEW here:
  * pivot = rolling max over the previous N YEARS (not the all-time cummax)
  * a minimum-history requirement (>= N*252 prior rows) so a "5-year high" cannot be
    printed by a stock with 8 months of data
  * the ATH-overlap variants incl / excl / athonly
  * pivot AGE (the level must have held for X months) and a tightness filter
  * split-scale blackout (multi-year lookbacks straddle unadjusted corporate actions)
  * risk-based position sizing (the site's own mechanic), a +25% take-profit, and
    Indian FY (1-April) loss-netting for tax
"""
from __future__ import annotations

import re
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
if not ROOT.exists():                      # laptop smoke-test fallback
    ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parents[1]
DB = ROOT / 'backtest_data' / 'market_data.db'
R142 = ROOT / 'research' / '142_bananapatterns_replication' / 'scripts'
sys.path.insert(0, str(R142))
import bluesky_replay as br              # noqa: E402  (the decoded engine)

CAPITAL = 1_000_000
TV_FLOOR = br.TV_FLOOR                    # Rs 5 cr 20d median traded value
ETF_RE = br.ETF_RE
CAP_PCT = 0.30                            # site's hard cap: no position > 30% of capital
YR = 252                                  # trading days per year
CACHE = STUDY / 'results' / 'frames_cache.npz'


# ───────────────────────────────── data ─────────────────────────────────
def load_wide(min_rows=260, use_cache=True):
    """Full-history wide frames (close/high/low/open/volume). Cached to npz."""
    if use_cache and CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        dates = pd.DatetimeIndex(z['dates'])
        syms = list(z['syms'])
        out = {k: pd.DataFrame(z[k], index=dates, columns=syms)
               for k in ('close', 'high', 'low', 'open', 'volume')}
        print(f'frames from cache: {out["close"].shape}', flush=True)
        return out
    conn = sqlite3.connect(str(DB))
    syms = [r[0] for r in conn.execute(
        "select symbol from (select symbol, count(*) n from market_data_unified "
        f"where timeframe='day' group by symbol) where n >= {min_rows}")]
    print(f'{len(syms)} symbols with >={min_rows} daily rows', flush=True)
    cols = {k: {} for k in ('close', 'high', 'low', 'open', 'volume')}
    t0 = time.time()
    for i, s in enumerate(syms):
        df = pd.read_sql_query(
            "select date, open, high, low, close, volume from market_data_unified "
            "where symbol=? and timeframe='day' order by date", conn, params=(s,))
        df['date'] = pd.to_datetime(df['date'].str[:10])
        df = df.drop_duplicates('date').set_index('date').sort_index()
        for k in cols:
            cols[k][s] = df[k]
        if (i + 1) % 600 == 0:
            print(f'  loaded {i+1}/{len(syms)} ({time.time()-t0:.0f}s)', flush=True)
    conn.close()
    out = {k: pd.DataFrame(v).astype('float32') for k, v in cols.items()}
    out = {k: v.sort_index() for k, v in out.items()}
    print(f'wide frames: {out["close"].shape} ({time.time()-t0:.0f}s)', flush=True)
    CACHE.parent.mkdir(exist_ok=True)
    np.savez_compressed(CACHE, dates=out['close'].index.values,
                        syms=np.array(out['close'].columns, dtype=object),
                        **{k: v.values for k, v in out.items()})
    return out


# ───────────────────────── common mechanics (r/142) ─────────────────────────
def common(w, rs_min=70.0):
    """Eligibility, RS percentile, prior all-time-high close, min-history counter."""
    close, high, low, vol = w['close'], w['high'], w['low'], w['volume']
    tv20 = (close * vol).rolling(20).median()
    tv_prev = tv20.shift(1)
    elig = tv_prev >= TV_FLOOR
    elig[[c for c in close.columns if ETF_RE.search(c)]] = False

    score = (2 * (close / close.shift(63) - 1) + (close / close.shift(126) - 1)
             + (close / close.shift(189) - 1) + (close / close.shift(252) - 1))
    rs = (score.where(elig).rank(axis=1, pct=True) * 100).shift(1)

    athp = close.shift(1).cummax()                       # prior all-time-high CLOSE
    nrows = close.notna().cumsum().shift(1)              # prior rows available per symbol
    return dict(close=close, high=high, low=low, tv_prev=tv_prev, elig=elig,
                rs=rs, athp=athp, nrows=nrows, rs_ok=(rs >= rs_min))


def split_blackout_events(close, tol=0.12):
    """Detect unadjusted corporate-action price-scale steps (see STATUS §2)."""
    r = close / close.shift(1)
    flag = ((r < 0.55) | (r > 1.85))
    ev = []
    fac = (2.0, 2.5, 3.0, 4.0, 5.0, 10.0)
    rr = r.values
    fl = flag.fillna(False).values
    idx_i, idx_j = np.nonzero(fl)
    for i, j in zip(idx_i, idx_j):
        v = rr[i, j]
        if not np.isfinite(v) or v <= 0:
            continue
        f = 1.0 / v if v < 1 else v
        if any(abs(f - k) < tol * k for k in fac):
            ev.append((i, j))
    return ev


def blackout_mask(shape, events, window, fwd=20):
    """True = TRADEABLE. Blank a symbol for [d-window, d+fwd] around each scale break."""
    ok = np.ones(shape, dtype=bool)
    for i, j in events:
        ok[max(0, i - window):min(shape[0], i + fwd + 1), j] = False
    return ok


# ───────────────────────── the multi-year signal ─────────────────────────
def myb_signal(cm, n_years, level='close', athvar='incl', age_months=0,
               maxdist=0.20, tight=None, blackout=None, min_history=True):
    """Return (TRIG bool array, PIV float array) aligned to cm['close'].

    PIV[t] = max of `level` series over [t-1 .. t-N*252]; trigger = close > PIV with the
    r/142 setup, RS and liquidity conditions, plus the age / ATH-variant / tightness filters.
    """
    close = cm['close']
    src = close if level == 'close' else cm['high']
    W = int(n_years * YR)
    piv = src.shift(1).rolling(W, min_periods=1).max()   # see myb_sweep.signal_cache

    prev_close = close.shift(1)
    ok = cm['elig'] & cm['rs_ok'] & piv.notna()
    ok &= (prev_close < piv) & (prev_close >= (1 - maxdist) * piv)
    if min_history:
        ok &= (cm['nrows'] >= W)

    if age_months:
        X = int(round(age_months * YR / 12))
        # the level must be at least X months old  <=>  no new W-window high in the last X days
        old = src.shift(1 + X).rolling(W - X, min_periods=1).max()
        ok &= (old >= piv)

    athp = cm['athp']
    if athvar == 'excl':
        ok &= (piv < athp) & (close <= athp)
    elif athvar == 'athonly':
        ok &= (piv >= 0.999 * athp)

    if tight:
        rng = cm['high'].rolling(60).max() / cm['low'].rolling(60).min() - 1
        ok &= (rng.shift(1) <= tight)

    trig = (ok & (close > piv)).fillna(False).values
    if blackout is not None:
        trig &= blackout
    return trig, piv.values.astype('float32')


def oa_signal(cm, maxdist=0.20):
    """Open Alpha / Blue-Sky signal set (r/142 decoded) — for the overlap measurement."""
    close, athp = cm['close'], cm['athp']
    prev_close = close.shift(1)
    setup = ((prev_close < athp) & (prev_close >= (1 - maxdist) * athp)
             & cm['elig'] & cm['rs_ok'] & athp.notna())
    return (setup & (close > athp)).fillna(False).values, athp.values.astype('float32')


# ───────────────────────── extended simulator ─────────────────────────
def simulate_ext(seed, days_idx, dates, C, O, PIV, TRAIL, TRIG, weak_arr,
                 cost=0.0025, stop=0.08, slots=16, size_pct=0.0625, risk_pct=None,
                 cap_pct=CAP_PCT, take_profit=None, fill_close=False,
                 fill_realistic=True, stcg=0.20, ltcg=0.125, cash_yield=0.05,
                 fy_tax=True, capital=CAPITAL, selection='random'):
    """r/142's `simulate` + risk-based sizing, take-profit, and Indian-FY tax netting.

    `simulate_ext` reduces EXACTLY to bluesky_replay.simulate when risk_pct=None,
    take_profit=None, fy_tax=False and cap_pct>=1 (asserted by --selftest).
    """
    rng = np.random.default_rng(seed)
    cash = float(capital)
    positions = []                       # (col, entry_i, buy, qty)
    trades = []                          # (col, entry_i, exit_i, buy, sell, reason)
    equity = np.empty(len(days_idx), dtype=float)
    passed_up = 0
    tax_pot = 0.0
    d0 = dates[days_idx[0]]
    cur_key = (d0.year - (1 if fy_tax and d0.month < 4 else 0)) if fy_tax else d0.year
    y_day = 1.0 + cash_yield / 252.0

    for k, i in enumerate(days_idx):
        d = dates[i]
        if cash_yield and cash > 0:
            cash *= y_day
        key = (d.year - (1 if d.month < 4 else 0)) if fy_tax else d.year
        if stcg and key != cur_key:
            if tax_pot > 0:
                cash -= tax_pot
            tax_pot = 0.0
            cur_key = key
        # ── entries ──
        if not weak_arr[i]:
            cand = np.nonzero(TRIG[i])[0]
            if len(cand):
                mtm = sum(q * (C[i, c] if not np.isnan(C[i, c]) else b)
                          for c, _, b, q in positions)
                eq = cash + mtm
                if selection == 'random':
                    cand = rng.permutation(cand)
                for c in cand:
                    if len(positions) >= slots:
                        passed_up += 1
                        continue
                    piv = float(PIV[i, c])
                    if fill_close:
                        fill = float(C[i, c])
                    else:
                        fill = max(piv, float(O[i, c])) if fill_realistic else piv
                    if not np.isfinite(fill) or fill <= 0:
                        continue
                    size = (risk_pct * eq / stop) if risk_pct else (size_pct * eq)
                    size = min(size, cap_pct * eq)
                    qty = int(size / fill)
                    if qty < 1 or cash < qty * fill * (1 + cost):
                        passed_up += 1
                        continue
                    cash -= qty * fill * (1 + cost)
                    positions.append((c, i, fill, qty))
        # ── exits at the close ──
        still = []
        for c, ei, b, q in positions:
            cl = C[i, c]
            if np.isnan(cl):
                still.append((c, ei, b, q))
                continue
            reason = None
            if cl <= b * (1 - stop):
                reason = 'stop'
            elif take_profit and cl >= b * (1 + take_profit):
                reason = 'tp'
            elif (TRAIL is not None and i > ei and not np.isnan(TRAIL[i, c])
                  and cl < TRAIL[i, c]):
                reason = 'trail'
            if reason:
                cash += q * float(cl) * (1 - cost)
                if stcg:
                    pnl = q * (float(cl) - b)
                    held = (dates[i] - dates[ei]).days
                    tax_pot += (ltcg if held > 365 else stcg) * pnl
                trades.append((c, ei, i, b, float(cl), reason))
            else:
                still.append((c, ei, b, q))
        positions = still
        mtm = sum(q * (C[i, c] if not np.isnan(C[i, c]) else b)
                  for c, _, b, q in positions)
        equity[k] = cash + mtm

    last = days_idx[-1]
    for c, ei, b, q in positions:
        cl = C[last, c]
        trades.append((c, ei, last, b, float(cl) if not np.isnan(cl) else b, 'open_marked'))
    return equity, trades, passed_up


# ───────────────────────────── stats ─────────────────────────────
def stats_from(equity, dates_used, trades, capital=CAPITAL, dates=None):
    e = pd.Series(equity, index=dates_used)
    yrs = (dates_used[-1] - dates_used[0]).days / 365.25
    cagr = (e.iloc[-1] / capital) ** (1 / yrs) - 1
    dd = float((e / e.cummax() - 1).min())
    rets = np.array([t[4] / t[3] - 1 for t in trades]) if trades else np.array([])
    hold = ([(dates[t[2]] - dates[t[1]]).days for t in trades]
            if (trades and dates is not None) else [])
    # longest losing streak
    streak = mx = 0
    for r in sorted(zip([t[2] for t in trades], rets)) if len(rets) else []:
        streak = streak + 1 if r[1] <= 0 else 0
        mx = max(mx, streak)
    yearly = e.groupby(e.index.year).last()
    yr = yearly.pct_change()
    yr.iloc[0] = yearly.iloc[0] / capital - 1
    return dict(
        final=float(e.iloc[-1]), x=float(e.iloc[-1] / capital), cagr=cagr * 100,
        dd=dd * 100, calmar=(cagr / abs(dd)) if dd < 0 else np.nan, n=len(trades),
        win=float((rets > 0).mean() * 100) if len(rets) else 0.0,
        mean=float(rets.mean() * 100) if len(rets) else 0.0,
        median=float(np.median(rets) * 100) if len(rets) else 0.0,
        avg_win=float(rets[rets > 0].mean() * 100) if (rets > 0).any() else 0.0,
        avg_loss=float(rets[rets <= 0].mean() * 100) if (rets <= 0).any() else 0.0,
        max_lose_streak=mx, avg_hold=float(np.mean(hold)) if hold else 0.0,
        trades_yr=len(trades) / yrs,
        yearly={int(k): round(v * 100, 2) for k, v in yr.items()}), e


def weak_gate(close, dates, gate_sma=200, on=False):
    """NaN-robust NIFTYBEES < SMA gate (the r/142 phantom-row scar)."""
    if not on:
        return np.zeros(len(dates), dtype=bool)
    nb = close['NIFTYBEES'].dropna()
    weak = (nb < nb.rolling(gate_sma).mean()).shift(1)
    return weak.reindex(dates).ffill().fillna(False).astype(bool).values


# ───────────────────────────── self-test ─────────────────────────────
def selftest():
    """Assert simulate_ext == bluesky_replay.simulate with the extras off."""
    w = load_wide()
    cm = common(w)
    close = cm['close']
    dates = close.index
    OA, ATH = oa_signal(cm)
    trail = close.rolling(50).mean().values
    C, O = close.values, w['open'].values
    days = np.array([i for i, d in enumerate(dates) if '2015-01-01' <= str(d.date()) <= '2018-12-31'])
    weak = np.zeros(len(dates), dtype=bool)
    RS, TVp = cm['rs'].values, cm['tv_prev'].values
    e1, t1, _ = br.simulate(7, 'random', days, dates, C, w['high'].values, O, ATH, trail,
                            RS, TVp, OA, weak, True, 0.0025, stop=0.08, slots=16,
                            size_pct=0.0625, stcg=0.20, ltcg=0.125, cash_yield=0.05)
    e2, t2, _ = simulate_ext(7, days, dates, C, O, ATH, trail, OA, weak, cost=0.0025,
                             stop=0.08, slots=16, size_pct=0.0625, cap_pct=1e9,
                             fy_tax=False, stcg=0.20, ltcg=0.125, cash_yield=0.05)
    ok = np.allclose(e1, e2) and len(t1) == len(t2)
    print(f'SELFTEST equal={ok}  br_final={e1[-1]:,.0f}  ext_final={e2[-1]:,.0f}  '
          f'trades {len(t1)} vs {len(t2)}')
    assert ok, 'simulate_ext diverges from the r/142 engine'


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        selftest()
    elif '--build-cache' in sys.argv:
        load_wide(use_cache=False)
