"""
Weekly HMA 30/44 + MACD(21,39,9) + RSI(9/3/21W) retracement-reversal swing engine.
research/93 — Nitin Hulaji setup (Market Aur Main Ep.5). Causal, look-ahead-safe.
See HMA30_44_MACD_RSI_WEEKLY_SWEEP_STATUS.md sections 1-4 for the locked spec.

Entry (weekly bar i, all causal, LONG only as taught):
  1. close>HMA30, cross-above happened within recent_win weeks, close still <= max(HMA30,HMA44)
  2. MACD hist turning up after a >=neg_bars consecutive below-zero run (run ended within recent_win)
  3. RSI(9) 3-SMA crossed above its 21-WMA within recent_win weeks and is still above
Entry at next week's open. SL = last confirmed 2/2 fractal low (0.1% buffer, fallback 10w min-low).
Target = 26w max-high. Stop-first same-bar. Time-stop 52w. Random-entry control shares the
exact exit mechanics (the drift benchmark that r/87/88/91 made mandatory).
"""
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

DB = Path(__file__).resolve().parents[3] / 'backtest_data' / 'market_data.db'


# ---------------------------------------------------------------- data loading
def load_daily(symbol, db=DB, start=None, end=None):
    con = sqlite3.connect(str(db))
    q = ("SELECT date, open, high, low, close, volume FROM market_data_unified "
         "WHERE symbol=? AND timeframe='day' ")
    args = [symbol]
    if start:
        q += "AND date>=? "; args.append(start)
    if end:
        q += "AND date<=? "; args.append(end)
    q += "ORDER BY date"
    df = pd.read_sql_query(q, con, params=args, parse_dates=['date'])
    con.close()
    if df.empty:
        return df
    df = df.drop_duplicates('date').reset_index(drop=True)
    df = df[(df[['open', 'high', 'low', 'close']] > 0).all(axis=1)].reset_index(drop=True)
    return df


def to_weekly(df):
    if df.empty:
        return df
    w = (df.set_index('date')
           .resample('W-FRI')
           .agg({'open': 'first', 'high': 'max', 'low': 'min',
                 'close': 'last', 'volume': 'sum'})
           .dropna(subset=['open'])
           .reset_index())
    return w


def list_universe(db=DB, min_weeks=200, min_price=20.0, min_turnover=1e7,
                  exclude_substr=('NIFTY', 'BANKNIFTY', 'SENSEX', 'VIX', 'BEES',
                                  'ETF', 'GOLD', 'LIQUID', 'SILVER')):
    """All daily symbols passing basic depth/liquidity screens. Survivorship caveat applies."""
    con = sqlite3.connect(str(db))
    syms = [r[0] for r in con.execute(
        "SELECT DISTINCT symbol FROM market_data_unified WHERE timeframe='day'")]
    con.close()
    out = []
    for s in syms:
        u = s.upper()
        if any(x in u for x in exclude_substr):
            continue
        out.append(s)
    return sorted(out), dict(min_weeks=min_weeks, min_price=min_price,
                             min_turnover=min_turnover)


# ---------------------------------------------------------------- indicators
def _wma(s, n):
    w = np.arange(1, n + 1, dtype=float)
    return s.rolling(n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


def hma(s, n):
    half, root = max(1, int(round(n / 2))), max(1, int(round(np.sqrt(n))))
    return _wma(2 * _wma(s, half) - _wma(s, n), root)


def rsi_wilder(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def add_indicators(w, hma_fast=30, hma_slow=44, macd_fast=21, macd_slow=39,
                   macd_sig=9, rsi_len=9, rsi_ma=3, rsi_wma_len=21):
    c = w['close']
    w['hma_f'] = hma(c, hma_fast)
    w['hma_s'] = hma(c, hma_slow)
    macd = c.ewm(span=macd_fast, adjust=False).mean() - c.ewm(span=macd_slow, adjust=False).mean()
    w['hist'] = macd - macd.ewm(span=macd_sig, adjust=False).mean()
    r = rsi_wilder(c, rsi_len)
    w['rsi_ma'] = r.rolling(rsi_ma).mean()
    w['rsi_wma'] = _wma(r, rsi_wma_len)
    return w


# ---------------------------------------------------------------- helpers
def confirmed_pivot_low(l, i, wing=2):
    """Index of most recent fractal low (strict vs `wing` bars each side) confirmed by bar i."""
    for j in range(i - wing, wing - 1, -1):
        lo = l[j]
        if all(lo < l[j - k] for k in range(1, wing + 1)) and \
           all(lo < l[j + k] for k in range(1, wing + 1)):
            return j
    return None


def _simulate(w, ent, stop, target, max_hold=52, cost_bps=25.0):
    """Long from open[ent] with fixed stop/target on weekly bars. Returns trade dict."""
    o = w['open'].values; h = w['high'].values; l = w['low'].values; c = w['close'].values
    dates = w['date'].values
    n = len(w)
    entry = o[ent]
    exit_price = None; exj = None; reason = None
    for k in range(ent, n):
        if l[k] <= stop:                                   # stop-first (conservative)
            exit_price = min(o[k], stop); exj = k; reason = 'sl'; break
        if h[k] >= target:
            exit_price = max(o[k], target); exj = k; reason = 'target'; break
        if (k - ent) >= max_hold:
            exit_price = c[k]; exj = k; reason = 'time'; break
    if exit_price is None:
        exj = n - 1; exit_price = c[exj]; reason = 'end'
    gross = (exit_price - entry) / entry
    net = gross - cost_bps / 1e4
    risk = (entry - stop) / entry
    R = net / risk if risk > 1e-6 else np.nan
    return dict(entry_date=str(dates[ent])[:10], exit_date=str(dates[exj])[:10],
                entry=round(float(entry), 3), exit=round(float(exit_price), 3),
                stop=round(float(stop), 3), target=round(float(target), 3),
                reason=reason, hold_wks=int(exj - ent),
                gross_pct=round(float(gross) * 100, 4), net_pct=round(float(net) * 100, 4),
                risk_pct=round(float(risk) * 100, 4),
                R=round(float(R), 3) if not np.isnan(R) else None)


def _levels(w, i, target_lb=26, stop_buffer=0.001, pivot_wing=2):
    """Stop/target computed from data <= signal bar i. Returns (stop, target) or None."""
    l = w['low'].values; h = w['high'].values
    entry_ref = w['open'].values[i + 1] if i + 1 < len(w) else w['close'].values[i]
    pj = confirmed_pivot_low(l, i, wing=pivot_wing)
    stop = l[pj] * (1 - stop_buffer) if pj is not None else None
    if stop is None or stop >= entry_ref:
        stop = l[max(0, i - 9):i + 1].min() * (1 - stop_buffer)
    if stop >= entry_ref:
        return None
    target = h[max(0, i - target_lb + 1):i + 1].max()
    if target <= entry_ref:
        return None
    return stop, target


# ---------------------------------------------------------------- signals + backtest
def find_signals(w, neg_bars=8, recent_win=4):
    """Boolean array: triple-aligned setup on completed bar i."""
    c = w['close'].values; hf = w['hma_f'].values; hs = w['hma_s'].values
    hist = w['hist'].values; rm = w['rsi_ma'].values; rw = w['rsi_wma'].values
    n = len(w)
    above30 = c > hf
    cross30 = np.zeros(n, bool)
    cross30[1:] = above30[1:] & ~above30[:-1]
    rsi_above = rm > rw
    rsi_cross = np.zeros(n, bool)
    rsi_cross[1:] = rsi_above[1:] & ~rsi_above[:-1]
    neg = hist < 0
    # length of consecutive below-zero run ending at each bar
    run = np.zeros(n, int)
    for i in range(n):
        run[i] = run[i - 1] + 1 if (i > 0 and neg[i]) else (1 if neg[i] else 0)
    sig = np.zeros(n, bool)
    for i in range(2, n):
        if np.isnan(hf[i]) or np.isnan(hs[i]) or np.isnan(rw[i]) or np.isnan(hist[i]):
            continue
        lo = max(0, i - recent_win + 1)
        # 1. HMA zone: above 30, crossed recently, not extended past both HMAs
        if not (above30[i] and cross30[lo:i + 1].any() and c[i] <= max(hf[i], hs[i])):
            continue
        # 2. MACD: hist rising; a >=neg_bars below-zero run ended within recent_win
        if not (hist[i] > hist[i - 1] and (run[lo:i + 1] >= neg_bars).any()):
            continue
        # 3. RSI: 3-SMA above 21-WMA, crossed recently
        if not (rsi_above[i] and rsi_cross[lo:i + 1].any()):
            continue
        sig[i] = True
    return sig


def backtest(w, neg_bars=8, recent_win=4, target_lb=26, max_hold=52, cost_bps=25.0):
    if len(w) < 100:
        return []
    sig = find_signals(w, neg_bars=neg_bars, recent_win=recent_win)
    trades = []
    n = len(w)
    i = 0
    while i < n - 1:
        if not sig[i]:
            i += 1; continue
        lv = _levels(w, i, target_lb=target_lb)
        if lv is None:
            i += 1; continue
        stop, target = lv
        t = _simulate(w, i + 1, stop, target, max_hold=max_hold, cost_bps=cost_bps)
        trades.append(t)
        i = i + 1 + t['hold_wks'] + 1      # no overlap: resume after exit bar
    return trades


def random_control(w, setup_trades, mult=3, seed=0, target_lb=26, max_hold=52,
                   cost_bps=25.0):
    """Random entries in the same symbol, drawn from the same calendar years as the
    setup trades, identical stop/target mechanics. The drift benchmark."""
    if not setup_trades or len(w) < 100:
        return []
    years = sorted({t['entry_date'][:4] for t in setup_trades})
    yr = pd.to_datetime(w['date']).dt.year.astype(str).values
    valid = [i for i in range(50, len(w) - 1) if yr[i] in years]
    if not valid:
        return []
    rng = np.random.RandomState(seed)
    picks = rng.choice(valid, size=min(len(setup_trades) * mult, len(valid)),
                       replace=len(valid) < len(setup_trades) * mult)
    trades = []
    for i in sorted(int(x) for x in picks):
        lv = _levels(w, i, target_lb=target_lb)
        if lv is None:
            continue
        trades.append(_simulate(w, i + 1, *lv, max_hold=max_hold, cost_bps=cost_bps))
    return trades


# ---------------------------------------------------------------- metrics
def summarize(trades):
    if not trades:
        return dict(n=0)
    net = np.array([t['net_pct'] for t in trades]) / 100.0
    gross = np.array([t['gross_pct'] for t in trades]) / 100.0
    Rs = np.array([t['R'] for t in trades if t['R'] is not None])
    wins = net[net > 0]; losses = net[net <= 0]
    n = len(net)
    t_stat = float(net.mean() / (net.std(ddof=1) / np.sqrt(n))) if n > 1 and net.std() > 0 else 0.0
    pf = float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float('inf')
    return dict(
        n=n,
        gross_exp_pct=round(float(gross.mean()) * 100, 4),
        net_exp_pct=round(float(net.mean()) * 100, 4),
        net_exp_R=round(float(Rs.mean()), 3) if len(Rs) else None,
        win_pct=round(float((net > 0).mean()) * 100, 1),
        pf=round(pf, 2),
        t_stat=round(t_stat, 2),
        avg_hold_wks=round(float(np.mean([t['hold_wks'] for t in trades])), 1),
        total_net_pct=round(float(net.sum()) * 100, 2))


def run_symbol(symbol, seed=0, min_weeks=200, min_price=20.0, min_turnover=1e7,
               **params):
    """Returns (setup_trades, setup_summary, control_summary) or (None, reason, None)."""
    df = load_daily(symbol)
    if df.empty:
        return None, 'no data', None
    w = to_weekly(df)
    if len(w) < min_weeks:
        return None, 'short history', None
    if df['close'].median() < min_price:
        return None, 'penny', None
    if (df['close'] * df['volume']).median() < min_turnover:
        return None, 'illiquid', None
    w = add_indicators(w)
    trades = backtest(w, **params)
    ctrl = random_control(w, trades, seed=seed, **{k: v for k, v in params.items()
                                                  if k in ('target_lb', 'max_hold', 'cost_bps')})
    return trades, summarize(trades), summarize(ctrl)
