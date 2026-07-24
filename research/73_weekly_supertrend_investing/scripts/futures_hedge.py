"""
MODELED futures-hedge version of the winner (NIFTYBEES · daily ST(7,3)).

Instead of SELLING the ETF on a red flip (equity CGT + T+1 settlement + liquid-fund tax),
you KEEP the ETF and SHORT NIFTY futures against it. While hedged, long-ETF + short-future
≈ a synthetic T-bill earning the carry (≈ risk-free): net daily = hedged_net/252.
Wins vs cash-rotation: (a) ETF never sold → NO equity CGT realised (deferred like B&H),
(b) NO T+1 settlement lag (hedge is intraday, ETF stays put), (c) margin funded by PLEDGING
the ETF (no extra capital). Costs: futures roll/txn per on-off; futures carry gains taxed at slab.

*** MODELED, not data-driven: the DB has no NIFTY futures series, so the hedge carry is an
assumption (sensitivity shown). Flag loudly; validate on real futures data before any capital. ***
"""
import os, sys, sqlite3
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, ROOT)
from services.technical_indicators import calc_supertrend

DB = os.path.join(ROOT, 'backtest_data', 'market_data.db')
START = '2010-01-01'
HEDGE_COST = 0.0004   # futures round-trip per hedge on/off (STT on sell + brokerage + 1 tick), on notional


def load(sym):
    c = sqlite3.connect(DB)
    df = pd.read_sql_query("SELECT date,open,high,low,close FROM market_data_unified "
                           "WHERE symbol=? AND timeframe='day' ORDER BY date", c, params=(sym,))
    c.close()
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['open', 'high', 'low', 'close']).set_index('date')
    return df[df['close'] > 0]


def stats(nav):
    nav = nav.dropna(); yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    c = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1
    dd = (nav / nav.cummax() - 1).min()
    r = nav.pct_change().dropna(); sh = (r.mean() * 252) / (r.std() * np.sqrt(252)) if r.std() else 0
    return c * 100, dd * 100, (c / abs(dd) if dd else 0), sh


def hedge_sim(df, dirn, hedged_net):
    """Hold ETF always. Red flip -> hedged (earn carry). Green -> unhedged (earn ETF). No equity tax, no lag."""
    ret = df['close'].pct_change().fillna(0).values
    d = dirn.reindex(df.index).ffill().values
    hd = (1 + hedged_net) ** (1 / 252) - 1
    n = len(ret); v = 1.0; state = 0   # 0=unhedged(full ETF), 1=hedged
    nav = np.empty(n); nav[0] = v; nsw = 0
    for i in range(1, n):
        sig = d[i - 1]
        want = 1 if sig == -1 else 0
        if want != state:
            v *= (1 - HEDGE_COST); nsw += 1; state = want   # put on / take off the hedge
        v *= (1 + (hd if state == 1 else ret[i]))
        nav[i] = v
    return pd.Series(nav, index=df.index), nsw


df = load('NIFTYBEES'); df = df[df.index >= START]
_, dser = calc_supertrend(df[['high', 'low', 'close']].assign(open=df['open']), 7, 3)
dirn = pd.Series(dser.values, index=df.index)
bh = (1 + df['close'].pct_change().fillna(0)).cumprod()
sbh = stats(bh)
yrs = (df.index[-1] - df.index[0]).days / 365.25

print(f"NIFTYBEES · daily ST(7,3) — MODELED futures-hedge vs alternatives ({df.index.min().date()}..{df.index.max().date()})\n")
print(f"{'Approach':<44}{'CAGR':>7}{'MaxDD':>8}{'Calmar':>7}{'Sharpe':>7}{'sw/yr':>7}")
print(f"{'BUY & HOLD (defers equity tax)':<44}{sbh[0]:>6.1f}%{sbh[1]:>7.1f}%{sbh[2]:>7.2f}{sbh[3]:>7.2f}{'0':>7}")
print(f"{'CASH-ROTATION realistic (net all tax) [ref]':<44}{'7.8%':>7}{'-14.3%':>8}{'0.46':>7}{'~0.9':>7}{'5.0':>7}")
print("  --- futures-hedge: REAL-DATA carry (NSE bhavcopy: hedge-on gross ~+3.1%, net ~+2-3%) ---")
for hn in [0.020, 0.026, 0.032, 0.046]:
    nav, nsw = hedge_sim(df, dirn, hn)
    s = stats(nav)
    tag = " (real ~net)" if hn in (0.020, 0.026, 0.032) else " (old modeled)"
    lbl = f"Futures-hedge, carry {hn*100:.1f}% net{tag}"
    print(f"{lbl:<44}{s[0]:>6.1f}%{s[1]:>7.1f}%{s[2]:>7.2f}{s[3]:>7.2f}{nsw/yrs:>7.1f}")
print("\n(equity CGT on the ETF is DEFERRED in both B&H and the hedge — fair, like-for-like.)")
print("DONE")
