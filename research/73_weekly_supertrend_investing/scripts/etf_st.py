"""
Direct test of the user's idea: trade the INDEX ETF ITSELF on daily SuperTrend
(buy ST-green, to cash on ST-red), vs buy-and-hold the ETF. CLEAN — no survivorship,
one liquid instrument. Net of cost + STCG/LTCG. NIFTYBEES + any other ETFs in the DB.
"""
import os, sys, sqlite3
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, ROOT)
from services.technical_indicators import calc_supertrend

DB = os.path.join(ROOT, 'backtest_data', 'market_data.db')
COST = 0.0015
IDLE_D = (1 + 0.065) ** (1 / 252) - 1
START = '2010-01-01'


def load(sym):
    c = sqlite3.connect(DB)
    df = pd.read_sql_query("SELECT date,open,high,low,close FROM market_data_unified "
                           "WHERE symbol=? AND timeframe='day' ORDER BY date", c, params=(sym,))
    c.close()
    if df.empty:
        return df
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['open', 'high', 'low', 'close']).set_index('date')
    return df[df['close'] > 0]


def stats(nav):
    nav = nav.dropna()
    if len(nav) < 50:
        return None
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    c = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1
    dd = (nav / nav.cummax() - 1).min()
    r = nav.pct_change().dropna()
    sh = (r.mean() * 252) / (r.std() * np.sqrt(252)) if r.std() else 0
    return c * 100, dd * 100, (c / abs(dd) if dd else 0), sh


def st_timed(df, atr, mult, tax=False):
    """Trade the ETF on daily ST: hold when green (dir==1), cash when red. Causal (act next day)."""
    _, d = calc_supertrend(df[['high', 'low', 'close']].assign(open=df['open']), atr, mult)
    ret = df['close'].pct_change().fillna(0).values
    expo = (pd.Series(d.values, index=df.index) == 1).shift(1).fillna(True).astype(float).values
    n = len(ret)
    v = 1.0; basis = 1.0; entry_i = 0; nsw = 0
    nav = np.empty(n)
    for i in range(n):
        prev = expo[i - 1] if i > 0 else 1.0
        cur = expo[i]
        if cur != prev:
            nsw += 1
            v *= (1 - COST)
            if tax and prev > 0.5 and cur <= 0.5:
                g = v - basis
                if g > 0:
                    hy = (df.index[i] - df.index[entry_i]).days / 365.25
                    v -= g * (0.10 if hy > 1 else 0.15)
            if cur > 0.5:
                basis = v; entry_i = i
        v *= (1 + (ret[i] if cur > 0.5 else IDLE_D))
        nav[i] = v
    return pd.Series(nav, index=df.index), nsw


ETFS = ['NIFTYBEES', 'JUNIORBEES', 'BANKBEES', 'GOLDBEES', 'NIFTYMIDCAP150',
        'MIDCAPETF', 'NIFTYIETF', 'SETFNIF50', 'MID150BEES', 'MOM100']
for sym in ETFS:
    df = load(sym)
    if df.empty:
        continue
    df = df[df.index >= START]
    if len(df) < 250:
        continue
    bh = pd.Series((1 + df['close'].pct_change().fillna(0).values).cumprod(), index=df.index)
    sbh = stats(bh)
    print(f"\n===== {sym}  ({df.index.min().date()}..{df.index.max().date()}) =====")
    print(f"{'variant':<16}{'CAGR':>7}{'MaxDD':>8}{'Calmar':>7}{'Sharpe':>7}{'sw/yr':>7}{'ntCAGR':>8}{'ntCal':>7}")
    print(f"{'BUY & HOLD':<16}{sbh[0]:>6.1f}%{sbh[1]:>7.1f}%{sbh[2]:>7.2f}{sbh[3]:>7.2f}{'0':>7}{sbh[0]:>7.1f}%{sbh[2]:>7.2f}")
    yrs = (df.index[-1] - df.index[0]).days / 365.25
    for atr, mult in [(7, 3), (10, 3), (14, 3)]:
        nav, nsw = st_timed(df, atr, mult, tax=False)
        navt, _ = st_timed(df, atr, mult, tax=True)
        s = stats(nav); st = stats(navt)
        print(f"{'ST(%d,%d)'%(atr,mult):<16}{s[0]:>6.1f}%{s[1]:>7.1f}%{s[2]:>7.2f}{s[3]:>7.2f}"
              f"{nsw/yrs:>7.1f}{st[0]:>7.1f}%{st[2]:>7.2f}")
print("\nDONE")
