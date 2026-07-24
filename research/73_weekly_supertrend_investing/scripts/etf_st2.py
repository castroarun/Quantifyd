"""
Expand the ETF-timing test: on NIFTYBEES, compare ST vs moving-average gates as the
whole-instrument timing signal (which is best & cheapest?). Net cost+tax, per-year.
All signals are INDEX-LEVEL on the ETF itself. Buy when signal risk-ON, to cash when risk-OFF.
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
    df['date'] = pd.to_datetime(df['date'])
    df = df.dropna(subset=['open', 'high', 'low', 'close']).set_index('date')
    return df[df['close'] > 0]


def run_signal(df, riskon, tax=False):
    """riskon = bool Series (True=hold ETF). Causal: act next day. Returns nav, n_switches."""
    ret = df['close'].pct_change().fillna(0).values
    expo = riskon.reindex(df.index).ffill().shift(1).fillna(True).astype(float).values
    n = len(ret); v = 1.0; basis = 1.0; entry_i = 0; nsw = 0; nav = np.empty(n)
    for i in range(n):
        prev = expo[i - 1] if i > 0 else 1.0
        cur = expo[i]
        if cur != prev:
            nsw += 1; v *= (1 - COST)
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


def stats(nav):
    nav = nav.dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    c = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1
    dd = (nav / nav.cummax() - 1).min()
    r = nav.pct_change().dropna()
    sh = (r.mean() * 252) / (r.std() * np.sqrt(252)) if r.std() else 0
    return c * 100, dd * 100, (c / abs(dd) if dd else 0), sh


df = load('NIFTYBEES')
c = df['close']
_, d7 = calc_supertrend(df[['high', 'low', 'close']].assign(open=df['open']), 7, 3)
_, d10 = calc_supertrend(df[['high', 'low', 'close']].assign(open=df['open']), 10, 3)
sigs = {
    'ST(7,3)':  pd.Series(d7.values, index=df.index) == 1,
    'ST(10,3)': pd.Series(d10.values, index=df.index) == 1,
    '50-DMA':   c > c.rolling(50).mean(),
    '100-DMA':  c > c.rolling(100).mean(),
    '200-DMA':  c > c.rolling(200).mean(),
}
df = df[df.index >= START]
for k in sigs:
    sigs[k] = sigs[k][sigs[k].index >= START]

bh = pd.Series((1 + df['close'].pct_change().fillna(0).values).cumprod(), index=df.index)
sbh = stats(bh)
yrs = (df.index[-1] - df.index[0]).days / 365.25
print(f"NIFTYBEES timing signals ({df.index.min().date()}..{df.index.max().date()})")
print(f"{'signal':<12}{'CAGR':>7}{'MaxDD':>8}{'Calmar':>7}{'Sharpe':>7}{'sw/yr':>7}{'ntCAGR':>8}{'ntCal':>7}")
print(f"{'BUY&HOLD':<12}{sbh[0]:>6.1f}%{sbh[1]:>7.1f}%{sbh[2]:>7.2f}{sbh[3]:>7.2f}{'0':>7}{sbh[0]:>7.1f}%{sbh[2]:>7.2f}")
navs = {}
for k, sig in sigs.items():
    nav, nsw = run_signal(df, sig, tax=False)
    navt, _ = run_signal(df, sig, tax=True)
    s = stats(nav); st = stats(navt); navs[k] = (nav, navt)
    print(f"{k:<12}{s[0]:>6.1f}%{s[1]:>7.1f}%{s[2]:>7.2f}{s[3]:>7.2f}{nsw/yrs:>7.1f}{st[0]:>7.1f}%{st[2]:>7.2f}")

print("\nPer-year NET-tax CAGR%: BUY&HOLD vs ST(7,3) vs 100-DMA vs 200-DMA")
def yr(nav, y):
    s = nav[nav.index.year == y]
    return 100 * (s.iloc[-1] / s.iloc[0] - 1) if len(s) >= 2 else float('nan')
print(f"{'Yr':<6}{'B&H':>8}{'ST7':>8}{'100DMA':>8}{'200DMA':>8}")
for y in range(2010, 2027):
    print(f"{y:<6}{yr(bh,y):>8.1f}{yr(navs['ST(7,3)'][1],y):>8.1f}"
          f"{yr(navs['100-DMA'][1],y):>8.1f}{yr(navs['200-DMA'][1],y):>8.1f}")
print("\nDONE")
