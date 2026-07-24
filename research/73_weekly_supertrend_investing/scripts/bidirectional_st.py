"""
Does adding a SHORT sleeve help? Compare, on NIFTY daily ST(7,3), 2010-2026:
  - Buy & hold
  - Long+cash    : long when green, CASH (liquid ~4.5% net) when red        [winner, no futures]
  - Long+hedge   : long when green, FLAT (short future, earn carry) when red [the futures winner]
  - BIDIRECTIONAL: long when green, SHORT (short future) when red            [the user's idea]
  - Short-only   : cash when green, short when red   [diagnostic: does the short side make money?]
Carry ~3.2% net (real NSE bhavcopy). Costs per switch scale with exposure change.
"""
import os, sys, sqlite3
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, ROOT)
from services.technical_indicators import calc_supertrend

DB = os.path.join(ROOT, 'backtest_data', 'market_data.db')
START = '2010-01-01'
CARRY = (1 + 0.032) ** (1 / 252) - 1     # net carry while short/hedged (real ~3.2%)
LIQ = (1 + 0.045) ** (1 / 252) - 1       # net liquid while in cash
COSTU = 0.0004                            # cost per 1.0 unit of exposure change


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


def sim(ret, dirn, green_expo, red_expo, red_mode):
    """expo (+1 long / 0 flat / -1 short). red_mode: 'cash'|'carry'|'short' sets the non-index yield."""
    d = dirn.values
    n = len(ret); v = 1.0; prev_e = green_expo; nav = np.empty(n); nav[0] = v
    for i in range(1, n):
        sig = d[i - 1]
        e = green_expo if sig == 1 else red_expo
        if e != prev_e:
            v *= (1 - COSTU * abs(e - prev_e)); prev_e = e
        if e == 1:                      # long: index return
            r = ret[i]
        elif e == -1:                   # short: -index + carry (short a contango future collects carry)
            r = -ret[i] + CARRY
        else:                           # flat
            r = CARRY if red_mode == 'carry' else LIQ
        v *= (1 + r); nav[i] = v
    return pd.Series(nav, index=df.index)


df = load('NIFTYBEES'); df = df[df.index >= START]
ret = df['close'].pct_change().fillna(0)
bh = (1 + ret).cumprod()


def daily_dir(atr, mult):
    _, d = calc_supertrend(df[['high', 'low', 'close']].assign(open=df['open']), atr, mult)
    return pd.Series(d.values, index=df.index)


def weekly_dir(atr, mult):
    w = df.resample('W-FRI').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
    _, d = calc_supertrend(w[['high', 'low', 'close']].assign(open=w['open']), atr, mult)
    ws = pd.Series(d.values, index=w.index)
    return ws.reindex(df.index, method='ffill').fillna(1)   # last completed week's signal, mapped to daily


def yr(nav, y):
    s = nav[nav.index.year == y]
    return 100 * (s.iloc[-1] / s.iloc[0] - 1) if len(s) >= 2 else float('nan')


for sig_name, dirn in [('DAILY ST(7,3)', daily_dir(7, 3)), ('WEEKLY ST(10,3)', weekly_dir(10, 3))]:
    red_idx = ret[(dirn.shift(1) == -1)]
    grn_idx = ret[(dirn.shift(1) == 1)]
    print(f"\n================  {sig_name}  ================")
    print(f"Time in RED: {100*(dirn==-1).mean():.0f}%   Index annualised return in "
          f"GREEN {100*((1+grn_idx.mean())**252-1):+.1f}% | RED {100*((1+red_idx.mean())**252-1):+.1f}%   "
          f"(short wins only if RED is clearly negative)")
    books = {
        'Buy & hold':          sim(ret.values, dirn, 1, 1, 'carry'),
        'Long+cash (liquid)':  sim(ret.values, dirn, 1, 0, 'cash'),
        'Long+hedge (winner)': sim(ret.values, dirn, 1, 0, 'carry'),
        'BIDIRECTIONAL L/S':   sim(ret.values, dirn, 1, -1, 'short'),
        'Short-only (diag)':   sim(ret.values, dirn, 0, -1, 'short'),
    }
    print(f"{'Book':<24}{'CAGR':>7}{'MaxDD':>8}{'Calmar':>7}{'Sharpe':>7}")
    for name, nav in books.items():
        s = stats(nav)
        print(f"{name:<24}{s[0]:>6.1f}%{s[1]:>7.1f}%{s[2]:>7.2f}{s[3]:>7.2f}")
    print("  per-year CAGR%: BH | hedge | bidir")
    for y in range(2010, 2027):
        print(f"    {y}  {yr(bh,y):6.1f} | {yr(books['Long+hedge (winner)'],y):6.1f} | {yr(books['BIDIRECTIONAL L/S'],y):6.1f}")
print("\nDONE")
