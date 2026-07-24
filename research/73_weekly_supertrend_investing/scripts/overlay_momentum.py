"""
Apply the research/73 crash overlay (NIFTY daily ST(7,3)) to our BEST-CAGR book —
the research/75 Nifty-250 momentum book (nav_base.csv, ~31.9% CAGR / −31.6% DD, already
monthly-gated). When NIFTY's daily ST(7,3) is RED, pull the whole book to cash (earn carry/
liquid); when GREEN, hold the book. Does a faster daily overlay cut the drawdown further?
"""
import os, sys, sqlite3
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, ROOT)
from services.technical_indicators import calc_supertrend

DB = os.path.join(ROOT, 'backtest_data', 'market_data.db')
COST = 0.0015          # per full-book switch to/from cash (delivery RT on the whole book)
NAV = os.path.join(ROOT, 'research', '75_nifty250_momentum_top15', 'results', 'nav_base.csv')


def stats(nav):
    nav = nav.dropna(); yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    c = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1
    dd = (nav / nav.cummax() - 1).min()
    r = nav.pct_change().dropna(); sh = (r.mean() * 252) / (r.std() * np.sqrt(252)) if r.std() else 0
    return c * 100, dd * 100, (c / abs(dd) if dd else 0), sh


# momentum book NAV -> daily returns
mom = pd.read_csv(NAV, index_col=0, parse_dates=True)['nav']
mret = mom.pct_change().fillna(0)

# NIFTY (NIFTYBEES) daily ST(7,3) direction, aligned
c = sqlite3.connect(DB)
nb = pd.read_sql_query("SELECT date,open,high,low,close FROM market_data_unified WHERE symbol='NIFTYBEES' "
                       "AND timeframe='day' ORDER BY date", c)
c.close()
nb['date'] = pd.to_datetime(nb['date']); nb = nb.set_index('date'); nb = nb[nb['close'] > 0]
_, dser = calc_supertrend(nb[['high', 'low', 'close']].assign(open=nb['open']), 7, 3)
dirn = pd.Series(dser.values, index=nb.index).reindex(mom.index, method='ffill').fillna(1)


def overlay(idle_annual, tax=False):
    idle = (1 + idle_annual) ** (1 / 252) - 1
    r = mret.values; d = dirn.values; idx = mom.index
    n = len(r); v = 1.0; state = 1; basis = 1.0; entry_i = 0
    nav = np.empty(n); nav[0] = v
    for i in range(1, n):
        want = 1 if d[i - 1] == 1 else 0
        if want != state:
            v *= (1 - COST)
            if tax and state == 1 and want == 0:      # liquidating book -> realise STCG(20% momentum convention)
                g = v - basis
                if g > 0:
                    hy = (idx[i] - idx[entry_i]).days / 365.25
                    v -= g * (0.10 if hy > 1 else 0.20)
            if want == 1:
                basis = v; entry_i = i
            state = want
        v *= (1 + (r[i] if state == 1 else idle))
        nav[i] = v
    return pd.Series(nav, index=idx)


base = mom / mom.iloc[0]
sb = stats(base)
print(f"Momentum book (research/75 nifty250, already gated) — apply NIFTY daily-ST(7,3) overlay")
print(f"period {mom.index[0].date()}..{mom.index[-1].date()}\n")
print(f"{'Version':<40}{'CAGR':>7}{'MaxDD':>8}{'Calmar':>7}{'Sharpe':>7}{'sw/yr':>7}")
print(f"{'BASE momentum book (no overlay)':<40}{sb[0]:>6.1f}%{sb[1]:>7.1f}%{sb[2]:>7.2f}{sb[3]:>7.2f}{'—':>7}")
yrs = (mom.index[-1] - mom.index[0]).days / 365.25
for lbl, idle, tax in [('+ overlay, cash@6.5% gross (pre-tax)', 0.065, False),
                       ('+ overlay, cash@4.5% NET liquid (pre-tax)', 0.045, False),
                       ('+ overlay, cash@4.5% NET liquid (net STCG)', 0.045, True),
                       ('+ overlay, hedge-carry@3.2% (proxy)', 0.032, False)]:
    nav = overlay(idle, tax)
    s = stats(nav)
    nsw = int((dirn.shift(1) != dirn).sum())
    print(f"{lbl:<40}{s[0]:>6.1f}%{s[1]:>7.1f}%{s[2]:>7.2f}{s[3]:>7.2f}{nsw/yrs:>7.1f}")
print(f"\nTime NIFTY-ST red (book pulled to cash): {100*(dirn==-1).mean():.0f}%")
print("DONE")
