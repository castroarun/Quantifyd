"""
Realistic frictions on the WINNER (NIFTYBEES timed by daily ST(7,3)):
 (a) idle cash parked in a LIQUID fund earning a rate NET of its expense + slab tax
     (gross ~6.5% → net ~4.5% for a 30%-slab investor after ~0.5% expense),
 (b) SETTLEMENT LAG — T+1 on the ETF: on a red flip you exit but proceeds sit in transit
     (0%) for `settle_gap` days before the liquid fund starts; on a green flip you re-enter
     the market `reentry_lag` days LATE (sell liquid T+1 → buy ETF), missing those days.
Plus equity STCG/LTCG on NIFTYBEES gains at each exit. Reports CAGR/MaxDD/Calmar.
"""
import os, sys, sqlite3
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, ROOT)
from services.technical_indicators import calc_supertrend

DB = os.path.join(ROOT, 'backtest_data', 'market_data.db')
COST = 0.0015          # per switch (ETF side; LIQUIDBEES has no STT, tiny spread)
START = '2010-01-01'


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


def sim(df, dirn, idle_net, reentry_lag, settle_gap, tax=False):
    ret = df['close'].pct_change().fillna(0).values
    d = dirn.reindex(df.index).ffill().values
    idle_d = (1 + idle_net) ** (1 / 252) - 1
    n = len(ret); v = 1.0; basis = 1.0; entry_i = 0
    state = 1                 # 1=IN NIFTYBEES, 0=OUT (liquid/transit)
    reentry_at = None; days_out = 10**9; nsw = 0
    nav = np.empty(n); nav[0] = v
    for i in range(1, n):
        sig = d[i - 1]                       # signal from prior close, act today
        if state == 1 and sig == -1:         # EXIT at today's open
            v *= (1 - COST); nsw += 1
            if tax:
                g = v - basis
                if g > 0:
                    hy = (df.index[i] - df.index[entry_i]).days / 365.25
                    v -= g * (0.10 if hy > 1 else 0.15)
            state = 0; days_out = 0; reentry_at = None
        elif state == 0:
            if sig == -1:
                reentry_at = None            # cancel a pending re-entry (whipsaw) → stay out
            elif sig == 1 and reentry_at is None:
                reentry_at = i + reentry_lag  # schedule delayed re-entry
            if reentry_at is not None and i >= reentry_at:
                v *= (1 - COST); nsw += 1; state = 1; basis = v; entry_i = i; reentry_at = None
        # accrue today's return
        if state == 1:
            v *= (1 + ret[i])
        else:
            v *= (1 + (0.0 if days_out < settle_gap else idle_d))
            days_out += 1
        nav[i] = v
    return pd.Series(nav, index=df.index), nsw


df = load('NIFTYBEES'); df = df[df.index >= START]
_, dser = calc_supertrend(df[['high', 'low', 'close']].assign(open=df['open']), 7, 3)
dirn = pd.Series(dser.values, index=df.index)
bh = (1 + df['close'].pct_change().fillna(0)).cumprod()
sbh = stats(bh)
yrs = (df.index[-1] - df.index[0]).days / 365.25

scen = [
    ('IDEAL (6.5% idle, no lag)',       6.5, 0, 0),
    ('Liquid net 4.5%, lag0',           4.5, 0, 0),
    ('REALISTIC: 4.5% net, lag1, gap1', 4.5, 1, 1),
    ('CONSERV: 4.5% net, lag2, gap2',   4.5, 2, 2),
    ('Cash 0% (no liquid), lag1',       0.0, 1, 1),
]
print(f"NIFTYBEES · daily ST(7,3) — realistic idle & settlement ({df.index.min().date()}..{df.index.max().date()})\n")
print(f"{'Scenario':<34}{'CAGR':>7}{'MaxDD':>8}{'Calmar':>7}{'sw/yr':>7}{'  |  net-tax:':>12}{'CAGR':>7}{'Calmar':>7}")
print(f"{'BUY & HOLD':<34}{sbh[0]:>6.1f}%{sbh[1]:>7.1f}%{sbh[2]:>7.2f}{'0':>7}{'':>12}{sbh[0]:>6.1f}%{sbh[2]:>7.2f}")
for lbl, idle, lag, gap in scen:
    nav, nsw = sim(df, dirn, idle / 100, lag, gap, tax=False)
    navt, _ = sim(df, dirn, idle / 100, lag, gap, tax=True)
    s = stats(nav); st = stats(navt)
    print(f"{lbl:<34}{s[0]:>6.1f}%{s[1]:>7.1f}%{s[2]:>7.2f}{nsw/yrs:>7.1f}{'':>12}{st[0]:>6.1f}%{st[2]:>7.2f}")
print("\nDONE")
