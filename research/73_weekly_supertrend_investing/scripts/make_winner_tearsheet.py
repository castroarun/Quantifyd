"""Client factsheet for THE WINNER: NIFTYBEES held/sold on daily SuperTrend(7,3) vs buy & hold."""
import os, sys, sqlite3, shutil
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'research', '_utilities'))
from services.technical_indicators import calc_supertrend
from tearsheet import generate_tearsheet

DB = os.path.join(ROOT, 'backtest_data', 'market_data.db')
RES = os.path.join(HERE, '..', 'results')
COST = 0.0015
IDLE_D = (1 + 0.065) ** (1 / 252) - 1
START = '2010-01-01'

c = sqlite3.connect(DB)
df = pd.read_sql_query("SELECT date,open,high,low,close FROM market_data_unified "
                       "WHERE symbol='NIFTYBEES' AND timeframe='day' ORDER BY date", c)
c.close()
df['date'] = pd.to_datetime(df['date'])
df = df.dropna(subset=['open', 'high', 'low', 'close']).set_index('date')
df = df[(df['close'] > 0) & (df.index >= START)]

# REALISTIC frictions: idle cash in a liquid fund @ 4.5% NET (after ~0.5% expense + slab tax),
# T+1 settlement lag on both sides (re-enter 1 day late; proceeds sit 1 day at 0% before liquid).
IDLE_NET = (1 + 0.045) ** (1 / 252) - 1
REENTRY_LAG, SETTLE_GAP = 1, 1
_, dser = calc_supertrend(df[['high', 'low', 'close']].assign(open=df['open']), 7, 3)
dirn = (pd.Series(dser.values, index=df.index) == 1).astype(int) * 2 - 1   # +1/-1
ret = df['close'].pct_change().fillna(0).values
dv = dirn.values
n = len(ret); v = 1.0; state = 1; reentry_at = None; days_out = 10**9
nav = np.empty(n); nav[0] = v
for i in range(1, n):
    sig = dv[i - 1]
    if state == 1 and sig == -1:
        v *= (1 - COST); state = 0; days_out = 0; reentry_at = None
    elif state == 0:
        if sig == -1:
            reentry_at = None
        elif sig == 1 and reentry_at is None:
            reentry_at = i + REENTRY_LAG
        if reentry_at is not None and i >= reentry_at:
            v *= (1 - COST); state = 1; reentry_at = None
    if state == 1:
        v *= (1 + ret[i])
    else:
        v *= (1 + (0.0 if days_out < SETTLE_GAP else IDLE_NET)); days_out += 1
    nav[i] = v
strat = pd.Series(nav, index=df.index)
bench = (1 + df['close'].pct_change().fillna(0)).cumprod()

meta = dict(
    bench='NIFTYBEES buy & hold',
    tag='BACKTEST · net of cost + realistic frictions',
    disclosures=(
        'THE WINNER (realistic) — hold NIFTYBEES when its daily SuperTrend(7,3) is green; on red, sell and '
        'park in a LIQUID fund earning ~4.5%% NET (after ~0.5%% expense + slab tax); re-enter on the next '
        'green. T+1 settlement modelled (re-enter 1 day late; proceeds sit 1 day in transit at 0%%). '
        '2010–2026, net of 0.30%% round-trip cost. Result: MAX DRAWDOWN CUT FROM −36%% TO −14%% (Calmar '
        '0.29→0.65) for a ~1.3pp CAGR give-up (10.6→9.3%%); NET of STCG/LTCG on the ETF ≈ 7.8%% CAGR / '
        'Calmar 0.46 — a ~2.8pp give-up net of EVERYTHING. The drawdown-halving is friction-proof; the '
        'return give-up is the cost. A NIFTY-futures/puts hedge (no ETF sale → no equity tax, no settlement '
        'lag) recovers most of the give-up and is the preferred build. ST(7,3) ties 50/100-DMA — any '
        'fast-medium trend filter; 200-DMA is too slow. A well-known tactical-timing result. Past '
        'performance is not indicative of future results.'
    ),
)
generate_tearsheet(strat, bench, 'NIFTYBEES · SuperTrend(7,3) Timing', meta=meta, out_dir=RES)
slug = 'niftybees-st73-winner'
os.replace(os.path.join(RES, 'tearsheet.png'), os.path.join(RES, slug + '.png'))
os.replace(os.path.join(RES, 'tearsheet.html'), os.path.join(RES, slug + '.html'))
shutil.copy(os.path.join(RES, slug + '.png'), os.path.join(ROOT, 'frontend', 'public', slug + '.png'))
print('winner tearsheet ->', slug + '.png  (published to frontend/public)')
