"""Single YoY table: DD10 / SMA200 / no-gate standalone + their 50-50 momentum
blends. Per year: median-across-seeds annual return AND median-across-seeds
intra-year max drawdown. Output: results/gate_yoy_full.csv (ret and dd rows).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / 'scripts'))
import bluesky_replay as br

print('loading frames ...', flush=True)
w = br.load_frames('2004-06-01', trail_sma=20)
close, high, open_, athcp, sma, tv20 = (w[k] for k in
    ('close', 'high', 'open', 'athcp', 'sma50', 'tv20'))
etf = [c for c in close.columns if br.ETF_RE.search(c)]
tv_prev = tv20.shift(1)
prev_close = close.shift(1)
elig = tv_prev >= br.TV_FLOOR
elig[etf] = False
score = 2*(close/close.shift(63)-1) + (close/close.shift(126)-1) \
    + (close/close.shift(189)-1) + (close/close.shift(252)-1)
rs = (score.where(elig).rank(axis=1, pct=True)*100).shift(1)
setup = (prev_close < athcp) & (prev_close >= 0.8*athcp) & elig & (rs >= 70.0)
trig = (setup & (close > athcp) & athcp.notna()).fillna(False).values
dates = close.index
C, H, O, ATH, S = close.values, high.values, open_.values, athcp.values, sma.values
RSv, TVv = rs.values, tv_prev.values
days = np.array([i for i, d in enumerate(dates) if str(d.date()) >= '2006-01-01'])
nb = close['NIFTYBEES'].dropna()


def align(raw):
    return raw.shift(1).reindex(dates).ffill().fillna(False).astype(bool).values


GATES = {'DD10': align(nb < 0.9 * nb.rolling(252).max()),
         'SMA200': align(nb < nb.rolling(200).mean()),
         'NoGate': np.zeros(len(dates), dtype=bool)}

mom = pd.read_csv(ROOT / 'research' / '75_nifty250_momentum_top15' / 'results' / 'nav_armed_spec.csv',
                  index_col=0, parse_dates=True)['nav']


def yearly(nav):
    """(annual returns %, intra-year maxDD %) per calendar year."""
    ret = nav.resample('YE').last().pct_change() * 100
    first_year = nav.index[0].year
    ret.loc[ret.index[ret.index.year == first_year + 0]] = np.nan  # partial first yr handled below
    rets, dds = {}, {}
    for yr, seg in nav.groupby(nav.index.year):
        prev_end = nav[nav.index.year < yr]
        base = prev_end.iloc[-1] if len(prev_end) else seg.iloc[0]
        rets[yr] = (seg.iloc[-1] / base - 1) * 100
        run = pd.concat([pd.Series([base]), seg])
        dds[yr] = float((run / run.cummax() - 1).min() * 100)
    return rets, dds


cols = {}
for gname, wk in GATES.items():
    navs = []
    for seed in range(1, 11):
        eq, _, _ = br.simulate(seed, 'random', days, dates, C, H, O, ATH, S,
                               RSv, TVv, trig, wk, True, 0.0025, stop=0.08, slots=8)
        navs.append(pd.Series(np.asarray(eq, dtype=float), index=dates[days]))
    # standalone
    rr, dd = [], []
    for nav in navs:
        r, d = yearly(nav)
        rr.append(pd.Series(r)); dd.append(pd.Series(d))
    cols[gname] = (pd.concat(rr, axis=1).median(axis=1), pd.concat(dd, axis=1).median(axis=1))
    # blend
    idx = navs[0].index.intersection(mom.index)
    rr, dd = [], []
    for nav in navs:
        b_m = nav.loc[idx].resample('ME').last().pct_change().fillna(0)
        m_m = mom.loc[idx].resample('ME').last().pct_change().fillna(0)
        blend = (1 + 0.5*b_m + 0.5*m_m).cumprod()
        r, d = yearly(blend)
        rr.append(pd.Series(r)); dd.append(pd.Series(d))
    cols[f'{gname}_blend'] = (pd.concat(rr, axis=1).median(axis=1),
                              pd.concat(dd, axis=1).median(axis=1))
    print(f'{gname} done', flush=True)

out = {}
for name, (r, d) in cols.items():
    out[f'{name}_ret'] = r.round(1)
    out[f'{name}_dd'] = d.round(1)
df = pd.DataFrame(out)
df.to_csv(STUDY / 'results' / 'gate_yoy_full.csv')
print(df.to_string())
print('DONE', flush=True)
