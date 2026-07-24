"""
Fair-benchmark test (research/73): isolate what the weekly-ST TIMING adds on top of
just owning the SAME survivorship-biased Nifty200 basket.

Baselines (all on the identical 200 names, 2010-2026):
  A) EW-B&H-200: equal-weight all listed names, monthly rebalance to equal weight, NEVER time (own the basket).
  B) ST-noexit : same 16-slot ST engine but IGNORE flip-down (buy on flip-up, never sell) -> isolates the EXIT.
  C) ST-core   : the winner (buy flip-up, sell flip-down)  [read from saved equity]
  NIFTYBEES    : the (unfair) Nifty50 index benchmark, for reference.
Excess of C over A = the honest "timing edge net of survivorship+breadth".
"""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from st_weekly_engine import get_conn, load_daily, to_weekly, ROOT

RES = os.path.join(HERE, '..', 'results')
START, END = '2010-01-01', '2026-07-07'


def cagr_dd(nav):
    nav = nav.dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    c = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1
    dd = (nav / nav.cummax() - 1).min()
    r = nav.pct_change().dropna()
    sharpe = (r.mean() * 52) / (r.std() * np.sqrt(52)) if r.std() else 0
    return c * 100, dd * 100, (c / abs(dd) if dd else 0), sharpe, nav.iloc[-1] / nav.iloc[0]


conn = get_conn()
syms = [s.strip() for s in pd.read_csv(os.path.join(ROOT, 'backtest_data', 'nifty200_official.csv'))['Symbol'].dropna()]
# weekly close panel (survivorship-matched basket)
closes = {}
for s in syms:
    d = load_daily(conn, s)
    if d.empty:
        continue
    w = to_weekly(d)['close']
    closes[s] = w
nb = to_weekly(load_daily(conn, 'NIFTYBEES'))['close']
conn.close()

px = pd.DataFrame(closes).sort_index()
px = px[(px.index >= START) & (px.index <= END)]
rets = px.pct_change()

# A) EW-B&H-200: each week, equal-weight across names that have a valid return; monthly reweight is implicit
# (weekly equal-weight rebalance = upper bound on 'own the basket'; also do a buy&hold drift version)
ew_weekly = rets.mean(axis=1, skipna=True).fillna(0)          # equal-weight, weekly rebalanced
ew_navA = (1 + ew_weekly).cumprod()

# buy&hold drift (equal weight at start among names listed at start, let it drift, add new names at equal weight on listing)
# simpler robust proxy: equal-weight monthly rebalance (A) is the standard "own the basket". Use A.

# C) ST-core equity (saved) — use ITS weekly grid as the common calendar
stC = pd.read_csv(os.path.join(RES, 'g4_Nifty200_core_nogate_equity.csv'), index_col=0, parse_dates=True).iloc[:, 0]
base = stC.index
stC = stC / stC.iloc[0]
navA = ew_navA.reindex(base).ffill().bfill()
navA = navA / navA.iloc[0]

# A2) TRUE buy&hold-drift, fixed capital: ₹1 into each name AVAILABLE IN THE FIRST 8 WEEKS,
# held to end, never rebalanced. Excludes later-listers (RVNL etc.) -> LESS survivorship, most conservative.
start_win = px.index[:8]
at_start = [s for s in px.columns if px.loc[start_win, s].notna().any()]
p0 = px[at_start].apply(lambda col: col.dropna().iloc[0])
val = px[at_start].div(p0)                 # each name = growth of ₹1 from its start price
navD = val.mean(axis=1, skipna=True).ffill()   # equal-₹ at start, drift thereafter
navD = navD.reindex(base).ffill().bfill()
navD = navD / navD.iloc[0]
print(f"(drift basket = {len(at_start)} names available in first 8 weeks of 2010)")
nbA = nb.reindex(base).ffill().bfill()
nbA = nbA / nbA.iloc[0]

print(f"{'Book':<34}{'CAGR':>7}{'MaxDD':>8}{'Calmar':>8}{'Sharpe':>8}{'Total':>8}")
for name, nav in [('ST-core (16-slot, flip in/out) [C]', stC),
                  ('EW-B&H weekly-reb, SAME 200 [A]', navA),
                  ('B&H-drift fixed-cap, start-names [A2]', navD),
                  ('NIFTYBEES (Nifty50 index)', nbA)]:
    c, dd, cal, sh, tot = cagr_dd(nav)
    print(f"{name:<34}{c:>6.1f}%{dd:>7.1f}%{cal:>8.2f}{sh:>8.2f}{tot:>7.1f}x")

cA = cagr_dd(stC)[0] - cagr_dd(navA)[0]
cN = cagr_dd(stC)[0] - cagr_dd(nbA)[0]
print(f"\nExcess CAGR of ST-core over EW-B&H-SAME-200 (honest timing edge): {cA:+.1f} pp/yr")
print(f"Excess CAGR of ST-core over NIFTYBEES (the headline, incl. survivorship+breadth): {cN:+.1f} pp/yr")
print(f"=> survivorship+breadth explains ~{100*(cN-cA)/cN:.0f}% of the headline; timing ~{100*cA/cN:.0f}%")

# per-year excess of ST vs EW-B&H-200
print("\nper-year (yr, ST%, EWbasket%, excess pp):")
for y in range(navA.index[0].year, navA.index[-1].year + 1):
    s = stC[stC.index.year == y]; a = navA[navA.index.year == y]
    if len(s) < 2:
        continue
    sr = 100 * (s.iloc[-1] / s.iloc[0] - 1); ar = 100 * (a.iloc[-1] / a.iloc[0] - 1)
    print(f"  {y}  {sr:6.1f}  {ar:6.1f}  {sr-ar:+6.1f}")
