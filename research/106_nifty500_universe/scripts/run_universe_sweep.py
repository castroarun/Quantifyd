"""Research 106d — universe bake-off with sizing optimisation.

Universes (traded-value rank bands; the band proxies track real indices at 0.94-0.99 correlation):
  N200      ranks   1-200  = the live book
  N250      ranks   1-250  = Nifty LargeMidcap 250 (Nifty 100 + Midcap 150) -> full midcap included
  MID150    ranks 101-250  = Nifty Midcap 150 only  -> pure midcap, large caps excluded
  N500      ranks   1-500  = reference
For each, sweep hold / buffer / pool to find each universe's OWN optimum, so the comparison is
best-vs-best rather than one universe handicapped by another's sizing.
Rules otherwise unchanged: rsblend momentum, 15-day Donchian stop, weekly NIFTYBEES>100-SMA cash gate,
monthly rotate-only rebalance. 2011-2026, net of 0.15%/leg, cash 6.5%, STCG 20% tracked.
"""
import importlib.util, csv, sys
from pathlib import Path
import numpy as np, pandas as pd, logging
logging.disable(logging.WARNING)
spec = importlib.util.spec_from_file_location(
    "m75", "/home/arun/quantifyd/research/75_nifty250_momentum_top15/scripts/run_nifty250_momentum.py")
m75 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m75)
BENCH, EXCLUDE = m75.BENCH, m75.EXCLUDE
RES = Path("/home/arun/quantifyd/research/106_nifty500_universe/results")
START = pd.Timestamp("2011-01-01"); CASH_Y, STCG, RT = 0.065, 0.20, 0.0015
close, tv = m75.load()
idx = close.index[(close.index >= START) & (close.index <= pd.Timestamp("2026-08-01"))]
ME = [pd.Timestamp(x) for x in sorted(set(pd.Series(idx, index=idx).groupby([idx.year, idx.month]).max().values))]
_iso = idx.isocalendar()
WK = set(pd.Series(idx, index=idx).groupby([_iso.year.values, _iso.week.values]).last().values)
ROLL_LOW = close.rolling(15, min_periods=15).min().shift(1); NBX = close[BENCH].ffill()
print("precomputing monthly scores + ranks ...", flush=True)
SNAP = {}
for d in ME:
    h = close.loc[:d].ffill()
    if len(h) <= 253:
        continue
    adv = tv.loc[:d].tail(126).median().dropna()
    adv = adv[[s for s in adv.index if s not in EXCLUDE and s != BENCH]]
    sc = None
    for L, w in ((126, 0.5), (252, 0.5)):
        p0, p1 = h.iloc[-L - 1], h.iloc[-1]
        r = (p1 / p0) / (p1[BENCH] / p0[BENCH]) * w
        sc = r if sc is None else sc.add(r, fill_value=np.nan)
    ok = [s for s in sc.index if s in adv.index and pd.notna(sc[s])]
    SNAP[d] = (sc[ok], adv.sort_values(ascending=False))
print(f"  {len(SNAP)} month-ends", flush=True)


def run(lo, hi, hold, buffer, pool):
    cash, held, prev, derisked, tax = 1.0, {}, None, False, 0.0
    nav_pre, nav_post = [], []
    st = dict(sells=0)

    def E():
        return sum(v[0] for v in held.values())

    def sell(s, d):
        nonlocal cash, tax
        v, b, bd = held.pop(s)
        if v > b and (d - bd).days < 365:
            tax += (v - b) * STCG
        cash += v * (1 - RT); st['sells'] += 1
    for d in idx:
        if held and prev is not None:
            for s, v in held.items():
                p1 = close.at[d, s] if s in close.columns else np.nan
                p0 = close.at[prev, s] if s in close.columns else np.nan
                if pd.notna(p1) and pd.notna(p0) and p0 > 0:
                    v[0] += v[0] * (p1 / p0 - 1.0)
        if cash > 0:
            cash *= (1 + CASH_Y) ** (1 / 252)
        if held:
            lows = ROLL_LOW.loc[d]
            for s in list(held):
                p1 = close.at[d, s] if s in close.columns else np.nan
                ln = lows.get(s, np.nan)
                if pd.notna(p1) and pd.notna(ln) and p1 < ln:
                    sell(s, d)
        if d in WK:
            b = NBX.loc[:d].dropna()
            if len(b) >= 100 and b.iloc[-1] < b.tail(100).mean():
                for s in list(held):
                    sell(s, d)
                derisked = True
            else:
                derisked = False
        if d in SNAP and not derisked:
            sc, rank = SNAP[d]
            uni = rank.index[lo:hi]
            sub = sc[[s for s in uni if s in sc.index]]
            etf = list(sub.sort_values(ascending=False).index[:pool])
            if etf:
                top, buf = etf[:hold], set(etf[:buffer])
                for s in list(held):
                    if s not in buf:
                        sell(s, d)
                cand = (list(held) + [x for x in top if x not in held])[:hold]
                tgt = [s for s in cand if s in close.columns and pd.notna(close.at[d, s])]
                per = (E() + cash) / hold
                for s in tgt:
                    if s not in held and cash > 0:
                        buy = min(per, cash)
                        if buy > 0:
                            held[s] = [buy * (1 - RT), buy, d]; cash -= buy
        nav_pre.append((d, E() + cash)); nav_post.append((d, E() + cash - tax)); prev = d

    def blk(p):
        n = pd.DataFrame(p, columns=["d", "v"]).set_index("d")["v"]
        y = (n.index[-1] - n.index[0]).days / 365.25
        c = (n.iloc[-1] / n.iloc[0]) ** (1 / y) - 1
        dr = n.pct_change().dropna()
        sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
        dd = ((n - n.cummax()) / n.cummax()).min()
        return c * 100, dd * 100, sh, (c / abs(dd) if dd < 0 else 0)
    c1, d1, s1, k1 = blk(nav_pre); c2, d2, s2, k2 = blk(nav_post)
    yrs = (idx[-1] - idx[0]).days / 365.25
    return dict(cagr=round(c1, 1), maxdd=round(d1, 1), sharpe=round(s1, 2), calmar=round(k1, 2),
                net_cagr=round(c2, 1), net_calmar=round(k2, 2), sells_yr=round(st['sells'] / yrs, 1))


UNIV = [("N200  (1-200, LIVE)", 0, 200),
        ("N250  (1-250, LargeMid250)", 0, 250),
        ("N51-250 (ex top-50 megacaps)", 50, 250),
        ("MID150 (101-250, pure midcap)", 100, 250),
        ("N500  (1-500)", 0, 500)]
SIZES = [(6, 16, 22), (8, 22, 30), (10, 28, 38), (12, 33, 45), (16, 44, 60)]
F = ["universe", "lo", "hi", "hold", "buffer", "pool", "cagr", "net_cagr", "maxdd",
     "sharpe", "calmar", "net_calmar", "sells_yr"]
w = csv.DictWriter(open(RES / "universe_sweep.csv", "w", newline=""), fieldnames=F); w.writeheader()
print(f"\n{'universe':>30} {'h/b/p':>11} {'CAGR':>6} {'netCAGR':>8} {'MaxDD':>7} {'Sharpe':>6} {'netCal':>7} {'sell/yr':>8}", flush=True)
best = {}
for uname, lo, hi in UNIV:
    for hold, buf, pool in SIZES:
        r = run(lo, hi, hold, buf, pool)
        w.writerow(dict(universe=uname, lo=lo, hi=hi, hold=hold, buffer=buf, pool=pool, **r))
        tag = f"{hold}/{buf}/{pool}"
        print(f"{uname:>30} {tag:>11} {r['cagr']:>5.1f}% {r['net_cagr']:>7.1f}% {r['maxdd']:>6.1f}% "
              f"{r['sharpe']:>6.2f} {r['net_calmar']:>7.2f} {r['sells_yr']:>8.1f}", flush=True)
        if uname not in best or r['net_calmar'] > best[uname][0]:
            best[uname] = (r['net_calmar'], tag, r)
        sys.stdout.flush()
    print(flush=True)
print("=== BEST CONFIG PER UNIVERSE (by net Calmar) ===", flush=True)
for u, (k, tag, r) in sorted(best.items(), key=lambda x: -x[1][0]):
    print(f"  {u:>30} {tag:>11}  netCAGR {r['net_cagr']:>5.1f}%  MaxDD {r['maxdd']:>6.1f}%  "
          f"netCalmar {k:.2f}  Sharpe {r['sharpe']:.2f}", flush=True)
print("done", flush=True)
