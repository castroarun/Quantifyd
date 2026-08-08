"""Research 106b — is Nifty-500 handicapped by using Nifty-200 SIZING?
Arun's hypothesis: with a 2.5x wider universe, a top-22 buffer is proportionally much tighter, so names
fall out of it more often -> more churn. Test proportional scaling: hold/buffer/pool scaled with the
universe. Reports turnover + realized STCG to verify the churn mechanism, not just the outcome.
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
SNAP = {}
for d in ME:
    h = close.loc[:d].ffill()
    if len(h) <= 253:
        continue
    adv = tv.loc[:d].tail(126).median()
    sc = None
    for L, w in ((126, 0.5), (252, 0.5)):
        p0, p1 = h.iloc[-L - 1], h.iloc[-1]
        r = (p1 / p0) / (p1[BENCH] / p0[BENCH]) * w
        sc = r if sc is None else sc.add(r, fill_value=np.nan)
    ok = [s for s in sc.index if s not in EXCLUDE and s != BENCH
          and pd.notna(sc[s]) and pd.notna(adv.get(s, np.nan))]
    SNAP[d] = (sc[ok], adv[ok])


def run(band, hold, buffer, pool):
    cash, held, prev, derisked, tax = 1.0, {}, None, False, 0.0
    nav_pre, nav_post = [], []
    st = dict(sells=0, buys=0, donch=0, buf_rot=0)

    def E():
        return sum(v[0] for v in held.values())

    def sell(s, d, why="x"):
        nonlocal cash, tax
        v, b, bd = held.pop(s)
        if v > b and (d - bd).days < 365:
            tax += (v - b) * STCG
        cash += v * (1 - RT); st['sells'] += 1
        if why == "donch":
            st['donch'] += 1
        elif why == "buf":
            st['buf_rot'] += 1
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
                    sell(s, d, "donch")
        if d in WK:
            b = NBX.loc[:d].dropna()
            if len(b) >= 100 and b.iloc[-1] < b.tail(100).mean():
                for s in list(held):
                    sell(s, d, "gate")
                derisked = True
            else:
                derisked = False
        if d in SNAP and not derisked:
            sc, adv = SNAP[d]
            uni = adv.sort_values(ascending=False).index[:band]
            sub = sc[[s for s in uni if s in sc.index]]
            etf = list(sub.sort_values(ascending=False).index[:pool])
            if etf:
                top, buf = etf[:hold], set(etf[:buffer])
                for s in list(held):
                    if s not in buf:
                        sell(s, d, "buf")
                cand = (list(held) + [x for x in top if x not in held])[:hold]
                tgt = [s for s in cand if s in close.columns and pd.notna(close.at[d, s])]
                per = (E() + cash) / hold
                for s in tgt:
                    if s not in held and cash > 0:
                        buy = min(per, cash)
                        if buy > 0:
                            held[s] = [buy * (1 - RT), buy, d]; cash -= buy; st['buys'] += 1
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
                net_cagr=round(c2, 1), net_calmar=round(k2, 2),
                sells_yr=round(st['sells'] / yrs, 1), buf_rot_yr=round(st['buf_rot'] / yrs, 1),
                donch_yr=round(st['donch'] / yrs, 1), stcg=round(tax * 100, 1))


grid = [
    ("N200 hold8 buf22 pool30 (LIVE)", 200, 8, 22, 30),
    ("N500 hold8 buf22 pool30 (naive)", 500, 8, 22, 30),
    ("N500 hold12 buf33 pool45", 500, 12, 33, 45),
    ("N500 hold16 buf44 pool60", 500, 16, 44, 60),
    ("N500 hold20 buf55 pool75 (full prop)", 500, 20, 55, 75),
    ("N500 hold8 buf44 pool60 (wide buf only)", 500, 8, 44, 60),
    ("N200 hold16 buf44 pool60 (control)", 200, 16, 44, 60),
]
F = ["config", "band", "hold", "buffer", "pool", "cagr", "net_cagr", "maxdd", "sharpe",
     "calmar", "net_calmar", "sells_yr", "buf_rot_yr", "donch_yr", "stcg"]
w = csv.DictWriter(open(RES / "n500_sizing.csv", "w", newline=""), fieldnames=F); w.writeheader()
print(f"{'config':>40} {'CAGR':>6} {'netCAGR':>8} {'MaxDD':>7} {'netCal':>7} {'sell/yr':>8} {'bufrot/yr':>10}", flush=True)
for lbl, band, hold, buf, pool in grid:
    r = run(band, hold, buf, pool)
    w.writerow(dict(config=lbl, band=band, hold=hold, buffer=buf, pool=pool, **r))
    print(f"{lbl:>40} {r['cagr']:>5.1f}% {r['net_cagr']:>7.1f}% {r['maxdd']:>6.1f}% "
          f"{r['net_calmar']:>7.2f} {r['sells_yr']:>8.1f} {r['buf_rot_yr']:>10.1f}", flush=True)
    sys.stdout.flush()
print("done", flush=True)
