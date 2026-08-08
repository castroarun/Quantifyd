"""Calibrate a smallcap-leadership proxy against the REAL Nifty Smallcap 250 ETF (HDFCSML250, 2023-02+),
then re-run the regime study with the best-agreeing definition.

The old proxy (small = traded-value ranks 301-500) was only 68% directionally correct because genuine
smallcaps trade thinly and rank far below 500 — that bucket was really mid-caps. Test wider/deeper
buckets to find one that tracks the real index, then redo the N200-vs-N500 regime split + switched book.
"""
import importlib.util, sqlite3, json
from pathlib import Path
import numpy as np, pandas as pd, logging
logging.disable(logging.WARNING)
spec = importlib.util.spec_from_file_location(
    "m75", "/home/arun/quantifyd/research/75_nifty250_momentum_top15/scripts/run_nifty250_momentum.py")
m75 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m75)
BENCH, EXCLUDE = m75.BENCH, m75.EXCLUDE
RES = Path("/home/arun/quantifyd/research/106_nifty500_universe/results")
close, tv = m75.load()
con = sqlite3.connect("/home/arun/quantifyd/backtest_data/market_data.db")
real = pd.Series({pd.Timestamp(d): c for d, c in con.execute(
    "SELECT date,close FROM market_data_unified WHERE symbol='HDFCSML250' AND timeframe='day' AND close>0")}
).sort_index()
nb = close[BENCH].ffill()
cm = real.index.intersection(nb.index)
rs_real_m = (real[cm] / nb[cm]).dropna().resample("ME").last()
real6 = rs_real_m / rs_real_m.shift(6) - 1

FULL = close.index[close.index >= pd.Timestamp("2011-01-01")]
ME = [pd.Timestamp(x) for x in sorted(set(pd.Series(FULL, index=FULL).groupby([FULL.year, FULL.month]).max().values))]
print("precomputing monthly traded-value ranks ...", flush=True)
RANK = {}
for d in ME:
    adv = tv.loc[:d].tail(126).median().dropna()
    adv = adv[[s for s in adv.index if s not in EXCLUDE and s != BENCH]]
    RANK[d] = adv.sort_values(ascending=False)


def bucket_rs(lo_s, hi_s, lo_l, hi_l):
    """Cumulative RS of a 'small' traded-value slice vs a 'large' slice, rebuilt monthly."""
    S, L, prev = [], [], None
    for d in ME:
        r = RANK[d]
        if prev is not None:
            for a, b, store in ((lo_s, hi_s, S), (lo_l, hi_l, L)):
                names = r.index[a:b]
                rr = []
                for s in names:
                    if s in close.columns:
                        p0, p1 = close.at[prev, s], close.at[d, s]
                        if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                            rr.append(p1 / p0 - 1)
                store.append((d, float(np.mean(rr)) if rr else 0.0))
        prev = d
    si = (1 + pd.Series(dict(S))).cumprod(); li = (1 + pd.Series(dict(L))).cumprod()
    return si / li


DEFS = [("small 301-500 vs 1-100 (OLD)", 300, 500, 0, 100),
        ("small 500-900 vs 1-100", 500, 900, 0, 100),
        ("small 600-1200 vs 1-100", 600, 1200, 0, 100),
        ("small 400-1000 vs 1-50", 400, 1000, 0, 50),
        ("small 700-1400 vs 1-100", 700, 1400, 0, 100),
        ("small 250-750 vs 1-100", 250, 750, 0, 100)]
print(f"\n{'definition':>32} {'agree%':>8} {'corr':>7} {'now 6m':>9} {'now 12m':>9}", flush=True)
best = None
for lbl, a, b, c_, d_ in DEFS:
    rs = bucket_rs(a, b, c_, d_)
    p6 = rs / rs.shift(6) - 1
    j = pd.DataFrame({"r": real6, "p": p6}).dropna()
    if len(j) < 5:
        print(f"{lbl:>32}   insufficient overlap"); continue
    agree = ((j.r > 0) == (j.p > 0)).mean() * 100
    corr = j.r.corr(j.p)
    n6 = p6.dropna().iloc[-1] * 100
    n12 = (rs / rs.shift(12) - 1).dropna().iloc[-1] * 100
    print(f"{lbl:>32} {agree:>7.0f}% {corr:>+7.2f} {n6:>+8.1f}% {n12:>+8.1f}%", flush=True)
    if best is None or (agree, corr) > (best[0], best[1]):
        best = (agree, corr, lbl, rs)
print(f"\nBEST PROXY: {best[2]}  (agreement {best[0]:.0f}%, corr {best[1]:+.2f})", flush=True)
rs_best = best[3]
sig = (rs_best / rs_best.shift(6) - 1) > 0
json.dump({"proxy": best[2], "agree": best[0], "corr": best[1],
           "signal": [[str(k.date()), bool(v)] for k, v in sig.items() if pd.notna(v)]},
          open(RES / "smallcap_signal_calibrated.json", "w"))
print(f"smallcaps lead in {sig.mean()*100:.0f}% of months under the calibrated proxy", flush=True)
print("wrote smallcap_signal_calibrated.json", flush=True)
