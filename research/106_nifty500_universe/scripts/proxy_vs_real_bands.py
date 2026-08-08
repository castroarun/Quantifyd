"""How good are my traded-value PROXIES for each market band, vs REAL index/ETF trackers?
Bands: Nifty 200 (ranks 1-200), Nifty 500 (1-500), Nifty 201-500 (the mid/small slice), plus the
smallcap slice already shown. Reports overlap, return correlation and direction agreement."""
import importlib.util, sqlite3
import numpy as np, pandas as pd, logging
logging.disable(logging.WARNING)
spec = importlib.util.spec_from_file_location(
    "m75", "/home/arun/quantifyd/research/75_nifty250_momentum_top15/scripts/run_nifty250_momentum.py")
m75 = importlib.util.module_from_spec(spec); spec.loader.exec_module(m75)
BENCH, EXCLUDE = m75.BENCH, m75.EXCLUDE
con = sqlite3.connect("/home/arun/quantifyd/backtest_data/market_data.db")

print("=== searching for REAL trackers of the 200 / 500 / midsmall bands ===")
pats = ["%NIF100%", "%NIFTY100%", "%N100%", "%NIF200%", "%NIFTY200%", "%N200%", "%NIF500%",
        "%NIFTY500%", "%N500%", "%500%", "%MIDSML%", "%MIDSMALL%", "%NEXT50%", "%JUNIOR%",
        "%MID150%", "%MIDCAP150%", "%SML250%", "%SMALL250%"]
found = {}
for p in pats:
    for s, n, a, b in con.execute(
            "SELECT symbol,COUNT(*),MIN(date),MAX(date) FROM market_data_unified "
            "WHERE timeframe='day' AND close>0 AND symbol LIKE ? GROUP BY symbol HAVING COUNT(*)>150", (p,)):
        found[s] = (n, a, b)
for s, (n, a, b) in sorted(found.items(), key=lambda x: -x[1][0]):
    print(f"  {s:18} {n:>6} rows  {a} .. {b}")

close, tv = m75.load()
nb = close[BENCH].ffill()
FULL = close.index[close.index >= pd.Timestamp("2014-01-01")]
ME = [pd.Timestamp(x) for x in sorted(set(pd.Series(FULL, index=FULL).groupby([FULL.year, FULL.month]).max().values))]
print("\nbuilding traded-value rank buckets ...", flush=True)
RANK = {}
for d in ME:
    adv = tv.loc[:d].tail(126).median().dropna()
    adv = adv[[s for s in adv.index if s not in EXCLUDE and s != BENCH]]
    RANK[d] = adv.sort_values(ascending=False)


def band_index(lo, hi):
    """Equal-weight monthly-rebalanced index of traded-value ranks [lo,hi)."""
    out, prev = [], None
    for d in ME:
        r = RANK[d]
        if prev is not None:
            rr = []
            for s in r.index[lo:hi]:
                if s in close.columns:
                    p0, p1 = close.at[prev, s], close.at[d, s]
                    if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                        rr.append(p1 / p0 - 1)
            out.append((d, float(np.mean(rr)) if rr else 0.0))
        prev = d
    return (1 + pd.Series(dict(out))).cumprod()


PROX = {"Nifty 200 proxy (tv 1-200)": band_index(0, 200),
        "Nifty 500 proxy (tv 1-500)": band_index(0, 500),
        "Nifty 201-500 proxy (tv 200-500)": band_index(200, 500),
        "Smallcap proxy (tv 301-500)": band_index(300, 500)}
REALS = {}
for sym in found:
    ser = pd.Series({pd.Timestamp(d): c for d, c in con.execute(
        "SELECT date,close FROM market_data_unified WHERE symbol=? AND timeframe='day' AND close>0", (sym,))
    }).sort_index()
    if len(ser) > 200:
        REALS[sym] = ser.resample("ME").last()

print(f"\n=== PROXY vs REAL: monthly-return correlation & direction agreement ===")
print(f"{'proxy':>34} {'real tracker':>14} {'months':>7} {'corr':>7} {'agree':>7} {'proxy CAGR':>11} {'real CAGR':>10}")
for pname, pser in PROX.items():
    pm = pser.pct_change().dropna()
    best = []
    for rname, rser in REALS.items():
        rm = rser.pct_change().dropna()
        j = pd.DataFrame({"p": pm, "r": rm}).dropna()
        if len(j) < 18:
            continue
        corr = j.p.corr(j.r); agree = ((j.p > 0) == (j.r > 0)).mean() * 100
        yrs = (j.index[-1] - j.index[0]).days / 365.25
        pc = ((1 + j.p).prod() ** (1 / yrs) - 1) * 100
        rc = ((1 + j.r).prod() ** (1 / yrs) - 1) * 100
        best.append((corr, rname, len(j), agree, pc, rc))
    for corr, rname, n, agree, pc, rc in sorted(best, reverse=True)[:3]:
        print(f"{pname:>34} {rname:>14} {n:>7} {corr:>+7.2f} {agree:>6.0f}% {pc:>10.1f}% {rc:>9.1f}%")
    print()
