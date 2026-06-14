"""
research/64 MASTER comparison (user request): solo / factor-only / factor+assets /
original 3-asset — all on the SAME clean window so they are directly comparable.
Clean factors only (Momentum/Value/Alpha + Nifty); Quality cleaned-indicative; Low-Vol
excluded (corrupt index). Gold/Nasdaq = ETF. Monthly, net 20bps.

Run on VPS: venv/bin/python research/64_factor_index_rotation/scripts/master_compare.py
"""
import datetime as dt, sys, csv
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RES = HERE.parent / "results"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "63_gtaa_etf_rotation" / "scripts"))
from gtaa_engine import load_monthly_closes, _metrics
from services.kite_service import get_kite

TOKS = {"Momentum": 290057, "Value": 267529, "Alpha": 265993, "Quality": 272393, "Nifty": 256265}
CLEAN = {"Momentum", "Value", "Alpha", "Nifty"}
k = get_kite()


def daily(tok):
    out = []; s = dt.datetime(2015, 1, 1)
    while s < dt.datetime(2026, 6, 13):
        e = min(s + dt.timedelta(days=1400), dt.datetime(2026, 6, 13))
        try: out += k.historical_data(tok, s, e, "day")
        except: pass
        s = e + dt.timedelta(days=1)
    df = pd.DataFrame(out); df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df.drop_duplicates("date").set_index("date")["close"].astype(float)


def clean_monthly(s):
    r = s.pct_change().clip(-0.18, 0.18)
    return (s.iloc[0] * (1 + r.fillna(0)).cumprod()).resample("ME").last()


m = {}
for nm, t in TOKS.items():
    raw = daily(t)
    m[nm] = raw.resample("ME").last() if nm in CLEAN else clean_monthly(raw)
fac = pd.DataFrame(m)
etf = load_monthly_closes(["GOLDBEES", "MON100"]).rename(columns={"GOLDBEES": "Gold", "MON100": "Nasdaq"})
panel = fac.join(etf, how="outer").sort_index()
start = panel[["Gold", "Nasdaq", "Momentum"]].dropna().index.min()
rets = panel.loc[start:].pct_change()
W = f"{panel.loc[start:].index.min():%Y-%m}->{panel.index.max():%Y-%m}"
print(f"window {W} ({len(rets.dropna())} mo)\n")


def run(assets, mode="invvol", cost=20.0, lb=12):
    R = rets[assets].dropna()
    if len(assets) == 1:
        s = R.iloc[:, 0].dropna(); s = s[s.index > R.index[0]]
        return _metrics(R.iloc[:, 0].dropna())
    prev = pd.Series(0.0, index=assets); net = []; idx = R.index
    for i in range(len(idx) - 1):
        t1 = idx[i + 1]
        if mode == "equal" or i < lb:
            w = pd.Series(1/len(assets), index=assets)
        else:
            vol = R.iloc[i-lb+1:i+1].std(); iv = 1/vol.replace(0, np.nan); w = (iv/iv.sum()).fillna(1/len(assets))
        net.append((t1, float((w*R.loc[t1]).sum()) - float((w-prev).abs().sum())*(cost/1e4))); prev = w
    return _metrics(pd.Series(dict(net)).sort_index())


BOOKS = [
    # (group, label, assets, mode)
    ("SOLO", "Nasdaq only", ["Nasdaq"], "-"),
    ("SOLO", "Gold only", ["Gold"], "-"),
    ("SOLO", "Momentum only", ["Momentum"], "-"),
    ("SOLO", "Alpha only", ["Alpha"], "-"),
    ("SOLO", "Value only", ["Value"], "-"),
    ("SOLO", "Quality only (ind)", ["Quality"], "-"),
    ("SOLO", "Nifty only", ["Nifty"], "-"),
    ("FACTOR-ONLY", "Value+Momentum", ["Value", "Momentum"], "invvol"),
    ("FACTOR-ONLY", "Value+Alpha", ["Value", "Alpha"], "invvol"),
    ("FACTOR-ONLY", "Momentum+Alpha", ["Momentum", "Alpha"], "invvol"),
    ("FACTOR-ONLY", "Value+Momentum+Alpha", ["Value", "Momentum", "Alpha"], "invvol"),
    ("FACTOR-ONLY", "Value+Quality (ind)", ["Value", "Quality"], "invvol"),
    ("FACTOR-ONLY", "5 clean-ish factors", ["Value", "Momentum", "Alpha", "Quality"], "invvol"),
    ("FACTOR+1 ASSET", "Value+Gold", ["Value", "Gold"], "invvol"),
    ("FACTOR+1 ASSET", "Value+Nasdaq", ["Value", "Nasdaq"], "invvol"),
    ("FACTOR+1 ASSET", "Momentum+Gold", ["Momentum", "Gold"], "invvol"),
    ("FACTOR+ GOLD+NASDAQ", "Value+Gold+Nasdaq", ["Value", "Gold", "Nasdaq"], "invvol"),
    ("FACTOR+ GOLD+NASDAQ", "Momentum+Gold+Nasdaq", ["Momentum", "Gold", "Nasdaq"], "invvol"),
    ("FACTOR+ GOLD+NASDAQ", "Alpha+Gold+Nasdaq", ["Alpha", "Gold", "Nasdaq"], "invvol"),
    ("ORIGINAL 3-ASSET", "Nifty+Gold+Nasdaq (equal)", ["Nifty", "Gold", "Nasdaq"], "equal"),
    ("ORIGINAL 3-ASSET", "Nifty+Gold+Nasdaq (invvol)", ["Nifty", "Gold", "Nasdaq"], "invvol"),
]

rows = []
for grp, lbl, a, mode in BOOKS:
    mm = run(a, "invvol" if mode == "-" else mode)
    rows.append((grp, lbl, mode, mm["cagr"]*100, mm["max_drawdown"]*100, mm["calmar"], mm["sharpe"]))

print(f"{'group':18s} {'book':28s} {'wt':6s} {'CAGR':>6s} {'MaxDD':>7s} {'Calmar':>7s} {'Sharpe':>6s}")
for grp, lbl, mode, cg, dd, cal, sh in rows:
    print(f"{grp:18s} {lbl:28s} {mode:6s} {cg:6.1f} {dd:7.1f} {cal:7.2f} {sh:6.2f}")

print("\n=== sorted by Calmar ===")
for grp, lbl, mode, cg, dd, cal, sh in sorted(rows, key=lambda x: -x[5]):
    print(f"  {cal:5.2f}  {lbl:28s} ({grp}, {mode})  CAGR={cg:.1f}% DD={dd:.1f}%")

with open(RES / "master_compare.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["group", "book", "weighting", "cagr", "max_dd", "calmar", "sharpe"])
    for r in rows:
        w.writerow([r[0], r[1], r[2], round(r[3], 2), round(r[4], 2), round(r[5], 3), round(r[6], 3)])
print("\nwrote results/master_compare.csv ; window", W)
