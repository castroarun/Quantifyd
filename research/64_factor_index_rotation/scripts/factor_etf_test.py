"""
research/64: replace-Nifty test on REAL factor ETFs (user request, max available history).
Fetches the longest-history ETF per factor (incl. Low-Vol & Quality that the index series
corrupted), builds the common window, runs sleeve+Gold+Nasdaq (equal & inverse-vol) +
factor-only baskets + the 3-asset baseline. Real NAV data (price), net 20bps.

Run on VPS: venv/bin/python research/64_factor_index_rotation/scripts/factor_etf_test.py
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
k = get_kite()

# candidate ETFs per factor (token, tradingsymbol); pick the longest-history one
CAND = {
    "Momentum": [(2094593, "MOMOMENTUM_MO"), (2709761, "MOM30IETF"), (2737409, "MOMOMENTUM_ABSL")],
    "Quality":  [(1847809, "SBIETFQLTY"), (2736641, "NIFTYQLITY"), (2881281, "HDFCQUAL")],
    "Value":    [(2882561, "HDFCVALUE"), (2771201, "MOVALUE")],
    "LowVol":   [(2209793, "LOWVOL1"), (2215425, "MOLOWVOL")],
    "Alpha":    [(1897473, "ALPHA_KOTAK")],
}


def daily(tok):
    out = []; s = dt.datetime(2014, 1, 1)
    while s < dt.datetime(2026, 6, 13):
        e = min(s + dt.timedelta(days=1400), dt.datetime(2026, 6, 13))
        try: out += k.historical_data(tok, s, e, "day")
        except: pass
        s = e + dt.timedelta(days=1)
    if not out: return None
    df = pd.DataFrame(out); df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df.drop_duplicates("date").set_index("date")["close"].astype(float)


print("Picking longest-history ETF per factor:")
chosen = {}
for fac, cands in CAND.items():
    best = None; bname = None
    for tok, nm in cands:
        s = daily(tok)
        if s is None or len(s) < 100:
            print(f"    {nm:16s} no/short data"); continue
        # sanity: reject corrupt (daily vol absurd)
        v = s.pct_change().std() * np.sqrt(252) * 100
        bad = int((s.pct_change().abs() > 0.08).sum())
        tag = "OK" if v < 40 and bad < 15 else "SUSPECT"
        print(f"    {nm:16s} {s.index.min().date()} -> {s.index.max().date()} n={len(s)} vol={v:.0f}% bad={bad} {tag}")
        if tag == "OK" and (best is None or s.index.min() < best.index.min()):
            best = s; bname = nm
    if best is not None:
        chosen[fac] = best; print(f"  -> {fac}: {bname} (from {best.index.min().date()})")

# monthly panel
fac_m = pd.DataFrame({f: s.resample("ME").last() for f, s in chosen.items()})
etf = load_monthly_closes(["NIFTYBEES", "GOLDBEES", "MON100"]).rename(
    columns={"GOLDBEES": "Gold", "MON100": "Nasdaq", "NIFTYBEES": "Nifty"})
panel = fac_m.join(etf, how="outer").sort_index()
allf = list(chosen)
start = panel[allf + ["Gold", "Nasdaq"]].dropna().index.min()
rets = panel.loc[start:].pct_change()
W = f"{panel.loc[start:].index.min():%Y-%m}->{panel.index.max():%Y-%m}"
print(f"\nCommon window (all factor ETFs + Gold + Nasdaq): {W} ({len(rets.dropna())} mo)\n")

print("Per-ETF (real NAV):")
for c in allf + ["Nifty", "Gold", "Nasdaq"]:
    mm = _metrics(rets[c].dropna()); print(f"  {c:9s} CAGR={mm['cagr']*100:6.1f}% vol={rets[c].std()*np.sqrt(12)*100:5.1f}% DD={mm['max_drawdown']*100:6.1f}%")

print("\nCorrelation (real ETF monthly returns):")
print(rets[allf + ["Nifty"]].corr().round(2).to_string())


def run(assets, mode="invvol", cost=20.0, lb=6):
    R = rets[assets].dropna()
    if len(assets) == 1: return _metrics(R.iloc[:, 0])
    prev = pd.Series(0.0, index=assets); net = []; idx = R.index
    for i in range(len(idx) - 1):
        t1 = idx[i + 1]
        if mode == "equal" or i < lb:
            w = pd.Series(1/len(assets), index=assets)
        else:
            vol = R.iloc[max(0, i-lb+1):i+1].std(); iv = 1/vol.replace(0, np.nan); w = (iv/iv.sum()).fillna(1/len(assets))
        net.append((t1, float((w*R.loc[t1]).sum()) - float((w-prev).abs().sum())*(cost/1e4))); prev = w
    return _metrics(pd.Series(dict(net)).sort_index())


print(f"\n=== REPLACE-NIFTY on REAL ETFs ({W}), sleeve+Gold+Nasdaq ===")
rows = []
for s in ["Nifty"] + allf:
    for mode in ("equal", "invvol"):
        mm = run([s, "Gold", "Nasdaq"], mode)
        rows.append((f"{s}+Gold+Nasdaq", mode, mm));
    me = run([s, "Gold", "Nasdaq"], "equal"); mi = run([s, "Gold", "Nasdaq"], "invvol")
    print(f"  {s:9s}  equal Cal={me['calmar']:.2f}(CAGR{me['cagr']*100:.0f}%,DD{me['max_drawdown']*100:.0f}%)  invvol Cal={mi['calmar']:.2f}(CAGR{mi['cagr']*100:.0f}%,DD{mi['max_drawdown']*100:.0f}%)")

print("\n=== factor-only & baseline (same window) ===")
for lbl, a, mode in [("LowVol+Quality (factor-only)", ["LowVol", "Quality"], "invvol"),
                      ("All factor ETFs (factor-only)", allf, "invvol"),
                      ("Nifty+Gold+Nasdaq baseline", ["Nifty", "Gold", "Nasdaq"], "equal")]:
    a = [x for x in a if x in rets.columns]
    mm = run(a, mode); print(f"  {lbl:32s} Cal={mm['calmar']:.2f} CAGR={mm['cagr']*100:.1f}% DD={mm['max_drawdown']*100:.1f}%")

with open(RES / "factor_etf_replace_nifty.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["book", "weighting", "cagr", "max_dd", "calmar", "sharpe", "window"])
    for name, mode, mm in rows:
        w.writerow([name, mode, round(mm['cagr']*100, 2), round(mm['max_drawdown']*100, 2), round(mm['calmar'], 3), round(mm['sharpe'], 3), W])
print("\nwrote results/factor_etf_replace_nifty.csv")
