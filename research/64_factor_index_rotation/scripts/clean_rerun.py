"""
research/64 CLEAN re-run after discovering Quality & LowVol index series are corrupt
(bad prints: Quality 150% daily vol/+472% print; LowVol 308%/±50-104%).

- Rebuild every factor from Kite daily, CLEAN bad prints (clip daily |move|>18% as
  spike+revert artifacts, reconstruct price), resample month-end.
- Clean correlation matrix; replace-Nifty single-sleeve ranking (sleeve+Gold+Nasdaq);
  re-confirm the G2 winner. Clean factors (Mom/Value/Alpha) reported as-is; Quality/LowVol
  reported as "cleaned-indicative" (use the factor ETF before trusting live).

Run on VPS: venv/bin/python research/64_factor_index_rotation/scripts/clean_rerun.py
"""
import datetime as dt, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RES = HERE.parent / "results"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "research" / "63_gtaa_etf_rotation" / "scripts"))
from gtaa_engine import load_monthly_closes, _metrics
from services.kite_service import get_kite

TOKS = {"Momentum": 290057, "Quality": 272393, "Value": 267529, "LowVol": 272137,
        "Alpha": 265993, "Nifty": 256265}
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
    """Remove bad prints: clip daily returns to +/-18%, rebuild price, then month-end."""
    r = s.pct_change()
    r_clip = r.clip(-0.18, 0.18)
    price = s.iloc[0] * (1 + r_clip.fillna(0)).cumprod()
    return price.resample("ME").last()


monthly = {}
for nm, t in TOKS.items():
    raw = daily(t)
    monthly[nm] = raw.resample("ME").last() if nm in CLEAN else clean_monthly(raw)
fac = pd.DataFrame(monthly)
etf = load_monthly_closes(["GOLDBEES", "MON100"]).rename(columns={"GOLDBEES": "Gold", "MON100": "Nasdaq"})
panel = fac.join(etf, how="outer").sort_index()
start = panel[["Gold", "Nasdaq", "Momentum"]].dropna().index.min()
rets = panel.loc[start:].pct_change()
print(f"window {panel.loc[start:].index.min().date()} -> {panel.index.max().date()} ({len(rets.dropna())} mo)\n")

print("Per-factor (clean; Quality/LowVol spike-cleaned):")
for c in TOKS:
    m = _metrics(rets[c].dropna())
    print(f"  {c:9s} CAGR={m['cagr']*100:6.1f}% vol={rets[c].std()*np.sqrt(12)*100:5.1f}% DD={m['max_drawdown']*100:6.1f}%")

print("\nCLEAN factor cross-correlation (monthly):")
print(rets[list(TOKS)].corr().round(2).to_string())


def book(assets, mode="equal", cost=20.0, lb=12):
    R = rets[assets].dropna(); prev = pd.Series(0.0, index=assets); net = []; idx = R.index
    for i in range(len(idx) - 1):
        t1 = idx[i + 1]
        if mode == "equal" or i < lb:
            w = pd.Series(1/len(assets), index=assets)
        else:
            vol = R.iloc[i-lb+1:i+1].std(); iv = 1/vol.replace(0, np.nan); w = (iv/iv.sum()).fillna(1/len(assets))
        net.append((t1, float((w*R.loc[t1]).sum()) - float((w-prev).abs().sum())*(cost/1e4))); prev = w
    return _metrics(pd.Series(dict(net)).sort_index())


print("\n=== REPLACE NIFTY (sleeve + Gold + Nasdaq), inverse-vol, net 20bps ===")
res = []
for s in ["Nifty", "Momentum", "Value", "Alpha", "Quality", "LowVol"]:
    m = book([s, "Gold", "Nasdaq"], "invvol")
    flag = "" if s in CLEAN else "  (cleaned-indicative)"
    res.append((s, m)); print(f"  {s:9s} CAGR={m['cagr']*100:5.1f}% DD={m['max_drawdown']*100:6.1f}% Calmar={m['calmar']:.2f} Sharpe={m['sharpe']:.2f}{flag}")

print("\n=== G2 winner re-confirm (clean) ===")
for lbl, a in [("Nifty+Gold+Nasdaq equal", ["Nifty", "Gold", "Nasdaq"]),
               ("Momentum+Gold+Nasdaq invvol", ["Momentum", "Gold", "Nasdaq"]),
               ("Value+Gold+Nasdaq invvol", ["Value", "Gold", "Nasdaq"])]:
    mode = "equal" if "equal" in lbl else "invvol"
    m = book(a, mode); print(f"  {lbl:32s} CAGR={m['cagr']*100:5.1f}% DD={m['max_drawdown']*100:6.1f}% Calmar={m['calmar']:.2f}")

fac.to_csv(RES / "factor_monthly_closes_CLEAN.csv")
rets[list(TOKS)].corr().round(3).to_csv(RES / "factor_corr_CLEAN.csv")
print("\nsaved CLEAN csvs")
