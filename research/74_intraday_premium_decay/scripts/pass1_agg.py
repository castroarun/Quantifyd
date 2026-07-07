"""research/72 pass 1 (aggregation) — reads the saved per-day-per-minute ATM paths and produces
the averaged decay-then-rise report. Merge-based normalisation (avoids the groupby.apply gotcha)."""
import pandas as pd
from pathlib import Path

OUT = Path("/home/arun/quantifyd/research/74_intraday_premium_decay/results")
REF = "14:00"
SHOW = [f"{h:02d}:{m:02d}" for h in (14, 15) for m in range(0, 60, 5)]
R = pd.read_csv(OUT / "atm_minute_paths.csv", dtype={"min": str})
n0 = len(R)
R = R[(R["spot"] > 20000) & (R["spot"] < 28000) & (R["prem"] > 0)].copy()   # drop garbage spot/prem rows
print("rows: %d -> %d after spot/prem sanity filter" % (n0, len(R)))


def get_ref(g):
    r = g[g["min"] == REF]
    r = r.iloc[0] if len(r) else g.sort_values("min").iloc[0]
    return r[["prem", "iv", "ce", "pe", "spot"]]


refdf = R.groupby("day").apply(get_ref, include_groups=False).add_suffix("_ref").reset_index()
R = R.merge(refdf, on="day", how="left")
for c in ["prem", "ce", "pe"]:
    R[c + "_idx"] = R[c] / R[c + "_ref"] * 100
R["iv_d"] = R["iv"] - R["iv_ref"]
R["spot_move"] = (R["spot"] - R["spot_ref"]).abs()


def report(sub, label):
    a = sub.groupby("min").agg(prem_idx=("prem_idx", "mean"), prem_abs=("prem", "mean"), iv=("iv", "mean"),
                               iv_d=("iv_d", "mean"), ce_idx=("ce_idx", "mean"), pe_idx=("pe_idx", "mean"),
                               spot_move=("spot_move", "median"), n=("day", "nunique")).reset_index().sort_values("min")
    print(f"\n=== {label}  (n≈{sub['day'].nunique()} days) ===")
    print(f"{'min':6}{'prem_idx':>9}{'prem_Rs':>8}{'atmIV%':>8}{'IVd':>7}{'medMove':>8}{'n':>4}")
    for _, r in a[a["min"].isin(SHOW)].iterrows():
        print(f"{r['min']:6}{r['prem_idx']:9.1f}{r['prem_abs']:8.0f}{r['iv']:8.2f}{r['iv_d']:7.2f}{r['spot_move']:8.1f}{int(r['n']):4}")
    tw = a[(a["min"] >= "14:20") & (a["min"] <= "15:30")]
    tr = tw.loc[tw["prem_idx"].idxmin()]
    e = a[a["min"] == "15:25"]
    msg = f"  -> TROUGH {tr['min']} prem_idx {tr['prem_idx']:.1f} (IV {tr['iv']:.2f}%)"
    if len(e):
        msg += f" ; by 15:25 prem_idx {e['prem_idx'].iloc[0]:.1f} (rise {e['prem_idx'].iloc[0]-tr['prem_idx']:+.1f}pts, IVd {e['iv_d'].iloc[0]-tr['iv_d']:+.2f})"
    print(msg)
    return a


allA = report(R, "ALL DAYS")
report(R[R["dte"] <= 1], "DTE 0-1 (near expiry)")
report(R[R["dte"] >= 2], "DTE 2+ (far)")
print("\n=== is the 15:15+ rise vega (both legs up) or directional (|dSpot| grows)? ===")
for m in ["14:00", "14:55", "15:00", "15:05", "15:10", "15:15", "15:20", "15:25"]:
    r = allA[allA["min"] == m]
    if len(r):
        print(f"  {m}: CE_idx {r['ce_idx'].iloc[0]:6.1f}  PE_idx {r['pe_idx'].iloc[0]:6.1f}  atmIV {r['iv'].iloc[0]:5.2f}  |dSpot| {r['spot_move'].iloc[0]:4.1f}")
allA.to_csv(OUT / "allday_minute_agg.csv", index=False)
print("\nwrote allday_minute_agg.csv")

# per-minute chart data for the 3 splits (13:55-15:30)
chart = []
for lbl, sub in [("all", R), ("expiry", R[R["dte"] <= 1]), ("far", R[R["dte"] >= 2])]:
    a = sub.groupby("min").agg(prem_idx=("prem_idx", "mean"), iv=("iv", "mean"), n=("day", "nunique")).reset_index()
    a = a[(a["min"] >= "13:55") & (a["min"] <= "15:30")].copy()
    a["split"] = lbl
    chart.append(a)
cd = pd.concat(chart).round(2)
cd.to_csv(OUT / "chart_data.csv", index=False)
print("CHARTDATA_START")
print(cd.to_csv(index=False))
print("CHARTDATA_END")
