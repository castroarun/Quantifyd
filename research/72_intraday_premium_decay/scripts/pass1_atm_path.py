"""research/72 pass 1 — intraday ATM straddle premium + IV path in the final ~2 hours, averaged
across all recorded days, to time the decay-then-rise Arun observed (~15:00 crush / ~15:15 pickup)
and decompose IV (vega) vs direction (delta). Read-only on options_data.db (~3s cadence, 53 days).

Logic: front weekly expiry, ATM strike re-picked each snapshot (direction-neutral), premium =
ATM_CE.ltp+ATM_PE.ltp, IV = mean(ATM CE/PE iv). Resample 1-min, index premium to 14:00=100,
average across days. A RISE in the (always-ATM) premium while time-to-expiry falls ⟹ IV rose."""
import sqlite3, datetime as dt
import pandas as pd
from pathlib import Path

DB = "/home/arun/quantifyd/backtest_data/options_data.db"
OUT = Path("/home/arun/quantifyd/research/72_intraday_premium_decay/results")
con = sqlite3.connect(DB)
REF, WIN0, WIN1 = "14:00", "13:40:00", "15:30:00"
SHOW = [f"{h:02d}:{m:02d}" for h in (14, 15) for m in range(0, 60, 5)]

days = pd.read_sql("SELECT DISTINCT date(snapshot_time) d FROM option_chain ORDER BY d", con)["d"].tolist()
rows = []
for d in days:
    exps = pd.read_sql(f"SELECT DISTINCT expiry_date e FROM option_chain "
                       f"WHERE snapshot_time BETWEEN '{d}T09:00' AND '{d}T15:31'", con)["e"].dropna().tolist()
    exps = sorted(exps)
    if not exps:
        continue
    front = next((e for e in exps if e[:10] >= d), exps[0])
    dte = (dt.date.fromisoformat(front[:10]) - dt.date.fromisoformat(d)).days
    q = (f"SELECT snapshot_time, instrument_type, ltp, iv, underlying_spot FROM option_chain "
         f"WHERE snapshot_time BETWEEN '{d}T{WIN0}' AND '{d}T{WIN1}' AND expiry_date='{front}' "
         f"AND instrument_type IN ('CE','PE') "
         f"AND CAST(ROUND(underlying_spot/50.0)*50 AS INT)=CAST(ROUND(strike/50.0)*50 AS INT)")
    df = pd.read_sql(q, con)
    if df.empty:
        continue
    df["min"] = pd.to_datetime(df["snapshot_time"]).dt.strftime("%H:%M")
    g = df.groupby(["min", "instrument_type"]).agg(ltp=("ltp", "mean"), iv=("iv", "mean"),
                                                   spot=("underlying_spot", "mean"))
    ce, pe = g.xs("CE", level=1), g.xs("PE", level=1)
    for m in sorted(set(ce.index) & set(pe.index)):
        rows.append(dict(day=d, dte=dte, min=m, prem=ce.loc[m, "ltp"] + pe.loc[m, "ltp"],
                         iv=(ce.loc[m, "iv"] + pe.loc[m, "iv"]) / 2, ce=ce.loc[m, "ltp"],
                         pe=pe.loc[m, "ltp"], spot=(ce.loc[m, "spot"] + pe.loc[m, "spot"]) / 2))
R = pd.DataFrame(rows)
R.to_csv(OUT / "atm_minute_paths.csv", index=False)
print("days with data:", R["day"].nunique(), "| minute-rows:", len(R))


def _norm(grp):
    ref = grp[grp["min"] == REF]
    ref = ref.iloc[0] if len(ref) else grp.sort_values("min").iloc[0]
    grp = grp.copy()
    grp["prem_idx"] = grp["prem"] / ref["prem"] * 100
    grp["iv_d"] = grp["iv"] - ref["iv"]
    grp["ce_idx"] = grp["ce"] / ref["ce"] * 100
    grp["pe_idx"] = grp["pe"] / ref["pe"] * 100
    grp["spot_move"] = (grp["spot"] - ref["spot"]).abs()
    return grp


R = R.groupby("day", group_keys=False).apply(_norm)


def report(sub, label):
    a = sub.groupby("min").agg(prem_idx=("prem_idx", "mean"), iv=("iv", "mean"), iv_d=("iv_d", "mean"),
                               ce_idx=("ce_idx", "mean"), pe_idx=("pe_idx", "mean"),
                               spot_move=("spot_move", "mean"), n=("day", "nunique")).reset_index().sort_values("min")
    print(f"\n=== {label}  (n≈{sub['day'].nunique()} days) ===")
    print(f"{'min':6}{'prem_idx':>9}{'atmIV%':>8}{'IVd':>7}{'|dSpot|':>8}{'n':>4}")
    for _, r in a[a["min"].isin(SHOW)].iterrows():
        print(f"{r['min']:6}{r['prem_idx']:9.1f}{r['iv']:8.2f}{r['iv_d']:7.2f}{r['spot_move']:8.1f}{int(r['n']):4}")
    tw = a[(a["min"] >= "14:20") & (a["min"] <= "15:30")]
    tr = tw.loc[tw["prem_idx"].idxmin()]
    e = a[a["min"] == "15:25"]
    print(f"  -> TROUGH {tr['min']} prem_idx {tr['prem_idx']:.1f} (IV {tr['iv']:.2f}%)", end="")
    if len(e):
        print(f" ; by 15:25 prem_idx {e['prem_idx'].iloc[0]:.1f} (rise {e['prem_idx'].iloc[0]-tr['prem_idx']:+.1f}pts, IV {e['iv_d'].iloc[0]-tr['iv_d']:+.2f})")
    else:
        print()
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
print("\nwrote results/atm_minute_paths.csv + allday_minute_agg.csv")
