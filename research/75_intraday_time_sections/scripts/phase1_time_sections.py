"""research/75 Phase 1 — section the trading day and measure NIFTY movement + ATM premium decay
per section, grouped by DTE and weekday. Tests Arun's "10:45-14:00 lull" hypothesis. Read-only on
options_data.db (per-minute NIFTY chain w/ IV, 53 days). Premium decay% = the short-straddle P&L driver."""
import sqlite3, datetime as dt
import numpy as np, pandas as pd
from pathlib import Path

DB = "/home/arun/quantifyd/backtest_data/options_data.db"
OUT = Path("/home/arun/quantifyd/research/75_intraday_time_sections/results")
con = sqlite3.connect(DB)
SECTIONS = [("OPEN", "09:15", "10:00"), ("MORN", "10:00", "10:45"),
            ("LULL", "10:45", "14:00"), ("AFT", "14:00", "15:00"), ("CLOSE", "15:00", "15:30")]
SEC_HRS = {n: (dt.datetime.strptime(e, "%H:%M") - dt.datetime.strptime(s, "%H:%M")).seconds / 3600 for n, s, e in SECTIONS}

days = pd.read_sql("SELECT DISTINCT date(snapshot_time) d FROM option_chain ORDER BY d", con)["d"].tolist()
rows = []
for d in days:
    exps = pd.read_sql(f"SELECT DISTINCT expiry_date e FROM option_chain WHERE snapshot_time BETWEEN '{d}T09:00' AND '{d}T15:31'", con)["e"].dropna().tolist()
    exps = sorted(exps)
    if not exps:
        continue
    front = next((e for e in exps if e[:10] >= d), exps[0])
    dte = (dt.date.fromisoformat(front[:10]) - dt.date.fromisoformat(d)).days
    wd = dt.date.fromisoformat(d).strftime("%a")
    q = (f"SELECT snapshot_time, instrument_type, ltp, iv, underlying_spot FROM option_chain "
         f"WHERE snapshot_time BETWEEN '{d}T09:15:00' AND '{d}T15:30:00' AND expiry_date='{front}' "
         f"AND instrument_type IN ('CE','PE') "
         f"AND CAST(ROUND(underlying_spot/50.0)*50 AS INT)=CAST(ROUND(strike/50.0)*50 AS INT)")
    df = pd.read_sql(q, con)
    df = df[(df.underlying_spot > 20000) & (df.underlying_spot < 28000)]
    if df.empty:
        continue
    df["hm"] = pd.to_datetime(df["snapshot_time"]).dt.strftime("%H:%M")
    g = df.groupby(["hm", "instrument_type"]).agg(ltp=("ltp", "mean"), iv=("iv", "mean"), spot=("underlying_spot", "mean"))
    try:
        ce, pe = g.xs("CE", level=1), g.xs("PE", level=1)
    except KeyError:
        continue
    mins = sorted(set(ce.index) & set(pe.index))
    m = pd.DataFrame({"hm": mins,
                      "prem": [ce.loc[x, "ltp"] + pe.loc[x, "ltp"] for x in mins],
                      "iv": [(ce.loc[x, "iv"] + pe.loc[x, "iv"]) / 2 for x in mins],
                      "spot": [(ce.loc[x, "spot"] + pe.loc[x, "spot"]) / 2 for x in mins]}).set_index("hm")
    for name, s, e in SECTIONS:
        seg = m[(m.index >= s) & (m.index < e)]
        if len(seg) < 3:
            continue
        p0, p1 = seg["prem"].iloc[0], seg["prem"].iloc[-1]
        sp = seg["spot"]
        rows.append(dict(day=d, dte=dte, wd=wd, section=name,
                         prem_decay_pct=(p1 - p0) / p0 * 100,            # negative = decay (short profit)
                         prem_decay_per_hr=(p1 - p0) / p0 * 100 / SEC_HRS[name],
                         nifty_net=sp.iloc[-1] - sp.iloc[0],
                         nifty_range=sp.max() - sp.min(),
                         nifty_range_per_hr=(sp.max() - sp.min()) / SEC_HRS[name],
                         iv_chg=seg["iv"].iloc[-1] - seg["iv"].iloc[0]))
R = pd.DataFrame(rows)
R.to_csv(OUT / "section_metrics.csv", index=False)
print("days:", R["day"].nunique(), "| section-rows:", len(R))

order = [n for n, _, _ in SECTIONS]


def show(sub, label):
    a = sub.groupby("section").agg(decay=("prem_decay_pct", "mean"), decay_hr=("prem_decay_per_hr", "mean"),
                                   nifty_rng=("nifty_range", "mean"), rng_hr=("nifty_range_per_hr", "mean"),
                                   absnet=("nifty_net", lambda x: np.mean(np.abs(x))), n=("day", "nunique"))
    a = a.reindex(order)
    print(f"\n=== {label} ===")
    print(f"{'section':7}{'decay%':>8}{'decay%/hr':>10}{'nifty_rng':>10}{'rng/hr':>8}{'|net|':>7}{'n':>4}")
    for sec in order:
        if sec in a.index and not pd.isna(a.loc[sec, "n"]):
            r = a.loc[sec]
            print(f"{sec:7}{r['decay']:8.1f}{r['decay_hr']:10.1f}{r['nifty_rng']:10.0f}{r['rng_hr']:8.0f}{r['absnet']:7.0f}{int(r['n']):4}")


show(R, "ALL DAYS")
show(R[R.dte <= 1], "DTE 0-1 (near expiry)")
show(R[R.dte >= 2], "DTE 2+ (far)")
for wd in ["Mon", "Tue", "Wed", "Thu", "Fri"]:
    show(R[R.wd == wd], "Weekday " + wd)
print("\nDONE — LULL should show the least |net|/range and slowest decay%/hr if the hypothesis holds.")
