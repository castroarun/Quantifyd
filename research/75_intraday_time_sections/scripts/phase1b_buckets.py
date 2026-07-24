"""research/75 Phase 1b — DATA-DRIVEN intraday profile: 30-min buckets across the whole session,
letting the natural rhythm of premium decay + NIFTY movement emerge (not pre-imposing sections).
Saves fullday_minute_paths.csv (reusable). Read-only on options_data.db (53 days)."""
import sqlite3, datetime as dt
import numpy as np, pandas as pd
from pathlib import Path

DB = "/home/arun/quantifyd/backtest_data/options_data.db"
OUT = Path("/home/arun/quantifyd/research/75_intraday_time_sections/results")
con = sqlite3.connect(DB)

# 30-min buckets 09:15 -> 15:30
edges = []
t = dt.datetime.strptime("09:15", "%H:%M")
while t <= dt.datetime.strptime("15:15", "%H:%M"):
    e = min(t + dt.timedelta(minutes=30), dt.datetime.strptime("15:30", "%H:%M"))
    edges.append((t.strftime("%H:%M"), e.strftime("%H:%M")))
    t += dt.timedelta(minutes=30)
BHRS = {s: (dt.datetime.strptime(e, "%H:%M") - dt.datetime.strptime(s, "%H:%M")).seconds / 3600 for s, e in edges}

days = pd.read_sql("SELECT DISTINCT date(snapshot_time) d FROM option_chain ORDER BY d", con)["d"].tolist()
rows, paths = [], []
for d in days:
    exps = pd.read_sql(f"SELECT DISTINCT expiry_date e FROM option_chain WHERE snapshot_time BETWEEN '{d}T09:00' AND '{d}T15:31'", con)["e"].dropna().tolist()
    exps = sorted(exps)
    if not exps:
        continue
    front = next((e for e in exps if e[:10] >= d), exps[0])
    dte = (dt.date.fromisoformat(front[:10]) - dt.date.fromisoformat(d)).days
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
    m = pd.DataFrame({"hm": mins, "prem": [ce.loc[x, "ltp"] + pe.loc[x, "ltp"] for x in mins],
                      "iv": [(ce.loc[x, "iv"] + pe.loc[x, "iv"]) / 2 for x in mins],
                      "spot": [(ce.loc[x, "spot"] + pe.loc[x, "spot"]) / 2 for x in mins]}).set_index("hm")
    for x in mins:
        paths.append(dict(day=d, dte=dte, hm=x, prem=m.loc[x, "prem"], iv=m.loc[x, "iv"], spot=m.loc[x, "spot"]))
    for s, e in edges:
        seg = m[(m.index >= s) & (m.index < e)]
        if len(seg) < 3:
            continue
        p0, p1 = seg["prem"].iloc[0], seg["prem"].iloc[-1]
        sp = seg["spot"]
        rows.append(dict(day=d, dte=dte, bucket=s,
                         decay_hr=(p1 - p0) / p0 * 100 / BHRS[s],
                         rng_hr=(sp.max() - sp.min()) / BHRS[s],
                         absnet=abs(sp.iloc[-1] - sp.iloc[0]),
                         iv_chg=seg["iv"].iloc[-1] - seg["iv"].iloc[0]))
pd.DataFrame(paths).to_csv(OUT / "fullday_minute_paths.csv", index=False)
R = pd.DataFrame(rows)
R.to_csv(OUT / "bucket_metrics.csv", index=False)
border = [s for s, _ in edges]


def show(sub, label):
    a = sub.groupby("bucket").agg(decay_hr=("decay_hr", "mean"), rng_hr=("rng_hr", "mean"),
                                  absnet=("absnet", "mean"), iv=("iv_chg", "mean"), n=("day", "nunique")).reindex(border)
    print(f"\n=== {label} — 30-min buckets (decay%/hr negative=faster decay) ===")
    print(f"{'bucket':7}{'decay%/hr':>10}{'rng/hr':>8}{'|net|':>7}{'ivΔ':>7}{'n':>4}")
    for b in border:
        if b in a.index and not pd.isna(a.loc[b, "n"]):
            r = a.loc[b]
            bar = "#" * int(min(abs(r["rng_hr"]) / 6, 30))
            print(f"{b:7}{r['decay_hr']:10.1f}{r['rng_hr']:8.0f}{r['absnet']:7.0f}{r['iv']:7.2f}{int(r['n']):4}  {bar}")


show(R, "ALL DAYS")
show(R[R.dte <= 1], "DTE 0-1 (near expiry)")
show(R[R.dte >= 2], "DTE 2+ (far)")
print("\nDONE — read the profile: slowest-decay + smallest rng/|net| buckets = the true lull; bursts at open/close.")
