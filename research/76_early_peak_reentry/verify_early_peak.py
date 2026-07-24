"""research/76 — VERIFY the "early-peak then erode" hypothesis for the 09:16 ATM short straddle.
User observation: the trade goes into profit in the first 30-60 min, then volatility erodes it.
Reconstructs the 09:16 ATM straddle P&L path across ALL recorded days (options_data.db), fixed-strike
(held), from the reliable underlying_spot TABLE + ±3-min premium windows. P&L(T) = (credit_0916 - prem_T)*65
per lot. Read-only. Reports mean path, per-day peak timing, and the early-peak-vs-EOD give-back."""
import sqlite3, datetime as dt
import numpy as np, pandas as pd

DB = "/home/arun/quantifyd/backtest_data/options_data.db"
con = sqlite3.connect(DB)
LOT = 65
TIMES = ["09:16", "09:31", "09:46", "10:16", "10:46", "11:15", "12:00", "13:00", "14:00", "15:00", "15:15"]


def win(hhmm, b=2, a=3):
    t = dt.datetime.strptime(hhmm, "%H:%M")
    return (t - dt.timedelta(minutes=b)).strftime("%H:%M"), (t + dt.timedelta(minutes=a)).strftime("%H:%M")


def spot(d, hhmm):
    lo, hi = win(hhmm)
    return con.execute(f"SELECT AVG(spot_price) FROM underlying_spot WHERE snapshot_time BETWEEN "
                       f"'{d}T{lo}:00' AND '{d}T{hi}:00' AND spot_price BETWEEN 20000 AND 28000").fetchone()[0]


def prem(d, front, strike, hhmm):
    lo, hi = win(hhmm)
    r = dict(con.execute(f"SELECT instrument_type, AVG(ltp) FROM option_chain WHERE snapshot_time BETWEEN "
                         f"'{d}T{lo}:00' AND '{d}T{hi}:00' AND expiry_date='{front}' "
                         f"AND CAST(ROUND(strike) AS INT)={strike} AND instrument_type IN ('CE','PE') "
                         f"GROUP BY instrument_type").fetchall())
    return (r['CE'] + r['PE']) if (r.get('CE') and r.get('PE')) else None


days = pd.read_sql("SELECT DISTINCT date(snapshot_time) d FROM option_chain ORDER BY d", con)["d"].tolist()
rows = []
for d in days:
    exps = pd.read_sql(f"SELECT DISTINCT expiry_date e FROM option_chain WHERE snapshot_time BETWEEN '{d}T09:00' AND '{d}T15:31'", con)["e"].dropna().tolist()
    exps = sorted(exps)
    if not exps:
        continue
    front = next((e for e in exps if e[:10] >= d), exps[0])
    dte = (dt.date.fromisoformat(front[:10]) - dt.date.fromisoformat(d)).days
    s0 = spot(d, "09:16")
    if not s0:
        continue
    atm = int(round(s0 / 50) * 50)
    cred = prem(d, front, atm, "09:16")
    if not cred:
        continue
    path = {}
    ok = True
    for t in TIMES:
        p = prem(d, front, atm, t)
        if p is None:
            ok = False
            break
        path[t] = (cred - p) * LOT   # short-straddle MTM P&L per lot
    if not ok:
        continue
    rows.append(dict(day=d, dte=dte, atm=atm, cred=cred, **{f"t_{t}": path[t] for t in TIMES}))

R = pd.DataFrame(rows)
print(f"days reconstructed: {len(R)}  (DTE0-1: {(R.dte<=1).sum()}, DTE2+: {(R.dte>=2).sum()})")
print("\n=== mean P&L path (per 1 lot=65), across all days ===")
print("  time    meanP&L   %days>0   median")
for t in TIMES:
    col = R[f"t_{t}"]
    print("  %-6s  %+7.0f    %3.0f%%    %+7.0f" % (t, col.mean(), 100*(col > 0).mean(), col.median()))

# peak timing: which time bucket holds each day's max P&L
peakcol = R[[f"t_{t}" for t in TIMES]].idxmax(axis=1).str[2:]
print("\n=== per-day PEAK P&L time (where the max occurs) ===")
vc = peakcol.value_counts().reindex(TIMES).fillna(0).astype(int)
for t in TIMES:
    print("  %-6s  %2d days (%3.0f%%)" % (t, vc[t], 100*vc[t]/len(R)))
early = peakcol.isin(["09:16", "09:31", "09:46", "10:16"])
print("  --> peak within first 60 min (<=10:16): %d/%d = %.0f%%" % (early.sum(), len(R), 100*early.mean()))

# early-peak vs EOD give-back
R["peak60"] = R[["t_09:31", "t_09:46", "t_10:16"]].max(axis=1)   # best of +15/+30/+60
R["eod"] = R["t_15:15"]
R["giveback"] = R["peak60"] - R["eod"]
print("\n=== early-peak (best of +15/30/60min) vs EOD (15:15) ===")
print("  mean peak60 = %+.0f | mean EOD = %+.0f | mean give-back = %+.0f/lot" % (R.peak60.mean(), R.eod.mean(), R.giveback.mean()))
print("  days where peak60 > EOD (erosion happened): %d/%d = %.0f%%" % ((R.giveback > 0).sum(), len(R), 100*(R.giveback > 0).mean()))
for lbl, m in [("DTE0-1", R.dte <= 1), ("DTE2+", R.dte >= 2)]:
    s = R[m]
    print("  %-7s n=%2d  peak60 %+.0f  EOD %+.0f  give-back %+.0f  erosion-days %.0f%%" % (
        lbl, m.sum(), s.peak60.mean(), s.eod.mean(), s.giveback.mean(), 100*(s.giveback > 0).mean()))
R.to_csv("/home/arun/quantifyd/research/76_early_peak_reentry/results/paths.csv", index=False)
print("\nsaved paths.csv | DONE")
