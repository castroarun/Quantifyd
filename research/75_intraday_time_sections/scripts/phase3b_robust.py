"""research/75 Phase 3b — ROBUST re-run (recording is fine; the 7-day loss was my query). Uses the
reliable underlying_spot TABLE for the ATM + ±3-min windows for the fixed-strike premiums. Same
policies: A=hold all day, B=skip lull (close 11:15, re-enter fresh ATM 13:00), Bcond=skip only if in
profit at 11:15. Net of round-trip slippage, by DTE. options_data.db, per lot 65."""
import sqlite3, datetime as dt
import numpy as np, pandas as pd
from pathlib import Path

DB = "/home/arun/quantifyd/backtest_data/options_data.db"
OUT = Path("/home/arun/quantifyd/research/75_intraday_time_sections/results")
con = sqlite3.connect(DB)
LOT = 65


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
    a0 = int(round(s0 / 50) * 50)
    cred0, buy11, exit15 = prem(d, front, a0, "09:16"), prem(d, front, a0, "11:15"), prem(d, front, a0, "15:15")
    s1 = spot(d, "13:00")
    a1 = int(round(s1 / 50) * 50) if s1 else None
    cred1 = prem(d, front, a1, "13:00") if a1 else None
    exit1 = prem(d, front, a1, "15:15") if a1 else None
    if None in (cred0, buy11, exit15, cred1, exit1):
        continue
    rows.append(dict(day=d, dte=dte, a0=a0, a1=a1, cred0=cred0, buy11=buy11, exit15=exit15, cred1=cred1, exit1=exit1))
R = pd.DataFrame(rows)
R["A"] = (R.cred0 - R.exit15) * LOT
R["Bg"] = ((R.cred0 - R.buy11) + (R.cred1 - R.exit1)) * LOT
R["prof11"] = R.cred0 > R.buy11
R.to_csv(OUT / "phase3b_daily.csv", index=False)
print("days recovered: %d  (was 7) | re-centered (a1!=a0) on %d" % (len(R), (R.a0 != R.a1).sum()))


def s(x):
    return "tot %+.0f  mean %+.0f/d  win%%%.0f" % (x.sum(), x.mean(), 100 * (x > 0).mean())


for sl in [0.0, 0.0015, 0.002]:
    extra = (R.buy11 + R.cred1) * LOT * sl
    B = R.Bg - extra
    Bc = pd.Series(np.where(R.prof11, B, R.A))
    print("\n=== slippage %.2f%%/leg ===" % (sl * 100))
    print("  A hold:      %s" % s(R.A))
    print("  B skip:      %s  | B-A tot %+.0f" % (s(B), (B - R.A).sum()))
    print("  Bcond skip+: %s  | Bc-A tot %+.0f" % (s(Bc), (Bc - R.A).sum()))
    if sl == 0.0015:
        for lbl, m in [("DTE0-1", R.dte <= 1), ("DTE2+", R.dte >= 2)]:
            print("     %-7s n=%2d  A %+.0f/d  B %+.0f/d  B-A %+.0f/d  win(B-A>0)%.0f%%" % (
                lbl, m.sum(), R.A[m].mean(), B[m].mean(), (B - R.A)[m].mean(), 100 * ((B - R.A)[m] > 0).mean()))
print("\nDONE")
