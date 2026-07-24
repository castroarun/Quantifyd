"""research/75 Phase 3 — does SKIPPING the lull beat holding all day for the short 09:16 ATM straddle?
A = hold 09:16->15:15. B = close at 11:15, re-enter fresh ATM at 13:00, hold to 15:15. Bcond = only
skip the lull if the morning is in profit at 11:15. Net of the extra round-trip slippage. Fixed-strike
premiums (held position). Read-only on options_data.db. Per lot = 65; short: P&L = credit - buyback."""
import sqlite3, datetime as dt
import numpy as np, pandas as pd
from pathlib import Path

DB = "/home/arun/quantifyd/backtest_data/options_data.db"
OUT = Path("/home/arun/quantifyd/research/75_intraday_time_sections/results")
con = sqlite3.connect(DB)
LOT = 65


def prem(d, front, strike, hhmm):
    q = (f"SELECT instrument_type, AVG(ltp) v FROM option_chain WHERE snapshot_time BETWEEN "
         f"'{d}T{hhmm}:00' AND '{d}T{hhmm}:59' AND expiry_date='{front}' "
         f"AND CAST(ROUND(strike) AS INT)={strike} AND instrument_type IN ('CE','PE') GROUP BY instrument_type")
    r = dict(con.execute(q).fetchall())
    if r.get('CE') and r.get('PE'):
        return r['CE'] + r['PE']
    return None


def spot(d, front, hhmm):
    v = con.execute(f"SELECT AVG(underlying_spot) FROM option_chain WHERE snapshot_time BETWEEN "
                    f"'{d}T{hhmm}:00' AND '{d}T{hhmm}:59' AND expiry_date='{front}'").fetchone()[0]
    return v


days = pd.read_sql("SELECT DISTINCT date(snapshot_time) d FROM option_chain ORDER BY d", con)["d"].tolist()
rows = []
for d in days:
    exps = pd.read_sql(f"SELECT DISTINCT expiry_date e FROM option_chain WHERE snapshot_time BETWEEN '{d}T09:00' AND '{d}T15:31'", con)["e"].dropna().tolist()
    exps = sorted(exps)
    if not exps:
        continue
    front = next((e for e in exps if e[:10] >= d), exps[0])
    dte = (dt.date.fromisoformat(front[:10]) - dt.date.fromisoformat(d)).days
    s0 = spot(d, front, '09:16')
    if not s0 or not (20000 < s0 < 28000):
        continue
    atm0 = int(round(s0 / 50) * 50)
    cred0, buy11, exit15 = prem(d, front, atm0, '09:16'), prem(d, front, atm0, '11:15'), prem(d, front, atm0, '15:15')
    s1 = spot(d, front, '13:00')
    atm1 = int(round(s1 / 50) * 50) if s1 and 20000 < s1 < 28000 else None
    cred1 = prem(d, front, atm1, '13:00') if atm1 else None
    exit1 = prem(d, front, atm1, '15:15') if atm1 else None
    if None in (cred0, buy11, exit15, cred1, exit1):
        continue
    rows.append(dict(day=d, dte=dte, atm0=atm0, atm1=atm1, cred0=cred0, buy11=buy11, exit15=exit15, cred1=cred1, exit1=exit1))

R = pd.DataFrame(rows)
R["A"] = (R.cred0 - R.exit15) * LOT
R["Bg"] = ((R.cred0 - R.buy11) + (R.cred1 - R.exit1)) * LOT
R["prof11"] = R.cred0 > R.buy11
print("days:", len(R), "| re-centered strike (atm1!=atm0) on %d days" % (R.atm0 != R.atm1).sum())


def summ(x):
    return "tot Rs%+.0f  mean Rs%+.0f/day  win%%%.0f" % (x.sum(), x.mean(), 100 * (x > 0).mean())


for s in [0.0, 0.001, 0.0015, 0.002]:
    extra = (R.buy11 + R.cred1) * LOT * s      # 4 extra legs (buy 2 @11:15 + sell 2 @13:00)
    B = R.Bg - extra
    Bc = np.where(R.prof11, B, R.A)
    print("\n=== slippage %.2f%%/leg ===" % (s * 100))
    print("  A (hold all day):  %s" % summ(R.A))
    print("  B (skip lull):     %s   |  B - A = Rs%+.0f total" % (summ(B), (B - R.A).sum()))
    print("  Bcond (skip if +): %s   |  Bc - A = Rs%+.0f total" % (summ(pd.Series(Bc)), (Bc - R.A).sum()))
    if s == 0.0015:
        print("  -- by DTE (B-A per day, +ve = skipping helps) --")
        for lbl, m in [("DTE 0-1", R.dte <= 1), ("DTE 2+", R.dte >= 2)]:
            dd = (B - R.A)[m]
            print("     %-8s B-A mean Rs%+.0f/day (n=%d)  |  A mean Rs%+.0f  B mean Rs%+.0f" % (lbl, dd.mean(), m.sum(), R.A[m].mean(), B[m].mean()))
print("\nDONE")
