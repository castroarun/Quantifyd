#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A worked example of the Wed->Fri condor and its 2x-credit stop, on REAL recorded quotes."""
import sqlite3
from datetime import datetime, date

ROOT = "/home/arun/quantifyd"
c = sqlite3.connect(f"file:{ROOT}/backtest_data/options_data.db?mode=ro", uri=True)
c.row_factory = sqlite3.Row
QTY = 130


def chain_at(day, t_from, expiry):
    ts = c.execute("SELECT MAX(snapshot_time) FROM option_chain WHERE symbol='NIFTY' "
                   "AND date(snapshot_time)=? AND time(snapshot_time)<=?", (day, t_from)).fetchone()[0]
    if not ts:
        return None, None, {}
    rows = c.execute("SELECT strike, instrument_type, ltp, underlying_spot FROM option_chain "
                     "WHERE symbol='NIFTY' AND snapshot_time=? AND expiry_date=? AND ltp>0",
                     (ts, expiry)).fetchall()
    spot = next((r["underlying_spot"] for r in rows if r["underlying_spot"]), None)
    book = {(int(r["strike"]), r["instrument_type"]): r["ltp"] for r in rows}
    return ts, spot, book


# find the Wednesdays we have, with their Tuesday expiry
weds = []
for d in [r[0] for r in c.execute("SELECT DISTINCT date(snapshot_time) FROM option_chain "
                                  "WHERE symbol='NIFTY' ORDER BY 1")]:
    if datetime.fromisoformat(d).weekday() != 2:
        continue
    e = c.execute("SELECT MIN(expiry_date) FROM option_chain WHERE symbol='NIFTY' "
                  "AND date(snapshot_time)=? AND expiry_date>=?", (d, d)).fetchone()[0]
    if e:
        weds.append((d, e))

print("Wednesdays available in the recorded chain:", [w[0] for w in weds])

for day, exp in weds[-3:]:
    ts, spot, book = chain_at(day, "15:25:00", exp)
    if not spot:
        continue
    atm = round(spot / 50) * 50
    Ks, Kl = atm + 100, atm + 350          # short CE, long CE
    Ps, Pl = atm - 100, atm - 350          # short PE, long PE
    need = [(Ks, "CE"), (Kl, "CE"), (Ps, "PE"), (Pl, "PE")]
    if any(k not in book for k in need):
        print(f"\n{day}: some strikes absent, skipping")
        continue

    sce, lce = book[(Ks, "CE")], book[(Kl, "CE")]
    spe, lpe = book[(Ps, "PE")], book[(Pl, "PE")]
    credit = (sce + spe) - (lce + lpe)
    stop_at = credit * 2.0
    dte = (datetime.fromisoformat(exp).date() - datetime.fromisoformat(day).date()).days

    print("\n" + "=" * 78)
    print(f"WEDNESDAY {day}  (expiry {exp}, DTE {dte})   NIFTY = {spot:,.0f}   snapshot {ts[11:19]}")
    print("=" * 78)
    print(f"  ATM (rounded to 50) = {atm:,}\n")
    print(f"  {'leg':<26s} {'strike':>7s} {'premium':>9s} {'x130':>11s}")
    print(f"  {'SELL CE (ATM+100)':<26s} {Ks:>7,} {sce:>9.2f} {sce*QTY:>+11,.0f}")
    print(f"  {'SELL PE (ATM-100)':<26s} {Ps:>7,} {spe:>9.2f} {spe*QTY:>+11,.0f}")
    print(f"  {'BUY  CE wing (+250)':<26s} {Kl:>7,} {lce:>9.2f} {-lce*QTY:>+11,.0f}")
    print(f"  {'BUY  PE wing (-250)':<26s} {Pl:>7,} {lpe:>9.2f} {-lpe*QTY:>+11,.0f}")
    print(f"  {'-'*58}")
    print(f"  {'NET CREDIT RECEIVED':<26s} {'':>7s} {credit:>9.2f} {credit*QTY:>+11,.0f}   <- you are paid this")
    print()
    print(f"  STOP LEVEL = 2 x credit = {stop_at:.2f} points")
    print(f"     -> if it would cost {stop_at*QTY:,.0f} to buy the structure back, close all 4 legs.")
    print(f"     -> that is a loss of about {(credit - stop_at)*QTY:+,.0f} (the credit, minus what you pay).")
    print(f"  MAX LOSS if it just sat there to expiry: "
          f"{(credit - 250)*QTY:+,.0f}  (wings cap it at 250 pts wide)")
    print()

    # walk forward to Friday close
    print(f"  {'what it costs to close':<26s} {'spot':>9s} {'structure':>10s} {'vs stop':>18s}")
    for nd in range(1, 3):
        dd = (datetime.fromisoformat(day).date()).toordinal() + nd
        s = date.fromordinal(dd).isoformat()
        ts2, sp2, bk2 = chain_at(s, "15:25:00", exp)
        if not sp2 or any(k not in bk2 for k in need):
            continue
        val = (bk2[(Ks, "CE")] + bk2[(Ps, "PE")]) - (bk2[(Kl, "CE")] + bk2[(Pl, "PE")])
        wd = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][datetime.fromisoformat(s).weekday()]
        hit = "STOPPED OUT" if val >= stop_at else f"safe ({val/credit:.2f}x credit)"
        print(f"  {wd + ' ' + s:<26s} {sp2:>9,.0f} {val:>10.2f} {hit:>18s}")
        if wd == "Fri":
            pnl = (credit - val) * QTY - 4 * 20
            print(f"\n  --> EXIT FRIDAY: sold at {credit:.2f}, bought back at {val:.2f}  "
                  f"=  P&L {pnl:+,.0f} (after Rs20/leg)")
c.close()
