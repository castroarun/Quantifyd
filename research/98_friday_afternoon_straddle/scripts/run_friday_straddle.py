# -*- coding: utf-8 -*-
"""research/98 — Friday afternoon NIFTY ATM short straddle. Enter 14:00, exit 15:05. Try every
management style Arun listed: no-mgmt hold, combined-premium SL, per-leg SL (close both / close hit
leg + hold other / close hit leg + trail other). Net-of-cost (ltp +/- slippage + brokerage), on the
real per-minute chain. Small sample (14 Fridays) -> exploratory (SIGNAL hunt), not a validated edge."""
import sqlite3, os, csv
from collections import defaultdict
import numpy as np
from datetime import date

DB = "backtest_data/options_data.db"
SYM = "NIFTY"
LOT = 75
QTY = LOT * 1                     # 1 lot; report per-lot Rs
BROK = 25                         # Rs/order
SLIP = float(os.environ.get("SLIP", "0.01"))
ENTRY, EXIT = "14:00", "15:05"
TRAIL_RET = 0.25                  # surviving-leg trail: exit if it bounces 25% off its running low
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "friday_cells.csv")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

con = sqlite3.connect("file:%s?mode=ro" % DB, uri=True)
con.row_factory = sqlite3.Row

spot_rows = con.execute("SELECT snapshot_time, spot_price FROM underlying_spot "
                        "WHERE symbol=? AND spot_price>0 ORDER BY snapshot_time", (SYM,)).fetchall()
spot_by_day = defaultdict(list)
for r in spot_rows:
    spot_by_day[r["snapshot_time"][:10]].append((r["snapshot_time"][11:16], r["spot_price"]))
expiries = sorted({r["expiry_date"][:10] for r in
                   con.execute("SELECT DISTINCT expiry_date FROM option_chain WHERE symbol=?", (SYM,))})


def at(series, hhmm):
    for s in series:
        if s[0] >= hhmm:
            return s
    return None


def leg_series(day, exp, strike, typ):
    rows = con.execute(
        "SELECT snapshot_time, ltp, bid, ask, volume, oi FROM option_chain "
        "WHERE symbol=? AND expiry_date LIKE ? AND strike=? AND instrument_type=? "
        "AND substr(snapshot_time,1,10)=? ORDER BY snapshot_time", (SYM, exp + "%", strike, typ, day)).fetchall()
    return [(r["snapshot_time"][11:16], r["ltp"] or 0, r["volume"] or 0, r["oi"] or 0) for r in rows]


def net_leg(entry_ltp, exit_ltp):
    sell = entry_ltp * (1 - SLIP); buy = exit_ltp * (1 + SLIP)
    return (sell - buy) * QTY - 2 * BROK


# ---- management sims: each returns net Rs for the straddle on one Friday ----
def sim_hold(path, ce_e, pe_e):
    last = path[-1]
    return net_leg(ce_e, last[1]) + net_leg(pe_e, last[2])


def sim_comb(path, ce_e, pe_e, x):
    thr = (ce_e + pe_e) * (1 + x)
    for _, ce, pe in path:
        if ce + pe >= thr:
            return net_leg(ce_e, ce) + net_leg(pe_e, pe)
    return sim_hold(path, ce_e, pe_e)


def sim_both(path, ce_e, pe_e, x):
    for _, ce, pe in path:
        if ce >= ce_e * (1 + x) or pe >= pe_e * (1 + x):
            return net_leg(ce_e, ce) + net_leg(pe_e, pe)
    return sim_hold(path, ce_e, pe_e)


def sim_other(path, ce_e, pe_e, x):
    ceo = peo = True; cex = pex = None
    for _, ce, pe in path:
        if ceo and ce >= ce_e * (1 + x): cex = ce; ceo = False
        if peo and pe >= pe_e * (1 + x): pex = pe; peo = False
    last = path[-1]
    if ceo: cex = last[1]
    if peo: pex = last[2]
    return net_leg(ce_e, cex) + net_leg(pe_e, pex)


def sim_trail(path, ce_e, pe_e, x):
    ceo = peo = True; cex = pex = None; trailing = None; runmin = None
    for _, ce, pe in path:
        if ceo and ce >= ce_e * (1 + x):
            cex = ce; ceo = False
            if peo and trailing is None: trailing = "pe"; runmin = pe
        if peo and pe >= pe_e * (1 + x):
            pex = pe; peo = False
            if ceo and trailing is None: trailing = "ce"; runmin = ce
        if trailing == "pe" and peo:
            runmin = min(runmin, pe)
            if pe >= runmin * (1 + TRAIL_RET): pex = pe; peo = False
        if trailing == "ce" and ceo:
            runmin = min(runmin, ce)
            if ce >= runmin * (1 + TRAIL_RET): cex = ce; ceo = False
    last = path[-1]
    if ceo: cex = last[1]
    if peo: pex = last[2]
    return net_leg(ce_e, cex) + net_leg(pe_e, pex)


VARIANTS = [("hold", None)]
for x in (0.20, 0.30, 0.40):
    VARIANTS += [("comb_%d" % (x * 100), ("comb", x)), ("both_%d" % (x * 100), ("both", x)),
                 ("other_%d" % (x * 100), ("other", x)), ("trail_%d" % (x * 100), ("trail", x))]
SIM = {"comb": sim_comb, "both": sim_both, "other": sim_other, "trail": sim_trail}

days = sorted(d for d in spot_by_day if date(*map(int, d.split("-"))).weekday() == 4)
res = defaultdict(list); perfri = []
for day in days:
    fut = [e for e in expiries if e >= day]
    if not fut:
        continue
    exp = fut[0]; dte = (date(*map(int, exp.split("-"))) - date(*map(int, day.split("-")))).days
    s0 = at(spot_by_day[day], ENTRY)
    if not s0:
        continue
    atm = round(s0[1] / 50) * 50
    ce = leg_series(day, exp, atm, "CE"); pe = leg_series(day, exp, atm, "PE")
    ce0, pe0 = at(ce, ENTRY), at(pe, ENTRY)
    if not ce0 or not pe0 or ce0[1] <= 0 or pe0[1] <= 0:
        continue
    if (ce0[2] <= 0 and ce0[3] <= 0) or (pe0[2] <= 0 and pe0[3] <= 0):
        continue
    ce_e, pe_e = ce0[1], pe0[1]
    # build aligned path 14:00..15:05
    cem = {t: l for t, l, _, _ in ce}; pem = {t: l for t, l, _, _ in pe}
    mins = sorted(t for t in cem if ENTRY <= t <= EXIT and t in pem and cem[t] > 0 and pem[t] > 0)
    path = [(t, cem[t], pem[t]) for t in mins]
    if len(path) < 10:
        continue
    row = {"day": day, "dte": dte, "atm": atm, "credit": round(ce_e + pe_e, 1)}
    for name, spec in VARIANTS:
        pnl = sim_hold(path, ce_e, pe_e) if spec is None else SIM[spec[0]](path, ce_e, pe_e, spec[1])
        res[name].append(pnl); row[name] = round(pnl)
    perfri.append(row)

# ---- report ----
print("Friday 14:00 -> 15:05 NIFTY ATM short straddle — management bake-off (1 lot=%d, net, %d%% slip)" % (QTY, SLIP * 100))
print("Fridays used: %d  (DTE~4)\n" % len(perfri))
print("%-10s %5s %9s %8s %8s %9s %9s %6s" % ("variant", "n", "total", "avg", "median", "worst", "best", "win%"))
rows_out = []
for name, _ in VARIANTS:
    a = np.array(res[name], float)
    if not len(a):
        continue
    rows_out.append((name, len(a), a.sum(), a.mean(), np.median(a), a.min(), a.max(), 100 * (a > 0).mean()))
for r in sorted(rows_out, key=lambda x: -x[3]):
    print("%-10s %5d %+9d %+8d %+8d %+9d %+9d %6.0f" % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]))

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["day", "dte", "atm", "credit"] + [n for n, _ in VARIANTS])
    w.writeheader()
    for row in perfri:
        w.writerow(row)
print("\nper-Friday detail ->", OUT)
print("\n%-12s %6s %7s %8s %8s %8s" % ("Friday", "credit", "hold", "both_30", "other_30", "trail_30"))
for row in perfri:
    print("%-12s %6.0f %+7d %+8d %+8d %+8d" % (row["day"], row["credit"], row["hold"], row.get("both_30", 0), row.get("other_30", 0), row.get("trail_30", 0)))
