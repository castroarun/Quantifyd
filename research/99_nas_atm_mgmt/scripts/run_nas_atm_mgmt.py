# -*- coding: utf-8 -*-
"""research/99 — NAS 9:16 ATM straddle management bake-off on the recorded NIFTY chain (~70 days).
System: 09:16 sell ATM CE+PE, per-leg SL, 1st SL closes the stopped leg, survivor managed. Test:
  SL level {30,40,50}% x survivor policy {close-both, hold-naked, trail-20, trail-30}, + hold baseline.
'trail_X' = retrace stop on the survivor (exit if it rises X% off its running low after the SL) — a clean
proxy for the live ST(7,3) naked-leg trail. Net-of-cost. Reports ALL days + the SL-FIRED subset (where
today's −₹6,857 damage happened)."""
import sqlite3, os
from collections import defaultdict
import numpy as np
from datetime import date

DB = "backtest_data/options_data.db"; SYM = "NIFTY"; LOT = 75; QTY = LOT
BROK = 25; SLIP = float(os.environ.get("SLIP", "0.01"))
ENTRY, EOD = "09:16", "15:15"

con = sqlite3.connect("file:%s?mode=ro" % DB, uri=True)
spot = defaultdict(dict)
for st, sp in con.execute("SELECT snapshot_time,spot_price FROM underlying_spot WHERE symbol=? AND spot_price>0", (SYM,)):
    spot[st[:10]][st[11:16]] = sp
exps = sorted({r[0][:10] for r in con.execute("SELECT DISTINCT expiry_date FROM option_chain WHERE symbol=?", (SYM,))})


def spot_at(day, hhmm):
    ks = sorted(k for k in spot[day] if k >= hhmm)
    return spot[day][ks[0]] if ks else None


def load_day(day, exp):
    d = defaultdict(dict)
    for st, strike, typ, ltp in con.execute(
        "SELECT snapshot_time,strike,instrument_type,ltp FROM option_chain "
        "WHERE symbol=? AND snapshot_time BETWEEN ? AND ? AND expiry_date LIKE ?",
        (SYM, day + "T09:15:00", day + "T15:20:00", exp + "%")):
        d[(int(strike), typ)][st[11:16]] = ltp or 0
    return d


def net_leg(e, x):
    return (e * (1 - SLIP) - x * (1 + SLIP)) * QTY - 2 * BROK


def sim(path, ce_e, pe_e, slpct, policy, ret=None):
    ce_sl, pe_sl = ce_e * (1 + slpct), pe_e * (1 + slpct)
    ceo = peo = True; cex = pex = None; fired = False; surv = None; smin = None
    for _, ce, pe in path:
        if not fired:
            hit = "ce" if (ceo and ce >= ce_sl) else ("pe" if (peo and pe >= pe_sl) else None)
            if hit:
                fired = True
                if hit == "ce":
                    cex = ce; ceo = False; surv = "pe"; smin = pe
                else:
                    pex = pe; peo = False; surv = "ce"; smin = ce
                if policy == "closeboth":
                    if ceo: cex = ce; ceo = False
                    if peo: pex = pe; peo = False
        else:
            if policy == "trail":
                if surv == "ce" and ceo:
                    smin = min(smin, ce)
                    if ce >= smin * (1 + ret): cex = ce; ceo = False
                elif surv == "pe" and peo:
                    smin = min(smin, pe)
                    if pe >= smin * (1 + ret): pex = pe; peo = False
    last = path[-1]
    if ceo: cex = last[1]
    if peo: pex = last[2]
    return net_leg(ce_e, cex) + net_leg(pe_e, pex), fired


VARIANTS = [("hold", None)]
for sl in (0.30, 0.40, 0.50):
    s = int(sl * 100)
    VARIANTS += [("sl%d_closeboth" % s, (sl, "closeboth", None)),
                 ("sl%d_holdnaked" % s, (sl, "holdnaked", None)),
                 ("sl%d_trail20" % s, (sl, "trail", 0.20)),
                 ("sl%d_trail30" % s, (sl, "trail", 0.30))]

res = defaultdict(list); res_fired = defaultdict(list); nfired = defaultdict(int)
days = sorted(spot.keys())
for day in days:
    fut = [e for e in exps if e >= day]
    if not fut:
        continue
    exp = fut[0]
    sE = spot_at(day, ENTRY)
    if not sE:
        continue
    atm = round(sE / 50) * 50
    chain = load_day(day, exp)
    ce, pe = chain.get((atm, "CE"), {}), chain.get((atm, "PE"), {})
    mins = sorted(t for t in ce if ENTRY <= t <= EOD and t in pe and ce[t] > 0 and pe[t] > 0)
    if len(mins) < 30:
        continue
    ce_e, pe_e = ce[mins[0]], pe[mins[0]]
    path = [(t, ce[t], pe[t]) for t in mins]
    for name, spec in VARIANTS:
        if spec is None:
            last = path[-1]; pnl = net_leg(ce_e, last[1]) + net_leg(pe_e, last[2]); fired = False
        else:
            pnl, fired = sim(path, ce_e, pe_e, spec[0], spec[1], spec[2])
        res[name].append(pnl)
        if name.startswith("sl30") and spec:
            nfired["sl30"] += 1 if fired else 0
        if fired:
            res_fired[name].append(pnl)

N = len(res["hold"])
print("NAS 9:16 ATM straddle management bake-off — %d days, net, 1 lot=%d, %d%% slip" % (N, QTY, SLIP * 100))
print("(sl30 first-leg SL fired on %d of %d days)\n" % (nfired.get("sl30", 0), N))
print("=== ALL DAYS ===")
print("%-16s %9s %8s %8s %9s %9s %6s" % ("variant", "total", "avg", "median", "worst", "p10", "win%"))
for name, _ in VARIANTS:
    a = np.array(res[name], float)
    print("%-16s %+9d %+8d %+8d %+9d %+9d %6.0f" % (name, a.sum(), a.mean(), np.median(a), a.min(), np.percentile(a, 10), 100 * (a > 0).mean()))

print("\n=== SL-FIRED DAYS ONLY (where the survivor management matters) ===")
print("%-16s %5s %9s %8s %9s %9s %6s" % ("variant", "n", "total", "avg", "worst", "p10", "win%"))
for name, spec in VARIANTS:
    if spec is None:
        continue
    a = np.array(res_fired.get(name, []), float)
    if not len(a):
        continue
    print("%-16s %5d %+9d %+8d %+9d %+9d %6.0f" % (name, len(a), a.sum(), a.mean(), a.min(), np.percentile(a, 10), 100 * (a > 0).mean()))
