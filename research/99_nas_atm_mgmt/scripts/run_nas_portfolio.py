# -*- coding: utf-8 -*-
"""research/99b — Validate the 3 NAS 9:16 ATM systems as a PORTFOLIO. Same 09:16 ATM straddle entry;
the point of each is a DIFFERENT management so they diversify. Model each system's real rule, get daily
P&L series, measure correlation + who-wins-when + portfolio stats. Goal: judge complementarity, and any
change to ATM must not duplicate ATM2/ATM4.
  ATM  : 30% per-leg SL -> close stopped leg -> TRAIL naked survivor (retrace-25% proxy for ST(7,3))
  ATM2 : +/-0.4% underlying move-stop -> close BOTH, one-and-done
  ATM4 : 30% per-leg SL -> ROLL stopped leg to new ATM (max 1) -> hold to EOD
Net-of-cost. ~70 days. Also tests candidate ATM changes and their effect on the PORTFOLIO."""
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


def sim_atm(path, ce_e, pe_e, sl=0.30, ret=0.25):
    ce_sl, pe_sl = ce_e * (1 + sl), pe_e * (1 + sl)
    ceo = peo = True; cex = pex = None; fired = False; surv = None; smin = None
    for _, ce, pe, sp in path:
        if not fired:
            hit = "ce" if (ceo and ce >= ce_sl) else ("pe" if (peo and pe >= pe_sl) else None)
            if hit:
                fired = True
                if hit == "ce": cex = ce; ceo = False; surv = "pe"; smin = pe
                else: pex = pe; peo = False; surv = "ce"; smin = ce
        else:
            if surv == "ce" and ceo:
                smin = min(smin, ce)
                if ce >= smin * (1 + ret): cex = ce; ceo = False
            elif surv == "pe" and peo:
                smin = min(smin, pe)
                if pe >= smin * (1 + ret): pex = pe; peo = False
    last = path[-1]
    if ceo: cex = last[1]
    if peo: pex = last[2]
    return net_leg(ce_e, cex) + net_leg(pe_e, pex)


def sim_movestop(path, ce_e, pe_e, spot0, mv=0.004):
    for _, ce, pe, sp in path:
        if sp and abs(sp - spot0) / spot0 >= mv:
            return net_leg(ce_e, ce) + net_leg(pe_e, pe)
    last = path[-1]
    return net_leg(ce_e, last[1]) + net_leg(pe_e, last[2])


def sim_closeboth(path, ce_e, pe_e, sl=0.30):
    ce_sl, pe_sl = ce_e * (1 + sl), pe_e * (1 + sl)
    for _, ce, pe, sp in path:
        if ce >= ce_sl or pe >= pe_sl:
            return net_leg(ce_e, ce) + net_leg(pe_e, pe)
    last = path[-1]
    return net_leg(ce_e, last[1]) + net_leg(pe_e, last[2])


def sim_hold(path, ce_e, pe_e):
    last = path[-1]
    return net_leg(ce_e, last[1]) + net_leg(pe_e, last[2])


def sim_roll(chain, path, ce_e, pe_e, atm0, sl=0.30):
    """30% SL -> close stopped leg, SELL new same-type leg at current ATM (max 1 roll), hold to EOD."""
    ce_sl, pe_sl = ce_e * (1 + sl), pe_e * (1 + sl)
    ceo = peo = True; cex = pex = None; rolled = False
    roll_e = roll_series = roll_open = None; rtype = None
    for t, ce, pe, sp in path:
        if not rolled:
            hit = "ce" if (ceo and ce >= ce_sl) else ("pe" if (peo and pe >= pe_sl) else None)
            if hit:
                rolled = True
                natm = round(sp / 50) * 50 if sp else atm0
                if hit == "ce":
                    cex = ce; ceo = False; rtype = "CE"
                else:
                    pex = pe; peo = False; rtype = "PE"
                roll_series = chain.get((natm, rtype), {})
                rv = roll_series.get(t)
                if rv and rv > 0:
                    roll_e = rv; roll_open = True
        # survivor + rolled leg just held to EOD (max 1 roll, no further mgmt)
    last = path[-1]
    if ceo: cex = last[1]
    if peo: pex = last[2]
    pnl = net_leg(ce_e, cex) + net_leg(pe_e, pex)
    if roll_open and roll_e:
        # close rolled leg at EOD
        rx = None
        for k in sorted(roll_series, reverse=True):
            if roll_series[k] > 0:
                rx = roll_series[k]; break
        if rx:
            pnl += net_leg(roll_e, rx)
    return pnl


rows = defaultdict(list); days_used = []
for day in sorted(spot.keys()):
    fut = [e for e in exps if e >= day]
    if not fut:
        continue
    exp = fut[0]
    ks = sorted(k for k in spot[day] if k >= ENTRY)
    if not ks:
        continue
    spot0 = spot[day][ks[0]]; atm = round(spot0 / 50) * 50
    chain = load_day(day, exp)
    ce, pe = chain.get((atm, "CE"), {}), chain.get((atm, "PE"), {})
    mins = sorted(t for t in ce if ENTRY <= t <= EOD and t in pe and ce[t] > 0 and pe[t] > 0)
    if len(mins) < 30:
        continue
    ce_e, pe_e = ce[mins[0]], pe[mins[0]]
    path = [(t, ce[t], pe[t], spot[day].get(t)) for t in mins]
    days_used.append(day)
    rows["ATM(trail)"].append(sim_atm(path, ce_e, pe_e))
    rows["ATM2(move)"].append(sim_movestop(path, ce_e, pe_e, spot0))
    rows["ATM4(roll)"].append(sim_roll(chain, path, ce_e, pe_e, atm))
    # candidate ATM replacements (to test portfolio effect / non-duplication)
    rows["cand:ATM_closeboth"].append(sim_closeboth(path, ce_e, pe_e))
    rows["cand:ATM_hold"].append(sim_hold(path, ce_e, pe_e))
    rows["cand:ATM_sl50trail"].append(sim_atm(path, ce_e, pe_e, sl=0.50, ret=0.25))

N = len(days_used)
print("NAS 9:16 ATM PORTFOLIO validation — %d days, net, 1 lot=%d\n" % (N, QTY))


def stats(a):
    a = np.array(a, float)
    return a.sum(), a.mean(), a.std(), a.min(), 100 * (a > 0).mean()


print("%-20s %9s %7s %7s %9s %6s" % ("system", "total", "avg", "std", "worst", "win%"))
for k in ["ATM(trail)", "ATM2(move)", "ATM4(roll)", "cand:ATM_closeboth", "cand:ATM_hold", "cand:ATM_sl50trail"]:
    s = stats(rows[k])
    print("%-20s %+9d %+7d %7d %+9d %6.0f" % (k, s[0], s[1], s[2], s[3], s[4]))

# correlation of the 3 live systems
live = ["ATM(trail)", "ATM2(move)", "ATM4(roll)"]
M = np.array([rows[k] for k in live], float)
C = np.corrcoef(M)
print("\nCorrelation (the 3 live systems):")
print("            " + " ".join("%12s" % k for k in live))
for i, k in enumerate(live):
    print("%-12s" % k + " ".join("%12.2f" % C[i, j] for j in range(3)))

# portfolio = equal-weight sum of the 3 live
port = M.sum(axis=0)
ps = stats(port)
print("\nPORTFOLIO (ATM+ATM2+ATM4): total %+d  avg %+d  std %d  worst %+d  win%% %.0f" % (ps[0], ps[1], ps[2], ps[3], ps[4]))
print("  diversification: portfolio std %d vs sum-of-stds %d (lower = diversified)" %
      (ps[2], sum(np.std(rows[k]) for k in live)))

# portfolio if ATM replaced by each candidate
print("\nPortfolio if ATM(trail) is REPLACED by a candidate (ATM2+ATM4 unchanged):")
for cand in ["cand:ATM_closeboth", "cand:ATM_hold", "cand:ATM_sl50trail"]:
    p2 = np.array(rows[cand], float) + M[1] + M[2]
    s = stats(p2)
    corr_atm2 = np.corrcoef(rows[cand], rows["ATM2(move)"])[0, 1]
    corr_atm4 = np.corrcoef(rows[cand], rows["ATM4(roll)"])[0, 1]
    print("  %-20s port total %+d avg %+d std %d worst %+d win%% %.0f | corr vs ATM2 %.2f ATM4 %.2f" %
          (cand.replace("cand:", ""), s[0], s[1], s[2], s[3], s[4], corr_atm2, corr_atm4))
