# -*- coding: utf-8 -*-
"""research/97 Layer A — SENSEX per-system stop calibration on real SENSEX chains.
Short ATM straddle entered 09:16, held to 15:15, one-and-done stop (isolates stop quality).
Net-of-cost: SELL at bid / BUY-back at ask (falls back to ltp +/- modelled half-spread), + brokerage.
Liquidity guard: entry strike must have volume>0 or oi>0. Split DTE0 (Thu expiry) vs DTE1 (Wed).
Stops: move0.4 (current ATM2), rupee_X {1000..5000}/lot, legSL30 (current ATM/ATM4), netpct_Y.
Rank by net Rs/lot per trade with worst-trade tail. This answers: what stop for SENSEX ATM2, and
is the 30% per-leg SL ok on expiry."""
import sqlite3, os, csv
from collections import defaultdict
import numpy as np

DB = "backtest_data/options_data.db"
LOT, LOTS = 20, 2
QTY = LOT * LOTS                      # 40 contracts/leg
BROK = 80                            # ~Rs/leg/side (BFO approx)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "layerA_cells.csv")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

con = sqlite3.connect("file:%s?mode=ro" % DB, uri=True)
con.row_factory = sqlite3.Row

# --- day list + spot path (SENSEX) ---
spot_rows = con.execute("SELECT snapshot_time, spot_price FROM underlying_spot "
                        "WHERE symbol='SENSEX' AND spot_price>0 ORDER BY snapshot_time").fetchall()
spot_by_day = defaultdict(list)
for r in spot_rows:
    spot_by_day[r["snapshot_time"][:10]].append((r["snapshot_time"][11:16], r["spot_price"]))

expiries = sorted({r["expiry_date"][:10] for r in
                   con.execute("SELECT DISTINCT expiry_date FROM option_chain WHERE symbol='SENSEX'")})


def nearest_expiry(day):
    fut = [e for e in expiries if e >= day]
    return fut[0] if fut else None


def dte_of(day, exp):
    from datetime import date
    d = date(*map(int, day.split("-"))); e = date(*map(int, exp.split("-")))
    return (e - d).days


def leg_series(day, exp, strike, typ):
    """minute series [(hhmm, ltp, bid, ask, vol, oi)] for one option that day."""
    rows = con.execute(
        "SELECT snapshot_time, ltp, bid, ask, volume, oi FROM option_chain "
        "WHERE symbol='SENSEX' AND expiry_date LIKE ? AND strike=? AND instrument_type=? "
        "AND substr(snapshot_time,1,10)=? ORDER BY snapshot_time",
        (exp + "%", strike, typ, day)).fetchall()
    out = []
    for r in rows:
        out.append((r["snapshot_time"][11:16], r["ltp"] or 0, r["bid"] or 0, r["ask"] or 0,
                    r["volume"] or 0, r["oi"] or 0))
    return out


def at(series, hhmm):
    """value at or just after hhmm."""
    for s in series:
        if s[0] >= hhmm:
            return s
    return None


STOPS = ["move0.4"] + ["rupee_%d" % x for x in (1000, 1500, 2000, 2500, 3000, 4000, 5000)] \
        + ["legSL30"] + ["netpct_%d" % y for y in (50, 75, 100, 150)]

# results[(stop, dte)] = list of net pnl per trade
res = defaultdict(list)
detail = []  # per (day,dte,stop) for audit
days = sorted(spot_by_day.keys())
n_used = 0
for day in days:
    exp = nearest_expiry(day)
    if not exp:
        continue
    dte = dte_of(day, exp)
    if dte not in (0, 1):
        continue
    sp = spot_by_day[day]
    s0 = at(sp, "09:16")
    if not s0:
        continue
    spot0 = s0[1]
    atm = round(spot0 / 100) * 100
    ce = leg_series(day, exp, atm, "CE")
    pe = leg_series(day, exp, atm, "PE")
    ce0, pe0 = at(ce, "09:16"), at(pe, "09:16")
    if not ce0 or not pe0:
        continue
    # liquidity guard at entry
    if (ce0[4] <= 0 and ce0[5] <= 0) or (pe0[4] <= 0 and pe0[5] <= 0):
        continue
    # v2 cost model: fill at LTP (reliable) minus modelled slippage per side (bid/ask too noisy:
    # 35% missing ask, p90 spread 60% of ltp). SLIP env override for cost-sensitivity.
    SLIP = float(os.environ.get("SLIP", "0.01"))

    def _ltp(x):  # x = leg tuple (hhmm,ltp,bid,ask,vol,oi); fall back to mid if ltp==0
        return x[1] if x[1] > 0 else ((x[2] + x[3]) / 2 if x[2] and x[3] else x[1])
    ce_sell = _ltp(ce0) * (1 - SLIP)
    pe_sell = _ltp(pe0) * (1 - SLIP)
    if ce_sell <= 0 or pe_sell <= 0:
        continue
    credit = ce_sell + pe_sell
    n_used += 1

    # build aligned minute grid from 09:16..15:15
    def mtm_at(hhmm):
        c, p = at(ce, hhmm), at(pe, hhmm)
        s = at(sp, hhmm)
        if not c or not p or not s:
            return None
        c_ltp = c[1] if c[1] > 0 else (c[2] + c[3]) / 2 if c[2] and c[3] else c[1]
        p_ltp = p[1] if p[1] > 0 else (p[2] + p[3]) / 2 if p[2] and p[3] else p[1]
        return s[1], c, p, c_ltp, p_ltp

    grid = [t for t in [x[0] for x in ce] if "09:16" <= t <= "15:15"]
    grid = sorted(set(grid))

    def buyback(c, p):
        cb = _ltp(c) * (1 + SLIP)   # BUY-back at ltp + slippage
        pb = _ltp(p) * (1 + SLIP)
        return cb + pb

    def net_pnl(exit_buy):
        return round((credit - exit_buy) * QTY - 4 * BROK)   # 2 legs x 2 sides

    # evaluate each stop as one-and-done
    for stop in STOPS:
        exit_buy = None
        for t in grid:
            m = mtm_at(t)
            if m is None:
                continue
            spot_t, c, p, c_ltp, p_ltp = m
            cur = c_ltp + p_ltp
            trig = False
            if stop == "move0.4":
                trig = abs(spot_t - spot0) / spot0 >= 0.004
            elif stop.startswith("rupee_"):
                X = int(stop.split("_")[1])
                trig = (credit - cur) * QTY <= -(X * LOTS)
            elif stop == "legSL30":
                trig = c_ltp >= 1.3 * ce_sell or p_ltp >= 1.3 * pe_sell
            elif stop.startswith("netpct_"):
                Y = int(stop.split("_")[1])
                trig = (cur - credit) >= (Y / 100.0) * credit
            if trig:
                exit_buy = buyback(c, p); break
        if exit_buy is None:
            lc, lp = at(ce, "15:15"), at(pe, "15:15")
            if lc and lp:
                exit_buy = buyback(lc, lp)
            else:
                exit_buy = ce[-1][3] or ce[-1][1] + pe[-1][3] or pe[-1][1]
        pnl = net_pnl(exit_buy)
        res[(stop, dte)].append(pnl)
    detail.append((day, dte, atm, round(credit, 1)))

# --- write cells + print ---
with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["stop", "dte", "n", "total", "avg_per_trade", "avg_per_lot", "median", "worst", "p10", "win%"])
    for stop in STOPS:
        for dte in (0, 1):
            a = np.array(res.get((stop, dte), []), float)
            if not len(a):
                continue
            w.writerow([stop, dte, len(a), int(a.sum()), int(a.mean()), int(a.mean() / LOTS),
                        int(np.median(a)), int(a.min()), int(np.percentile(a, 10)),
                        round(100 * (a > 0).mean())])

print("SENSEX Layer-A stop calibration — ATM straddle 09:16, 2 lots (QTY %d), net, one-and-done" % QTY)
print("entries used: %d  (DTE0 + DTE1 across cycles)" % n_used)
for dte in (0, 1):
    print("\n=== DTE%d ===" % dte)
    print("%-12s %4s %9s %9s %9s %9s %9s %6s" % ("stop", "n", "total", "avg/tr", "avg/lot", "median", "worst", "win%"))
    rows = []
    for stop in STOPS:
        a = np.array(res.get((stop, dte), []), float)
        if not len(a):
            continue
        rows.append((stop, len(a), a.sum(), a.mean(), a.mean() / LOTS, np.median(a), a.min(), 100 * (a > 0).mean()))
    for r in sorted(rows, key=lambda x: -x[3]):
        print("%-12s %4d %+9d %+9d %+9d %+9d %+9d %6.0f" % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]))
print("\nwrote", OUT)
