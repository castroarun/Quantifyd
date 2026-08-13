# -*- coding: utf-8 -*-
"""research/98b — Friday NIFTY ATM straddle: sweep ENTRY x EXIT window (14:00..15:15), hold only.
Net-of-cost. 14 Fridays. FAST: one range query per Friday via the (symbol,snapshot_time) index."""
import sqlite3, os
from collections import defaultdict
import numpy as np
from datetime import date

DB = "backtest_data/options_data.db"; SYM = "NIFTY"; LOT = 75; QTY = LOT
BROK = 25; SLIP = float(os.environ.get("SLIP", "0.01"))
ENTRIES = ["14:00", "14:15", "14:30", "14:45", "15:00"]
EXITS = ["14:30", "14:45", "15:00", "15:10", "15:15"]

con = sqlite3.connect("file:%s?mode=ro" % DB, uri=True)
spot = defaultdict(dict)
for st, sp in con.execute("SELECT snapshot_time,spot_price FROM underlying_spot WHERE symbol=? AND spot_price>0", (SYM,)):
    spot[st[:10]][st[11:16]] = sp
exps = sorted({r[0][:10] for r in con.execute("SELECT DISTINCT expiry_date FROM option_chain WHERE symbol=?", (SYM,))})


def spot_at(day, hhmm):
    ks = sorted(k for k in spot[day] if k >= hhmm)
    return spot[day][ks[0]] if ks else None


def load_day(day, exp):
    """(strike,type) -> {hhmm: ltp} for 13:55..15:20 that day, nearest expiry only. One indexed query."""
    d = defaultdict(dict)
    for st, strike, typ, ltp in con.execute(
        "SELECT snapshot_time,strike,instrument_type,ltp FROM option_chain "
        "WHERE symbol=? AND snapshot_time BETWEEN ? AND ? AND expiry_date LIKE ?",
        (SYM, day + "T13:55:00", day + "T15:20:00", exp + "%")):
        d[(int(strike), typ)][st[11:16]] = ltp or 0
    return d


def val_at(series, hhmm):
    ks = sorted(k for k in series if k >= hhmm and series[k] > 0)
    return series[ks[0]] if ks else None


def net_leg(e, x):
    return (e * (1 - SLIP) - x * (1 + SLIP)) * QTY - 2 * BROK


res = defaultdict(list)
fridays = [d for d in sorted(spot) if date(*map(int, d.split("-"))).weekday() == 4]
for day in fridays:
    fut = [e for e in exps if e >= day]
    if not fut:
        continue
    exp = fut[0]
    chain = load_day(day, exp)
    for E in ENTRIES:
        sE = spot_at(day, E)
        if not sE:
            continue
        atm = round(sE / 50) * 50
        ce, pe = chain.get((atm, "CE"), {}), chain.get((atm, "PE"), {})
        ceE, peE = val_at(ce, E), val_at(pe, E)
        if not ceE or not peE:
            continue
        for X in EXITS:
            if X <= E:
                continue
            ceX, peX = val_at(ce, X), val_at(pe, X)
            if not ceX or not peX:
                continue
            res[(E, X)].append(net_leg(ceE, ceX) + net_leg(peE, peX))

print("Friday NIFTY ATM straddle — ENTRY x EXIT window (hold, net, 1 lot=%d). cell = avg Rs/lot [win%%,worst]\n" % QTY)
print("entry\\exit " + " ".join("%14s" % x for x in EXITS))
for E in ENTRIES:
    cells = []
    for X in EXITS:
        a = np.array(res.get((E, X), []), float)
        cells.append("%14s" % ("%+d[%.0f,%d]" % (a.mean(), 100 * (a > 0).mean(), a.min()) if len(a) and X > E else "-"))
    print("%-9s " % E + " ".join(cells))

print("\nTop windows by avg Rs/lot (n>=10):")
allc = [((E, X), np.array(v, float)) for (E, X), v in res.items() if len(v) >= 10]
for (E, X), a in sorted(allc, key=lambda kv: -kv[1].mean())[:8]:
    print("  %s->%s  avg %+d  median %+d  worst %+d  best %+d  win %.0f%%  (n=%d)" %
          (E, X, a.mean(), np.median(a), a.min(), a.max(), 100 * (a > 0).mean(), len(a)))
