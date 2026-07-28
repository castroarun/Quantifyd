# -*- coding: utf-8 -*-
"""Squeeze-family ENTRY-TRIGGER bake-off. Isolate the entry: short the ATM straddle at the trigger
time, hold to 15:15 the SAME way for every variant, so the only difference is WHEN you enter.
Triggers: 09:16 (ref) | ATR squeeze (current) | fixed times | price/strike-move (+/-50, +/-100 from
the 09:16 spot). Net-of-cost (brokerage), on the recorded NIFTY chain, full + recent-window."""
import os, sys
sys.path.insert(0, "/home/arun/quantifyd")
sys.path.insert(0, "/home/arun/quantifyd/research/90_nas_portfolio_bracket/scripts")
import numpy as np, pandas as pd
from datetime import time as dtime
from engine_mtm import load_day, days, LOT, LOTS, BROK_PER_LEG  # noqa

QTY = LOT * LOTS
SQ_START, WIN_END, EXIT = dtime(9, 30), dtime(14, 30), dtime(15, 15)


def squeeze_t0(spot_s, times):
    c5 = spot_s.resample("5min").agg(["first", "max", "min", "last"]).dropna()
    if len(c5) < 55:
        return next((t for t in times if t.time() >= SQ_START), None)
    c5.columns = ["open", "high", "low", "close"]
    pc = c5["close"].shift(1)
    tr = pd.concat([c5["high"] - c5["low"], (c5["high"] - pc).abs(), (c5["low"] - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    sma = atr.rolling(50).mean()
    tt = c5.index.to_series().dt.time
    sq = (atr < sma) & (tt >= SQ_START) & (tt <= WIN_END)
    if not sq.any():
        return None
    return min((t for t in times if t >= sq[sq].index[0]), default=None)


def time_t0(times, hh, mm):
    return next((t for t in times if t.time() >= dtime(hh, mm)), None)


def price_t0(spot_s, times, thr):
    ent = [t for t in times if t.time() >= dtime(9, 16)]
    if not ent:
        return None
    s0 = float(spot_s.loc[ent[0]])
    for t in times:
        if t < ent[0]:
            continue
        if t.time() > WIN_END:
            break
        if t in spot_s.index and abs(float(spot_s.loc[t]) - s0) >= thr:
            return t
    return None


def trade(b, t0):
    if t0 is None or t0.time() > EXIT:
        return None
    chain, spot_s, times = b["chain"], b["spot_s"], b["times"]
    def prem(ts, t):
        ta, la, _, _ = chain[ts]; i = np.searchsorted(ta, np.datetime64(t), side="right") - 1
        return float(la[i]) if i >= 0 and la[i] and la[i] > 0 else None
    def tsym(strike, typ):
        for ts, (_, _, st, ty) in chain.items():
            if int(st) == int(strike) and ty == typ:
                return ts
        return None
    atm = round(float(spot_s.loc[t0]) / 50) * 50
    ce, pe = tsym(atm, "CE"), tsym(atm, "PE")
    if not ce or not pe:
        return None
    ce_e, pe_e = prem(ce, t0), prem(pe, t0)
    if ce_e is None or pe_e is None:
        return None
    lastt = [t for t in times if t.time() <= EXIT][-1]
    ce_x, pe_x = prem(ce, lastt) or ce_e, prem(pe, lastt) or pe_e
    return round(((ce_e - ce_x) + (pe_e - pe_x)) * QTY - 2 * BROK_PER_LEG)


TRIGS = [("0916", "09:16 (ref)"), ("squeeze", "ATR squeeze"),
         ("t0930", "time 09:30"), ("t1000", "time 10:00"), ("t1030", "time 10:30"),
         ("t1100", "time 11:00"), ("t1200", "time 12:00"),
         ("px50", "price ±50 (1 strike)"), ("px100", "price ±100 (2 strikes)")]

res = {k: [] for k, _ in TRIGS}
for i, day in enumerate(days):
    b = load_day(day)
    if b is None:
        continue
    ss, tt = b["spot_s"], b["times"]
    t0s = {"0916": time_t0(tt, 9, 16), "squeeze": squeeze_t0(ss, tt),
           "t0930": time_t0(tt, 9, 30), "t1000": time_t0(tt, 10, 0), "t1030": time_t0(tt, 10, 30),
           "t1100": time_t0(tt, 11, 0), "t1200": time_t0(tt, 12, 0),
           "px50": price_t0(ss, tt, 50), "px100": price_t0(ss, tt, 100)}
    for k, t0 in t0s.items():
        p = trade(b, t0)
        if p is not None:
            res[k].append((day, p))
    if i % 10 == 0 or i == len(days) - 1:
        print("  replay %d/%d %s" % (i, len(days), day), flush=True)

keys = sorted(days); half = len(keys) // 2
print("\n=== SQUEEZE-family ENTRY-TRIGGER bake-off — short ATM straddle held to 15:15, 2 lots (QTY %d), net ===" % QTY)
print("%-22s %5s %9s %8s %8s %6s %9s %9s %9s" % ("trigger", "nDays", "total", "avg/tr", "median", "win%", "best", "worst", "recent"))
rows_out = []
for k, lab in TRIGS:
    rows = res[k]; p = np.array([x[1] for x in rows], float)
    if not len(p):
        print("%-22s  (never fired)" % lab); continue
    recent = sum(x[1] for x in rows if x[0] >= keys[half])
    med = float(np.median(p))
    rows_out.append((lab, len(p), p.sum(), p.mean(), med, 100 * (p > 0).mean(), p.max(), p.min(), recent))
    print("%-22s %5d %+9d %+8d %+8d %6.0f %+9d %+9d %+9d" %
          (lab, len(p), p.sum(), p.mean(), med, 100 * (p > 0).mean(), p.max(), p.min(), recent))

print("\nRanked by AVG per trade (the fair entry-quality metric, since trigger fire-counts differ):")
for r in sorted(rows_out, key=lambda x: -x[3]):
    print("  %-22s avg %+7d/tr  (n=%d, total %+d, recent %+d)" % (r[0], r[3], r[1], r[2], r[8]))
