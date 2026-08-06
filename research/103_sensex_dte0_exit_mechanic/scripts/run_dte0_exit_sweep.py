#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/103 — SENSEX ATM2 expiry-day exit-mechanic sweep (MODELED straddle).

We have no real SENSEX (BFO) option intraday history, only the SENSEX index 1-min. So the ATM
short-straddle premium path is Black-Scholes-modeled off the index path + an assumed IV. This
CANNOT claim an absolute rupee edge; it is a *relative* exit-rule comparison whose whole point is
the loss-when-stopped distribution BY DTE (the expiry-gamma-trap test). See STATUS-MD sections 1-4.

Entry 09:16, square by 15:25. Expiry = next Thursday 15:30 (current SENSEX weekly convention;
applied to the whole sample as a documented modeling simplification — the gamma physics that drive
the comparison depend on T, not on the historical expiry calendar).

Output: results/exit_sweep.csv  (rule x iv x dte_bucket aggregate — the deliverable table).
"""
import sqlite3, math, csv, os, sys
from datetime import datetime, timedelta
from collections import defaultdict

DB = "/home/arun/quantifyd/backtest_data/market_data.db"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "exit_sweep.csv")
LOT = 20                      # SENSEX lot
R = 0.065
IVS = [0.10, 0.12, 0.15, 0.18]
COST_PER_LOT = 200.0         # ~Rs160 brokerage + ~Rs40 modeled slippage per round-trip/lot
SLIP_PT = 0.5                # premium pts slippage per leg (already folded into COST_PER_LOT proxy)
ENTRY_HM = (9, 16)
EXIT_HM = (15, 25)

# exit rules: (name, kind, param). kind in {hold, move, move_re, rupee}
RULES = [
    ("HOLD_EOD",        "hold",    None),
    ("MOVE_0.4pct",     "move",    0.004),
    ("MOVE_0.6pct",     "move",    0.006),
    ("MOVE_0.4_REENTER","move_re", 0.004),
    ("RUPEE_2500",      "rupee",   2500.0),
    ("RUPEE_3500",      "rupee",   3500.0),
]

def ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_straddle(S, K, T, iv):
    """ATM-ish straddle price (call+put) per index point."""
    if T <= 0:
        c = max(S - K, 0.0); p = max(K - S, 0.0); return c + p
    v = iv * math.sqrt(T)
    if v < 1e-9:
        c = max(S - K, 0.0); p = max(K - S, 0.0); return c + p
    d1 = (math.log(S / K) + (R + 0.5 * iv * iv) * T) / v
    d2 = d1 - v
    disc = math.exp(-R * T)
    call = S * ncdf(d1) - K * disc * ncdf(d2)
    put = K * disc * ncdf(-d2) - S * ncdf(-d1)
    return call + put

def next_thursday_expiry(d):
    """Next Thursday >= d, at 15:30."""
    off = (3 - d.weekday()) % 7          # Thu = weekday 3
    exp = d + timedelta(days=off)
    return datetime(exp.year, exp.month, exp.day, 15, 30)

def load_days():
    c = sqlite3.connect("file:%s?mode=ro" % DB, uri=True)
    rows = c.execute("SELECT date,open,high,low,close FROM market_data_unified "
                     "WHERE symbol='SENSEX' AND timeframe='minute' ORDER BY date").fetchall()
    c.close()
    days = defaultdict(list)
    for ds, o, h, l, cl in rows:
        dt = datetime.strptime(ds[:19], "%Y-%m-%d %H:%M:%S")
        days[dt.date()].append((dt, cl))
    return days

def simulate(bars, S0, K, expiry_dt, iv, kind, param):
    """Return (net_per_lot, exit_reason, gross_loss_if_stopped_or_None, n_reentries)."""
    ent0 = bs_straddle(S0, K, max((expiry_dt - bars[0][0]).total_seconds(), 60) / (365*24*3600), iv)
    cur_S0, cur_K, cur_ent = S0, K, ent0
    reent = 0
    stopped_loss = None
    exit_reason = "EOD"
    exit_mtm = 0.0
    for (dt, S) in bars:
        T = max((expiry_dt - dt).total_seconds(), 60) / (365*24*3600)
        strad = bs_straddle(S, cur_K, T, iv)
        mtm_pts = (cur_ent - strad)                 # short: +ve = profit (per index pt)
        mtm_lot = mtm_pts * LOT
        move = (S - cur_S0) / cur_S0
        if kind == "move" and abs(move) >= param:
            exit_reason = "MOVE_STOP"; exit_mtm = mtm_lot
            if mtm_lot < 0: stopped_loss = -mtm_lot
            break
        if kind == "move_re" and abs(move) >= param:
            if mtm_lot < 0: stopped_loss = -mtm_lot
            reent += 1
            if reent > 1:                            # ATM2 re-centers once, then holds
                exit_reason = "MOVE_STOP_2"; exit_mtm = mtm_lot; break
            # re-center: bank this leg's mtm, open a fresh ATM straddle at current spot
            exit_mtm += mtm_lot
            cur_S0 = S; cur_K = round(S / 100.0) * 100.0
            cur_ent = bs_straddle(S, cur_K, T, iv)
            continue
        if kind == "rupee" and mtm_lot <= -param:
            exit_reason = "RUPEE_STOP"; exit_mtm = mtm_lot
            stopped_loss = -mtm_lot
            break
    else:
        # reached end without stop -> square at last bar
        dtN, SN = bars[-1]
        T = max((expiry_dt - dtN).total_seconds(), 60) / (365*24*3600)
        exit_mtm += (cur_ent - bs_straddle(SN, cur_K, T, iv)) * LOT
    net = exit_mtm - COST_PER_LOT * (1 + reent)      # extra round-trip cost per re-entry
    return net, exit_reason, stopped_loss, reent

def main():
    days = load_days()
    print("loaded %d trading days: %s -> %s" % (len(days), min(days), max(days)), flush=True)
    agg = defaultdict(lambda: {"n":0, "net":[], "stops":0, "losses":[]})
    for d in sorted(days):
        bars = [(dt, cl) for (dt, cl) in days[d] if cl and cl > 0]
        # entry bar >= 09:16, exit window <= 15:25
        entry = [b for b in bars if (b[0].hour, b[0].minute) >= ENTRY_HM]
        entry = [b for b in entry if (b[0].hour, b[0].minute) <= EXIT_HM]
        if len(entry) < 5:
            continue
        S0 = entry[0][1]; K = round(S0 / 100.0) * 100.0
        exp = next_thursday_expiry(d)
        dte = (exp.date() - d).days
        dte_b = "0" if dte == 0 else ("1" if dte == 1 else "2plus")
        for iv in IVS:
            for (name, kind, param) in RULES:
                net, reason, sloss, reent = simulate(entry, S0, K, exp, iv, kind, param)
                a = agg[(name, iv, dte_b)]
                a["n"] += 1; a["net"].append(net)
                if sloss is not None:
                    a["stops"] += 1; a["losses"].append(sloss)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fields = ["rule","iv","dte_bucket","n_days","mean_net_per_lot","median_net","win_pct",
              "stop_rate_pct","mean_loss_when_stopped","p95_loss_when_stopped"]
    def pct(v, p):
        if not v: return 0.0
        s = sorted(v); i = min(len(s)-1, int(p*len(s)))
        return s[i]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for (name, iv, dte_b) in sorted(agg):
            a = agg[(name, iv, dte_b)]
            nets = a["net"]; losses = a["losses"]
            w.writerow(dict(
                rule=name, iv=iv, dte_bucket=dte_b, n_days=a["n"],
                mean_net_per_lot=round(sum(nets)/len(nets), 1) if nets else 0,
                median_net=round(pct(nets, 0.5), 1),
                win_pct=round(100*sum(1 for x in nets if x > 0)/len(nets), 1) if nets else 0,
                stop_rate_pct=round(100*a["stops"]/a["n"], 1) if a["n"] else 0,
                mean_loss_when_stopped=round(sum(losses)/len(losses), 1) if losses else 0,
                p95_loss_when_stopped=round(pct(losses, 0.95), 1) if losses else 0,
            ))
    print("wrote", OUT, flush=True)
    # quick console readout: the DTE0 head-to-head at base IV 0.12
    print("\n=== DTE0 @ IV12% — the gamma-trap test ===", flush=True)
    for (name, kind, param) in RULES:
        a = agg.get((name, 0.12, "0"))
        if a and a["net"]:
            ml = (sum(a["losses"])/len(a["losses"])) if a["losses"] else 0
            print("  %-18s mean_net/lot=%+8.0f  win%%=%4.0f  stop%%=%4.0f  meanLoss@stop=%7.0f" % (
                name, sum(a["net"])/len(a["net"]),
                100*sum(1 for x in a["net"] if x>0)/len(a["net"]),
                100*a["stops"]/a["n"], ml), flush=True)

if __name__ == "__main__":
    main()
