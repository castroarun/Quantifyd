"""research/111 - CSL BEST-CONFIG sweep: ENTRY time x EXIT time x combined-SL x DTE,
NIFTY + SENSEX, on raw ~3-sec snaps with the accepted dwell mechanic (2 consecutive
breaching snaps -> market exit at next snap). Strike = ATM at the ENTRY moment.
Per-day sparse-coverage days are tagged (resolution ladder rule) not dropped.
Writes results/entry_exit_sweep.json; prints top configs per index x DTE."""
import json, sqlite3
from bisect import bisect_right
from datetime import datetime, timedelta

DB = "/home/arun/quantifyd/backtest_data/options_data.db"
OUT = "/home/arun/quantifyd/research/111_sensex_manual_mgmt/results/entry_exit_sweep.json"
CFG = {"NIFTY": {"step": 50, "qty": 650, "lots": 10}, "SENSEX": {"step": 100, "qty": 100, "lots": 5}}
ENTRIES = ["09:16", "09:20", "09:25", "09:30", "10:00", "10:30", "11:15", "12:00", "13:00", "14:00"]
EXITS = ["11:00", "12:00", "13:00", "14:00", "15:00", "15:20"]
SLS = (20, 25, 30, 40, 999)
COST = 160
MIN_HOLD_MIN = 45

oc = sqlite3.connect(DB)

def tdte(E, day):
    dd = datetime.strptime(day, "%Y-%m-%d").date(); ed = datetime.strptime(E, "%Y-%m-%d").date()
    n, cur = 0, dd + timedelta(days=1)
    while cur <= ed:
        if cur.weekday() < 5: n += 1
        cur += timedelta(days=1)
    return n

def mins(h):  # "HH:MM" -> minutes
    return int(h[:2]) * 60 + int(h[3:5])

def agg(f):
    c = pk = dd = 0
    for v in f: c += v; pk = max(pk, c); dd = min(dd, c - pk)
    n = len(f)
    return dict(total=round(sum(f)), mean=round(sum(f) / n), win=round(100 * sum(1 for v in f if v > 0) / n),
                maxdd=round(dd), n=n, ratio=(round(sum(f) / abs(dd), 1) if dd < 0 else 99.0))

grid = {}      # (sym,dte,entry,exit,sl) -> [(day,pnl),...]
sparse = {"NIFTY": 0, "SENSEX": 0}
meta = {}
for SYM, cfg in CFG.items():
    days = [r[0] for r in oc.execute(
        "SELECT DISTINCT substr(snapshot_time,1,10) FROM underlying_spot WHERE symbol=? AND spot_price>0 ORDER BY 1", (SYM,))]
    meta[SYM] = {"days": len(days), "from": days[0] if days else None, "to": days[-1] if days else None,
                 "lots": cfg["lots"], "qty": cfg["qty"]}
    print("START %s days=%d" % (SYM, len(days)), flush=True)
    for i, day in enumerate(days):
        exps = sorted({r[0] for r in oc.execute(
            "SELECT DISTINCT expiry_date FROM option_chain WHERE symbol=? AND substr(snapshot_time,1,10)=? AND expiry_date>=?",
            (SYM, day, day))})
        if not exps: continue
        E = exps[0]; k = tdte(E, day)
        sp = [(r[0][11:19], float(r[1])) for r in oc.execute(
            "SELECT snapshot_time,spot_price FROM underlying_spot WHERE symbol=? AND substr(snapshot_time,1,10)=? AND spot_price>0 ORDER BY snapshot_time", (SYM, day))]
        if not sp: continue
        st = [a for a, _ in sp]
        legcache = {}
        def leg(strike, ty):
            key = (strike, ty)
            if key not in legcache:
                legcache[key] = [(r[0][11:19], float(r[1])) for r in oc.execute(
                    "SELECT snapshot_time,ltp FROM option_chain WHERE symbol=? AND expiry_date=? AND strike=? AND instrument_type=? AND substr(snapshot_time,1,10)=? AND ltp>0 ORDER BY snapshot_time",
                    (SYM, E, strike, ty, day))]
            return legcache[key]
        day_sparse = False
        for ent_t in ENTRIES:
            j = bisect_right(st, ent_t + ":59")
            if not j or st[j - 1] < ent_t[:2]: continue
            if st[j - 1][:5] < ent_t[:5]:      # no spot near entry minute -> skip this entry today
                if st[j - 1] < (datetime.strptime(ent_t, "%H:%M") - timedelta(minutes=5)).strftime("%H:%M"):
                    continue
            K = round(sp[j - 1][1] / cfg["step"]) * cfg["step"]
            ce, pe = leg(K, "CE"), leg(K, "PE")
            if not (ce and pe): continue
            pk_ = [a for a, _ in pe]; pv = [v for _, v in pe]
            path = []
            for t, cv in ce:
                if t < ent_t + ":00" or t > "15:20:59": continue
                jp = bisect_right(pk_, t)
                if jp: path.append((t, cv + pv[jp - 1]))
            if len(path) < 10:
                day_sparse = True; continue
            ent = path[0][1]
            for ex_t in EXITS:
                if mins(ex_t) - mins(ent_t) < MIN_HOLD_MIN: continue
                sub = [p for p in path if p[0] <= ex_t + ":59"]
                if len(sub) < 5: continue
                for sl in SLS:
                    thr = ent * (1 + sl / 100.0); streak = 0; pnl = None
                    if sl < 900:
                        for m in range(1, len(sub)):
                            if sub[m][1] >= thr:
                                streak += 1
                                if streak >= 2:
                                    nx = sub[m + 1][1] if m + 1 < len(sub) else sub[m][1]
                                    pnl = (ent - nx) * cfg["qty"] - COST
                                    break
                            else: streak = 0
                    if pnl is None: pnl = (ent - sub[-1][1]) * cfg["qty"] - COST
                    grid.setdefault((SYM, k, ent_t, ex_t, sl), []).append((day, round(pnl)))
        if day_sparse: sparse[SYM] += 1
        if i % 10 == 0: print("%s %d/%d" % (SYM, i, len(days)), flush=True)

out = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "sparse_days": sparse,
       "meta": meta, "cells": [], "best": {}}
series_map = {}
for (SYM, k, ent_t, ex_t, sl), fp in grid.items():
    if len(fp) < 6: continue
    fp = sorted(fp)
    a = agg([v for _, v in fp])
    cell = {"sym": SYM, "dte": k, "entry": ent_t, "exit": ex_t,
            "sl": ("none" if sl == 999 else sl), **a,
            "series": [[d, v] for d, v in fp]}   # per-cell daily series -> page can stack slots
    out["cells"].append(cell)
    series_map[(SYM, k, ent_t, ex_t, sl)] = fp
# best config per (sym,dte) by ratio among n>=8, WITH its daily P&L series (for page curves)
for SYM in CFG:
    out["best"][SYM] = {}
    for k in range(5):
        cand = [c for c in out["cells"] if c["sym"] == SYM and c["dte"] == k and c["n"] >= 8]
        if not cand: continue
        cand.sort(key=lambda c: -(c["ratio"] if c["ratio"] is not None else -9))
        b = dict(cand[0])
        key = (SYM, k, b["entry"], b["exit"], (999 if b["sl"] == "none" else b["sl"]))
        b["series"] = [[d, v] for d, v in series_map.get(key, [])]
        out["best"][SYM][str(k)] = b
json.dump(out, open(OUT, "w"))
for extra in ("/home/arun/quantifyd/static/app/straddles/csl_best_configs.json",
              "/home/arun/quantifyd/frontend/public/straddles/csl_best_configs.json"):
    json.dump(out, open(extra, "w"))
print("\nTOP CONFIGS by ratio (n>=8):")
for SYM in CFG:
    for k in range(5):
        cand = [c for c in out["cells"] if c["sym"] == SYM and c["dte"] == k and c["n"] >= 8]
        cand.sort(key=lambda c: -(c["ratio"] if c["ratio"] else -9))
        print("%s DTE%d:" % (SYM, k))
        for c in cand[:3]:
            print("   in %s out %s SL%-4s  tot %+9d mean %+7d win %2d%% dd %+8d ratio %s n=%d" % (
                c["entry"], c["exit"], str(c["sl"]), c["total"], c["mean"], c["win"], c["maxdd"], c["ratio"], c["n"]))
print("DONE")
