"""research/125 - EXPIRY-AFTERNOON straddle: fine (entry x exit x SL) grid on DTE0 days
only (NIFTY Tuesdays, SENSEX Thursdays), spot-ATM, raw ~3-sec chain snaps, dwell SL,
r/123 net cost model. Also builds a per-15-min calmness map of the expiry afternoon.
Prints AlgoTest reference cell (13:45->15:00) explicitly."""
import json, sqlite3
from bisect import bisect_right
from datetime import datetime, timedelta

DB = "/home/arun/quantifyd/backtest_data/options_data.db"
OUT = "/home/arun/quantifyd/research/125_expiry_afternoon_straddle/results/expiry_afternoon.json"
CFG = {"NIFTY": {"step": 50, "qty": 650, "lots": 10}, "SENSEX": {"step": 100, "qty": 100, "lots": 5}}
ENTRIES = ["12:00", "12:30", "12:45", "13:00", "13:15", "13:30", "13:45", "14:00", "14:15", "14:30"]
EXITS = ["14:00", "14:15", "14:30", "14:45", "15:00", "15:10", "15:20"]
SLS = (20, 25, 30, 40, 999)
SLIP_PT = 0.50
BROK_PER_LEGSIDE_PER_LOT = 30.0
SENS_SLIPS = (0.25, 0.50, 1.00)
MIN_HOLD_MIN = 30

def cost_of(c, slip=SLIP_PT):
    return slip * c["qty"] * 4 + BROK_PER_LEGSIDE_PER_LOT * 4 * c["lots"]

oc = sqlite3.connect(DB)

def mins(h):
    return int(h[:2]) * 60 + int(h[3:5])

def agg(f):
    c = pk = dd = 0
    for v in f: c += v; pk = max(pk, c); dd = min(dd, c - pk)
    n = len(f)
    return dict(total=round(sum(f)), mean=round(sum(f) / n), win=round(100 * sum(1 for v in f if v > 0) / n),
                maxdd=round(dd), n=n, ratio=(round(sum(f) / abs(dd), 1) if dd < 0 else 99.0))

grid = {}
calm = {}   # (sym, bucket) -> [per-day mean abs 1-min delta as pct of 13:00 prem], drift
meta = {}
for SYM, cfg in CFG.items():
    days = [r[0] for r in oc.execute(
        "SELECT DISTINCT substr(snapshot_time,1,10) FROM underlying_spot WHERE symbol=? AND spot_price>0 ORDER BY 1", (SYM,))]
    used = []
    for day in days:
        hit = oc.execute(
            "SELECT 1 FROM option_chain WHERE symbol=? AND snapshot_time>=? AND snapshot_time<=? AND expiry_date=? LIMIT 1",
            (SYM, day + "T09:00:00", day + "T16:00:00", day)).fetchone()
        if hit:
            used.append((day, day))
    # today (still trading) excluded from stats: partial afternoon would poison cells
    today = datetime.now().strftime("%Y-%m-%d")
    used = [u for u in used if u[0] != today]
    meta[SYM] = {"expiry_days": len(used), "from": used[0][0] if used else None,
                 "to": used[-1][0] if used else None, "lots": cfg["lots"], "qty": cfg["qty"],
                 "cost": round(cost_of(cfg))}
    print("START %s expiry days=%d" % (SYM, len(used)), flush=True)
    for i, (day, E) in enumerate(used):
        sp = [(r[0][11:19], float(r[1])) for r in oc.execute(
            "SELECT snapshot_time,spot_price FROM underlying_spot WHERE symbol=? AND snapshot_time>=? AND snapshot_time<=? AND spot_price>0 ORDER BY snapshot_time",
            (SYM, day + "T00:00:00", day + "T23:59:59"))]
        if len({round(v, 2) for _, v in sp}) < 50:
            print("  %s SKIP frozen-chain guard" % day, flush=True); continue
        st = [a for a, _ in sp]
        legcache = {}
        def leg(strike, ty):
            key = (strike, ty)
            if key not in legcache:
                legcache[key] = [(r[0][11:19], float(r[1])) for r in oc.execute(
                    "SELECT snapshot_time,ltp FROM option_chain WHERE symbol=? AND snapshot_time>=? AND snapshot_time<=? AND expiry_date=? AND strike=? AND instrument_type=? AND ltp>0 ORDER BY snapshot_time",
                    (SYM, day + "T00:00:00", day + "T23:59:59", E, strike, ty))]
            return legcache[key]
        # ---- calm map from the 13:00-ATM straddle path (minute-sampled) ----
        j13 = bisect_right(st, "13:00:59")
        if j13:
            K13 = round(sp[j13 - 1][1] / cfg["step"]) * cfg["step"]
            ce, pe = leg(K13, "CE"), leg(K13, "PE")
            if ce and pe:
                pk_ = [a for a, _ in pe]; pv = [v for _, v in pe]
                bymin = {}
                for t, cv in ce:
                    if t < "12:00:00" or t > "15:20:59": continue
                    jp = bisect_right(pk_, t)
                    if jp: bymin[t[:5]] = cv + pv[jp - 1]
                ks = sorted(bymin)
                ref = bymin.get("13:00") or (bymin[ks[0]] if ks else None)
                if ref:
                    for a, b in zip(ks, ks[1:]):
                        if mins(b) - mins(a) > 3: continue
                        bucket = "%02d:%02d" % (mins(a) // 15 * 15 // 60, mins(a) // 15 * 15 % 60)
                        d = calm.setdefault((SYM, bucket), {"absd": [], "drift": []})
                        d["absd"].append(abs(bymin[b] - bymin[a]) / ref * 100)
                        d["drift"].append((bymin[b] - bymin[a]) / ref * 100)
        # ---- fine afternoon grid ----
        for ent_t in ENTRIES:
            j = bisect_right(st, ent_t + ":59")
            if not j or st[j - 1][:5] < ent_t[:5]:
                if not j or st[j - 1] < (datetime.strptime(ent_t, "%H:%M") - timedelta(minutes=5)).strftime("%H:%M"):
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
            if len(path) < 10: continue
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
                                    pnl = (ent - nx) * cfg["qty"] - cost_of(cfg)
                                    break
                            else: streak = 0
                    if pnl is None: pnl = (ent - sub[-1][1]) * cfg["qty"] - cost_of(cfg)
                    grid.setdefault((SYM, ent_t, ex_t, sl), []).append((day, round(pnl)))
        print("  %s %d/%d done" % (SYM, i + 1, len(used)), flush=True)

out = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "meta": meta,
       "cost_model": {"slip_pt": SLIP_PT, "brok": BROK_PER_LEGSIDE_PER_LOT, "sens": list(SENS_SLIPS)},
       "cells": [], "calm": {}}
for (SYM, ent_t, ex_t, sl), fp in grid.items():
    if len(fp) < 6: continue
    a = agg([v for _, v in sorted(fp)])
    out["cells"].append({"sym": SYM, "entry": ent_t, "exit": ex_t,
                         "sl": ("none" if sl == 999 else sl), **a,
                         "series": sorted(fp)})
for (SYM, bucket), d in sorted(calm.items()):
    out["calm"].setdefault(SYM, {})[bucket] = {
        "absd_pct_min": round(sum(d["absd"]) / len(d["absd"]), 3),
        "drift_pct_min": round(sum(d["drift"]) / len(d["drift"]), 3), "n": len(d["absd"])}
import os
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, "w"))

print("\n== CALM MAP (per 15-min bucket: mean |1-min move| %% of 13:00 premium / mean drift) ==")
for SYM in CFG:
    row = out["calm"].get(SYM, {})
    print(SYM + ": " + " | ".join("%s %s/%s" % (b, v["absd_pct_min"], v["drift_pct_min"]) for b, v in sorted(row.items())))
print("\n== TOP 8 by ratio (n>=8) per venue ==")
for SYM, cfg in CFG.items():
    cand = [c for c in out["cells"] if c["sym"] == SYM and c["n"] >= 8]
    cand.sort(key=lambda c: -(c["ratio"] if c["ratio"] is not None else -9))
    for c in cand[:8]:
        print("%s %s->%s SL%-4s tot %+8d mean %+6d (%+5d/lot) win %2d%% dd %+7d ratio %5.1f n=%d" % (
            SYM, c["entry"], c["exit"], str(c["sl"]), c["total"], c["mean"], c["mean"] / cfg["lots"],
            c["win"], c["maxdd"], c["ratio"], c["n"]))
print("\n== ALGOTEST REFERENCE CELL 13:45->15:00 (all SLs) ==")
for SYM, cfg in CFG.items():
    for c in out["cells"]:
        if c["sym"] == SYM and c["entry"] == "13:45" and c["exit"] == "15:00":
            print("%s SL%-4s tot %+8d mean %+6d (%+5d/lot) win %2d%% dd %+7d ratio %5.1f n=%d" % (
                SYM, str(c["sl"]), c["total"], c["mean"], c["mean"] / cfg["lots"], c["win"], c["maxdd"], c["ratio"], c["n"]))
print("DONE")
