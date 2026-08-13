"""research/111 - POST-CSL MANAGEMENT replay (NIFTY 09:16->15:20, combined-SL 20%, 3 lots):
what happens AFTER the combined-premium stop triggers?
  BASE   - close both legs, done for the day (the deployed NAS_COMB20 book)
  TRAIL  - close only the LOSER leg (the one that expanded); keep the winner and trail it:
           exit when its premium bounces >=30% off its post-trigger running low (2-snap dwell)
           [the nas_916_atm-style 'let the winner run' management]
  SHIFT  - close both, immediately sell a NEW straddle at the NEW ATM strike (max 3 shifts/day,
           no new entry after 14:30); each cycle carries its own 20% combined-SL
           [the ATM4-style 'recenter the straddle' management]
Raw ~3-sec chain, dwell mechanic identical to the paper executor. Costs: Rs160 per straddle
cycle (TRAIL = same 2 closing trades as BASE -> same cost).
Writes results/csl_mgmt_replay.json + prints the comparison table."""
import json, sqlite3
from bisect import bisect_right
from datetime import datetime, timedelta

DB = "/home/arun/quantifyd/backtest_data/options_data.db"
OUT = "/home/arun/quantifyd/research/111_sensex_manual_mgmt/results/csl_mgmt_replay.json"
SYM, STEP, QTY, LOTS = "NIFTY", 50, 195, 3
ENT_T, EX_T = "09:16", "15:20"
SL = 0.20
TRAIL_BOUNCE = 1.30      # winner exit: premium >= post-trigger low * 1.30
MAX_SHIFTS = 3
SHIFT_CUTOFF = "14:30"   # no new straddle after this
COST = 160               # per straddle cycle

oc = sqlite3.connect(DB)

def day_chain(day, E):
    """One scan: full NIFTY chain for the day/expiry -> {(strike,type): [(t,ltp)...]}"""
    m = {}
    for t, k, ty, v in oc.execute(
        "SELECT snapshot_time,strike,instrument_type,ltp FROM option_chain INDEXED BY idx_oc_symbol_time "
        "WHERE symbol=? AND snapshot_time BETWEEN ? AND ? AND expiry_date=? AND ltp>0 ORDER BY snapshot_time",
            (SYM, day + "T09:00:00", day + "T15:26:00", E)):
        m.setdefault((k, ty), []).append((t[11:19], float(v)))
    return m

def series_at(leg, t):
    ts = [a for a, _ in leg]
    j = bisect_right(ts, t)
    return leg[j - 1][1] if j else None

def dwell_walk(path, thr, start_i=1):
    """Return (trigger_exit_index, None) using 2-consecutive-snap dwell -> exit NEXT snap."""
    streak = 0
    for m in range(start_i, len(path)):
        if path[m][1] >= thr:
            streak += 1
            if streak >= 2:
                return min(m + 1, len(path) - 1)
        else:
            streak = 0
    return None

days = [r[0] for r in oc.execute(
    "SELECT DISTINCT substr(snapshot_time,1,10) FROM underlying_spot WHERE symbol=? AND spot_price>0 ORDER BY 1", (SYM,))]
res = {"BASE": [], "TRAIL": [], "SHIFT": []}
detail = []
for di, day in enumerate(days):
    exps = sorted({r[0] for r in oc.execute(
        "SELECT DISTINCT expiry_date FROM option_chain INDEXED BY idx_oc_symbol_time WHERE symbol=? AND snapshot_time BETWEEN ? AND ? AND expiry_date>=?",
        (SYM, day + "T09:00:00", day + "T15:26:00", day))})
    if not exps: continue
    E = exps[0]
    sp = [(r[0][11:19], float(r[1])) for r in oc.execute(
        "SELECT snapshot_time,spot_price FROM underlying_spot WHERE symbol=? AND snapshot_time BETWEEN ? AND ? AND spot_price>0 ORDER BY snapshot_time",
        (SYM, day + "T09:00:00", day + "T15:26:00"))]
    if not sp: continue
    spt = [a for a, _ in sp]
    j = bisect_right(spt, ENT_T + ":59")
    if not j or spt[j - 1] < "09:11": continue
    K0 = round(sp[j - 1][1] / STEP) * STEP
    chain = day_chain(day, E)

    def comb_path(K, t0):
        ce, pe = chain.get((K, "CE"), []), chain.get((K, "PE"), [])
        if not (ce and pe): return None, None, None
        pk = [a for a, _ in pe]; pv = [v for _, v in pe]
        path = []
        for t, cv in ce:
            if t < t0 or t > EX_T + ":59": continue
            jp = bisect_right(pk, t)
            if jp: path.append((t, cv + pv[jp - 1]))
        return (path if len(path) >= 10 else None), ce, pe

    path, ce0, pe0 = comb_path(K0, ENT_T + ":00")
    if not path: continue
    ent = path[0][1]; t_ent = path[0][0]
    ce_e = series_at(ce0, t_ent); pe_e = series_at(pe0, t_ent)
    thr = ent * (1 + SL)
    trig = dwell_walk(path, thr)

    # ---- BASE
    if trig is None:
        base = (ent - path[-1][1]) * QTY - COST
    else:
        base = (ent - path[trig][1]) * QTY - COST
    res["BASE"].append((day, round(base)))

    # ---- TRAIL
    if trig is None:
        trail = base
        trail_note = "no-trigger"
    else:
        t_tr = path[trig][0]
        ce_tr = series_at(ce0, t_tr); pe_tr = series_at(pe0, t_tr)
        # loser = leg that expanded more vs its entry
        if (ce_tr / ce_e) >= (pe_tr / pe_e):
            lose_e, lose_x, win_leg, win_e = ce_e, ce_tr, pe0, pe_e
        else:
            lose_e, lose_x, win_leg, win_e = pe_e, pe_tr, ce0, ce_e
        wp = [(t, v) for t, v in win_leg if t_tr <= t <= EX_T + ":59"]
        lo = wp[0][1] if wp else win_e; wx = wp[-1][1] if wp else win_e
        streak = 0
        for m in range(1, len(wp)):
            v = wp[m][1]
            lo = min(lo, v)
            if v >= lo * TRAIL_BOUNCE:
                streak += 1
                if streak >= 2:
                    wx = wp[min(m + 1, len(wp) - 1)][1]; break
            else:
                streak = 0
        else:
            wx = wp[-1][1] if wp else win_e
        trail = ((lose_e - lose_x) + (win_e - wx)) * QTY - COST
        trail_note = "trailed"
    res["TRAIL"].append((day, round(trail)))

    # ---- SHIFT
    total = 0.0; shifts = 0; cur_path = path; cur_ent = ent; cur_thr = thr; ok = True
    while True:
        tr2 = dwell_walk(cur_path, cur_thr)
        if tr2 is None:
            total += (cur_ent - cur_path[-1][1]) * QTY - COST
            break
        t_x = cur_path[tr2][0]
        total += (cur_ent - cur_path[tr2][1]) * QTY - COST
        if shifts >= MAX_SHIFTS or t_x > SHIFT_CUTOFF + ":59":
            break
        jj = bisect_right(spt, t_x)
        if not jj: break
        Kn = round(sp[jj - 1][1] / STEP) * STEP
        npath, _, _ = comb_path(Kn, t_x)
        if not npath: break
        shifts += 1
        cur_path = npath; cur_ent = npath[0][1]; cur_thr = cur_ent * (1 + SL)
    res["SHIFT"].append((day, round(total)))
    detail.append({"day": day, "trigger": trig is not None, "shifts": shifts,
                   "base": round(base), "trail": round(trail), "shift": round(total)})
    if di % 10 == 0:
        print("%d/%d %s" % (di, len(days), day), flush=True)

def agg(fp):
    f = [v for _, v in fp]
    c = pk = dd = 0
    for v in f: c += v; pk = max(pk, c); dd = min(dd, c - pk)
    n = len(f)
    return dict(total=round(sum(f)), mean=round(sum(f) / n), win=round(100 * sum(1 for v in f if v > 0) / n),
                maxdd=round(dd), n=n, ratio=(round(sum(f) / abs(dd), 1) if dd < 0 else 99.0))

out = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
       "cfg": {"sym": SYM, "entry": ENT_T, "exit": EX_T, "sl": 20, "lots": LOTS, "qty": QTY,
               "trail_bounce": TRAIL_BOUNCE, "max_shifts": MAX_SHIFTS, "shift_cutoff": SHIFT_CUTOFF},
       "arms": {k: {"stats": agg(v), "series": v} for k, v in res.items()},
       "detail": detail}
json.dump(out, open(OUT, "w"))
trig_days = sum(1 for d in detail if d["trigger"])
print("\nPOST-CSL MANAGEMENT — NIFTY 09:16->15:20 CSL20, %d lots, n=%d days (%s -> %s), SL triggered on %d days" % (
    LOTS, len(res["BASE"]), days[0], days[-1], trig_days))
print("%-8s %10s %8s %5s %10s %7s" % ("ARM", "TOTAL", "MEAN/d", "WIN%", "MAXDD", "RATIO"))
for k in ("BASE", "TRAIL", "SHIFT"):
    a = agg(res[k])
    print("%-8s %+10d %+8d %4d%% %+10d %7.1f" % (k, a["total"], a["mean"], a["win"], a["maxdd"], a["ratio"]))
print("DONE")
