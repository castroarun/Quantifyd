"""research/111 sec 19 - SPOT-ATM vs FORWARD-ATM strike selection.
Motivation (2026-08-19 live obs): TB-SENSEX entry skew CE-PE=124 vs spot-K=+40 ->
implied forward ~84pts above cash. Question: does picking K nearest the SYNTHETIC
FORWARD (F = K_spot + CE(K_spot) - PE(K_spot) at entry) beat the tested spot-ATM?
Replays the FROZEN book configs (TB-NIFTY, TB-SENSEX, COMB) on the recorded chain,
both arms, same windows/dwell-SL/costs. Reports: how often K differs, head-to-head
on differing days, and overall. Fast pattern: INDEXED BY idx_oc_symbol_time + T-bounds."""
import json, sqlite3
from bisect import bisect_right
from datetime import datetime, timedelta

DB = "/home/arun/quantifyd/backtest_data/options_data.db"
OUT = "/home/arun/quantifyd/research/111_sensex_manual_mgmt/results/sec19_fwd_atm.json"
CFG = json.load(open("/home/arun/quantifyd/backtest_data/csl_paper_config.json"))["books"]
BOOKS = [
    ("CSL_TIMEB_NIFTY", "NIFTY", 50, 520),
    ("CSL_TIMEB_SENSEX", "SENSEX", 100, 160),
    ("NAS_COMB20", "NIFTY", 50, 130),
]
COST = 160
oc = sqlite3.connect(DB)

def tdte(E, day):
    dd = datetime.strptime(day, "%Y-%m-%d").date(); ed = datetime.strptime(E, "%Y-%m-%d").date()
    n, cur = 0, dd + timedelta(days=1)
    while cur <= ed:
        if cur.weekday() < 5: n += 1
        cur += timedelta(days=1)
    return n

def day_chain(sym, day, E):
    m = {}
    for t, k, ty, v in oc.execute(
        "SELECT snapshot_time,strike,instrument_type,ltp FROM option_chain INDEXED BY idx_oc_symbol_time "
        "WHERE symbol=? AND snapshot_time BETWEEN ? AND ? AND expiry_date=? AND ltp>0 ORDER BY snapshot_time",
            (sym, day + "T09:00:00", day + "T15:26:00", E)):
        m.setdefault((k, ty), []).append((t[11:19], float(v)))
    return m

def path_of(chain, K, t0, t1):
    ce, pe = chain.get((K, "CE"), []), chain.get((K, "PE"), [])
    if not (ce and pe): return None
    pk = [a for a, _ in pe]; pv = [v for _, v in pe]
    out = []
    for t, cv in ce:
        if t < t0 or t > t1: continue
        jp = bisect_right(pk, t)
        if jp: out.append((t, cv + pv[jp - 1]))
    return out if len(out) >= 10 else None

def replay(path, sl_pct):
    ent = path[0][1]; thr = ent * (1 + sl_pct / 100.0); streak = 0
    for m in range(1, len(path)):
        if path[m][1] >= thr:
            streak += 1
            if streak >= 2:
                nx = path[min(m + 1, len(path) - 1)]
                return ent - nx[1], "SL"
        else:
            streak = 0
    return ent - path[-1][1], "TIME"

def agg(f):
    c = pk = dd = 0
    for v in f: c += v; pk = max(pk, c); dd = min(dd, c - pk)
    n = len(f)
    return (round(sum(f)), round(dd), (round(sum(f) / abs(dd), 1) if dd < 0 else 99.0),
            round(100 * sum(1 for v in f if v > 0) / n))

results = {}
for bk, sym, step, qty in BOOKS:
    sched = CFG.get(bk, {})
    days = [r[0] for r in oc.execute(
        "SELECT DISTINCT substr(snapshot_time,1,10) FROM underlying_spot WHERE symbol=? AND spot_price>0 ORDER BY 1", (sym,))]
    rs, rf, diffs, basis = [], [], [], []
    for day in days:
        exps = sorted({r[0] for r in oc.execute(
            "SELECT DISTINCT expiry_date FROM option_chain INDEXED BY idx_oc_symbol_time WHERE symbol=? AND snapshot_time BETWEEN ? AND ? AND expiry_date>=?",
            (sym, day + "T09:00:00", day + "T15:26:00", day))})
        if not exps: continue
        E = exps[0]; k_dte = str(tdte(E, day))
        c = sched.get(k_dte)
        if not c: continue
        sp = [(r[0][11:19], float(r[1])) for r in oc.execute(
            "SELECT snapshot_time,spot_price FROM underlying_spot WHERE symbol=? AND snapshot_time BETWEEN ? AND ? AND spot_price>0 ORDER BY snapshot_time",
            (sym, day + "T09:00:00", day + "T15:26:00"))]
        if not sp: continue
        spt = [a for a, _ in sp]
        j = bisect_right(spt, c["entry"] + ":59")
        if not j or spt[j - 1] < (datetime.strptime(c["entry"], "%H:%M") - timedelta(minutes=5)).strftime("%H:%M"):
            continue
        S = sp[j - 1][1]
        Ks = round(S / step) * step
        chain = day_chain(sym, day, E)
        t0, t1 = c["entry"] + ":00", c["exit"] + ":59"
        ps = path_of(chain, Ks, t0, t1)
        if not ps: continue
        # forward from parity at the spot-ATM strike, at entry snap
        ce_l = chain.get((Ks, "CE"), []); pe_l = chain.get((Ks, "PE"), [])
        def at(leg, t):
            ts = [a for a, _ in leg]
            jj = bisect_right(ts, t)
            return leg[jj - 1][1] if jj else None
        t_ent = ps[0][0]
        ce0, pe0 = at(ce_l, t_ent), at(pe_l, t_ent)
        if ce0 is None or pe0 is None: continue
        F = Ks + (ce0 - pe0)
        Kf = round(F / step) * step
        basis.append(round(F - S, 1))
        sl = float(c["sl"]) if str(c["sl"]) != "none" else 50.0
        v_s, why_s = replay(ps, sl)
        if Kf == Ks:
            v_f = v_s
        else:
            pf = path_of(chain, Kf, t0, t1)
            if not pf: continue
            v_f, _ = replay(pf, sl)
            diffs.append((day, k_dte, Ks, Kf, round(v_s * qty), round(v_f * qty)))
        rs.append(v_s * qty - COST); rf.append(v_f * qty - COST)
    a_s, a_f = agg(rs), agg(rf)
    results[bk] = {"n": len(rs), "diff_days": len(diffs), "basis_mean": round(sum(basis) / max(len(basis), 1), 1),
                   "basis_p90": sorted(basis)[int(0.9 * len(basis))] if basis else None,
                   "spot_atm": a_s, "fwd_atm": a_f, "diffs": diffs}
    print("%s (qty %d): n=%d days, K differs on %d (basis mean %+0.1f, p90 %+0.1f)" % (
        bk, qty, len(rs), len(diffs), results[bk]["basis_mean"], results[bk]["basis_p90"] or 0))
    print("   SPOT-ATM: total %+9d dd %+8d ratio %5.1f win %d%%" % a_s)
    print("   FWD-ATM : total %+9d dd %+8d ratio %5.1f win %d%%" % a_f)
    if diffs:
        d_s = sum(x[4] for x in diffs); d_f = sum(x[5] for x in diffs)
        print("   differing days only: SPOT %+d vs FWD %+d (%d days)" % (d_s, d_f, len(diffs)))

json.dump({"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "results": results}, open(OUT, "w"))
print("DONE")
