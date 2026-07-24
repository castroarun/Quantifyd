#!/usr/bin/env python3
"""research/90: NSR-W entry-TIME sweep on the 1-min chain replay.
Same spec as run_replay_nsrw.py; only the Monday entry snapshot varies:
09:16 / 09:20 / 09:30 / 09:45 / 10:00 / 11:00 / 13:00 / 15:14.
Excludes 2026-04-20 (recorder started 13:56). Aggregates by entry time."""
import datetime as dt
import os
import sqlite3
import time
from collections import defaultdict

BASE = "/home/arun/quantifyd"
RESDIR = os.path.join(BASE, "research/90_nifty_strangle_rules/results")
DB = os.path.join(BASE, "backtest_data/options_data.db")
ENTRY_TIMES = ["09:16", "09:20", "09:30", "09:45", "10:00", "11:00", "13:00", "15:14"]
T_GRID = [60, 100, 140]
STOP_MULT = 2.0
PT_GRID = [0.4, 0.5, 0.6, 0.7, 0.8, None]


def log(m):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def mid(b, a, l):
    return (b + a) / 2 if (b and a and b > 0 and a > 0) else (l or 0)


def simulate(times, snaps, entry_t, exp, deadline, T, pt):
    chain0 = snaps[entry_t]
    spot0 = max((v[5] for v in chain0.values()), default=0)
    if not spot0:
        return None
    best = {"PE": None, "CE": None}
    for (k, ot), v in chain0.items():
        m = mid(v[1], v[2], v[0])
        if m < 1.5 or (v[3] == 0 and v[4] == 0):
            continue
        if (ot == "PE" and k >= spot0) or (ot == "CE" and k <= spot0):
            continue
        if best[ot] is None or abs(m - T) < abs(best[ot][1] - T):
            best[ot] = (k, m, v)
    if not best["PE"] or not best["CE"]:
        return None
    strike = {"PE": best["PE"][0], "CE": best["CE"][0]}
    live = {"PE": best["PE"][2][1] or best["PE"][2][0],
            "CE": best["CE"][2][1] or best["CE"][2][0]}
    credit = sum(live.values())
    sl = {s: STOP_MULT * live[s] for s in live}
    last_q, realized, rolled = {}, 0.0, False
    entry_prem, exit_prem = credit, 0.0
    for t in times:
        if t <= entry_t:
            continue
        ch = snaps[t]
        for side in list(live):
            q = ch.get((strike[side], side))
            if q:
                last_q[side] = q
        hit = next((s for s in list(live) if last_q.get(s) and last_q[s][0] >= sl[s]), None)
        if hit:
            q = last_q[hit]
            fill = q[2] if q[2] > 0 else q[0]
            realized += live[hit] - fill
            exit_prem += fill
            del live[hit]
            if not live:
                return realized - cost(entry_prem, exit_prem), "STOP2"
            if not rolled:
                rolled = True
                spot_t = max((v[5] for v in ch.values()), default=0)
                nb = None
                for (k, ot), v in ch.items():
                    if ot != hit:
                        continue
                    m = mid(v[1], v[2], v[0])
                    if m < 1.5 or (v[3] == 0 and v[4] == 0):
                        continue
                    if spot_t and ((ot == "PE" and k >= spot_t) or (ot == "CE" and k <= spot_t)):
                        continue
                    if nb is None or abs(m - T) < abs(nb[1] - T):
                        nb = (k, m, v)
                if nb:
                    nk, _, nv = nb
                    np_ = nv[1] or nv[0]
                    strike[hit], live[hit], sl[hit] = nk, np_, STOP_MULT * np_
                    last_q[hit] = nv
            continue
        comb = 0.0
        ok = True
        for s in live:
            q = last_q.get(s)
            if not q:
                ok = False
                break
            comb += mid(q[1], q[2], q[0])
        if ok and live and pt:
            profit = sum(live.values()) - comb + realized
            if profit >= pt * credit:
                c2 = sum((last_q[s][2] if last_q[s][2] > 0 else last_q[s][0]) for s in live)
                return sum(live.values()) - c2 + realized - cost(entry_prem, exit_prem + c2), "PT"
        if (deadline and t[:10] == deadline and t[11:16] >= "15:15") or \
                (t[:10] == exp and t[11:16] >= "09:30"):
            c2 = sum((last_q[s][2] if last_q[s][2] > 0 else last_q[s][0]) for s in live)
            return sum(live.values()) - c2 + realized - cost(entry_prem, exit_prem + c2), "TIME"
    return None


def cost(ep, xp):
    return 0.0025 * (ep + xp) + 0.10


def main():
    t0 = time.time()
    con = sqlite3.connect(DB)
    c = con.cursor()
    c.execute("SELECT DISTINCT substr(snapshot_time,1,10) FROM underlying_spot WHERE symbol='SENSEX' ORDER BY 1")
    days = [r[0] for r in c.fetchall()]
    mondays = [d for d in days if dt.date.fromisoformat(d).weekday() == 0 and d != "2026-04-20"]
    c.execute("SELECT DISTINCT expiry_date FROM option_chain WHERE symbol='SENSEX' ORDER BY 1")
    expiries = [r[0] for r in c.fetchall()]

    agg = defaultdict(list)
    for entry_day in mondays:
        d0 = dt.date.fromisoformat(entry_day)
        exp = next((e for e in expiries if 6 <= (dt.date.fromisoformat(e) - d0).days <= 12), None)
        if not exp:
            continue
        ed = dt.date.fromisoformat(exp)
        cand = [d for d in days if d >= entry_day and (ed - dt.date.fromisoformat(d)).days == 1]
        deadline = cand[0] if cand else None
        cur = con.cursor()
        cur.execute(
            "SELECT snapshot_time, strike, instrument_type, ltp, bid, ask, volume, oi, underlying_spot "
            "FROM option_chain WHERE symbol='SENSEX' AND expiry_date=? AND snapshot_time>=? AND snapshot_time<=? "
            "ORDER BY snapshot_time", (exp, entry_day + "T00:00:00", exp + "T23:59:59"))
        snaps = defaultdict(dict)
        times = []
        for st, k, ot, ltp, bid, ask, vol, oi, us in cur:
            if not times or times[-1] != st:
                times.append(st)
            snaps[st][(float(k), ot)] = (ltp or 0, bid or 0, ask or 0, vol or 0, oi or 0, us or 0)
        if not times:
            continue
        for ET in ENTRY_TIMES:
            entry_t = next((t for t in times if t[:10] == entry_day and t[11:16] >= ET), None)
            if not entry_t:
                continue
            for T in T_GRID:
                for pt in PT_GRID:
                    r = simulate(times, snaps, entry_t, exp, deadline, T, pt)
                    if r:
                        agg[(ET, T, pt)].append(r[0])
        log(f"{entry_day} done")

    import statistics as st

    def line(key):
        s = agg.get(key, [])
        if not s:
            return "     -"
        return f"{st.mean(s):6.1f}"

    log("=== GRID mean net pts/week (13 mondays) — rows=entry time, cols=PT ===")
    for T in T_GRID:
        log(f"-- T{T}   " + "  ".join(f"PT{int(p*100) if p else 'no'}" for p in PT_GRID))
    for T in T_GRID:
        log(f"-- T{T}")
        for ET in ENTRY_TIMES:
            log(f"  {ET}: " + " ".join(line((ET, T, p)) for p in PT_GRID))
    log("=== detail at PT50 by entry time ===")
    for T in T_GRID:
        for ET in ENTRY_TIMES:
            s = agg.get((ET, T, 0.5), [])
            if not s:
                continue
            n = len(s)
            mu = st.mean(s)
            sd = st.stdev(s) if n > 1 else 0
            log(f"  T{T} {ET}: mean={mu:6.2f} t={mu/(sd/n**0.5) if sd else 0:5.2f} "
                f"wins={sum(1 for x in s if x>0)}/{n} worst={min(s):7.1f}")
    log("=== detail at 09:16 by PT ===")
    for T in T_GRID:
        for p in PT_GRID:
            s = agg.get(("09:16", T, p), [])
            if not s:
                continue
            n = len(s)
            mu = st.mean(s)
            sd = st.stdev(s) if n > 1 else 0
            log(f"  T{T} PT{p}: mean={mu:6.2f} t={mu/(sd/n**0.5) if sd else 0:5.2f} "
                f"wins={sum(1 for x in s if x>0)}/{n} worst={min(s):7.1f}")
    log(f"DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
