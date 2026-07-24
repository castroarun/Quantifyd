#!/usr/bin/env python3
"""research/90: NSR-W v1.0 replay P&L travel — same sim as run_replay_nsrw.py but
records the mark-to-market P&L path (5-min sampled + every event) per cycle.
Output: results/replay_paths.json for the visual report."""
import datetime as dt
import json
import os
import sqlite3
import time
from collections import defaultdict

BASE = "/home/arun/quantifyd"
RESDIR = os.path.join(BASE, "research/90_nifty_strangle_rules/results")
DB = os.path.join(BASE, "backtest_data/options_data.db")
T_GRID = [20, 30]
STOP_MULT = 2.0
PT_FRAC = 0.5


def log(m):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def mid(b, a, l):
    return (b + a) / 2 if (b and a and b > 0 and a > 0) else (l or 0)


def main():
    t0 = time.time()
    con = sqlite3.connect(DB)
    c = con.cursor()
    c.execute("SELECT DISTINCT substr(snapshot_time,1,10) FROM underlying_spot WHERE symbol='NIFTY' ORDER BY 1")
    days = [r[0] for r in c.fetchall()]
    mondays = [d for d in days if dt.date.fromisoformat(d).weekday() == 0]
    c.execute("SELECT DISTINCT expiry_date FROM option_chain WHERE symbol='NIFTY' ORDER BY 1")
    expiries = [r[0] for r in c.fetchall()]

    cycles_out = []
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
            "FROM option_chain WHERE symbol='NIFTY' AND expiry_date=? AND snapshot_time>=? AND snapshot_time<=? "
            "ORDER BY snapshot_time", (exp, entry_day + "T00:00:00", exp + "T23:59:59"))
        snaps = defaultdict(dict)
        times = []
        for st, k, ot, ltp, bid, ask, vol, oi, us in cur:
            if not times or times[-1] != st:
                times.append(st)
            snaps[st][(float(k), ot)] = (ltp or 0, bid or 0, ask or 0, vol or 0, oi or 0, us or 0)
        if not times:
            continue
        entry_t = next((t for t in times if t >= entry_day + "T15:14:00" and t[:10] == entry_day), times[0])
        chain0 = snaps[entry_t]
        spot0 = max((v[5] for v in chain0.values()), default=0)
        if not spot0:
            continue

        for T in T_GRID:
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
                continue
            pe_k, ce_k = best["PE"][0], best["CE"][0]
            pe0 = best["PE"][2][1] or best["PE"][2][0]
            ce0 = best["CE"][2][1] or best["CE"][2][0]
            credit = pe0 + ce0
            sl = {"PE": STOP_MULT * pe0, "CE": STOP_MULT * ce0}
            strike = {"PE": pe_k, "CE": ce_k}
            live = {"PE": pe0, "CE": ce0}
            last_q, realized, rolled = {}, 0.0, False
            events = [{"t": entry_t, "label": f"ENTRY {int(pe_k)}PE+{int(ce_k)}CE cr {credit:.0f}"}]
            path = [[entry_t, 0.0]]
            exit_t = reason = None
            gross = None

            def mtm():
                comb = 0.0
                for s in live:
                    q = last_q.get(s)
                    if not q:
                        return None
                    comb += mid(q[1], q[2], q[0])
                return sum(live.values()) - comb + realized

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
                    del live[hit]
                    events.append({"t": t, "label": f"STOP {hit}@{fill:.0f}"})
                    if not live:
                        exit_t, reason, gross = t, "STOP2_FLAT", realized
                        path.append([t, realized])
                        break
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
                            events.append({"t": t, "label": f"ROLL->{int(nk)}{hit}"})
                    v = mtm()
                    if v is not None:
                        path.append([t, round(v, 1)])
                    continue
                v = mtm()
                if v is None:
                    continue
                if t[15] in "05" and t[17:19] == "00":  # every 5 min
                    path.append([t, round(v, 1)])
                if v >= PT_FRAC * credit:
                    cost = sum((last_q[s][2] if last_q[s][2] > 0 else last_q[s][0]) for s in live)
                    gross = sum(live.values()) - cost + realized
                    exit_t, reason = t, "PT"
                    events.append({"t": t, "label": "PROFIT-TAKE"})
                    path.append([t, round(gross, 1)])
                    break
                if (deadline and t[:10] == deadline and t[11:16] >= "15:15") or \
                        (t[:10] == exp and t[11:16] >= "09:30"):
                    cost = sum((last_q[s][2] if last_q[s][2] > 0 else last_q[s][0]) for s in live)
                    gross = sum(live.values()) - cost + realized
                    exit_t, reason = t, "TIME"
                    events.append({"t": t, "label": "TIME-EXIT"})
                    path.append([t, round(gross, 1)])
                    break
            if exit_t is None:
                v = mtm() or realized
                gross, exit_t, reason = v, times[-1], "OPEN"
            cycles_out.append(dict(week=entry_day, T=T, expiry=exp, credit=round(credit, 1),
                                   pe=f"{int(pe_k)}PE@{pe0:.1f}", ce=f"{int(ce_k)}CE@{ce0:.1f}",
                                   exit=exit_t, reason=reason, gross=round(gross, 1),
                                   events=events, path=path))
            log(f"{entry_day} T{T}: {reason} {gross:.1f} pathpts={len(path)}")
    json.dump(cycles_out, open(os.path.join(RESDIR, "replay_paths.json"), "w"))
    log(f"DONE {len(cycles_out)} cycles in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
