#!/usr/bin/env python3
"""research/90: daily 09:16 Rs-premium strangle, intraday, 1-min chain replay.
Enter first snapshot >=09:16, nearest expiry, strikes by premium target
T in {10,15,20,25,30}; sell BID; stop 2x LTP->ASK; PT 50% mid; one roll-away;
survivors ride; time exit same day >=15:15. See DAILY_0916 STATUS-MD."""
import csv
import datetime as dt
import os
import sqlite3
import time
from collections import defaultdict

BASE = "/home/arun/quantifyd"
RESDIR = os.path.join(BASE, "research/90_nifty_strangle_rules/results")
DB = os.path.join(BASE, "backtest_data/options_data.db")
T_GRID = [10, 15, 20, 25, 30]
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
    c.execute("SELECT DISTINCT expiry_date FROM option_chain WHERE symbol='NIFTY' ORDER BY 1")
    expiries = [r[0] for r in c.fetchall()]

    out = open(os.path.join(RESDIR, "daily916_cycles.csv"), "w", newline="")
    w = csv.DictWriter(out, fieldnames=["day", "T", "expiry", "dte", "entry_t", "pe", "ce",
                                        "credit", "exit_t", "reason", "gross", "net"])
    w.writeheader()
    rows_out = []

    for day in days:
        d0 = dt.date.fromisoformat(day)
        exp = next((e for e in expiries if (dt.date.fromisoformat(e) - d0).days >= 0
                    and e >= day), None)
        if not exp:
            continue
        dte = (dt.date.fromisoformat(exp) - d0).days
        cur = con.cursor()
        cur.execute(
            "SELECT snapshot_time, strike, instrument_type, ltp, bid, ask, volume, oi, underlying_spot "
            "FROM option_chain WHERE symbol='NIFTY' AND expiry_date=? AND snapshot_time LIKE ? "
            "ORDER BY snapshot_time", (exp, day + "%"))
        snaps = defaultdict(dict)
        times = []
        for st, k, ot, ltp, bid, ask, vol, oi, us in cur:
            if not times or times[-1] != st:
                times.append(st)
            snaps[st][(float(k), ot)] = (ltp or 0, bid or 0, ask or 0, vol or 0, oi or 0, us or 0)
        if not times:
            continue
        entry_t = next((t for t in times if t[11:16] >= "09:16"), None)
        if not entry_t or entry_t[11:16] > "09:30":  # day with late data start
            continue
        chain0 = snaps[entry_t]
        spot0 = max((v[5] for v in chain0.values()), default=0)
        if not spot0:
            continue

        for T in T_GRID:
            best = {"PE": None, "CE": None}
            for (k, ot), v in chain0.items():
                m = mid(v[1], v[2], v[0])
                if m < 1.0 or (v[3] == 0 and v[4] == 0):
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
            if abs(pe0 - T) > 0.6 * T or abs(ce0 - T) > 0.6 * T:
                continue  # no strike anywhere near target (e.g. 0DTE deep grid gaps)
            credit = pe0 + ce0
            sl = {"PE": STOP_MULT * pe0, "CE": STOP_MULT * ce0}
            strike = {"PE": pe_k, "CE": ce_k}
            live = {"PE": pe0, "CE": ce0}
            last_q, realized, rolled = {}, 0.0, False
            entry_prem, exit_prem = credit, 0.0
            exit_t = reason = None
            gross = None
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
                        exit_t, reason, gross = t, "STOP2", realized
                        break
                    if not rolled:
                        rolled = True
                        spot_t = max((v[5] for v in ch.values()), default=0)
                        nb = None
                        for (k, ot), v in ch.items():
                            if ot != hit:
                                continue
                            m = mid(v[1], v[2], v[0])
                            if m < 1.0 or (v[3] == 0 and v[4] == 0):
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
                if ok and live:
                    profit = sum(live.values()) - comb + realized
                    if profit >= PT_FRAC * credit:
                        cost = sum((last_q[s][2] if last_q[s][2] > 0 else last_q[s][0]) for s in live)
                        gross = sum(live.values()) - cost + realized
                        exit_prem += cost
                        exit_t, reason = t, "PT"
                        break
                if t[11:16] >= "15:15":
                    cost = sum((last_q[s][2] if last_q[s][2] > 0 else last_q[s][0]) for s in live)
                    gross = sum(live.values()) - cost + realized
                    exit_prem += cost
                    exit_t, reason = t, "TIME"
                    break
            if exit_t is None:
                continue
            net = gross - (0.0025 * (entry_prem + exit_prem) + 0.10)
            row = dict(day=day, T=T, expiry=exp, dte=dte, entry_t=entry_t[11:16],
                       pe=f"{int(pe_k)}@{pe0:.1f}", ce=f"{int(ce_k)}@{ce0:.1f}",
                       credit=round(credit, 1), exit_t=exit_t[11:16], reason=reason,
                       gross=round(gross, 2), net=round(net, 2))
            w.writerow(row)
            rows_out.append(row)
        out.flush()
    out.close()

    import statistics as st
    log(f"cycles={len(rows_out)}")
    for T in T_GRID:
        sel = [r["net"] for r in rows_out if r["T"] == T]
        if not sel:
            continue
        n = len(sel)
        mu = st.mean(sel)
        sd = st.stdev(sel) if n > 1 else 0
        log(f"T{T}: n={n} mean={mu:.2f} t={mu/(sd/n**0.5) if sd else 0:.2f} "
            f"win={100*sum(1 for x in sel if x>0)/n:.0f}% worst={min(sel):.1f} sum={sum(sel):.0f}")
        for lab, lo, hi in (("DTE0", 0, 0), ("DTE1-2", 1, 2), ("DTE3+", 3, 99)):
            s2 = [r["net"] for r in rows_out if r["T"] == T and lo <= r["dte"] <= hi]
            if s2:
                log(f"   {lab}: n={len(s2)} mean={st.mean(s2):.2f} sum={sum(s2):.0f} worst={min(s2):.1f}")
    log(f"DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
