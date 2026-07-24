#!/usr/bin/env python3
"""research/90: NSR-W v1.0 intraday replay on options_data.db (1-min chain).

Spec (locked, MONDAY_20RS_STRANGLE_WEEKLY_SWEEP_STATUS.md #6): Monday entry,
next-week expiry (DTE 6-12), strikes by premium target (T=20/30 per leg, MID),
sell at BID, GTT stop 2.0x per leg (trigger LTP, fill ASK), PT 50% combined MID,
ONE roll-away (same T rule), 2nd stop -> flat, time exit DTE<=1 @ <=15:15 ASK.
See NSRW_V1_CHAIN_REPLAY_1MIN_RUN_STATUS.md.
"""
import csv
import datetime as dt
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
LOTQTY = 650  # 10 lots, 1 pt = Rs.650


def log(msg):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def mid(bid, ask, ltp):
    if bid and ask and bid > 0 and ask > 0:
        return (bid + ask) / 2
    return ltp or 0


def main():
    t0 = time.time()
    con = sqlite3.connect(DB)
    c = con.cursor()
    c.execute("SELECT DISTINCT substr(snapshot_time,1,10) FROM underlying_spot "
              "WHERE symbol='NIFTY' ORDER BY 1")
    days = [r[0] for r in c.fetchall()]
    mondays = [d for d in days if dt.date.fromisoformat(d).weekday() == 0]
    c.execute("SELECT DISTINCT expiry_date FROM option_chain WHERE symbol='NIFTY' ORDER BY 1")
    expiries = [r[0] for r in c.fetchall()]
    log(f"{len(days)} days, {len(mondays)} mondays, {len(expiries)} expiries")

    out = open(os.path.join(RESDIR, "replay_nsrw_cycles.csv"), "w", newline="")
    w = csv.DictWriter(out, fieldnames=[
        "entry_time", "expiry", "T", "pe_k", "ce_k", "pe0", "ce0", "credit",
        "spot0", "events", "exit_time", "exit_reason", "gross", "net",
        "rupees_10lots", "status"])
    w.writeheader()

    exp_cache = {}

    def load_week(exp, d_from, d_to):
        key = (exp, d_from)
        if key in exp_cache:
            return exp_cache[key]
        cur = con.cursor()
        cur.execute(
            "SELECT snapshot_time, strike, instrument_type, ltp, bid, ask, volume, oi, underlying_spot "
            "FROM option_chain WHERE symbol='NIFTY' AND expiry_date=? "
            "AND snapshot_time>=? AND snapshot_time<=? ORDER BY snapshot_time",
            (exp, d_from + "T00:00:00", d_to + "T23:59:59"))
        snaps = defaultdict(dict)   # time -> (strike, type) -> row
        times = []
        for st, k, ot, ltp, bid, ask, vol, oi, us in cur:
            if not times or times[-1] != st:
                times.append(st)
            snaps[st][(float(k), ot)] = (ltp or 0, bid or 0, ask or 0, vol or 0, oi or 0, us or 0)
        exp_cache.clear()  # keep memory bounded; one week at a time
        exp_cache[key] = (times, snaps)
        return exp_cache[key]

    for entry_day in mondays:
        d0 = dt.date.fromisoformat(entry_day)
        exp = next((e for e in expiries if 6 <= (dt.date.fromisoformat(e) - d0).days <= 12), None)
        if not exp:
            continue
        ed = dt.date.fromisoformat(exp)
        # time exit fires on the recorded day with cal DTE == 1 (or expiry morning as backstop)
        cand = [d for d in days if d >= entry_day and (ed - dt.date.fromisoformat(d)).days == 1]
        exit_deadline_day = cand[0] if cand else None
        times, snaps = load_week(exp, entry_day, exp)
        if not times:
            continue
        entry_t = next((t for t in times if t >= entry_day + "T15:14:00"
                        and t[:10] == entry_day), times[0])
        chain0 = snaps[entry_t]
        spot0 = max((v[5] for v in chain0.values()), default=0)
        if not spot0:
            continue

        for T in T_GRID:
            # pick strikes: OTM, liquid, MID nearest T
            best = {"PE": None, "CE": None}
            for (k, ot), v in chain0.items():
                m = mid(v[1], v[2], v[0])
                if m < 1.5 or (v[3] == 0 and v[4] == 0):
                    continue
                if ot == "PE" and k >= spot0:
                    continue
                if ot == "CE" and k <= spot0:
                    continue
                if best[ot] is None or abs(m - T) < abs(best[ot][1] - T):
                    best[ot] = (k, m, v)
            if not best["PE"] or not best["CE"]:
                continue
            pe_k, ce_k = best["PE"][0], best["CE"][0]
            pe0 = best["PE"][2][1] or best["PE"][2][0]   # sell at BID
            ce0 = best["CE"][2][1] or best["CE"][2][0]
            credit = pe0 + ce0
            sl = {"PE": STOP_MULT * pe0, "CE": STOP_MULT * ce0}
            strike = {"PE": pe_k, "CE": ce_k}
            live_credit = {"PE": pe0, "CE": ce0}
            last_q = {}
            realized, rolled = 0.0, False
            exit_prem_sum, events = 0.0, [f"ENTRY {entry_t[11:16]} {int(pe_k)}PE@{pe0:.1f}+{int(ce_k)}CE@{ce0:.1f}"]
            entry_prem = credit
            exit_time = exit_reason = None
            gross = None

            for t in times:
                if t <= entry_t:
                    continue
                ch = snaps[t]
                for side in list(live_credit):
                    q = ch.get((strike[side], side))
                    if q:
                        last_q[side] = q
                # stop check on LTP
                hit = None
                for side in list(live_credit):
                    q = last_q.get(side)
                    if q and q[0] >= sl[side]:
                        hit = side
                        break
                if hit:
                    q = last_q[hit]
                    fill = q[2] if q[2] > 0 else q[0]  # ASK
                    realized += live_credit[hit] - fill
                    exit_prem_sum += fill
                    events.append(f"STOP {t[5:16]} {int(strike[hit])}{hit}@{fill:.1f}")
                    del live_credit[hit]
                    if not live_credit:
                        exit_time, exit_reason = t, "STOP2_FLAT"
                        gross = realized
                        break
                    if not rolled:
                        rolled = True
                        spot_t = max((v[5] for v in ch.values()), default=0)
                        nbest = None
                        for (k, ot), v in ch.items():
                            if ot != hit:
                                continue
                            m = mid(v[1], v[2], v[0])
                            if m < 1.5 or (v[3] == 0 and v[4] == 0):
                                continue
                            if ot == "PE" and spot_t and k >= spot_t:
                                continue
                            if ot == "CE" and spot_t and k <= spot_t:
                                continue
                            if nbest is None or abs(m - T) < abs(nbest[1] - T):
                                nbest = (k, m, v)
                        if nbest:
                            nk, _, nv = nbest
                            nprem = nv[1] or nv[0]  # sell at BID
                            strike[hit] = nk
                            live_credit[hit] = nprem
                            sl[hit] = STOP_MULT * nprem
                            entry_prem += nprem
                            last_q[hit] = nv
                            events.append(f"ROLL {t[5:16]} -> {int(nk)}{hit}@{nprem:.1f}")
                    continue
                # PT on combined MID
                comb_mid = 0.0
                ok = True
                for side in live_credit:
                    q = last_q.get(side)
                    if not q:
                        ok = False
                        break
                    comb_mid += mid(q[1], q[2], q[0])
                if ok and live_credit:
                    profit = sum(live_credit.values()) - comb_mid + realized
                    if profit >= PT_FRAC * credit:
                        cost = sum((last_q[s][2] if last_q[s][2] > 0 else last_q[s][0])
                                   for s in live_credit)
                        gross = sum(live_credit.values()) - cost + realized
                        exit_prem_sum += cost
                        exit_time, exit_reason = t, "PT"
                        events.append(f"PT {t[5:16]}")
                        break
                # time exit: DTE-1 day at 15:15, or expiry-day 09:30 backstop
                if (exit_deadline_day and t[:10] == exit_deadline_day and t[11:16] >= "15:15") \
                        or (t[:10] == exp and t[11:16] >= "09:30"):
                    cost = sum((last_q[s][2] if last_q[s][2] > 0 else last_q[s][0])
                               for s in live_credit)
                    gross = sum(live_credit.values()) - cost + realized
                    exit_prem_sum += cost
                    exit_time, exit_reason = t, "TIME"
                    events.append(f"TIME {t[5:16]}")
                    break

            status = "CLOSED"
            if exit_time is None:  # ran out of data (week still open)
                comb = sum(mid(last_q[s][1], last_q[s][2], last_q[s][0])
                           for s in live_credit if s in last_q)
                gross = sum(live_credit.values()) - comb + realized
                exit_prem_sum += comb
                exit_time, exit_reason, status = times[-1], "OPEN_M2M", "OPEN"
                events.append(f"OPEN m2m {times[-1][5:16]}")
            net = gross - (0.0025 * (entry_prem + exit_prem_sum) + 0.10)
            w.writerow(dict(entry_time=entry_t, expiry=exp, T=T, pe_k=int(pe_k),
                            ce_k=int(ce_k), pe0=round(pe0, 2), ce0=round(ce0, 2),
                            credit=round(credit, 2), spot0=round(spot0, 1),
                            events=" | ".join(events), exit_time=exit_time,
                            exit_reason=exit_reason, gross=round(gross, 2),
                            net=round(net, 2),
                            rupees_10lots=round(net * LOTQTY),
                            status=status))
            log(f"{entry_day} T{T}: {exit_reason} net={net:.1f} ({' | '.join(events)})")
        out.flush()
    out.close()
    log(f"ALL DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
