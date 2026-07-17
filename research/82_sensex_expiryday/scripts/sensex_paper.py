#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SENSEX expiry-day PAPER book (research/82). Standalone; places NO orders.

WHY: NIFTY expires Tuesday, SENSEX Thursday. NAS already harvests NIFTY's 0/1-DTE edge (Mon/Tue).
SENSEX's Thursday expiry lets the SAME expiry-day systems cover Wed(DTE1)/Thu(DTE0) — the days
NAS-OPT leaves idle. This paper-trades it and records the evidence.

TWO variants, mirroring the NAS shapes, entered 09:20 every weekday:
  straddle : SELL ATM CE+PE, per-leg 30% stop, exit 15:15        (the NAS-916/ATM shape)
  strangle : SELL ~0.4%-OTM CE+PE, +/-0.4% move-stop, exit 14:45  (the NAS-OPT shape)

RUNS EVERY DAY (user's ask), but never mixes the evidence:
  DTE<=1 (Wed/Thu) -> signal_class 'system'         <- the strategy; the headline number
  DTE>=2           -> signal_class 'observational'  <- recorded only, never quoted as the system
(Exactly the split NAS-OPT uses, after research/79 showed the far-DTE days are EV-negative there.)

Marks off the real recorded chain (options_data.db). Cron-driven (no Flask restart).
State: backtest_data/sensex_paper_state.json | Output: static/app/sensex_paper.json
Usage: sensex_paper.py            -> today (enter/mark/exit)
       sensex_paper.py --backfill -> replay every recorded day (resumable; skips days already done)
"""
import json, os, sqlite3, sys, time
from collections import defaultdict
from datetime import datetime, timedelta

ROOT = "/home/arun/quantifyd"
OPT = f"{ROOT}/backtest_data/options_data.db"
STATE = f"{ROOT}/backtest_data/sensex_paper_state.json"
OUT = f"{ROOT}/static/app/sensex_paper.json"

SYM, STEP = "SENSEX", 100
LOT, LOTS = 20, 2
QTY = LOT * LOTS                 # 40
BROK = 20                        # per leg, round trip
MOVE = 0.004                     # 0.4% OTM + 0.4% underlying move-stop
SL_MULT = 1.30                   # 30% per-leg stop (straddle)
T_ENTRY = 9 * 3600 + 20 * 60
T_STRAD_EXIT = 15 * 3600 + 15 * 60
T_STRANGLE_EXIT = 14 * 3600 + 45 * 60


def tsec(s):
    return int(s[11:13]) * 3600 + int(s[14:16]) * 60 + int(s[17:19])


def hm(sec):
    return "%02d:%02d" % (sec // 3600, (sec % 3600) // 60)


def load():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {"history": []}


def save(st):
    json.dump(st, open(STATE, "w"), indent=1)


def day_chain(c, day, E):
    lo, hi = day + "T00:00", day + "T23:59"
    rows = c.execute("SELECT snapshot_time,strike,instrument_type,ltp,underlying_spot FROM option_chain "
                     "WHERE symbol=? AND snapshot_time>=? AND snapshot_time<=? AND expiry_date=? AND ltp>0 "
                     "ORDER BY snapshot_time", (SYM, lo, hi, E)).fetchall()
    chain, spot = defaultdict(dict), {}
    for st_, k, ot, ltp, us in rows:
        t = tsec(st_)
        chain[t][(int(k), ot)] = ltp
        if us:
            spot[t] = float(us)
    return chain, spot, sorted(chain)


def run_day(c, day, E, dte):
    """Return list of trade dicts for this day (straddle + strangle)."""
    chain, spot, times = day_chain(c, day, E)
    if not times:
        return []
    s0 = next((spot[t] for t in times if t >= T_ENTRY and t in spot), None)
    if not s0:
        return []
    cls = "system" if dte <= 1 else "observational"
    wd = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][datetime.fromisoformat(day).weekday()]

    def entry_px(K, ot):
        for t in times:
            if t >= T_ENTRY and (K, ot) in chain[t]:
                return chain[t][(K, ot)], t
        return None, None

    def px_at(tt, K, ot, fb):
        best = fb
        for t in times:
            if t > tt:
                break
            if (K, ot) in chain[t]:
                best = chain[t][(K, ot)]
        return best

    out = []
    # ---- straddle: ATM, per-leg 30% stop, exit 15:15 ----
    atm = round(s0 / STEP) * STEP
    ce0, t_in = entry_px(atm, "CE")
    pe0, _ = entry_px(atm, "PE")
    if ce0 and pe0:
        def leg_exit(K, ot, e):
            cap = e * SL_MULT
            for t in times:
                if T_ENTRY <= t <= T_STRAD_EXIT:
                    p = chain[t].get((K, ot))
                    if p is not None and p >= cap:
                        return p, t, "SL"
            return px_at(T_STRAD_EXIT, K, ot, e), T_STRAD_EXIT, "time1515"
        cx, tcx, rc = leg_exit(atm, "CE", ce0)
        px_, tpx, rp = leg_exit(atm, "PE", pe0)
        pnl = ((ce0 - cx) + (pe0 - px_)) * QTY - 2 * BROK
        legs = [
            {"type": "CE", "side": "SELL", "strike": atm, "qty": QTY, "entry": round(ce0, 2),
             "ltp": round(cx, 2), "pnl": round((ce0 - cx) * QTY)},
            {"type": "PE", "side": "SELL", "strike": atm, "qty": QTY, "entry": round(pe0, 2),
             "ltp": round(px_, 2), "pnl": round((pe0 - px_) * QTY)},
        ]
        out.append(dict(day=day, weekday=wd, dte=dte, signal_class=cls, variant="straddle",
                        entry_spot=round(s0, 2), ce_strike=atm, pe_strike=atm,
                        ce_entry=ce0, pe_entry=pe0, credit=round(ce0 + pe0, 2),
                        ce_exit=cx, pe_exit=px_, exit_reason=("SL" if "SL" in (rc, rp) else "time1515"),
                        entry_time=hm(t_in), exit_time=hm(max(tcx, tpx)), pnl=round(pnl), legs=legs))
    # ---- strangle: ~0.4% OTM, 0.4% move-stop, exit 14:45 ----
    ck = round(s0 * (1 + MOVE) / STEP) * STEP
    pk = round(s0 * (1 - MOVE) / STEP) * STEP
    ce0, t_in = entry_px(ck, "CE")
    pe0, _ = entry_px(pk, "PE")
    if ce0 and pe0:
        et = None
        for t in times:
            if T_ENTRY <= t <= T_STRANGLE_EXIT and spot.get(t) and abs(spot[t] - s0) / s0 >= MOVE:
                et = t
                break
        tx = et or T_STRANGLE_EXIT
        cx = px_at(tx, ck, "CE", ce0)
        px_ = px_at(tx, pk, "PE", pe0)
        pnl = ((ce0 - cx) + (pe0 - px_)) * QTY - 2 * BROK
        legs = [
            {"type": "CE", "side": "SELL", "strike": ck, "qty": QTY, "entry": round(ce0, 2),
             "ltp": round(cx, 2), "pnl": round((ce0 - cx) * QTY)},
            {"type": "PE", "side": "SELL", "strike": pk, "qty": QTY, "entry": round(pe0, 2),
             "ltp": round(px_, 2), "pnl": round((pe0 - px_) * QTY)},
        ]
        out.append(dict(day=day, weekday=wd, dte=dte, signal_class=cls, variant="strangle",
                        entry_spot=round(s0, 2), ce_strike=ck, pe_strike=pk,
                        ce_entry=ce0, pe_entry=pe0, credit=round(ce0 + pe0, 2),
                        ce_exit=cx, pe_exit=px_, exit_reason=("move0.4%" if et else "time1445"),
                        entry_time=hm(t_in), exit_time=hm(tx), pnl=round(pnl), legs=legs))
    return out


def publish(st):
    h = sorted(st["history"], key=lambda x: (x["day"], x["variant"]))
    def curve(rows):
        cum, out = 0, []
        for r in sorted(rows, key=lambda x: x["day"]):
            cum += r["pnl"]
            out.append([r["day"], cum])
        return out
    payload = {
        "name": "SENSEX expiry-day (paper)",
        "subtitle": "SENSEX expires Thursday — fills the Wed/Thu days NAS-OPT leaves idle. "
                    "Entry 09:20 daily. DTE0/1 = the system; other DTEs recorded as observation only.",
        "study": "/app/backtest/fardte-rescue",
        "mode": "paper", "lots": LOTS, "qty": QTY, "lot_size": LOT,
        "updated": (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %H:%M:%S"),
        "today_day": (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d"),
        "variants": {},
        "history": list(reversed(h))[:400],
    }
    for v in ("straddle", "strangle"):
        rows = [r for r in h if r["variant"] == v]
        sysr = [r for r in rows if r["signal_class"] == "system"]
        obsr = [r for r in rows if r["signal_class"] != "system"]
        def agg(rs):
            if not rs:
                return {"n": 0, "pnl": 0, "win": None}
            return {"n": len(rs), "pnl": sum(r["pnl"] for r in rs),
                    "win": round(100 * sum(1 for r in rs if r["pnl"] > 0) / len(rs))}
        payload["variants"][v] = {"all": agg(rows), "system": agg(sysr), "observational": agg(obsr),
                                  "curve": curve(sysr), "curve_all": curve(rows)}
    json.dump(payload, open(OUT, "w"), indent=1)


def main():
    st = load()
    done = {(r["day"], r["variant"]) for r in st["history"]}
    c = sqlite3.connect(f"file:{OPT}?mode=ro", uri=True)
    backfill = "--backfill" in sys.argv

    if backfill:
        rows = c.execute("SELECT substr(snapshot_time,1,10) d, MIN(expiry_date) e FROM option_chain "
                         "WHERE symbol=? AND expiry_date>=substr(snapshot_time,1,10) GROUP BY d ORDER BY d",
                         (SYM,)).fetchall()
        targets = [(d, e) for d, e in rows]
    else:
        today = (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d")
        e = c.execute("SELECT MIN(expiry_date) FROM option_chain WHERE symbol=? AND snapshot_time>=? "
                      "AND snapshot_time<=? AND expiry_date>=?",
                      (SYM, today + "T00:00", today + "T23:59", today)).fetchone()[0]
        targets = [(today, e)] if e else []

    for i, (day, E) in enumerate(targets):
        if not E:
            continue
        dte = (datetime.fromisoformat(E).date() - datetime.fromisoformat(day).date()).days
        # Today's row is LIVE — recompute it every run so the marks tick. Past days are frozen.
        _today = (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d")
        if day != _today and all((day, v) in done for v in ("straddle", "strangle")):
            continue
        if day == _today:
            st["history"] = [r for r in st["history"] if r["day"] != _today]
            done = {(r["day"], r["variant"]) for r in st["history"]}
        t0 = time.time()
        try:
            trades = run_day(c, day, E, dte)
        except Exception as ex:
            print("  %s ERR %s" % (day, ex)); sys.stdout.flush(); continue
        for t in trades:
            if (t["day"], t["variant"]) not in done:
                st["history"].append(t); done.add((t["day"], t["variant"]))
        save(st); publish(st)          # save after EVERY day -> resumable
        if trades:
            print("  [%d/%d] %s DTE%d %s  %s" % (i + 1, len(targets), day, dte, trades[0]["signal_class"],
                  " ".join("%s=%+d" % (t["variant"][:4], t["pnl"]) for t in trades)))
            sys.stdout.flush()
    c.close()
    publish(st)
    h = st["history"]
    print("\nrecorded %d trades over %d days" % (len(h), len({r["day"] for r in h})))
    for v in ("straddle", "strangle"):
        sysr = [r["pnl"] for r in h if r["variant"] == v and r["signal_class"] == "system"]
        obsr = [r["pnl"] for r in h if r["variant"] == v and r["signal_class"] != "system"]
        if sysr:
            print("  %-9s SYSTEM(DTE0/1) n=%-3d total %+7d  mean %+6.0f  win %3.0f%%" % (
                v, len(sysr), sum(sysr), sum(sysr) / len(sysr),
                100 * sum(1 for x in sysr if x > 0) / len(sysr)))
        if obsr:
            print("  %-9s observational n=%-3d total %+7d  mean %+6.0f  win %3.0f%%" % (
                v, len(obsr), sum(obsr), sum(obsr) / len(obsr),
                100 * sum(1 for x in obsr if x > 0) / len(obsr)))
    print("DONE")


if __name__ == "__main__":
    main()
