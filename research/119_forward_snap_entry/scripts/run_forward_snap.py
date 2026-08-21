#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/119 - Forward-ATM vs Spot-ATM entry for the live CSL books.

ONE varied axis: the entry strike rule.
  A (status quo, csl_paper_exec.py): K = round(spot/step)*step
  B (forward-ATM, nas_atm_executor.py): K_A -> forward = K_A + (CE-PE) -> re-round;
     fall back to K_A if the new strike has no usable quotes.

Everything else frozen to backtest_data/csl_paper_config.json: per-DTE entry/exit
windows and combined-SL, 2-poll dwell, 50% backstop on sl=none, EOD force.
Real 1-min chain, READ-ONLY. Net of costs.
"""
import sqlite3, csv, os, json, time, sys
from datetime import date, timedelta

Q = "/home/arun/quantifyd/"
CHAIN = Q + "backtest_data/options_data.db"
CFG = Q + "backtest_data/csl_paper_config.json"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
DETAIL = os.path.join(RES, "fs_detail.csv")
LOG = os.path.join(RES, "run.log")

MKT = {
    "NIFTY":  {"step": 50,  "lot": 65, "slip": 0.5, "wd2dte": {0: 1, 1: 0, 2: 4, 3: 3, 4: 2}},
    "SENSEX": {"step": 100, "lot": 20, "slip": 1.0, "wd2dte": {0: 3, 1: 2, 2: 1, 3: 0, 4: 4}},
}
CHG = 30.0        # Rs per leg-side per lot
BACKSTOP = 0.50
EOD_FORCE = "15:26"
BOOKS = {  # study label -> (config book key, symbol)
    "COMB_NIFTY":   ("NAS_COMB20", "NIFTY"),
    "COMB_SENSEX":  ("CSL30F_SENSEX", "SENSEX"),
    "TIMEB_NIFTY":  ("CSL_TIMEB_NIFTY", "NIFTY"),
    "TIMEB_SENSEX": ("CSL_TIMEB_SENSEX", "SENSEX"),
}

FIELDS = ["book", "sym", "day", "wd", "dte", "arm", "dwell", "entry_hm", "exit_cfg", "sl",
          "K_spot", "K_used", "spot0", "forward", "fwd_gap", "strike_changed",
          "ce0", "pe0", "credit", "skew", "ba_pct", "exit_hm", "exit_comb", "reason",
          "spot_exit", "idx_move_pct", "gross_lot", "net_lot", "lot"]


def log(m):
    with open(LOG, "a") as f:
        f.write(m + "\n")
    print(m, flush=True)


def rec_days(c, sym):
    return [r[0] for r in c.execute(
        "SELECT DISTINCT substr(snapshot_time,1,10) d FROM download_log WHERE symbol=? ORDER BY d",
        (sym,))]


def load_day(c, sym, day):
    """-> (fexp, per_minute {hm: {strike: {'CE':(ltp,bid,ask),'PE':...}}}, spot_by_hm)"""
    rows = c.execute(
        "SELECT snapshot_time, expiry_date, strike, instrument_type, ltp, bid, ask, underlying_spot "
        "FROM option_chain WHERE symbol=? AND snapshot_time LIKE ? AND ltp IS NOT NULL",
        (sym, day + "%")).fetchall()
    if not rows:
        return None
    exps = sorted({e for (_, e, _, _, _, _, _, _) in rows if e and e >= day})
    if not exps:
        return None
    fexp = exps[0]
    per, spot = {}, {}
    for st, e, k, it, ltp, bid, ask, sp in rows:
        if e != fexp:
            continue
        hm = st[11:16]
        per.setdefault(hm, {}).setdefault(k, {})[it] = (ltp, bid, ask)
        if sp and hm not in spot:
            spot[hm] = sp
    return fexp, per, spot


def trading_dte(day, fexp, calset, lastrec):
    """business days strictly after `day` up to `fexp`, skipping observed holidays."""
    d0, d1 = date.fromisoformat(day), date.fromisoformat(fexp)
    if d1 <= d0:
        return 0
    n, x = 0, d0 + timedelta(days=1)
    while x <= d1:
        if x.weekday() < 5:
            iso = x.isoformat()
            if iso in calset or iso > lastrec:
                n += 1
        x += timedelta(days=1)
    return n


def pick_entry(per, spot, entry_hm, step, arm):
    """-> dict with K_spot, K_used, spot0, forward, ce0, pe0, entry_hm  (or None)"""
    mins = sorted(per)
    ehm = next((h for h in mins if h >= entry_hm), None)
    if not ehm:
        return None
    sp0 = spot.get(ehm) or next((spot[h] for h in mins if h >= entry_hm and h in spot), None)
    if not sp0:
        return None
    KA = round(sp0 / step) * step
    a = per[ehm].get(KA, {})
    if "CE" not in a or "PE" not in a:
        return None
    ceA, peA = a["CE"][0], a["PE"][0]
    if not (ceA and peA and ceA > 0 and peA > 0):
        return None
    fwd = KA + (ceA - peA)
    KB = round(fwd / step) * step
    out = dict(K_spot=KA, spot0=sp0, forward=fwd, entry_hm=ehm)
    if arm == "A" or KB == KA:
        out.update(K_used=KA, ce0=ceA, pe0=peA, leg=a)
        return out
    b = per[ehm].get(KB, {})
    if "CE" in b and "PE" in b and b["CE"][0] and b["PE"][0] and b["CE"][0] > 0 and b["PE"][0] > 0:
        out.update(K_used=KB, ce0=b["CE"][0], pe0=b["PE"][0], leg=b)
    else:
        out.update(K_used=KA, ce0=ceA, pe0=peA, leg=a)   # same fallback as the suite
    return out


def replay(per, spot, K, ehm, credit, sl, exit_hm, dwell):
    """-> (exit_hm, exit_comb, reason, spot_exit)"""
    thr = (1 + (BACKSTOP if sl == "none" else sl / 100.0)) * credit
    streak, seen, lastrow = 0, False, None
    for hm in sorted(per):
        if hm < ehm:
            continue
        d = per[hm].get(K, {})
        if "CE" not in d or "PE" not in d:
            continue
        comb = d["CE"][0] + d["PE"][0]
        lastrow = (hm, comb)
        breach = comb >= thr
        if dwell == 0:                       # sensitivity: exit on the breach minute itself
            if breach:
                return hm, comb, "SL_IMMEDIATE", spot.get(hm)
        elif seen and streak >= dwell:       # live: N consecutive breach polls -> exit next poll
            return hm, comb, "SL_DWELL", spot.get(hm)
        streak = streak + 1 if breach else 0
        seen = True
        if hm >= exit_hm:
            return hm, comb, "TIME_EXIT", spot.get(hm)
        if hm >= EOD_FORCE:
            return hm, comb, "EOD_FORCE", spot.get(hm)
    if lastrow:
        return lastrow[0], lastrow[1], "EOD", spot.get(lastrow[0])
    return None


def spot_at(spot, hm):
    ks = [h for h in sorted(spot) if h <= hm]
    return spot[ks[-1]] if ks else None


def main():
    dwells = [2, 0] if "--dwell-sens" in sys.argv else [2]
    os.makedirs(RES, exist_ok=True)
    cfg = json.load(open(CFG))["books"]
    c = sqlite3.connect("file:%s?mode=ro" % CHAIN, uri=True)
    cal = {s: rec_days(c, s) for s in ("NIFTY", "SENSEX")}
    out = open(DETAIL, "w", newline="")
    w = csv.DictWriter(out, fieldnames=FIELDS)
    w.writeheader()
    n = 0
    for sym in ("NIFTY", "SENSEX"):
        M = MKT[sym]
        days = cal[sym]
        calset = set(days)
        lastrec = days[-1]
        books = [(lbl, cfg[key]) for lbl, (key, s) in BOOKS.items() if s == sym]
        log("== %s: %d recorded days (%s .. %s)" % (sym, len(days), days[0], days[-1]))
        for day in days:
            d = load_day(c, sym, day)
            if not d:
                log("%s %s: no chain" % (sym, day))
                continue
            fexp, per, spot = d
            dte_cal = trading_dte(day, fexp, calset, lastrec)
            wd = date.fromisoformat(day).weekday()
            dte_map = M["wd2dte"].get(wd)
            # the live daemon keys its config off the weekday map -> replay that exactly.
            # (dte_cal is only a cross-check; recording gaps can make it disagree.)
            if dte_map is not None and dte_map != dte_cal:
                log("%s %s: DTE map=%s cal=%s (fexp %s) -> using MAP (live behaviour)" % (sym, day, dte_map, dte_cal, fexp))
            dte = dte_map if dte_map is not None else dte_cal
            for lbl, bcfg in books:
                cc = bcfg.get(str(dte))
                if not cc:
                    continue
                for arm in ("A", "B"):
                    e = pick_entry(per, spot, cc["entry"], M["step"], arm)
                    if not e:
                        continue
                    credit = e["ce0"] + e["pe0"]
                    for dw in dwells:
                        r = replay(per, spot, e["K_used"], e["entry_hm"], credit, cc["sl"], cc["exit"], dw)
                        if not r:
                            continue
                        xhm, xcomb, reason, sx = r
                        sx = sx or spot_at(spot, xhm)
                        gross = (credit - xcomb) * M["lot"]
                        net = gross - 4 * M["slip"] * M["lot"] - 4 * CHG
                        ce_t = e["leg"]["CE"]
                        pe_t = e["leg"]["PE"]
                        ba = []
                        for (l, b, a) in (ce_t, pe_t):
                            if b and a and a > b and l and l > 0:
                                ba.append(100.0 * (a - b) / ((a + b) / 2.0))
                        w.writerow({
                            "book": lbl, "sym": sym, "day": day, "wd": wd, "dte": dte, "arm": arm,
                            "dwell": dw, "entry_hm": e["entry_hm"], "exit_cfg": cc["exit"], "sl": cc["sl"],
                            "K_spot": e["K_spot"], "K_used": e["K_used"], "spot0": round(e["spot0"], 2),
                            "forward": round(e["forward"], 2), "fwd_gap": round(e["forward"] - e["spot0"], 2),
                            "strike_changed": int(e["K_used"] != e["K_spot"]),
                            "ce0": e["ce0"], "pe0": e["pe0"], "credit": round(credit, 2),
                            "skew": round(e["ce0"] - e["pe0"], 2),
                            "ba_pct": round(sum(ba) / len(ba), 3) if ba else "",
                            "exit_hm": xhm, "exit_comb": round(xcomb, 2), "reason": reason,
                            "spot_exit": round(sx, 2) if sx else "",
                            "idx_move_pct": round(100.0 * (sx - e["spot0"]) / e["spot0"], 4) if sx else "",
                            "gross_lot": round(gross), "net_lot": round(net), "lot": M["lot"]})
                        n += 1
            out.flush()
            time.sleep(0.02)
    out.close()
    log("rows written: %d -> %s" % (n, DETAIL))


if __name__ == "__main__":
    main()
