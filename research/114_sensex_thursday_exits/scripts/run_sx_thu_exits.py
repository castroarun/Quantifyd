#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/114 — SENSEX Thursday (DTE0) exit-rule bake-off on the real 1-min chain.

One entry (09:16 ATM straddle, the live suite's behaviour), 17 recorded Thursdays,
per lot (20). Every variant sees the SAME minute series, so differences are the rule
alone. Net of costs. Writes an incremental CSV + a summary.
"""
import sqlite3, csv, os, json
from datetime import date

Q = "/home/arun/quantifyd/"
CHAIN = Q + "backtest_data/options_data.db"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
DETAIL = os.path.join(RES, "sx_thu_detail.csv")
SUMMARY = os.path.join(RES, "sx_thu_summary.csv")
LOG = os.path.join(RES, "run.log")

LOT = 20
SLIP = 1.0          # pts per leg-side (SENSEX spreads are wide)
CHG = 30.0          # Rs per leg-side per lot
ENTRY_HM = "09:16"
EOD_HM = "15:15"


def log(m):
    with open(LOG, "a") as f:
        f.write(m + "\n")
    print(m, flush=True)


def thursdays(c):
    days = [r[0] for r in c.execute(
        "SELECT DISTINCT substr(snapshot_time,1,10) d FROM download_log "
        "WHERE symbol='SENSEX' ORDER BY d")]
    return [d for d in days if date.fromisoformat(d).weekday() == 3]


def day_series(c, day):
    """(strike, credit, [(hm, ce, pe, comb)]) for the 09:16 ATM straddle, front expiry."""
    rows = c.execute(
        "SELECT snapshot_time, expiry_date, strike, instrument_type, ltp, underlying_spot "
        "FROM option_chain WHERE symbol='SENSEX' AND snapshot_time LIKE ? AND ltp IS NOT NULL",
        (day + "%",)).fetchall()
    if not rows:
        return None
    exps = sorted({e for (_, e, _, _, _, _) in rows if e and e >= day})
    if not exps:
        return None
    fexp = exps[0]
    per = {}
    spot0 = None
    for st, e, k, it, ltp, sp in rows:
        if e != fexp:
            continue
        hm = st[11:16]
        d = per.setdefault(hm, {})
        d.setdefault(k, {})[it] = ltp
        if hm <= ENTRY_HM and sp:
            spot0 = sp
    if not spot0:
        return None
    entry_hm = min((h for h in per if h >= ENTRY_HM), default=None)
    if not entry_hm:
        return None
    K = round(spot0 / 100) * 100
    e0 = per[entry_hm].get(K, {})
    if "CE" not in e0 or "PE" not in e0:
        return None
    ser = []
    for hm in sorted(per):
        if hm < entry_hm or hm > EOD_HM:
            continue
        d = per[hm].get(K, {})
        if "CE" in d and "PE" in d:
            ser.append((hm, d["CE"], d["PE"], d["CE"] + d["PE"]))
    return (K, e0["CE"] + e0["PE"], e0["CE"], e0["PE"], ser) if len(ser) > 30 else None


def variants():
    v = [("HOLD", {}), ("TP1667", {"tp": 1667}), ("TP1000", {"tp": 1000}),
         ("TP2500", {"tp": 2500}), ("TP4000", {"tp": 4000}),
         ("LEG30", {"leg": 0.30}), ("LEG30_TP1667", {"leg": 0.30, "tp": 1667}),
         ("COMB25", {"comb": 0.25}), ("COMB30", {"comb": 0.30}), ("COMB40", {"comb": 0.40}),
         ("STOP1300", {"stop": 1300}), ("STOP1300_TP1667", {"stop": 1300, "tp": 1667}),
         ("TRAIL2000_350", {"arm": 2000, "give": 350}),
         ("TRAIL3000_1000", {"arm": 3000, "give": 1000}),
         ("TIME1200", {"time": "12:00"}), ("TIME1300", {"time": "13:00"}),
         ("TIME1400", {"time": "14:00"})]
    return v


def replay(cfg, credit, ce0, pe0, ser):
    """Returns (exit_hm, exit_comb, reason, legs_traded)."""
    peak = 0.0
    armed = False
    for hm, ce, pe, comb in ser:
        pnl = (credit - comb) * LOT
        if pnl > peak:
            peak = pnl
        # per-leg stop: either leg 30% above its own entry -> exit whole straddle
        if cfg.get("leg"):
            if ce >= ce0 * (1 + cfg["leg"]) or pe >= pe0 * (1 + cfg["leg"]):
                return hm, comb, "LEG_SL", 2
        if cfg.get("comb") and comb >= credit * (1 + cfg["comb"]):
            return hm, comb, "COMB_SL", 2
        if cfg.get("stop") and pnl <= -cfg["stop"] * 1:
            return hm, comb, "BOOK_STOP", 2
        if cfg.get("tp") and pnl >= cfg["tp"] * 1:
            return hm, comb, "TP", 2
        if cfg.get("arm"):
            if not armed and pnl >= cfg["arm"]:
                armed = True
            if armed and pnl <= peak - cfg["give"]:
                return hm, comb, "TRAIL", 2
        if cfg.get("time") and hm >= cfg["time"]:
            return hm, comb, "TIME", 2
    return ser[-1][0], ser[-1][3], "EOD", 2


def main():
    os.makedirs(RES, exist_ok=True)
    c = sqlite3.connect("file:%s?mode=ro" % CHAIN, uri=True)
    days = thursdays(c)
    log("SENSEX Thursdays: %d (%s .. %s)" % (len(days), days[0], days[-1]))
    with open(DETAIL, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "day", "strike", "credit",
                                          "exit_hm", "exit_comb", "reason", "gross", "net"])
        w.writeheader()
        for day in days:
            d = day_series(c, day)
            if not d:
                log("%s skip (no clean series)" % day)
                continue
            K, credit, ce0, pe0, ser = d
            for name, cfg in variants():
                hm, comb, reason, legs = replay(cfg, credit, ce0, pe0, ser)
                gross = (credit - comb) * LOT
                net = gross - (legs * 2 * SLIP * LOT) - (legs * 2 * CHG)
                w.writerow({"variant": name, "day": day, "strike": K,
                            "credit": round(credit, 2), "exit_hm": hm,
                            "exit_comb": round(comb, 2), "reason": reason,
                            "gross": round(gross), "net": round(net)})
    # summary
    import statistics as st
    rows = list(csv.DictReader(open(DETAIL)))
    out = []
    for name, _ in variants():
        vr = [r for r in rows if r["variant"] == name]
        if not vr:
            continue
        nets = [float(r["net"]) for r in vr]
        out.append({"variant": name, "n": len(nets), "total": round(sum(nets)),
                    "mean": round(st.mean(nets)), "median": round(st.median(nets)),
                    "win_pct": round(100 * sum(1 for x in nets if x > 0) / len(nets)),
                    "worst": round(min(nets)), "best": round(max(nets)),
                    "p25": round(sorted(nets)[max(0, len(nets) // 4 - 1)])})
    out.sort(key=lambda r: -r["total"])
    with open(SUMMARY, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        for r in out:
            w.writerow(r)
    log("== SENSEX THURSDAY EXIT BAKE-OFF (net Rs per LOT, %d Thursdays) ==" % out[0]["n"])
    for r in out:
        log("  %-16s total=%7d mean=%6d med=%6d win=%3d%% worst=%7d best=%6d"
            % (r["variant"], r["total"], r["mean"], r["median"], r["win_pct"],
               r["worst"], r["best"]))


if __name__ == "__main__":
    main()
