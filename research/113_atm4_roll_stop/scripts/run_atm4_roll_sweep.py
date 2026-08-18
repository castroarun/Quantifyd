#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/113 — ATM4 roll-leg stop calibration on real 1-min NIFTY option chain.

Faithful replay of services/nas_atm4_executor.py mechanics (entry 09:16 ATM straddle,
per-leg 1.3x SL, roll-to-match on first SL, CASE-2 tighten on second SL, EOD 15:15),
with ONE axis varied: the ROLLED leg's stop rule (9 variants, see STATUS-MD sec 4).
Incremental CSV; resumable; --aggregate-only to re-summarise.
"""
import sqlite3, csv, os, sys
from datetime import date, timedelta

DB = "/home/arun/quantifyd/backtest_data/options_data.db"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
DAILY = os.path.join(RES, "atm4_roll_daily.csv")
SUMMARY = os.path.join(RES, "atm4_roll_summary.csv")
LOG = os.path.join(RES, "run.log")

LOT = 65                       # NIFTY lot; live ATM4 runs qty 130 = 2 lots
SLIP = 0.5                     # pts per leg-side
CHG = 30.0                     # Rs per leg-side per lot (brokerage+stat, stated in STATUS)
ENTRY_HM = "09:16"
EOD_HM = "15:15"
STEP = 50
ROLL_MIN_OTM = 50

VARIANTS = ["SQ", "P150", "P200", "F8", "F12", "SURV", "MAXV", "NOSL", "MIN15", "NOROLL"]

FIELDS = ["variant", "day", "dte", "gross", "net", "legs_traded", "rolled",
          "roll_prem", "roll_sl_gap", "roll_restopped", "roll_restop_min",
          "first_sl_hm", "exit_detail"]


def log(m):
    with open(LOG, "a") as f:
        f.write(m + "\n")
    print(m, flush=True)


def trading_dte(day_s, exp_s):
    d = date.fromisoformat(day_s); e = date.fromisoformat(exp_s)
    n, cur = 0, d
    while cur < e:
        cur += timedelta(days=1)
        if cur.weekday() < 5:
            n += 1
    return n


def day_snaps(c, day):
    """{'HH:MM': (spot, {strike:{'CE':ltp,'PE':ltp}})} for the front weekly expiry + expiry str."""
    rows = c.execute(
        "SELECT snapshot_time,expiry_date,strike,instrument_type,ltp,underlying_spot "
        "FROM option_chain WHERE symbol='NIFTY' AND snapshot_time LIKE ? AND ltp IS NOT NULL",
        (day + "%",)).fetchall()
    if not rows:
        return {}, None
    exps = sorted({e for (_, e, _, _, _, _) in rows if e and e >= day})
    if not exps:
        return {}, None
    fexp = exps[0]
    snaps = {}
    for st, e, k, it, ltp, sp in rows:
        if e != fexp:
            continue
        hm = st[11:16]
        d = snaps.setdefault(hm, [None, {}])
        if sp:
            d[0] = sp
        d[1].setdefault(k, {})[it] = ltp
    return {hm: (v[0], v[1]) for hm, v in snaps.items() if v[0]}, fexp


def find_roll_strike(chain, spot, inst, target):
    """Replicates _find_roll_strike: OTM scan from >=50 pts, premium closest to target."""
    atm = round(spot / STEP) * STEP
    best_k, best_p, best_d = None, None, float("inf")
    for i in range(15):
        k = atm + i * STEP if inst == "CE" else atm - i * STEP
        if (k - spot if inst == "CE" else spot - k) < ROLL_MIN_OTM:
            continue
        p = chain.get(k, {}).get(inst)
        if not p or p <= 0:
            continue
        d = abs(p - target)
        if d < best_d:
            best_d, best_k, best_p = d, k, p
        elif best_k is not None and p < target:
            break
    return best_k, best_p


def roll_sl(variant, rp, price_x):
    if variant in ("SQ", "MIN15"):
        return rp * 1.3
    if variant == "P150":
        return rp * 1.5
    if variant == "P200":
        return rp * 2.0
    if variant == "F8":
        return rp + max(0.3 * rp, 8.0)
    if variant == "F12":
        return rp + max(0.3 * rp, 12.0)
    if variant == "SURV":
        return price_x * 1.3
    if variant == "MAXV":
        return max(price_x, rp) * 1.3
    return None  # NOSL


def replay(variant, snaps):
    """Replay one day for one variant. Returns row dict fields (gross/net per lot etc.)."""
    hms = sorted(snaps)
    ehm = next((h for h in hms if h >= ENTRY_HM), None)
    if ehm is None:
        return None
    spot0, chain0 = snaps[ehm]
    atm = round(spot0 / STEP) * STEP
    e = chain0.get(atm, {})
    if "CE" not in e or "PE" not in e:
        return None
    legs = {  # sym -> dict
        "CE": {"strike": atm, "entry": e["CE"], "sl": e["CE"] * 1.3, "open": True, "exit": None},
        "PE": {"strike": atm, "entry": e["PE"], "sl": e["PE"] * 1.3, "open": True, "exit": None},
    }
    sides = 4  # entry 2 sells + eventual 2 buys minimum; recount at end via legs_traded
    n_legs = 2
    rolled = False
    roll_leg = None          # side name of rolled leg ('CE'/'PE')
    roll_info = {"prem": None, "gap": None, "restop": 0, "restop_min": None, "hm": None}
    price_x_saved = None
    second_done = False
    first_hm = None
    detail = []

    def prem(chain, side):
        L = legs[side]
        return chain.get(L["strike"], {}).get(side)

    for hm in hms:
        if hm <= ehm:
            continue
        if hm >= EOD_HM:
            break
        spot, chain = snaps[hm]
        # collect breaches this minute among open legs with a stop
        br = []
        for side in ("CE", "PE"):
            L = legs[side]
            if not L["open"] or L["sl"] is None:
                continue
            p = prem(chain, side)
            if p is not None and p >= L["sl"]:
                br.append((p - L["sl"], side, p))
        if not br:
            continue
        br.sort(reverse=True)  # biggest breach first
        for _, side, p in br:
            L = legs[side]
            if not L["open"]:
                continue
            other = "PE" if side == "CE" else "CE"
            O = legs[other]
            if not rolled and not second_done:
                # ---- FIRST SL ----
                L["open"] = False; L["exit"] = p; L["reason"] = "SL1"
                first_hm = hm
                detail.append("%s SL1 %s@%.1f" % (hm, side, p))
                px = prem(chain, other)
                if px is None or px <= 0 or not O["open"]:
                    if O["open"]:
                        O["open"] = False; O["exit"] = px or 0; O["reason"] = "BOUNDARY"
                    second_done = True
                    continue
                price_x_saved = px
                if variant == "NOROLL" or (variant == "MIN15" and px < 15.0):
                    O["sl"] = px * 1.3
                    rolled = True   # consumes the one roll slot (no new leg)
                    detail.append("%s no-roll survivor_sl=%.1f" % (hm, O["sl"]))
                    continue
                k, rp = find_roll_strike(chain, spot, side, px)
                if k is None:
                    O["open"] = False; O["exit"] = px; O["reason"] = "BOUNDARY"
                    second_done = True
                    detail.append("%s no-strike boundary" % hm)
                    continue
                legs[side] = {"strike": k, "entry": rp, "sl": roll_sl(variant, rp, px),
                              "open": True, "exit": None, "is_roll": True}
                O["sl"] = px * 1.3
                rolled = True
                n_legs += 1
                roll_leg = side
                roll_info["prem"] = rp
                sl_v = legs[side]["sl"]
                roll_info["gap"] = (sl_v - rp) if sl_v else None
                roll_info["hm"] = hm
                detail.append("%s roll %s->%d@%.1f sl=%s" % (hm, side, k, rp,
                              ("%.1f" % sl_v) if sl_v else "none"))
            elif not second_done:
                # ---- SECOND SL ----
                L["open"] = False; L["exit"] = p; L["reason"] = "SL2"
                second_done = True
                detail.append("%s SL2 %s@%.1f" % (hm, side, p))
                if L.get("is_roll"):
                    roll_info["restop"] = 1
                    if roll_info["hm"]:
                        h1, m1 = map(int, roll_info["hm"].split(":"))
                        h2, m2 = map(int, hm.split(":"))
                        roll_info["restop_min"] = (h2 * 60 + m2) - (h1 * 60 + m1)
                if O["open"] and price_x_saved:
                    O["sl"] = price_x_saved  # flat tighten (CASE 2)
            else:
                # ---- THIRD breach: close remaining ----
                L["open"] = False; L["exit"] = p; L["reason"] = "SL3"
                detail.append("%s SL3 %s@%.1f" % (hm, side, p))

    # EOD square-off
    eod = next((h for h in reversed(hms) if h < EOD_HM), None)
    _, chain_e = snaps[eod]
    for side in ("CE", "PE"):
        L = legs[side]
        if L["open"]:
            p = prem(chain_e, side)
            L["open"] = False; L["exit"] = p if p is not None else L["entry"]
            L["reason"] = "EOD"

    gross = sum((L["entry"] - L["exit"]) for L in legs.values() if L["exit"] is not None)
    # legs actually traded this day (each = 1 sell + 1 buy = 2 sides)
    sides = n_legs * 2
    net = gross - sides * SLIP
    gross_rs = gross * LOT
    net_rs = net * LOT - sides * CHG
    return {"gross": round(gross_rs), "net": round(net_rs), "legs_traded": n_legs,
            "rolled": int(rolled and roll_leg is not None),
            "roll_prem": roll_info["prem"], "roll_sl_gap": roll_info["gap"],
            "roll_restopped": roll_info["restop"], "roll_restop_min": roll_info["restop_min"],
            "first_sl_hm": first_hm, "exit_detail": ";".join(detail)[:220]}


def aggregate():
    import statistics as st
    rows = list(csv.DictReader(open(DAILY)))
    out = []
    for v in VARIANTS:
        vr = [r for r in rows if r["variant"] == v]
        if not vr:
            continue
        nets = [float(r["net"]) for r in vr]
        rolls = [r for r in vr if r["rolled"] == "1"]
        restop = [r for r in rolls if r["roll_restopped"] == "1"]
        fast = [r for r in restop if r["roll_restop_min"] and float(r["roll_restop_min"]) <= 15]
        rec = {"variant": v, "days": len(vr), "net_total": round(sum(nets)),
               "net_mean": round(st.mean(nets)), "net_median": round(st.median(nets)),
               "p05": round(sorted(nets)[max(0, int(0.05 * len(nets)) - 1)]),
               "win_pct": round(100 * sum(1 for x in nets if x > 0) / len(nets)),
               "n_rolls": len(rolls),
               "restop_pct": round(100 * len(restop) / len(rolls)) if rolls else "",
               "restop_le15m_pct": round(100 * len(fast) / len(rolls)) if rolls else ""}
        # per-DTE net means
        for d in (0, 1, 2, 3, 4):
            dn = [float(r["net"]) for r in vr if r["dte"] == str(d)]
            rec["dte%d_mean" % d] = round(st.mean(dn)) if dn else ""
        out.append(rec)
    with open(SUMMARY, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        for r in out:
            w.writerow(r)
    log("== SUMMARY (net Rs/lot, %d days) ==" % max(r["days"] for r in out))
    for r in sorted(out, key=lambda x: -x["net_total"]):
        log("  %-7s total=%7d mean=%5d med=%5d p05=%6d win=%2s%% rolls=%s restop=%s%% (<=15m %s%%)"
            % (r["variant"], r["net_total"], r["net_mean"], r["net_median"], r["p05"],
               r["win_pct"], r["n_rolls"], r["restop_pct"], r["restop_le15m_pct"]))


def main():
    os.makedirs(RES, exist_ok=True)
    if "--aggregate-only" in sys.argv:
        aggregate(); return
    c = sqlite3.connect("file:%s?mode=ro" % DB, uri=True)
    days = [r[0] for r in c.execute(
        "SELECT DISTINCT substr(snapshot_time,1,10) d FROM download_log "
        "WHERE symbol='NIFTY' ORDER BY d")]
    days = [d for d in days if d < date.today().isoformat()]  # exclude today (partial)
    done = set()
    if os.path.exists(DAILY):
        done = {(r["variant"], r["day"]) for r in csv.DictReader(open(DAILY))}
        log("resume: %d rows already done" % len(done))
    else:
        with open(DAILY, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    log("days=%d variants=%d" % (len(days), len(VARIANTS)))
    for i, day in enumerate(days):
        if all((v, day) in done for v in VARIANTS):
            continue
        snaps, fexp = day_snaps(c, day)
        if not snaps or len(snaps) < 100:
            log("%s skip (snaps=%d)" % (day, len(snaps)))
            continue
        dte = trading_dte(day, fexp)
        with open(DAILY, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            for v in VARIANTS:
                if (v, day) in done:
                    continue
                r = replay(v, snaps)
                if r is None:
                    continue
                r.update(variant=v, day=day, dte=dte)
                w.writerow(r)
        if (i + 1) % 10 == 0:
            log("[%d/%d] %s done" % (i + 1, len(days), day))
    log("sweep complete")
    aggregate()


if __name__ == "__main__":
    main()
