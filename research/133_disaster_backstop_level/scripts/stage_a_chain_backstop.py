#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/133 Stage A - recorded 1-minute SENSEX chain: what would each backstop level
have done on the DTE0 13:00->15:20 stop-less TimeB straddle?

READ-ONLY on backtest_data/options_data.db. Writes results/stage_a_days.csv,
results/stage_a_levels.csv, results/r114_reconciliation.txt.

DTE0 is derived from the CHAIN (front expiry == the day itself), never from the weekday -
SENSEX expiry moved Fri -> Tue -> Thu inside our recorded history.

Cost model: the MEASURED outcome-aware model from research/122
(stage_a_alldays.py :: cost_per_lot) - entry slippage 0, time exit +0.178 pt/leg-side,
forced/stop exit +6.548 pt/leg-side, plus the exact Zerodha F&O option rate card.
"""
import sqlite3, csv, os, json
from datetime import date

Q = "/home/arun/quantifyd/"
CHAIN = Q + "backtest_data/options_data.db"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
LOG = os.path.join(RES, "stage_a.log")

LOT = 20          # SENSEX. option_chain.lot_size is WRONG - do not read it.
STEP = 100        # strike step
LOTS_LIVE = 6     # CSL_TIMEB_SENSEX DTE0 after the 2026-08-27 scale-down (qty 120)

ENTRY_M = 13 * 60 + 0
EXIT_M = 15 * 60 + 20

# --- MEASURED cost model (r/122, 2026-08-25) --------------------------------
SLIP_ENTRY = 0.0
SLIP_TIME = 0.178
SLIP_STOP = 6.548
NLOTS_REF = 10


def cost_per_lot(credit, exitp, lot, reason):
    sell = credit * lot
    buy = exitp * lot
    tot = sell + buy
    brok = 80.0 / NLOTS_REF
    stt = 0.001 * sell
    txn = 0.0003503 * tot
    ipft = 0.0000050 * tot
    sebi = 0.0000010 * tot
    stamp = 0.00003 * buy
    gst = 0.18 * (brok + txn + ipft + sebi)
    slip = 2 * SLIP_ENTRY + 2 * (SLIP_STOP if reason in ("SL", "BACKSTOP") else SLIP_TIME)
    return brok + stt + txn + ipft + sebi + stamp + gst + slip * lot


# credit-relative levels + fixed rupee-per-lot levels
FRAC_LEVELS = [0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.75, 1.00]
RS_LEVELS = [1000, 1500, 2000, 2500, 3000, 4000, 5000]   # Rs per lot of adverse premium


def log(m):
    with open(LOG, "a") as f:
        f.write(m + "\n")
    print(m, flush=True)


def hm2m(h):
    return int(h[:2]) * 60 + int(h[3:5])


def m2hm(m):
    return "%02d:%02d" % (m // 60, m % 60)


def all_days(c):
    return [r[0] for r in c.execute(
        "SELECT DISTINCT substr(snapshot_time,1,10) d FROM download_log "
        "WHERE symbol=? ORDER BY d", ("SENSEX",))
        if date.fromisoformat(r[0]).weekday() < 5]


def load_day(c, day):
    """-> (front_expiry, {min: spot}, {min: {strike: (ce,pe)}}) or None on a guard."""
    rows = c.execute(
        "SELECT snapshot_time, expiry_date, strike, instrument_type, ltp, underlying_spot "
        "FROM option_chain WHERE symbol=? AND snapshot_time>=? AND snapshot_time<? "
        "AND ltp IS NOT NULL", ("SENSEX", day, day + "z")).fetchall()
    if not rows:
        return None, "nodata"
    last_snap = max(r[0] for r in rows)
    if last_snap[11:16] < "15:15":
        return None, "partial(last=%s)" % last_snap[11:16]
    exps = sorted({e for (_, e, _, _, _, _) in rows if e and e >= day})
    if not exps:
        return None, "noexpiry"
    fexp = exps[0]
    spot, chain = {}, {}
    for st, e, k, it, ltp, sp in rows:
        mi = hm2m(st[11:16])
        if sp and mi not in spot:
            spot[mi] = sp
        if e != fexp:
            continue
        chain.setdefault(mi, {}).setdefault(k, {})[it] = ltp
    if len(set(spot.values())) < 50:
        return None, "frozen(%d distinct spots)" % len(set(spot.values()))
    ch2 = {}
    for mi, ks in chain.items():
        ch2[mi] = {k: (v["CE"], v["PE"]) for k, v in ks.items() if "CE" in v and "PE" in v}
    return (fexp, spot, ch2), "ok"


def series(chain, spot, K, m0, m1):
    """the combined-premium series for strike K over [m0, m1]. -> (credit, [(min, comb)], spot0)."""
    if m0 not in chain or K not in chain[m0]:
        return None
    ce0, pe0 = chain[m0][K]
    credit = ce0 + pe0
    if credit <= 0:
        return None
    s0 = spot.get(m0)
    path = []
    for mi in range(m0 + 1, m1 + 1):
        d = chain.get(mi)
        if not d or K not in d:
            continue
        ce, pe = d[K]
        path.append((mi, ce + pe, spot.get(mi)))
    if not path:
        return None
    return credit, path, s0


def apply_level(credit, path, thresh, dwell):
    """thresh = combined premium at/above which the backstop is armed.
    dwell=1 -> exit at the SAME minute the level is first touched.
    dwell=2 -> live model: level seen at m, confirmed, exit at the NEXT available print.
    -> (exit_min, exit_comb, reason, first_touch_min, first_touch_comb) or None if never."""
    for i, (mi, comb, _sp) in enumerate(path):
        if comb >= thresh:
            if dwell == 1:
                return mi, comb, "BACKSTOP", mi, comb
            if i + 1 < len(path):
                nm, nc, _ = path[i + 1]
                return nm, nc, "BACKSTOP", mi, comb
            return mi, comb, "BACKSTOP", mi, comb
    return None


def main():
    os.makedirs(RES, exist_ok=True)
    open(LOG, "w").close()
    c = sqlite3.connect("file:%s?mode=ro" % CHAIN, uri=True)
    days = all_days(c)
    log("SENSEX recorded weekdays: %d  (%s .. %s)" % (len(days), days[0], days[-1]))

    dayf = ["day", "weekday", "expiry", "strike", "spot0", "credit", "credit_rs_lot",
            "hold_exit_comb", "hold_gross_lot", "hold_net_lot", "hold_net_6L",
            "mae_pts", "mae_frac", "mae_hm", "mae_rs_lot",
            "und_exc_pts", "und_exc_bp", "und_exc_hm", "term_move_pts", "term_move_bp",
            "n_mins"]
    fday = open(os.path.join(RES, "stage_a_days.csv"), "w", newline="")
    wday = csv.DictWriter(fday, fieldnames=dayf)
    wday.writeheader()

    lvlf = ["day", "arm", "kind", "level", "thresh_comb", "dwell", "fired", "fire_hm",
            "touch_comb", "exit_comb", "overshoot_pts", "gross_lot", "net_lot", "net_6L",
            "hold_net_lot", "hold_net_6L", "delta_lot", "delta_6L", "recovered"]
    flvl = open(os.path.join(RES, "stage_a_levels.csv"), "w", newline="")
    wlvl = csv.DictWriter(flvl, fieldnames=lvlf)
    wlvl.writeheader()

    WD = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    r114 = []          # (day, credit, net_hold_916, net_leg30_916)
    kept = 0
    for day in days:
        d, why = load_day(c, day)
        if not d:
            log("  %s SKIP %s" % (day, why))
            continue
        fexp, spot, chain = d
        if fexp != day:
            continue                      # not DTE0 -> not this book's cell
        mins = sorted(m for m in chain if 9 * 60 + 15 <= m <= EXIT_M)
        if len(mins) < 200:
            log("  %s SKIP thin (%d mins)" % (day, len(mins)))
            continue
        # ---- the live cell: 13:00 -> 15:20 --------------------------------
        m0 = ENTRY_M if ENTRY_M in chain else min(
            (m for m in mins if ENTRY_M <= m <= ENTRY_M + 10), default=None)
        if m0 is None:
            log("  %s SKIP no 13:00 print" % day)
            continue
        sp0 = spot.get(m0)
        if not sp0:
            log("  %s SKIP no 13:00 spot" % day)
            continue
        K = round(sp0 / STEP) * STEP
        s = series(chain, spot, K, m0, EXIT_M)
        if not s:
            log("  %s SKIP no series at K=%d" % (day, K))
            continue
        kept += 1
        credit, path, _ = s
        mae, mae_m = 0.0, m0
        exc, exc_m = 0.0, m0
        term = 0.0
        for mi, comb, sp in path:
            if comb - credit > mae:
                mae, mae_m = comb - credit, mi
            if sp:
                e = abs(sp - sp0)
                if e > exc:
                    exc, exc_m = e, mi
                term = sp - sp0
        hold_m, hold_comb, _ = path[-1]
        hold_gross = (credit - hold_comb) * LOT
        hold_net = hold_gross - cost_per_lot(credit, hold_comb, LOT, "TIME")
        wday.writerow(dict(
            day=day, weekday=WD[date.fromisoformat(day).weekday()], expiry=fexp,
            strike=K, spot0=round(sp0, 2), credit=round(credit, 2),
            credit_rs_lot=round(credit * LOT),
            hold_exit_comb=round(hold_comb, 2), hold_gross_lot=round(hold_gross),
            hold_net_lot=round(hold_net), hold_net_6L=round(hold_net * LOTS_LIVE),
            mae_pts=round(mae, 2), mae_frac=round(mae / credit, 4), mae_hm=m2hm(mae_m),
            mae_rs_lot=round(mae * LOT),
            und_exc_pts=round(exc, 1), und_exc_bp=round(1e4 * exc / sp0, 1),
            und_exc_hm=m2hm(exc_m), term_move_pts=round(term, 1),
            term_move_bp=round(1e4 * term / sp0, 1), n_mins=len(path)))

        arms = ([("FRAC%02d" % round(100 * L), "frac", L, credit * (1.0 + L)) for L in FRAC_LEVELS] +
                [("RS%d" % r, "rs", r, credit + r / float(LOT)) for r in RS_LEVELS])
        for arm, kind, level, thresh in arms:
            for dwell in (1, 2):
                res = apply_level(credit, path, thresh, dwell)
                if res is None:
                    wlvl.writerow(dict(day=day, arm=arm, kind=kind, level=level,
                                       thresh_comb=round(thresh, 2), dwell=dwell, fired=0,
                                       fire_hm="", touch_comb="", exit_comb=round(hold_comb, 2),
                                       overshoot_pts="", gross_lot=round(hold_gross),
                                       net_lot=round(hold_net),
                                       net_6L=round(hold_net * LOTS_LIVE),
                                       hold_net_lot=round(hold_net),
                                       hold_net_6L=round(hold_net * LOTS_LIVE),
                                       delta_lot=0, delta_6L=0, recovered=""))
                    continue
                em, ec, reason, tm, tc = res
                gross = (credit - ec) * LOT
                net = gross - cost_per_lot(credit, ec, LOT, "BACKSTOP")
                wlvl.writerow(dict(
                    day=day, arm=arm, kind=kind, level=level, thresh_comb=round(thresh, 2),
                    dwell=dwell, fired=1, fire_hm=m2hm(em), touch_comb=round(tc, 2),
                    exit_comb=round(ec, 2), overshoot_pts=round(tc - thresh, 2),
                    gross_lot=round(gross), net_lot=round(net),
                    net_6L=round(net * LOTS_LIVE), hold_net_lot=round(hold_net),
                    hold_net_6L=round(hold_net * LOTS_LIVE),
                    delta_lot=round(net - hold_net), delta_6L=round((net - hold_net) * LOTS_LIVE),
                    recovered=1 if hold_net > net else 0))

        # ---- r/114 reconciliation arm: 09:16 -> 15:15, HOLD and per-leg 30% ----
        m916 = 9 * 60 + 16
        mm = m916 if m916 in chain else min((m for m in mins if m916 <= m <= m916 + 10),
                                            default=None)
        if mm is not None and spot.get(mm):
            sp916 = spot[mm]
            K9 = round(sp916 / STEP) * STEP
            if mm in chain and K9 in chain[mm]:
                ce0, pe0 = chain[mm][K9]
                cr9 = ce0 + pe0
                end9 = 15 * 60 + 15
                lce, lpe = ce0, pe0
                sce, spe = None, None      # per-leg 30% stop fills
                for mi in range(mm + 1, end9 + 1):
                    dd = chain.get(mi)
                    if not dd or K9 not in dd:
                        continue
                    ce, pe = dd[K9]
                    lce, lpe = ce, pe
                    if sce is None and ce >= ce0 * 1.30:
                        sce = ce
                    if spe is None and pe >= pe0 * 1.30:
                        spe = pe
                ex9 = lce + lpe
                g9 = (cr9 - ex9) * LOT
                n9 = g9 - cost_per_lot(cr9, ex9, LOT, "TIME")
                exl = (sce if sce is not None else lce) + (spe if spe is not None else lpe)
                gl = (cr9 - exl) * LOT
                nl = gl - cost_per_lot(cr9, exl, LOT,
                                       "SL" if (sce is not None or spe is not None) else "TIME")
                r114.append((day, cr9, n9, nl))

    fday.close()
    flvl.close()
    log("kept %d DTE0 sessions with a clean 13:00 series" % kept)

    with open(os.path.join(RES, "r114_reconciliation.txt"), "w") as f:
        if r114:
            hs = [x[2] for x in r114]
            ls = [x[3] for x in r114]
            f.write("r/114 reconciliation - SENSEX DTE0, 09:16 entry -> 15:15, MEASURED costs\n")
            f.write("n = %d sessions (%s .. %s)\n" % (len(r114), r114[0][0], r114[-1][0]))
            f.write("HOLD   mean %+.0f/lot  median %+.0f  win %.0f%%  worst %+.0f  best %+.0f\n"
                    % (sum(hs) / len(hs), sorted(hs)[len(hs) // 2],
                       100.0 * sum(1 for x in hs if x > 0) / len(hs), min(hs), max(hs)))
            f.write("LEG30  mean %+.0f/lot  median %+.0f  win %.0f%%  worst %+.0f\n"
                    % (sum(ls) / len(ls), sorted(ls)[len(ls) // 2],
                       100.0 * sum(1 for x in ls if x > 0) / len(ls), min(ls)))
            f.write("\n[r/114 published: HOLD +2,630/lot, 92%% win, n=12; LEG30 -227/lot, 25%% win]\n")
            f.write("[r/131 B0916 + measured costs: HOLD +2,831/lot]\n\nper-day:\n")
            for d_, cr_, h_, l_ in r114:
                f.write("  %s credit=%7.2f HOLD=%+8.0f LEG30=%+8.0f\n" % (d_, cr_, h_, l_))
        else:
            f.write("no 09:16 series available\n")
    log(open(os.path.join(RES, "r114_reconciliation.txt")).read())
    log("DONE stage A")


if __name__ == "__main__":
    main()
