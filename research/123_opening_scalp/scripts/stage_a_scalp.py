#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/123 Stage A + Stage 0 - the opening scalp on the REAL 1-minute option chain.

Decomposition (STATUS-MD s2): "2x qty, book half at T" = the deployed 1x position
(unchanged) + an INDEPENDENT 1x ATM-straddle scalp of duration T entered at the
system's own entry minute. This script replays that scalp on every recorded day:

  * entries: NIFTY 09:16 / 09:30 / 10:00, SENSEX 09:16 / 10:30
    (the deployed morning entry minutes; afternoon cells excluded by scope)
  * horizons: T = +20/25/30/35/40/45/50/55/60/65 minutes from the ACTUAL fill minute
  * defence arms per scalp:
      NOSTOP            bare scalp
      CSL15/20/25/30    tight combined-SL (generic-straddle ask)
      PERLEG30          each leg stopped at leg_entry x 1.30 (ATM/ATM4 family)
      RUP2500           combined MTM loss >= Rs2500/lot (NIFTY ATM2, r/96)
      MOVE04            |spot - spot0| >= 0.4% -> exit both legs (SENSEX ATM2)
  * costs: the scalp pays its OWN full round trip: 4 leg-sides x slip x lot
    + 4 x Rs30  ->  Rs250/lot NIFTY, Rs200/lot SENSEX (r/120..122 convention)

STAGE 0 rows (cell=STAGE0): for every deployed morning book on its LIVE DTE days,
the GROSS open P&L per lot at fixed wall-clock minutes 09:36..10:21 (step 5), with
the system's own stop applied (booked value once stopped). No exit costs - this is
the mark Arun sees on the screen. Aggregation to book-level rupees in analyze.

RECON rows: the three deployed morning TimeB windows replayed full-length with the
deployed stop, to reconcile against r/122's atlas before anything is published.

Guards carried from r/120..122: frozen-chain holiday guard (<50 distinct spot
prints), partial-session guard (last snap < 15:15). READ-ONLY on the DB.
Writes results/stage_a_scalp.csv
"""
import sqlite3, csv, os
from datetime import date, timedelta

Q = "/home/arun/quantifyd/"
CHAIN = Q + "backtest_data/options_data.db"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
LOG = os.path.join(RES, "stage_a.log")

VENUE = {
    "NIFTY":  dict(lot=65, step=50,  slip=0.5),
    "SENSEX": dict(lot=20, step=100, slip=1.0),
}
CHG = 30.0
LEG_SIDES = 4
T_GRID = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
ENTRIES = {"NIFTY": [556, 570, 600], "SENSEX": [556, 630]}   # 09:16 09:30 10:00 | 09:16 10:30
ARMS = ["NOSTOP", "CSL15", "CSL20", "CSL25", "CSL30", "PERLEG30", "RUP2500", "MOVE04"]
STAGE0_MINUTES = list(range(576, 622, 5))   # wall clock 09:36 .. 10:21

# Deployed morning books -> (venue, entry_min, live DTE set, arm per DTE, lots per DTE)
# 2026 era: NIFTY exp Tue (Mon=1 Tue=0 Thu=3 Fri=2), SENSEX exp Thu (Wed=1 Thu=0).
BOOKS = [
    ("NAS_N_ATM",    "NIFTY", 556, {1: ("PERLEG30", 3), 0: ("PERLEG30", 3), 2: ("PERLEG30", 3)}),
    ("NAS_N_ATM2",   "NIFTY", 556, {1: ("RUP2500", 3), 0: ("RUP2500", 3), 2: ("RUP2500", 3)}),
    ("NAS_N_ATM4",   "NIFTY", 556, {1: ("PERLEG30", 3), 0: ("PERLEG30", 3), 2: ("PERLEG30", 3)}),
    ("NAS_N_COMB",   "NIFTY", 556, {0: ("CSL25", 3), 1: ("CSL30", 3), 2: ("CSL30", 3), 3: ("CSL20", 5)}),
    ("TIMEB_N_TUE",  "NIFTY", 570, {0: ("CSL25", 8)}),
    ("TIMEB_N_FRI",  "NIFTY", 600, {2: ("CSL20", 8)}),
    ("NAS_S_ATM",    "SENSEX", 556, {1: ("PERLEG30", 3), 0: ("NOSTOP", 3)}),   # leg-SL off DTE0 (r/114)
    ("NAS_S_ATM4",   "SENSEX", 556, {1: ("PERLEG30", 3), 0: ("NOSTOP", 3)}),
    ("NAS_S_ATM2",   "SENSEX", 556, {1: ("MOVE04", 3), 0: ("MOVE04", 3)}),
    ("CSL30F_S_WED", "SENSEX", 556, {1: ("CSL30", 3)}),
    ("TIMEB_S_WED",  "SENSEX", 630, {1: ("CSL20", 8)}),
]
RECON = [  # label, venue, entry_min, exit_min, arm  (reconcile vs r/122 atlas)
    ("RECON_TUE_N_0930_1100_SL25", "NIFTY", 570, 660, "CSL25", 0),
    ("RECON_FRI_N_1000_1200_SL20", "NIFTY", 600, 720, "CSL20", 2),
    ("RECON_WED_S_1030_1200_SL20", "SENSEX", 630, 720, "CSL20", 1),
]


def log(m):
    with open(LOG, "a") as f:
        f.write(m + "\n")
    print(m, flush=True)


def m2hm(m):
    return "%02d:%02d" % (m // 60, m % 60)


def trading_dte(day, exp):
    d0, d1 = date.fromisoformat(day), date.fromisoformat(exp)
    n, d = 0, d0
    while d < d1:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n += 1
    return n


def all_days(c, sym):
    return [r[0] for r in c.execute(
        "SELECT DISTINCT substr(snapshot_time,1,10) d FROM download_log WHERE symbol=? ORDER BY d",
        (sym,)) if date.fromisoformat(r[0]).weekday() < 5]


def load_day(c, sym, day):
    rows = c.execute(
        "SELECT snapshot_time, expiry_date, strike, instrument_type, ltp, underlying_spot "
        "FROM option_chain WHERE symbol=? AND snapshot_time>=? AND snapshot_time<? "
        "AND ltp IS NOT NULL", (sym, day, day + "z")).fetchall()
    if not rows:
        return None
    last_snap = max(r[0] for r in rows)
    if last_snap[11:16] < "15:15":
        return None
    exps = sorted({e for (_, e, _, _, _, _) in rows if e and e >= day})
    if not exps:
        return None
    fexp = exps[0]
    spot, chain = {}, {}
    for st, e, k, it, ltp, sp in rows:
        mi = int(st[11:13]) * 60 + int(st[14:16])
        if sp and mi not in spot:
            spot[mi] = sp
        if e != fexp:
            continue
        d = chain.setdefault(mi, {}).setdefault(k, {})
        d[it] = ltp
    if len(set(spot.values())) < 50:
        return None
    ch2 = {}
    for mi, ks in chain.items():
        ch2[mi] = {k: (v["CE"], v["PE"]) for k, v in ks.items() if "CE" in v and "PE" in v}
    return fexp, spot, ch2


def walk(chain, spot, K, m0, m_end, lot):
    """One pass m0+1..m_end. Returns the minute-path and, per arm, the stop event.
    path: list of (mi, ce, pe, comb, sp). stops: arm -> (stop_minute, booked_comb)."""
    if m0 not in chain or K not in chain[m0]:
        return None
    ce0, pe0 = chain[m0][K]
    credit = ce0 + pe0
    if credit <= 0:
        return None
    s0 = spot.get(m0)
    if not s0:
        return None
    path = []
    stops = {}
    # PERLEG30 leg state
    leg_stop = {"CE": ce0 * 1.30, "PE": pe0 * 1.30}
    leg_booked = {}          # leg -> booked ltp
    rup_pts = 2500.0 / lot   # RUP2500 combined-rise threshold in points
    mae = 0.0
    exc = 0.0
    for mi in range(m0 + 1, m_end + 1):
        d = chain.get(mi)
        if not d or K not in d:
            continue
        ce, pe = d[K]
        comb = ce + pe
        sp = spot.get(mi)
        adv = comb - credit
        if adv > mae:
            mae = adv
        if sp:
            e = abs(sp - s0)
            if e > exc:
                exc = e
        # combined-SL family + rupee stop + move stop (single event, book whole comb)
        for arm, thr in (("CSL15", 0.15), ("CSL20", 0.20), ("CSL25", 0.25), ("CSL30", 0.30)):
            if arm not in stops and comb >= credit * (1.0 + thr):
                stops[arm] = (mi, comb)
        if "RUP2500" not in stops and comb >= credit + rup_pts:
            stops["RUP2500"] = (mi, comb)
        if "MOVE04" not in stops and sp and abs(sp - s0) >= 0.004 * s0:
            stops["MOVE04"] = (mi, comb)
        # per-leg
        for leg, ltp in (("CE", ce), ("PE", pe)):
            if leg not in leg_booked and ltp >= leg_stop[leg]:
                leg_booked[leg] = (mi, ltp)
        path.append((mi, ce, pe, comb, sp))
    if not path:
        return None
    return dict(credit=credit, ce0=ce0, pe0=pe0, s0=s0, path=path, stops=stops,
                leg_booked=leg_booked, mae=mae, exc=exc)


def value_at(w, arm, mT):
    """Exit combined value + reason + exit minute for `arm` with horizon end mT
    (book stop value if the stop fired at/before mT, else last mark <= mT).
    Also returns exc/term within the horizon."""
    marks = [(mi, ce, pe, comb, sp) for (mi, ce, pe, comb, sp) in w["path"] if mi <= mT]
    if not marks:
        return None
    last = marks[-1]
    exc = 0.0
    term = 0.0
    for mi, ce, pe, comb, sp in marks:
        if sp:
            e = abs(sp - w["s0"])
            if e > exc:
                exc = e
            term = sp - w["s0"]
    mae = max((c - w["credit"] for (_, _, _, c, _) in marks), default=0.0)
    if arm == "PERLEG30":
        out_ce, out_pe = last[1], last[2]
        reason, exit_m = "TIME", last[0]
        for leg, idx in (("CE", 1), ("PE", 2)):
            if leg in w["leg_booked"] and w["leg_booked"][leg][0] <= mT:
                if leg == "CE":
                    out_ce = w["leg_booked"][leg][1]
                else:
                    out_pe = w["leg_booked"][leg][1]
                reason, exit_m = "LEGSL", max(exit_m if reason == "LEGSL" else 0,
                                             w["leg_booked"][leg][0])
        if reason == "TIME":
            exit_m = last[0]
        return dict(exit_comb=out_ce + out_pe, reason=reason, exit_m=exit_m,
                    exc=exc, term=term, mae=mae)
    ev = w["stops"].get(arm) if arm != "NOSTOP" else None
    if ev and ev[0] <= mT:
        return dict(exit_comb=ev[1], reason="SL", exit_m=ev[0], exc=exc, term=term, mae=mae)
    return dict(exit_comb=last[3], reason="TIME", exit_m=last[0], exc=exc, term=term, mae=mae)


def main():
    os.makedirs(RES, exist_ok=True)
    open(LOG, "w").close()
    c = sqlite3.connect("file:%s?mode=ro" % CHAIN, uri=True)
    fg = ["venue", "day", "weekday", "expiry", "dte_trd", "cell", "book", "entry_target",
          "entry_hm", "T", "arm", "strike", "spot0", "credit", "exit_hm", "exit_comb",
          "reason", "gross", "net", "mae_pts", "und_exc_bp", "term_bp"]
    fout = open(os.path.join(RES, "stage_a_scalp.csv"), "w", newline="")
    w = csv.DictWriter(fout, fieldnames=fg)
    w.writeheader()
    WD = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    for sym, V in VENUE.items():
        lot, step, slip = V["lot"], V["step"], V["slip"]
        cost = LEG_SIDES * slip * lot + LEG_SIDES * CHG
        days = all_days(c, sym)
        log("%s: %d candidate days (cost Rs%.0f/lot RT)" % (sym, len(days), cost))
        kept = 0
        for day in days:
            d = load_day(c, sym, day)
            if not d:
                log("  %s %s SKIP" % (sym, day))
                continue
            fexp, spot, chain = d
            dte = trading_dte(day, fexp)
            wd = WD[date.fromisoformat(day).weekday()]
            mins = sorted(chain.keys())
            kept += 1
            nrows = 0
            walks = {}   # (entry_min) -> walk to session-relevant end
            for s in ENTRIES[sym]:
                m0 = next((m for m in mins if s <= m <= s + 10), None)
                if m0 is None:
                    continue
                sp0 = spot.get(m0)
                if not sp0:
                    continue
                K = round(sp0 / step) * step
                wk = walk(chain, spot, K, m0, min(m0 + 105, 15 * 60 + 20), lot)
                if not wk:
                    continue
                walks[s] = (m0, K, wk)
                # ---- scalp grid rows
                for T in T_GRID:
                    for arm in ARMS:
                        r = value_at(wk, arm, m0 + T)
                        if not r:
                            continue
                        gross = (wk["credit"] - r["exit_comb"]) * lot
                        w.writerow(dict(
                            venue=sym, day=day, weekday=wd, expiry=fexp, dte_trd=dte,
                            cell="SCALP", book="", entry_target=m2hm(s), entry_hm=m2hm(m0),
                            T=T, arm=arm, strike=K, spot0=round(sp0, 2),
                            credit=round(wk["credit"], 2), exit_hm=m2hm(r["exit_m"]),
                            exit_comb=round(r["exit_comb"], 2), reason=r["reason"],
                            gross=round(gross), net=round(gross - cost),
                            mae_pts=round(r["mae"], 2),
                            und_exc_bp=round(1e4 * r["exc"] / sp0, 1),
                            term_bp=round(1e4 * r["term"] / sp0, 1)))
                        nrows += 1
            # ---- Stage 0 rows: deployed books on their live DTE, marks at wall clock
            for book, bsym, s, dmap in BOOKS:
                if bsym != sym or dte not in dmap or s not in walks:
                    continue
                arm, lots = dmap[dte]
                m0, K, wk = walks[s]
                for mc in STAGE0_MINUTES:
                    if mc <= m0:
                        continue
                    r = value_at(wk, arm, mc)
                    if not r:
                        continue
                    gross = (wk["credit"] - r["exit_comb"]) * lot   # per lot, no costs
                    w.writerow(dict(
                        venue=sym, day=day, weekday=wd, expiry=fexp, dte_trd=dte,
                        cell="STAGE0", book="%s|lots%d" % (book, lots),
                        entry_target=m2hm(s), entry_hm=m2hm(m0), T=mc - m0, arm=arm,
                        strike=K, spot0=round(wk["s0"], 2), credit=round(wk["credit"], 2),
                        exit_hm=m2hm(mc), exit_comb=round(r["exit_comb"], 2),
                        reason=r["reason"], gross=round(gross), net=round(gross),
                        mae_pts=round(r["mae"], 2),
                        und_exc_bp=round(1e4 * r["exc"] / wk["s0"], 1),
                        term_bp=round(1e4 * r["term"] / wk["s0"], 1)))
                    nrows += 1
            # ---- reconciliation rows (full deployed windows)
            for lbl, rsym, s, m_end, arm, rdte in RECON:
                if rsym != sym or dte != rdte:
                    continue
                m0 = next((m for m in mins if s <= m <= s + 10), None)
                if m0 is None:
                    continue
                sp0 = spot.get(m0)
                if not sp0:
                    continue
                K = round(sp0 / step) * step
                wk = walk(chain, spot, K, m0, m_end, lot)
                if not wk:
                    continue
                r = value_at(wk, arm, m_end)
                if not r:
                    continue
                gross = (wk["credit"] - r["exit_comb"]) * lot
                w.writerow(dict(
                    venue=sym, day=day, weekday=wd, expiry=fexp, dte_trd=dte,
                    cell=lbl, book="", entry_target=m2hm(s), entry_hm=m2hm(m0),
                    T=m_end - m0, arm=arm, strike=K, spot0=round(sp0, 2),
                    credit=round(wk["credit"], 2), exit_hm=m2hm(r["exit_m"]),
                    exit_comb=round(r["exit_comb"], 2), reason=r["reason"],
                    gross=round(gross), net=round(gross - cost),
                    mae_pts=round(r["mae"], 2),
                    und_exc_bp=round(1e4 * r["exc"] / sp0, 1),
                    term_bp=round(1e4 * r["term"] / sp0, 1)))
                nrows += 1
            log("  %s %s %s dte=%d rows=%d" % (sym, day, wd, dte, nrows))
            fout.flush()
        log("%s: kept %d days" % (sym, kept))
    fout.close()
    log("DONE")


if __name__ == "__main__":
    main()
