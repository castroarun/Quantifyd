#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/120 Stage A - Friday intraday window sweep on the REAL 1-minute option chain.

For every recorded Friday and every venue, sell the ATM straddle at a window start minute
and buy it back at the window end (or on a combined stop), and record net P&L per lot plus
the maximum adverse excursion inside the window.

Two things are computed for every day:
  1. the PRE-REGISTERED GRID  : starts 09:20..14:30 in 15-min steps x durations 45/60/90/120/HOLD
  2. the ALL-START BASELINE   : the same durations from EVERY minute in the session

(2) is the control that (1) must beat. Short straddles decay all day, so ANY window shows a
positive mean; the only interesting question is whether a chosen window beats the average
window of the same length. Without this control the surface is unreadable (research/115).

READ-ONLY on the DB. Writes results/stage_a_trades.csv (grid, per day) and
results/stage_a_allstarts.csv (per day x duration x start minute).
"""
import sqlite3, csv, os, sys
from datetime import date

Q = "/home/arun/quantifyd/"
CHAIN = Q + "backtest_data/options_data.db"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
LOG = os.path.join(RES, "stage_a.log")

VENUE = {
    "NIFTY":  dict(lot=65, step=50,  slip=0.5),
    "SENSEX": dict(lot=20, step=100, slip=1.0),
}
CHG = 30.0            # Rs per leg-side per lot
LEG_SIDES = 4         # 2 legs x (sell + buy back)
SESS_START = "09:20"
SESS_END = "15:20"
EXCLUDE_DAYS = {"2026-08-21"}   # today - market still open, partial session

GRID_STARTS = []
_m = 9 * 60 + 20
while _m <= 14 * 60 + 30:
    GRID_STARTS.append("%02d:%02d" % (_m // 60, _m % 60))
    _m += 15
if "14:30" not in GRID_STARTS:
    GRID_STARTS.append("14:30")
DURATIONS = [45, 60, 90, 120, "HOLD"]
ARMS = [("SL20", 0.20), ("NOSTOP", None)]


def log(m):
    with open(LOG, "a") as f:
        f.write(m + "\n")
    print(m, flush=True)


def hm2m(h):
    return int(h[:2]) * 60 + int(h[3:5])


def m2hm(m):
    return "%02d:%02d" % (m // 60, m % 60)


def fridays(c, sym):
    days = [r[0] for r in c.execute(
        "SELECT DISTINCT substr(snapshot_time,1,10) d FROM download_log WHERE symbol=? ORDER BY d",
        (sym,))]
    return [d for d in days if date.fromisoformat(d).weekday() == 4 and d not in EXCLUDE_DAYS]


def load_day(c, sym, day):
    """-> (front_expiry, spot{minute:px}, chain{minute:{strike:(ce,pe)}})"""
    rows = c.execute(
        "SELECT snapshot_time, expiry_date, strike, instrument_type, ltp, underlying_spot "
        "FROM option_chain WHERE symbol=? AND snapshot_time>=? AND snapshot_time<? "
        "AND ltp IS NOT NULL", (sym, day, day + "z")).fetchall()
    if not rows:
        return None
    exps = sorted({e for (_, e, _, _, _, _) in rows if e and e >= day})
    if not exps:
        return None
    fexp = exps[0]
    spot, chain = {}, {}
    for st, e, k, it, ltp, sp in rows:
        mi = hm2m(st[11:16])
        if sp and mi not in spot:
            spot[mi] = sp
        if e != fexp:
            continue
        d = chain.setdefault(mi, {}).setdefault(k, {})
        d[it] = ltp
    # HOLIDAY GUARD (found 2026-08-21): the recorder polls on exchange holidays too and
    # captures a FROZEN chain - every minute identical, so every window books exactly
    # -cost and looks like a real losing day. 2026-05-01 and 2026-06-26 are both such
    # days (1 distinct spot all session; market_data_unified has 0 rows for them).
    if len(set(spot.values())) < 50:
        return None
    ch2 = {}
    for mi, ks in chain.items():
        ch2[mi] = {k: (v["CE"], v["PE"]) for k, v in ks.items() if "CE" in v and "PE" in v}
    return fexp, spot, ch2


def replay(chain, spot, K, m0, m1, sl_pct):
    """Sell ATM K straddle at m0, cover at m1 (or on combined stop).
    Returns dict or None. MAE fields are in POINTS of combined premium."""
    if m0 not in chain or K not in chain[m0]:
        return None
    ce0, pe0 = chain[m0][K]
    credit = ce0 + pe0
    if credit <= 0:
        return None
    s0 = spot.get(m0)
    mae_full = 0.0          # worst combined-premium excursion over the WHOLE window (no stop)
    mae_to_exit = 0.0       # worst excursion up to the actual exit
    und_exc = 0.0           # worst absolute underlying excursion from entry spot
    stopped_at = None
    last_m, last_comb = m0, credit
    for mi in range(m0 + 1, m1 + 1):
        d = chain.get(mi)
        if not d or K not in d:
            continue
        ce, pe = d[K]
        comb = ce + pe
        last_m, last_comb = mi, comb
        adv = comb - credit
        if adv > mae_full:
            mae_full = adv
        if stopped_at is None:
            if adv > mae_to_exit:
                mae_to_exit = adv
            if sl_pct is not None and comb >= credit * (1.0 + sl_pct):
                stopped_at = (mi, comb)
        sp = spot.get(mi)
        if sp and s0:
            e = abs(sp - s0)
            if e > und_exc:
                und_exc = e
    if last_m == m0:
        return None
    if stopped_at is not None:
        exit_m, exit_comb, reason = stopped_at[0], stopped_at[1], "SL"
    else:
        exit_m, exit_comb, reason = last_m, last_comb, "TIME"
    return dict(credit=credit, exit_m=exit_m, exit_comb=exit_comb, reason=reason,
                mae_full=mae_full, mae_to_exit=mae_to_exit, und_exc=und_exc, spot0=s0)


def main():
    os.makedirs(RES, exist_ok=True)
    open(LOG, "w").close()
    c = sqlite3.connect("file:%s?mode=ro" % CHAIN, uri=True)
    m_start, m_end = hm2m(SESS_START), hm2m(SESS_END)

    fg = ["venue", "day", "expiry", "dte_cal", "arm", "start", "dur", "end", "clamped",
          "strike", "spot0", "credit", "exit_hm", "exit_comb", "reason",
          "gross", "net", "mae_full_rs", "mae_exit_rs", "und_exc_pts", "und_exc_bp"]
    ag = ["venue", "day", "arm", "dur", "start", "net", "mae_full_rs", "und_exc_bp"]
    fgrid = open(os.path.join(RES, "stage_a_trades.csv"), "w", newline="")
    wgrid = csv.DictWriter(fgrid, fieldnames=fg); wgrid.writeheader()
    fall = open(os.path.join(RES, "stage_a_allstarts.csv"), "w", newline="")
    wall = csv.DictWriter(fall, fieldnames=ag); wall.writeheader()

    for sym, V in VENUE.items():
        lot, step, slip = V["lot"], V["step"], V["slip"]
        cost = LEG_SIDES * slip * lot + LEG_SIDES * CHG
        days = fridays(c, sym)
        log("%s: %d complete Fridays %s .. %s  (cost Rs%.0f/lot round trip)"
            % (sym, len(days), days[0], days[-1], cost))
        for day in days:
            d = load_day(c, sym, day)
            if not d:
                log("  %s %s SKIP no data" % (sym, day)); continue
            fexp, spot, chain = d
            dte_cal = (date.fromisoformat(fexp) - date.fromisoformat(day)).days
            mins = sorted(m for m in chain if m_start <= m <= m_end)
            if len(mins) < 200:
                log("  %s %s SKIP thin (%d mins)" % (sym, day, len(mins))); continue
            ngrid = 0
            for arm, slp in ARMS:
                # --- pre-registered grid ---
                for st in GRID_STARTS:
                    m0 = hm2m(st)
                    if m0 not in chain:
                        # nearest available minute at/after the nominal start (snapshot gaps).
                        # The CELL KEEPS ITS NOMINAL LABEL so the surface has no holes.
                        m0 = min((m for m in mins if m >= hm2m(st) and m <= hm2m(st) + 10),
                                 default=None)
                        if m0 is None:
                            continue
                    sp0 = spot.get(m0)
                    if not sp0:
                        continue
                    K = round(sp0 / step) * step
                    for dur in DURATIONS:
                        m1 = m_end if dur == "HOLD" else min(m0 + dur, m_end)
                        clamped = 1 if (dur != "HOLD" and m0 + dur > m_end) else 0
                        if m1 <= m0:
                            continue
                        r = replay(chain, spot, K, m0, m1, slp)
                        if not r:
                            continue
                        gross = (r["credit"] - r["exit_comb"]) * lot
                        wgrid.writerow(dict(
                            venue=sym, day=day, expiry=fexp, dte_cal=dte_cal, arm=arm,
                            start=st, dur=dur, end=m2hm(m1), clamped=clamped,
                            strike=K, spot0=round(sp0, 2), credit=round(r["credit"], 2),
                            exit_hm=m2hm(r["exit_m"]), exit_comb=round(r["exit_comb"], 2),
                            reason=r["reason"], gross=round(gross), net=round(gross - cost),
                            mae_full_rs=round(r["mae_full"] * lot),
                            mae_exit_rs=round(r["mae_to_exit"] * lot),
                            und_exc_pts=round(r["und_exc"], 1),
                            und_exc_bp=round(1e4 * r["und_exc"] / sp0, 1)))
                        ngrid += 1
                # --- all-start baseline (every minute) ---
                for dur in DURATIONS:
                    for m0 in mins:
                        if dur == "HOLD":
                            m1 = m_end
                            if m1 - m0 < 30:
                                continue
                        else:
                            m1 = m0 + dur
                            if m1 > m_end:
                                continue
                        sp0 = spot.get(m0)
                        if not sp0:
                            continue
                        K = round(sp0 / step) * step
                        r = replay(chain, spot, K, m0, m1, slp)
                        if not r:
                            continue
                        gross = (r["credit"] - r["exit_comb"]) * lot
                        wall.writerow(dict(venue=sym, day=day, arm=arm, dur=dur,
                                           start=m2hm(m0), net=round(gross - cost),
                                           mae_full_rs=round(r["mae_full"] * lot),
                                           und_exc_bp=round(1e4 * r["und_exc"] / sp0, 1)))
            log("  %s %s exp=%s dte_cal=%d mins=%d gridrows=%d" % (sym, day, fexp, dte_cal, len(mins), ngrid))
            fgrid.flush(); fall.flush()
    fgrid.close(); fall.close()
    log("DONE")


if __name__ == "__main__":
    main()
