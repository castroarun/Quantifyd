#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/122 Stage A - ALL-WEEKDAY window sweep on the REAL 1-minute option chain.

Extends research/120's Friday harness (stage_a_windows.py) to every recorded day,
both venues, labelling each day by TRADING-day DTE derived from the chain itself.

Per day x window x arm: sell the ATM straddle at the window start, cover at the end
or on a combined stop; record credit, net P&L per lot, MAE (premium points), and the
maximum underlying excursion in bp. The grid is the atlas grid pre-registered in the
STATUS-MD: starts 09:20 -> 14:30 in 30-min steps x durations 60/90/120/HOLD(15:20),
PLUS the five deployed TimeB windows as explicitly named cells.

Guards carried over from r/120 + r/121:
  * frozen-chain holiday guard: reject any day with < 50 distinct spot prints
    (known: 2026-05-01, 2026-05-28, 2026-06-26)
  * partial-session guard: reject any day whose last snapshot is before 15:15
READ-ONLY on the DB.  Writes results/stage_a_alldays.csv
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
CHG = 30.0            # Rs per leg-side per lot
LEG_SIDES = 4         # 2 legs x (sell + buy back)

# --- MEASURED cost model (2026-08-25) ----------------------------------------
# From 443 real live leg-sides (Kite fill vs chain LTP, same minute), live NAS 916 books:
#   entry (sell) -0.228 pt (favourable -> booked 0) | time-exit +0.178 | SL-exit +6.548
# Slippage lives in stop-outs. Charges are the exact Zerodha F&O option rate card; brokerage
# is Rs20 per ORDER (4 per straddle round trip), so it is divided across the study lot count.
SLIP_ENTRY = 0.0
SLIP_TIME  = 0.178
SLIP_STOP  = 6.548
NLOTS_REF  = 10

def cost_per_lot(credit, exitp, lot, reason):
    sell = credit * lot; buy = exitp * lot; tot = sell + buy
    brok = 80.0 / NLOTS_REF          # Rs20 x 4 orders, spread over the study lot count
    stt = 0.001 * sell
    txn = 0.0003503 * tot
    ipft = 0.0000050 * tot
    sebi = 0.0000010 * tot
    stamp = 0.00003 * buy
    gst = 0.18 * (brok + txn + ipft + sebi)
    slip = 2 * SLIP_ENTRY + 2 * (SLIP_STOP if reason == "SL" else SLIP_TIME)
    return brok + stt + txn + ipft + sebi + stamp + gst + slip * lot
SESS_END_M = 15 * 60 + 20

# atlas grid: starts 09:20..14:20 step 30 + 14:30; durations 60/90/120/HOLD
GRID_STARTS = []
_m = 9 * 60 + 20
while _m <= 14 * 60 + 20:
    GRID_STARTS.append(_m)
    _m += 30
GRID_STARTS.append(14 * 60 + 30)
DURATIONS = [60, 90, 120, "HOLD"]

# the five deployed windows, replayed on EVERY day/venue so each becomes an atlas
# row on its own venue-DTE and a comparison row everywhere else.
DEPLOYED_WINDOWS = [  # (label, start_hm, end_hm)
    ("DEP_0930_1100", "09:30", "11:00"),
    ("DEP_1000_1200", "10:00", "12:00"),
    ("DEP_1030_1200", "10:30", "12:00"),
    ("DEP_1300_1400", "13:00", "14:00"),
    ("DEP_1300_1520", "13:00", "15:20"),
]
ARMS = [("SL20", 0.20), ("SL25", 0.25), ("NOSTOP", None)]


def log(m):
    with open(LOG, "a") as f:
        f.write(m + "\n")
    print(m, flush=True)


def hm2m(h):
    return int(h[:2]) * 60 + int(h[3:5])


def m2hm(m):
    return "%02d:%02d" % (m // 60, m % 60)


def trading_dte(day, exp):
    """trading days strictly between day and expiry, weekends excluded (expiry day
    itself = DTE0). Holidays inside the gap are ignored -> label can overstate by
    one on holiday weeks; acceptable for labelling."""
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
    if last_snap[11:16] < "15:15":          # partial session
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
    if len(set(spot.values())) < 50:        # frozen-chain holiday guard (r/120)
        return None
    ch2 = {}
    for mi, ks in chain.items():
        ch2[mi] = {k: (v["CE"], v["PE"]) for k, v in ks.items() if "CE" in v and "PE" in v}
    return fexp, spot, ch2


def replay(chain, spot, K, m0, m1, sl_pct):
    if m0 not in chain or K not in chain[m0]:
        return None
    ce0, pe0 = chain[m0][K]
    credit = ce0 + pe0
    if credit <= 0:
        return None
    s0 = spot.get(m0)
    mae_full = 0.0
    und_exc = 0.0
    term_move = 0.0
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
        if stopped_at is None and sl_pct is not None and comb >= credit * (1.0 + sl_pct):
            stopped_at = (mi, comb)
        sp = spot.get(mi)
        if sp and s0:
            e = abs(sp - s0)
            if e > und_exc:
                und_exc = e
            term_move = sp - s0
    if last_m == m0:
        return None
    if stopped_at is not None:
        exit_m, exit_comb, reason = stopped_at[0], stopped_at[1], "SL"
    else:
        exit_m, exit_comb, reason = last_m, last_comb, "TIME"
    return dict(credit=credit, exit_m=exit_m, exit_comb=exit_comb, reason=reason,
                mae_full=mae_full, und_exc=und_exc, term_move=term_move, spot0=s0)


def main():
    os.makedirs(RES, exist_ok=True)
    open(LOG, "w").close()
    c = sqlite3.connect("file:%s?mode=ro" % CHAIN, uri=True)

    fg = ["venue", "day", "weekday", "expiry", "dte_trd", "dte_cal", "cell", "arm",
          "start", "end", "dur", "strike", "spot0", "credit", "exit_hm", "exit_comb",
          "reason", "gross", "net", "mae_full_rs", "mae_full_pct",
          "und_exc_pts", "und_exc_bp", "term_move_bp"]
    fout = open(os.path.join(RES, "stage_a_alldays.csv"), "w", newline="")
    w = csv.DictWriter(fout, fieldnames=fg)
    w.writeheader()

    WD = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    for sym, V in VENUE.items():
        lot, step, slip = V["lot"], V["step"], V["slip"]
        cost = cost_per_lot(100.0, 85.0, lot, "TIME")  # nominal, for the log only
        days = all_days(c, sym)
        log("%s: %d candidate days %s..%s (cost Rs%.0f/lot RT)" % (sym, len(days), days[0], days[-1], cost))
        kept = 0
        for day in days:
            d = load_day(c, sym, day)
            if not d:
                log("  %s %s SKIP (holiday/partial/no data)" % (sym, day))
                continue
            fexp, spot, chain = d
            dte_t = trading_dte(day, fexp)
            dte_c = (date.fromisoformat(fexp) - date.fromisoformat(day)).days
            wd = WD[date.fromisoformat(day).weekday()]
            mins = sorted(m for m in chain if 9 * 60 + 15 <= m <= SESS_END_M)
            if len(mins) < 200:
                log("  %s %s SKIP thin (%d mins)" % (sym, day, len(mins)))
                continue
            kept += 1
            # cells: grid + deployed windows
            cells = []
            for s in GRID_STARTS:
                for dur in DURATIONS:
                    m1 = SESS_END_M if dur == "HOLD" else min(s + dur, SESS_END_M)
                    if m1 <= s:
                        continue
                    cells.append(("G_%s_%s" % (m2hm(s), "H" if dur == "HOLD" else str(dur)),
                                  s, m1, dur))
            for lbl, sh, eh in DEPLOYED_WINDOWS:
                cells.append((lbl, hm2m(sh), hm2m(eh), hm2m(eh) - hm2m(sh)))
            nrows = 0
            for lbl, s, m1, dur in cells:
                m0 = s if s in chain else min((m for m in mins if s <= m <= s + 10), default=None)
                if m0 is None:
                    continue
                sp0 = spot.get(m0)
                if not sp0:
                    continue
                K = round(sp0 / step) * step
                for arm, slp in ARMS:
                    r = replay(chain, spot, K, m0, m1, slp)
                    if not r:
                        continue
                    gross = (r["credit"] - r["exit_comb"]) * lot
                    _c = cost_per_lot(r["credit"], r["exit_comb"], lot, r["reason"])
                    w.writerow(dict(
                        venue=sym, day=day, weekday=wd, expiry=fexp, dte_trd=dte_t,
                        dte_cal=dte_c, cell=lbl, arm=arm, start=m2hm(s), end=m2hm(m1),
                        dur=("HOLD" if dur == "HOLD" else dur),
                        strike=K, spot0=round(sp0, 2), credit=round(r["credit"], 2),
                        exit_hm=m2hm(r["exit_m"]), exit_comb=round(r["exit_comb"], 2),
                        reason=r["reason"], gross=round(gross), net=round(gross - _c),
                        mae_full_rs=round(r["mae_full"] * lot),
                        mae_full_pct=round(100.0 * r["mae_full"] / r["credit"], 2),
                        und_exc_pts=round(r["und_exc"], 1),
                        und_exc_bp=round(1e4 * r["und_exc"] / sp0, 1),
                        term_move_bp=round(1e4 * r["term_move"] / sp0, 1)))
                    nrows += 1
            log("  %s %s %s exp=%s dte_trd=%d rows=%d" % (sym, day, wd, fexp, dte_t, nrows))
            fout.flush()
        log("%s: kept %d days" % (sym, kept))
    fout.close()
    log("DONE")


if __name__ == "__main__":
    main()
