#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/118 Stage A — characterise EVERY recorded SENSEX day on the real 1-min chain.

Not just Wednesdays and Thursdays, and not just the losers: all five weekdays in the
recorded window (2026-04-20 .. 2026-08-20) so the "Wednesday is dangerous" claim has a
control group.

Per day we record, for the live construction (09:16 ATM straddle, exit 15:15):
  * front expiry + DTE (so the day is labelled by DTE, not just by calendar name)
  * entry credit, and the underlying level at 09:16
  * terminal move from the 09:16 level (signed and absolute)
  * maximum adverse excursion of the UNDERLYING and the minute it happened
  * maximum adverse excursion of the STRADDLE mark (rupees/lot) and its minute
  * whether the day trended or spiked-and-reverted
  * HOLD P&L net of costs, and the worst mark-to-market the position ever showed
    (so we can see which winners nearly became losers)

READ-ONLY on both databases. Re-runnable, self-contained.
"""
import csv
import os
import sqlite3
from datetime import date

Q = "/home/arun/quantifyd/"
CHAIN = "file:" + Q + "backtest_data/options_data.db?mode=ro"
MKT = "file:" + Q + "backtest_data/market_data.db?mode=ro"

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
OUT = os.path.join(RES, "stage_a_day_characterisation.csv")
LOG = os.path.join(RES, "stage_a.log")

LOT = 20            # SENSEX lot size
SLIP = 1.0          # points per leg-side
CHG = 30.0          # Rs per leg-side per lot
COST_NET = (2 * 2 * SLIP * LOT) + (2 * 2 * CHG)   # = 200 Rs/lot round trip, both legs
ENTRY_HM = "09:16"
EOD_HM = "15:15"
WD = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

FIELDS = [
    "day", "weekday", "front_expiry", "expiry_weekday", "dte", "is_expiry_day",
    "spot0916", "strike", "credit", "ce0", "pe0",
    "spot_eod", "move_term", "abs_move_term",
    "mae_spot_pts", "mae_spot_hm", "mae_spot_dir",
    "comb_max", "comb_max_hm", "comb_min", "comb_min_hm",
    "hold_gross", "hold_net", "worst_mtm_net", "worst_mtm_hm", "best_mtm_net",
    "move_over_credit", "mae_over_credit", "revert_ratio", "shape", "near_miss",
    "n_minutes", "spot_src",
]


def log(m):
    with open(LOG, "a") as f:
        f.write(m + "\n")
    print(m, flush=True)


def recorded_days(c):
    return [r[0] for r in c.execute(
        "SELECT DISTINCT substr(snapshot_time,1,10) d FROM download_log "
        "WHERE symbol='SENSEX' ORDER BY d")]


def spot_path(m, day):
    """1-min underlying OHLC for the day keyed by HH:MM (from market_data, index series)."""
    rows = m.execute(
        "SELECT date, open, high, low, close FROM market_data_unified "
        "WHERE symbol='SENSEX' AND timeframe='minute' AND date >= ? AND date < ? "
        "ORDER BY date", (day + " 00:00:00", day + " 23:59:59")).fetchall()
    return {r[0][11:16]: (r[1], r[2], r[3], r[4]) for r in rows}


def chain_spot_path(c, day):
    """Fallback when the index minute series has a gap: the recorder's own per-minute
    underlying_spot. No intrabar high/low, so excursions are point-in-time (slightly
    conservative). Flagged in the output via spot_src."""
    rows = c.execute(
        "SELECT snapshot_time, spot_price FROM underlying_spot "
        "WHERE symbol='SENSEX' AND snapshot_time LIKE ? AND spot_price IS NOT NULL "
        "ORDER BY snapshot_time", (day + "%",)).fetchall()
    d = {}
    for st, p in rows:
        d[st[11:16]] = (p, p, p, p)
    return d


def chain_day(c, day):
    """(front_expiry, strike, ce0, pe0, [(hm, ce, pe, comb)]) for the 09:16 ATM straddle."""
    rows = c.execute(
        "SELECT snapshot_time, expiry_date, strike, instrument_type, ltp, underlying_spot "
        "FROM option_chain WHERE symbol='SENSEX' AND snapshot_time LIKE ? AND ltp IS NOT NULL",
        (day + "%",)).fetchall()
    if not rows:
        return None
    exps = sorted({str(e)[:10] for (_, e, _, _, _, _) in rows if e and str(e)[:10] >= day})
    if not exps:
        return None
    fexp = exps[0]
    per, spot0 = {}, None
    for st, e, k, it, ltp, sp in rows:
        if str(e)[:10] != fexp:
            continue
        hm = st[11:16]
        per.setdefault(hm, {}).setdefault(k, {})[it] = ltp
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
    if len(ser) <= 30:
        return None
    return fexp, K, e0["CE"], e0["PE"], spot0, ser


def main():
    os.makedirs(RES, exist_ok=True)
    open(LOG, "w").close()
    c = sqlite3.connect(CHAIN, uri=True)
    m = sqlite3.connect(MKT, uri=True)
    days = recorded_days(c)
    log("Stage A: %d recorded SENSEX chain days, %s .. %s" % (len(days), days[0], days[-1]))

    out = []
    for day in days:
        d = chain_day(c, day)
        if not d:
            log("  %s %s  SKIP (no clean 09:16 ATM series)" % (day, WD[date.fromisoformat(day).weekday()]))
            continue
        fexp, K, ce0, pe0, spot0, ser = d
        credit = ce0 + pe0
        sp = spot_path(m, day)
        spot_src = "index_minute"
        if len(sp) < 200:
            sp = chain_spot_path(c, day)
            spot_src = "chain_spot"
        # underlying reference: the option-chain's own 09:16 spot (what the trade saw)
        ref = spot0
        # --- underlying excursions, intrabar, from 09:16 to 15:15 ---
        mae, mae_hm, mae_dir = 0.0, "", ""
        eod_close = None
        nmin = 0
        for hm in sorted(sp):
            if hm < ENTRY_HM or hm > EOD_HM:
                continue
            o, hi, lo, cl = sp[hm]
            nmin += 1
            eod_close = cl
            up, dn = hi - ref, ref - lo
            if up > mae:
                mae, mae_hm, mae_dir = up, hm, "UP"
            if dn > mae:
                mae, mae_hm, mae_dir = dn, hm, "DOWN"
        if eod_close is None:
            log("  %s SKIP (no minute price series)" % day)
            continue
        move_term = eod_close - ref
        # --- straddle mark path ---
        comb_max = max(s[3] for s in ser)
        comb_max_hm = [s[0] for s in ser if s[3] == comb_max][0]
        comb_min = min(s[3] for s in ser)
        comb_min_hm = [s[0] for s in ser if s[3] == comb_min][0]
        hold_gross = (credit - ser[-1][3]) * LOT
        hold_net = hold_gross - COST_NET
        worst_mtm_net = (credit - comb_max) * LOT - COST_NET
        best_mtm_net = (credit - comb_min) * LOT - COST_NET
        # --- shape ---
        revert = (abs(move_term) / mae) if mae > 0 else 0.0
        if mae < 0.5 * credit:
            shape = "CALM"
        elif revert >= 0.70:
            shape = "TREND"
        elif revert <= 0.40:
            shape = "SPIKE_REVERT"
        else:
            shape = "MIXED"
        ed = date.fromisoformat(fexp)
        out.append({
            "day": day, "weekday": WD[date.fromisoformat(day).weekday()],
            "front_expiry": fexp, "expiry_weekday": WD[ed.weekday()],
            "dte": (ed - date.fromisoformat(day)).days,
            "is_expiry_day": 1 if fexp == day else 0,
            "spot0916": round(ref, 2), "strike": K, "credit": round(credit, 2),
            "ce0": round(ce0, 2), "pe0": round(pe0, 2),
            "spot_eod": round(eod_close, 2), "move_term": round(move_term, 1),
            "abs_move_term": round(abs(move_term), 1),
            "mae_spot_pts": round(mae, 1), "mae_spot_hm": mae_hm, "mae_spot_dir": mae_dir,
            "comb_max": round(comb_max, 2), "comb_max_hm": comb_max_hm,
            "comb_min": round(comb_min, 2), "comb_min_hm": comb_min_hm,
            "hold_gross": round(hold_gross), "hold_net": round(hold_net),
            "worst_mtm_net": round(worst_mtm_net), "worst_mtm_hm": comb_max_hm,
            "best_mtm_net": round(best_mtm_net),
            "move_over_credit": round(abs(move_term) / credit, 3),
            "mae_over_credit": round(mae / credit, 3),
            "revert_ratio": round(revert, 3), "shape": shape,
            "near_miss": 1 if (hold_net > 0 and worst_mtm_net <= -1500) else 0,
            "n_minutes": nmin, "spot_src": spot_src,
        })
        log("  %s %s dte=%d credit=%6.1f move=%7.1f mae=%6.1f@%s hold_net=%7d worst_mtm=%8d %s"
            % (day, out[-1]["weekday"], out[-1]["dte"], credit, move_term, mae, mae_hm,
               hold_net, worst_mtm_net, shape))

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in out:
            w.writerow(r)
    log("wrote %s (%d days)" % (OUT, len(out)))


if __name__ == "__main__":
    main()
