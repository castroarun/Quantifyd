#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""research/118 Stage A2 — REAL SENSEX option prices, every trading day 2024-01 .. 2026-08.

`market_data.db :: bse_options_bhav` holds the BSE daily bhavcopy for SENSEX options
(2024-01-01 .. 2026-08-04, 289,859 rows). That is real traded option data for ~630 days,
where the 1-minute chain recorder only covers ~53. It lets us replay a held ATM straddle
for two and a half years instead of twelve Wednesdays.

Construction (deliberately the same shape as the live book):
  * ATM strike chosen from the 09:16 SENSEX level in market_data (CAUSAL — we never use
    the bhavcopy's own `underlying`, which is an end-of-day figure and would be look-ahead)
  * sell CE+PE at that strike at the day's OPEN price, buy back at the day's CLOSE
  * front weekly expiry only; both legs must have real traded volume and open interest
  * DTE and expiry weekday come from the data, so days are labelled by DTE, not by
    calendar name (the SENSEX weekly expiry day moved twice in this window)

Cross-era unit is POINTS (lot size changed from 10 to 20 inside the window); we also
report rupees at the current lot of 20 so the numbers line up with research/114.

DATA TRAP FOUND AND HANDLED — on rows where trade_date == expiry_date the BSE file
overwrites `close`, `settle` AND `underlying` with the index settlement level (e.g. every
option on 2026-07-30 shows close = 77928.15). Using that column on expiry days produces
nonsense of the order of -Rs3,000,000/lot. Open/high/low are fine. So:

  * DTE > 0  -> buy back at the bhavcopy CLOSE (a real traded price)
  * DTE == 0 -> buy back at INTRINSIC against the 15:15 index level from the 1-minute
                series, which is what an expiring straddle is worth. Marked in the output
                as terminal_src, and cross-validated against the 1-minute chain on the
                overlapping days in Stage C.

Open interest also legitimately collapses to 0 on expiry rows in the 2024 file, so the OI
filter is applied only to DTE > 0. A sanity band on credit and terminal (as a fraction of
spot) guards against any remaining bad prints.

READ-ONLY. Re-runnable.
"""
import csv
import os
import sqlite3
from datetime import date

Q = "/home/arun/quantifyd/"
MKT = "file:" + Q + "backtest_data/market_data.db?mode=ro"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
OUT = os.path.join(RES, "stage_a2_bhav_straddle_daily.csv")
ERAS = os.path.join(RES, "stage_a2_expiry_eras.csv")
LOG = os.path.join(RES, "stage_a2.log")

LOT_NOW = 20
SLIP = 1.0
CHG = 30.0
COST_RS = (2 * 2 * SLIP * LOT_NOW) + (2 * 2 * CHG)     # 200 Rs / lot, both legs round trip
MIN_VOL = 1000         # traded quantity on the leg that day (liquidity filter, binding rule)
MIN_OI = 1000          # applied to DTE>0 only: OI legitimately goes to 0 on expiry rows
CREDIT_BAND = (0.0005, 0.05)    # sane ATM straddle credit as a fraction of spot
WD = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

FIELDS = ["day", "weekday", "front_expiry", "expiry_weekday", "dte", "is_expiry_day",
          "era", "ref0916", "ref1515", "strike", "lot_size",
          "ce_open", "pe_open", "credit", "ce_close", "pe_close", "terminal",
          "terminal_src", "pnl_pts", "net_rs_lot20", "ce_vol", "pe_vol", "ce_oi", "pe_oi",
          "moneyness_pts"]


def log(m):
    with open(LOG, "a") as f:
        f.write(m + "\n")
    print(m, flush=True)


def real_expiries(m):
    """Expiries that actually expired inside our window: the last day they were quoted
    equals the expiry date itself. That is a data-derived definition, not an assumption."""
    rows = m.execute(
        "SELECT expiry_date, MAX(trade_date), COUNT(DISTINCT trade_date) "
        "FROM bse_options_bhav WHERE symbol='SENSEX' GROUP BY expiry_date "
        "ORDER BY expiry_date").fetchall()
    return sorted(e[:10] for e, last, nd in rows if str(last)[:10] == str(e)[:10] and nd >= 3)


def era_of(day, exp_wd_by_month):
    return exp_wd_by_month.get(day[:7], "?")


def main():
    os.makedirs(RES, exist_ok=True)
    open(LOG, "w").close()
    m = sqlite3.connect(MKT, uri=True)

    exps = real_expiries(m)
    log("data-derived real expiries: %d, %s .. %s" % (len(exps), exps[0], exps[-1]))

    # --- expiry-day era map, derived from the data (modal expiry weekday per month) ---
    by_month = {}
    for e in exps:
        by_month.setdefault(e[:7], []).append(WD[date.fromisoformat(e).weekday()])
    era_map = {mo: max(set(v), key=v.count) for mo, v in by_month.items()}
    with open(ERAS, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["month", "modal_expiry_weekday", "expiries", "weekdays_seen"])
        for mo in sorted(era_map):
            w.writerow([mo, era_map[mo], len(by_month[mo]), "|".join(by_month[mo])])
    log("expiry eras written -> %s" % ERAS)
    seq = [(mo, era_map[mo]) for mo in sorted(era_map)]
    cur, start = None, None
    for mo, wd in seq:
        if wd != cur:
            if cur:
                log("   era %s .. %s : weekly expiry on %s" % (start, mo, cur))
            cur, start = wd, mo
    log("   era %s .. %s : weekly expiry on %s" % (start, seq[-1][0], cur))

    # --- causal 09:16 SENSEX level, and the 15:15 level for expiry-day intrinsic ---
    ref, ref1515 = {}, {}
    rows = m.execute(
        "SELECT date, close FROM market_data_unified WHERE symbol='SENSEX' "
        "AND timeframe='minute' AND date >= '2023-12-01' ORDER BY date").fetchall()
    for dt, cl in rows:
        d, hm = dt[:10], dt[11:16]
        if hm >= "09:16" and d not in ref:
            ref[d] = cl
        if hm <= "15:15":
            ref1515[d] = cl
    log("09:16 reference levels: %d days" % len(ref))

    days = [r[0][:10] for r in m.execute(
        "SELECT DISTINCT trade_date FROM bse_options_bhav WHERE symbol='SENSEX' "
        "ORDER BY trade_date")]
    log("bhav trading days: %d, %s .. %s" % (len(days), days[0], days[-1]))

    out, skipped = [], {}
    for day in days:
        if day not in ref:
            skipped["no 09:16 level"] = skipped.get("no 09:16 level", 0) + 1
            continue
        fexp = next((e for e in exps if e >= day), None)
        if not fexp:
            skipped["no front expiry"] = skipped.get("no front expiry", 0) + 1
            continue
        K = round(ref[day] / 100.0) * 100
        rows = m.execute(
            "SELECT option_type, open, close, volume, open_interest, lot FROM bse_options_bhav "
            "WHERE symbol='SENSEX' AND trade_date=? AND expiry_date=? AND strike=?",
            (day, fexp, float(K))).fetchall()
        leg = {r[0]: r for r in rows}
        if "CE" not in leg or "PE" not in leg:
            skipped["no ATM pair"] = skipped.get("no ATM pair", 0) + 1
            continue
        ce, pe = leg["CE"], leg["PE"]
        is_exp = 1 if fexp == day else 0
        if None in (ce[1], pe[1]) or ce[1] <= 0 or pe[1] <= 0:
            skipped["no open price"] = skipped.get("no open price", 0) + 1
            continue
        if (ce[3] or 0) < MIN_VOL or (pe[3] or 0) < MIN_VOL:
            skipped["low volume"] = skipped.get("low volume", 0) + 1
            continue
        if not is_exp and ((ce[4] or 0) < MIN_OI or (pe[4] or 0) < MIN_OI):
            skipped["low OI"] = skipped.get("low OI", 0) + 1
            continue
        credit = ce[1] + pe[1]
        if not (CREDIT_BAND[0] * ref[day] <= credit <= CREDIT_BAND[1] * ref[day]):
            skipped["credit out of band"] = skipped.get("credit out of band", 0) + 1
            continue
        # --- terminal: bhav close for DTE>0; intrinsic vs the 15:15 index for DTE0,
        #     because the file overwrites close/settle with the index on expiry rows ---
        if is_exp:
            if day not in ref1515:
                skipped["no 15:15 level"] = skipped.get("no 15:15 level", 0) + 1
                continue
            terminal = abs(ref1515[day] - K)
            tsrc = "intrinsic_1515"
        else:
            if None in (ce[2], pe[2]) or ce[2] < 0 or pe[2] < 0:
                skipped["no close price"] = skipped.get("no close price", 0) + 1
                continue
            terminal = ce[2] + pe[2]
            tsrc = "bhav_close"
        if terminal > CREDIT_BAND[1] * ref[day]:
            skipped["terminal out of band"] = skipped.get("terminal out of band", 0) + 1
            continue
        pnl_pts = credit - terminal
        ed = date.fromisoformat(fexp)
        out.append({
            "day": day, "weekday": WD[date.fromisoformat(day).weekday()],
            "front_expiry": fexp, "expiry_weekday": WD[ed.weekday()],
            "dte": (ed - date.fromisoformat(day)).days,
            "is_expiry_day": is_exp,
            "era": era_map.get(day[:7], "?"),
            "ref0916": round(ref[day], 2),
            "ref1515": round(ref1515.get(day, 0) or 0, 2),
            "strike": K, "lot_size": ce[5],
            "ce_open": ce[1], "pe_open": pe[1], "credit": round(credit, 2),
            "ce_close": ("" if is_exp else ce[2]), "pe_close": ("" if is_exp else pe[2]),
            "terminal": round(terminal, 2), "terminal_src": tsrc,
            "pnl_pts": round(pnl_pts, 2),
            "net_rs_lot20": round(pnl_pts * LOT_NOW - COST_RS),
            "ce_vol": ce[3], "pe_vol": pe[3], "ce_oi": ce[4], "pe_oi": pe[4],
            "moneyness_pts": round(K - ref[day], 1),
        })
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in out:
            w.writerow(r)
    log("kept %d days, skipped %s" % (len(out), skipped))
    bydte = {}
    for r in out:
        bydte[r["dte"]] = bydte.get(r["dte"], 0) + 1
    log("kept by DTE: %s" % sorted(bydte.items()))
    bywd = {}
    for r in out:
        bywd[r["weekday"]] = bywd.get(r["weekday"], 0) + 1
    log("kept by weekday: %s" % sorted(bywd.items()))
    log("wrote %s" % OUT)


if __name__ == "__main__":
    main()
