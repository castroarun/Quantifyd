#!/usr/bin/env python3
"""
research/174 — P0b: does a NIFTY ATM option pair actually TRADE at each entry tenor?

Read-only against backtest_data/market_data.db (nse_options_bhav = real NSE bhavcopy).

For every (expiry, target entry-DTE) pair we ask, on the session a trader would have
stood on:
  - is there an ATM strike LISTED with both legs priced?          (listed)
  - is there an ATM strike with both legs actually TRADED that day? (tradeable)
  - how far from spot is the nearest tradeable pair?
  - what volume / OI does it carry?

Binding rule (research/89): a strike with a settlement print and zero volume is NOT
fillable. Everything here reports listed and tradeable separately so the gap is visible.

Output: results/liquidity_by_dte.csv, one row per (expiry, target DTE).
"""
import csv
import os
import sqlite3
import sys
from bisect import bisect_left
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
RESULTS.mkdir(exist_ok=True)
OUT = RESULTS / "liquidity_by_dte.csv"

TARGET_DTES = [45, 60, 75, 90, 120, 150, 180, 240, 270, 365, 545, 730]
SYMBOL = os.environ.get("R174_SYMBOL", "NIFTY")
UNDERLYING = {"NIFTY": "NIFTY50", "BANKNIFTY": "BANKNIFTY"}[SYMBOL]

FIELDS = ["symbol", "target_dte", "expiry", "entry_date", "actual_dte", "spot",
          "n_strikes", "listed_strike", "listed_dist_pct", "listed_prem",
          "trad_strike", "trad_dist_pct", "trad_prem",
          "trad_ce_contracts", "trad_pe_contracts", "trad_ce_oi", "trad_pe_oi",
          "n_traded_strikes", "expiry_total_contracts"]


def db_path():
    for p in [Path("/home/arun/quantifyd/backtest_data/market_data.db"),
              HERE.parents[2] / "backtest_data" / "market_data.db"]:
        if p.exists():
            return str(p)
    raise FileNotFoundError("market_data.db not found")


def dstr(d):
    return d.strftime("%Y-%m-%d")


def dparse(s):
    return datetime.strptime(s[:10], "%Y-%m-%d")


def prev_session(days, target):
    i = bisect_left(days, target)
    if i < len(days) and days[i] == target:
        return target
    return days[i - 1] if i > 0 else None


def main():
    con = sqlite3.connect("file:%s?mode=ro" % db_path(), uri=True)
    days = [r[0] for r in con.execute(
        "SELECT DISTINCT trade_date FROM nse_options_bhav WHERE symbol=? "
        "AND trade_date>='2011-01-01' ORDER BY trade_date", (SYMBOL,))]
    spot = {r[0][:10]: float(r[1]) for r in con.execute(
        "SELECT date, close FROM market_data_unified WHERE symbol=? AND timeframe='day' "
        "ORDER BY date", (UNDERLYING,)) if r[1]}
    expiries = [r[0] for r in con.execute(
        "SELECT DISTINCT expiry_date FROM nse_options_bhav WHERE symbol=? ORDER BY expiry_date",
        (SYMBOL,))]
    print("sessions=%d  expiries=%d  spot_days=%d" % (len(days), len(expiries), len(spot)),
          flush=True)

    done = set()
    if OUT.exists():
        with open(OUT) as f:
            done = {(r["symbol"], r["target_dte"], r["expiry"]) for r in csv.DictReader(f)}
        print("resuming, %d rows already present" % len(done), flush=True)
    new = not OUT.exists()
    fh = open(OUT, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new:
        w.writeheader()

    n = 0
    for exp in expiries:
        exp_dt = dparse(exp)
        for tgt in TARGET_DTES:
            key = (SYMBOL, str(tgt), exp)
            if key in done:
                continue
            ed = prev_session(days, dstr(exp_dt - timedelta(days=tgt)))
            if not ed or ed >= exp:
                continue
            s = spot.get(ed)
            if not s:
                continue
            rows = con.execute(
                "SELECT strike, option_type, close, contracts, open_interest "
                "FROM nse_options_bhav WHERE symbol=? AND expiry_date=? AND trade_date=?",
                (SYMBOL, exp, ed)).fetchall()
            if not rows:
                continue
            chain = {}
            tot_contracts = 0
            for k, ot, c, ct, oi in rows:
                if ot not in ("CE", "PE"):
                    continue
                tot_contracts += (ct or 0)
                chain.setdefault(float(k), {})[ot] = (c or 0.0, ct or 0, oi or 0)

            listed = trad = None
            ld = td = 1e18
            n_traded = 0
            for k, legs in chain.items():
                if "CE" not in legs or "PE" not in legs:
                    continue
                ce, pe = legs["CE"], legs["PE"]
                if ce[0] <= 0 or pe[0] <= 0:
                    continue
                d = abs(k - s)
                if d < ld:
                    listed, ld = k, d
                if ce[1] > 0 and pe[1] > 0:
                    n_traded += 1
                    if d < td:
                        trad, td = k, d

            row = dict(symbol=SYMBOL, target_dte=tgt, expiry=exp, entry_date=ed,
                       actual_dte=(exp_dt - dparse(ed)).days, spot=round(s, 2),
                       n_strikes=len(chain), n_traded_strikes=n_traded,
                       expiry_total_contracts=tot_contracts)
            if listed is not None:
                lg = chain[listed]
                row.update(listed_strike=listed, listed_dist_pct=round(100 * ld / s, 3),
                           listed_prem=round(lg["CE"][0] + lg["PE"][0], 2))
            if trad is not None:
                tg = chain[trad]
                row.update(trad_strike=trad, trad_dist_pct=round(100 * td / s, 3),
                           trad_prem=round(tg["CE"][0] + tg["PE"][0], 2),
                           trad_ce_contracts=tg["CE"][1], trad_pe_contracts=tg["PE"][1],
                           trad_ce_oi=tg["CE"][2], trad_pe_oi=tg["PE"][2])
            w.writerow(row)
            n += 1
            if n % 500 == 0:
                fh.flush()
                print("  %d rows written (last %s dte=%d)" % (n, exp, tgt), flush=True)
    fh.close()
    print("DONE, %d new rows -> %s" % (n, OUT), flush=True)


if __name__ == "__main__":
    main()
