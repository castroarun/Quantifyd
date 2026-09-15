#!/usr/bin/env python3
"""
research/174 — P0b v2: per SESSION and per target tenor, what is the best contract a
trader could actually sell, and does its ATM pair trade?

v1 conflated dead weeklies with real monthlies: at 45 DTE in 2016-2024 it reported 41%
tradeable because NSE lists weeklies 5-9 weeks out that never print a single contract.
A trader does not sell those. v2 asks the right question:

  standing on session D and wanting ~T days of tenor, look at every expiry listed that
  day whose DTE falls in the acceptance window around T, and take the one whose ATM pair
  is most liquid. Report whether ANY of them is fillable, and how good the best one is.

Acceptance window: DTE in [T*0.70, T*1.45] for T<=90, [T*0.75, T*1.30] for T>90.

Output: results/liquidity_v2.csv, one row per (session, target tenor).
"""
import csv
import os
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
RESULTS.mkdir(exist_ok=True)
OUT = RESULTS / "liquidity_v2.csv"

TARGETS = [45, 60, 90, 120, 180, 270, 365, 545, 730]
SYMBOL = os.environ.get("R174_SYMBOL", "NIFTY")
UNDERLYING = {"NIFTY": "NIFTY50", "BANKNIFTY": "BANKNIFTY"}[SYMBOL]
STEP = int(os.environ.get("R174_STEP", "1"))       # sample every Nth session

FIELDS = ["symbol", "session", "target_dte", "n_candidate_expiries", "best_expiry",
          "best_dte", "spot", "atm_strike", "atm_dist_pct", "atm_prem",
          "ce_contracts", "pe_contracts", "ce_oi", "pe_oi",
          "n_traded_strikes_in_expiry", "expiry_contracts", "fillable"]


def db_path():
    for p in [Path("/home/arun/quantifyd/backtest_data/market_data.db"),
              HERE.parents[2] / "backtest_data" / "market_data.db"]:
        if p.exists():
            return str(p)
    raise FileNotFoundError("market_data.db not found")


def dparse(s):
    return datetime.strptime(s[:10], "%Y-%m-%d")


def window(t):
    return (t * 0.70, t * 1.45) if t <= 90 else (t * 0.75, t * 1.30)


def main():
    con = sqlite3.connect("file:%s?mode=ro" % db_path(), uri=True)
    spot = {r[0][:10]: float(r[1]) for r in con.execute(
        "SELECT date, close FROM market_data_unified WHERE symbol=? AND timeframe='day'",
        (UNDERLYING,)) if r[1]}
    sessions = [r[0] for r in con.execute(
        "SELECT DISTINCT trade_date FROM nse_options_bhav WHERE symbol=? "
        "AND trade_date>='2015-01-01' ORDER BY trade_date", (SYMBOL,))]
    sessions = sessions[::STEP]
    print("sessions=%d step=%d" % (len(sessions), STEP), flush=True)

    done = set()
    if OUT.exists():
        with open(OUT) as f:
            done = {(r["symbol"], r["session"], r["target_dte"]) for r in csv.DictReader(f)}
        print("resuming, %d rows present" % len(done), flush=True)
    new = not OUT.exists()
    fh = open(OUT, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new:
        w.writeheader()

    n = 0
    for si, sess in enumerate(sessions):
        s = spot.get(sess)
        if not s:
            continue
        if all((SYMBOL, sess, str(t)) in done for t in TARGETS):
            continue
        rows = con.execute(
            "SELECT expiry_date, strike, option_type, close, contracts, open_interest "
            "FROM nse_options_bhav WHERE symbol=? AND trade_date=?", (SYMBOL, sess)).fetchall()
        chains = defaultdict(dict)
        exp_contracts = defaultdict(int)
        for e, k, ot, c, ct, oi in rows:
            if ot not in ("CE", "PE"):
                continue
            exp_contracts[e] += (ct or 0)
            chains[e].setdefault(float(k), {})[ot] = (c or 0.0, ct or 0, oi or 0)
        sd = dparse(sess)
        dtes = {e: (dparse(e) - sd).days for e in chains}

        for t in TARGETS:
            if (SYMBOL, sess, str(t)) in done:
                continue
            lo, hi = window(t)
            cands = [e for e, d in dtes.items() if lo <= d <= hi]
            row = dict(symbol=SYMBOL, session=sess, target_dte=t,
                       n_candidate_expiries=len(cands), spot=round(s, 2), fillable=0)
            best = None
            for e in cands:
                ch = chains[e]
                # nearest-to-spot strike with BOTH legs traded and priced
                pick, pd_ = None, 1e18
                ntr = 0
                for k, legs in ch.items():
                    if "CE" not in legs or "PE" not in legs:
                        continue
                    ce, pe = legs["CE"], legs["PE"]
                    if ce[0] <= 0 or pe[0] <= 0 or ce[1] <= 0 or pe[1] <= 0:
                        continue
                    ntr += 1
                    d = abs(k - s)
                    if d < pd_:
                        pick, pd_ = k, d
                if pick is None:
                    continue
                legs = ch[pick]
                vol = legs["CE"][1] + legs["PE"][1]
                cand = (vol, e, pick, pd_, legs, ntr)
                if best is None or vol > best[0]:
                    best = cand
            if best:
                _, e, k, d, legs, ntr = best
                row.update(best_expiry=e, best_dte=dtes[e], atm_strike=k,
                           atm_dist_pct=round(100.0 * d / s, 3),
                           atm_prem=round(legs["CE"][0] + legs["PE"][0], 2),
                           ce_contracts=legs["CE"][1], pe_contracts=legs["PE"][1],
                           ce_oi=legs["CE"][2], pe_oi=legs["PE"][2],
                           n_traded_strikes_in_expiry=ntr,
                           expiry_contracts=exp_contracts[e], fillable=1)
            w.writerow(row)
            n += 1
        if si % 200 == 0:
            fh.flush()
            print("  session %d/%d (%s) rows=%d" % (si, len(sessions), sess, n), flush=True)
    fh.close()
    print("DONE %d rows -> %s" % (n, OUT), flush=True)


if __name__ == "__main__":
    main()
