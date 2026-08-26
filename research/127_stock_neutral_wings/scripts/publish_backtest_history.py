#!/usr/bin/env python3
"""Publish the BACKTESTED trade history of the live book's own structure.

The /app/stock-wings page shows what the paper book holds now. This answers the
next question — "show me the same structure historically" — from the study that
justified the book, so the live rows can be read against 3,700 prior instances
of the identical construction rather than against nothing.

Structure is the adopted C1 ruleset, unchanged: enter 45 DTE, exit 21 DTE,
shorts +/-2.5% of spot, wings 7% out, no stop, TP at 50% of credit, and the same
liquidity gate the live book screens on (ATM volume >= 100, thinnest wing >= 10).

net = gross - 0.5% of premium turnover, the study's cost model. READ ONLY.
"""
import csv
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "phase_b2_trades.csv")
OUT = [os.path.join(ROOT, "static", "app", "stock_wings_backtest.json"),
       os.path.join(ROOT, "frontend", "public", "stock_wings_backtest.json")]
CONFIG = "C1_E45X21W7K25_noSL"
ATM_VOL_MIN, WING_VOL_MIN, COST = 100, 10, 0.005


def agg(rows):
    if not rows:
        return None
    n = len(rows)
    nets = [r["net_pct"] for r in rows]
    w = [x for x in nets if x > 0]
    return dict(n=n, win=round(100.0 * len(w) / n, 1),
                avg=round(sum(nets) / n, 4),
                med=round(sorted(nets)[n // 2], 4),
                best=round(max(nets), 3), worst=round(min(nets), 3),
                avg_credit=round(sum(r["credit_pct"] for r in rows) / n, 3),
                avg_hold=round(sum(r["hold_days"] for r in rows) / n, 1))


def main():
    trades = []
    with open(SRC) as f:
        for r in csv.DictReader(f):
            if r["config"] != CONFIG:
                continue
            try:
                if float(r["atm_vol"]) < ATM_VOL_MIN or float(r["wing_vol_min"]) < WING_VOL_MIN:
                    continue
                g, t = float(r["gross_pct"]), float(r["turnover_pct"])
            except (ValueError, TypeError):
                continue
            trades.append(dict(
                symbol=r["symbol"], expiry=r["expiry"], entry=r["entry_date"],
                exit=r["exit_date"], reason=r["exit_reason"], year=int(r["year"]),
                spot=round(float(r["S0"]), 1),
                kce=float(r["Ks_ce"]), kpe=float(r["Ks_pe"]),
                wce=float(r["Kc"]), wpe=float(r["Kp"]),
                hold_days=int(float(r["hold_days"])),
                credit_pct=round(100 * float(r["credit_pct"]), 3),
                gross_pct=round(100 * g, 3),
                net_pct=round(100 * (g - COST * t), 3)))
    trades.sort(key=lambda r: (r["entry"], r["symbol"]))

    by_sym, by_year, by_reason = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in trades:
        by_sym[r["symbol"]].append(r)
        by_year[r["year"]].append(r)
        by_reason[r["reason"]].append(r)

    payload = dict(
        config=CONFIG, source="research/127 phase B2 (G3 universal ruleset)",
        cost_model="net = gross - 0.5%% of premium turnover",
        gate=dict(atm_vol_min=ATM_VOL_MIN, wing_vol_min=WING_VOL_MIN),
        overall=agg(trades),
        by_symbol=sorted(
            [dict(symbol=s, **agg(v)) for s, v in by_sym.items()],
            key=lambda d: -d["n"]),
        by_year=[dict(year=y, **agg(by_year[y])) for y in sorted(by_year)],
        by_reason=sorted([dict(reason=k, **agg(v)) for k, v in by_reason.items()],
                         key=lambda d: -d["n"]),
        trades=trades)

    for p in OUT:
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p + ".tmp", "w") as f:
                json.dump(payload, f, separators=(",", ":"))
            os.replace(p + ".tmp", p)
        except Exception as e:
            print("  publish %s failed: %s" % (p, e))
    o = payload["overall"]
    print("%d backtested trades of the SAME structure, %d symbols"
          % (o["n"], len(by_sym)))
    print("  net %+.3f%% of spot per trade, %.1f%% win, avg hold %.1f days"
          % (o["avg"], o["win"], o["avg_hold"]))
    print("  size: %.0f KB" % (os.path.getsize(OUT[0]) / 1024.0))


if __name__ == "__main__":
    sys.exit(main())
