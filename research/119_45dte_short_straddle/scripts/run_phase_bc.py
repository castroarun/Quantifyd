#!/usr/bin/env python3
"""
Phase B — monitoring-timeframe bake-off (daily / 60m / 30m / 15m / 5m).
Phase C — India VIX percentile filter (rank vs previous 252 sessions) at entry.

Trades are built ONCE from real bhav data; every cell re-evaluates the same set of
trades under a different exit-check frequency and/or entry filter.

Intraday marks are RECONSTRUCTED (see STATUS section 3) — real 5-min NIFTY spot,
forward + IV backed out of real option closes, snapped back to the real price at
every daily close. iv_mode 'prev' is causal; 'same' is the anticipatory bracket.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine45 import (connect, trading_days, monthly_expiries, nifty_daily_close,
                      nifty_5min, india_vix_daily, vix_rank_series,
                      build_trade, run_daily, run_intraday, summarise, QTY)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
os.makedirs(RES, exist_ok=True)

WIN_LO, WIN_HI = "2019-01-01", "2026-06-30"
TFS = [("daily", None), ("60m", 60), ("30m", 30), ("15m", 15), ("5m", 5)]
VIX_THRESH = [("none", -1.0), (">25", 25.0), (">50", 50.0), (">75", 75.0)]


def main():
    con = connect()
    days = trading_days(con, "2018-06-01")
    spot = nifty_daily_close(con)
    exps = monthly_expiries(con, days, "2018-06-01", "2026-08-31")
    vrank = vix_rank_series(india_vix_daily(con))

    trades = []
    for ym, exp in exps.items():
        t = build_trade(con, exp, days, spot)
        if t and WIN_LO <= t["entry_date"] <= WIN_HI:
            t["vix_rank"] = vrank.get(t["entry_date"])
            trades.append(t)
    print("trades built: %d  (%s .. %s)" %
          (len(trades), trades[0]["entry_date"], trades[-1]["entry_date"]))
    missing_vix = sum(1 for t in trades if t["vix_rank"] is None)
    print("trades with no VIX rank: %d" % missing_vix)

    d0 = min(t["entry_date"] for t in trades)
    d1 = max(t["time_exit_date"] for t in trades)
    bars = nifty_5min(con, d0, d1)
    print("5-min spot days loaded: %d (%s .. %s)" %
          (len(bars), min(bars) if bars else "-", max(bars) if bars else "-"))
    covered = sum(1 for t in trades
                  if all(r["date"] in bars for r in t["path"][1:]))
    print("trades with FULL 5-min spot coverage: %d / %d" % (covered, len(trades)))

    grid, results_by_cell = [], {}
    for tf_name, tf_min in TFS:
        iv_modes = ["real"] if tf_min is None else ["prev", "same"]
        for iv in iv_modes:
            closed, modelled_exits = [], 0
            for t in trades:
                if tf_min is None:
                    r = run_daily(t)
                    src = "real"
                else:
                    r, src = run_intraday(t, bars, tf_min, iv_mode=iv)
                if src == "modelled":
                    modelled_exits += 1
                r["vix_rank"] = t["vix_rank"]
                closed.append(r)
            results_by_cell[(tf_name, iv)] = closed
            for vname, vthr in VIX_THRESH:
                sel = [r for r in closed
                       if vthr < 0 or (r["vix_rank"] is not None and r["vix_rank"] > vthr)]
                if not sel:
                    continue
                s = summarise([dict(x) for x in sel])
                grid.append(dict(tf=tf_name, iv_mode=iv, vix=vname,
                                 modelled_exits=modelled_exits, **{
                                     k: s[k] for k in
                                     ("trades", "win_rate", "avg_premium", "total_gross",
                                      "total_net", "avg_net", "t_stat", "max_dd",
                                      "avg_win", "avg_loss", "best", "worst",
                                      "target", "stop", "time", "total_net_rs", "max_dd_rs")}))
            print("  done %s/%s  (%d exits taken on modelled marks)" % (tf_name, iv, modelled_exits))

    out = os.path.join(RES, "grid_tf_vix.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(grid[0].keys()))
        w.writeheader()
        for g in grid:
            w.writerow(g)
    print("\nwrote %s (%d cells)" % (out, len(grid)))

    # ---------------- Phase B table: monitoring frequency, no VIX filter -----
    print("\n=== PHASE B — monitoring frequency (no VIX filter) ===")
    print("%-8s %-5s %5s %7s %10s %9s %6s %10s %-10s" %
          ("TF", "IV", "n", "win%", "net pts", "net/tr", "t", "MaxDD", "T/S/21DTE"))
    for g in grid:
        if g["vix"] != "none":
            continue
        print("%-8s %-5s %5d %7.1f %10.1f %9.1f %6.2f %10.1f %d/%d/%d" %
              (g["tf"], g["iv_mode"], g["trades"], g["win_rate"], g["total_net"],
               g["avg_net"], g["t_stat"], g["max_dd"], g["target"], g["stop"], g["time"]))

    # ---------------- Phase C table: VIX filter ------------------------------
    print("\n=== PHASE C — VIX percentile filter (rank vs previous 252 sessions) ===")
    for tf_name, iv in [("daily", "real"), ("60m", "prev"), ("30m", "prev")]:
        if (tf_name, iv) not in results_by_cell:
            continue
        print("\n-- %s monitoring (%s IV) --" % (tf_name, iv))
        print("%-6s %5s %7s %10s %10s %9s %6s %10s %-10s" %
              ("VIX", "n", "win%", "avgPrem", "net pts", "net/tr", "t", "MaxDD", "T/S/21DTE"))
        for g in grid:
            if g["tf"] != tf_name or g["iv_mode"] != iv:
                continue
            print("%-6s %5d %7.1f %10.1f %10.1f %9.1f %6.2f %10.1f %d/%d/%d" %
                  (g["vix"], g["trades"], g["win_rate"], g["avg_premium"], g["total_net"],
                   g["avg_net"], g["t_stat"], g["max_dd"], g["target"], g["stop"], g["time"]))

    # ---------------- cost sensitivity on the daily cell ---------------------
    print("\n=== cost sensitivity (daily monitoring, no filter) ===")
    base = results_by_cell[("daily", "real")]
    for slip in (0.0, 0.0025, 0.005, 0.01, 0.02):
        s = summarise([dict(x) for x in base], slip_pct=slip)
        print("  slip %4.2f%% per side -> net/tr %7.1f pts (Rs %8.0f)  total %9.1f  t=%5.2f  avgCost %.1f" %
              (slip * 100, s["avg_net"], s["avg_net_rs"], s["total_net"], s["t_stat"], s["avg_cost"]))

    # per-trade ledger for the 60m causal cell (the video's timeframe)
    if ("60m", "prev") in results_by_cell:
        cell = results_by_cell[("60m", "prev")]
        summarise([dict(x) for x in cell])  # populate cost/net
        for r in cell:
            r["cost_pts"] = r.get("cost_pts", 0.0)
        out2 = os.path.join(RES, "trades_60m.csv")
        rows = [dict(x) for x in cell]
        summarise(rows)
        with open(out2, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print("\nwrote %s" % out2)


if __name__ == "__main__":
    main()
