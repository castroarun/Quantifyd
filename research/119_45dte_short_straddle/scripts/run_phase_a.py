#!/usr/bin/env python3
"""
Phase A — replicate the published table on REAL NIFTY option prices.
Daily-close exit monitoring. Writes results/trades_daily.csv + prints the comparison.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine45 import (connect, trading_days, monthly_expiries, nifty_daily_close,
                      build_trade, run_daily, run_touch_bracket, summarise, fmt_row,
                      QTY, LOTS)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
os.makedirs(RES, exist_ok=True)

PUBLISHED = dict(trades=83, win_rate=69.9, avg_premium=758.9, total=5951.6, per_trade=71.7,
                 avg_win=196.1, avg_loss=-216.8, best=805.3, worst=-1062.6, mdd=-1062.6,
                 target=1, stop=4, time=78)


def main():
    con = connect()
    days = trading_days(con, "2018-06-01")
    spot = nifty_daily_close(con)
    exps = monthly_expiries(con, days, "2018-06-01", "2026-08-31")
    print("monthly expiries found: %d  (%s .. %s)" %
          (len(exps), min(exps.values()), max(exps.values())))
    import datetime as _dt
    wk = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    cal = ["%s(%s)" % (e, wk[_dt.date(*map(int, e.split("-"))).weekday()])
           for e in sorted(exps.values())]
    print("expiry calendar:", " ".join(cal))

    trades, skipped = [], []
    for ym, exp in exps.items():
        t = build_trade(con, exp, days, spot)
        if not t:
            skipped.append(exp)
            continue
        r = run_daily(t)
        sd, td = run_touch_bracket(t)
        r["touch_stop_day"] = sd["date"] if sd else ""
        r["touch_target_day"] = td["date"] if td else ""
        r["month"] = ym
        trades.append(r)
    print("built %d trades, skipped %d (%s)" % (len(trades), len(skipped), skipped[:6]))

    # --- window conventions -------------------------------------------------
    by_entry = [t for t in trades if "2019-01-01" <= t["entry_date"] <= "2026-06-30"]
    by_expiry = [t for t in trades if "2019-01-01" <= t["expiry"] <= "2026-06-30"]
    print("\nwindow by ENTRY  date 2019-01-01..2026-06-30 : %d trades" % len(by_entry))
    print("window by EXPIRY date 2019-01-01..2026-06-30 : %d trades" % len(by_expiry))

    for label, ts in (("BY_ENTRY", by_entry), ("BY_EXPIRY", by_expiry)):
        s = summarise(list(ts))
        print("  " + fmt_row(label, s))

    ts = by_entry if abs(len(by_entry) - 83) <= abs(len(by_expiry) - 83) else by_expiry
    chosen = "BY_ENTRY" if ts is by_entry else "BY_EXPIRY"
    print("\n=== HEADLINE (%s, daily-close monitoring, real bhav prices) ===" % chosen)

    s_gross = summarise(list(ts), slip_pct=0.0)
    s_net = summarise(list(ts), slip_pct=0.0025)

    rows = [
        ("Trades",              "%d" % PUBLISHED["trades"],       "%d" % s_net["trades"]),
        ("Win rate %",          "%.1f" % PUBLISHED["win_rate"],   "%.1f" % s_net["win_rate"]),
        ("Avg premium sold",    "%.1f" % PUBLISHED["avg_premium"], "%.1f" % s_net["avg_premium"]),
        ("Exits T/S/21DTE",     "%d/%d/%d" % (PUBLISHED["target"], PUBLISHED["stop"], PUBLISHED["time"]),
                                "%d/%d/%d" % (s_net["target"], s_net["stop"], s_net["time"])),
        ("Total P&L pts",       "%.1f" % PUBLISHED["total"],      "%.1f gross / %.1f net" % (s_gross["total_net"], s_net["total_net"])),
        ("Avg P&L per trade",   "%.1f" % PUBLISHED["per_trade"],  "%.1f gross / %.1f net" % (s_gross["avg_net"], s_net["avg_net"])),
        ("Avg win / avg loss",  "%.1f / %.1f" % (PUBLISHED["avg_win"], PUBLISHED["avg_loss"]),
                                "%.1f / %.1f" % (s_net["avg_win"], s_net["avg_loss"])),
        ("Best trade",          "%.1f" % PUBLISHED["best"],       "%.1f" % s_net["best"]),
        ("Worst trade",         "%.1f" % PUBLISHED["worst"],      "%.1f" % s_net["worst"]),
        ("Max drawdown pts",    "%.1f" % PUBLISHED["mdd"],        "%.1f" % s_net["max_dd"]),
    ]
    print("%-22s | %-18s | %s" % ("Metric", "Published", "Ours (real data)"))
    print("-" * 78)
    for a, b, c in rows:
        print("%-22s | %-18s | %s" % (a, b, c))

    print("\n10 lots (qty %d): total net Rs %.0f | per trade Rs %.0f | MaxDD Rs %.0f" %
          (QTY, s_net["total_net_rs"], s_net["avg_net_rs"], s_net["max_dd_rs"]))
    print("avg round-trip cost: %.1f pts (Rs %.0f at %d lots)" %
          (s_net["avg_cost"], s_net["avg_cost"] * QTY, LOTS))
    print("t-stat (net, per trade): %.2f" % s_net["t_stat"])

    # --- per-year -----------------------------------------------------------
    print("\nPer-year (net):")
    years = sorted({t["entry_date"][:4] for t in ts})
    for y in years:
        yy = [t for t in ts if t["entry_date"][:4] == y]
        sy = summarise(list(yy))
        print("  %s  n=%2d  net=%8.1f pts (Rs %9.0f)  win=%5.1f%%  worst=%8.1f  T/S/E=%d/%d/%d" %
              (y, sy["trades"], sy["total_net"], sy["total_net_rs"], sy["win_rate"],
               sy["worst"], sy["target"], sy["stop"], sy["time"]))

    # --- concentration ------------------------------------------------------
    srt = sorted(ts, key=lambda t: t["net_pts"])
    print("\nWorst 5:", ["%s %.0f" % (t["entry_date"], t["net_pts"]) for t in srt[:5]])
    print("Best  5:", ["%s %.0f" % (t["entry_date"], t["net_pts"]) for t in srt[-5:]])
    tot = sum(t["net_pts"] for t in ts)
    top3 = sum(t["net_pts"] for t in srt[-3:])
    print("top-3 trades = %.0f pts = %.0f%% of total net %.0f" %
          (top3, 100.0 * top3 / tot if tot else 0, tot))

    # --- touch bracket ------------------------------------------------------
    nstop_touch = sum(1 for t in ts if t["touch_stop_day"])
    ntgt_touch = sum(1 for t in ts if t["touch_target_day"])
    print("\nTouch bracket (real bhav daily leg highs/lows — bound on ANY intraday scheme):")
    print("  trades where CE.high+PE.high ever pierced the 200%% stop : %d / %d" % (nstop_touch, len(ts)))
    print("  trades where CE.low +PE.low  ever pierced the 50%% target : %d / %d" % (ntgt_touch, len(ts)))
    print("  (daily-close rule fired: stop %d, target %d)" % (s_net["stop"], s_net["target"]))

    out = os.path.join(RES, "trades_daily.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ts[0].keys()))
        w.writeheader()
        for t in ts:
            w.writerow(t)
    print("\nwrote %s (%d rows)" % (out, len(ts)))


if __name__ == "__main__":
    main()
