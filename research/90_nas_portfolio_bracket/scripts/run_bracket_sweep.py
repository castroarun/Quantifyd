# -*- coding: utf-8 -*-
"""Replay the 3 live 916 systems across all recorded days, build the combined per-minute
portfolio P&L path per day, then sweep a daily TP/SL bracket and rank by P&L + risk.

See NAS_PORTFOLIO_BRACKET_1MIN_SWEEP_STATUS.md for the full ask / base / plan."""
import csv
import json
import os
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, "/home/arun/quantifyd")
sys.path.insert(0, HERE)

import numpy as np
from engine_mtm import days, load_day, sim_loaded   # noqa

RESULTS = os.path.join(BASE, "results")
os.makedirs(RESULTS, exist_ok=True)
PATHS_JSON = os.path.join(RESULTS, "day_paths.json")

# current-live configs: (label, entry_mode, mgmt, knobs)
SYSCFG = [
    ("916 ATM",  "t916", "SL_ST",      dict(stop_mode="premium", sl_mult=1.3,
                                            naked_method="st", st_period=7, st_mult=2.0)),
    ("916 ATM2", "t916", "CASCADE",    dict(stop_mode="move", move_pct=0.004,
                                            move_stop_reenter=True)),
    ("916 ATM4", "t916", "ROLL_MATCH", dict(stop_mode="premium", sl_mult=1.3)),
]


def build_paths():
    if os.path.exists(PATHS_JSON):
        with open(PATHS_JSON) as f:
            return json.load(f)
    day_paths = {}
    n = len(days)
    for i, day in enumerate(days, 1):
        b = load_day(day)
        if b is None:
            continue
        per_sys = []          # list of dict{ts_str: val}
        grid = set()
        for label, em, mg, knobs in SYSCFG:
            out = sim_loaded(b, em, mg, record_mtm=True, **knobs)
            mtm = out[1] if isinstance(out, tuple) else []
            d = {}
            for ts, v in mtm:
                d[ts] = float(v)
                grid.add(ts)
            per_sys.append(d)
        if not grid:
            continue
        gts = sorted(grid)
        # forward-fill each system onto the union grid, sum
        path = []
        last = [0.0] * len(per_sys)
        for ts in gts:
            tot = 0.0
            for si, d in enumerate(per_sys):
                if ts in d:
                    last[si] = d[ts]
                tot += last[si]
            path.append((ts, round(tot, 1)))
        day_paths[day] = path
        if i % 8 == 0 or i == n:
            print("  replay %d/%d days (%s)" % (i, n, day), flush=True)
    with open(PATHS_JSON, "w") as f:
        json.dump(day_paths, f)
    return day_paths


def outcome(path, tp, sl):
    """First minute the combined P&L breaches TP or SL -> book that value; else EOD."""
    for _ts, v in path:
        if tp is not None and v >= tp:
            return v
        if sl is not None and v <= sl:
            return v
    return path[-1][1]


def metrics(dailies):
    a = np.array(dailies, dtype=float)
    cum = np.cumsum(a)
    peak = np.maximum.accumulate(cum)
    maxdd = float((cum - peak).min())          # <= 0
    total = float(a.sum())
    return dict(total=round(total), maxdd=round(maxdd), worst=round(float(a.min())),
                best=round(float(a.max())), std=round(float(a.std())),
                winrate=round(float((a > 0).mean()) * 100, 1),
                calmar=(round(total / abs(maxdd), 2) if maxdd < 0 else None))


def main():
    t0 = datetime.now()
    print("Building per-day combined portfolio paths ...", flush=True)
    day_paths = build_paths()
    keys = sorted(day_paths.keys())
    print("Days with paths: %d (%s .. %s)" % (len(keys), keys[0], keys[-1]), flush=True)

    TPS = [None, 3000, 4000, 5000, 6000, 8000, 10000, 12000, 15000]
    SLS = [None, -3000, -4000, -5000, -6000, -8000, -10000, -12000, -15000]

    half = len(keys) // 2
    first, second = keys[:half], keys[half:]

    rows = []
    for tp in TPS:
        for sl in SLS:
            dailies = [outcome(day_paths[d], tp, sl) for d in keys]
            m = metrics(dailies)
            m["tp"] = tp if tp is not None else 0
            m["sl"] = sl if sl is not None else 0
            m["h1"] = round(sum(outcome(day_paths[d], tp, sl) for d in first))
            m["h2"] = round(sum(outcome(day_paths[d], tp, sl) for d in second))
            rows.append(m)

    base = next(r for r in rows if r["tp"] == 0 and r["sl"] == 0)

    fields = ["tp", "sl", "total", "calmar", "maxdd", "worst", "best", "std", "winrate", "h1", "h2"]
    with open(os.path.join(RESULTS, "sweep_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})

    def show(title, key, rev=True, n=12):
        print("\n=== %s ===" % title)
        print("%7s %7s %8s %7s %8s %8s %7s %6s %7s %7s" %
              ("TP", "SL", "total", "calmar", "maxDD", "worst", "std", "win%", "h1", "h2"))
        for r in sorted(rows, key=lambda x: (x[key] is None, x[key]), reverse=rev)[:n]:
            print("%7s %7s %+8d %7s %+8d %+8d %7d %6.1f %+7d %+7d" %
                  (r["tp"] or "-", r["sl"] or "-", r["total"],
                   ("%.2f" % r["calmar"]) if r["calmar"] is not None else "-",
                   r["maxdd"], r["worst"], r["std"], r["winrate"], r["h1"], r["h2"]))

    print("\nBASELINE (ride to EOD, no bracket): total=%+d  calmar=%s  maxDD=%+d  worst=%+d  win%%=%.1f  [h1 %+d | h2 %+d]"
          % (base["total"], base["calmar"], base["maxdd"], base["worst"], base["winrate"], base["h1"], base["h2"]))
    show("TOP 12 by TOTAL net P&L", "total")
    show("TOP 12 by CALMAR (total / |maxDD|)", "calmar")

    # the 5k/5k the user originally asked about
    r55 = next(r for r in rows if r["tp"] == 5000 and r["sl"] == -5000)
    print("\n5k/5k bracket: total=%+d  calmar=%s  maxDD=%+d  worst=%+d  win%%=%.1f  [h1 %+d | h2 %+d]"
          % (r55["total"], r55["calmar"], r55["maxdd"], r55["worst"], r55["winrate"], r55["h1"], r55["h2"]))

    print("\nElapsed %.0fs" % (datetime.now() - t0).total_seconds())


if __name__ == "__main__":
    main()
