# -*- coding: utf-8 -*-
"""SENSEX portfolio bracket — same study as research/90 but on the recorded SENSEX chain, to give
the SENSEX 3-system book its OWN per-lot stop (the deployed -Rs1300/lot is NIFTY-calibrated).

Replays SENSEX ATM (SL_ST) + ATM2 (move 0.4% re-center) + ATM4 (roll) at 2 lots each (QTY 40),
builds the combined per-minute portfolio path per day, sweeps a daily TP/SL bracket, and reports the
best STOP per lot (6 lots = 2 x 3). Also confirms whether a take-profit helps on SENSEX (it did not
on NIFTY)."""
import csv, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
sys.path.insert(0, "/home/arun/quantifyd"); sys.path.insert(0, HERE)
import numpy as np
from engine_mtm_sensex import days, load_day, sim_loaded   # noqa

RESULTS = os.path.join(BASE, "results"); os.makedirs(RESULTS, exist_ok=True)
PATHS_JSON = os.path.join(RESULTS, "sensex_day_paths.json")
TOTAL_LOTS = 6

SYSCFG = [
    ("SENSEX ATM",  "t916", "SL_ST",      dict(stop_mode="premium", sl_mult=1.3,
                                               naked_method="st", st_period=7, st_mult=2.0)),
    ("SENSEX ATM2", "t916", "CASCADE",    dict(stop_mode="move", move_pct=0.004, move_stop_reenter=True)),
    ("SENSEX ATM4", "t916", "ROLL_MATCH", dict(stop_mode="premium", sl_mult=1.3)),
]


def build_paths():
    if os.path.exists(PATHS_JSON):
        return json.load(open(PATHS_JSON))
    day_paths = {}
    for i, day in enumerate(days, 1):
        b = load_day(day)
        if b is None:
            continue
        per_sys, grid = [], set()
        for _lab, em, mg, knobs in SYSCFG:
            out = sim_loaded(b, em, mg, record_mtm=True, **knobs)
            mtm = out[1] if isinstance(out, tuple) else []
            d = {ts: float(v) for ts, v in mtm}
            grid |= set(d.keys())
            per_sys.append(d)
        if not grid:
            continue
        last = [0.0] * len(per_sys); path = []
        for ts in sorted(grid):
            tot = 0.0
            for si, d in enumerate(per_sys):
                if ts in d:
                    last[si] = d[ts]
                tot += last[si]
            path.append((ts, round(tot, 1)))
        day_paths[day] = path
        if i % 8 == 0 or i == len(days):
            print("  replay %d/%d (%s)" % (i, len(days), day), flush=True)
    json.dump(day_paths, open(PATHS_JSON, "w"))
    return day_paths


def outcome(path, tp, sl):
    for _ts, v in path:
        if tp is not None and v >= tp:
            return v
        if sl is not None and v <= sl:
            return v
    return path[-1][1]


def metrics(dailies, keys, half):
    a = np.array(dailies, float); cum = np.cumsum(a)
    dd = float((cum - np.maximum.accumulate(cum)).min()); tot = float(a.sum())
    return dict(total=round(tot), maxdd=round(dd), worst=round(float(a.min())),
                win=round(float((a > 0).mean()) * 100, 1),
                calmar=(round(tot / abs(dd), 2) if dd < 0 else None),
                h1=round(float(a[:half].sum())), h2=round(float(a[half:].sum())))


def main():
    print("Building SENSEX per-day combined paths ...", flush=True)
    dp = build_paths(); keys = sorted(dp.keys()); half = len(keys) // 2
    print("Days: %d (%s..%s)" % (len(keys), keys[0], keys[-1]), flush=True)

    # intraday range distribution (for choosing the level)
    peaks = [max(v for _t, v in dp[d]) for d in keys]
    troughs = [min(v for _t, v in dp[d]) for d in keys]
    eods = [dp[d][-1][1] for d in keys]
    import statistics as st
    print("\nSENSEX combined book (6 lots) intraday range:")
    print("  PEAK median %+d p90 %+d max %+d | TROUGH median %+d p10 %+d min %+d | EOD median %+d"
          % (st.median(peaks), sorted(peaks)[int(0.9*len(peaks))], max(peaks),
             st.median(troughs), sorted(troughs)[int(0.1*len(troughs))], min(troughs), st.median(eods)))

    TPS = [None, 3000, 4000, 5000, 6000, 8000, 10000, 12000, 15000]
    SLS = [None, -2000, -3000, -4000, -5000, -6000, -8000, -10000, -12000]
    rows = []
    for tp in TPS:
        for sl in SLS:
            m = metrics([outcome(dp[d], tp, sl) for d in keys], keys, half)
            m["tp"] = tp or 0; m["sl"] = sl or 0
            rows.append(m)
    with open(os.path.join(RESULTS, "sensex_sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tp", "sl", "total", "calmar", "maxdd", "worst", "win", "h1", "h2"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in ["tp", "sl", "total", "calmar", "maxdd", "worst", "win", "h1", "h2"]})

    base = next(r for r in rows if r["tp"] == 0 and r["sl"] == 0)
    print("\nBASELINE (no bracket): total %+d calmar %s maxDD %+d worst %+d win %.1f [h1 %+d|h2 %+d]"
          % (base["total"], base["calmar"], base["maxdd"], base["worst"], base["win"], base["h1"], base["h2"]))

    print("\n=== STOP-ONLY curve (no target), by SL  [per-lot = SL/6] ===")
    print("%8s %8s %8s %7s %9s %8s %8s %8s" % ("SL", "perlot", "total", "calmar", "maxDD", "worst", "h1", "h2"))
    for sl in [-2000, -3000, -4000, -5000, -6000, -8000, -10000, -12000]:
        r = next(x for x in rows if x["tp"] == 0 and x["sl"] == sl)
        print("%8d %8.0f %+8d %7s %+9d %+8d %+8d %+8d"
              % (sl, sl / TOTAL_LOTS, r["total"], ("%.2f" % r["calmar"]) if r["calmar"] else "-",
                 r["maxdd"], r["worst"], r["h1"], r["h2"]))

    print("\n=== TOP 8 by TOTAL and by CALMAR ===")
    for key in ("total", "calmar"):
        print("-- by %s --" % key)
        for r in sorted(rows, key=lambda x: (x[key] is None, x[key]), reverse=True)[:8]:
            print("  TP %5s SL %6s -> total %+7d calmar %5s maxDD %+8d [h1 %+d|h2 %+d]"
                  % (r["tp"] or "-", r["sl"] or "-", r["total"],
                     ("%.2f" % r["calmar"]) if r["calmar"] else "-", r["maxdd"], r["h1"], r["h2"]))


if __name__ == "__main__":
    main()
