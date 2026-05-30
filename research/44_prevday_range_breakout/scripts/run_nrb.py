"""Prev-day range-compression breakout sweep (Phase 1).

Per symbol: load daily bars (setup: box, compression flags, ATR) and 5-min bars
(trigger + intraday mgmt). For each trading day with a prev day, find the first
5-min close beyond the prev-day range, then simulate every SL x target x hold
combo and route the result into each compression bucket the prev day satisfies.
Per-symbol x variant tallies are appended immediately (crash-safe, resumable).

Usage:
  python run_nrb.py                       # full run
  python run_nrb.py --only RELIANCE,SBIN  # smoke
  python run_nrb.py --aggregate-only
"""
from __future__ import annotations

import argparse
import csv
import logging
import sqlite3
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from nrb import simulate, sl_price, SL_DEFS, TARGETS, HOLDS  # noqa: E402

DB = ROOT / "backtest_data" / "market_data.db"
RESULTS = HERE.parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
STATS_CSV = RESULTS / "symbol_variant_stats.csv"

INDEX_SYMBOLS = {"NIFTY50"}
EXCLUDE = {"BANKNIFTY"}
SESS_LO, SESS_HI, BREAK_MIN = "09:15", "15:25", "09:20"
COST_BPS = 6.0
COMPRESSIONS = ("NONE", "NR7", "NR4", "PCT25", "INSIDE")
VARIANTS = [f"{comp}|{sl}|{tg}|{hd}" for comp in COMPRESSIONS
            for sl in SL_DEFS for tg in TARGETS for hd in HOLDS]

STATS_FIELDS = ["cohort", "symbol", "variant", "trades", "wins", "sumR_gross",
                "sum_cost", "sumR_net", "win_R", "loss_R", "sum_hold",
                "n_long", "wins_long", "sumR_net_long", "win_R_long", "loss_R_long",
                "n_short", "wins_short", "sumR_net_short", "win_R_short", "loss_R_short"]


def wilder_atr_daily(h, l, c, period=14):
    n = len(c)
    tr = np.empty(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    atr = np.full(n, np.nan)
    if n > period:
        atr[period] = tr[1:period + 1].mean()
        for i in range(period + 1, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def load_daily(con, symbol):
    rows = con.execute("SELECT date,open,high,low,close FROM market_data_unified "
                       "WHERE timeframe='day' AND symbol=? ORDER BY date",
                       (symbol,)).fetchall()
    if len(rows) < 30:
        return None
    dates = [r[0][:10] for r in rows]
    o = np.array([r[1] for r in rows], float)
    h = np.array([r[2] for r in rows], float)
    l = np.array([r[3] for r in rows], float)
    c = np.array([r[4] for r in rows], float)
    atr = wilder_atr_daily(h, l, c)
    rng = h - l
    return dict(dates=dates, idx={d: i for i, d in enumerate(dates)},
                o=o, h=h, l=l, c=c, atr=atr, rng=rng)


def load_5min_by_day(con, symbol):
    rows = con.execute("SELECT date,open,high,low,close FROM market_data_unified "
                       "WHERE timeframe='5minute' AND symbol=? ORDER BY date",
                       (symbol,)).fetchall()
    by_day = defaultdict(list)
    for dt, o, h, l, c in rows:
        t = dt[11:16]
        if SESS_LO <= t <= SESS_HI:
            by_day[dt[:10]].append((t, o, h, l, c))
    return by_day


def compression_flags(D, i):
    """Which compression buckets does prev day (i-1) satisfy? i is the entry day."""
    p = i - 1
    flags = {"NONE": True, "NR7": False, "NR4": False, "PCT25": False, "INSIDE": False}
    rng = D["rng"]
    if p >= 7:
        flags["NR7"] = rng[p] == np.min(rng[p - 6:p + 1])
    if p >= 4:
        flags["NR4"] = rng[p] == np.min(rng[p - 3:p + 1])
    if p >= 20:
        flags["PCT25"] = rng[p] <= np.percentile(rng[p - 20:p], 25)
    if p >= 1:
        flags["INSIDE"] = D["h"][p] < D["h"][p - 1] and D["l"][p] > D["l"][p - 1]
    return flags


def new_tally():
    return {v: dict(trades=0, wins=0, sumR_gross=0.0, sum_cost=0.0, sumR_net=0.0,
                    win_R=0.0, loss_R=0.0, sum_hold=0.0,
                    n_long=0, wins_long=0, sumR_net_long=0.0, win_R_long=0.0, loss_R_long=0.0,
                    n_short=0, wins_short=0, sumR_net_short=0.0, win_R_short=0.0, loss_R_short=0.0)
            for v in VARIANTS}


def _add(tv, r_gross, cost_R, hold_days, direction):
    r_net = r_gross - cost_R
    tv["trades"] += 1
    tv["sumR_gross"] += r_gross
    tv["sum_cost"] += cost_R
    tv["sumR_net"] += r_net
    tv["sum_hold"] += hold_days
    tv["wins"] += 1 if r_net > 0 else 0
    if r_net >= 0:
        tv["win_R"] += r_net
    else:
        tv["loss_R"] += r_net
    if direction > 0:
        tv["n_long"] += 1
        tv["sumR_net_long"] += r_net
        tv["wins_long"] += 1 if r_net > 0 else 0
        if r_net >= 0:
            tv["win_R_long"] += r_net
        else:
            tv["loss_R_long"] += r_net
    else:
        tv["n_short"] += 1
        tv["sumR_net_short"] += r_net
        tv["wins_short"] += 1 if r_net > 0 else 0
        if r_net >= 0:
            tv["win_R_short"] += r_net
        else:
            tv["loss_R_short"] += r_net


def run_symbol(con, symbol):
    D = load_daily(con, symbol)
    if D is None:
        return None
    by_day = load_5min_by_day(con, symbol)
    if not by_day:
        return None
    tally = new_tally()
    daily_tuples = list(zip(D["o"], D["h"], D["l"], D["c"]))
    for d, bars in by_day.items():
        i = D["idx"].get(d)
        if i is None or i < 1:
            continue
        pdh, pdl = D["h"][i - 1], D["l"][i - 1]
        prevrange = pdh - pdl
        datr = D["atr"][i - 1]
        if prevrange <= 0 or np.isnan(datr):
            continue
        # --- find first 5-min close beyond the box (>= 09:20) ---
        direction = entry = ebar_lo = ebar_hi = None
        ebar_idx = None
        for k, (t, o, h, l, c) in enumerate(bars):
            if t < BREAK_MIN:
                continue
            if c > pdh:
                direction, entry, ebar_lo, ebar_hi, ebar_idx = 1, c, l, h, k
                break
            if c < pdl:
                direction, entry, ebar_lo, ebar_hi, ebar_idx = -1, c, l, h, k
                break
        if direction is None:
            continue
        # arrays after the breakout bar (entry-day intraday mgmt)
        post = bars[ebar_idx + 1:]
        d5h = np.array([b[2] for b in post], float)
        d5l = np.array([b[3] for b in post], float)
        d5c = np.array([b[4] for b in post], float)
        daily_after = daily_tuples[i + 1:]
        flags = compression_flags(D, i)
        for sl_def in SL_DEFS:
            SL = sl_price(sl_def, direction, entry, pdl, pdh, prevrange, datr,
                          ebar_lo, ebar_hi)
            R = abs(entry - SL)
            if R <= 0:
                continue
            cost_R = (COST_BPS / 1e4 * abs(entry)) / R
            for tg in TARGETS:
                for hd in HOLDS:
                    out = simulate(direction, entry, SL, tg, prevrange,
                                   d5h, d5l, d5c, daily_after, hd)
                    if out is None:
                        continue
                    r_gross, hold_days = out
                    for comp in COMPRESSIONS:
                        if flags[comp]:
                            _add(tally[f"{comp}|{sl_def}|{tg}|{hd}"],
                                 r_gross, cost_R, hold_days, direction)
    return tally


def cohort_of(symbol):
    if symbol in INDEX_SYMBOLS:
        return "nifty50"
    if symbol in EXCLUDE:
        return None
    return "stocks"


def all_symbols(con):
    return [r[0] for r in con.execute(
        "SELECT DISTINCT symbol FROM market_data_unified WHERE timeframe='5minute' "
        "ORDER BY symbol").fetchall()]


def done_symbols():
    if not STATS_CSV.exists():
        return set()
    with open(STATS_CSV) as f:
        return {row["symbol"] for row in csv.DictReader(f)}


def append_stats(symbol, cohort, tally):
    write_header = not STATS_CSV.exists()
    with open(STATS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=STATS_FIELDS)
        if write_header:
            w.writeheader()
        for variant, tv in tally.items():
            if tv["trades"] == 0:
                continue
            w.writerow(dict(cohort=cohort, symbol=symbol, variant=variant, **{
                k: (round(v, 4) if isinstance(v, float) else v) for k, v in tv.items()}))


def _pf(win_r, loss_r):
    return round(win_r / abs(loss_r), 3) if loss_r < 0 else "inf"


def aggregate():
    if not STATS_CSV.exists():
        print("no stats yet")
        return
    agg = {"stocks": defaultdict(lambda: defaultdict(float)),
           "nifty50": defaultdict(lambda: defaultdict(float))}
    keys = [k for k in STATS_FIELDS if k not in ("cohort", "symbol", "variant")]
    with open(STATS_CSV) as f:
        for row in csv.DictReader(f):
            a = agg[row["cohort"]][row["variant"]]
            for k in keys:
                a[k] += float(row[k])
    fields = ["variant", "comp", "sl", "tgt", "hold", "trades", "win_pct",
              "net_exp_R", "net_PF", "gross_exp_R", "net_total_R", "avg_hold_d",
              "n_long", "long_win_pct", "long_exp_R", "long_PF",
              "n_short", "short_win_pct", "short_exp_R", "short_PF"]
    for cohort, by_variant in agg.items():
        rows = []
        for variant, a in by_variant.items():
            tr = a["trades"]
            if tr == 0:
                continue
            comp, sl, tg, hd = variant.split("|")
            nl, ns = a["n_long"], a["n_short"]
            rows.append(dict(
                variant=variant, comp=comp, sl=sl, tgt=tg, hold=hd, trades=int(tr),
                win_pct=round(100 * a["wins"] / tr, 1),
                net_exp_R=round(a["sumR_net"] / tr, 4),
                net_PF=_pf(a["win_R"], a["loss_R"]),
                gross_exp_R=round(a["sumR_gross"] / tr, 4),
                net_total_R=round(a["sumR_net"], 1),
                avg_hold_d=round(a["sum_hold"] / tr, 2),
                n_long=int(nl),
                long_win_pct=round(100 * a["wins_long"] / nl, 1) if nl else 0,
                long_exp_R=round(a["sumR_net_long"] / nl, 4) if nl else 0,
                long_PF=_pf(a["win_R_long"], a["loss_R_long"]),
                n_short=int(ns),
                short_win_pct=round(100 * a["wins_short"] / ns, 1) if ns else 0,
                short_exp_R=round(a["sumR_net_short"] / ns, 4) if ns else 0,
                short_PF=_pf(a["win_R_short"], a["loss_R_short"]),
            ))
        rows.sort(key=lambda r: r["net_exp_R"], reverse=True)
        out = RESULTS / f"ranking_{cohort}.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        elig = [r for r in rows if r["trades"] >= 300]
        print(f"== {cohort}: {len(rows)} variants -> {out.name} (top by net exp, >=300 trades) ==")
        for r in elig[:10]:
            print(f"   {r['variant']:24s} n={r['trades']:6d} wr={r['win_pct']:4.1f}% "
                  f"NET={r['net_exp_R']:+.3f}R PF={r['net_PF']} hold={r['avg_hold_d']:.1f}d "
                  f"totR={r['net_total_R']:.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.aggregate_only:
        aggregate()
        return
    con = sqlite3.connect(DB)
    syms = all_symbols(con)
    if args.only:
        wanted = set(args.only.split(","))
        syms = [s for s in syms if s in wanted]
    done = done_symbols()
    todo = [s for s in syms if s not in done and cohort_of(s) is not None]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(syms)} symbols | {len(done)} done | {len(todo)} to process")
    t0 = time.time()
    for n, sym in enumerate(todo, 1):
        cohort = cohort_of(sym)
        st = time.time()
        tally = run_symbol(con, sym)
        if tally is None:
            print(f"[{n}/{len(todo)}] {sym} no data", flush=True)
            continue
        append_stats(sym, cohort, tally)
        tt = sum(tv["trades"] for tv in tally.values())
        print(f"[{n}/{len(todo)}] {sym:12s} ({cohort:7s}) {tt:6d} trade-rows "
              f"{time.time()-st:4.1f}s elapsed={time.time()-t0:6.0f}s", flush=True)
    print(f"done {len(todo)} symbols in {time.time()-t0:.0f}s")
    aggregate()


if __name__ == "__main__":
    main()
