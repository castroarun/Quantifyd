"""Prev-day range breakout — Phase 2: confluence-filter gating.

Carries forward the Phase-1 least-bad core (compression in {NONE,NR7,PCT25} x SL
in {BOX,HALF} x target in {2R,3R,TRAIL} x SWING) and routes each breakout into
every filter-combo it satisfies:

  cprok     current-day CPR is narrow-to-normal (width <= trailing-60d 75th pctile)
  htf       daily trend agrees: prev close vs SMA50 (long: above, short: below)
  highbeta  stock-level: beta vs NIFTY50 (full-period daily returns) >= BETA_HI

Per-direction win/PF tracked from the start. Crash-safe incremental tallies.

Usage:
  python run_nrb_p2.py [--only ...] [--aggregate-only]
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
from nrb import simulate, sl_price  # noqa: E402
from run_nrb import (load_daily, load_5min_by_day, compression_flags,  # noqa: E402
                     cohort_of, all_symbols, COST_BPS, BREAK_MIN)

DB = ROOT / "backtest_data" / "market_data.db"
RESULTS = HERE.parent / "results"
STATS_CSV = RESULTS / "symbol_variant_stats_p2.csv"

CORE_COMP = ("NONE", "NR7", "PCT25")
CORE_SL = ("BOX", "HALF")
CORE_TG = ("2R", "3R", "TRAIL")
HOLD = "SWING"
CPR_PCTL = 75            # narrow-to-normal = width <= trailing 75th pctile
BETA_HI = 1.2
FILTERS = [
    ("none", ()),
    ("cprok", ("cprok",)),
    ("htf", ("htf",)),
    ("highbeta", ("highbeta",)),
    ("cprok+htf", ("cprok", "htf")),
    ("htf+highbeta", ("htf", "highbeta")),
    ("cprok+highbeta", ("cprok", "highbeta")),
    ("all", ("cprok", "htf", "highbeta")),
]
VARIANTS = [f"{c}|{s}|{t}|{fn}" for c in CORE_COMP for s in CORE_SL
            for t in CORE_TG for (fn, _) in FILTERS]

STATS_FIELDS = ["cohort", "symbol", "variant", "trades", "wins", "sumR_gross",
                "sum_cost", "sumR_net", "win_R", "loss_R", "sum_hold",
                "n_long", "wins_long", "sumR_net_long", "win_R_long", "loss_R_long",
                "n_short", "wins_short", "sumR_net_short", "win_R_short", "loss_R_short"]


def _sma(x, w):
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.full(len(x), np.nan)
    if len(x) >= w:
        out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def load_nifty_returns(con):
    rows = con.execute("SELECT date,close FROM market_data_unified "
                       "WHERE timeframe='day' AND symbol='NIFTY50' ORDER BY date"
                       ).fetchall()
    dates = [r[0][:10] for r in rows]
    c = np.array([r[1] for r in rows], float)
    ret = np.full(len(c), np.nan)
    ret[1:] = c[1:] / c[:-1] - 1.0
    return {d: ret[i] for i, d in enumerate(dates)}


def compute_beta(D, nifty_ret):
    c = D["c"]
    sret = np.full(len(c), np.nan)
    sret[1:] = c[1:] / c[:-1] - 1.0
    xs, ys = [], []
    for i, d in enumerate(D["dates"]):
        nr = nifty_ret.get(d)
        if nr is not None and not np.isnan(sret[i]) and not np.isnan(nr):
            xs.append(nr)
            ys.append(sret[i])
    if len(xs) < 100:
        return 0.0
    xs, ys = np.array(xs), np.array(ys)
    var = np.var(xs)
    return float(np.cov(ys, xs)[0, 1] / var) if var > 0 else 0.0


def cpr_width_series(D):
    """width fraction per daily bar (CPR derived from that day -> applies next day)."""
    h, l, c = D["h"], D["l"], D["c"]
    P = (h + l + c) / 3.0
    bc = (h + l) / 2.0
    tc = 2 * P - bc
    return np.abs(tc - bc) / np.where(c > 0, c, np.nan)


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
    side = "long" if direction > 0 else "short"
    tv[f"n_{side}"] += 1
    tv[f"sumR_net_{side}"] += r_net
    tv[f"wins_{side}"] += 1 if r_net > 0 else 0
    if r_net >= 0:
        tv[f"win_R_{side}"] += r_net
    else:
        tv[f"loss_R_{side}"] += r_net


def run_symbol(con, symbol, nifty_ret):
    D = load_daily(con, symbol)
    if D is None:
        return None
    by_day = load_5min_by_day(con, symbol)
    if not by_day:
        return None
    sma50 = _sma(D["c"], 50)
    cprw = cpr_width_series(D)
    beta = compute_beta(D, nifty_ret)
    highbeta = beta >= BETA_HI
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
        direction = entry = ebar_lo = ebar_hi = ebar_idx = None
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
        post = bars[ebar_idx + 1:]
        d5h = np.array([b[2] for b in post], float)
        d5l = np.array([b[3] for b in post], float)
        d5c = np.array([b[4] for b in post], float)
        daily_after = daily_tuples[i + 1:]
        comp_flags = compression_flags(D, i)

        # --- filter flags for this breakout ---
        # cprok: prev-day CPR width (applies to entry day) vs trailing 60d 75th pctile
        cprok = False
        if i - 1 >= 0 and not np.isnan(cprw[i - 1]):
            lo = max(0, i - 61)
            hist = cprw[lo:i - 1]
            hist = hist[~np.isnan(hist)]
            if len(hist) >= 20:
                cprok = cprw[i - 1] <= np.percentile(hist, CPR_PCTL)
        htf = False
        if not np.isnan(sma50[i - 1]):
            htf = (D["c"][i - 1] > sma50[i - 1]) if direction > 0 else (D["c"][i - 1] < sma50[i - 1])
        fl = {"cprok": cprok, "htf": htf, "highbeta": highbeta}

        for sl_def in CORE_SL:
            SL = sl_price(sl_def, direction, entry, pdl, pdh, prevrange, datr, ebar_lo, ebar_hi)
            R = abs(entry - SL)
            if R <= 0:
                continue
            cost_R = (COST_BPS / 1e4 * abs(entry)) / R
            for tg in CORE_TG:
                out = simulate(direction, entry, SL, tg, prevrange, d5h, d5l, d5c,
                               daily_after, HOLD)
                if out is None:
                    continue
                r_gross, hold_days = out
                for comp in CORE_COMP:
                    if not comp_flags[comp]:
                        continue
                    for fn, reqs in FILTERS:
                        if reqs and not all(fl[k] for k in reqs):
                            continue
                        _add(tally[f"{comp}|{sl_def}|{tg}|{fn}"],
                             r_gross, cost_R, hold_days, direction)
    return tally


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
        print("no p2 stats")
        return
    agg = {"stocks": defaultdict(lambda: defaultdict(float)),
           "nifty50": defaultdict(lambda: defaultdict(float))}
    keys = [k for k in STATS_FIELDS if k not in ("cohort", "symbol", "variant")]
    with open(STATS_CSV) as f:
        for row in csv.DictReader(f):
            a = agg[row["cohort"]][row["variant"]]
            for k in keys:
                a[k] += float(row[k])
    fields = ["variant", "comp", "sl", "tgt", "filters", "trades", "win_pct",
              "net_exp_R", "net_PF", "gross_exp_R", "net_total_R", "avg_hold_d",
              "n_long", "long_win_pct", "long_exp_R", "long_PF",
              "n_short", "short_win_pct", "short_exp_R", "short_PF"]
    for cohort, by_variant in agg.items():
        rows = []
        for variant, a in by_variant.items():
            tr = a["trades"]
            if tr == 0:
                continue
            comp, sl, tg, fn = variant.split("|")
            nl, ns = a["n_long"], a["n_short"]
            rows.append(dict(
                variant=variant, comp=comp, sl=sl, tgt=tg, filters=fn, trades=int(tr),
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
        out = RESULTS / f"ranking_p2_{cohort}.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        elig = [r for r in rows if r["trades"] >= 300]
        print(f"== {cohort}: {len(rows)} variants -> {out.name} (top net exp, >=300 trades) ==")
        for r in elig[:12]:
            print(f"   {r['variant']:26s} n={r['trades']:6d} wr={r['win_pct']:4.1f}% "
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
    nifty_ret = load_nifty_returns(con)
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
        tally = run_symbol(con, sym, nifty_ret)
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
