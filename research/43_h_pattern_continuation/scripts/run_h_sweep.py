"""H-Pattern continuation sweep driver.

Per symbol: load 5-min bars (session 09:15-15:25), Wilder-14 ATR on the
continuous series, then per trading day detect H setups and simulate all 15
entry x target variants in BOTH directions (bearish on raw arrays, bullish on
negated arrays). Per-symbol x variant tallies are appended to a CSV immediately
so the run is crash-safe and resumable (skips symbols already present).

Usage:
  python run_h_sweep.py                 # full run (stocks + NIFTY50)
  python run_h_sweep.py --aggregate-only  # rebuild ranking_*.csv from tallies
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
from h_pattern import process_day, VARIANTS, ENTRY_STYLES, TARGETS  # noqa: E402

DB = ROOT / "backtest_data" / "market_data.db"
RESULTS = HERE.parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
STATS_CSV = RESULTS / "symbol_variant_stats.csv"
SAMPLE_CSV = RESULTS / "h_patterns_sample.csv"

INDEX_SYMBOLS = {"NIFTY50"}        # its own cohort
EXCLUDE = {"BANKNIFTY"}            # index, not requested
SESS_LO, SESS_HI = "09:15", "15:25"

STATS_FIELDS = ["cohort", "symbol", "variant", "trades", "wins_net",
                "sumR_gross", "sum_cost", "sumR_net", "sum_win_R_net",
                "sum_loss_R_net", "n_long", "n_short", "sumR_net_long",
                "sumR_net_short"]


def wilder_atr(high, low, close, period=14):
    n = len(close)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    atr = np.empty(n)
    atr[:period] = np.nan
    if n > period:
        atr[period] = tr[1:period + 1].mean()
        for i in range(period + 1, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def load_symbol(con, symbol):
    rows = con.execute(
        "SELECT date,open,high,low,close FROM market_data_unified "
        "WHERE timeframe='5minute' AND symbol=? ORDER BY date", (symbol,)
    ).fetchall()
    if not rows:
        return None
    dates = [r[0] for r in rows]
    o = np.array([r[1] for r in rows], float)
    h = np.array([r[2] for r in rows], float)
    l = np.array([r[3] for r in rows], float)
    c = np.array([r[4] for r in rows], float)
    # session filter (drop the spurious 18:00+ bars)
    t = np.array([d[11:16] for d in dates])
    keep = (t >= SESS_LO) & (t <= SESS_HI)
    dates = [d for d, k in zip(dates, keep) if k]
    o, h, l, c = o[keep], h[keep], l[keep], c[keep]
    if len(c) < 100:
        return None
    atr = wilder_atr(h, l, c)
    day = np.array([d[:10] for d in dates])
    return dates, o, h, l, c, atr, day


def new_tally():
    return {v: dict(trades=0, wins_net=0, sumR_gross=0.0, sum_cost=0.0,
                    sumR_net=0.0, sum_win_R_net=0.0, sum_loss_R_net=0.0,
                    n_long=0, n_short=0, sumR_net_long=0.0, sumR_net_short=0.0)
            for v in VARIANTS}


def run_symbol(con, symbol, cohort, sample_writer, sample_left):
    data = load_symbol(con, symbol)
    if data is None:
        return None, sample_left
    _dates, _o, h, l, c, atr, day = data
    tally = new_tally()
    # iterate per trading day
    uniq = np.unique(day)
    for d in uniq:
        m = day == d
        if m.sum() < 20:
            continue
        hh, ll, cc, aa = h[m], l[m], c[m], atr[m]
        if np.isnan(aa).all():
            continue
        aa = np.nan_to_num(aa, nan=0.0)
        # bearish-H (short) on raw; bullish-H (long) on negated arrays
        for direction, (H, L, C) in ((-1, (hh, ll, cc)), (+1, (-ll, -hh, -cc))):
            for (variant, r_gross, cost_R, dirn, entry_orig) in \
                    process_day(H, L, C, aa, direction):
                r_net = r_gross - cost_R
                tv = tally[variant]
                tv["trades"] += 1
                tv["sumR_gross"] += r_gross
                tv["sum_cost"] += cost_R
                tv["sumR_net"] += r_net
                tv["wins_net"] += 1 if r_net > 0 else 0
                if r_net >= 0:
                    tv["sum_win_R_net"] += r_net
                else:
                    tv["sum_loss_R_net"] += r_net
                if dirn < 0:
                    tv["n_short"] += 1
                    tv["sumR_net_short"] += r_net
                else:
                    tv["n_long"] += 1
                    tv["sumR_net_long"] += r_net
                if sample_left[0] > 0 and variant == "breakout|2R":
                    sample_writer.writerow([symbol, d, "short" if direction < 0
                                            else "long", round(entry_orig, 2),
                                            round(r_gross, 3), round(cost_R, 3)])
                    sample_left[0] -= 1
    return tally, sample_left


def cohort_of(symbol):
    if symbol in INDEX_SYMBOLS:
        return "nifty50"
    if symbol in EXCLUDE:
        return None
    return "stocks"


def all_symbols(con):
    rows = con.execute(
        "SELECT DISTINCT symbol FROM market_data_unified WHERE timeframe='5minute' "
        "ORDER BY symbol").fetchall()
    return [r[0] for r in rows]


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
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in tv.items()}))


def aggregate():
    """Sum per-symbol tallies into per-variant rankings, one CSV per cohort."""
    if not STATS_CSV.exists():
        print("no stats yet")
        return
    agg = {"stocks": defaultdict(lambda: defaultdict(float)),
           "nifty50": defaultdict(lambda: defaultdict(float))}
    keys = ("trades", "wins_net", "sumR_gross", "sum_cost", "sumR_net",
            "sum_win_R_net", "sum_loss_R_net", "n_long", "n_short",
            "sumR_net_long", "sumR_net_short")
    with open(STATS_CSV) as f:
        for row in csv.DictReader(f):
            a = agg[row["cohort"]][row["variant"]]
            for k in keys:
                a[k] += float(row[k])
    fields = ["variant", "entry", "target", "trades", "win_rate_net_pct",
              "gross_exp_R", "net_exp_R", "avg_cost_R", "net_total_R",
              "net_profit_factor", "avg_win_R", "avg_loss_R", "n_long",
              "n_short", "net_exp_long_R", "net_exp_short_R"]
    for cohort, by_variant in agg.items():
        rows = []
        for variant, a in by_variant.items():
            tr = a["trades"]
            if tr == 0:
                continue
            entry, target = variant.split("|")
            wins = a["wins_net"]
            losses = tr - wins
            sl = a["sum_loss_R_net"]
            pf = (a["sum_win_R_net"] / abs(sl)) if sl < 0 else float("inf")
            rows.append(dict(
                variant=variant, entry=entry, target=target, trades=int(tr),
                win_rate_net_pct=round(100 * wins / tr, 1),
                gross_exp_R=round(a["sumR_gross"] / tr, 4),
                net_exp_R=round(a["sumR_net"] / tr, 4),
                avg_cost_R=round(a["sum_cost"] / tr, 4),
                net_total_R=round(a["sumR_net"], 1),
                net_profit_factor=round(pf, 3) if pf != float("inf") else "inf",
                avg_win_R=round(a["sum_win_R_net"] / wins, 3) if wins else 0,
                avg_loss_R=round(sl / losses, 3) if losses else 0,
                n_long=int(a["n_long"]), n_short=int(a["n_short"]),
                net_exp_long_R=round(a["sumR_net_long"] / a["n_long"], 4) if a["n_long"] else 0,
                net_exp_short_R=round(a["sumR_net_short"] / a["n_short"], 4) if a["n_short"] else 0,
            ))
        rows.sort(key=lambda r: r["gross_exp_R"], reverse=True)
        out = RESULTS / f"ranking_{cohort}.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"== {cohort} ({sum(r['trades'] for r in rows)} trades) -> {out.name}")
        for r in rows[:6]:
            print(f"   {r['variant']:16s} n={r['trades']:6d} wr={r['win_rate_net_pct']:5.1f}% "
                  f"GROSS={r['gross_exp_R']:+.3f}R NET={r['net_exp_R']:+.3f}R "
                  f"cost={r['avg_cost_R']:.3f} PF={r['net_profit_factor']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="process only N new symbols (smoke test)")
    ap.add_argument("--only", type=str, default="", help="comma-list of symbols (smoke test)")
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
    print(f"{len(syms)} symbols total | {len(done)} done | {len(todo)} to process")

    sample_writer = None
    sample_left = [0]
    if not SAMPLE_CSV.exists():
        sf = open(SAMPLE_CSV, "w", newline="")
        sample_writer = csv.writer(sf)
        sample_writer.writerow(["symbol", "date", "direction", "entry_px",
                                 "gross_R_breakout2R", "cost_R"])
        sample_left = [60]
    else:
        sf = open(SAMPLE_CSV, "a", newline="")
        sample_writer = csv.writer(sf)

    t0 = time.time()
    for i, sym in enumerate(todo, 1):
        cohort = cohort_of(sym)
        st = time.time()
        tally, sample_left = run_symbol(con, sym, cohort, sample_writer, sample_left)
        if tally is None:
            print(f"[{i}/{len(todo)}] {sym} ({cohort}) no data", flush=True)
            continue
        append_stats(sym, cohort, tally)
        tt = sum(tv["trades"] for tv in tally.values())
        print(f"[{i}/{len(todo)}] {sym:12s} ({cohort:7s}) {tt:5d} trade-rows "
              f"{time.time()-st:4.1f}s  elapsed={time.time()-t0:6.0f}s", flush=True)
    sf.close()
    print(f"done {len(todo)} symbols in {time.time()-t0:.0f}s")
    aggregate()


if __name__ == "__main__":
    main()
