"""H-Pattern Phase 2 — confluence-filter gating on the P1 winner.

Carries forward BREAKOUT entry + MM/3R targets (both directions), and routes each
filled breakout trade into every filter-combo it satisfies. Filters use the
PREVIOUS session's daily data (no look-ahead):

  pdl     breakout level breaks prev-day low (short) / high (long)
  cpr_w   prev-day CPR narrow (width/price < CPR_W_THRESH) -> trend day
  cpr_pos entry below CPR bottom (short) / above CPR top (long)
  trend   daily close vs SMA200 agrees with the trade direction

One breakout|target simulation per setup; the trade is added to all combos whose
flags it meets, so 20 variants are tallied in a single pass. Gross/net kept
separate (same cost model as P1). Crash-safe: per-symbol tallies appended to a
resume CSV; re-running skips done symbols.

Usage:
  python run_h_p2.py                 # full run
  python run_h_p2.py --aggregate-only
  python run_h_p2.py --only RELIANCE,TCS,NIFTY50   # smoke
"""
from __future__ import annotations

import argparse
import bisect
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
from h_pattern import find_setups, simulate, COST_BPS  # noqa: E402
# reuse the P1 loaders for parity
from run_h_sweep import (load_symbol, cohort_of, all_symbols,  # noqa: E402
                         wilder_atr, SESS_LO, SESS_HI)

DB = ROOT / "backtest_data" / "market_data.db"
RESULTS = HERE.parent / "results"
STATS_CSV = RESULTS / "symbol_variant_stats_p2.csv"

CPR_W_THRESH = 0.005          # narrow CPR = width < 0.5% of price
TARGETS_P2 = ("MM", "3R")
COMBOS = [
    ("none", ()),
    ("pdl", ("pdl",)),
    ("cpr_w", ("cpr_w",)),
    ("cpr_pos", ("cpr_pos",)),
    ("trend", ("trend",)),
    ("pdl+trend", ("pdl", "trend")),
    ("cpr_pos+trend", ("cpr_pos", "trend")),
    ("pdl+cpr_pos", ("pdl", "cpr_pos")),
    ("cpr_pos+cpr_w", ("cpr_pos", "cpr_w")),
    ("all", ("pdl", "cpr_pos", "trend")),
]
VARIANTS_P2 = [f"{t}|{name}" for t in TARGETS_P2 for (name, _) in COMBOS]

STATS_FIELDS = ["cohort", "symbol", "variant", "trades", "wins_net",
                "sumR_gross", "sum_cost", "sumR_net", "sum_win_R_net",
                "sum_loss_R_net", "n_long", "n_short", "sumR_net_long",
                "sumR_net_short", "wins_long", "wins_short",
                "win_R_long", "loss_R_long", "win_R_short", "loss_R_short"]


def _sma(x, w):
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.full(len(x), np.nan)
    if len(x) >= w:
        out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def load_daily_context(con, symbol):
    """Return (sorted daily dates, list of ctx) where ctx[i] describes the
    session that closed on dates[i] -- prev-day H/L/C, CPR, close, SMA200."""
    rows = con.execute(
        "SELECT date,high,low,close FROM market_data_unified "
        "WHERE timeframe='day' AND symbol=? ORDER BY date", (symbol,)).fetchall()
    if len(rows) < 50:
        return [], []
    dates = [r[0][:10] for r in rows]
    H = np.array([r[1] for r in rows], float)
    L = np.array([r[2] for r in rows], float)
    C = np.array([r[3] for r in rows], float)
    sma200 = _sma(C, 200)
    ctx = []
    for i in range(len(dates)):
        P = (H[i] + L[i] + C[i]) / 3.0
        bc = (H[i] + L[i]) / 2.0
        tc = 2 * P - bc
        ctx.append(dict(
            pdl=L[i], pdh=H[i],
            cpr_top=max(tc, bc), cpr_bot=min(tc, bc),
            cpr_w_frac=abs(tc - bc) / C[i] if C[i] else 9.9,
            close=C[i], sma200=sma200[i]))
    return dates, ctx


def prev_ctx(daily_dates, daily_ctx, day):
    """Context of the session strictly before `day` (the one whose levels we use)."""
    i = bisect.bisect_left(daily_dates, day) - 1
    if i < 0:
        return None
    c = daily_ctx[i]
    if np.isnan(c["sma200"]):
        # still usable for pdl/cpr; trend just can't pass
        return c
    return c


def trade_flags(direction, entry, ctx):
    """Boolean flags for one filled breakout trade (original price space)."""
    if ctx is None:
        return None
    short = direction < 0
    sma_ok = not np.isnan(ctx["sma200"])
    return {
        "pdl": (entry <= ctx["pdl"]) if short else (entry >= ctx["pdh"]),
        "cpr_w": ctx["cpr_w_frac"] < CPR_W_THRESH,
        "cpr_pos": (entry < ctx["cpr_bot"]) if short else (entry > ctx["cpr_top"]),
        "trend": sma_ok and ((ctx["close"] < ctx["sma200"]) if short
                             else (ctx["close"] > ctx["sma200"])),
    }


def new_tally():
    return {v: dict(trades=0, wins_net=0, sumR_gross=0.0, sum_cost=0.0,
                    sumR_net=0.0, sum_win_R_net=0.0, sum_loss_R_net=0.0,
                    n_long=0, n_short=0, sumR_net_long=0.0, sumR_net_short=0.0,
                    wins_long=0, wins_short=0, win_R_long=0.0, loss_R_long=0.0,
                    win_R_short=0.0, loss_R_short=0.0)
            for v in VARIANTS_P2}


def _add(tv, r_gross, cost_R, r_net, dirn):
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
        tv["wins_short"] += 1 if r_net > 0 else 0
        if r_net >= 0:
            tv["win_R_short"] += r_net
        else:
            tv["loss_R_short"] += r_net
    else:
        tv["n_long"] += 1
        tv["sumR_net_long"] += r_net
        tv["wins_long"] += 1 if r_net > 0 else 0
        if r_net >= 0:
            tv["win_R_long"] += r_net
        else:
            tv["loss_R_long"] += r_net


def run_symbol(con, symbol):
    data = load_symbol(con, symbol)
    if data is None:
        return None
    _dates, _o, h, l, c, atr, day = data
    daily_dates, daily_ctx = load_daily_context(con, symbol)
    tally = new_tally()
    for d in np.unique(day):
        m = day == d
        if m.sum() < 20:
            continue
        hh, ll, cc, aa = h[m], l[m], c[m], atr[m]
        if np.isnan(aa).all():
            continue
        aa = np.nan_to_num(aa, nan=0.0)
        ctx = prev_ctx(daily_dates, daily_ctx, d) if daily_dates else None
        for direction, (H, L, C) in ((-1, (hh, ll, cc)), (+1, (-ll, -hh, -cc))):
            n = len(C)
            setups = find_setups(H, L, C, aa, n)
            for s in setups:
                for tk in TARGETS_P2:
                    out = simulate(s, "breakout", tk, H, L, C, n)
                    if out is None:
                        continue
                    r_gross, entry, stop = out
                    R = stop - entry
                    cost_R = (COST_BPS / 1e4 * abs(entry)) / R if R > 0 else 0.0
                    r_net = r_gross - cost_R
                    entry_orig = entry if direction < 0 else -entry
                    fl = trade_flags(direction, entry_orig, ctx)
                    for name, reqs in COMBOS:
                        if reqs:
                            if fl is None or not all(fl[k] for k in reqs):
                                continue
                        _add(tally[f"{tk}|{name}"], r_gross, cost_R, r_net, direction)
    return tally


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


def done_symbols():
    if not STATS_CSV.exists():
        return set()
    with open(STATS_CSV) as f:
        return {row["symbol"] for row in csv.DictReader(f)}


def aggregate():
    if not STATS_CSV.exists():
        print("no p2 stats yet")
        return
    agg = {"stocks": defaultdict(lambda: defaultdict(float)),
           "nifty50": defaultdict(lambda: defaultdict(float))}
    keys = ("trades", "wins_net", "sumR_gross", "sum_cost", "sumR_net",
            "sum_win_R_net", "sum_loss_R_net", "n_long", "n_short",
            "sumR_net_long", "sumR_net_short", "wins_long", "wins_short",
            "win_R_long", "loss_R_long", "win_R_short", "loss_R_short")
    with open(STATS_CSV) as f:
        for row in csv.DictReader(f):
            a = agg[row["cohort"]][row["variant"]]
            for k in keys:
                a[k] += float(row[k])

    def pf(win_r, loss_r):
        return round(win_r / abs(loss_r), 3) if loss_r < 0 else "inf"

    fields = ["variant", "target", "filters", "trades", "win_rate_pct",
              "net_exp_R", "net_PF", "gross_exp_R", "avg_cost_R", "net_total_R",
              "n_long", "long_win_pct", "long_net_exp_R", "long_PF",
              "n_short", "short_win_pct", "short_net_exp_R", "short_PF"]
    for cohort, by_variant in agg.items():
        rows = []
        for variant, a in by_variant.items():
            tr = a["trades"]
            if tr == 0:
                continue
            target, filters = variant.split("|")
            nl, ns = a["n_long"], a["n_short"]
            rows.append(dict(
                variant=variant, target=target, filters=filters, trades=int(tr),
                win_rate_pct=round(100 * a["wins_net"] / tr, 1),
                net_exp_R=round(a["sumR_net"] / tr, 4),
                net_PF=pf(a["sum_win_R_net"], a["sum_loss_R_net"]),
                gross_exp_R=round(a["sumR_gross"] / tr, 4),
                avg_cost_R=round(a["sum_cost"] / tr, 4),
                net_total_R=round(a["sumR_net"], 1),
                n_long=int(nl),
                long_win_pct=round(100 * a["wins_long"] / nl, 1) if nl else 0,
                long_net_exp_R=round(a["sumR_net_long"] / nl, 4) if nl else 0,
                long_PF=pf(a["win_R_long"], a["loss_R_long"]),
                n_short=int(ns),
                short_win_pct=round(100 * a["wins_short"] / ns, 1) if ns else 0,
                short_net_exp_R=round(a["sumR_net_short"] / ns, 4) if ns else 0,
                short_PF=pf(a["win_R_short"], a["loss_R_short"]),
            ))
        rows.sort(key=lambda r: r["short_net_exp_R"], reverse=True)
        out = RESULTS / f"ranking_p2_{cohort}.csv"
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"== {cohort} -> {out.name} (sorted by short net exp)")
        print(f"   {'variant':18s}{'  LONG: n   win%   exp   PF':28s}{'  SHORT: n   win%   exp   PF':28s}")
        for r in rows[:8]:
            print(f"   {r['variant']:18s} "
                  f"L {r['n_long']:6d} {r['long_win_pct']:5.1f}% {r['long_net_exp_R']:+.3f} {r['long_PF']:>5} | "
                  f"S {r['n_short']:6d} {r['short_win_pct']:5.1f}% {r['short_net_exp_R']:+.3f} {r['short_PF']:>5}")


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
    for i, sym in enumerate(todo, 1):
        cohort = cohort_of(sym)
        st = time.time()
        tally = run_symbol(con, sym)
        if tally is None:
            print(f"[{i}/{len(todo)}] {sym} no data", flush=True)
            continue
        append_stats(sym, cohort, tally)
        tt = sum(tv["trades"] for tv in tally.values())
        print(f"[{i}/{len(todo)}] {sym:12s} ({cohort:7s}) {tt:6d} trade-rows "
              f"{time.time()-st:4.1f}s elapsed={time.time()-t0:6.0f}s", flush=True)
    print(f"done {len(todo)} symbols in {time.time()-t0:.0f}s")
    aggregate()


if __name__ == "__main__":
    main()
