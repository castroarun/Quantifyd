"""research/135 Stage G - CORRECTION to Stage F's momentum arm.

Stage F compared the Turtle books against a hand-rolled reconstruction of "the
momentum book's rules" and got 12.6% CAGR - essentially benchmark - which is
wildly below research/75's published 31.9%. Arun flagged it. Three defects:

  1. WRONG UNIVERSE. Stage F ranked momentum inside the 78 F&O large caps.
     The book's real universe is a point-in-time top-250 by traded value
     (Nifty LargeMidcap 250 proxy). Cross-sectional momentum lives in the
     mid/small tail; restricting it to 78 large caps removes the edge.
  2. WRONG RULES - it conflated research/75's backtest with the LIVE paper
     book's extra machinery, bolting a daily 15-day Donchian stop onto a
     monthly rebalance book that has no such stop.
  3. IDLE-CASH BUG. Names stopped out mid-month, and everything sold at a
     risk-off gate, could not be re-bought until the next month-end - so the
     book sat in cash through recoveries.

This stage drives research/75's OWN runner instead of re-implementing it, and
separates the two questions Stage F confounded:

  G1  reproduce r/75 A1 on its own universe   -> must land near 31.9% CAGR
  G2  the SAME rules restricted to the 78 F&O names -> the like-for-like number
  G3  Turtle-OPT lifted onto the broad top-250 universe -> does the Turtle
      benefit from the wider universe too, or is the gap really about rules?

G1 vs G2 measures the UNIVERSE. G2 vs Turtle-on-78 measures the RULES.
G3 is the fair fight: both systems on the universe each would actually trade.
"""
from __future__ import annotations

import csv
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R81 = HERE.parents[1] / "81_swing_edge_discovery"
for p in (str(HERE), str(R81), str(R81.parents[1])):
    if p not in sys.path:
        sys.path.insert(0, p)

from engine import metrics                               # noqa: E402
from turtle_core import turtle_positions, book_nav       # noqa: E402

RESULTS = HERE.parent / "results"
R75_PATH = Path("/home/arun/quantifyd/research/75_nifty250_momentum_top15/"
                "scripts/run_nifty250_momentum.py")
START = pd.Timestamp("2006-01-01")
END = pd.Timestamp("2026-08-29")


def load_r75():
    spec = importlib.util.spec_from_file_location("r75", str(R75_PATH))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def curve_from_run(res):
    """r/75's run() returns its _stats dict, which carries the daily NAV under
    'nav' (plus its own cagr/dd/calmar, which we print to cross-check ours)."""
    if isinstance(res, dict):
        return res["nav"], res
    if isinstance(res, pd.Series):
        return res, None
    return pd.Series([v for _, v in res],
                     index=pd.DatetimeIndex([d for d, _ in res])), None


def show(label, eq, out_rows):
    cs = metrics.curve_stats(eq.dropna())
    row = {"arm": label, "cagr": round(cs["cagr"] * 100, 2),
           "sharpe": round(cs["sharpe"], 3), "sortino": round(cs["sortino"], 3),
           "max_dd": round(cs["max_dd"] * 100, 2),
           "calmar": round(cs["calmar"], 3),
           "mult": round(float(eq.dropna().iloc[-1] / eq.dropna().iloc[0]), 1),
           "years": round(cs["years"], 2)}
    out_rows.append(row)
    print(f"  {label:34s} CAGR {row['cagr']:6.2f}%  DD {row['max_dd']:7.2f}%  "
          f"Cal {row['calmar']:5.2f}  Sh {row['sharpe']:5.2f}  {row['mult']:7.1f}x",
          flush=True)
    return row


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    m = load_r75()
    print("loading r/75 price panel ...", flush=True)
    close, tv = m.load()
    print(f"  {close.shape[1]} symbols {close.index.min().date()}..{close.index.max().date()}",
          flush=True)
    ema = {p: close.ewm(span=p, adjust=False, min_periods=p).mean()
           for p in (50, 100, 200)}
    nbx_ema100 = close[m.BENCH].ffill().ewm(span=100, adjust=False,
                                            min_periods=100).mean()

    # ---- the 78-name F&O universe Stage F used --------------------------
    from services.data_manager import FNO_LOT_SIZES
    from engine import loader as eloader
    fno = set()
    for sym in sorted(FNO_LOT_SIZES.keys()):
        df = eloader.load_bars(sym, "day", start="2004-01-01", end="2026-08-29")
        if len(df) >= 300 and len(eloader.ca_gap_flags(df)) <= 5:
            fno.add(sym)
    print(f"  F&O universe: {len(fno)} names", flush=True)

    orig_pit = m.rs2.pit_universe

    print("\n=== G1: reproduce r/75 A1 BASE on its own top-250 universe ===")
    r = m.run(close, tv, ema, nbx_ema100, score="ret252", N=15, use_stack=True,
              use_gate=True, rt=m.RT, stcg=0.0, start=START, gate_freq="monthly")
    eq_g1, st1 = curve_from_run(r)
    show("MOM r/75 A1  · top-250", eq_g1, rows)
    print(f"   r/75 own stats: CAGR {st1['cagr']:.1f}%  DD {st1['dd']:.1f}%  "
          f"Cal {st1['calmar']:.2f}  {st1['mult']:.0f}x")
    print("   (published r/75 A1: 31.9% CAGR, -31.6% DD, Calmar 1.01, 292x)")

    print("\n=== G2: SAME rules, universe restricted to the 78 F&O names ===")
    m.rs2.pit_universe = lambda tv_, c_, d_, b_: orig_pit(tv_, c_, d_, b_) & fno
    r = m.run(close, tv, ema, nbx_ema100, score="ret252", N=15, use_stack=True,
              use_gate=True, rt=m.RT, stcg=0.0, start=START, gate_freq="monthly")
    eq_g2, _ = curve_from_run(r)
    show("MOM r/75 A1  · F&O-78 only", eq_g2, rows)
    m.rs2.pit_universe = orig_pit

    # benchmark on the same window
    bn = close[m.BENCH].loc[close.index >= START].dropna()
    show("NIFTYBEES buy & hold", bn / bn.iloc[0], rows)

    # ---- G3: Turtle on the BROAD universe -------------------------------
    print("\n=== G3: Turtle-OPT (20/10, no stop, EQ) on the broad top-250 ===")
    # union of PIT membership at each month-end, so we only pull what we need
    idx = close.index[(close.index >= START) & (close.index <= END)]
    _s = pd.Series(idx, index=idx)
    me = sorted(set(_s.groupby([idx.year, idx.month]).max().values))
    pit_by_date = {}
    union = set()
    for d in me:
        u = orig_pit(tv, close, pd.Timestamp(d), "n250") - set(m.EXCLUDE) - {m.BENCH}
        pit_by_date[pd.Timestamp(d)] = u
        union |= u
    print(f"  PIT union over {len(me)} month-ends: {len(union)} distinct names",
          flush=True)

    t0 = time.time()
    bars, closes = {}, {}
    for i, sym in enumerate(sorted(union)):
        try:
            df = eloader.load_bars(sym, "day", start="2004-01-01", end="2026-08-29")
        except Exception:
            continue
        if len(df) < 300 or len(eloader.ca_gap_flags(df)) > 5:
            continue
        bars[sym] = df
        closes[sym] = df["close"]
    print(f"  OHLC loaded for {len(bars)} names ({time.time()-t0:.0f}s)", flush=True)

    # a position may only be OPENED if its symbol was in the PIT universe at the
    # most recent month-end on or before the entry date
    me_idx = pd.DatetimeIndex(sorted(pit_by_date))

    def eligible(sym, day):
        pos = me_idx.searchsorted(day, side="right") - 1
        if pos < 0:
            return False
        return sym in pit_by_date[me_idx[pos]]

    positions = []
    for sym, df in bars.items():
        for p in turtle_positions(df, sym, 20, 10, None, 1, 0.5):
            if eligible(sym, p["units"][0]["day"]):
                positions.append(p)
    print(f"  {len(positions)} PIT-eligible turtle positions", flush=True)

    nb_raw = close[m.BENCH].ffill()
    gate200 = (nb_raw.shift(1) > nb_raw.rolling(200).mean().shift(1)).to_dict()
    cal = idx
    eq_g3 = book_nav(positions, closes, gate200, cal, cap=12, sizing="EQ",
                     unit_frac=0.10, stop_mult=None, costs_on=True)
    show("TURTLE-OPT · top-250", eq_g3, rows)

    # and the same Turtle on the 78-name universe, same 2006 window, for the
    # apples-to-apples row
    pos78 = [p for sym in fno if sym in bars
             for p in turtle_positions(bars[sym], sym, 20, 10, None, 1, 0.5)]
    eq_g4 = book_nav(pos78, closes, gate200, cal, cap=12, sizing="EQ",
                     unit_frac=0.10, stop_mult=None, costs_on=True)
    show("TURTLE-OPT · F&O-78 only", eq_g4, rows)

    pd.DataFrame(rows).to_csv(RESULTS / "stage_G_universe.csv", index=False)
    curves = pd.DataFrame({"MOM_250": eq_g1, "MOM_FNO78": eq_g2,
                           "TURTLE_250": eq_g3, "TURTLE_FNO78": eq_g4,
                           "BENCH": bn / bn.iloc[0]})
    curves.to_csv(RESULTS / "stage_G_curves.csv")

    print("\n=== per-year (%) ===")
    per = {}
    for k in curves.columns:
        per[k] = (metrics.per_year_table(curves[k].dropna().pct_change())["return"]
                  * 100).round(1)
    pt = pd.DataFrame(per)
    print(pt.to_string())
    pt.to_csv(RESULTS / "stage_G_per_year.csv")
    for name, yrs in (("2006-2017", range(2006, 2018)),
                      ("2018-2023", range(2018, 2024)),
                      ("2024-2026", range(2024, 2027))):
        sel = pt.loc[[y for y in yrs if y in pt.index]]
        print(f"  {name}: " + "  ".join(f"{a}={sel[a].mean():6.1f}%" for a in pt.columns))

    print("\nSTAGE G COMPLETE", flush=True)


if __name__ == "__main__":
    main()
