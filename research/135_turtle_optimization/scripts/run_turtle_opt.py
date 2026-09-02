"""research/135 - staged Turtle optimization sweep.

Stages (gates: do not spend the next stage's compute until this one passes):
  A  channel (n_in x n_out) x stop_mult            44 cells   IS 2005-2017
  B  pyramiding: max_units x add_step               7 cells   IS
  C  sizing / cap / gate ablation                 ~10 cells   IS
  D  finalist + incumbent + benchmark, OOS 2024+    once

Append-only CSVs, completed labels skipped -> re-launch is always safe.
"""
from __future__ import annotations

import csv
import os
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

from engine import loader, metrics                      # noqa: E402
from turtle_core import turtle_positions, book_nav      # noqa: E402

RESULTS = HERE.parent / "results"
SPLITS = {
    "IS": ("2005-01-01", "2017-12-31"),
    "VAL": ("2018-01-01", "2023-12-31"),
    "OOS": ("2024-01-01", "2026-08-29"),
}
FIELDS = ["stage", "label", "split", "n_in", "n_out", "stop_mult", "max_units",
          "add_step", "sizing", "cap", "gate", "cagr", "sharpe", "sortino",
          "max_dd", "calmar", "dd_days", "cagr_gross", "n_pos", "n_units"]

_BARS = {}
_CLOSES = {}
_GATE = None
_CAL = {}
_POSCACHE = {}


def setup():
    global _GATE
    from services.data_manager import FNO_LOT_SIZES
    t0 = time.time()
    for sym in sorted(FNO_LOT_SIZES.keys()):
        df = loader.load_bars(sym, "day", start="2004-01-01", end="2026-08-29")
        if len(df) >= 300 and len(loader.ca_gap_flags(df)) <= 5:
            _BARS[sym] = df
            _CLOSES[sym] = df["close"]
    print(f"universe: {len(_BARS)} symbols  ({time.time()-t0:.0f}s)", flush=True)

    nb = loader.load_bars("NIFTYBEES", "day", start="2003-01-01", end="2026-08-29")
    _GATE = (nb["close"].shift(1) > nb["close"].rolling(200).mean().shift(1))

    allday = set()
    for c in _CLOSES.values():
        allday |= set(c.index)
    allday = pd.DatetimeIndex(sorted(allday))
    for k, (s, e) in SPLITS.items():
        _CAL[k] = allday[(allday >= pd.Timestamp(s)) & (allday <= pd.Timestamp(e))]
        print(f"  {k}: {len(_CAL[k])} sessions {_CAL[k][0].date()}..{_CAL[k][-1].date()}",
              flush=True)


def get_positions(n_in, n_out, stop_mult, max_units, add_step):
    key = (n_in, n_out, stop_mult, max_units, add_step)
    if key in _POSCACHE:
        return _POSCACHE[key]
    pos = []
    for sym, df in _BARS.items():
        pos.extend(turtle_positions(df, sym, n_in, n_out, stop_mult,
                                    max_units, add_step))
    _POSCACHE[key] = pos
    if len(_POSCACHE) > 8:
        _POSCACHE.pop(next(iter(_POSCACHE)))
    return pos


def run_cell(stage, label, split="IS", n_in=20, n_out=10, stop_mult=2.0,
             max_units=1, add_step=0.5, sizing="EQ", cap=12, gate_on=True,
             unit_frac=0.10, risk_pct=0.01, put_payoff=None, put_cost_bps=0.0,
             gate_override=None, positions=None, size_stop_mult=None):
    pos = positions if positions is not None else \
        get_positions(n_in, n_out, stop_mult, max_units, add_step)
    cal = _CAL[split]
    g = gate_override if gate_override is not None else _GATE
    gate = {d: True for d in cal} if not gate_on else g.to_dict()
    # the position's stop and the N-sizing risk distance are separate things:
    # a no-stop book can still be vol-sized off a notional 2N distance
    risk_dist = size_stop_mult if size_stop_mult is not None else stop_mult
    kw = dict(closes=_CLOSES, gate=gate, cal=cal, cap=cap, sizing=sizing,
              unit_frac=unit_frac, risk_pct=risk_pct, stop_mult=risk_dist,
              put_payoff=put_payoff, put_cost_bps=put_cost_bps)
    eq_net = book_nav(pos, costs_on=True, **kw)
    eq_gr = book_nav(pos, costs_on=False, **kw)
    cs = metrics.curve_stats(eq_net)
    gs = metrics.curve_stats(eq_gr)
    eq_net.to_csv(RESULTS / f"nav_{stage}_{label}_{split}.csv")
    lo, hi = pd.Timestamp(SPLITS[split][0]), pd.Timestamp(SPLITS[split][1])
    npos = sum(1 for p in pos if lo <= p["units"][0]["day"] <= hi)
    nun = sum(len(p["units"]) for p in pos if lo <= p["units"][0]["day"] <= hi)
    return {"stage": stage, "label": label, "split": split, "n_in": n_in,
            "n_out": n_out, "stop_mult": stop_mult, "max_units": max_units,
            "add_step": add_step, "sizing": sizing, "cap": cap,
            "gate": int(gate_on), "cagr": round(cs["cagr"] * 100, 2),
            "sharpe": round(cs["sharpe"], 3), "sortino": round(cs["sortino"], 3),
            "max_dd": round(cs["max_dd"] * 100, 2),
            "calmar": round(cs["calmar"], 3), "dd_days": cs["dd_duration_days"],
            "cagr_gross": round(gs["cagr"] * 100, 2), "n_pos": npos, "n_units": nun}


def writer(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            done = {(r["label"], r["split"]) for r in csv.DictReader(f)}
    else:
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    def emit(row):
        with open(path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
    return done, emit


def stage_a():
    path = RESULTS / "stage_A_channel_stop.csv"
    done, emit = writer(path)
    grid = []
    for n_in in (20, 40, 55, 80):
        for n_out in (10, 20, 40):
            if n_out > n_in:
                continue
            for sm in (None, 1.5, 2.0, 3.0):
                grid.append((n_in, n_out, sm))
    print(f"\n=== STAGE A: {len(grid)} cells (IS 2005-2017) ===", flush=True)
    for k, (n_in, n_out, sm) in enumerate(grid, 1):
        label = f"C{n_in}_{n_out}_S{'none' if sm is None else sm}"
        if (label, "IS") in done:
            continue
        t0 = time.time()
        row = run_cell("A", label, "IS", n_in=n_in, n_out=n_out, stop_mult=sm)
        emit(row)
        print(f"[A {k}/{len(grid)}] {label:18s} {time.time()-t0:5.0f}s  "
              f"CAGR {row['cagr']:6.2f}%  DD {row['max_dd']:7.2f}%  "
              f"Cal {row['calmar']:5.2f}  Sh {row['sharpe']:5.2f}  n={row['n_pos']}",
              flush=True)


def stage_b():
    """Rule 4 - pyramiding. The one mechanic of the attached spec r/83 never
    tested. Carried on TWO bases (no-stop winner + faithful 2N) so the answer
    is not conditional on Stage A's stop verdict."""
    path = RESULTS / "stage_B_pyramid.csv"
    done, emit = writer(path)
    grid = []
    for (n_in, n_out) in ((20, 10), (40, 10)):
        for sm in (None, 2.0):
            grid.append((n_in, n_out, sm, 1, 0.5))
            for mu in (2, 3, 4):
                for step in (0.5, 1.0):
                    grid.append((n_in, n_out, sm, mu, step))
    # robustness: does the pyramid benefit survive in OTHER channels, or is it
    # a 20/10 artefact? (40/10 degraded -> must not trust a single channel)
    for (n_in, n_out) in ((20, 20), (55, 10), (55, 20), (80, 20)):
        for mu in (1, 2, 4):
            grid.append((n_in, n_out, None, mu, 0.5))
    print(f"\n=== STAGE B: {len(grid)} cells (pyramiding, IS) ===", flush=True)
    for k, (n_in, n_out, sm, mu, step) in enumerate(grid, 1):
        label = f"C{n_in}_{n_out}_S{'none' if sm is None else sm}_U{mu}_A{step}"
        if (label, "IS") in done:
            continue
        row = run_cell("B", label, "IS", n_in=n_in, n_out=n_out, stop_mult=sm,
                       max_units=mu, add_step=step)
        emit(row)
        print(f"[B {k}/{len(grid)}] {label:26s} CAGR {row['cagr']:6.2f}%  "
              f"DD {row['max_dd']:7.2f}%  Cal {row['calmar']:5.2f}  "
              f"Sh {row['sharpe']:5.2f}  units={row['n_units']}", flush=True)


def _best(csv_path, split="IS"):
    d = pd.read_csv(csv_path)
    d = d[d.split == split]
    return d.loc[d.calmar.idxmax()].to_dict()


def stage_c():
    """Book construction ablation on the Stage-B winner: sizing (Rule 2),
    position cap, gate on/off, unit fraction."""
    b = _best(RESULTS / "stage_B_pyramid.csv")
    sm = None if pd.isna(b["stop_mult"]) else float(b["stop_mult"])
    base = dict(n_in=int(b["n_in"]), n_out=int(b["n_out"]), stop_mult=sm,
                max_units=int(b["max_units"]), add_step=float(b["add_step"]))
    print(f"\n=== STAGE C: base={base} (from B winner {b['label']}) ===", flush=True)
    path = RESULTS / "stage_C_book.csv"
    done, emit = writer(path)

    cells = [("EQ_cap12_gate", dict(sizing="EQ", cap=12, gate_on=True))]
    for cap in (8, 20):
        cells.append((f"EQ_cap{cap}_gate", dict(sizing="EQ", cap=cap, gate_on=True)))
    for uf in (0.06, 0.15):
        cells.append((f"EQ_uf{uf}_gate", dict(sizing="EQ", cap=12, gate_on=True,
                                              unit_frac=uf)))
    cells.append(("EQ_cap12_nogate", dict(sizing="EQ", cap=12, gate_on=False)))
    for rp in (0.005, 0.01, 0.02):
        smx = sm if sm else 2.0
        cells.append((f"Nrisk{rp}_cap12_gate",
                      dict(sizing="N", cap=12, gate_on=True, risk_pct=rp,
                           stop_mult_override=smx)))
    for label, kw in cells:
        if (label, "IS") in done:
            continue
        kw = dict(kw)
        smo = kw.pop("stop_mult_override", None)
        if smo is not None:
            kw["size_stop_mult"] = smo
        row = run_cell("C", label, "IS", **base, **kw)
        emit(row)
        print(f"[C] {label:22s} CAGR {row['cagr']:6.2f}%  DD {row['max_dd']:7.2f}%  "
              f"Cal {row['calmar']:5.2f}  Sh {row['sharpe']:5.2f}", flush=True)


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    setup()
    stage_a()
    print("\nSTAGE A COMPLETE", flush=True)
    stage_b()
    print("\nSTAGE B COMPLETE", flush=True)
    stage_c()
    print("\nSTAGE C COMPLETE", flush=True)


if __name__ == "__main__":
    main()
