"""research/153 — IPO-Base sweep runner. Incremental, resume-safe, seed-ensembled.

Phases
  g1a  signal geometry   : age x base-length x depth x RS-policy      (book held fixed)
  g1b  exits x book      : stop x trail x take-profit x slots/sizing  (top geometries)
  g2   mechanics         : fill, gate, cost ladder, sizing family, structure stop

Every cell is evaluated on TWO windows (W1 = the site's 2020-2025, W2 = 2006->now) with a
10-seed random-selection ensemble; medians and worst seed are recorded. Ranking metric
(pre-registered): median after-tax CAGR on W2, gated on positive per-trade expectancy net
of 25 bps/side in BOTH windows.
"""
from __future__ import annotations

import csv
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
sys.path.insert(0, str(HERE))
import ipo_replay as ir  # noqa: E402

WINDOWS = {"w1": ("2020-01-01", "2025-12-31"), "w2": ("2006-01-01", "2026-09-04")}
SEEDS = list(range(1, 11))

BASE = dict(max_age_m=12, min_bars=25, L=40, max_depth=0.35, rs_policy="off",
            rs_min=70.0, tight_max=None, pivot_mode="close",
            stop=0.08, trail=50, target=None, slots=8, size_pct=0.1875,
            risk_pct=None, stop_mode="pct", cost=0.0025, gate=False,
            fill_close=False)

FIELDS = (["label"] + list(BASE.keys()) + ["n_signals"] +
          [f"{w}_{m}" for w in WINDOWS for m in
           ("cagr", "cagr_lo", "cagr_hi", "cagr_worst", "dd", "dd_worst", "calmar",
            "n", "tpy", "win", "mean", "netexp", "avg_win", "avg_loss", "hold",
            "inv", "streak")] + ["secs"])


def run_cell(ctx, cfg, seeds=SEEDS, collect_curves=False):
    trig, piv, lo = ir.build_trigger(
        ctx, max_age_m=cfg["max_age_m"], min_bars=cfg["min_bars"], L=cfg["L"],
        max_depth=cfg["max_depth"], rs_policy=cfg["rs_policy"], rs_min=cfg["rs_min"],
        tight_max=cfg["tight_max"], pivot_mode=cfg["pivot_mode"])
    sma = ctx.sma(cfg["trail"])
    weak = ctx.WEAK if cfg["gate"] else ctx.NOWEAK
    out = {"n_signals": int(trig.sum())}
    curves = {}
    for wk, (a, b) in WINDOWS.items():
        days = np.array([i for i, d in enumerate(ctx.dates) if a <= str(d.date()) <= b])
        du = ctx.dates[days]
        rows = []
        for sd in seeds:
            eq, trd, _, inv = ir.simulate_ipo(
                sd, days, ctx.dates, ctx.C, ctx.O, piv, lo, sma, ctx.RSF, ctx.TVp,
                trig, weak, cost=cfg["cost"], stop=cfg["stop"], slots=cfg["slots"],
                size_pct=cfg["size_pct"], risk_pct=cfg["risk_pct"],
                stop_mode=cfg["stop_mode"], target=cfg["target"],
                fill_close=cfg["fill_close"])
            st, e = ir.stats_from(eq, du, trd, invested=inv)
            rows.append(st)
            if collect_curves and wk == "w2":
                curves[f"seed{sd}"] = e
        d = pd.DataFrame(rows)
        out[f"{wk}_cagr"] = round(float(d.cagr.median()), 2)
        out[f"{wk}_cagr_lo"] = round(float(d.cagr.min()), 2)
        out[f"{wk}_cagr_hi"] = round(float(d.cagr.max()), 2)
        out[f"{wk}_cagr_worst"] = round(float(d.cagr.min()), 2)
        out[f"{wk}_dd"] = round(float(d.dd.median()), 2)
        out[f"{wk}_dd_worst"] = round(float(d.dd.min()), 2)
        out[f"{wk}_calmar"] = round(float(d.cagr.median() / abs(d.dd.median()))
                                    if d.dd.median() else np.nan, 3)
        out[f"{wk}_n"] = int(d.n.median())
        out[f"{wk}_tpy"] = round(float(d.tpy.median()), 1)
        out[f"{wk}_win"] = round(float(d.win.median()), 1)
        out[f"{wk}_mean"] = round(float(d["mean"].median()), 3)
        out[f"{wk}_netexp"] = round(float(d["mean"].median()) - 200 * cfg["cost"], 3)
        out[f"{wk}_avg_win"] = round(float(d.avg_win.median()), 2)
        out[f"{wk}_avg_loss"] = round(float(d.avg_loss.median()), 2)
        out[f"{wk}_hold"] = round(float(d.hold.median()), 0)
        out[f"{wk}_inv"] = round(float(d.invested_pct.median()), 1)
        out[f"{wk}_streak"] = int(d.max_loss_streak.median())
    return out, curves


def sweep(ctx, cells, path):
    done = set()
    if path.exists():
        with open(path) as f:
            done = {r["label"] for r in csv.DictReader(f)}
        print(f"resuming: {len(done)} cells already done", flush=True)
    else:
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    t0 = time.time()
    for i, (label, cfg) in enumerate(cells, 1):
        if label in done:
            continue
        t = time.time()
        out, _ = run_cell(ctx, cfg)
        row = {"label": label, **cfg, **out, "secs": round(time.time() - t, 1)}
        row = {k: row.get(k) for k in FIELDS}
        with open(path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
        el = time.time() - t0
        print(f"[{i}/{len(cells)}] {label:<52} sig {out['n_signals']:5d} | "
              f"w2 CAGR {out['w2_cagr']:6.2f}% DD {out['w2_dd']:7.2f}% "
              f"netexp {out['w2_netexp']:+6.2f}% tpy {out['w2_tpy']:5.1f} | "
              f"w1 CAGR {out['w1_cagr']:6.2f}% netexp {out['w1_netexp']:+6.2f}% "
              f"[{time.time()-t:.0f}s, {el/60:.1f}m]", flush=True)


def cells_g1a():
    out = []
    for age, L, dep, pol in itertools.product(
            (3, 6, 12, 24), (15, 25, 40, 60), (0.20, 0.30, 0.40, 0.60),
            ("off", "relaxed", "short70", "short80")):
        cfg = dict(BASE)
        cfg.update(max_age_m=age, L=L, max_depth=dep, min_bars=max(25, L))
        if pol.startswith("short"):
            cfg.update(rs_policy="short", rs_min=float(pol[5:]))
        else:
            cfg.update(rs_policy=pol, rs_min=70.0)
        out.append((f"g1a_a{age}_L{L}_d{int(dep*100)}_{pol}", cfg))
    return out


def cells_g1b(tops):
    out = []
    for gi, g in enumerate(tops):
        for stop, trail, tgt, (slots, size) in itertools.product(
                (0.07, 0.08, 0.10), (20, 30, 50, 150), (None, 0.25),
                ((5, 0.30), (8, 0.1875), (10, 0.15), (16, 0.0625))):
            cfg = dict(g["cfg"])
            cfg.update(stop=stop, trail=trail, target=tgt, slots=slots, size_pct=size)
            out.append((f"g1b_{g['tag']}_s{int(stop*100)}_t{trail}"
                        f"_{'tp25' if tgt else 'notp'}_n{slots}", cfg))
    return out


if __name__ == "__main__":
    phase = sys.argv[1]
    ctx = ir.Ctx()
    if phase == "g1a":
        sweep(ctx, cells_g1a(), RES / "g1a_sweep.csv")
    elif phase == "g1b":
        # DELIBERATELY DIVERSE representatives of the G1a plateau, not the 6 peak cells:
        # peak-picking here would carry the multiple-testing inflation into the exit sweep.
        reps = [("a3_L25_d30", 3, 25, 0.30), ("a6_L25_d30", 6, 25, 0.30),
                ("a12_L40_d30", 12, 40, 0.30), ("a24_L15_d30", 24, 15, 0.30)]
        tops = []
        for tag, age, L, dep in reps:
            cfg = dict(BASE)
            cfg.update(max_age_m=age, L=L, max_depth=dep, min_bars=max(25, L),
                       rs_policy="off", rs_min=70.0)
            tops.append(dict(tag=tag, cfg=cfg))
        print("geometries:", [t["tag"] for t in tops], flush=True)
        sweep(ctx, cells_g1b(tops), RES / "g1b_sweep.csv")
    print("PHASE DONE", flush=True)
