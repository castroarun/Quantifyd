"""Phase 20 (revised) — event timeline for the KEPT SMOOTHEST
(all-at-once exit; the 25%/wk staggered version was rejected).

Anti-overplot design: instead of one crowded marker row (where 11
events in a tight cluster merged into ~2 blobs), each event TYPE gets
its OWN stacked strip rendered as an eventplot of thin vertical ticks
— so clusters show as honest comb-density you can count, and types
never overlap each other. Top panel = equity exposure %. Per-strip
title carries the exact total count.

Reuses Phase-19 `run()` with chunk=1.0 (all-at-once) + log=True.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

SC = Path(__file__).resolve().parent
RES = SC.parent / "results"
_s = importlib.util.spec_from_file_location("p19",
                                            str(SC / "19_smoothest_staggered.py"))
p19 = importlib.util.module_from_spec(_s); _s.loader.exec_module(p19)
rs2 = p19.rs2


def main():
    print("Loading ...", flush=True)
    close, tv = rs2.load()
    print("SMOOTHEST all-at-once (chunk=1.0), logging events ...",
          flush=True)
    r = p19.run(close, tv, 1.0, log=True)
    ev = r["events"]; ex = r["expo"]
    ev.to_csv(RES / "phase20_events_allatonce.csv", index=False)

    order = [
        ("ENTRY", "#159e3b", "Entry (new holding)"),
        ("EXIT_REGIME_CHUNK", "#d62728", "Regime exit → cash (Nifty<100SMA)"),
        ("EXIT_PERSTOCK_SMA", "#ff7f0e", "Exit: stock < own 100-SMA"),
        ("EXIT_TRAIL12", "#8e44ad", "Exit: 12% trailing stop"),
        ("EXIT_RS_ROTATION", "#777777", "Exit: RS rotation (out of top-22)"),
    ]
    n = len(order)
    fig, axes = plt.subplots(n + 1, 1, figsize=(15, 9), sharex=True,
                             gridspec_kw={"height_ratios":
                                          [2.2] + [1] * n})
    ax0 = axes[0]
    ax0.fill_between(ex.index, ex.values * 100, 0, color="#9fb8d6",
                     alpha=0.6)
    ax0.set_ylabel("Exposure %"); ax0.set_ylim(-5, 108)
    ax0.set_title("SMOOTHEST (all-at-once exit — the kept config) — "
                  "exposure & event timeline, 2014–2026  ·  "
                  "staggered-exit variant was REJECTED", fontsize=11)
    ax0.grid(alpha=0.2)
    for i, (typ, col, lab) in enumerate(order):
        ax = axes[i + 1]
        dts = pd.to_datetime(ev[ev["type"] == typ]["date"])
        if len(dts):
            ax.eventplot([mdates.date2num(dts)], colors=col,
                         lineoffsets=0.5, linelengths=0.9, linewidths=0.9)
        ax.set_yticks([])
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"{lab}\n(n={len(dts)})", rotation=0, ha="right",
                      va="center", fontsize=8.5)
        ax.grid(axis="x", alpha=0.18)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    out = RES / "smoothest_allatonce_timeline.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    vc = ev["type"].value_counts().to_dict()
    print("event counts (all-at-once):", vc, flush=True)
    print(f"saved -> {out.name}", flush=True)


if __name__ == "__main__":
    main()
