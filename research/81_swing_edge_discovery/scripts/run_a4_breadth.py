"""EXP-A4: BREADTH replication of gap-up + ORB long across the full backfilled
5-min universe. Pre-registered BEFORE the broad data existed (written while
the backfill was still running) — the strongest possible guard against
selection: the rule is fully locked, only the universe widens.

Cells (+2 ledger): W {6,12}, gap>=0.25%, LONG, ts4, stop=OR-low. IS window
2015-02-01..2021-09-30 (per-symbol from its first bar). Costs 3bp slippage.
PASS gate (decided now): pooled net t >= 3 AND >=60% of symbols net-positive
AND F&O-subset consistent with the 9-name A3 result. OOS untouched.

Run (VPS): venv/bin/python3 research/81_swing_edge_discovery/scripts/run_a4_breadth.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]
for p in (str(STUDY), str(ROOT), str(STUDY / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from engine import loader                              # noqa: E402
from engine.backtester import BTConfig, run_symbol     # noqa: E402
from engine.costs import CostConfig                    # noqa: E402
from run_g3_orb_validation import orb_entries          # noqa: E402
from run_setups_battery import d2_entries, c2_entries  # noqa: E402

RESULTS = STUDY / "experiments" / "A4_breadth" / "results"
IS_START, IS_END = "2015-02-01", "2021-09-30"
MIN_SESSIONS = 500          # need meaningful IS history post-backfill
CFG = BTConfig(cost=CostConfig(product="FUTURES_PROXY"), time_stop_sessions=4)
SKIP = {"NIFTY50", "INDIAVIX", "BANKNIFTY"}


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    from services.data_manager import FNO_LOT_SIZES
    fno = set(FNO_LOT_SIZES.keys())

    # drifted symbols excluded until repair report says refetched
    import json
    rep_file = STUDY / "results" / "repair_report.json"
    bad = set()
    if rep_file.exists():
        rep = json.loads(rep_file.read_text())
        refetched = set(rep.get("refetched", []))
        bad = {d["symbol"] for d in rep.get("drifted", [])} - refetched
        bad |= set(rep.get("errors", {}).keys())
    print(f"excluding {len(bad)} unrepaired/error symbols")

    universe = [s for s in loader.list_symbols("5minute")
                if s not in SKIP and s not in bad]
    print(f"universe: {len(universe)} symbols")

    all_tr = []
    used = skipped_thin = 0
    for i, sym in enumerate(universe, 1):
        df = loader.load_bars(sym, "5minute", start=IS_START, end=IS_END)
        if df.empty or df.index.normalize().nunique() < MIN_SESSIONS:
            skipped_thin += 1
            continue
        used += 1
        cells = [("W6", orb_entries(df, 6, IS_START, IS_END, 0.0025)),
                 ("W12", orb_entries(df, 12, IS_START, IS_END, 0.0025)),
                 ("D2_OL", d2_entries(df, "OL_long")),
                 ("C2_CPR", c2_entries(df, "long"))]
        for lbl, ent in cells:
            tr = run_symbol(df, ent, CFG, symbol=sym)
            tr["cell"] = lbl
            tr["fno"] = sym in fno
            all_tr.append(tr)
        if i % 40 == 0:
            print(f"[{i}/{len(universe)}] used={used} "
                  f"({(time.time()-t0)/60:.1f}m)", flush=True)

    trades = pd.concat(all_tr, ignore_index=True)
    trades.to_csv(RESULTS / "a4_trades.csv", index=False)
    print(f"\nsymbols used: {used} (thin-skipped {skipped_thin})")

    def report(g, label):
        r = g["net_ret"].to_numpy(float)
        if len(r) < 2:
            print(f"{label}: n<2")
            return
        t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r)))
        sym_pnl = g.groupby("symbol")["net_ret"].sum()
        yr = pd.to_datetime(g["entry_time"]).dt.year
        yr_net = g.groupby(yr)["net_ret"].mean()
        print(f"{label}: n={len(r)} syms={sym_pnl.size} "
              f"gross={g['gross_ret'].mean()*1e4:.1f}bps "
              f"net={r.mean()*1e4:.1f}bps t={t:.2f} "
              f"sym_pos={(sym_pnl>0).mean():.2f} yr_pos={(yr_net>0).mean():.2f}",
              flush=True)

    print("\n=== A4 breadth results (net 3bp, IS) ===")
    for w in ("W6", "W12", "D2_OL", "C2_CPR"):
        g = trades[trades["cell"] == w]
        report(g, f"{w} ALL")
        report(g[g["fno"]], f"{w} F&O-only")
        report(g[~g["fno"]], f"{w} non-F&O")
    print(f"\ndone in {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
