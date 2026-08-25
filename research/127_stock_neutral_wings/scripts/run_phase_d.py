#!/usr/bin/env python3
"""research/127 Phase D — G3 robustness runs around C1.
C1 = E45/X21/W7/K2.5/noSL/TP50. Neighborhood + DTE-window placebo + entry-lag.
Output: results/phase_d_trades.csv"""
import sqlite3, time, csv
from pathlib import Path
import pandas as pd
import run_phase_b as B

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS / "phase_d_trades.csv"
C1 = dict(dte_entry=45, dte_exit=21, tp=0.50, sl=None, wing_pct=0.07, k=0.025)

CFGS = [
    ("N_X18", dict(C1, dte_exit=18)), ("N_X24", dict(C1, dte_exit=24)),
    ("N_W6",  dict(C1, wing_pct=0.06)), ("N_W8", dict(C1, wing_pct=0.08)),
    ("N_K2",  dict(C1, k=0.02)), ("N_K3", dict(C1, k=0.03)),
    ("P_E35", dict(C1, dte_entry=35)), ("P_E55", dict(C1, dte_entry=55)),
    ("LAG1",  dict(C1, lag=1)),
]

def main():
    conn = sqlite3.connect(B.E.db_path())
    syms = [r[0] for r in conn.execute(
        "SELECT symbol, COUNT(*) c FROM nse_options_bhav "
        "WHERE symbol NOT IN ('NIFTY','BANKNIFTY') GROUP BY symbol HAVING c>500 ORDER BY symbol")]
    done = set()
    if OUT.exists():
        d = pd.read_csv(OUT, usecols=["config","symbol"]).drop_duplicates()
        done = set(zip(d["config"], d["symbol"]))
    hdr = not OUT.exists()
    for i, s in enumerate(syms):
        todo = [(l, p) for (l, p) in CFGS if (l, s) not in done]
        if not todo: continue
        t0 = time.time()
        piv = B.load_chain(conn, s); spot = B.E.load_daily(s, conn)
        if piv is None or spot.empty: continue
        n = 0
        with open(OUT, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=B.FIELDS)
            if hdr: w.writeheader(); hdr = False
            for label, p in todo:
                try: rows = B.run_cfg_symbol(piv, spot["close"], s, label, p)
                except Exception as ex: print(f"{s}/{label}: ERROR {ex}", flush=True); rows = []
                for r_ in rows: w.writerow(r_)
                n += len(rows)
        print(f"[{i+1}/{len(syms)}] {s}: {n} ({time.time()-t0:.0f}s)", flush=True)
    print("DONE ->", OUT, flush=True)

if __name__ == "__main__":
    import logging; logging.disable(logging.WARNING)
    main()
