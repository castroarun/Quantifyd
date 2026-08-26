#!/usr/bin/env python3
"""research/127 Phase B3 — tight premium stops on the C1 structure, full history.
Motivated by the paper book's Jun-Aug 2026 window where SL125 flipped -9k to +51k
on 18 trades; this tests whether that generalizes over 628+ trades / 10 years.
Output: results/phase_b3_trades.csv (compare vs phase_b2 C1_E45X21W7K25_noSL)."""
import sqlite3, time, csv
from pathlib import Path
import pandas as pd
import run_phase_b as B

RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS / "phase_b3_trades.csv"
C1 = dict(dte_entry=45, dte_exit=21, tp=0.50, wing_pct=0.07, k=0.025)

CFGS = [
    ("SL125", dict(C1, sl=1.25)),
    ("SL150", dict(C1, sl=1.50)),
    ("SL175", dict(C1, sl=1.75)),
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
