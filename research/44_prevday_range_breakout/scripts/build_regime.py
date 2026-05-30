"""Build a market-regime series from the synthetic equal-weight market index
(median daily return across all stocks). Emits results/market_regime.csv with
CAUSAL regime flags (each day's flag uses the PRIOR day's level vs its MA, so it's
known before the session).
"""
from __future__ import annotations

import csv
import sqlite3
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from run_nrb_p3 import build_market_returns  # noqa: E402

DB = ROOT / "backtest_data" / "market_data.db"
OUT = HERE.parent / "results" / "market_regime.csv"


def sma(x, w):
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.full(len(x), np.nan)
    if len(x) >= w:
        out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def main():
    con = sqlite3.connect(DB)
    print("building synthetic market index...")
    mret = build_market_returns(con)
    dates = sorted(mret)
    ret = np.array([mret[d] for d in dates])
    level = np.cumprod(1.0 + ret)        # index level
    s20, s50, s200 = sma(level, 20), sma(level, 50), sma(level, 200)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "level", "sma50", "reg20", "reg50", "reg200"])
        for i, d in enumerate(dates):
            # causal: use prior day's level vs prior day's MA
            j = i - 1
            r20 = int(j >= 0 and not np.isnan(s20[j]) and level[j] > s20[j])
            r50 = int(j >= 0 and not np.isnan(s50[j]) and level[j] > s50[j])
            r200 = int(j >= 0 and not np.isnan(s200[j]) and level[j] > s200[j])
            w.writerow([d, round(level[i], 4),
                        round(s50[i], 4) if not np.isnan(s50[i]) else "",
                        r20, r50, r200])
    print(f"wrote {len(dates)} rows -> {OUT.name} "
          f"({dates[0]} -> {dates[-1]})")


if __name__ == "__main__":
    main()
