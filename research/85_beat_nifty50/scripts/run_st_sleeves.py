"""research/85 addendum EXP-ST1 (user idea): per-stock daily SuperTrend(7,3)
sleeves on Nifty 50 — long when green, park in liquid fund (6.5% gross ~ 4.5%
net slab-taxed -> use 5.5%) when red. Case 1 = all 50 equal sleeves; Case 2 =
top-10 by market cap (today's list — survivorship-flagged). Costs 15bp/side.
Prior art (weekly-supertrend study): per-stock ST timing LOST to B&H by
~3.5pp/yr on Nifty 200 — this is the N50/liquid-parking variant, run to spec.
Window 2005-2023 (OOS 2024+ untouched for this family).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R81 = HERE.parents[1] / "81_swing_edge_discovery"
for p in (str(R81), str(R81.parents[1]), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from engine import loader, metrics                     # noqa: E402
from run_beat_n50 import N50                           # noqa: E402
from services.technical_indicators import calc_supertrend  # noqa: E402

START, END = "2005-01-01", "2023-12-31"
LIQUID_DAILY = 0.055 / 252
COST_SIDE = 0.0015
TOP10 = ["RELIANCE", "HDFCBANK", "TCS", "BHARTIARTL", "ICICIBANK",
         "SBIN", "INFY", "HINDUNILVR", "ITC", "LT"]


def sleeve_returns(df: pd.DataFrame) -> pd.Series:
    _, direction = calc_supertrend(df, atr_period=7, multiplier=3.0)
    green = (direction > 0).shift(1).fillna(False)      # signal yesterday -> today
    ret = df["close"].pct_change().fillna(0.0)
    out = np.where(green, ret, LIQUID_DAILY)
    flip = green.astype(int).diff().abs().fillna(0)
    out = out - flip.to_numpy() * COST_SIDE
    return pd.Series(out, index=df.index)


def case_nav(universe, bars, label, bench):
    sleeves = []
    for sym in universe:
        if sym in bars:
            sleeves.append(sleeve_returns(bars[sym]).rename(sym))
    panel = pd.concat(sleeves, axis=1)
    port = panel.mean(axis=1, skipna=True)
    port = port[(port.index >= START) & (port.index <= END)]
    nav = (1 + port).cumprod()
    cs = metrics.curve_stats(nav)
    b = bench.reindex(nav.index).ffill()
    cb = metrics.curve_stats(b / b.iloc[0])
    print(f"\n{label} ({panel.shape[1]} sleeves): CAGR {cs['cagr']:.1%} vs bench "
          f"{cb['cagr']:.1%} -> EXCESS {cs['cagr']-cb['cagr']:+.1%} | "
          f"MaxDD {cs['max_dd']:.0%} (bench {cb['max_dd']:.0%}) | "
          f"Sharpe {cs['sharpe']:.2f} Calmar {cs['calmar']:.2f}")
    ry = metrics.per_year_table(port)["return"]
    by = metrics.per_year_table(b.pct_change())["return"]
    exc = ((ry - by) * 100).round(1)
    print(f"  yearly excess: {dict(exc)}  beat {int((exc>0).sum())}/{len(exc)} yrs",
          flush=True)
    # B&H of the SAME basket (the fair comparison, per prior study)
    bh = pd.concat([bars[s]["close"].pct_change() for s in universe if s in bars],
                   axis=1).mean(axis=1)
    bh = bh[(bh.index >= START) & (bh.index <= END)]
    bnav = (1 + bh.fillna(0)).cumprod()
    cbh = metrics.curve_stats(bnav)
    print(f"  same-basket B&H: CAGR {cbh['cagr']:.1%} MaxDD {cbh['max_dd']:.0%} "
          f"-> ST timing adds {cs['cagr']-cbh['cagr']:+.1%} CAGR, "
          f"{(cs['max_dd']-cbh['max_dd'])*100:+.0f}pp DD", flush=True)


def main():
    bars = {}
    for sym in N50:
        df = loader.load_bars(sym, "day", start="2004-01-01", end=END)
        if len(df) >= 500:
            bars[sym] = df
    bench = loader.load_bars("NIFTYBEES", "day", start="2004-01-01", end=END)["close"]
    print(f"universe: {len(bars)} names")
    case_nav(list(bars.keys()), bars, "CASE1 all-N50 ST(7,3)+liquid", bench)
    case_nav(TOP10, bars, "CASE2 top10-mcap ST(7,3)+liquid", bench)
    print("\nST-SLEEVES COMPLETE", flush=True)


if __name__ == "__main__":
    main()
