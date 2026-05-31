"""Produce the full report suite for the regime-gated midcap-momentum strategy:
client tearsheet (PNG+HTML) + exact stats + yearly-vs-Nifty table, into
research/41/reports/.
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

import run_overlay as ov
rs2 = ov.rs2
UTIL = Path(__file__).resolve().parents[2] / "_utilities"
sys.path.insert(0, str(UTIL))
from tearsheet import generate_tearsheet, _metrics, _yearly  # noqa: E402

REPORTS = Path(__file__).resolve().parents[2] / "41_midsmall400_mq_concentrated" / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

close, tv = rs2.load()
strat = ov.run(close, tv, "cash", strikes=ov.STRK)["nav"]
strat = strat / strat.iloc[0]
bench = close[ov.BENCH].reindex(strat.index).ffill()
bench = bench / bench.iloc[0]

generate_tearsheet(
    strat, bench,
    name="Midcap Momentum — RS Concentrated (Regime-Gated)",
    meta=dict(bench="Nifty 50", tag="BACKTEST 2014-2026",
              disclosures="Backtest, net of 0.4% round-trip cost; idle cash @6.5% during "
                          "risk-off. Point-in-time mid-cap liquidity band; survivorship "
                          "partially controlled. Past performance is not indicative of future "
                          "results. For discussion only."),
    out_dir=REPORTS)

sm, bm = _metrics(strat, 0.065), _metrics(bench, 0.065)
sy, by = _yearly(strat), _yearly(bench)
yrs = sorted(set(sy.index) & set(by.index))
yt = pd.DataFrame({"strategy_%": (sy * 100).round(1), "nifty50_%": (by * 100).round(1)})
yt["excess_pp"] = (yt["strategy_%"] - yt["nifty50_%"]).round(1)
yt.loc[yrs].to_csv(REPORTS / "yearly_vs_nifty.csv")

stats = dict(
    cagr=sm["cagr"]*100, bench_cagr=bm["cagr"]*100, excess=(sm["cagr"]-bm["cagr"])*100,
    total=sm["total"], bench_total=bm["total"], vol=sm["vol"]*100, bench_vol=bm["vol"]*100,
    sharpe=sm["sharpe"], bench_sharpe=bm["sharpe"], sortino=sm["sortino"],
    maxdd=sm["maxdd"]*100, bench_maxdd=bm["maxdd"]*100, calmar=sm["calmar"],
    bench_calmar=bm["calmar"], best_yr=max(sy.values)*100, worst_yr=min(sy.values)*100,
    beat_pct=float(np.mean([sy[y] > by[y] for y in yrs]))*100,
    pos_months=float((strat.pct_change().dropna() > 0).mean())*100,
    years=sm["yrs"], start=str(strat.index[0].date()), end=str(strat.index[-1].date()))
(REPORTS / "stats.json").write_text(json.dumps({k: round(v, 2) if isinstance(v, float) else v
                                                for k, v in stats.items()}, indent=2))
print(json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in stats.items()}, indent=2))
print("\nYEARLY:\n", yt.loc[yrs].to_string())
print("\nwrote ->", REPORTS)
