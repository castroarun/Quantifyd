"""Run the MQ (Momentum + Quality) portfolio backtest (PS30 baseline) and produce
the client tearsheet + stats, per the Quant-Research Playbook reporting standard.
Benchmark = Nifty 50 (NIFTYBEES) over the same window.
"""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "_utilities"))

from services.mq_backtest_engine import MQBacktestEngine  # noqa: E402
from services.mq_portfolio import MQBacktestConfig  # noqa: E402
from tearsheet import generate_tearsheet, _metrics, _yearly  # noqa: E402

REPORTS = ROOT / "research" / "02_mq_portfolio_optimization" / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
DB = ROOT / "backtest_data" / "market_data.db"

# PS30 baseline — CLEAN capital allocation (equity 0.80 + debt 0.20 = 1.00). NOTE:
# the often-quoted "EQ95" baseline sets equity 0.95 + debt 0.20 = 1.15, which makes
# the portfolio's total_value curve start at ~Rs.1.15cr from Rs.1cr capital and
# inflates the engine's nominal CAGR (it divides by Rs.1cr) by ~6pp vs the actual
# simulated path. We use the clean 80/20 so the factsheet path-CAGR is un-inflated
# and internally consistent (path CAGR == engine CAGR).
config = MQBacktestConfig(
    start_date="2023-01-01", end_date="2025-12-31",
    initial_capital=10_000_000, portfolio_size=30,
    equity_allocation_pct=0.80, hard_stop_loss=0.50, rebalance_ath_drawdown=0.20,
)

print("preloading MQ data...", flush=True)
universe, price_data = MQBacktestEngine.preload_data(config)
print("running MQ backtest (PS30)...", flush=True)
engine = MQBacktestEngine(config, preloaded_universe=universe, preloaded_price_data=price_data)
result = engine.run()

# daily_equity already stores portfolio.total_value (equity + cash + debt + NIFTYBEES)
# per mq_portfolio.py:1059 — it IS the full portfolio path. Use it directly.
nav = pd.Series(result.daily_equity, dtype=float)
nav.index = pd.to_datetime(nav.index)
nav = nav.sort_index()
nav = nav / nav.iloc[0]

con = sqlite3.connect(DB)
b = pd.read_sql_query(
    "SELECT date,close FROM market_data_unified WHERE symbol='NIFTYBEES' AND "
    "timeframe='day' AND date>=? AND date<=? ORDER BY date", con,
    params=[nav.index[0].strftime("%Y-%m-%d"), nav.index[-1].strftime("%Y-%m-%d")])
con.close()
b["date"] = pd.to_datetime(b["date"])
bench = b.set_index("date")["close"]

generate_tearsheet(
    nav, bench,
    name="MQ Portfolio — Momentum + Quality (PS30)",
    meta=dict(bench="Nifty 50", tag="BACKTEST 2023-2025",
              disclosures="Backtest, NET of full Indian transaction costs (brokerage+STT+GST+"
                          "stamp+slippage). Nifty-500 universe, 30 names, semi-annual rebalance, "
                          "ATH-drawdown + 50% hard-stop exits, Darvas breakout top-ups, idle cash "
                          "in NIFTYBEES. Past performance is not indicative of future results. "
                          "For discussion only."),
    out_dir=REPORTS)

sm, bm = _metrics(nav, 0.065), _metrics(bench / bench.iloc[0], 0.065)
sy, by = _yearly(nav), _yearly(bench / bench.iloc[0])
yrs = sorted(set(sy.index) & set(by.index))
yt = pd.DataFrame({"strategy_%": (sy * 100).round(1), "nifty50_%": (by * 100).round(1)})
yt["excess_pp"] = (yt["strategy_%"] - yt["nifty50_%"]).round(1)
yt.loc[yrs].to_csv(REPORTS / "yearly_vs_nifty.csv")

stats = dict(
    engine_cagr=result.cagr, engine_sharpe=result.sharpe_ratio, engine_sortino=result.sortino_ratio,
    engine_maxdd=result.max_drawdown, engine_calmar=result.calmar_ratio,
    total_return_pct=result.total_return_pct, final_value=result.final_value,
    win_rate=result.win_rate, total_trades=result.total_trades, total_topups=result.total_topups,
    ts_cagr=sm["cagr"]*100, bench_cagr=bm["cagr"]*100, excess=(sm["cagr"]-bm["cagr"])*100,
    total_x=sm["total"], bench_total_x=bm["total"], maxdd=sm["maxdd"]*100, bench_maxdd=bm["maxdd"]*100,
    sharpe=sm["sharpe"], bench_sharpe=bm["sharpe"], calmar=sm["calmar"],
    beat_pct=float(np.mean([sy[y] > by[y] for y in yrs]))*100,
    start=str(nav.index[0].date()), end=str(nav.index[-1].date()))
(REPORTS / "stats.json").write_text(json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                                                for k, v in stats.items()}, indent=2))
print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in stats.items()}, indent=2))
print("\nYEARLY:\n", yt.loc[yrs].to_string())
print("\nwrote ->", REPORTS)
