
import sys, sqlite3
from pathlib import Path
import pandas as pd
sys.path.insert(0, "/home/arun/quantifyd/research/_utilities")
from tearsheet import generate_tearsheet
BASE = Path("/home/arun/quantifyd/research/75_nifty250_momentum_top15")
nav = pd.read_csv(BASE/"results"/"nav_base.csv", index_col=0, parse_dates=True)["nav"]
con = sqlite3.connect("/home/arun/quantifyd/backtest_data/market_data.db")
b = pd.read_sql("SELECT date,close FROM market_data_unified WHERE symbol='NIFTYBEES' AND timeframe='day' AND close IS NOT NULL ORDER BY date", con, parse_dates=["date"]).set_index("date")["close"]
con.close()
b = b.reindex(nav.index).ffill(); b = b/b.iloc[0]; nav = nav/nav.iloc[0]
meta = dict(bench="Nifty 50 (NIFTYBEES)", tag="BACKTEST 2006-2026",
    disclosures=("Backtest, not live. 2006-2026, survivorship-FREE PIT top-250-by-traded-value proxy "
      "for Nifty LargeMidcap 250 (early years ~218 names). Net of 0.30% round-trip; 20% STCG shown "
      "separately, NOT in NAV. Daily-marked drawdown. Faithful replication of the 'Only Momentum "
      "Strategy' video: top-15 by 12m momentum, per-stock 50>100>200 EMA filter, NIFTYBEES>100EMA "
      "cash gate, monthly. Past performance does not guarantee future returns."))
png = generate_tearsheet(nav, b, "Nifty-250 Momentum Top-15 (faithful video replication)", meta, str(BASE/"results"))
print("WROTE", png)
