"""Intraday 75% WR Quest — three-system live trading engine.

Three walk-forward-validated 5-min intraday systems:
  1. Diamond Short      — 25 stocks, single 09:45 IST scan
  2. Long Mean-Reversion — 15 stocks, continuous 11:15-13:15 scan
  3. Long Trend-Cont    — 30 stocks, continuous 09:15-10:30 scan

All exit on TP 0.5% / SL 1.5% / EOD 15:25 IST.

Default mode = PAPER (no Kite orders). See config.py:
  DIAMOND_SHORT_DEFAULTS / LONG_MR_DEFAULTS / LONG_TC_DEFAULTS

Source: research/37_intraday_75wr_quest/INTRADAY_75WR_5MIN_SWEEP_RESULTS.md
"""
