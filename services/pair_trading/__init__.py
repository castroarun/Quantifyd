"""Config D — 6-Pair Cointegrated Pair Trading (carry-forward F&O).

Daily EOD process at 16:00 IST (post F&O close):
  1. Fetch latest daily-close prices for both legs of each pair
  2. Compute spread = log(P_a) - alpha - beta * log(P_b)
  3. Compute z-score on rolling lookback (20/40 days per pair)
  4. ENTRY (long spread)  if z <= -entry_z: BUY pair-A futures + SELL pair-B futures
  5. ENTRY (short spread) if z >= +entry_z: SELL pair-A futures + BUY pair-B futures
  6. EXIT on first of: |z| crosses 0 (mean revert) | |z| >= stop_z | hold-cap days

Default mode = PAPER (no Kite orders). See config.py:PAIR_TRADING_DEFAULTS.

Source: research/39_carry_forward_75wr_quest/CARRY_FORWARD_75WR_DAILY_SWEEP_RESULTS.md
"""
