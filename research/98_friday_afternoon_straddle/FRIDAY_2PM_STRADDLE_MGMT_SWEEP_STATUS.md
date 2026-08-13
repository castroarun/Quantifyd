# Friday 14:00 ATM Straddle -> 15:05 exit — management bake-off

STATUS: RUNNING (2026-07-30). NIFTY ATM short straddle every Friday, enter 14:00 exit 15:05, DTE~4.
Management grid: hold | combined-premium SL | per-leg SL (close both / hold other / trail other) at 20/30/40%.
Net-of-cost (ltp+/-1% slip + brokerage), real per-minute chain, 14 Fridays (small sample -> exploratory).
Data: options_data.db NIFTY option_chain + underlying_spot.
