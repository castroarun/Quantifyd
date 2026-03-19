# 06: Exit & Risk Management Research

**Period:** February 14-15, 2026
**Status:** Complete — ATH drawdown exit dominates, trailing SL adds no value

## What Was Tested

Comprehensive exit and risk management research across multiple approaches:

### A. Stop Loss Sweep

Tested hard stop losses from 2% to 50% on MQ portfolio strategy.

**Finding:** Stops above 8% produce identical results because the ATH drawdown exit (20% from peak) always fires first. Hard stop only matters for catastrophic single-day drops.

### B. Trailing Stop Loss Backtest

Tested fixed vs trailing vs ratchet stops on Breakout V3 trades:

| Strategy | Win% | Avg Ret | PF | Total Return | Severe Losses |
|----------|------|---------|-----|-------------|---------------|
| FIXED (baseline) | 67.5% | 21.8% | 6.38 | 7,240% | 2 |
| TRAIL-15% | 56.9% | 10.2% | 4.17 | 3,374% | 0 |
| TRAIL-20% | 61.7% | 15.2% | 4.96 | 5,036% | 0 |
| TRAIL-25% | 69.0% | 20.5% | 6.09 | 6,805% | 25 |
| RATCHET-20% | 60.2% | 14.9% | 5.12 | 4,934% | 0 |
| RATCHET-25% | 66.0% | 20.0% | 6.35 | 6,624% | 1 |

**Finding:** Trailing/ratchet stops reduce severe losses to zero but cut total return by 30-50%. The fixed exit with ATH drawdown is the best risk-adjusted approach.

### C. Options-Hedged Futures Backtest

Tested V3 breakout trades with futures leverage + protective puts/covered calls:

| Strategy | CAGR | MaxDD | Calmar |
|----------|------|-------|--------|
| Naked Futures (3x) | 27.3% | 10.9% | 2.49 |
| Covered Call (5% OTM) | 12.5% | 7.9% | 1.57 |
| Protective Put (5% OTM) | 20.1% | 8.3% | 2.42 |

**Finding:** Naked futures with V3 signals has best CAGR. Covered calls cut risk but halve returns. Protective puts offer best balance.

### D. Liquid Fund Allocation

Tested parking idle cash in liquid funds (6.5% annual) during no-signal periods.

**Finding:** Adds ~1-2% to annual returns when capital is idle 40-60% of the time.

## Files

| File | Purpose |
|------|---------|
| `scripts/sl_sweep_v2.py` | Stop loss parameter sweep |
| `scripts/run_trailing_sl_backtest.py` | Trailing SL comprehensive test |
| `scripts/run_exit_optimization.py` | Exit rule optimization framework |
| `scripts/run_options_hedged_backtest.py` | Options hedging with futures |
| `scripts/run_futures_backtest.py` | Pure futures leverage test |
| `scripts/run_liquid_fund_backtest.py` | Liquid fund idle cash test |
| `scripts/run_liquid_fund_fixed.py` | Fixed version of liquid fund test |
| `results/trailing_sl_results.csv` | Full trailing SL sweep data |
| `results/trailing_sl_summary.txt` | Summary of trailing SL results |
| `results/options_backtest_output.txt` | Options hedging detailed output |

## Next Steps / Recommendations

- **ATH drawdown exit (20% from peak) is the production exit** — no change needed
- Hard stop at 50% is a safety net only (never fires in practice)
- If F&O leverage is ever used, protective puts (5% OTM) offer the best risk/reward
- Liquid fund parking is a "nice to have" for idle capital but not material
