# Technical Indicators Optimization - Observations

**Started:** 2026-02-07 11:58
**Paused:** 2026-02-07 15:40
**Status:** R8 IN PROGRESS (5/36 complete)

---

## Executive Summary

Tested 115 technical indicator configurations across 4 rounds (R5-R8):
- **R5 (EMA):** 54 configs - COMPLETE
- **R6 (RSI/Stochastics):** 36 configs - COMPLETE
- **R7 (Ichimoku):** 20 configs - COMPLETE
- **R8 (Supertrend):** 36 configs - 5/36 (14%) RUNNING
- **R9 (Top-Down):** Pending
- **R10 (Combined):** Pending

**Total runtime so far:** ~4 hours

---

## Top 10 Configurations Found

| Rank | Config | CAGR | MaxDD | Calmar | Round | Notes |
|------|--------|------|-------|--------|-------|-------|
| 1 | **RSI21_40_80** | 22.0% | ~19.8% | 1.11 | R6 | **BEST ABSOLUTE RETURNS** |
| 2 | ICHI9_26 | 21.1% | ~20.1% | 1.05 | R7 | Standard Ichimoku |
| 3 | ST7_2.5 | 20.3% | 21.0% | 0.97 | R8 | Best Supertrend returns |
| 4 | ST7_2.0 | 19.1% | 22.2% | 0.86 | R8 | Baseline Supertrend |
| 5 | **EMA50_200_cro_exitcro** | 19.1% | ~13.4% | **1.43** | R5 | **BEST CALMAR RATIO** |
| 6 | EMA9_21_pri | 17.7% | ~16.9% | 1.05 | R5 | Fast EMA |
| 7 | ST7_3.0 | 15.1% | 16.9% | 0.89 | R8 | Higher multiplier |
| 8 | RSI14_STOCH14 | 15.2% | 16.3% | 0.93 | R6 | Combined momentum |
| 9 | **ST7_2.5_exitFlip** | 12.0% | 8.5% | **1.42** | R8 | **EXCELLENT RISK-ADJUSTED** |
| 10 | STOCH21_5_80_20 | 16.2% | 17.0% | 0.95 | R6 | Best Stochastics |

---

## Key Findings

### 1. RSI with Wider Bands Wins
- **RSI21_40_80** achieved best absolute returns (22.0% CAGR)
- Slower RSI period (21 vs 7/14) reduces whipsaws
- Wider entry band (40) catches more of the uptrend
- Higher exit threshold (80) avoids premature exits

### 2. Exit Signals Trade Returns for Safety
- Exit signals consistently reduce CAGR by 30-70%
- BUT dramatically reduce drawdowns (EMA50/200 with exit: 13.4% MaxDD)
- Result: Much higher Calmar ratios (1.43 vs 0.86)
- **Best example:** ST7_2.5_exitFlip - only 12% CAGR but 8.5% MaxDD (Calmar=1.42)

### 3. Slower Indicators Outperform Faster Ones
- RSI21 > RSI14 > RSI7
- EMA50/200 > EMA20/50 > EMA9/21
- Ichimoku 9/26 optimal (standard settings)
- Supertrend ATR7 multiplier 2.5 optimal

### 4. Combined Indicators Underperform
- ICHI + EMA combinations: 14-16% CAGR (worse than standalone)
- RSI + Stochastics: 10-15% CAGR (worse than RSI alone)
- Single indicators with optimal parameters beat combinations

### 5. Supertrend Observations (R8 Partial)
- Multiplier 2.5 is optimal (20.3% CAGR)
- Higher multipliers (3.0+) become too restrictive
- ATR period 7 best so far (faster response)
- Using flip as exit: Low returns but excellent Calmar (1.42)

---

## R8 Supertrend Results (5/36 Complete)

| Config | ATR | Mult | Exit | CAGR | MaxDD | Calmar |
|--------|-----|------|------|------|-------|--------|
| ST7_2.0 | 7 | 2.0 | None | 19.1% | 22.2% | 0.86 |
| **ST7_2.5** | 7 | 2.5 | None | **20.3%** | 21.0% | 0.97 |
| ST7_2.0_exitFlip | 7 | 2.0 | Flip | 5.6% | 5.4% | 1.05 |
| ST7_3.0 | 7 | 3.0 | None | 15.1% | 16.9% | 0.89 |
| **ST7_2.5_exitFlip** | 7 | 2.5 | Flip | 12.0% | **8.5%** | **1.42** |

**Remaining R8 configs:** 31 (ATR periods 10/14 with various multipliers and exit modes)

---

## Pending Work

### R8 Supertrend (31/36 remaining)
Still need to test:
- ATR period 10: multipliers 2.0, 2.5, 3.0, 3.5 (×2 with/without exit)
- ATR period 14: multipliers 2.0, 2.5, 3.0, 3.5 (×2 with/without exit)
- Various combined configs

### R9 Top-Down (22 configs)
- Weekly trend filter ON/OFF
- Daily confirmation ON/OFF
- Best indicators from R5-R8 with multi-timeframe alignment

### R10 Combined Best (TBD)
- Top 3-5 indicators combined with various weighting

---

## How to Resume Optimization

### Option 1: Continue R8 from config 6
```bash
cd C:\Users\Castro\Documents\Projects\Covered_Calls
python run_technical_optimization.py --round R8 --workers 3 --skip-first 5
```

### Option 2: Restart R8 from beginning
```bash
cd C:\Users\Castro\Documents\Projects\Covered_Calls
python run_technical_optimization.py --round R8 --workers 3
```

### Option 3: Run all remaining rounds (R8-R10)
```bash
cd C:\Users\Castro\Documents\Projects\Covered_Calls
python run_technical_optimization.py --round all --workers 3
```

**Note:** The optimization script auto-saves results to `technical_optimization_results.csv` after each round.

---

## Background Task Info

- **Task ID:** b63e283
- **Output file:** `C:\Users\Castro\AppData\Local\Temp\claude\c--Users-Castro-Documents-Projects-Covered-Calls\tasks\b63e283.output`
- **Status at pause:** R8 running, 5/36 complete

To check if still running:
```powershell
Get-Content 'C:\Users\Castro\AppData\Local\Temp\claude\c--Users-Castro-Documents-Projects-Covered-Calls\tasks\b63e283.output' -Tail 20
```

---

## Technical Implementation Notes

### Files Created/Modified
- `services/technical_indicators.py` - EMA, RSI, Stochastics, Ichimoku, Supertrend calculators
- `services/mq_portfolio.py` - Added 15+ technical indicator config fields
- `services/mq_backtest_engine.py` - Added `_check_technical_entry()` and `_check_technical_exit()`
- `run_technical_optimization.py` - Parallel optimization runner with 5-min status updates

### Performance Issues Noted
- Supertrend with high multipliers (3.0+) creates very restrictive filters
- Only 14-20 initial positions instead of 30
- Causes more frequent rebalancing and slower execution
- Each Supertrend config takes ~15-30 minutes vs ~3-5 minutes for EMA/RSI

---

*Last updated: 2026-02-07 15:40*
