# Research Index

All research conducted for Quantifyd, organized chronologically by execution order.

**Total: 216 files across 12 research phases + utilities**

---

## Folder Guide

| # | Folder | Period | What | Key Results |
|---|--------|--------|------|-------------|
| 01 | `01_covered_call_baseline/` | Dec 2025 | Initial covered call strategy optimizer & data simulator | Baseline strategy, superseded by MQ |
| 02 | `02_mq_portfolio_optimization/` | Jan-Feb 2026 | MQ (Momentum+Quality) portfolio parameter sweeps — portfolio size, stops, allocations | PS10 = 48.66% CAGR, PS30 = 32.19% CAGR |
| 03 | `03_breakout_v3_analysis/` | Feb 12-15 | Breakout V3 consolidation pattern analysis, filter optimization, trade verification | 65% WR, PF 1.70, Pine scripts for TradingView |
| 04 | `04_technical_indicator_sweep/` | Feb 7-16 | Technical indicators (EMA, RSI, SuperTrend, MACD, ADX) applied to MQ entry | All indicators HURT MQ — baseline wins |
| 05 | `05_crash_filter_confluence/` | Feb 14 | Crash detection filters, signal confluence, TTM Squeeze analysis | Crash filter useful, TTM inconclusive |
| 06 | `06_exit_risk_management/` | Feb 14-15 | Stop loss sweeps, trailing SL, exit rules, options hedging, futures, liquid funds | ATH drawdown exit dominates, stops >8% identical |
| 07 | `07_strategy_exploration/` | Feb 16-17 | Broad multi-strategy sweeps — momentum, mean reversion, hybrid, price action, EMA crossover, long/short | Per-stock analysis, 20Y exploration results |
| 08 | `08_ipo_strategy/` | Feb 17-18 | IPO (new listing) momentum strategy research across 5 sweep phases | 65% WR sweep, practical parameter sets |
| 09 | `09_mq_advanced_variants/` | Feb 18-20 | MQ concentration, KC6 models, MQ+technical hybrid, longterm (v1-v4), COVID analysis, model portfolio | Concentration = #1 CAGR lever, COVID recovery analysis |
| 10 | `10_combined_mq_v3_overlay/` | Feb 20 | Combined MQ + Breakout V3 overlay, 15-year backtest, rebalance optimization | Equity curves, MQ ATH trailing Pine script |
| 11 | `11_cpr_intraday_strategy/` | Mar 16 | CPR (Central Pivot Range) intraday strategy — baseline, BB+KC, RSI+Stoch, SuperTrend, regime filters | 79-stock full & OOS results, MQ correlation |
| 12 | `12_combined_swing_strategy/` | Mar 17 | Combined swing system (EMA trend + RSI mean reversion + NR7 breakout) on 60-min data, 113 config sweep | Best: E25/60 ADX20, 8.19% CAGR — underperforms index |

---

## Folder Structure Convention

Each research folder follows this layout:

```
NN_research_name/
├── scripts/          # Python scripts that ran the research
├── results/          # CSV outputs, JSON data, sample trades
├── reports/          # HTML dashboards & visual reports (if any)
├── verification/     # Trade verification logs, charts (if any)
├── pine_scripts/     # TradingView Pine scripts (if any)
├── logs/             # Computation logs (if any)
└── *.md              # Research summary / findings documentation
```

## Utilities

`_utilities/` — Helper scripts (code generators, data backfill tool, logo exploration)

---

## Best Results Summary

| Strategy | Best Config | CAGR | MaxDD | Sharpe | Calmar |
|----------|------------|------|-------|--------|--------|
| MQ Concentrated (PS10) | PS10_SEC70_POS30_TOP30_BIM | **48.66%** | 26.35% | 1.30 | 1.85 |
| MQ Balanced (PS30) | PS30_HSL50_ATH20_EQ95 | 32.19% | 27.0% | 1.05 | 1.19 |
| MQ + SuperTrend | STREND_atr7_m3.0 | 27.79% | 16.65% | — | 1.67 |
| Breakout V3 (KC6) | KC6 baseline | — | — | — | PF 1.70 |
| Combined Swing | E25_60_ADX20_R14_30_P20_10 | 8.19% | 8.0% | 1.40 | 1.03 |

**Key insight:** MQ portfolio strategy dominates. Technical indicators and swing trading on 60-min data cannot compete with fully-invested momentum+quality stock selection.

## 49 — volbreak_pdh_30min (2026-06-01)
**Vol > own 50-day MA + break prev-day high, 30-min intraday LONG**, all-exit
sweep (EOD/HARD_SL/2R/Chandelier/Supertrend), RSI axis, 8-name G1 smoke,
2018–2026, 6bps. **VERDICT: NO EDGE (net of cost)** — gross faint (best HARD_SL
+0.081R), every exit net-negative @6bps (best −0.029R, PF 0.95), no per-year
persistence, RSI no help. Cost (~0.08–0.11R) eats the edge; intraday < swing
(cf research/44). Shelved before the 30k-cell sweep. See research/49.../RESULTS.md.


## 50 - NAS 8-System 28-Day Live/Paper Review
**Real recorded fills of all 8 NAS systems, 2026-04-20 to 06-02 (25 days, 180 trades).** Factsheet: per-system equity (raw + per-lot), per-day x system heatmap, drawdowns, stats. **VERDICT: SIGNAL/AUDIT ONLY (28-day single regime, not validation).** Surfaced a data bug: the OTM trade recorder books exit=Rs0 -> Squeeze OTM (+115k) & 916 OTM (+204k) show fake 100% win / Rs0 DD, inflating the combined headline to Rs592k (~319k phantom). Trustworthy ATM half: 916 ATM2 +137k / ATM +68k / ATM4 +65k positive; Squeeze ATM/ATM2 ~flat-negative; 9:16 family beat squeeze-entry family. Phase 2 re-prices actual legs vs recorded option_chain for true P&L. See research/50_nas_28day_live_review/results/RESULTS.md.

## 51 - NAS Systems: true replay on recorded NIFTY weekly chain
**All 8 NAS systems replayed (rules re-run) on 28 days of recorded option_chain, lots=2 normalized.** Combined **net Rs-54,123 (NEGATIVE)**; only Squeeze ATM +Rs1.7k, 916 ATM2 cascade -23.9k worst, 9:16 family < squeeze-entry, day-win 32-44%. **VERDICT: NO EDGE this regime** (agrees with research/50 actuals ~flat). Fixed OTM roll-churn (1252->157 legs). 28d single-regime + squeeze-entry/1-min approximations = signal not validation. See research/51_nas_chain_replay/results/.

## 52 - NAS optimization (disciplined: 28d signal + years validation)
Found an evidence-graded playbook, NOT a dredged winner. (1) Edge at 1 DTE. (2) **Tight opening-range -> range day, robust over 6yr/1565d, positive EVERY year 2020-26 (corr 0.52)** - sell on tight open, skip wide. (3) Stop = **+/-0.4% underlying-move** (converges on 28d-real +2yr-stress; beats 1.3x premium whipsaw and no-stop whose 2yr worst-day = -58.8k). (4) Diversify across 916 vs Squeeze families (uncorrelated), drop OTM losers. Recommended config: 1-DTE ATM straddle, tight-open only, cross-family, +/-0.4% move stop. See research/52_nas_optimization/results/RESULTS_COMBINED.md.