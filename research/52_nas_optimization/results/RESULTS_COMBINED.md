# NAS Optimization — Combined Verdict (research/52)

**Method (binding, QUANT_RESEARCH_PLAYBOOK):** no brute-force grid on 28 days. The 28-day
real option chain gives SIGNAL; years of real NIFTY price action give VALIDATION/tail.
Every claim below is graded by evidence strength.

## Validated playbook (evidence-graded)

| # | Finding | Evidence | Strength |
|---|---|---|---|
| 1 | **Edge is at 1 DTE** — 0 DTE flat/neg, far-DTE (4-6) bleeds | research/51 replay, 28d REAL premiums (1-DTE +₹37k) | 28d → signal |
| 2 | **Tight opening range → calm/range day; sell then, skip wide opens** | regime_long: 6 yrs / 1,565 days, corr 0.52, **positive EVERY year 2020-26**, tight 63% vs wide 31% range-day rate | **strong, all-regime** |
| 3 | **Stop = ±0.4% underlying-move (or max-loss ₹2-3k); NOT 1.3-1.5× premium (whipsaw), NOT no-stop** | converges: 28d real (undl0.4% 1-DTE +₹16k, worst −₹4.8k) AND 2-yr stress (undl0.4% tightest tail −₹7.9k vs no-stop −₹58.8k) | **strong on tail** |
| 4 | **Diversify across families (916 vs Squeeze ≈ uncorrelated); drop OTM sleeves** | research/52(b) daily-P&L corr; subset halves DD vs all-8 | directional (28d) |

## Recommended configuration (to prototype → deploy)
**Sell 1-DTE ATM straddle, ONLY on tight-opening-range days, diversified across families,
with a ±0.4% underlying-move stop, exit 14:45/EOD.** This stacks the four edges above.

## Caveats (standing)
- 28-day option premiums = signal, not validation. Multi-year tests use REAL NIFTY paths +
  a BS premium model → the **ranking/tail is robust; absolute P&L levels are model-dependent**
  (the real 28-day premiums are the level truth, and showed 1-DTE profitable).
- Squeeze-entry reconstruction is approximate; 1-min SL/ST cadence; no bid/ask slippage.
- Forward validation accrues as the VPS options recorder keeps building real history.

## Artifacts
- (a) `scan_sensitivity.png` / `RESULTS_scan.md` — DTE/strike/SL/exit one-factor sensitivity
- (a-ii) `stop_design.png` / `RESULTS_stopdesign.md` — stop designs, 28d real premiums
- (a-iii) `stress_stops.png` / `RESULTS_stress.md` — stop designs stressed over 2 yrs real paths
- (b) `diversification.png` / `RESULTS_diversification.md` — correlation + uncorrelated subset
- (c) `regime.png` / `RESULTS_regime.md` — opening-range regime (5-min, 451 days)
- (c-long) `regime_long.png` / `RESULTS_regime_long.md` — opening-range validated 6 yrs / 1,565 days
