# Volume Breakout — Findings (PROMISING, with caveats)

**Status:** Viable v2 configuration identified. Proceed to out-of-sample test +
paper trade before live.

## Thesis

A 5-min bar that prints >N× its 20-bar avg volume AND closes beyond the prior
10-bar range signals institutional order flow. Enter on next bar's open, stop
at the breakout bar's opposite edge, target 2.5R.

Momentum-family (correlated with ORB on regime, but operationally distinct:
different trigger, different time distribution, **different universe** so no
account-netting conflict with ORB's 15 stocks).

## v1 result (broad 15-stock universe)

| Variant | Trades | WR | PF | Net P&L (2yr) |
|---|---:|---:|---:|---:|
| spike_2x | 488 | 40.8% | 0.65 | −Rs 152K |
| spike_3x | 270 | 38.1% | 0.61 | −Rs 107K |
| spike_2x_rsi | 439 | 40.1% | 0.66 | −Rs 130K |
| spike_3x_rsi | 253 | 39.9% | 0.67 | −Rs 81K |

Portfolio-wide break-even *before* costs; costs eat the edge (0.15% round-trip).

**Per-stock diagnostic revealed strong dispersion:**
- Quality names (ASIANPAINT 11.93 PF, LT 1.49, JSWSTEEL 1.27, HEROMOTOCO 1.16,
  EICHERMOT 1.14) — breakouts resolve cleanly
- News/PSU names (WIPRO 0.18, INDUSINDBK 0.19, HCLTECH 0.33, BANKBARODA 0.41)
  — too many fake-out reversals

## v2 result (curated 9-stock universe, R:R 2.5)

| Variant | Trades | WR | PF | Net P&L | CAGR | Sharpe | MaxDD | Calmar |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rr_2.5 spike_3x (BEST) | 121 | 43.8% | **1.35** | +Rs 35,485 | +38.2% | **1.96** | 3.7% | **10.41** |
| rr_2.0 spike_3x | 121 | 45.5% | 1.29 | +Rs 29,513 | +31.2% | 1.86 | 3.8% | 8.32 |
| rr_2.0 spike_2x | 208 | 43.8% | 1.14 | +Rs 23,415 | +16.0% | 0.94 | 6.3% | 2.54 |

**Universe:** ASIANPAINT, LT, JSWSTEEL, HEROMOTOCO, EICHERMOT, MARUTI, TITAN,
HINDALCO, JINDALSTEL

## Caveats (read before getting excited)

1. **Hindsight bias on universe selection.** We picked the 9 stocks *from*
   v1's per-stock PF. Classic overfit risk — the same 9 may not be tomorrow's
   winners. Out-of-sample validation is needed before committing capital.
2. **Single-name concentration.** ASIANPAINT alone contributes +Rs 19,235 on
   just 6 trades — 54% of total profit. Remove it and PF drops to ~1.15.
   The system is fragile without the best stock's contribution.
3. **Low trade frequency.** 121 trades / 2 years = ~60/year. Roughly 1 trade
   every 4 trading days across 9 stocks. Statistical power is modest.
4. **JINDALSTEL bleeds.** −Rs 6,264 on 39 trades in the best variant. Worth
   dropping or investigating why breakouts fail there specifically.
5. **Costs sensitivity.** At 0.15% round-trip, costs consumed ~30% of gross
   P&L. Real-world slippage on these mid-caps may be worse than assumed.

## Per-stock contributions (rr_2.5 spike_3x)

| Stock | Trades | Net P&L | % of portfolio |
|---|---:|---:|---:|
| ASIANPAINT | 6 | +Rs 19,235 | 54% |
| MARUTI | 9 | +Rs 5,155 | 15% |
| HINDALCO | 14 | +Rs 4,222 | 12% |
| HEROMOTOCO | 12 | +Rs 4,065 | 11% |
| JSWSTEEL | 12 | +Rs 3,978 | 11% |
| LT | 6 | +Rs 2,675 | 8% |
| EICHERMOT | 8 | +Rs 2,568 | 7% |
| TITAN | 15 | −Rs 150 | 0% |
| JINDALSTEL | 39 | −Rs 6,264 | −18% |

Dropping JINDALSTEL: net Rs 41,749 on 82 trades, even cleaner curve.

## Recommended next steps (before going live)

1. **Out-of-sample test.** Run v2 on pre-2024 data if we can backfill more,
   OR reserve last 6 months (Oct 2025 – Mar 2026) as OOS.
2. **Drop JINDALSTEL**, consider also dropping TITAN (noisy flat).
3. **Paper trade 4-6 weeks** on VPS with a Kite executor — same infrastructure
   pattern as ORB live engine.
4. **Live with small capital first** — Rs 50K-1L risk exposure, scale only if
   paper+live both match backtest expectations.

## Artifacts

- `scripts/run_volume_breakout.py` — v1 broad universe
- `scripts/run_volume_breakout_v2.py` — v2 curated universe + R:R sweep
- `results/trades.csv`, `trades_v2.csv`
- `results/summary.csv`, `summary_v2.csv`
- `results/daily_pnl.csv` — for future correlation analysis vs ORB
- `logs/run_v1.log`, `run_v2.log`
