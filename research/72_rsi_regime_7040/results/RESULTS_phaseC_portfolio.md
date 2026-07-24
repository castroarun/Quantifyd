# Phase C — RSI Momentum-Regime SLOT PORTFOLIO: does diversification beat the Nifty?

## VERDICT: **SIGNAL (weak) — NOT A STRATEGY.** Only 1 of 40 configs clears the gate, by a hair.

Long-only slot portfolio; per name enter at close when RSI(14) >= ENTRY (fill a free slot,
strongest RSI first), exit at close when RSI < EXIT. 15 bps/side, idle cash 0%.
Window 2015-01-01 -> 2026-07-07 (11.5y). Benchmark NIFTYBEES B&H over identical window:
**CAGR 10.95%, MaxDD -36.34%.**

Gate = net CAGR >= 1.5x bench (>= 16.4%) AND net MaxDD < bench (< 36.3%). Ranked by net Calmar.

## Result
- **Gate passers: 1 / 40 -> `nifty50_N20_E60X30`** (net CAGR 16.76%, MaxDD -23.85%, Calmar 0.70,
  Sharpe 1.14, beat 1.53x, 29.6 trades/yr). Clears 1.5x by 0.03 with a much shallower DD.
- Beat-ratio range across the sweep: **0.21x - 1.53x**. Most configs LOSE to the index outright.
- The DD gate is easy (36/40 have lower DD than the crash-heavy index B&H); the RETURN gate is
  what kills nearly everything — RSI-timing spends much of its life in cash and misses upside.

## Top-8 by net Calmar
| label | names | netCAGR | netMaxDD | Calmar | Sharpe | trades/yr | beat | lowerDD | GATE |
|---|---|---|---|---|---|---|---|---|---|
| nifty50_N20_E60X30 | 43 | 16.76 | -23.85 | 0.70 | 1.14 | 29.6 | 1.53 | Y | **PASS** |
| nifty50_N15_E60X30 | 43 | 16.13 | -22.93 | 0.70 | 1.07 | 22.8 | 1.47 | Y | no |
| nifty200_N20_E70X40 | 139 | 16.35 | -29.06 | 0.56 | 1.03 | 77.5 | 1.49 | Y | no |
| nifty50_N10_E60X30 | 43 | 13.85 | -24.71 | 0.56 | 0.89 | 15.5 | 1.26 | Y | no |
| nifty200_N15_E60X30 | 139 | 14.24 | -27.60 | 0.52 | 0.85 | 24.1 | 1.30 | Y | no |
| nifty200_N20_E60X40 | 139 | 15.57 | -30.58 | 0.51 | 0.95 | 90.8 | 1.42 | Y | no |
| nifty200_N20_E60X30 | 139 | 14.63 | -28.74 | 0.51 | 0.88 | 32.5 | 1.34 | Y | no |
| nifty200_N15_E70X40 | 139 | 15.97 | -31.75 | 0.50 | 0.96 | 59.1 | 1.46 | Y | no |

## Winner per-year (net %) vs NIFTYBEES
| yr | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| strat | -3.6 | 0.7 | 42.0 | 2.6 | 11.8 | 32.7 | 48.3 | 16.5 | 30.4 | 26.1 | 2.2 | -2.3 |
| bench | -4.2 | 4.0 | 29.9 | 4.8 | 13.6 | 15.4 | 26.0 | 5.5 | 21.0 | 10.4 | 11.7 | -6.2 |

Edge is concentrated in strong-trend years (2017, 2020, 2021, 2024). In 2025 it LAGGED the
index (2.2 vs 11.7). Not a persistent every-year edge.

## Honest caveats
1. **SURVIVORSHIP BIAS (loud).** Universe = TODAY's Nifty50/200 membership applied to 2015-26.
   Names that dropped out of the index (losers) are absent -> results optimistically biased.
   43/50 Nifty50 and 139/200 Nifty200 names had data since <=2015 (>=1500 rows) and qualified.
2. **Single-factor / correlation.** One signal (RSI) across highly-correlated Indian large-caps;
   diversification helps DD (lower than index) but the names co-move, so it does NOT manufacture
   a return edge — it mostly trims crash depth by sitting in cash.
3. **No cash yield.** Idle-slot cash earns 0%. A 6% yield would only IMPROVE these numbers
   (the book is frequently partly in cash), so the gate result is conservative in that respect.
4. **Marginal passer.** The single winner clears 1.5x by 0.03 and is the highest-N / lowest-threshold
   corner (N20, 60/30) — i.e. "hold more names, stay invested longer," which is close to just owning
   the index with a mild crash-timing overlay. A barely-passing single cell does not survive
   multiple-testing skepticism.

## Read
Diversifying the RSI regime system across Nifty50/200 does NOT convincingly beat the index.
It reliably lowers drawdown (cash during regime-off), but only one corner of a 40-cell grid
edges past 1.5x CAGR, on survivorship-biased data, with the edge concentrated in a few bull
years. Verdict: **SIGNAL, not a STRATEGY.**

## Next levers (if pursued)
- De-bias: point-in-time index membership (kills the survivorship inflation) before trusting any beat.
- Add the 6% idle-cash yield explicitly and re-rank (helps most at high N / frequent-cash configs).
- Robustness on N20/60-30: rsi_len sweep, per-year walk-forward, and vs an equal-weight always-in
  basket of the SAME 43 names (is the RSI overlay adding anything over just holding them?).
