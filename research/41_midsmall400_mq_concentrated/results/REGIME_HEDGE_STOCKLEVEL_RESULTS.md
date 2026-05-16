# Regime / Hedge / Stock-Level Risk — Consolidated Results (Phases 09–11)

Refines the locked mid-cap winner. Core held constant throughout:
`mid_120d_N15 + q0.5` (RS-120 vs NIFTYBEES, N=15, monthly, top-22
buffer, 0.4% RT, 6.5% bear-cash, 2014–2026). Only the regime / hedge /
stock-level layer changes, so every delta is attributable. Companion to
`MIDCAP_RS120_REGIME_MOMENTUM_DETAILED_REPORT.md`.

## Phase 09 — regime-filter alternatives + ATH layer

The 200-SMA gate is laggy/whipsaws. Swept 9 regime rules:

| Regime | CAGR | post-tax | MaxDD | Sharpe | Calmar |
|---|---|---|---|---|---|
| OFF | 37.0 | 30.9 | −29.6 | 1.35 | 1.25 |
| SMA200 (old lock) | 35.3 | 29.4 | −24.6 | 1.53 | 1.44 |
| **SMA100** ★ | 35.1 | 29.5 | **−16.4** | 1.66 | **2.14** |
| SMA50 | 29.7 | 23.6 | −19.1 | 1.55 | 1.56 |
| cross 50/200 | 31.9 | 26.5 | −33.3 | 1.30 | 0.96 |
| DD-from-1yr-high>10% | 31.5 | 26.0 | −31.3 | 1.24 | 1.01 |
| 3m-momentum<0 | 31.4 | 26.1 | −21.9 | 1.48 | 1.44 |
| volspike (ATR) | 33.9 | 27.0 | −33.4 | 1.40 | 1.02 |
| SMA200+vol | 33.8 | 27.6 | −20.9 | 1.54 | 1.61 |

**SMA100 wins** — ~same CAGR as SMA200, MaxDD −24.6→−16.4, Calmar
1.44→2.14. ATH-layer on SMA100: **+ATH≤10% entry additive** (35.2 /
29.3 / −15.1 / Sharpe 1.78 / **Calmar 2.33**); **+20% trail INERT**
(RS buffer rotates losers before −20%).
ATR/vol-spike (user idea) FAILED (−33% DD); NIFTYBEES has no true OHLC
so ATR is a close-to-close proxy — flagged, but weak as implementable.

## Phase 10 — drawdown hedge overlay

| Config | CAGR | post-tax | MaxDD | Sharpe | Calmar |
|---|---|---|---|---|---|
| SMA100→cash (Ph09 best) | 35.2 | 29.3 | −15.1 | 1.78 | 2.32 |
| **SMA100→beta-hedge hr1.0** | **42.8** | **34.0** | −22.7 | 1.83 | 1.89 |
| SMA100→beta hr0.5 | 37.8 | 29.5 | −24.9 | 1.58 | 1.52 |
| OFF no-hedge | 32.7 | 24.8 | −32.8 | 1.32 | 1.00 |
| always hedge 0.25/0.40/0.60 | 27.8→20.9 | 20.5→14.4 | ~−28 | <1.3 | <1.0 |

**Regime-triggered beta-hedge (in risk-off, keep the stocks, short 1×
Nifty) = highest post-tax CAGR of the whole project: 34.0%.** In gated
months it's long top-RS / short Nifty → harvests the RS spread as
market-neutral alpha instead of dead cash (per-year: 2023 +70 vs cash
+40; 2020 +108 vs +86; 2024 +63 vs +45). It does NOT reduce drawdown
(−22.7 vs cash −15.1; mid-cap β>1 under-hedged) — it is a **return
amplifier with contained DD vs ungated (−33%)**, not a DD reducer.
Permanent hedge = bleeds the bull, rejected. Covered calls = NOT
modelled / rejected (caps the right-tail that *is* the CAGR; the
rotating mid-cap holdings mostly lack liquid options — only ~22 of the
whole mid band is F&O).

## Phase 11 — stock-level vs market-level risk control

| Config | post-tax | MaxDD | Calmar |
|---|---|---|---|
| SMA100 mkt (Ph09 winner) | 29.3 | −15.1 | 2.32 |
| OFF + trail 15 / 12 / 10 | ~25 | ~−32 | ~1.0 |
| OFF + perStockSMA100 | 24.9 | −30.2 | 1.10 |
| perStockSMA only (no mkt) | 24.9 | −30.2 | 1.10 |
| SMA100 + perStockSMA | 29.6 | −15.1 | 2.35 |
| SMA100 + trail12 | 29.4 | −15.1 | 2.34 |
| **SMA100 + perStockSMA + trail12** | **29.6** | **−15.1** | **2.36** |

**Stock-level control alone cannot replace the market gate** (no-gate
variants stuck ~−30/−32%, Calmar ~1.0–1.1) — bottom-up stops fire only
*after* each name falls; in a broad bear that's too late for a
concentrated book. **On top of the gate they add a small, free gain**
(Calmar 2.32→2.36, +0.3pp post-tax, same −15.1% DD).

## Final — two recommended systems (risk choice)

Both supersede the original SMA200 lock (29.4% / −24.6% / Calmar 1.44):

1. **SMOOTHEST (best risk-adjusted):**
   `mid_120d_N15 + q0.5 + SMA100 regime + ATH≤10% entry +
   per-stock-SMA100 + 12% trail` →
   **29.6% post-tax / −15.1% MaxDD / Sharpe 1.80 / Calmar 2.36.**
2. **MAX RETURN:**
   `… + SMA100→beta-hedge hr1.0` (short 1× Nifty in risk-off vs cash) →
   **34.0% post-tax / −22.7% MaxDD / Calmar 1.89.**

Rejected (honest): ATR/vol regime, 20% trail (inert), permanent hedge,
hr0.5, covered calls, any stock-level-only control. Parent-study
caveats still bind (price-path quality ≠ fundamentals; PIT liquidity
proxy; LTCG not netted; no OOS on these refinements yet; no live
wiring). Files: `scripts/09_*,10_*,11_*`,
`results/phase09_*.csv / phase10_*.csv / phase11_*.csv`.
