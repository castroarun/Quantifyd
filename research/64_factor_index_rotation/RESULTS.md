# RESULTS — Nifty Factor-Index Rotation / Diversification (research/64)

**Verdict: research/63's "diversification beats selection" does NOT transfer to the Nifty
factor indices — they are too correlated (mean 0.65; 0.79–0.91 vs Nifty) to diversify, so
equal-weighting them tops out at Calmar 0.76 (best pure-factor book = Momentum+LowVol,
17.4% CAGR but −23% DD). The real win is COMBINING: use the best single factor (Momentum)
as the equity sleeve inside the research/63 asset trio and weight by inverse-vol →
Momentum + Gold + Nasdaq (inverse-vol), monthly = Calmar 1.77, CAGR 22.1%, MaxDD −12.5%,
cost-insensitive. That marginally beats research/63's Nifty-based book (1.75 / 19.7% /
−11.3%) by upgrading the equity sleeve and taming Nasdaq's vol. Label: STRATEGY
(candidate) — an INCREMENTAL upgrade to research/63, not a standalone factor edge. Adding
ALL factors dilutes it (1.18) — concentrate the equity sleeve into one factor.**

Windows: factor-only 2010-26 (192m, price-return indices); combined 2016-26 (ETF era).
Net 20 bps/side. Snapshot 2026-06-12. 56 configs.

---

## 1. The decisive G1 fact — factors are mostly beta

Monthly-return correlations (2010-26): mean off-diagonal **0.65** (research/63 asset trio
was ~0.0–0.25). Momentum/Quality/Value/Alpha cluster 0.71–0.90 and sit **0.79–0.91 vs
Nifty**. **Low-Vol is the lone diversifier (0.42–0.47** to everything, incl. Nifty).

## 2. The three families (net 20 bps)

| Family | Best config | CAGR | MaxDD | Calmar |
|---|---|---|---|---|
| (a) Factor rotation (2010-26) | top-3 blend + trend-gate-to-cash | 12.1% | −18.1% | 0.67 |
| (b) Factor baskets (2010-26) | **Momentum + Low-Vol, equal-wt** | 17.4% | −22.9% | **0.76** |
| (c) Combined factor+asset (2016-26) | **Momentum + Gold + Nasdaq, inverse-vol** | **22.1%** | **−12.5%** | **1.77** |
| Benchmark | NIFTYBEES buy & hold | 10.1% | −28.8% | 0.35 |
| Reference | research/63 Nifty+Gold+Nasdaq equal | 19.7% | −11.3% | 1.75 |

## 3. Winner vs research/63 (the "did we beat it?" answer)

| Metric | research/64 winner (Mom+Gold+Nasdaq invvol) | research/63 (Nifty+Gold+Nasdaq EW) |
|---|---|---|
| CAGR | **22.1%** | 19.7% |
| MaxDD | −12.5% | **−11.3%** |
| Calmar | **1.77** | 1.75 |
| Sharpe | **1.68** | 1.57 |

Marginal (1.77 vs 1.75 is within period-noise) but directionally real: **Momentum is a
better equity sleeve than plain Nifty, and inverse-vol down-weights Nasdaq's 24% vol.**

## 4. Robustness

- **Cost-insensitive:** Calmar 1.80 → 1.75 across 0/10/20/40 bps (low turnover book).
- **Per-year (winner vs Nifty):** beats Nifty **7/11 years**; worst year −4.2% (2022).
  2024 +23%, 2026 +27% excess.
- **Inverse-vol matters here:** Mom+Gold+Nasdaq equal = 1.48 vs inverse-vol = 1.77 (Nasdaq's
  high vol over-weighted by equal). Unlike the factor baskets where inv-vol ≈ equal.
- **Concentration finding:** AllFactors+Gold+Nasdaq = 1.18 (dilution); one factor + 2 assets wins.
- **Sanity:** research/63 book reproduced at 1.75 on this independent engine path.

## 5. Honest caveats

1. **Period dependence (biggest):** combined book is 2016-26 — same golden decade as
   research/63 (Nasdaq +24.6% INR, Momentum's strong run). Treat ~22% CAGR as an upper bound;
   the low-DD/Calmar property is the robust part.
2. **Mixed data:** Momentum is a PRICE-return INDEX (understates dividends ~1.5%/yr → its real
   edge over Nifty is *larger*); Gold/Nasdaq are ETF prices. Live version must use the Momentum
   ETF NAV (MOMOMENTUM/MOM30IETF, listed ~2022 → short history, recheck tracking/capacity).
3. **Marginal vs research/63:** the improvement (1.77 vs 1.75) is inside noise; the *defensible*
   claims are structural (factors don't diversify; Momentum > Nifty as a sleeve; inv-vol tames
   the book), not the 2nd-decimal Calmar.
4. **56 configs / multiple testing:** winner is structurally simple and on a stable plateau
   (any cross-asset combo with a momentum-ish equity sleeve lands ~1.6–1.8), but haircut the
   headline.
5. **Backtest, net of 20 bps. Synthetic 6%/yr cash leg** (fixes research/63's LIQUIDBEES≈0
   issue). Nothing live. Past performance ≠ future.

## 6. Next levers

1. Swap the Momentum INDEX for the Momentum ETF NAV (TR + real cost) once ≥5y history — confirm
   the sleeve upgrade survives the tradable instrument.
2. Add a **market trend overlay** (partial-cash when the book's assets are below MA) for the
   tail-risk the 2016-26 window never showed.
3. Test **Low-Vol as a 4th sleeve with a market gate** — it's the one in-set diversifier; may
   cut DD below −12% in a real risk-off.
4. Paper-forward the winner alongside the research/63 book (both trivial monthly crons).

---

*Reproducibility: `scripts/{g1_probe,g2_sweep,g2_finalize}.py`; factor indices via Kite tokens
(see g1_probe FACTORS); snapshot 2026-06-12; cost 20 bps. Verdict: **STRATEGY (candidate)** —
incremental upgrade to research/63; factor selection/diversification alone is SIGNAL not strategy.*
