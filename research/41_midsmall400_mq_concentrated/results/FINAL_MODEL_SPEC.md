# Final Model — Faithfully-Tested Spec (no options hedge)

Consolidated, decision-grade specification. Options-put hedge **dropped**
(no historical Nifty option/IV data in-house — verified: 15 days of 2026
snapshots only; not honestly backtestable). Everything below is
faithfully backtested on price data, 2014→2026 (~12.1y, incl. 2018/
2020/2022/2025 bears).

## Shared core (identical for all three; monthly)

| Element | Rule |
|---|---|
| Universe | Survivorship-free PIT **mid-cap liquidity band** = rank 101–250 by trailing-6-month median (close×volume), rebuilt every month from ~1,623 NSE daily symbols. Not index membership. |
| Signal | **Relative Strength** RSᵢ = (Pᵢ[t]/Pᵢ[t−120]) / (NIFTYBEES[t]/NIFTYBEES[t−120]); rank high→low. |
| Quality (q0.5) | Last 252 trading days → 12 consecutive 21-day blocks; keep only if **≥ 6 of 12 blocks** ended higher than they started. |
| Entry filter | Price **≥ 90% of point-in-time all-time high** (within 10% of ATH). |
| Hold | Top **15**, equal-weight. |
| Rotation | Monthly; **top-22 retention buffer** — a holding is sold only when it drops out of the top 22 by RS (low churn). |
| Costs / cash / tax | 0.4% round-trip on turnover; idle/cash 6.5% p.a.; post-tax = net 20% STCG on lots held < 365 days (LTCG not modelled). |

## The three candidates — differ only in the risk layer

| | Regime gate | Risk-off action | Stock-level exits |
|---|---|---|---|
| **SMOOTHEST** | NIFTYBEES vs its 100-day SMA, **checked WEEKLY** (Phase-15 result) | **Liquidate to cash @6.5%** until risk-on | per-stock-100-SMA exit **+ 12% trailing stop** (applied at month-ends) |
| **MAX-RETURN** | NIFTYBEES vs 100-SMA, **checked month-end** | **Stay invested + short 1× Nifty** notional, rolled monthly while risk-off; removed when risk-on | none |
| **FORTIFIED** | NIFTYBEES vs 100-SMA, **checked month-end** | **Stay invested + short 1× Nifty** | per-stock-100-SMA + 12% trailing stop |

Phase-15 note: a *weekly* regime check materially helps **only SMOOTHEST**
(daily-DD −24.1→−22.2, Calmar 1.52→1.65, CAGR flat; *daily* over-
whipsaws — 105 flips, CAGR falls). For MAX-RETURN/FORTIFIED the regime
clock is ~irrelevant (Calmar pinned ~1.0 across month/week/day) — their
drawdown is mid-cap-β>1 under-hedging in bears, not regime timing — so
they stay month-end.

## Headline performance (each on its own honest engine)

| System | post-tax CAGR @20% | MaxDD | Calmar | Final ×, 2014–26 | Character |
|---|---|---|---|---|---|
| **SMOOTHEST** (weekly) | ~30.2% | −22.2% daily-strict / −15.1% month-end | 1.65 d-strict / 2.36 m-end | ~40× | calmest compounder |
| **MAX-RETURN** | 34.0% | −22.7% (month-end engine) | 1.89 | ~75× | highest return |
| **FORTIFIED** | 33.7% | −22.3% | 1.92 | ~75× | Max-Return + per-stock brakes |

Nifty-50 over the same window ≈ **13% CAGR / ~4.7×**. All three ≈ **3×
the index CAGR**.

> Engine note: Smoothest's two MaxDD figures use different rulers —
> daily-marked (Phase 15, stricter) vs month-end (Phase 11). Max-Return/
> Fortified are month-end-engine (Phase 10/13). Compare each system's
> trade-off on its own ruler; do not cross-compare the decimals.

## Binding caveats

1. Price-path "quality" is **not** fundamentals; the index's Quality
   leg is not replicated — we beat its *return* via momentum.
2. PIT universe is a **liquidity proxy**, not point-in-time index
   membership (~68/100 overlap with today's MQ100).
3. The Nifty short (Max-Return/Fortified) is modelled **frictionless** —
   no futures roll/basis/margin, and 1× **under-hedges** mid-cap β>1,
   so live results would be somewhat worse than shown.
4. LTCG not netted; STCG@20% is.
5. No walk-forward OOS on the Phase 09–15 refinements (the parent
   RS study was OOS-validated; these overlays were tuned in-sample).
6. **Nothing is wired live.** Real-capital deployment is a separate
   decision and would need live paper-validation first.

## Status

3 named candidates locked; pick is the user's open decision. Options
hedge closed (no data). Files: `scripts/02,09,10,11,13,15_*.py`;
`results/REGIME_HEDGE_STOCKLEVEL_RESULTS.md`,
`FINAL_SYSTEMS_YEARLY_BREAKUP.md`, `phase15_decoupled_regime.csv`,
`final_systems_pl_overlay.png`; live log
`REGIME_ALTS_ATH_LAYER_HEDGE_DAILY_RUN_STATUS.md`.
