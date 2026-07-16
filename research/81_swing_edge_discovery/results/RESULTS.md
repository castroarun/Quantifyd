# research/81 — Swing Edge Discovery: RESULTS

**VERDICT: SIGNAL (not investable as tested) — real edge, visibly DECAYING through
the out-of-sample window.** The gap-up + opening-range-breakout long system is
genuinely there (IS t=5.6 across 77 F&O names; OOS trade-level still net-positive,
t=2.57) but its per-trade edge halved out-of-sample and turned NEGATIVE in 2026
(+33 bps 2024 → +5 bps 2025 → −27 bps 2026). Neither pre-declared book passes the
acceptance gates on OOS. Do NOT deploy; paper-monitor at most. This is the OOS
look doing exactly its job — IS+Val alone would have funded the 2026 bleed.

## Study scope (2026-07-15 → 07-16)

Brief: `docs/Trading-sytem-research-prompt-fable.md`. ~170 pre-registered cells
across 8 families + 6 portfolio constructions + 1 OOS touch, all logged in
`EDGE_DISCOVERY_81_STUDY_STATE.md`. Data: 5-min backfill 2015→2024 for 381
symbols executed as part of the study (+ splice/adjustment repair of 6 corrupted
F&O series + BANKNIFTY 2015+ via index token).

## League table (all net of futures-proxy costs)

| System | IS | Val/OOS | Verdict |
|---|---|---|---|
| Gap≥0.25% + ORB long ≤4d (stocks W12, 77 F&O) | +20.9bps t=5.62 | OOS +10.4bps t=2.57, decaying 2024→2026 (+33/+5/−27) | **SIGNAL, decaying** |
| Same, NIFTY index W12 | +29bps t=3.0 | Val +15bps; OOS n=52 (<100 gate), +21bps t=1.25, 2026 negative | **SIGNAL, insufficient OOS n** |
| G4 10-name book (Sharpe 1.00, DD −17% on IS+Val) | passed ex-Calmar | **OOS CAGR −1.6%, Sharpe −0.10** | **FAILS OOS** |
| Breadth book (6 constructions) | best Sharpe 0.58, DD −42% | OOS +0.9% CAGR | **capacity-constrained + FAILS** |
| O=L break long (D2) | t=3.2 on megacaps | breadth: t=0.87 | does not generalize |
| CPR-open long (C2) | t=2.4 megacaps | breadth: negative | does not generalize |
| Deep-z reversion short (B1/B2) | +32bps t=1.5, 8-10/13yr | not taken further | SIGNAL, thin |
| Donchian daily / squeeze / EOD-carry / 5d-RS / PDH-PWH / MA crosses / first-candle coin-toss (A7) | all negative | — | **NO EDGE** (7 families killed) |

## Key learnings (worth more than the backtests)

1. **Every short-side mirror of every setup loses net** on Indian liquid names at
   2-4d horizons — the market's long drift + costs make short swing untenable here.
2. **Morning-strength continuation was the only real anomaly family** — and even
   it decays: strongest 2017-2021, weaker 2022-2024, gone/negative 2025-2026.
   Likely crowded away (ORB is retail-mainstream post-2023).
3. **OR-width inverts equal-risk sizing**: narrow-range breakouts lose; sizing by
   1/stop-width loads up exactly on them. Equal-notional is the safe default.
4. **Breadth ≠ implementable**: 12 signals/day × 4-day holds ≈ 3× any sane book;
   which-subset-you-take dominates results. Trade-level t-stats do not equal a book.
5. **Data landmines found & fixed**: 6 F&O symbols carried mixed split-adjustment
   bases (KOTAKBANK showed a fake 401% jump); catchup path can't fetch index
   tokens (NIFTY50 5-min still ends 2026-03-25 — repair queued).

## Honest caveats

- Universe survivorship-biased (today's 381 names) — per-trade OOS decay is if
  anything UNDERSTATED by this.
- Futures modeled from cash (user-approved proxy); slippage 1-3bp assumed.
- Index OOS has n=52 (< the 100-trade gate) and truncates 2026-03 (data gap).
- 2026 negativity is 6.5 months of data — decay vs regime not fully separable;
  the label stays SIGNAL either way.

## Next levers (not started)

- Decay-gate research: regime filter that would have exited the family in 2025
  (e.g. rolling 12m trade-level t < 0 → flat) — pre-register before testing.
- OR-width quality filter (discovery #3) as an entry refinement.
- B-family (deep-z short fade) 5-min timing on the now-deep broad universe.
- Index catchup fix via token map (NIFTY50/INDIAVIX post-2026-03 5-min).

**OOS ledger: consumed 2026-07-16 for the ORB family (index cell, stock cells
W6+W12, G4 book, breadth v6 book). No further OOS looks for these systems.**

Reproducibility: engine @ research/81_swing_edge_discovery/engine (32 unit
assertions); data snapshot = VPS market_data.db as of 2026-07-16 22:40 IST;
scripts in scripts/; all runs logged in experiment STATUS files.
