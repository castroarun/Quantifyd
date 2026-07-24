# RESULTS — RSI 70/40 Momentum-Regime Timing (research/72)

**VERDICT: SIGNAL, not a clean STRATEGY for the stated goal.** The literal single-name RSI 70/40
system is **NO EDGE**; filters do not rescue it. Diversified across a universe, the RSI momentum-
regime becomes a **real, OOS-robust momentum-breadth signal** — but it faces a hard **return/drawdown
frontier**: you can beat the Nifty by ~2.8× on a broad universe *at roughly index-level drawdown*, OR
get lower drawdown on blue chips *at only ~1.5× return* — **no pure-RSI config delivers "beat the index
by ≥50% AND clearly lower drawdown" together.** The one config that technically clears the literal gate
does so on a razor-thin DD margin and rests on survivorship + small-cap-capacity flattery. Pushed to
where it works, the idea converges on the fund's **existing regime-gated momentum book (research/41/62)**
— with a *cruder* entry signal — so it adds no new alpha over what we already run.

Data snapshot: VPS `market_data.db` (5.24 GB), 2026-07-07. Window 2015-01-01 → 2026-07-07. Benchmark
**NIFTYBEES** buy-&-hold. Costs modelled as round-trip bps (default 15), gross vs net + cost-stress.

---

## 1. Phase A — single-name RELIANCE (G1/G2): NO EDGE

Literal ask (RSI14, enter ≥70, exit <40), net 15bps:

| System | Net CAGR | MaxDD | Calmar | Sharpe |
|---|---|---|---|---|
| RSI 70/40 (the ask) | **4.19%** | −23.3% | 0.18 | 0.36 |
| NIFTYBEES (index) | 10.95% | −36.3% | 0.30 | 0.79 |
| RELIANCE buy-&-hold | 17.13% | −45.1% | 0.38 | 0.73 |

**75-cell threshold×length sweep** (entry 60–80 × exit 30–50 × len 7/14/21): best `beat_bench_ratio`
= **0.97× — not one cell beats the index**; every cell loses to RELIANCE B&H; response surface is noisy
(no monotonic plateau). Lower DD comes only from sitting in cash (~37% exposure), not skill. RSI≥70
enters late in the move; RSI<40 exits after the drop. **Single-name RSI-regime timing = NO EDGE.**
(`results/phaseA_sweep.csv`)

## 2. Phase B — filter overlays (do MA/ADX/wRSI/ST/Donchian rescue it?): NO

On RELIANCE + an 8-name blue-chip basket, each filter added as a marginal entry gate. Decomposition of
any Calmar gain into "adds return" vs "just cuts exposure":

- Only **SMA-200 / weekly-RSI≥50** genuinely add return (~+1.0pp CAGR, ~+3.6pp lower DD, exposure barely
  touched) — but the basket still nets only **2.74% vs index 10.95% (~0.25×)**.
- **ADX / Supertrend / weekly-RSI≥60** improve Calmar almost entirely by **parking capital in cash**
  (exposure −7 to −9pp) — the "looks better because I stopped playing" illusion, not skill.
- **Donchian alt-exit is destructive** (whipsaw, 185 trades, net worse).
- **Not one filtered config beats NIFTYBEES.** (`results/phaseB_filters.csv`, `RESULTS_phaseB_filters.md`)

## 3. Phase C — slot-based portfolio (the real lever): 1 of 40 barely passes

N equal-capital slots; enter names on RSI≥ENTRY (highest-RSI first if oversubscribed), exit on RSI<EXIT;
Nifty50 & Nifty200 current membership (survivorship-biased), net 15bps.

| Config | Net CAGR | MaxDD | Calmar | Sharpe | beat | lowerDD | GATE |
|---|---|---|---|---|---|---|---|
| **nifty50 N20 60/30 (winner)** | **16.76%** | −23.85% | 0.70 | 1.14 | 1.53× | Y | **PASS** |
| nifty50 N15 60/30 | 16.13% | −22.93% | 0.70 | 1.07 | 1.47× | Y | no |
| nifty200 N20 70/40 | 16.35% | −29.06% | 0.56 | 1.03 | 1.49× | Y | no |
| NIFTYBEES | 10.95% | −36.34% | 0.30 | 0.79 | — | — | — |

Exactly **1 / 40** cells clears the literal gate (≥1.5× CAGR + lower DD), by 0.03. The **DD half is easy**
(36/40 beat the crash-heavy index DD by de-risking; Calmar ≈ doubles the index across the board). The
**return half kills nearly everything** (beat-range 0.21×–1.53×). Winner's edge concentrates in trend
years (2017/2020/2021/2024); it **lagged in 2025 (+2.2% vs index +11.7%)**. (`results/phaseC_portfolio.csv`)

## 4. Phase D — adversarial robustness (survivorship + OOS + params)

Winner config (N20, 60/30) on **today's Nifty50 (43 qualified)** vs a **broad 533-name 2015-vintage
universe** (survivorship-reduced — includes names that later fell out; still excludes truly delisted):

| Universe / window | Net CAGR | MaxDD | Calmar | Sharpe |
|---|---|---|---|---|
| Nifty50 full (survivorship-flattered) | 16.76% | −23.85% | 0.70 | 1.14 |
| **Broad 533 full** | **29.31%** | **−45.11%** | 0.65 | 1.12 |
| Broad — cost 30bps | 28.76% | −45.66% | 0.63 | 1.10 |
| Broad — IS 2015–2020 | 19.35% | −38.13% | 0.51 | 1.09 |
| **Broad — OOS 2021–2026** | **51.75%** | −44.08% | 1.17 | 1.68 |
| Nifty50 — OOS 2021–2026 | 19.57% | −20.99% | 0.93 | 1.27 |
| NIFTYBEES full / OOS | 10.95% / 11.84% | −36.3% / −16.1% | 0.30 / 0.73 |

Findings: (a) **the edge is STRONGER out-of-sample** (2021–26) than in-sample → *not* an overfit peak;
(b) parameter neighborhood on broad is a **plateau** (Calmar 0.59–0.75), not a lone spike; (c) but the
**broad book's drawdown blows out to −45%** — the high return is high-beta small/midcap momentum, and its
DD is *worse* than the index. The return/DD tension is structural. (`results/phaseD_robustness.json`)

## 5. Phase E — market regime gate (can we get BOTH?): only marginally

Added a NIFTY-200DMA market gate (go to cash when NIFTYBEES < 200-DMA), net 15bps. **Net CAGR only**
(the script's gross column is a known churn-inflated artifact; cost-stress ladder below is the real
cost-sensitivity):

| Config | Net CAGR | MaxDD | Calmar | Sharpe | beat | lowerDD | GATE |
|---|---|---|---|---|---|---|---|
| broad, no gate | 30.3% | −42.1% | 0.72 | 1.15 | 2.77× | No | no |
| broad, gate=block entries | 31.1% | −37.4% | 0.83 | 1.24 | 2.84× | No | no |
| **broad, gate=exit-all** | **30.4%** | **−35.3%** | 0.86 | 1.39 | 2.78× | **Yes** | **PASS** |
| broad, gate=exit-all, 30bps | 28.3% | −38.4% | 0.74 | 1.31 | 2.59× | No | no |
| broad, gate=exit-all, 50bps | 25.5% | −42.9% | 0.59 | 1.20 | 2.33× | No | no |
| broad gate exit-all — OOS 21–26 | 48.8% | −29.3% | 1.67 | 1.65 | 4.12× | No | no |
| nifty50, gate=exit-all | 9.5% | −23.1% | 0.41 | 0.81 | 0.86× | Yes | no (gate hurts blue chips) |

The 200DMA gate trims the broad DD from −42% → −35% while holding ~30% CAGR — the **one config that
technically clears both goals** (2.78× and −35.3% < −36.3%). But the DD margin is **razor-thin** (basically
index-level, not "lower drawdowns" in any meaningful sense), it **reverts to FAIL at 30bps cost**, and it
**fails lowerDD out-of-sample** (−29% vs index −16% in 2021–26). On blue chips the gate *hurts* (whipsaw).

---

## Honest caveats (biases, modelled assumptions, sample limits)

1. **Survivorship — the dominant caveat.** Both universes use names that *survived* to today with long
   history. Truly delisted/failed names are absent → the broad 29–30% CAGR is materially inflated. A
   point-in-time universe would lower it (unmeasured here).
2. **Capacity / liquidity — likely fatal for the high-return book.** The broad book's return comes from
   picking the highest-RSI names among 533, i.e. often illiquid small/microcaps. Entry/exit fills at the
   close are optimistic; at any real AUM, impact erodes it. **research/62 already established this exact
   pattern**: lower-cap momentum is a *gross-only mirage at size* (top-200 net-optimal). The 30% here is a
   retail-scale, un-capacity-checked figure and should be read as such.
3. **Single-factor / correlation.** All slots are long-momentum equities → one bet; the −35 to −45% DDs
   are the cluster-stress showing through even with N=20.
4. **Multiple testing.** ~40 (Phase C) + ~12 (Phase D) + ~14 (Phase E) configs tried; the "passing" cells
   are the best corners → deflate their claims. The literal-gate passes are 1-in-40 and 1-in-14 corners.
5. **No cash yield** on idle cash (0% modelled) — would modestly help the low-exposure / gated books.
6. **Costs** modelled as flat bps, no slippage/impact/STCG tax. STCG (holds are weeks–months) would cut
   net further for the high-turnover gated books.

## Verdict labels by phase

- Single-name RSI 70/40 (± filters): **NO EDGE / CONCLUDED.**
- Diversified RSI momentum-regime portfolio: **SIGNAL** (real, OOS-robust momentum breadth) — but **not a
  STRATEGY** meeting the stated dual objective; the return/DD frontier and capacity/survivorship caveats
  keep it from a clean G4 pass.

## Next levers (if pursued)

1. **Liquidity-floored, capacity-aware universe** (turnover ≥ ₹25cr, top-200/liquid-midcap) + market-
   impact model — the honest capacity test. Expectation (per research/62): the 30% collapses toward the
   blue-chip ~16% once tradeable. **Do this before believing the high number.**
2. **Vol-targeting / risk-parity sizing** instead of equal-weight, to force DD below the index while
   keeping return — the only realistic path to "both goals."
3. **Recognise the convergence:** this idea, at its best, *is* the regime-gated momentum book already in
   production (research/41 midcap RS-momentum, research/62 Momentum-30, Calmar ~1.7, live paper). RSI≥70 is
   a **cruder entry** than the RS-momentum ranking we already use. Highest-EV move = **improve the existing
   book**, not productionise RSI-regime, which adds no new alpha.

**Reproducibility:** engine `scripts/rsi_regime_engine.py`, portfolio `scripts/portfolio_engine.py`,
runners `run_phaseA_sweep.py` / `run_phaseD_robustness.py` / `run_phaseE_regime_gate.py`; all on VPS
`/home/arun/quantifyd/research/72_rsi_regime_7040/`. Snapshot 2026-07-07, NIFTYBEES benchmark, 15bps base.
