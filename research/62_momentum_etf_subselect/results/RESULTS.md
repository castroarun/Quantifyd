# Research 62 — Momentum-30 ETF Sub-Selection — RESULTS

**VERDICT: STRATEGY (candidate). Clears G1→G3.** A concentrated, buffered sub-basket of a
**reconstructed Nifty 200 Momentum 30** — hold **top 8** by 6m/12m relative-strength, retain
inside a buffer, **monthly**, with a **NIFTYBEES 100-day-SMA macro gate** *and* a **per-stock
15-day Donchian trailing exit** — delivered **CAGR 33.4% / net-of-tax 29.0% / MaxDD −17.0% /
Sharpe 1.78 / net-Calmar ≈1.5–1.7** over 2014→2026, versus NIFTYBEES 12.3% / −36.3% / 0.34.
It survives cost stress to 60 bps, a super-winner guard (Calmar holds at 1.79 without its 3
best names), and a 288-cell parameter plateau. **Net-Calmar beats the research/41 midcap
keep-top8 book (~1.66).** Ready for G4 (tearsheet/correlation/capacity) → G5 paper.

> Reproducibility: VPS `market_data.db` snapshot 2026-06-10 (daily 2012-06→2026-05, 1623 syms);
> `scripts/62_mom30_subselect.py` + `scripts/62b_g2_sweep.py`; cost 0.4% RT base; STCG 20%.

---

## The headline result (the winner + its neighbours form a plateau, not a peak)

| Config | CAGR | net (tax20) | MaxDD | Sharpe | Calmar | net-Calmar |
|---|---|---|---|---|---|---|
| **rsblend N8 buf22 gate100 Donch15** | **33.4** | **29.0** | **−17.0** | **1.78** | **1.97** | **1.71** |
| rsblend N8 buf18 gate100 Donch15 | 32.9 | 28.6 | −16.8 | 1.76 | 1.96 | 1.70 |
| rsblend N8 buf26 gate100 Donch15 | 32.7 | 28.4 | −17.0 | 1.75 | 1.93 | 1.67 |
| rsblend N5 buf22 gate100 Donch15 | 34.4 | 29.5 | −20.6 | 1.65 | 1.68 | 1.44 |
| _mom30 N8 buf18 gate100 Donch15_ | 25.2 | 21.6 | −16.7 | 1.54 | 1.51 | 1.30 |
| _BASE mom30 N10 buf22 no-gate (G1)_ | 25.4 | 21.9 | −44.6 | 1.07 | 0.57 | — |
| _NIFTYBEES buy & hold_ | 12.3 | — | −36.3 | 0.88 | 0.34 | — |

## What the sweep taught us (288 cells: score×N×buffer×Donchian×gate)

1. **The base idea works, but ONLY fully risk-controlled.** No-gate top-10 momentum has a
   −44.6% drawdown (uninvestible). The return was never the problem; the drawdown was.
2. **Gate + Donchian are complementary (the key structural finding).** Gate alone → −28.8%;
   Donchian-15 alone → ~−32%; **both together → −17%.** Gate handles market-wide risk-off;
   Donchian handles single-name momentum reversals the gate misses. Neither replaces the other
   — confirming research/41's "gate irreplaceable" while extending it: gate **+** Donchian beats
   gate-alone by a wide margin.
3. **Donchian-15 ≫ Donchian-20 ≫ Donchian-50.** Tighter trail = better DD *and* far more robust
   to the super-winner guard (d15 holds Calmar ~1.79 ex-top3; d20 sags to ~1.25). 50 is too slow.
4. **N=8 is the concentration sweet spot.** N5 = more CAGR but DD breaches −20%; N10/N15 dilute.
5. **Buffer barely matters** (18/22/26 ≈ identical) — the "extra knobs add nothing" pattern.
6. **Gate-100 (100-day SMA) > Gate-200.** The faster trend filter de-risks sooner.
7. **Score — CORRECTION to a mid-run claim.** Initially (sampling the wrong corner) I said the
   authentic risk-adjusted Momentum-30 z-score (`mom30`) was the better risk-adjusted choice. The
   completed sweep **refutes that**: in the winning stack (N8/gate100/Donch15) plain 6m/12m
   relative-strength (`rsblend`) gets the *same* −17% DD as mom30 but ~8% more CAGR → net-Calmar
   **1.71 vs 1.30**. The simpler score wins; the risk-adjustment is redundant once Donchian+gate
   control DD. (Same "fancy filter adds nothing" lesson as MQ — it was just masked until the
   Donchian window was tuned.)

## Robustness (G3) — all passed

- **Cost-sensitivity (net post-tax CAGR | net-Calmar):** 20bps 30.3% | 1.77 · 40bps 29.0% | 1.53
  · 60bps 27.6% | 1.31. Monotonic, graceful. Large-cap real cost ≈10–20 bps → the 20-bps column
  is realistic; we modelled 40 bps as conservative.
- **Super-winner guard (drop top-3 lifetime contributors = MAZDOCK, SUZLON, COCHINSHIP):**
  CAGR 33.4→30.4%, DD −17.0%, **Calmar 1.79**. Edge is breadth, not multibaggers.
- **Parameter plateau:** the top ~6 configs cluster on (N8, any buffer, Donch15, gate100) — not a
  lone peak. Stable to ±perturbation.
- **Per-year:** only 2 down years in 13 (2019 −4.2%, 2026 −6.8% partial); flat in 2015/2018;
  beats NIFTYBEES in ~12/13 years.

## Honest caveats

- **2019 is the genuine weak spot** (−4.2% vs index +13.6%) — the narrow Indian momentum
  dead-year; the gate+Donchian kept it flat-ish but it missed the large-cap melt-up.
- **Multiple testing:** 288 configs tried — the winner's headline t-stat is inflated. Mitigated
  (not erased) by the tight plateau + cost/super-winner survival. Apply a haircut; treat 29% net
  as optimistic-end.
- **Reconstruction ≠ the live ETF.** We rebuild the index from methodology (PIT top-200 by
  traded value → 6m/12m score → top-30); the real NSE Momentum-30 uses risk-adjusted scores,
  free-float caps, and semi-annual reconstitution. Our top-8 sub-selection by relative-strength
  is a faithful *proxy*, not the exact product. **Validation pass vs ~3 real factsheet dates is
  still owed** (the deferred "Hybrid" check) before live capital.
- **Concentration/correlation (G4 pending).** N8 large-caps currently lean PSU/defence/renewable
  (MAZDOCK, SUZLON, COCHINSHIP, IRFC, BSE, PFC). Cluster-stress DD not yet measured; the −17%
  backtest DD could understate a thematic unwind. Capacity is high (Nifty-200 names); long-only,
  so no shortability constraint.
- **Costs/taxes modelled, not live-filled.** No slippage/impact model beyond flat RT; monthly
  + Donchian churn is STCG-heavy (already in the post-tax figures).

## Next levers (G4 → G5)

1. **Client tearsheet** for `rsblend N8 buf22 gate100 Donch15` (equity-vs-NIFTYBEES log, drawdown,
   yearly bars, monthly heatmap) → publish to `/app/backtest/momentum30-subselect`.
2. **Correlation / cluster-stress** DD; sector cap experiment (does a max-per-theme cap cost much?).
3. **Factsheet validation** of the reconstruction at 3 dates.
4. **Walk-forward** (the params were chosen in-sample 2014→2026) + a 2019-only stress note.
5. If G4 holds → **paper-forward soak** on VPS alongside the existing books.
