# Research 144 — True North (Momentum-30 Sub-Selection) Re-Assessment — RESULTS

**VERDICT: CONCLUDED — THE INCUMBENT STANDS. No change is adopted under the pre-registered
rule.** The deployed spec (NIFTYBEES-100SMA weekly liquidate-all gate, top-8, buffer-22,
Donchian-15, monthly rebalance) survived a 71-cell gate bake-off, a 27-cell action/frequency
sweep, a 240-run slots×exits sweep, and a 12-rebalance-offset robustness ensemble, all ranked
AFTER 20% STCG / 12.5% LTCG (fiscal-year loss-netted) and 0.3% costs. Two challengers beat the
+1pp CAGR margin (n5/Donch15: +2.0pp; n8/Donch20: +1.9pp) but **both fail the second adoption
condition** — W2 (2020→now) net-tax Calmar 1.29/1.28 vs incumbent 1.31 — and both carry
materially worse tail drawdowns (worst-offset DD −34%/−32% vs incumbent −28%). "No change" is
the honest outcome; the sub-threshold findings are documented below for Arun's discretion.

> Engine: `scripts/tn_sweep.py` — deployed-faithful (no-trim top-up rebalance per
> `momentum_paper.py` `live_rebalance_trim=False`, exact per-side costs, PIT traded-value
> universe, ETFs excluded). Windows: WA 2012→2026-09 (primary, all gate series defined),
> W1 2016-06→2019-12, W2 2020→now, W0 2006-04→now. Snapshot: VPS `market_data.db` 2026-09-03.
> All CAGR/DD figures below are NET of cost AND tax on WA unless marked gross.

## (a) Data hygiene — momentum engine audit

- **2026-01-15 phantom-row purge CONFIRMED on VPS** (0 rows remain).
- Residual phantom-signature rows (OHLC=prev close, vol 0) exist on **2025-03-18 (766 syms)**
  and **2024-01-15 (348 syms)** — but these are genuinely-untraded small-caps on days the
  market traded normally (RELIANCE/TCS/NIFTYBEES fine). **Benign for a Nifty-200 system**;
  flag for a future small-cap purge pass.
- `momentum_paper.py` indicators are NaN-robust: `_gate_risk_off` and `_donchian_low` dropna()
  per-series before rolling; `_rs_basket` ffills and per-symbol NaN-checks. No union-alignment
  NaN poisoning path found.
- Gate series current: NIFTYBEES to 2026-09-03 (same-day). NIFTY50/500/MIDCAP150/SMLCAP250
  exist 2011→2026-08-28 (4 sessions stale — irrelevant for research, but note if ever used live).
- **Live observation (not a defect):** on 2026-09-03 NIFTYBEES closed 272.40 vs SMA100 272.95 —
  marginally risk-OFF. The paper book correctly still shows gate ON because the check is weekly
  (Friday). Expect a possible liquidation at this week's check if it stays below.

## (b) Gate bake-off (71 cells: 5 series × 14 constructions + no-gate; N8/D15/weekly/cash fixed)

**The inherited gate won its own bake-off.** Net-tax WA, top rows + notable losers:

| Gate | CAGR | MaxDD | Calmar | W1 | W2 | 2006+ DD |
|---|---|---|---|---|---|---|
| **NIFTYBEES SMA100 (INCUMBENT)** | 20.9 | **−23.7** | **0.88** | 15.3 | 27.3 | **−23.7** |
| NIFTY50 SMA100 (near-twin) | 20.6 | −23.7 | 0.87 | 15.5 | 28.2 | −52.0¹ |
| NIFTYBEES EMA100 | 21.1 | −27.6 | 0.77 | 13.8 | 27.6 | −27.6 |
| NIFTYBEES XO20/100 | 18.2 | −26.0 | 0.70 | 13.3 | 27.4 | −26.0 |
| NIFTYBEES DD15 (drawdown gate) | 22.3 | −39.6 | 0.56 | 15.3 | 35.5 | −39.6 |
| NO GATE | 23.9 | −46.5 | 0.51 | 15.1 | 39.7 | −52.0 |
| NIFTYBEES B&H benchmark | 12.7 | −36.3 | 0.35 | 13.2 | 11.8 | −59.7 |

¹ index series start 2011 → risk-on through 2008; only NIFTYBEES-based gates protect 2008.

Findings: (1) no alternative SERIES beats NIFTYBEES — broader/mid/small-cap gate series are all
worse; (2) no alternative CONSTRUCTION beats SMA100 on DD-constrained CAGR — the OA-style
DD-from-high gates (DD8-15) and momentum-negative gates that worked for ATH-breakout are WRONG
for this book (−34..−47% DDs): a momentum book needs the faster trend gate; (3) MA crossovers
are the worst family. The r/41→r/62 inheritance was, by luck or wisdom, already optimal.

## (c) Gate action, check frequency, slots, exits — with offset-robustness bands

**Action × frequency** (n8/D15, net-tax WA, offset 0): block-new-entries-only (21.12/−22.05/
Cal 0.96) and cash+monthly-check (21.24/−21.61/0.98) both nudged past the incumbent
(20.90/−23.67/0.88); halve-exposure adds CAGR but fails DD (−33); daily liquidation is worse
(whipsaw); block is check-frequency-invariant.

**BUT the 12-offset ensemble (rebalance anchored 0..11 trading days before month-end — the
deterministic analogue of OA's seed variance) REVERSED the offset-0 ranking:**

| Config (net-tax WA) | CAGR med [min..max] | DD med / worst | Cal med | W1 med | W2 med |
|---|---|---|---|---|---|
| **INCUMBENT cash n8 D15** | 20.7 [14.9..25.1] | −25.1 / **−28.3** | 0.88 | 13.6 | 27.3 |
| BLOCK n8 D15 | 20.4 [14.2..24.9] | −23.6 / −31.9 | 0.90 | 13.7 | 27.8 |
| CASH monthly-check n8 D15 | 20.7 [14.4..25.5] | −23.9 / −30.1 | 0.90 | 13.2 | 28.5 |
| CASH n5 D15 | **22.7** [14.9..28.9] | −27.5 / −34.3 | 0.91 | **17.9** | **31.6** |
| CASH n8 D20 | **22.6** [16.0..26.1] | −27.5 / −32.2 | 0.88 | 18.2 | 31.2 |
| BLOCK n5 D15 | 21.8 [13.8..28.3] | −28.9 / −36.8 | 0.82 | 16.8 | 30.8 |

- **Block-new-only vs liquidate-all (Arun's priority #1):** across offsets they are a WASH
  (median 20.4 vs 20.7). Block's tax-churn saving is real but small — median gross→net drag
  4.13pp/yr (incumbent) vs 3.91pp/yr (block), i.e. **~0.22pp/yr saved** — because FY loss
  netting absorbs most of the mass-realization hit and the Donchian stops realize most
  positions anyway. Block's median DD is ~1.5pp better but its WORST-offset DD is 3.6pp worse
  (−31.9 vs −28.3): retained positions ride corrections the gate would have side-stepped.
  Cash-yield fairness checked: at 5% instead of 6.5% idle yield, incumbent 20.9→20.0, block
  21.1→20.2 (offset 0) — block is NOT rescued or hurt by the cash-yield assumption.
  **Verdict: no adoption; the incumbent's worst-case is the best of all six finalists.**
- **Exits:** Donchian-15 is the best-DD exit at every n (r/62 confirmed under every gate
  variant tested — the OA "exits flip when the gate changes" risk did NOT materialize here).
  ATR(20)×3 trail and per-stock SMA50/100 trails are strictly worse (DD −31..−41). Donch20
  gains CAGR at DD cost (see table). Donch-none confirms the stop is load-bearing (DD −36+).
- **Slots:** n5 lifts CAGR ~+2pp after-tax with ~+2.4pp median DD and −34% worst-offset tail;
  n≥10 dilutes CAGR AND worsens DD (W1 collapses to 9-12%). The OA lesson "wider book lifts
  the worst case" does NOT transfer to deterministic rank-based momentum — width only dilutes.
  n8 remains the sweet spot for the stated −30% DD tolerance.

## (d) After-tax adoption test (pre-registered: >1pp median net-tax CAGR across 12 offsets AND W1+W2 Calmar ≥ incumbent AND plateau)

| Challenger | ΔCAGR med | W1 Cal (inc 0.54) | W2 Cal (inc 1.31) | Adopt? |
|---|---|---|---|---|
| CASH n5 D15 | **+2.01pp** ✓ | 0.70 ✓ | 1.29 ✗ (−0.02) | **NO** — fails W2 Calmar; worst-offset DD −34.3 |
| CASH n8 D20 | **+1.92pp** ✓ | 0.62 ✓ | 1.28 ✗ (−0.03) | **NO** — fails W2 Calmar; worst-offset DD −32.2 |
| BLOCK n8 D15 | −0.34pp ✗ | — | — | NO |
| CASH monthly n8 D15 | −0.01pp ✗ | — | — | NO |

Sensitivities (offset 0, net-tax): cost 0.5% RT costs ~1.3pp CAGR uniformly (no ranking flips);
cash yield 5% vs 6.5% costs ~0.9-1.0pp uniformly (no flips). Gross figures for every cell are
in the CSVs (tax0 rows); tax drag is 3.9-4.6pp/yr across finalists.

## Open Alpha blend (Arun's priority #2)

50-50 monthly-rebalanced, both legs AFTER-TAX, 30 OA seeds (adopted spec: trail-15 SMA, −8%
stop, 16 slots @6.25%, no gate, 25bps, cash 5%), window 2006-04→2026-09:

| TN leg | Blend CAGR med [min..max] | Blend DD med | Calmar | corr daily/monthly | TN solo | OA solo med |
|---|---|---|---|---|---|---|
| **cash (incumbent)** | **27.4** [26.7..28.5] | **−16.4** | **1.68** | 0.40 / 0.54 | 19.5 / −23.7 | 34.9 / −26.4 |
| block | 27.2 [26.5..28.3] | −16.4 | 1.67 | 0.43 / 0.57 | 19.1 / −22.1 | " |

**Answer: the softer gate action makes TN a slightly WORSE blend partner, not better** — block
stays invested through corrections, so its correlation to OA RISES (0.43/0.57 vs 0.40/0.54)
and the blend gains nothing. The incumbent's hard risk-off cash is exactly what diversifies
the always-in OA leg. The blend itself is the standout portfolio fact: **~27.4% after-tax CAGR
at −16.4% DD (Calmar 1.68)** — a better DD than either leg alone (TN −23.7, OA −26.4), robust
across all 30 seeds (blend DD range just −15.9..−16.7).

## (e) Recommendations

1. **ADOPT: nothing.** The deployed True North spec stands as-is — gate series, construction,
   action, frequency, n=8, buffer 22, Donchian-15 all survived their sweeps. No change to
   `services/momentum_paper.py`.
2. **REJECT: gate replacements** (all series/constructions/action/frequency variants), ATR and
   SMA-trail exits, n≥10 books, halve-exposure gate. Evidence in phases A-D.
3. **ARUN'S DISCRETION (sub-threshold, documented, NOT recommended):** if he ever wants a
   CAGR-tilted variant and accepts −32..−34% worst-case DD, CASH n5/D15 (+2.0pp after-tax) or
   CASH n8/D20 (+1.9pp, best worst-offset CAGR floor 16.0%) are the two honest candidates.
   They fail the pre-registered W2-Calmar condition by 0.02-0.03 — real money should not move
   on that; a paper A/B sleeve could.
4. **BLEND (needs Arun's sizing decision, separate study):** the 50-50 TN+OA book is better
   than either alone on every risk measure. If pursued, register it as its own book with its
   own STATUS doc; keep the TN leg's incumbent liquidate-all gate (better diversifier).
5. **Watch this week:** NIFTYBEES is a hair below its 100-SMA — the live paper book may
   correctly liquidate at the Friday weekly check. No action needed; just don't be surprised.

## (f) What was NOT tested, and why

- **Score/ranking variants** (12−1 momentum, risk-adjusted z-blend, lookback sweeps): settled
  by r/62 (rsblend beat mom30 z-score on the winning stack) and r/75 (score choice ≈
  immaterial); re-testing would be multiple-testing without a new hypothesis.
- **Buffer sweep**: r/62 showed 18/22/26 indistinguishable; held at round(2.75n).
- **Official-constituent PIT universe**: the real Nifty-200 membership history isn't
  reconstructable; the traded-value proxy is the honest stand-in (stated divergence from the
  deployed official-list universe).
- **Intra-month re-entry after Donchian stops** and **gate re-entry faster than month-end**:
  r/62 tested weekly re-entry (hurts); not re-run.
- **Slippage/impact beyond flat 0.3%** (0.5% stressed): large-cap book at ₹20L — capacity not
  binding.
- **LTCG ₹1.25L exemption and loss carry-forward**: ignored (both directions small, net
  conservative).
- **OA blend at other weights / with the n5 TN leg**: out of scope of the asked 50-50 question.

## Seven deadly sins — how controlled

Look-ahead: causal ranks/indicators, shift(1) stops, same-close execution as the live engine.
Survivorship: PIT traded-value universe. Overfitting: metric + margin pre-registered before
any run; 12-offset ensemble; two validation windows; incumbent-default-wins (and it did).
Cost neglect: 0.3% baked, 0.5% stressed, tax modeled with FY netting. Regime: W1 momentum
dead-zone + W2 + 2008 (W0) all reported. Correlation: blend corr measured (0.40 daily to OA).
Capacity: Nifty-200 large caps at ₹20L — non-binding.

## Reproducibility

VPS `/home/arun/quantifyd/research/144_truenorth_reassessment/`: `scripts/tn_sweep.py`
(phases smoke/A/B/C/D), `scripts/tn_blend.py`, results in `results/phase[A-D]_*.csv`,
`results/blend_oa.csv`, `results/phaseD_peryear.csv`, finalist NAVs `results/nav_*_tax1.csv`.
Data: market_data.db snapshot 2026-09-03. Total compute ~25 min.
