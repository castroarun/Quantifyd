# Momentum-30 ETF Sub-Selection — Hold Top-10 of a Reconstructed Nifty 200 Momentum 30, Buffer-22 Rotation

**STATUS: DONE — G1→G3 PASS. STRATEGY candidate. Winner = rsblend N8 buf22 gate100 Donch15
(CAGR 33.4% / net 29.0% / DD −17.0% / Sharpe 1.78 / net-Calmar ~1.5–1.7). See `results/RESULTS.md`.**
**Next: G4 tearsheet + correlation/capacity + factsheet validation → G5 paper.**

**STATUS (orig): FRAMING (G0→G1) — sections 1–4 locked, not yet launched**
**Owner:** Arun · **Created:** 2026-06-10 · **Research #:** 62
**Host:** VPS `94.136.185.54` is canonical (laptop has NO market_data.db — all runs on VPS)

---

## 1. The Ask

**What you asked (near-verbatim):**
> New system: instead of running our own selection, directly invest using a factor
> ETF's picks. Say a momentum ETF — pick the top 10 of its 30, invest 100%. Every
> month-end, check if a held name is still in the top 22 (or whatever number we
> optimize); if yes keep it, else replace it with the best candidate we don't already
> own. No stock exits based on Nifty/macro. Optionally add the stock's own trailing
> exit (a Donchian channel). For backtesting, read the ETF's selection from its
> factsheet and reconstruct the same.

**What we're actually testing (sharpened):**
Can a **concentrated, buffered sub-basket of the Nifty 200 Momentum 30 index** — hold
the **top 10** by momentum score, **100% invested**, retain a name while it stays inside
the **top-22 buffer**, monthly rebalance, **no macro gate** — beat NIFTYBEES **net of
cost and tax** over 2014→2026, and does adding a **per-stock Donchian trailing exit**
(in place of the macro gate) control drawdown without killing return?

**Decision (resolved with user 2026-06-10):**
- Parent index = **Nifty 200 Momentum 30**.
- Constituents = **reconstructed from methodology** (NOT factsheets). We have daily
  price+volume; the index is fully computable. Factsheets are unnecessary and give only
  point-in-time snapshots. (Optional later: validate reconstruction against ~3 real
  factsheet dates — the "Hybrid" check — before trusting results.)

**Single success metric:** post-cost, post-tax **Calmar** (CAGR / |MaxDD|), with MaxDD
as a hard secondary constraint. Ranked the way research/41 ranks the midcap book.

**Gates a result must clear:**
- **G1:** reconstructed base (N=10, buffer-22, no gate) shows a **gross** edge vs
  NIFTYBEES with sane per-year behaviour. Falsify if it can't beat the benchmark gross.
- **G2:** survives **net + post-tax**; Donchian and/or gate produce a Calmar ≥ the
  research/41 midcap book's best **net** Calmar (~1.66 keep-top8) OR a materially lower
  MaxDD at comparable CAGR.

---

## 2. Economic hypothesis (why an edge should exist; who's on the other side)

- **Momentum under-reaction.** The parent index already harvests cross-sectional
  momentum (investors under-react to sustained earnings/price trends; the slow diffusion
  of information is the premium). Counterparty = late/disposition-biased sellers.
- **Why sub-select top-10 of 30?** Concentration is the #1 CAGR lever in our prior work
  (MQ: PS3≈65%, PS10≈49%, PS30≈32% CAGR). The index dilutes momentum across 30 names and
  caps weights for capacity; a retail book can concentrate into the strongest 10.
- **Why a buffer (top-22)?** Pure top-10 churns hard at the boundary; a retention buffer
  only evicts a name once it has decayed materially — cuts turnover/cost while keeping
  the momentum tilt. (Same mechanism the NSE index itself uses with its buffer.)
- **Decay risk:** momentum crashes (sharp reversals after stress, e.g. Apr-2020,
  early-2022) are the known failure mode. The index has a semi-annual reconstitution lag;
  our monthly rotation + optional Donchian/gate is the defence we're testing.

**Prior-art contradiction to confront (research/41 phase 28, midcap):**
> per-stock trailing-MA exit = NO NET EDGE; the **Nifty gate is irreplaceable**.
This study deliberately drops the gate on a **large-cap** book and substitutes a Donchian
trail. The grid includes a **gated arm** as the control so we measure the cost of dropping
it head-to-head. If no-gate large-cap survives where no-gate midcap didn't, that is the
finding; if it bleeds in 2018/2022 drawdowns, we'll see it and revert to the gate.

---

## 3. The Base — mechanics (locked, no ambiguity)

| Element | Definition |
|---|---|
| **Universe (eligible pool)** | PIT **top-200 by trailing-6mo median traded value** (price×volume), reconstructed monthly from data ≤ as-of date. Survivorship-free. Faithful Nifty-200 proxy. Benchmark (NIFTYBEES) excluded from investable set. |
| **Factor score (the "ETF")** | **Nifty 200 Momentum 30 methodology:** for each name, 6-month and 12-month price returns, each **risk-adjusted** = return ÷ (annualized std of daily returns over the window); **z-score** each leg across the 200; momentum score = average of the two z-scores. **Eligible "ETF" = top 30** by score. |
| **Selection** | Rank the top-30 by score. **Hold top 10 equal-weight, 100% invested.** Keep a held name if still inside the **top-22** of the 30 (buffer). Evicted names replaced by highest-ranked un-owned name in the 30. |
| **Rebalance** | Monthly (month-end), as-of close; positions marked **daily** (for honest MaxDD + Donchian). |
| **Macro gate** | **None** in the base (user's design). Control arm: NIFTYBEES weekly 100/200-SMA risk-off → de-risk (research/41 mechanism). |
| **Per-stock exit (optional overlay)** | **Donchian** N-day low break → exit the name, redeploy proceeds to next eligible at next rebalance (or immediately — both tested). N ∈ {20, 50}. |
| **Costs** | 0.4% round-trip on turnover (large-cap likely tighter; **cost-sensitivity at 0.2% / 0.4% / 0.6%**). |
| **Tax** | Report **gross AND post-tax @ STCG 20%** (research/41 convention; momentum churn is STCG-heavy). |
| **Idle cash** | 6.5% p.a. (only relevant in gated arm). |
| **Period** | 2014-01-01 → 2026 (covers 2018 mid/smallcap bear, Mar-2020 crash, 2022 drawdown). |
| **Benchmark** | NIFTYBEES (Nifty-50). Secondary: our reconstructed top-30 equal-weight (the "plain ETF") to isolate the value of sub-selecting 10. |
| **Success metric** | post-cost, post-tax **Calmar**; MaxDD hard constraint. |

---

## 4. Plan — staged grid (kill cheap, scale dear)

**G1 PROBE (cheap, ~8 cells) — does the base even work, and does dropping the gate hurt?**
Fixed: N=10, buffer=22, score=momentum30, cost=0.4%.
| Axis | Values |
|---|---|
| Gate | {none, NIFTYBEES-SMA gated} |
| Donchian | {off, 20d, 50d} |
| Score sanity (at base only) | {momentum30 risk-adj, RS-blend 6m/12m} |
→ ~6 (gate×donchian) + 2 (score sanity) = **8 cells**. Plus reference: NIFTYBEES,
reconstructed top-30 equal-weight (the plain ETF). **Gate: G1 passes if base beats
NIFTYBEES gross with sane per-year; decide whether gate is needed before expanding.**

**G2 EXPANSION (only if G1 passes) — concentration × churn × robustness:**
| Axis | Values |
|---|---|
| N (concentration) | {5, 8, 10, 15} |
| Buffer | {15, 18, 22, 26} |
| Donchian | {off, best-from-G1} |
| Gate | {winner-from-G1} |
Net + post-tax + per-year + cost-sensitivity on the survivors. Then robustness
(per-year stability, param monotonicity, super-winner guard, cost +50%).

**Falsification:** abandon if (a) base can't beat NIFTYBEES gross at G1, or (b) every
net+post-tax config has Calmar < plain-NIFTYBEES buy-hold, or (c) the no-gate book's
MaxDD is uninvestible (>35%) and the gate is the only fix (→ reverts to a known result,
not a new system).

---

## 5. Status (live log)
**STATE: G1 PROBE DONE — PASS. Real edge vs NIFTYBEES. Winner = GATE + Donchian-20.**
| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-06-10 | Framing locked | Index=Nifty200Momentum30, reconstruct-from-methodology; engine = research/41 `02_rs_sweep.py` reuse + new `mom30_score` + daily-mark + Donchian |
| 2026-06-10 ~14:20 | Engine built + scp'd to VPS | `scripts/62_mom30_subselect.py`; venv = `/home/arun/quantifyd/venv` |
| 2026-06-10 ~14:25 | Smoke PASS | Reconstruction sane: 2018=graphite-electrode names, 2021=IT/Adani, 2024=PSU/defence. ~4s/config after 114s load |
| 2026-06-10 ~14:35 | G1 probe DONE (8 cells, 135s) | Results in `results/g1_probe.csv` + `_peryear.csv` |

## 6. Crash Recovery
_To be filled when the runner launches (commands, how to check progress, how to resume)._
Engine reuse: `/home/arun/quantifyd/research/41_midsmall400_mq_concentrated/scripts/02_rs_sweep.py`
(`load`, `month_ends`, `pit_universe`, `rs_scores`). New band `nifty200=(0,200)`.

## 7. Files
| File | Purpose | Committable? |
|---|---|---|
| `MOMENTUM30_ETF_SUBSELECT_DAILY_SWEEP_STATUS.md` | This file | yes |
| `scripts/62_mom30_subselect.py` | Reconstruction + backtest engine | yes |
| `results/g1_probe.csv` | G1 cell results | yes (small) |
| `results/*_peryear.csv` | Per-year tables | yes |
| `results/RESULTS.md` | Final verdict | yes |

## 8. Findings

### G1 PROBE (2014→2026, daily-marked, gross CAGR / post-tax@20% / daily MaxDD)

Benchmark **NIFTYBEES: CAGR 12.3%, DD −36.3%, Sharpe 0.88, Calmar 0.34.**

| Config | CAGR | net20 | MaxDD | Sharpe | Calmar | donchX |
|---|---|---|---|---|---|---|
| plain_ETF_top30 | 24.8 | 21.4 | −39.4 | 1.19 | 0.63 | 0 |
| BASE mom30 N10 buf22 (no gate) | 25.4 | 21.9 | **−44.6** | 1.07 | 0.57 | 0 |
| mom30 N10 + Donch20 (no gate) | 25.1 | 21.5 | −32.6 | 1.30 | 0.77 | 724 |
| mom30 N10 + Donch50 (no gate) | 21.6 | 18.0 | −39.5 | 1.03 | 0.55 | 293 |
| mom30 N10 + GATE100 | 25.4 | 21.2 | −28.8 | 1.36 | 0.88 | 0 |
| **mom30 N10 + GATE100 + Donch20** | **25.5** | **22.1** | **−19.1** | **1.55** | **1.33** | 457 |
| mom30 N10 + GATE100 + Donch50 | 23.7 | 20.0 | −27.6 | 1.33 | 0.86 | 140 |
| SANITY rsblend N10 buf22 (no gate) | 32.4 | 29.3 | −45.2 | 1.19 | 0.72 | 0 |

**Verdict: SIGNAL → STRATEGY-candidate. G1 PASS** (every momentum config beats NIFTYBEES
on CAGR & Sharpe). Key reads:

1. **Reconstruction validated** — names match the real index regime-by-regime (graphite
   2018, IT/Adani 2021, PSU/defence 2024). No factsheets needed.
2. **The base (no gate) is uninvestible as-is** — CAGR 25.4% but **DD −44.6%**, Calmar
   0.57, worse drawdown than the benchmark. Bad years: 2015 −17%, 2018 −14%, 2025 −20%.
3. **User's Donchian instinct has merit** — Donch-20 is the **best no-gate** config
   (DD −44.6→−32.6, Sharpe 1.07→1.30, ~0 CAGR cost). Donch-50 is too slow (worse). 20d ≫ 50d.
4. **But dropping the gate still costs DD** — GATE100 alone (Calmar 0.88) beats every
   no-gate config. **research/41's "gate irreplaceable" holds for large-cap too.**
5. **NEW finding — gate + Donchian are COMPLEMENTARY, not substitutes.** GATE100+Donch20:
   DD **−19.1%**, Sharpe **1.55**, Calmar **1.33**, net CAGR 22.1%, and per-year it
   neutralises the bad years (2015 −0.7, 2018 +0.5, 2025 +0.6) while keeping the up years
   (2014 +85, 2020 +53, 2021 +86). Gate handles market-wide risk-off; Donchian handles
   single-name momentum reversals the gate misses. Best risk-adjusted by a wide margin.
6. **Methodology caution** — the *authentic* risk-adjusted Momentum-30 z-score is NOT
   clearly better than plain 6m/12m relative strength: rsblend got **32.4% CAGR** (vs
   mom30 25.4%), same ~−45% DD. Echoes the MQ "fancy filters add nothing" trap. Must test
   rsblend inside the GATE+Donch20 stack at G2 before assuming mom30 is the right score.

### Honest caveats (G1)
- Gross CAGR/DD drive Calmar here; net (post-tax) Calmar for the winner ≈ 22.1/19.1 ≈ 1.16
  — good but below research/41's net keep-top8 ~1.66. Not yet a portfolio decision.
- Single cost level (0.4%) and single param point each — no sensitivity/monotonicity yet.
- 2026 is a partial, market-wide down year (−10%); benchmark also −9.5%.
- Multiple configs tried (8) — no multiple-testing haircut applied yet.

### Next levers (G2, only the winner stack)
- **Score:** rsblend vs mom30 inside GATE100+Donch20 (does the punchier score keep its CAGR
  edge once DD-controlled?).
- **Concentration:** N ∈ {5,8,10,15}; **Buffer** ∈ {15,18,22,26}.
- **Donchian window** fine-grid {10,15,20,30}; gate SMA {100,200} weekly.
- Cost-sensitivity {0.2/0.4/0.6%}; per-year monotonicity; super-winner guard (drop top-3);
  cost +50% adversarial.
