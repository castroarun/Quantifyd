# MidSmallcap400-MQ Concentrated Rotation — Detailed Methodology & Results Report

Single comprehensive reference for the research/41 study. For the short
verdict see `MIDCAP_RS120_REGIME_MOMENTUM_RESULTS.md`; for live picks see
`LIVE_TOP15_WITH_FUNDAMENTALS.md`;
for the running log see the STATUS-MD. This document is the full detail:
universe, selection, filtering/ranking, rotation, costs/tax, and every
comparison run.

---

## 0. The question

Out of the Nifty MidSmallcap-400 Momentum-Quality space (NSE index ~20%
CAGR), can a concentrated, frequently-rotated stock-selection rule
*consistently and robustly* beat the index's ~20% CAGR — validated
survivorship-free, with honest drawdown, tax and out-of-sample treatment?
**Hurdle = ~20% CAGR.**

---

## 1. Stock universe

Two universes, by design (locked scope, STATUS-MD §2):

### 1a. Backtest universe — survivorship-free, point-in-time (PIT)

The validated numbers use this, **not** today's MQ100 (using today's
constituents on past dates = survivorship/look-ahead bias).

- **Source pool:** every NSE symbol with daily data in
  `market_data.db` (~1,623 symbols, 2000→2026).
- **Ranking metric:** trailing **126-trading-day median daily traded
  value** = close × volume (a liquidity/size proxy; true market-cap PIT
  not freely available).
- **Eligibility:** ≥ 75 priced bars in the lookback window, median
  traded value > 0.
- **Rebalanced monthly** using only data available up to that date (no
  look-ahead).
- **Bands** (0-based rank, top-100 dropped as large-cap):
  | Band | Rank range | Meaning |
  |---|---|---|
  | `mid` | 101–250 | mid-cap liquidity band ← **chosen** |
  | `small` | 251–500 | small-cap band |
  | `combo` | 101–500 | mid + small combined |
- A separate semi-annual reconstruction (`01_reconstruct_universe.py`,
  LARGE_CAP_EXCLUDE=100, UNIVERSE_SIZE=400) sanity-checked the proxy:
  ~68/100 of today's supplied MQ100 fall in the reconstructed 101–500
  band → the liquidity proxy captures most of the same space but is
  **owned as a proxy, not the real index membership** (caveat).

### 1b. Live-pick universe — user-supplied MQ100

For *today's* actionable list only: the 100 MQ100 constituents the user
supplied (`universe_mq100_2026-05-15.csv`). 4 ticker renames remapped
(360ONE→IIFLWAM, UNOMINDA→MINDAIND, NAVA→NAVABHARAT, GVT&D→GET&D);
91/100 have ≥120d history; 9 lack data (recent listings).

---

## 2. Selection criteria — Relative Strength (RS)

Core signal, identical in backtest and live:

```
RS_i = (P_i[t] / P_i[t-L]) / (BENCH[t] / BENCH[t-L])
```

- `P_i` = stock close, `BENCH` = **NIFTYBEES** (Nifty-50 ETF; full daily
  history 2005→2026). RS is a *ratio* so the ETF price scale cancels —
  faithful Nifty-50 proxy.
- **Benchmark bug (disclosed, not hidden):** the first sweep used
  `NIFTY50`, whose daily series in the DB only exists 2023-03→2026-03
  (740 bars). For 2014→2022 RS was undefined → the book sat 100% in
  cash at 6.5% for 8 of 12 years. That run's "RS fails, 0/75 beat 20%"
  was a **fabricated negative** and is **void**. Fixed → NIFTYBEES; all
  reported numbers are from the corrected run only.
- **Lookback `L` swept:** 55d, 120d, 126d (~6m), 252d (~1y), and a
  126+252 blend. **120d chosen** (see §6 comparison).
- **Ranking:** within the chosen band, sort eligible names by RS
  descending; the top slots (subject to filters below) are held
  equal-weight.

---

## 3. Applied filtering / ranking stack (the winning config)

**Recommended config = `q0.5_dd__v__REG` on the `mid_120d_N15` core.**
Pipeline at each monthly rebalance, in order:

1. **Universe gate** — reconstruct PIT `mid` band (rank 101–250 by
   126d median traded value) as of that date.
2. **Regime gate (market on/off)** — if NIFTYBEES close < its **200-
   session SMA** → hold **zero equity**, sit in cash @6.5% p.a. that
   month. Else proceed. *This is the primary drawdown lever* — it
   sidesteps broad small-cap bear beta (2018, 2025).
3. **RS ranking** — compute RS-120 vs NIFTYBEES for all band members,
   sort descending.
4. **Quality screen (price-path proxy, `q0.5`)** — walking down the RS
   rank, keep a name only if **≥ 50% of its trailing-12-month 21-day
   blocks were positive** (return-consistency). This is **not**
   fundamentals — no ROE/D-E/EPS. Free PIT fundamentals don't exist;
   using current fundamentals on past dates = look-ahead. So the
   validated edge is pure price/momentum + regime (caveat #2).
5. **Volume-breakout confirmation — TESTED AND REJECTED.** Requiring
   recent 20d vol ≥ k×prior-60d vol collapsed CAGR to ~17–23% and
   *worsened* drawdown in every variant (it blocks the very momentum
   entries RS selects). The `v` axis is **OFF** in the winner.
6. **Own-DD cap — optional, OFF in the headline.** Capping a name's
   own trailing-12m max-DD (`dd-0.4/-0.5`) trims ~2pp more portfolio
   DD for ~2pp less CAGR; a conservative variant uses it (§6).
7. **Fill to N=15**, equal-weight, applying the buffer (§4).

Fundamentals enter **nowhere** in this pipeline. They appear only as a
**post-selection human annotation** on the live top-15
(`LIVE_TOP15_WITH_FUNDAMENTALS.md`): current ROE/D-E/PAT-growth/ROCE
web-sourced *after* the 15 are chosen, as a sanity flag (flagged
GMDCLTD weak); it does not re-rank or remove anything.

---

## 4. Rotation: checks, period, frequency

- **Frequency:** **monthly** (rebalance on each month-end bar).
- **Portfolio size N:** 15 (swept 10/15/20/25/30 — §6).
- **Retention buffer = N × 1.5 = top-22 (hysteresis):**
  - A held name is **kept** while it remains in the **top-22** by RS
    (not just top-15).
  - Only names that fall **out of the top-22** are sold.
  - Freed slots refill from the **top-15** down.
  - Purpose: kills churn from a stock oscillating around rank #15 →
    lower turnover → less cost **and less STCG** (directly improves the
    post-tax number). Worked example: a holding at rank #19 is *kept*
    (within 22), not sold.
- **Regime check** runs first every month (can flatten the whole book
  to cash irrespective of RS).
- **Cash when flat:** idle/bear-state cash earns **6.5% p.a.** (debt),
  modelled explicitly — not 0%.
- **Turnover cost:** **0.4% round-trip** applied on the fraction of the
  book that changes each month (brokerage+STT+impact, small-cap level).

---

## 5. Cost & tax model

| Item | Treatment |
|---|---|
| Transaction cost | 0.4% round-trip on monthly turnover |
| Idle/bear cash | +6.5% p.a. (debt) |
| **STCG** (held <365d, sold at gain) | Modelled in Phase 04: **15%** (pre-Jul-2024) and **20%** (current) |
| **LTCG** | **Not modelled** — monthly rotation is overwhelmingly short-term so the omission is small; it errs toward *understating* total tax. Stated, not hidden. |
| Window | 2014-01-01 → 2026 (12.1y); includes 2018-19 small-cap bear, Mar-2020, 2022, 2025 drawdown |

---

## 6. Comparison against other runs / parameters

### 6a. RS-alone sweep — 75 configs (3 bands × 5 lookbacks × 5 sizes)

Corrected run. **75/75 beat the 20% hurdle** raw (CAGR 25–41%).
Selected RS-alone leaders:

| Config | CAGR | Sharpe | MaxDD | Calmar | top-3 share |
|---|---|---|---|---|---|
| mid_126d_6m_N10 | 40.7% | 1.35 | −33.5% | 1.21 | 14.1% |
| combo_blend_6m12m_N25 | 40.4% | 1.39 | −38.8% | 1.04 | 10.1% |
| mid_120d_N10 | 39.9% | 1.34 | −34.6% | 1.15 | 16.2% |
| **mid_120d_N15** (chosen core) | **38.3%** | **1.39** | **−29.8%** | **1.29** | **11.9%** |
| mid_120d_N20 | 35.8% | 1.39 | −28.1% | 1.27 | 9.6% |
| mid_126d_6m_N25 | 34.6% | 1.41 | −25.5% | 1.36 | 8.3% |

**Lookback comparison (the key fitted knob):**
- `55d`: highest raw CAGR in places but **worst drawdown bucket
  −54% to −66%** — noise-chasing; it only "won" the *void* run #1
  because that run saw only 2023–26.
- `120d` / `126d_6m` / `blend`: best risk-adjusted (Calmar >1.2,
  DD ~−28 to −36%). **120d chosen.**
- `252d (1y)`: deepest holes among long lookbacks.

**Universe comparison:** `mid` = best risk-adjusted (lower DD, Calmar
>1.2); `small` = more CAGR but −40 to −66% DD; `combo` between.

**Size N comparison:** N=10 marginally higher CAGR but higher top-3
dependence; N=15–25 the robust sweet spot; **N=15 chosen.**

### 6b. Super-winner robustness (false-indication guard)

Re-ran the top-12 configs **forbidding their 3 best lifetime
contributors**. **12 configs still beat 20% (ex-top-3 CAGR 34–39%);
top-3 profit share only ~8–15%.** → the edge is **breadth, not 1–2
multibaggers**. (Run #1 falsely showed 0 robust — the cash artifact.)

### 6c. Per-year (bear exposure, RS-alone leaders)

Sample (`mid_blend_6m12m_N10`): 2017 +129%, 2020 +68%, 2021 +120%,
2023 +66% carry the CAGR; **2018 −14%, 2025 −11%** are the bleed years.
Lumpy, not smooth — the drawdown is real.

### 6d. Phase 03 — 53 drawdown-control overlays on `mid_120d_N15`

Goal: shrink the −30% DD toward the index's −24% **without** dropping
CAGR below 35%. Baseline `mid_120d_N15`: 38.4% / −29.8% / Calmar 1.29.

**Goal-test winners (DD shallower AND CAGR ≥ 35%):**

| Config | CAGR | Sharpe | MaxDD | Calmar | Note |
|---|---|---|---|---|---|
| **`q0.5_dd__v__REG`** ★ | **35.3%** | **1.53** | **−24.6%** | **1.44** | quality0.5 + regime; best in study |
| `q0.5_dd__v__nor` | 37.0% | 1.35 | −29.6% | 1.25 | quality only — ~neutral on DD |

**Top configs by Calmar (drawdown-efficiency):**

| Config | CAGR | MaxDD | Calmar |
|---|---|---|---|
| q0.5_dd__v__REG | 35.3% | −24.6% | 1.44 |
| q0.58_dd__v__REG | 33.5% | −24.5% | 1.37 |
| q0.5_dd-0.5_v__REG | 32.2% | −23.4% | 1.37 |
| q0.5_dd-0.4_v__REG | 30.6% | **−22.5%** | 1.36 |
| q__dd__v__REG (regime only) | 34.8% | −26.4% | 1.32 |

Findings: **regime filter = the DD lever** (every `_REG` trims 3–7pp);
**volume confirm = poison** (collapses CAGR everywhere); mild quality
screen compounds with regime (lifts Sharpe 1.40→1.53). Conservative
pick `q0.5_dd-0.4_v__REG` gets DD to −22.5% (shallower than the index)
at 30.6% CAGR.

### 6e. Phase 04 — out-of-sample + post-tax (on the chosen `q0.5_dd__v__REG`)

**A. Sub-period stability — PASS** (fixed config, disjoint halves):

| Window | CAGR | MaxDD | Sharpe |
|---|---|---|---|
| Full 2014–2026 | 35.3% | −24.6% | 1.53 |
| H1 2014–2019 | 32.2% | −24.6% | 1.46 |
| H2 2020–2026 | 37.3% | −14.7% | 1.54 |

Edge strong in **both** halves — not a single-regime artifact.

**B. Walk-forward lookback selection — PASS.** Each year 2019→2026 the
RS lookback was re-picked by best trailing-3y Calmar (no peeking) and
traded that year, chained: **33.1% CAGR** vs static L=120 **35.0%**
over 2019–2026. The procedure only ever picked `120d`/`126d_6m` (never
`55d`/`252d`) → lookback choice **robust, not lucky**.

**C. Post-tax (STCG) — PASS the hurdle:**

| | CAGR | MaxDD | Sharpe | Drag |
|---|---|---|---|---|
| Gross | 35.3% | −24.6% | 1.53 | — |
| Net STCG @15% | 30.4% | −25.1% | 1.38 | −4.9pp |
| Net STCG @20% (current) | 28.9% | −25.3% | 1.33 | −6.4pp |

**Post-tax 28.9% still clears the ~20% hurdle by ~9pp.**

### 6f. Run comparison summary

| Run | Universe/benchmark | Verdict | Status |
|---|---|---|---|
| Run #1 (RS sweep) | NIFTY50 (2023+ only) | "0/75 beat 20%" | **VOID** — 8/12y in cash |
| Run #2 (RS sweep) | NIFTYBEES (2005+) | 75/75 beat 20%; 12 robust | valid |
| Phase 03 (53 overlays) | NIFTYBEES | `q0.5_dd__v__REG` 35.3%/−24.6% | valid |
| Phase 04 (OOS+tax) | NIFTYBEES | stable both halves; 28.9% post-tax | PASS |

---

## 7. Final answer & honest caveats

**The winning system:** monthly-rotated, equal-weight **15-stock**
portfolio chosen by **RS-120 vs NIFTYBEES** within the **PIT mid-cap
liquidity band**, gated by a **NIFTYBEES-200DMA market regime switch**
and a **price-path consistency screen (≥50% positive months)**, with a
**top-22 retention buffer**, 0.4% round-trip cost, 6.5% bear-cash.
→ **35.3% gross / 28.9% post-tax (20% STCG) CAGR at −24.6% drawdown**,
OOS-stable, robust to losing its 3 best names. Beats the ~20% hurdle by
a wide margin even after tax.

**Caveats (binding — cite with the numbers):**
1. Run #1 void (benchmark-data artifact) — never cite its numbers.
2. **No fundamentals in the strategy.** "Quality" = price-path proxy.
   The index's actual Quality leg is *not* replicated — we beat its
   *return* via momentum, not its method. Fundamentals are a live-list
   annotation only.
3. PIT universe is a **liquidity-traded-value proxy**, not real index
   membership (~68/100 MQ100 overlap).
4. Drawdown is real (~−25% even after the regime filter).
5. **LTCG not modelled** — slightly understates total tax.
6. Live list is as-of the laptop snapshot date — re-run
   `05_live_top15.py` on the VPS for a current-dated list.
7. No performance guarantee. A measured, validated edge — not certainty.
   Real-capital deployment is a user decision; nothing is wired live.

**Genuine next phase (not done):** put *real* fundamentals into
selection (a quant-quality factor in the backtest) — requires a **paid
point-in-time fundamentals source** (Capitaline/CMIE/Refinitiv);
free sources can't supply as-of-date financials without look-ahead.

---

## 8. File map

| File | Purpose |
|---|---|
| `scripts/01_reconstruct_universe.py` | PIT universe reconstruction + MQ100 sanity |
| `scripts/02_rs_sweep.py` | RS-alone 75-config sweep + super-winner robustness |
| `scripts/03_rs_quality_volume.py` | 53 drawdown-control overlays |
| `scripts/04_walkforward.py` | OOS sub-period + walk-forward + post-tax |
| `scripts/05_live_top15.py` | today's live top-15 generator |
| `results/rs_sweep_ranking.csv` | all 75 RS configs |
| `results/rs_sweep_robustness.csv` | 12 robust (ex-top-3) configs |
| `results/rs_sweep_top5_peryear.csv` | per-year returns, top-5 |
| `results/phase03_overlay_sweep.csv` | all 53 overlay results |
| `results/phase04_walkforward.csv` | walk-forward year-by-year picks |
| `results/phase04_chosen_gross_nav.csv` | chosen-config NAV curve |
| `results/MIDCAP_RS120_REGIME_MOMENTUM_RESULTS.md` | short verdict + caveats |
| `results/LIVE_TOP15_WITH_FUNDAMENTALS.md` | live picks + fundamentals overlay |
| `results/MIDCAP_RS120_REGIME_MOMENTUM_DETAILED_REPORT.md` | **this document** |
| `results/MIDCAP_WINNER_YOY_VS_BENCHMARKS.md` | yearly vs Nifty-50 (§10) |
| `scripts/06_smallcap_overlay_oos.py` + `SMALLCAP_RSBLEND_REGIME_MOMENTUM_RESULTS.md` | small-cap variant (explored, not preferred) |
| `scripts/07_combo_overlay_oos.py` + `COMBO_RSBLEND_REGIME_MOMENTUM_RESULTS.md` | combo variant (dominated by mid) |
| `MIDSMALL400_MQ_CONCENTRATED_DAILY_SWEEP_STATUS.md` | running log / crash recovery |

---

## 9. Universe decision — MID vs SMALL vs COMBO (LOCKED: MID)

Same regime+quality overlay + OOS + post-tax pipeline run on all three
PIT bands, apples-to-apples. **Mid is the locked recommended system.**

| (gated champion, post-tax @20% STCG) | Post-tax CAGR | MaxDD | Sharpe | Gross Calmar | OOS H1/H2 | F&O stocks in band |
|---|---|---|---|---|---|---|
| **MID** `q0.5_dd__v__REG` ✅ | **28.9%** | **−24.6%** | 1.53 | **1.44** | 32.2 / 37.3 | **22 / 150** |
| SMALL `q0.5_dd-0.4_REG` | 30.2% | −28 to −30% | 1.56 | 1.27 | 35.0 / 35.1 | **1 / 250** (IRCTC) |
| COMBO `q0.58_dd-0.4_REG` | 28.1% | −30.6% | 1.31 | 1.13 | 32.0 / 33.8 | 23 / 400 |

**Why MID (decisive):**
1. **Shallowest drawdown** — −24.6% (index-level) vs small −28/−30%,
   combo −30.6%. Phase-03's whole purpose was DD control; mid wins it.
2. **Best risk-adjusted** — Calmar 1.44 (small 1.27, combo 1.13).
3. **COMBO is strictly dominated** — lower post-tax CAGR than mid
   (28.1 < 28.9) *and* deeper DD. No preference picks combo.
4. **SMALL's only edge is +1.3pp CAGR**, erased by deeper DD, lower
   Calmar, a needed extra junk-filter, **and near-zero F&O liquidity
   (1 vs 22 names)** — small's real costs likely exceed the modelled
   0.4% RT, so its 30.2% is optimistic; mid's 28.9% is trustworthy.
5. **Smaller working universe** — 150 names (vs 250 / 400): easier to
   reconstruct, monitor and reason about.

Verdict: **MID `q0.5_dd__v__REG` is THE system.** Small = a
higher-pain alternative only for those who will tolerate ~−30% for
~1pp; combo = never.

## 10. Year-by-year vs Nifty 50 (gross)

Full table + data-honesty note: `MIDCAP_WINNER_YOY_VS_BENCHMARKS.md`.
Headline: **beat Nifty 50 in 10 of 13 years**; CAGR 35.3% gross
(28.9% post-tax) vs Nifty 50 13.6% over 12.1y. The 3 lag years
(2018/2019/2025) are the regime-gated risk-off years (sat in cash
through small-cap bears — controlled give-back is the edge). Nifty
100 / Midcap150 / Smallcap250 YoY pending a Kite index-history pull on
the VPS (not fabricated; columns intentionally blank till real data).
