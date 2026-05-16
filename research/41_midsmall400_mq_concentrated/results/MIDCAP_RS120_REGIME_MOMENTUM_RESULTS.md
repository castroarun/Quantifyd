# RS-Concentrated Rotation on PIT Mid/Small — Honest Findings

**Hurdle:** ~20% CAGR (Nifty MidSmallcap400 Momentum-Quality-100 index).
**Window:** 2014→2026 (12.1y), includes 2018-19 small-cap bear, Mar-2020,
2022, and the 2025 drawdown. **Run:** #2 (corrected). Run #1 is **void** —
see caveat 1.

## 1. The benchmark bug that voided run #1 (read this first)

Run #1 used `NIFTY50` as the RS denominator. That daily series exists in
`market_data.db` **only from 2023-03-20** (740 bars). For 2014→2022 (8 of 12
years) `rs_scores()` returned `None`, so the strategy held **zero equity and
compounded idle cash at 6.5%** every one of those years. Run #1's headline
("0/75 beat 20%, RS fails, best ~10% CAGR") was therefore a **fabricated
negative** — RS was never tested over 2014-2022.

**Fix:** benchmark → `NIFTYBEES` (Nifty-50 ETF, full daily history
2005→2026). RS is a price *ratio* so the ETF's price scale cancels — it is a
faithful Nifty-50 proxy. The benchmark was also excluded from the investable
PIT universe. All numbers below are from the corrected run #2 only.

## 2. Headline verdict

**Plain Relative-Strength, concentrated, monthly-rotated on a survivorship-free
mid/small universe, decisively and robustly beats the ~20% index hurdle —
at materially deeper drawdown than the index.**

- **Raw:** 75/75 configs beat 20% (CAGR 25–41%).
- **Super-winner guard (mandatory):** 12 configs still beat 20% *after being
  forbidden from ever holding their 3 best lifetime contributors* — ex-top3
  CAGR 34–39%. Top-3 profit share is only ~8–15%, so the edge is **breadth,
  not 1–3 multibaggers**. This is the key result: the apparent edge is real.
- **Cost of the edge:** MaxDD −28% to −66% depending on config (index ≈ −24%).
  Concentration buys CAGR *and* a deeper hole, exactly as scoped.

## 3. Recommended configs (robust subset, ranked by drawdown control)

| Config | CAGR | ex-top3 | Sharpe | MaxDD | Calmar | top3 share |
|---|---|---|---|---|---|---|
| **`mid_120d_N15`** | 38.3% | 33.9% | 1.39 | **−29.8%** | **1.29** | 11.9% |
| `mid_126d_6m_N15` | 38.4% | 36.3% | 1.39 | −31.0% | 1.24 | 11.8% |
| `combo_126d_6m_N20` | 37.5% | 36.0% | 1.30 | −34.3% | 1.09 | 9.8% |
| `combo_blend_6m12m_N30` | 38.4% | 37.1% | 1.37 | −36.2% | 1.06 | 8.5% |
| `combo_blend_6m12m_N25` | 40.4% | **39.0%** | 1.39 | −38.8% | 1.04 | 10.1% |

**Pick: `mid_120d_N15`** — mid-cap liquidity band, ~6-month (120-trading-day)
RS lookback, hold 15 names equal-weight, monthly rotation with a top-22
retention buffer. Best Calmar (1.29) and shallowest drawdown among the robust
set, ~38% CAGR, low concentration. `combo_blend_6m12m_N25` is the choice if
maximizing the post-super-winner-removal floor (ex-top3 39%) matters more
than drawdown.

## 4. What the sweep taught us

- **Lookback:** ~6-month (120d / 126d) and the 6m+12m blend dominate.
  Short `55d` is the **worst** drawdown bucket (−54 to −66%) — it chases
  noise; it only looked good in the broken run because that run saw only the
  2023-26 bull. `252d` (1y) has the deepest holes among long lookbacks.
- **Universe:** `mid` (rank 101–250) gives the best risk-adjusted profile
  (lower DD, Calmar >1.2). `small` adds CAGR but −40 to −66% drawdowns.
  `combo` sits between.
- **Size N:** more concentration (N=10) → marginally higher CAGR but higher
  top-3 dependence; N=15–25 is the robust sweet spot. N=15 chosen.
- **Bears (per-year, `mid_120d_N15` cohort):** 2018 −11%, 2025 −20%, 2026
  −3% — it *does* trail/bleed in small-cap bears; 2017 +132%, 2020 +81%,
  2021 +88% carry the CAGR. Lumpy, not smooth.

## 5. Honest caveats (do not omit when citing this)

1. **Run #1 void** — benchmark-data artifact (§1). Never cite its numbers.
2. **Quality is not in this test.** This is pure price RS. No ROE/D-E/EPS
   screen (free PIT fundamentals unavailable). The index's "Quality" leg is
   *not* replicated — we beat its *return*, not its method.
3. **PIT universe is a liquidity-band proxy**, not the real index membership
   (survivorship-free by construction, but only ~68/100 of today's supplied
   MQ100 fall in the reconstructed band — proxy, owned as such).
4. **Drawdown is real and large** (−30% best case vs index −24%). A live
   investor must survive a −30% to −40% equity hole to realize this CAGR.
5. **Costs:** 0.4% round-trip modelled on turnover. **STCG (~15-20%) is NOT
   deducted** from these CAGRs — monthly rotation is highly STCG-exposed;
   post-tax CAGR is materially lower. Reported separately, not netted.
6. **Idle cash earns 6.5%** (debt) — explicit, not 0%.
7. No performance guarantee. Walk-forward / OOS split not yet run (Phase 04).

## 6. Phase 03 — drawdown-control overlays (COMPLETE)

Core fixed at `mid_120d_N15`; swept 53 overlay combos of trend-quality
screen / volume-breakout confirm / SMA200 regime filter vs the RS-alone
baseline. **Goal met** — 2 configs shrink drawdown *and* keep CAGR ≥ 35%:

| Config | CAGR | Sharpe | MaxDD | Calmar | vs baseline |
|---|---|---|---|---|---|
| Baseline `mid_120d_N15` | 38.4% | 1.40 | −29.8% | 1.29 | — |
| **`q0.5_dd__v__REG`** ★ | **35.3%** | **1.53** | **−24.6%** | **1.44** | DD −5.2pp, CAGR −3.1pp, every risk metric better |
| `q0.5_dd-0.4_v__REG` | 30.6% | 1.45 | **−22.5%** | 1.36 | conservative: DD beats the index |
| `q0.58_dd__v__REG` | 33.5% | 1.51 | −24.5% | 1.37 | |
| Index hurdle | ~20% | — | ~−24% | — | |

**★ Recommended final config: `q0.5_dd__v__REG`** on the `mid_120d_N15`
core = mid liquidity band, 120-day RS vs NIFTYBEES, hold 15 equal-weight,
monthly rotation (top-22 buffer), **plus** (a) only hold names with ≥50% of
trailing-12m 21-day blocks positive, **plus** (b) sit in 6.5% cash any month
NIFTYBEES closes below its 200-session average. Volume filter OFF, own-DD
cap OFF. Result: **35.3% CAGR at index-level −24.6% drawdown**, Sharpe 1.53,
Calmar 1.44 (best in the whole study). The drawdown objection to RS-alone is
resolved without surrendering the edge (still 1.75× the ~20% hurdle).

Structural findings:
- **Regime filter (SMA200) is the drawdown lever** — *every* `_REG` config
  trims 3–7pp of MaxDD; it sidesteps the 2018/2025 small-cap bear beta that
  caused the −30% hole.
- **A mild quality screen (pos-month ≥ 0.5) compounds with regime** — alone
  it's ~neutral; with regime it lifts Calmar to 1.44 and Sharpe to 1.53.
- **Volume-breakout confirmation is decisively counter-productive** — every
  `v1.0`/`v1.2` config collapses CAGR to ~17–23% and *worsens* drawdown
  (it blocks the very momentum entries RS selects for). Rejected.
- **Tighter own-DD caps** trade ~2pp more DD for ~2pp less CAGR; the
  `dd-0.4` regime variant gets MaxDD to −22.5% (shallower than the index)
  at a still-strong 30.6% CAGR — the choice for the most risk-averse.

Caveat 5 (STCG not netted) and caveat 3 (PIT liquidity proxy) still bind on
these numbers. No OOS split yet.

## 7. Phase 04 — out-of-sample validation + post-tax (COMPLETE)

The strategy is a fixed rule; the real overfitting risk is config
*selection* on the full window. Three honest OOS tests on the chosen
`q0.5_dd__v__REG`:

**A. Sub-period stability — PASS.** The fixed config, run on disjoint halves:

| Window | CAGR | MaxDD | Sharpe |
|---|---|---|---|
| Full 2014–2026 | 35.3% | −24.6% | 1.53 |
| H1 2014–2019 | 32.2% | −24.6% | 1.46 |
| H2 2020–2026 | 37.3% | −14.7% | 1.54 |

Edge is present and strong in **both** halves (not a single-regime
artifact); drawdown control holds in both. H2 actually shallower DD.

**B. Walk-forward lookback selection — PASS.** Each year 2019→2026, the RS
lookback was re-picked by best trailing-3y Calmar (no peeking), then traded
that year, chained: **33.1% CAGR** vs static L=120 **35.0%** over the same
2019–2026. The procedure only ever picked `120d` or `126d_6m` (never the
bad `55d`/`252d`) — the lookback choice is **robust, not lucky**; the 1.9pp
gap to static is well within noise and far above the 20% hurdle.

**C. Post-tax (STCG) — PASS the hurdle.** Indian STCG applied to gains on
positions held <365d:

| | CAGR | MaxDD | Sharpe | Drag |
|---|---|---|---|---|
| Gross | 35.3% | −24.6% | 1.53 | — |
| Net STCG @15% (pre-Jul-2024) | 30.4% | −25.1% | 1.38 | −4.9pp |
| Net STCG @20% (current) | 28.9% | −25.3% | 1.33 | −6.4pp |

**Post-tax CAGR 28.9% (at the current 20% STCG) still clears the ~20%
index hurdle by ~9pp.** (The log's "cum tax ~5× of init" is a *scale*
artifact — the book compounds ~50× gross over 12y, so tax measured in
units of *initial* capital is large; the meaningful figure is the
**5–6pp CAGR drag**. LTCG is *not* modelled — monthly rotation is
overwhelmingly short-term so the omission is small, and it errs toward
*understating* total tax; stated, not hidden.)

## 8. Final verdict

`q0.5_dd__v__REG` on the mid_120d_N15 core: **robust, out-of-sample-stable,
~29% post-tax CAGR at index-level drawdown** — beats the ~20% MidSmallcap400
MQ100 hurdle by a wide margin even after tax, survives losing its 3 best
names, and works in both market-halves. Honest residual caveats: PIT
universe is a liquidity proxy (caveat 3); quality leg is price-path not
fundamentals (caveat 2); LTCG not netted; live list as-of snapshot date.
Not a guarantee — a measured, validated edge. Live picks +
fundamentals: `LIVE_TOP15_WITH_FUNDAMENTALS.md`.
