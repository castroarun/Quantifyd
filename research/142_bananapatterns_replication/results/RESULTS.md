# Phase-5 Results — Corrected Harness, After-Tax Adoption, Blend Capstone (2026-09-02)

**HARNESS BUG DISCLOSED:** the Phase-4 sweep runner recomputed trail-SMAs on a
calendar-aligned matrix, NaN-poisoning ~1.18M cell-days and silently disabling
trail exits (found via A/B probe `scripts/probe_diff.py`; e.g. no-floor 517×
inflated → corrected 301×). Every adoption decision re-made on the corrected
CLI engine, with a new gate per Arun: **decisions are made NET OF TAX** (20%
STCG / 12.5% LTCG on net realized gains, modelled in-sim, `--stcg`).

**FINAL ADOPTED SPEC (pre-tax reference):** decoded rules + ₹5cr/day liquidity
floor (NO mcap floor), 200-DMA gate ON, −8% stop KEPT (stop-off reversed on the
corrected engine: 29.3% < 31.8%), 50-SMA trail, realistic fills, 25bps —
**301× / 31.8% [25.4..37.6] / −45.7%** (2006→Aug 2026, 10-seed medians).

**TAXABLE-ACCOUNT PICK — trail-20 variant (Arun's balance principle):**
after-tax trail 50/20/15 = 25.7%/28.0%/29.8% CAGR at −47.8/−33.4/−36.3% DD.
Tax scales with gains, so trail-20's edge survives: +2.3pp net-of-tax and 14pp
less drawdown than trail-50; the IS-decade return deficit (pre-tax 28.3 vs
30.8) roughly equalizes after tax while its DD advantage held in BOTH decades.
Trail-15 declined (further churn; edge more recent-era; 15↔20 ranges overlap).

**Correlation matrix (daily / monthly, 2006→Jul 2026 common window):**
BlueSky(trail-20)↔Momentum 0.45 / 0.51 · BlueSky↔Nifty50 0.30 / 0.33 ·
Momentum↔Nifty50 0.44 / 0.44 · 50-50 blend↔Nifty50 — / 0.44. The half-correlation
between the legs is the raw material of the blend's gains; expect convergence in
crashes (both are long smallcap momentum).

**CAPSTONE — 50-50 blend with research/75 momentum (monthly rebalanced):**
corr 0.29 daily / 0.52 monthly; blend median **33.0% [30.1..36.1] at −27.5% DD
(worst seed −32.3%)** — beats BOTH legs on CAGR and DD (staggered drawdowns +
rebalancing premium + volatility-drag reduction). Same physics as research/63
GTAA. Best construction found in the whole study. Caveat: crash-time
correlation convergence; blend is pre-tax on the BlueSky leg (momentum leg
low-churn/LTCG-friendly).

Published (single consolidated update): corrected Phase-4 table + after-tax
verdict + blend table + config-B-on-2020-25 row + harness-bug caveat + charts
regenerated from the adopted-spec equity — commit `3f717e2`.

---

# Phase-4 Results — Optimization Sweep (2026-09-02) — SUPERSEDED (harness bug; see Phase-5)

**VERDICT: the decoded baseline is near-optimal for its family; the sweep's chief
fruit is a CORRECTION, not a betterment.** 38+10 cells (six axes OFAT + plateau
neighbors + combos + sizing×slots), 2006→2026-08, config-D stack, 10-seed
ensembles per cell (`results/p4_sweep.csv`).

**THE CORRECTION (publish loudly):** the Phase-3 claim "mcap floor = risk filter
(−0.7pp CAGR for −14pp DD)" was an ARTIFACT. The mcap snapshot had only 925/2,321
symbols when config D first ran, so the floor silently excluded ~1,400
unknown-mcap names. With the completed snapshot (2,042 known): floor = 297× /
31.7% / −36.7%; **no floor = 517× / 35.3% / −36.9% — same DD, floor is pure
return drag. New headline = the no-floor book (≈ config B).** Study page updated.

Findings (medians): trail 15/20-SMA spikes (562×/387×) FAIL the plateau test
(SMA 25-40 collapse to 163-189×) → rejected; long trails 75-200 (LTCG hope)
destroy the book (14-21% CAGR, −47/−56% DD) → rejected; removing the −8% stop is
mildly positive alone (326×, −34.8%) but unstable in combos → optional; slots
INERT (cash binds at ~5 concurrent — sizing is the true constraint); smaller-
size×more-slots slightly worse (concentration is the engine); RS threshold /
basing depth / gate-DMA show no clean dose-response; Arun's adaptive-mcap
switching is MOOT (≡ no-floor: the gate already blocks weak-day entries; and the
"bull-mode gap" that motivated it dissolved — config B on 2020-25 as an ensemble
is 37.6% median vs D's 37.3%, the 58.3% figure was one lucky seed).

**Churn (STCG question):** BlueSky ≈ 24 trades/yr × 18.75% ≈ 4.5× book turnover,
all STCG; Momentum r/75 ≈ 0.38× turnover with LTCG-eligible winners. For a
taxable account momentum keeps a structural 2-4pp/yr after-tax advantage at
equal gross CAGR. Longer BlueSky trails cannot close this (see above).

---

# Phase-3 Results — 20-Year Robustness (2026-09-02)

**VERDICT: STRATEGY (candidate).** The decoded blue-sky construction survives
2006–2025 net-of-cost with realistic fills, reported as 10-seed selection
ensembles (medians [min..max]):

| Config | Terminal × | CAGR | MaxDD | Signals |
|---|---|---|---|---|
| A gate-ON, their fills, gross | 398 [228..813] | 34.9% | −44.0% | 16,612 |
| B gate-ON, real fills, net 25bps | 287 [138..758] | 32.7% | −45.7% | 16,612 |
| C gate-OFF, real fills, net 25bps | 225 [108..413] | 31.1% | −45.0% | 16,612 |
| **D = B + mcap≥₹500cr PIT proxy (HEADLINE)** | **203 [136..367]** | **30.4% [27.9..34.4]** | **−31.5%** | 8,069 |
| NIFTYBEES B&H | 10.25 | 12.3% | −59.7% | — |
| research/75 momentum (reference, net) | — | 31.9% | −31.6% | — |

- **The mcap floor (Arun's point-in-time challenge) is a RISK filter:** −0.7pp
  CAGR for 14pp less drawdown. Calmar ~0.96 ≈ research/75's tier. Two independent
  Indian momentum/breakout constructions converge on ~30-32% net — a strong
  plausibility check.
- Worst years (config D medians): 2008 −20.1%, 2018 −13.4%, 2016 −2.5%,
  2025 +10.3%. No dead decade; falsification criterion (pre-2015 ≲ NIFTYBEES) NOT
  met — though pre-2015 is survivorship-flattered (2006 coverage = 528 surviving
  symbols).
- **Published to the app (register of record):** `/app/backtest/bluesky-ath-breakout-research142`
  with the client tearsheet, the vs-indices growth chart (NIFTYBEES /
  NIFTYMIDCAP150 / NIFTYSMLCAP250, 2011→2025) and full caveats.
- Arun's regime challenge honoured: the sample has zero negative INDEX years but
  the strategy's edge is smallcap BREADTH, not index level; NIFTYBEES 2024 +10.4%
  / 2025 +11.7% while config D printed +52.8% / +10.3% — and 2026 YTD tape is
  negative. Paper-soak before capital.
- Honest caveats and next levers: see the study page + STATUS-MD P3 section.
  Next: optimization program on top of config D (publish betterments to the same
  study), STCG-net modelling, G5 paper book decision.

---

# Phase-2 Results — Full Blue-Sky Replication (2026-09-02)

**VERDICT: RULES REPRODUCED, RETURNS NOT. STRATEGY-family real but bull-sample;
published magnitude unconfirmed.** Running their exact, fully-decoded rules
(close above prior ATH-close → fill at pivot; IBD-RS≥70; 20d-median traded value
≥₹5cr; −8% close-stop; 50-SMA close-trail; 8 slots, 18.75% sizing, no costs) over
the full 2,321-symbol universe, 2020–2025:

| Run | Terminal | CAGR | MaxDD (daily) | Trades | Win% |
|---|---|---|---|---|---|
| Published (site, PROVISIONAL) | 33.74× | 79.8% | "worst fall" −11.4% | 272 | 52% |
| Faithful (RS-desc selection) | 11.01× | 49.2% | −31.5% | 175 | 42% |
| Random selection ×5 (seeds) | 6.5–15.1× | 36.6–57.2% | −22 to −32% | ~180 | 44–46% |
| + skip-weak-markets ON | 15.73× | 58.3% | −22.0% | 141 | 48% |
| + realistic fills (open-gap) | 9.88× | 46.5% | −26.7% | | |
| + costs 25bps/side | 9.34× | 45.1% | −31.6% | | |
| + exits at next open | 14.94× | 57.0% | −28.7% | | |

**Key findings**

1. **Path dependence is enormous.** 8 slots choosing from 10,691 signals → six
   equally-valid selection paths span 6.5×–15.1×. Any single backtest number from
   this construction (theirs included) is one draw from a wide distribution.
   Their 33.74× is >2× the best of six honest paths.
2. **Their −11.4% "worst fall" is not reproducible at any marking frequency**
   (our best variant: −22.0% daily, −15.3% monthly). Treat their risk figure as
   unreliable — the site itself stamps everything PROVISIONAL.
3. **Signal-close trail exits are NOT an inflation lever** — exiting next open
   actually performed BETTER (+35% terminal, single-path caveat). Retracting the
   Phase-1b suspicion. Fantasy pivot fills (−10%) and costs (−15%) are real drags.
4. **Skip-weak-markets ON improves risk-adjusted results** (58.3% CAGR, −22% DD,
   fewer/better trades) — consistent with research/71's 200DMA-gate finding.
   Their showcase run has it OFF.
5. Signal pool 10.7k vs their 5.7k — residual gap likely their mcap ≥₹500cr
   filter (no shares-outstanding history on our side) + universe differences.
   Recall of their exact trade list is structurally low (2–6/54) because of slot
   path-divergence, NOT because the rules differ — 48/51 of their trades pass
   every decoded condition on their exact entry dates.

**Honest caveats:** survivorship on BOTH sides (Kite lists only current
instruments; 700/1,762 download failures include delisted names); 85% of the
sample is a strong market — 2022 is the only weak year and prints 3.6–19% across
variants; no mcap filter; RS formula and selection inferred (best-evidence) not
disclosed; single-path variance dominates ±10% differences between variants.

**Bottom line for Arun:** the construction is a real, codeable bull-market
breakout book worth ~40–58% CAGR at −22 to −32% drawdown in THIS 6-year bull
sample — genuinely strong, but not 79.8%/−11.4%, and its weak-regime behaviour
is untested by this sample. Before any capital: G3 robustness (pre-2020 history,
per-year stability, walk-forward) and comparison against the live momentum-paper
book and research/75 (31.9% net CAGR, 20-year validated).

**Next levers:** (a) extend backtest to 2006–2025 with the gate ON; (b) mcap
proxy filter; (c) portfolio-level seed ensemble (report the distribution, not a
path); (d) tearsheet + publish to /app/backtest when concluded.

---

# Phase-1 Results — BananaPatterns Trade-Level Replication (2026-09-01)

## PHASE 1b UPDATE (same evening) — Blue-sky ground truth: ENGINE FULLY SOLVED

Arun re-ran the site's backtest with the **Blue sky** screen (8 pos, 1.5% risk, cap 30%:
₹10L → ₹3.37Cr, 79.8% CAGR PROV, 272 trades, median +0.6%); 51 trades transcribed to
`data/trades_groundtruth_bluesky.csv`. After repairing our DB (see below):

- **ENTRY SOLVED: pivot = the ALL-TIME-HIGH CLOSE.** Buy price equals the prior
  all-time-high close EXACTLY (0.00%) on ~35/51 trades, within 0.6% on nearly all
  others. Entry = buy-stop at that level, filled intraday the day price crosses it.
  A demerger resets the ATH context (STAR post-spin-off). Full rule:
  liquidity floor → within 20% of ATH-close → RS 70+ → buy-stop at ATH-close →
  −8% stop (close-basis, gap fills) → trail = exit at the close that breaks 50-SMA.
- **EXITS: 37/39 exact day+price** on blue-sky list (the 2 misses were our own
  unadjusted-bonus rows: ANANDRATHI, ECLERX — both fixed by re-download).
- **DB REPAIR executed on VPS** (`scripts/repair_data.py --apply`): 8 scale-broken
  symbols backed up to `market_data_unified_bak142`, deleted, re-fetched adjusted;
  18 missing symbols downloaded fresh (24/26 ok; E2E returns no data from Kite,
  BONDADA is BSE-only — both stay absent). Residual "breaks" are genuine demergers
  (SUVEN 2020, STAR 2024) / 2008 crash days. POCL found 2.5×-scale AFTER repair
  list was frozen — include it in the full-DB repair.
- **FULL-DB SCAN: 72/1,666 daily symbols carry ≥1 suspected unadjusted split/bonus**
  (close ratio ≤0.62 day-over-day) — infrastructure defect beyond this study; needs
  its own repair pass (some flags are genuine demergers — review before deleting).
- Remaining unknowns for the full replica: their RS formula (threshold 70), the
  selection rule when candidates exceed the 8 slots (5,399 passed up), and the
  30%-of-capital position cap mechanics.

---

## Original Phase-1 (VCP-screen ground truth) below

**VERDICT (Phase 1): REPRODUCIBLE.** Their published backtest is a real, rule-driven
engine, not fabricated numbers — we reproduced their exits to the exact day and exact
price on 22 of 23 closed trades, and their entries sit within ~0.02–1.1% of recent
swing-high pivots in our data. The match gate (≥80%) is PASSED on exits (96%) and
passed-with-caveats on entries. Optimization/controls (Phase 2) are now unlocked per
Arun's sequencing rule.

## What their engine actually does (reverse-engineered, evidence-backed)

| Rule | Inferred mechanic | Evidence |
|---|---|---|
| Trail exit | Exit booked AT THE CLOSE of the day that closes below the 50-SMA | 20+ trades: their sell price == that close to the paisa, same date |
| −8% stop | Evaluated on the CLOSE (not intraday touch), gap-aware fill below the stop level | KFINTECH sold at −9.48% (below stop = gap); GULFOILLUB did NOT stop despite intraday low ≤ stop px |
| Entry | Buy-stop at a pattern pivot ≈ recent swing high (varying 5–75d lookback); filled at pivot price intraday | 26/27 data-present entries within ~1% of a prior swing high; entry-day high always above buy (+0.4% to +10.7%) |

## The four inflation levers found (carry into Phase 2 honestly)

1. **Trail exit at signal close is look-ahead-ish**: you can't know the close broke
   the 50-DMA until the close; a real system sells the NEXT day. Our convention grid
   shows next-open exits matched only 14/23 — i.e., their booked prices are
   systematically the earlier/better price. Repricing all trail exits to next-open
   is a Phase-2 sensitivity.
2. **At-the-pivot fills are sometimes fantasy**: on ~4–6 entries our data shows the
   day OPENED above their buy price (CHOLAFIN, CHOICEIN, ORIENTCEM, LAURUSLABS,
   SMLMAH) — a real buy-stop fills at the open, worse. On the rest, the day's high ran
   1–10% above the pivot, so at-pivot vs breakout-close is worth several % per trade.
3. **Open positions marked to year-end close** (CUPID +201% still "open") — standard
   but flatters the terminal number.
4. **Risk dial = leverage**: their 2%-risk rerun is the identical 173 trades at 25%
   position size (vs 18.75%) — same stream, bigger bets, worst fall −12.3%→−14.9%.

## Data gaps ON OUR SIDE (action items)

- **market_data.db daily series are NOT retroactively split-adjusted.** MCX / HEG /
  NAZARA / SMLMAH / MUFIN / KFINTECH show entries "20–80% below ATH" only because our
  pre-split rows kept the old price scale (MCX 2025 split; entry-day ratio to their
  price is ~1.0 while old highs are ~5×). CUPID is scale-shifted 5.00× outright.
  **This affects ANY ATH/52wk-high computation on this DB, not just this study.**
- **11 of 35 symbols absent from our universe**: APARINDS, ASHAPURMIN, BONDADA, E2E,
  GLOBUSSPR, LUMAXTECH, MAHSCOOTER, PGIL, SUVEN, TFCILTD, V2RETAIL — including two of
  their three biggest winners (E2E +124%, i.e., the tails live in names we don't
  track). Phase 2 needs a universe extension download (VPS-only, per binding rule).

## Honest caveats

- Ground truth = 40 trades transcribed from screenshots of their VCP-screen run
  (2024-08→2025-12 window); the full 173-trade list and the Blue-sky screen's list are
  not yet in hand. Entry-pivot inference is approximate (their pattern detector is
  proprietary); exits are exact.
- Our DB's split-adjustment defect means "distance from ATH" screens can't be
  faithfully computed until fixed; entry-day prices matched because recent rows agree.

## Next levers (Phase 2 gate now open)

1. Arun: re-run their backtest with **Blue sky** selected; share the trade table →
   swap in as ground truth for the screen we actually target.
2. Fix/refresh split-adjusted daily history for affected symbols; extend universe to
   their liquidity floor (mcap ≥₹500cr, ₹5cr+/day traded).
3. Full-universe faithful replication 2020–2025, then the controls suite: next-open
   trail fills, open-above-pivot fill realism, costs, survivorship, tail-removal
   (super-winner guard), selection-rule sensitivity, 200DMA gate on/off.

*Reproducibility: script `scripts/validate_trades.py` + `scripts/entry_diag.py` on VPS
`backtest_data/market_data.db` (snapshot as of 2026-09-01); output
`results/trade_match.csv`.*

---

## ADDENDUM 2026-09-03 — Gate audit, bake-off, and spec revision (Arun-approved)

**Bug found:** phantom 15-Jan-2026 holiday rows (526 symbols, O=H=L=C=prev close, vol 0,
one Feb-17 Kite batch) NaN-poisoned `rolling(200).mean()` on the union-aligned NIFTYBEES
series → `close < NaN` = False → **the weak-market gate was silently OPEN in every
backtest/seeding run from late-Apr-2026**. The live engine (dropna) was never affected.
Rows purged (full 2015+ scan: no other date matches); gate code made NaN-robust in
`bluesky_replay.py` / `seed_paper_state.py`.

**Gate bake-off (72 cells × 2 windows + 20y finals + Donchians + plateau + after-tax +
30-seed paired test):** SMA200 gate REFUTED on any series/length/type. DD10 (block >10%
below 252d high) is real 2008 insurance (30/30 paired seeds, +9.6pp) at a real premium
(−1.6pp CAGR/yr, loses on 20/30 seeds) — NOT adopted per Arun's balance principle.
Smallcap/midcap gate series strangle the book (blocked 27-41% of days). Breadth gates
catastrophic. Full grids: `results/gate_bakeoff.csv`, `gate_finals.csv`,
`gate_yoy_full.csv`, `gate_paired_test.csv`.

**Adopted spec (2026-09-03): trail-20, −8% stop, 16 slots @6.25%, NO market gate.**
30-seed evidence: median 37.8% CAGR (pre-tax, 2006→26), worst-seed 33.6% (vs 31.9% at
8 slots), spread halved, losing years near-deterministic (2008 band −14.5..−12.3),
2026 zero losing paths. Book re-seeded (seed 5, 1,310 trades, 15 open), deposits
carried, dividend HWM re-anchored ₹11,35,026. Known model gap for the Dec-12 restudy:
sim holds idle cash at 0% (no CASHIETF yield) — understates all configs, most for
gated ones.
