# Tri-Sleeve Book — RS-Momentum Base + KC6 Options Overlay + Short/Hedge Sleeve

**STATUS: DESIGN LOCKED — building combined engine. No run launched yet.**

One Rs.1-crore book. Sleeve 1 (RS-momentum, the locked research/41 MID
`q0.5_dd__v__REG`) owns the equity. Sleeve 2 (KC6 monetised as defined-risk
option spreads) and Sleeve 3 (short/hedge — covered calls **vs** collar **vs**
systematic short, decided by backtest) are overlays funded as **margin
against the whole equity/debt book + option premium** — the margin capacity
available to KC6 is constant and **regime-independent** (the book's holdings
collateralise it whether risk-on equity or risk-off debt; the regime gate does
NOT change funds available to KC6). Goal: lift post-tax CAGR and/or cut the
−24.6% drawdown of the base **without a separate capital carve-out**.

This file is the sole crash-recovery source. If Claude disappears mid-build or
mid-run, Arun resumes from §6 using only this file.

---

## 1. Headline

**Tri-Sleeve single-book overlay system, daily/monthly cadence, 2014→2026,
post-tax, validated against base-alone.** STATUS: DESIGN LOCKED.

Base alone (research/41, the hurdle to beat): **35.3% gross / 28.9% post-tax
CAGR, −24.6% MaxDD, Sharpe 1.53, Calmar 1.44.** The combined system must beat
this on post-tax Calmar (or clearly cut DD at ≤ same CAGR) to justify the
added machinery — otherwise the honest verdict is "base alone wins, don't add."

---

## 2. The Ask

**What the user asked (near-verbatim):**
> Another session built `midcap-rs120-regime-momentum` (~30% CAGR). We've
> already done KC6 (good CAGR). I want a combined system of 3:
> 1. Base = the RS momentum one.
> 2. Overlay = KC6 — whenever we get an opportunity, deploy debit spreads or
>    credit spreads to monetise the KC6 opportunities.
> 3. The above 2 are long-only — add some kind of short system. Can be as
>    simple as covered-call opportunities, or protective strategies offering a
>    hedge at sustainable cost, or a regular short system using spreads/futures
>    — we need to work on this.
> Record status/progress in a relevantly-named MD throughout — crash recovery.

**What we are actually building (cleaned up, decisions locked via the 4
clarifying questions on 2026-05-17):**

A single-capital-pool backtest (one Rs.1cr book, 2014-01-01→2026, post-tax,
OOS-aware) that layers three sleeves and measures the **incremental**
post-tax CAGR / MaxDD / Sharpe / Calmar each sleeve adds over the locked
base, so we can decide what (if anything) is worth wiring live.

- **Sleeve 1 — Base, owns the book.** Exact locked research/41 system: MID
  `q0.5_dd__v__REG` — monthly-rotated 15-stock equal-weight, RS-120 vs
  NIFTYBEES, PIT mid-cap liquidity band (rank 101–250), NIFTYBEES-200DMA
  regime gate (flat to cash @6.5% in bear), price-path quality screen (≥50%
  positive 21-day blocks), top-22 retention buffer, 0.4% round-trip cost.
  **Not re-optimised. Replayed verbatim** so the comparison is honest.
- **Sleeve 2 — KC6 as options, funded from the same book.** Run the KC6
  mean-reversion signal (Close < KC(6,1.3·ATR) lower **and** Close >
  SMA-200, universe-ATR crash filter) on the F&O-eligible names. Each
  KC6 entry is expressed **both** as a bull-put **credit** spread and a
  bull-call **debit** spread; per name we pick the better expression by
  realised post-cost expectancy + an option-liquidity filter (user
  decision: "backtest both, pick per name liquidity"). Defined risk only.
  **Capital: margin against the whole equity/debt book — a constant,
  regime-independent capacity.** The regime gate flattening Sleeve 1 to
  cash does NOT add to or subtract from KC6's available funds; KC6's only
  cap is its own swept max-risk-% of book (§4). Bear-regime cash is not a
  special KC6 pool.
- **Sleeve 3 — Short / hedge, decided by backtest.** Scaffold all three
  and compare (user decision: "decide after backtesting all three"):
  - **A. Covered calls** — write OTM calls (target delta ~0.2–0.3) against
    the F&O-eligible Sleeve-1 holdings. Income, no extra capital, caps
    upside on called names.
  - **B. Collar / protective hedge** — covered-call premium funds long
    protective puts (stock or NIFTY index). Near-zero-cost DD cap; goal
    is to push −24.6% toward ~−15%.
  - **C. Systematic short** — bear-put spreads / short futures on the
    weakest-RS or breakdown names. Independent short alpha; own risk
    budget; most complex.
  Winner picked by post-tax Calmar contribution to the combined book.

- **Success criterion (single, ranked):** post-tax (20% STCG) **Calmar**
  of the combined book. Gates a config must clear to be "worth it":
  (i) post-tax CAGR ≥ base 28.9% **OR** MaxDD shallower than −24.6% at
  CAGR ≥ ~25%; (ii) Sharpe ≥ base 1.53; (iii) edge present in **both**
  OOS halves (2014–19 / 2020–26), not a single-regime artifact.

---

## 3. The Base — exact mechanics being layered (NO ambiguity)

### 3.1 Sleeve 1 — RS-momentum (replayed, not re-tuned)

Locked config = research/41 **`q0.5_dd__v__REG` on `mid_120d_N15`**.
Pipeline at each month-end rebalance, in order:

1. **PIT universe gate** — rank all NSE daily symbols by trailing-126d
   median traded value (close×vol); keep band rank **101–250** (`mid`),
   eligibility ≥ 75 priced bars, exclude NIFTYBEES.
2. **Regime gate** — NIFTYBEES close vs its 200-session SMA. Below →
   **0% equity**, whole book to cash @6.5% p.a. that month. (This is the
   primary DD lever. It does NOT change funds available to Sleeve 2 —
   KC6 margin is regime-independent against the book.)
3. **RS-120 ranking** — `RS_i = (P_i[t]/P_i[t-120]) / (BEES[t]/BEES[t-120])`,
   sort descending.
4. **Quality screen `q0.5`** — keep a name only if ≥ 50% of its
   trailing-12m 21-day blocks were positive.
5. Volume confirmation **OFF** (tested, poison).
6. Own-DD cap **OFF** in headline.
7. **Fill to N=15**, equal-weight, **top-22 retention buffer** (a held
   name kept while still in top-22 by RS; freed slots refill from top-15).
- Cost 0.4% round-trip on monthly turnover. Idle/bear cash 6.5% p.a.
- Window 2014-01-01→2026 (12.1y, includes 2018-19, Mar-2020, 2022, 2025).
- **Source of truth:** `research/41_.../scripts/03_rs_quality_volume.py`
  (config `q0.5_dd__v__REG`) + `02_rs_sweep.py` for universe/RS. The
  Sleeve-1 driver here must reproduce 35.3% gross / −24.6% DD to ±0.3pp
  as a self-check before any overlay is added.

### 3.2 Sleeve 2 — KC6 → defined-risk option spreads

- **Signal (entry trigger), per name, daily bar:** `Close < KC_lower`
  where `KC_lower = EMA(6) − 1.3·ATR(6)` **AND** `Close > SMA(200)`.
  Block ALL new entries when universe-ATR ratio
  (`ATR(14)/avg ATR(14, 50)`) ≥ 1.3 (KC6 crash filter).
- **Universe (REFINED 2026-05-17):** KC6's **native Nifty500 ∩ F&O
  universe** (~86 names, `services/data_manager.FNO_LOT_SIZES` ∩ DB
  daily symbols) — NOT restricted to the Sleeve-1 mid band. Rationale:
  KC6 is its own overlay funded by the book's (regime-independent)
  margin; clipping it to the 22 mid-band F&O names would gut a
  known-good system (KC6 native = 2,482 trades / 65% WR / PF 1.70 over
  20y on Nifty500). The earlier "22 mid names" line was an over-
  restriction and is corrected here. Sleeve 1 and Sleeve 2 universes
  are intentionally decoupled (one book funds both).
- **Option expression (both tested, per-name pick):**
  - **Credit:** sell bull-put spread — short put ~at/just-OTM, long put
    one strike-interval lower. Profit = bounce + theta. Max loss =
    (width − net credit)·lot.
  - **Debit:** buy bull-call spread — long call ~ATM, short call at the
    KC6-mid target. Profit = directional bounce. Max loss = net debit·lot.
- **Pricing:** flat-IV Black-Scholes — the **established repo convention**
  (`COLLAR_DEFAULTS`: `iv_assumed=0.25`, `risk_free_rate=0.065`). We have
  **no historical option chains** (`data/iv_history.db` empty) — see
  caveat C1. Sensitivity sweep IV ∈ {0.20, 0.25, 0.30} mandatory.
- **Exit:** mirror KC6 on the underlying — target at KC6 mid, 5% SL,
  15% TP, 15-day max hold; force-close ≤ 3 DTE (gamma); roll month if
  ≤ 7 DTE (reuse `COLLAR_DEFAULTS` expiry rules).
- **Funding:** margin against the whole equity/debt book — **constant,
  regime-independent**. Not drawn from a bear-cash pool. Position size =
  `COLLAR_DEFAULTS`-style 1-lot units, capped so total Sleeve-2 defined
  risk ≤ a swept % of book (§4) — that risk-% is the ONLY constraint on
  Sleeve-2 size, not the regime state.
- **Source of truth:** `services/collar_engine.py`, `services/kc6_scanner.py`,
  `config.py` `KC6_DEFAULTS` / `COLLAR_DEFAULTS`.

### 3.3 Sleeve 3 — short / hedge (3 variants, A/B/C)

- **A. Covered calls:** for each F&O-eligible Sleeve-1 holding, sell a
  ~30-delta (≈5% OTM, `COLLAR_DEFAULTS.call_otm_pct`) monthly call;
  collect premium; if ITM at expiry, deliver-or-roll (modelled as upside
  cap on that name for the cycle). No extra capital.
- **B. Collar:** A + use the collected call premium to buy a ~5% OTM
  protective put on the same names (or NIFTY index, swept). Net cost
  ≈ 0 to small bleed; objective DD ≤ −15%.
- **C. Systematic short:** scan the **weakest-RS** decile of the mid band
  (or breakdown: Close < SMA200 and falling) → bear-put spread (defined
  risk) or short future (capped notional). Independent risk budget
  (swept % of book). Crash filter inverted (allow shorts in high-ATR).
- All three reuse the flat-IV BS pricer + the same expiry/roll rules.
- **Only the winner (by post-tax Calmar contribution) is carried into
  the final recommended system.** The other two stay in the grid as
  documented comparisons.

---

## 4. Plan — variant grid + cell count

**Phase 0 — Sleeve-1 fidelity check (gate; no overlay).**
Replay `q0.5_dd__v__REG`; assert CAGR 35.3%±0.3 / MaxDD −24.6%±0.3.
1 cell. If it fails, STOP and fix the driver before anything else.

**Phase 1 — Sleeve 2 (KC6 options) alone over base.**
| Axis | Values |
|---|---|
| Expression | credit, debit, per-name-best |
| IV assumption | 0.20, 0.25, 0.30 |
| Sleeve-2 max risk % of book | 5%, 10%, 15% |
= 3 × 3 × 3 = **27 cells** (+ base = 28).

**Phase 2 — Sleeve 3 variants over (base + best Sleeve-2).**
| Axis | Values |
|---|---|
| Variant | A covered-call, B collar, C systematic-short, OFF |
| Strike OTM % | 3%, 5%, 7% |
| (C only) short risk % of book | 5%, 10% |
= A/B: 3 OTM each = 6; C: 3 OTM × 2 risk = 6; OFF = 1 → **13 cells**.

**Phase 3 — Combined best + robustness.**
- Best Phase-1 × best Phase-3 variant, full combined book.
- OOS split 2014–19 / 2020–26 (must hold in both).
- Post-tax @15% and @20% STCG.
- IV stress re-confirm on the chosen combo.
≈ **8 cells**.

**Total ≈ 1 + 28 + 13 + 8 ≈ 50 cells.** Each combined-book year-walk is
heavier than a plain RS sweep (options leg per signal per day) — expect
minutes/cell. Runs on **VPS** (canonical host, binding rule). Incremental
CSV per cell, chat check-in ~every 4 min while running.

**Known-skipped a priori:** Sleeve-2 debit + IV 0.30 + 5% risk combos
that Phase-1 shows dominated are not re-run in Phase-3.

---

## 5. Status (live running log)

**State header:** Phase = **ALL SLEEVES + COMBINED ENGINE BUILT & RUN
(laptop smoke)**. STATUS: DONE pending canonical VPS re-run for sign-off.
Verdict written → `results/RESULTS.md`. Honest read: overlays do NOT
transform the base; only KC6 *credit* spreads at realistic IV add a
modest Calmar gain; Sleeve 3 should be OFF (short sleeve is
drawdown-destructive; covered-calls neutral-to-negative). Next: push to
VPS, re-run `04_combined_book.py` there for canonical sign-off.

| Date/time IST | Event | Notes |
|---|---|---|
| 2026-05-17 | research/42 created; design questions answered | Capital=one-book-overlay; Sleeve3=decide-by-backtest; KC6=both-spreads-per-name; path=backtest-first |
| 2026-05-17 | Base read (research/41 detailed report + STATUS) | Locked config `q0.5_dd__v__REG`, 35.3%/−24.6%/1.53/1.44 |
| 2026-05-17 | Existing infra found | `services/collar_engine.py`+`collar_db.py` (KC6-signal options collar, flat-IV BS); research/41 scripts 10/21 prior hedge work; `COLLAR_DEFAULTS` in config.py |
| 2026-05-17 | STATUS-MD §1–8 written (this file) | Crash-recovery doc in place BEFORE any run |
| 2026-05-17 | `01_sleeve1_base_replay.py` built | Reuses research/41 `02_rs_sweep.py` loaders; replays locked `q0.5_dd__v__REG`; exposes per-month timeline (regime/holdings/cash/RS-rank) for overlays |
| 2026-05-17 | **Phase-0 GATE: PASS** (laptop smoke-test) | Replay = **35.28% CAGR / −24.58% MaxDD / Sharpe 1.53 / Calmar 1.44** vs research/41 target 35.3 / −24.6 / 1.53 / 1.44. 147 months, 118 RISK-ON / **29 bear-cash**. Snapshot DB matches VPS — re-confirm on VPS for the real sweep. |
| 2026-05-17 | **Capital-model correction (user)** | KC6 funding = margin against the whole equity/debt book, **constant & regime-independent** — NOT the bear-cash pool. §1/§2/§3.2/C3 corrected. Bear-cash framing removed. |
| 2026-05-17 | Sleeve-2 universe REFINED | KC6 native Nifty500∩F&O (~86), not mid-band-22 (would gut KC6). §3.2 corrected. |
| 2026-05-17 | `02_sleeve2_kc6_options.py` built + smoke-tested | Reuses `kc6_scanner.compute_indicators` + shared `bs_price`/strike/expiry from `collar_engine`. 80 F&O names, **361 KC6 trades 2014→26, 65–67% WR** (= KC6 native WR — fidelity OK). |
| 2026-05-17 | `03_sleeve3_variants.py` built + smoke-tested (after fixing a double-/s0 bug + making otm a real C width axis) | A/B/C all run off the Sleeve-1 timeline. Standalone smoke: A/B −ve, C +ve — **but this was PATH-BLIND (raw sum), later overturned by 04.** |
| 2026-05-17 | `04_combined_book.py` built + full ~31-cell sweep run (laptop smoke) | Single-book compose + risk-% caps + OOS. **Result OVERTURNS the standalone Sleeve-3 read:** path-aware, C systematic-short **deepens MaxDD to −34…−45%** (Calmar 0.74–1.28) — harmful. Only KC6 **credit** spreads add honest value (Calmar 1.44→~1.50, DD slightly shallower). Debit's big CAGR is an optimistic-IV artifact. **Sleeve 3 → OFF.** `results/RESULTS.md` written. |

**Live findings (partial):**
- Sleeve-1 is bit-faithful to the locked base — the combined comparison
  rests on a sound foundation.
- **29 of 147 months are bear/all-cash.** This is a Sleeve-1 risk
  observation only — it does NOT gate Sleeve 2. Per the locked capital
  model (user correction 2026-05-17), KC6's funds = margin against the
  whole equity/debt book, **constant and regime-independent**; the
  bear-cash months neither add to nor restrict KC6 capacity. The only
  cap on Sleeve-2 size is its swept max-risk-% of book (§4). What still
  matters in those 29 months: the KC6 crash filter (universe-ATR ≥ 1.3)
  will independently suppress most Sleeve-2 *entries* there — a signal
  effect, not a funding effect.
- **Sleeve-2 standalone (smoke):** 361 KC6 spread trades 2014→26,
  65–67% WR (= KC6 native WR — signal faithfully reproduced). **Debit
  bull-call dominates credit under flat-IV BS** (+Rs313–606k vs
  −Rs44k/+Rs57k total; debit better at low IV, credit better at high
  IV). Honest read: credit's apparent weakness is partly the C1 model
  bias (flat IV understates crash-time credit); debit's edge sits where
  BS is ~fair, so it is the more trustworthy of the two. Final arbiter
  = per-name best + IV sweep inside the combined engine. Rs ~11k
  defined risk/spread, ≤5 concurrent — modest book-margin footprint.
- **Sleeve-3 (smoke) answers the user's open question early:**
  **covered-calls (A) and collar (B) DESTROY value on this base**
  (−1.6 to −3.1% of final book, every OTM): writing calls on a momentum
  book sells off exactly the fat-tail winners that ARE the momentum
  edge, and only 131/~1770 held name-months are even F&O-eligible (the
  RS book is mostly illiquid non-F&O mid-caps — a structural mismatch
  between "covered calls" and *this* base). **Systematic-short (C) is
  the only additive Sleeve-3 variant** (+4.5 to +15.6%, best at tight
  3% width). STRONG caveat: C's magnitude is optimistic — flat-IV BS
  (C1) + only a 2% round-trip proxy for borrow/impact + weakest-RS
  mid-caps are the least shortable names in reality. Treat C as
  "directionally helps, magnitude not bankable." Provisional Sleeve-3
  pick = **C**, pending post-tax combined-engine confirmation.
- **⚠ CORRECTION (combined engine 04 overturned the above).** The
  standalone C "+15%" was a raw P&L SUM — path-blind. The path-aware,
  risk-capped combined engine shows **C deepens MaxDD to −34…−45%**
  (Calmar 0.74–1.28): shorting weakest-RS mid-caps bleeds through
  momentum bull years. A/B only look positive *atop the S2-debit CAGR*;
  standalone they are negative. **Final Sleeve-3 verdict = OFF (no
  variant earns its risk).** The only honest overlay is KC6 **credit**
  spreads at realistic/high IV (Calmar 1.44→~1.50, DD ~0.5–0.8pp
  shallower). Debit's large CAGR is an IV0.20 artifact (fades to ~base
  at IV0.30) — not bankable (C1). Full honest verdict +
  recommendation = `results/RESULTS.md`.

---

## 6. Crash Recovery — resume WITHOUT Claude

**Where things stand:** design locked, combined engine not yet built,
**no backtest launched**. Nothing live, no VPS process running. Safe to
do nothing; the base system (research/41) is independent and unaffected.

**To resume the build (the next concrete step):**
1. Read this file §3 + §4 — the full locked spec.
2. Reference scripts to copy from:
   - Sleeve 1 base replay: `research/41_midsmall400_mq_concentrated/scripts/03_rs_quality_volume.py`
     (config `q0.5_dd__v__REG`) and `02_rs_sweep.py` (universe + RS-120).
   - Sleeve 2/3 options: `services/collar_engine.py`, `services/kc6_scanner.py`,
     `config.py` `KC6_DEFAULTS` / `COLLAR_DEFAULTS`.
   - Prior hedge attempts (don't reinvent): `research/41_.../scripts/10_hedge_overlay.py`,
     `21_put_hedge_sketch.py`, `26_cashflow_policy.py`.
3. Build under `research/42_tri_sleeve_rs_kc6_overlay/scripts/`:
   `01_sleeve1_base_replay.py` → must reproduce 35.3%/−24.6% (Phase-0
   gate) → `02_sleeve2_kc6_options.py` → `03_sleeve3_variants.py` →
   `04_combined_book.py` (the sweep runner) → `05_oos_posttax.py`.
4. All read `backtest_data/market_data.db` (canonical on **VPS** — per
   binding rule, run backtests on VPS not laptop). Idempotent;
   incremental CSV.

**To check what finished (once a run exists):**
`tail research/42_.../results/*.log`; `wc -l results/combined_sweep.csv`;
compare against the ~50-cell plan in §4.

**To check if a VPS run is alive:**
`ssh arun@94.136.185.54 'pgrep -af 04_combined_book'` (paramiko if
OpenSSH password auth fails — see memory `vps_ssh_paramiko.md`).

**Files NOT to touch:** anything under
`research/41_midsmall400_mq_concentrated/` (the base study is COMPLETE
and frozen — Sleeve 1 only *reads* its logic, never edits it).
`services/collar_engine.py` / `kc6_scanner.py` (shared live code — copy
patterns into research/42 scripts, do not mutate the services).

**Files safe to inspect/edit:** everything under
`research/42_tri_sleeve_rs_kc6_overlay/`.

---

## 7. Files (output map)

| File | Purpose | Committable? |
|---|---|---|
| `TRI_SLEEVE_RS_KC6_HEDGE_DAILY_SWEEP_STATUS.md` | This crash-recovery doc | yes |
| `scripts/01_sleeve1_base_replay.py` | Replay locked RS base; Phase-0 fidelity gate | yes |
| `scripts/02_sleeve2_kc6_options.py` | KC6 → credit/debit spread, per-name pick | yes |
| `scripts/03_sleeve3_variants.py` | Covered-call / collar / systematic-short scaffolds | yes |
| `scripts/04_combined_book.py` | Single-book sweep runner (~50 cells) | yes |
| `scripts/05_oos_posttax.py` | OOS halves + post-tax @15/20% STCG | yes |
| `results/combined_sweep.csv` | Per-cell aggregate (small) | yes |
| `results/*_trades.csv` | Per-sleeve trade logs (heavy) | NO — gitignore |
| `results/*.log` | Per-phase progress log | NO — gitignore |
| `results/RESULTS.md` | Final honest verdict + sleeve-3 winner | yes |

---

## 8. Findings (during + final)

**None yet** — design only. To be filled live as Phase 0→3 run.

**Binding honesty caveats (cite WITH every number, do not hide):**

- **C1. No historical option chains.** `data/iv_history.db` is empty; the
  DB has only stock/index OHLCV. Every Sleeve-2/3 option leg is priced by
  **flat-IV Black-Scholes** (`iv_assumed`, `risk_free_rate=0.065`), the
  established repo convention (`COLLAR_DEFAULTS`). This **ignores the
  volatility smile, IV term structure, and IV spikes in crashes** — which
  systematically *flatters credit spreads* (real IV rises when KC6 fires
  on a selloff, so real credit collected > modelled) and is *roughly fair
  to debit spreads*. Mandatory IV sensitivity {0.20,0.25,0.30}. The
  options P&L is **indicative, not tradeable-accurate**. A real verdict on
  Sleeve 2/3 needs a paid historical options-chain source.
- **C2. Base is replayed, not re-fitted.** Sleeve 1 inherits all
  research/41 caveats verbatim (price-path "quality" is not fundamentals;
  PIT universe is a liquidity proxy ~68/100 MQ100 overlap; LTCG not
  modelled; ~−25% DD real). We do not re-open those.
- **C3. Single-book margin-capacity risk.** KC6/short-sleeve margin is
  collateralised by the same book, so aggregate overlay margin must stay
  within the book's true SPAN+exposure capacity at all times (NOT a
  bear-cash contention — funding is regime-independent, see §2). The
  engine models a hard book-level risk cap and flags any cell where
  combined overlay defined-risk would have exceeded that capacity
  (those cells are *infeasible*, not merely lower-return).
- **C4. Covered-call assignment is modelled as an upside cap**, not via
  a live option chain — conservative-ish but ignores early assignment /
  pin risk.
- **C5. No performance guarantee.** A measured, model-based edge. Nothing
  is wired live; real-capital deployment is a user decision after the
  RESULTS.md verdict.

**Decision rule for the final verdict:** if no combined cell beats base
post-tax Calmar 1.44 (or cut DD below −24.6% at CAGR ≥ ~25%) in **both**
OOS halves, the honest recommendation is **"base alone — the overlays do
not earn their complexity/risk."** That null result is a valid, expected
outcome and will be stated plainly.
