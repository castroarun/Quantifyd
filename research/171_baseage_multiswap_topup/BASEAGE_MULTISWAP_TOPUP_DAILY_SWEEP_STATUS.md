# OA · Base Age — swap MORE than one holding a night, and/or redeploy into EXISTING winners — STATUS: PRE-REGISTERED

**Book:** OA — Open Alpha · Base Age (research/161 adopted spec), the book that converts to
live today under research/165 with the single-swap rule **OA-ROT-1** staged behind a switch.

**STATUS: DONE — 13-Sep-2026. Verdict: Axis A NO EDGE, Axis B NO EDGE, Axis C SIGNAL but NOT
ADOPTED. The live book keeps OA-ROT-1 exactly as staged. Full write-up:
[`results/RESULTS.md`](results/RESULTS.md).**

---

## 1. The Ask

**What Arun asked (13-Sep-2026, verbatim):**

> "what if we swap the last 2 ranks instead of 1? can we find out an optimized number? or
> maybe swap the lowest ranked one(s) and instead of new entrants, top up the highest running
> ones existing within the portfolio?"

**What is actually being tested — three questions, kept separate:**

1. **How many holdings should leave on a swap evening?** OA-ROT-1 sells exactly one
   under-water holding per evening. Arun asks for two. The test sweeps `k = 1, 2, 3, 4, 6`
   and **all eligible**, where *eligible* means a holding more than the margin under water,
   ordered largest-loss-first — and asks whether there is an optimum, a plateau, or a hump.
2. **Where should the money go?** OA-ROT-1 gives the freed slot to the refused entrant with
   the highest 12-month relative strength. Arun asks whether it should instead **top up the
   strongest holdings the book already owns**. That is a different book: it spends the slot
   rather than refilling it, and it concentrates.
3. **Is a mixture better than either?** 50/50 between a new entrant and an existing winner,
   and "entrant if a signal is going begging, otherwise top up".

**What this study is NOT.** It does not touch a live file, it does not change OA-ROT-1 as
staged, and it does not re-litigate the entrant ranking (research/170 Part B settled that:
`rs252`). Nothing here is deployed. The result feeds the **26-Sep-2026** review of the live
Base Age conversion.

**Why it is worth asking.** research/166 did test `rot_max_per_day = 2` and `3` — but only on
the **tv20 entrant** construction, where it cost Calmar 0.673 → 0.599. OA-ROT-1 uses the
**rs252 entrant**, and research/170 Part B showed the entrant choice is worth 5.5 points of
drawdown on its own ("who leaves sets the return; who enters sets the drawdown"). The `k`
axis has therefore never been measured on the construction that is actually staged. The
top-up destination has **never been tested at all**, on any construction.

---

## 2. The Base — what is being tested

### 2.1 The book (unchanged from research/161's adopted spec)

| Element | Setting |
|---|---|
| Entry event | a new all-time-high **close**, prior ATH ≥ **60** trading bars old, the name having fallen ≥ **20%** below that prior ATH, 20-day median traded value ≥ **₹2 cr**, 60-bar per-symbol re-arm |
| Fill | the **next open** |
| Slots / size | **16 slots at 6.25% of NAV**, ₹10 lakh book, NSE cash CNC, long only |
| Contested slot | seeded **random** draw (the ensemble axis); the live `tv20` tie-break is reported as a control only |
| Exit | **SuperTrend(14, 4)** turning down on the **close**, sold at the next open. No hard stop, no target, no time stop |
| Costs | **25 bps a side** (40 and 60 bps as a sensitivity) |
| Tax | **20% STCG / 12.5% LTCG above 365 days**, Indian FY loss-netting, settled 1 April, modelled inside the engine — never a haircut |
| Idle cash | **5.2% p.a. post-tax**, credited daily (Arun's binding standard) |
| Window | **2005-01-03 → 2026-09-11** (21.7 years), 3,619 frozen events |
| Ensemble | **30 seeds**, seed base **7000** (seeds 7001–7030) — a set no cell in research/164, 166 or 170 has touched |

### 2.2 The incumbent and the reference

| name | rule |
|---|---|
| **INCUMBENT** | never swap. This is the book as research/161 adopted it and the thing every cell must beat |
| **OA-ROT-1** (reference) | research/170 Part B `X_entrs_unre_m010`, staged live today: on an evening when a qualifying signal is refused, sell the **single** holding more than **10%** under water (largest loss first), and buy the refused entrant with the highest **rs252**. **One swap per evening.** Both legs at the next open |

Every candidate is reported twice: **paired against the INCUMBENT** (the adoption test) and
**paired against OA-ROT-1** (the practical test — does it beat the rule already staged?).

### 2.3 The three new mechanics, exactly

**A. `k` — how many holdings leave per evening.** The engine already implements this
(`rot_max_per_day`): the refused entrants are walked in `rs252` order, and each one, in turn,
is matched against the **currently weakest** eligible holding. Because the eligible list is
rebuilt after each swap, `k = 2` sells the weakest and then the second-weakest — exactly
"the last 2 ranks". A swap only fires if that holding's loss is deeper than the margin, so
`k` is a **cap**, not a quota. **No code change is needed for axis A.**

Because each swap consumes one refused entrant, `k = 2` can only fire twice if **two**
signals were refused that evening. Three answers to "what if there is only one entrant", and
the obvious two are tested:

| variant | what happens to the second eligible loser |
|---|---|
| **`spill = none`** (the engine's own behaviour, the default) | nothing — it is kept |
| **`spill = cash`** | it is sold and the proceeds sit in cash at 5.2% |
| **`spill = topup`** | it is sold and the proceeds top up the strongest holding |

**B. Destination = top up an existing winner.** Same sell trigger (a holding more than the
margin under water). The proceeds do **not** buy a new entrant; they buy more of a holding
the book already owns, ranked **highest** by `X`:

| `X` | definition (read on the signal close, `close[i-1]`) |
|---|---|
| `rs` | `rs252` — 12-month price return |
| `unreal` | unrealised return against the position's average buy price |
| `cush` | `100 × (close / SuperTrend(14,4) line − 1)` — cushion above the trailing stop |

Crossed with **split** (all to the top one / equally across the top two) and a **position
cap** (2× / 3× / none, of the 6.25% target weight, measured against NAV on the signal close).
A top-up that would breach the cap is trimmed to the cap; if nothing can be bought, **the
sale is cancelled** — the same "cancel both legs" discipline OA-ROT-1 uses.

The **trigger** is tested two ways: `signal` (only on an evening when a qualifying signal was
refused — Arun's framing) and `any` (any evening with an eligible loser — pure internal
rotation, no signal needed).

**C. Hybrid.** `hybrid` splits the proceeds 50/50 between the best refused entrant and the
top existing winner. `els` ("entrant else top-up") takes the entrant when a signal is going
begging and tops up otherwise.

**D. Interaction with the tie-break and the 60-bar re-arm — the ruling, made in advance.**

- A **top-up is not an entry.** It consumes no event, occupies no slot and cannot re-arm
  anything: the 60-bar re-arm lives in the event generator (`build164.py`), per symbol,
  and is blind to the book. A topped-up name therefore behaves exactly as it did before for
  every purpose except its size.
- A top-up **resets nothing and hides nothing**: the position keeps its original `entry_i`
  (so the reported holding period stays honest), its `entry_px` becomes the **weighted
  average** buy price (so the −10% trigger is measured against what the book actually paid),
  and the purchase is recorded as its **own tax lot** so the LTCG clock is not handed to
  shares bought yesterday. A single-lot position — every cell that does no top-ups — is
  taxed by exactly the inherited arithmetic.
- A position **bought or topped up today cannot be sold today** (`last_buy_i >= i`).
- The `tv20` tie-break is untouched. It is a **control arm**, not a cell.

### 2.4 The metric and the adoption bar — pre-registered, applied unchanged

**Ranking metric: after-tax Calmar, with CAGR ≥ the INCUMBENT's** (the same eligibility
clause research/166 used, which is why a plain hard stop was excluded there).

**Adoption bar vs the INCUMBENT:** paired **≥ +0.10 Calmar** OR **≥ +2pp CAGR at no worse
drawdown**, on **≥ 20 of 30 seeds**, in **both** windows, with a plateau (both immediate
neighbours agreeing in direction), surviving **40 bps** a side.

**Windows:** fit **W1 = 2005-01 → 2015-12**, holdout **W2 = 2016-01 → 2026-09**. A cell whose
W1 exceeds its W2 by more than **4pp** of CAGR is treated as fitted and rejected.
**W3 = 2025-01 → 2026-09** is reported for every shortlisted cell because that is the regime
Arun will watch live — it is a *reporting* window, never a selection window. All window
drawdowns are measured from the running peak of the **full** curve.

**Nulls and controls (not selection cells):** random-swap at the matched rate for each `k`;
**sell-only** (sell the loser, do not redeploy) at each `k` — research/166 found sell-only
*loses* to doing nothing, and that must reproduce on the new seeds or something is wrong.

**Cell budget: ≤ 70 selection cells.** Because one cell costs ~6 seconds, every selection
cell is run at the full **30 seeds** rather than scanned at 10 — a strengthening of the
stated protocol, not a relaxation.

### 2.5 Harness

`sim171.py` is generated from research/170's `sim170.py` by **12 exact-string patches**
(`patch171.py`), each of which must match exactly once or the build aborts — the same
discipline research/170 used on research/160's engine and research/165 used on this one.
research/170's files are never edited. `patch171.py --verify` re-runs research/170's own
incumbent and OA-ROT-1 cells through **both** modules on the same seed and requires the NAV
curve and the trade list to be **bit-identical**. Then the two cells are re-run on
research/170's fresh seed base (1000) and must return its published rows:

| cell | required |
|---|---|
| `BASE_rand` (incumbent) | **20.945 / −34.045 / 0.611** |
| `X_entrs_unre_m010` (OA-ROT-1) | **22.58 / −31.78 / 0.710** |

No selection cell runs until both match.

---

## 3. The Plan — the pre-registered cell list

### Stage `proof` — harness (2 cells, seed base 1000, NOT selection)

`BASE_rand`, `X_entrs_unre_m010`.

### Stage `main` — 27 selection cells + 2 references + 5 controls, seed base 7000

**References (not selection):** `REF_base` (incumbent), `REF_rot1` (OA-ROT-1, `k = 1`).

**Axis A — how many leave (9 selection cells)**

| label | k | margin | spill |
|---|---|---|---|
| `A_k2` | 2 | 10% | none |
| `A_k3` | 3 | 10% | none |
| `A_k4` | 4 | 10% | none |
| `A_k6` | 6 | 10% | none |
| `A_kall` | 99 (all eligible) | 10% | none |
| `A_k2_m075` | 2 | 7.5% | none |
| `A_k2_m125` | 2 | 12.5% | none |
| `A_k2_spillcash` | 2 | 10% | cash |
| `A_k2_spilltopup` | 2 | 10% | top up (rs, top-1, cap 3×) |

`k = 4` is also the "all eligible, capped at 25% of a 16-slot book" cell Arun's framing asks
for; it is labelled `A_k4` and reported as both.

**Axis B — destination = top up an existing winner (18 selection cells)**

`X ∈ {rs, unreal, cush}` × `k ∈ {1, 2}` × `cap ∈ {2×, 3×, none}`, split = top-1, trigger =
signal, margin 10%. Labels `B_S_k<k>_<X>_c<cap>`.

**Axis C — hybrid (4 selection cells)**

| label | rule |
|---|---|
| `C_hyb_k1`, `C_hyb_k2` | proceeds split 50/50 between the best refused entrant and the top existing winner |
| `C_els_k1`, `C_els_k2` | entrant when a signal is refused, otherwise top up (trigger = any evening) |

**Controls (5, not selection):** `CTRL_sellonly_k1/k2/k3`, `CTRL_null_k2`, `CTRL_null_k3`.

### Stage `follow` — 4 selection cells (executing the pre-registered follow-ups)

The winning Axis-B configuration re-run with **split = top-2** at `k ∈ {1, 2}`, and with
**trigger = any evening** at `k ∈ {1, 2}`.

### Stage `plat` — plateau on whatever wins (≤ 4 selection cells)

The winner's two immediate neighbours on its own key axis.

### Stage `cost40` / `cost60` — re-scoring, not selection

The shortlist at 40 and 60 bps a side.

**Selection cells planned: 9 + 18 + 4 + ≤4 = ≤ 35, against a budget of 70.**

### Recorded per cell (beyond CAGR / MaxDD / Calmar / Sharpe)

swaps per year · top-ups per year · average invested % · cash refusals · slot refusals ·
turnover × NAV · tax paid in rupees · win rate · average win / loss · expectancy · max losing
streak · trades per year · share of profit from the ten best realisations · median position
as % of the held name's own tv20 · **max position weight ever reached** · and the
**eligibility histogram** — on how many refused-signal evenings were ≥ 1, ≥ 2, ≥ 3 holdings
simultaneously more than 10% under water. **That last number is reported before any CAGR**,
because if two eligible losers are rare, the whole `k = 2` question is moot regardless of
what the return column says.

---

## 4. Status log

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-09-13 ~09:30 | Context read | research/161, 164, 165 (both STATUS docs), 166, 170 Part B; playbook §4-§6; agent file |
| 2026-09-13 ~09:55 | Sections 1-4 written | Grid, metric and adoption bar locked before any cell |
| 2026-09-13 09:38 | `sim171.py` generated, no-op PROVED | 12 exact-string patches; NAV, trades, invested and book all bit-identical to research/170's `sim170.py` on the incumbent and on OA-ROT-1, seeds 1001 and 1017 |
| 2026-09-13 09:40 | Harness proof PASSED | incumbent 20.945 / −34.045 / 0.611 and OA-ROT-1 22.58 / −31.78 / 0.710 on research/170's own seed base — its published rows to the digit |
| 2026-09-13 09:41 | `main` launched, 38 cells × 30 seeds | seeds 7001-7030, 2 workers, nice 10 |
| 2026-09-13 09:45 | **Axis A settled inside four minutes** | `k = 3, 4, 6` and all-eligible returned bit-identical results; `k = 2` a wash against `k = 1`. The eligibility counter explains it: two holdings qualify together **0.5 times a year** once the rule runs |
| 2026-09-13 09:46 | **Axis B settled** | all 18 top-up cells lose Calmar to the incumbent; none beats OA-ROT-1 on any seed; single positions reach 52-81% of NAV |
| 2026-09-13 09:46 | `main` done in 249s | `C_els_k1` (entrant-else-top-up) led at Calmar 0.736 — flagged for the plateau and control clauses |
| 2026-09-13 09:52 | `measure` + `follow2` done | `CTRL_measure_k0` reproduced the incumbent exactly; the margin and cap plateaus on the rs252 variant FAILED; `CTRL_anycash_k1` (stop to cash) reached 0.692 of the 0.736 with no machinery |
| 2026-09-13 09:58 | `follow` (pre-registered Axis-B follow-ups) + cost ladder done | `C_els_k1` collapses to 0.638 at 40 bps, below OA-ROT-1's 0.684 |
| 2026-09-13 10:05 | `follow3` done — the cushion variant | `C_els_k1_cush` clears the letter of the bar (+0.128 on 30/30, survives 40 bps) but `C_cush_sigonly` returns **OA-ROT-1 to the digit**, proving the top-up leg contributes nothing on refused-signal evenings |
| 2026-09-13 10:12 | Reports written | `paired171.md`, `elig171.md`, `yoy171.md/.html`, `r171.png` |
| 2026-09-13 10:20 | RESULTS.md written, STATUS closed | 51 selection cells of a 70 budget; ≈ 2,820 simulations |

---

## 5. Crash Recovery — how Arun resumes without Claude

Everything runs on the VPS in `/home/arun/quantifyd/research/171_baseage_multiswap_topup`.
Nothing outside that folder is written until the publish step.

```bash
ssh arun@94.136.185.54
cd /home/arun/quantifyd/research/171_baseage_multiswap_topup

# 1. Is anything still running?
pgrep -af run171.py

# 2. How far did it get?  One row per finished cell; finished cells are SKIPPED on restart.
wc -l results/cells_*.csv
tail -5 results/main.log

# 3. Rebuild the engine and prove it is research/170's engine (30 seconds)
cd /home/arun/quantifyd
venv/bin/python3 research/171_baseage_multiswap_topup/scripts/patch171.py --verify
#   must print: NAV identical: True / trades identical: True  for BOTH cells

# 4. Re-run the harness proof (must reproduce research/170's published rows)
nice -n 10 venv/bin/python3 research/171_baseage_multiswap_topup/scripts/run171.py \
     --stage=proof --seedbase=1000 --seeds=30 --workers=2

# 5. Resume the main sweep (safe to re-run: it skips completed cells)
cd /home/arun/quantifyd && setsid nohup nice -n 10 venv/bin/python3 -u \
  research/171_baseage_multiswap_topup/scripts/run171.py --stage=main --seedbase=7000 \
  --seeds=30 --workers=2 \
  > research/171_baseage_multiswap_topup/results/main.log 2>&1 < /dev/null &
```

**Safe to inspect:** every file under `results/`. **Do not delete** `results/cells_*.csv` or
`results/seedstats_*.csv` — they are the resume state. **Never edit**
`research/170_qs_leeway_and_baseage_best_entrant/scripts/sim170.py`; `sim171.py` is generated
from it and `patch171.py` aborts if that file has drifted.

**Inputs, all read-only and frozen:** `research/164_baseage_slots_sizing/results/panel164.pkl`
(the price panel), `research/166_baseage_rotation_and_drift/results/st166.pkl` (SuperTrend
lines), `research/166_baseage_rotation_and_drift/results/events166.csv` (3,619 events).

---

## 6. Files

| file | purpose | committable |
|---|---|---|
| `BASEAGE_MULTISWAP_TOPUP_DAILY_SWEEP_STATUS.md` | this file | yes |
| `scripts/patch171.py` | generates `sim171.py` from research/170 `sim170.py` by 12 exact-string patches; `--verify` proves the no-op | yes |
| `scripts/sim171.py` | GENERATED — the engine. Do not hand-edit | yes |
| `scripts/run171.py` | the sweep runner, resume-safe, one row per completed cell | yes |
| `scripts/report171.py` | paired tests, YoY table, curves | yes |
| `scripts/publish171.py` | the app study page | yes |
| `results/cells_*.csv` | one summary row per cell | yes |
| `results/seedstats_*.csv` | one row per (cell, seed) | yes |
| `results/navs_*/*.npz` | NAV curves per cell (30 × 5,378 float32) | no — gitignored |
| `results/*.log` | run logs | yes (small) |
| `results/RESULTS.md` | the verdict | yes |

---

## 7. Findings

**The full write-up is `results/RESULTS.md`.** The three answers, in one line each:

1. **Swap two instead of one? No.** Two holdings are more than 10% under water on the same
   refused-signal evening **0.5 times a year** once the rule is running (4.2 times a year on a
   book that never swaps — the rule destroys its own second opportunity). `k = 3, 4, 6` and
   "all eligible" are **bit-identical on all 30 seeds**; `k = 2` costs −0.21pp of CAGR and
   −0.003 of Calmar at 25 bps and loses outright at 40 bps.
2. **Is there an optimum k? Yes: one.** The axis has only two distinguishable points.
3. **Top up existing winners instead of new entrants? No.** All 18 constructions lose Calmar to
   doing nothing; none beats OA-ROT-1 on a single seed of thirty. A single position reaches
   52-81% of NAV and the ten best trades supply 55% of book profit.

**The one cell that clears the bar is not Arun's idea.** `C_els_k1_cush` reaches Calmar 0.735
(+0.128 on 30/30, survives 40 bps) — but the control `C_cush_sigonly`, which restricts it to
refused-signal evenings, returns **OA-ROT-1 to the digit with zero top-ups**. Every point of its
advantage comes from selling on evenings when **no signal fired**, i.e. from an unconditional
−10% stop; and a plain −10% stop with no machinery at all reaches **0.683**. 94% of its return
edge sits in the 2016-2026 half of the history.

**Recommendation: the live Base Age book changes nothing.** Registered for the 26-Sep-2026
review: expect ~4.5 swaps a year and about **one occasion every two years** where a second
holding would also have qualified — and do not read that second loser as a missed opportunity.
