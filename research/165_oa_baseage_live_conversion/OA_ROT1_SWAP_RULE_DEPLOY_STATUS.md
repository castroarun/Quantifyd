# OA · Base Age — the OA-ROT-1 best-entrant SWAP rule, built into the staged conversion — STATUS: DONE, SWITCH STILL OFF

**OA (Open Alpha).** This document is the sole crash-recovery source for adding **one rule**
— the research/170 Part B best-entrant swap, **OA-ROT-1** — to the *already staged, still
switched-off* Base Age conversion of the live ₹6.18L Open Alpha book.

It is a **child of** `OA_BASE_AGE_LIVE_CONVERSION_DEPLOY_STATUS.md` in this same folder.
That document is still the runbook; this one records what the swap rule is, that it was
proved against research/170's own engine, and what it changes in the flip. **§9 below is the
runbook delta that gets folded into the parent's §9.**

- **Book:** Open Alpha REAL, Zerodha RA6610, CNC equity, NSE cash.
- **Switches:** `OA_RULESET` in `services/oa_real.py` — `'legacy'` (current, default) |
  `'baseage'`.  And now, beside it, `OA_ROT1` — `True` (staged) | `False` (swap disabled,
  Base Age kept).
- **Written:** 13-Sep-2026 (Sunday, market closed). Next session Monday 15-Sep-2026.
- **Nothing in this work has flipped a switch, placed an order, edited the crontab,
  restarted a service, or written `backtest_data/oa_real_state.json`.**

---

## 1. The Ask

**What Arun asked (verbatim, 13-Sep-2026 morning):**

> "Swap, entrant by relative strength 22.58% / −31.8% — I love this. Let's make changes to
> the live system later today, not now."

**What is actually being built:**

research/170 Part B re-ran research/166's post-hoc "highest-RS entrant" swap on a seed set
it had never seen (1001–1030) and it came back: **22.58% CAGR after tax, −31.78% max
drawdown, Calmar 0.710**, against the un-rotated incumbent's 20.95 / −34.05 / 0.611 —
**+0.105 paired Calmar on 30 of 30 fresh seeds**, +1.65pp CAGR on **60 of 60** pooled paths,
and 60 of 60 against a rate-matched random swap. research/170's own verdict was **SIGNAL,
not adopted** — it missed the pre-registered +0.10 Calmar bar, pooled, by four thousandths.

Arun has now read that evidence and **adopted the rule**. So this session:

1. **Builds OA-ROT-1 into the staged 18:50 evening job**, on the `'baseage'` branch only,
   behind its own `OA_ROT1` constant so it can be rolled back without losing Base Age.
2. **Proves it against research/170's own engine** — not against a re-implementation:
   research/170's `sim170.py` is patched by an auditable exact-string patch script to emit
   every rotation decision it makes, and the **live** `rot1_pick()` is asked to re-decide
   each one. The swap days and the swap pairs must match.
3. **Dry-runs it on the live book** as of the 11-Sep-2026 close.
4. Leaves **`OA_RULESET = 'legacy'`**. A `git pull` on the VPS still changes no running
   behaviour, and the legacy branch is byte-for-byte unaffected.

**What is explicitly NOT being done**, on Arun's own words ("later today, not now"): no
flip, no order, no crontab edit, no restart, no state write.

---

## 2. The Base — the rule, exactly

### 2.1 OA-ROT-1 as adopted (research/170 Part B `X_entrs_unre_m010`)

On an evening when **at least one qualifying Base Age signal fires AND the book cannot take
it** — no free slot, or cash below one slot of 6.25% of NAV:

| Element | Rule |
|---|---|
| **Trigger** | ≥ 1 qualifying signal refused **after** the SuperTrend exits and the normal entries have been queued |
| **Who leaves** | among the open positions, the one with the **largest loss versus its buy price**, measured on the **trigger close** (`100 × (close/buy − 1)`, lowest wins; ties broken by symbol, ascending) |
| **The margin** | the swap fires only if that loss is **deeper than 10%** — `close ≤ 0.90 × buy`. This is `(0 − weakest_unrealised) ≥ 10.0`, i.e. the engine's `rot_margin = 10.0` against `ent_score = 0` for a brand-new position |
| **Who enters** | among the refused qualifying signals, the one ranked **highest by 12-month relative strength**: `rs252 = close[t] / close[t−252] − 1`, computed on the **trigger close**. A name with fewer than 252 bars ranks **last**. Ties broken by symbol |
| **Not eligible to leave** | a position bought at the same open (impossible live: it is not in the book yet), and any position that **already has a SELL resting** — a SuperTrend exit that day is an ordinary exit, **not** a swap |
| **Not eligible to enter** | a name the book already holds, or one that already has an entry order armed |
| **Fill** | **both legs at the next open**, AMO placed the same evening: **SELL first, then BUY** |
| **Size** | 6.25% of NAV *after* the sale, funded from the proceeds; whole shares |
| **Cap** | **at most one swap per evening** (`rot_max_per_day = 1`) |
| **If unfundable** | if no refused entrant can be bought with one whole share out of `min(slot, cash + proceeds)`, **both legs are cancelled** — nothing is sold and nothing is bought |
| Everything else | **unchanged.** SuperTrend(14,4) exit, the 60-bar re-arm, the ₹2 cr liquidity floor, the traded-value tie-break for the normal entries, 16 slots at 6.25% |

**The order of operations inside the 18:50 job is the rule**, because it decides what counts
as "refused":

```
1. confirm()            ST(14,4) exits on the official close  -> AMO SELL (tag OA-EXIT)
2. baseage scan + plan  normal entries                        -> AMO BUY  (tag OA-ENTRY)
3. OA-ROT-1             only if a qualifying signal is STILL refused after 1 and 2
```

Step 1 frees slots and cash for step 2, exactly as research/170's engine does (a position
whose exit was queued on yesterday's close has already left `open_pos` before the rotation
block runs). A signal that step 2 could fund is never a swap.

### 2.2 The evidence being relied on (research/170 `results/RESULTS.md` Part B)

Window 2005-01-03 → 2026-09-11, 21.7 years, after tax (20% STCG / 12.5% LTCG, FY netting),
25 bps a side, idle cash 5.2% post-tax, 16 slots at 6.25%, ₹10 lakh book, **30 fresh seeds
(1001–1030) that no earlier cell had touched**.

| | incumbent (never swap) | **OA-ROT-1 (entrant = rs252)** | entrant = tv20 | entrant = base age | random swap, same rate |
|---|---:|---:|---:|---:|---:|
| CAGR after tax | 20.95% | **22.58%** | 22.77% | 22.50% | 20.80% |
| worst of 30 seeds | 19.42% | **21.78%** | 21.45% | 20.52% | 18.79% |
| MaxDD | −34.05% | **−31.78%** | −33.52% | −37.23% | −34.42% |
| Calmar | 0.611 | **0.710** | 0.671 | 0.605 | 0.603 |
| paired ΔCalmar vs incumbent | — | **+0.105 on 30/30** | +0.058 on 28/30 | −0.018 on 11/30 | +0.001 on 15/30 |
| pooled over both seed sets (60 paths) | — | **+0.096 ΔCalmar on 60/60, +1.65pp ΔCAGR on 60/60** | +0.051 on 55/60 | −0.013 on 26/60 | −0.037 on 18/60 |
| swaps / yr | 0.0 | **4.5** | 4.3 | 4.4 | 3.9 |

**Three facts that shaped the implementation, not just the decision:**

1. **Who leaves sets the return; who enters sets the drawdown.** All three entrant rules earn
   the same +1.6 to +1.8pp. The whole spread — 5.5 points of drawdown, 0.105 of Calmar —
   is in *which refused breakout you buy*. That is why the entrant ranking is `rs252` and
   why it is worth the machinery.
2. **The 10% margin is the only one that wins both windows.** 7.5% earns in 2005-2015 and
   *loses* in 2016-2026; 12.5% does the opposite. The working band is 7.5–12.5% and it is
   thin. The threshold is a constant in the source, not a tunable.
3. **research/170 did not adopt it.** It missed its own bar by 0.004 Calmar, and listed four
   reasons not to ship — the top one being "the book has never traded live". Arun has
   overridden that on the strength of 60/60. **That override is recorded here as a decision,
   not laundered into evidence**, and it is why the rollback is a separate one-word switch.

### 2.3 Deviations from the study — declared, not hidden

The parent STATUS already declares D1–D6. These are the ones OA-ROT-1 adds.

| # | Study / engine did | Live does | Why |
|---|---|---|---|
| **R1** | Decides on `close[i−1]` and fills both legs at `open[i]`, knowing `open[i]` exists | Decides on the **official close of the trigger day** in the 18:50 job and places two **AMO**s for the next open | Same information set — a close signal, a next-open fill. A live book cannot check tomorrow's open exists before it arms. Same order type and MARKET→LIMIT fallback as the staged entries and exits (parent D3) |
| **R2** | Sale proceeds = `shares × open[i] × (1 − cost)`; NAV for sizing = cash + close-of-day value of the rest of the book | Proceeds estimated at the **trigger close**; NAV for sizing = cash after the armed entries **+** estimated proceeds **+** the rest of the book marked at the trigger close **+** the armed entries at their estimated cost | Tomorrow's open is unknowable this evening. The estimate moves the *size* by the overnight gap, never the *decision* — the fire test does not read either number |
| **R3** | `rs252` from the study panel's forward-filled close, 252 **panel-calendar** bars back | `rs252` from the symbol's own daily bars, **252 rows back**, `volume > 0 and close > 0`, duplicates dropped keeping the last | The live scanner has no panel. For a name clearing the ₹2 cr liquidity floor the two calendars coincide except across a trading halt. **Measured, not asserted** — §8.1 reports every entrant-rank disagreement this causes |
| **R4** | A held position always has a close (forward-filled) | A held position whose latest DB bar is **not** the trigger session is **not eligible to be swapped out**, and says so | Same guard as `confirm()`: an un-refreshed bar is not a reading of the rule. Erring toward *not* selling |
| **R5** | The engine may try the next entrant when the top one cannot be bought with a whole share | **Same** — entrants are walked in `rs252` order and the first fundable one takes the slot | This is the engine's behaviour (`shares <= 0 → continue`), and it is also what makes research/166's "cancel both legs" clause correct: both legs are cancelled only when **no** entrant is fundable |
| **R6** | No 15:18 reading exists | The 15:18 close-proxy check is **unchanged** and says nothing about swaps | research/166's proposal text put the scoring at 15:18. The parent's D2 already settled that a proxy close is a forecast: the decision is made on the official close, in the evening, or it is a different rule |

### 2.4 Live state at the moment this work started (13-Sep-2026, 08:41 IST)

Read on the VPS, read-only:

- `OA_RULESET = 'legacy'` (line 58 of `services/oa_real.py`). Entry crons **112** and **114**
  still commented. HEAD `0e71e146`.
- **11 open positions**, all `entry_date` 2026-09-04; **cash ₹1,88,697.86 · capital
  ₹6,17,637.68**; 5 closed trades.
- Last official session in `market_data.db`: **2026-09-11**.
- The four staged files are byte-identical on laptop and VPS (md5 matched before any edit).
- **The parent's §8.0 blocker is unchanged and still blocks the flip**: `equity_executor.py`
  at 09:20 (crontab line 107) still tops up OA holdings with real orders. OA-ROT-1 does not
  touch it and does not resolve it.

| symbol | qty | buy | 11-Sep close | P&L % |
|---|---:|---:|---:|---:|
| INDSWFTLAB | 99 | 362.33 | 389.00 | +7.36% |
| SETL | 99 | 402.69 | 468.15 | +16.26% |
| WELCORP | 14 | 2,596.36 | 2,686.90 | +3.49% |
| SHILPAMED | 40 | 962.36 | 959.05 | −0.34% |
| **SBCL** | 34 | 1,115.48 | 1,088.00 | **−2.46%** ← the deepest loss in the book |
| IRISDOREME | 634 | 62.39 | 61.97 | −0.67% |
| INOXINDIA | 17 | 2,236.50 | 2,242.50 | +0.27% |
| MANINDS | 47 | 800.19 | 870.00 | +8.72% |
| SSWL | 106 | 358.30 | 371.00 | +3.54% |
| ENTERO | 21 | 1,843.72 | 1,806.90 | −2.00% |
| NITINSPIN | 61 | 636.65 | 652.00 | +2.41% |

---

## 3. The Plan

| Step | Deliverable | Gate |
|---|---|---|
| A | `services/oa_baseage_entry.py` — `rot1_pick()` (pure, testable) + `rot1_run()` (arms the two AMOs), wired into `run()` after the normal entries | imports clean; `--asof` replay prints a sane decision |
| B | `services/oa_real.py` — `OA_ROT1` constant, the two order tags, `place_exit_amo(tag=…)`, reconcile applying a swap as a swap | `OA_RULESET='legacy'` output unchanged; legacy diff is additive only |
| C | `scripts/patch_probe170.py` → `scripts/sim170_probe.py` — research/170's engine with a decision probe, generated by exact-string patches | the probe run reproduces research/170's own swap count for the adopted cell |
| D | **Replication gate R1** — every rotation decision point the engine makes, re-decided by the live `rot1_pick()` | **100% agreement on fire/no-fire, the name sold and the name bought**; every mismatch named |
| E | **Replication gate R2** — the live code walked end to end over the last 400 sessions on research/164's event list, from the live book's state **and** from an empty book | runs; swaps and converted refusals counted |
| F | **60-day walk on live capital** + the **Monday dry-run** on the real 11 positions | table produced, nothing written |
| G | Runbook delta (§9), register (`strategies.ts`), ops reviews, `TODO.md` | frontend build green, string present in the bundle |

### The grid there is to check

This is a rule adoption, not a sweep: **there are no cells**. research/170 already spent the
5 selection cells and the 2 plateau neighbours; re-running them here would be re-testing,
and the margin is deliberately not a tunable. The three things *measured* rather than
asserted are (D) decision-for-decision agreement with the engine, (E) that the live code
produces swaps at all on the live event stream, and (F) what would happen on Monday.

---

## 4. Status log

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-09-13 08:32 | Read the sources | parent STATUS (all), the four staged services files, research/170 STATUS + RESULTS Part B, `sim170.py` rotation block + `build_aux`, `run170b.py` cell list, research/166 harness inputs, `bt_core.Panel` |
| 2026-09-13 08:41 | Live state read on the VPS, read-only | `legacy`, 11 positions, cash ₹1,88,697.86, HEAD `0e71e146`, crons 112/114 still commented |
| 2026-09-13 08:55 | **§1–4 written before any code** | this file |
| 2026-09-13 09:20 | A + B built on the laptop, shipped to the VPS | switch still `'legacy'`; `OA_ROT1 = True` but unreachable |
| 2026-09-13 08:52 | C built; `--verify` proves `PROBE=None` is a no-op; 95 swaps = 4.4/yr | §8.0 |
| 2026-09-13 08:56 | Gate R1 pair-agreement 100.00% — but the **entrant RANKING only 91.8%** | investigated rather than accepted |
| 2026-09-13 08:58 | **`rs252` defect found and fixed**: read on the study's NIFTYBEES master calendar, not the symbol's own bars | §8.2 |
| 2026-09-13 09:02 | **Gate R1 re-run: PASS, 100.00% on 8,488 decisions; R3 ranking 100.00%** | §8.1 |
| 2026-09-13 09:02 | **Gate R2 walked, both starting books + the 60-day walk** | §8.3 — and the rule **loses** on this window |
| 2026-09-13 09:05 | **Dry-run: no swap possible Monday, and none even on a full book** | §8.4 |
| 2026-09-13 09:12 | Register: `strategies.ts` + LABS doc written; **`ops_center.py` and `TODO.md` SKIPPED — a concurrent session had landed `register170.py` on the VPS between this script being written and run** | the marker guard doing its job |
| 2026-09-13 09:18 | VPS copies pulled back to the laptop; register re-written to **amend** research/170's entries rather than duplicate them; applied | ops parses, 1 review dated 2027-03-13 |
| 2026-09-13 09:25 | `npm run build` green; `Swap rule OA-ROT-1` and `OA-ROT1-SELL` present in `static/app/assets/index-MIvD2W0u.js` | frontend-only, no restart |
| 2026-09-13 09:30 | **Committed on the VPS as `08496a7f`, NOT pushed** | 14 files: `services/oa_real.py`, `services/oa_baseage_entry.py`, `research/165/**`, `strategies.ts`, `ops_center.py`, LABS doc, `TODO.md` |
| 2026-09-13 09:32 | Post-commit safety re-check | `OA_RULESET='legacy'`, `OA_ROT1=True` (inert), crontab 112 and 114 still commented, state file still 11 positions / cash ₹1,88,697.86 |
| 2026-09-13 09:40 | **Follow-up 1: the rate gap settled** — research/170's own engine run over the same 400-session window, 30 seeds, empty book and the live 11 | **§8.0 — the engine also fires 5.7–7.9 swaps/yr there and also loses (6/30, 3/30). It is the period.** |
| 2026-09-13 09:45 | **Follow-up 2: `plan()` now sizes off the MARKED NAV** | §8.5 — one default changed; `marks`/`asof` added; stale names named in `ctx['unmarked']` |
| 2026-09-13 09:50 | Re-verified after the fix | dry-run **PAYTM ×21 @ ₹38,157** (matches the parent §8.4); **gate R1 still 8,488/8,488 = 100.00%**; R2 re-walked (§8.3), swaps 20→16 and 18→15 |
| 2026-09-13 09:55 | Committed on the VPS, not pushed | second commit; switch still `'legacy'` |

---

## 5. Crash Recovery — how Arun resumes without Claude

**Nothing here is half-applied.** OA-ROT-1 is unreachable until `OA_RULESET` becomes
`'baseage'`, and switchable off on its own after that.

```bash
# 1. Both switches, and what they are set to
ssh arun@94.136.185.54 "grep -n \"^OA_RULESET\|^OA_ROT1\" /home/arun/quantifyd/services/oa_real.py"
#    expected TODAY:  OA_RULESET = 'legacy'   and   OA_ROT1 = True   (inert while legacy)

# 2. The no-op proof: a git pull changed nothing that runs
ssh arun@94.136.185.54 "cd /home/arun/quantifyd && \
  venv/bin/python3 services/oa_real.py ruleset && \
  venv/bin/python3 services/oa_baseage_entry.py && \
  venv/bin/python3 -m py_compile services/oa_real.py services/oa_entry.py \
      services/oa_baseage.py services/oa_baseage_entry.py && echo COMPILE_OK"
#    expected: legacy / "the Base Age scanner is not the active ruleset" / COMPILE_OK

# 3. Is the live book untouched?
ssh arun@94.136.185.54 "cd /home/arun/quantifyd && venv/bin/python3 -c \"
import json;s=json.load(open('backtest_data/oa_real_state.json'))
print(len(s['positions']),'positions, cash',s['cash'],'capital',s['capital'])\""
#    expected: 11 positions, cash 188697.86, capital 617637.68

# 4. Re-run the replication gate (read-only, ~6 min, safe any time, needs ~4 GB RAM)
ssh arun@94.136.185.54 "cd /home/arun/quantifyd && nice -n 10 venv/bin/python3 \
  research/165_oa_baseage_live_conversion/scripts/rot1_gate.py > /tmp/rot1_gate.log 2>&1; \
  tail -50 /tmp/rot1_gate.log"

# 5. Re-run the Monday dry-run (read-only, no Kite, no state write)
ssh arun@94.136.185.54 "cd /home/arun/quantifyd && venv/bin/python3 \
  research/165_oa_baseage_live_conversion/scripts/rot1_dryrun.py > /tmp/rot1_dry.log 2>&1; \
  cat /tmp/rot1_dry.log"
```

**Files that must NOT be touched by this work:** `backtest_data/oa_real_state.json`,
`backtest_data/market_data.db`, `backtest_data/access_token.json`, any other executor,
`services/equity_executor.py`, the crontab.

**Safe to inspect:** everything under `research/165_oa_baseage_live_conversion/` and
`research/170_qs_leeway_and_baseage_best_entrant/`, the `services/oa_*.py` files,
`/tmp/rot1_*.log`.

**To abandon just the swap and keep Base Age:** set `OA_ROT1 = False`.
**To abandon everything:** leave `OA_RULESET = 'legacy'` — none of this code is reached.

---

## 6. Rollback

| Situation | Action |
|---|---|
| Flip has not happened | nothing to do; `OA_RULESET = 'legacy'` |
| Flipped, and the swap is behaving badly, but Base Age is fine | `sed -i "s/^OA_ROT1 = True/OA_ROT1 = False/" services/oa_real.py` — the evening job then runs exits + entries and stops there. No crontab change, no restart, no position touched |
| Flipped, and Base Age itself is to be abandoned | the parent STATUS §6 rollback, unchanged |
| A swap's AMOs are already resting | cancel both by hand in Kite. If only the SELL filled, the book is simply one position lighter with the cash in hand; the next evening's normal entry logic redeploys it. If only the BUY filled, the book is over one slot and the extra name is managed by SuperTrend(14,4) like any other |
| A swap already filled both legs | it is a completed exit and a completed entry. Nothing needs unwinding |

---

## 7. Files

| File | Purpose | Committable |
|---|---|---|
| `services/oa_real.py` | `OA_ROT1`, `ROT1_SELL_TAG` / `ROT1_BUY_TAG`, `place_exit_amo(tag=…)`, reconcile labels a swap-out | yes (modified) |
| `services/oa_baseage_entry.py` | `rs252()`, `rot1_pick()` (pure), `rot1_run()` (arms the two AMOs), `plan()` returns the refused rows | yes (modified) |
| `research/165_.../scripts/patch_probe170.py` | generates the probe engine from research/170's `sim170.py` by exact-string patches | yes |
| `research/165_.../scripts/sim170_probe.py` | generated; research/170's engine + a decision probe | yes (small) |
| `research/165_.../scripts/rot1_gate.py` | gates R1 and R2 | yes |
| `research/165_.../scripts/rot1_dryrun.py` | the Monday dry-run on the live 11 | yes |
| `research/165_.../scripts/rot1_rs_diag.py` | the rs252 calendar diagnostic that found the §8.2 defect | yes |
| `research/165_.../scripts/register_rot1.py` | the marker-guarded register pass (index, ops, labs, TODO) | yes |
| `research/165_.../results/rot1_gate.json` | the gate summary | yes |
| `research/165_.../results/rot1_decisions.csv` | one row per disagreement — **empty, there are none** | gitignored (`*.csv`) |
| `research/165_.../results/rot1_walk.csv` | the 400-session live-code walk, both starting books, 87 KB | gitignored (`*.csv`) |
| `research/165_.../results/rot1_rs_diag.csv` | rs252 per frozen event: study / own-bars / DB-calendar / live, 304 KB | gitignored (`*.csv`) |
| `frontend/src/data/strategies.ts` | the Open Alpha register row | yes (modified) |
| `research/111_.../scripts/ops_center.py` | the 2027-03-13, 2026-09-26 and 2026-09-15 reviews | yes (modified) |
| `docs/LABS_AND_JOBS_REFERENCE.md` · `TODO.md` | the mirrors | yes (modified) |

Commit **`08496a7f`** on the VPS, **not pushed**. The `*.csv` results are excluded by the
repo's existing ignore rule; re-generate them with the two commands in §5.
| `research/165_.../OA_ROT1_SWAP_RULE_DEPLOY_STATUS.md` | this file | yes |

---

## 8. Findings

### 8.0 THE RATE GAP: **it is the period, not the live code.** The study's own engine loses on this window too

**The question.** The pick logic matches research/170's engine 8,488 / 8,488 (§8.1), yet the
live-code walk fired ~12 swaps a year over 2025-01-31 → 2026-09-11 and lost on every arm,
against the study's 4.5/yr and +0.105 paired Calmar on 30/30. Either the window is a bad
stretch for the rule, or the live code differs somewhere in the *"the book cannot take it"*
condition — cash refusals under real integer sizing, the NAV basis, the tie-break.

**The test.** research/170's **own engine, rule untouched**, run over **exactly that window**:
400 sessions, 563 events, once from an empty book at the live book's capital and once
**opening with the live book's eleven positions and its ₹1.88L of cash**. 30 seeds under the
study's random contested-slot draw, plus the live `select='tv'` tie-break.
(`scripts/rot1_window.py`. The engine is `sim170_probe.py`, whose only additions are the
decision probe and two lines that set the opening cash and opening positions from `cfg` — no
rule, filter or ordering touched, and `--verify` re-proves the no-op at 5 patches.)

| research/170's engine, this window | swaps/yr | NAV no-rot | NAV OA-ROT-1 | ΔNAV | **OA-ROT-1 wins** |
|---|---:|---:|---:|---:|---:|
| empty book, random, 30 seeds | **7.56** | ₹10,19,557 | ₹9,56,157 | **−₹75,356** | **6 / 30** |
| empty book, `tv` tie-break | 5.67 | ₹10,72,185 | ₹9,55,515 | −₹1,16,670 | 0 / 1 |
| **live 11 positions + ₹1.88L, random, 30 seeds** | **7.88** | ₹6,42,722 | ₹6,01,502 | **−₹44,996** | **3 / 30** |
| live 11 positions, `tv` tie-break | 7.56 | ₹6,41,010 | ₹6,22,623 | −₹18,387 | 0 / 1 |

**The answer: the gap is the period.** On its own engine, on this window, the rule fires at
**5.7 – 7.9 swaps a year — not 4.5 — and it LOSES**, winning **6 of 30** and **3 of 30** paired
paths. Both the elevated rate and the negative sign reproduce without any live code involved.
A 1.6-year stretch in which a rule that wins 30/30 over 21.7 years loses is exactly what a
real-but-small long-run edge looks like from close up; it is not a defect.

**The residual, stated rather than rounded away.** After the §8.5 sizing fix the live walk
fires **16 swaps (10.1/yr) from the live book and 15 (9.4/yr) from an empty one**, against the
engine's 12 and 9 on the matched `tv` arm — still **1.3 – 1.7× the engine's rate on the same
window**. That residual is the **walk harness**, not the executor, and the cash-refusal counts
say so: the live walk books 360 cash refusals where the engine books ~184. The walk (a) credits
**no idle-cash yield**, (b) pays **no tax**, and (c) sizes off the **trigger close** while the
engine sizes at the **fill open** (deviation R2) — so it runs persistently tighter on cash,
refuses more signals, and therefore offers the rule more chances to fire. None of those three
is in `rot1_pick()`, which is the code that decides a swap and which is exact at 8,488 / 8,488.

**What this means for the flip.** Nothing here blocks it. What it changes is the expectation:
**do not expect ~4 swaps a year in the first months, and do not read an early losing stretch as
the rule being broken.** Both are inside what the study's own engine does on a window like this
one. The 2027-03-13 review's rate criterion is registered against ~4/yr and now has this table
to read it against.

### 8.0b The probe engine is a no-op — proved before it was used

`patch_probe170.py` regenerates research/170's `sim170.py` with **3 exact-string patches**,
each of which must match **exactly once** or the build aborts. With `PROBE = None` the
generated module is run against the original on the same seed and the same inputs:

```
NAV identical            : True
trades identical         : True
book identical           : True
swaps                    : 95          (21.7 years -> 4.4 swaps/yr, research/170 reports 4.5)
VERIFY OK - the probe copy is behaviourally identical with PROBE=None
```

So the thing the live code is being measured against is research/170's engine, not a
paraphrase of it.

### 8.1 Gate R1 — **PASS, 100.00%**

Six full engine paths over 2005-01-03 → 2026-09-11: the five fresh seeds **1001–1005** under
the study's own random contested-slot draw, plus one run under **`select='tv'`**, the
traded-value tie-break the live book actually uses (parent deviation D1). Each engine run
reproduces the published shape — CAGR 21.8–23.5%, **MaxDD −31.78% on every path**, Calmar
0.687–0.739 — against research/170's 22.58 / −31.78 / 0.710.

At every rotation decision point the engine recorded what it saw and what it did; the live
`services.oa_baseage_entry.rot1_pick()` was handed the same inputs and asked again.

| | |
|---|---:|
| Decision points | **8,488** (2005-07-11 → 2026-09-11) |
| Swaps the engine fired | **575** |
| Fire / no-fire agrees | **8,488 / 8,488 = 100.00%** |
| … and the name **SOLD** agrees | **8,488 / 8,488 = 100.00%** |
| … and the name **BOUGHT** agrees — **THE GATE** | **8,488 / 8,488 = 100.00%** |
| Restricted to the last 400 sessions (from 2023-12-14) | **2,299 / 2,299 = 100.00%** |

**No mismatches of any kind.** `results/rot1_decisions.csv` is empty by construction — it
holds one row per disagreement and there are none.

### 8.2 R3 — the one deviation that was real, found and closed

The gate's first run disagreed on **8.2% of contested entrant rankings**, and the cause was
not the rule. It was `rs252`.

research/161 built its price panel on **one master trading calendar — the distinct daily bars
of `NIFTYBEES` from 2005-01-03** (`bt161_sweep.py`), research/164 inherited it and
research/170 computes `rs252` on it. So "252 bars ago" in the published result means **252
sessions of that calendar**, with each symbol's close forward-filled onto it. The first live
implementation counted **252 of the symbol's own bars**, which is the natural reading of the
rule as written and is a different number for any name that missed a session in a year.

Measured against the **3,366 events in research/164's frozen list that carry an `rs252`**
(`scripts/rot1_rs_diag.py`):

| live reading | reproduces the study's rs252 exactly | picks the same top-ranked name on contested days |
|---|---:|---:|
| 252 of the symbol's **own bars** | 2,481 / 3,366 = **73.7%** | 794 / 810 = 98.02% |
| forward-filled on the **all-symbol DB calendar** | 2,788 / 3,366 = 82.8% | 806 / 810 = 99.51% |
| **the master `NIFTYBEES` calendar — DEPLOYED** | **3,357 / 3,366 = 99.73%** | **810 / 810 = 100.00%** |

`services/oa_baseage_entry.py` now carries `master_calendar()` and computes `rs252` on it.
After the change, the gate's own ranking check went to **4,472 / 4,472 = 100.00%** of
contested decision points, which is why R1 reads 100% and not 92.8%.

Two details behind the remaining 0.27%: the DB's all-symbol day calendar carries **6 sessions
the study's calendar does not** (2014-04-24, 2014-10-15, 2015-02-28, 2016-10-30, 2017-10-19
and one more — the phantom-holiday rows this repo has met before), and nine events differ in
the fourth decimal. Neither changes a ranking, which is the only thing the rule reads.

**This is the whole value of running the gate.** The rule was right the first time; the input
it reads was not, and nothing in the rule's own logic would ever have surfaced it.

### 8.3 Gate R2 — the live code, walked, and it does NOT flatter the rule

The staged evening job replayed day by day over the **last 400 sessions, 2025-01-31 →
2026-09-11**, on research/164's frozen event list (562 events, 471 symbols), calling the live
`plan()` and the live `rot1_pick()` for every decision. Two starting books, each run with and
without the swap.

**Re-run 13-Sep after the §8.5 marked-NAV fix; these are the current numbers.**

| starting book | swaps | buys | exits | refused cash | refused slot | end | **NAV** |
|---|---:|---:|---:|---:|---:|---:|---:|
| the live 11 positions + ₹1.88L | **16** | 95 | 92 | 360 | 116 | 14 pos | **₹4,83,793** |
| the same, OA-ROT-1 off | 0 | 84 | 79 | 115 | 357 | 16 pos | **₹5,35,718** |
| empty book, ₹6,17,638 | **15** | 97 | 83 | 407 | 68 | 14 pos | **₹7,25,291** |
| the same, OA-ROT-1 off | 0 | 82 | 67 | 409 | 66 | 15 pos | **₹8,62,125** |

60 sessions at the live book's capital, the comparison the parent's §8.2 made for the
un-rotated spec:

| | swaps | buys | exits | refused cash | refused slot | end | NAV |
|---|---:|---:|---:|---:|---:|---:|---:|
| with OA-ROT-1 | **11** | 30 | 14 | 7 | 100 | 16 pos, cash ₹5,342 | ₹6,62,083 |
| un-rotated | 0 | 21 | 6 | 37 | 68 | 15 pos, cash ₹18,837 | ₹6,66,387 |

**Three things have to be said plainly.**

**1. The machinery works and converts refusals into entries.** Every arm with the swap on takes
MORE entries than the same arm without it (84 → 95, 82 → 97, 21 → 30) and fills the book
earlier, which is the mechanism the rule exists for: a refused signal gets funded by selling
the book's worst holding. (Refusal counts move the other way here because a book that keeps
buying stays closer to empty on cash and so refuses more of what comes after — which is also
most of the residual rate gap in §8.0.) The swap log
reads exactly as the rule specifies — *"swap: sold HINDCOPPER (−13.3%) for RACLGEAR (rs252
rank 1 of 2)"* — and the live-book arm's first swap, *"sold SETL (−63.3%)"*, is the
anachronism working as intended: the live book's Sep-2026 buy prices against Apr-2025 closes
make every position look ruined, the rule fires, and the book normalises. That is the edge
case exercised, not an economic claim.

**2. On this window the rule LOST, on all three arms** — −₹51,925 over 400 sessions from the
live book, −₹1,36,834 from an empty one, −₹4,304 over 60 days. **And so does research/170's
own engine on the same window, winning only 6 of 30 and 3 of 30 paired paths (§8.0).** That is
the pairing that matters: the loss is a property of 2025-26, not of this implementation. Arun
should go into the flip expecting the first months to be able to look like this.

**3. The swap RATE is roughly double the study's long-run 4.5/yr — and so is the engine's, on
this window.** The live walk now produces **16 in 1.6 years (≈ 10.1/yr)** and 15 from an empty
book (≈ 9.4/yr); research/170's engine on the same window produces **5.7 – 7.9/yr** (§8.0). The
window carries most of it and the walk harness's cash simplifications carry the rest. The log
shows genuine churn chains (NGLFINE bought 22-May, swapped out 04-Jun; V2RETAIL in 04-Jun, out
12-Jun) — which the engine permits identically. **The 2027-03-13 review's pass criterion is
written against ~4 occurrences a year; §8.0's table is what it should be read against**, and
the day-one and 26-Sep checks below count swaps for exactly that reason.

### 8.4 Dry run on the REAL book, 11-Sep-2026 close — **no swap is possible on Monday**

`scripts/rot1_dryrun.py`, read-only: no Kite, no lock, no state write, run with
`OA_RULESET='legacy'`.

**11 of 16 slots used, 5 free. Positions ₹4,33,173 + cash ₹1,88,698 = NAV ₹6,21,871.**

| symbol | qty | buy | 11-Sep close | P&L % | swap-out rank |
|---|---:|---:|---:|---:|---:|
| **SBCL** | 34 | 1,115.48 | 1,088.00 | **−2.46%** | **1 — deepest loss** |
| ENTERO | 21 | 1,843.72 | 1,806.90 | −2.00% | 2 |
| IRISDOREME | 634 | 62.39 | 61.97 | −0.67% | 3 |
| SHILPAMED | 40 | 962.36 | 959.05 | −0.34% | 4 |
| INOXINDIA | 17 | 2,236.50 | 2,242.50 | +0.27% | 5 |
| NITINSPIN | 61 | 636.65 | 652.00 | +2.41% | 6 |
| WELCORP | 14 | 2,596.36 | 2,686.90 | +3.49% | 7 |
| SSWL | 106 | 358.30 | 371.00 | +3.54% | 8 |
| INDSWFTLAB | 99 | 362.33 | 389.00 | +7.36% | 9 |
| MANINDS | 47 | 800.19 | 870.00 | +8.72% | 10 |
| SETL | 99 | 402.69 | 468.15 | +16.26% | 11 |

**(a) The book as it is.** One qualifying signal on the 11-Sep close — **PAYTM**, base 1,191
bars, depth 82.4%, TV ₹479.61 cr, **rs252 +44.8%** — and the book **takes** it at **21 shares,
₹38,157**, leaving ₹1,50,541: 5 free slots and ₹1,88,698 of cash fund it outright. **OA-ROT-1 cannot fire at Monday's open.** It only
ever looks at a signal the book had to turn away, and nothing was turned away.

**(b) The book if it were full — the first hypothetical.** Suppose all 16 slots were taken
and the cash gone, so Monday's signal were refused. The holding with the largest loss is
**SBCL at −2.46%**. That is **not** worse than −10%, so **no swap would fire even then**. SBCL
would have to fall a further **7.7%, to ₹1,003.93**, before it qualified. Nobody is being
asked to approve a sale.

### 8.5 An inconsistency in the PARENT's entry sizing — found by this work and **now FIXED**

`services/oa_baseage_entry.plan()` is called by `run()` with `nav=None`, and its fallback was
the book's **cost-plus-cash NAV** — ₹4,19,225 of cost + ₹1,88,698 = **₹6,07,923**, slot
₹37,995 — while the study, research/170's engine and `rot1_pick()` all size off the **marked**
NAV, ₹6,21,871, slot ₹38,867. On Monday's book that was **PAYTM ×20 from the code against
the ×21 the parent STATUS §8.4 prints** — and worse, it meant the two legs of a single
evening were sized on two different bases: an entry at cost NAV and a swap at marked NAV.

**Fixed 13-Sep-2026 at Arun's instruction.** `plan()` now marks the book to the **last official
close** by default (`marks` / `asof` let a caller pass closes it has already read; an explicit
`nav` still overrides). A position whose latest bar is stale falls back to its buy price and is
named in `ctx['unmarked']` and printed by `run()`, so a silent mis-mark is not possible.
Re-verified:

- **The dry run now arms PAYTM ×21 at ₹38,157**, cash left ₹1,50,541 — the code and the
  parent STATUS §8.4 now agree.
- **Gate R1 re-run: still 8,488 / 8,488 = 100.00%.** R1 exercises `rot1_pick()` directly and
  never touches `plan()`, so this is a confirmation rather than a re-measurement.
- **Gate R2 re-walked**; §8.3 carries the post-fix numbers. Swaps fell from 20 to 16 on the
  live book and 18 to 15 on an empty one — the fix moved the live rate about 20% closer to
  the engine's on the same window.

Both legs of an OA-ROT-1 evening are now sized on one basis, and it is the study's.

### 8.6 The no-op proof — a `git pull` still changes nothing that runs

```
$ venv/bin/python3 services/oa_real.py ruleset
legacy
$ venv/bin/python3 -c "... print(OA_RULESET, OA_ROT1, _book_tags())"
ruleset legacy rot1 True
book tags ('OA-TOPUP', 'OA-ENTRY', 'OA-EXIT')      <-- the swap tags are NOT added under legacy
$ venv/bin/python3 services/oa_baseage_entry.py
OA_RULESET is 'legacy' - the Base Age scanner is not the active ruleset; nothing scanned, nothing placed.
$ venv/bin/python3 -m py_compile services/oa_real.py services/oa_entry.py \
      services/oa_baseage.py services/oa_baseage_entry.py && echo COMPILE_OK
COMPILE_OK
```

**The legacy branch is untouched by design, not by luck.** Every edit to `oa_real.py` is
either a new constant, a new function, or guarded:

| Edit | Why legacy is unaffected |
|---|---|
| `OA_ROT1`, `ROT1_MARGIN_PCT`, `ROT1_MAX_PER_DAY`, `ROT1_SELL_TAG`, `ROT1_BUY_TAG` | new constants, read only from the Base Age evening job |
| `_book_tags()` | returns `BOOK_TAGS` **unchanged** unless `OA_RULESET == 'baseage'`, so `reconcile` reads exactly the orders it read on 11-Sep |
| `reconcile` tag tracking | the swap tags cannot appear on any order under legacy (nothing places them), so `reason` stays `rule_exit` and `src` stays `executor` |
| `place_exit_amo(..., tag=None)` | defaults to `EXIT_TAG`; every existing caller passes nothing |
| `place_buy(..., tag=ENTRY_TAG)` | same |
| `plan()` returning `turned_away` / `cash_after_entries` / `armed_cost` | additive keys on `ctx`; the orders and refusals it returns are byte-for-byte what they were |
| `run()` no longer returning early when there is nothing to arm | Base Age module only — unreachable under legacy |

`backtest_data/oa_real_state.json` untouched: still **11 positions, cash ₹1,88,697.86,
capital ₹6,17,637.68**. No restart is needed for the same reason the parent gave in its §8.5 —
nothing that reads either switch runs inside gunicorn.

---

## 9. Runbook delta — what changes in the parent's §9

**Nothing new to flip.** `OA_RULESET = 'baseage'` enables OA-ROT-1 along with Base Age; there
is no second switch to throw, no extra cron line, no new job. The 18:50 entry cron
(crontab line **112**) already runs exits, then entries, then — from this commit — the swap.

Fold these into `OA_BASE_AGE_LIVE_CONVERSION_DEPLOY_STATUS.md` §9:

**STEP 2b — SETTLED 13-Sep-2026, no action required.** The live entry scanner now sizes off
the **marked** NAV, like the study and like the swap leg (§8.5). Monday's PAYTM is ×21 at
₹38,157 in both the code and the parent's dry-run. The only remaining pre-flip blocker is
the 09:20 top-up (parent §8.0).

**STEP 3 — the switch, with the swap's own rollback beside it.**

```bash
ssh arun@94.136.185.54 "grep -n \"^OA_RULESET\|^OA_ROT1\" /home/arun/quantifyd/services/oa_real.py"
# expected before the flip:  OA_RULESET = 'legacy'   /   OA_ROT1 = True
```

**STEP 6 — first-night watch, two extra lines to look for** in `/tmp/oa_entry.log`, AFTER the
entry block:

```
OA-ROT-1: no qualifying signal was refused tonight; nothing to swap.        <- the normal night
OA-ROT-1: N qualifying signal(s) refused; ranking them by 12-month relative strength
   refused <SYM>  close ...  rs252 +NN.N%  TV ... cr                        <- one per refusal
OA-ROT-1 swap: sold <X> (-12.3%) for <Y> (rs252 rank 1 of 3)
   SELL placed <id> as MARKET|LIMIT
   BUY  placed <id> as MARKET|LIMIT
```

and the two refusals that are **not** failures:

```
OA-ROT-1: deepest loss is SBCL -2.46%, not worse than -10.0% - no swap
OA-ROT-1: both legs cancelled: <X> is -12.4% but Rs 38,867 buys no whole share of any of the 2 refused signal(s)
```

**STEP 7 — next morning**, `reconcile` now labels the two legs. A swap-out appears in the
trade table with `reason: 'rot1_swap_out'`, not `'rule_exit'`, and the entrant position
carries `src: 'rot1'`. If a swap fired and both show up as ordinary exits and entries, the
tags did not reach the broker — check `kite.orders()` for the `OA-ROT1-*` tags before
trusting any later swap-rate count.

### Rollback — the swap alone

```bash
ssh arun@94.136.185.54 "cd /home/arun/quantifyd && \
  sed -i \"s/^OA_ROT1 = True/OA_ROT1 = False/\" services/oa_real.py && \
  venv/bin/python3 -c \"import sys;sys.path.insert(0,'.');from services.oa_real import OA_ROT1;print(OA_ROT1)\""
# must print: False
```

The evening job then runs exits and entries and stops. **No crontab change, no restart, no
position touched**, and Base Age keeps running. Use this — not the ruleset switch — if the
swap is what is misbehaving. If a swap's AMOs are already resting, cancel both by hand in
Kite; the partial-fill cases are in §6.

### Added to the day-one checklist (2026-09-15)

- [ ] Did the 18:50 log show an **OA-ROT-1 block at all** — even the "nothing to swap" line?
      If the block is missing, the swap code did not run and the flip is only half applied.
- [ ] If a swap fired: are **both** legs in `kite.orders()`, tagged `OA-ROT1-SELL` and
      `OA-ROT1-BUY`, and did the **SELL go in first**?
- [ ] Did `reconcile` label the swap-out `rot1_swap_out` rather than `rule_exit`?
- [ ] Was the name sold the **deepest loss** in the book, and was it worse than **−10%**?
      Check against `mark`'s own P&L column; if not, stop and set `OA_ROT1 = False`.
- [ ] Was the name bought the **highest rs252** among the refused signals, and is that name
      **not** one the book already held?

### Added to the 26-Sep-2026 review

- [ ] **How many swaps in two weeks?** research/170 expects ~4 a year; this session's
      400-session walk of the live code produced ~12 a year and the 60-day walk ~46 (§8.3).
      Record the actual count — it is the input to the 2027-03-13 pass criterion.
- [ ] **How many refusals were converted?** Count `refused for cash` / `no free slot` lines
      that were followed by a swap, against those that were not.
- [ ] **Did any swapped-in name get swapped out again within ten sessions?** The walk showed
      genuine churn chains. Two or three of those in a fortnight is the rule working; a
      weekly rotation of the same slot is not.
- [ ] **P&L attribution of the swaps**, both legs: what the sold name did after it was sold,
      and what the entrant did. Over the only window that could be walked, the rule **lost**
      (§8.3). Two weeks will not settle that, but the ledger has to start.
- [ ] **Read the swap rate against §8.0's table, not against the bare 4.5/yr.** research/170's
      own engine fires 5.7-7.9 swaps a year on a 2025-26-shaped window and loses there too, so
      an early losing stretch at ~10 swaps a year is inside what the rule does — not evidence
      that it is broken. The entry-sizing NAV question is closed (§8.5).

**Flip record:** 2026-09-13 13:33 IST — **FLIPPED by Arun's go.** OA_RULESET=baseage (reads back `baseage`), crontab installed from /tmp/ct.new (backup /tmp/ct.bak.20260913-133230, 135 lines before and after, diff = exactly lines 107 and 112): 18:50 entry job live, 09:20 equity_executor restricted to `--book ipo-base`, 09:25 re-arm still commented. Broker order book empty at flip time. No restart. OA_ROT1=True (swap live under baseage). First evening: Mon 14-Sep 18:50.
