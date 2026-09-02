# research/135 — Turtle (Dennis/Covel) Optimization on Indian Equities: channel × stop × pyramiding × sizing

STATUS: DONE — CONCLUDED, NO DEPLOY

## 1. The Ask

**What you asked:** "Turtle trading system — pls test the attached system rules
on our stocks data in the database, and give a simple report; also see if this
can be optimized for better numbers / CAGR / Calmar."

**What we are actually testing.** The *faithful test* of the attached rules is
already done — `research/83_turtle_equities` (CONCLUDED 2026-07-17) ran
S1(20/10) + S2(55/20), both directions, 2N stops, IS 2005-2017 + book bakeoff
2005-2023. That answers "does it work". This study answers the **second** half
of the ask, which r/83 explicitly left unconsumed:

> "pyramiding untested (phase 2 not earned); IS+Val only (OOS 2024+ unconsumed)"

So r/135 = **optimization only**, on the long side (shorts are permanently
closed by r/81 + r/82 + r/83 — not re-tested, not re-openable):

1. **Rule 1 (entry) / Rule 5 (exit) parameter surface** — is 20/10 + 55/20 a
   peak or a plateau? Sweep the channel grid and look for a *region*, not a
   maximum.
2. **Rule 3 (stop) calibration** — is 2N right? Sweep {no stop, 1.5N, 2N, 3N}.
3. **Rule 4 (PYRAMIDING) — the genuine gap.** The attached rules add up to
   4 units every ½N in favour, moving all stops to 2N from the last unit.
   r/83 never tested this. This is the one mechanic of the attached system
   that has never been measured here.
4. **Rule 2 (N-sizing) re-examination** — r/83 found N-sizing LOSES to
   equal-notional at 0.75% risk. The attached spec says **1%**. Test whether
   the sizing verdict is a risk-level artefact or a real inversion.

**Success criterion (pre-declared):** Calmar over the book NAV, tie-break
Sharpe, subject to MaxDD <= 1.2x the r/83 incumbent (C_turtleEQ, −31.9%).
An "optimization" only counts if the winner sits on a **plateau** (its
parameter neighbours are also top-quartile) AND survives the OOS shot.

## 2. The Base — what is being tested

- **Signal (entry):** daily close > highest high of prior `n_in` sessions ->
  enter next open. Long only.
- **N:** ATR20 (simple mean of true range) at the signal bar; fixed for the
  life of the position (faithful Turtle).
- **Stop:** entry_price − `stop_mult` × N. Evaluated intrabar; gap-through
  fills at the open. On each pyramid add, ALL stops move to
  last_unit_entry − stop_mult × N.
- **Pyramid (Rule 4):** add a unit each time price advances `add_step` × N
  above the last unit's entry, up to `max_units`. Intrabar fill at the level
  (or the open if it gaps past).
- **Exit (Rule 5):** daily close < lowest low of prior `n_out` sessions ->
  exit ALL units next open. Trailing channel IS the exit; no time stop, no
  profit target.
- **Universe:** F&O names (`FNO_LOT_SIZES`, ~78 after CA-guard <=5 gap flags
  and >=300 bars). Survivorship-biased — stated, not fixed.
- **Costs:** FUTURES_PROXY, 3bp/side slippage; every headline reported gross
  AND net; slippage 2x sensitivity on the finalist.
- **Gate:** entries only when NIFTYBEES close > its 200DMA (both shifted 1
  day). Positions ride their own exits. Same gate for every arm.
- **Book:** equal-notional per UNIT, position cap, 1.5x book-notional cap,
  25% max single-position notional. Daily MTM NAV.

### Periods (ledger discipline)

| Split | Range | Use |
|---|---|---|
| IS | 2005-01-01 .. 2017-12-31 | selection — all sweeping happens here |
| VAL | 2018-01-01 .. 2023-12-31 | robustness; r/83 warns this era is ~flat |
| OOS | 2024-01-01 .. 2026-08-29 | **held out; consumed ONCE on the finalist** |

## 3. Plan — staged grid

Stage gates: do not spend the next stage's compute until the current passes.

| Stage | Axis | Cells | Held fixed |
|---|---|---|---|
| A | `n_in` {20,40,55,80} × `n_out` {10,20,40} (n_out<=n_in) × `stop_mult` {none,1.5,2.0,3.0} | 11 × 4 = **44** | no pyramid, EQ sizing, cap 12, gate ON |
| B | pyramiding: `max_units` {1,2,3,4} × `add_step` {0.5N,1.0N} | **7** (max_units=1 shared) | Stage-A winner region |
| C | sizing EQ vs N-risk {0.5%,1%,2%}; cap {8,12,20}; gate {on,off} | **~10** | Stage-B winner |
| D | OOS single shot: finalist + r/83 incumbent + NIFTYBEES | **3** | — |

**Effective trials ~61.** Multiple-testing control: plateau requirement (above)
+ a single OOS consumption + per-year stability reporting. A winner that is a
lone spike in the parameter map is reported as OVERFIT, not as an upgrade.

**Falsification (declared up front):** if no cell's Calmar beats the r/83
incumbent (0.45) by more than the spread of its own parameter neighbourhood,
the verdict is **NO OPTIMIZATION FOUND** and the attached rules stand as-is.

## 4. Status

| Date/time | Event | Notes |
|---|---|---|
| 2026-08-30 19:50 IST | Study pre-registered | Grid locked before any run |
| 2026-08-30 19:58 IST | Simulator sanity check PASSED | synthetic trend: 4 units at 0.5N, conservative gap-fills |
| 2026-08-30 20:02 IST | Stage A done (44 cells) | no-stop beats every stop level, monotone; 20/10 best channel |
| 2026-08-30 20:12 IST | Stage B done (40 cells) | pyramid helps ONLY 20/10, degrades 5 other channels -> flagged OVERFIT |
| 2026-08-30 20:14 IST | Stage C done (9 cells) | equal-notional beats N-sizing at every risk level (4th time) |
| 2026-08-30 20:23 IST | Stage E BUG FOUND | put overlay double-counted premium (-99.9% absurd); hand-check caught it |
| 2026-08-30 20:31 IST | Stage E re-run after fix | sleeve -6.9%/yr at 2x ATM, matches hand-check; OTM beats ATM everywhere |
| 2026-08-30 20:34 IST | Stage F done — OOS CONSUMED ONCE | every Turtle arm NEGATIVE OOS; benchmark +5.25% |
| 2026-08-30 20:40 IST | Full-period run + RESULTS.md written | VERDICT: CONCLUDED, no deploy |
| 2026-08-30 21:05 IST | **Arun challenged the momentum arm** | chart showed momentum at ~benchmark vs r/75's published 31.9% |
| 2026-08-30 21:20 IST | Stage G: momentum arm was WRONG 3 ways | wrong universe (F&O-78 not top-250), wrong rules (live-book Donchian stop bolted on), idle-cash re-entry bug |
| 2026-08-30 21:28 IST | Stage G reproduces r/75 to within 0.1pt | 31.78% vs published 31.9% — engine driven correctly |
| 2026-08-30 21:32 IST | Report + RESULTS.md corrected | conclusion STRENGTHENED: momentum wins every era |

## 5. Crash Recovery

Everything runs on the VPS (`94.136.185.54`, `/home/arun/quantifyd`), python
`/home/arun/quantifyd/venv/bin/python`. The laptop copy is NOT a git checkout;
scripts are synced by SFTP.

```
# what finished?
ssh arun@94.136.185.54 'tail -40 /tmp/turtle135.log; wc -l /home/arun/quantifyd/research/135_turtle_optimization/results/*.csv'
# still alive?
ssh arun@94.136.185.54 'pgrep -af run_turtle_opt'
# resume (skips cells already in the CSV)
ssh arun@94.136.185.54 'cd /home/arun/quantifyd && nohup venv/bin/python research/135_turtle_optimization/scripts/run_turtle_opt.py > /tmp/turtle135.log 2>&1 &'
```

Results CSVs are append-only and the runner skips completed labels, so a
re-launch is always safe. Do not delete `results/*.csv` mid-run.

## 6. Files

| File | Purpose | Committable |
|---|---|---|
| `scripts/turtle_core.py` | unit-level Turtle simulator + book NAV | yes |
| `scripts/run_turtle_opt.py` | staged sweep runner | yes |
| `results/stage_*.csv` | per-cell book metrics | yes (small) |
| `results/RESULTS.md` | final verdict | yes |

## 7. Findings

See `results/RESULTS.md` for the full write-up. Headline:

1. **The attached spec taken literally is the worst book tested** — 1.7% CAGR,
   -67.9% MaxDD, Calmar 0.02 (2005-2026 net), losing to buy-and-hold by ~11
   points of CAGR at more drawdown.
2. **The optimization comes from DELETING rules.** Drop Rule 3 (2N stop),
   Rule 4 (pyramiding) and Rule 2 (N-sizing); keep Rule 1/5 at the original
   20/10. Calmar 0.02 -> 0.53, CAGR 1.7% -> 15.9%.
3. **Rule 3 is the single most damaging rule.** Mean Calmar by stop across 11
   channels: none 0.65 > 3N 0.51 > 2N 0.46 > 1.5N 0.45 — monotone, and removing
   the stop improves drawdown too (-32.1% vs -34.7%).
4. **The plateau test earned its keep.** Pyramiding scored the study's best IS
   Calmar (1.06) but improved only 1 of 6 channels -> flagged OVERFIT before the
   holdout; VAL then confirmed the collapse (1.06 -> 0.08).
5. **OOS 2024-2026 is negative for every Turtle arm** (TT_OPT -8.3% vs
   benchmark +5.3%); era means decay monotonically +27.4% -> +14.8% -> -5.4%.
6. **CORRECTED (Stage G) — the momentum book beats every Turtle variant in
   EVERY era, by far more than first reported**: 31.78% CAGR vs 15.97% at the
   same drawdown (−31.7% vs −31.8%), 299× vs 21×, and +21.0%/yr in the held-out
   window where every Turtle arm loses. The original 12.58% momentum figure was
   a broken reconstruction (wrong universe + live-book stop bolted on + idle-cash
   re-entry bug); Stage G drives r/75's own runner and reproduces its published
   number to 0.1pt. The gate choice is era-unstable; the put overlay does not
   rescue the book (5%-OTM does consistently beat ATM).
7. **The two systems want OPPOSITE universes.** Momentum gains +11.6 pts of CAGR
   moving from the 78 F&O large caps to the top-250; the Turtle gets *worse*
   there (Calmar 0.50 → 0.24) — breakout-and-trail on thinner mid-caps buys
   whipsaws, not trends.

**VERDICT: CONCLUDED — NO DEPLOY.**
