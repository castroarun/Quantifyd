# Open Alpha — does a freed slot have to wait a day? STATUS: DONE - NO EDGE

**research/157_same_day_slot_swap** · opened 08-Sep-2026 21:45 IST

## 1. The ask

**What Arun asked:** *"so a slot freed by today's exit is available to the next day's scan
— why the 1 day delay? was it intentional? canu test the same day swap?"*

**What we are testing:** In `bluesky_replay.simulate`, the per-bar loop runs **entries
first, then exits at the close**. A position that exits on bar *i* therefore frees its slot
only for bar *i+1*. Nothing in the study documents this as a decision — the code is
literally `# entries` … `# exits at close`, with no comment — so the one-day delay looks
like an artefact of block order rather than a modelled constraint.

Does running **exits before entries on the same bar** — so a name that stops out at today's
close is replaced by today's best qualifying breakout — change the outcome beyond noise?

**Why it matters now:** Open Alpha has no live entry scanner at all, so I am about to build
one. The ordering decides what that scanner does: a same-day swap means the 15:18 exit
check must also place the replacement buy in the same pass; a next-day refill means the
scan is a separate overnight job. Building first and testing after would be backwards.

## 2. The base — what is being tested

Unchanged from r/142's adopted 16-slot spec, which is what the live book runs:

| Rule | Value |
|---|---|
| Universe | daily bars, ≥260 rows; ETFs excluded (`BEES|ETF|LIQUID|GILT|SENSEX|NIF*50`) |
| Liquidity | 20-day median traded value ≥ ₹5 cr, on the **previous** bar |
| Setup | `prev_close < ATH-close` **and** `prev_close ≥ 0.8 × ATH-close` |
| RS | IBD-style percentile of `2·r63 + r126 + r189 + r252`, shifted 1, **≥ 70** |
| Trigger | `close > ATH-close` (ATH-close = expanding max of close, shifted 1) |
| Book | 16 slots, 6.25% of equity each, cash-constrained |
| Exits | −8% stop from fill; close below the trailing SMA (not on the entry bar) |
| Selection | **highest RS first** — deterministic, and how the live book was seeded |
| Costs | 25 bps per side |

**The single variable:** the order of the entry and exit blocks within one bar.
Nothing else differs between the two arms.

## 3. Plan

Two arms, one shared set of frames and signals so the inputs are provably identical:

| Arm | Order within a bar | Meaning |
|---|---|---|
| **A — next day** (control) | entries → exits | today's exit frees a slot for tomorrow |
| **B — same day** | exits → entries | today's exit frees a slot for today |

**Null control first.** Arm A is run through a *forked* copy of `simulate`, and its equity
curve must match the untouched `bluesky_replay.simulate` **exactly**. If the fork does not
reproduce the original, nothing else in this study means anything and it stops there.

Selection is `rs`, which is deterministic — there is no seed ensemble to run and no
selection luck to average away. The comparison is a single clean paired path.

**Reported:** CAGR, max drawdown, Calmar, trade count, win rate, per-year returns for both
arms, and the difference. Net of 25 bps/side.

**What would make this ADOPT:** arm B ahead on return *and* not worse on drawdown, by a
margin that is not a rounding artefact, with the per-year table showing it is not one lucky
year. Anything less and the delay stays — an unvalidated live change to a real-money book
needs a reason, and "it seemed tidier" is not one.

## 4. Status

| Date/time | Event | Notes |
|---|---|---|
| 08-Sep 21:45 IST | Folder + this doc written | before any run, per the binding rule |
| 08-Sep 21:52 IST | First run: B beat A by +12.57pp | too good - held back for checking |
| 08-Sep 21:58 IST | LOOK-AHEAD found in arm B | entries fill at the OPEN, exits at the CLOSE: it sold at the close and spent the proceeds that morning |
| 08-Sep 22:05 IST | Honest re-run, both arms filling at the close | +0.53pp, 4 of 7 years - noise |
| 08-Sep 22:06 IST | RESULTS.md written, verdict NO EDGE | the delay stays; entries remain an after-close job |

## 5. Crash recovery

```bash
cd /home/arun/quantifyd
tail -40 research/157_same_day_slot_swap/results/run.log     # progress
pgrep -af swap_test.py                                       # still alive?
# resume (idempotent, ~5 min: it rebuilds frames then runs both arms)
nohup venv/bin/python3 -u research/157_same_day_slot_swap/scripts/swap_test.py \
  > research/157_same_day_slot_swap/results/run.log 2>&1 &
```

Do not edit `research/142_bananapatterns_replication/scripts/bluesky_replay.py` — this
study imports it read-only and forks `simulate` into its own file.

## 6. Files

| File | Purpose | Committable |
|---|---|---|
| `OA_SAME_DAY_SLOT_SWAP_DAILY_SWEEP_STATUS.md` | this doc | yes |
| `scripts/swap_test.py` | both arms + the fidelity control | yes |
| `results/run.log` | progress | yes (small) |
| `results/RESULTS.md` | verdict | yes |

## 7. Findings

**NO EDGE - see `results/RESULTS.md`.** Honest same-day swap is worth +0.53pp of CAGR on a
4-of-7-year split, with a slightly worse per-trade average. The one-day delay stays.

The first run said +12.57pp and was **look-ahead**: entries fill at the OPEN while exits
happen at the CLOSE, so running exits first sold at the close and spent the proceeds that
morning. 96% of the apparent edge was the bug.

Consequence for the build: the entry scanner runs after the close and arms a buy-stop for
the next session. It is NOT part of the 15:18 exit pass.
