# V2 Iron Fly — the live book trades an expiry that was never tested

STATUS: **FINDING — no deploy. A live-vs-backtest construction mismatch, documented in our own files.**
research/141 · 2026-09-01

---

## 1. The question

Arun, weighing arming the V2 book: *"is this next week options or current week?"* and then
*"If live engine does the 2nd nearest - then thr must hv been a study done by us/claude project."*

## 2. The answer: there was no such study

`research/60`'s STATUS doc lists expiry as **lever #5** of its sweep grid —
*"Expiry | nearest weekly / 2nd-weekly / monthly | baseline: 2nd-weekly"* — with test order
0 -> 1 -> 2 -> 4 -> **5** -> 3 -> 6 -> 7.

**Lever 5 was never run.** AlgoTest caps entry at 4 trading days before expiry, which forces the
FRONT weekly. The doc records the drift in its own words:

> "Config actually tested **(drifted from spec, both noted)** ... **front weekly**, entry **4 TD
> before expiry** (**AlgoTest max; not the 2nd-weekly 12-DTE carry**)"

and the Final-structure line hedges: *"positional (**front/2nd weekly**, 4TD entry / 1TD roll)"*.

So the 2nd-nearest weekly is a **design intention inherited from the original V2 spec, never
validated** — and `services/v2_ironfly_api.py` implements it via `_second_weekly()`.

## 3. What the choice is worth — same engine, only the expiry varied

VIX>=13, 10 lots, 2019-2026, our bhavcopy engine, net of Rs20/leg + 0.25% slippage.

| construction | arm C (no stop) | Calmar | arm D (+2% stop) | Calmar |
|---|---:|---:|---:|---:|
| **FRONT weekly, 4 TD** — what AlgoTest tested | **+Rs15,16,346** | **0.59** | +Rs9,52,919 | 0.32 |
| **2nd weekly, any DTE>=4** — **what runs LIVE** | **+Rs35,315** | **0.01** | +Rs15,69,127 | 0.36 |
| 2nd weekly, 8 TD — the written plan | -Rs1,50,957 | -0.04 | -Rs7,84,248 | -0.08 |

**On the tested arm the expiry is worth Rs14.8 lakh.** And the plan's own specification (2nd
weekly at 8 TD) is **negative on all four arms** — it is fortunate AlgoTest could not express it.

## 4. The two expiries want OPPOSITE stop settings

| | no stop -> with stop |
|---|---|
| FRONT weekly | +Rs15.16L -> +Rs9.53L — the stop **HURTS** (3-day hold; it whipsaws) |
| 2nd weekly | +Rs35k -> +Rs15.69L — the stop **HELPS** decisively (longer hold, more gamma days) |

So the live book's deployed 2% stop is **right for what it actually trades** — but for reasons no
study established. The study locked arm C (stopless-equivalent, front weekly); the live book's
best arm is D (stopped, 2nd weekly). Different construction, different optimum, same label.

## 5. Three live-vs-tested divergences, for the record

| | AlgoTest tested | Live engine |
|---|---|---|
| **expiry** | **front weekly** | **2nd-nearest** — MISMATCH |
| **inside-week skip** | **not tested** | **also skips** — MISMATCH |
| wings | +/-250, later swept to 2.0% of ATM | 2.0% of ATM — matches |
| stop / target / VIX / CPR | 2% move / 40% / >=13 / <0.10% | identical — matches |

## 6. Limits of this evidence

Our engine prices daily closes, so arms B and D approximate the intraday stop by exiting at the
close of the day the 2% move happened - biased against the stop by an unknown margin. Arms A and
C are stopless CONTROLS, not proposals. Absolute levels will not match AlgoTest. What makes the
comparison credible is that **only the expiry changed between rows**; everything else is held
fixed, and the gap is far too large to be an artefact of the stop approximation.

## 7. What would settle it

An AlgoTest re-run exporting the **front-weekly, 4 TD, VIX>=13** trade CSV. That is the one
configuration where AlgoTest and our engine can be placed side by side, and it would restore the
per-trade data (streaks, per-year) that `research/60` did not retain — its results are documented
but the export itself is not in the repo.

**Nothing deployed. This is a finding about what the live book is, not a proposal to change it.**
