# V2 Gap-Day Opening-Range Stop — Frequency & Behaviour Study

**STATUS: DONE** · **VERDICT: KEEP AS A LOW-FREQUENCY RISK OVERLAY** (not a backtested net-P&L edge —
a fill-quality / churn-avoidance tweak on rare gap days; forward-validating live via the engine.)

## The ask
On a day NIFTY opens *outside* the V2 fly's ±2% band, the old engine insta-exited on the first 1-min
candle (~09:16) at the gapped premium, then could re-enter at 09:20 (churn). New rule (now live, paper):
no stop action for the first 5 min, then exit on a 1-min close beyond the 09:15–09:20 opening range
(OR) high/low — both sides; normal days keep the 2% move-stop. Question: is this worth it, how often
does it fire, and does waiting help or hurt?

## Why there's no native backtest
- **AlgoTest** cannot express an opening-range-conditional stop on a positional book.
- **Multi-year NIFTY 1-min** is unavailable (Kite caps minute history ~60 days; our DB doesn't hold it),
  so an exact OR-vs-insta fly P&L over years can't be reconstructed.
- Feasible instead: a **frequency + underlying-behaviour** read on NIFTY daily (2019–2026), which is
  enough to size the decision. Options P&L is inferred qualitatively (short fly = short gamma:
  continuation = more loss, revert = recovery; the long wings cap the tail either way).

## What was run
`research/63_v2_gap_or_stop/scripts/gapstudy.py` — NIFTY daily (tok 256265), 1846 bars, 2019-01-01→2026-06-12.
For each session: overnight gap = (open − prior_close)/prior_close; flag |gap| ≥ 2%. Per gap day measure
the further run beyond the open in the gap direction (adverse for a short fly), the revert back toward the
prior close (favourable), and whether the day *closed back inside* the prior-close ±2% band.

## Findings
- **Rare event:** 40 / 1845 sessions = **2.17%** (~5.4/yr) opened ≥2% gapped. **But heavily COVID-skewed**
  — 22 of 40 fall in Feb–Jun 2020. Post-2020 (2021–2026) it's only ~14 days over 5.5 yrs ≈ **2–3/yr** in
  normal regimes.
- **Split direction:** 18 gap-UP, 22 gap-DN.
- **~28% fully revert** — closed back inside the prior-close ±2% band on 11/40 days. These are exactly the
  days where the old insta-exit dumps at the worst print and the position would have recovered → the
  OR-stop's win cases.
- **Adverse continuation is usually modest:** median further-run beyond the open = **1.10%** (gap-UP 0.70%,
  gap-DN 1.21%). Post-2020 days are mostly 0.0–1.2%.
- **The big damage days (further-run 4–6%) are ALL COVID-2020 gap-down crashes** that kept falling. On those
  the OR-stop exits too (price breaks the OR downward) — it does *not* save you on a true crash, but the
  **defined-risk wings cap the loss regardless**, so waiting 5 min vs insta is ~a wash there.

## Interpretation (for the short fly)
- **Win case (~28%, revert days):** OR-stop avoids the worst-fill insta-exit and often holds through to
  recovery. Clear improvement.
- **Cost case (continuation days):** OR-stop still exits, just slightly later — median ~1% more adverse
  move, and that incremental loss is **bounded by the wings**. Modest, capped downside.
- **Crash case (rare, COVID-like):** wash on timing; wings are the real protection either way.
- **Churn:** structurally eliminated — the exit can only fire after 09:20 while still holding, so no
  same-day re-entry.

Net: a **low-frequency, asymmetric-favourable** risk tweak. It is **not** a net-P&L alpha edge and isn't
claimed as one; it improves fill quality and removes churn on the handful of gap days per year.

## Caveat on the engine's trigger
The engine flags `gap_day` when today's **open is >2% from `entry_spot`** (which may be 1–4 days old), so it
fires on *drift-to-2%* opens as well as true overnight gaps — i.e. **more often than the 5.4/yr overnight
count**. That's consistent with the rule as stated ("open outside our boundary"); on a drift day the OR sits
near the open, so the OR-stop just behaves like a tight stop around the opening range. Acceptable.

## Forward validation (the real test)
The engine now records `gap_day`, `or_high`, `or_low` on the open position and logs `gap_or_break` exits.
Plan: accumulate live gap-day cases in paper; for each, compare the OR-exit outcome vs the hypothetical
insta-2% exit. After a handful of live cases, confirm the win/cost split matches this study before relying
on it in live mode.
