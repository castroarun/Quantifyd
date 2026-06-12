# Phase 27 — Weekly Re-Entry vs Month-End Re-Entry — RESULTS

**Verdict: SPLIT — weekly re-entry is a STRATEGY IMPROVEMENT for `allcash`, but
a NET NEGATIVE (drawdown blow-out) for `keeptop8`. Do not treat "faster re-entry"
as universally good — it depends on whether you de-risked fully or partially.**

Daily-marked engine, weekly 100-SMA gate, 2014-2026, VPS canonical data.
Both `month-end (BASE)` configs reproduce published Phase 22/24 exactly → faithful.

| Config | CAGR | Post-tax@20% | MaxDD | Sharpe | Calmar | Re-entry ev | Fills |
|---|---|---|---|---|---|---|---|
| allcash  month-end (BASE) | 34.2% | 28.4% | −22.2% | 1.82 | 1.54 | 0 | 108 |
| **allcash  WEEKLY-reentry** | **35.5%** | **29.0%** | **−20.7%** | **1.84** | **1.72** | 24 | 124 |
| keeptop8 month-end (BASE) | 33.6% | 28.3% | −20.2% | 1.71 | 1.66 | 0 | 99 |
| keeptop8 WEEKLY-reentry | 32.2% | 26.7% | **−28.5%** | 1.61 | 1.13 | 22 | 121 |

## The finding

1. **`allcash` + weekly re-entry WINS on every axis** — CAGR +1.3pp, post-tax
   +0.6pp, drawdown *shallower* (−22.2→−20.7%), Calmar 1.54→1.72 — at a cost of
   only 24 re-entry events over 12y (~2/yr, +16 fills). The locked "slow-in" left
   return on the table here. When you're 100% in cash, getting back in faster
   captures recoveries you were otherwise sitting out.

2. **`keeptop8` + weekly re-entry LOSES badly** — CAGR −1.4pp, post-tax −1.6pp,
   and drawdown BLOWS OUT −20.2→−28.5% (Calmar 1.66→1.13). Because keep-top8
   already holds your 8 strongest *through* the dip, re-entering the other 7 on
   the first green week buys into **false dawns** — they roll over and you now
   hold 15 falling mid-caps. This is the same failure mode Phase 25 found with
   "gated refill". The slow month-end re-entry is PROTECTIVE for keep-top8.

## The unifying insight

**Fast re-entry helps when you de-risked FULLY (cash), hurts when you de-risked
PARTIALLY (keep-8).** Keep-8 already gives you recovery participation via the 8
you never sold; bolting fast re-entry on top over-exposes you to whipsaws.

## Per-year tells (the mechanism)

| Year | allcash M-E | allcash WK | keeptop8 M-E | keeptop8 WK | Note |
|---|---|---|---|---|---|
| 2022 | +1.1% | **+17.2%** | +0.2% | +3.1% | choppy recovery — slow-in sat in cash through rallies |
| 2020 | +85.7% | +77.5% | +69.2% | +60.7% | sharp V — fast re-entry slightly whipsawed |
| 2025 | +5.3% | −5.4% | −6.9% | **−14.2%** | false-dawn year — fast re-entry into keep-8 doubled the loss |

So even for `allcash`, weekly re-entry isn't free (it lost ~8pp in the 2020 V and
2025); it wins on the 12y net because 2022-type choppy recoveries dominate.

## Recommendation

- The **finalized/productized SMOOTHEST is `allcash` base** → **weekly re-entry is
  an adopt-worthy upgrade** (Calmar 1.54→1.72, post-tax +0.6pp). Worth promoting
  to a locked variant after a per-year robustness re-read.
- If instead the book runs **`keeptop8`** (the client-smoothness option that never
  goes 100%→0%) → **keep the month-end re-entry; do NOT add weekly buy-back.**
- The two client-relevant finalists are therefore: `allcash + weekly-reentry`
  (best Calmar 1.72, but all-or-nothing exits) vs `keeptop8 + month-end`
  (Calmar 1.66, never dumps the whole book). Pick on client risk-appetite, not
  on a few bps of Calmar.

## Caveats

- NIFTYBEES gate uses close-to-close (no true OHLC) — unchanged from the study.
- 2025/2026 are partial / recent; the keep-8 weekly blow-out leans on 2025.
- Cost model is the study's turnover proxy (0.4% RT × name-turnover), same on both
  sides → fair for comparison; absolute live costs may differ.
- Single window; not re-OOS'd. Next step if adopting `allcash+weekly`: walk-forward
  + per-year stability re-read (the 2020/2025 give-backs are the stress cases).
