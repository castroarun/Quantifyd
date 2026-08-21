# Forward-ATM vs Spot-ATM Entry — Do the CSL Books Sell a Skewed Straddle?

STATUS: **RUNNING** (launched 2026-08-21 by the ops session; executed by a research agent)

## 2. The Ask

**What Arun asked (2026-08-21):** after noticing the 9:16 suite entered NIFTY at 24300 this
morning while COMB and TimeB entered the same instant at 24250 — "check why different strikes
were taken in the morning" — and, on being shown the cause, asked for it to be studied before
any change.

**The cause, already established (do not re-derive).** The two families define ATM
differently, in code:

- **916 suite** (`services/nas_atm_executor.py`): starts at spot-ATM, then **re-snaps to the
  synthetic forward** — `forward = K + (CE − PE)`, re-rounded — landing on the strike where the
  straddle is actually balanced. Has logging and a no-quote fallback.
- **CSL daemon** (`research/111_.../csl_paper_exec.py`): `K = round(spot/step)*step` only.
  **No forward snap exists.**

On 2026-08-21 at 09:16, spot 24,262:

| Strike | CE | PE | gap | chosen by |
|---|---|---|---|---|
| 24250 | 123.45 | 60.75 | **62.7** | COMB, TimeB |
| 24300 | ~88 | ~82 | **6.2** | ATM, ATM2, ATM4 |

The CSL books sold a straddle 62 points off balance — short a CE effectively ITM against the
forward, i.e. a directional short rather than a neutral short-vol position.

> **The question: across the recorded chain, does forward-snapping the CSL books' entry change
> their results — and by enough to justify changing a live entry rule?**

## 3. The Base

- **Data:** `options_data.db :: option_chain`, 1-minute, ~85 days, 2026-04-20 → 2026-08-21,
  NIFTY and SENSEX, READ-ONLY.
- **Constructions replayed:** the live CSL books as configured — COMB (09:16 to 15:20, per-DTE
  combined-SL) and TimeB (per-DTE windows) — per venue, per DTE.
- **The single varied axis:** entry strike = `spot-ATM` (status quo) vs `forward-ATM` (the
  suite's rule: K + CE − PE, re-rounded, same no-quote fallback). Everything else frozen.
- **Costs:** 0.5 pt/leg-side NIFTY, 1.0 SENSEX, plus Rs30/leg-side/lot.

## 4. Plan — what must be measured

1. **How often do the two even differ?** If they agree on most days this is a non-issue.
   Report the distribution of |forward − spot| and how often the strike actually changes.
2. **P&L:** net total, mean, median, win%, worst day, p05 — per venue, per DTE, both rules.
3. **Skew at entry:** the CE−PE gap under each rule. This is the mechanism; if forward-ATM
   does not materially reduce it, the premise is wrong and the study ends there.
4. **Directionality:** does the spot-ATM entry carry a systematic delta — does its P&L
   correlate with the day's underlying move where the balanced entry's does not? That is the
   real risk being carried and may matter more than the mean.
5. **Monotonicity:** does the difference scale with the size of the spot-forward gap? A real
   effect should; noise will not.
6. **State the blocker plainly:** research/111 validated COMB/TimeB **with spot-ATM entries**.
   Changing the entry rule invalidates that basis unless forward-ATM is neutral-or-better.

**Success criterion:** forward-ATM must beat or match spot-ATM on net P&L AND reduce entry skew
AND reduce directional exposure. Anything less means leave the live rule alone.

## 5. Status

| Date/time | Event | Notes |
|---|---|---|
| 2026-08-21 ~11:2x IST | Divergence spotted live, traced to the forward snap | brief written, agent launched |

## 6. Crash Recovery

Read-only; no live state touched. Scripts in `scripts/`, outputs in `results/`. Re-run with
`venv/bin/python3`. Query the chain per-day (27M rows).

## 7. Files

| File | Purpose | Committable |
|---|---|---|
| `FORWARD_SNAP_ENTRY_1MIN_SWEEP_STATUS.md` | this file | yes |
| `scripts/*.py` | the A/B replay | yes |
| `results/*.csv` | per-day, per-rule detail | yes |
| `results/RESULTS.md` | verdict + recommendation | yes |

## 8. Findings

(to be written by the research agent)
