# Forward-ATM vs Spot-ATM Entry — Do the CSL Books Sell a Skewed Straddle?

STATUS: **DONE** — verdict **NO EDGE**; leave the live CSL entry rule alone.
(launched 2026-08-21 by the ops session; executed by a research agent, same day)

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
| 2026-08-21 11:35 IST | Research agent picked up; spec read, harness pattern taken from research/114 | READ-ONLY on options_data.db |
| 2026-08-21 11:45 IST | `scripts/run_forward_snap.py` written and launched | A/B replay of COMB+TimeB x NIFTY/SENSEX, 86 recorded days, dwell 2 (live) + dwell 0 (sensitivity) |
| 2026-08-21 11:52 IST | DTE keying fixed to the live weekday map | recording gaps (2026-04-21..24 absent) made a calendar-derived DTE disagree once; live keys off `wd2dte`, so we replay that |
| 2026-08-21 12:1x IST | First full pass done, 964 rows; sensitivity arm found buggy | the "no-dwell" arm was exiting unconditionally on minute 2 rather than on a breach — fixed and re-run rather than reported |
| 2026-08-21 12:5x IST | Re-run complete, all six measurements produced | `results/fs_detail.csv` (964 rows), `results/analysis.txt` |
| 2026-08-21 13:0x IST | **DONE — verdict NO EDGE.** `results/RESULTS.md` written, INDEX updated | recommendation: no change to `csl_paper_exec.py` |

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

**Verdict: NO EDGE. Leave the live CSL entry rule alone.** Full write-up, tables and sins
accounting in `results/RESULTS.md`.

Scoring against the success criterion set in §4 — B must beat-or-match on net P&L **and**
reduce entry skew **and** reduce directional exposure:

| # | Criterion | Result | Verdict |
|---|---|---|---|
| 1 | Beat or match net P&L | pooled **–65 Rs/lot/day (t –1.79)**; negative in 11 of 14 book×DTE cells; worst day worse in both venues | **FAIL** (at best a coin-flip) |
| 2 | Reduce entry skew | mean \|skew\| NIFTY 21.4→12.8 (t –5.74), SENSEX 65.9→27.8 (t –7.46); mean signed skew → ~0 | **PASS** |
| 3 | Reduce directional exposure | arm A has **no significant tilt to remove** (t 1.13 NIFTY / 0.66 SENSEX) and arm B's slope is *larger*, not smaller, in every cut | **FAIL** |

1 of 3 → **no change.**

The six required measurements:

1. **Divergence is real and frequent** — B picks a different strike on **31% of NIFTY** and
   **48% of SENSEX** entries; mean |forward−spot| 16.4 / 54.4 pts. It is a cost-of-carry
   basis and it ladders monotonically with DTE (NIFTY DTE0 gap −2.2 / 3% changed → SENSEX
   DTE4 +108.1 / 76% changed). The 2026-08-21 09:16 case reproduces exactly in the replay.
2. **P&L**: A total 141,427 vs B 125,818 Rs/lot over 241 paired book-days; mean 587 vs 522.
   On the changed-strike days only: NIFTY −119 (t –1.02), SENSEX −195 (t –1.50), B better
   on 41%. Nothing significant — but nothing gained, and B's tail is worse.
3. **Entry skew**: B removes it, decisively (see table above). The mechanism is confirmed;
   the premise about its *consequences* is what fails.
4. **Directionality**: what drives these books is |move|, not direction — r −0.49 to −0.73
   (t up to −11.7) on absolute move vs r 0.06–0.17 (n.s.) on signed move. B is *more*
   short-gamma (−4,270 vs −3,667 NIFTY; −6,654 vs −6,150 SENSEX), which is economically
   right and strategically the wrong direction for a risk argument.
5. **Monotonicity**: absent. (B−A) by |gap| quartile is flat on NIFTY (+0/−50/−50/−44) and
   *shrinks* with gap on SENSEX (+0/−220/−104/−54); regression t −1.41 / −0.24. The skew
   reduction over the same buckets is strongly monotone — a real mechanic with no P&L
   consequence at this holding period.
6. **Cost of switching**: liquidity is a non-issue (B's bid-ask is equal-or-tighter,
   0.217% vs 0.220% NIFTY, 0.253% vs 0.262% SENSEX). The genuine cost is premium — B
   collects **−0.73 pts NIFTY (t −4.38)** and **−2.35 pts SENSEX (t −4.98)** less credit,
   because straddle premium is minimised at the forward.

**The blocker, plainly:** research/111 fitted and froze every per-DTE window and combined-SL
level against spot-ATM straddles. Forward-ATM changes the credit significantly, which moves
the SL threshold `(1+sl)·credit`, which moves when the stop fires — so this is not a
cosmetic re-centring, it shifts the whole calibrated exit stack off its validation basis.
Only worth doing if forward-ATM were neutral-or-better. It is not.

**Robustness**: same conclusion with the SL dwell removed (NIFTY −20 t −0.64, SENSEX −78
t −1.30); unchanged when the in-flight 2026-08-21 is excluded; (B−A) flips sign month to
month in both venues. **Main caveat**: one 4-month regime, n=17–18 per book×DTE cell.

**Recommendation**: no change to `csl_paper_exec.py`; no change to the 916 suite either
(its forward snap is a different, defensible trade-off, and moving it would invalidate its
own basis for the same reason). Record the strike disagreement as an expected, harmless
property so the next morning's divergence does not restart this investigation.
