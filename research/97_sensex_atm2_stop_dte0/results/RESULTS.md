# research/97 — SENSEX Exit-Stack Calibration — RESULTS

## Verdict: **INCONCLUSIVE (G2) — DO NOT deploy a stop level from this sample.** One actionable read (30% SL bad on expiry); one flag (DTE1 looks unprofitable). Keep current stops as tail insurance until more expiry cycles (esp. a trend expiry) are recorded.

Run: 2026-07-30 (flat book, market open, no live trades). Layer A only; Layer B deferred (same
sample limitation). Data: `options_data.db` SENSEX chain, 14 expiry cycles, 28 day-entries (14 DTE0 +
14 DTE1). Short ATM straddle 09:16 → 15:15, one-and-done stop, 2 lots (QTY 40). Net-of-cost.

## Cost model
v1 (sell@bid/buy@ask) discarded — SENSEX chain bid/ask too noisy (35% missing ask, p90 spread = 60%
of ltp) → manufactured losses. v2 = fill at **ltp ± 1% slippage/side + brokerage**; liquidity guard
(entry strike vol>0 or oi>0). v1 and v2 agree directionally, so the reads below are cost-robust.

## Layer A — net Rs/trade, best→worst

### DTE0 (Thursday expiry)
| stop | avg/tr | worst | win% |
|---|---|---|---|
| netpct_150 (≈hold) | **+5,878** | −272 | 86 |
| rupee_5000 | +4,544 | −11,716 | 79 |
| netpct_75/100 | +4,178..+4,407 | −13k..−17k | 79 |
| rupee_2500 | −212 | −9,130 | 50 |
| move0.4 (current ATM2) | −383 | −3,421 | 36 |
| rupee_1500/2000 | −922 / −1,425 | −9,130 | 43 |
| **legSL30 (current ATM/ATM4)** | **−964** | −2,381 | **14** |

### DTE1 (Wednesday) — every variant negative
| stop | avg/tr | worst | win% |
|---|---|---|---|
| legSL30 | −1,291 | −3,474 | 7 |
| rupee_3000 | −1,433 | −8,178 | 43 |
| move0.4 | −2,249 | −6,179 | 7 |
| netpct_150 | −3,033 | −36,056 | 43 |

## Reads
1. **30% per-leg SL is bad on expiry (answers the user's Q2 directly): DTE0 win-rate 14%, −964/tr.**
   It exits on premium noise and forgoes the expiry theta crush. The tight rupee stops and the ±0.4%
   move-stop are also all negative on DTE0. On DTE0 the winning behaviour is to HOLD.
2. **BUT "hold wins on DTE0" is a sample artifact** — all 14 in-sample expiries pinned (biggest DTE0
   move +0.74%; whole-window max |move| = −1.61% on one DTE1). A short straddle held through a *trending*
   expiry loses big on the ITM leg at settlement; there is **no such day in the sample**. The stop's
   entire value is that rare tail → this sample cannot price it. Do NOT read this as "remove the stop."
   (Today, 2026-07-30, was a mild-trend expiry, +0.43%; the full straddle still netted ~+₹1,900 because
   the PE decay covered the CE loss — consistent with "hold usually fine on moderate moves," tail unknown.)
3. **DTE1 intraday short straddles look structurally unprofitable net-of-cost** (every stop negative;
   small intraday decay can't beat moves+costs). Thin/tail-driven (14 days) — flag for more data:
   SENSEX may not warrant DTE1 entries.

## Recommendation (no live change)
- **Do NOT deploy a calibrated SENSEX stop level from this sample** — it would optimize for pinned days
  and remove the tail insurance, the opposite of risk management. The guardian-flagged NIFTY-borrowed
  numbers (±0.4% move-stop, −₹1,300/lot, +₹1,667 TP, ₹2,500/lot) stay in place as provisional insurance.
- **Actionable, low-regret:** the **30% per-leg SL demonstrably hurts on DTE0** (whipsaws the theta
  crush) — candidate to loosen/disable it on expiry-day for ATM/ATM4. Still verify against a trend
  expiry before deploying.
- **Gather more cycles** (especially a trending expiry) before any stop-level calibration. Re-run this
  sweep as the sample grows. Layer B (book-level stop/TP/trail) deferred — same benign-sample limitation
  would make its numbers equally unreliable.
- **DTE1 profitability** is a separate open question worth its own study (is the SENSEX Wed entry +EV at all?).

## Files
- `scripts/run_sensex_atm2_stop_sweep.py` — Layer A runner (SLIP env = slippage/side; default 0.01)
- `results/layerA_cells.csv` — per-cell net/tail/win
