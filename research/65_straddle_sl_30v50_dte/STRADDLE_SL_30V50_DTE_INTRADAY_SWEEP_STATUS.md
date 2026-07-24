# Short ATM Straddle — 30% vs 50% per-leg SL, by DTE (NIFTY, per-minute chain)

STATUS: RUNNING | cheap probe (G1/G2), sample-limited

## The Ask
**What you asked:** "backtest 30% vs 50% [per-leg SL on the cascade ATM straddle] with our
available database, focus on different DTEs."
**What we're testing:** For a short ATM NIFTY straddle entered in the morning, does a **50%
per-leg stop** beat the live system's **30%** stop — net of cost — and does the answer depend
on **DTE at entry**? Exit policy = the cascade-ATM2 behaviour: **on the FIRST leg to breach its
SL, close BOTH legs** (no trail, no re-enter); otherwise hold to the 15:15 EOD square-off.

## Economic hypothesis (G0)
A short straddle is short gamma / long theta. A **tight stop (30%)** cuts losers fast but gets
**whipsawed out** on intraday noise that later reverts (you pay the stop, then the move comes
back — today's 09:16 gap-then-revert is the canonical case). A **wider stop (50%)** rides through
noise to harvest theta, but takes a **bigger loss when the move is real and continues** (gap-and-
go). So the SL width is a noise-vs-trend trade-off; the net winner is **regime/DTE dependent** —
0-DTE has huge gamma (stops hit easily, premium decays fast) vs higher DTE (calmer, more theta to
collect). Counterparty: directional/gamma buyers. This is **exit-tuning on an existing system**,
not a new edge → judged at G2 (net mechanics) / G3 (stability), NOT a from-scratch G0-G1.

## The Base (mechanics — locked)
- **Universe/instrument:** NIFTY weekly ATM straddle (nearest expiry), 1 lot (qty 65).
- **Entry:** the per-minute snapshot nearest **09:20 IST** each day; ATM strike = round(spot/50)*50;
  entry premiums = ATM CE & PE `ltp` at that minute (nearest expiry).
- **Exit policy (primary = cascade-ATM2):** walk minute-by-minute to 15:15; if **either** leg's
  ltp ≥ entry×(1+SL) → **close BOTH legs** at that minute's ltps (first-breach). Else close both at
  the 15:15 ltp. (Secondary policy reported: per-leg SL, keep survivor to EOD.)
- **SL grid:** {30%, 50%} + **no-stop (EOD only)** baseline.
- **Cost:** parameterised. Base = ₹20/order ×4 + slippage 0.75 pt/leg ×2 legs round-trip ≈
  **₹20×4 + 0.75×65×4 = ₹275/straddle** (~4.2 premium-pts). Report **gross AND net** + a
  cost-sensitivity (0 / base / 2×).
- **Grouping:** by **DTE = (nearest-expiry − entry-date) in calendar days**. Also a per-day table.
- **Success metric:** mean **net** P&L per straddle (₹) by DTE; secondary win% + worst-day.

## The Plan (grid + cells)
- Days: all full trading days in `options_data.db.option_chain` (2026-04-21 → ~06-13, ~38 days).
- Cells = days × {SL30, SL50, NOSTOP} × {exit-both, keep-survivor}. Aggregate by DTE.
- Falsification: if **50% is not clearly better net** (or only wins on ≤1 DTE bucket on a handful
  of days), verdict = NO ROBUST EDGE for widening the stop / SIGNAL at best.

## Data reality (G-integrity — stated loudly)
- `option_chain`: per-minute NIFTY chain, **2026-04-20 13:56 → 2026-06-16** (~2 months). Full
  ltp/spot/Greeks. `option_ohlc` EMPTY (no deep history). **~38 trading days only.**
- ⇒ This is a **2-month, single-regime sample**. DTE buckets ≈ 6–10 days each. **Any result is a
  SIGNAL at best, not robust.** No multi-year, no walk-forward possible. Will report per-day so the
  thinness is visible.

## Status (live log)
| Time | Event |
|---|---|
| 2026-06-16 10:2x IST | STATUS written; runner built; launching on VPS (data is there) |

## Files
| File | Purpose | Committable |
|---|---|---|
| this doc | framing + crash-recovery | yes |
| scripts/run_sl_sweep.py | the sweep runner | yes |
| results/sl_dte_sweep.csv | per-day per-policy P&L | yes (small) |
| results/RESULTS.md | verdict | yes |
