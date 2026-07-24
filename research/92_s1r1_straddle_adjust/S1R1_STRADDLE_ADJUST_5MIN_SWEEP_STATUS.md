# S1/R1-Break Straddle Adjustment — Cut Loser + ST(7,3) Trail (price vs premium), 5-min, 64 days

STATUS: DONE

## 2. The Ask
**What Arun asked:** New NAS system. Short a straddle at 09:16 (same as 916). Only when a 5-min candle
CLOSES above R1 or below S1 do we adjust: cut the losing side and trail the other side with
SuperTrend(7,3). Test ST trailing on BOTH the 5-min price candle AND the naked option premium — see
which works better.

**What we're testing:** Over the recorded NIFTY chain (options_data.db, 64 days, per-minute), replay:
short ATM straddle 09:16 → hold (no per-leg SL) → on the FIRST 5-min close beyond prev-day R1/S1, cut
the losing leg and trail the survivor. Compare four modes:
- HOLD    — never adjust, hold both legs to 15:15 (pure straddle baseline)
- CUT     — on break, cut loser, hold survivor to 15:15 (no trail) — isolates the cut
- PRICE   — on break, cut loser, trail survivor with ST(7,3) on the 5-min NIFTY price
- PREMIUM — on break, cut loser, trail survivor with ST(7,3) on the survivor's option premium
Which of PRICE / PREMIUM trails better, and does the adjustment beat HOLD.

## 3. The Base
- Entry: 09:16 ATM straddle (nearest 50), 2 lots, QTY 130 (NIFTY lot 65). No per-leg SL.
- Pivots: standard floor pivots from the PREVIOUS trading day's regular-session OHLC (from
  underlying_spot): P=(H+L+C)/3, R1=2P-L, S1=2P-H.
- Break: first 5-min candle (built from per-minute underlying_spot) whose CLOSE > R1 (up) or < S1
  (down), after entry.
- Adjust: up-break -> the CALL is the loser (cut CE, keep PE); down-break -> cut PE, keep CE.
- Trail: ST(7,3) on 5-min candles.
    - PRICE: survivor PE exits when price trend flips DOWN; survivor CE exits when it flips UP.
    - PREMIUM: survivor exits when its own premium ST flips UP (premium rising = short losing).
- Force exit 15:15. Brokerage Rs80/leg.
- Optimism: LTP fills, no slippage, 1-min/5-min resolution.

## 4. Plan
- 4 modes x 64 days. Rank by total net, Calmar, win-rate; break-days-only comparison (the modes only
  differ on break days). Report which trail wins and whether the system beats HOLD.
- If promising: parameter check (ST mult 2 vs 3), per-year/per-half stability, net-of-cost sensitivity.

## 5-8: results/ (s1r1_results.csv, RESULTS.md). Runner:
research/92_s1r1_straddle_adjust/scripts/run_s1r1.py. Crash recovery: re-run the runner.
