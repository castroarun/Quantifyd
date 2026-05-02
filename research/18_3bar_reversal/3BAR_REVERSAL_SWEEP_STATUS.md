# 3-Bar Reversal — Live Status

**Started:** 2026-04-24 post-market session
**Goal:** Test if a "3-bar reversal" price-action pattern (strong green →
strong red → strong green close above prior 2 highs) with MACD confirmation
is a viable intraday system on 5/10/15-min timeframes. Mirror pattern for
shorts.

## Pattern definition

### LONG setup (bullish reversal)
- **Bar 1:** strong GREEN — body ≥60% of (high-low), close > open
- **Bar 2:** strong RED — body ≥60% of (high-low), close < open
- **Bar 3:** strong GREEN, body ≥60%, AND `close > max(bar1.high, bar2.high)`

Entry: next bar's open after bar 3 close.
Initial stop: low of bar 1 (default). Variant: low of bar 2.
Target: 2× SL distance from entry (R:R 1:2).

### SHORT setup (bearish reversal — mirrored)
- **Bar 1:** strong RED · **Bar 2:** strong GREEN · **Bar 3:** strong RED that
  closes below `min(bar1.low, bar2.low)`
- Stop: high of bar 1. Target: 2× SL distance.

### Common rules
- Time stop: 24 bars (no infinite hold)
- EOD safety exit: 15:15
- Max 1 trade/stock/day
- Entry window: 09:45 → 14:00
- Sizing: Rs 2,500 risk per trade, Rs 3L cap, 0.15% round-trip costs

## MACD confirmation — 2 variants

| Variant | Condition for LONG | Condition for SHORT |
|---|---|---|
| `macd_cross` | macd line just crossed ABOVE signal line | macd just crossed BELOW signal |
| `macd_zero` | macd line > 0 (positive territory) | macd line < 0 |

MACD(12, 26, 9) — standard.

## Phase plan

### Phase 1 — Signal × TF × MACD (6 cells)
3 TFs (5-min, 10-min, 15-min) × 2 MACD conditions × both directions per cell.

### Phase 2 — Confluence filters (top variant from Phase 1 + each)
- + CPR direction (close vs prior-day pivot)
- + RSI > 50 long / < 50 short
- + Stochastic bullish/bearish crossover in OS/OB zone
- + Higher TF support (60-min EMA21 slope agrees)
- + Bollinger Band — close above middle (long) / below (short)
- + Confluence stack (CPR + HTF + RSI50)

### Phase 3 — SL/Target tweaks on best
- SL below bar 1 + 2R target (baseline)
- SL below bar 2 (tighter, higher risk-of-stop-out)
- SL below bar 1 + 1.5R target
- SL below bar 1 + 3R target

**Universe:** 15 non-ORB liquid F&O stocks (research/16 set).
**Period:** 2024-03-18 → 2026-03-12.

## Pass criteria
PF ≥ 1.20 AND Sharpe ≥ 0.8 (basic profitability gate).
Walk-forward later if any cell passes.

---

## Status

| Phase | Status | Output |
|---|---|---|
| Engine build | ✅ done | `scripts/run_3bar.py` |
| Phase 1 sweep (6 cells) | ✅ done (365s) | `results/phase1_summary.csv` |
| Phase 2 filters (7 cells) | ✅ done (250s) | `results/phase2_summary.csv` |
| Phase 3 tweaks (4 cells) | ✅ done (97s) | `results/phase3_summary.csv` |
| Findings | ✅ done — DEAD | `FINDINGS.md` |

### Verdict — all 17 cells negative

| Phase | Best PF |
|---|---:|
| Phase 1 (signal + TF + MACD) | 0.59 |
| Phase 2 (+ filters) | 0.59 (no improvement) |
| Phase 3 (SL/target tweaks) | **0.65** (still negative) |

WR structurally locked at 33-38% across all variants. Pattern + MACD doesn't have edge on Indian cash 5/10/15-min bars. Same finding pattern as research/13/15/16 — generic intraday directional signals fail consistently.

## Crash recovery (for the human)

### Rerun any phase
```bash
cd c:/Users/Castro/Documents/Projects/Covered_Calls
python research/18_3bar_reversal/scripts/run_3bar.py --phase 1
python research/18_3bar_reversal/scripts/run_3bar.py --phase 2
python research/18_3bar_reversal/scripts/run_3bar.py --phase 3
```

### Inspect what's done
```bash
ls research/18_3bar_reversal/results/
tail -3 research/18_3bar_reversal/results/progress.txt
```

### Files written incrementally per phase
- `results/phase{N}_summary.csv` — variant metrics (written at end of phase)
- `results/phase{N}_trades.csv` — per-trade log per phase
- `results/progress.txt` — heartbeat with last completed cell + timestamp

### What NOT to touch
- The pattern detection logic in `run_3bar.py`
- The 15-stock universe (matches research/16 for cross-comparison)
