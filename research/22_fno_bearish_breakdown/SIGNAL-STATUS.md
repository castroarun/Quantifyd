# Signal 1 — F&O Bearish 252-day Breakdown — STATUS

## Goal

Validate a SHORT-side mirror of research/21's vol_3x winner: 252-day low
breakdown + volume confirmation + 200-SMA regime, sized for bear put debit
spreads on the F&O 76-symbol universe.

Pass criteria (OOS 2023-2025): PF ≥ 1.20 AND Sharpe ≥ 0.8 AND MaxDD ≤ 30%.

## Plan

| # | Variant | vol | ATR floor | ADX |
|---|---|---|---|---|
| 1 | baseline           | 2.5x | no  | no |
| 2 | vol_3x             | 3.0x | no  | no |
| 3 | vol_3x_atr_floor   | 3.0x | yes | no |
| 4 | vol_3x_adx         | 3.0x | no  | yes (>25) |
| 5 | vol_3x_confluence  | 3.0x | yes | yes (>25) |

## Status

DONE — all 5 variants run, walk-forward complete on top 3.

**Verdict: DOA. All 5 variants negative full-period; OOS PF=0.61-0.69, Sharpe<0.**

Full-period (2018-2025):

| Variant            | Trades | WR%  | PF   | CAGR%  | Sharpe | MaxDD% |
|--------------------|--------|------|------|--------|--------|--------|
| baseline           |  96    | 25.0 | 0.71 | -1.83  | -0.22  | 26.37  |
| vol_3x             |  68    | 23.5 | 0.57 | -1.98  | -0.32  | 18.97  |
| vol_3x_atr_floor   |  68    | 23.5 | 0.57 | -1.98  | -0.32  | 18.97  |
| vol_3x_adx         |  49    | 20.4 | 0.51 | -1.74  | -0.28  | 15.68  |
| vol_3x_confluence  |  49    | 20.4 | 0.51 | -1.74  | -0.28  | 15.68  |

Walk-forward (OOS 2023-2025) — all FAIL. Indian F&O universe is too long-biased
for symmetric 252-day breakdown shorts; new 252-day lows usually mean-revert
rather than follow through. ATR-floor filter is degenerate after vol_3x.

Outputs in `results/`: `summary.csv`, `walk_forward.csv`, equity curves, universe.

## Crash recovery

Sweep is single-process; resume from scratch is cheap (~3-4 min).
```bash
cd research/22_fno_bearish_breakdown
python scripts/run_bearish_sweep.py            # ~3 min
python scripts/walk_forward_bearish.py         # ~2 min
```

## Final aggregation

`FINDINGS.md` gets the verdict + tradable variant (or DOA if none pass).
