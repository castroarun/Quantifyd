# Intraday Stocks Buy/Sell System Discovery — 9 Signal Families, 150 Names, 5-min (2015→)

STATUS: RUNNING (wave-1 screen)

## 1. The Ask

**What you asked:** "frame, backtest, optimize a new intraday stocks based
buying selling system... explore every angle — price action, indicator
combinations, timeframes, CPRs, support/resistances, everything possible...
CAGR must be meaningful, at least 20%, but go for the highest possible."

**What we're testing:** Across 150 liquid names (74 F&O + 76 deepest-history
others), 5-min bars 2015-02→, do any of 9 pre-registered intraday signal
families produce net-positive per-trade returns strong enough to compound
into a ≥20% CAGR book after realistic intraday cash costs?

**The governing arithmetic (kill-line):** 20% CAGR ≈ +8bps/day net on fully
deployed capital. One capital-turn/day ⇒ candidate families need ≥ +10-15
bps/trade net (t≥3) to survive book construction. Anything below is noise
against the cost floor.

## 2. The Base

- **Costs:** intraday cash equity ≈ 10bps round-trip (brokerage+STT 0.025%
  sell+charges+2-3bp slippage/side). Sensitivity at 15bps reported for
  passers. This is the hurdle that killed intraday ORB (r/89: negative every
  era) — stated up front.
- **Fills:** signal on 5-min close t → enter next bar open; exit 15:15 bar
  close (wave 1 = hold-to-EOD; exit engineering only for wave-2 passers).
  One trade per family-cell per name per day (first trigger).
- **Splits:** IS 2015-02→2021-09 · Val 2021-10→2023-12 · **OOS 2024+
  QUARANTINED** (fresh families; the consumed r/81 ORB cells are excluded
  from this study's grid — FAILED_ORB fades the pattern, it does not re-test it).
- **Controls:** every cell scored raw AND as excess vs a TIME-MATCHED
  baseline (mean same-entry-slot→EOD return across all days/names) — the
  intraday analogue of the drift control that exposed r/87.
- **Gate (wave 1):** excess net > 0 AND t ≥ 3.0 (large n demands more) AND
  ≥55% names positive AND both IS halves (2015-18 / 2018-21) same sign.

## 3. Wave-1 families (pre-registered, 30 cells)

| # | Family | Signal (L; S = mirror) | Variants |
|---|---|---|---|
| 1 | TREND_BRK | break of day-range high at/after {11:00, 12:30} | 2×2dir |
| 2 | VWAP_RECLAIM | ≥{6,12} bars below VWAP, then close above | 2×2 |
| 3 | VWAP_FADE | extension ≥{1.5%, 2.5%} from VWAP → fade | 2×2 |
| 4 | CPR_TREND | narrow CPR (<0.3%) + open above TC + first 15m up | 1×2 |
| 5 | PIV_BOUNCE | tag S1 (±0.15%) + 5m reversal close | 1×2 |
| 6 | MORN_FADE | first-hour move ≤−{1.5%, 2.5%} + 15m higher close | 2×2 |
| 7 | FAILED_ORB | OR15 break that closes back inside within 30m → fade | 1×2 |
| 8 | PD_LEVEL | tag prev-day low + reversal (mirror: prev-day high) | 1×2 |
| 9 | LATE_MOM | 14:15 price in top{/bottom} 20% of day range → ride close | 1×2 |

Ledger: +30 (program total 391 + 30 = 421). Wave 2 (only for passers):
timeframe variants (15m/30m triggers), exits (trail/stop/target), filters
(CPR-width, gap, VIX), book construction with CAGR/DD, cost sensitivity.

## 4. Status / event log

| Date/time | Event | Notes |
|---|---|---|
| 2026-08-09 ~19:50 IST | Pre-registration written; runner authored | 30 cells |
| (launch row in log) | Wave-1 launched on VPS | results/wave1.log |

## 5. Crash recovery

- Progress: `tail research/109_intraday_stocks/results/wave1.log` (per-symbol
  progress rows); results accumulate in `results/wave1_cells.csv`
- Alive: `ps -eo args | grep '[r]un_wave1'`
- Resume: rerun `venv/bin/python3 research/109_intraday_stocks/scripts/run_wave1.py`
  — per-symbol checkpoint (`results/wave1_done_syms.txt`), skips done names.
- Data read-only from `backtest_data/market_data.db`.

## 6. Files

| File | Purpose | Committable |
|---|---|---|
| `scripts/run_wave1.py` | streaming 9-family screen | yes |
| `results/wave1_trades.parquet` | per-trade rows (symbol, cell, slot, ret) | if <20MB |
| `results/wave1_cells.csv` | per-cell aggregates | yes |
| `results/RESULTS.md` | verdict | yes (after) |

## 7. Findings

(populated during/after run)
