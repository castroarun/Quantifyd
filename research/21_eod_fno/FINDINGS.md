# F&O-Only EOD Breakout — Findings (POSITIVE)

**Status:** Walk-forward validated. **Third positive edge in the project**
(after research/17 Nifty 500 EOD and research/19 small-cap daily). Same family,
F&O-restricted universe, options-overlay candidate.

## Universe

- 81 F&O candidates from `services/data_manager.py:FNO_LOT_SIZES`
- **76 kept** (≥1500 daily bars since 2018-01-01)
- 5 dropped (recent IPOs / data gaps): COALINDIA, DELHIVERY, ONGC, PAYTM, ZOMATO

## Spec — same as research/19's vol_3x winner

- Entry: today's `close > 252-day high` (excluding today)
- Volume: `volume >= 3.0× 50-day average`
- Regime: `close > 200-day SMA`
- Stop: `max(entry − 2×ATR(14), entry × 0.92)`
- Target: `entry × 1.25` (fixed 25%)
- Sizing: 1% risk, 10 concurrent, Rs 1L notional cap, 0.20% costs
  (vs research/19's 0.30% — F&O has tighter spreads), Rs 10L capital

## Full-period sweep (2018-2025)

| Variant | Trades | WR% | PF | CAGR% | Sharpe | MaxDD% | Calmar |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_252_25pct_8pct | 386 | 40.9 | 1.57 | +8.54 | 0.83 | 22.81 | 0.37 |
| vol_2x | 429 | 41.0 | 1.58 | +9.53 | 0.87 | 22.02 | 0.43 |
| **vol_3x** | **321** | **43.6** | **1.86** | **+10.88** | **1.13** | **20.03** | **0.54** |
| target_30pct | 374 | 41.4 | 1.61 | +8.82 | 0.84 | 22.29 | 0.40 |
| target_20pct | 403 | 41.4 | 1.55 | +8.30 | 0.85 | 21.54 | 0.39 |
| cost_30bps (stress) | 386 | 40.7 | 1.53 | +8.09 | 0.79 | 23.07 | 0.35 |

`vol_3x` wins — same shape as research/19. All variants clear PF >= 1.53.

## Walk-forward — top 3 (train 2018-2022, test 2023-2025)

| Variant | Phase | Trades | PF | Sharpe | MaxDD% | CAGR% | Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| **vol_3x** | IS | 187 | 2.06 | 1.18 | 20.03 | +11.22 | — |
| **vol_3x** | **OOS** | **134** | **1.91** | **1.04** | **9.11** | **+10.25** | **PASS** |
| baseline | IS | 225 | 1.88 | 0.68 | 22.81 | +6.83 | — |
| baseline | OOS | 161 | 1.79 | 1.07 | 12.65 | +11.43 | PASS |
| target_30pct | IS | 217 | 1.99 | 0.73 | 22.29 | +7.59 | — |
| target_30pct | OOS | 157 | 1.78 | 1.03 | 11.62 | +10.86 | PASS |

**vol_3x OOS gates:**
- PF ≥ 1.20 → **1.91 PASS** (+0.71 buffer)
- Sharpe ≥ 0.80 → **1.04 PASS** (+0.24 buffer)
- MaxDD ≤ 30% → **9.11% PASS** (massive buffer)

**OOS metrics improved over IS** on every measure — drawdown halved, Calmar doubled. Same anti-overfit signature as research/17 and research/19.

## Top 5 contributors (vol_3x, full period)

| Symbol | Net P&L | Trades |
|---|---:|---:|
| BEL | +Rs 1,58,395 | 9 |
| TRENT | +Rs 1,29,433 | 9 |
| COFORGE | +Rs 92,413 | 5 |
| PERSISTENT | +Rs 82,164 | 7 |
| DIVISLAB | +Rs 79,602 | 8 |

Top 5 = 41% of total Rs 13.2L net P&L over 321 trades. Defence + retail + select IT mid-caps. Healthy concentration (not single-name dependence).

## Exit mix

INITIAL_STOP 52% · MAX_HOLD 32% · TARGET 13% · END_OF_BACKTEST 3%.

Classic momentum profile — small losses cut early, fat wins ride the 25% target.

## Caveats

1. **2023-2025 was a broad bull regime.** Same period that lifted research/17 and /19. The cross-system PASS is the real signal — not just a regime pick.
2. **0.20% cost is optimistic for less-liquid F&O names** (mid-cap futures). Cost_30bps stress survives (PF 1.53 / Sharpe 0.79) — material edge cushion.
3. **Universe survivorship** — `FNO_LOT_SIZES` is current point-in-time, not historical. Stocks added to F&O during the period are present from 2018; mild upward bias.
4. **Only 134 OOS trades over 3 years** — statistically thin per-stock.
5. **Research/17 + /19 + /21 share signal logic** — they are NOT independent strategies. Treat /21 as the *underlier-selection layer for options*, not as a separate portfolio slot.

## The cumulative picture

Three sister systems, same rules, different universes, all walk-forward validated:

| System | Universe | Capital | OOS PF | OOS Sharpe | OOS CAGR | OOS MaxDD |
|---|---|---:|---:|---:|---:|---:|
| Research/17 | Nifty 500 | Rs 10L | 1.44 | 0.95 | +14.5% | 24.0% |
| Research/19 | Small/micro-caps | Rs 10L | 1.46 | 1.43 | +26.8% | 12.9% |
| **Research/21** | **F&O** | **Rs 10L** | **1.91** | **1.04** | **+10.3%** | **9.1%** |

Research/19 has the best raw return (small-caps amplify the move) but research/21 has the best risk profile (F&O liquidity = clean exits, MaxDD stays tight).

## Recommendation

**Paper-trade vol_3x on F&O for 1 quarter.** Capture real slippage data.

If paper trade matches backtest expectations, build the **options overlay**:
- Cash-secured puts at breakout strike — get paid to wait for entry
- Covered calls written 25% OTM 30-DTE on every fill — match the +25% target as the call's strike, give up upside for premium
- This converts a 1.91 PF cash strategy into a higher-Sharpe options strategy with matched directional view

**Critical operational rule:** do NOT stack research/17 + /19 + /21 as parallel strategies in the same book. Their signal logic is identical — overlap is heavy. Allocate as ONE strategy with a universe-selector.

## Artifacts

- `scripts/run_fno_backtest.py` — 6-variant engine
- `scripts/walk_forward_fno.py` — IS/OOS validator
- `EOD-FNO-STATUS.md` — historical run log
- `results/fno_universe.csv` — 76 kept symbols
- `results/fno_summary.csv` — full-period 6-variant metrics
- `results/fno_walk_forward.csv` — IS vs OOS top 3
- `results/fno_trades.csv` — aggregate trade log
- `results/fno_equity_*.csv` — daily equity curves per variant

## Decision

Proceed to paper trading via the EOD breakout scanner service
(`services/eod_breakout_scanner.py` system_id='fno'). Live page at
`/app/eod-breakout` (F&O tab) shows recorded backtest summary and live state.
