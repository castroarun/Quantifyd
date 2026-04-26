# F&O Multi-Signal + Spread Structure Deploy — Live Status

**Started:** 2026-04-26 evening session
**Goal:** Optimize 6 trend-detection signals on the F&O universe (each with
filter sweeps), build a debit-spread structure generator, expose both via
the live `/app/eod-breakout` page, and persist spread structures on each
signal for paper-trading.

## Plan

### Backend
- [x] `services/spread_structure.py` — Black-Scholes-priced bull-call /
      bear-put debit spread generator (auto-detects strike intervals,
      theoretical premium, max profit/loss, breakeven, R:R)
- [x] `services/eod_breakout_db.py` — added `direction` and `spread_structure`
      (JSON) columns to `eod_signals` with idempotent ALTER migration
- [x] `services/eod_breakout_scanner.py` — import spread_structure module
- [ ] `services/eod_breakout_scanner.py` — call `build_spread_for_signal`
      at signal-fire time, persist spread JSON on each row
- [ ] `app.py` — surface spread JSON in `/api/eod/<sys>/signals` response

### Frontend
- [ ] `frontend/src/pages/EodBreakout.tsx` — render spread structure
      below each signal row (long/short strikes, debit, max profit, R:R,
      breakeven %), plus a collapsible adjustment-playbook section

### Optimization sweep — agent running in background
6 signal families × 5 filter variants each = 30 backtests. Walk-forward
(train 2018-2022, test 2023-2025) on best variant per family.

| # | Signal family | Folder | Direction | Status |
|---|---|---|---|---|
| 1 | Bearish breakdown (mirror of /21) | `research/22_fno_bearish_breakdown/` | SHORT | running |
| 2 | Donchian 55-day | `research/23_fno_donchian_55/` | LONG | running |
| 3 | RS leadership (60d, Z-score vs Nifty) | `research/24_fno_rs_leadership/` | LONG | running |
| 4 | Golden / death cross + EMA + Stoch | `research/25_fno_golden_cross/` | LONG+SHORT | running |
| 5 | ADX trend strength | `research/26_fno_adx_trend/` | LONG+SHORT | running |
| 6 | Pullback in trend (RSI<40 in uptrend) | `research/27_fno_pullback_trend/` | LONG+SHORT | running |

PEAD intentionally skipped — earnings data lift too heavy for this session.

Pass criteria per variant: OOS PF ≥ 1.20, Sharpe ≥ 0.8, MaxDD ≤ 30%.

### Spread structure design (locked from `services/spread_structure.py`)

For LONG signals — **Bull Call Spread**:
- Long ATM call (strike = round(spot))
- Short OTM call at +25% strike (matches research/17-21 +25% target rule)
- Theoretical economics via Black-Scholes (30% IV default, 30 DTE)
- Sanity check on Rs 1000 stock: long 1000 CE + short 1250 CE, debit ~Rs 36,
  max profit Rs 213, R:R = 5.84:1, breakeven Rs 1037 (3.7% above spot)

For SHORT signals — **Bear Put Spread** (mirrored).

## Adjustment playbook (already coded in `services/spread_structure.py`)

| Situation | Adjustment | When |
|---|---|---|
| Stock stalls between breakeven and target | Roll short leg up | RSI < 50 in expected uptrend |
| Stock approaches target before expiry | Convert to butterfly | Spot at 80%+ of way to short strike, 14+ DTE left |
| Stock breaks down below long strike | Convert to back spread | Spot reaches breakeven adverse, trend reversal flag |
| Volatility spikes mid-trade | Roll out one cycle | IV percentile > 75 pre-event |
| Partial profit + reversal threat | Sell short put/call below long strike | 50%+ max profit captured, signal weakening |

## Status (live)

| Phase | Status | Output |
|---|---|---|
| Spread generator + sanity check | ✅ done | `services/spread_structure.py` |
| DB schema migration (direction + spread_structure cols) | ✅ done | `services/eod_breakout_db.py` |
| Scanner imports spread module | ✅ done | `services/eod_breakout_scanner.py` |
| Scanner persists spread JSON on signals | ✅ done | scan_eod() builds + writes JSON |
| API surfaces spread JSON | ✅ done | `/api/eod/<sys>/signals` returns spread_structure |
| Frontend renders spread + adjustment playbook | ✅ done | `EodBreakout.tsx` (commit `1d31d79`) |
| 6-signal optimization (agent) | ✅ done | `research/22*` through `research/27*` |
| Deploy spread layer to VPS | ✅ done | service restarted 2026-04-26 |
| Wire winning signals into scanner | pending | next code change |

## Final agent findings — walk-forward results (OOS = 2023-01-01 to 2025-12-31)

Pass criteria: PF ≥ 1.20 AND Sharpe ≥ 0.8 AND MaxDD ≤ 30% on OOS.

| Family | Best variant | Dir | OOS PF | Sharpe | MaxDD | CAGR | Trades | Verdict |
|---|---|---|---|---|---|---|---|---|
| 22 Bearish breakdown | baseline | SHORT | 0.69 | -0.41 | 7.4% | -1.35% | 32 | **DOA** (all 5 fail) |
| 23 Donchian 55 | **55_vol_3x_atr_floor** | LONG | **1.97** | **1.25** | **7.74%** | **+13.47%** | 175 | **PASS — top pick** |
| 24 RS leadership | rs_60d_z2.0 | LONG | 1.53 | 0.99 | 9.35% | +11.41% | 217 | PASS |
| 25 Golden cross | gc_ema_20_50 | LONG | 1.46 | 0.90 | 24.16% | +11.03% | 251 | PASS |
| 25 Golden cross (alt) | gc_sma_50_200_adx | LONG | 1.82 | 0.81 | 6.02% | +4.41% | 78 | PASS but thin/low CAGR |
| 25 Death cross | all 6 | SHORT | 0.31-0.76 | <0 | 15-63% | neg | **DOA** |
| 26 ADX trend | **adx_30** | LONG | **1.77** | **1.30** | 16.47% | **+15.50%** | 225 | **PASS — highest CAGR** |
| 26 ADX (other) | adx_25_pure | LONG | 1.22 | 0.44 | 15.08% | +4.70% | 260 | overfit (IS 2.57 → OOS 1.22) |
| 26 ADX trend | all SHORT | SHORT | 0.60-0.67 | <-0.5 | 50-67% | neg | **DOA** |
| 27 Pullback | **pb_rsi40** | LONG | **1.55** | **1.01** | 18.19% | **+12.99%** | 233 | **PASS — non-correlated** |
| 27 Pullback (SHORT) | all 5 | SHORT | 0.45-0.78 | <0 | 11-47% | neg | **DOA** |

### Hard verdicts
- **24 of 24 SHORT-direction variants failed.** Indian F&O 76 universe is structurally long-biased; bear-put spreads from these signals lose money. Drop SHORT side.
- **All 5 LONG winners are trend-continuation breakouts** that overlap heavily with research/21's 252-day Donchian (already live). True correlation analysis (per-day signal overlap) is the next step before deploying all five.
- **Confluence filters underperformed** — minimal-filter variants tended to win OOS. The "filters help" hypothesis didn't hold.
- OOS regime (2023-25) was a strong Indian bull market — bear-regime stress (2008-10, 2020-Q1) is the missing test.

### Recommended deploy ranking

1. **Donchian 55 `55_vol_3x_atr_floor`** — best risk-adjusted (PF 1.97, MaxDD 7.74%). Tighter than research/21 (which uses 252-day breakout).
2. **ADX `adx_30`** — highest OOS CAGR (15.50%), Sharpe 1.30. Strict ADX>30 — not the same trade as Donchian since it requires established trend strength, not a fresh breakout.
3. **Pullback `pb_rsi40`** — most non-correlated entry style (buy-the-dip vs breakout). Lower Sharpe but adds diversity.

RS leadership `rs_60d_z2.0` and golden cross `gc_ema_20_50` are deployable but second-tier — RS will likely co-fire with Donchian; cross-system has 24% MaxDD which is on the edge of the 30% gate.

### Per-signal artifacts

For each `research/<NN>/results/` folder:
- `summary.csv` — full sweep (5-12 variants, IS metrics)
- `walk_forward.csv` — top 3 variants run train 2018-2022 / test 2023-2025
- `equity_<variant>.csv` — daily equity curves
- `trades_<best>.csv` — full trade log for the winner
- `universe.csv` — F&O symbol list used
- `SIGNAL-STATUS.md` — verdict + tradable variant (verdicts in Status section since FINDINGS.md was harness-blocked)

## Crash recovery — for the human

### If Claude crashes mid-build

The `services/spread_structure.py` module is standalone — runs `python services/spread_structure.py` to verify economics.

Scanner integration is partial — search for `spread_structure` references in `services/eod_breakout_scanner.py` to see how far it got.

### If the agent crashes mid-sweep

Each research folder has its own `SIGNAL-STATUS.md`. Check:
```bash
for d in research/22* research/23* research/24* research/25* research/26* research/27*; do
  echo "=== $d ==="
  tail -3 "$d/SIGNAL-STATUS.md" 2>/dev/null
  ls "$d/results/" 2>/dev/null
done
```

Each research folder is independent — re-run any one with:
```bash
python research/<NN>_<name>/scripts/run_<name>_sweep.py
python research/<NN>_<name>/scripts/walk_forward_<name>.py
```

### How to test the spread generator standalone

```bash
python services/spread_structure.py
```

Outputs theoretical bull call + bear put structures for a Rs 1000 stock.

### How to manually fire a scan + verify spread structures persist

```bash
curl -X POST http://localhost:5000/api/eod/scan
# After it returns:
sqlite3 backtest_data/eod_breakout.db \
  "SELECT signal_date, symbol, direction, spread_structure FROM eod_signals ORDER BY id DESC LIMIT 5;"
```

### Final aggregation

When agent completes:
1. Read each `research/<NN>/results/walk_forward.csv` for the best variant per family
2. Build a comparison table: 6 signals × OOS metrics
3. Pick top 2-3 to add to live as additional `system_id`s in `eod_breakout_scanner.py`
4. Spread structure already plumbed — they get spreads "for free"
5. Update FuturePlans status entries
6. Commit + push

## Files NOT to touch

- `research/17_eod_breakout_scan/`, `research/19_smallcap_daily/`, `research/21_eod_fno/` — validated, frozen
- `services/orb_*` — live ORB code path, untouched by this work
- `services/nas_*` — live NAS code path, untouched
