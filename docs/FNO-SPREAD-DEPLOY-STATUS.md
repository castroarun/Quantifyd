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
| Scanner persists spread JSON on signals | RUNNING (in progress) | scan_eod() will need update |
| API surfaces spread JSON | pending | `/api/eod/<sys>/signals` |
| Frontend renders spread + adjustment playbook | pending | `EodBreakout.tsx` |
| 6-signal optimization (agent) | RUNNING | `research/22*` through `research/27*` |
| Aggregate findings + commit | pending | this doc + commit message |

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
