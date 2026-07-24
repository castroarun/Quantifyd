# Momentum-30 ₹20L book — Go-Live Runbook

`services/momentum_paper.py` · page `/app/momentum-paper` · DB `backtest_data/momentum_paper.db`

The module is **PAPER by default**. Real Kite CNC orders fire **only** when the persisted
`live_mode` setting is `"1"`. Every buy/sell funnels through `_buy`/`_sell`, so flipping that one
flag arms the whole book. Built 2026-07-05 (flag OFF). First possible live action = the end-July
monthly rebalance (~Jul 31, 14:45 IST).

## How the book acts (unchanged from paper)

| Job | Cron (Mon–Fri) | Fires when | Action |
|---|---|---|---|
| `rebalance_job` → `monthly_job` | 14:45 | **last trading day of the month** | Re-rank Nifty-200, rotate to top-8 (buffer-22), **buys** |
| `eod_job` → `weekly_job` | 15:15 | last trading day of the **week** | NIFTYBEES 100-DMA gate; risk-OFF → liquidate all |
| `eod_job` → `daily_job` | 15:15 | every day | Donchian-15 per-stock stop; mark NAV; accrue cash yield |

Gate is currently **risk-ON** (NIFTYBEES > 100-DMA). Positions open only at month-end.

## LIVE vs PAPER rebalance — important difference

- **PAPER**: liquidate the whole book and rebuild to clean equal weight each month (cost modeled).
- **LIVE** (`_rebalance_live_delta`, rotate-only): sell only names leaving the target, buy only
  brand-new names (cash-aware, equal-weight budget), **kept winners ride as-is** (no top-up/trim).
  This avoids paying brokerage + 20% STCG on the whole book every month. Mild equal-weight drift is
  accepted (winners run). Weight top-up/trim is deferred behind `CFG['live_rebalance_trim']`.

## Guardrails (live mode)

- Whole-share (integer) qty for CNC delivery; skips if budget < 1 share.
- Refuses to place orders outside 09:15–15:30 IST (`_market_open_now`).
- Per-order value cap `CFG['live_max_order_value']` (₹15L default).
- Reads back `average_price` as the true fill; logs a **SLIPPAGE** alert if fill deviates > 1%.
- A failed/rejected BUY records nothing; a failed SELL leaves the position held (loud error) — never
  a phantom state.

## Flip to LIVE (do this once, before the end-July rebalance)

1. **Fund the Zerodha account** with the intended live capital.
2. Ensure the Kite token is valid (auto-login runs pre-open; the book calls `get_kite_with_refresh`).
3. Flip the flag **and set live capital** in one call:

   ```bash
   curl -s -X POST http://127.0.0.1:5000/api/momentum-paper/toggle-mode \
     -H 'Content-Type: application/json' -d '{"live": true, "capital": <RUPEES>}'
   # → {"live_mode": true, "mode": "LIVE", "capital": <RUPEES>}
   ```

   `mode` on `/api/momentum-paper/state` will read `LIVE`. No restart needed (flag is read live).

4. After the first live rebalance fills, **reconcile book vs broker**:

   ```bash
   curl -s http://127.0.0.1:5000/api/momentum-paper/reconcile
   # → {"live": true, "match": true, "diffs": []}   ← diffs must be empty
   ```

## Kill switch (back to PAPER, positions untouched)

```bash
curl -s -X POST http://127.0.0.1:5000/api/momentum-paper/kill-switch
# → live_mode OFF. No further real orders. Existing broker holdings are NOT squared off —
#   let the next risk-off gate exit them, or square off manually in Kite.
```

## Watch during a live session

- `journalctl -u quantifyd -f | grep MP-LIVE` — order placed / FILLED / SLIPPAGE / FAILED / MISMATCH.
- `/api/momentum-paper/reconcile` after any fills.
- The book is EOD-only — nothing fires intraday except the 14:45 / 15:15 jobs.

## Files / recovery

| File | Purpose |
|---|---|
| `services/momentum_paper.py` | Book + live layer (`_place_cnc_market`, `_rebalance_live_delta`, `_toggle_mode`, `_kill_switch`, `reconcile_holdings`) |
| `services/momentum_paper.py.bak_live` | Pre-live-layer backup (on VPS) |
| `services/momentum_paper.py.bak_panelcache` | Pre-perf-fix backup (on VPS) |
| `backtest_data/momentum_paper.db` | Book state (`mp_positions`, `mp_closed`, `mp_state`, `mp_nav`, `mp_fills`) |

## Still open before first live trade

- Set the **live capital** amount (user: "different amount" — TBD; pass via toggle).
- Confirm **rotate-only** rebalance policy (v1 default) vs full equal-weight.
- Frontend **LIVE/PAPER badge + toggle control** on `/app/momentum-paper` (build on VPS).
- All backend changes are **uncommitted** on the VPS → include in the next git sweep.
