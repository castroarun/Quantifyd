# NAS 8-System 28-Day Live/Paper Performance Review — Real Recorded Trades

STATUS: DONE (phase 1) — phase 2 (chain re-price) in progress

> **KEY FINDING:** the OTM trade recorder books exit premium = ₹0 → Squeeze OTM
> (+₹115k) and 916 OTM (+₹204k) show fake 100% win / ₹0 drawdown and inflate the
> combined headline (~₹319k of the ₹592k is phantom). ATM systems (real exits) are
> trustworthy: 916 ATM2 +137k, 916 ATM +68k, 916 ATM4 +65k positive; Squeeze ATM/ATM2
> slightly negative. → Phase 2 re-prices actual legs against recorded option_chain to
> get true P&L for all 8.

## 1. The Ask

**What you asked:** "since u have 28 days of good nifty options data, can u run all
our NAS systems across them on all days and come up with a comprehensive report with
visual P/L curves per system, groupings per day per system, drawdown curves etc."

**What we're actually producing:** A comprehensive performance report of the **actual
recorded trades** of all 8 NAS systems over the window where we have data
(2026-04-20 → 2026-06-02). These are REAL fills (live/paper at live quotes), not
synthetic/BS premiums and not a replay — so it is a faithful audit of what the
systems did on real markets, with per-system equity curves, per-day×system P/L grid,
and drawdown curves.

## 2. The Base — what's measured

- **Source:** the 8 `backtest_data/nas*trading.db` trade tables (`nas_trades` /
  `nas_atm_trades`), `net_pnl` (gross − brokerage) per completed strangle, since
  2026-04-20.
- **Systems:** Squeeze OTM, ATM, ATM2, ATM4 + 9:16 OTM, ATM, ATM2, ATM4.
- **Metrics:** total/avg net P/L, per-lot-normalized P/L, win rate, max drawdown,
  best/worst day, trade & active-day counts, exit-reason mix.

## 3. Caveats (honesty first — this is NOT strategy validation)

- **Tiny sample:** ~28 trading days, single regime. Signal/behavior audit only.
- **Lot-size change:** early trades ran lots=10, later dropped to lots=2 → raw
  cumulative curves are NOT apples-to-apples over time. Report shows BOTH raw and
  **per-lot-normalized**.
- **Mixed paper/live + messy tail:** restarts, today's (2026-06-02) exit/adjustment
  bugs, ghost legs and MANUAL_FLATTEN_RECONCILE trades distort the last day; flagged.
- A clean normalized **replay on recorded option_chain** is a separate phase-2 build.

## 4. Plan

1. Read 8 trade tables → combined frame (this file written first, per playbook).
2. Generate factsheet PNG: KPI strip, combined equity+drawdown, per-system cumulative
   (raw + per-lot), per-day×system heatmap, per-system stats table.
3. Write RESULTS.md with the honest verdict + caveats.

## 5. Status (live log)

| Date/time | Event | Notes |
|---|---|---|
| 2026-06-02 ~15:25 IST | Folder + STATUS written | building generator |

## 6. Files

| File | Purpose | Committable? |
|---|---|---|
| `scripts/generate_report.py` | Report generator | yes |
| `results/nas_28day_review.png` | Factsheet | yes |
| `results/RESULTS.md` | Verdict | yes |
| `results/per_trade.csv` | Flattened trades used | yes (small) |
