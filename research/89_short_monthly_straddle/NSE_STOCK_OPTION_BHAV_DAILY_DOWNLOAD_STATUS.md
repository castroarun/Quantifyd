# NSE F&O Stock+Index Option EOD History Download — 2016→now into `nse_options_bhav`

**STATUS: RUNNING (background on VPS)** · Started 2026-07-22 ~10:30 IST

## The Ask
User: "There's no recorded stock option IV history anywhere — please download it into our
database alongside the existing stock data." Goal: real EOD option prices for the F&O stock
universe (+ indices) so stock straddle premiums / IV are REAL, not modeled — enabling a
decision-grade re-run of the research/89 flip hypothesis on stocks.

## What it does
- Source: **NSE F&O bhavcopy archives** (free, no Kite). Coverage probe confirmed HTTP 200 for
  every year 2016→2026 from the VPS, both URL formats, stock options present.
- Extends the production `download_nse_bhav.py` machinery to **81 F&O stocks** (read live from
  `services/data_manager.py::FNO_LOT_SIZES`) + NIFTY/BANKNIFTY; parses OPTSTK/STO in addition to
  OPTIDX/IDO.
- Keeps only a **near-ATM band** (|strike/spot−1| ≤ 0.25) and **expiries ≤ 75 days** to stay
  tractable (~12k rows/day). Spot = UndrlygPric (UDiFF) / daily close / max-OI proxy.
- Writes into the SAME `nse_options_bhav` table in `market_data.db` (INSERT OR IGNORE, dedup).
  Resume-safe: skips dates that already have stock rows.
- IV is NOT stored here — it is computed at analysis time by inverting Black-Scholes on the EOD
  close vs the underlying daily close (repo `implied_vol()` bisection helper).

## Scope
- Range: 2016-01-01 → yesterday. ~2,500 trading days. Est. runtime ~2–2.5 h at 2s rate limit.
- Est. size: ~12k rows/day × 2.5k days ≈ **~30M rows (~3 GB)** added to `market_data.db`.

## Status log
| Time (IST) | Event | Notes |
|---|---|---|
| 2026-07-22 10:24 | Test run 9 days OK | ~12k rows/day, 80 syms, ATM filter + insert verified; table → 1.51M rows |
| 2026-07-22 10:26 | Full backfill launched (VPS, PID 1574770) | 2744 days todo; 2016 has ~42 syms/3.6k rows/day → grows to 80 syms/12k rows/day by 2026 |
| 2026-07-22 10:29 | Real-IV pipeline validated on INDEX | `run_g4_realiv_flip.py` on NIFTY/BANKNIFTY: 56 real monthly trades; real ATM VRP ratio 1.15; after-spike iron fly +73bps (t1.3, n small). Stock leg pending backfill. |
| — | Background poll watching for completion | ETA ~13:00 IST; on finish → run `run_g4_realiv_flip.py` (all symbols) for the decisive stock test |

## Crash Recovery (resume without Claude)
- Runner: `research/89_short_monthly_straddle/scripts/download_nse_bhav_stocks.py` (on VPS).
- Progress log: `results/NSE_BHAV_STOCK_DOWNLOAD.log` — `tail -f` it.
- Check alive: `ps aux | grep download_nse_bhav_stocks | grep -v grep`.
- **Resume (idempotent):** just re-run
  `cd /home/arun/quantifyd/research/89_short_monthly_straddle && nohup venv/bin/python3 scripts/download_nse_bhav_stocks.py > results/nse_dl_stdout.log 2>&1 &`
  It skips dates already downloaded and continues. Safe to run repeatedly.
- Writes only to `nse_options_bhav` (INSERT OR IGNORE) — never touches `market_data_unified`
  or the live `quantifyd` service. No market-hours risk (read-only NSE fetch).

## Output
| File | Purpose |
|---|---|
| `nse_options_bhav` table (market_data.db) | Real EOD stock+index option OHLC/OI/settle |
| `results/NSE_BHAV_STOCK_DOWNLOAD.log` | Live progress + heartbeat |
| `scripts/download_nse_bhav_stocks.py` | The downloader |
| `scripts/probe_nse_coverage.py` | Year-by-year coverage probe (done) |
