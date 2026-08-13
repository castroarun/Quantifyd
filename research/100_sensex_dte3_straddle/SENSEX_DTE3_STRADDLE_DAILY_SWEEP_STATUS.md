# SENSEX DTE-3 Short Straddle — Multi-Year Study on Real BSE Bhavcopy (2024→now)

STATUS: RUNNING (Phase 1 — BSE bhavcopy download)

## The Ask

**What Arun asked:** After the NIFTY DTE-3 straddle study concluded (robust short-DTE
edge, best variant DTE-3 open-entry/hold-to-DTE-1, 3% next-week wings), do the *same*
exercise for SENSEX — "our funds will remain not utilized for the other days and sensex
expiry is different day." First a quick recorder look (done, b), now the proper
multi-year study (a): "lets move on to (a)… download it [the BSE data + India VIX]."

**What we're actually testing:** On years of REAL SENSEX EOD option premiums, does the
DTE-3 short straddle (enter ~3 calendar days before the SENSEX weekly expiry at the
option OPEN, hold to DTE-1, exit at CLOSE) show the same positive, theta-harvesting edge
NIFTY did — net of slippage, OI-filtered, VIX-gated, and stable across years? And does
its Monday→Wednesday cycle complement NIFTY's Friday→Monday cycle so combined capital is
utilized ~4 of 5 weekdays?

## The Base — what's being tested

- **Signal:** sell ATM SENSEX straddle (CE+PE at nearest strike to entry-day underlying).
- **Entry:** DTE-3 = nearest trade day on/before (weekly expiry − 3 days). SENSEX expiry
  is Thursday → entry ≈ Monday. Enter at the option daily OPEN (morning proxy).
- **Exit:** DTE-1 (≈ Wednesday), at the option daily CLOSE.
- **Wings (fly variant):** long ±3% strikes in the NEXT-week expiry (fallback same-week),
  matching the NIFTY optimum. BANKEX kept as a bonus symbol.
- **Costs:** 0.3% slippage on every leg + flat ₹160/round-trip; OI filter ATM≥ (scaled),
  wings≥ (scaled) — real traded contracts only (research/89 binding rule).
- **VIX gate:** India VIX 13–28 (NIFTY-implied, ~0.95 corr to SENSEX vol) — report BOTH
  gated and ungated since 2024–26 is a low-VIX regime.
- **Universe/period:** SENSEX weekly options, 2024-01 → today (~130 weeklies). 2023 (BSE
  relaunch) needs the legacy pre-UDiFF format — deferred unless results warrant.
- **Success criterion:** positive net mean/trade with M/SD comparable to NIFTY, ≥ most
  years positive, monotone-ish DTE, survives the VIX gate.

## Plan

1. **Download** BSE F&O UDiFF bhavcopy 2024-01→today → `bse_options_bhav` (SENSEX+BANKEX).
2. **Refresh INDIAVIX** (stale at 2026-06-12) → current, via Kite.
3. **DTE sweep** (DTE 1..5) naked + 3% fly, VIX gated + ungated, per-year table.
4. **Compare vs NIFTY**; write RESULTS.md verdict; publish to /app if STRATEGY-grade.

## Status

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-08-05 ~ | Feasibility confirmed | UDiFF URL works VPS-side 2024-01-02→2026-07-31; SENSEX IDO rows present w/ OHLC+UndrlygPric+OI+lot |
| 2026-08-05 ~ | Download launched | bse_bhav_downloader.py detached on VPS, logging to results/download.log |

## Crash Recovery

- **Download progress:** `tail -f research/100_sensex_dte3_straddle/results/download.log`
  or `sqlite3 backtest_data/market_data.db "SELECT symbol,COUNT(*),MIN(trade_date),MAX(trade_date) FROM bse_options_bhav GROUP BY symbol"`.
- **Resume download:** re-run `python3 research/100_sensex_dte3_straddle/scripts/bse_bhav_downloader.py`
  — it skips trade_dates already loaded. Safe to re-run anytime.
- **Alive check:** `ps aux | grep bse_bhav_downloader`.
- **Do NOT** drop `bse_options_bhav` mid-run. Safe to inspect it read-only.

## Files

| File | Purpose | Committable? |
|---|---|---|
| `scripts/bse_bhav_downloader.py` | BSE UDiFF downloader → bse_options_bhav | yes |
| `scripts/sensex_study.py` | DTE sweep naked+fly (written Phase 3) | yes |
| `results/download.log` | Download progress | yes (small) |
| `results/RESULTS.md` | Final verdict | yes |
| `bse_options_bhav` (table in market_data.db) | Raw premiums | n/a (in DB) |

## Findings

_(pending download + study)_
