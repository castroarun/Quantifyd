# Intraday Alternative-Information Systems — Cross-Sectional RS, Flow Proxies, Event Drift (5-min, 150 Names)

STATUS: RUNNING (wave 1)

## 1. The Ask

**What you asked:** pursue the door left open by research/109 — intraday
edges from *information sources other than single-name price/indicator
signals*: event/news reactions, cross-sectional intraday relative strength,
order-flow proxies.

**What we're testing:** With per-day features built from 5-min OHLCV across
150 names (2015→2023), do CROSS-SECTIONAL and FLOW-BASED signals at 10:15
carry net-positive edge to the close — where absolute price signals (r/109)
could not clear the ~10bps cost floor?

**Honesty note:** no true news/order-book data exists in our DB; "event" =
RVOL+gap proxy, "flow" = OHLCV-derived imbalance. Stated as such.

## 2. The Base

- Features per (name, day), all computed by 10:15: overnight gap; morning
  return 09:15→10:15; first-hour RVOL (vs trailing 20-day same-window avg);
  first-hour up-bar volume share (VOLIMB); first-hour mean close-location
  value (CLV). Entry = 10:15 5-min open; exit = 15:15 close. Costs 10bps RT.
- Splits: IS 2015-02→2021-09 · Val 2021-10→2023-12 · **OOS 2024+ untouched.**
- Control: cross-sectional cells are benchmarked against the same-day
  universe mean 10:15→EOD return (market-neutral by construction); absolute
  net also reported. Gates: excess>0, t≥3, ≥55% names+ (where applicable),
  halves same-sign; Val: t≥2, no flip.

## 3. Wave-1 cells (pre-registered, 14)

| Family | Cells |
|---|---|
| XMOM: decile rank of morning return | top-decile L, bottom-decile S, L-S spread |
| XREV: reversed | bottom-decile L, top-decile S |
| EVENT (RVOL≥3 & \|gap\|≥1%) | follow gap dir; fade gap dir (2) |
| VOLIMB first hour | ≥0.65 L, ≤0.35 S |
| CLV first hour | ≥0.7 L, ≤0.3 S |
| RVOL alone ≥3 | with-morning-trend L and S |

Ledger: +14 (program total 456 + 14 = 470).

## 4. Status / event log

| Date/time | Event |
|---|---|
| 2026-08-09 ~22:20 IST | Pre-registered; runner authored; launched — log results/wave1.log |

## 5. Crash recovery

- `tail research/110_intraday_altinfo/results/wave1.log`; features accumulate
  in `results/features.csv` (per-symbol checkpoint `done_syms.txt`); rerun
  the same script to resume; aggregation prints at end of log.

## 6. Files

| File | Purpose |
|---|---|
| `scripts/run_altinfo_wave1.py` | feature build + cross-sectional evaluation |
| `results/features.csv` | per-(name,day) features |
| `results/RESULTS.md` | verdict (after) |
