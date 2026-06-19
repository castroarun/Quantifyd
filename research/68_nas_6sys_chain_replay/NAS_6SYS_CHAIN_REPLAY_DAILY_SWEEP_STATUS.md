# NAS 6 ATM Systems — True Chain Replay + Improvement Sweep on 41-Day NIFTY Per-Minute Option Data

STATUS: DONE — verdict SIGNAL (see results/RESULTS.md). Best stack = SmartGate + ST×3.0 +
ATM2 move-stop: +₹130k/41d lots2, Calmar 19.1, holds train+test. Paper-forward before any live change.

---

## 1. The Ask

**What you asked:** "can u run these 6 systems on the options data we have been collecting
in our database (at least 30+ days) and produce a P&L curve report apart from other stats
and also investigate if we can improve the system to its best performance."

**What we're actually testing:** Replay the **6 ATM NAS option-selling systems that go live
Monday** — squeeze family (`nas_atm`, `nas_atm2`, `nas_atm4`) + 9:16 family (`nas_916_atm`,
`nas_916_atm2`, `nas_916_atm4`) — bar-by-bar against the **recorded per-minute NIFTY weekly
option chain** (`backtest_data/options_data.db`, 41 trading days 2026-04-20→06-19), pricing
every fill from real recorded premiums. Produce: per-system + combined **equity (P&L) curve**,
full stats (Net, ₹/day, day-win%, MaxDD, Sharpe, PF, per-DTE), and then **investigate the two
live improvement levers** (research/54 left only these alive): (a) ±0.4% underlying **move-stop**
vs the 1.3× premium SL, (b) **~100pt-OTM strangle** vs ATM, plus **DTE-gating** (the one robust
prior finding). Single success metric: **net ₹/day per system** and **combined Calmar/MaxDD**;
gate to clear = net-positive after ₹160/strangle cost AND a sane DTE/weekday stability.

## 2. Economic hypothesis

Short ATM straddle/strangle harvests **intraday theta + variance risk premium** on NIFTY; the
counterparty is hedgers/long-gamma buyers overpaying for protection. It decays/inverts on
**trend days** (gamma losses) — which is why the 1.3× premium stop and the survivor management
exist. The prior robust finding: **the VRP is concentrated near expiry (1-DTE)** and **bleeds
on far-DTE** (more time for a trend to develop, less theta/day). Improvement thesis: a stop on
the **underlying move** (not premium noise) should bound the trend-day tail better, and selling
**slightly OTM** trades a little theta for a wider break-even.

## 3. The Base (mechanics being replayed)

- **Universe/period:** NIFTY weekly options, front expiry, 41 days (2026-04-20→06-19), per-minute.
- **Entry:** 916 family = first snapshot ≥ 09:16 (exact). Squeeze family = first 5-min bar with
  Wilder-ATR(14) < SMA(ATR,50) in 09:30–14:30 (reconstructed from per-min `underlying_spot`; approx).
- **Strike:** ATM = nearest 50 to spot (offset knob for the OTM lever).
- **Per-leg SL:** entry × 1.30 (premium stop), checked per minute.
- **Management (per variant):** ATM=`SL_ST` (close stopped leg, survivor trails SuperTrend(7,2));
  ATM2=`CASCADE` (any-leg SL → close BOTH, re-enter ≤5×); ATM4=`ROLL_MATCH` (roll stopped leg once
  to premium-matched strike, then survivor trails ST on 2nd SL).
- **Exit:** force squareoff 15:15 (live), residual EOD.
- **Cost:** ₹160/strangle (₹40×2 fills × 2 legs); LTP fills, **no bid/ask slippage** (optimistic).
- **Size:** lots=2 (QTY 130) normalized for cross-system comparison; live 916=1 lot (halve those
  three for the real-money book — reported separately).

## 4. The Plan (grid + cell count)

**Phase A — Baseline (run_baseline.py):** 6 systems × 41 days, as-live, faithfulness-fixed
(15:15 exit, Wilder ATR). Output: legs/per-day/equity/summary CSVs + factsheet PNG. = 6 cells.

**Phase B — Improvement sweep (run_improve.py):** reuse engine with lever axes —
- STOP_MODE ∈ {premium_1.3x, move_0.4pct} (2)
- STRIKE ∈ {ATM, 100-OTM} (2)
- DTE-GATE ∈ {all, 0+1-DTE-only} (2) — applied as a reporting/portfolio filter
× 6 systems = up to 48 cells (cheap; one DB pass each). Rank by net ₹/day + combined MaxDD/Calmar.

**Falsification:** if the baseline 6-book is net-negative after cost on the 0+1-DTE slice AND no
lever flips it positive with stable per-DTE behavior → **NO EDGE / keep paper-only, don't scale.**

---

## 5. Status (live log)

| Date/time | Event | Notes |
|---|---|---|
| 2026-06-19 ~16:0x IST | Folder + STATUS created (sections 1–4) | research/68; engine pulled from research/51 |
| 2026-06-19 ~16:1x IST | Engine + baseline built, smoke-tested | naked-survivor ST path verified on 05-12 |
| 2026-06-19 ~16:3x IST | Baseline DONE (6 systems × 41 days, 6m33s) | combined +₹16.4k lots2 / +₹27.9k live-lots; factsheet written |
| 2026-06-19 ~16:3x IST | Improvement sweep launched (day-outer) | STOP × STRIKE × DTE-gate, 4 configs |

## 6. Crash Recovery

- Engine + runners: `research/68_nas_6sys_chain_replay/scripts/{engine.py,run_baseline.py,run_improve.py}`
- Data (read-only): `backtest_data/options_data.db` (VPS). Outputs: `research/68_.../results/*.csv,*.png`.
- Resume: re-run `./venv/bin/python3 research/68_nas_6sys_chain_replay/scripts/run_baseline.py`
  (idempotent — overwrites outputs). Then `run_improve.py`. No background state; ~minutes per pass.
- Safe to inspect: all results/*.csv. Do NOT touch options_data.db (live recorder writes it).

## 7. Files

| File | Purpose | Committable? |
|---|---|---|
| `scripts/engine.py` | Parametrized replay engine (stop-mode + strike-offset knobs) | yes |
| `scripts/run_baseline.py` | 6-system baseline + factsheet | yes |
| `scripts/run_improve.py` | Lever sweep | yes |
| `results/baseline_legs.csv` | per-leg trades | yes (small) |
| `results/baseline_per_day.csv` / `_equity.csv` / `_summary.csv` | P&L curve + stats | yes |
| `results/nas6_baseline.png` | factsheet | yes |
| `results/RESULTS.md` | verdict | yes |

## 8. Findings

(pending run)
