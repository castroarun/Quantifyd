# SENSEX Exit-Stack Calibration (per-system stop + book-level stop/TP/trail) — 0/1-DTE Sweep on Real SENSEX Chains

STATUS: **DONE — INCONCLUSIVE (G2): sample too benign to calibrate a stop level; 30% SL bad on expiry; DTE1 looks unprofitable. No deploy. See results/RESULTS.md**

Research #: **97** · folder `research/97_sensex_atm2_stop_dte0/`

---

## 1. The Ask

**What you asked (2026-07-30, SENSEX expiry day):**
> "On an expiry day, having the management point still 286 pts away means the risk is heavier — by the time price reaches there one leg is deep ITM and the other has no premium left to support it. Assess this. Also assess if the 30% SL on expiry is ok. We did the rupee-stop fix for NIFTY yesterday. Calibrate the right SENSEX stop first, then deploy — combined study with the 30% SL."

**What we're actually testing:**
On real recorded SENSEX weekly-option chains (2026-04-20 → 2026-07-30, 70 trading days ≈ 14 expiry cycles), for the SENSEX short-premium systems entered on **DTE1 (Wed)** and **DTE0 (Thu)**, calibrate the **entire SENSEX exit stack on SENSEX data** — two layers:
- **Layer A — per-system stop:** current ±0.4% move-stop (ATM2) vs a **rupee MTM stop swept across ₹/lot** vs the **30% per-leg SL** (ATM/ATM4) vs net-%; rank by net-of-cost per-trade P&L with a tail gate; DTE0 vs DTE1.
- **Layer B — book-level (combined) exits** across all 3 SENSEX systems (the research/90-91 portfolio guardrail): the **combined daily STOP (₹/lot)** and the **daily take-profit** (fixed vs trailing-lock vs none), currently −₹1,300/lot stop + fixed +₹1,667/lot TP — both NIFTY-calibrated. Sweep stop ₹/lot, TP ₹/lot, and trail (arm/giveback ₹/lot); DTE0 vs DTE1.

Goal = a **SENSEX-calibrated** exit stack (per-system stop + combined stop + TP/trail) to replace the guardian-flagged NIFTY-borrowed numbers, so every deploy is validated not assumed.

**Why now:** the *whole* SENSEX exit stack is NIFTY-borrowed/provisional:
- `SENSEX_ATM2_DEFAULTS` reverted to ±0.4% move-stop 2026-07-29 (guardian HIGH: "rupee stop calibrated on NIFTY chains ONLY"). ₹2,500/lot not transferable — NIFTY lot 75 vs SENSEX lot 20 ⇒ same ₹/lot = ~33-pt vs ~125-pt premium tolerance.
- `nas_portfolio_stop.py` docstring: "the −1,300/lot is **NIFTY-calibrated** (SENSEX has ~2 live days); a SENSEX-chain calibration can refine it later." The +₹1,667 TP (research/91) rested on ~2 SENSEX days.
Now we have 70 real SENSEX-chain days — measure SENSEX's own numbers for all of it.

## 2. The Base — what's being tested

- **Data:** `backtest_data/options_data.db` → `option_chain WHERE symbol='SENSEX'` (8.28M rows, minute snapshots, incl. `ltp,bid,ask,oi,volume,iv,delta,underlying_spot`) + `underlying_spot` for SENSEX spot. 70 distinct days.
- **Systems (short ATM straddle at entry):**
  - **ATM2** — short ATM CE+PE (both legs ATM strike). Entry 09:16 (916 semantics) on the live-matrix days.
  - **ATM** — same entry; per-leg 30% SL + naked-survivor ST(7,3) trail.
  - **ATM4** — same entry; 30% SL → roll-to-match (max 1), else boundary exit.
- **Lot/size:** SENSEX lot 20, 2 lots/leg (live size). Report ₹/lot to compare vs NIFTY.
- **Stops tested (grid axis):**
  1. `move0.4` — ±0.4% underlying move-stop + re-center (**current SENSEX ATM2 baseline**).
  2. `rupee_X` — exit BOTH legs when net MTM ≤ −₹X/lot, one-and-done. X ∈ {1000,1500,2000,2500,3000,4000,5000}.
  3. `legSL30` — per-leg exit at entry×1.3 (**current ATM/ATM4**), + their trail/roll.
  4. `netpct_Y` — exit both at net loss = Y% of credit collected. Y ∈ {50,75,100,150}.
- **Exit backstop:** EOD square-off 15:15 (all variants).
- **Period / cohorts:** all 14 cycles; **split DTE0 (Thu) vs DTE1 (Wed)** — calibrate each separately (the user's core point is a DTE0 effect).
- **Success criterion:** rank by **net-of-cost ₹/lot per trade**, gated on (a) worst-trade / max-intraday-DD tail, (b) stability across cycles, (c) DTE0 and DTE1 both acceptable. A SIGNAL (positive per-trade) ≠ a deployable stop until the tail + stability gates pass.

## 3. Guards (seven deadly sins — how each is controlled)

- **Look-ahead:** decisions use only chain data at/ before the decision minute; fills at next-minute bid/ask.
- **Cost neglect:** **net-of-cost mandatory** — model per-leg slippage from the chain **bid/ask spread** (not a flat %), + brokerage/taxes; report gross, net, and a cost-sensitivity (½×, 1×, 2× spread).
- **Capacity / liquidity (research/89 lesson — BINDING):** filter to strikes with **real traded volume/OI** at the snapshot; a roll/exit that would price against a stale far-OTM quote is rejected. No fills on zero-volume strikes.
- **Overfitting / multiple-testing:** the rupee sweep is judged on **monotonicity/plateau**, not peak-pick; a lone peak between neighbours that lose is discarded.
- **Regime dependence:** per-cycle table + DTE0-vs-DTE1 split; flag if the winner rides one or two cycles.
- **Survivorship / single-factor:** all cycles in-window included; both legs modelled (no dropping the losing leg).
- **DTE realism:** DTE0 uses the actual Thu chain (fast theta/gamma); no DTE-agnostic shortcuts.

## 4. Plan — grid + cell count

**Layer A — per-system stop**

| Axis | Values | n |
|---|---|---|
| System | ATM2, ATM, ATM4 | 3 |
| Stop family | move0.4(1) + rupee(7) + legSL30(1) + netpct(4) | 13 |
| DTE | 0 (Thu), 1 (Wed) | 2 |

≈ **3 × 13 × 2 = 78 cells**, each over ~14 expiry cycles. (Some system×stop pairs are the live default and serve as the control.)

**Layer B — book-level (combined) exits** — sum the 3 SENSEX systems' intraday P&L per day, apply a venue exit, measure the venue's net-of-cost daily P&L distribution.

| Axis | Values | n |
|---|---|---|
| Combined daily STOP (₹/lot) | off, 800, 1000, 1300*, 1600, 2000 | 6 |
| Daily TAKE-PROFIT | none, fixed{1000,1667*,2500}, trail{arm 1500/gb 300, arm 2000/gb 350, arm 2500/gb 500} | 7 |
| DTE | 0 (Thu), 1 (Wed) | 2 |

≈ **6 × 7 × 2 = 84 cells** over ~14 days each (*= current live value, the control). Layer B runs **on top of** the Layer-A per-system stop chosen as the leading candidate, so the combined result reflects the full stack.
Total ≈ **78 (A) + 84 (B) ≈ 162 cells.** Rank Layer B by **venue net daily P&L, worst-day tail, and % of days the TP/stop actually improves vs hold-to-EOD** — separately DTE0 vs DTE1 (the user's core hypothesis: SENSEX fades harder on DTE0, so a TP/tighter-stop should help more there).
Deliverable at G4: a client tearsheet per finalist stop (KPI strip, per-cycle bars, DTE0-vs-DTE1, worst-trade) via `research/_utilities/tearsheet.py`.

## 5. Status (live log)

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-07-30 10:4x | STATUS-MD written, study QUEUED | Data confirmed: 8.28M SENSEX chain rows, 70 days |
| — | (pending) runner written | after 15:30 |
| — | (pending) sweep launched on VPS | after 15:30 |
| — | (pending) RESULTS.md + verdict | — |

## 6. Crash Recovery

- This file is the source of truth. To resume without Claude: the runner (once written) lives in `scripts/`, reads `backtest_data/options_data.db` (read-only), writes incremental `results/*.csv`. Re-run `python3 research/97_sensex_atm2_stop_dte0/scripts/<runner>.py` — it skips cells already in the output CSV.
- Do NOT run during 09:15–15:30 IST (competes with live NAS crons). Safe after close.
- The DB is **read-only** for this study — never written.

## 7. Files (output map)

| File | Purpose | Committable? |
|---|---|---|
| `SENSEX_ATM2_STOP_DTE0_CALIB_SWEEP_STATUS.md` | This file | yes |
| `scripts/run_sensex_atm2_stop_sweep.py` | Sweep runner (to write) | yes |
| `results/cells.csv` | Per-cell net/gross/tail/stability | yes (small) |
| `results/*_tearsheet.png` | Finalist factsheets | yes |
| `results/RESULTS.md` | Verdict (NO EDGE / SIGNAL / STRATEGY) + deploy recommendation | yes |

## 8. Findings

- (pending run)
- **Decision it feeds:** the validated SENSEX exit stack, deployed only after it clears —
  - Layer A: the stop value/family for `SENSEX_ATM2_DEFAULTS` (+ a read on the 30% SL for ATM/ATM4),
  - Layer B: the SENSEX per-venue values in `services/nas_portfolio_stop.py` (`STOP_PER_LOT`, `tp_per_lot`, and whether SENSEX should switch from fixed-TP to a trail),
  replacing the NIFTY-borrowed −₹1,300/lot stop, +₹1,667/lot TP, and the ₹2,500/lot per-system value.
