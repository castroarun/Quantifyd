# SENSEX ATM2 Expiry-Day Exit Mechanic — 0.4% Move-Stop vs ₹-Stop vs Hold, Modeled Straddle on 1-min

**STATUS: DONE** — G2 diagnostic. Trap CONFIRMED; ₹2,500 rupee-stop fix REFUTED; keep move-stop. Research/103.

> ⚠️ **MODELED STUDY (low confidence by construction).** We have NO real SENSEX (BSE/BFO) option
> intraday history — only the SENSEX **index** 1-min. The straddle premium path is therefore
> **Black-Scholes-modeled** off the index path + an assumed IV. This can **flag or disprove** the
> expiry-gamma trap (a spot-path-driven, gamma-driven phenomenon the model captures well) but it
> **cannot claim an absolute rupee edge** and is not a STRATEGY result. Gate ceiling: G2 diagnostic.

---

## 1. The Ask (restated precisely)

**What Arun asked:** fast-follow #2 — review whether SENSEX ATM2's 0.4% underlying move-stop is a
bad exit on 0-DTE (Thursday expiry), where premiums are tiny and gamma huge (the research/96
expiry-gamma concern, which was fixed for NIFTY via a ₹2,500/lot rupee stop but left on SENSEX).

**What we're actually testing:** For an ATM SENSEX short straddle entered 09:16 and squared by
15:25, across **all available expiry-day types (DTE0 Thu, DTE1 Wed)**, does the **0.4% move-stop**
crystallise a materially **larger rupee loss-when-triggered on DTE0** than a **DTE-agnostic
₹2,500/lot rupee MTM stop** — and does either beat **plain EOD-hold** — on the MODELED straddle?
Single success metric: **mean net P&L per lot per expiry-day, by DTE**, plus the **loss-when-stopped
distribution by DTE** (the direct test of the gamma trap). Gate: this is a **G2 diagnostic** — it
informs a live-param decision but no live change ships off this alone (playbook: don't overfit a
small sample; robustness before "it works").

## 2. Economic hypothesis (G0)

A short ATM straddle is short gamma. Near expiry, **gamma per point of spot explodes** while the
premium collected shrinks. A **fixed % move-stop (0.4%)** exits on a fixed *spot distance*
(≈0.004×78,700 ≈ 315 pts) **regardless of DTE**, so on DTE0 that same 315-pt move corresponds to a
much larger *rupee* loss than on DTE3+ (loss ≈ ½·Γ·Δspot², and Γ↑ sharply into expiry). A
**rupee/MTM stop** caps the *rupee* loss at a constant (₹2,500/lot) whatever the DTE — so it should
dominate the move-stop specifically on DTE0. Counterparty: expiry-day directional/pinning flow.
Decay risk: none — it's a mechanical exit-rule comparison, not an alpha claim.

**Falsification:** if the move-stop's DTE0 loss-when-triggered distribution is **not** materially
worse than the rupee-stop's (similar mean/95th pct loss), the expiry-gamma concern is **unfounded**
and ATM2 stays as-is. If EOD-hold beats both on mean net, stops are net-harmful (keep-simple).

## 3. The Base (mechanics — locked)

- **Underlying data:** `market_data.db` symbol=SENSEX timeframe=minute (1-min index OHLC).
- **Straddle:** SELL ATM CE + ATM PE at 09:16 close (strike = round(spot/100)*100). Qty = 1 lot
  (20) for per-lot reporting; scale is linear.
- **Premium model:** Black-Scholes reprice each 1-min bar from the index close, strike, r=0.065,
  **IV assumed** (base 12% annualised; sensitivity 10/12/15/18% — a key robustness axis since we
  have no real IV). Entry premium = BS at 09:16; each bar MTM = entry − current BS (short).
- **DTE:** days to Thursday weekly expiry (Thu=0, Wed=1, Tue=2 …). SENSEX weekly expiry = Thursday.
- **Exit variants (the grid):** see §4. All also force EOD square-off at 15:25.
- **Cost:** ₹40/order × 4 legs = ₹160/round-trip/lot; slippage 0.5 premium pt/leg (modeled, since
  BSE bid/ask is noisy — research/97 lesson). Report **gross AND net**.
- **Success metric:** mean **net P&L per lot**, split by DTE bucket; plus loss-when-stopped
  distribution (mean, 95th pct) by DTE — the direct gamma-trap test.

## 4. Plan (grid + cell count)

Exit rules × IV assumptions, per expiry-day, over the full backfilled sample:

- **Exit rule (6):** `HOLD_EOD` · `MOVE_0.4%` (current) · `MOVE_0.6%` · `RUPEE_2500/lot` ·
  `RUPEE_3500/lot` · `MOVE_0.4%_REENTER` (re-center once, as live ATM2 does).
- **IV assumption (4):** 10% · 12% · 15% · 18% annualised.
- **DTE buckets:** report each rule/IV split by DTE 0 / 1 / 2+ (the study's whole point is the DTE0
  row vs the rest).

Cells = 6 × 4 = **24 configs**, each evaluated over every trading day in the backfill (~1,100 days
2021→now → ~230 DTE0 Thursdays, ~230 DTE1 Wednesdays). Held fixed: entry time 09:16, 1-lot sizing,
r=0.065, cost model. Deliberately skipped: multi-lot (linear), real-IV surface (unavailable).

## 5. Status (live log)

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-08-06 ~11:20 | Study framed, STATUS written | Feasibility gate: SENSEX 1-min was only 15 days → backfill required first |
| 2026-08-06 ~11:25 | SENSEX 1-min backfill launched + DONE | 2021-01-01→2026-08-06, 504k rows, 1,353 days |
| 2026-08-06 ~11:35 | Sweep ran (1,353 days × 6 rules × 4 IV) | results/exit_sweep.csv written |
| 2026-08-06 ~11:40 | DONE — RESULTS.md written | Trap CONFIRMED (3.5× DTE0); rupee-stop fix REFUTED; keep move-stop |

## 6. Crash recovery

- **Backfill:** `tail /tmp/sensex_backfill.log`; resumable+idempotent — just re-run
  `./venv/bin/python3 scripts/dl_sensex_1min.py --backfill` (skips days already present).
- **Check data:** `SELECT COUNT(*),MIN(date),MAX(date) FROM market_data_unified WHERE symbol='SENSEX' AND timeframe='minute'`.
- **Sweep:** rerun `research/103_sensex_dte0_exit_mechanic/scripts/run_dte0_exit_sweep.py`
  (reads market_data.db, writes results/exit_sweep.csv incrementally; safe to re-run).
- Do NOT touch `market_data.db` writes from the laptop (VPS-canonical rule).

## 7. Files

| File | Purpose | Committable? |
|---|---|---|
| `SENSEX_ATM2_DTE0_EXIT_1MIN_SWEEP_STATUS.md` | This live-status doc | yes |
| `scripts/run_dte0_exit_sweep.py` | Modeled-straddle exit sweep runner | yes |
| `results/exit_sweep.csv` | Per-config aggregate (small) | yes |
| `results/RESULTS.md` | Final verdict | yes |

## 8. Findings

See `results/RESULTS.md` for the full write-up. Headline (all IV-invariant; absolute P&L unreliable
due to modeled IV):

1. **Gamma trap CONFIRMED** — the 0.4% move-stop's mean loss-when-triggered is **₹2,481/lot on DTE0
   vs ₹708 on DTE2+ (~3.5×, IV12%)**, fires on 82% of DTE0s. Same shape at every IV.
2. **₹2,500 rupee-stop fix REFUTED** — ties the move-stop on DTE0 (no help) and is ~2× worse on
   DTE2+ (lets normal-day losses run to ₹2,700 vs the move-stop's ₹708). Porting research/96's NIFTY
   fix to SENSEX would be net-negative.
3. **The 0.4% move-stop is the least-bad rule on every DTE**; HOLD_EOD is worst. Tighter/faster
   beats looser/hold.
4. **Decision: keep ATM2's move-stop as-is (Option A).** The real DTE0 lever is sizing/participation
   (DTE0 losses dwarf DTE1/2+ under every rule), which needs REAL BFO option intraday to study
   properly — we only have modeled premium and daily chain snapshots today.
