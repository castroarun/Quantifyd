# NAS Session Handoff — 2026-07-30 (SENSEX exit-stack + related)

Context carried from a Holdings/chart-wall session. Everything below is on the VPS
(`arun@94.136.185.54:/home/arun/quantifyd`). Market was OPEN when this was written
(Thu 30-Jul, SENSEX expiry, DTE0) — so nothing new was deployed; several items are staged/queued for after 15:30.

---

## A. THE HEADLINE ISSUE — SENSEX exit stack is entirely NIFTY-borrowed

**User's concern (valid):** on DTE0, SENSEX ATM2's **±0.4% move-stop** sits ~286 pts away = essentially at the straddle break-evens. By the time spot travels there, the losing leg is deep ITM (delta≈1, unbounded) while the winning leg's premium is already ~fully decayed (gain capped) → the "management point" is too wide for expiry gamma. Matches research/96's "expiry-gamma trap" finding.

**Ground truth from config (verified 07-30):**
- `SENSEX_ATM2_DEFAULTS` (config.py ~L554): `move_stop_pct=0.004`, `rupee_stop_per_lot=0`, `move_stop_reenter=True`. The move-stop overrides are at **L560–562**.
- This override is **deliberate** — a comment at L556–559 says it was added **2026-07-29 as a guardian HIGH finding**: the ₹2,500/lot rupee stop was *calibrated on NIFTY chains ONLY, never approved for SENSEX*. (NIFTY lot 75 vs SENSEX lot 20 ⇒ same ₹/lot = ~33-pt vs ~125-pt premium tolerance — not transferable.)
- **CORRECTION to note:** earlier in the session I told the user "SENSEX ATM2 uses the ₹5k rupee stop" — that was WRONG. It uses the move-stop. The per-leg 30% SL (`leg_sl_pct=0.3`) is present but INERT (executor guard disables per-leg SL when move-stop is active).

**Full SENSEX exit stack today (all NIFTY-borrowed / provisional):**

*Per-system* (config.py):
| System | Stop | On stop | Target |
|---|---|---|---|
| SENSEX ATM | 30% per-leg | trail survivor to cost, re-enter ≤5 | none |
| SENSEX ATM2 | ±0.4% move-stop (~±310 pts) | close both, re-center | none |
| SENSEX ATM4 | 30% per-leg | roll once (max_rolls=1) then trail | none |
All: entry 09:16 (till 14:50), EOD square-off **15:15**.

*Book-level* (`services/nas_portfolio_stop.py`, checked every 10s, per-venue, sums the 3 systems' live day-P&L):
- `STOP_PER_LOT = 1300` → combined daily **STOP = −₹1,300 × lots deployed** (flatten whole venue).
- SENSEX `tp_per_lot = 1667` → combined daily **TAKE-PROFIT = +₹1,667/lot fixed** (SENSEX fades into close → banks it). **No trail** on SENSEX.
- NIFTY differs: **no fixed TP**, uses a **trailing lock** (arm +₹2,000/lot, giveback ₹350/lot).
- Docstring admits: *"the −1,300/lot is NIFTY-calibrated (SENSEX has ~2 live days); a SENSEX-chain calibration can refine it later."* The +1,667 TP (research/91) also rested on ~2 SENSEX days.

**Decision taken:** DO NOT blind-deploy NIFTY numbers to SENSEX. **Calibrate on SENSEX data first, then deploy after sign-off.**

---

## B. QUEUED STUDY — research/97 (calibrate the whole SENSEX exit stack)

Folder on VPS: `research/97_sensex_atm2_stop_dte0/` — STATUS-MD written:
`SENSEX_ATM2_STOP_DTE0_CALIB_SWEEP_STATUS.md` (status = QUEUED).

**Data confirmed feasible:** `backtest_data/options_data.db` → `option_chain WHERE symbol='SENSEX'` = **8.28M rows, 70 trading days (Apr20→Jul30), every minute 09:15→15:30** (≈14 weekly expiry cycles, real DTE0/DTE1). The executor docstring's "backfill hasn't run" warning is STALE.

**Two layers to sweep (net-of-cost, bid-ask slippage, traded-volume/OI liquidity filter per research/89, DTE0-vs-DTE1 split):**
- **Layer A — per-system stop (~78 cells):** ATM2 ±0.4% move vs rupee ₹/lot sweep {1000..5000} vs net-%; ATM/ATM4 30% SL vs alternatives.
- **Layer B — book-level (~84 cells):** combined daily STOP {off,800,1000,1300*,1600,2000} × TP {none, fixed 1000/1667*/2500, trail arm/gb 1500-300 / 2000-350 / 2500-500}, run on top of the best Layer-A stop. (* = current live control.)

**Output feeds:** validated SENSEX numbers for BOTH `SENSEX_ATM2_DEFAULTS` (per-system stop) AND `services/nas_portfolio_stop.py` (`STOP_PER_LOT`, `tp_per_lot`, fixed-vs-trail). **Nothing deploys without user approval.**

**To run:** after 15:30 (backtests compete with the live per-minute crons). A one-shot reminder cron `c87455f2` was set for 15:35 IST **but it is session-only and dies when this session closes** → in the NAS session, just run it per the STATUS-MD after close.

---

## C. COUNTERFACTUAL RECORDING (user's ask: keep recording till 15:15 after TP/SL books out)

**Goal:** after the book hits +₹10k TP (or the stop) and flattens, still record what *would* have happened to 15:15, for later study of whether the TP/SL exits are optimal.

**Finding — mostly ALREADY captured, no live shadow job needed:**
- The chain recorder logs **every SENSEX strike's price every minute to 15:30 daily** (verified: 07-24 & 07-29 both span 09:15→15:30, ~384 strikes) → the "held-to-EOD" path of any flattened leg is already in `options_data.db`.
- The portfolio stop flattens via `exit_all_positions(reason)` with `reason ∈ {PORTFOLIO_STOP, PORTFOLIO_TP, PORTFOLIO_TRAIL}` → the booking event *should* be tagged on each leg (exit_price, exit_time, reason) in the positions DB.
- ⚠️ **OPEN / UNVERIFIED:** a scan of all 6 position DBs for `exit_reason LIKE 'PORTFOLIO%'` returned **ZERO rows** — meaning either the portfolio stop/TP has **never actually fired-and-recorded** yet, or `exit_all_positions` does **not** store that exit_reason. **NAS session must verify** the booking legs get tagged; if not, add a tiny **booking-event ledger** (legs + entry + booked fill + spot + time + reason) — deploy after close. That's the only possible new code; the till-EOD path itself is free from the chain recorder.
- **Fold into research/97:** add a "booked-out vs held-to-EOD" reconstruction (join positions-DB portfolio closures ↔ options_data.db to 15:15) — works for past and forward.

---

## D. ALREADY DONE THIS SESSION (deployed + committed to `main`)

- **ATM4 roll bug FIXED** (`services/sensex_scanner.py`): the SENSEX ATM4 roll failed on 07-29 because `build_tradingsymbol` got `pos['expiry_date']` as a **string** but only coerced date/datetime → cache miss → "no contract" on every candidate → boundary exit instead of rolling. Fix coerces `'YYYY-MM-DD'` string → date. Verified, deployed (restart 07-29 evening), committed **9235b8e**. *(Relevant: today is SENSEX expiry — the roll now works.)*
- **git:** committed all outstanding work (**51e1e03**) + the fix; working tree clean except two daily `snapshots_backup_*` dirs (intentionally excluded).
- (Non-NAS, for completeness) Holdings `/app/holdings` **Charts** tab shipped over prior turns: chart wall of all 36 holdings, candlesticks w/ zoom-pan, live intraday "today" candle, name watermark; daily OHLC cron 16:10 IST. All committed.
- (One-off, closed) A manual ATM4-roll **replica** short strangle was placed 07-29 (77500PE/77800CE, 2 lots) and managed by a VPS monitor to a breakeven close at EOD — DONE, flat, no residue.

---

## E. OPEN ITEMS / NEXT ACTIONS (for the NAS session)

1. **Run research/97** after close → SENSEX-calibrated numbers (per-system stop + book-level stop/TP/trail).
2. **Deploy validated stops** after the study clears + user sign-off (config.py `SENSEX_ATM2_DEFAULTS`; `nas_portfolio_stop.py` SENSEX venue). Backend → restart → after 15:30 only.
3. **Verify PORTFOLIO_* exit tagging** in positions DBs; add booking-event ledger if missing (for the counterfactual anchor).
4. **30% SL on expiry (ATM/ATM4):** covered by research/97 Layer A.
5. **Today's live SENSEX ATM2:** if still open, manage manually to ~−₹5k net; don't ride the 286-pt move-stop into DTE0 gamma. (Combined guardrails active: −₹1,300/lot stop, +₹1,667/lot fixed TP.)

## Key files
- `config.py` — `SENSEX_ATM2_DEFAULTS` ~L554 (move-stop override L560-562); `NAS_ATM2_DEFAULTS` ~L489-492 (NIFTY rupee stop, the calibrated one).
- `services/nas_portfolio_stop.py` — `STOP_PER_LOT=1300`; VENUES (NIFTY trail arm2000/gb350; SENSEX tp_per_lot=1667); `compute_venue` / `check_and_apply`.
- `services/sensex_scanner.py` — `build_tradingsymbol` (roll fix).
- `services/sensex_executors.py` — SensexAtm/Atm2/Atm4 (rules inherited from NIFTY, venue overridden).
- `backtest_data/options_data.db` — `option_chain` (SENSEX 8.28M rows, 09:15→15:30 daily).
- `research/97_sensex_atm2_stop_dte0/SENSEX_ATM2_STOP_DTE0_CALIB_SWEEP_STATUS.md` — the study design.
