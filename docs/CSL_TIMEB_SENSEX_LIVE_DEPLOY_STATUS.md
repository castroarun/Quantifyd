# SENSEX Wednesday restructure: TB-SENSEX → REAL @ 8 lots · suite Wednesday → PAPER

STATUS: **EXECUTED 2026-08-18 ~19:5x IST — verify tomorrow 09:12/10:30, then mark DONE** (user
decisions 2026-08-18 evening; SUPERSEDES the earlier 6L-only version of this doc)

> **Laptop:** `c:\Users\arunc\Documents\Projects\Covered_Calls\docs\CSL_TIMEB_SENSEX_LIVE_DEPLOY_STATUS.md`
> **VPS:** `/home/arun/quantifyd/docs/CSL_TIMEB_SENSEX_LIVE_DEPLOY_STATUS.md`

## 1. The decisions (user, 18-AUG)

Per the Wednesday numbers (window +₹1,612/day at 5L is the ONLY earning construction; suite's
per-leg mechanic ≈ −₹137/lot on Wed with a −₹17k/lot p05 tail):

1. **Tomorrow (Wed) only the TB-SENSEX 10:30→12:00 window trades REAL. The SENSEX suite's
   Wednesday goes to PAPER** (Thursday stays real — it's the harvest day).
2. **TB-SENSEX resized 6L → 8 lots (qty 160)** — notional parity with NIFTY TB@8L
   (8×20×~78,000 ≈ ₹1.25Cr vs 8×65×~24,300 ≈ ₹1.26Cr), and goes **REAL** (mode live).

## 2. Changes (three edits, all before 09:12 tomorrow; no gunicorn restart needed)

### A. Day-matrix: suite Wednesday real OFF
`/home/arun/quantifyd/backtest_data/nas_day_matrix.json`: for `sensex_atm`, `sensex_atm2`,
`sensex_atm4` set `"dte": {"1": false}` (keep `"0": true` = Thursday). `gate()` reads at decision
time — effective immediately; paper-shadow continues on Wed automatically (evidence keeps flowing).
Do NOT touch NIFTY rows. Verify via GET `/api/nas/day-matrix`.

### B. Executor: TB-SENSEX 8L + live
`research/111_sensex_manual_mgmt/scripts/csl_paper_exec.py` BOOKS:
`"CSL_TIMEB_SENSEX": {**SENSEX_MKT, "lots": 6, "qty": 120, "cfg_from": "lab"},`
→ `"lots": 8, "qty": 160, "cfg_from": "lab", "mode": "live"` (+ dated comment).
The order layer is venue-generic (BFO seg, marketable-LIMIT ±3%, fill-anchored SLs, per-leg
idempotent exits, naked-leg unwind, gates) — no other code change. py_compile + commit.

### C. Backfill: match sizes
`csl_paper_backfill.py`: same `"lots": 8, "qty": 160` so history regenerates at deployed size
(records carry their own lots; mixed history self-describes).

Also: frontend rules text "CSL_TIMEB_SENSEX (6 lots · qty 120" → "(8 lots · qty 160 — REAL,
notional-parity with NIFTY TB@8L"; npm build (safe anytime).

## 3. ⚠ THE MARGIN DECISION (Thursday, not tomorrow — surface to the user)

Tomorrow (Wed) is trivial: TB-SENSEX 8L alone ≈ ₹13L vs ~₹45L net (NIFTY is ex-Wed; suite Wed
now paper). **Thursday is the squeeze**: SENSEX suite 9L real (~₹15L) + TB-SENSEX full-day 8L
(need ₹17.2L at its 09:20 entry with 1.3× headroom) + NIFTY COMB 2L (~₹3.3L) + NIFTY TB 8L
full-day Thu (~₹13L) → available at TB-SENSEX's entry ≈ 45 − 31 ≈ **₹14L < ₹17.2L need → the
margin gate will likely PAPER-FALLBACK TB-SENSEX on Thursday** (safe, by design, but Thursday
is the venue's harvest day). Options for the user: (a) add/pledge ~₹5–8L before Thursday,
(b) accept Thursday fallback (Wed window real, Thu paper), (c) trim another book Thursday.
Record the user's choice in the Revision Log. Do NOT silently lower the headroom factor.

## 4. Verify (tomorrow)

09:12: `CSL_TIMEB_SENSEX plan [LIVE]: DTE1 10:30->12:00 SL20 qty 160 (8 lots)`; suite logs show
Wed paper-mode orders (matrix gate); 10:30 ENTER [LIVE] with BFO fill prices + REAL popup +
tagged orderbook pair; log first-fill slippage vs LTP (>0.5% of premium → flag band review);
record `"source": "REAL"`; guardian clean. Append all to Revision Log; STATUS → DONE.

## 5. Registry
Ops Center items already dated: 19-AUG first-REAL-window verify · 04-SEP Wednesday review (its
question is now partially ANSWERED BY ACTION — suite Wed to paper; the review validates with
shadow data). Update the review note when marking this DONE.

## 6. Revision Log (append)
| Date/time | Event | Evidence |
|---|---|---|
| 2026-08-18 late eve | Superseding instructions written (8L + Wed-suite-paper) | — |
| 2026-08-18 ~19:5x | A: matrix `sensex_*` dte1→false (dte0 stays true); backup `.bak_tbsx` | verified via matrix json + gate() |
| 2026-08-18 ~19:5x | B: exec `CSL_TIMEB_SENSEX` → lots 8 / qty 160 / mode live (+dated comment); py_compile OK | csl_paper_exec.py:26 |
| 2026-08-18 ~19:5x | C: backfill → 8L/160 | csl_paper_backfill.py:24 |
| 2026-08-18 ~19:5x | D: frontend — Straddles.tsx text, Nas.tsx TimeB-SENSEX card→LIVE 8L + tradebook qty 160, strategies register (TB-SENSEX status live/8L/since 19-Aug + suite-Wed-paper changelog); npm build OK | commit on main |
| 2026-08-18 ~19:5x | **Margin note updated for the 2-lot suite resize done midday**: Thursday suite is now 6L (~₹10L) not 9L (~₹15L) → available at TB-SENSEX's 09:20 Thu entry ≈ 45 − (10 + 3.3 + 13) ≈ **₹18.7L vs ₹17.2L need → likely CLEARS, but only ~₹1.5L spare**. User decision (a/b/c) still owed — surfaced in chat. | notional math in chat |
| 2026-08-18 ~20:0x | **§3 MARGIN DECISION RESOLVED — user: "I have enough margin for Thursday."** Kite equity screenshot: available ₹44,68,010 / used ₹0.59L / cash ₹2.76L. Thursday stack ≈ ₹43.5L incl. TB-SENSEX 1.3× headroom → clears with ~₹1.2L spare. No top-up; thin-headroom accepted — paper-fallback remains the safety if SPAN gaps up. | Kite equity screenshot in chat |
