# CSL_TIMEB_SENSEX → REAL MONEY (6 lots, BFO) — Deployment Instructions

STATUS: **PLANNED** (user decision 2026-08-18 evening: "this has to be real live")

> **Laptop:** `c:\Users\arunc\Documents\Projects\Covered_Calls\docs\CSL_TIMEB_SENSEX_LIVE_DEPLOY_STATUS.md`
> **VPS:** `/home/arun/quantifyd/docs/CSL_TIMEB_SENSEX_LIVE_DEPLOY_STATUS.md`
> Context: `docs/LIVE_TRADING_SYSTEM_RULES.md` + `docs/THE_STACK_FULL_LIVE_DEPLOY_STATUS.md` (the
> NIFTY sleeves' live order layer — already built, venue-generic, proven since Mon 17-AUG).

## 1. The change (one line of config — the order layer already handles BFO)

`/home/arun/quantifyd/research/111_sensex_manual_mgmt/scripts/csl_paper_exec.py` BOOKS:
`"CSL_TIMEB_SENSEX": {**SENSEX_MKT, "lots": 6, "qty": 120, "cfg_from": "lab"},`
→ append `"mode": "live"` (+ dated comment). That's the entire trading change: `place_market()`
uses `B["seg"]` (=BFO) and the marketable-LIMIT/±3%/tick-0.05 logic is venue-generic; gates
(master mode, kill flag, freeze flag), fill-anchored SLs, per-leg idempotent exits, naked-leg
unwind, margin gate, REAL-tagged records/alerts all apply automatically.

**Deploy before the next 09:12 cron** (executor is standalone; no gunicorn restart). If deployed
tonight, the first REAL window is **tomorrow Wed 10:30→12:00 SL20 (DTE1)** — note this is the
venue's danger day, but the TB window IS the studies' prescription for SENSEX-Wed (compressed
slice + tight combined SL); Thursday (DTE0 full-day SL30) follows as the venue's harvest day.

## 2. Checks the implementing session must make

1. **Margin arithmetic:** SENSEX 6L short straddle ≈ ₹10–12L (contract ≈ NIFTY's; the gate's
   `MARGIN_PER_LOT=165000` estimate is close enough). Concurrent worst case is THURSDAY:
   SENSEX suite 9L (~₹15L) + TB-SENSEX 6L (~₹10L) + NIFTY books closed (Thu = suite shadow,
   COMB live 2L ~₹3L, TB-NIFTY 8L live Thu full-day ~₹13L!) → peak ≈ ₹41L vs ~₹45L net.
   **TIGHT.** Verify with `--probe` (prints avail vs need) and check `k.margins()` net that
   morning; the live gate falls back to paper if short — that is acceptable behavior, not a bug.
2. **BFO liquidity:** SENSEX weeklies are thinner than NIFTY. The ±3% marketable-LIMIT band
   should still fill instantly ATM; watch the FIRST fill's slippage vs LTP in the revision log —
   if >0.5% of premium, flag for band review.
3. **Tag length:** `("CSL_" + "CSL_TIMEB_SENSEX")[:20]` = `CSL_CSL_TIMEB_SENSEX` (20 chars, fits) —
   verify the tag actually applied in the orderbook (Kite truncates silently).
4. Do NOT touch: the frozen schedule (incl. Thu SL30 = "never stopless" insurance), the 6L size,
   any NIFTY book, the SENSEX suite.

## 3. Verify (first live morning)

`/tmp/csl_paper.log`: `CSL_TIMEB_SENSEX plan [LIVE]` at 09:12; at the window: `ENTER [LIVE]` with
FILL prices; desktop popup tagged REAL; Kite orderbook BFO pair with the tag; record row
`"source": "REAL"` after exit; guardian cycle clean. Add all of it to the Revision Log below.

## 4. Registry (already done by the originating session)

Ops Center: "TB-SENSEX first REAL window verify" (due 19-AUG) + "SENSEX Wednesday exposure
review" (due 04-SEP) are registered. On completion, update this STATUS → DONE + Revision Log.

## 5. Revision Log (append)

| Date/time | Event | Evidence |
|---|---|---|
| 2026-08-18 | Instructions written | — |
