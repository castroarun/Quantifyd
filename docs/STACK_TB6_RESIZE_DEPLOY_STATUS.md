# THE STACK resize 6/2/2 → 6/2/6 — TB-CSL to 6 lots (sec-18b decision) — Instructions

STATUS: **PLANNED** (decision by Arun 2026-08-14 ~14:40 IST: "lets go for 6/2/6")

> **Laptop:** `c:\Users\arunc\Documents\Projects\Covered_Calls\docs\STACK_TB6_RESIZE_DEPLOY_STATUS.md`
> **VPS:** `/home/arun/quantifyd/docs/STACK_TB6_RESIZE_DEPLOY_STATUS.md`
> Context docs: `LIVE_TRADING_SYSTEM_RULES.md`, `THE_STACK_FULL_LIVE_DEPLOY_STATUS.md`,
> research/111 STATUS §18/§18b (the evidence), `/app/straddles#portfolio-lab` (TB-OVERWEIGHT ×3 row = this config).

## 1. The decision

Stack becomes **LIVE suite 6L + NAS_COMB20 2L + CSL_TIMEB_NIFTY 6L = 14 lots** (ex-Wed, unchanged days).
Evidence: §18b grid — best cell (+₹3,75,440 / −₹12,130 / ratio 31.0 ex-Wed, 34d); marginal TB lots cost
~₹72 DD per +₹66k profit vs ~₹8,669 for COMB. COMB deliberately stays 2L (its axis lowers the ratio).
Recorded caveat: TB history is in-sample-flattered; a real in-window SL day at 6L ≈ −₹12–15k (model), vs
−₹1.2k worst in backfill. User accepts; the lab's live-first rows self-audit this as real TB days accrue.
LIVE suite and all other books: NO change. Wednesday stays off. All frozen configs untouched.

## 2. Changes (execute in order; ~15 min total)

### A. Executor (the only live-behavior change)
`/home/arun/quantifyd/research/111_sensex_manual_mgmt/scripts/csl_paper_exec.py` BOOKS:
`"CSL_TIMEB_NIFTY": {**NIFTY_MKT, "lots": 2, "qty": 130, "cfg_from": "lab", "mode": "live"},`
→ `"lots": 6, "qty": 390`. Nothing else in BOOKS. The margin gate self-adjusts
(need = 6 × 1.65L × 1.3 ≈ ₹12.9L during the TB window only; account net was ₹46.1L on 08-14 —
comfortably clears suite ~₹10L + COMB ~₹3.3L + TB ~₹12.9L, but the gate re-checks live at entry
and falls back to paper if short — leave that mechanism alone).
Deploy any time: the running process doesn't reload; effective next cron (Mon 09:12).
Keep the laptop copy `c:\Users\arunc\Documents\Projects\Covered_Calls\csl_paper_exec.py` in sync.

### B. Backfill sizes
`csl_paper_backfill.py` BOOKS: same `"lots": 6, "qty": 390` for CSL_TIMEB_NIFTY, so nightly
regenerated history is at deployed size. (Records carry their own lots/qty — mixed history is fine.)

### C. Portfolio lab — CRITICAL correctness step
`portfolio_lab.py` currently does `comp["TBCSL_2L"] = book_daily("CSL_TIMEB_NIFTY")` with NO
per-record scaling → after the resize, new 6L records would silently mix with 2L history.
Fix `book_daily` to normalize per record: scale each record's pnl by `TARGET / r.get("lots", TARGET)`
with the component's target lots (LIVE normalization stays 2L/system). Then restructure components:
- `TBCSL_6L` (target 6) replaces `TBCSL_2L` in NAMES/matrix/components.
- DEPLOYED stack row → `("THE STACK (DEPLOYED 14L ex-Wed 6/2/6): LIVE + COMB + TB-CSL", 14, [...TBCSL_6L])`.
- Keep ONE reference row `("pre-18b stack 10L (TB@2L)", 10, [... TBCSL_2L_ref])` where the ref
  component is the 6L series × (2/6) — so the before/after comparison stays visible on the page.
- Drop the now-redundant ×2/×3 virtual rows (the deployed row IS ×3); update `verdict`/`basis` strings.
Run it once after editing; verify `/app/straddles#portfolio-lab` shows the 14L deployed row.

### D. Page texts
`frontend/src/pages/Straddles.tsx`: rules block "CSL_TIMEB_NIFTY (2 lots · qty 130" → "(6 lots ·
qty 390 — sec-18b overweight"; grep for other stale "2 lots" TB mentions. `npm run build` (safe anytime).

### E. Docs + TODO close-out
- `LIVE_TRADING_SYSTEM_RULES.md` §0/§6: TB-CSL 6L/390, stack total 14L.
- `THE_STACK_FULL_LIVE_DEPLOY_STATUS.md` event log: one row for this resize (date, commit).
- `TODO.md`: update the stack gate line (TB reweight decision RESOLVED by §18b — remove the
  "~4wk TB reweight" gate, keep Mon-fills verify + suite-Friday review + 15-SEP checkpoint).
- This file: STATUS → DONE + Revision Log (commits, py_compile output, lab regen output).

## 3. Verification (Monday 09:12+)
`/tmp/csl_paper.log`: `CSL_TIMEB_NIFTY plan [LIVE]: ... qty 390 (6 lots)`; at its window entry
(Mon = DTE1 13:00→14:00 SL20): marketable-LIMIT fills for qty 390, REAL popup, margin gate log clean.
Note Monday is ALSO the sleeves' first REAL-fill morning (LIMIT fix) — one verification covers both.

## 4. Rollback
Set lots/qty back to 2/130 in the two BOOKS dicts (+ lab targets); effective next morning. No state migration needed.
