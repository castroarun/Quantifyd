# THE STACK — FULL LIVE: all systems real-money on tested days (Mon/Tue/Thu/Fri), 2 lots each

STATUS: **PART A DONE (suite live Thu+Fri from TODAY 14-AUG) · PART B (sleeves) PLANNED** (updated 2026-08-14 07:35 IST)

> **Absolute path (laptop):** `c:\Users\arunc\Documents\Projects\Covered_Calls\docs\THE_STACK_FULL_LIVE_DEPLOY_STATUS.md`
> **VPS (canonical repo):** `/home/arun/quantifyd/docs/THE_STACK_FULL_LIVE_DEPLOY_STATUS.md`
> Prerequisite reading: `docs/LIVE_TRADING_SYSTEM_RULES.md` (current live rules, verified 08-14) and
> `docs/THE_STACK_NIFTY_EXWED_DEPLOY_STATUS.md` (the 2-lot ex-Wed deploy, DONE 08-13).

---

## 1. The Ask (user decision, verbatim intent)

*"deploy all the systems tested as-is with all the tested params and live days as-is except the
lot sizes of 2 each"* — i.e. make deployment match **Matrix A (as-tested)** exactly, with real money:

| System | Mon (DTE1) | Tue (DTE0) | Wed | Thu (DTE3) | Fri (DTE2) | Params (FROZEN — do not touch) |
|---|---|---|---|---|---|---|
| nas_916_atm | 🔴 REAL | 🔴 REAL | off | 🔴 **REAL (new)** | 🔴 **REAL (new)** | per-leg SL30 + trail-to-cost + re-enter ≤5 |
| nas_916_atm2 | 🔴 REAL | 🔴 REAL | off | 🔴 **REAL (new)** | 🔴 **REAL (new)** | ₹2,500/lot rupee stop, one-and-done |
| nas_916_atm4 | 🔴 REAL | 🔴 REAL | off | 🔴 **REAL (new)** | 🔴 **REAL (new)** | per-leg SL30 + roll once |
| NAS_COMB20 | 🔴 **REAL (new)** | 🔴 **REAL (new)** | off | 🔴 **REAL (new)** | 🔴 **REAL (new)** | 09:16→15:20, combined-SL per DTE: 25/30/30/20 (DTE0/1/2/3) |
| CSL_TIMEB_NIFTY | 🔴 **REAL (new)** | 🔴 **REAL (new)** | off | 🔴 **REAL (new)** | 🔴 **REAL (new)** | windows: DTE0 09:30→11:00 SL25 · DTE1 13:00→14:00 SL20 · DTE2 10:00→12:00 SL20 · DTE3 09:16→15:20 SL20 |

All at **2 lots** (NIFTY qty 130/leg). Portfolio stop (−1,300/lot), NIFTY profit-trail (arm 2,000 /
give 350), ATM2 rupee stop: **unchanged** (they auto-scale per-lot). Wednesday stays off everywhere.

**OUT of scope / unchanged:** SENSEX books (live Wed/Thu as they are), evidence paper books
(NAS_C20_TRAIL, NAS_C20_SHIFT, CSL30F_NIFTY, CSL30F_SENSEX, CSL_TIMEB_SENSEX — remain PAPER).

## 1.1 Risk acknowledgments (record, then proceed — the user has decided)

- Thu (DTE3) for the LIVE suite earned **+₹297 over 10 days** in tests (≈ zero; may be net-negative
  after costs). Fri (+₹31k/10d) contradicts older research (r/51/79: far-DTE bleeds). Deployed anyway per decision.
- The two sleeves go live **ahead of** the paper-soak checkpoint (~15-SEP). Their SLs/windows are
  in-sample picks (walk-forward showed ≈half the in-sample edge OOS). Deployed anyway per decision.
- New margin need: 2 more short straddles × 2 lots ≈ **₹5–7L additional** (verify in step D0 —
  account balance was ~₹24.8L; the live suite already consumes ~3 straddles × 2 lots).

---

## 2. Part A — LIVE suite: enable Thu + Fri real-money (config-only, ~5 min)

Edit `/home/arun/quantifyd/backtest_data/nas_day_matrix.json`: for `nas_916_atm`, `nas_916_atm2`,
`nas_916_atm4` set `"dte": {"0": true, "1": true, "2": true, "3": true, "4": false}` (currently 0/1
only). Keep `live: true`, gap flags false. Do it via the UI at `/app/nas-config` (preferred — it
POSTs `/api/nas/day-matrix`) or a python edit of the JSON. `gate()` reads the JSON at decision time —
**no restart needed**. Do NOT touch the three `sensex_*` rows or the squeeze/OTM rows.
Verify: GET `/api/nas/day-matrix` shows the new flags; next Thu/Fri morning log shows live-mode orders.

---

## 3. Part B — Sleeves to REAL MONEY (the build: order execution for the CSL executor)

`csl_paper_exec.py` is a **simulator** — it polls LTPs and records fills, it places no orders.
Going real requires adding a Kite order layer. Scope it as a *minimal, guarded* addition, reusing
the proven patterns in `services/nas_atm_executor.py`:

### B1. Per-book mode flag
In BOOKS: `"NAS_COMB20": {..., "mode": "live"}`, `"CSL_TIMEB_NIFTY": {..., "mode": "live"}` —
every other book stays `"mode": "paper"` (default when absent). Also honor two global gates before
ANY real order: `backtest_data/nas_manual_freeze.flag` absent AND
`backtest_data/nas_master_mode.json == {"mode":"live"}` (the same master switch as the suite —
one kill lever for the whole stack).

### B2. Order functions (mirror nas_atm_executor patterns)
- `place_order(k, tradingsymbol, side, qty)`: Kite `place_order(variety=regular, exchange=NFO,
  product=MIS, order_type=MARKET)`. Wrap in try/except with **timeout retry** — on a read-timeout,
  DO NOT assume failure: poll `k.orders()` for the tag/symbol before re-placing (the 2026-08-06
  SENSEX incident: a timed-out exit was never retried and left a naked short — fix `b9349e1`;
  copy that logic).
- Entry (state OPEN transition): SELL CE + SELL PE market, qty 130 each, then read actual fill
  prices from the order book (`k.order_history`) and use FILL prices as `ce0/pe0/credit`
  (not LTPs) so SL thresholds are anchored to reality.
- Exit (SL_DWELL / TIME_EXIT / EOD_FORCE / backstop): BUY back both legs market, with the retry-
  verify loop above; record fill prices in the day record.
- Order tags: `tag="CSL_"+book` (Kite tag ≤20 chars) so `/nas-guardian` and manual audits can
  attribute orders.

### B3. Safety rails (all mandatory)
1. **Margin pre-check** at entry: `k.margins()` available cash > 1.5× estimated straddle margin,
   else log + fall back to paper for the day (never partial-leg).
2. **One-leg-filled guard:** if leg 1 fills and leg 2 rejects, immediately buy back leg 1 (never
   carry a naked single leg), alert `PLACE_FAIL`.
3. **50% disaster backstop** stays armed on every book (already in the executor).
4. **15:26 hard force** stays; MIS auto-squareoff by broker ~15:20-15:25 is the backstop's backstop —
   set book exits at 15:20 (already) so the broker never has to.
5. Desktop alerts: push_event source **"REAL"** for live books (`push_event` currently hardcodes
   "PAPER" — parametrize). The Windows watcher + page already render REAL red.
6. Records: `"source": "REAL"` on live-book records — nas_baseline/portfolio_lab/leaderboard then
   tag everything correctly with zero further changes.
7. Token freshness: executor starts 09:12; the 08:5x auto-login cron refreshes the token. Guard:
   if `k.profile()` fails at startup, log + run the day in paper (do not die).

### B4. What NOT to change
The trading logic — entry times, windows, dwell, per-DTE SLs, ex-Wed gating, qty 130 — is already
exactly the tested configuration. The build adds ONLY the order layer + rails. Do not "improve"
mechanics in the same change.

### B5. Deploy + verify
- Edit on laptop copy + sftp to VPS (`research/111_sensex_manual_mgmt/scripts/csl_paper_exec.py`),
  `py_compile` check, commit. Executor is cron-standalone: **no gunicorn restart needed**; deploy
  any time before next 09:12.
- Verify next morning: `/tmp/csl_paper.log` shows `NAS_COMB20 [LIVE]` plan lines; Kite orderbook
  shows the 09:16 sell pair with tag `CSL_NAS_COMB20`; desktop alert arrives tagged REAL;
  `/app/straddles#csl-paper` record row shows REAL badge.
- First live day: watch the first SL/exit cycle end-to-end (fills recorded, no phantom).

---

## 4. Sequencing recommendation (compresses risk without delaying much)

Day 1 (config day): Part A (suite Thu/Fri live — zero new code) + build Part B, deploy with sleeves
still `"mode": "paper"`. Day 2: after one clean morning of logs, flip the two books to
`"mode": "live"` (one-line change, no restart). This keeps "all systems live on tested days" within
~2 sessions while ensuring the new order code's first run isn't with real money. If the user wants
same-day: flip immediately after B5's py_compile + a `--probe` run (resolves legs + margin check
without orders).

## 5. Event log (append as you execute)

| Date/time | Event | Notes |
|---|---|---|
| 2026-08-14 07:21 IST | **PART A EXECUTED**: day-matrix dte2+dte3 -> true for nas_916_atm/atm2/atm4 | verified via live /api/nas/day-matrix; master=live; TODAY Fri (DTE2) is a real-money day at 2 lots/system |
| 2026-08-14 07:35 IST | Part B/C/E instructions finalized after full infra recon | sleeves: build today, first live day Mon 17-AUG (see Part C timing note) |

## 6. Rollback / kill

- Whole stack to paper instantly: `POST /api/nas/kill-switch` (suite) + set the two BOOKS back to
  `"mode": "paper"` (sleeves; takes effect next poll-cycle day) or touch
  `backtest_data/nas_manual_freeze.flag` (blocks everything incl. paper).
- Open sleeve positions on kill: buy back manually in Kite app (they're plain MIS straddles) —
  the executor's records can be reconciled after (see 08-06 incident playbook: read the ORDERBOOK
  before calling a mismatch phantom).

## 7. Files to touch

| File | Change |
|---|---|
| `backtest_data/nas_day_matrix.json` | NIFTY 916 trio: dte 2,3 → true (Part A) |
| `research/111_sensex_manual_mgmt/scripts/csl_paper_exec.py` | order layer + mode flags + rails (Part B) |
| `docs/THE_STACK_FULL_LIVE_DEPLOY_STATUS.md` | this file: STATUS → DONE + event log |
| `TODO.md` | link this deploy; supersede the "sleeves paper until 15-SEP" checkpoint note (decision changed) |


---

## Part C — Integration with the EXISTING live infrastructure (recon 2026-08-14; do not miss, do not alter)

The NAS book is protected by this standing machinery (crontab-verified). The sleeve build must
**honor** it where it gates trading, **extend** it where sleeves would otherwise be invisible,
and **change none of it**:

| Existing piece | Cadence | Sleeve-build obligation |
|---|---|---|
| `services/nas_kill_switch.py` + `backtest_data/nas_kill.flag` | persistent | **HONOR**: before ANY real sleeve order, also check `nas_kill.flag` absent (in addition to `nas_manual_freeze.flag` + master mode). One panic button must stop the whole stack. |
| `scripts/killflag_premarket_check.py` | 09:05 | nothing to do (it guards the flag itself) |
| `scripts/nas_live_guardian.py` | */5 min 9-15h | **EXTEND**: add a `check_csl_live_books()` — reconcile csl_paper_state OPEN books vs Kite positions/orderbook (tag prefix `CSL_`), alert on: order placed but no position, position with no book record, exit older than 2 min with position still open. Follow the file's existing check/alert pattern; do NOT restructure it. |
| `scripts/nas_fail_rejected.py` | */2 min | **EXTEND (small)**: include tag-prefix `CSL_` orders in its rejected/failed sweep, or replicate its check inside the executor's retry loop — either way rejected sleeve orders must not die silently. |
| `scripts/nas_integrity_watchdog.py` + `services/nas_watchdog.py` (candle/pipeline freeze, SMTP email) | */5 min | no change needed (sleeves poll REST LTP, not the ticker) — but **REUSE its SMTP helper** for the sleeve PLACE_FAIL / naked-leg email alert instead of inventing a channel. |
| `scripts/dump_nas_mtm.py` (per-minute MTM snapshots) | * 9-15h | optional, recommended: append sleeve live MTM so the future all-stack portfolio-overlay study (§17 revisit) gets real synchronized curves. |
| `scripts/preopen_restart.sh` 09:00 + `auto_login.sh` 08:50 + `token_heal.sh` 09:06 | daily | executor starts 09:12 — AFTER token heal; keep the token-freshness guard (paper-fallback) anyway. |
| `snapshot_nas_eod.py` 15:32 / `nas_analyzer.py` 15:45 / EOD report | daily | no change; the 15:40 regen already publishes sleeve records. |
| nas-live-guardian **agent** (harness, laptop) | on demand / periodic review | **EXTEND its instructions**: add the CSL live books to its hunting list (stops-not-firing, churn, P&L misreads, orderbook-vs-state reconciliation) — same failure classes, new books. |
| Manual-trade etiquette (NRML ignored; manual exit = system stands down that day) | standing | implement for sleeves: if the user manually exits a sleeve position (detected in reconciliation), do NOT re-enter that book that day. |

**Timing note (why sleeves are NOT live today despite the "from today" instruction):** Part B is new
real-order code. Deploying it untested into today's 09:16/10:00 windows repeats the exact failure
class of the 08-06 incident (unverified exit paths). Today the suite trades Fri real (done, Part A);
the implementing session builds Part B today against the checklist above, runs `--probe` + one paper
morning if possible, and the sleeves' first real-money day is **Mon 17-AUG** — every tested day
thereafter. If the user explicitly overrides for a same-day flip, the minimum bar is: py_compile +
`--probe` clean + margin check + manual watch of the first entry.

---

## Part E — MANDATORY completion log (for post-implementation verification)

The implementing session MUST append to this file a **Revision Log** with, per change:

1. File + function touched, one-line intent, and the **commit hash** (small commits, not one blob).
2. Verbatim output of: `py_compile` on every touched file; `csl_paper_exec.py --probe` run;
   the margin pre-check dry run; `GET /api/nas/day-matrix` after any matrix change.
3. First live morning evidence: `/tmp/csl_paper.log` plan + entry lines, Kite orderbook screenshot
   or `k.orders()` dump showing the `CSL_*` tags and fill prices, the day-record JSON with
   `"source": "REAL"`, and the desktop-alert event entries.
4. Guardian evidence: one `/tmp/nas_guardian.log` cycle showing `check_csl_live_books` running clean.
5. Any deviation from these instructions, with reason (deviations without a logged reason = defect).

The originating session (this one) will then verify the implementation elaborately against this log +
live state. Do not mark STATUS DONE until the Revision Log is complete.

| 2026-08-14 07:5x IST | Sleeves LIVE order layer deployed (probe green: gates+legs+margin net Rs46.1L); popups for all books (nas_alert_feed + dual-feed watcher) | commit 33d5f50 |
| 2026-08-14 07:5x IST | **DECISION (user, after sibling stop-by-DTE study input): suite Thu (DTE3) real-money REVERTED to shadow; Fri (DTE2) stays live.** Sibling study: NIFTY per-leg-30% net-negative DTE2+ (-Rs303/lot overall); sleeves' combined-SL is the validated DTE2+ mechanic and carries Thu exposure as paper | day-matrix dte3=false, dte2=true |
