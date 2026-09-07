# CSL-60 DTE-0 — NIFTY Expiry-Day 09:16 ATM Straddle, 60% Per-Leg Stop → PAPER BOOK

STATUS: DEPLOYED (2026-09-07) — first live paper trade Tue 2026-09-08 09:16

## 1. The Ask

**What you asked (2026-09-07):** "nifty 60 SL is the best one we have selected for
Nifty, i think for DTE1 and 0 ... is it already enabled for paper trades?" then
"i want to see it in this page as a new position/trade" (the /app/nas Trade Book).

**What we're actually deploying:** the winning configuration from the AlgoTest
study (research/136, §0d-viii — rank 1 of 98 systems) as a live **paper** book,
rendered as a new group in the NAS Trade Book. **DTE-0 ONLY** — the study
explicitly dropped DTE-1 (t 0.91, Net/DD 0.94 at 60%, OOS-negative years; only a
lone 200%-stop cell ever passed, i.e. the multiple-testing false-positive rate).
If Arun wants DTE-1 shadowed anyway it becomes a second, separately-tagged book —
not silently folded into this one. **Arun took that option on 2026-09-07** ("use
today's options data capture and playback and put the positions"): the service
now also trades DTE-1 days into a `book='dte1'` SHADOW ledger, rendered as its
own Trade Book group (NIFTY CSL60 · DTE-1 shadow) and reported separately
(`shadow` block in the JSON). The DTE-0 record stays pure.

## 2. The Base — rules (locked, AlgoTest-parity)

| Rule | Value |
|---|---|
| Underlying | NIFTY weekly options, ATM (strike nearest spot, 50-step) |
| Trade days | **Expiry day only (DTE-0)** — nearest weekly expiry == today (Tuesdays currently) |
| Entry | **09:16** — sell 1 ATM CE + 1 ATM PE at recorded 09:16 LTP |
| Size | **10 lots (qty 650)** — matches the study, 1 premium pt = ₹650 |
| Per-leg stop | LTP ≥ **1.60 × entry** → that leg exits at the breaching LTP (not the theoretical 1.60×) |
| Trail-to-BE | when the first leg stops, the surviving leg's stop tightens to **its own entry price** (AlgoTest "Trail SL to Break-even – All Legs") |
| Square-off | **Partial** — legs exit independently |
| Time exit | **15:15** close whatever survives |
| Event skip | Union Budget + LS-election result days (calendar rule; next: 2027-02-01) |
| Missed entry | one-shot: if the 09:16 chain snapshot is absent or first evaluation is after 09:30 → **skip the day**, log MISSED (916 semantics — never replay late) |
| Costs (reporting) | study-parity model: 0.59% of premium turnover + ₹80/trade; gross and turnover stored so the model stays query-time |

**Study baseline to validate against** (research/136 §0d-viii, ex-events, 10 lots):
net ₹23,69,304 / 294 trades / mean ₹8,059 / median +₹8,927 / WR 63.6% / MaxDD
−₹1,51,578 / worst −₹52,458 / lose-streak 6 / t 4.28 / OOS t 2.24.
Study page: http://94.136.185.54:5000/app/straddle-study

## 3. Architecture — no live engine touched, no restart needed

Per the 2026-08-17 binding (UX/registry work never touches trading logic) and to
keep THE STACK's executors pristine, this is a **standalone paper service** in
the straddle45_paper.py mould:

- `services/csl60_paper.py` — cron every minute (Mon–Fri 09:00–15:59); exits
  instantly unless today is NIFTY weekly expiry. Reads ONLY
  `backtest_data/options_data.db` (`option_chain` recorded 1-min quotes +
  `underlying_spot`). Writes its own `backtest_data/csl60_paper.db` and
  publishes `static/app/csl60_paper.json` (+ `frontend/public/` copy for
  future builds). fcntl lock + atomic tmp→rename (statefile-race binding).
- **NAS Trade Book display**: `frontend/src/pages/Nas.tsx` additionally fetches
  `/app/csl60_paper.json` and renders a "CSL 60 · DTE-0" paper group. Frontend
  build on VPS — served without restart.
- Cron install follows the crontab-safety procedure (backup → temp file →
  count check → install).

Fidelity note: the study was built on AlgoTest 1-minute bars, so a 1-minute
recorded-chain paper book is resolution-consistent with what it validates.
(The no-5-min binding is satisfied: this is 1-min recorded data, the accepted
fallback; 3-sec ticks are not recorded for the full chain.)

## 4. Plan

1. STATUS-MD (this file) ✅
2. Verify option_chain cadence/coverage for today (background query) 
3. `services/csl60_paper.py` + smoke test on today's recorded data (read-only)
4. Cron entry (safe procedure)
5. Nas.tsx trade-book group + VPS build
6. Registrations: /app/strategies index (new paper row), Ops & Review Center
   (job + 30-session review), TODO.md, tracker event log
7. First live paper trade: **Tue 2026-09-08 09:16** (NIFTY weekly expiry)

## 5. Status / event log

| Date/time | Event | Notes |
|---|---|---|
| 2026-09-07 11:3x IST | Build started; STATUS-MD written | Market open — no engine files touched, no restart planned |
| 2026-09-07 ~12:00 IST | Feed verified | option_chain full 1-min cadence (135/135 minutes), 09:16 ATM rows clean, nearest expiry = 2026-09-08 (Tue) |
| 2026-09-07 ~12:10 IST | services/csl60_paper.py shipped; replay 2026-09-01 verified | CE SL at 1.60x exact, PE trailed to BE and stopped 13:51 — mechanics correct |
| 2026-09-07 ~12:20 IST | Seed backfill: 12 expiry days (16-Jun→1-Sep) | +₹24,116 net cum, 58% WR, worst −₹28,170 (10 lots); square-off bug (missing 15:15 minute) found+fixed via 15:30 window |
| 2026-09-07 ~12:30 IST | Cron installed (safety procedure, 97→99 lines) | `* 9-15 * * 1-5 ... csl60_paper.py mark`; no-ops on non-expiry days |
| 2026-09-07 ~12:40 IST | NAS Trade Book wired (frontend-only, no restart) | Nas.tsx: sleeve def + /app/csl60_paper.json fetch + adapter; planned row shows 'Tue 09:16' on non-expiry days |
| 2026-09-07 ~12:50 IST | Registered: /app/strategies (paper row) + Ops Center (job + review due 2026-11-30) | Study-slot gap noted: /app/backtest factsheet entry owed |
| 2026-09-07 ~12:10 IST | DTE-1 SHADOW added on Arun's instruction | `book` column (dte0/dte1); mark() trades DTE-1 too, tagged; `seed-dte1` backfills prior sessions; separate Trade Book group; JSON `shadow` block keeps ledgers apart |
| 2026-09-07 12:06 IST | Today's positions LIVE from recorded capture | DTE-1 shadow ATM 23850: CE in 71.35 / PE in 70.1 at 09:16, SLs 114.2/112.2, both OPEN (CE +6,142 / PE −11,765 at 12:06) |

## 6. Crash recovery

- State DB: `backtest_data/csl60_paper.db` (tables `legs`, `days`); published
  JSON: `static/app/csl60_paper.json`.
- Is the cron alive? `crontab -l | grep csl60`; log `/tmp/csl60_paper.log`.
- Manual tick: `cd /home/arun/quantifyd && ./venv/bin/python3 services/csl60_paper.py mark`
- Show state: `./venv/bin/python3 services/csl60_paper.py show`
- The service is idempotent per minute — re-running never duplicates legs
  (PRIMARY KEY on trade_date+leg).
- Nothing here places orders or touches live engines; killing the cron simply
  stops the paper book.

## 7. Files

| File | Purpose | Committable? |
|---|---|---|
| `services/csl60_paper.py` | Paper executor (cron, 1-min) | yes |
| `backtest_data/csl60_paper.db` | Book state | no (data) |
| `static/app/csl60_paper.json` | Published state for the NAS page | no (regenerated) |
| `frontend/src/pages/Nas.tsx` | Trade-book group render | yes |
| This file | Deploy record | yes |
