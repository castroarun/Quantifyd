# THE STACK (NIFTY): LIVE suite + COMB sleeve + TB-CSL, 2 lots/system (10L total), ex-Wednesday — Deployment Instructions

STATUS: **DONE — DEPLOYED LIVE 2026-08-13 22:33 IST** — config at 2 lots/system; NAS master mode flipped PAPER->LIVE; first real-money trade Mon 2026-08-17 (NIFTY 916 DTE1); Fri 08-14 runs paper (DTE2/DTE4, no live day).
> Executor directive during run: *"do not alter anything in the live system except the lots."* Portfolio stop-loss + profit-trail (nas_portfolio_stop.py) and ALL system-level / per-lot SLs UNTOUCHED; only lots_per_leg 3->2 changed.

> **Absolute path of this file (laptop):** `c:\Users\arunc\Documents\Projects\Covered_Calls\docs\THE_STACK_NIFTY_EXWED_DEPLOY_STATUS.md`
> **Same file on VPS (canonical repo):** `/home/arun/quantifyd/docs/THE_STACK_NIFTY_EXWED_DEPLOY_STATUS.md`
> Carry this file to the implementing session. Update the STATUS line + event log below as you execute.

---

## 1. The Ask

**What Arun asked (2026-08-13):** "ok lets implement this THE STACK: LIVE + COMB + TB-CSL · ex-Wed" + follow-up: **"lot sizes to be 2 per system"**

**What we're actually implementing:** make the deployed paper/live books physically match the
portfolio configuration that research/111 §15–16 converged on — three complementary NIFTY
components at **2 lots per system (user 2026-08-13)**, Wednesday (trading-DTE 4) excluded
for the stack's books. Stack total = 6L + 2L + 2L = **10 lots**:

| Component | Vehicle | Lots (target) | Wednesday | Evidence today |
|---|---|---|---|---|
| LIVE suite | nas_916_atm + atm2 + atm4 (live NAS book) | **3 → 2 each = 6L (step B0: config.py + post-close restart)** | already gated OFF (day-matrix DTE0/1 only) | real money (but see §4.3 master=paper) |
| COMB sleeve | `NAS_COMB20` paper book (per-DTE CSL 25/30/30/20/30) | **3L → 2L, qty 130 (step B)** | **to be gated off (step A)** | backfill + paper from 14-AUG |
| TB-CSL | `CSL_TIMEB_NIFTY` paper book (frozen per-DTE windows) | **12L → 2L, qty 130 (step B)** | **to be gated off (step A)** | backfill + paper from 14-AUG |

Reference numbers (lab 2026-08-13, at the OLD study basis LIVE 9L + sleeves 3L — scale rupee
figures by ≈2/3 for the new 2-lot sizing; the ratio is size-invariant): THE STACK ex-Wed =
**+₹3,64,608, MaxDD −₹17,501, ratio 20.8, avg component corr 0.18, n≈57 ex-Wed days** (mixed evidence: LIVE real+shadow, sleeves
backfill/model — see /app/straddles#portfolio-lab for the always-current version).

**NOT in scope:** SENSEX books (CSL_TIMEB_SENSEX 6L, CSL30F_SENSEX), the evidence books
(CSL30F_NIFTY, NAS_C20_TRAIL, NAS_C20_SHIFT — these KEEP trading all days incl. Wednesday
to collect OOS evidence), and flipping NAS master mode to live (user-only decision, §4.3).

---

## 2. Why (one paragraph of context)

All three live NAS systems are the same 09:16 ATM straddle differing only in post-SL management,
so replacing their mechanics with CSL would collapse diversification (corr 0.99). The winning
structure keeps the LIVE suite as-is and adds two low-correlation sleeves: COMB (full-day CSL
replacement arm, corr 0.31 to LIVE) and TB-CSL (time-blocked windows, corr 0.18 to LIVE, 0.28
to COMB). Every arm's worst DTE is Wednesday (DTE4); ex-Wed the stack's ratio roughly triples.
Full derivation: `/home/arun/quantifyd/research/111_sensex_manual_mgmt/SENSEX_MANUAL_STRADDLE_MGMT_FORENSIC_STATUS.md` §14–16.

---

## 3. Current state inventory (verified 2026-08-13 ~23:00 IST)

- **Paper executor (7 books):** `/home/arun/quantifyd/research/111_sensex_manual_mgmt/scripts/csl_paper_exec.py`
  — cron `12 9 * * 1-5` (crontab of user arun on VPS). Books dict keys: CSL_TIMEB_NIFTY (12L/qty 780),
  CSL_TIMEB_SENSEX (6L/120), NAS_COMB20 (3L/195), NAS_C20_TRAIL (3L/195), NAS_C20_SHIFT (3L/195),
  CSL30F_NIFTY (3L/195), CSL30F_SENSEX (3L/60). Local working copy: `c:\Users\arunc\Documents\Projects\Covered_Calls\csl_paper_exec.py`.
- **Frozen config:** `/home/arun/quantifyd/backtest_data/csl_paper_config.json` — per-book per-DTE
  {entry, exit, sl}. Already re-frozen twice (SL∅→30 SENSEX DTE0; NAS_COMB20 per-DTE SLs). Every
  change must append to `refreeze_note`.
- **State/publish:** `/home/arun/quantifyd/backtest_data/csl_paper_state.json` → copied to
  `/home/arun/quantifyd/static/app/csl_paper.json`; intraday `static/app/csl_paper_live.json`;
  alert events feed inside state (Windows watcher `scripts/csl_alert_watcher.pyw` polls it, 30s).
- **Backfill (nightly):** `/home/arun/quantifyd/research/111_sensex_manual_mgmt/scripts/csl_paper_backfill.py`
  → `static/app/csl_paper_backfill.json`. Runs in `/home/arun/quantifyd/research/58_intraday_recenter_straddle/scripts/regen_straddles.sh`
  (cron `40 15 * * 1-5`) along with `nas_baseline.py`, `portfolio_lab.py`, `strategy_rankings.py`.
- **Portfolio lab:** `/home/arun/quantifyd/research/111_sensex_manual_mgmt/scripts/portfolio_lab.py`
  → `static/app/straddles/portfolio_lab.json` → rendered at `/app/straddles#portfolio-lab`.
  Currently scales TB-CSL by `3/12.0` (line `comp["TBCSL_3L"] = book_daily("CSL_TIMEB_NIFTY", scale=3 / 12.0)`).
- **NAS day-matrix:** `/home/arun/quantifyd/backtest_data/nas_day_matrix.json` — NIFTY 916 books
  already dte {0,1} only + live:true; **master mode `backtest_data/nas_master_mode.json` = "paper" since 05-AUG**.
- **Frontend page:** `/home/arun/quantifyd/frontend/src/pages/Straddles.tsx` (laptop working copy
  `c:\Users\arunc\Documents\Projects\Covered_Calls\Straddles_CUR.tsx` — upload via sftp, then build).
  Rules block text mentions "CSL_TIMEB_NIFTY (12 lots · qty 780)" — must change in step B.
- **VPS access:** paramiko, password regex from
  `C:\Users\arunc\.claude\projects\c--Users-arunc-Documents-Projects-Covered-Calls\memory\vps_ssh_paramiko.md`
  (`re.search(r"password='([^']+)'", ...)`). Single spaced connections (sshd rate-limits storms).
  Git commits happen ON THE VPS (`cd /home/arun/quantifyd && git add ... && git commit -F /tmp/msg.txt && git push`),
  redact PAT in push output with `sed -E 's#ghp_[A-Za-z0-9]+#ghp_***#g'`. Co-author line:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

## 4. Implementation steps (execute in order)

### Step A — Gate the two stack sleeves out of Wednesday (DTE4)

Edit `/home/arun/quantifyd/backtest_data/csl_paper_config.json` with a python snippet on the VPS:

```python
import json
from datetime import datetime
p = "/home/arun/quantifyd/backtest_data/csl_paper_config.json"
cfg = json.load(open(p))
removed = {}
for bk in ("NAS_COMB20", "CSL_TIMEB_NIFTY"):
    if "4" in cfg["books"].get(bk, {}):
        removed[bk] = cfg["books"][bk].pop("4")
cfg["refrozen_at"] = datetime.now().isoformat()[:16]
cfg["refreeze_note"] = cfg.get("refreeze_note", "") + \
    " | <DATE>: THE STACK ex-Wed deploy - DTE4 removed from NAS_COMB20 + CSL_TIMEB_NIFTY (docs/THE_STACK_NIFTY_EXWED_DEPLOY_STATUS.md)"
json.dump(cfg, open(p, "w"), indent=1)
print("removed:", removed)
```

The executor already logs "no config for DTE4 — skip today" and skips cleanly (verified behavior).
Do NOT touch DTE4 for CSL30F_NIFTY / NAS_C20_TRAIL / NAS_C20_SHIFT (evidence books) or SENSEX books
(SENSEX "DTE4" is Friday, not Wednesday — different venue calendar; leave alone).

**Guard:** the executor's `freeze_config()` re-ADDS missing fixed-cfg books' full 5-DTE schedules
only when the book key is absent entirely — a popped DTE key stays popped. But NAS_COMB20 is
`cfg_from: "fixed"`: confirm after edit that `cfg["books"]["NAS_COMB20"]` still exists with keys
0-3 only (if the whole book key were deleted, freeze_config would recreate all 5 — don't delete the book).

### Step B0 — LIVE suite 3 lots → 2 lots per system (REAL-MONEY CONFIG — handle with care)

`/home/arun/quantifyd/config.py`: in `NAS_916_ATM_DEFAULTS`, `NAS_916_ATM2_DEFAULTS`,
`NAS_916_ATM4_DEFAULTS` (lines ~516-532): `'lots_per_leg': 3,` → `'lots_per_leg': 2,` and in
`NAS_ATM_DEFAULTS` (~line 449) `'paper_lots_per_leg': 3,` → `'paper_lots_per_leg': 2,` (keeps
paper-shadow size matched to live — the standing convention). Extend the inline comment trail:
`# 2026-08-1x: user 3->2 (THE STACK, 2 lots per system)`.
**config.py is read by the Flask process → requires `sudo systemctl restart quantifyd` — ONLY
after 15:30 IST, never during market hours.** SENSEX lots: leave untouched (out of scope).
Do NOT change `nas_day_matrix.json` (already correct) or master mode (§4.3).
**Do NOT touch `rupee_stop_per_lot: 2500` (ATM2) or the portfolio stop (-1300/lot)** — both are
per-lot and auto-scale with the 3→2 resize; both are validated (research/96, research/90) parts
of the LIVE suite the stack deliberately keeps. Sleeves get NO such overlays (STATUS §17 test).

### Step B — Resize the two paper sleeves to 2L (qty 130 = 2 × lot 65)

1. In `/home/arun/quantifyd/research/111_sensex_manual_mgmt/scripts/csl_paper_exec.py` (and the
   laptop copy `c:/Users/arunc/Documents/Projects/Covered_Calls/csl_paper_exec.py` in sync):
   - `"CSL_TIMEB_NIFTY": {**NIFTY_MKT, "lots": 12, "qty": 780, "cfg_from": "lab"},`
     → `"CSL_TIMEB_NIFTY": {**NIFTY_MKT, "lots": 2, "qty": 130, "cfg_from": "lab"},`
   - `NAS_COMB20`: `"lots": 3, "qty": 195` → `"lots": 2, "qty": 130` (keep cfg_from/fixed_cfg as-is).
   - CSL30F_NIFTY / NAS_C20_TRAIL / NAS_C20_SHIFT are NAS-suite A/B evidence books sized to match
     the live suite: **if step B0 ships, resize these three to 2L/130 as well** (A/B stays
     size-matched); if B0 is deferred, leave them at 3L.
2. Same lots/qty changes in `csl_paper_backfill.py` BOOKS dict so the nightly backfill regenerates
   history at the new sizes (records carry their own `lots`/`qty`; mixed history stays self-describing).
3. In `portfolio_lab.py`: make scaling per-record — inside `book_daily`, scale each record by
   `TARGET_LOTS / r.get("lots", TARGET_LOTS)` with `TARGET_LOTS = 2` (robust across mixed
   12L/3L/2L history), and normalize LIVE-suite days by `2 / b["lots"]` instead of `3 / b["lots"]`.
   Rename component keys `*_3L` → `*_2L` / `LIVE_SUITE_9L` → `LIVE_SUITE_6L` and update the
   `basis` string to "LIVE 6L (2L/system) + sleeves 2L each = 10L stack".
4. Frontend rules text in `frontend/src/pages/Straddles.tsx`: "CSL_TIMEB_NIFTY (12 lots · qty 780)"
   → "(2 lots · qty 130 — stack weight)"; "NAS_COMB20 (3 lots · qty 195" → "(2 lots · qty 130";
   grep for other stale "12 lots"/"3 lots" NIFTY mentions (schedule card, day-curve notes).
5. SENSEX book (CSL_TIMEB_SENSEX 6L) untouched — not part of this NIFTY stack.

### Step C — Mark the stack as DEPLOYED in the lab

In `portfolio_lab.py`, PORTS list: rename label `"THE STACK: LIVE + COMB + TB-CSL"` →
`"THE STACK (DEPLOYED 10L ex-Wed): LIVE + COMB + TB-CSL"` and update the `verdict` string with
the deployment date and the 2-lots-per-system basis. Optional but nice: default-chart the ex-Wed row in `Straddles.tsx`
(`findIndex(... q2.scope === 'all')` → `'ex-Wed'`).

### Step D — Deploy + rebuild (all safe any time of day)

None of this touches gunicorn (executor is a standalone cron process; frontend is static):

```
# from laptop (paramiko pattern): sftp put the 3 edited scripts + Straddles.tsx, then on VPS:
cd /home/arun/quantifyd && venv/bin/python3 research/111_sensex_manual_mgmt/scripts/csl_paper_backfill.py   # ~75 min, run detached: setsid nohup ... > /tmp/csl_backfill.log 2>&1 &  (do NOT run 09:00-15:40 IST)
cd /home/arun/quantifyd && venv/bin/python3 research/111_sensex_manual_mgmt/scripts/portfolio_lab.py        # fast
export PATH=/home/arun/.nvm/versions/node/v20.20.2/bin:$PATH && cd /home/arun/quantifyd/frontend && npm run build
# git add the changed files + frontend/public/straddles/portfolio_lab.json, commit (Co-Authored-By line), push (PAT-redacted)
```

If not running the backfill manually, the nightly 15:40 regen does it — the lab just shows 12L-scaled
TB-CSL history until then (fine if step B.3 used the per-record lots scaling).

### Step E — Verify (next trading morning, ~09:15 IST)

```
tail -30 /tmp/csl_paper.log     # on VPS
```
Expect: `CSL_TIMEB_NIFTY plan: DTE<d> ... qty 130 (2 lots)`; `NAS_COMB20 ... qty 130 (2 lots)`; `NAS_COMB20 plan` with the per-DTE SL;
**on a Wednesday**: `NAS_COMB20: no config for DTE4 — skip today` + same for CSL_TIMEB_NIFTY, while
CSL30F_NIFTY / TRAIL / SHIFT still plan. Desktop alert should fire on the first entry (~09:16 or
book window). After 15:40: `/app/straddles#portfolio-lab` shows the DEPLOYED label, TB-CSL component
sources gaining `PAPER` days at 3L.

### 4.3 USER-ONLY decision (do not automate)

The LIVE suite component is real-money **only if** NAS master mode is 'live'. It has been
**'paper' since 05-AUG** (`backtest_data/nas_master_mode.json`, last live orders 04-AUG).
Flipping back is Arun's explicit call via /app/nas-config. THE STACK runs meaningfully in
all-paper mode meanwhile; just report which mode each component was in (nas_baseline already
tags days REAL/PAPER by actual order mode).

---

## 5. Event log (append as you execute)

| Date/time | Event | Notes |
|---|---|---|
| 2026-08-13 23:30 IST | Instructions written (this file), STATUS PLANNED | source session had full research/111 §14-16 context |
| 2026-08-13 22:05 IST | Backup taken | backups/the_stack_deploy_20260813_220505 (8 files) |
| 2026-08-13 22:12 IST | Step A: DTE4 popped from NAS_COMB20 + CSL_TIMEB_NIFTY (paper sleeves ex-Wed) | keys now 0-3 |
| 2026-08-13 22:14 IST | Step B0: config.py NAS_916 lots_per_leg 3->2 + paper_lots 3->2 (SENSEX 3 untouched) | import OK |
| 2026-08-13 22:18 IST | Step B: sleeves 12L/3L->2L/130 (exec+backfill); portfolio_lab per-record 2L + DEPLOYED label; Straddles.tsx text | lab: STACK ex-Wed +243,074 DD -11,667 ratio 20.8 |
| 2026-08-13 22:21 IST | Restart quantifyd (book flat) -> 2-lot config live; master STILL paper | service active, no errors |
| 2026-08-13 22:30 IST | Frontend built on VPS | static/app |
| 2026-08-13 22:33 IST | master mode flipped PAPER->LIVE (user consent); armed-check OK (SL monitors 10s, no errors) | first live Mon 08-17 |
| 2026-08-13 22:40 IST | User directive: only lots in live system -> portfolio stop + all SLs UNTOUCHED; committed to git; wrote LIVE rules doc | STATUS DONE |

---

## 6. Crash recovery / verify without Claude

- Is the config gated? `python3 -c "import json; print(json.load(open('/home/arun/quantifyd/backtest_data/csl_paper_config.json'))['books']['NAS_COMB20'].keys())"` → should NOT contain '4'.
- Book sizes: grep `CSL_TIMEB_NIFTY` in `csl_paper_exec.py` → lots 2, qty 130; `config.py` → `lots_per_leg: 2` in all three NAS_916_*_DEFAULTS.
- Morning behavior: `/tmp/csl_paper.log` (see step E). Executor cron: `crontab -l | grep csl_paper`.
- Lab freshness: `generated_at` inside `http://94.136.185.54:5000/app/straddles/portfolio_lab.json`.
- Nothing here requires a gunicorn restart; if the page hangs, check §"wedged worker" precedent —
  restart `quantifyd` only OUTSIDE 09:15–15:30 IST.

## 7. Files touched (all committable)

| File | Change |
|---|---|
| `backtest_data/csl_paper_config.json` | pop DTE "4" from NAS_COMB20 + CSL_TIMEB_NIFTY, refreeze note (NOT in git — runtime state; the refreeze note IS the audit trail) |
| `config.py` | NAS_916_* lots_per_leg 3→2 + paper_lots_per_leg 3→2 (restart AFTER 15:30 IST only) |
| `research/111_sensex_manual_mgmt/scripts/csl_paper_exec.py` | CSL_TIMEB_NIFTY 12L→2L, NAS_COMB20 3L→2L (+ evidence books 3L→2L if B0 ships) |
| `research/111_sensex_manual_mgmt/scripts/csl_paper_backfill.py` | same resize |
| `research/111_sensex_manual_mgmt/scripts/portfolio_lab.py` | per-record lots scaling + DEPLOYED label |
| `frontend/src/pages/Straddles.tsx` | rules text 12L→3L, DEPLOYED label default-chart ex-Wed |
| `docs/THE_STACK_NIFTY_EXWED_DEPLOY_STATUS.md` | this file: STATUS → DONE + event log |
| `TODO.md` | move stack-deploy item to Done |
