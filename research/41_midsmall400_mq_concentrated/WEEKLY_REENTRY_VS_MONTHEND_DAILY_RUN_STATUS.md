# Weekly Re-Entry vs Month-End Re-Entry — Is "Fast-Out / Slow-In" Actually Optimal?

**STATUS: DONE**  ·  research/41 · Phase 27 · daily-marked engine · VPS canonical

**VERDICT (see results/RESULTS.md): SPLIT** — weekly re-entry IMPROVES `allcash`
(Calmar 1.54→1.72, post-tax +0.6pp, DD −22.2→−20.7) but HURTS `keeptop8`
(DD blows out −20.2→−28.5, Calmar 1.66→1.13, buys false dawns à la Phase 25).
Fast re-entry helps when fully de-risked (cash), hurts when partially (keep-8).
Both `month-end (BASE)` configs reproduced published Phase 22/24 exactly.

---

## 1. The Ask

**What you asked:** *"The engine sells on the weekly clock but only buys back at
month-end. Is this how it was backtested? Did we try the weekly clock buy-back also?"*

**What we're actually testing:** Every prior phase varied the **speed of exit**
(Phase 15 daily/weekly/monthly regime gate; Phase 19 staggered chunked exit) but
**re-entry was hardcoded to month-end in all of them** — the "fast-out / slow-in"
asymmetry was never A/B tested. This run adds a **weekly re-entry** switch: on the
first weekly risk-ON bar *after* a de-risk, immediately rebuild to full (15 names,
current eligibles) instead of waiting for the next monthly rebalance. Everything
else identical. Question: **does faster re-entry beat the locked month-end re-entry
on net (post-tax) risk-adjusted return, or does it just add whipsaw + cost?**

## 2. The Base (locked SMOOTHEST core — unchanged)

- **Universe:** PIT mid-cap liquidity band (rank 101–250), survivorship-free, monthly.
- **Signal:** RS-120 vs NIFTYBEES; q0.5 price-path quality; ATH≤10% entry.
- **Hold:** top 15 equal-weight, monthly rotation, top-22 buffer.
- **Stock-level exits:** per-stock 100-SMA + 12% trail, at month-ends.
- **Market gate:** NIFTYBEES vs its 100-day SMA, checked **WEEKLY** (Phase-15 lock).
- **Risk-off action:** `allcash` (sell all) OR `keeptop8` (keep 8 strongest, cash 7).
- **Costs:** 0.4% round-trip on turnover; 6.5% p.a. idle cash; post-tax = 20% STCG.
- **Window:** 2014-01-01 → 2026, daily-marked drawdown. Data: VPS `market_data.db`.

**The ONE thing changed:** *when* the cash gets redeployed after a de-risk —
month-end (baseline) vs the first risk-on week (new).

## 3. Plan — 4 configs × {gross, post-tax@20%} = 8 runs

| # | Risk-off action | Re-entry timing |
|---|---|---|
| 1 | allcash | month-end (BASE — reproduces Phase 22) |
| 2 | allcash | **WEEKLY** |
| 3 | keeptop8 | month-end (BASE — reproduces Phase 22/24) |
| 4 | keeptop8 | **WEEKLY** |

**Sanity gate:** configs 1 & 3 must reproduce the published Phase 22/24 numbers
(allcash ≈ 34.2% CAGR / −22.2% DD; keeptop8 ≈ 33.6% / −20.2%). If they don't, the
refactor is unfaithful and the weekly-reentry delta is void.

**Success criterion:** a weekly-reentry variant *wins* only if it beats its
month-end twin on **post-tax CAGR AND/OR Calmar** without a material drawdown
blow-out. Report also: re-entry-event count, total fill count, and cost drag —
to see whether speed bought return or just churn.

## 4. Status (live log)

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-06-10 ~12:30 | STATUS written, runner staged | sections 1–4 locked pre-launch |
| 2026-06-10 ~12:35 | Launched on VPS (venv python) | first attempt died (system py, no numpy); relaunched OK |
| 2026-06-10 ~12:40 | DONE, 161s total | both BASE configs reproduced Phase 22/24 exactly → faithful |
| 2026-06-10 ~12:40 | RESULTS.md written, verdict SPLIT | allcash WK wins (Cal 1.72); keeptop8 WK loses (DD −28.5) |

## 5. Crash Recovery (resume without Claude)

- **Script:** `research/41_midsmall400_mq_concentrated/scripts/27_weekly_reentry.py` (on VPS)
- **Check progress:** `tail -f /tmp/phase27.log` — prints one line per config with CAGR/DD/Calmar/reentry-count.
- **Is it alive?** `ps aux | grep 27_weekly_reentry`
- **Re-run from scratch:** `cd /home/arun/quantifyd && nohup python3 research/41_midsmall400_mq_concentrated/scripts/27_weekly_reentry.py > /tmp/phase27.log 2>&1 &`
- **Output:** `research/41_.../results/phase27_weekly_reentry.csv` (+ `_peryear.csv`), written incrementally after each config — safe to inspect mid-run.
- **Do NOT touch:** `market_data.db`, scripts 01–26.

## 6. Files

| File | Purpose | Committable? |
|---|---|---|
| `scripts/27_weekly_reentry.py` | This runner | yes |
| `WEEKLY_REENTRY_VS_MONTHEND_DAILY_RUN_STATUS.md` | This file | yes |
| `results/phase27_weekly_reentry.csv` | 4-config summary | yes (small) |
| `results/phase27_weekly_reentry_peryear.csv` | per-year returns | yes (small) |
| `/tmp/phase27.log` | run log | no |

## 7. Findings

_Pending run completion._

## 8. Verdict

_Pending — RESULTS to follow with the honest read (does fast re-entry beat slow, net of tax/cost?)._
