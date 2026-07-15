# EDGE_DISCOVERY_81_STUDY_STATE — research/81 Swing Edge Discovery (crash-recovery master)

> Study brief: `docs/Trading-sytem-research-prompt-fable.md` (mission, rules,
> acceptance gates). Doctrine: `research/QUANT_RESEARCH_PLAYBOOK.md` + `.claude/CLAUDE.md`.
> A fresh Claude session resumes from THIS file + those two. Canonical copy
> lives at VPS `/home/arun/quantifyd/EDGE_DISCOVERY_81_STUDY_STATE.md`; laptop copy is a mirror
> (laptop working dir is NOT a git checkout — author locally, scp to VPS, run on VPS).

**Last updated:** 2026-07-15 ~19:00 IST

## 1. Current phase & sub-task

**Phase 0 audit DONE (report: `research/81_swing_edge_discovery/reports/00_data_audit.md`)
— awaiting user sign-off.** Backfill RUNNING (launched ~18:46 IST, ETA ~17-19h,
monitor: `NIFTY500_HISTORY_BACKFILL_5MIN_RUN_STATUS.md`). Engine build starting
(`engine/costs.py` written).

**Audit verdict:** deep series usable. Repair queue (post-backfill):
(1) splice-inconsistent symbols (KOTAKBANK 401% fake jump found; full-universe
splice re-check after backfill) → delete + fresh re-download of 5-min series;
(2) 22 daily symbols with <95% session coverage incl. COALINDIA 40%, ONGC 55%
→ re-download daily; (3) loader must drop 2,550 nonpositive-price 5-min rows,
~160 OHLC-violation rows, out-of-hours bars, and <30-bar (muhurat) sessions.
321 |gap|>25% daily events across 182 symbols = unadjusted corporate actions
or circuits → corporate-action guard needed for any cash-equity strategy.

## 2. Study definition (locked with user, 2026-07-15)

- Mission: discover/validate/rank MANY automatable swing systems (hold ≤ 3–4
  trading days), net-of-cost, per brief §6 acceptance gates (Sharpe ≥1, Calmar ≥1,
  MaxDD ≤20%, WF-eff ≥0.5, ≥100 OOS trades, PF ≥1.3, expectancy >2× cost).
- **Data reality (audited 2026-07-15, VPS market_data.db):**
  - 5-min: 381 syms total — but 369 only from **2024-03-18**; 10 large-caps from
    2018; NIFTY50+INDIAVIX from 2015-02. NIFTY50 5-min stale (ends 2026-03-25);
    ~150 names stale at 2026-05-15. BANKNIFTY 5-min from 2024-03 only.
  - Daily: 1,642 syms from 2000 (deep). 60-min: 95 syms 2018→2026-03.
  - **No futures data.** User decision: treat cash series AS the futures proxy
    with a futures cost model (lot size, margin, futures STT) for F&O names.
- **User decisions:** (a) HYBRID data plan — backfill 5-min 2015→2024 from Kite
  on VPS while study proceeds on deep data; (b) cash-as-futures-proxy.
- Universe is survivorship-biased (today's names) → state in every report;
  headline claims prefer index/liquid F&O large-caps.

## 3. Experiments log

| ID | Family | What | Result file | Verdict |
|---|---|---|---|---|
| (none yet) | | | | |

**Total experiment count (multiple-testing ledger): 0**

## 4. In progress + how to resume

- **5-min history backfill (2015→2024, ~370 syms, ~17–19h):**
  status: `research/81_swing_edge_discovery/NIFTY500_HISTORY_BACKFILL_5MIN_RUN_STATUS.md`
  (has full resume commands + checkpoint path). Chained: forward catch-up
  (`scripts/backfill_market_data_vps.py --timeframe 5minute`) after it.
- **Phase 0 audit:** report goes to `research/81_swing_edge_discovery/reports/00_data_audit.md`.
  If interrupted: re-run `research/81_swing_edge_discovery/scripts/data_audit.py` on VPS.

## 5. OOS-touch ledger

Split convention: IS = first 60%, Val = next 20%, OOS = final 20% (chronological,
per series actually used). **No system has consumed its OOS look yet.**

## 6. Known issues / decisions

- Laptop `Covered_Calls` dir ≠ git repo (partial snapshot). All canonical work on
  VPS repo; scp files over; commit on VPS.
- 60-min table stale (ends 2026-03) and only 95 syms → prefer building 15/30/60-min
  and daily aggregates FROM 5-min (and daily table for pre-2015).
- Options strategies out of scope v1; ML out of scope v1 (interpretable rules only).
- App recording: publish per playbook §8.5 to `frontend/src/data/backtests.ts`
  (+ tearsheet PNG in `frontend/public/`), build on VPS (laptop frontend stale).

## 7. Next 3 actions

1. Launch backfill on VPS (smoke 1 symbol → nohup full run) + monitor.
2. Phase 0 audit on deep series → `reports/00_data_audit.md` → **pause for user sign-off**.
3. Build canonical backtest engine (`research/81_swing_edge_discovery/engine/`) + unit tests.
