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
| EXP-A1 | A momentum | Donchian N-day breakout, daily F&O, 2-4d hold, 24 cells, IS 05-17 | `experiments/A1_donchian_daily/results/a1_ranking.csv` | **NO EDGE** (all cells net-neg; longs gross-neg t≈−8..−10; best short gross +4-6bps < 9.6bps cost) |
| EXP-B1 | B mean-rev | z-score dip-buy/rip-fade, daily F&O, 2-4d hold, 16 cells, IS 05-17 | `experiments/B1_zscore_daily/results/b1_ranking.csv` | **WEAK SIGNAL** — gate t≥3 missed (best +31.7bps net, t=1.5) but monotone in z, sma20-target dominant, short side > long |
| EXP-B2 | B mean-rev | deeper z {2.5,3.0} follow-up + per-year regime check, 8 cells | `experiments/B2_deep_z_daily/results/b2_ranking.csv` | **SIGNAL (not investable)** — z3.0 plateaus; per-year clean (8-10/13 yrs positive, NOT a 2008 artifact); max t=1.5 dies to multiple-testing haircut. Family B daily CLOSED; 5-min timing + sleeve candidacy recorded as next levers |
| EXP-C1 | C volatility | BB-squeeze→breakout daily, 16 cells | `experiments/C1_squeeze_daily/results/c1_ranking.csv` | **NO EDGE** — 4 positive cells but spike-not-plateau (longs only ts4, shorts only ts2); fails stability |
| EXP-D1 | D intraday→swing | EOD strength (CLV) carry, 16 cells | `experiments/D1_eod_strength_daily/results/d1_ranking.csv` | **NO EDGE** — longs negative gross; best short +6.6bps t=1.2 |
| EXP-E1 | E cross-sectional | 5-day RS decile rotation, fresh entrants, 16 cells | `experiments/E1_xsec_rs_daily/results/e1_ranking.csv` | **NO EDGE** — all 16 cells net-negative both directions/skips |

**Total experiment count (multiple-testing ledger): 96 cells (A1 24, B1 16, B2 8, C1 16, D1 16, E1 16)**

**Meta-finding (daily variants):** on daily bars with forced 2–4 session exits,
the F&O universe offers at best a weak deep-reversion short-side signal.
Breakout/momentum needs longer holds than the mandate allows; rotation/reversal
is cost-eaten. The study's remaining hope: 5-min families (F index ORB, A2
stock ORB, B-timing refinement) where entry precision is worth bps and index
futures carry the lowest cost.

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
