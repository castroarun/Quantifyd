# SENSEX Manual Straddle Management — Forensic Reconstruction → Automated System (research/111)

**STATUS: PHASE 1 (forensic) — data snapshotted, reconstruction underway** · Started 2026-08-12 19:20 IST

## 1. Headline
Arun's 2026-08-12 manual SENSEX options day (DTE1, 13-AUG weekly): short straddle entered
post-ATR-squeeze at 12:03; the CALL side breached a ~30% per-leg SL twice in a trending
rally and was ROLLED UP each time ("range expanded"), puts rolled up behind the move, then
the book was TIGHTENED once the move spent itself. Net day ~ -Rs171 (breakeven) on a day
that would have badly hurt a static short straddle. Goal: codify this management as a
state-machine and test it across entry conditions on the recorded SENSEX chain (79 days).

## 2. The Ask (user, 2026-08-12)
"Read today's manual SENSEX options trading with respect to price movements. Entry was after
ATR squeeze but the main focus should be how positions were managed post 30% SL breach on
one side — the range is then expanded, then closed in after some confirmed bullish moves.
Structure this; we might automate it against different entry conditions — 9:16, ATR squeeze,
ATR+BB squeeze, time-based entries, staggered time-based entries — and study the sweet spot
of theta decay from our options database. Proper reports as a page like other systems."

## 3. The Base — what is being tested
- **Underlying/venue:** SENSEX weekly options (BFO), Thursday expiry; trade day was DTE1.
- **Recorded data:** options_data.db option_chain has SENSEX at full depth — 9.45M rows,
  79 days, 2026-04-20 → 2026-08-12 (same recorder as NIFTY); underlying_spot per-minute.
- **Trade record:** data/kite_snapshot_2026-08-12.json — full orderbook (24 SENSEX orders),
  tradebook (80 fills), positions. Snapshotted 19:17 IST before broker-day rollover.
- **Management state-machine to codify (v0 from the fills):**
  1. ENTER short ATM-ish straddle/strangle (size S).
  2. Per-leg SL: if a leg's premium rises >= ~30% over entry -> BUY it back (breach).
  3. On breach: EXPAND — re-sell the tested side 300-700 pts further OTM (roll up/down).
  4. Trail the untested side toward the move (roll it tighter, bank its decay).
  5. If the roll is breached again -> expand again (repeat).
  6. Once the move stalls ("confirmed" spent — define objectively: e.g. N candles without a
     new extreme / re-entry inside CPR-R1), CLOSE IN: roll the far side back tighter.
  7. Square off by close (or defined exit).
- **Success criterion:** does managed-short-vol beat (a) static straddle same entry, (b)
  straddle + hard 30% stop (no re-entry), on net P&L, win%, maxDD, worst-day, at stated qty?

## 4. Plan — phases & grid
| Phase | What | Output |
|---|---|---|
| 1 FORENSIC | Reconstruct 12-Aug trade vs SENSEX 5-min price, CPR, ATR/BB squeeze; annotated timeline + charts | results/FORENSIC.md + chart data |
| 2 CODIFY | State-machine spec + replay engine on recorded chain | scripts/engine |
| 3 BAKE-OFF | Entries: 09:16 / ATR squeeze / ATR+BB squeeze / fixed-time grid / staggered cascade x mgmt on-off | ranked table |
| 4 THETA | SENSEX EOD-decay by DTE + time-of-day (reuse NIFTY treatment) -> entry sweet spot | table+curves |
| 5 REPORT | /app page card + factsheet, wired into daily regen + leaderboard | page |
Grid sizing at phase 3; ~79 recorded days is the sample — single-regime caveat applies to all.

## 5. Status log
| Time (IST) | Event |
|---|---|
| 2026-08-12 19:17 | Kite snapshot captured (24 orders/80 fills/9 flat positions) — day P&L ~ -Rs171 |
| 2026-08-12 19:20 | Folder research/111_sensex_manual_mgmt created; this STATUS written |

## 6. Crash recovery
- Raw trade data: data/kite_snapshot_2026-08-12.json (IMMUTABLE — do not regenerate; Kite
  only serves same-day). All analysis is reproducible from it + options_data.db.
- Nothing long-running yet. Scripts land in scripts/, outputs in results/.

## 7. Files
| File | Purpose | Committable |
|---|---|---|
| data/kite_snapshot_2026-08-12.json | Raw broker record of the manual day | yes (small) |
| SENSEX_MANUAL_STRADDLE_MGMT_FORENSIC_STATUS.md | This doc | yes |

## 8. Findings so far (2026-08-12 evening)
- Manual day decoded: entry 12:03 near day low post-squeeze; CE breached ~+38% at 14:32 in the
  14:30-14:55 ~350pt rip; rolled up twice (78000/78100 -> 78400), puts trailed up, tightened after
  the stall; net ~ -Rs171 (breakeven on a violent V-day). 15:36 square-off = broker MIS auto-square.
- REPLAY (scripts/mgmt_replay.py -> results/mgmt_replay.json, 77 days, qty 100 = 5 lots, 09:20 entry):
  HARD30 -17,505 total (win 58%, maxDD -94,860) vs MANAGED -114,270 (win 53%, maxDD -164,890).
  MANAGED WINS ONLY ON DTE1 (Wed fat-tail): -2,048/day vs -4,038, win 56% vs 50%. Breach on 75% of
  days at 09:20. KNOWN BUG: STATIC arm duplicated HARD30 (breach branch leaked) — fix + rerun owed.
  BIG CAVEAT: manual entry was 12:03 post-squeeze, not 09:20 — entry-condition bake-off is decisive.
- NAS comparison (corrected after user challenge): trade-level stop-exits 9-68% by book; per-leg
  30% SLs touch MOST trades via orders (nas_916_atm: 55 SL_HIT orders/64 trades). nas_atm4 7,459
  SL_HIT order rows = logging churn, dedup before quoting.
- NAS PORTFOLIO (results/nas_portfolio.json, 6 books, 82 days): +384,635 total, mean +4,691/day,
  day-win 52%, worst day -125,769, maxDD -194,990. Avg pairwise corr 0.20 -> 59% DD reduction vs
  sum-of-DDs. 916 one-shots = ALL the profit (+4.53L, small DDs); all-day atm/atm2 lose -1.08L with
  -1.45L DDs -> recommend dropping them. Tail days still co-move (all short vol).
## 9. Next steps
1. Fix STATIC arm (skip breach check when mode==STATIC) + rerun (~1h detached).
2. Entry bake-off: 09:16 / ATR squeeze / ATR+BB squeeze / fixed-time grid / staggered.
3. SENSEX theta EOD-decay by DTE (reuse NIFTY treatment). 4. Report page + leaderboard entry.

## 10. ENTRY x EXIT x SL sweep — BEST CONFIGS (2026-08-13, 3-sec dwell, ~15d/cell)
NIFTY (10 lots): DTE0 09:30->11:00 SL25-30 (+198,967, win93%, DD-1,687, r117.9) | DTE1 13:00->14:00
(+56,848 r40.8) | DTE2 10:00->12:00 (+82,003 r74.4) | DTE3 FULL DAY any-SL (+169,968 r121.8) |
DTE4 10:30->12:00 (+46,810 r4.9 — Wed becomes tradable in the window).
SENSEX (5 lots): DTE0 FULL DAY no-SL (+204,435, win93%, DD-775, r263.8; live=wide 40-50% backstop) |
DTE1 10:30->12:00 (+25,785 r4.3) | DTE2 09:25->11:00 (+53,465, DD-30 = artifact, flag) |
DTE3 13:00->14:00 (+16,200 r9.8) | DTE4 10:30->12:00 (+22,590 r26.1).
KEY INSIGHT: the TIME WINDOW is the edge — SL level (20/25/30) rarely binds inside the windows;
time-boxed exits cut DD ~10-25x while keeping most of the profit. 30% stays as disaster backstop.
CAVEATS: ~1500-cell grid maxima on 15-16d cells = multiple-testing; validate OOS/paper before live.
Data: results/entry_exit_sweep.json + csl_3sec_dwell.json (full-days). Next: NAS comparisons w/
visuals (live-first) -> portfolio optimization -> hub page + factsheet + final conclusion.

## 11. STUDY CONCLUDED (2026-08-13) — hub live, paper validation running
All four final deliverables shipped: (1) best configs per index x DTE incl. entry/exit timing
(Lab, weekly Fri 15:45 refresh); (2) NAS-vs-CSL comparisons + charts; (3) portfolio scan
(optimal 2:0:1:1) -> actioned as PAPER books NIFTY 12 lots + SENSEX 6 lots (frozen 13-AUG
config, cron 09:12, dwell mechanic, 50% backstop on none-SL); (4) hub + conclusion on
/app/straddles (#hub). VERDICT: strong in-sample SIGNAL - the window is the edge; combined-SL
>> per-leg; schedule > stop-tuning; CSL x NAS corr ~0. STRATEGY decision ~mid-Sep from paper.


## 12. WALK-FORWARD HONESTY CHECK (2026-08-13) + full-schedule decision
Configs picked on Apr-Jun only, scored on unseen Jul-Aug (from per-cell series):
NIFTY book IS +434,670 -> OOS +203,954 (~-35%/day; DDs 5-10x bigger OOS).
SENSEX book IS +234,115 -> OOS +124,160 (~-45%/day).
=> The in-sample smoothness was selection-polish, but the edge SURVIVES OOS on both
indices. ROBUST cells: NIFTY DTE0 (09:30-11:00, OOS +67.7k dd -1.7k), NIFTY DTE3
full-day (OOS +74.1k dd -648), SENSEX DTE0 full-day (OOS +89.5k, 100% win, 0 DD).
FRAGILE: DTE1 both venues (SENSEX DTE1 OOS negative; NIFTY DTE1 win 93->50, unstable
window pick), SENSEX DTE3. Realistic paper-book bar: ~6.4k/day NIFTY + ~4k/day SENSEX.
USER DECISION: paper books trade the FULL frozen schedule (no de-rating) - data decides
at the ~mid-Sep checkpoint.


## 13. Command Center reorg + NAS-COMB20 book + CPR probe (2026-08-13 evening)
- Unified Strategy Leaderboard v2: ALL short-vol systems in one ranked table (straddle
  family + NAS 916x3 + SENSEX atm2 + CSL paper books when they trade) with per-row links
  to card / backtest report / tearsheet. NAS SENSEX atm2 ranks #2 (Calmar 5.7), 916_atm #3.
- Paper executor now runs 3 books: CSL_NIFTY 12L, CSL_SENSEX 6L, NAS_COMB20 3L (NIFTY
  09:16->15:20 combined-20% - the live A/B against nas_916_atm's per-leg mechanic).
- CPR width probe (prior-day CPR vs full-day straddle pnl, ~50d/index): NO actionable
  edge (corr +0.06/+0.15); NIFTY narrow-CPR win-rate tilt (83% vs 62%) on watch-list only.
  S/R-level interaction study deferred - data does not currently justify it.

