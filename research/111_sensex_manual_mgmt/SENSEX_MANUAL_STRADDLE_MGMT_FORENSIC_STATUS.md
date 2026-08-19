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



---

## 14. Post-CSL management arms (2026-08-13 evening) — STATUS: RUNNING

**What you asked:** on NAS_COMB20, once the combined-SL triggers there is no trail — test
(a) closing only the hit side and TRAILING the surviving leg (ATM-style), and (b) SHIFTING
the straddle to the new ATM on CSL hit (ATM4-style rollout). Add as NEW versions, don't replace.

**Base:** NIFTY 09:16->15:20, combined-premium 20% SL, 2-snap dwell, 3 lots (qty 195),
raw ~3-sec chain, Rs160/cycle. Arms: BASE (close both, done) / TRAIL (close loser leg,
winner trailed: exit on >=30% bounce off its post-trigger low, dwell-2) / SHIFT (close both,
re-sell new ATM straddle, own 20% CSL, max 3 shifts/day, no new entry after 14:30).

**Plan:** replay all ~80 recorded NIFTY days -> results/csl_mgmt_replay.json; if an arm
beats BASE on ratio with sane DD, deploy as new paper books (NAS_C20_TRAIL / NAS_C20_SHIFT).
Runner: scripts/csl_mgmt_replay.py -> /tmp/csl_mgmt.log. Also this evening: books renamed
CSL_NIFTY->CSL_TIMEB_NIFTY, CSL_SENSEX->CSL_TIMEB_SENSEX (config/state/backfill migrated).

| 2026-08-13 20:0x IST | RE-FREEZE: books renamed CSL_*->CSL_TIMEB_*; all SL 'none' cells -> SL30 (user: never run stopless) | config migrated in place; backfill history re-syncs at next 15:40 regen |

### Sec 14 RESULT (2026-08-13 21:15 IST) — STATUS: DONE

NIFTY 09:16->15:20 CSL20, 3 lots, n=51 days (20-APR->13-AUG), SL triggered 10 days:

| Arm | Total | Mean/d | Win | MaxDD | Ratio |
|---|---|---|---|---|---|
| BASE (close both, done) | +115,939 | +2,273 | 71% | -15,953 | 7.3 |
| TRAIL (loser out, trail winner 30%-bounce) | +122,765 | +2,407 | 71% | -16,530 | 7.4 |
| SHIFT (re-center new ATM, max 3) | +133,864 | +2,625 | 75% | -27,507 | 4.9 |

Trigger-days-only: BASE -89,458 / TRAIL -82,632 / SHIFT -71,533. TRAIL beat BASE 6/10.
VERDICT: **SIGNAL, not conclusive** — n=10 trigger days. SHIFT recovers the most on bad
days but at ~1.7x the drawdown (ratio 4.9 vs 7.3 — risk-adjusted WORSE). TRAIL ~neutral.
Neither replaces BASE; the three books (NAS_COMB20 / NAS_C20_TRAIL / NAS_C20_SHIFT, 3 lots
each) run as live paper A/Bs from 14-AUG — let the data decide. Data lesson recorded:
options_data.db snapshot_time uses ISO 'T' separator; window predicates must use
INDEXED BY idx_oc_symbol_time + T-format bounds (substr() dodges the index).


---

## 15. NAS suite vs CSL-replacement vs HYBRID (2026-08-13 ~21:45) - STATUS: RUNNING

**Reframe (user):** goal is the best PORTFOLIO of complementary systems, not silo wins.
BASE = the live 3-system suite (ATM/ATM2/ATM4) as it trades today. COMB = per-leg SL +
trailing fully replaced by combined-SL X% (X swept 20/25/30/40, per-DTE view). HYBRID =
CSL as trigger, then each system's own management (ATM: loser out + trail winner to cost +
re-enter <=5; ATM4: roll to new ATM once; ATM2: both-out = COMB by construction).

**Key structural fact from config.py:** all three live systems are the SAME 09:16 ATM
straddle differing only in post-SL management (leg SL30; ATM trail-to-cost+re-enter,
ATM2 one-and-done + Rs2500/lot rupee stop, ATM4 roll-once). Full CSL replacement therefore
collapses the suite to 3 identical books - diversification dies; HYBRID preserves it.

**Plan:** replay all arms on identical recorded days (3-sec dwell, 15:15 exit, Rs160/cycle,
3 lots/system); suites SUITE_BASE / SUITE_COMB (=3x COMB) / SUITE_HYB (COMB+HYB_ATM+HYB_ATM4);
correlations; live actuals from nas_baseline.json as ground truth (portfolio stop caveat).
NIFTY first; SENSEX later. Runner: scripts/nas_suite_csl_replay.py -> /tmp/nas_suite.log.

### Sec 15 RESULT (2026-08-13 22:00 IST) - STATUS: DONE

n=51 replay days (20-APR->13-AUG), 3 lots/system, all arms on identical days.

Singles: BASE_ATM -20,367 (r-0.4) / BASE_ATM2 +31,779 (2.4) / BASE_ATM4 +34,525 (1.3);
COMB25 +126,198 (6.1) ~ COMB30 +130,955 (5.3) plateau; HYB_ATM~=COMB (corr .96);
HYB_ATM4 more total, worse DD. Suites (9L): SUITE_BASE +45,937 (0.6) / SUITE_COMB30
+392,865 (5.3) / SUITE_HYB30 +394,507 (5.7). LIVE actuals same window +98,940
(model omits portfolio stop -> modeled BASE pessimistic; live > model).

**Portfolio finding (the headline): corr(LIVE suite, COMB30) = 0.31.** The CSL sleeve is
genuinely complementary to the live suite; full replacement would also collapse the suite
to 3 identical books (all are the same ATM straddle differing only post-SL management).
LIVE + COMB30 (12L): +225,149, DD -41,921, ratio 5.4 vs LIVE alone 1.7.

Per-DTE apt CSL (in-sample, n=9-11/DTE): DTE0:25 DTE1:30 DTE2:30 DTE3:20(SL never binds)
DTE4:30 (NOTE: DTE4/Wed NEGATIVE at every SL - consistent with all prior research).
ACTION: NAS_COMB20 book re-frozen to per-DTE SLs 25/30/30/20/30 (user-delegated choice);
name kept for continuity. VERDICT: **SIGNAL** - keep live suite, grow the CSL sleeve as
the complement; paper books adjudicate OOS. SENSEX version: pending (NIFTY-first per user).

### Sec 15b - PER-DTE ELIMINATION CHECK (2026-08-13 22:15 IST) - user hypothesis VALIDATED

User: a system inferior overall may be superior once a weak DTE is eliminated -> then no
replacement needed. Result: **every arm's worst DTE is DTE4/Wednesday** (live suite lost
-58,701 on 9 Wednesdays at 3L/system-normalized). Ex-Wednesday, the LIVE suite jumps from
ratio 1.7 -> 6.2 (+157,640/34d) - COMPETITIVE with COMB25 (8.0) / COMB30 (6.5): the live
suite does NOT need mechanic replacement, it needs to not trade NIFTY Wednesdays.
Standout: HYB_ATM4_30 ex-Wed ratio 15.4 (+168,072, dd -10,928, 3L). Portfolio
LIVE(9L)+COMB30(3L) ex-Wed: +289,765, dd -19,381, ratio 15.0 (n=34).
Consistent with research/79+90 doctrine (NIFTY edge Mon/Tue, SENSEX covers Wed/Thu).
Caveat: n=9-11 Wednesdays, in-sample. Script: scripts/per_dte_elimination_check.py.
RECOMMENDATION (pending user): gate NIFTY books out of DTE4/Wed (day-matrix already
exists for live suite); keep paper books trading all DTEs for OOS evidence.


## 16. OPTIONS PORTFOLIO LAB (2026-08-13 ~23:00) - the convergence point

Where the research landed: **THE STACK = LIVE suite (9L, Mon/Tue-gated) + COMB sleeve
(NAS_COMB20 3L, per-DTE CSL) + TB-CSL (3L-scaled)** - avg cross-corr ~0.26, ratio 7.6
all-days (study window). Ex-Wed variants tracked. SHIFT candidate (HYB_ATM4_30, ex-Wed
ratio 15.4 in-sample) rides as MODEL until NAS_C20_SHIFT accrues live paper days.
Recorded as a LIVING page section: /app/straddles#portfolio-lab, regenerated daily 15:40
by scripts/portfolio_lab.py (live-first merge: paper records override backfill; source
mix reported per component). Corr matrix + all-days/ex-Wed portfolio rows on-page.


## 17. Portfolio SL / profit-trail overlay on THE STACK sleeves (2026-08-14 ~00:15) - NO, keep as-is

User asked whether THE STACK needs a portfolio-level SL or profit trail like the live system.
Tested on synchronized per-minute sleeve curves (COMB 3L + TB-CSL 3L-scaled, backfill, n=51/40):
**every overlay is worse than none.** Ex-Wed: none +259,223 (ratio 24.1) vs pstop-1300/lot
+204,936 (12.1) vs best trail +178,201 (21.8). Mechanisms: portfolio stop double-stops
(fires mid-dip while one book's CSL is already handling it - MaxDD WORSENS); profit trail
amputates back-loaded theta afternoons (win% up, total down 80-130k). The live suite keeps
its research/90 stop (it compensates for leaky per-leg SLs); the sleeves' CSL+windows+Wed-gate
IS the risk system. All-stack overlay untestable yet (no LIVE intraday history; nas_mtm
snapshots accruing since 2026-08-11 - revisit in 4-6 weeks). Caveats: minute-mark fills,
modeled backfill curves. Script: scripts/sleeve_pstop_test.py.


## 18. TB-CSL overweight test (2026-08-14) - idea SOUND, sizing STAGED

User hypothesis: overweight TB-CSL (time-boxed, ~1-2h exposure vs 6h) vs others. In-sample
answer is emphatic (ex-Wed, 34d: 6/2/2 ratio 20.8 -> 4/2/4 = 31.6 -> 2/2/6 = 52.7; additive
TB@6L ratio 31.0 with DD barely moving) BUT TB's history is the in-sample-optimized champion:
its windows were CHOSEN on this data (walk-forward halves the edge), its worst recorded day
(-1,398 @2L) understates a real SL20 window hit (~-4-5k @2L model), and it has ZERO live days.
DECISION PATH: keep 6/2/2 through the ~4-week live soak; if TB's REAL days confirm the
structural profile, step 5/2/3 -> 4/2/4 fortnightly. Genuine structural point: TB margin is
consumed only ~2h/day (capital efficiency real regardless). Script: scripts/tb_overweight_test.py.

### Sec 18b - COMB x TB overweight GRID (2026-08-14 ~14:30) - TB is the lever, COMB is not

Grid (LIVE 6L fixed, ex-Wed n=34, NET/DD/ratio): TB axis RAISES ratio monotonically
(20.8 -> 26.3 -> 31.0 at COMB@2); COMB axis LOWERS it (20.8 -> 15.5 -> 13.3 at TB@2).
Marginal +2L from 6/2/2: TB +66,183 net for -72 DD; COMB +71,796 net for -8,669 DD
(~120x the DD cost for the same profit). Cause: COMB is full-day, corr 0.31-0.37 to LIVE
(bad days coincide); TB windowed, corr 0.18 (TB@6 even HEDGES COMB@4: DD -20,336 -> -17,400).
Best cell 6/2/6 = +375,440 / -12,130 / ratio 31.0. In-sample caveat strongest on TB
(worst recorded day -391 @2L vs ~-4-5k modeled SL hit). Lab rows: TB@4L + TB@6L stacks
live on /app/straddles#portfolio-lab (daily live-first regen). Sizing decision: PENDING user.
Script: scripts/comb_tb_overweight_grid.py.

**Sec 18b DECISION (2026-08-14 14:4x IST, user): stack resized to 6/2/6 (TB-CSL -> 6 lots, 14L total).** Instructions: docs/STACK_TB6_RESIZE_DEPLOY_STATUS.md (execute in build-context session; effective Mon 17-AUG).


## 19. Spot-ATM vs Forward-ATM strike selection (2026-08-19) - STATUS: RUNNING

Live observation (TB-SENSEX 10:30 entry): CE-PE skew 124 vs spot-K +40 -> implied forward
~84pts above cash; fills verified mid-market, spot-ATM selection correct per mechanic.
Question: would selecting K nearest the SYNTHETIC FORWARD (parity at the spot-ATM strike)
improve the books? Replay of frozen configs (TB-N 8L, TB-SX 8L, COMB 2L) on the recorded
chain, both arms, same windows/dwell-SL/Rs160. Runner: scripts/sec19_fwd_atm.py.

### Sec 19 RESULT (2026-08-19) - STATUS: DONE - KEEP SPOT-ATM (current mechanic validated)

| Book | n | K differs | basis mean/p90 | SPOT-ATM | FWD-ATM |
|---|---|---|---|---|---|
| TB-NIFTY 8L | 60 | 16 | +8.6/+42 | +423,456 dd -4,328 r97.8 | +386,224 dd -17,744 r21.8 |
| TB-SENSEX 8L | 81 | 42 | +45.9/+114 | +333,680 dd -45,304 r7.4 | +312,864 dd -48,816 r6.4 |
| COMB 2L | 43 | 9 | +6.5/+37 | +111,908 r13.9 | +115,522 r14.3 (noise) |

Even on SENSEX - where the forward premium is largest and K differs on HALF the days -
spot-ATM beats forward-ATM on totals, DD and ratio; on NIFTY forward-ATM quadruples the
DD. Working hypothesis: the basis decays intraday, and the spot-centered straddle is
short the basis-rich call side, harvesting that decay; forward-centering gives it away.
NO mechanic change. Live fills note (10:30 TB-SX): CE 282.92/PE 158.70 both mid-market
vs recorded chain - BFO marketable-LIMIT slippage cleared. Caveat: differing-day n=16/42/9,
in-sample. Script: scripts/sec19_fwd_atm.py; results/sec19_fwd_atm.json.

### Sec 19b (2026-08-19) - SX-assessment first run catches Thursday SL config; re-frozen SL30->SL40

New SENSEX assessment section flagged SX-TB-WINDOWS: frozen Thu (09:20-15:20 SL30) at 1%% of
sweep-best. Cells (16 Thursdays, 5L): SLnone +229,155/dd-775/r295.7/win94%% vs SL30
+81,040/dd-34,630/r2.3/win62%% - expiry gamma noise trips every pct stop, then premium decays
(the stop-by-DTE HOLD finding, quantified). USER DECISION: SL40 compromise (+119,960/r4.0)
- keeps a pct stop per never-stopless preference while halving the noise-trips vs SL30.
Effective 2026-08-20 09:12 (first real Thursday). 50%% disaster backstop unchanged.
Also first-run: SX-CROSS-CORR full -0.04 (excellent diversification) but recent15 0.47 - watch.

**Sec 19c (2026-08-19): NIFTY-Thu (DTE3) SL20 QUESTIONED AND KEPT (user).** SL20/25/30/40 historically identical on NIFTY Thursdays (no stop ever fired; SL20 was a tie-break). Loosening to 30/40 offered for OOS robustness (post the SENSEX-Thu noise-trip lesson); user decision: keep SL20 as frozen. Applies to TB-NIFTY DTE3 + NAS_COMB20 DTE3. No config change.
