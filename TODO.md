# Covered_Calls — TODO

Cross-session source of truth for pending work. Each item: what / why / when.

## ⏳ 2026-08-31 — research/138 phase 2: the live book, and COMB20's day allocation

**Two decisions are owed from Arun. Nothing was deployed.**

**Correction first:** every portfolio number I gave earlier in that session pooled the
live systems with 13 paper shadows. The real-money book is **7 sleeves over 43 days:
+₹2,38,557, maxDD −₹36,082, return/DD 6.61, t 2.32** — healthier than the pooled
−₹5.99L / 2.05 I reported. Live roster comes from `nas_day_matrix.json` (`live:true`
+ enabled DTEs) and `csl_paper_exec.py` `BOOKS` (`"mode":"live"`): the three NIFTY 9:16
systems (Mon+Tue), the three SENSEX systems (Wed+Thu), and `NAS_COMB20` (2 lots, Mon+Tue).

**Also retracted:** an intermediate claim that Friday was the best untraded day
(+₹2.86L, t 2.18) was a **lot-sizing artefact** — raw sums over a shadow running 1 to
10 lots. At a constant 10 lots those Fridays are +₹83,407 at **t 0.41**, with three days
carrying more than the whole total. Noise, not an opportunity. (Consistent with the
already-overdue ops review "Suite FRIDAY (DTE2) — keep live or revert".)

**The finding.** `NAS_COMB20` is the only live sleeve losing money (−₹10,089, t −1.00),
and the cause is **which days it trades**, not its parameters. The 9:16 suite and the
held-straddle/combined-stop mechanic have near-opposite weekday edges:

| | Mon | Tue | Wed | Thu | Fri |
|---|---:|---:|---:|---:|---:|
| 9:16 suite (t) | **2.89** | −0.24 | 0.50 | −0.37 | 0.41 |
| held straddle + CSL (t) | 1.23 | 1.85 | 0.35 | **3.85** | 0.94 |

The live config runs **both on Mon+Tue**. COMB20 therefore stacks correlated size onto
Monday — the suite's strongest day, which needs no help — and sits out Thursday, its own
strongest cell *and* a day the suite loses on. Live Monday record: 3 sessions, all
losing, −₹56,630 at 10-lot equivalent.

**Thursday is settled as a day, not a stop.** Re-priced from the full recorded premium
path, DTE3 is *identical* from SL20 through no stop at all (₹1,55,265, t 3.85, DD
−₹15,790, **0/18 stops fired**). The paper twin `NAS_COMB20_THU` already runs SL20 =
the same thing. Only **size** and **margin** separate it from the headline.

**DECISION 1 — COMB20's Monday cell: keep, shrink or stop?** It has lost all three live
Mondays and duplicates the suite's best day. The replay says Monday is mildly positive at
every stop (t 1.0–1.6) but with ~4× Thursday's drawdown, and its edge is almost entirely
in the first half of the sample (₹1,31,845 → ₹12,960 across halves). No parameter fixes it.

**DECISION 2 — Thursday: wait, or move now?** Recommendation is **wait**. NIFTY was pulled
off Thursdays on 2026-08-27 for SENSEX-expiry margin and that constraint stands; the live
Thursday record is 2 sessions and negative. Let `NAS_COMB20_THU` reach ~8 paper Thursdays
(already registered: ops review due 2026-10-30, now carrying this evidence).

Either change is a **strategy change** — its own STATUS-MD, its own evidence, an
after-15:40 deploy. Study: `research/138_comb20_dte_allocation/results/RESULTS.md`.

## ⏳ 2026-08-31 — Confirm which CSL/COMB books are really live (flag vs comments)

`is_live_book()` requires `"mode": "live"`; **only `NAS_COMB20` has it**. Comments assert
`CSL_TIMEB_SENSEX` and `CSL_TIMEB2_LIVE` are REAL money. By that function both evaluate as
paper and `/app/straddles` renders them as paper. Either they execute via another path, or
two books believed live are not trading real money. **Not edited** — live-trading code.
Check the broker tradebook for a fill tagged to either. Ops review due **2026-09-05**.


## ✅ 2026-08-30 — research/135 Turtle system: tested + optimized → CONCLUDED, NO DEPLOY

Arun sent the classic Dennis/Covel Turtle rules (5-rule breakout system) and asked to test them on
our stocks data, then optimize for CAGR/Calmar, then fold in the momentum book's gate + put buying
and compare everything. Done end-to-end on the VPS. **No action owed — logged for the record.**

**What we found**

- The attached spec **taken literally is the worst book in this engagement**: 1.67% CAGR,
  −67.9% MaxDD, Calmar 0.02 (2005–2026 net). Loses to NIFTYBEES by ~11 points of CAGR at
  more drawdown, and spends 4,226 days (11+ yrs) below its prior peak.
- **The optimization is entirely subtractive.** Drop Rule 3 (2N stop — the single most damaging
  rule; mean Calmar none 0.65 > 3N 0.51 > 2N 0.46 > 1.5N 0.45, monotone, and removing it improves
  drawdown too), drop Rule 4 (pyramiding: with the ratcheting stop, Calmar 0.47→0.08 as units
  1→4), drop Rule 2 (N-sizing loses to equal-notional — **4th** independent time). Keep
  Rule 1/5 at the original 20/10. Result: Calmar 0.02 → 0.53, CAGR 1.7% → 15.9%.
- **OOS 2024–2026 (held out, consumed once) is NEGATIVE for every Turtle arm** — optimized
  −8.3% CAGR vs benchmark +5.3%. Era means decay monotonically: +27.4% → +14.8% → −5.4%.
- **CORRECTED after Arun challenged the chart:** the momentum book beats every Turtle variant
  in **every** era — 31.78% CAGR vs 15.97% at identical drawdown, 299x vs 21x, and **+21.0%/yr
  in the held-out window** where every Turtle arm loses. The first momentum arm (12.58%) was a
  broken hand-rolled reconstruction (wrong universe, live-book stop bolted on, idle-cash bug);
  Stage G drives research/75's own runner and reproduces its published 31.9% to 0.1pt.
- **The two systems want OPPOSITE universes:** momentum gains +11.6 pts of CAGR moving from the
  78 F&O large caps to the PIT top-250; the Turtle gets *worse* there (Calmar 0.50 -> 0.24).
- Momentum **gate is era-unstable** (200DMA best IS, 100SMA best VAL — no stable winner).
  **Put overlay does not rescue the book**; matches the live momentum book's own
  `hedge_enabled=False` note. One stable sub-finding: **5%-OTM beats ATM in every pairing.**

**Reusable lessons banked** — hard stops on multi-week equity trend books cost return AND add
drawdown; pyramiding tied to a ratcheting stop is actively destructive; equal-notional is settled
(stop re-testing vol-sizing); 5%-OTM index puts dominate ATM for hedging.

**Process note:** a put-overlay bug (premium expensed AND its decay marked from the same level —
double-counting ~1.8% of NAV per roll, producing absurd −99.9% results) was caught by a hand-check
and fixed mid-study; invalid output retained as `_INVALID_stage_E_premium_doublecount.csv`.

Docs: `research/135_turtle_optimization/results/RESULTS.md` · STATUS doc in same folder ·
`research/INDEX.md` row 135 · report artifact published.

---

## 🔴 2026-08-28 — Stored 5-min FIRST BAR of the day is wrong on ~half of sessions (NOT FIXED)
Found while validating the N500M CCRB fix. The first 5-minute bar of each session, as stored in
`market_data.db`, disagrees with Kite's final value on roughly half of sessions:

| check | DLF | HDFCBANK | ITC |
|---|---|---|---|
| API daily open == API first-5min open | 30/30 | 30/30 | 30/30 |
| DB daily open == API daily open | 29/30 | 29/30 | 29/30 |
| **DB first-5min == API first-5min** | **14/30** | **17/30** | **19/30** |

So the daily bars are fine and the underlying relationship is exact — it is the stored 5-minute
series that is off, by 0.1–0.6%, systematically on the first bar.

**Likely mechanism:** `refresh_5min` runs every 5 minutes and captures the 09:15 candle while it
is still forming; `data_manager._store_data` then inserts only timestamps it does not already
hold (`df_new = df_insert[~df_insert['date'].isin(existing_dates)]`) and never replaces one, so
the partial values are frozen permanently. The module docstring claimed "Idempotent — uses
INSERT OR REPLACE", which is not what the code does (docstring corrected 08-28, no behaviour
change).

**Who reads this:** vol-BO (the half of N500M that actually trades), ORB, and every intraday
backtest run off `market_data.db`. Any signal keyed on the opening candle is affected.

**Fix (own change, own testing, after-15:40 deploy):** make the writer correct a row when the
newly fetched candle differs, or refuse to store the current (incomplete) candle at all and only
persist bars whose window has closed. Prefer the latter — simpler, and never writes a value it
will have to take back. Then backfill/repair the historical first bars from the API.

Evidence: `research/N500M_CCRB_DEAD_RULES_FORENSIC_STATUS.md` §6.


## 🔴 2026-08-28 — N500M: half the book (15 CCRB rules) has NEVER fired, since May
Investigated "why has N500M not traded since 20 Aug". Infrastructure is healthy — all four
jobs run on time, data refresh 27/27 — but **every one of the 15 CCRB rules is skipped every
day, and always has been**: 1,230 daily-state rows, 100% `setup_reason='skip:no_setup_row'`,
and all 32 trades the book has ever taken are `volbo`.

**Root cause:** CCRB's setup gate needs a row keyed on *today* in `daily_setup_table`, which
needs today's **daily** bar. `services/market_data_refresh.py` refreshes `timeframe="5minute"`
only, so `market_data.db` has no `day` bar for today at any point in the session (verified
mid-session: DLF/HDFCBANK/ITC latest `day` = 2026-08-27, latest `5minute` = today 12:40).
`today_setup` is therefore always `None` → `skip:no_setup_row` → CCRB can never fire.
vol-BO is unaffected: its gate runs intraday off 5-minute bars.
(First hypothesis — "precompute at 09:10 is before the 09:15 open" — was REFUTED: the bar is
missing mid-session too.)

**Fix (NOT applied; needs after-15:40 deploy):** `daily_setup_table` needs exactly one field
from today's bar — `today["open"]` — and that is already available as the open of today's
first 5-minute bar. Preferred: synthesise today's daily row from the first 5-min bar and move
the CCRB precompute to ~09:20. Alternatives: add `day` to the intraday refresher; or recompute
lazily inside the scan.

**Treat as a strategy change, not just a bug fix** — it switches on 15 rules with no live or
paper record. Before deploying: (a) verify the synthesised open matches the true daily open
over ~30 sessions, (b) replay the CCRB gate over 3 months to see the would-have-qualified rate
against the bake-off's expectation, (c) deploy after 15:40 and watch the first signals.

Full evidence: `research/N500M_CCRB_DEAD_RULES_FORENSIC_STATUS.md`.

## ⚠️ 2026-08-28 12:22:45 IST — quantifyd restarted DURING market hours (not this session)
`systemd: Stopping quantifyd.service … Started` at 12:22:45, `NRestarts=0` (deliberate, not a
crash). Not from this session — it was doing frontend builds, a static-feed write and a git
push; there is **no auto-deploy/pull cron** on the box, so a push cannot cause a restart. The
reflog shows other sessions committing at 12:06, 12:25 and 12:45 (holdings/Dad-account work),
so the restart came from one of those or by hand.
**Consequence checked:** the live NAS-ATM2 leg (NIFTY2690124250PE) was under active SL
monitoring at 12:15; after the restart the monitor picked it back up and was still ticking at
12:46, so nothing was left unwatched. Flagging it because it breaks the no-restart-before-15:40
rule, and concurrent sessions each need to honour it.


## ✅ 2026-08-26 — 'silent' paper books audited: three were fine, the projection was wrong
Checked every book that read as idle or never-traded on /app/strategies and /app/overview.
- **Pairs — WORKING.** Daily 16:00 scan runs in paper mode, 6 pairs evaluated. Two OPEN cohorts:
  COFORGE-HCLTECH (short, entry z 2.10, 08-20) and BAJFINANCE-KOTAKBANK (long, z −2.31, 08-21).
  NAV ₹9.81L, unrealised −₹18,552. Zero closed trades yet — hence the false 'never traded'.
- **Breakout ₹10L — WORKING, gate-blocked.** One entry ever: NAVINFLUOR ₹1.25L on 08-06, still held,
  unrealised −₹1,402, NAV ₹10.06L. Latest NAV rows show `gate: OFF`, i.e. NIFTY below the 200-DMA
  gate, so no new entries — correct behaviour, not a fault. Weekly cadence: mid-week runs only
  update cash/NAV.
- **MST — enabled + paper, flat, no signal.** Boots `enabled=True paper_mode=True state=NO_POSITION`;
  all 10 legacy legs closed by STALE_CLEARED_20260817. No flip since. ⚠️ Its 30-min seed buffer
  ends **2026-07-16** (`[MST] Seeded 250 historical 30-min bars (last=2026-07-16 15:15:00)`) — worth
  confirming that a 5-week-old seed plus live 5-min-derived bars gives a sane indicator window.
  Review already due 2026-09-01.
- **NWV — idle 23 days**, weekly jobs registered (Sun 22:00 state, Mon 09:46 view). Not yet explained;
  next item to look at.
- **ORB Cash — RECOVERED.** 85 closed trades, 39 in 30d, traded today, flat at close. Net −₹27,773.

**FIXED (commit `288a924`):** `services/book_liveness.py` counted closed trades only, so any book
holding a position read as dead and was dropped from the Overview table. It now also counts open
positions per book; `days_idle` keeps its meaning (days since last exit), new `days_since_activity`
counts an entry too, and the pages show `holds N`. Read-only projection — no engine touched.
Frontend built and live; **the API field arrives with the 09:00 pre-open restart** (no manual
restart taken: the book was flat on options, 4 manual CNC holdings only).

**Still open from this pass:** (a) why NWV has not traded in 23 days; (b) the MST stale-seed question above.

## ⏳ 2026-08-25 — research/127 STRATEGY-candidate: stock 45→21 DTE winged strangle — next: margin check, then paper book
One universal ruleset across ~80 F&O stocks (real NSE bhav EOD): sell ±2.5% strangle at
45 DTE, buy 7% wings, NO stop, TP 50%, exit 21 DTE, ATM vol≥100 + wings traded. Net
+0.264%S0/trade (t=5.06, n=628); portfolio 39% CAGR dense-era at modeled margin, 21%/−10%DD
at 2× margin; corr to NIFTY −0.09 (+EV in NIFTY crash months). G3 PASSED (super-winner, OOS,
liquidity-monotone, DTE placebo 35/55≈0, lag). Full verdict research/127_stock_neutral_wings/
results/RESULTS.md (commit 964753e). **PAPER BOOK LIVE 2026-08-25 evening** at /app/stock-wings (services/stock_wings_paper.py,
cron 16:20 bhav-stocks + 16:50/20:30 seed-mark; seeded from 01-Jun: 18 replayed closes,
10 open Sep-29 positions, NAV Rs20.41L). Strategies index + Ops Center + LABS ref updated.
**Open items:** (1) real margin check via Kite basket
margin API (G4 full pass gate); (2) paper book on top-liquidity tier (5-10 slots); (3) earnings
-date source → test earnings-skip; (4) tearsheet + publish to /app/backtest registry.

## 2026-08-25 - 45-DTE straddle paper book: VIX-rank filter ADOPTED + LIQUID1 idle sweep

**Filter is now the plan: India VIX PERCENTILE RANK > 25** vs the previous 252 sessions - NOT a VIX
level of 25 (that would be 7 of 89 entries; the rank keeps 61 of 89). Study basis: Calmar 1.09 vs
0.66 unfiltered, zero losing years vs one, worst trade halved (-Rs 1.58L -> -Rs 88k at 3 lots), for
Rs 1.1L less profit over 7.5 years.

Arun's call: sub-threshold campaigns are **still paper-traded but tagged OFF-PLAN**, so the filter's
value is measured live rather than assumed. The open Sep-29 position entered at rank 22.6 and is
retained with that tag. Book now reports NAV "as traded" AND "on plan" side by side.
All 3 closed trades were ON-PLAN, so realised P&L is unchanged at Rs 1,31,704.

**Idle cash -> LIQUID1** (Kotak Nifty 1D Rate Liquid ETF). Chosen over 17 other NSE liquid ETFs:
highest measured yield (5.11%) and 3x the depth of any other GROWTH-structure liquid ETF not already
in use (LIQUIDCASE is pledged, CASHIETF is Momentum's). The Rs 1000.00-pinned ones (LIQUIDBEES etc)
are the daily-dividend model the project already avoids. Accrual uses the ETF's REAL close-to-close
prices over each flat span - Rs 3,544 earned over 3 spans so far.

**Also fixed a latent bug:** the monthly-expiry picker would have chosen the 31-Dec-26 legacy Thursday
contract (listed 2024) over the real 29-Dec-26 Tuesday monthly, and would have fired mid-Nov. Now
guarded by the prevailing monthly weekday, derived from data rather than hardcoded.

**STILL OPEN:** (1) stress-margin test - the blocking item for live (Ops review 2026-09-30);
(2) the live mark does NOT evaluate target/stop intraday - exits are only tested on EOD closes, which
contradicts the hourly-monitoring recommendation. Small fix, not yet done.


## 2026-08-24 - Monday dropped from live TimeB, and the Best-Config Lab now reports NET

**Monday is dark live; Friday stays.** r/122's atlas condemned the Monday cell (R:R@p95 1:11.8,
modelled P(loss) 52%) as the third independent study to do so. Arun dropped Mon, briefly dropped
Fri too, then kept Friday on its KEEP verdict (93% win, 1:6.9). Live TB-NIFTY = **Tue DTE0 +
Fri DTE2**. The Monday cell keeps trading on PAPER as `CSL_TIMEB_NIFTY_MON` (8L) so the Nov
re-run has evidence. Thursday SENSEX 10-lot bump **DECLINED** - stays 8 lots (the p95 tail
~Rs38k at 8L was the deciding number). Commit 00d62a3.

**research/124 - can Monday be rehabilitated? NO EDGE, CONCLUDED.** Swept 3,014 cells (137
windows <=120min x 11 stop arms incl. rupee stops) on 17 Mondays + the 2015-> calm-zone study.
137 cells passed the raw screen; **zero survived Westfall-Young + label-shuffle**. The best cell
(+5,880@8L) sits at p=0.33 against best-of-2,000-random-draws (null-95 = +7,280) - i.e. it is
what mining noise produces. Monday is the WIDEST morning of the week, not the calmest; the only
calm zone (lunch) earns nothing. Kill-sheet: revisit 09:16-11:16 + Rs1,000 rupee stop at >=40
recorded Mondays. Commits f17a6e9, a9498a2.

**The Best-Config Lab was near-GROSS - fixed.** It charged a flat COST=160 per straddle round
trip (Rs16/lot on 10 NIFTY lots): brokerage scale only, ZERO slippage, fills at observed chain
prices. Monday DTE1 read +3,703/day 94% win; net of real costs it is **+1,363/day 75%**, which
reconciles with r/122 (+1,200 median @8L) and r/124 (+992). Cost is now venue/size-aware
(0.5pt/leg-side x4 + Rs30/leg-side/lot = Rs2,500 NIFTY, Rs800 SENSEX), the panel STATES the
basis, and every DTE carries a 0.25/0.50/1.00pt sensitivity band. **Re-costed live rows:
Tue +10,164 · Fri +3,122 · Thu +13,270 · Wed +367 (was +2,708 - effectively dead).**
Commit 15edff9; full sweep regen ran after 15:40 via scripts/deferred_lab_regen.sh.

**NAS watchdog was mailing ~12x/day about PAPER books - fixed.** (1) No check looked at leg mode,
so the paper squeeze books raised naked-leg FAILs on positions with no broker order, and day-P&L
vs max-loss summed paper + live. Every check is now live-only. (2) The de-dupe hashed the full
detail text, which carries live ST/SL values - each 5-min poll looked like a NEW fail. Signature
is now the failure identity with numbers stripped. Verified 13 OK / 0 fail. Commit 3e1288d.

**Live NAS today: flat, -Rs3,601** (916-ATM +2,132 · ATM2 -5,252 · ATM4 -481) - well inside the
-Rs7,800 venue floor, portfolio stop never armed. Both naked survivors exited ST_EXIT @15.70,
which also CONFIRMS the a792136 trail-counter fix fires live.

**PENDING - needs Arun's call:** the 1.0x lot bump, reshaped by the Monday drop. Atlas says Tue
8->10L is earned (R:R@p95 1:1.5); TB-SENSEX Wed 8L / Thu 5L unchanged unless he says otherwise.
Also still open: unified per-system position ledger (was due today).


## 2026-08-24 - 45-DTE straddle now runs as a 3-lot PAPER book (/app/straddle45)

`services/straddle45_paper.py` + `backtest_data/straddle45_paper.db`. Seeds completed campaigns
from REAL NSE bhavcopy closes and marks the open position from the broker. Publishes a static
JSON (`/app/straddle45_paper.json`) so the page needs NO API route and NO backend restart.

State at open: **3 closed, realised +Rs 1,31,704** (2026-05-15/06-09 +453.8 pts, 2026-06-12/07-07
+31.0, 2026-07-10/08-04 +190.6 - all TIME_21DTE) and **1 LIVE: Sep-29 monthly, entered 14-Aug @
24350, credit 749.7 pts, marked ~597 from Kite LTP, MTM ~+Rs 29,900, exit due 08-Sep**.

**Found and fixed: bhav had NO download cron** - it sat stale from 2026-07-21 to 2026-08-24, so
every EOD-priced book was silently ageing. Backfilled the gap and added a 16:10 daily cron.
Also fixed a real bug in the paper engine: `prev_session()` collapsed FUTURE dates onto the last
known session, which closed the open trade 18 days early and invented entries for long-dated
contracts. Future dates now stay unknown.

Crons: mark */5 during market hours; bhav 16:10; seed+mark 16:20. All registered in ops_center
(new group) with two dated REVIEWS. Strategies register flipped parked -> paper.

**Data limit worth remembering:** the 1-min recorder only picks a contract up at ~27 DTE, so it
can NEVER price a 45-DTE entry. Entries come from the EOD close; the live mark uses Kite LTP
until the contract enters the recorder window, then the 1-min feed takes over.

**PENDING before this can go live:** (1) STRESS-MARGIN test - reconstruct per-lot SPAN 2019-26 and
re-run with a margin-call rule (Ops review due 2026-09-30); (2) paper-vs-study tracking review
after ~3 completed campaigns (due 2026-11-30); (3) correlation vs THE STACK / NAS / straddle books.


## 2026-08-24 - 45-DTE straddle: control-room page LIVE at /app/straddle45 (NOT ARMED)

Arun asked for an app page under Live, in line with NAS/Momentum. Built:
`frontend/src/pages/Straddle45.tsx` + module.css, route + sidebar entry, and a row in the
Strategies register (status PARKED - nothing is armed, there is no executor). The page carries
a **lot selector defaulted to 3** and everything on it re-prices from that: capital to block,
margin headroom, KPI strip, the year-by-year table, and the payoff graph. Study is stored in
POINTS so rupees derive as pts x 65 x lots.

Margin measured LIVE from Kite `basket_order_margins` on 24-Aug-2026 (read-only, no orders):
**Rs 2.13-2.42L/lot, NRML = MIS to the rupee** (no intraday benefit on short index options).
Margin does NOT rise into expiry - it is flat 1-22 DTE and rises with tenor. What does move it
is **moneyness**: +26% once spot is 3% away, +41% at 5%, and down-moves cost more than up-moves.
Since a >=3% move happens in 66% of campaigns, the page sizes capital on the **3%-adverse
margin (Rs 2.69L/lot) + 2x MaxDD** - at 3 lots that is Rs 12.0L.

**PENDING before this can ever be armed:** (1) the STRESS-MARGIN test (reconstruct per-lot SPAN
across 2019-26 and re-run with a margin-call rule) - still the one blocking item; (2) an executor
+ positions feed, none exists today; (3) correlation vs THE STACK / NAS / straddle paper books.


## 2026-08-20 - research/119 45-DTE short straddle: STRATEGY-CANDIDATE - stress-margin test OWED

Arun asked us to verify Sandeep Rao's "The Long & The Short Ep. 48" 45-DTE NIFTY short-straddle
backtest, test 1h vs 30m monitoring, and his VIX percentile filter. **His table replicates on real
NSE bhavcopy** (89 trades vs 83, win 70.8 vs 69.9%): net **+78.0 pts/trade, t 3.12**.
**On Arun's margin basis (Rs 3L/lot x 10 lots, Rs 36L blocked; NIFTY lot is 65 not 75, so 1 pt =
Rs 650): CAGR 11.47% vs NIFTY 11.60%, MaxDD -13.8% vs the index's -38.4%, Calmar 0.83 vs 0.30.**
Six positive years, worst -3.2% (2019), 2020 POSITIVE. Monitoring settled on REAL 1-minute data:
0 of 60 sessions in the DTE>=21 band travelled >=50% from their close, and the 3 real 45-DTE trades
our recorder overlaps stayed within 0.55x-1.08x of credit - neither trigger ever approached.
Hourly beats daily on the tail only (worst trade -29%, MaxDD -23%); 30m/15m/5m identical.
VIX >25 is the best risk-adjusted cell (Calmar 1.05); >75 is the WORST on capital (6.95% CAGR) and
his 85.7% win rate does not reproduce.

**2026-08-23 - Phase E (delta management) REFUTED.** Arun asked: keep the straddle until an x%
underlying move, then exit and redeploy at the new ATM. Swept 7 thresholds x 3 arms x 3 re-entry
caps x close/intraday triggers on real bhavcopy prices: **every variant loses to holding.** A cycle
cut by the move rule realises **-28.6 pts (38% win)**; a cycle left to run to 21 DTE earns **+83.0
pts (81% win)**. Best managed cell keeps 36% of the return. Cost is only ~12 of the 67-pt shortfall
- the rest is forfeited theta. Cutting on UP moves costs ~3x cutting on down moves. **Actionable:
to reduce drawdown, size down rather than manage - hold @ 5 lots (6.73% CAGR / 9.0% DD / Calmar
0.75) dominates the best managed arm @ 10 lots (5.16% / 9.6% / 0.54).** No further management
variants worth testing on this structure.

**PENDING - the one thing blocking a live decision: STRESS-MARGIN TEST.** Rs 3L/lot is today's
margin at India VIX 10.83. VIX peaked 83.61 on 2020-03-24 and SPAN scales with vol, so Rs 36L would
very likely have been breached in Mar-2020 - forcing a top-up or liquidation exactly when the book
was losing. Reconstruct per-lot SPAN+exposure across 2019-26 and re-run the equity curve with a
margin-call rule. **Until then 11.47% CAGR is an UPPER BOUND.** Also owed before any sizing:
correlation vs THE STACK / NAS / straddle paper books (all short-vol, all lose the same week).
NO deploy, NO paper book opened yet.
Report: /app/backtest/nifty-45dte-short-straddle - Full: research/119_45dte_short_straddle/results/RESULTS.md



## 2026-08-20 — research/114 SENSEX Thursday exits: SIGNAL — deploy decision PENDING

Arun questioned the venue TP after it closed the SENSEX book at 12:13 (+10,194). Two
independent tests agree it hurts: tp_validation (34 mixed sessions, -4,241 per TP day) and
research/114 (12 clean Thursdays). The bake-off found the bigger problem: the suite's
per-leg 30% stop turns +2,630/lot/day at 92% win into -227 at 25% win on expiry day.
Recommendation, needs sign-off: (1) drop per-leg 30% on SENSEX Thursday, (2) raise TP to
~4,000/lot or retire it, (3) keep the 50% disaster backstop, (4) add no book stop.
CAVEAT: 12 benign Thursdays cannot price the tail - this is "remove destructive stops",
NOT "run naked". Full: research/114_sensex_thursday_exits/results/RESULTS.md

## 2026-08-19 — 1.0x scale plan LOCKED (Option B) + Thursday restructure DEPLOYED

Data-driven split (lab per-DTE cells): Thursday is BOTH venues' best day (NIFTY DTE3 mean
16,956/91% · SENSEX DTE0 14,322/94%) — so NIFTY Thursday is NOT dark: it runs via a new
dedicated **CSL_TIMEB_NIFTY_THU book at 3 lots (entry 09:25)** — the max that clears every
1.3x margin gate at Rs44.7L capital. TB-SENSEX restricted to **Wed+Thu only** (its Mon/Tue/Fri
cells are the grid's weakest and collide with NIFTY). Main TB-N dark Thu. Config json is FROZEN
so the trims persist. **PENDING Mon 24-Aug: bump TB-N and TB-SX 8 -> 10 lots** (registered in
Ops). Weekly targets & progress page: /app/scaleup (append actuals every Friday).

## ✅ 2026-08-18 — research/113: ATM4 roll-leg stop — SIGNAL — DEPLOY STAGED (restart 15:40)

Arun watched the live ATM4 roll (24150 PE @12.1, SL 15.7) die in 6 minutes and asked for a
data assessment. 81 days real 1-min NIFTY chain, 63 roll events: live rule (1.3x roll_prem)
is the churniest variant (32% restop, 6% <=15min). Rolling itself is strongly validated
(never-roll = -49k vs +143k). **DEPLOYED: rolled-leg SL = max(price_x, roll_prem) x 1.3 (MAXV — Arun refinement over SURV; best tail p05 -1,067, 19% restop, +8.5k over old rule)** — beats
live rule on every metric incl. DTE0. One line in services/nas_atm4_executor.py (~L395),
SIGNED OFF by Arun 2026-08-18 midday; code patched + committed, deferred restart scheduled 15:40. Verify review in Ops Center (due 2026-08-28). Full verdict: research/113_atm4_roll_stop/results/RESULTS.md.
Re-check when the data window doubles (~late Sep 2026).

## ✅ 2026-08-20 — Live-vs-app reconciliation job LIVE (11:00 + 14:00, email + WhatsApp)

Arun spotted NIFTY COMB showing **2 lots on the app while the account held 5**. Diagnosis: the
trade was RIGHT and the page was wrong. `csl_paper_config.json` records the 19-Aug decision —
*"NIFTY Thursday CONSOLIDATED into NAS_COMB20 DTE3 5L/qty325 09:16-15:20 SL20 ... Total NIFTY-Thu
size unchanged at 5L"* — and the broker order confirms it: one SELL, qty 325, tag `CSL_NAS_COMB20`,
avg 133.95/73.80. The page renders the book's static 2L/130 because `csl_paper_live.json` carries no
lots/qty, so displayed size AND P&L were 2.5x light (+Rs 448 shown vs +Rs 1,218 real).

**Built: `scripts/live_vs_app_recon.py`** — read-only, cron **11:00 + 14:00 Mon-Fri**, emails and
WhatsApps the report every run (`get_notification_service()`), writes `static/app/live_recon.json`
and appends to `docs/LIVE_RECON_LOG.md`.

Four checks: **ORPHAN** (broker leg no book claims) · **SIZE** (broker qty != app qty) ·
**GHOST** (app records a leg the broker lacks) · **NAKED** (short option with no SL resting at the
exchange). Manual equity holdings are INFO, not alerts.

Getting it honest took four passes, each a lesson about where truth lives:
1. `*_positions WHERE exit_time IS NULL` → 16 false GHOSTs (May/June expiries never marked closed).
2. App state endpoints → SENSEX legs invisible (sensex_live.json has no leg detail) = false ORPHAN.
3. `config.paper_trading_mode` → reads False on the 916 arms while they trade paper-shadow.
4. **The day/gap matrix ALSO lies** (`nas_916_atm: live=true`) — a per-DTE gate forced paper anyway.
**Truth = each position row's own `mode` column.** Only `mode='live'` legs are expected at the broker.

First clean run 11:57 IST: 7 broker legs, **0 alerts**, 4 NAKED warnings, 3 manual holdings.
It resolves COMB20 correctly at 325 = 325.

**The 4 NAKED warnings are real and worth a decision:** SENSEX 77500 CE/PE and NIFTY 24200 CE/PE are
short with software-side stops only — the same exposure as the 2026-08-17 incident.

**STILL OPEN (display bug):** make the Straddles page show each CSL sleeve's EFFECTIVE per-DTE
lots/qty instead of the static book default. The data is already in `csl_paper_config.json`; the
page just needs to read it (or the feed writer should emit it, which touches the executor).

## 2026-08-20 — P0 ORB Cash: paper stops fire instantly (my entry fix exposed a 2nd defect)

The 05-May entry fix works — ORB booked 8 positions today, the first since May. But **all 8 closed
at exactly their stop 13-42 seconds after entry** (`SL_HIT_EXCHANGE`), and today's -Rs 21,029 is
**fictitious**. Arun caught it: AXISBANK entered 1,247.00 at 09:40:05 and "stopped" at 1,236.20 at
09:40:47, a price it never traded after entry (it rallied to 1,250.10, +1.22%).

**No real money at any point** — every order today is `PAPER-xxxx`; no broker exposure.

**Root cause:** `_kite_order_history()` (orb_live_engine.py) returns a SYNTHETIC
`status: "COMPLETE"` in paper mode, by design, "so SL-poll loops terminate cleanly". The
exchange-SL reconciliation at ~line 1560 (`Step 1b`) reads any COMPLETE as "the stop filled" and
closes the position at `average_price or pos['sl_price']` — `average_price` is None for the
synthetic row, so it books the exit at exactly the stop. Before my fix no positions existed, so
this path never ran; fixing entries exposed it.

**Fix (engine code — needs approval + after-15:40 deploy):**
1. Skip Step 1b entirely in paper mode: `if self._is_paper() or str(sl_order_id).startswith('PAPER-'): continue`
   — let the existing price-based SL monitor decide stops from the LTP, as it does for live.
2. Record `paper_mode=1` on positions created in paper mode. Today's 8 rows are written with
   `paper_mode=0`, so paper trades masquerade as live in orb_positions (pollutes the journal and
   the liveness reader).
3. Delete or tag today's 8 phantom rows so the book's record is not poisoned.

**Meanwhile:** ORB Cash should go to Off (mode toggle, reversible, no code) or it keeps writing
false trades every few minutes. Awaiting Arun's word.

## 2026-08-19 — KC6 audit: paper all along, but mislabelled in two places

Chased the "KC6 shows 5 trades in 30 days while the register says parked" flag from the liveness
projection. Verdict: **never real money, but the records are wrong in two ways.**

- Every KC6 order ever placed is `status='PAPER'` / `'PAPER_TARGET'` with `kite_order_id NULL`.
  No broker exposure at any point.
- It was NOT dormant: 6 closed trades total, **5 between 23 Jul and 17 Aug** (BANDHANBNK −4,999,
  ADANIGREEN +935, PHOENIXLTD +1,582, ADANIPORTS +2,756, NH −4,979), net **−₹4,202**, 67% win.
- `config.enabled` is **False** now with 0 open positions, so "parked" is accurate today — the
  register's old note ("scheduler runs, unfunded") just undersold a month of paper activity.
  Register corrected 19 Aug.

**STILL OPEN — journal mislabel:** `services/journal/sources/kc6_source.py:66` hardcodes
`'mode': 'LIVE'`, so ₹5,528 of PAPER P&L sits in a ledger that is supposed to be live-only. Same
class of problem as the ORB-Index and ORB-Cash sources. Fix it with the journal live-only work
(item 4 of the app-review follow-ups) — one filter fixes all three.

**Also noted:** the service was restarted at **14:08 IST on 19 Aug, during market hours** (not by
this session). Side effect: the ORB entry fix and the liveness endpoint went live early, and the
15:40 deferred restart was therefore cancelled as redundant. Worth knowing who/what triggered it —
the standing rule is no restart before 15:40.

## 2026-08-19 — Per-book activity audit: 5 of 6 are fine, ORB was the only break

Arun: "i see no paper trades in orb cash ... nwv, n500, mst, 175wr, pairs ... breakout 10L as well".
Audited each one against its own store rather than its page.

| Book | Trades | Last | Status |
|---|---|---|---|
| **NWV** | **2 cycles, both winners** | 03 Aug | WORKING. 27 Jul bullish -> PT **+Rs 14,586**; 03 Aug neutral -> TIME **+Rs 12,187**. 10 + 17 Aug the weekly view said `ignore`, so it correctly skipped (`SKIP_IGNORE` rows in history). Next decision Mon 09:50. |
| **N500M** | 31 since 08 May | 17 Aug | WORKING. +Rs 13,852, 58% wins. Sparse by design — the per-stock volbo trigger fires ~2x/week across 27 names. |
| **I75WR** | 1 | 18 Aug | WORKING. First paper trade booked the day after enabling: AARTIIND SHORT 570 @525.85 -> EOD 530.10, -Rs 2,422. 35 other signals correctly BLOCKED. |
| **Pairs** | 0 | — | WORKING, no trigger yet. The 16:00 scan logs all 6 cohort pairs with their z-scores; none has breached the entry band since it was enabled on 17 Aug. |
| **MST** | 0 since re-enable | 07 May | WORKING, no trigger yet. Booted FLAT, 30-min bars aggregating; needs a stochastic cross with >=6 DTE. |
| **Breakout Rs10L** | 1 open | 06 Aug | WORKING. Gate ON only 1 of 33 sessions since 01 Jul; that window bought NAVINFLUOR (-4.6%). Gate settled by research/109 — do not re-litigate. |
| **ORB Cash** | 0 since 05 May | 05 May | **WAS BROKEN** — undefined-name in the entry path, fix deploys at 15:40 today, first paper trades expected Thu 20 Aug. |

**Conclusion: only ORB was broken.** The other five are running correctly and are simply
low-frequency; two of them (NWV, N500M) have real P&L that the pages never show.

**ROOT CAUSE OF THE PERCEPTION GAP (and the fix worth building):** every one of these pages shows
*today* and hides the book's own history. NWV has +Rs 26,773 of closed cycles sitting in a JSON the
page does not render; N500M has 31 closed trades and an EMPTY `n500m_equity` table so there is no
curve to draw. This is finding #14 in `docs/APP_ASSESSMENT_2026-08-17.md`.

**TO BUILD (read-only projection, no engine touched — inside the standing guardrail):**
a per-book history footer on every paper page — last trade date, days since, trades in 30d,
cumulative net, win rate, and a sparkline derived from the trade table. Plus the liveness rule
(mode + last-trade + days-idle) so "is this thing running?" is answered on the page itself instead
of by a database query.

## 2026-08-19 — Breakout gate question: ALREADY ANSWERED by research/109 (no new study run)

Arun, looking at 4 strong qualifiers the gate skipped: "the qualifying stocks picked up are excellent
picks, in fact entry must hv been few days earlier too". Before running a gate-cost study I checked
the shelf — **research/109 already swept exactly this**, from 2006, same rules and costs.

Its verdict: **changing the gate threshold is a dead end; cadence was the lever.**

| Gate (daily cadence) | CAGR | MaxDD | Calmar |
|---|---|---|---|
| ma200 (current) | 18.9% | -33.7% | 0.56 |
| ma150 | 17.5% | -30.7% | 0.57 |
| ma200 hysteresis +/-3% | 16.4% | -28.7% | 0.57 |
| ma200 hysteresis +/-1% | 16.1% | -29.8% | 0.54 |
| ema200 | 14.9% | -34.5% | 0.43 |
| ma200 + rising slope | 13.6% | -30.7% | 0.44 |

Note WHY hysteresis was tested: the book was paralysed with NIFTY 0.19% below its 200-DMA — the same
complaint as today (NIFTYBEES -1.38% below). It bought essentially nothing (0.57 vs 0.56). The lever
found instead was the WEEKLY cadence (same 18.9% CAGR, MaxDD -33.7% -> -27.3%, Calmar 0.56 -> 0.69),
and that is already live in the book (`decision_cadence="weekly"`).

Live-window context: since inception 01 Jul the gate has been ON for **1 of 33 sessions** (06 Aug).
That single open window is what bought NAVINFLUOR, currently -4.6%. A 33-session sample cannot
overturn a 20-year sweep, and re-testing a settled negative is the multiple-testing sin the playbook
warns about — so **no new gate study was run**.

**QUEUED (after 15:40, not during market hours):** the one interaction research/109 did NOT test —
hysteresis +/-1% *combined with* the weekly cadence (its gate variants were all swept at daily
cadence). ONE pre-registered cell vs the base config, using the existing harness
`research/109_breakout_gate_freq/scripts/run_breakout_opt.py`. Deliberately not launched during
market hours: heavy compute on this box has starved live monitors before.

Housekeeping from the app review, same folder: **research/109 is used TWICE**
(`109_breakout_gate_freq` and `109_intraday_stocks`) and neither is in `research/INDEX.md`.

## 2026-08-19 — P0 ORB Cash: paper entries never booked since 05 May (FIX WRITTEN, NOT APPLIED)

Arun asked why the paper books are not trading. ORB Cash is not "filtered out" — it is **broken**.

**Root cause:** `services/orb_live_engine.py:1870` calls
`self._verify_order(kite, order_id_str, instrument, 'entry')` where **`kite` is undefined** in that
scope. Every entry raises `NameError: name 'kite' is not defined`; the enclosing `except` logs
"Entry order FAILED", writes a REJECTED order row and returns None, so the caller skips on and
**no position is ever recorded**. The exit path (line 2284) does it correctly, guarded by
`if not self._is_paper(): kite = self._get_kite()`.

**Regression:** commit `03fc917` (2026-05-05) added the `_kite_place_order` paper wrapper, removing
the inline `kite = self._get_kite()` that had been defining the name. Last position recorded
05 May 13:05. Since then PAPER-placed orders equal REJECTED rows 1:1 every month (Jun 225/225,
Jul 200/200, Aug 83/83) with **0 positions**.

**Severity:** in LIVE mode the real Kite order is placed BEFORE the NameError fires, so the engine
would hold an untracked real intraday position with no SL and no monitoring. Currently PAPER, so
nothing is at risk today — but the Live button is one click away.

**Fix (one line, mirrors the exit path) — NOT applied; needs approval + after-15:40 deploy:**

    if not self._is_paper():
        kite = self._get_kite()
        self._verify_order(kite, order_id_str, instrument, 'entry')

Verify next session: `orb_positions` gains rows, and PAPER-placed no longer pairs 1:1 with REJECTED.

Full forensic: `research/113_orb_paper_entry_forensic/ORB_PAPER_ENTRY_FORENSIC_STATUS.md`.

**Side finding — this kills the filter-cost study for this question.** The filters were never the
reason the book is flat. Stored counterfactuals in `backtest_data/orb_backtest.db` (186 run days,
2025-08-18 to 2026-08-18): TAKEN 675 trades **+Rs 34,260**; BLOCKED 376 signals would-be
**-Rs 25,104** — the filter stack SAVED about Rs 25k over the year. Separately, 1,105 ERROR rows in
that table point at data-pipeline noise in the 15:45 backtest job, worth its own look.

## 2026-08-17 — App review follow-ups (PENDING — from `docs/APP_ASSESSMENT_2026-08-17.md`)

Independent structural review of the whole app (37 pages, 391 backend routes, 90 scheduled jobs,
128 research folders). Full findings + evidence in the doc; rendered artifact "Quantifyd App
Assessment". **Guardrail (Arun, binding, now in `.claude/CLAUDE.md`): none of this touches live or
paper trading logic** — read-only projections, display/routing, shared components over existing
endpoints, and docs only.

In the order to do them:

1. **Dead-link hygiene** (half a day) — (a) Settings nav item points at `/settings`, which has no
   React and no Flask route: build the page or remove the item; (b) `nas_analyzer.py:5`,
   `options_outlier_scan.py:5` and `docs/LABS_AND_JOBS_REFERENCE.md:46-47` all say `/app/reports`,
   which never existed (the page is `/app/report`, nav label "Performance", title "NAS performance
   report" — pick one name); (c) `/api/v2-ironfly/` bare-prefix fetch in `Straddles.tsx` matches no
   route; (d) ORB Index (`/strangle`) has no nav entry — put it in Paper Books; (e) NotFound has no
   way back.
2. **Register the morning token chain in the Ops Centre** — `auto_login.sh` 08:50, `token_heal.sh`
   09:06, `preopen_restart.sh` 09:00, `killflag_premarket_check.py` 09:05 appear in NEITHER
   `ops_center.py` nor the labs doc, and a stale-token cascade is a known way to lose the 09:16
   one-shot. Ten-minute edit, highest-consequence gap found. Then: ops page diffs
   `scheduler.get_jobs()` + `crontab -l` against the curated table with paper/parked families
   filtered out by design (Arun's ruling: ops covers live + research/re-assessment only), showing
   in-scope-but-unregistered jobs as **UNREGISTERED**. Also missing and in scope:
   `sensex_live_writer`, `publish_nifty_5m`, `gen_momentum_scan`, holdings jobs (3 + 2 cron),
   `db_integrity_watchdog`, `instruments_dump`, `premarket_brief_*`, and the research recorders
   (`sl_reanchor_shadow`, `dl_sensex_1min`, r/56, r/80, r/82, r/90 travel, mentor capture). Four
   more are in the labs doc but not ops_center: `snapshot_nas_eod`, `dump_nas_mtm`,
   `options_study_agg`, `backup_to_github_release`.
3. **Liveness rule per system** — mode + last trade + last signal + days idle, computed from the
   trade tables, shown on the register and on each page. This is what would have caught I75WR
   (7 jobs, empty DB), Pairs (same) and MST (dead since 07 May) months ago.
4. **Journal = exactly the live book** (Arun's ruling: live-only by design). Today it misses 3 of
   the 5 live systems — TB-CSL NIFTY, NAS_COMB20 (both in `csl_paper_state.json`) and Momentum ₹3L
   — while ingesting 3 non-live ones: ORB Index 349 trades (−₹2,02,510), ORB Cash 46 (−₹21,888),
   KC6 6 (−₹5,528), i.e. ≈ −₹2.3L of paper/parked P&L inside a real-money ledger. Cheapest first
   step needs no new code: filter to live strategies. Then add a CSL source + a momentum source and
   retire/tag the orb, strangle and kc6 sources.
5. **Shared page furniture** — `<PageHeader>` (name, purpose, mode chip, size, Rules/Study/Journal
   links), `<ModeChip>` and `<ModeControl>` to end the nine-words-for-three-states problem
   (LIVE/PAPER/REAL/ARMED/Off/Disabled/Parked/"Live trading"/"Paper trading"), one money formatter
   (formatPnl 16 pages vs toLocaleString 18 vs Math.round 17; "₹" 15 pages vs "Rs" 12). Apply to
   the five paper-book pages first — their stylesheets are byte-identical (md5 `434740f8…`), so they
   collapse into one `<PaperBook>` driven by a config record.
6. **Register's second pass** — derive mode/size/last-activity instead of declaring them (the
   hand-maintained register repeats the ops-centre failure mode); keep rules, evidence and change
   log by hand. Do this AFTER 3, or it just moves the drift.

Also queued from the same review: per-book history footer (N500M shows today only while its DB holds
31 trades / 25 sessions / +₹13,852, and `n500m_equity` has 0 rows); generate `research/INDEX.md`
(47 folders unindexed, and `109` is used by two folders); decide the 7 live-but-unlinked Jinja
dashboards (`/agent /kc6 /collar /maruthi /bnf /tactical /trident` — `/maruthi` still offers controls
for a strategy with 9 known correctness bugs); route-level `React.lazy` + polling tiers (one 1.25 MB
chunk; 23 pages polling 1–60s with no shared policy); accessibility basics (12 aria attrs total,
`:focus-visible` in 1 of 30 stylesheets).

## 2026-08-17 — Paper trading switched on for the labs (MST included — activates 09:00 Tue 18 Aug)

Arun: "lets do paper trading for all of these... create a section Live above paper books".

DONE:
- **I75WR** — all 3 configs off → PAPER (persisted in `backtest_data/intraday_75wr_mode_overrides.json`).
  Jobs were registered all along but every config sat at mode=off, so its DB is empty to date.
- **Pairs** — off → PAPER. `/api/pair_trading/toggle-mode` only patches the running process, so
  `config.py PAIR_TRADING_DEFAULTS['enabled']` was flipped to True as well (survives restart).
- **N500M** — already PAPER and trading since 08 May: 31 closed trades / 25 sessions / 58% win /
  **+₹13,852**. Nothing to enable.
- **NWV** — already running as a paper book (`services/nwv_trade.py`, JSON state).
- Sidebar: new **Live** section (NAS, Straddles) above Paper Books; NWV / N500M / MST / I75WR /
  Pairs moved into Paper Books. Register updated to match (23 systems).

**MST — DONE, activates at the 09:00 pre-open restart (Tue 18 Aug).** Arun: "MST shud be paper
trading". The blocker was its stale state, now cleared:

- 6 rows still marked OPEN from 07 May were the incident itself — **ids 1,2 real legs**
  (`paper_mode=0`: BUY NIFTY 24450 CE @266.80 / SELL 24650 CE @173.65, 65 qty each) on the
  **expired 2026-05-19 weekly**, plus ids 7-10 priced **0.00** from the frozen-tick
  `credit_too_low: credit=0/lot` rolls (see `mst_events` 11:15 and 14:45 that day).
- All 6 marked CLOSED with `exit_reason='STALE_CLEARED_20260817'`, **rows kept, P&L left NULL**
  (no knowable exit). A `state_cleared` event records why. DB backup:
  `backtest_data/mst_trading.db.bak_20260817_mst_paper`.
- `config.py MST_DEFAULTS['enabled'] = True`; `paper_trading_mode` True, `live_trading_enabled`
  False. The do-not-go-live note is preserved and extended: **LIVE stays barred** until the
  2026-05-15 causes are closed (tick-pipeline freeze, spurious credit_too_low rolls, rejected
  real-leg closes).
- Activation needs one process restart, which the existing `0 9 * * 1-5 preopen_restart.sh` cron
  performs — so the engine boots FLAT in paper mode at 09:00, before the open. No manual step.
- **Verify after 09:00 Tue:** boot log should NOT say "State RESTORED from N open legs";
  `/api/mst/state` should show `enabled: true, paper_trading_mode: true` and a flat state.
- **WATCH (this is what killed it in May):** any 0.00-priced leg or repeated `credit_too_low` roll
  in `mst_events` means the tick-freeze is back. Worth a dated review — suggest 2026-09-01.

**OPEN, separate:** the two REAL legs from 07 May expired 19 May with **no exit ever recorded** in
the app, so that real-money outcome is missing from the ledger. It is knowable from the broker
statement / May tradebook only. Decide whether to reconstruct it (the journal is live-only by
design, and MST was live then).

~~PENDING — MST cannot be enabled yet.~~ Original note: every boot logged
`[MST] State RESTORED from 6 open legs → state=DEBIT_OPEN_L1 direction=1 L1_anchor=24450
expiry=2026-05-19`. Those legs are from 07 May on an expiry three months dead; switching mode to
paper would have the engine manage phantom legs on an expired series and produce garbage marks.
TO FIX: close out the 6 stale rows in `mst_positions` (mark CLOSED with a note — do not delete the
07 May record), reset the engine state to FLAT, then `POST /api/mst/toggle-mode {"mode":"paper"}`
(that route is `@login_required`, so it needs a browser session or a cookie).

ALSO PENDING — **N500M page shows today only.** The 25-session history sits in `n500m_positions`
and `n500m_equity` is empty (0 rows), so there is no equity curve. Worth surfacing past trades +
a cumulative curve on `/app/n500m` the way the other paper books do.

## ✅ 2026-08-17 — Strategies page → register of live / paper / parked systems (SHIPPED)

Arun picked mocks **A (Register)** and **C (Spec)** — both built at `/app/strategies` behind a
Register / Spec switch in the page header (choice persists in localStorage; clicking a system name
or its Rules chip opens that system's Spec). Boxed metric tiles dropped at his request — the Spec
pane opens on an unboxed stat line with hairline dividers.

- `frontend/src/data/strategies.ts` — **the register of record**: 19 systems, each with status,
  size, one-line rule, the rules as they run, source-of-truth doc path, published studies (real
  slugs → `/app/backtest/<slug>`) or a visible "Study pending" gap, and a dated change log.
  Plus `LAB_PAGES` so the no-capital lab pages aren't silently dropped.
- `frontend/src/pages/Strategies.tsx` + `.module.css` — rewritten; old StrategyCard grid and
  "Today at a glance" mini-stats removed.
- Status split resolved from the repo: Momentum ₹3L = LIVE (real money), ORB Cash = PAPER
  (resumed 08-17, paper-aware margin gate), KC6 = PARKED (scheduler runs, unfunded).
- Day P&L only where a feed already exists (ORB `/api/orb/state`, ORB Index `/api/strangle/state`,
  NAS NIFTY 8×state + SSE LTPs, NAS SENSEX `/app/sensex_live.json`); everything else reads "—".

Mocks: `frontend/mockups/strategies/01_register_three_ways.html`. `tsc --noEmit` clean, built,
frontend-only (no restart). OPEN: KC6 rows in the journal are tagged LIVE from the dashboard's own
mode flag — confirm before reading them as real money. Uncommitted.

Binding rule already added to `.claude/CLAUDE.md` — **THE STRATEGIES INDEX IS THE REGISTER OF
RECORD**: any system going live/paper/parked, changing size, or changing a rule updates the
index in the same commit (status, size, rule line, Rules/Study/Dashboard links, dated change-log
entry). Uncommitted as of writing.

## ✅ 2026-08-17 — Journal NAS filter fixed (live) + sidebar page shortcuts

Journal showed "0 trades · Rs0" with the NAS chip selected even though the 15:50 sync had
projected all 10 of today's NAS cycles. Cause: the chip sends `strategy=NAS` and
`journal_db.py` matched `strategy = ?` exactly, while stored labels are per-system
(`NAS-ATM2`, `NAS-916-ATM`, `SENSEX-ATM4`, `ORB-INDEX-OR60-STD`...). Added
`STRATEGY_FAMILIES` + `strategy_clause()` (family → LIKE patterns, unmapped value still
matches exactly) and used it in `list_trades`, `daily_summary`, `equity_curve`,
`r_distribution`. NAS = `NAS-%` + `SENSEX-%`. Verified over HTTP after the 18:06 restart:
NAS Aug = 11 sessions / 101 trades / +₹1,79,688; 17 Aug = 10 cycles, net −₹13,228.

Also: single-letter page shortcuts in the sidebar (`frontend/src/components/Sidebar/hotkeys.ts`
is the single source for both the badge and the key listener) — J journal, N nas, H holdings,
M momentum, O opt-study, etc. Keyed by route, so moving an item between sections keeps its letter.
`journal_db.py` + mockups + CLAUDE.md still uncommitted.

## ✅ 2026-08-17 — NAS ST-trail confirm-counter bug FIXED + DEPLOYED (restart 15:48 IST, commit a792136)

The intrabar ST-trail on naked ATM/ATM4 survivors never fired: the confirm counter lived in a
throwaway LOCAL var (`self_atm_breach_ticks`) that read the instance attr via getattr but never
wrote it back, so it reset to 1 every tick and never reached NAKED_TRAIL_CONFIRM_TICKS(3). Live
symptom today: a NIFTY 24300 CE survivor breached its ~58.6 trail continuously for 40+s stuck at
"1/3", never exiting (Arun had to exit manually). Renamed to instance attrs
`self._atm_breach_ticks` / `self._atm4_breach_ticks` (ATM + ATM4). Deployed after close, service
healthy, committed+pushed **a792136**.
VERIFY (review 2026-08-21): next naked survivor should log 1/3->2/3->3/3 then TRAIL EXIT intrabar.

## 2026-08-17 — Unified per-system position ledger (PENDING — "fix the overall stuff")

Root cause of today's COMB mess: the portfolio stop flattens at the ACCOUNT level (it bought back
COMB's CE without COMB knowing — COMB isn't even in the portfolio stop's system list), and COMB
runs as a separate in-memory daemon. No shared tagged ledger, so trackers drift from the broker the
moment any actor (system exit / portfolio stop / manual) touches a shared symbol. Also had to KILL
the whole CSL daemon to stop COMB's 15:20 phantom-CE exit (freeze/kill flags don't gate an
already-open book's exit; restart double-enters) — no per-book pause exists.

BUILD: one position ledger where every leg is tagged by owning system; the portfolio stop, COMB, and
manual fills all reconcile through it; ~~broker-qty ASSERT before any exit buy~~ **DONE 2026-08-18**
(commit 0b581e7 - CSL exit path now checks broker-held qty: manual close -> reconcile, partial -> capped
buyback); a per-book pause flag so one sleeve halts without killing the daemon or stranding the others. Registered in Ops Center REVIEWS (due 2026-08-24).
Today's COMB rightful trade recorded as PAPER in csl_paper_state.json (−₹121 TIME_EXIT) for the
later rules-vs-actual assessment.

## ✅ 2026-08-14 — Momentum live book: top-up ledger bug FIXED + DEPLOYED (restart 2026-08-16 21:08 IST)

Arun added ₹1L via `/app/momentum-paper` "Immediate equal top-up". Broker fills were correct
(BHEL 134 / NATIONALUM 147 / LAURUSLABS 30 / RADICO 11 / POWERINDIA 1) but the ledger corrupted:

1. `_buy()` used `INSERT OR REPLACE` → the top-up OVERWROTE the held row, wiping prior qty,
   cost basis and entry_date. `_sell()` sells the RECORDED qty, so the next Donchian stop would
   have sold 45 BHEL and ORPHANED 89 shares at the broker with no stop on them.
2. `cash_deposit()` double-deducted the spend (`_buy` already deducts cash) — cash ₹78,113 short.

DONE: ledger repaired by replaying `mp_fills` + asserting vs `_broker_qty()`; NAV back to
₹396,204 (capital ₹4L, cash ₹142,289). Code patched in `services/momentum_paper.py` at 14:57.

DEPLOYED Sun 2026-08-16 21:08 IST (market closed). Verified by FUNCTIONAL test on a DB copy,
not a source grep: top-up of a held name took qty 134 -> 144 (accumulated, not overwritten),
entry_date preserved (STCG clock intact), cost basis = weighted average 415.54, cash debited
once. `live_armed` now also exposed by the state API (was returning None) and reads True.
"Immediate equal top-up" is safe to use again.

Add-funds UI now shows the MINIMUM for a full even split (dearest share price x n_holdings =
Rs1,78,600 today, constrained by POWERINDIA) plus a live breakdown of what deploys vs what falls
back to cash and which names get skipped. Frontend-only, shipped 2026-08-14.

## ✅ 2026-08-14 — research/112 fresh-deposit deployment timing: SIGNAL

Settles which policy applies to DEPOSITED cash (r/108 monthly = stop-out cash; r/41 ph-27 weekly
= all-cash gate re-entry; neither covered deposits). 4 arms × 12 phase offsets, identical cash
flows, 2011-2026 net of STCG. Winner: **immediate EVEN top-up of names already HELD (12/12
phases)**; filling empty slots fast loses (false-dawn penalty, consistent with r/108). Edge is
small — +0.8% terminal over 15.6y (~5bps/yr), 12 phases share one price history so not 12
independent trials. Live `immediate` mode already implements the winner.
TODO: make `immediate` the DEFAULT deposit mode when the gate is risk-ON (park only on risk-OFF).
Details: research/112_deposit_timing/results/RESULTS.md

## 2026-08-09 — research/110 alt-info intraday CONCLUDED: NO EDGE (0/14) — intraday line CLOSED
Cross-sectional RS dead both directions; event-proxy fade fake (flips in Val); flow
proxies negative everywhere. 58 intraday constructions total across r/89+109+110:
OHLCV-derived intraday edge does not clear costs. Would need external data (news
feeds, order-book depth) to reopen. Both new paper books (ORB revival Rs10L,
OHOL first-candle 1-lot) LIVE from Mon 2026-08-10; /app pages pending.

## 2026-08-09 — research/109 intraday stocks CONCLUDED: NO EDGE net of costs
9 families/35 cells/1.94M trades: best signal (narrow-CPR trend-day short) carries
+5-9bps real excess vs ~10bps cost floor; all Val nets negative. 20%-CAGR intraday
goal ruled out for price/indicator signals; route ambition via multi-day books +
ORB revival decision (still pending above) + NAS. OOS 2024+ untouched.

## ✅ RESOLVED 2026-08-10 — research/89 ORB reassessment: revival armed as a PAPER book

**Closed 2026-08-17 during the app review:** the Rs-capped paper revival went live 10 Aug
(ORB Revival ₹10L, `/app/orb-paper`, multi-day signal only — never the intraday variant that
killed the live book). ORB Cash itself resumed PAPER 17 Aug. Original entry below.

### (original) 2026-08-09 — research/89 ORB reassessment CONCLUDED — revival decision PENDING (Arun)

- ~~DONE 2026-08-14~~ **THE STACK DEPLOYED LIVE** - suite REAL Mon/Tue/Fri 2L; sleeves live-armed, first REAL fills Mon 17-AUG post LIMIT-fix; Thu shadow per stop-by-DTE evidence. Docs: THE_STACK_FULL_LIVE_DEPLOY_STATUS.md + LIVE_TRADING_SYSTEM_RULES.md. NEXT GATES: Mon 09:16 sleeve fills verify; ~2wk suite-Friday review; ~4wk TB reweight sec-18; 15-SEP paper checkpoint.
Verdict: live intraday build was the failure (never validated, negative every era);
multi-day gap-ORB long alive (+16-22bps/trade 2024-26, IN-SAMPLE — OOS consumed).
Best never-died config: 90-min OR, gap>=0.4%, 4-day hold, long, NIFTY>50DMA gate.
Offer on the table: Rs-capped sleeve PAPER book, 90-day soak. Do NOT re-arm live.
Details: research/89_orb_reassessment/results/RESULTS.md

## PENDING 2026-08-31 -- 45-DTE straddle: stress margin ANSWERED, sizing decision owed BEFORE go-live
`research/119.../scripts/margin_stress_live.py` measured the moneyness x DTE margin surface
against the real broker (a straddle m% offside == a strike m% away today, so the adverse-move
axis is bought from Kite, not modelled). Two corrections mattered: compare **margin + MTM loss**
to capital (margin alone never breaches), and use the **pre-premium-credit** margin, since
`final` credits an inflated ITM premium that was never received.
**Result: 3 lots BREACH the Rs 11.96L reserve at an 8% adverse move (110-115% of capital), and
sit at 95% by 5%.** Running 3 lots through +/-8% needs Rs 13.10L. Safe size at this reserve is
**2 lots**; 3 lots wants ~Rs 17-18L once a vol spike is allowed for (extrapolated, not measured).
**And this is a BEST case** -- India VIX was 10.6, near the floor, so the true breach is BELOW 8%.
DECISION OWED (Arun's -- it is a strategy parameter, nothing was changed): cut to 2 lots, or raise
the reserve, before this book goes live alongside NAS. The VOL axis remains unmeasured (5 recorder
days spanning 0.65 VIX pts) -- that half stays dated 2026-11-30.
Detail: `research/119_45dte_short_straddle/results/RESULTS.md` section 7b.

## PENDING 2026-08-27 -- research/134 follow-ups: weight the long-equity books against the short-vol book
r/134 CONCLUDED: the short-vol book's enemy is the **low-vol melt-up, not the crash** -- it has never
lost in a NIFTY down-trend in 75 months (+7.85% in Apr-2020). So no put hedge / jade lizard / skewed
condor is wanted. The offset is **plain long equity at ~25-35% of combined risk**, which we already
run (Momentum-30 Rs20L, Breakout Rs10L, HA-2green Rs20L). Verdict is **weighting guidance, not a new
edge claim** -- nothing is deployed off it yet. Two things owed before acting:
1. **Re-run Stage B/C with the real books' monthly series** in place of the NIFTY proxy -- those books
   are higher-beta than the index, so the realised shape will be noisier than tested.
2. **Per-leg attribution** -- does an index-level sleeve protect the C1 *stock* book, or only the NIFTY
   straddle? research/128 found stock-strangle tails are idiosyncratic, so this cannot be assumed.
Standing caveat for any sizing decision: only **7 down-trend months** in the sample, one fast V-shaped
crash, **no slow grinding bear**. The down-tail is unmeasured, not proven safe -- a multi-quarter bear
would hurt the equity sleeve *and* test the claim that the neutral book is safe there.
Study: `research/134_directional_diversifier/results/RESULTS.md`.

## ⏳ 2026-08-07 — Breakout paper book: cash-model v2 (settlement realism) DEPLOYED, activates TODAY 15:32 IST
`services/breakout_paper.py` rewritten (commit `f45f619`): 4 cash buckets — one slot's ₹
held as a SETTLED buy buffer (earns 0), liquid fund earns 6.5% from T+1, redemptions +
equity sale proceeds settle T+1, a buy triggers a same-day fund redemption so tomorrow's
slot is ready. One-time migration recasts the whole history from fills+NAV dates (interest
₹7,022 → ₹5,298 as of 08-06). Arun approved restarts → one-shot self-removing cron installed:
`32 15 7 8 *` runs `scripts/bp_v2_restart.sh` (post-close restart + auto-verify, log
`/tmp/bp_v2_activation.log`), so TODAY'S 15:45 daily job already runs the new model.
**Verify after 15:33**: the log should say `cash-model v2 live: True`; `/app/breakout-paper`
should show CASH (fund) + BUFFER rows and `bp_state` `cash_model_v2=true`. (Fallback if the
cron missed: `sudo /bin/systemctl restart quantifyd` after 15:30.) Side observation: someone/
something restarted quantifyd at 12:49 IST today DURING market hours — not this session; check
whether it was intentional (other session's NAS deploy?) or a crash-restart.
Frontend already live. Backtest evidence: research/71 G5b (`g5b_cash_ledger.py`) — realistic
18.8% CAGR / −30.5% DD / Calmar 0.62; naive instant-cash model overstates ~0.9% CAGR;
gate-aware buffer (park during risk-OFF) worth +0.85% CAGR — available via
`CFG['buffer_gate_aware']=True` if wanted (default = always-on buffer per Arun's spec).

## ✅ INCIDENT FIXED 2026-08-05 — paper-book state-file race (both weekly books) — commit `abca8ef`
Symptom Arun spotted: NSR-W ₹30 book flat all week while ₹20 book traded. Root cause:
**unlocked concurrent APScheduler jobs doing load-modify-write on the same JSON state.**
Three casualties: (a) NWV JL state CORRUPTED Mon 12:45 (monitor vs :45 pivot check) →
every NWV job crashed all week, position unmanaged Mon 12:45→Wed (audit vs recorded
chain: P&L stayed −₹3.8k…+₹6.5k, NO missed PT/stop; resumed cleanly); (b) NSR-W t30
Monday DTE≤1 TIME close (−₹4,030 @3.35) was **un-done twice** by the stale monitor save
(Mon AND Tue), finally "closing" Wed at a phantom entry-price fill (−₹16,218 — wrong);
(c) t30 missed Monday's new entry (old Tue-expiry cycle still open at 15:14).
FIXES: threading lock + atomic tmp+os.replace saves in BOTH services; Monday entry now
TIME-closes a DTE≤1 leftover first, then enters; t30 history repaired to the true
−₹4,030 Mon close; NSR-W card gained a COMPLETED WEEKS table (per-leg in/out datetimes,
min/max, close reasons). Restarted 15:5x (post-close). **LESSON (generalize): any
multi-job JSON-state paper service MUST lock its load→save and write atomically —
check ha_paper/breakout_paper/momentum_paper/nas services for the same pattern.

## ★ 2026-07-30 (Thu, SENSEX expiry DTE0) — manual close + research/97
- **Live event:** SENSEX rallied +0.43%; the short-CE side bled. Arun manually closed the whole
  SENSEX book (CEs then PEs) — followed the handoff's own E5 guidance ("manage to ~−₹5k, don't ride
  the 286-pt move-stop into DTE0 gamma"). Net day ≈ +₹1,900 (PE decay covered the CE loss).
- **✅ Book LOCKED:** broker flat; **kill flag ARMED** (blocks all NAS+SENSEX entries/re-entries);
  live phantom legs reconciled (no orders). 17 paper-shadow legs left (harmless). Master mode = `mixed`.
  **⚠️❌ MISSED — the kill flag was NOT cleared before Monday.** It carried into Mon 2026-08-03 (a live
  NIFTY day) and BLOCKED the 09:16 live entry (missed, one-shot, unrecoverable). Cleared 08-03 10:34 IST.
  Day-matrix verified: Tue 08-04 → nas_916_atm/atm2/atm4 LIVE, squeeze+OTM paper. **LESSON: a kill flag
  armed at session-end must have a scheduled un-arm (cron), not just a TODO line — a TODO note didn't
  survive the session gap.** Live re-armed for Tue 08-04 (real money) — flagged to Arun.
- **✅ Verified the two 07-29 staged deploys landed** (entry-fill reconciliation + SENSEX ATM2 scope fix).
- **✅ research/97 DONE — INCONCLUSIVE (G2). NO SENSEX stop deploy.** SENSEX exit-stack calibration on
  real chains (14 cycles, DTE0 vs DTE1). Findings: (1) **30% per-leg SL is BAD on expiry** — DTE0 win 14%,
  −964/tr, whipsaws the theta crush (answers "is 30% SL ok on expiry?" = NO); (2) on DTE0 hold/loose wins
  BUT only because **no trending expiry is in the sample** (all moves <0.75%) → can't price the tail the
  stop exists for → do NOT read as "remove the stop"; (3) **DTE1 intraday short straddles look
  structurally unprofitable** net-of-cost (flag: maybe no SENSEX Wed entry). **Recommendation: keep the
  NIFTY-borrowed stops as provisional tail insurance; gather more cycles (esp. a trend expiry) before any
  calibration. Layer B deferred (same benign-sample limit).** `research/97_.../results/RESULTS.md`.
- **[ ] Follow-ups from research/97:** (a) loosen/disable the 30% per-leg SL on DTE0 for ATM/ATM4 (low-regret,
  verify vs a trend expiry first); (b) separate study — is the SENSEX DTE1 (Wed) entry +EV at all?; (c) re-run
  the sweep as more expiry cycles accrue.

## ★ PENDING — guardian findings 2026-07-29 (SENSEX live validation)
1. **entry-fill reconciliation: STAGED, one-shot 15:45 07-29** (`/home/arun/fillfix_stage/`):
   async `_reconcile_entry_fill` in nas_atm_executor base (all ATM-family + SENSEX) — writes Kite
   average_price back to entry_price, rescales sl_price by fill/quote. Verify deploy.log post-close.
   NOTE: nas_executor.py (OTM/base, paper-only systems) NOT covered — extend if those ever go live.
2. **Add SENSEX coverage to scripts/nas_live_guardian.py** — still open (only remaining guardian item).
3. STAGED via 15:33 one-shot 07-29: SENSEX ATM2 scope fix (rupee stop = NIFTY-only; restored 0.4%
   move-stop) + venue-aware lot divisor. Verify /home/arun/atm2fix_stage/deploy.log post-close.
4. **Travel page live-weeks = actual paper fills: DONE 07-29** — new inject_travel.py deployed &
   run (30 cycles, 2 live-book); engine path_week accumulation goes live with the 15:45 restart
   (charts fill from Thu; Mon–Wed path synthesized flat).

## ★ NAS live-book — in flight + queue (2026-07-27)
**Live schedule armed:** NIFTY-916 live Mon/Tue, SENSEX live Wed/Thu, else paper-shadow (2 lots, recorded).
**Portfolio risk manager** (`services/nas_portfolio_stop.py`, 10s job): STOP −₹1,300/lot both venues;
NIFTY **trailing profit-lock** arm ₹2,000/lot + give-back ₹350/lot (committed 07-27, activates at the
after-close / pre-open restart); SENSEX **TP** +₹1,667/lot. + 15:16 EOD square-off backstop + BFO naked-
survivor auto-arm. Guardian (`.claude/agents/nas-live-guardian.md`) mandate broadened → full periodic
**SYSTEM REVIEW** (performance · per-system contribution · pattern-drift/edge-decay · param re-calibration
· exec health · **RED/AMBER/GREEN** + ranked recs). First review run in progress 07-27.

### 2026-07-28 (Tue, expiry-week) — manual close + ATM2 exit redesign
- **Live event:** Arun manually closed the live 09:16 ATM2/ATM/ATM4 book on an **expiry-gamma
  exit concern** (0.4% spot-move stop crystallises an asymmetric loss near expiry — losing leg
  balloons, OTM leg already ~0, no cushion). Broker FLAT, phantom DB legs reconciled (no orders).
- **Rest of today = PAPER:** all 8 NAS variants forced `paper` via `/api/nas/master-mode`; kill
  flag cleared so paper entries continue + record. **⚠️ master mode persisted as `paper` — MUST flip
  back to `live` before Wed's SENSEX session** (write `backtest_data/nas_master_mode.json`={"mode":"live"}
  and/or POST master-mode live; folded into the post-close deploy below).
- **[ ] STAGED — post-close deploy (after 15:30 IST): ATM2 exit redesign (research/96, APPROVED).**
  Replace the 0.4% move-stop with a **DTE-agnostic ₹2,500/lot rupee MTM stop**, **drop** the 30%
  per-leg SL, **one-and-done** (no re-center). **ATM2 only, both variants** (`nas_atm2` +
  `nas_916_atm2`); other 6 have `move_stop_pct=None`, untouched. Calibration (68d): ₹2,500/lot
  near-expiry +2,153/tr vs current +1,386, tail ≈ same, and fixes the current stop's far-DTE bleed.
  **Exact edits + deploy checklist: `research/96_atm2_exit_rupee_stop/ATM2_EXPIRY_EXIT_RUPEE_STOP_STATUS.md`.**
  Bundle the master-mode→live flip + restart + Wed-day-matrix verify with this deploy.
- **Finding (paper, sign-off needed) — squeeze 2nd-sleeve shape (research/96 §stack test):** stacking
  a 2nd ATM straddle at the SAME strike deepens the combined worst day (−37k vs −32k for 916-alone) —
  confirms the concentration risk. An **OTM strangle ±100** is better: higher total (+93.3k best) with
  NO tail worsening. Combined with research/95 (squeeze timing sub-optimal) → "if stacking at all, stack OTM."
- **Finding (11.5yr NIFTY 5-min):** a tight morning (consolidation by 09:30) does NOT foreshadow a
  bigger breakout — calm mornings mostly stay calm (corr +0.58, P(≥1% rest-of-day move) 13% tight vs
  36% wide). Volatility persists intraday; the squeeze selects calmer days (lower per-unit risk).
- **[ ] Optional (not approved):** add a `_broker_holds_any` guard to `exit_all_positions` (EOD/emergency
  path is unguarded — a phantom short would be bought-to-cover into a NEW long; today the move-stop guard
  caught it first). One-liner. Arun deferred; re-offer if desired.

**QUEUED (take up in order, only after the guardian report + the trailing-stop restart):**
- **Options Behaviour Study page** `/app/options-study` (React + uPlot) — ATM straddle (CE+PE combined) + OTM.
  - [x] **Phase 1 LIVE (2026-07-27):** NIFTY ATM straddle — intraday curve (+CE/PE split, day picker),
    all-days normalised-100 overlay w/ median path, clickable daily-decay strip. `scripts/options_study_agg.py`
    → `static/app/options_study.json` (67 days, 5-min series + daily summary); daily 15:45 cron appends.
  - [x] **Phase 1b enhancements (07-27):** weekday filter, start→close time window (all charts window-aware
    + aligned), NIFTY spot dotted on intraday (right axis), median-decay-by-weekday chart.
  - [x] **Phase 2 LIVE (07-27):** OTM strangles (agg stores ±100/200/300pt series); ATM-vs-OTM median overlay,
    median-decay-by-DTE chart, weekday×DTE decay heatmap — all window-aware.
  - [ ] **Phase 3:** weekly rollup + BANKNIFTY/SENSEX.
- [x] **Squeeze-ATM entry-trigger study DONE (research/95, 07-27):** SIGNAL/actionable — the ATR **squeeze
  trigger is SUB-OPTIMAL**. Early time entry wins: 09:30 +₹633/tr, 09:16 +₹576/tr BEAT squeeze +₹407/tr;
  late (10:00+) and price ±100 LOSE (11-12:00 ~ −₹1,000/tr). Edge = enter early to bank the morning theta;
  the squeeze wait gives it up + skips 15 no-squeeze days. **Recommend: paper squeeze family (nas_atm/atm2/
  atm4) drop the squeeze wait → enter 09:16/09:30** (sign-off needed). results/RESULTS.md.

## ✅ RESOLVED 2026-07-27 — research/94: NWV → jade lizard / IC automation

**Closed 2026-08-17 during the app review:** `services/nwv_trade.py` PAPER book was deployed
27 Jul with the only rule that survived the bake-off (never roll; exit the threatened side on a
close beyond S1/R2). Directional JL/IC mapping stays NO EDGE. Original entry below.

### (original) ★ DECISION PENDING — research/94: NWV → jade lizard / iron condor automation — 2026-07-27
Arun's ask: automate the Nifty Weekly View into JL/IC trades ("construct like so" =
his live 27-Jul position: short 23450 PE / long 22900 PE / short 24500 CE / long
24700 CE, 10 lots, 4-Aug = pivot-anchored S1/R2 asymmetric condor). **Bake-off DONE
same day on real option EOD 2020-02→2026-07** (318 Mondays, replayed live engine,
net of costs, r/89 liquidity rule): `research/94_nwv_jade_lizard_ic/results/RESULTS.md`.
**Verdict: NO EDGE for the directional mapping; user's exact construction ≈ breakeven
always-on (+₹145/wk, PF 1.01) and NEGATIVE on BULL-view weeks (−₹1.0k/wk, t −0.18 —
this week's deployment is its weakest bucket). Only weak SIGNAL: NEUTRAL-week
far-OTM premium selling (true JL naked S1 put: +₹14.3k/wk, PF 1.99, t 2.22 — weak
after ~90 cells; tail −₹395k/wk at 10 lots; ₹10-11L margin). ICs flat everywhere.
Bear-view inversion re-confirmed (bull structures win on bear weeks, t~1.2).**
**RESOLVED same day — Arun picked (b): his JL template, ALL non-ignore weeks, PAPER.**
`services/nwv_trade.py` BUILT + DEPLOYED (registered in app.py after nsrw, `.bak_nwvtrade`;
activates at next 09:00 pre-open restart — no market-hours restart done). Mon 09:50 entry
from live Phase-0 view, next-wk expiry, 10 lots, sells@bid/buys@ask, PT50/stop−1×,
Fri 15:15 out; W2026-07-27 cycle SEEDED from Arun's real fills (credit 44.44pts) so the
paper book mirrors his live 4-Aug position. Kill: POST /api/nwv-trade/kill-switch.
**PHASE-2 (adjustments — Arun asked when/how to adjust): NEVER ROLL, EXIT.** Both roll
styles (defensive roll-away AND credit-chase = his W30 habit) re-widen the tail; best =
**exit threatened side on daily close beyond weekly S1/R2** (+₹5.8k/wk PF 1.62 t 2.48 vs
hold +₹4.0k PF 1.29; worst −₹230k→−₹144k; fixes 2021). Wired into executor as 15:25
pivot-exit job (combo with PT/stop untested — paper book is the forward test). 4th
independent confirmation: r/92 hold>adjust, June morph net-neg, mentor W30.
- [x] **Card BUILT on /app/nwv (2026-07-27, VPS bundle index-B4ev1EO_.js)**: level-watch strip
  (S1 / spot / R2 + distances + 30-min check rule), legs table, MTM, PT/stop ₹, history,
  kill button. `frontend/src/pages/NwvPaperCard.tsx` + Nwv.tsx/module.css patched ON VPS
  (laptop frontend stale — do NOT scp laptop copies of Nwv.tsx/css over).
- [x] **Book ACTIVATED intraday 07-27 11:38** via standalone one-day runner
  (`research/94.../scripts/standalone_today_runner.py`, exits 15:31); 30-MIN pivot checks
  (:15/:45) per phase-3 (30m monotonic best: t 3.10, worst wk −₹74.5k, maxDD −₹1.42L vs
  daily t 2.48). Executor pivot job moved to 30-min cadence.
- [x] Verified 07-31: /api/nwv-trade/state 200; **JL WEEK 1: PT hit Tue 07-28 11:01,
  all 4 legs closed, net +₹14,586** (robot banked +50% of credit in ~25h).
- [x] **Leg-detail upgrade BOTH books (2026-07-31)**: per-leg px_max/px_min tracking +
  stop/exit reason_detail + full leg snapshots in history (nsrw_paper.py VPS-patched
  `.bak_legdetail` — laptop nsrw copy STALE v1.2; nwv_trade.py updated); recorder
  backfill script `research/94.../scripts/backfill_leg_maxmin.py` (option_chain.snapshot_time);
  cards show pretty legs ("NIFTY 23550 PE · 4 Aug"), entry/exit datetimes, Min/Max
  columns, reason text (Nas.tsx NsrwBook + NwvPaperCard, bundle index-DAQlahUo.js).
  Notable: stopped strangle week — 23550 PE spiked 112.75→6; 24450 CE stopped @65.25;
  new 23950 PE maxed 43.55 vs 44.2 stop (0.65 pts from re-stop).
- [x] **RESTARTED 07-31 11:04 IST** (user-cleared mid-market: "no live trades" — verified
  first: only a paper NAS-OPT position open, master-mode intact after). Live max/min
  tracking confirmed ticking (23950 PE min updated live); backfill re-run post-restart.
- [x] Git commit + push DONE 07-31 (`e5409d8`): both services, cards, bundle,
  research/94 scripts, book states. (research/94 folder + app.py were already in `51e1e03`.)
- [ ] Git commit research/94 + services/nwv_trade.py + app.py (on VPS).
- [ ] Watch Fri 15:15 exit + weekly /trade-mentor comparison: Arun's manual JL vs robot.
Prior Phase-1 design: `docs/NWV-PHASE1-TRADE-PLAN.md`. Infra byproduct: NIFTY50 30-min
derived from 5-min through 2026-07-16 (was stuck 2026-05-05); script in research/94 scripts/.

## ★ LIVE — research/90 NSR-W v1.2: **G5 PAPER BOOK LIVE on VPS** (2026-07-24) — first auto-entry Mon 07-27 15:14
`services/nsrw_paper.py` — Mon 15:14 entry, next-wk expiry, ₹30/leg 10 lots, GTT stop 2×, PT50,
one roll-away, EOD recenter 1.5× (user's idea — beat exit-heavy-leg, t 5.84), out DTE≤1. Card +
positions on /app/nas; study card /app/backtest/nifty-strangle-rules-research90 (embeds travel
report); travel page auto-regens 15:55 cron, LIVE-PAPER chips for weeks ≥07-27. Kill:
POST /api/nsrw/kill-switch. **SENSEX tested → NO (66-day replay: all morning entries deeply
negative, best cell noise; wide BSE spreads).** WATCH: first live cycle vs replay; weekly
/trade-mentor review = Arun vs robot. Prior detail below.

## (superseded header) research/90 G2 notes — 2026-07-24
**G2 DONE (pessimistic gap-aware fills, 58k rows, 22s):** monthly stop-family SURVIVES at 2.0–2.5×
(best: 2.5%OTM + stop2.5× + PT50 → net 47.8 pts/cycle, t 2.61, worst −301) but **stop 1.5× monthly
DIES under real fills** (post-22 negative). Post-stop answer: **monthly = flat both** (roll re-fattens
tail −161→−670); **weekly = roll-away-once at stop 1.5× = best family in study (t 4.73, p5 −39,
7/8 yrs positive, 2020 flat)**. Indicator exits (ATR/ADX/VIX-jump) all lose to premium stop; VIX≥1.25×
entry = higher mean, 2.7× tail (rejected for loss-min objective). Monthly condor UNTESTABLE at EOD
(stale wing marks — worst exceeds structural cap). NSR v0.9 spec: RESULTS §5. **NEXT: (a) G5 paper
book — NSR monthly + weekly-roll sleeves, 10 lots, alongside straddle V1/V2 books, weekly
human-vs-robot mentor comparison; (b) replay W30 on chain recorder; (c) CPR/VIX entry gates with
regime controls.** Original G0/G1 detail below.
W30 mentor review (`mentor/reviews/2026-W30.md`) proved manual strangle management is a measurable
drag (untouched Monday strangle +₹12.7k vs 22-leg managed +₹6.8k; root habit = calm-day
credit-chasing rolls toward spot; margin measured 97% utilized). Arun approved building a
rules-based NIFTY strangle system (entry/exit/adjust/react; CPR + VIX gates; emotions out).
**G1 DONE same day — SIGNAL, PASS → G2** (`research/90_nifty_strangle_rules/results/RESULTS.md`):
monthly strangle + per-leg premium stop 2.0–2.5× = net t≈2.0–2.4, tail cut 6× (worst −1,878→−298
pts), monotonic stop family; giveback harmful; weekly arm t 2.5 but gap-tail unfixable at EOD →
wings/intraday. VIX≥16 helps monthly/hurts weekly; narrow weekly CPR GOOD (opposite r/67 sign —
regime confound, don't gate yet). **NEXT = G2:** (1) pessimistic gap-aware stop fills
(make-or-break), (2) iron-condor arm (fixes 97%-margin problem), (3) per-year tables + 2020
isolation, (4) r/89 reconciliation memo, (5) chain-recorder intraday validation incl. W30 replay.
Runner: `research/90_nifty_strangle_rules/scripts/run_g1_daily_sweep.py` (12s on VPS).
NEW INFRA: mentor daily account capture cron LIVE on VPS (15:45 IST →
`/home/arun/mentor/daily/*.json`) — weekly reviews data-complete without Console exports.
Mentor system: `.claude/skills/trade-mentor/SKILL.md` + `mentor/LEDGER.md` (invoke /trade-mentor).
W30 review COMPLETE: `mentor/reviews/2026-W30.md`.

## ✅ CONCLUDED 2026-07-26 — research/93 (VPS numbering): HMA 30/44 weekly swing (Nitin Hulaji, Market Aur Main Ep.5) — **SIGNAL (not investable)**
Arun's ask: test the video's weekly swing system — HMA30/44 retracement zone + MACD(21,39,9)
histogram turn after ≥8 bars below zero + RSI(9) 3-SMA × 21-WMA cross; SL below swing low,
target prior swing high. Full daily universe (629 names pass screens) resampled to weekly, 2001–2026.
- **Per-trade edge REAL**: net(25bps) +4.62%/tr (n 4,537) vs year-matched random-entry control
  +1.45% → **+3.17%/tr, Welch t 7.15**; all 27 sweep cells beat control (flat grid); both
  decade-halves positive; super-winner-proof; costs irrelevant at 12.6-wk holds.
- **G4 book FAILS**: 20-slot 5%-NAV = 6.70% CAGR / DD −48.9% vs NIFTYBEES 12.75% / −58%;
  idle-cash-in-index variant worse (8.93%, DD −63.8%). Post-crash signal clustering (52/wk vs
  20 slots) turns away 65% of candidates in the best vintages; median trade −6.1%, 61% stop-outs
  (tail-carried); 2020 outlier (+49.7%/tr; ex-2020 diff ≈ +1.7%).
- Untested: video's +3%-day→sell-10% overlay (moot until a book beats the index).
- **If ever revisited**: NIFTY>200DMA regime gate (r/71/75 precedent), contention ranking
  instead of alphabetical, trailing exit instead of fixed target.
**Phase 2 (2026-07-27, optimization for investability): improved but verdict unchanged.**
Donchian-10w trail replaces target → per-trade net +11.11%/tr, PF 2.72, t 13.9 (2.4× the
taught target rule — r/71's "never a target" again). Best book (trail, 40×2.5%, ungated):
**15.04% CAGR / Sharpe 0.87 / DD −51.2% / Calmar 0.29** vs NIFTYBEES 12.75% / 0.73 / −58% /
0.22 — beats the index on all headline metrics but FAILS the pre-set MaxDD ≤35% bar; excess
lumpy (−28pp 2018, −24pp 2025); best-of-14-cells haircut. **Structural finding: regime gate
HURTS retracement-reversal systems** (alpha fires below the 40w SMA — 2009/2020/2023
vintages); R:R contention ranking never helps. Calmar 0.29 ≪ existing books → shelve.
Verdict: `research/93_hma_weekly_swing/results/RESULTS.md`. STATUS-MD:
`HMA30_44_MACD_RSI_WEEKLY_SWEEP_STATUS.md`. Publish-to-app (backtests.ts card) = optional chore.

## ✅ CONCLUDED 2026-07-24 — research/91: 20/200-SMA "Picture of Power" retrace-break (iFundTraders RBI&GO) — **NO EDGE**
Arun's ask: test the iFundTraders "RBI & GO" setup from the video clips — buy a red pause-bar's
high (sell a green pause-bar's low, short mirror "NARROW TO WIDE") when price is NEAR a **rising**
20-SMA stacked over the 200-SMA, hold while trending, exit on 2–3×ATR drift away. Tested long+short,
5/15/30-min + daily, 12 deep 5-min names 2015→now, gross+net@5bps.
- **G1:** loses **gross** on 5-min (the taught TF) — long −0.006% / short −0.005%/tr, net −0.056%,
  win 28%, avg hold 3.8 bars, t≈−38. Tight red-low stop shredded by noise; ATR target rarely hit.
- **G2 (84 cells):** no cell clears gross>0∧net>0∧t≥3. Stricter "rising 20-SMA" = worse; overnight
  hold + SMA-cross exit barely help; only daily-LONG net-positive but t 1.1–1.8 (n≤233); daily
  SHORT mirror loses (asymmetry ⇒ not a real structure).
- **G3 drift control:** setup +0.78%/tr < random-entry-in-uptrend +1.03% < all-regime-bars +0.90%.
  Daily "profit" = 100% survivor drift; the pause/near mechanics **subtract** 0.12–0.25%/tr.
Verdict: `research/91_sma20_200_pullback/results/RESULTS.md`. SHELVE — do not re-litigate intraday
(loses gross). Engine + G1/G2/G3 runners committed. Mandatory drift-control rule (r/87-88) applied.

## ✅ RESOLVED 2026-07-21 — research/86 HA 2-green-no-wick 30m LONG: G5 paper book BUILT

**Closed 2026-08-17 during the app review:** the ₹20L paper book has been live since 21 Jul
(`services/ha_paper.py`, 81 sleeves, cash sleeves — the construction question was answered by
shipping it). Soak review ~Oct 2026. Original entry below for the evidence trail.

### (original) ★ PENDING DECISION — research/86 HA 2-green-no-wick 30m LONG: build the G5 paper book? — 2026-07-20
**STRATEGY CANDIDATE — the first full survivor of the r/81-86 program** (IS t3.7 → Val t6.0 →
OOS t3.7 PASSED; OOS book 11.6% CAGR vs bench 5.6%, DD −11%, Calmar 1.03, beat bench all 3 OOS
years incl. the 2026 down-tape). OOS consumed. Watch-item: per-trade fade 47→36→25bps across
splits. NEXT: G5 paper book — construction choice needed (cash-CNC sleeves vs futures subset;
fractional per-name sizing is the practical question). Verdict: `research/86_heikin_patterns/results/RESULTS.md`.

## ✅ CONCLUDED 2026-07-22 — research/89: Short straddle (calm + flip + real-IV mgmt) — NO ROBUST TRADEABLE EDGE
User idea: sell monthly straddles into predicted-calm stocks; later reframe: don't hold a month,
manage actively (take profit / cut on criteria) for better-probability-of-calm shorter holds.
Findings: (1) **sell-into-calm is INVERTED** — calm is the WORST time (vol mean-reverts up; calm
persists only ~35%/mo, 53-73% shorter). (2) Built **REAL NSE F&O stock+index option EOD history
2016→now** into `backtest_data/market_data.db` table `nse_options_bhav` (30.3M rows, 83 syms;
`download_nse_bhav_stocks.py`; IV via BS inversion) — permanent asset, removes the "no real stock
IV" blocker. (3) Mgmt reframe CONFIRMED useful: take-25/50%-profit beats hold-to-expiry, ~18d hold,
iron fly caps tail (per tastytrade/OptionAlpha/Varsity). (4) INDEX real-IV: short-vol strong pre-2021
(+315bps/trade OOS) but **DECAYED to ≈0 post-2022** retail options boom. (5) STOCKS looked huge
(+146bps t16 every year) but **G6 LIQUIDITY FILTER KILLS IT** — iron fly +140→−82bps (t−7.9) once
you require real ATM volume≥50; **105% of the apparent profit came from untraded stale-priced
options**; only 9/39 liquid names positive (noise). **BINDING LESSON: any options backtest here MUST
filter real traded volume/OI.** Verdict: NO ROBUST TRADEABLE EDGE — don't trade. Full writeup:
`research/89_short_monthly_straddle/results/RESULTS.md`. Reusable: engine + run_g4/g5/g6.

## ✅ 2026-07-20 — Momentum-paper weekly gate re-entry LIVE (+ market-hours restart incident, no harm)
Patched `services/momentum_paper.py`: when FULLY in cash and the weekly gate is ON, re-enter the
top-8 immediately (r/41-P27 validated) instead of waiting for month-end. Verified working: book
re-entered 2026-07-20 (ADANIENSOL, POWERINDIA, GVT&D, LAURUSLABS, ADANIPOWER, BHARATFORG, BHEL,
IDEA; 100% invested, gate ON). INCIDENT: the deploy restart accidentally executed 10:24 IST Mon
(market hours; Friday's "tonight" approval executed after a session gap without re-checking the
clock). nas-live-guardian full battery = PASS (SL monitors re-armed in 23s, ticker live, Kite↔DB
reconciled, no unmanaged exposure). Prevention memorized: time-window approvals VOID after gaps;
standalone date+market-hours check before any VPS state change.

## ✅ CONCLUDED 2026-07-17 — research/83: Turtle (Dennis) on F&O equities — shorts closed at ALL horizons; turtle-EQ recorded, no book change
T1: turtle multi-week shorts = worst short result yet (t −11, S2 0% yrs) → with r/81+82 the equity
short-swing question is PERMANENTLY closed. T2 bakeoff: turtle S1+S2+2N EQUAL-notional beats live-book
rules IS (Calmar 0.45 vs 0.37; N-sizing loses — 3rd sizing failure) but family ~flat 2018-23 → user
chose RECORD ONLY; live breakout-paper soak decides. OOS unconsumed. `research/83_turtle_equities/results/RESULTS.md`.

## ✅ CONCLUDED 2026-07-17 — research/82: Medium-swing 5-15d long+short — shorts NO EDGE (final), longs converge on live r/71 book
User-mandated extension of research/81 to 5-15 session holds incl. futures shorts. **Shorts: all 24
cells negative — directional short swing dead across the whole 1-15d spectrum (combined r/81+r/82).**
Longs: real cyclical breakout edge at 10-15d (t 3.7-4.0 IS, Val + but purely 2020-21) — same family as
research/71, already LIVE as the breakout paper book; per-trade 200DMA gating fails (lags tops). No new
build; **OOS unconsumed.** Verdict: `research/82_medium_swing/results/RESULTS.md`; crash doc
`MEDIUM_SWING_82_STUDY_STATE.md`.

## ✅ CONCLUDED 2026-07-16 — research/81: Swing Edge Discovery — **SIGNAL (decaying), not investable**
Two-day systematic study (~170 pre-registered cells, 8 families, 6 book constructions,
1 authorized OOS touch). Gap-up+ORB long was real (IS t=5.6 / 77 F&O names) but the OOS
look caught temporal decay: +33bps 2024 → +5 2025 → −27 2026; both books fail gates.
7 families buried with data (incl. all shorts, MA crosses, coin-toss first-candle).
Byproducts: 5-min history 2015→2024 backfilled for 381 symbols + adjustment repairs
(KOTAKBANK-class splice bugs) + BANKNIFTY 5-min 2015+. Full verdict:
`research/81_swing_edge_discovery/results/RESULTS.md`. Crash doc: `EDGE_DISCOVERY_81_STUDY_STATE.md`.
Follow-ups ALL CLOSED 2026-07-17: study published to /app/backtest/gaporb-morning-strength-research81
(+factsheet); NIFTY50/INDIAVIX 5-min repaired to current via index tokens; OR-width filter too weak
(sizing lesson stands); B-family 5-min timing = noise; NO workable trailing decay-gate (6-12m health
gates lag abrupt decay — general live-book kill-switch lesson). Nothing further queued.

## ~~★ ACTIVE~~ — research/81: Swing Edge Discovery study (brief: docs/Trading-sytem-research-prompt-fable.md) — 2026-07-15
Multi-family systematic search for automatable 2-4-day swing systems, net-of-cost.
Crash-recovery master: `EDGE_DISCOVERY_81_STUDY_STATE.md` (repo root, VPS canonical).
- [x] Phase 0 data audit (user signed off) + unit-tested canonical engine (32 asserts)
- [x] 5-min history backfill 2015→2024 for ~370 names LAUNCHED on VPS (~20h; resumable;
  STATUS: `research/81_swing_edge_discovery/NIFTY500_HISTORY_BACKFILL_5MIN_RUN_STATUS.md`)
- [x] Night-1 IS screens (108 cells): A1/C1/D1/E1 **NO EDGE**; B1+B2 **SIGNAL** (deep-z
  short-side reversion, +32bps, 8-10/13 yrs pos, t1.5); F1 **SIGNAL — best** (NIFTY ORB
  long 4d-hold, +15bps net @1bp cost, t2.3, 6/7 yrs pos & improving)
- [ ] Post-backfill repair pass (splice refetch incl. KOTAKBANK, BANKNIFTY via token
  260105, 22 daily-hole symbols) → re-audit
- [ ] EXP-F2 filters on F1 locked cell → Val confirmation; EXP-A2 stock ORB; B 5-min timing

## ✅ DONE 2026-07-08 — research/75: faithful backtest of the "Nifty-250 Momentum" video
Replicated the Quantinuous "Only Momentum Strategy You Need for Nifty 250" video on
survivorship-free data, 2006–2026. **Verdict: STRATEGY-candidate — replicates & EXCEEDS the
claimed return (net 31.9% CAGR / 292× vs advertised 27% / 100×) but DD is deeper (−31.6%
daily vs claimed −23%; the −23% only shows on modern 2014+ w/ risk-adj momentum).** Key
attribution: **the NIFTYBEES>100EMA cash gate is the whole risk story** (remove it → DD −66%);
**the video's per-stock 50>100>200 EMA filter is inert-to-harmful** (removing it *raises* CAGR
to 34.7%). Cost-robust, low turnover. **Not new alpha** — same family as the live momentum-paper
₹20L book (research/62). Files: `research/75_nifty250_momentum_top15/` (RESULTS.md/_P2/_P3 + tearsheet.png).
**Phase 2/3 (2026-07-21):** universe×momentum sweep → best risk-adjusted = **midcap + 6-month RS
(Calmar 1.26)**; combo (mid+small) = highest CAGR 43.5% but −42% DD (uninvestable). **Gate is
IRREPLACEABLE** — no per-stock quality/ATH/exit combo substitutes (best gate-less DD −46%).
**PUBLISHED:** `/app/backtest/nifty250-momentum-video-research75` (built on VPS) + Artifact
`claude.ai/code/artifact/f7cccc3d`. ⚠ research/75 folder **not git-pushed** → app page's GitHub links 404 until pushed.

## ★ QUEUED — Aurum: arm the research/75 winner as a selectable engine (paper-first) — 2026-07-21

User approved (full gated process; wait for phase-3 winner — now known). Arm **gated midcap RS-120/126**
(a higher-CAGR/higher-DD sibling of Aurum's existing `midcap_smoothest`) as a distinct selectable engine
in the `aurum` repo strategy registry, **paper-only** (`EXECUTION_LIVE_ENABLED=False`). **NEXT = write the
GATE-A design doc** (`aurum/docs/`) for user approval BEFORE any code. Confirm exact spec at GATE-A: bare
research/75 variant (higher return, −29% DD) vs adding smoothest DD-filters. Parity-check vs research/75 +
tests before GATE-B. Note: winner ≈ Aurum's default family, so this is a more-aggressive variant, not new alpha.

## ✅ LIVE 2026-07-07 — 9:16 NAS systems armed REAL MONEY (2 lots, all weekdays)
`nas_916_atm/atm2/atm4` → `live=True` on all 5 weekday DTEs, **2 lots**; squeeze `nas_atm/atm2/atm4`
forced **PAPER** (`live=False`, shadow kept); master-mode=**live**. ATM2 keeps the 0.4% move-stop
(revalidated best on 53d; strike-gate mechanic C was worse). Activates at the 09:00 preopen restart;
first live fire 09:16 Wed 07-08. **Wed is knowingly −EV** (≈−₹2k/lot ATM2; Thu flat, Fri +, Mon/Tue
edge) — user chose all-days. Commit `530d99c`. Kill: `/api/nas/kill-switch`. Paper-shadow stays 10 lots.

## ★ QUEUED — research/75 book-level P&L trailing-stop (optimize, then implement) — 2026-07-07
NAS book intraday P&L hit **+₹75k then gave back to +₹40k (~47%) in minutes** (short-gamma straddle
book). Want an optimized **trailing profit-stop**: flatten the whole NAS book when day-P&L retraces X
from its running peak. User: first-pass assessment done; **IMPLEMENT later**.
- **First-pass (14-day single-straddle proxy) = too thin to lock a number.** Directional hint: ARM the
  trail only after a real profit (~₹2k/lot), then lock on a **~25–30% retrace** from peak (helped 4 /
  hurt 0 in that subset); arming on small peaks hurts more than helps. Proxy script `/tmp/pnl_trail.py` (VPS).
- **Do it properly:** reconstruct the ACTUAL multi-system book intraday P&L from recorded trades
  (`nas_*_positions` entry/exit) marked per-minute vs `options_data.db`, ALL sessions = the real curve.
  NO stored intraday P&L series exists (app computes it live). Sweep (arm-₹, retrace-%) + abs-₹ variant;
  objective = maximize total locked = give-back saved − winners cut. Then implement as a book overlay.

## ★ ACTIVE — V2 executor + inside-week breakout sleeve (build) — 2026-06-10
Spec: `research/61_v2_feature_attribution/V2_EXECUTOR_AND_BREAKOUT_SLEEVE_BUILD_SPEC.md`.
- [x] **research/61 causal-feature attribution DONE.** Only vol-COMPRESSION separates losing weeks:
  daily CPR<0.10% + **inside-week** (NEW, independent). Combo skip → Calmar 1.03→**2.00**, DD −1.17L→−0.78L.
  RSI/MAs/Ichimoku/pivots/range-breaks = no signal. App study UPDATED (new "Causal-feature forensic" block).
- [x] Decisions LOCKED: V2 live gates on **combo skip (CPR<0.10% OR inside-week)**; paper-first, SHORT
  (~2-4wk) compute-confirm window then promote; 10 lots/650 (~₹9.6L margin).
- [x] Inside-week breakout sleeve (paper-only): UP-break→call DEBIT spread (runner edge); DOWN-break→
  broken-wing fly skewed down (no edge, premium+capped). Case A late-entry sim FAILED calib (needs AlgoTest);
  bear-rescue filters FAILED (n=156).
- [x] **Pure signal layer DONE** `services/v2_breakout_signals.py` (smoke-tested). NB: market_data.db NIFTY50
  daily STALE (ends 2026-03-19) → executor pulls fresh daily bars from Kite.
- [x] **EXECUTOR BUILT + DEPLOYED 2026-06-10** (user cleared restart, no live trades that Wed). `services/
  v2_ironfly_api.py` (mirrors nas_opt.py: paper executor + `register(app,scheduler)`) + `services/
  v2_breakout_signals.py`. Routes `/api/v2-ironfly/{state,scan,kill-switch}` + `/api/v2-breakout/state`;
  APScheduler entry(09:20)/monitor(3min)/breakout(15:20) mon-fri. app.py patched (1-line register, `.bak_v2if`).
  Straddles.tsx "V2 Engine" card wired + frontend rebuilt. VERIFIED: paper fly entered live (SELL 23350 CE/PE +
  BUY 23850/22850, net 352.5, VIX 15.5, exp 06-23) + monitor marks P&L. PAPER-only (force_paper). DB
  `backtest_data/v2_ironfly_trading.db`.
- [ ] Promote to live after ~2-4wk paper compute-confirm (verify CPR+inside-week day-by-day vs backtest); set
  force_paper=False + live_weekdays. Optional: watchdog coverage + SSE stream (currently 30s poll).
- [ ] AlgoTest (USER): (a) Case A conditional-late-entry run; (b) Case B call-debit-spread on inside-week up-break.

## ✅ research/62 — Momentum-30 ETF sub-selection — STRATEGY candidate (G1→G3 PASS) — 2026-06-10
Folder: `research/62_momentum_etf_subselect/` (STATUS-MD + `results/RESULTS.md`). Runs on VPS.
New system: piggyback a factor index instead of our own selection. **Reconstructed Nifty 200
Momentum 30 from methodology** (NO factsheets — PIT top-200 by traded value → 6m/12m score →
top-30), then hold a concentrated buffered sub-basket. Reuses research/41 `02_rs_sweep.py` core
(`pit_universe`/`rs_scores`) + new daily-marked engine + Donchian + gate.
- [x] G1 probe (8 cells) + G2 sweep (288 cells) DONE on VPS. **Winner = `rsblend N8 buf22
  gate100 Donch15`: CAGR 33.4% / net-tax 29.0% / MaxDD −17.0% / Sharpe 1.78 / net-Calmar ~1.5–1.7**
  (beats NIFTYBEES 12.3%/−36% AND research/41 keep-top8 ~1.66).
- [x] KEY FINDING: **gate + Donchian are complementary** (gate alone −29%, both −17%) — confirms
  research/41 "gate irreplaceable", extends it. Donch-15 ≫ 20 ≫ 50. N8 sweet spot. Buffer irrelevant.
  Plain 6m/12m RS beats the fancy risk-adjusted score once DD-controlled.
- [x] Robustness PASS: cost-stress to 60bps (monotonic), super-winner guard (Calmar holds 1.79
  without top-3 names = breadth not multibaggers), 288-cell plateau, 11/13 yrs beat index.
- [x] **TEARSHEET + PUBLISHED** to `/app/backtest/momentum30-subselect` (4th card). NB: build the
  React app ON THE VPS (laptop `frontend/` is stale — a laptop build dropped the V2 study; see
  memory `laptop_frontend_stale_build_on_vps`). All 4 study slugs verified in live bundle.
- [ ] **G4 next:** **tighten the universe definition** (currently loose = "any stock with data" +
  ≥75-day floor) → add explicit floors: listing-age ≥252d, price ≥₹20, ABSOLUTE turnover ≥₹25cr,
  data-completeness ≥90%, equities-only, THEN top-200 by traded value. **Tune the floors AGAINST a
  real factsheet** (do the factsheet validation first, then pick floors that reproduce the index's
  actual holdings) — user explicitly deferred this to G4 (2026-06-11). Plus: correlation/cluster-
  stress DD (N8 leans PSU/defence); walk-forward + 2019-stress note. Then → G5 paper soak on VPS.
- WHY paused before G4: natural gate checkpoint — confirm with user whether to build the tearsheet/
  publish now or park as a validated candidate.
- [x] **PHASE 2 — universe-band capacity study DONE (2026-06-30).** Scripts `62d_universe_bands.py`
  (sqrt market-impact model, fixed-AUM), `62e_combos.py` (multi-sleeve combos), `62f_runner_capture.py`,
  `62g_fairgate_diversified.py`. Findings: (a) **top200 net-optimal at every AUM** (net Cal 1.34/0.71/0.25
  @₹1/10/50cr); top500/small higher GROSS but NEGATIVE net at ₹10cr (participation 1,000–98,000× ADV =
  untradeable). (b) **No combo beats top200.** (c) **Runner-capture:** held 5/5 of in-universe runners
  (+130–250%/name); 25/30 big runners live in 200–500 (outside our net) AND un-tradeable. (d) **Fair-gate
  correction:** band-matched gate lifts smallcap GROSS Cal 1.09→1.67 (I'd under-rated lower-cap) but net
  still collapses (capacity wall). (e) **Diversified midcap sleeve** (N≈30, fair-gated) IS tradeable
  (participation→0.8×) but net Cal ~0.44 < top200. **Verdict: top200 stands; lower-cap momentum is a
  gross-only mirage at size.** All in STATUS-MD Phase 2a–2d.
- [x] **DE-CORRELATION BLEND test DONE (2026-06-30, `62h_blend.py`):** corr top200↔div-midcap = **0.69**
  (too high); blending monotonically LOWERS Calmar gross (2.21→1.49) and net@₹10cr (0.71→0.60). A midcap
  sleeve does NOT help even as a diversifier. **Lower-cap momentum CONCLUSIVELY rejected at every angle**
  (concentrated/fair-gated/diversified/blended). top200 stands. Phase 2 fully closed.
- [ ] **G4 still owed** (deferred): tighten universe (floors, factsheet-tuned), correlation/cluster-stress,
  walk-forward, then G5 paper. **NB: Phase-2 scripts + STATUS/CSVs are UNCOMMITTED on VPS+laptop** → next git sweep.

## ✅ research/62 LIVE PAPER BOOK — ₹20L Momentum-30 deployed (G5 soak) — 2026-06-30
`services/momentum_paper.py` (PAPER only, never places orders) + `/api/momentum-paper/*` +
`/app/momentum-paper` page + sidebar "Momentum ₹20L" + LIVE-BOOK badge/CTA on the backtest card.
DB `backtest_data/momentum_paper.db`. Registered in app.py after v2_ironfly. Backend restarted
16:19 IST 2026-06-30 (after close). Frontend built ON VPS (bundle index-CWcU0nQO.js).
- [x] **Universe = the OFFICIAL NSE Nifty 200** (niftyindices.com CSV, cached `backtest_data/
  nifty200_official.csv`, market-cap defined) — NOT the traded-value proxy (user corrected this;
  proxy was only a backtest-PIT necessity). Refreshes exactly the 200 (not 381). Fallback to
  traded-value if list unfetchable. ETFs excluded (SILVERBEES/GOLDBEES bug fixed).
- [x] Rules automated (APScheduler): daily 15:45 mark+Donchian-15 · weekly Fri 15:50 NIFTYBEES-100DMA
  gate · monthly last-trading-day 15:55 rebalance (top-8 / buffer-22). Idle cash earns 6.5% (liquid
  fund). Net ~0.3% RT; STCG 20% shown separately. Closed trades shown with exit reason.
- [x] SEEDED 2026-06-30: ₹20L, gate RISK-OFF (NIFTYBEES −0.44% vs 100DMA on fresh data) → in CASH;
  target basket computed (POWERINDIA/GVT&D/LAURUSLABS/ADANIPOWER/IDEA/ADANIENSOL/BHARATFORG/BHEL).
  Re-entry is MONTH-END (next rebalance end-July) once gate flips risk-on — matches research/62 winner.
- [ ] Monitor the soak; verify daily/weekly/monthly jobs fire. When gate flips risk-on at a month-end,
  confirm it deploys the 8 and that Donchian/gate exits log correctly.
- [x] **PERF FIX 2026-07-05 — page was stuck on "Loading paper book…" forever.** `get_state()` took
  13–35s: `_panel()` reads the ENTIRE daily table (1015d × 1642 syms) + pivots it, TWICE/request,
  uncached, on single-worker GIL-bound gunicorn. Fixed: memoize `_panel` by (start, DB-mtime)
  (`_PANEL_CACHE`), fetch panel once in get_state, + daemon pre-warm thread in `register()`. Warm now
  0.02–0.09s. Deployed via SIGHUP (weekend). Backups `momentum_paper.py.bak_panelcache`. UNCOMMITTED.
- [x] **GATE/BUY TIMING confirmed from code:** gate evaluated in `weekly_job` (last trading day of
  week ~15:15) — flipped **risk-ON Fri 07-03** (NIFTYBEES +1.01% vs 100DMA). But BUYS only happen in
  `monthly_job`=`rebalance_job` (cron 14:45, guarded `_is_last_trading_day()` of MONTH). So first
  8-stock basket buys on **last trading day of July 2026 (~Jul 31)** IF gate still risk-ON then
  (monthly_job re-checks). Month-end re-entry is by design (matches research/62 winner).
- [x] **LIVE-EXECUTION PATH BUILT + DEPLOYED (flag OFF) 2026-07-05.** Per user "build now, flip when I
  say · MARKET orders · capital set at flip". `services/momentum_paper.py` now has a real Kite **CNC
  MARKET** order layer gated by persisted `live_mode` (default OFF=PAPER, verified). All flow funnels
  through `_buy`/`_sell` → one switch arms the whole book. Adds: `_place_cnc_market` (place+poll fill,
  read `average_price`), integer-share qty, `_market_open_now` guard, per-order value cap, slippage
  alert, `reconcile_holdings` (book vs Kite, alert-only), partial-sell support. **LIVE monthly rebalance
  is ROTATE-ONLY** (`_rebalance_live_delta`): sell names leaving target, buy brand-new names cash-aware
  equal-weight, kept winners RIDE — NOT the paper liquidate-and-rebuild (that would churn+tax the whole
  book monthly). New endpoints `/api/momentum-paper/{toggle-mode,kill-switch,reconcile}`; `get_state`
  now returns `mode`/`live_mode`. **20/20 simulated-live tests PASS** (`/tmp/test_momentum_live.py`,
  fake order layer + temp DB, no real orders). Backup `momentum_paper.py.bak_live`. Runbook:
  `docs/MOMENTUM_LIVE_RUNBOOK.md`. UNCOMMITTED on VPS → next git sweep.
- [ ] **BEFORE FLIPPING LIVE (user decisions still open):** (a) set the **live capital** amount (user
  said "different amount" — not yet given; pass via toggle `{"capital": <rupees>}`). (b) Confirm the
  **rotate-only vs full-equal-weight** rebalance policy (v1 = rotate-only, no top-up/trim of kept
  names; `CFG['live_rebalance_trim']` reserved for future). (c) Frontend LIVE/PAPER badge + toggle
  control on `/app/momentum-paper` (build on VPS). (d) First live action would be the ~Jul-31
  rebalance — flip + fund the Zerodha account before then; run `reconcile` after the first fills.

## ⏸ QUEUED — re-test Phase 2 lower-cap with OFFICIAL market-cap indices (user flagged) — 2026-06-30
Phase 2 (research/62) mid/small/micro bands were by TRADED VALUE (liquidity rank), NOT market cap —
labels were loose. Capacity verdict is robust (liquidity-driven), but the midcap/smallcap PERFORMANCE
claims (e.g. smallcap fair-gated Cal 1.67) should be re-tested on the REAL indices. Lists already
cached on VPS: `backtest_data/niftymidcap150_official.csv` (150), `niftysmallcap250_official.csv` (250).
- [ ] Re-run the band study (62d/62e/62g) using official Nifty Midcap 150 / Smallcap 250 membership
  (current list as a modern-period proxy; PIT history still owed for full rigor). Expect capacity to
  still bind, but get honest labels + numbers. Then update STATUS-MD Phase 2 with the correction.

## ⏸ QUEUED (start ONLY after the V2-executor thread closes) — "Weekend-theta" iron fly variant — 2026-06-10
User-tried variant; user runs AlgoTest, Claude analyzes (separate system + separate assessment). **A couple
more versions of this coming.**
- **Structure:** same 2.0% wings + 2.0% underlying move-stop as V2, BUT **enter DTE-2 (Friday), exit DTE-1
  (Monday)** — capture the 2 weekend days' theta, close Monday. Short hold across the weekend.
- **Data scope (critical):** ONLY the weeks where **NIFTY weekly expiry was TUESDAY** (shifted from Thursday),
  so DTE-2 = Fri, DTE-1 = Mon, expiry = Tue. Need to identify/confirm that exact date window in the AlgoTest data.
- **Filter:** same CPR / inside-week skip MAY apply — but inside-week check uses the **CURRENT week of entry
  (the Friday's week)**, not the last completed week (note the causal subtlety: at Fri the current week's H/L are
  nearly fully formed — assess look-ahead carefully when we get there).
- [ ] Await user's AlgoTest exports (+ the other versions), then structure + assess as a standalone system.

## Straddle V1 — DTE-conditional move-stop (1-DTE → 0.5%, 0-DTE → 0.4%) — 2026-06-08
Page: `/app/straddles` · live logger `research/58_intraday_recenter_straddle/scripts/straddle_paper_live.py` (`V1_TRIG = 0.4`).
- **Why:** current V1 stop is a flat ±0.4% underlying-move stop for BOTH 0- and 1-DTE.
  0.4% IS backtested (research/52 stop_design: 0.4% beat 0.6/0.8/1.0% undl-move + all
  premium/maxloss stops, best net AND bounded worst-day; 1-DTE-only +₹15,988). BUT the
  grid jumped 0.4→0.6 (0.5% never tested) and was never split by DTE.
- **New evidence (user):** in another Claude session, **0.5% for 1-DTE was tested over 2+
  years on algotest.in** — user has all the details written down and will bring them.
- [x] **DONE 2026-06-08 (user-confirmed).** DTE-conditional stop wired in
  `straddle_paper_live.py` (`v1trig = 0.5 if dte(E) == 1 else 0.4`). Paper-only cron, no restart.
- [ ] Optionally re-run our own recorded-chain split sweep (0-DTE 0.4 fixed; 1-DTE {0.4,0.5,0.6}) to cross-check.

## Straddle live ticking — real-time SSE (NAS-style) — DONE 2026-06-08
- **Why:** `/app/straddles` legs only refreshed on the 5-min cron JSON → looked frozen.
- [x] Interim (no restart): cron bumped to 1-min + 1-min intraday grid + page poll 30s +
  per-leg trade-book table with **In/Out time columns** + collapsible **V1 & V2 rules** block.
- [x] **SSE DEPLOYED 2026-06-08 (after close).** `/api/straddles/stream` added to `app.py`:
  resolves V1/V2 leg tradingsymbols from `option_chain`, live `kite.ltp()` re-price every ~3s,
  payload `{type:tick, systems:{v1,v2:{ce_ltp,pe_ltp,ce_pnl,pe_pnl,pnl_now}}}`. `Straddles.tsx`
  opens one `EventSource`, overlays pnl_now + leg LTP/P&L on the cron base, shows a LIVE pulse.
  **Deployed without sudo** (passwordless sudo NOT configured): `SIGHUP` to the gunicorn master
  (runs as `arun`) graceful-reloads workers → re-imports `app.py`, zero downtime. Verified
  streaming live (v1 +39,360 / v2 −9,163). Bundle `index-C6k7-Uxf.js`.

## Straddle V2 — algotest optimization (research/60) — base LOCKED 2026-06-08
STATUS: `research/60_v2_straddle_optimization/V2_BIWEEKLY_STRADDLE_ALGOTEST_OPTIMIZATION_SWEEP_STATUS.md`.
User runs backtests on algotest.in; Claude structures + analyzes (net of taxes + ₹20/order + 0.25% slip).
- [x] **Wing width LOCKED = 2.0% of ATM (= ±500 today).** %-of-ATM sweep (2.0/2.5/3.0%) resolved the
  index-drift confound; 2.0% best (Calmar 0.70 ex-COVID), wider strictly worse. Width sweep CLOSED.
- [x] **VIX floor LOCKED = ≥13** (Claude pulled India VIX from Kite, daily-open proxy): 2023 flips
  green, +8.5L, Calmar 0.76; ≥14 = max risk-adj (Calmar 0.94). Script `scripts/vix_overlay_2pct.py`.
- [x] **SL SWEEP DONE 2026-06-08 → full base LOCKED = 2.0% wings + 2.0% underlying move-stop + VIX≥13.**
  Stop sweep @VIX≥13: Calmar PEAKS at 2.0% (0.76→**1.03**→0.62 across 1.5/2.0/2.5%); +₹8.80L, DD −₹1.17L,
  7/8 green. Conservative alt VIX≥14 = 8/8 green (+₹8.16L). Wings are the real risk control (stop = sweet-spot,
  not plateau → "~2% wide stop"). Replaces old 1.5% spec. **PUBLISHED:** /app/backtest/v2-nifty-ironfly-sl-vix
  (+ factsheet PNG; standalone HTML at laptop `research_v2_locked_factsheet.html`).
- [ ] **★ CRITICAL — Phase 2 profit-target sweep** on the 2%+2%-stop+VIX≥13 base. PT ∈ {25%, 55%, 70%, none}
  (40% already in hand). Fire 4 algotest runs; Claude computes year-wise/Calmar + VIX overlay. THEN entry-time sweep.
  (User flagged 2026-06-08: this is the next must-do; do not skip.)
- [x] **Conditional-attribution study DONE 2026-06-08 → CPR-COMPRESSION OVERLAY found + WALK-FORWARD VALIDATED.**
  Losses concentrate in volatility compression, flagged by NARROW PRIOR-DAY DAILY CPR. **Skip entries when
  CPR width < ~0.10% of spot** (|TC−BC|/spot from prior-day H/L/C). On VIX≥13 book: +CPR≥0.10% → 147t,
  +₹11.0L, **Calmar 0.95→1.59, 7/8 green**; +CPR & skip Jan/Aug/Sep → 116t, +₹11.85L, Calmar 1.71, **8/8 green**.
  Filter RAISES return AND CUTS drawdown. **Walk-forward:** train-half threshold (≈0.12%) applied blind to
  test half lifts Calmar 1.13→2.81 (2023-26) and 1.11→2.08 (2019-22); skipped bucket negative in BOTH halves.
  Directional skew NOT supported (it's a regime skip, not a tilt). Mechanism: compression → expansion → short
  gamma run over. Detail in STATUS doc + memory.
- [ ] **CPR overlay — forward-validate before adopting (candidate, NOT yet in locked base/app study).**
  (1) paper-forward on the live book; (2) check AlgoTest native CPR filter, else compute CPR from NIFTY daily
  in the live V2 engine and skip narrow days; (3) test a WEEKLY-CPR variant; (4) once confirmed, fold into the
  locked base + update /app/backtest/v2-nifty-ironfly-sl-vix.
- [ ] Re-spec wing as % live if NIFTY moves materially (rebuild as ±500 pts at today's level).
- [ ] **MARGIN CORRECTION (page shows wrong RoM).** Verified Zerodha SPAN via Kite margin API (2026-06-08):
  ±500 iron fly = **₹8,24,580 / 10 lots (₹82,458/lot)**; naked straddle ₹21.0L/10 lots. Earlier ₹95,802/lot
  was ~16% high. Corrected RoM on ₹8.25L: **14.6%/yr simple / ~10.5% CAGR / ~9.7%/yr on 1.5× buffered capital**.
  Update /app/backtest study metrics+caveat once user picks the basis to display. NB: current-level snapshot —
  2019 margin was ~half (lower notional); RoM is simple, not compounding (fixed lots).
- [~] **MONTHLY positional fly — SHELVED 2026-06-08: NOT FEASIBLE on AlgoTest (platform-blocked).** AlgoTest's
  positional entry is weekly-cadence-oriented (entry capped ~4 TD-before-expiry); a true monthly book needs
  entry ~18-20 TD before monthly expiry + ~1-month hold, which it can't express. Forcing expiry=Monthly gave
  only **6 sporadic Friday fills over 6 years** (whole years missing) — an artifact, not a backtest; re-run
  reproduced it (structural, not a stray filter). REVISIT only if AlgoTest adds a calendar/weekday entry, OR
  if we acquire a historical MONTHLY option-chain data source (local recorder has only ~2 months since
  Apr-2026, not 2019+) and self-backtest. Not worth pursuing now. Weekly remains the tradeable cadence.

## Straddle live V2 — wire card to the research/57 engine — 2026-06-08
- **Why:** the live V2 card currently tracks only the **core short straddle** (CE+PE); the backtested
  V2 system is a full **iron fly** (±500 wings) with 1.5% stop / +40% PT / re-enter / roll / VIX≥13.
- [ ] Wire the live card (`straddle_paper_live.py` + `Straddles.tsx`) to run the research/57 engine
  (`research/57_positional_straddle_biweekly/scripts/biweekly_paper.py`) so V2 shows the **wing legs**,
  the locked rules, and **each entry's entry/exit time + short exit reason** (stop / PT / roll). The
  RulesBlock footnote already flags this gap. Frontend + cron-script change (no backend restart needed
  unless a new API route is added).

## Research 56 — NIFTY 30-min Double-Supertrend options book — SIGNAL (in-sample), 2026-06-04
Folder: `research/56_nifty_dual_supertrend/` (STATUS + RESULTS + scripts).
- [x] As-specced always-on credit book = **NO EDGE** (−₹17k–62k/6wk, gross neg too):
  trailing stop flips at turning points → late entries into neg-skew spreads.
- [x] User refinements **layering (stack/convert) + bi-weekly expiry (2nd-nearest Tue,
  skip front weekly)** → near break-even (best V3S −₹8.5k, gross −₹4.6k).
- [x] **ENTRY-TIMING FIX = the unlock.** Enter on first pullback-and-resume inside the
  MST regime (not on the flip). Clean MONOTONIC dose-response. Pure-pullback (V4) =
  **first NET-POSITIVE: +₹4,529/6wk/1lot, gross +₹5,306, worst −₹3,319, 12 trades.**
  → **SIGNAL, not yet a strategy** (n=12, 6wk, one regime; edge is selectivity, not
  always-on). Best engine = `scripts/g2c_layered_engine.py` (V4, bi-weekly, stack).
- [x] Spike protection (defined-risk wing) WORKS — worst bounded.
- [x] **PAPER forward-logger LIVE on VPS** (2026-06-04) — standalone cron
  (`scripts/nifty_dst_paper.py`, no gunicorn restart), paper-only 1 lot, logs to
  `results/paper_dst.db`. Recovery doc: `NIFTY_DST_PAPER_FORWARD_RUN_STATUS.md`.
  Robustness (G2f): survives 2× costs, monotonic in OTM/wing/period, but FLIPS
  NEGATIVE at MST mult 6 (1 yellow flag). Capital: 1 lot needs ~₹90k peak margin
  (~5.2%/6wk in-sample); scales linearly (10 lots ≈ +₹46.5k on ~₹9L, worst −₹33k).
- [ ] **Validate SIGNAL→STRATEGY:** let paper logger accumulate ≥50–100 forward
  trades across ≥2 regimes; compare realized vs backtest; THEN consider sizing up.
  Do NOT size to 10 lots on the 12-trade in-sample number.
- [ ] Alt EV+ use: same regime as flat/hedge OVERLAY on live RS-momentum/MQ books.

## Research 55 — MTF Compression Breakout (smallcap runner pattern) — CONCLUDED 2026-06-04
Folder: `research/55_mtf_compression_breakout/` (STATUS + RESULTS + g1-g4 scripts).
- [x] **VERDICT: NO ALPHA (beta).** User idea: daily uptrend + 30m above weekly CPR +
  5m prev-day-coil/narrow-CPR/PDR-break + volume (refs TDPOWERSYS/DATAPATTNS/KMEW).
  Tested 4 ways — largecap-5m (n1424), smallcap-5m 2024-26 (n631), DAILY full-universe
  1099 names 2018-26 (n7501). On every trailing exit the breakout entry LOSES to a plain
  "hold the uptrend" baseline (daily Supertrend: SIGNAL +0.33R vs BASE +0.93R). **Volume
  spike consistently HURTS** (refuted all 4 runs). Only crumb: +0.04R on tight R-targets.
  Examples = survivorship (user's own caution). Killed before any big sweep.
- [x] **One real insight:** compression filter beat baseline ONLY in 2022 (bear) → it has
  *defensive* value. Revisit ONLY as a risk-off/regime filter on the MQ momentum book,
  never as an entry trigger. The baseline ("own uptrending names, trail Supertrend") IS
  the edge — that's the MQ book (32-48% CAGR); improve it, don't overlay breakouts.

## NWV Phase 1 — Trade execution & management
Design doc: `docs/NWV-PHASE1-TRADE-PLAN.md`. Builds on the live Phase-0 view
engine on the Quantifyd host (`94.136.185.54:/home/arun/quantifyd`).

### Blocked on user sign-off (decisions, see doc §9)
- [ ] Confirm **next-week expiry** (changes the locked Phase-0 current-week rule).
- [ ] Confirm **"CPR R1" = weekly R1 pivot** (`nwv_weekly_state.pivot_r1`).
- [ ] Confirm **IC-morph definition** (add upside short-call spread R1/R1+200 to the put debit spread).
- [ ] Confirm **conviction gating** (default: trade 5 lots on any directional view).

### Investigations — DONE v1 (see doc §10; low confidence, n=21, one regime)
- [x] **A. Adjustment point — BIGGEST WIN.** Morph = **add a BULL PUT spread** (not a call spread) → all-put condor/butterfly. Bearish book −₹2.4k/wk (PF 0.65) → **+₹2.1k/wk (PF 2.41)**, tail −19k→−6/−10k. Best placement: condor band near existing short strike (butterfly = tightest tail). Recenter-at-price is worse. My first call-spread version was wrong (it backfired).
- [x] **C. Stop timeframe** — 15m ≈ 30m; **use 30-min close beyond R1/S1**. ~3x baseline expectancy, tail −19k→−14k. Role = backstop when no morph trigger.
- [x] **B. Friday exit** — leans earlier (09:45 > 15:15) but model-based; robust call = exit Friday. Profit-take 75% ≈ neutral.
- [x] **EXTEND to 2020** (73 wks, 6 regimes; modeled BS, 22pt error) — see doc §12. **Morph REVERSES: net negative across regimes** (caps 4 big winners −137k vs saves 18 losers +95k). 2024-25 morph win was a pure-uptrend artifact. **Stop is the only robust edge** (+₹125/wk, helps every year). Bullish mirror also net-negative.

### Revised core (regime-tested)
Bear/bull debit spread → **30-min R1/S1 stop (PRIMARY management)** → **Friday exit**.
**Morph DEMOTED to experimental** — only worth revisiting as a **loss-gated** trigger (morph only a trade already underwater, so it can never cap a winner). Conviction gating: none yet.

### Bearish-signal diagnosis — DONE (doc §14)
- [x] **Bearish view is directionally INVERTED** — when it fires NIFTY rises +0.6% avg, falls by Fri only 37% (vs 44% base). Weak Monday open mean-reverts up. So a bear *debit* spread is the worst vehicle (wrong way + long theta).
- [x] **Skewed-IC test** — on BEAR weeks every IC beats the debit spread; **bull-skew IC** −₹2.4k→+₹2.7k/wk (PF 1.71 real, only positive structure modeled). Neutral IC nearly as good + more intuitive.

### REVISED directional structure (new core)
- **Bearish view → SLIGHTLY-BEARISH IRON CONDOR (LOCKED 2026-06-01)** — centre offset −50: short call ≈ spot+200, short put ≈ spot−300, 200 wings, 50%-credit TP, −1× stop, Friday time-stop. NOT a bear debit spread. (−50 tilt ≈ neutral in execution due to 100-pt strike rounding → mild bearish lean at ≈zero cost; +₹2,372/wk PF 1.43 real. Don't skew past −75: expectancy drops, modeled goes clearly negative.)
- **Bullish view → bull debit spread** (drift-aligned, capped risk) or bull-skew IC.
- Mind IC gap/crash tail (worst wk −19k..−32k modeled); 4-leg fills erode edge.

### Open / next
- [ ] **Engine question:** the bearish matrix branch precedes UP-moves — fix/invert/filter it in Phase-0, or formally redefine "bearish view" as "elevated-chop" → IC. (Bigger than Phase-1.)
- [ ] Intraday PT test for any debit legs (EOD granularity missed the intra-week excursions).
- [ ] Validate IC edge with real fills/slippage modelled (4 legs × 5 lots).
- [ ] (optional) loss-gated morph v2.

### Build (after design locked)
- [ ] `services/nwv_trade.py` — spread construction from view + pivots (5 lots, 200-wide, ~40% debit).
- [ ] 15-min R1/S1 structural-stop monitor (reuse ticker infra).
- [ ] 30-min stochastic monitor + IC-morph executor (reuse Tier-2c IC wing code).
- [ ] Friday exit scheduler.
- [ ] Paper-trade one full week before going live NRML.

## NAS live options (8 variants on 94.136.185.54)

### Resolved 2026-06-01 (live)
- [x] **Bug #1 — OTM cross-variant roll routing.** The OTM tick-adjustment shared
  one token pool (Squeeze-OTM + 9:16-OTM) but always fired through the *squeeze*
  executor/DB → 9:16-OTM rolls failed `position not found` and never executed
  (silently, all morning). **Fixed** (`nas_ticker.py`, commit `3adc074`, pushed):
  route each roll to the owning variant's executor/DB, re-subscribe full pool,
  skip cross-leg roll when >1 strangle in pool (guard). Deployed + verified live.
- [x] Synced the user's manual 10:08 OTM roll into the 916-OTM DB (PE 23350 →
  PE 23250 @ 14.35). App display now matches broker.
- [x] **Re-synced today's recorded entry/exit prices to actual broker fills**
  (entries per-leg by order-id, exits by symbol buy-back avg). Realized
  −5,057 → **−5,317 = broker exact**. 4 DBs backed up (`.pxbak_*`). CAVEAT: open
  legs that close later today will again record the SL-trigger price (not fill)
  until the code fix below ships — do a final EOD re-sync for the day's report.

### NAS-OPT new paper variant (research/54 system) — 2026-06-03
- [x] **Backtest performance report** — `research/54.../results/nasopt_perf.png` (P&L curve+drawdown+KPIs),
  `nasopt_trades.csv`, `RESULTS_nasopt_report.md`. 29d: 13 trades, +₹20,409, 69% win, maxDD −2,695.
- [x] **Paper module** `services/nas_opt.py` — built + live-validated (reads options recorder, trades
  0/1-DTE only, ±0.4% move-stop, paper-only); `register()` adds 3 API routes + entry/monitor/exit jobs.
  `nas_opt_trading.db` backfilled with the 13 backtest trades. py_compile clean.
- [x] **Wiring DEPLOYED LIVE 2026-06-03 (commit 188b145)** — user cleared mid-market deploy (no trades
  today, all flat). NAS-OPT registered: /api/nas-opt/state|trades|equity live, entry(09:20)/monitor(1min)/
  exit(14:45) paper jobs scheduled. First paper entry expected next Mon/Tue (0/1-DTE) at 09:20.
- [x] **Dashboard card DEPLOYED LIVE 2026-06-03 (commit 4061e54)** — NAS-OPT card added to
  `frontend/src/pages/Nas.tsx` (total P&L, trades, win rate, SVG equity curve, today status). Built on
  laptop (node v24, pulled frontend source), pushed bundle `index-dmozehmb.js` → `static/app/`; source +
  bundle committed to git (durable, survives future rebuilds). Confirmed in served bundle. Hard-refresh
  /app/nas to see it. (Laptop `frontend/` is now a build checkout — re-pull fresh before next edit.)
- **NAS-OPT IS COMPLETE + RUNNING IN PAPER. No action needed — let it accrue paper P&L; watch /app/nas.**
- [ ] **PARKED (user will trigger) — flip NAS-OPT to LIVE.** NOT a toggle: `services/nas_opt.py` is
  paper-only by design (no Kite-order code; marks P&L from the recorder). Live-flip = a small build —
  add the real-order execution path (place Kite orders on entry + on each exit), behind a paper/live
  flag (mirror nas_atm_executor's `paper_trading_mode` + live branch), with fill read-back + a kill
  switch. Only build when the user says NAS-OPT paper is working well and asks to go live.

### Operating schedule — LOCKED 2026-06-03 (user directive)
- [ ] **Live only Mon/Tue/Fri; PAPER every other day; mode-tagged — DEPLOY after
  close 2026-06-03.** User: trade LIVE only Fri/Mon/Tue; on all other days run the
  same signals as PAPER (DB + P&L + EOD report, no real Kite orders) so we never
  stop collecting data; every trade/P&L/order tag must say paper vs live. **Built +
  dry-run-validated** (Mon/Tue/Fri→LIVE, Wed/Thu→PAPER): adds `live_weekdays=(0,1,4)`
  + `max_dte_at_entry=None` to NAS_DEFAULTS & NAS_ATM_DEFAULTS, empties `skip_weekdays`,
  and makes `_place_order`/exit in both executors day-aware (`_is_paper`). Patcher
  staged on VPS `_nas_paperdays_patch.py` (live files untouched); after-close deploy
  scheduled. **DTE gate (max_dte=1, commit bec1ac4) is OFF operationally** — now only a
  backtest-study question (see research item below). Mode column already in DB; deploy
  step verifies/adds the tag in EOD report + Nas.tsx trade table.
- [x] **NAS system-improvement BACKTEST — research/54 DONE 2026-06-03 (verdict CONCLUDED).**
  `research/54_nas_tune_newsys/` (real recorded NIFTY chain, 29d, net-of-cost). 3 new angles
  tested: **IV-level filter = NO EDGE** (DTE proxy: all-day corr +0.41 but within-1DTE −0.14);
  **defined-risk iron-flies = NO EDGE** (cost premium, cut edge to ~0, far wings don't cap the
  −20k intraday tail); **weekday×DTE map** confirms Mon(1DTE) +2,284/day, Tue(0) +395, Fri(4)
  −70 flat, Wed(6)/Thu(5) bleed → **Mon/Tue/Fri-live is data-consistent** (excludes the 2
  bleeders). Winner: naked straddle + ±0.4% move stop (+1,412/day 0-1DTE, worst −3,260). See
  `research/54.../results/RESULTS.md`. **6 new angles tested total** (stages 1-6): IV filter ❌,
  iron-flies ❌, late entry ❌, intraday re-entry ❌ (HURTS — re-sells into the trend), directional
  skew ❌ (neutral), multi-feature calm-classifier ❌ (no better than opening-range alone; prior-day
  feats useless) — **1 keeper: ~100pt-OTM strangle + move-stop beats ATM straddle (monotonic, net+tail)**.
  FINAL refined system: 1-DTE · ~100pt-OTM strangle · 09:20 entry · ±0.4% move-stop · ONE-AND-DONE ·
  tight-opening-range days · exit 14:45 · cross-family. Edge = day-selection + stop + modest-OTM, NOT
  structures/filters/re-entry/skew/classifiers. Sole implementation lever = the move-stop upgrade below.
- [ ] **TOP UPGRADE — replace per-leg 1.3× premium stop with ±0.4% underlying-move stop (HIGH).**
  **Status 2026-06-03: DESIGN LOCKED + kept safely here; user said BUILD-but-DEPLOY-LATER, so it is
  NOT yet coded into the live ticker (money-path — deserves its own focused build+test session).**
  Why: single actionable finding from research/54 + research/52. Premium stops whipsaw (scan:
  1.3× = −₹13,983 vs move-stop positive on same chain); the move-stop triggers on REAL adverse
  moves → no whipsaw AND bounded tail (2yr stress −7.9k vs no-stop −58.8k).
  **WHERE THE CURRENT STOP FIRES (investigated):** NOT in `_place_order` — it fires in
  `services/nas_ticker.py` on each tick via `if ltp >= sl_price` in the per-family SL handlers
  (`_check_atm_sl`/`_check_atm2_sl`/`_check_atm4_sl` ≈ lines 786-790 / 1021-1025 / 1141-1145) and
  the OTM cross-leg path. `sl_price = entry_premium × 1.30` is set in `_place_order`/DB.
  **DESIGN (move-stop):**
    1. Capture `entry_spot` (live NIFTY underlying at fill time) per strangle at entry — add to the
       in-memory leg slot (`_atm_*_legs`) AND persist (new `entry_spot` col on nas_positions /
       nas_atm_positions, nullable) so it survives a restart/reconcile.
    2. In the ticker's tick/candle handler (it already holds the live NIFTY spot), add a per-strangle
       check: `if abs(spot - entry_spot)/entry_spot >= 0.004: exit FULL strangle (both legs)` via the
       owning variant's executor — same exit path the SL handler already calls.
    3. Stop policy decision (pick at build): (a) REPLACE the 1.3× premium SL with the move-stop, or
       (b) move-stop PRIMARY + keep a WIDE premium SL (e.g. 2.5×) as a backstop. Research favours the
       move-stop; a wide backstop is cheap insurance. Config: add `move_stop_pct: 0.004` to
       NAS_DEFAULTS + NAS_ATM_DEFAULTS; gate behind a flag (`use_move_stop`) for safe rollout.
    4. Exit = full strangle (research used full-strangle exit on the move trigger), NOT naked-survivor.
    5. STRIKES (research/54 Stage 4, signal): pair the move-stop with **~100pt-OTM strikes (1-2 strikes
       OTM each side), 09:20 entry** — beats ATM straddle monotonically on net (+1,412→+1,570/day) AND
       tail (−3,260→−2,695). Modest-OTM = less gamma into the move; the move-stop still caps the tail.
  **VALIDATION already done:** the move-stop *strategy* is proven on the real chain (research/54
  stage1/3: 0/1-DTE +1,412/day, worst −3,260) and 2yr stress (research/52). The BUILD step still
  needs: offline replay of the executor path + a paper-soak before going wide.
  **ROLLOUT:** build → py_compile + logic unit-test → stage patcher (do NOT apply) → deploy AFTER
  CLOSE behind `use_move_stop`, PAPER first (pairs with paper-all-days) → watch a few sessions →
  flip live. Sequence AFTER tonight's paper-days deploy (same ticker/executor files — rebase on that).
- [ ] **App↔broker DESYNC prevention (user request — HIGH).** The reconciler
  (`_nas_run_reconciler`, app.py:145) only reconciles ENTRY orders
  (PENDING→ACTIVE/FAILED + partial-entry orphan close). It does NOT compare
  ACTIVE DB legs vs broker NET positions, so a manually/externally-closed
  ACTIVE leg stays "active" in the app (2026-06-01: squeeze-ATM2 PE 23550 closed
  at broker @147.15 but app showed it active; reconciler logged orphans=0).
  Fix: add a position-level broker recon to the 3-min job — per symbol, sum
  DB-active qty across variants vs broker net short; DB>broker → ALERT (+ auto-
  close where one variant owns the symbol); broker-only short → ALERT (untracked
  live leg). CAVEAT: shared-strike legs net at broker → attribution ambiguous
  (same root as single-slot bug) → safe v1 = read-only ALERT, auto-correct only
  when unambiguous. Deploy + test after close (auto-close on live broker state
  is sensitive). Stopgap until then: manual reconciliation on each user trade.
- [ ] **Single naked/monitor slot per family → multi-naked legs unmanaged + ATM2
  monitor bumped (HIGH).** Ticker has ONE `_atm_naked_leg`/`_atm4_naked_leg` +
  one `atm/atm2/atm4_option_legs` slot per family, but squeeze+916 both active
  create 2+ naked legs / 2 straddles → only one is monitored; the others get
  `sl=999999` with no working ST and no tick-SL (2026-06-01: 4 naked legs, only
  2 in slots, both `st_value=None`; squeeze-ATM2 PE breached SL unmonitored).
  ST also needs 8 candles (40min) and the shared buffer resets each time another
  leg goes naked → never computes. Fix: per-position naked-ST monitors + per-
  variant option-leg slots. After close.
- [ ] **Full per-variant OTM split — ELEVATED (now leaves legs unmanaged live).**
  The bug-#1 guard *pauses* cross-leg rolls whenever Squeeze-OTM AND 9:16-OTM are
  both active. 2026-06-01 the 11:00 squeeze made both active → squeeze-OTM PE
  23350 ran to 39.2 (2.6× the CE's 15.1, well past the 2.0 trigger) with NO
  auto-roll; user had to roll it manually (per-leg 2× SL still protected). Fix:
  in `nas_ticker._check_premium_tick`, group pooled legs by strangle_id and run
  the cross-leg compare + roll INDEPENDENTLY per 2-leg strangle, with
  per-strangle state (`_adj_triggered`/`_adj_next_direction`/`_adj_confirm`
  keyed by sid). Replaces the blunt `len!=2` guard. Live auto-order change →
  deploy + test after close.
- [x] **ATM-V4 roll parity — DONE (deployed 2026-06-02, commit `cf54fb8`).**
  User chose true premium parity. `_find_roll_strike` rewritten: scans OTM
  strikes from a 50-pt floor (`roll_min_otm=50`) OUTWARD and picks the strike
  whose premium is *closest to the surviving leg* (no more ≥100-OTM outward-only
  undershoot). Validated by `tests/test_nas_per_strangle_roll`-sibling
  `tests/test_v4_roll_strike.py` (replays real 09:19 2026-06-02 prices: NEW
  picks CE 23350 @36.7 vs OLD CE 23400 @23.6 for target 42.2; PE side also
  matches; 50-pt floor respected) — ALL PASS. Restart clean, ticker reconnected.
- [ ] **SECURITY — rotate VPS GitHub PAT.** The VPS git remote URL embeds the
  PAT in cleartext (`https://ghp_…@github.com/...`) — recurrence of the
  2026-05-19 leak. Rotate the token, set remote to tokenless HTTPS + credential
  helper. Why: a working-dir read or backup tarball exposes write access.
- [ ] **Record ACTUAL fills, not signal/trigger prices (durable P&L fix).** Root
  cause of the app↔broker P&L gap: executors write entry = quoted premium at
  decision and exit = SL-trigger LTP, NOT the broker fill avg. Fix: after each
  order COMPLETEs, read back `average_price` (order_id → `orders()`) and store
  THAT as entry/exit across all executors. **+ SLIPPAGE GUARD (user request):**
  if |fill − expected| exceeds a threshold (e.g. >5% or >N pts), log a
  `SLIPPAGE ALERT` for investigation (fast-fill/illiquid leg). After close;
  touches every executor's order path — too risky live.
- [ ] **Trade Book — subtle SL column (user request).** Add an `SL` column after
  `ENTRY→EXIT` showing the fixed level (1.30× entry, muted) or **`ST`** for
  naked SuperTrend-managed survivors (`sl_price=999999`). Needs a FRONTEND
  REBUILD — VPS has the source (`frontend/src/pages/Nas.tsx`) but NO node/npm
  toolchain; build off-box and deploy the bundle after close (mid-session bundle
  swap risks breaking the live monitoring view). Grid is at Nas.tsx ~L826/L850.
- [x] **ATM strike snaps to the FORWARD, not spot — DEPLOYED 06-01 (commit
  `57eb8c2`, restarted/verified live).** `nas_atm_executor.execute_strangle_entry`
  now derives the live synthetic forward = `strike + (CE − PE)` at the
  spot-nearest strike and re-snaps ATM to it (spot fallback on any quote
  failure, so never worse than before). Fixes the call-rich imbalance from
  spot-rounding when futures trade over spot. Live-tested: spot-ATM 23600 gap
  42.5 → fwd-ATM 23650 gap 7.8. Applies to all 3 ATM variants (shared method).
  The 3 imbalanced 23550 straddles from 11:00 left running (SL-protected, user
  agreed). FOLLOW-UP (lower priority): also fix `nas_scanner.py:593` stale
  candle-close spot used by non-ATM scan paths.
- [ ] **ATM2 same-strike re-entry churn — FIX = skip re-entry when ATM unchanged
  (user decision; deploy AFTER CLOSE).** On SL-BOTH, 916-ATM2 closes both legs
  and re-enters a fresh ATM straddle even when the market whipsawed back to the
  SAME strike (2026-06-01: closed 23600 @11:32:55 → re-sold 23600 @11:32:58 =
  pure churn, not re-centering). Cycled 3× (10:03/11:09/11:32) net +₹544 today
  (chop), but trends would churn losses+slippage. FIX (`nas_atm2_executor.py`
  re-entry path ~L165): on SL-BOTH, FIRST compute the new forward-ATM strike;
  if it == the strike being tested, **do NOT close at all — hold the straddle
  and reset the per-leg SLs in place** (recompute 1.3× off current premiums, no
  orders). Only close+re-enter when the ATM has genuinely moved to a new strike.
  (User refinement 06-01: closing+reopening the same strike is pure churn, not
  re-centering — avoid the round-trip entirely.) Applies to both ATM2 variants.
  Needs design care (SL-reset semantics). Deploy + test after close.
- [ ] **Ticker keeps STALE leg SL after ATM2 cascade re-entry (log noise).**
  After a cascade re-enters the same symbol, the ticker still compares ltp to
  the *old* straddle's SL → repeated false `SL TICK ... >= <old SL>` +
  `no actions taken`. Harmless (executor enforces the real SL via 10s poll), but
  re-subscribe ATM2 legs after re-entry to refresh cached SLs. After close.
- [ ] **null `pnl_inr` on closed legs.** Closed positions return `pnl_inr=null`
  from the API/DB (UI computes P&L itself), so server-side realized-P&L tally
  reads 0. Persist realized P&L on close. Cosmetic for trading; fixes monitoring.
- [ ] **Watchdog tz bug.** `[NAS-WD] can't compare offset-naive and offset-aware
  datetimes` → mis-reports `outside_market`/stale candle. Cosmetic (ticker is
  fine); normalize tz in the watchdog candle lookup.
- [ ] **Reconcile local repo with origin.** Origin is at `3adc074`; local is
  behind (`8129661`) with an uncommitted parallel MQ/research workstream. Pull
  after close (no nas_ticker.py conflict). Also bake the standalone-app
  manifest/favicon (runtime-patched on VPS `static/app/`) into source.
- [ ] Investigate 08:55 Monday cron `auto_login.sh` failure (http=000; token
  refreshed manually at 09:04). Check before next session's pre-open.

## Research log
- [x] **research/73 — Weekly SuperTrend (10,3) trend-following — CONCLUDED 2026-07-07: NO INVESTABLE TIMING EDGE (headline was a benchmark artifact).**
  YouTube system (Vijay Khant): buy weekly ST(10,3) green / exit blind on red / size 5-7% / book 40/40/20 / +5 hacks.
  Tested core on Nifty50/200/Midcap150/Smallcap250/Nifty500, net 0.30% RT + STCG/LTCG, 2010-26 (VPS folder
  `research/73_weekly_supertrend_investing`; engine `st_weekly_engine.py`, g1/g3/g4 + `fair_bench.py`/`all_bands_fair.py`,
  RESULTS.md). **FIRST PASS looked great (Nifty200 17.5% CAGR / −31.7% DD / Calmar 0.55, "+6.9pp over NIFTYBEES") but
  the CORRECTION (same day) killed it: that was a BENCHMARK ARTIFACT — a survivorship-selected TODAY's-Nifty200 book
  vs the Nifty 50 INDEX. Fair test vs equal-weight buy-&-hold of the SAME names: the ST timing LOSES on EVERY band —
  Nifty50 −6.6 / Nifty200 −3.5 / Midcap150 −6.4 / Smallcap250 −2.8 / Nifty500 −4.4 pp/yr, at equal-or-worse Calmar
  (basket wins except Smallcap, where ST only helps by cutting the basket's −54% DD).** The basket beats Nifty50 by
  +8..+11pp on every band = the whole headline (survivorship + Nifty200-breadth). Per-trade ENTRY edge is real
  (G1 +5.2pp vs random-hold) but swamped at book level by time out-of-market in a bull → SIGNAL≠STRATEGY (same
  lesson as research/49 "beta not alpha"). Also proven: the guest's own 40/40/20 booking (17.5→8.8%) and a regime
  gate (17.5→11%) both HURT. Merit = none as timing; at best a mild de-risk overlay on a basket you'd hold anyway
  (poor trade: −4pp DD for −3pp CAGR). PUBLISHED + CORRECTED `/app/backtest/weekly-supertrend-nifty200` (added
  the deciding fair-benchmark table, all-index table, year-by-year). Honest way to the ~20% = own the basket (with
  its survivorship caveat) or improve the existing regime-gated momentum book (Cal ~1.7). (All files on VPS.)
  **PHASE 2 (2026-07-08) — the redemption:** ST DOES work as a MARKET-LEVEL CRASH OVERLAY (not per-name). Hold
  the basket always; a DAILY ST(7,3) on the index flattens the whole book in downtrends → **pre-tax Calmar
  0.56→1.28** (Nifty200 DD −39%→−15%) for ~2pp CAGR; consistent all bands, robust across fast family (dST 7/10/20
  + 50DMA); **200-DMA HURTS (0.45)**. Tax is the real cost (liquidating the cash book ~2.5 sw/yr → net Calmar
  1.01) → **build as a NIFTY-futures/puts hedge (no sale = no tax event)**. `crash_overlay.py` on VPS; app study
  + RESULTS Phase-2 updated. **TWO NEXT-LEVERS opened:** (1) implement the overlay as a Nifty-futures hedge +
  re-measure net (incl. roll/basis/tracking); (2) swap the LIVE momentum book's (research/62) MA gate for a
  daily-ST(7,3) gate and re-test — dST beat the 200-DMA here.
  **GATE CROSS-CHECK DONE (2026-07-08) — REJECTED.** `research/62.../scripts/62i_st_gate.py`: on the LIVE
  momentum book (rsblend N8 buf22 donch15, net STCG20%, 2014–26) swapping the 100-DMA gate for a daily-ST gate
  is WORSE — net Calmar 100-DMA **1.71** vs dST(7,3) 1.33 / dST(10,3) 1.25 / 50-DMA 0.99. ST gates twitchier
  (30–36 de-risk events vs 23), give up ~6pp CAGR for no DD benefit. **KEEP the live 100-DMA gate.** (Engine got
  a backward-compat `gate_roff` param, `.bak_stgate` kept; services/momentum_paper.py untouched.)
  **PHASE 3 DONE (2026-07-08) — the cleanest tradeable finding.** `etf_st.py`/`etf_st2.py`: trend-time the
  actual INDEX ETF itself (NIFTYBEES; index-level, no survivorship, infinite capacity). Net-of-tax ~1.5pp CAGR
  give-up (10.6→9.0%) but **DD MORE THAN HALVED (−36→−14%)**, Calmar 0.29→0.53, Sharpe 0.75→1.11 (~2×). Pre-tax
  give-up ~zero. ST(7,3) marginally best (fewest switches→least tax) but 50/100-DMA tied — any fast-medium
  filter; **200-DMA HURTS** (halves CAGR). Robust NIFTYBEES/JUNIORBEES/BANKBEES; GOLDBEES no. Well-known
  Faber-style timing, clean+scalable not novel. Published as ★★ Phase 3 on the app study. **NEW next-levers:**
  (i) futures/puts-hedge implementation to kill the ~1.5pp tax drag; (ii) multi-ETF trend-timed sleeve (equity
  + gold, though gold didn't respond to ST). NB: the STOCK-LEVEL per-name ST (Phase 1) loses; only INDEX-LEVEL
  works — always label which.
  **PHASE 3b + WINNER REFRAME (2026-07-08).** User pushed for realism on the ETF winner: idle cash in a LIQUID
  fund earns NET of its expense+slab tax (~6.5%→~4.5% net) + T+1 settlement lag (`settlement_liquid.py`). REALISTIC
  NIFTYBEES·ST(7,3): **9.3% CAGR / DD −14.3% / Calmar 0.65; net-of-ALL-tax 7.8% / Calmar 0.46** — DD-halving is
  friction-PROOF, but give-up grows to ~1.3pp pre-tax / **~2.8pp net-tax** (earlier ~1.5pp was too kind). Roughly
  **Sharpe-NEUTRAL (0.33 vs 0.34) — a drawdown-reduction overlay, not a return-enhancer.** Liquid fund essential
  (worth ~1.8pp). Study + HTML report REFRAMED to LEAD with the winner (own dark factsheet `niftybees-st73-winner.png`,
  realistic numbers) not the Phase-1 illusion; added settlement table + Phase2-vs-3 note (P2 Calmar 1.28 > P3 0.65
  only because P2 times the survivorship-inflated basket — mirage). Clean HTML report live
  `/app/weekly-supertrend-report.html`, linked from the study card.
  **PHASE 3c — MODELED futures-hedge DONE (2026-07-08)** `futures_hedge.py`. Keep the ETF (never sold → no
  equity CGT, deferred like B&H; no T+1 lag; margin by pledging the ETF) + SHORT NIFTY futures on the red signal;
  hedged ≈ synthetic T-bill (carry ≈ risk-free). **RECOVERS the whole give-up: ~B&H return 10.6% CAGR at HALF the
  drawdown (−14.4%, Calmar 0.74 vs 0.29, Sharpe 1.10)** — the near-free-lunch, via the tax structure. Published
  as ★★ section 05 on the study + HTML report. **⚠ MODELED, not backtested — DB has NO NIFTY futures series, so
  the ~4.6% carry is an ASSUMPTION** (sensitivity 4.0/4.6/5.2% → Cal 0.71/0.74/0.76). Hidden risk: crash-time
  BACKWARDATION (short carry goes negative exactly when hedged).
  **PHASE 3d — REAL-DATA VALIDATED (2026-07-08)** `kite_futures_probe.py` + `build_real_basis.py`. Kite only
  serves the current contract, but NSE F&O BHAVCOPY archives ARE reachable from the VPS → pulled **196 real
  NIFTY near-month future basis points** across COVID/2022/2018 crashes + normal months (`real_basis.csv`).
  **Findings: (1) backwardation risk CONFIRMED — COVID 52% of days negative, clustering when hedge is ON; my
  +4.6% modeled carry was too kind. (2) But BOUNDED — real hedge-on carry still +3.1% mean/+1.1% median (the
  −20..−46%/yr extremes are near-expiry annualisation artifacts). (3) Re-run with real ~+3% carry (incl. crash
  backwardation): hedge = ~9.9% CAGR / −14.8% DD / Calmar 0.67 / Sharpe 1.03** (vs B&H 10.5%/−36%/0.29; cash-rot
  7.8%/0.46). Recovers MOST of the give-up (~0.6pp vs B&H), halves DD, and GENUINELY improves Sharpe (unlike the
  Sharpe-neutral cash version). Study + HTML report updated with validated numbers + real-basis backwardation
  table. **Remaining before capital: full DAILY basis series (vs 196-pt crash sample) for path-exact P&L + a
  paper-forward soak of the futures roll execution.** NSE bhavcopy downloader is reusable for the full series.
  **PHASE 3e — bidirectional long/short? TESTED (daily+weekly ST), REJECTED (2026-07-08)** `bidirectional_st.py`.
  Idea: go net SHORT (not flat) when ST red. Short side is a structural LOSER — during ST-red the index STILL
  RISES (+6%/yr daily, +19%/yr weekly; slow filter shorts into the recovery). Short-only ~0 (daily +0.8%) /
  negative (weekly −1.9%) at huge DD; bidirectional cuts CAGR (9.9→6.6% daily, 6.3→0.3% weekly) and ~DOUBLES
  drawdown (−15→−25% daily, −31→−51% weekly, worse than B&H). Weekly worse than daily throughout. **STAY
  LONG-ONLY** — winner stands. Study + HTML report updated.
  **PHASE 3f — apply the overlay to our BEST-CAGR book? TESTED (2026-07-08)** `overlay_momentum.py`. Best recent
  CAGR = research/75 nifty250 momentum (combo__ret252 46.5% gross but −42% DD, lower-cap mirage). Overlaid the
  NIFTY daily-ST(7,3) crash filter on the tradeable base NAV (31.9% CAGR/−31.6% DD, already gated): cuts DD to
  −22% and PRE-TAX lifts Calmar 1.01→1.21, BUT **net of STCG it HURTS (0.93 < 1.01)** — liquidating a high-gain
  momentum book ~5×/yr triggers heavy tax + forgoes ~30%/yr while out. Hedge version (1.14) avoids tax but NIFTY
  futures don't cleanly hedge midcaps (optimistic). **KEY LESSON: the crash overlay's value is INVERSELY related
  to the underlying's return** — it's an index-ETF tool (low-return/high-DD), NOT for a high-Calmar momentum book
  (de-risk that with its own gate). Confirms the Phase-3c gate cross-check. research/73 design space now fully
  explored.
- [x] **research/72 — RSI 70/40 momentum-regime timing — CONCLUDED 2026-07-07: SIGNAL, not a clean STRATEGY.**
  User idea: enter stock when daily RSI closes ≥70, exit when RSI closes <40; RELIANCE base, expand to
  Nifty universe; aim = beat Nifty by ≥50% with lower DD. Master-orchestrator + 2 fan-out agents on VPS.
  Folder `research/72_rsi_regime_7040/` (engine `rsi_regime_engine.py` + `portfolio_engine.py`, phases A-E,
  RESULTS.md). **Findings:** (A) single-name RELIANCE 70/40 = **NO EDGE** (net 4.2% vs index 10.9% / stock
  B&H 17.1%; 0/75 threshold cells beat index — RSI≥70 enters late, <40 exits after the drop). (B) filters
  (MA/ADX/wRSI/ST/Donchian) **don't rescue it** — only SMA200/wRSI add ~1pp, rest just cut exposure (Calmar
  illusion); 0 configs beat index. (C) diversified slot-portfolio = **real OOS-robust momentum-breadth
  signal** but a **return/DD frontier**: broad-533 universe 2.8× index CAGR (29%) at ~index DD (−45%); blue-
  chip Nifty50 1.5× (16.8%) at lower DD (−24%) — not both. (D) edge STRONGER out-of-sample (2021-26 broad
  net 51.8%) → not overfit; param plateau. (E) 200DMA regime gate → 1 config technically passes both
  (broad exit-all 2.78×, −35.3% < −36.3%) but razor-thin, fails at 30bps + OOS. **Dominant caveats:
  survivorship + capacity** (high return = illiquid small/midcaps; research/62 already showed lower-cap
  momentum is a gross-only mirage at size). **Convergence:** at its best this IS the existing regime-gated
  momentum book (research/41/62, Calmar ~1.7) with a cruder entry → adds no new alpha. Next levers:
  liquidity-floored capacity test, vol-target sizing, or just improve the existing book. Files UNCOMMITTED
  on VPS+laptop → next git sweep.
- [x] **REC Supertrend always-on futures — CONCLUDED 2026-06-07: NO ROBUST EDGE.**
  (VPS `research/48_covered_calls_cpr_st/`: rec_st_sweep/deep/rupee, st_basket_15m,
  rec_donchian.) Daily loses to B&H. 15-min REC looked strong (OOS +29% CAGR, plateau,
  per-year+, cost-robust, ₹98k/yr/lot) BUT **basket validation (381 F&O names) killed it**:
  beats B&H only 30% of names, **11% of risers**, median Sharpe −0.37 → REC was a lucky
  single-name draw, not an edge. Donchian = peer (same fate). Also: CPR-ST morning options
  (System A+B) earlier CONCLUDED NO EDGE (real India VIX, now in DB, showed no gap-day crush).
- [x] **research/49 — volbreak_pdh_30min — CONCLUDED 2026-06-01: NO EDGE (both
  intraday AND positional).** Vol>own-50d-MA + break prev-day-high, 30-min long.
  *Intraday:* every exit net-negative @6bps (best −0.029R, PF 0.95) — cost eats it.
  *Positional (user request):* multi-day hold flipped numbers positive (daily-
  Supertrend net +0.701R / PF 1.54, several policies clear the bar) — BUT the
  **placebo/benchmark kill** showed it's **pure beta, not alpha**: SIGNAL ≈
  BREAK_ONLY ≈ random-day BASELINE for every exit; volume filter adds nothing
  (slightly hurts), prev-day-high break adds nothing over a random entry. The
  +0.70R is just large-cap drift in the 2018–25 bull. Did NOT run the 30k-cell
  sweep. RESULTS: `research/49.../results/RESULTS.md`.
- [!] **Restored 2026-06-01:** `.claude/CLAUDE.md` + `research/QUANT_RESEARCH_PLAYBOOK.md`
  had been DELETED from this laptop folder; recovered from Claude file-history (v3,
  May 31). Not yet committed/pushed — at risk again until version-controlled.

## Notes
- NIFTY lot size = 65 (2026). 5 lots = 325 contracts/leg.
- Reference spread (Sensibull): 23600/23400 PE, ~78 debit, R/R 1.56, max loss ≈ ₹25k @ 5 lots.
