# Covered_Calls — TODO

Cross-session source of truth for pending work. Each item: what / why / when.

## DONE 2026-09-15 - NIFTY long-dated short premium: 2 / 3 / 6 / 12-month tenors are all worse than the live 45-DTE book, and no stop beats no stop

Arun: *"now that we are live with 45 DTE, can we test for more like 2 months away
straddles/strangles/condors with different DTE entries and DTE exits and different stop losses,
same for other liquid ones 3 months, 6 months away, 1 year away etc?"* and *"sls can be combined
premium, single side, underlying price move, vix, relative vix or combinations."*
research/174, published at `/app/backtest/longdated-short-premium-research174`.

**NO EDGE beyond 45 DTE - CONCLUDED. Nothing deployed, nothing live touched.**

**Every tenor Arun named loses, and the decay is monotone rather than a peak.** Fixed rule -
enter at tenor T, exit at 21 DTE, no stop, 0.75% slippage, 25-contract liquidity floor on both
legs, 2015-2026: 45d +56.6 pts a trade at t 2.96, 60d +46.3 at t 1.36, 75d +30.5 at t 0.78, 90d
**-5.5**, 105d -32.2, 180d -70.4, 210d -164.3. Win rate falls smoothly 73.6 / 66.7 / 59.0 / 48.0;
drawdown grows 1,243 / 3,671 / 4,795 points. **Of 624 cells exactly TWO clear t = 2, and both are
the live book.** Selling more premium earns less: the 365-day straddle collects 1,906 points
against 640, nets nothing, and blocks 2.7x the margin for twelve times as long.

**One year and beyond is killed on the data, not the P&L.** The at-the-money call at 365 DTE
trades **54 contracts a day**; the two-year contracts trade **four**. And past ~105 DTE the strike
grid widens to 1,000-1,500 points, so the nearest listed strike is 3% from spot - a long-dated ATM
straddle cannot be placed at all.

**The eighth refutation of stops, including the three genuinely new families.** All seven families
lose PAIRED against no-stop at every tenor, and the damage is monotone in how often the stop fires
(move stop: 86% fired -49.7 pts, 61% -68.5, 41% -65.2, 25% -58.6, 10% -22.7). **Single-side was the
most promising untested idea and is refuted specifically**: at matched fire rates it is
indistinguishable from a whole-position stop, so the theta-preservation hypothesis is dead. Stops
ARE real insurance at a real premium - VIX rank above 60 cuts the worst trade from -1,049 to -464
points - but research/119 already showed the cheaper way to buy that is fewer lots.

**The mechanism, which is the reusable part.** Chop the 45-to-21 DTE hold into five five-day pieces
that each re-pick the ATM strike: +68.5 gross / +56.6 net becomes +45.2 gross / **-12.8 net**.
Re-centring gives up a third of the gross edge and pays four extra round trips. Every individual
DTE window across the contract life is statistically indistinguishable from zero. **The return is
holding ONE STRIKE through the drift, not harvesting theta** - which is why research/119 phases E
and G failed, and why all seven stop families fail here.

**Also settled.** Condors and winged straddles are dead on the index at all six tenors (consistent
with research/128; NOT a contradiction of research/127, whose wings work on idiosyncratic stock
tails). The exit-DTE choice is **not resolvable** at n=140 - 9, 14, 21 and 27 DTE have overlapping
+-2SE intervals, so the live 21-DTE rule is defensible and unrefuted but should not be called
optimised. The live **VIX-rank entry filter is validated and monotone**: 27.8 / 37.3 / 41.5 percent
a year on measured margin at off / above-25 / above-50, both halves positive throughout.

**A correction I made to myself mid-study.** An interim table showed the 45-DTE 5% strangle beating
the live straddle on t-stat, worst trade and return on margin. Paired on the same 140 entry days it
**loses** - median -26.4 points, wins on 55 of 140. The advantage was lower variance and a smaller
margin block, not more earnings. Retracted. It remains open only as a capital-allocation question
(does freeing Rs 0.65L per lot and giving up 26 points a trade help the whole book?), which is a
Capital Desk question and would need its own study and its own deploy.

**By-product: a repo-wide data defect found and permanently repaired.** `nse_options_bhav` was
missing every NIFTY expiry beyond ~75 DTE for **2016-01 to 2024-02**, plus 2026-04/05/06 and
2026-09 - eight years - because `MAX_DTE = 75` in research/89's stock-bhav downloader resumed by
trade-date, marking those sessions done so the uncapped production downloader skipped them forever.
2,102 sessions re-downloaded with zero errors, **+2,836,719 rows merged**, every year 2015-2026 now
complete. The download ran during the session but wrote to a staging file; the merge was held until
15:40 IST so it never took a write lock on the database the live executors read.

**Registered:** a 15-Sep-2027 review that re-measures long-dated listing and liquidity before this
line may be reopened (with the exact thresholds that would justify it), and a 15-Mar-2027 review
requiring any future short-premium management proposal to cite the eight refutations and say why it
is not a ninth.

---

## QUEUED 2026-09-15 -- NEXT STUDY after research/174 closes: single-stock always-on directional system (Arun)
**Arun's words (near-verbatim):** "earlier i used to trade manually on maruti always on... using the mst system
in our app... master super trend 7,5, child ST 7,2... with futures long/short on mst, options selling hedge on
cst. the aim is to find some other system apart from nifty/indexes, im inclined towards directional... need not
be by % move but even with options system to manage, so it need not be fully directionally trending... and it
can be on a single stock / few stocks - like taking ALL signals instead of scanning for signals on stocks....
say we might end up finding supertrend 7,3 all signals works on reliance over a good run, or ema crossover on
hdfcbank works with either futures or options or both or so.... take this up after current task is fully done"

**Restated:** a per-stock, always-in, every-signal system on one or a few liquid F&O names -- direction from a
trend rule (SuperTrend / EMA crossover / MST master+child), expressed in futures, options, or both -- as a
non-index, directional complement to the short-vol book. Dispatch `quant-researcher`; it must ask the intake
questions (which names, which instruments, what "always-on" means when flat is not allowed) before sweeping.

**Prior art the study MUST cite, not rediscover:**
- **The exact trap:** research/48 REC SuperTrend positional -- 15-min looked great on ONE name, basket validation
  killed it: a lucky single-name overfit. "Take all signals on RELIANCE" is that shape. The study is worthless
  without (a) the same rule on a basket of names, (b) a random-entry null on the same name, (c) a window split.
- `memory/maruthi_algo_bugs.md`: the live Maruthi MST algo was DISABLED 2026-03-25 with 9 critical bugs. If any
  MST code is reused, read that first. Design in `memory/maruthi_strategy.md`. App page at /mst.
- research/56 dual-SuperTrend 30-min options: NO NET EDGE standalone, only EV+ as a flat/hedge overlay.
- research/81/82/83: every SHORT-side equity signal tested lost (1-15d, and Turtle on F&O equities). A
  long/short system must show its short leg earns on its own or drop it.
- research/135 Turtle: optimisation was SUBTRACTIVE; the plateau test caught the overfit. Same discipline here.
- research/134: the short-vol book bleeds in UP-trends, so the value of a directional complement is in
  up-runs -- the blend test against the live 45-DTE book is the deciding metric, not standalone CAGR.
- Data: `market_data.db` is NOT split-adjusted retroactively (memory 2026-09-01) -- verify any single-name
  daily series before trusting a multi-year run on it. 60-min bars exist for 93 names 2018-2025; 5-min only
  for 10. Options on stocks: `nse_options_bhav` daily, volume/OI filter binding.

**Do not start until research/174 has a RESULTS.md verdict.**

## ✅ 2026-09-15 — The daily check watched 10 of 19 jobs, and a log with no timestamp could never clear a fixed error

Arun, told that the new-listing onboarding job had died on Monday's holiday: *"this shud hv been
listed in our app section on failed items existing"*. Correct, and it was invisible for two
separate reasons.

**1. Nine jobs were never registered.** `JOBS` in `scripts/mpf_health.py` listed the ten that
place or price orders. Everything that FEEDS them could fail nightly in silence: new-listing
onboarding, the weekly listing table, the three chart bakers, the book curves, the IPO
reconcile, the cash-park sweep, the dividend declaration. Onboarding is the one that decides
whether a newly listed stock can ever be traded at all, so its silence was the expensive kind.
All nineteen are registered now, each with the time it is due.

**2. Weekly jobs had no way to say so.** The listing table runs Sunday 10:15; a daily check with
no notion of "which days" would have called it overdue every weekday. `JOBS` rows now carry the
weekdays they are due on.

**3. The holiday branch hid real errors.** The 15-Sep holiday fix reported every job OK on a
closed day, returning *before* reading the log. Right for a job that stood down; wrong for one
that ran anyway and crashed — which is exactly what Monday's two unguarded Kite jobs did. A
traceback is now reported whatever the calendar says, flagged "ran on a CLOSED day and ERRORED";
only "it did not run" is excused on a holiday.

**4. The real one — "judge today's lines" needs a date stamp, and 16 of 19 logs have none.**
The 15-Sep fix scanned the log tail and, if today's ISO date appeared, judged only from there.
With no stamp to find it fell back to the whole 4 KB tail, so a crash fixed a week ago failed
the row every evening forever. Registering nine more jobs made this immediate: **five of the
nineteen lit up red on fossils** — the `SEEN_ORDERS` NameError fixed 13-Sep (IPO engine + IPO
recon), Monday's token rejection (universe refresh + new listings), and the 9-Sep
`could not convert string to float: '303.70.'` bug (OA entries) that prompted this script.

The checker now keeps its own **byte watermark per log** (`backtest_data/mpf_health_job_marks.json`)
and reads only what has been appended since it last looked. Independent of log formatting; the
window is frozen per calendar day so a second run in an evening still sees the morning; a log
shorter than its watermark (rotated or truncated) is read whole; a log seen for the first time
is watched from now and the row says so.

**Verified** by appending a synthetic `ValueError` to `/tmp/oa_reconcile.log` — the row went
`X Open Alpha recon ERRORED: ValueError: synthetic watcher test` — then truncating back to the
recorded 2,980 bytes, after which the row returned to OK on the next run.

All nineteen logs were read by hand before installing the watermark, to be sure nothing real was
being buried: the only tracebacks present are the five fossils above.

**Also:** `scripts/onboard_new_listings.py` was the last mpf job still outside
`scripts/on_trading_day.sh`, which is why it called Kite on Monday and was rejected. Wrapped
(crontab backed up to `/tmp/ct.bak.onboard.20260915-132646`, 137 lines before and after).

Renders on the Capital Desk daily-check card; no restart needed (cron script + static JSON).
`mpf_health.json` now carries 37 checks, 19 of them jobs.

### ⏳ Open, from the same pass

- **Five listed NSE equities are absent from `market_data.db`** — DCM, GAUDIUMIVF, MANUGRAPH,
  PRANAV, TCIFINANCE. They appear nowhere in the onboarding log, so they listed (or relisted)
  after Friday's last successful run; Monday's died on the holiday token. Confirm tonight's
  17:30 run picks them up. Coverage is otherwise 2,454 of 2,459 (99.8%).
- **The refresh cohorts can still lock a name out forever.** `refresh_daily_universe.py` runs
  `n >= 260 AND last bar within 30 days` alongside `n < 260 AND last bar within 180 days`. A
  young name that falls more than 180 days behind drops out of both and is never refreshed
  again — the same trap as the old 30-day rule, one cohort further out. No holding is near it,
  but it should be closed rather than relied upon not to happen.

## ✅ 2026-09-13 — research/172: the 52-week-high / 52-week-low channel on Nifty 100 — NO EDGE as written, SIGNAL when optimised, nothing adopted

52W. Arun: *"buy a stock when a day closes above its 52-week high and exit when a day closes
its 52-week low. apply this to simple nifty 50 and nifty next 50 stocks... Test it, optimize
it comprehensively and report back."* ₹1 crore, 20 slots @ 5%, next-open fills on both legs,
15 bps a side, after tax, idle cash 5.2%, 2006→2026. ~600 cells.

- **The literal rule: 14.25% CAGR / −46.7% / Calmar 0.305.** Beats NIFTYBEES (11.37% / −59.7%)
  and is beaten by everything that matters: its own random-entry null (median **16.47%**,
  Calmar 0.352 over 30 draws), equal-weight buy-and-hold of the same hundred names (19.22%),
  and, on the exit specifically, doing nothing at all — hold-forever returns 13.94%, so the
  52-week-low exit buys +0.31pp of CAGR for ₹2.05 crore of tax.
- **The exit is the whole optimisation.** SuperTrend(14,4) tops the 21-exit axis (median
  Calmar 0.521) — the THIRD independent confirmation after r/159 and r/161. The literal
  52-week-low exit ranks 18th of 21. Swapping it: **14.95% / −26.0% / Calmar 0.575.**
- **The entry lookback barely matters** — a flat plateau from 126 to 504 days. The NIFTYBEES
  200-SMA gate does not help this family. RS-rank slot contention is noise.
- **The kill: a momentum-matched null.** Random names from the top half of the same universe
  by 252-day relative strength return 16.79% (Calmar 0.584); trend-matched too, 17.48%. The
  system is **below the entire 30-draw range of both on CAGR**. The 52-week high is a
  low-resolution momentum proxy and True North's ranking dominates it.
- **Survivorship premium on the current official Nifty-100 CSVs: ~4.2pp of CAGR.** On a
  point-in-time liquidity top-100 the same rules return 10.77% — below the index.
- **Blend:** correlation 0.689 daily to OA·Base Age (bar 0.40); best cell +0.033 Calmar at
  −0.67pp CAGR on 22/30 paths (bar +0.10), while plain 5.2% cash wins 30/30 at every weight.

**Nothing adopted. No live book, engine or register row changed.** Three things banked for
reuse: the third ST(14,4) confirmation; a number to subtract (~4.2pp) whenever a study screens
the current NIFTY50/NIFTYNEXT50 CSVs; and a **momentum-matched null** that is much sharper
than a plain random draw and should join the standard control set for long-equity studies.


**PHASE 2 (same day, Arun mid-turn: *"u can add some stop loss variations/trailing SL etc,
try different combinations as well"*) — ~300 more cells, every stack run on BOTH the 189-day
and the literal 252-day entry so each comparison is paired. Verdict unchanged, and sharper:**

- **Initial hard stops from the entry price do nothing, and the paired entry is what proves
  it.** The −8% cell scores Calmar 0.408 on the 189d entry and 0.301 on the 252d entry —
  below its own no-stop line of 0.305. A −30% stop moves book drawdown only −46.7% → −44.0%:
  a stop on one position cannot fix a drawdown made of twenty positions falling together.
  What it does buy is a win rate collapsing 68.1% → 25.8% and a losing streak going 6 → 30.
- **Trailing beats fixed-from-entry** (Calmar 0.43–0.59 vs 0.29–0.41 — the r/71 ordering
  reproduces). Chandelier is **monotonic in width** (2×→0.281, 3×→0.498, 4×→0.512); the
  percentage trail is **twin-peaked** at −10% and −20%, and only −20% survives both entries.
- **One combination of dozens adds anything — a time stop. 52W STOPPED** = 20% trail from
  the highest close since entry + sell after 63 days if not up: **16.73% / −27.64% / Calmar
  0.605 (189d)** and **16.54% / −26.57% / 0.622 (252d)**; 12-offset band 0.603
  [0.565..0.615] and 0.640 [0.622..0.660]; survives 45 bps.
- **A hard stop wider than the trail is inert BY CONSTRUCTION** (−20% stop + −20% trail
  reproduces the bare trail to the digit). Breakeven moves and profit-locks are washes or
  entry-specific noise. Blocking re-entry after a stop-out for good starves the book (96
  trades, 8.8% CAGR). **The book-level −20% drawdown kill is a disaster: −2.66% CAGR at
  −81.4% drawdown on 15,990 trades**, re-arming into the same falling tape.
- **The auto-ranked Calmar winner (0.606) was a trap** and the pre-registered clauses caught
  it: its −15% neighbour returns −1.06% CAGR, it collapses to −7.47% at 30 bps, one of its
  twelve start-offsets scores Calmar −0.009.
- **THE DECISIVE RESULT.** Re-running the momentum-matched nulls on the winner's own exit
  stack: they **beat it on RETURN on 21–29 of 30 draws** (medians 18.16 / 18.71 / 17.27 /
  17.20%) and **lose to it on CALMAR on 27–30 of 30**. **The 52-week high carries no return
  information and real risk information** — it tells you which stock's path will be
  smoother, not which stock will go up. In Phase 1 the system sat at the null's Calmar
  median, a coin flip; the stop stack is what makes the risk edge visible.
- Blend still fails: **+0.043 Calmar at 10% weight against a +0.10 bar**, cash still wins
  30/30 at every weight, correlation to OA·Base Age still **0.679** daily.
- **A pre-registered gate was deliberately overridden** (the nulls were to be skipped unless
  the winner beat 52W OPT by +0.05 Calmar; it cleared by +0.030/+0.047). Disclosed in
  `scripts/run172e.py`, the STATUS log and RESULTS.md §15 — not done quietly.

`research/172_52wk_channel_n100/results/RESULTS.md` ·
http://94.136.185.54:5000/app/backtest/52wk-channel-n100-research172

## ✅ 2026-09-13 — research/173: IPO Base exits at the NEXT OPEN cost nothing versus the study's close exit — IMMATERIAL (it is ~1pp better)

IPO. The live book now sells at the next open for an exit decided on the close (MARKET AMO,
LIMIT −2% fallback; commit 01bc4bce); research/167 sold at the signal close. On research/167's
own engine and panel, same exits and 30 paired seeds, 2006→2026, after tax:

- Close exit (as published) 21.80% / −26.63% / Calmar 0.819, reproduced exactly.
- Next open 22.97% / −26.10% / 0.880: **+1.18pp CAGR, the close exit wins on 0 of 30 seeds**.
- Next open with the 2% floor 23.42% / −26.16% / 0.895: **+1.59pp, 0 of 30**.
- Both halves and 5.2% cash agree. The pre-registered bar (a cost of >1.0pp or >0.10 Calmar on ≥20/30 seeds) is not met; the sign is favourable.
- Overnight gap on the exit signals: mean +0.46%, 4.3% of exits open more than 2% lower, worst −6.85%.
- The floor bites on ~1 exit a year (5.3%; stops 10.6%) for 1.1 extra days on average.
- Cost side: worst-seed DD is 1.3pp deeper on the unfloored arm.

**Do NOT build a 15:05 close-proxy exit**: it moves toward the arm that loses on 30 of 30 seeds.
No live change, nothing owed.
`research/173_ipo_exit_next_open/results/RESULTS.md`

## ✅ 2026-09-13 — IPO Base exits are automated: placed by the book, booked only when the broker fills

Arun: *"pls automate this"*. Before, a live IPO exit booked the sale at the signal close and alerted
"Place it" — no order, and the reconcile ignored sells. Now the 18:45 run marks the exit due, places a
next-open SELL tagged IPO-EXIT (MARKET, LIMIT −2% fallback, retried next evening if refused, CRITICAL
alert on refusal), and the sale is booked only on the broker's fill at the broker's price. Exiting
positions free their slot for arming (as the backtest does); the 09:20 order job reconciles first so a
morning sale's cash is counted. Paper mode unchanged. Tested with a fake broker.

**Settled 13-Sep-2026 by research/173: IMMATERIAL.** Selling at the next open does not cost Spec A anything against selling at the signal close. On 30 paired seeds it was +1.18pp CAGR and +0.048 Calmar over 2006-2026, and the close exit won 0 of 30 seeds in every window. Names that close through a stop or the 50-day trail tend to open slightly higher, and names that close through +25% tend to keep going. The only give-back is a 1.3pp deeper worst-seed drawdown. A same-day 15:05 close-proxy exit would move the book back toward the losing arm and should not be built. First live exit check registered.

- Status doc: `docs/IPO_BASE_AUTOMATED_EXITS_DAILY_DEPLOY_STATUS.md`

---

## ✅ 2026-09-13 — Every True North, Open Alpha and IPO Base job now respects NSE holidays

Arun: *"this has to be the case for all jobs - monthly/weekly etc. for tn, oa, ipo all"*. Audit:
True North's month-end, backstop and report jobs already asked the calendar; its weekly gate was
fixed earlier today. **Not safe:** Open Alpha's 18:50 order job (its duplicate check reads the
broker's day-scoped order book, so a holiday re-run could re-place Sunday's PAYTM order), IPO's
09:20 order job and the cash park (weekday + time only), and Open Alpha's 18:46 mark (a fake flat
day in its value curve). Now: a trading-day wrapper in front of all 13 Open Alpha / IPO / executor /
cash-park cron jobs, plus in-code guards on the executor, the cash park and True North's 09:20
reconcile. Fails open if the calendar is unreadable. First live test: Mon 14-Sep.

**Owed:** `config/nse_holidays_2027.json` before 1-Jan-2027 (review 15-Dec) — without it every 2027
weekday counts as a trading day.

- Status doc: `docs/HOLIDAY_GUARD_ALL_BOOK_JOBS_DEPLOY_STATUS.md`

---

## ✅ 2026-09-13 — True North: the weekly gate now runs on the last TRADING day of the week

Arun: *"pls fix this"*. The weekly gate check skipped weekends but not NSE holidays, and the 15:05
end-of-day job had no holiday guard. In a holiday-Friday week (2-Oct, 25-Dec-2026) the Thursday got
no gate check and the check fired on the holiday itself, where orders are refused — the gate action
slipped a week. Now the helper asks the trading calendar (like the month-end check already did) and
the end-of-day job skips holidays. Tested on the real 2026 calendar with every step stubbed. Loads
at the next 09:00 restart. Check registered for 2-Oct. No rule changed.

- Status doc: `docs/TN_WEEKLY_GATE_HOLIDAY_FIX_DEPLOY_STATUS.md`

---

## ⏳ 2026-09-16 — Idle-cash park: IPO Base and Open Alpha idle cash into CASHIETF — built, SWITCHED OFF, live test due

Arun (13-Sep): *"can we do arb fund for ipo cash?"* → then chose the **automated liquid-ETF sweep**.

**Why not the arbitrage fund.** Kite Connect cannot place mutual-fund orders — official docs:
*"Order placement can't be done, as order placement needs payment from the user's bank account."*
Only exchange-traded instruments can be automated. Arbitrage scores kept for the 15-Dec cash-yield
review at `backtest_data/arb_fund_scores_20260913.json` (Tata 7.47%, Kotak 7.46% three-year, no
losing month). Memory: `kite-mf-orders-not-via-api`.

**The rule: never park money a buy could need before the next release.**

| | IPO Base | Open Alpha · Base Age |
|---|---|---|
| Buys placed | 09:20 by the executor | 18:50 AMOs, executed at the open |
| Kept as cash | ₹10,000 buffer — the executor sells ETF first when buy-stops need it | (free slots + 1 swap) × 6.25% × NAV × 1.10 |
| Parks today | up to ₹1.76L (₹10,000 during the test) | nothing — 5 free slots need more than it holds |

A book's `cash` still includes parked money at cost, so sizing is unchanged; NAV adds the gain; anything
that sends an order uses free cash. The park run also holds back cash tied up in the book's own resting
buy orders. 21 tests pass, including the executor's release path on a fake broker. States untouched.

**Next — 16-Sep, after the 15-Sep rebalance lands and IPO takes the cash in at 18:45:** switch IPO on with the ₹10,000 cap, watch one park at 15:10
and one explicit release, reconcile against the broker, then lift the cap. Steps in the Ops & Review
Centre entry. **Owed at switch-on:** changelog entries on both rows of the Strategies register.

- Status doc: `docs/CASH_PARK_LIQUID_ETF_IPO_OA_DAILY_DEPLOY_STATUS.md`

---

## ⏳ 2026-09-15 — Capital Desk: ONE-OFF REBALANCE to TN 37.5 / OA 37.5 / IPO 25 runs Tuesday 15-Sep 09:45

**Rescheduled 13-Sep:** moved from Mon 14-Sep-2026, an NSE holiday (Ganesh Chaturthi). The armed Monday job was killed before it could try to sell into a closed market.

Arun (13-Sep): *"now that entire TN is in cash fund, can v now make the distribution in the correct
ratio?"* — approved as "fix IPO, move all Monday".

**Fixed first, deployed 13-Sep:** a Capital Desk deposit into the already-live IPO book never reached
the cash it buys with. `services/ipo_paper.py` copied `ipo_funded` into its own capital and cash only on
a paper→live switch (once, 8-Sep). Now a FUNDING SYNC runs on every live cycle; a withdrawal that would
take cash below zero is refused and alerted. Tested with a pretend deposit and a pretend oversized
withdrawal, nothing saved, state byte-identical.

**The move (13-Sep values; recomputed live on Tuesday):**

| Book | Now | Target | Move |
|---|---|---|---|
| True North | ₹9,15,949 | ₹6,62,333 | −₹2,53,615 (₹27,578 cash + ~₹2.26L CASHIETF) |
| Open Alpha · Base Age | ₹6,21,068 | ₹6,62,333 | +₹41,265 |
| IPO Base | ₹2,29,205 | ₹4,41,556 | +₹2,12,350 |

Overrides, for this transfer only, the 05-Sep rule that True North is never sold to rebalance.

**Job:** `scripts/deferred_rebalance_20260915.sh` → `scripts/rebalance_mpf_20260915.py --execute`. Dry run
passed 13-Sep. Check `logs/rebalance_mpf_20260915.json` on Tuesday; IPO's cash updates at the 18:45 run.

**Known cost:** the moved money sits as plain cash in Open Alpha and IPO — neither sweeps idle cash into
a fund. About ₹1,100 a month on the moved sum, and ~₹3.75L across the two books is already idle.
Arun asked (13-Sep) whether IPO's cash can go to an arbitrage fund — see the next entry once decided.

---

## ✅ 2026-09-13 — OA · Base Age: "swap the last 2 instead of 1?" and "top up the winners we already hold?" — both NO EDGE, nothing changed

Arun: *"what if we swap the last 2 ranks instead of 1? can we find out an optimized number? or
maybe swap the lowest ranked one(s) and instead of new entrants, top up the highest running ones
existing within the portfolio?"* — research/171, published at
`/app/backtest/baseage-multiswap-topup-research171`.

**Swap two instead of one? No, and there is no optimum to find.** Once OA-ROT-1 is running, two
holdings are more than 10% under water on the same refused-signal evening **0.5 times a year** —
eleven occasions in 21.7 years, because the rule keeps removing the loser so a second never
accumulates. `k = 3, 4, 6` and "all eligible" are **bit-identical on all 30 seeds**. `k = 2` costs
−0.21pp of CAGR and −0.003 of Calmar and turns into a real loss at 40 bps.

**Top up existing winners instead of buying the new breakout? No — the clearest negative in the
study.** All 18 constructions lose Calmar to doing nothing and **not one beats the staged rule on
a single seed out of thirty**. A top-up spends the slot instead of refilling it, and a single
position reaches 52–81% of NAV while the ten best trades come to supply 55% of book profit.

**One cell the study invented clears the bar and is refused anyway.** Restrict it to the evenings
Arun described and it returns OA-ROT-1 to the digit with zero top-ups — all of its edge is an
unconditional −10% stop, 94% of it sits in the 2016-2026 half, and it takes one name to 40.8% of NAV.

**Nothing deployed. The live Base Age book keeps OA-ROT-1 exactly as research/165 staged it.**

Registered: the 26-Sep-2026 live-conversion review now also checks how often TWO holdings are
simultaneously eligible (expect ~0.5 a year); and a new 13-Mar-2027 review asks whether the
unconditional −10% hard stop — now surfaced three times at Calmar 0.683–0.692 with no machinery —
deserves its own study with its own drawdown bar.

---

## ✅ 2026-09-13 — IPO Base: the live book now runs the spec it is funded on (MIN_BARS 60 → 25), and the Capital Desk is on 37.5 / 37.5 / 25

Arun: *"Capital Desk target from 40/40/20 to 37.5/37.5/25 is your call - lets do this. but b4
that, pls chk this..."* — and the check (research/169) is what found the problem.

**The universe question.** IPO Base's rules fail on every other universe tried — Nifty-50-like
through all stocks, 0 of 16 beat their random-entry control. The return lives in a stock's first
weeks after listing, so IPO Base stays the only third sleeve.

**The problem it found.** `services/ipo_paper.py` ran `MIN_BARS = 60`. research/167 validated
**25**. The 6-Sep reason for 60 read research/153's `where n >= 60` as "60 bars on the signal day";
it counts a symbol's rows in the database today. On research/167's own engine: 25 bars 21.80%, 40
bars 17.87% (and loses to random on 27 of 30), 60 bars 11.57%. Funding a 25% sleeve on the 60-bar
book would have cost the blend 2.46 points of CAGR on every one of 30 paths.

**Done, in one change.**
- `MIN_BARS = 25`, the misreading explained in place, spec version `r167-A-mb25` logged on state.
- Capital Desk targets **TN 37.5 / OA 37.5 / IPO 25** through the Capital Desk route, with its
  changelog entry.
- Two figures corrected everywhere they had spread: 1.56% is the **median** position at ₹10L (the
  90th percentile is 9.05%), and the edge over random is **+2.25pp** on a clean panel, tying random
  young names since 2016 — not +4.78pp.
- Register, IPO page, portfolio report and the research/167 study entry updated.

**Nothing traded differently.** The gate is on (NIFTYBEES 2.34% below its 150-day average), so no
buy-stop is armed for 14-Sep under either floor. CORDELIA would have triggered at 25 bars.

- Status doc: `research/167_ipo_base_honest_reopt/IPO_BASE_MIN_BARS_25_DAILY_DEPLOY_STATUS.md`
- Review: 15-Oct-2026 — early-entry fills, capacity, and which age band entries actually fall in.

---

## 🔴 2026-09-13 — research/169: why IPO Base is IPO-specific — the rules do NOT transplant, and **the live IPO book runs MIN_BARS 60, not the validated 25**

Arun asked why the IPO system only trades IPOs, and what happens on Nifty 50 / 100 / 200 / Midcap / 500 /
Smallcap / all stocks, before moving the Capital Desk to TN 37.5 / OA 37.5 / IPO 25.
Published at `/app/backtest/ipo-rules-universe-transplant-research169`. **Nothing live was changed.**

**IPO — transplants: NO EDGE.** 0 of 16 size-universe transplants (age band removed, or seasoned names
only) beat their own date-matched random-entry control: 2.3-9.0% after tax at −34% to −54% drawdown.
A point-in-time market-cap re-run 2018+ agrees. Every transplant lowers the three-sleeve blend on 30 of 30
paths. **Why it is IPO-specific:** the return lives in a stock's first months after listing — age ≤ 6m
22.39%, ≤ 12m 16.28%, ≤ 24m 14.68%, no limit 6.97% — and past six months random young names beat the
breakout.

**IPO — what is OWED (by Arun, before the 26-Sep funding call; review registered 19-Sep-2026):**
- **Decide MIN_BARS.** `services/ipo_paper.py` runs 60; Spec A was validated at 25. The 6-Sep comment
  misread the harness (`n >= 60` counts rows over the whole DB today, not bars at the signal date).
  research/167's own engine: **21.80% at 25, 11.57% at 60**. Inside TN/OA/IPO 37.5/37.5/25 the 60-bar
  book costs **−2.46pp CAGR on 30/30 paths** and beats risk-matched cash by only
  +0.61pp. Changing it = its own STATUS doc + capacity check on entries
  25-60 sessions after listing + after-15:40 deploy.
- **Read research/167's null claim as +2.25pp, not +4.78pp** — +4.62pp in 2006-15, −0.18pp (13/30)
  in 2016-26 on a NaN-robust panel. The sleeve is a young-listing cohort harvest with a good exit.
- **Capacity**: research/167's "p90 1.56% of traded value at ₹10 L" is the MEDIAN; the p90 is 9.05%.
  The ₹20-25 L hard cap in the Strategies register is optimistic on the tail.

- Full write-up: `research/169_ipo_rules_universe_transplant/results/RESULTS.md`
- Status doc: `research/169_ipo_rules_universe_transplant/IPO_RULES_UNIVERSE_TRANSPLANT_DAILY_SWEEP_STATUS.md`

## ✅ 2026-09-13 — research/170: QS rank leeway is a dead axis — and Base Age's "best entrant" pick **replicated on fresh seeds** and still sits ON the bar

Two unrelated questions Arun asked on the morning of 13-Sep-2026, on two different books.
Published at `/app/backtest/qs-leeway-and-baseage-best-entrant-research170`. **Nothing is deployed.**

**Part A — Quality Summit rank leeway: CONCLUDED, NO ADOPTION.** Arun asked for "a leeway,
say within top 25" for a holding that slips in the rank. **The book already keeps a holding to
rank 23** (`ceil(buffer 1.5 × N 15)`), so a name at rank 16 is not sold today. More to the
point, instrumenting the sale reason for the first time shows **only 14.5% of sales are rank
sales** — 85.5% are the name leaving the near-all-time-high band, the liquidity floor or the
screen, and no leeway of any width can touch those. Paired on the same 12 rebalance-day offsets,
**every** width loses: rank 15/20/26/30/38/45 give −2.02 / −0.15 / −0.09 / −0.39 / −0.57 /
−0.84 pp of CAGR, winning 3/4/5/4/3/2 of 12. Arun's own proposal (rank 26) is a dead wash.
Widening **does** cut churn (74→54 trades/yr, hold 66→93 days) and **cannot** cut tax: trades
held beyond 365 days rise only 0.3%→1.6%, so the 12.5% long-term rate never arrives. The
*other* leeway — keep a name after it leaves the band — is the **worst cell in the study**
(−1.42pp CAGR, −0.146 Calmar, 5.5 extra points of drawdown). k = 0.90 stands: it is the Calmar
peak at every leeway, and k = 0.85 clears the fit window and reverses in the holdout — the
**second independent replication** of research/162's reversal.

**Part B — Base Age best-qualifying entrant: SIGNAL, confirmed, NOT adopted.** research/166
found, *after* seeing results, that giving a freed slot to the highest-RS refused entrant lifts
the rotation rule to Calmar 0.715. research/170 named the rule and the three candidates
**before** running anything and re-ran them on **seeds 1001–1030, which no cell in r/164 or
r/166 had ever touched**. It came back: **22.58% after tax, −31.78% drawdown, Calmar 0.710**
against the incumbent's 20.95% / −34.05% / 0.611 — paired **+0.105 Calmar on 30/30**, beating
a rate-matched random swap **+0.115 on 30/30**, winning **both** windows on 30/30, and holding at
40 and 60 bps. Pooled over all 60 paths: **+0.096 against a pre-registered bar of +0.100**. Three
independent evaluations have now landed at +0.094, +0.096 and +0.105. **The effect is real and
small, and the threshold sits on top of it.**

**The new mechanical fact worth carrying forward: who leaves the book sets the return; who enters
it sets the drawdown.** All three entrant priorities earn the same +1.65 to +1.77pp of CAGR; the
entire 5.5-point drawdown spread is in which refused breakout you buy (rs252 −31.78%, tv20
−33.52%, oldest base −37.23% — and the oldest-base variant actually **loses** to the incumbent
on Calmar).

**SUPERSEDED THE SAME DAY — OA-ROT-1 ADOPTED by Arun, 13-Sep-2026.** This study's standing
decision was that the live Base Age book converts *without* rotation. Arun read Part B and
overrode it: *"Swap, entrant by relative strength 22.58% / −31.8% — I love this. Let's make
changes to the live system later today, not now."* research/165 has built the rule into the
staged conversion behind its own OFF switch `OA_ROT1`; `OA_RULESET` is still `'legacy'` and
nothing is running. The **dated review 2027-03-13** stays, amended: it now asks about the live
SWAP RATE first (~4/yr expected, ~12/yr in research/165's walk of the live code) and the P&L
attribution of both legs, then re-runs the five Part-B cells against the **same +0.10 bar,
unchanged**. See the conversion entry below and
`research/165_oa_baseage_live_conversion/OA_ROT1_SWAP_RULE_DEPLOY_STATUS.md`.

## 🟢 2026-09-13 — Open Alpha · Base Age live conversion — FLIPPED 13:33 IST Sunday (Arun: "ok, go"); first evening Mon 14-Sep 18:50

**Flip record (13-Sep-2026 13:33 IST):** switch `OA_RULESET=baseage` (reads back), crontab installed from /tmp/ct.new (backup `/tmp/ct.bak.20260913-133230`, 135 lines, diff = lines 107 + 112 only): 18:50 entry job live, 09:20 top-up restricted to `--book ipo-base` (decision taken on the recommended path; Arun said go without objecting), 09:25 re-arm stays commented. Broker order book empty at flip. No restart. OA_ROT1=True. WATCH: Mon 18:50 `/tmp/oa_entry.log` (exit confirm block, PAYTM ARM, which AMO type Kite accepted); Tue 09:35 reconcile + `/tmp/equity_executor.log` must say ipo-base only. Rollback: STATUS §9. Reviews: 15-Sep day-one, 26-Sep first two weeks.

Arun: "we must convert the existing OA trades into this OA base age system, manage the exits and continue to be live with further trades." Code is on the VPS behind `OA_RULESET` in `services/oa_real.py` (default `legacy`, so nothing has changed; commits 2a0f9074, b1eed5a1). Replication gate 100.00% vs the research/164 3,619-event list; all 11 live positions HOLD under SuperTrend(14,4) (13-26% above the line); Monday only candidate is PAYTM (21 sh, about Rs 38,157). **BLOCKER before the flip:** `services/equity_executor.py` (crontab line 107, 09:20) still tops up OA holdings with real OA-TOPUP orders - settle it (crontab-only `--book ipo-base`, or a guarded engine change with its own STATUS) or do not flip. Unverified until the first evening: which AMO order type Kite accepts (MARKET, else LIMIT at close +/-2%). Runbook + rollback: `research/165_oa_baseage_live_conversion/OA_BASE_AGE_LIVE_CONVERSION_DEPLOY_STATUS.md` section 9. Register row updated (Open Alpha - Base Age (converting)); reviews 2026-09-15 and 2026-09-26 in the Ops Centre.

**EXTENDED 13-Sep-2026 — the conversion now also carries OA-ROT-1, still switched off.** Arun
adopted research/170 Part B (*"Swap, entrant by relative strength 22.58% / −31.8% — I love this.
Let's make changes to the live system later today, not now."*). On an evening when a qualifying
signal is refused for a slot or for cash, the 18:50 job sells the holding whose loss against its
buy price is worse than **−10%** on the trigger close and buys the refused signal with the
**highest 12-month relative strength**, both as AMOs for the next open (SELL first), tagged
`OA-ROT1-SELL` / `OA-ROT1-BUY`, **one swap a night**. Study: 22.58% after tax / −31.78% DD /
Calmar 0.710 against the un-rotated 20.95 / −34.05 / 0.611; **+0.105 paired Calmar on 30/30 fresh
seeds, +1.65pp CAGR on 60/60 pooled paths**. research/170 itself did **not** adopt it (it missed
its own pre-registered +0.10 bar by 0.004), so the rule has its **own OFF switch `OA_ROT1`** —
set it to `False` to stop the swap and keep Base Age, no crontab change, no restart.

**Replication gate PASS, 100.00%:** 8,488 of 8,488 rotation decisions across six engine paths
agree with research/170's own engine on fire/no-fire, the name sold and the name bought. The gate
also caught a real defect before it could reach live: `rs252` must be read on the study's
**NIFTYBEES master calendar**, not on 252 of a symbol's own bars (73.7% vs 99.7% reproduction of
the frozen figures; 16 of 810 contested days would have bought a different name).

**Two cautions to carry into the flip.** (1) Walked over the **last 400 sessions on the live
code**, the rule **LOST** on all three arms (−₹23,082 from the live book) and fired **~12 swaps a
year against the study's 4.5** — one 1.6-year path with no seeds, but it is the only walk of the
live code that exists. (2) `plan()` sizes the entry slot off **cost-plus-cash NAV** while the
study and `rot1_pick()` use the **marked** NAV — the difference between Monday's PAYTM at ×20 and
the ×21 printed above. **Not changed**; a question for Arun before the flip.

**Dry run on the 11-Sep close:** no swap can fire Monday (5 free slots and ₹1.88L of cash take
PAYTM outright, so nothing is refused), and **none would fire even on a full book** — the deepest
loss is **SBCL at −2.46%**, which would have to fall a further 7.7% to qualify.

**Runbook delta + the swap's own rollback:**
`research/165_oa_baseage_live_conversion/OA_ROT1_SWAP_RULE_DEPLOY_STATUS.md` §9.

## ✅ 2026-09-13 — research/168: the three-sleeve blend settles the IPO question — **adopt the RE-FIT, fund it at 25%; the INCUMBENT sleeve does not earn a place at any weight**

The adoption blocker named in research/167 section 8 item 1, and the last open piece of the IPO
line. r/167 had produced a better IPO standalone book that was a WORSE pairwise diversifier, so
only the blend could decide which was worth more to the portfolio. It is now run.

**Answer 1 — the re-fit is worth more, unanimously.** At the same weight, same rebalance, same 30
paired paths: **+1.73pp of blend CAGR and +0.054 Calmar on 30 of 30 paths** at a 25% weight;
+2.42pp / +0.107 Calmar at 35%; +3.45pp / +0.267 at 50%. The sign is unanimous at **every** weight
from 5% to 50%, on **both** cost bases, under **monthly rebalancing and under pure drift**, and in
**both halves** of the window (WA +0.38pp 24/30, WB +1.38pp 30/30 against the two-sleeve book).

**Answer 2 — r/167's correlation-based worry was backwards, and the inference from it is
retracted.** The refit IS the worse pairwise diversifier (monthly 0.348 to True North and 0.329 to
OA·BaseAge, against the incumbent's 0.259 and 0.319) and is still far the better blend sleeve,
because the correlation rise is swamped by the return improvement. **Pairwise correlation was the
wrong screen for this decision.**

**Answer 3 — the INCUMBENT sleeve fails the pre-registered bar at every weight.** It costs the
blend CAGR at every weight (**0 of 30** paths positive, −0.32pp at 10% to −2.03pp at 50%) and never
reaches +0.10 Calmar (best **+0.089** at a 30% weight). Against plain cash **at the same portfolio
drawdown** it is worth only **+1.17pp of CAGR at 25%**, +1.25pp at its best, and it **loses** at
50% (5/30). IPO-INC is insurance with a premium, and the premium is roughly what an arbitrage fund
would have paid for nothing. IPO-A beats cash-at-equal-drawdown by **+2.52pp at 25% and +4.79pp at
50%, on 30 of 30 paths at every weight**.

**Answer 4 — r/167's 2008 black mark washes out in the blend.** The 10.7pp standalone gap
(−10.3% for the refit vs +0.4% for the incumbent) becomes **2.5pp** at blend weight (−19.3% vs
−16.8%), and **both** arms improve on the two-sleeve book's −22.3%. Plain cash at the same weight
delivers the same 2008 cushion the incumbent does, for free. The crash-cushion argument does not
rescue the incumbent.

**RECOMMENDED WEIGHT — True North 37.5% / OA · Base Age 37.5% / IPO-A 25%, rebalanced monthly:**
**21.18% CAGR after tax [worst path 19.17%] / −24.01% MaxDD [worst path −26.39%] / Calmar 0.885**
over 2006-04-03 → 2026-09-03, against the two-sleeve book's 20.28% / −26.91% / 0.749 — **+0.91pp
CAGR and +2.9pp of drawdown on 30 of 30 paths**. 20% is the floor at which the bar clears on every
path; 35% is where the refit's edge over the incumbent also clears the pre-registered magnitude;
25% is the capacity-aware choice inside that band, fundable on a book up to ₹80–100 L because the
sleeve itself caps at ~₹20–25 L.

**Reported but NOT recommended:** the Calmar surface peaks at a **45–60%** IPO-A weight (1.05–1.06,
a flat plateau) and the unconstrained 231-combination simplex wants **TN 45 / OA 0 / IPO 55**.
Three reasons not to: the sleeve's ₹20–25 L capacity ceiling, no held-out period anywhere in the
chain (r/167 chose the trail and gate on this same window, and these weights were chosen on it
again), and it would delete a live book with twenty years of its own evidence.

**Two method findings worth carrying forward.**
1. **Calmar cannot adjudicate a cash null.** Cash has zero drawdown, so Calmar rises without bound
   with the cash weight — 100% cash scores Calmar **infinity** — and a weight-matched comparison
   flatters cash above about a 30% weight. The decision-grade form is the **risk-matched cash
   null**: solve the cash weight that reproduces the candidate blend's drawdown, then compare CAGR.
2. **A bug of mine, caught before it reached the report.** The first blend engine measured each
   rebalance period's returns from the rebalance day itself rather than the previous close, which
   **discarded the return of every rebalance day** and manufactured a fake frequency premium
   (monthly appeared to cost 1.8pp of CAGR against drift; quarterly to earn +1.4pp). The tells were
   a non-monotonic frequency response and a move far too large for the change made. Fixed, with a
   self-test that a 100% single-sleeve blend must reproduce that sleeve exactly under every
   frequency. Real effect: monthly is worth **+0.54pp of CAGR over drift**, monotone, phase
   dispersion under 0.005 of Calmar. **Any "quarterly beats monthly" reading from that session is
   retracted.**

**A separate finding that is NOT this study's question, now registered:** the two-sleeve book itself
prefers a True North tilt — TN 85 : OA 15 scores Calmar 0.826 against 50:50's 0.749, and every top
simplex cell pushes OA·BaseAge toward zero. That is a re-weighting of the live pair and needs its
own study (Ops Centre review, 2026-09-26). Every headline figure in r/168 holds the deployed 50:50
ratio fixed precisely so the IPO question is answered on its own.

**Housekeeping.** Every sleeve was re-measured at the common **5.2% post-tax idle-cash standard**
(research/163) through a reproduction gate first — True North bit-exact (0 of 5,066 rows differ),
OA·BaseAge bit-exact on all 30 paths, both IPO arms exact to +0.000pp against r/167's stage 9 —
because the four curves had been produced at 6.5% / 5.5% / 5.0% while the books hold 57% / 27% / 68%
cash, and blending them as produced would have biased the weights. One basis is NOT harmonised and
is inherited, not introduced: True North runs at 15 bps a side (r/144's `rt = 0.003`) against 25 bps
for the other two; every conclusion was re-measured with True North also at 25 bps and is unchanged.

**Nothing was deployed by this study.** `services/` and `frontend/` were not touched. The re-fitted
spec is already live in the IPO paper book (see the 12-Sep entry below); this study says the
**weight** it should eventually be funded at is **25%**, not the 0–15% a correlation-only reading of
r/167 would have suggested.

- Full write-up: `research/168_three_sleeve_blend/results/RESULTS.md`
- Status doc: `research/168_three_sleeve_blend/THREE_SLEEVE_BLEND_IPO_WEIGHT_DAILY_SWEEP_STATUS.md`
- Per-year house table: `research/168_three_sleeve_blend/results/peryear_table.md`
- The test that kills the incumbent: `research/168_three_sleeve_blend/results/risk_matched_cash_null.csv`

## ✅ 2026-09-12 (evening) — every Momentum Portfolio book now credits idle cash at **5.2% post-tax — the arbitrage-fund rate** (research/163)

Arun: *"the Momentum Portfolio report `/app/mpf-report` must credit idle cash at 5.2% a year
post-tax on EVERY book (was 5.0%) … idle cash is kept in the best post-tax cash instrument …
nothing else about any system changes."*

**Nothing about any system changed.** No entry, exit, stop, slot count, universe or gate moved;
no executor, no cron, no live DB, no backend restart. One input moved on all five books, each
re-run on its own study's engine.

**Why 5.2%, and it is now written on the page.** Idle cash is assumed to sit in an **arbitrage
fund**, which carries **equity taxation** — 20% STCG on units churned inside a year, 12.5% LTCG
beyond a year, ~0.25% exit load inside a month — so **~6.5% pre-tax** at 2025-26 cash-futures
spreads is **~5.2% post-tax**. A liquid ETF is taxed at slab and would be only ~3.5% post-tax at
30% (`LIQUIDCASE` / `LIQUIDADD` / `LIQUIDBETF` in `market_data.db` realised **5.4–5.5% pre-tax
in 2025**, ~**5.0% annualised in 2026**). Operating rule assumed: **bulk in the arbitrage fund,
a liquid-ETF buffer** for money needed at the next open, because arbitrage redemptions settle
**T+1**. It is a **flat assumption, not a measured yield** — hence the dated review below.

**Before → after, after tax, both windows.**

| Window | Row | CAGR 5.0 → 5.2 | Δ |
|---|---|---|---|
| 20.4y | True North | 18.56% → **18.69%** | +0.13 |
| 20.4y | Open Alpha · Base Age | 19.93% → **19.99%** | +0.06 |
| 20.4y | IPO Base | 15.10% → **15.26%** | +0.16 |
| 20.4y | TN + Base Age 50-50 | 19.80% → **19.89%** | +0.09 |
| 20.4y | NIFTYBEES | 10.58% → 10.58% | 0.00 — bit-identical, it holds no cash |
| 2018 | True North | 19.80% → **19.93%** | +0.13 |
| 2018 | Open Alpha · Base Age | 25.54% → **25.57%** | +0.03 |
| 2018 | Quality Summit | 20.90% → **20.63%** | −0.27 (path re-draw, see below) |
| 2018 | IPO Base | 12.97% → **13.08%** | +0.11 |
| 2018 | TN + Base Age 50-50 | 23.40% → **23.48%** | +0.08 |
| 2018 | NIFTYBEES | 10.92% → 10.92% | 0.00 |

Open Alpha · ATH + VIX: 19.23 / −34.15 / 0.56 → **20.44 / −32.92 / 0.62**, 30-seed band
15.60–22.57%. **Nothing reorders on either window.**

**Each move is the arithmetic.** Twenty extra basis points are earned only on the cash share, so
the gain is ≈ (1 − invested) × 0.2 pp — and the ordering proves the mechanism: least-invested
book gains most (IPO Base, 32% invested, +0.16), most-invested gains least (Quality Summit, 91%,
+0.01), fully-invested gains nothing. Every book passed a **paired** consistency test
(`scripts/check_cash052.py`, PASS).

**Every harness proved itself before it was believed.** No curve was written until the same
script reproduced that book's published 5.0% curve — True North 5,066/5,066 rows, Base Age
**bit-exact on all 30 paths**, IPO Base 5,128/5,128 and its published 15.00%, ATH + VIX
2,642/2,642 *and* the published 19.23 / −34.15 / 0.56 exactly, Quality Summit bit-exact on its
frozen panel. Two scripts stopped at their own gates and the gates were fixed to test the right
thing, not loosened.

**The finding worth remembering: single-path re-draw swamps the cash effect.** Changing the cash
rate changes integer share counts, which changes whether a buy is affordable, which re-draws
every later selection — worth up to **±2 points** of CAGR on one path against the 0.02–0.16 the
rate is actually worth. So the drawn path can move 30× too far or the wrong way. Handled by
freezing the drawn seed at the one the 5.0% page drew and testing consistency paired across the
ensemble; the ATH + VIX row now publishes its band and says to read the band, not the point.

**Not yet at 5.2%, and the page says so:** the entry-surface / null / gate-bake-off tables
(70 rows × 30 seeds, ~4 h) are still research/159's 5.0% run. `scripts/aftertax_all_052.py` is
running; the generator switches to it automatically on the next regen and meanwhile **labels
that section with the rate it used**. Ordering there is unaffected.

**Evidence:** `research/163_mpf_cash_yield_harmonisation/MPF_CASH_YIELD_5P2_DAILY_RUN_STATUS.md`,
`results/cash052/RESULTS_CASH052.md`, scripts `{tn,ba,ipo,oa_vix,qs}_cash052.py` +
`build_inputs_052.py` + `check_cash052.py`.
**Page:** http://94.136.185.54:5000/app/mpf-report

### ⏳ OWED (Arun) — move True North's idle cash to an arbitrage fund + liquid buffer

**Operational cash management, NOT a model or executor change.** True North's idle cash sits in a
liquid instrument today and the book is **in cash 57% of the time**, so it is the book with the
most riding on where that cash actually sits. Move the bulk into an **arbitrage fund** and keep a
**liquid-ETF buffer sized for the gate's re-entry**: the NIFTYBEES 100-SMA weekly gate
**liquidates the whole book** and then re-buys 8 names, so the buffer must cover a **full
re-entry within T+1** of a redemption — otherwise the redemption has to be placed **the day the
gate signals**. **Record the fund chosen and the date in the Capital Desk / True North dashboard
note.** Do not touch any executor for this.

### ⏳ Dated review — **2026-12-15**, PENDING (top of the Ops & Review Centre)

*"Momentum Portfolio - idle cash instrument: pick the arbitrage fund, add the liquid-ETF buffer,
measure the realised post-tax yield."* Tasks: **(0)** the owed action above; (1) name the
instruments held and the buffer size; (2) measure the **realised** post-tax yield on the idle
balance since the switch, with its source (broker/AMC statement, not a quoted headline yield);
(3) re-state the page's cash line as that measured number.
**PASS** = the report's cash line reads a **measured** number with its source named. If it
differs from 5.2% by more than **0.5 points**, re-run the curves via
`research/163_.../scripts/{tn,ba,ipo,oa_vix,qs}_cash052.py` → `build_inputs_052.py` →
`check_cash052.py` → `research/_utilities/mpf_report_build.py`. Each script reproduces its own
published curve at the old yield before changing it, so the re-run is self-gating.
Mirrored in `docs/LABS_AND_JOBS_REFERENCE.md`.


---

## ✅ 2026-09-12 — IPO Base: the research/167 spec is DEPLOYED to the live book, and the book was found DEAD for four sessions

Arun: *"also implement this in the current IPO system, make the required changes... the one
running positions, ensure the exit/management is within the new framework, also list down and
ensure the code is armed for new entrants the next trading day. u may list down the potential
candidates in the IPO page as a section if not already there"*.

Deployed at 23:45 IST on a Saturday, so outside market hours on a non-trading day. No backend
restart was needed — this book runs from cron, not from the Flask process.

### THE OUTAGE CAME FIRST, because it is worse than the spec change

**The live IPO book had not run since 8 September.** Commit `3829ad71` (3-Sep, "IPO Base marks
intraday") rewrote the constants block in `services/ipo_paper.py` and silently dropped two
lines: `IPO_TAG` and `SEEN_ORDERS`. Both are read **only** inside the `mode == 'live'` branch,
so nothing failed while the sleeve was on paper. Then it was funded with real money on 8-Sep
and **every nightly cycle and every reconcile crashed on a `NameError` from that moment**.

- Four trading sessions with no exit evaluation and no buy-stops armed, on real money.
- Nothing alerted. The cron redirects the traceback into `/tmp/ipo_paper.log`, which no
  monitor reads.
- Re-checked: KISSHT would have been **HELD** on 9, 10 and 11 September under both the old and
  the new exit rules, so **no exit was actually missed**. That was luck, not design.
- Constants restored verbatim from `3bc0b4f4`. The book now runs.

**The broader fix is still open and matters more than this one book:** a crashing cron on a
live book should raise an alert, not write a traceback to a file nobody reads. Every
`*_paper.py` and `*_real.py` cron shares that shape. Review registered for **19-Sep-2026**.

### What changed in the spec (research/167 Spec A)

| Dial | Was | Now | Worth |
|---|---|---|---|
| Trail | close below the 20-day average | close below the **50-day** average | ~+7pp CAGR, and the **only** dial that lifts this book above random stock selection |
| Stop | 8% below the fill | **10%** below the fill | ~+0.8pp; 10% is the plateau centre, 8% was one notch tight |
| Market gate | none | **no new entries while NIFTYBEES closes below its 150-day average** | 17.8pp of drawdown removed on 30 of 30 paths, for a coin flip on return |
| Fill | `max(pivot, open)` | `max(pivot, open)` **only if the day's high reached the pivot** | correctness, not return — see below |

Target (+25%), base geometry, universe, liquidity floor, 8 slots at 18.75% and the tie-break
are all unchanged, and all were re-confirmed on the honest entry.

### The fill defect, fixed in the same pass

The book booked a fill at `max(pivot, open)` **without checking that the day's high ever
reached the pivot**, so a buy-stop resting above a market that never got there was recorded as
a filled position. research/167 measured it at about 1.5% of signals, every one of them
flattering. The day's high is now required, and a non-fill is logged with the high that
missed, and shown on the page.

### Running positions brought onto the new framework

A stop is stored on the position row as a price at entry, so a position bought under the old
dials would have run the old rule for its whole life. `migrate_spec()` re-bases it once,
idempotently, keeping the prior value as `stop_prev` so the change is auditable:

- **KISSHT** stop **297.94 → 291.47** (0.92 → 0.90 of the 323.85 fill). Looser, which is the
  intended direction.
- Its trail moved from the 20-day average at **310.43** to the 50-day at **314.97** — slightly
  *tighter* for this particular name, because its recent closes sit below its older ones.
- Last close 328.40, so it is 12.7% above the new stop, 4.3% above the new trail, and 1.4% into
  a +25% target at 404.81. It survives every new exit.

### Armed for the next session: NOTHING, and the reason is the new gate

`NIFTYBEES` closed at **267.40** on 11-Sep against its 150-day average of **273.81**, i.e.
**2.34% below it**. The gate is **ON**, so the book arms no buy-stops for Monday. Zero names
triggered in any case. Held positions are untouched — the gate blocks buying only.

The pipeline behind that, on the page now:

| Stock | State | To pivot | Close | Pivot | Base depth | Traded value |
|---|---|---|---|---|---|---|
| CMRGREEN | watching | 6.7% away | 216.50 | 231.03 | 11.2% | ₹12.6 cr |
| VEDPOWER | watching | 11.2% away | 33.73 | 37.50 | 10.0% | ₹41.8 cr |
| VAML | watching | 11.8% away | 422.60 | 472.60 | 9.4% | ₹256.3 cr |

A **watching** row already satisfies every rule except the trigger: inside the age band, base
no deeper than 30%, clears the ₹5 cr liquidity floor, not already extended. One close above the
pivot makes it an order — unless the gate is on that evening.

### Page and registry

- New **Candidate pipeline** card on `/app/ipo-paper`: what triggered and was armed, what
  triggered and was passed over **and why**, the near-pivot watchlist, and any buy-stop that
  did not fill with the high that missed it. An empty armed list reads very differently when
  six names triggered and the gate blocked them than when nothing triggered.
- Study published at `/app/backtest/ipo-base-honest-reopt-research167`. The research/153 page's
  verdict now opens by pointing forward to it, because every performance figure on that page is
  the look-ahead one.
- Strategies register: the IPO row now carries the research/167 rules, and **two drifts are
  corrected** — it still said `paper` on ₹10L notional while the sleeve has been **live on
  ₹2,28,711 since 8-Sep** holding a real position. A hard capacity cap of ₹20–25L is recorded
  on the row.

### Still open

1. ~~**The three-sleeve blend**~~ — **ANSWERED 13-Sep-2026 by research/168: fund the re-fit at
   25%** (True North 37.5 / OA Base Age 37.5 / IPO-A 25, monthly → 21.18% CAGR after tax /
   −24.01% DD / Calmar 0.885 against the two-sleeve 20.28 / −26.91 / 0.749). The re-fit beats the
   incumbent on **30 of 30 paired paths at every weight tested**, and the correlation worry quoted
   here was **backwards** — the refit is the worse pairwise diversifier and still the far better
   blend sleeve. The **incumbent** sleeve fails the bar at every weight and is worth only ~1pp of
   CAGR over plain cash at equal drawdown. Remaining item, owed by **Arun, not research**: the
   funding call (Ops Centre review 2026-09-26), and a live-fill check before going past 25%
   (review 2026-11-28).
2. ~~**2008**~~ — **largely closed by research/168.** The 10.7pp standalone gap becomes **2.5pp at
   blend weight** (−19.3% for the refit vs −16.8% for the incumbent at a 25% weight), and **both**
   arms improve on the two-sleeve book's −22.3%. Plain cash at the same weight delivers the same
   2008 cushion the incumbent does, for free — so the crash-cushion argument does not rescue the
   old spec. The book is still not a crash cushion and should not be relied on as one.
3. **Cron-crash alerting** across every paper and real book (above).
4. **The rename defect** (`LOTUSDEV` → `LOTUSDEV-BE`) hits this book hardest of the three. Ten
   of eleven stale young names are missing from the instrument dump. It corrupts live signals,
   not the backtest. Still not fixed.

---

## ✅ 2026-09-12 — research/167: IPO Base re-optimised on a placeable entry — **the adopted spec has NO EDGE; the re-fit is a STRATEGY candidate, nothing deployed**

Arun: *"The largest piece of work not started is the one you named: improving IPO Base. proceed"*.

**The correction first.** research/153 published 31.0% for IPO Base. That figure rests on the
same-bar look-ahead fill this project found across three books (research/158, research/159): the
trigger is a close above the pivot and the fill is that same day's open, which no order can place.
Measured on the entry the **live book actually uses** — next-day buy-stop at the broken pivot,
filled `max(pivot, open)` — the adopted spec returns **14.90% after tax, −38.6% drawdown,
Calmar 0.386**.

**And at those parameters it is NO EDGE, not merely weaker.** Against a date-matched random-entry
null — same days, same number of entries, names drawn at random from the same young-and-liquid
universe, the same fill convention and the same gate on both arms — it returns 14.90% against the
null's 15.11% and wins only **14 of 30 paired seeds**. Gated, 8 of 30. research/153's own null
showed +5.8pp, but it was run on the close-fill arm rather than the live next-day-stop arm.

**One dial carries the entire edge, and it is the trail.** Real-minus-null in points of CAGR along
the trail axis, 30 paired seeds at every cell, stop 15%:

| trail | 10 | 15 | 20 | 30 | 40 | **50** | 60 | 75 | 100 |
|---|---|---|---|---|---|---|---|---|---|
| edge (pp) | −0.29 | −0.88 | −0.06 | +1.54 | +2.80 | **+4.91** | +3.28 | +2.05 | +1.21 |
| real wins /30 | 11 | 5 | 13 | 30 | 30 | 30 | 30 | 30 | 27 |

Zero or negative across the incumbent's whole region (≤20), unanimous across the entire 30–75
band, and the same shape at a second independent stop value. A smooth plateau maximum, not a spike.

**Spec A — the recommendation, not an adoption.** Three dials change from research/153: trail
SMA-20 → **SMA-50**, stop 8% → **10%**, and a **new** gate blocking new entries while NIFTYBEES sits
below its 150-day SMA. Everything else in research/153 survives re-fitting: the base geometry
(age ≤6 months, 25-bar base, depth ≤30%, RS off) tops a 128-cell grid, and 8 slots at 18.75%
beats 5×20%, 10×10% and 16×6.25%.

| | incumbent | **Spec A** |
|---|---|---|
| CAGR after tax (30-seed median) | 14.90% | **21.80%** |
| seed band | 13.11 – 16.73 | **20.83 – 23.19** |
| CAGR with idle cash at 0% | 11.08% | **17.97%** |
| max drawdown, median / worst seed | −38.6% / −46.4% | **−26.6% / −32.9%** |
| Calmar, median / worst seed | 0.386 / 0.282 | **0.819 / 0.634** |
| mean invested fraction | 31.8% | 36.3% |
| per-trade expectancy after 50 bps | +2.31% | **+6.24%** |
| win rate / longest losing streak | 39.9% / 18 | 49.0% / 11 |
| cost ladder 25 / 40 / 60 bps | 14.90 / 12.94 / 10.47 | 21.80 / 20.81 / 19.02 |

**The gate is insurance with no premium:** a coin flip on return (14 of 30 paired seeds) that
removes **17.8 points of drawdown on 30 of 30 paths**. SMA-200 agrees, SMA-100 fails, so the
region is bounded at both ends.

**Cash yield answered, because Arun asked.** Every figure above credits 5% on idle cash, as the
engine default does and as the live books genuinely earn by sweeping to CASHIETF. At zero yield
the incumbent falls to 11.08% — **a quarter of its headline was the sweep** — and Spec A to
17.97%, which is 17.6% of its headline.

**NOTHING DEPLOYED, and two things block the call.**

1. **The 3-sleeve blend was never run.** It is the question that decides adoption, and the refit
   argues against itself here: it **raises** correlation to the other books (0.282 weekly to OA
   Base Age and 0.256 to True North, against the incumbent's 0.245 and 0.211). A sleeve can be
   worth more standalone and less to the portfolio.
2. **2008 is the honest black mark.** The refit loses 10.3% in 2008 where the incumbent made
   +0.4% — the fast SMA-20 trail that costs 7 points a year in normal times is exactly what
   sidestepped that crash. The gate recovers part of it (−18.3% ungated → −10.3% gated), not all.

**Read the headline down.** ~350 cells were scored, so 21.80% should be read as **19–22%**; the
trail and the gate were both chosen after seeing the data, with only the 2006–2015 / 2016–2026
split as out-of-sample evidence (both pass: +8.5% and +5.5% per trade); and **capacity caps this
sleeve at roughly ₹20–25L permanently** — the p90 position is 1.56% of the name's own 20-day
traded value at ₹10L and about 90% of a day's volume at ₹1cr.

**Two live-book defects surfaced, neither touched.**

- `services/ipo_paper.py` books `fill = max(pivot, open)` **without checking the day's high
  reached the pivot**, so a buy-stop that never triggered is recorded as filled. 1.5% of signals;
  it makes the paper book's fills slightly optimistic against this study.
- The rename defect (`LOTUSDEV` → `LOTUSDEV-BE`) hits **this book hardest of the three**, because
  it trades exactly the young, thin names NSE moves to trade-for-trade. Ten of eleven stale young
  names are missing from the instrument dump. It corrupts live signals, not the backtest.

**Not tested:** the blend (above), VIX gates, risk-based sizing and the structure stop, pivot on
highs rather than closes, a walk-forward beyond the two-window split, and — the one worth
returning to — **a re-fit of the 15:10 close-fill arm**, which scored 17.39% against 14.90% at the
*incumbent's* parameters and was never re-optimised. Possibly ~2pp left on the table.

- Full write-up: `research/167_ipo_base_honest_reopt/results/RESULTS.md`
- Status doc: `research/167_ipo_base_honest_reopt/IPO_BASE_HONEST_ENTRY_DAILY_SWEEP_STATUS.md`
- Index row filed; review registered for **2026-09-26**, the same date as the Base Age paper-book
  call, so the two sleeves are decided together.
- Folder renumbered 163 → 167: a sibling session already held 163, and 164–166 are taken.

---

## ✅ 2026-09-12 — research/164: Open Alpha · Base Age slot count and position size finally tested — **16 × 6.25% survives, no spec change**

Arun: the sixteen-slot, 6.25%-per-slot book was never tested for Base Age. It was **inherited**
from the old Open Alpha, whose 680-cell sweep (research/142) was scored entirely against a
**same-bar look-ahead entry** — and research/158 / research/159 showed those surfaces INVERT
once the entry is made placeable. research/161 swept age, depth, volume, saucer, exits and hard
stops but has **no slot column at all**. That gap is now closed.

**Nothing was deployed or papered. No engine under `services/` was touched, no backend restart,
no frontend change** (another agent held `frontend/` for research/163).

**Harness proof first.** Reproduced research/161's published winner to the last digit at 5.5%
idle cash (21.26% / −34.80% / Calmar 0.618, worst seed 19.87%) and research/163's independent
5.0% re-run exactly (20.94% / −35.50% / 0.601, worst 19.81%, 72.9% invested), on an event list
rebuilt from scratch that contains the same 3,619 events.

| Question | Answer |
|---|---|
| How many slots? | About **ten**, not sixteen — Calmar humps at 9-12 (0.629 / 0.669 / 0.670 / 0.633 vs 0.601). Worth +1.1 to +1.4pp CAGR, 25-27 of 30 paired seeds, both windows, on a plateau, surviving 40 and 60 bps. **Below the pre-registered bar** of +0.10 Calmar or +2pp CAGR. 0 of 32 cells clear it. |
| At what size per slot? | **Keep 6.25%, tied to the slot count.** With the count fixed, shrinking the position only de-levers — not one of the 11 eligible cells comes from that axis. |
| Does a cash buffer help at 5%? | **No.** Best buffered cell 15.46% / −23.69% / 0.647; 11 slots fully invested gives a better ratio (0.670) AND 6.6pp more return. |
| Who wins a contested slot? | **The most liquid candidate** — the only rule that beats the random null at both slot counts (+2.60pp at 8, 30/30 seeds; +0.78pp at 16, 29/30), both windows. Relative strength wins at 8 and LOSES at 16, so it is noise here. |
| Change the spec before 26-Sep? | **No.** Nothing cleared the bar, and the two things the bar does not measure — outlier dependence (ten best trades go from 35.7% to 53.4% of profit) and capacity (median position 0.43% → 0.77% of the name's own traded value, ×10 at ₹1 crore) — both argue against concentrating. |

**The finding that matters most is not on any of the four axes.** At every slot count, the
commonest reason a qualifying signal is NOT taken is that the book **has no cash**, not that it
has no free slot: 3,619 events → 688 entries, 977 refused for want of a slot, **1,955 refused
for want of cash**. The book never trims a winner, so a few bloated positions absorb 95% of NAV
while slots sit nominally free. **Position drift is the untested first-order knob; the slot
count is second-order.**

**Pending — dated review 2026-10-10 (registered in the Ops & Review Center):** test position
drift on Base Age — trim a bloated winner toward its target weight, or size the next entry to
available cash instead of skipping it. Do not re-open the slot count until that is answered.

**Pending — not yet published.** `research/164_baseage_slots_sizing/results/PUBLISH_NOTE.md`
holds the exact `BacktestStudy` entry and the two charts to draw; publish once the other
agent's `frontend/` work is committed, then `cd frontend && npm run build` (frontend-only, safe
any hour).

Evidence: `research/164_baseage_slots_sizing/results/RESULTS.md`.

## ✅ 2026-09-12 — research/163: every book on /app/mpf-report now credits idle cash at 5%, and the last "NOT MEASURED" is gone

Arun: *"every book must credit idle cash at the SAME rate, 5% a year, post-tax"*, plus three
display asks on the same page. **Nothing about any system changed** — no rule, no engine under
`services/`, no live or paper book, no backend restart.

**What was inconsistent.** True North's curve came from research/144, which assumed **6.5%** idle
cash, and Open Alpha · Base Age's from research/161, which assumed **5.5%**. Every other book was
already at 5%. True North holds cash 57% of the time, so the assumption alone was worth about a
point a year to it.

**What was done.** Both books were re-run on their OWN engines with `cash_y` / `idle_yield` at 5%.
Each re-run first reproduced its published curve at the old yield: True North matched
`nav_INC_cash_n8_d15_tax1.csv` on 5,052 of 5,066 bars to machine precision (the last 14 differ
because `market_data.db` has been refreshed since 3-Sep — worth +0.05pp, reported separately), and
Base Age matched `curves161.npz['WINNER']` **bit-exactly**.

| Row, 20.4-year window | before | after | Δ CAGR |
|---|---|---|---|
| True North | 19.48% / −23.67% / 0.82 | **18.56% / −24.95% / 0.74** | −0.92 |
| Open Alpha · Base Age | 20.27% / −32.45% / 0.62 | **19.93% / −32.73% / 0.61** | −0.34 |
| TN + Base Age 50-50 | 20.42% / −25.24% / 0.81 | **19.80% / −25.68% / 0.77** | −0.62 |

IPO Base, Quality Summit, Open Alpha · ATH + VIX and NIFTYBEES are **bit-identical on both
windows**, asserted by `scripts/check_unchanged.py`. Nothing reorders. The post-tax check passed on
all four engines: every one credits the yield daily and none taxes it.

**Open Alpha · Base Age is now MEASURED at 72.89% invested** (30-seed median, band 72.73–73.05%),
with a daily series at `research/163_.../results/baseage_invested_daily.csv`. That **contradicts
the handover's unsourced ~67%** by about six points; the measurement is used. Consistency check:
the measured series predicts 0.127pp of CAGR per 0.5pp of yield, the paired per-seed median delta
is 0.180pp with an SE of ~0.13pp — agreement within noise.

**Charts redrawn** on the same page: line-weight hierarchy inverted (the compared books lead, the
blend is a thin dashed slate), drawdown fills removed, growth panel drawn weekly / measured daily,
correlation heatmaps shrunk to ~5in with the colourbar dropped and the diagonal greyed, and capped
at 560px on the page.

**What this leaves.** Nothing is owed and no periodic job was created — the generator now defaults
to research/163's curve files. Still open from before: a daily invested series for True North, IPO
Base and Quality Summit (only Base Age has one), and the proper blend/allocation study.

- Full write-up: `research/163_mpf_cash_yield_harmonisation/results/RESULTS.md`
- Status doc: `research/163_mpf_cash_yield_harmonisation/MPF_CASH_YIELD_HARMONISATION_DAILY_RUN_STATUS.md`
- Page: http://94.136.185.54:5000/app/mpf-report

## ✅ 2026-09-12 — research/162: Quality Summit could NOT be improved, and the quality screen does NOT belong inside Base Age

Arun said **"go"** at ~23:35 on 11-Sep and asked for the overlay review that research/160 had
booked for **10-Oct-2026** to be done now. Both were done the same night. **Nothing is deployed,
papered, or changed** — no engine, no live book, no `strategies.ts`, no `/app/mpf-report`. The one
operational consequence is subtraction: the October review slot is freed.

**Part A — can Quality Summit earn more and fall less? CONCLUDED, no adoption.**
The incumbent (r/160 Family B `b7`, k=0.90, 15 names) stands: **21.19% after tax / −37.1% /
Calmar 0.58**, reproduced bit-identically before anything was changed. The three axes r/160 never
tried were swept — ATR trails, alternative ranking axes, inverse-vol sizing (the sector cap was
dropped: **no sector field exists anywhere in this project**). The best cell — keep the screen,
widen the near-ATH band from 0.90 to 0.85, cut to ten names, size inverse-vol — beat the incumbent
by **+6.06pp CAGR and +0.282 Calmar on 12 of 12 rebalance offsets** in the fit window, on a
verified 24-cell plateau. **The pre-registered holdout returned −3.48pp on 3 of 12 and −0.152
Calmar on 1 of 12**, 9.22pp below its fit window against a 4pp limit written down in advance, and
it loses money without its ten best trades. Both plateau neighbours fail the same way.

**Part B — does the quality screen help inside Open Alpha · Base Age? NO EDGE. Review CLOSED.**
r/161's engine byte-identical (its no-mask control reproduces **21.26% / −34.80% / Calmar 0.618**
exactly). Applying each screen to ENTRIES only: **not one screen wins on a single seed out of
thirty** on return, in either window, under either missing-data policy. b7 costs −9.71pp; the
screen as Arun wrote it costs −17.77pp at 18.8% invested. It is starvation, not selection —
qualifying signals fall 3,619 → 468 → 76. **The quality-screen line is closed permanently**, as
that review's own text instructed on a fail.

**Part C — portfolio fit: DILUTIVE.** Against True North + Base Age 50-50 monthly (the honest pair,
24.42% / −13.71% / Calmar 1.769 on monthly marks), adding Quality Summit at 10/20/33% is beaten by
plain **cash at the same weight on 360 of 360 paths**. Monthly correlation to Base Age **0.717**.

**What this leaves for someone to pick up**

- Nothing is owed. One dated review was registered: **2027-09-12 — re-open the Quality Summit
  optimisation ONLY when the holdout has grown a year** (re-running it on the same window is
  holdout mining). The pass criterion is unchanged and written into the Ops Centre entry.
- **If Arun ever wants a lower-drawdown near-ATH momentum book**, the screen is the honest way to
  get it: it takes **12.7 to 16.9 points off the maximum drawdown on 12 of 12 offsets** for 1.3
  points of CAGR. That is insurance with a premium, not an edge, and it is not what was asked for.
- A method note worth keeping: without the pre-registered W1/W2 split and the 4pp rule, a 12-of-12
  offset sweep with a verified plateau would have been published as an improvement.

Study: **http://94.136.185.54:5000/app/backtest/quality-summit-optimisation-research162**
Verdicts + caveats: `research/162_quality_summit_optimisation/results/RESULTS.md`
Pre-registration + live log + crash recovery:
`research/162_quality_summit_optimisation/QUALITY_SUMMIT_OPTIMISATION_DAILY_SWEEP_STATUS.md`

---

## ✅ 2026-09-11 — **ONE report page for the Momentum Portfolio** — `/app/mpf-report` is live

Arun asked for a single page that says what the book actually is, replacing the roster study page
as the entry point. Built as a React page at **`/app/mpf-report`** (sidebar: Holdings → MPF
Report). **Nothing was lost:** every study page under `/app/backtest/<slug>` stays as the archive
and every section links back to it. The roster study entry in `backtests.ts` was deliberately NOT
edited — a second session was editing it at the same time — and `strategies.ts` was not touched,
because no status, size or rule changed.

**What the page is.** Post-tax only, on one basis, with every table stating WHICH SYSTEMS, WHICH
WINDOW, WHICH BASIS. The **correction leads**, above any return figure. Then a five-question Q&A,
the headline table on the **full 20.4-year common period**, the portfolio view, a clearly
secondary **2018-2026 section** where Quality Summit can be compared, four per-system sections of
six blocks each (Rules · Mechanics · Evidence · What its distinctive piece is worth · Caveats ·
Links), the after-tax evidence behind the correction, and the dated reviews plus nine owed items.
Systems are NAMED, never versioned: **Open Alpha · Base Age** (r/161) and **Open Alpha · ATH +
VIX** (r/159).

**Headline, after tax, 2006-04-03 → 2026-09-03 (20.4 years):**

| System | CAGR | Max DD | Calmar | Growth of 100 | Avg invested |
|---|---:|---:|---:|---:|---:|
| TN + Base Age, 50-50 monthly *(computed by the report, not a study)* | **20.42%** | −25.24% | 0.81 | 4,442 | n/m |
| Open Alpha · Base Age | 20.27% | −32.45% | 0.62 | 4,334 | **not measured** |
| True North | 19.48% | **−23.67%** | **0.82** | 3,787 | 43% |
| IPO Base | 15.10% | −35.86% | 0.42 | 1,767 | 33% |
| NIFTYBEES | 10.58% | −59.71% | 0.18 | 780 | 100% |

**Nothing reaches 25% after tax on the long window.** The 50-50 blend is the strongest line on the
page — essentially all of Base Age's return with True North's shallower ride, and on the 2018
window it posts **Calmar 1.21**, the best number anywhere on it. That row is the report
generator's own arithmetic; **the blend/allocation study (handover owed item 7) is still NOT
STARTED**, and it is the only structure that plausibly clears Arun's 25% bar.

**Ten generated charts**, colours constant throughout (gold TN, green Base Age, coral Quality
Summit, purple IPO, blue blend, grey index): log growth + drawdown on both windows, yearly
grouped bars, rolling 3-year CAGR against the 25% bar, two correlation heatmaps, invested-vs-cash,
and a monthly-return heatmap per system.

**How to regenerate** (safe any time — read-only over research results, no DB, no engine):

```
cd /home/arun/quantifyd && venv/bin/python3 research/_utilities/mpf_report_build.py
export PATH=$HOME/.nvm/versions/node/v20.20.2/bin:$PATH; cd frontend && npm run build
```

Run it after ANY mpf system change or curve re-run, otherwise the page shows the previous
evidence. Registered in the Ops & Review Centre (`ops_center.py` GROUPS) and mirrored in
`docs/LABS_AND_JOBS_REFERENCE.md`.

**What is owed / known gaps** (all on the page itself, in the last section):

1. **Average invested for Open Alpha · Base Age is NOT MEASURED** — the handover asserts ~67% but
   no file carries it (`full_period.py` records `None`). The page renders a visible gap rather
   than a number it cannot point at. Fix: have the r/161 harness emit that column.
2. **IPO Base re-optimisation on the honest entry — NOT STARTED.** Top priority; its 680 cells
   were scored on the look-ahead entry and Open Alpha's trail surface inverted when corrected.
3. **Blend / allocation study — NOT STARTED.**
4. **No engine writes a daily invested-fraction series**, so the "when is each book in cash" chart
   is measured averages, not a strip over time.
5. **No pre-registered soak criterion exists for a Base Age paper book** — needed before 26-Sep if
   the answer is yes.
6. True North is still a single path while the others are ensembles; a like-for-like re-run is owed.

**Note on the evidence:** `research/159/results/after_tax_tables.csv` completed at 22:18 (the
other session's `c92a4749` / `eb3c9f71`) while this page was being built, so the page carries the
VIX-gate rows after tax too — as their OWN 2016-2026 table, never merged with the 2006-2026 price
gates.

- Page: `/app/mpf-report`
- Build doc + full provenance: `research/160_quality_growth_near_ath/MPF_REPORT_PAGE_BUILD_STATUS.md`
- Generator: `research/_utilities/mpf_report_build.py`
- Handover it was built from: `docs/MPF_UNIFIED_REPORT_HANDOVER_2026-09-11.md`

---

## ⏳ 2026-09-11 — **OA V2.0** (research/161): STRATEGY candidate — named by Arun, nothing deployed

Arun's follow-up to r/159. Three answers: **base age YES** (X≥60 bars + depth≥20% → 21.26% CAGR /
−34.80% DD, +2.60pp and −6.79pp over plain ATH), **volume NO** (better trades, worse book),
**saucer NO** (5.6 trades/yr). All five pre-registered criteria pass.

**The headline finding is about the EXIT, not the entry**: swapping OA's 15-SMA+−8% pair for
ST(14,4) on the SAME plain-ATH entries is worth **+11.85pp** (6.81% → 18.66%). Base age adds
+2.60pp on top of that.

**Recorded in the app 11-Sep-2026:** study page retitled "OA V2.0", trade list (689 trades,
median seed 20) and 30-seed summary exported and served at `/app/research161/`, reports linked,
OA-vs-OA-V2.0 differences table added, and an **OA V2.0** row added to the Strategies register
under Parked · not trading. Open Alpha's own status and rules were NOT changed — only a one-line
change-log pointer.

**Next steps, in order:**
1. **Test RS >= 70** on the OA V2.0 entry — it is Open Alpha's one remaining filter and the only
   axis from its spec this study did not sweep.
2. **Reconcile with `research/159_oa_honest_reoptimization`** — that study owns the entry-mechanic,
   trail and stop axes; OA V2.0's ST(14,4) result should be checked against their trail sweep
   before either is proposed as a change to the live book.
3. **Paper-book decision** — only after 1 and 2. It has never been papered and no order has been placed.

**Open for Arun:**
- Adopt the ST(14,4) exit for the ATH-breakout family? That is the single biggest lever found.
- Add the base-age + depth entry filter (X≥60, depth≥20%)?
- **Blocked:** the portfolio-fit test vs Open Alpha cannot run until `research/159_oa_honest_reoptimization`
  lands — OA's published 34.9% rests on a same-bar look-ahead fill, so the r/154 OA curve is unusable
  as a benchmark. A dated caveat was added to r/159's RESULTS.md and STATUS for the same reason.

- Study: `/app/backtest/ath-base-age-breakout-research161`
- STATUS: `research/161_ath_base_age_breakout/ATH_BASE_AGE_VOLUME_BREAKOUT_DAILY_SWEEP_STATUS.md`

---

## ✅ 2026-09-11 — research/159: rounding base → shelf breakout near the ATH — CONCLUDED, not deployed

Arun's own chart pattern (semi-circle base + volume accumulation + breakout, near the all-time high).
Built causally in three versions after two rejections from him; v3 reproduces his KMEW trade to the day.

**Verdict: SIGNAL, not STRATEGY — and no incremental value to the book.** Per-trade edge is real
(+11.45%/trade, beats a date-matched near-ATH control by +4.73pp) and it beats NIFTYBEES on both return
and drawdown (14.60% / −24.94% vs 12.29% / −59.71%). It FAILS two pre-registered criteria: the 20% CAGR
floor, and the pre-2016 window (8.64% vs the index's 12.68%). Ten of 472 trades carry the result, the
16-slot book is only ~40% invested at ~40 events a year, and correlation to Open Alpha (0.468 daily)
means every blend weight makes the book worse.

**Nothing to do.** No deployment, no paper book. The detector is kept and its live-candidate list is a
reasonable watchlist input to the existing Open Alpha process.

- Study: `/app/backtest/rounding-base-shelf-breakout-research159`
- STATUS: `research/159_rounding_base_breakout/ROUNDING_BASE_BREAKOUT_DAILY_SCREEN_STATUS.md`
- Results: `research/159_rounding_base_breakout/results/RESULTS.md`
- **Open question for Arun/coordinator:** research number **159 is claimed twice** — this study and
  `research/159_oa_honest_reoptimization`. Next free number is 161. Not renamed unilaterally.

---

## ✅ 2026-09-11 — research/160 DONE: Arun's Screener query tested — **FAMILY A: NO EDGE / FAMILY B: SIGNAL, not STRATEGY**

**NAMED "QUALITY SUMMIT" (Arun, 11-Sep-2026 late evening) = research/160 Family B** — near-ATH (close >= 0.9 x ATH close) + loose point-in-time quality screen (profitable 3 FY, ROE avg3 > 15, ROCE > 15 or lender, sales & profit growth 3y > 10, mcap > Rs 1,000 cr, NO D/E), top-15 by RS, monthly, next-open fills, no exit. Added as a fifth book on the honest-entries roster page `/app/backtest/mpf-honest-entries-roster-2026-09` with a common-window (2018-08 ->) table, YoY, correlations, a QS-vs-OA-v2 differences table and chart `mpf-honest-roster-2026-09-qs.png` (script `research/160_quality_growth_near_ath/scripts/roster_add_quality_summit.py`). RESEARCH ONLY — not in the Strategies index, not papered; if Arun wants it papered it needs a paper book + Strategies row + soak criterion like IPO Base.

**Fundamentals now live in the central store `backtest_data/fundamentals.db` (11-Sep-2026 evening, Arun: "log the fundamentals in the database for our other researches")** — point-in-time panel `features_pit_monthly` + raw Screener annual/quarterly/top-ratio tables + `symbol_map`; self-documented via `schema_notes`; loader `research/160_quality_growth_near_ath/scripts/load_fundamentals_db.py`. Refresh review 2027-02-01 in the Ops Centre.

All three legs complete (DATA, ENGINE, STUDY). **588 cells, published, nothing deployed and
nothing papered.** Study: `/app/backtest/quality-growth-near-ath-research160`

**The replication gate ruled before any backtest did: he does not run the screen he thinks he
runs.** On his real Zerodha history the written screen picks **8 of 69** holdings and **3 of 43**
observed buys; growth→15 gives 19, dropping D/E too gives 40, and **near-ATH alone gives 42 of
69**. So two labelled families were tested and never blurred.

| | CAGR after tax | MaxDD | Calmar | % invested | verdict |
|---|---:|---:|---:|---:|---|
| **Family A — the screen as written** | **10.77%** | −29.1% | 0.40 | **43%** | **NO EDGE** — below Midcap 150 (16.41%) |
| Family B — what he actually does | 21.19% | −37.1% | 0.58 | 91% | **SIGNAL, not STRATEGY** |
| Best of all 588 cells — **no screen at all** | 25.88% | −40.9% | 0.61 | 96% | clears 25% CAGR, fails Calmar 1.0 |
| Random-selection null (same universe) | 14.82% | −41.7% | 0.37 | 99% | **beats Family A** |

**Answer to his question — does it clear 25% after tax?** Yes, but only by **deleting the
screen**. The screen as written costs **−11.90pp on 12 of 12 paired offsets** against the identical
book unscreened, and at a 0% idle-cash assumption it returns **7.74%** (three of its eleven points
were cash yield on the 57% of the book it could not fill). Delete its ten best of 213 trades and
the trade-level compounding proxy falls to **0.32× — below one**.

**Twelve fundamental masks vs a pre-registered bar (+2pp CAGR or +0.15 Calmar on ≥8/12 paired
offsets): ZERO passed.** The best of them, the loosest, buys +0.106 Calmar for −1.32pp CAGR.

**What is actually doing the work:** relative-strength ranking, worth **+7.70pp** over the random
null. The best fundamental screen is worth −1.32pp.

**Specific, actionable for Arun:**
- **Drop the 20% growth bar and the D/E ≤ 0.2 test.** They cost ~11.8 points of CAGR between
  them. Growth is a monotonic downhill slope: >10 → 21.19%, >15 → 15.43%, >20 → 10.77%, >30 → 5.94%.
- **ROCE > 15 is completely inert** — it never rejects a name ROE has not already rejected.
- **D/E ≤ 0.2 is a sector exclusion, not a quality filter** — Screener carries no Borrowings row
  for banks/NBFCs, so it removes every financial by construction.
- **The "OPM steady or rising" step makes it worse** on all four readings of it.
- **The missing exit is not missing much**: across 285 exit×gate cells, no exit beats simply
  holding on either investable book, and the NIFTY-200SMA gate raises Calmar only by parking the
  book in cash.

**Do NOT add it to the book.** Monthly correlation **0.624 to Open Alpha** (0.730 for the
price-only version). Added to the deployed TN+OA pair at 10/20/33%, Calmar falls
2.369 → 2.232 → 2.017 → 1.693 monotonically, and **holding cash in its place wins on 360 of 360
paths**.

**The one open question (registered, due 2026-10-10):** does a loose quality gate help INSIDE
Open Alpha's own entries, as an overlay rather than a standalone book? That is the only version
the 0.73 correlation does not already answer. **Blocked until `research/159_oa_honest_reoptimization`
lands** — OA's published 34.9% rests on a same-bar look-ahead fill, so the overlay must be measured
against the honest OA curve.

**Reusable residual:** a point-in-time Screener panel (2,116 pages, 256,828 rows, 141 months,
only FY filed by the decision date) and **37 causal eligibility masks**. Refresh before reuse
(dated review, 2027-02-01) — Screener figures are restated, not as-reported.

- STATUS: `research/160_quality_growth_near_ath/QUALITY_GROWTH_NEAR_ATH_DAILY_SWEEP_STATUS.md`
- RESULTS: `research/160_quality_growth_near_ath/results/RESULTS.md`
- Caveat to carry: **8-year window** (a data fact — four filed FY do not exist before Aug-2018),
  contains the 2023-25 smallcap boom, and every book here is tail-carried (top 5% of trades =
  81-96% of all trade return).

## 🔴 2026-09-11 — RENAMED SYMBOLS GO STALE SILENTLY, and IPO Base is the most exposed

**Found by chasing why Sri Lotus Developers was not an IPO Base holding.** It was a valid
candidate and triggered twice (24-Sep-2025 and 07-Jan-2026, both inside its six-month
window), but the book did not exist until 08-Sep-2026, so nothing was there to take them.
That part is fine. What is not fine is what the check turned up.

**`LOTUSDEV` is not in the Kite instrument dump at all.** The tradeable symbol is
`LOTUSDEV-BE` ("SRI LOTUS DEVLPRS N RTY L"). The nightly refresh asks for history under the
dead name, receives nothing, treats that as "no new bars", and moves on. The database keeps
the old name frozen at the rename date — **126 days stale** as of 11-Sep.

**Eleven young names are stale and ten of them are missing from the dump.** Six freeze on
exactly the same day, which is a single batch migration to the trade-for-trade series:

| Symbol | Bars | Last bar | In dump |
|---|---|---|---|
| RNBDENIMS | 113 | 2026-02-17 | no |
| CHEMBONDCH | 189 | 2026-04-30 | no |
| AMANTA / LOTUSDEV / OMFREIGHT / RAJOOENG / SYSTMTXC / UFBL | 90-233 | **2026-05-08** | no |
| DAICHI / EBIX | 8-90 | 2026-08-26 | no |
| KALYANI-BE | 5 | 2026-08-28 | yes (genuinely new) |

**This is the same root cause as the OA scanner defect fixed this morning**, in a different
place. That fix made the SCANNER name a rename instead of skipping silently
(`services/oa_entry.py`, `renamed_to`). The REFRESH still asks for the dead symbol and gets
an empty answer it cannot distinguish from "nothing to do".

**Why IPO Base is the most exposed of the three books.** It trades young, thinly traded
names, which are exactly the ones NSE moves to the trade-for-trade series. A holding or
candidate inside its six-month window that gets renamed goes stale and becomes invisible at
the moment it is eligible. True North trades the liquid top-200 and Open Alpha has a Rs 5cr
floor, so both are far less exposed.

**The fix (NOT applied — a nightly data job, not trading logic, but still a real change):**
`scripts/refresh_daily_universe.py` should resolve each symbol against the instrument dump
before requesting history. Where the plain symbol is absent but a suffixed variant exists
(-BE, -BZ, -SM, -ST), either follow the rename or raise it. Returning quietly is the bug.

**Related, still open from this morning:** 154 of the scanner-eligible symbols in
`market_data.db` no longer trade under their stored name, 110 of them recoverable series
moves. The daily check counts the drift; nothing yet repairs it.

---

## 🔴 2026-09-11 — OPEN ALPHA WAS BUYING GOLD ETFs: the universe filter never excluded them

**Found while auditing the entry, and separate from it.** `services/oa_entry.py` filters the
universe with a TICKER regex, `(BEES|ETF|LIQUID|GILT|SENSEX|NIF[A-Z]*50)`. It was written
against the ETF names that existed when research/142 was built. The 2023-2025 wave of gold
and silver funds is named nothing like any of them, so **221 instruments that are not
companies were reaching an Indian EQUITY momentum book**:

| Kind | Symbols |
|---|---|
| Gold / silver funds | EGOLD, ESILVER, GOLD1, GOLD360, GOLDADD, GOLDAXIS, GOLDBETA, GOLDCASE, GROWWGOLD, HDFCGOLD, HDFCSILVER, LICMFGOLD, QGOLDHALF, SBISILVER, SILVER, SILVER1, SILVER360, SILVERADD, SILVERAG, SILVERBETA, TATAGOLD, TATSILV, AONEGOLD, AONESILVER, BBNPPGOLD, CHOICEGOLD, HSBCGOLD, IVZINGOLD, MOGOLD, MOSILVER, UNIONGOLD, GOLDBND, SILVERBND, SILVERCASE, GROWWSLVR |
| Index / sector funds | MON100 (Nasdaq 100), MAFANG (FANG+), ICICIB22 (Bharat 22), METAL, MODEFENCE, and ~180 more sector and index ETFs |

**Why it matters beyond tidiness.** Gold ran hard through 2024-2026. The fundamental-overlay
arm that treated missing data as eligible printed **+32% CAGR** — it was not measuring
Arun's screen at all, it was buying gold funds, which have no fundamentals because they are
not companies. That number is void and was never reported as a result.

**THE FIX — by what the instrument is CALLED, not by its ticker.** A longer ticker blacklist
works until the next fund launches, which is not a fix. The Kite instrument dump carries a
long name, and it separates the two cleanly:

```
HDFCGOLD    EQ   HDFC GOLD ETF                <- fund
CHOICEGOLD  EQ   CHOICE GOLD ETF              <- fund
AONESILVER  EQ   AONEAMC - AONESILVER         <- fund (AMC-dash naming)
SKYGOLD     EQ   SKY GOLD AND DIAMONDS        <- company
DECNGOLD    EQ   DECCAN GOLD MINES            <- company
SILVERTUC   EQ   SILVER TOUCH TECHNO          <- company
```

Every fund says so in its name; no operating company does. Built to
`backtest_data/etf_exclusions.json` by
`research/158_oa_arming_width/scripts/build_etf_list.py`:

| | Old ticker filter | New name-based list |
|---|---|---|
| Instruments excluded | 125 | **346** |
| Real companies wrongly excluded | — | **none** |
| Gold/silver-named symbols left in the universe | 21 | 6, all genuine companies |

The six kept are DECNGOLD, GOLDIAM, GOLDTECH-BE, SHANTIGOLD, SILVERTUC, SKYGOLD. The old
ticker regex is retained as a second net for funds that have since DELISTED and are absent
from today's dump. The list is committed so backtests reproduce without a network call.

**Still to do:**
- **`services/oa_entry.py` still has the old filter.** Only the research fork
  (`research/158_oa_arming_width/scripts/oa_entry_mechanics.py`) is fixed. OA entries are
  paused so nothing is at risk today, but the scanner MUST be fixed before they restart.
- **research/142's published numbers carry the same contamination** in their recent years,
  since that engine uses the same regex. Re-run owed if those figures are ever cited again.
- **Check True North and IPO Base for the same class of defect.** TN builds its universe from
  a point-in-time top-200-by-traded-value list and IPO Base from a vetted listing table, so
  both are probably clean, but neither was checked for fund contamination specifically.
- Rebuild `etf_exclusions.json` periodically — new funds list constantly. Candidate for the
  Ops & Review Centre as a dated recurring job.

---

## 🔴 2026-09-11 — OPEN ALPHA BUYING IS PAUSED: the entry cannot be placed (research/158)

**Arun action needed: decide what Open Alpha does next.** Selling is untouched and running,
so the open positions keep their −8% stop and 15-SMA trail. Only the two jobs that OPEN
positions are off (18:50 scan, 09:25 re-arm), commented out of the crontab with a backup at
`/tmp/mpf/ct.bak.20260911-111828`. Arun cancelled the four resting buy-stops by hand.

**What was found.** The published r/142 entry combines two things one order cannot have at
once: the pivot price available the moment a breakout starts, AND the knowledge that the
breakout will still be there at the closing bell. It counts a trade only when the CLOSE held
above the pivot, but prices it at that day's OPEN.

Proven on the source site's own 54 published trades, not on our simulation:

| Test | Result |
|---|---|
| Their entry-day close finished above the pivot | 49 of 50 (98%) |
| Their entry day opened above the price they booked | 9 of 50 |
| Failed resting-order fills in the 120 days before each entry | **348, or 7.0 per published trade** |

HCLTECH is the clearest case: their clean 10-Jan-2025 entry at 1972.20 follows twenty
earlier days when the same level was touched and the close fell back. A live order was
filled on 18-Dec-2024 and lost. The clean entry never happens.

**Every placeable entry, 2006→2026, 30 seeds, same book (16 slots @6.25%, −8% stop,
25bps):**

| Entry | CAGR | After tax | Max DD |
|---|---|---|---|
| Published (not placeable) | 40.8% | — | −33.9% |
| Stop above the breakout candle, trail-20 | 9.9% | — | −56.0% |
| Buy at the breakout close (~15:10) | 8.5% | 5.5% | −57.3% |
| Stop above the breakout candle, trail-15 | 6.5% | 3.7% | −71.1% |
| Next-day stop at the pivot | 2.7% | −0.5% | −72.6% |
| **Touch of the pivot (what the code does)** | **−1.4%** | — | **−81.8%** |
| Same-day abort + slot recycling | −21.0% | −22.2% | −99.5% |
| **NIFTYBEES held** | **11.5%** | — | **−59.7%** |

Nothing placeable beats simply holding the index. RS selection is no better than random, so
none of these understate the book. Zero costs and the inflated fill together still only
reach 11.9%, so this is not a friction problem.

**A second, separate defect in the same file.** `services/oa_entry.py` selects `close <
pivot` — names that have NOT broken out — and rests a stop at the high. The designed
rule (and the register, and the backtest) is `close > pivot`. The two can never pick the same
name on the same day. The 04-Sep seed used the correct condition; only scanner entries from
08-Sep did not.

**Open:**
- Arun's fundamental overlay (3y profit + sales growth >15%, D/E <=0.20, ROE and ROCE >15%,
  no negatives) is measuring now on Screener point-in-time annuals over Aug-2024→Sep-2026.
  It is the last idea on the table for rescuing the entry.
- Same-day abort WITHOUT slot recycling: queued.
- Two-year trade ledger on the app for manual verification: requested, not built.
- If nothing clears the index bar, the decision is whether Open Alpha continues at all.

Files: `research/158_oa_arming_width/` (STATUS-MD, scripts, all logs).
Commits: `18066bfc`, `f68efc9c`, `677b696a`.

---

## ⏳ 2026-09-11 — IPO Base: honest number is half the published one (research/158)

**No action forced; the book is sound and stays live.** IPO Base's live engine is CORRECT:
it triggers on tonight's close and fills the next morning. The r/153 study that justified it
is not — it enters on the same day the close clears the pivot, priced at that day's open,
which needs the close known at the open.

| IPO Base arm, 2006→2026, 30 seeds, 25bps, after tax | CAGR | Max DD | Calmar |
|---|---|---|---|
| Study headline (reproduced to within seed noise) | 31.5% | −20.9% | 1.52 |
| Study's own close-fill control | 17.5% | −32.1% | 0.53 |
| **Live engine: next-day stop at the broken pivot** | **15.0%** | **−37.6%** | **0.40** |
| NIFTYBEES held (pre-tax) | 11.5% | −59.7% | 0.19 |

It still beats the index on return AND on Calmar, and these are after-tax figures. Verdict:
survives at about half its advertised strength. Caveat: the book sits ~2/3 in cash by design,
so a standalone comparison with a fully invested index is not like for like.

**Owed:** re-publish the r/153 study page with the honest entry arm shown alongside the
headline, so `/app/backtest/ipo-base-breakout-research153` stops advertising 31%.

---

## ✅ 2026-09-11 — True North audited, no defect found (research/158)

Checked for the same defect class and cleared. Its engine holds only closing prices, so a
decide-at-the-close-fill-at-the-open mismatch is structurally impossible. The gate is
NaN-robust (computed on the dropna'd series, then reindexed — the exact fix r/142 needed
after phantom holiday rows silently disabled its gate for months). The universe is built
point-in-time. Every live dial matches the study: 8 slots, 22-name buffer, 200-name universe,
NIFTYBEES 100-SMA weekly gate to cash, 15-day Donchian, 0.3% round-trip, 6.5% idle cash,
month-end rebalance. **Published numbers stand.** Not re-verified: survivorship and the RS
formula, and a line-by-line implementation comparison (dials only).

---

## PENDING 2026-09-09 -- 45-DTE straddle LIVE executor deployed, ARMING is Arun's call
`services/straddle45_live.py` automates entry/management/exit for the research/119 book at
3 lots real money. Cron `*/2 15:18-15:30 Mon-Fri`. **Unarmed** unless `STRADDLE45_LIVE=1` is
exported in the cron line -- unarmed it dry-runs and logs the exact orders it would send.
KILL: `touch backtest_data/straddle45_KILL`. PANIC: `straddle45_live.py panic`.
**11-Sep is a SKIP as things stand** -- India VIX 11.15, rank 17.9, filter needs >25. The
executor will correctly do nothing unless vol rises. `STRADDLE45_OFF_PLAN=1` overrides;
deliberately NOT set.
Three date bugs were caught in pre-flight, each of which would have failed SILENTLY:
sessions() excluded today; entry_session() collapsed to TODAY for future dates (would have
entered 2 days early); exit_session() same collapse (would have closed the position on the
day it opened). All three now calendar-based and verified against the published plan
(entry 11-Sep, exit 06-Oct). **The order-placement path itself has never been exercised** --
first-fire review registered in Ops Center for 2026-09-12.
Still open: ring-fence Rs11.96L vs Rs13.5L (3 lots breaches at an 8% move), and the
stress-margin vol axis (dated 2026-11-30).
Deploy record: `research/119_45dte_short_straddle/NIFTY_45DTE_STRADDLE_LIVE_EXECUTOR_DEPLOY_STATUS.md`.

## ✅ 2026-09-10 — ALERTS NOW REACH THE PHONE AND THE INBOX (was: they went nowhere)

Email and ntfy both deliver. The daily check reported OK for the first time on 10-Sep, and
the 11-Sep entry run's rejection summary arrived on both channels.

Five separate faults stacked on top of each other, any one of which silenced everything:

| Fault | What it did |
|---|---|
| `_alert()` wrote to `/tmp/nas_alert_feed.log` | the **cron output log of a different job**. Nothing has ever read it. |
| the delivery import could never run | `python services/oa_real.py` puts `services/` on the path, not the repo root, so `from services.dividend_notify import ...` always raised |
| 1 of 14 cron jobs sourced `.env` | the other 13 had no credentials whatever was configured |
| the credential names did not match | `.env` held `TWILIO_ACCOUNT_SID` / `GMAIL_APP_PASSWORD`; the code looked for `TWILIO_SID` / `EMAIL_SMTP_PASS` |
| `.env` had no trailing newline | the appended `NTFY_TOPIC` glued onto the previous value, corrupting a TOTP secret **and** leaving the topic undefined |

**This is why SPORTKING sat a full day below its 15-SMA trail.** The 15:18 check produced the
exact sell order and wrote it to a log nobody reads.

Still open on the channels:

- **Enable "Instant delivery" in the ntfy Android app.** Without it, Firebase batches a burst
  and alerts arrive late. The code already paces sends 2.5s apart; the app setting is Arun's.
- **WhatsApp stays off deliberately.** Credentials are present; `WHATSAPP_ENABLED=1` turns it
  on, and Twilio bills per message. Email + ntfy cover the need for free.
- **The ntfy topic name is the only secret protecting the channel**, and real alerts carry
  position data over a public relay. Rotate the topic if it is ever pasted anywhere.

---

## ⏳ 2026-09-11 — Open Alpha entries: what the morning's run left open

The book is **16/16** (12 held + 4 resting buy-stops: CUPID, MOREPENLAB, AEROFLEX,
CYIENTDLM). Two scanner faults were fixed and pushed (`869290fd`):

- the order type now follows the clock — a live order inside market hours, an after-market
  order outside. The 09:25 re-arm cron **could never have worked** before this: it runs at
  09:25 and always asked for an AMO, which Kite refuses during the session.
- a circuit rejection is read by **which field** it caps. A capped order price is
  renegotiated up to four attempts, because both caps can be hit in sequence; a capped
  trigger means the pivot itself is outside today's band, so the name is skipped and the
  slot goes to the next ranked candidate.

Left open:

- **Three names are unreachable while their circuit bands stay where they are** — TBZ
  (trigger 555.30 vs cap 526.30), BIRLACABLE (423.65 vs 393.35), BODALCHEM (184.05 vs
  174.41). Each is a pivot above the upper circuit, so no breakout can print today. They
  re-enter the candidate list on their own as bands move; nothing to do, but if names keep
  landing here the RS ranking is feeding the scanner stocks already locked limit-up, which
  is worth measuring at the soak review.
- **BLISSGVS-BE was lost on 10-Sep to the old two-attempt budget** — the exchange had
  already named a workable price and the retry was spent. Fixed; recorded because it is the
  only *missed* entry of the episode rather than an unreachable one.
- **Point the universe refresh at renamed symbols.** 169 of the scanner-eligible symbols in
  `market_data.db` no longer trade under the stored name; 110 are recoverable series moves
  (MODISONLTD → MODISONLTD-BE). The scan now names which is which. Delisted history
  **stays in the DB on purpose** — removing it is survivorship bias, and r/142's universe
  is built from it. The daily check counts the drift so 169 does not quietly become 400.
- **True North and IPO Base still debit cash gross on a buy**, so their `Unreconciled` row
  will drift again the way Open Alpha's did. Open Alpha charges the measured 25 bps now.

---

## ⏳ 2026-09-08 — also left open by tonight's Open Alpha automation

- **IPO Base has the identical missing entry scanner.** `ipo_paper.py` finds its own
  candidates but, like Open Alpha before tonight, never places an entry. Build it the same
  way (`services/oa_entry.py` is the template) once Open Alpha has run clean for a few days.
- **The gap-ceiling deviation.** The study fills entries at `max(pivot, open)` with no
  ceiling; Kite refuses SL-M via API and the exchange caps a stop-limit's spread (~3%). A
  bigger gap will not fill where the backtest took it. Measure the miss rate at the soak
  review — gap-ups are exactly where breakout edges live, so this could matter more than
  its size suggests.
- ~~**`/api/momentum-paper/state` takes 10–16s**~~ — **done 10-Sep.** `scripts/gen_momentum_state.py`
  bakes the 46-key state to `static/app/momentum_state.json` and the page raw-fetches it.
  11.65s of API became ~4ms of file, the same pattern Open Alpha already used.
- **Two credential rotations still owed** — the `.env.bak` Kite leak (02-Sep), which
  reached GitHub, and the GitHub PAT in cleartext in the VPS git remote. A third scare on
  11-Sep was contained: `.env.fixbak` (a backup made while repairing a glued `.env` line)
  was committed with Twilio credentials in it, GitHub push protection refused the push, the
  commits were rewritten before anything left the machine, the file was shredded, and
  `.gitignore` now denies `.env*` rather than the two exact names the 02-Sep incident saw.
  **No rotation owed for that one** — it never reached the remote.

---

## ✅ 2026-09-08 — Momentum Portfolio (mpf): one summary panel across all three books — SHIPPED and restarted

**"mpf"** is Arun's shorthand for the Momentum Portfolio: the `/app/portfolio` tab group —
True North (`momentum-3l`), Open Alpha (`oa-real`), IPO Base (`ipo-paper`) and the Capital
Desk. A change to one book page's panel, holdings, charts or record strip is a change to
all of them.

### ✅ Done and live (frontend, commit `59111294`, no restart needed)

New shared `frontend/src/components/BookPanel/` (`BookPanel.tsx` + `BookCurve.tsx`) now
renders the summary on all three book pages, replacing three near-copies that had already
drifted apart on date format, timestamp placement and whether the return was shown at all.

- return sits **beside** the value; second line keeps only capital and inception
- freshness (`live · Ns ago · updated dd-Mon-yyyy HH:MM IST`) in the panel's top-right
- **one chevron** at the foot of the status line opens the book record + the curve vs
  Nifty 50. Closed it costs nothing. The standalone curve card is gone from all three pages
- holdings collapsible; on OA the charts moved to sit directly under the holdings
- `manual-assisted exits` → **order health** (`all orders filled` / `N orders unfilled`,
  red), read from each book's `failed_orders` ledger — the same one that raises the email
  and WhatsApp alert
- dates dd-Mon-yyyy everywhere (IPO Base had been rendering `8 Sep`)

### ✅ RESTARTS DONE — everything above is live (08-Sep-2026)

- 15:40:11 — deferred restart fired on its own gate (0 open strangle legs).
- 15:55:48 — a second restart, because the dashed-book-lines change landed *after* the
  first one. Clock checked (15:55 Tue, market closed); open risk checked and clear: the
  only options position was a **defined-risk IDEA call spread** (short 18CE / long 20CE,
  equal quantity, so the long leg caps the short), everything else CNC equity, and **zero
  resting orders**.

Verified live: 18 books in `/api/books/liveness` (oa-real 16 open, ipo-paper 1 open);
`/api/books/{momentum-3l,oa-real,ipo-paper,portfolio}/benchmarks` all 200; portfolio curve
n=22 at −2.25% with True North (#0F6E56) and Open Alpha (#A21CAF) dashed and on by
default; `slots_total = 8` in True North's state.

IPO Base is absent from the portfolio chart's series **by design** — it has fewer than two
curve points, and two points are a line, not a record. It joins itself once it has history.

### ⚠️ `/api/momentum-paper/state` measured 15.9s cold (9.7s warm)

Worse than the 0.6–3.5s recorded earlier. It does a live Kite quote **plus** a large pandas
pivot on the request path. The first-paint path now renders the complete panel from the
baked feed so the page no longer *looks* broken while waiting, but this endpoint is still
the slowest thing in the app and every True North page load pays it. Worth a session of
its own: cache the pivot, or bake the whole thing the way Open Alpha does.

### ⏳ PENDING — the 15:40 restart (armed 13:50, PID 2193127)

`services/book_liveness.py` is edited on disk but the running gunicorn has not loaded it.
`scripts/deferred_restart.sh` is armed and re-checks the clock at wake time.

**After the restart, verify:**

```bash
curl -s http://127.0.0.1:5000/api/books/liveness | python3 -c "import json,sys; b=json.load(sys.stdin)['books']; print(len(b), 'books'); print({k:b[k]['open_positions'] for k in ('oa-real','ipo-paper')})"
# expect: 18 books, {'oa-real': 16, 'ipo-paper': 1}
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:5000/api/books/oa-real/benchmarks    # expect 200 (404 before)
```

Then open `/app/portfolio`, expand the chevron on each of the three tabs and confirm the
record line appears on Open Alpha and IPO Base (both will honestly read
"holding N — nothing closed yet"; neither has a closed trade).

**If the restart did not fire:** `cat /tmp/deferred_restart.log`, then
`sudo systemctl restart quantifyd` yourself once `TZ=Asia/Kolkata date` is past 15:40.

**Rollback:** backups of every file touched are in `/tmp/mpf/*.bak`;
`git revert 59111294` undoes the whole change.

### ⏳ Also waiting on that same 15:40 restart (commit `da46984f`)

Capital Desk: slots not counts, "0 armed" in plain words, a **What happens next** schedule,
and the combined portfolio curve behind the same chevron.

```bash
curl -s http://127.0.0.1:5000/api/books/portfolio/benchmarks | python3 -c "import json,sys; d=json.load(sys.stdin); print('combined n=%d last=%+.2f%%' % (len(d['book']), d['book'][-1]['r'])); print('series:', sorted(d['series']))"
# expect: combined n=21 last=-2.70%  ·  series includes NIFTY50 + momentum-3l/oa-real
```

Then open `/app/capital`: the status line should read `5/8 True North holdings ·
16/16 Open Alpha holdings · 1/8 IPO on paper · no buy-stops for tomorrow`, and the chevron
at its right should open the combined curve with the three books toggleable.

**Maintenance coupling to know about:** the `What happens next` list in `CapitalDesk.tsx`
(`const DAY: Step[]`) MIRRORS the VPS crontab and `momentum_paper.register()`. It is
display-only and schedules nothing, but if a job moves, move it there too or the page will
calmly state a falsehood. A comment above the list says so.

**Incident, 08-Sep 14:20:** adding the slot count to `gen_momentum_live.py` produced a
repeated `n=` keyword — a SyntaxError in a script that runs every minute of market hours.
True North's live feed went stale for one mark before restore. Display data only, no
trading impact. The patch script now compiles the result before writing it; that guard
belongs in every patch that touches a cron-driven script.

### Notes worth keeping

- OA and IPO keep trades in **JSON state**, not SQLite, which is why they were never in
  `/api/books/liveness`. `JSON_BOOKS` in `book_liveness.py` now reads them.
- `/api/books/<id>/benchmarks` replaces the book-specific
  `/api/momentum-paper/benchmarks` for new work; the old route still serves True North.
- The curve is **time-weighted with flows backed out** — OA went 0 → ₹6.18L of capital in
  four days, which on raw NAV reads as a vertical climb where the book had done nothing.
- Both books' curves are near-empty today (OA 2 points, IPO 0). The component says so
  rather than drawing a two-point "record".

### ⏳ Still open on the mpf pages

- **Capital Desk** has not been moved onto `BookPanel` — it is not a book, but its metrics
  box should still speak the same language.
- The panel's `extraSub` (CAGR / worst drawdown) differs per book because each page
  computes it differently; worth folding into the shared component.


## ✅ 2026-09-07 — research/156 sector trend: NO EDGE (rotation) / NO ADDED VALUE (sector filter)

Full verdict: `research/156_sector_rotation/results/RESULTS.md` · study page
`/app/backtest/sector-trend-rotation-research156`.

Arun asked whether the sector indices can be read for trend — ride the leaders in proportions, or
drill into their leader stocks for a curated book above 20% CAGR — and whether it complements
TN+OA. Both branches were built with **no inherited design** from OA or TN and both failed.

- **Branch A (allocate across sectors): 0 of 1,440 configurations** clear the bar. Best 16.3%
  CAGR / −36.9% DD / Calmar 0.44 after tax. Equal-weighting all nine sectors gives 14.0% / 0.32;
  **Midcap 150 buy-and-hold gives 18.0% / 0.41 and beats every cell we built.** Momentum does rank
  sectors better than chance (94th–100th percentile of a 500-draw random-sector null) — and the
  ranking is worth less than the diversification it destroys. That is the r/63 lesson again.
- **"To-be trending" is empty.** Acceleration t = 1.37 (1m) / 1.79 (3m); breadth-change t = 0.36.
  Nothing anticipates leadership.
- **Branch B (sector as a universe filter): the sector layer adds nothing.** The sector-gated stock
  book returns 32.5% / Calmar 0.85 — and the *identical stock rule with no sector filter at all*
  returns 32.9% / 0.84 and wins 11 of 16 paired offsets. All the return is stock momentum, which
  TN and OA already harvest.
- **No complement value.** Correlation 0.41–0.54 to the live legs (ceiling 0.40); best blend gain
  +0.04 Calmar at slightly lower CAGR; **a plain cash sleeve at the same weight beats every
  candidate.**
- **Confirms and extends r/147** — its single SECROT cell reproduces after tax at 8.9% / −56.0% /
  0.16, the worst book in the study.

**A reusable data warning was produced and belongs to future studies, not just this one:** synthetic
sector proxies built from today's index membership out-drift the real sector indices by **+4 to
+14pp of CAGR per year**, and **shuffled industry labels reproduce most of the apparent "sector
momentum". Any future sector work must run the same-universe head-to-head and the shuffled-label
null before believing a wide-panel result.**

### Pending — the only follow-up worth having

**Back-fill the nine real NSE sector indices to their 2005 inception** (Ops review 2027-03-07).
The database starts them at 2015, which caps the sample at 11.7 years with one crash and no 2008.
NSE publishes the history. This is a data-acquisition task, not a modelling one, and it is the only
thing that could reopen this line honestly. It would also improve any future sector-aware work
(regime gates, sector caps on TN/OA) regardless of this verdict.


## ✅ 2026-09-06 — DECIDED: no gold sleeve for now (Arun)

The book stands at **True North 40 / Open Alpha 40 / IPO Base 20**, and gold is not in it.

This closes the adoption call left open by research/147 and re-raised by research/154, whose
constrained frontier wanted OA 40 / TN 25 / IPO 20 / GOLD 15. It is a decision, not a
refutation: gold's numbers stand (it lifts blend Calmar at ~zero correlation, and it earns in
the 2018 and 2022H1 grinds *and* in the 2020 crash). Arun has chosen not to add a fourth
sleeve while the third has never traded.

**What this parks, and what it does not:**
- The **four-sleeve study** (TN/OA/GOLD/MYB, Ops review 2026-11-30) keeps its gold-only null —
  that null is what makes the study honest, so it stays regardless of whether gold is held.
- research/154's **crash-tail retraction stands on its own** and is unaffected: the deployed
  pair's worst 20-year drawdown IS 2008 at −16.5% monthly / −17.15% daily. The re-audit of
  per-window drawdowns in r/146–r/153 is still owed.
- Revisit gold if the IPO sleeve clears its soak and the book still wants drawdown cover, or
  if a crash-alpha candidate re-opened by the retraction changes the picture.


## ✅ 2026-09-05 — research/153 IPO Base breakout: STRATEGY CANDIDATE — the first third sleeve to clear every leg of the bar

Full verdict: `research/153_ipo_base/results/RESULTS.md` · study page
`/app/backtest/ipo-base-breakout-research153`.

**The study was really a data-integrity study.** We have no listing-date table, and the obvious
proxy (a symbol's first row in `market_data_unified`) is only **70% accurate**. Bulk
data-onboarding waves masquerade as IPOs — 451 symbols start on 2005-01-03, 15 on 2025-05-26
including **ABB, listed in the 1990s** — and pre-listing junk rows sit on reused tickers
(DELHIVERY carries 8 rows at ₹5–11 from 2016 before its real ₹536 listing: a **93× jump inside
what a base window would measure**). A vetted table now exists at
`research/153_ipo_base/results/listing_dates.csv` (1,293 accepted listings, 2006–2026) and was
**validated before any backtest ran: 48/48 known NSE IPOs accepted, date exact to ±3 days for
47/48, 0/12 known onboardings leaked.** Reusable by any future study that needs listing dates.

**Adopted spec (IPO-Base MID):** listed within 6 months · 25-day base, depth ≤ 30% · buy-stop at
the base high · −8% close stop · exit below SMA-20 · **+25% take-profit** · 8 slots @ 18.75% ·
no market gate. Standalone, 30 seeds, after tax and 25 bps a side, 2006→Sep-2026:
**31.03% CAGR [28.82..33.44], worst seed 28.82%, −20.88% drawdown, Calmar 1.50**, 32.6 trades a
year at +4.89% per trade net.

**As a third sleeve at 20% beside True North + Open Alpha: +1.13pp CAGR, −3.63pp drawdown,
+0.56 Calmar** (27.14 / −16.42 / 1.65 → **28.27 / −12.79 / 2.21**), correlation **0.16** daily
to OA and **0.18** to TN — lower than OA↔TN at 0.42 — and it beats a plain-cash sleeve at the
same weight by 5.60pp of CAGR. **Every leg of the pre-registered bar is met with room.** On the
common 2015+ window it beats r/147's gold on return at comparable Calmar: gold buys Calmar by
*lowering* return, this buys it while *raising* return.

### ⏳ PENDING — Arun's adoption call on IPO-Base MID at 10–20% of the book
Nothing was deployed. If adopted, the next step is a **G5 paper soak with a pre-registered fill
criterion** (modeled vs actual fill within 0.5% of the pivot, miss rate < 15%) and a dated
review — because **the entire edge lives in getting filled AT the pivot**: filling at the
signal-day close instead costs **−14.08pp of CAGR and loses on 30 of 30 paired seeds**.
Registered in the Ops Centre for 2026-10-15.

### ⏳ PENDING — send the site's IPO-Base panel dials + claimed numbers when legible
No replication gate was run (the screenshots did not come through). The engine is built so the
gate is a one-command run. Note in advance that two dials the site exposes — *Trail 30-week*
and *Breakout close* — are the **worst** settings we tested, so their published figures cannot
be assumed comparable.

### ⏳ PENDING — fold IPO into the four-sleeve study already owed from r/152
The exploratory cell 40% OA / 40% TN / 10% gold / 10% IPO scored **29.05% / −11.55% /
Calmar 2.52** on 2015+. That is one un-swept cell, not a finding. It joins the r/152 four-sleeve
question under the same Ops Centre review (2026-11-30) and must be run with a **gold-only null**
— the real question is what each candidate adds *on top of gold*.

**Operator caveats to carry forward:** the book earned only the idle-cash yield in **2013 and
2014** (no trades at all — the Indian IPO pipeline supplied 8–17 usable listings a year in
2012–14 against 80–182 in 2021–25); 2020–2026 supplies much of the record; capacity is
comfortable to about a ₹10 cr portfolio and binds near ₹50 cr.

Hand-off for the follow-on correlation study (r/154) is written:
`research/153_ipo_base/results/ipo_equity_seeds.csv` (30 seeds, daily, after-tax, cash 5%)
+ `results/ipo_adopted_spec.json`.

## ✅ 2026-09-05 — research/152 Multi-Year Breakout: SIGNAL, NOT ADOPTED (screen itself = an Open Alpha duplicate)

Full verdict: `research/152_multiyear_breakout/results/RESULTS.md` · study page
`/app/backtest/multiyear-breakout-research152`.

Three separate answers to bananapatterns.com's "Multi-Year Breakout" screen:

1. **As published (any multi-year high) — KILL, it is Open Alpha.** 76-93% of its signals
   ARE OA signals; it captures 62-92% of everything OA fires on. Running it = running OA twice.
2. **The "multi-year" quality itself — NO EDGE.** Requiring the ceiling to have stood ≥6 /
   ≥12 months halves CAGR (22.8 → 12.9 → 11.1%) with Calmar flat. It is de-levering dressed
   as a chart pattern. Closed question — do not re-test.
3. **The distinctive residual — a real SIGNAL that is still not adopted.** A 3-year high that
   is NOT an all-time high, trail-15: **23.45% CAGR [21.74..25.37] / −25.3% DD / Calmar 0.93**
   after tax (30 seeds, 2010–2026), 1,929 trades, +3.78%/trade, robust to deleting its ten
   best trades. Holding overlap with OA only **3.8-4.4%** — different stocks, same factor
   (corr 0.43 daily / 0.53 monthly). Passes 4 of 5 pre-registered complement conditions and
   **fails the correlation leg**; on a like-for-like window **gold beats it as the third
   sleeve** (+0.282 vs +0.240 paired Calmar at 10% weight, at ~zero correlation).

**PENDING — highest-value follow-up: a pre-registered FOUR-SLEEVE study (TN / OA / GOLD / MYB).**
The exploratory probe (2015+, 30 paths, NOT pre-registered) put 80% TN+OA / 10% gold / 10% MYB
at 28.81% / −11.54% / **Calmar 2.43** (+0.628 paired, 30/30 paths) vs 2.08 for gold alone and
2.03 for MYB alone — they fail in different windows. That number must be re-earned under a
pre-registered weight grid and bar before it means anything. Registered in the Ops Centre for
2026-11-30.

Hand-off for the follow-on correlation study (r/154) is written:
`research/152_multiyear_breakout/results/myb_equity_seeds.csv` (30 seeds, daily, after-tax,
cash 5%) + `results/myb_adopted_spec.json`.

## ✅ 2026-09-02 — BananaPatterns replication (research/142): Phases 1+2 DONE — rules decoded & reproduced, published returns NOT

Full verdict: `research/142_bananapatterns_replication/results/RESULTS.md`. Engine decoded
(close>ATH-close trigger, at-pivot fill, IBD-RS≥70, ₹5cr TV floor, −8% close-stop, 50-SMA
close-trail, 8 slots). Honest full-universe replica: 6.5–15.7× across selection paths vs
their 33.74×; their −11.4% worst-fall unreachable (best −22% daily / −15.3% monthly).
**PHASE 3 (G3) DONE same night — verdict STRATEGY (candidate), PUBLISHED to
`/app/backtest/bluesky-ath-breakout-research142`** (tearsheet + vs-indices chart +
full caveats). 2006-25 net/real-fills 10-seed ensembles: headline config D (gate ON +
mcap≥₹500cr PIT proxy) median **30.4% CAGR [27.9-34.4] / −31.5% DD / ~203×** vs
NIFTYBEES 12.3%/−59.7%; converges with research/75 momentum (31.9%/−31.6%).
**PHASE 4 (optimization) also DONE 2026-09-02:** 48-cell sweep (6 axes + combos +
sizing×slots, 10-seed ensembles each) — no adoptable betterment except THE CORRECTION:
mcap-floor "risk filter" claim was an incomplete-snapshot artifact (925→2,042 symbols);
full data shows floor = pure return drag → **no-floor book is the headline: 517× /
35.3% / −36.9% (2006→Aug 2026 medians)**. Trail-15/20 spike rejected on plateau; slots
inert (sizing binds ~5 positions); adaptive mcap switching moot. Correction published
on the study page. Churn: BlueSky ~4.5× book/yr all-STCG vs momentum 0.38×/yr with
LTCG → momentum keeps 2-4pp/yr after-tax edge.

**PHASE 5 (2026-09-02, final): HARNESS BUG found in the sweep (trail-SMA NaN-poisoned →
trail exits disabled → inflated numbers); all decisions re-made on corrected engine +
Arun's NET-OF-TAX gate (STCG modelled in-sim via `--stcg`).** Locked outcomes:
(a) ADOPTED SPEC = decoded rules, no mcap floor, gate ON, stop KEPT, trail 50:
301×/31.8%/−45.7% pre-tax; (b) TAXABLE PICK = trail-20 variant: after-tax 28.0% @
−33.4% beats trail-50's 25.7% @ −47.8% (tax scales with gains — ranking survives);
(c) CAPSTONE = 50-50 monthly-rebalanced blend with r/75 momentum: 33.0% @ −27.5%,
beats both legs (corr 0.29/0.52) — best construction in the study; (d) stop-off and
mcap-floor rejections confirmed on corrected engine. All published in one consolidated
update (commit 3f717e2).

**G5 PAPER BOOK LIVE (2026-09-02, Arun: "go"):** `services/bluesky_paper.py` — ₹10L EOD
soak of the adopted trail-20 spec; cron 18:40 IST; dashboard `/app/bluesky-paper`;
Strategies index row; ops-center dated review **2026-12-05** (pass criterion
pre-registered in `research/142_.../BLUESKY_PAPER_DAILY_RUN_STATUS.md`). Study page also
gained the banana-style full trade book embed (1,082 trades, median seed). Remaining:
watch first runs this week; momentum-leg tax model for the blend.

**OPEN ALPHA FULL RESTUDY — REGISTERED for ~2026-12-12 (Arun 2026-09-03: "a complete
restudy and reassessment of the system"):** joint gate × entry × exit/SL combo
optimization + 16-slot sizing, AFTER the Dec-5 soak review (live fills inform entry
modeling). Context: the 2026-09-03 gate bake-off (research/142
GATE_BAKEOFF_DAILY_SWEEP_STATUS.md) found the old backtest gate NaN-disabled since
Apr-2026 (phantom 2026-01-15 holiday rows), refuted SMA200, and identified DD10
(block only >10% below 252d high) + 16 slots @6.25% (seed spread 6.7×→2.5×, worst
seed 30.4→32.5% CAGR) + 50-50 momentum blend (35.7% CAGR, −22.2% DD) as the candidate
package. Interim decisions PENDING from Arun: adopt DD10? 16 slots? phantom-row purge
(classifier-blocked, needs explicit go)? Then: re-seed OA book (current 5 positions
are spec-invalid gate-WEAK entries), restore ₹2.5L deposits + re-anchor dividend HWM,
update live engine + study page. Ops-registry REVIEWS entry dated 2026-12-12.

**DIVIDEND ENGINE LIVE (2026-09-03, Arun: "yes this is good, build for both"):**
Quarterly HWM dividend policy on BOTH sleeves — adopted spec = research/142
`dividend_sim_v2.py` variant E: **25% of new profit above the flow-adjusted HWM;
payout capped at last dividend +7.5%/qtr (smooth stepping income line); surplus →
liquid equalization reserve (~6% p.a.) that bridges dry quarters; honest cut+rebase
if the reserve empties; capital never invaded; positions never force-sold** (outflow
clipped at cash+CASHIETF). Build: `services/dividend_engine.py` (book adapters +
declaration math), `scripts/dividend_declare.py` cron **19:15 Mon-Fri** (idempotent,
acts ≤12 days after quarter end), notices via `services/dividend_notify.py`
(registrar-style email + WhatsApp both DORMANT until .env keys; desktop alert live),
`/api/sleeves/dividends` + preview endpoint, Dividends card on /app/sleeves,
ops-center GROUP + review dated **2026-10-01** (verify first declaration).
HWMs anchored at adoption: TN ₹9,37,525 (contributed — book underwater, correctly
pays nothing until recovery), OA ₹9,17,628 (NAV at adoption — backfilled 2020-26
history is capital, never distributable). ALSO FIXED: True North deposit/withdraw
endpoints didn't exist (Sleeves portal TN legs 404'd) — added
`/api/sleeves/truenorth/deposit|withdraw` writing cash/capital/fund_flows in
mp_state; Sleeves.tsx repointed. 10-yr sim tables:
`research/142_.../results/dividend_sim_v2_*.csv`.

**GO-LIVE ARCHITECTURE (decided with Arun 2026-09-02, build gated on the Dec-5 soak):**
books stay SEPARATE (own engines/state/kill-switches); a **Sleeve Allocator** layer sits
on top using unit-NAV accounting (deposits buy units at NAV per target split, withdrawals
redeem pro-rata; sleeve-level `adjust_capital` API per engine; monthly rebalance between
sleeves with ±10% bands). Combined REPORT layer built now: **/app/sleeves** (live 50-50
blend curve vs both legs + NIFTYBEES, correlation, DD tiles). Live entry mechanic locked:
evening AMO pivot buy-stops on all credible names → first-past-the-post → watcher cancels
on book-full (fill-at-close variant REFUTED: 536×→14.4× — the edge lives in the entry
price). **Standing infra item (independent):
full-DB split-adjustment repair — remaining scale-broken symbols beyond study set (was
72/1,666; extend_universe.py fixed those it touched; POCL-class stragglers remain).**

bananapatterns.com claims 64.5% CAGR (PROVISIONAL, 2020-25) on an ATH-breakout screen.
**Phase 1 (trade-level match, Arun's gate) PASSED** — see
`research/142_bananapatterns_replication/results/RESULTS.md`: exits reproduced 22/23 to
exact day+price (rules inferred: −8% stop on CLOSE with gap fills; 50-DMA trail booked AT
the signal close — optimistic), entries ≈ recent swing-high pivots within ~1% (at-pivot
buy-stop; some fills fantasy where the day opened above the pivot). Study lives on VPS +
laptop (SFTP'd; laptop copy is source — NOTE laptop project folder is NOT a git repo,
fix eventually). **Phase 2 pending:** (1) Arun re-runs the site backtest with "Blue sky"
selected and shares its trade table (screenshots were VCP-screen); (2) fix our
market_data.db split-adjustment defect (MCX/HEG/NAZARA/SMLMAH/MUFIN/KFINTECH old rows
unadjusted; CUPID 5× scale — affects ALL ATH/52wk screens, not just this study);
(3) extend universe (11/35 trade symbols absent, incl. big winner E2E); (4) full
2020-25 replication + controls (next-open trail fills, fill realism, costs,
survivorship, super-winner guard, 200DMA gate).

**UPDATE same evening — Phase 1b DONE, engine fully solved:** Blue-sky ground truth
(51 trades) validated — exits 37/39 exact; **entry pivot = ALL-TIME-HIGH CLOSE**
(buy == prior ATH-close to the paisa on ~35/51); study symbols repaired/downloaded on
VPS (`repair_data.py --apply`, backup in `market_data_unified_bak142`; E2E + BONDADA
unavailable from Kite/NSE; POCL still 2.5× scale — add to full repair). **NEW STANDING
ITEM: full-DB split-adjustment repair — 72/1,666 daily symbols have suspected
unadjusted splits/bonuses** (review demergers before deleting; script pattern in
research/142/scripts/repair_data.py). Phase-2 build next: point-in-time liquidity-floor
universe (ALL stocks passing mcap ≥₹500cr + ₹5cr/day, not just their picks), RS-formula
inference, 8-slot selection rule, then full 2020-25 replication + controls.

## ⏳ 2026-09-01 — Nifty CSL: DTE-0 book selected; TIME-GRID RUN SHEET (11 AlgoTest runs) + restart pending

**Where the study stands (see STATUS doc §0d):** all 14 NIFTY stops (10→300%) run and validated;
the stop is NOT an optimisable parameter (paired |t|<1.5 everywhere; risk monotonic, return not).
The book is **DTE-0 (expiry day) only** — 12/14 stops t≥3.0, 6/6 years positive, IS t=2.91 /
OOS t=2.59. DTE-1 DROPPED (OOS-negative). **Tradeability gate now binding: WR≥45% & losing
streak≤7** (rejects 10-30% stops — median trade negative). **BOOK: DTE-0 @ 60%** — ₹23.69L /
294 tr / WR 63.6% / median +₹8,927 / MaxDD −₹1.52L / streak 6 / t 4.28 / OOS t 2.24 @10 lots.
**SENSEX confirms on a 2nd index** (30SL/60SL, 814 tr, 2023-26): DTE-0 @30% = ₹14.26L / 172 tr /
t 3.61 / Net-DD 13.8 PASSES; every other SENSEX DTE rejects; paired 30-vs-60 not significant.

**☐ THE 11 TIME-GRID RUNS (Arun runs in AlgoTest; tracker rows 42-52 in CSL_TRACKER_v5.csv).**
Fixed: NIFTY ATM weekly straddle SELL 10 lots, SL 60% both legs Percent, Partial, trail-BE ON,
brokerage ON, slippage 0.5, DTE chip CLEARED (DTE split done offline — DTE-1 grid comes free),
2021-01-01→2026-08-31, export "Download trades" CSV into ~/Downloads with EXACT filename:

1. ☐ entry 09:25 → `algotest_entry0925.csv`   (P1 — most likely improvement)
2. ☐ entry 09:30 → `algotest_entry0930.csv`   (P1)
3. ☐ exit 15:00 → `algotest_exit1500.csv`     (P1 — dodge closing gamma)
4. ☐ exit 14:30 → `algotest_exit1430.csv`     (P1)
5. ☐ entry 09:20 → `algotest_entry0920.csv`   (P2 — plateau check around 09:16)
6. ☐ entry 09:45 → `algotest_entry0945.csv`   (P2)
7. ☐ entry 10:00 → `algotest_entry1000.csv`   (P2)
8. ☐ exit 14:00 → `algotest_exit1400.csv`     (P2)
9. ☐ exit 15:25 → `algotest_exit1525.csv`     (P2)
10. ☐ entry 10:30 → `algotest_entry1030.csv`  (P3)
11. ☐ entry 11:00 → `algotest_entry1100.csv`  (P3)

STOP after 1-4 if none beats baseline (DTE-0 net ₹23,69,304) by >₹1.5L — that's inside noise.
Analyser is pre-built & smoke-tested: `research/136_nifty_csl_portfolio/scripts/analyse_timegrid.py`
(auto-detects the files, reports DTE-0/1/2 side-by-side with both gates). DTE-1 revisit is
PRE-REGISTERED with a higher bar (both gates + plateau of ≥2 adjacent times + OOS positive alone).

**☐ RESTART PENDING (backend): Straddle Intraday Study deployed to VPS but gunicorn not yet
restarted** — Claude's restart was blocked by permission gate. Everything else is DONE and
COMMITTED (`6068d8a`, pushed): 16 runs / 21,172 trades in `backtest_data/algotest_studies.db`
(loader `scripts/load_algotest_studies.py`), query API `services/straddle_study_api.py`
(`/api/straddle-study/runs|query` — filter by index/SL/DTE/year-range/events, cost model &
lots-scale query-time, rank by net/WR/Calmar/Net-DD/PF/t/median/streak), React page
`/app/straddle-study` ("Straddle Study" in sidebar Options section), frontend BUILT on VPS.
To finish: `ssh arun@94.136.185.54 'sudo systemctl restart quantifyd'` (after-15:40 rule —
verified 16:30 IST when staged). Page renders now; API 404s until restart. When new AlgoTest
CSVs land, re-run: `python3 scripts/load_algotest_studies.py backtest_data/algotest_csv` on VPS
(scp new files there first).

## ⏳ 2026-08-31 — Nifty CSL (09:16 ATM straddle) → per-DTE configs → AlgoTest Portfolio [research/136]

Arun ran 10 AlgoTest backtests on "Nifty CSL" (NIFTY weekly ATM short straddle, 10 lots / Qty 650,
09:16 entry, 14:30 exit, 2021-08-31→2026-08-31) sweeping the per-leg stop 300→250→200→150→100→
75→60→50→30, plus one 15:15-exit run at SL 30. All 10 decoded (PDFs are image-only — page
images pulled via pypdf+PIL and read at native 3417×5280).

**Findings so far:** SL 30 @ 15:15 is best on P&L (₹25.75L) and Return/MaxDD (3.69); SL 30 @ 14:30
best on expectancy (0.30) with MaxDD only −₹1.31L. Wide stops (150-300) made NOTHING in 2021-2023
— all their profit is 2024-26 (regime-dependent, treat as failed). SL 60 is the most year-stable
(₹2.4-4.7L every year) → portfolio stability sleeve.

**Problems that must be fixed before any conclusion:** (a) SL 30 is the EDGE of the tested grid —
a boundary optimum, need 25/20/15/10; (b) "Include Brokerage" was OFF in all 10 runs; (c) CE leg
stop is Percent(%) while PE leg stop is Points(Pts) in every run; (d) the DTE filter chip changed
35→3→2 but trade count stayed 522 every time — **no run isolates a DTE**, so "DTE 1 and 2 work
best" is unverified; (e) no margin figure → no ROI/CAGR exists yet; (f) Partial-vs-Complete and
trail-to-breakeven were never isolated (run 1 vs run 2 changed 4 things at once).

**Next:** work the 64-row tracker at `research/136_nifty_csl_portfolio/results/CSL_TRACKER.csv`
in gate order — row 21 (brokerage ON) → 13 (SL 20) → 11/12/14 → 16/17 → 18/19 → 24-29 (DTE
isolation, the real question) → 31-39 (time grid) → 40-50 (re-entry / redeploy-new-ATM / basket
stops / strangle) → 51-59 (VIX, weekday, IS-OOS, margin) → 60-64 (AlgoTest Portfolio assembly).
Status doc: `research/136_nifty_csl_portfolio/NIFTY_CSL_ATM_STRADDLE_INTRADAY_SWEEP_STATUS.md`.

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

## ★ DECISION PENDING — research/94: NWV → jade lizard / iron condor automation — 2026-07-27
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

## ★ PENDING DECISION — research/86 HA 2-green-no-wick 30m LONG: build the G5 paper book? — 2026-07-20
**STRATEGY CANDIDATE — the first full survivor of the r/81-86 program** (IS t3.7 → Val t6.0 →
OOS t3.7 PASSED; OOS book 11.6% CAGR vs bench 5.6%, DD −11%, Calmar 1.03, beat bench all 3 OOS
years incl. the 2026 down-tape). OOS consumed. Watch-item: per-trade fade 47→36→25bps across
splits. NEXT: G5 paper book — construction choice needed (cash-CNC sleeves vs futures subset;
fractional per-name sizing is the practical question). Verdict: `research/86_heikin_patterns/results/RESULTS.md`.

## ⏰ REMINDER ~2026-09-15 — research/111 paper-verdict checkpoint
After ~4-6 weeks of CSL paper data (books live since 2026-08-14, frozen 13-AUG config):
compare paper vs in-sample expectations, re-run the weight scan (deliverable3_portfolio.py)
with real CSL streams, decide STRATEGY-upgrade / re-freeze / kill. Books: csl_paper_state.json.

## ✅ research/111 FINAL DELIVERABLES (user spec 2026-08-13) — ALL FOUR DELIVERED 13-AUG
1. **CSL best config per index (NIFTY & SENSEX separately)**: per-DTE sweet spots for the
   combined-SL **plus optimized ENTRY time and EXIT time** (drop the 9:16-only / hold-to-EOD
   assumption). Engine: entry×exit×SL×DTE sweep on 3-sec dwell data (1-min fallback per day,
   report n-days per resolution + live-first rule).
2. **Comparison vs existing NAS systems** — live AND backtested, NIFTY + SENSEX, individual
   systems AND paired portfolios (NIFTY NAS + NIFTY CSL; SENSEX NAS vs SENSEX CSL), with
   VISUAL comparisons (equity + DD curves).
3. **Best portfolio configurations** (weights/day-scheduling across the sleeves).
4. **One traversable hub** (page/section) linking all tables + tearsheets/charts, lots+days
   stated everywhere, ending in a CERTAIN conclusion. Findings so far live in
   research/111_sensex_manual_mgmt/ STATUS + results/.

## ★ NEXT UP (queued 2026-08-12, user-accepted order) — two straddle follow-ons

**① SENSEX manual-trade → automated system (research/NN, new).** Reconstruct Arun's 2026-08-12
manual SENSEX options trade vs price action — entry was post-ATR-squeeze; the REAL focus is the
management after a 30% SL breach on one leg (range expanded, then closed after confirmed bullish
moves). Phases: (1) reconstruct+narrate the trade timeline vs SENSEX/CPR/ATR/BB; (2) codify the
management state-machine (entry → per-leg 30% SL → on breach expand range → exit on defined
"confirmed bullish move"); (3) entry-condition bake-off: 09:16 fixed / ATR squeeze / ATR+BB
squeeze / time-based / staggered time-based; (4) SENSEX theta-decay sweet spot by DTE+time (reuse
the NIFTY EOD-decay treatment); (5) report page + factsheet per playbook. BLOCKED ON: Arun's
trade fills (Kite Console export / paste — Kite MCP unauthenticated) + verify SENSEX chain depth
in options_data.db (looked NIFTY-only; research/103 backfilled SENSEX 1-min underlying only).

**② V1+30%-SL straddle vs NAS-ATM 30%-SL systems — portfolio report.** Compare the new SL30
system (leaderboard #1, Calmar 6.2, 79 days, 10 lots) against every NAS ATM variant that runs a
30% per-leg SL. KEY NUANCE: straddle SL is on COMBINED premium; NAS is 30% PER LEG (per-leg fires
more often on one-sided spikes) — normalize and state clearly. One board: net/mean/win/MaxDD/
Calmar/SL-hit at stated lots, then correlation → pick best or combine into an efficient blended
book with combined equity curve. Report page like the others. Data: NAS paper/live records on VPS.

## ✅ 2026-08-12 — /app/straddles: SL30 system + leaderboard + variant lab (commits 4fe808c, b0b94d6)
Built from Arun's observation that a 30% combined-premium SL is rarely hit (verified: not hit 90%
of days, 25% hit on DTE0 where it caps the +73–186% gamma spikes). V1 entry + 30% SL backtest over
the recorded chain: +₹7.56L / 79 days / win 72% / SL-hit 9% / maxDD −₹1.22L at 10 lots — debuts
**#1 on the Strategy Leaderboard (Calmar 6.2)**. Page adds: leaderboard (A–F grades, hyperlinked
rows that scroll to each system's section), V2 variant lab (naked-vs-ironfly × stop sweep — calm-
regime finding: wide/no stop beats tight stops, but naked tail is unbounded), SL30 card (stats,
intraday + cumulative charts, by-DTE with mean+DD, lots always stated), exit-price column. All
regenerated by the daily 15:40 post-close cron. Opt-Study page also got NIFTY candles+CPR,
EOD-decay-by-DTE, sparkline wall (commits d5fe0d1…2d37dfe).

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

### research/151 — BananaPatterns "VCP" screen — DONE 2026-09-05, verdict NO EDGE

- Replication gate PARTIAL (62.2% joint match). Their exit engine reproduces 31/32 ground-truth trades exactly; their entry pivot is an exact prior close but carries no volatility-contraction structure, and no fixed lookback can fit it.
- Published claim (25.99x / +72.1% CAGR / -14.8% worst fall) REFUTED: 32.4% CAGR [6.5..61.6] at -34.5% on their own dials, after tax and costs, 30 seeds.
- Killed by its own null control: shrinking the pivot lookback toward no-pattern-at-all monotonically improves the book, so the screen subtracts value.
- Portfolio: corr 0.749 to the live Open Alpha book (bar <0.40); best blend weight adds +0.033 Calmar (bar +0.10) and loses to a plain cash sleeve at the same weight.
- Published at `/app/backtest/vcp-breakout-research151`. Deliverables for study r/154 in place: `research/151_vcp_breakout/results/vcp_equity_seeds.csv` (30 after-tax daily curves) and `vcp_adopted_spec.json`.
- Dated obligation registered in the Ops & Review Centre: re-open only on a published, reproducible VCP definition (due 2027-03-05).

### research/154 — Six-sleeve correlation & blend matrix — DONE 2026-09-05, verdict STRATEGY (candidate)

- **The deployed TN+OA pair's true 2008 drawdown is −16.5%, not −2.4%.** r/146 and r/151
  measured the 2008 window from 2008-01-01, which is after the Dec-2007 peak. 2008 is the
  pair's single deepest hole in twenty years. The standing claim that the TN gate plus OA's
  stops "already stripped the crash tail" is **withdrawn**.
  → **PENDING:** re-audit every per-window drawdown figure in r/146 through r/153 for the same
  window-start artefact.
- **VCP is Open Alpha.** 87.0% of OA's signals are VCP signals; 48.6% / 41.5% holding-day
  overlap; correlation 0.749 daily. MYB shares 90.2% of its signals with VCP. Both are retired
  from consideration permanently.
- **OA and IPO have never once held the same stock on the same day** (0.0% signal and 0.0%
  holding-day overlap, 2010–2026), at correlation 0.211 daily. Gold is ~0 to everything.
- **197 of 1,767 enumerated weight vectors** clear the pre-registered bar on all three panels
  against three nulls — a contiguous plateau. Recommended (constrained):
  **OA 40 / TN 25 / IPO 20 / GOLD 15 → 28.21% / −10.77% / Calmar 2.61** vs the pair's
  27.74% / −17.01% / 1.68 on 2006-04→2026-08.
  Deployable today without an unproven sleeve: **OA 60 / TN 15 / GOLD 25 → 28.02% / −13.31% /
  2.095**. r/147's 45/45/10 is NOT admitted (CAGR shortfall).
- **IPO is 80% cash** (19.6% invested; zero trades in 2013 and 2014). A cash null does not
  catch that, so a **beta-matched null** was built (IPO → 19.6% OA + 80.4% cash). Beyond ~20%
  IPO weight the extra Calmar is indistinguishable from de-levering on two of three panels.
- **Both r/152 open questions answered:** MYB+OA reproduces but is not actionable (2008 is
  unreachable by construction for a 3-year-high screen); the 80/10/10 four-sleeve probe is
  **REFUTED** against a gold-only null at the same satellite weight.
- **Data defect fixed:** r/147's gold-INR reference series was missing 40 of 274 months.
  Rebuilt at daily resolution, zero gaps, monthly correlation to real GOLDBEES 0.878.
  Lives in `research/154_multi_system_blends/results/gold_nav.csv` — never in market_data.db.
- Published at `/app/backtest/multi-system-blends-research154`.
- **ARUN DECIDES.** Nothing deployed; no live engine, crontab or spec touched.
  Dated obligation registered in the Ops & Review Centre (due 2026-10-15, merged with the
  r/153 adoption call; the r/152 four-sleeve review is marked DONE by this study).

## Done - 2026-09-05 - research/155 IPO idle-cash redeployment

- **CONCLUDED - the idle cash stays in cash.** Tested Arun's proposal to park the IPO sleeve's
  idle cash in Open Alpha / True North during listing droughts and pull it back when supply
  returns, with every pull-back friction modelled (25/40/60 bps both ways, tax on the realised
  gain with FY netting, T+1 settlement, pro-rata/LIFO/FIFO lot policy). The premise is
  CONFIRMED and the mechanism WORKS (0 missed entries in 20 years), but it can only touch 2.7%
  of the portfolio and buys +0.105pp CAGR / +0.006 Calmar - gone by 40 bps. Continuous
  redeployment costs 0.375 of Calmar and takes the sleeve's correlation to Open Alpha from
  0.21 to 0.90. A plain static TN35/OA35/IPO30 beats the whole mechanism.
  Published: `/app/backtest/ipo-idle-cash-redeployment-research155`.
  Nothing deployed; research/153's spec unchanged.
- **Dated review registered:** 31-Mar-2027 - revisit only if the IPO sleeve's weight exceeds
  30% or the pipeline has been in drought for more than 12 consecutive months.

- [ ] CSL-60 DTE-0 paper book: publish /app/backtest factsheet entry for the r/136 study (strategies.ts studyGap notes it); paper-soak review 2026-11-30 (Ops Center) — added 2026-09-07
