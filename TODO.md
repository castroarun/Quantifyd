# Covered_Calls — TODO

Cross-session source of truth for pending work. Each item: what / why / when.

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
