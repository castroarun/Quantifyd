# Covered_Calls — TODO

Cross-session source of truth for pending work. Each item: what / why / when.

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

### ✅ DONE 2026-06-12 — Live stops/churn incident: 5 fixes + NAS Live Guardian
Fast NIFTY rally (23325→23530); the 6 live variants' protective stops malfunctioned (book
+₹6k→−₹9k). Contained via manual-freeze, fixed once-and-for-all, deployed + verified after
close, freeze cleared (user OK). Forensic + crash-recovery:
`research/60_nas_churn_incident_fix/NAS_STOPS_CHURN_FORENSIC_STATUS.md`. Commits
`0dbb1d0 9c8a63a a963de7 0856fea af05acd`.
- [x] **#1a SL-skip** — `check_and_handle_sl` now REST-fetches a leg's premium instead of silently
  `continue`-ing when the ticker's `live_ltps` lacked it (all 3 ATM executors).
- [x] **#1b squeeze SL poll** — `_nas_squeeze_sl_monitor` 10s REST backstop (squeeze was ticker-only).
- [x] **#3 churn cooldown ROOT CAUSE** — `_pt` date-formats didn't match `datetime.isoformat()`
  (T + microseconds) → the 15-min `reentry_cooldown_min` was silently blind. Added
  `%Y-%m-%dT%H:%M:%S.%f`. NB: this time-gate now BLOCKS the "ATM2 same-strike re-entry churn"
  symptom listed below; the "hold straddle + reset SL in place when strike unchanged" refinement
  there is now OPTIONAL (not required for safety).
- [x] **#2 ST survivor exit** — fired only on SuperTrend flip + only at 5-min candle close. Added
  level-breach (`latest_close>st_val`) + tick-level (cache st_val, exit on any tick premium>ST).
  999999 naked sentinel preserved.
- [x] **#4 additive subscriptions** — same-strike legs share ONE token; triangular exclusion sets let
  a variant unsubscribe a sibling's live leg → tick gap → fed #1. Replaced with
  `_tokens_in_use_by_others()` (excludes ALL sibling maps). Also reduces the "stale leg SL after
  ATM2 cascade" log-noise item below.
- [x] **NAS Live Guardian** `scripts/nas_live_guardian.py` + agent `.claude/agents/nas-live-guardian.md`
  + `/nas-guardian` command. 4 tiers: live health (ticker/token/per-leg subscription+SL coverage/
  naked-ST/freeze) · behavioural audit of today's REAL trades (churn, SL-breach-without-exit, P&L
  reconcile vs Kite) · regression self-test (all 5 fixes intact) · opt-in `--firedrill` sandbox that
  drives the REAL SL path on a throwaway DB. Runs the 5-min monitor itself + escalates. `7514a0a b2fb2a5`.


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
