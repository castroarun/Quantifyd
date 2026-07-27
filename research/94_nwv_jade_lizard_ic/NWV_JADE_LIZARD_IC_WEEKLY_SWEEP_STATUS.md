# NWV Weekly View → Jade Lizard / Iron Condor Structure Bake-off — Real Option EOD 2020–2026

STATUS: DONE (2026-07-27) — verdict in `results/RESULTS.md`: NO EDGE for the
directional JL/IC mapping; weak SIGNAL (t≈2.2) for NEUTRAL-week far-OTM premium
selling; the user's exact construction ≈ breakeven over 2020-26.

## 2. The Ask

**What you asked (2026-07-27):** "Nifty weekly setup — lets move this further, and
automate… using jade lizards/iron condors…" followed by a screenshot of the live
manual position for this week's BULLISH view (NIFTY 4-Aug, 10 lots): LONG 22900 PE /
SHORT 23450 PE / SHORT 24500 CE / LONG 24700 CE — "we need to construct like so."

**What we're actually testing:** Across every NWV-view week 2020-02 → 2026-07
(replaying the live Phase-0 engine), which weekly credit structure — pivot-anchored
jade lizard / asymmetric condor (the user's construction) vs the fixed-offset iron
condors locked in the June study vs the incumbent debit spread — has the best
net-of-cost expectancy, tail, and per-year stability, per view bucket
(BEAR = bearish+neutral_to_bearish, BULL = bullish+neutral_to_bullish, NEUTRAL)?
The winner becomes the construction rule for the Phase-1 automated executor
(paper-first).

Upgrade vs the June study (`docs/NWV-PHASE1-TRADE-PLAN.md` §10–14): that study had
only 21 real directional weeks (2024-03+) and used modeled Black-Scholes for the
regime read (22-pt error). `nse_options_bhav` now holds REAL NIFTY option EOD
2016→2026-07-21 with weeklies traded throughout → the whole multi-regime window
runs on real prices. No modeled arm needed.

## 3. The Base — what's being tested

- **View source:** replay of the live engine (`services/nwv_engine.py`) via
  `research/nwv_phase1_regime.build_view` — weekly CPR width bucket × Monday first
  30-min candle vs CPR × gap dampener × monthly override. Views bucketed:
  BEAR (bearish, neutral_to_bearish), BULL (bullish, neutral_to_bullish),
  NEUTRAL (neutral). `ignore` weeks skipped (locked Phase-0 rule), counted.
- **Entry:** Monday EOD close prices (same as June study, comparability).
- **Expiry:** next-week (first expiry ≥6 calendar days out; ≈8 days — matches the
  user's live 4-Aug position entered 27-Jul and the Phase-1 doc rationale).
- **Structures** (all strikes from real traded chain; spot = Monday first-30m close,
  weekly pivots from prior-week H/L/C):
  - `debit` — ATM/200 debit spread, side-aligned (incumbent reference, hold + PT60)
  - `ic_neutral` — short ±250, 200 wings (100-strike)
  - `ic_lock-50` — the LOCKED 2026-06-01 slightly-bearish IC: short call +200, short put −300
  - `ic_bull` — bull-skew: short put −100, short call +400
  - `jl_fix250` — fixed-offset TRUE jade lizard: naked short put −250 + call credit spread +250/+450
  - `pv_userJL` — **the user's construction**: short put @ floor50(S1), long put 550 lower,
    short call @ floor50(R2), long call +200 (put-heavy asymmetric condor)
  - `pv_trueJL` — same anchors, NO long put (naked S1 put + R2/+200 call spread)
  - `pv_condorR1` — same put side, call side tighter at ceil50(R1)/+200
  - `pv_bearMir` — bear mirror: short call @ ceil50(R1) + 550 wing, short put @ floor50(S2) − 200 wing
- **Exits (credit structures):** checked at each daily EOD Mon→Fri:
  `pt50_stop1x` (TP at 50% of credit, stop at −1× credit — June-study incumbent),
  `pt50_stop2x` (stop −2× — r/90 found looser premium stops survive real fills),
  `hold` (Friday EOD, no management). Stop/TP fills at that day's EOD MTM —
  pessimistic under gaps (fill can be far worse than the trigger level).
- **Costs:** 0.5 pt per leg per side (entry+exit legs actually traded) — covers
  spread+slippage on liquid NIFTY weeklies; reported net.
- **Liquidity (BINDING r/89 rule):** every leg must show real traded contracts > 0
  on entry day, else the week is skipped (counted).
- **Sizing:** LOT 65 × 10 lots (matches the user's live position; June tables were 5 lots — halve for comparison).
- **Period:** 2020-02-03 → 2026-07-13 Mondays (30-min spot coverage bound).
- **Success criterion:** rank by net avg/week within each view bucket, gated on
  PF > 1.15, worst-week tail no worse than incumbent, and ≥4/7 years positive.
  A structure must beat the bucket's incumbent (BULL: debit spread; BEAR: ic_lock-50)
  to displace it in the executor.

## 4. Plan — variant grid

- 3 view buckets × 9 structures × 3 exit policies (debit: 2 policies) ≈ **~85 cells**,
  each over ~330 replayed weeks (real fills), single process, expected runtime < 5 min.
- Pre-req data fix: NIFTY50 30-min ends 2026-05-05 → derive 30-min bars from the
  5-min series (complete days only, through ~2026-07-16) so the replay covers May–Jul.
- Known skips: `ignore` weeks (no-trade rule), weeks with any untraded leg,
  weeks with <2 sessions.

## 5. Status (live log)

| Date/time | Event | Notes |
|---|---|---|
| 2026-07-27 10:4x IST | Study set up; STATUS written before launch | folder research/94_nwv_jade_lizard_ic |
| 2026-07-27 10:4x IST | 30-min derivation + sweep launching on VPS | |
| 2026-07-27 10:42 IST | 30-min gap filled from 5-min (572 bars, complete days → 2026-07-16) | idempotent script |
| 2026-07-27 10:44 IST | First launch failed (system python3 lacks dotenv) | relaunched with venv/bin/python3 |
| 2026-07-27 10:46 IST | Sweep DONE (<1 min): 318 Mondays, 128 ignore, BULL 40 / BEAR 36 / NEU 112 | results CSVs written |
| 2026-07-27 10:5x IST | Always-on control + t-stats run; RESULTS.md written | best cell t 2.22 (weak after ~90 cells) |
| 2026-07-27 11:0x IST | Arun picked paper mapping (b): his JL, all non-ignore weeks | via AskUserQuestion |
| 2026-07-27 11:1x IST | `services/nwv_trade.py` built, deployed, app.py registered, week seeded from live fills | live at next 09:00 restart |
| 2026-07-27 11:2x IST | Phase-2 adjustment sweep DONE: exits beat rolls; pivot exit-side t 2.48 | wired into executor (15:25 job) |
| 2026-07-27 11:3x IST | Phase-3 trigger-TF sweep: 30m monotonic best (t 3.10, worst −74.5k, maxDD −142k) | executor moved to 30-min pivot checks (:15/:45) |
| 2026-07-27 11:38 IST | **Book ACTIVATED intraday** via standalone one-day runner (Arun's ask) | first MTM +₹3,308; in-app jobs take over at next 09:00 restart |
| 2026-07-27 11:3x IST | S1 discrepancy explained: Zerodha chart weekly high differs ~39 pts | engine stays self-consistent (23,493.80) |
| 2026-07-27 11:5x IST | /app/nwv card BUILT on VPS (NwvPaperCard.tsx, bundle index-B4ev1EO_.js) | level watch + legs + MTM + history; API 404 until 09:00 restart (card explains) |

## 6. Crash Recovery (resume without Claude)

- All artifacts on VPS: `/home/arun/quantifyd/research/94_nwv_jade_lizard_ic/`
- Check progress: `tail -50 results/run.log`; results CSVs written at end of run
  (fast run — if `results/jl_ic_ranking.csv` exists, it finished).
- Re-run everything (idempotent):
  `cd /home/arun/quantifyd && python3 research/94_nwv_jade_lizard_ic/scripts/derive_30min_from_5min.py && python3 research/94_nwv_jade_lizard_ic/scripts/run_jl_ic_sweep.py | tee research/94_nwv_jade_lizard_ic/results/run.log`
- The derivation script DELETEs+reinserts only NIFTY50 30-min rows after
  2026-05-05 15:59 — safe to re-run; touches nothing else.
- Do NOT touch `backtest_data/market_data.db` tables other than described.

## 7. Files

| File | Purpose | Committable? |
|---|---|---|
| `NWV_JADE_LIZARD_IC_WEEKLY_SWEEP_STATUS.md` | this file | yes |
| `scripts/derive_30min_from_5min.py` | 30-min gap fill from 5-min | yes |
| `scripts/run_jl_ic_sweep.py` | replay + structure sweep | yes |
| `results/run.log` | run output | yes (small) |
| `results/jl_ic_ranking.csv` | per bucket×struct×exit summary | yes |
| `results/jl_ic_by_year.csv` | per-year stability | yes |
| `results/RESULTS.md` | final verdict | yes |

## 8. Findings

(pending run)
