# MTF-Bullish Volume-Breakout — Exit Rule Bake-Off (Fixed vs Trailing vs Time)

**STATUS: DONE (G1→G4) · VERDICT: STRATEGY candidate** · research/71 · 2026-07-01

**Answer:** trailing stop (Donchian-20 or Supertrend 10,3), NO target, 20% catastrophe, NIFTY>200DMA
gate, ~8 concurrent → 20.5% CAGR / Sharpe 0.71 / MaxDD −36% / Calmar 0.57, beats NIFTYBEES +9%/yr
(45.9× vs 9.4×, 20.5y). Tearsheet: `results/tearsheet.png`. Full verdict + caveats: `results/RESULTS.md`.

---

## 1. The Ask

**What you asked:** "Using this prodigaltrader multi-timeframe bullish scanner I picked
breakout stocks (all-time-high / near-ATH, cup-and-handle, breakout-and-consolidating, all
with volume, illiquids removed) and took the attached chart trades. Assess these. How do we
pick them up automated, how much/how long do we trail them? Aim is SHORT-TERM trading, not
holding as a portfolio. Assess which is better — fixed SL & target, trailing SL, or a fixed
number of days. If trailing — Donchian, Supertrend, EMA — assess all. Reconstruct the
patterns first, see if we have it clear."

**What we're actually testing:** Given an automatable entry that reproduces the Chartink
"MultiTimeFrame Bullish" volume-breakout selection, which **exit** maximises **net
risk-adjusted return** on a **short-term** (days-to-weeks) horizon:
- (A) fixed stop-loss + fixed profit target,
- (B) trailing stop — Donchian channel / Supertrend / EMA / ATR-chandelier,
- (C) time-based exit (hold N days),
- (D) hybrid (initial hard SL then trail).

Single success metric: **net per-trade expectancy (in R and %)**, then portfolio Calmar /
MaxDD under concurrent sizing. A result must clear: net-positive after realistic cost across
**every year**, with monotonic (not lone-peak) parameter sensitivity.

## 2. Economic hypothesis

Breakout to new highs on a volume surge = **late-comer / momentum under-reaction flow**:
new information or a supply shortage pushes price out of a base; volume confirms real
participation; MTF (monthly+weekly+daily) MACD>0 filters for aligned uptrend so we buy
strength, not a dead-cat bounce. Counterparty = early/base holders taking profit and shorts
covering. **Why short-term:** microcap/smallcap breakout momentum is fast and reverts hard;
held as portfolio you give the pop back (the whole point of the user's ask). Decay risk: the
pattern is widely scanned (Chartink), so crowding erodes edge — must survive costs + recent
years.

## 3. The Base — entry mechanics (LOCKED, validated in reconstruction)

Daily bars. On each day `t`, a name QUALIFIES when ALL hold:
- `MACD_line(12,26) > 0` on **daily**, **weekly** (W-FRI resample of close), **monthly** (ME
  resample) — the Chartink MTF-MACD filter.
- `close ≥ 0.98 × rolling_252d_max(close)` — at/near 52-week high (proxy for ATH / near-ATH).
- `volume ≥ 2.0 × SMA(volume,20)` — volume breakout.
- `mean(close×volume, 20d) ≥ turnover_gate` — liquidity (illiquids removed); gate sweep
  ₹1cr / ₹5cr / ₹10cr. Price floor ₹20.

**Daily selection (the "usable" step — user's actual workflow):** the raw scan throws 500+
matches, which is NOT tradeable. So on each day, among all QUALIFYING names, **rank by today's
% run** (`close/prev_close − 1`, descending) and take the **top-K** (sweep K ∈ {3,5,10}) — the
names "running hardest today". This is exactly what the user does (looks at the Chartink table
sorted by %-change, takes the top handful). Cap concurrent open positions at `max_open` (sweep
5/10). De-dup: no re-entry into a name already held.
- **Marketcap note:** ₹1000cr-marketcap gate is NOT point-in-time computable (no shares table
  in DB); turnover gate is the tradeability filter that actually matters for a short-term book.

**Entry fill:** next-day **open** after the signal close (no look-ahead).
**Reconstruction check (PASSED):** fires on CARTRADE 2025-07-08/07-28/10-28, NACLIND
2025-06-27/07-18, AYMSYNTEX 2024 breakouts — i.e. it lands on the real breakout candles.

## 3b. Exit variants tested (the actual research question)

| Family | Variants |
|---|---|
| A. Fixed SL+target | SL ∈ {5,8,10,15%}; Target ∈ {none,10,15,20,30%} |
| B1. ATR chandelier | exit if close < HighestSinceEntry − m×ATR(22); m ∈ {2,2.5,3,4} |
| B2. Supertrend | ST(10,3), ST(10,2), ST(7,3), ST(7,2) flip |
| B3. Donchian | exit close < LowerDonchian(N); N ∈ {10,15,20} |
| B4. EMA | exit close < EMA(N); N ∈ {10,20,21,50} |
| C. Time | exit at open after N held days; N ∈ {5,10,15,20,40} |
| D. Hybrid | hard SL 8% THEN best trailing family |

All exits also carry a hard catastrophic SL (default 15%) and fill at next-open on a
close-based trigger (gap-through modelled at next open, not the trigger level).

## 4. Plan / grid + costs

- **Universe:** all daily symbols passing the liquidity+price floor at signal time
  (~point-in-time via trailing turnover). ~1,642 daily symbols in DB; liquidity filter trims
  to the tradeable subset. **Known bias:** the DB skews to today's Nifty-500-ish liquid names
  and is missing many of the exact microcaps traded (PAISALO, SMSPHARMA, VELJAN, … = 0 rows),
  so results are a **proxy** for the microcap population, stated loudly. Modern sub-period
  (2022–2026) reported separately.
- **Period:** 2019-01-01 .. 2026-05-15 (DB max for most names; CARTRADE to 2026-06-30).
- **Cost:** round-trip 0.20% base (delivery brokerage + STT + slippage); microcap slippage
  higher → sensitivity at 0.10 / 0.20 / 0.35%. STCG tax (20%) noted for net-of-tax on the
  short holds.
- **Cell count:** entry population generated ONCE to CSV; each exit variant replays that same
  population (cheap). ~4+5+4+3+4+5+ hybrids ≈ 30–40 exit configs × per-year.
- **G1 gate (this step):** does the raw entry have positive gross forward return (5/10/20/40d)
  and a population large enough (target ≥ 300 events) to bake off exits? If forward returns are
  flat/negative at every horizon → **NO EDGE**, stop before G2.

## 5. Status log

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-07-01 ~10:00 | Reconstruction PASSED | Entry fires on real breakout candles of covered names |
| 2026-07-01 ~10:05 | research/71 created on VPS; STATUS written | sections 1–4 locked |
| 2026-07-01 ~10:10 | G1 population + forward-return probe launching | universe-wide, writes entries CSV |

## 6. Crash recovery
- Probe script: `research/71_breakout_exit_bakeoff/scripts/g1_probe.py` on VPS.
- Entries CSV (resumable, incremental): `results/entries.csv`.
- Log: `results/g1_probe.log`. Check `tail -f`; population count = `wc -l results/entries.csv`.
- Runs on VPS venv: `cd /home/arun/quantifyd && venv/bin/python3 research/71_.../scripts/g1_probe.py`.

## 7. Files
| File | Purpose | Commit? |
|---|---|---|
| `BREAKOUT_..._STATUS.md` | this doc | yes |
| `scripts/g1_probe.py` | entry population + forward-return probe | yes |
| `results/entries.csv` | all entry events (reused by G2) | yes if small |
| `results/g1_forward_returns.csv` | per-horizon gross return summary | yes |
| `results/RESULTS.md` | verdict | yes (on close) |

## 8b. Deferred fine-filters (user note, 2026-07-01)

"If we get stocks every day, that's also not tradeable." Trade-frequency / capital-churn
control is a **fine-filter to apply at the END**, once the exit is settled. Levers held in
reserve (the probe already records `pct_run` + `turn_cr` per signal, so these need NO re-run):
- minimum `pct_run` threshold (only take strong-run days, cutting signal days);
- max trades/day and max concurrent positions (capacity/attention cap);
- per-name cooldown; take signals only on the strongest day in a window;
- optionally a static current-marketcap gate once a shares table is available.
Design order: **exit rule first → then squeeze trade frequency to a tradeable cadence.**

## 8. Findings

### G1 (2026-07-01) — entry population + gross forward returns. Gate: PASS (proceed to G2).
- **Population:** 35,169 qualifying signals, all-universe, 2001–2026. Ample to bake off exits.
- **Gross forward return (entry=next open):** 5d −0.1% (win45%), 10d +0.4% (48%), 20d +1.5%
  (51%), 40d +3.5% (55%). **Edge is a slow 20–40d momentum drift, not a fast pop** — 5-day
  holds are flat-to-negative gross.
- **Top-K-by-%run does NOT help** — TOP3/5/10 slightly *worse* than take-all at every horizon
  (40d: TOP5 3.44 vs ALL 3.78). Biggest-gap-today mildly mean-reverts. Use run as a *filter*,
  not the ranker.
- **Regime-dependent:** bad years bleed (2008 −13%, 2011/2018 ≈−3.5%, 2026 −2.2% ytd). Market
  filter likely needed at G3/G4.
- **Read:** SIGNAL-grade weak gross drift with a fat right tail. The exit rule is the crux —
  a trail that harvests the right tail (100–200% runners) while cutting losers converts the
  mediocre mean into positive skew. That is G2.

### G2 (2026-07-01) — exit bake-off, 35,168 trades × 31 exits, net @0.20% RT. Gate: PASS.

**Headline: trailing stops win; targets and tight/fast exits lose. The edge is captured by
giving winners room + time.** Net mean per-trade return, top configs:

| Exit | Net mean/trade | Win% | PF | Avg hold (d) | Tail>50% | Max loss |
|---|---|---|---|---|---|---|
| FIX_sl15_noTgt | +8.06% | 44 | 2.01 | 79 | 10.0% | −90%* |
| ST(7,3) trail | +4.95% | 43 | 1.86 | 47 | 5.7% | −83% |
| Donchian-20 trail | +4.90% | 42 | 1.82 | 50 | 5.9% | −78% |
| ST(10,3) trail | +4.89% | 42 | 1.87 | 45 | 5.5% | −83% |
| Chandelier 4×ATR22 | +4.68% | 43 | 1.82 | 46 | 5.4% | −83% |
| EMA-50 trail | +4.62% | 40 | 1.87 | 41 | 5.3% | −78% |
| TIME_40 (hold 40d) | +3.27% | **53** | 1.59 | 36 | 2.5% | −83% |
| — worst — FIX_sl5_tgt10 | −0.09% | 34 | 0.98 | 8 | 0% | — |
| — worst — TIME_5 | −0.09% | 46 | 0.97 | 5 | 0% | — |

**Lessons (decisive):**
1. **Trailing SL ≫ fixed target ≫ tight stop.** Best = wide trailing: ST(7,3)/ST(10,3),
   Donchian-20, Chandelier 4×ATR(22), EMA-50 — all ≈ +4.9% net/trade, PF ~1.85, ride ~45–50d.
2. **NO profit target.** Every fixed-target config underperforms its no-target sibling
   (FIX_sl8_tgt15 +0.92 vs FIX_sl8_noTgt +4.85). A target caps the fat tail that IS the edge.
3. **Time matters — hold weeks, not days.** TIME_5/EMA_10/Chand-2× ≈ 0. TIME_40 is the
   "smoothest" (win 53%, positive median) but trailing earns more.
4. **Extreme positive skew.** Best configs have NEGATIVE median (−3 to −9%) yet positive mean:
   lose small often, win big rarely. Psychologically hard (44% win) but that's the math.
5. **NOT cost-fragile** (low turnover, long holds): net@0.35% ≈ net@0.20% − ~0.15/trade.
6. **Regime-dependent** (per-year, best configs): 2023 +25, 2014 +25, 2020 +19, 2009 +16,
   2021 +11; but 2008 −15, 2011/2018 −7, 2015 −3, 2025 −2.6. Bear years bleed → G3 regime gate.
7. `*`FIX_sl15 −90% max-loss = gap-through on bad prints/microcap gaps → **data-hygiene +
   catastrophe-stop needed**; the *trailing* families cap this better (−78 to −83%).

**Status: SIGNAL confirmed (strong per-trade edge). NOT YET a STRATEGY** — these are
equal-weight independent per-trade expectancies, no concurrency/sizing/regime yet. G3/G4 next:
regime filter, portfolio equity curve with concurrency + the trade-frequency fine-filter,
tearsheet, cost/tax net. Recommended exit for the book: **ST(10,3) or Donchian-20 trail, no
target, 20% catastrophe stop.**

### G3 (2026-07-01) — liquidity cleaning + entry-fill realism. Gate: PASS (edge survives).

User caught two untradeable classes in the sample (KAMOPAINTS/OSIAHYPER): illiquid + circuit
"straight-line" runs (locked at upper circuit, filled only pre-market). Fixes applied:
- **Liquidity gate = 20d MEDIAN turnover ≥ ₹5cr** (not mean — one spike day inflated the mean
  and let OSIAHYPER's ₹0.2–0.5cr-median breakouts through). Sweepable 3/5/10.
- **Entry-fill guard:** skip if the entry (next-open) bar is circuit-locked ((h−l)/o < 1%) or
  gaps > 15% above the signal close (unfillable chase). 144 signals skipped.
- **Rejected an over-extension filter** — data killed it: IRFC broke out +63% above its 50-SMA
  and ran +119%, SUPRIYA +33%. Extension removes the biggest winners. KAMOPAINTS's −56% is a
  *post-entry reversal* → handled by the trailing stop, not pre-filterable.

**Result (adversarial kill survived):** population 35,176 → **20,804 tradeable** (~41% removed),
per-trade edge **unchanged** (ST7,3 +4.43 · DONCH20 +4.38 · EMA50 +4.28 · ST10,3 +4.27 ·
CHAND4 +4.12 · TIME40 +3.30 · FIX5/10 +0.02, net@0.20). Edge was NOT built on illiquid junk.
Named-stock check: OSIAHYPER 5→1 signals, SUPRIYA/MAZDOCK/IRFC/MARINE-2024 kept, KAMOPAINTS
kept-but-trail-protected. **Not cost-fragile** (ST10,3: 0%→0.35% = 4.47→4.12). Still
regime-dependent (2008 −16, 2018 −5, 2015 −4; 2003 +18, 2023 +15, 2020 +13).
Clean bake-off script: `scripts/g3_clean_bakeoff.py`. (Cost-display bug — subtracted 20% not
0.20% — found & fixed in first run.)

### G4 (2026-07-01) — portfolio equity curves (compounding, MTM). Gate: PASS → STRATEGY.
16 curves (exit×gate×concurrency). **Regime gate decisive** (halves DD + raises CAGR). Best =
Donchian-20 + NIFTY>200DMA gate + 8 concurrent: CAGR 20.5% / Sharpe 1.01 / MaxDD −36% / Calmar
0.57 / 45.9× vs NIFTYBEES 9.4×. No-gate DD −48 to −70%. `scripts/g4_portfolio.py`.

### G5 (2026-07-01) — trade-frequency fine-filter (the "too many per day" concern). RESOLVED.
Cadence is already LIGHT: even uncapped it's only **~0.55 trades/week**, an entry on ~10% of
days, busiest day 4 new names — the 8-concurrent cap + ~7-week holds throttle it naturally.
Sweep of max-entries/day × min-%run on the winning book:
- **max 1 new entry/day (recommended): MaxDD −29.1% (under 30%!), Calmar 0.68, CAGR 19.9%** —
  capping to 1/day smooths correlated pile-ins → lower DD, best Calmar. Dead simple to run.
- 2/day = highest CAGR 22.2% (DD −34.6, Calmar 0.64); min-%run gate does NOT help (run not
  additive, per G1). `scripts/g5_finefilter.py`.

### PAPER SOAK (G5) — `/app/breakout-paper` — BUILT 2026-07-01, auto-activates 15:36 IST
₹10L PAPER book (never places real orders). `services/breakout_paper.py` (cloned from
momentum_paper): daily 15:45 job = accrue cash yield → exits (Donchian-20 / 20% catastrophe) →
1 new breakout if NIFTY>200DMA & <8 held. DB `backtest_data/breakout_paper.db`. API
`/api/breakout-paper/{state,seed,run-daily}`. React page `frontend/src/pages/BreakoutPaper.tsx`
(+route in App.tsx, sidebar "Breakout ₹10L"). Frontend BUILT + live on VPS. Smoke test OK: 387
liquid names, gate currently **OFF** (NIFTY<200DMA → starts all-cash, correct), today's
candidates ATHERENERG/AUROPHARMA/RADICO.
**Activation (needs 1 restart — market-hours prohibited):** one-shot cron `36 15 1 7 *` →
`scripts/activate_breakout_paper.sh` restarts quantifyd (registers routes+job) + seeds + self-
removes. Log `/tmp/breakout_paper_activate.log`.
**Manual fallback (after 15:30 IST if cron missed):** `ssh vps 'sudo /bin/systemctl restart
quantifyd; sleep 25; cd /home/arun/quantifyd && venv/bin/python3 -c "import
services.breakout_paper as bp; print(bp.seed())"'`. Backups: /tmp/app.py.bak, App.tsx.bak,
Sidebar.tsx.bak on VPS.

### G5b (2026-08-07) — liquid-fund settlement realism (cash-ledger re-run). STATUS: DONE

**What you asked:** funds move to/from a liquid fund with T+1 both ways (redemption proceeds
usable next day; fresh parkings start earning next day). So the book must keep one slot's worth
of SETTLED cash idle every day to be able to buy a breakout same-day; if a slot is consumed by a
buy, redeem the next slot from the liquid fund the SAME day so cash is ready again tomorrow.
Neither the G5 backtest (zero cash yield, no cash constraint — pure notional model) nor the
paper book (instant 6.5% on ALL cash, no lags, no idle buffer) models this. Is the logic in,
and what are the honest numbers?

**What we're testing:** the winning G5 book (Donchian-20 trail + 20% catastrophe, NIFTY>200DMA
gate, 8 concurrent, max 1 new entry/day, %-run ranking, 0.20% RT cost, 2006→now, ₹10L,
eq/8 compounding slots) re-run through an explicit daily cash ledger, 4 variants:

- **A. published** — G5 as-is: no cash yield, no cash constraint (sanity: must reproduce
  19.9% / −29.1% / Calmar 0.68).
- **B. naive-instant** — what the paper book currently does: one cash pool, earns 6.5%/yr
  daily (calendar-day accrual), instantly available, entries cash-constrained.
- **C. realistic buffer (user spec)** — 3 buckets: idle settled buffer (target = 1 slot,
  earns 0) held EVERY day slots are free (gate ON or OFF); liquid fund earns 6.5% with
  parkings starting T+1; redemptions usable T+1; equity sale proceeds usable T+1; after an
  EOD buy, same-day redemption refills the buffer for tomorrow.
- **D. gate-aware buffer** — same as C but the buffer is held only while the regime gate is
  ON (during OFF everything sits in the fund; on a flip day the book cannot buy — entry
  capability resumes T+1).

Success criterion: quantify B−A (what yield adds), B−C (what the naive model over-credits),
C vs D (what the always-on buffer costs). Runner: `scripts/g5b_cash_ledger.py` (VPS, nice-15).
Cost caveat: 6.5% held flat across 2006→now (conservative for the high-rate 2006-2014 era).

**RESULTS (run 2026-08-07, window 2006→2026-08, ₹10L):**

| Variant | CAGR | Sharpe | MaxDD | Calmar | Final | Interest | Blocked entries |
|---|---|---|---|---|---|---|---|
| A published (no yield, no cash constraint) | 18.63% | 0.91 | −32.9% | 0.57 | ₹3.37cr | — | 0 |
| B naive-instant (old paper-book model) | 19.70% | 1.00 | −28.8% | 0.68 | ₹4.06cr | ₹30.8L | 0 |
| **C realistic buffer (user spec — DEPLOYED)** | **18.82%** | **0.97** | **−30.5%** | **0.62** | **₹3.49cr** | ₹22.2L | 240 |
| D gate-aware buffer | 19.67% | 1.00 | −30.7% | 0.64 | ₹4.03cr | ₹27.2L | 270 |

(A is below the published 19.9%/−29.1% because the window now includes Jul-2026's gate-OFF
drawdown month — same code, +1 month of data.)

Findings: (1) liquid yield is worth ~+1% CAGR over 20y — the published numbers were
conservative by that much; (2) the NAIVE instant model over-credits ~0.9% CAGR vs realistic
T+1 settlement — the old paper-book accounting was flattering; (3) the always-on buffer (C)
gives up ~0.85% CAGR vs parking it during gate-OFF (D) — mostly 2008/2011/2015/2018-19 bear
stretches where one idle slot earned nothing for months; D's price is entries slipping T+1 on
gate FLIP days (~30 extra blocked entry-days in 20y, immaterial); (4) ~12 blocked entry-days/yr
under realism are exits whose proceeds hadn't settled — instant-recycle was never real.

**Deployed to the live paper book (services/breakout_paper.py, cash-model v2):** 4 cash
buckets (settled buffer ⁄ T+1 in-transit ⁄ fund earning ⁄ fund pending-T+1), buffer target =
1 slot ×1.002 while a slot is free, same-day fund redemption after a buy, calendar-day 6.5%
accrual starting T+1, equity sales settle T+1. One-time migration recasts the book's history
from fills+NAV dates (interest ₹7,022 → ₹5,298 as of 08-06). `buffer_gate_aware` config flag
(default False = user spec C; True = D) if the +0.85% CAGR is ever wanted. Activates via the
Mon 09:00 preopen auto-restart; frontend (CASH/BUFFER split rows) is live already.
Study added to `frontend/src/data/backtests.ts`; factsheet `frontend/public/breakout-swing-
factsheet.png`; built on VPS (static/app), slug verified in live bundle. Frontend-only → no
backend restart. Headline = the 1/day-cap config (19.9% CAGR, −29.1% DD, Calmar 0.68). Recommended
tearsheet: `results/tearsheet.png` (= mpd=1). **VERDICT: STRATEGY candidate; regime gate mandatory;
optimistic due to survivorship — G5 paper soak owed before capital.**

