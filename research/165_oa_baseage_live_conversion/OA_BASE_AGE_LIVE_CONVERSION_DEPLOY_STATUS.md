# Open Alpha · Base Age — LIVE CONVERSION of the real ₹6.18L book — STATUS: STAGED, SWITCH OFF

**OA (Open Alpha).** This document is the sole crash-recovery source for converting the
**live, real-money** Open Alpha book from its legacy rules (−8% hard stop + 15-SMA close
trail, buy-stop-at-the-pivot entry) to the **Base Age** spec adopted in research/161 and
re-confirmed in research/164. Everything below is built, proved and left **switched off**:
a `git pull` on the VPS changes no running behaviour.

- **Book:** Open Alpha REAL, Zerodha RA6610, CNC equity, NSE cash.
- **Switch:** `OA_RULESET` in `services/oa_real.py` — `'legacy'` (current, default) | `'baseage'`.
- **Written:** 12-Sep-2026 (Saturday, market closed). Next session Monday 15-Sep-2026 09:15 IST.
- **Nothing in this work has placed an order, written the state file, edited the crontab,
  or restarted the backend.**

---

## 1. The Ask

**What Arun asked (verbatim, 12-Sep-2026 19:25 IST):**

> "we must convert the existing OA trades into this OA base age system, manage the exits and
> continue to be live with further trades."

**What is actually being built:**

Open Alpha is live with real money (capital ₹6,17,637.68, 11 open positions bought
04-Sep-2026, cash ₹1,88,697.86). Its **entries have been paused since 11-Sep-2026** because
research/158 proved the published entry mechanic is not placeable *and* found the scanner's
condition inverted (`close < pivot` where the design says `close > pivot`). Its **exits are
still running** on the legacy stop and trail.

The successor spec is **Open Alpha · Base Age** (research/161, re-fitted in research/164).
The conversion must:

1. **Re-home the eleven existing positions onto the new exit** — they were bought on the
   old entry rule, but from the flip they are managed by SuperTrend(14,4) alone. No
   position is force-sold on conversion day; each is simply evaluated against the new rule
   from that evening's official close.
2. **Restart entries**, on the corrected Base Age signal, filled at the next day's open.
3. Ship behind a single switch that defaults to today's behaviour, with a runbook Arun can
   execute without Claude present.

---

## 2. The Base — what is being deployed

### 2.1 The adopted spec (research/161 §1, frozen by research/164 `build164.py`)

| Element | Rule |
|---|---|
| Universe | NSE cash daily bars from `backtest_data/market_data.db`, `timeframe='day'`, rows with `volume > 0 and close > 0`, duplicates dropped keeping the last, ≥ 90 bars of history |
| Fund exclusion | **by NAME** from `backtest_data/etf_exclusions.json` (`by_name`, 346 curated symbols) — never the old ticker regex alone |
| Split guard (universe) | ATH is computed only on the bars **after** the last day-over-day fall worse than **−35%**; the series is truncated there (carried from research/161 `ath_events.py`, because `market_data.db` is not retroactively split-adjusted) |
| Liquidity | 20-day **median traded value** (`close × volume`, `rolling(20, min_periods=10).median()`) **≥ ₹2 cr** on the trigger bar |
| Sanity | `abs(trigger-day return) ≤ 50%`; 20-day median volume (shifted one bar) must exist and be > 0 |
| **Signal** | on the day's **official close**: `close[t] > max(close[:t])` — the first close above the **prior all-time-high close** — where that prior ATH close is **≥ 60 trading bars old** (`x_bars ≥ 60`) **and** the close fell **≥ 20% below it in between** (`depth_pct ≥ 20`) |
| Re-arm | a symbol re-arms only **60 bars** after its last **kept** signal (greedy, earliest-first, on the post-filter list — exactly `build164.rearm`) |
| Volume filter | **NONE** (research/161: raises per-trade expectancy, lowers book CAGR, deepens drawdown) |
| Saucer filter | **NONE** (research/161: 5.90% CAGR, 5.6 trades/yr — far too rare to fill a book) |
| **Entry fill** | the **NEXT day's open** |
| **Exit** | **SuperTrend(14,4)** on daily close (`bt_core.supertrend_dir`, Wilder ATR, period 14, multiplier 4). When direction flips to −1 on a close, **sell at the next day's open** |
| Hard stop | **NONE.** The −8% stop is dropped |
| Time stop | **NONE** |
| Book | **16 slots at 6.25% of current NAV**, whole shares, buy only if cash ≥ the slot cost |
| Contested slot | more qualifying signals than free slots/cash → **largest 20-day traded value wins** |
| Idle cash | not this executor's concern |

Study support: research/161 `results/RESULTS.md` (30 seeds, 2005-01-03 → 2026-09-11, 25 bps
a side, after tax): **CAGR 21.26% [worst seed 19.87%], MaxDD −34.80%, Calmar 0.618, win rate
49.2%, expectancy +12.43%/trade, 31.7 trades/yr, max losing streak 14.** research/164
re-ran the same book at a 5.0% idle rate (20.94% / −35.50% / 0.601) and confirmed **16
slots at 6.25% survives** its own bake-off.

### 2.2 Deviations from the study — declared, not hidden

| # | Study did | Live does | Why |
|---|---|---|---|
| D1 | Contested slots resolved by a **seeded random draw** | **Largest 20-day traded value wins** | research/164 §3: the traded-value rule beat the random draw at both slot counts tested (+0.78pp at 16 slots, +2.60pp at 8), on 29/30 and 30/30 paired seeds, in both windows. It does not clear that study's adoption bar, but it **costs nothing, removes path randomness entirely, and slightly improves capacity** — and a live book cannot draw a random number and call it a rule |
| D2 | Exit signal read on the **daily close** | **Two-stage**: 15:18 close-proxy raises an *alert* only; the exit is **confirmed on the official close** in the evening job and only then is an order placed | A live 15:18 proxy would sell on an intraday wobble that the close reverses. The study's signal is a close signal; the only faithful live reading is the official close |
| D3 | Sell/buy at "the next open" | **AMO** (after-market order) placed the evening before, resting for the next open — MARKET preferred, **AMO LIMIT at last close ∓2% as the fallback**, logged either way | An AMO is the only order type that participates in the opening trade without a human at the screen. The ∓2% band is the same floor idea as the legacy `place_exit` |
| D4 | Universe funds excluded by a **substring pattern** (`ath_events.py` `ETF_PAT`) | Excluded by the **curated name list** in `etf_exclusions.json` | research/158's `etf_filter.py`: the substring rule both under-excludes (the 2023-25 gold/silver fund wave: `EGOLD`, `TATAGOLD`, `SILVER1`, …) and over-excludes real companies (`SKYGOLD`, `GOLDIAM`, `SILVERTUC`, `GOLDTECH`). Quantified in §8.1 as its own mismatch bucket |
| D5 | Exit evaluated every bar including the entry bar | Same, **except** a single-day close move of **≤ −40%** is treated as a split/bonus: **HOLD + alert, never sell** | Precedent `services/ipo_paper.py` `DATA_EVENT_DROP`. `market_data.db` is not split-adjusted; selling into a data artefact is a real-money loss |
| D6 | No re-arm concept at 09:25 | The **09:25 re-arm cron stays commented out and is retired** | The legacy 09:25 job re-placed intraday `regular` buy-stops because a *touch-of-the-pivot* entry needs a live trigger all session. A next-open fill needs no trigger at all: the AMO either participates in the opening trade or it does not, and a same-day replacement would be a different (and un-backtested) entry |

### 2.3 Live state at the moment this work started (12-Sep-2026)

- **11 open positions** (INDSWFTLAB, SETL, WELCORP, SHILPAMED, SBCL, IRISDOREME, INOXINDIA,
  MANINDS, SSWL, ENTERO, NITINSPIN), all `entry_date` 2026-09-04.
- **capital ₹6,17,637.68 · cash ₹1,88,697.86 · NAV 11-Sep ₹6,21,068 · invested (cost) ₹4,46,348**
- 5 closed trades (SPORTKING, KTKBANK, KMEW, TMB, IOLCP), all losses, realised **−₹8,666**.
- **Entry crons commented out**: crontab lines **112** (`50 18 * * 1-5 … oa_entry.py --arm`)
  and **114** (`25 9 * * 1-5 … oa_entry.py --arm`).
- **Crontab backup from the pause:** `/tmp/mpf/ct.bak.20260911-111828` (research/158 §6).
- Exits, marks, reconcile and the OHLC bakes all still run:
  `18 15 check --arm` · `* 9-15 mark` · `46 18 mark` · `50 15 / 52 18 / 35 9,11,13 reconcile --arm`.
- Nightly universe refresh: **17:45 Mon–Fri** (`scripts/refresh_daily_universe.py`), which is
  why the exit confirmation must run **after** it and never on a partial candle.
- **And a second live actor nobody flagged:** `services/equity_executor.py`, crontab line
  **107**, `20 9 * * 1-5 … --arm`, places real `OA-TOPUP` buys into this book's holdings.
  See **§8.0** — it is the one thing standing between this work and a flip.

---

## 3. The Plan

| Step | Deliverable | Gate |
|---|---|---|
| A | `services/oa_baseage.py` — the shared spec library: universe load, split cut, `supertrend_dir` copied byte-for-byte from `bt_core.py`, the event scan, the re-arm | imports clean on the VPS |
| B | `services/oa_baseage_entry.py` — the live scanner: correct `close > prior ATH close`, age/depth/re-arm/liquidity, name-based fund exclusion, 16 slots, cash check, traded-value tie-break, AMO buy for the next open | dry-run prints a sane candidate list |
| C | `services/oa_real.py` — `OA_RULESET`, `_st14` (DB + close-proxy), `confirm()` mode, baseage branches in `check()` / `mark()`, legacy path untouched | `OA_RULESET='legacy'` reproduces today's output exactly |
| D | `services/oa_entry.py` — dispatch to the baseage scanner under `'baseage'`; legacy path untouched | — |
| E | **Replication gate** — the new signal function over the last 400 trading days vs research/164 `results/events164.csv` | **≥ 95% exact symbol-by-date agreement, every mismatch explained** |
| F | **Conversion dry-run** on the live 11 + Monday's candidate entries | table produced, nothing written |
| G | Runbook, register (`strategies.ts`), ops reviews, `TODO.md` | frontend build green, string present in the bundle |

### The grid there is to check

This is a conversion, not a sweep: there are no cells. The two things that are *measured*
rather than asserted are (E) the replication rate against the frozen study event list, and
(F) the per-position verdict of the new exit on the eleven live names.

---

## 4. Status log

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-09-12 ~19:30 | Read the spec sources | `oa_real.py`, `oa_entry.py`, `bt_core.py`, `ath_events.py`, `build164.py`, r/161 + r/164 RESULTS, r/158 §6, `etf_filter.py`, live crontab |
| 2026-09-12 ~19:45 | STATUS §1-4 written before any code | this file |
| 2026-09-12 ~20:10 | A-D built and shipped to the VPS, switch at `'legacy'` | no cron, no restart, no order |
| 2026-09-12 ~20:25 | **Replication gate PASS** | see §8.1 |
| 2026-09-12 ~20:40 | **Conversion dry-run produced** | see §8.2 / §8.3 |
| 2026-09-12 ~20:50 | **BLOCKER found: `equity_executor.py` at 09:20 is a second live buyer on this book** | §8.0 — not mine to edit; must be settled before any flip |
| 2026-09-12 ~21:00 | Runbook written (§9); register + ops + TODO text staged in §10, files NOT edited (the 5.2% commit had not landed; head `60d30b36`) | switch still `'legacy'` |
| 2026-09-12 ~21:10 | Committed on the VPS, not pushed | `services/oa_*.py`, `research/165/**` only |

---

## 5. Crash Recovery — how Arun resumes without Claude

**Nothing here is half-applied.** The conversion is inert until `OA_RULESET` is changed.

```bash
# 1. Where is the switch, and what is it set to?
ssh arun@94.136.185.54 "grep -n \"^OA_RULESET\" /home/arun/quantifyd/services/oa_real.py"
#    expected TODAY:  OA_RULESET = 'legacy'

# 2. Are the entry crons still paused?  (lines 112 and 114 must start with '#')
ssh arun@94.136.185.54 "crontab -l | grep -n oa_entry"

# 3. Is the live book untouched?
ssh arun@94.136.185.54 "cd /home/arun/quantifyd && git status --short backtest_data/oa_real_state.json && venv/bin/python3 -c \"import json;s=json.load(open('backtest_data/oa_real_state.json'));print(len(s['positions']),'positions, cash',s['cash'],'capital',s['capital'])\""
#    expected: 11 positions, cash 188697.86, capital 617637.68

# 4. Re-run the replication gate (read-only, ~3 min, safe any time)
ssh arun@94.136.185.54 "cd /home/arun/quantifyd && nice -n 10 venv/bin/python3 research/165_oa_baseage_live_conversion/scripts/replicate165.py > /tmp/oa165_rep.log 2>&1; tail -40 /tmp/oa165_rep.log"

# 5. Re-run the conversion dry-run (read-only, no Kite, no state write)
ssh arun@94.136.185.54 "cd /home/arun/quantifyd && venv/bin/python3 research/165_oa_baseage_live_conversion/scripts/dryrun165.py > /tmp/oa165_dry.log 2>&1; cat /tmp/oa165_dry.log"
```

**Files that must NOT be touched by this work:** `backtest_data/oa_real_state.json`,
`backtest_data/market_data.db`, `backtest_data/access_token.json`, any other executor,
`frontend/` (other than `src/data/strategies.ts`), `research/163_*`,
`research/111_sensex_manual_mgmt/scripts/ops_center.py` and `TODO.md` until the concurrent
idle-cash session has committed.

**Safe to inspect:** everything under `research/165_oa_baseage_live_conversion/`, the two new
`services/oa_baseage*.py` files, `/tmp/oa165_*.log`.

**To abandon the whole thing:** `git revert` the commits listed in §7, or simply leave
`OA_RULESET = 'legacy'` — the new code is not reached.

---

## 6. Rollback

| Situation | Action |
|---|---|
| Flip has not happened | nothing to do; the switch is `'legacy'` |
| Flipped, and Arun wants out before the next session | set `OA_RULESET = 'legacy'`, re-comment the 18:50 entry line (crontab-safety procedure in §9), cancel any resting AMO by hand in Kite. The legacy `stop` field is still present on every position in the state file, so the old rule resumes with no data loss |
| Flipped, an AMO already filled | the position is a normal holding; it is managed by whichever ruleset is active. Nothing needs unwinding |

---

## 7. Files

| File | Purpose | Committable |
|---|---|---|
| `services/oa_baseage.py` | shared spec library: universe, split cut, SuperTrend(14,4), event scan, re-arm | yes |
| `services/oa_baseage_entry.py` | live Base Age entry scanner + AMO arming | yes |
| `services/oa_real.py` | `OA_RULESET`, `_st14`, `confirm()`, baseage exit branches | yes (modified) |
| `services/oa_entry.py` | dispatch to the baseage scanner under `'baseage'` | yes (modified) |
| `research/165_.../scripts/replicate165.py` | the replication gate vs `events164.csv` | yes |
| `research/165_.../scripts/dryrun165.py` | the conversion dry-run on the live 11 | yes |
| `research/165_.../results/replication165.csv` | per-mismatch detail | yes (small) |
| `research/165_.../results/dryrun165.md` | the table Arun approves | yes |
| `research/165_.../OA_BASE_AGE_LIVE_CONVERSION_DEPLOY_STATUS.md` | this file | yes |

---

## 8. Findings

### 8.0 THE BLOCKER — a SECOND live actor buys into this book every morning at 09:20

**This must be decided before the switch is flipped. It is not part of the conversion and it
was not in the brief; it was found while checking whether a gunicorn restart was needed.**

`services/equity_executor.py` runs from **crontab line 107** — `20 9 * * 1-5 … --arm` — with
real order-placing rights, and Open Alpha is one of its two books. Its OA leg
(`deploy_open_alpha`) reads `backtest_data/oa_real_state.json`, computes

```
target = (market value of the held positions + cash) / 16
```

and places **real CNC LIMIT buys tagged `OA-TOPUP`** into the positions the book **already
holds**, largest shortfall first, until the idle cash is gone. It has no kill flag set
(`backtest_data/executor_kill.flag` does not exist), and its idempotence is per
(book, symbol, **day**) — so it is not a one-off; it gets another go every morning.

**Why it conflicts with Base Age.** Base Age sizes **once, at the fill, at 6.25% of NAV, and
never adds**. Adding to a position days later — often one that has fallen, which is exactly
what "largest shortfall first" selects — is a rule the study never tested. Worse, it spends
the cash the new book needs: research/164's structural finding is that **cash, not slots, is
the binding constraint** (3,619 qualifying events → 688 entries, 977 refused for a slot,
**1,955 refused for cash**), and this session's own 60-day forward walk reproduced it — 21
entries refused for cash against 20 taken. Every rupee the 09:20 job spends on a top-up is a
Base Age entry that cannot be funded. The executor's own docstring already calls the OA
top-up "a deviation from the tested spec" with a study registered for 2026-10-31; under Base
Age it stops being a deviation and becomes a contradiction.

**The good news: on Monday's numbers it would place nothing.** With 11 positions and the
target struck over **16** slots, the per-slot target at Friday's close is ₹38,867, and every
held position is already within ₹1,875 of it — under the executor's own ₹2,000 `MIN_ORDER`
floor. So the idle ₹1,88,698 is, at these prices, mathematically un-deployable by it.

| symbol | value at 11-Sep close | gap to the ₹38,867 target | would it buy? |
|---|---|---|---|
| SETL | ₹46,347 | 0 | no |
| MANINDS | ₹40,890 | 0 | no |
| NITINSPIN | ₹39,772 | 0 | no |
| SSWL | ₹39,326 | 0 | no |
| IRISDOREME | ₹39,289 | 0 | no |
| INDSWFTLAB | ₹38,511 | ₹356 | no — under the ₹2,000 floor |
| SHILPAMED | ₹38,362 | ₹505 | no — under the floor |
| INOXINDIA | ₹38,123 | ₹744 | no — under the floor |
| ENTERO | ₹37,945 | ₹922 | no — under the floor |
| WELCORP | ₹37,617 | ₹1,250 | no — under the floor |
| SBCL | ₹36,992 | ₹1,875 | no — under the floor, by ₹125 |

**So the conflict is latent, not immediate — and the margin is ₹125.** A 5–6% fall in any one
of these names on Monday morning opens a gap above ₹2,000 and the job buys into it with cash
Base Age wants for its next signal. SBCL needs to fall **0.35%** to cross the line.

**What must be done before the flip, and it is NOT mine to do.** Editing
`services/equity_executor.py` is off-limits under the standing rule that a conversion never
touches another executor's trading logic. Two ways to settle it, in order of preference:

1. **Crontab-only (recommended, no code change at all).** Point line 107 at the other book:
   `venv/bin/python -u services/equity_executor.py --arm --book ipo-base`. The file stays
   byte-identical, the IPO Base deployment leg is untouched, and reverting is one word. This
   is a crontab edit, so it rides the same backup→temp→verify→install procedure as §9 step 4.
2. **A narrow guard in the executor**, as its own approved strategy change with its own
   STATUS doc and evidence: `if OA_RULESET == 'baseage': return` at the top of
   `deploy_open_alpha`. Cleaner in the long run, but it is an engine edit and must not ride
   in on the back of this conversion.

**If neither is done, do not flip.** Running Base Age entries and OA-TOPUP together is a book
that both concentrates into losers and starves its own signals, and it is not a book anyone
has measured.

---

### 8.1 Replication gate — **PASS, 100.00%**

`scripts/replicate165.py`, run against research/164's frozen `events164.csv` (3,619 events —
the adopted spec after filtering **and** the 60-bar re-arm, i.e. the exact list the published
numbers were computed from). Gate window: the last **400 trading sessions**, **2025-01-31 →
2026-09-11** (562 study events in it).

| Phase | What is being run | Study events | Scanner events | Exact matches | Rate | Misses | Extras |
|---|---|---|---|---|---|---|---|
| **A** | live scanner under **research/161's own universe rule** (its substring fund pattern; no last-bar allowance) — 2,492 symbols | 562 | 562 | **562** | **100.00%** | **0** | **0** |
| **B** | live scanner as it will actually run — **curated `etf_exclusions.json`** name list — 2,369 symbols | 562 | 558 | 554 | 98.58% | 8 | 4 |

**Phase A is the gate and it is exact.** Not 95%, not "within tolerance" — every one of the
562 study signals in the window is reproduced by `services/oa_baseage.py` on the same symbol
and the same date, with no signal invented. Full-history counts line up too (3,646 qualifying
events against the study's 3,619 — the difference is entirely events before the study panel's
2005-01-03 calendar start, which `build164.py` drops by construction).

**Phase B is deviation D4 measured, not a failure.** All twelve differences are named:

| Bucket | n | Symbols | Reading |
|---|---|---|---|
| Study kept it, the curated list excludes it | **8** | GROWWDEFNC, ICICIB22, MAHKTECH, MASPTOP50, MODEFENCE (×2), MOM100, MON100 | **Every one is a fund or an index product.** research/161's substring pattern let them through and the study traded them as if they were companies. Excluding them is a correction, not a loss |
| Study excluded it, the curated list keeps it | **3** | SHANTIGOLD, SILVERTUC, SKYGOLD | **Every one is a real operating company** whose ticker merely contains GOLD or SILVER. research/158's `etf_filter.py` warned about exactly these. Restoring them is a correction too |
| Trigger after the study list ends | **1** | PAYTM, 2026-09-11 | The study needs a **next** bar to book the entry open, so its list stops at 2026-09-10. The live scan fires on the day's close and arms for tomorrow — the last-bar allowance working as designed, and it is Monday's only candidate |

**Nothing is unexplained.** Per-row detail `results/replication165.csv`, summary
`results/replication165.json`.

### 8.2 Forward walk — the last 60 sessions at the live book's own capital

The same code path replayed end to end over **2026-06-19 → 2026-09-11** on ₹6,17,638 —
entries at the next open, the traded-value tie-break, the cash check, SuperTrend(14,4) exits
at the next open. This is the "last 60 days' would-have-been entries with sizes" the gate
asked for; full ledger `results/walk165.csv`.

| | |
|---|---|
| Buys filled | **20** |
| Sells (ST(14,4) flip, next open) | **5** |
| **Refused for cash** | **21** |
| Refused for a slot | **86** |
| End state | 15 positions, cash ₹34,631, **NAV ₹6,73,106** |

| fill date | symbol | qty | price | signal |
|---|---|---|---|---|
| 2026-06-22 | WABAG | 19 | ₹2,014.90 | ATH 19-Jun, base 378 bars, depth 45%, TV ₹77.6 cr |
| 2026-06-23 | AIAENG | 7 | ₹4,860.00 | base 460 bars, depth 37%, TV ₹35.0 cr |
| 2026-06-25 | SKMEGGPROD | 140 | ₹274.00 | base 694 bars, depth 70%, TV ₹16.9 cr |
| 2026-07-01 | AUROPHARMA | 24 | ₹1,578.70 | base 452 bars, depth 35%, TV ₹164.8 cr |
| 2026-07-03 | RATEGAIN | 42 | ₹949.80 | base 583 bars, depth 53%, TV ₹40.1 cr |
| 2026-07-30 | APCOTEXIND | 54 | ₹700.00 | base 1,031 bars, depth 56%, TV ₹5.0 cr |
| 2026-08-18 | SHANTIGOLD | 144 | ₹269.00 | base 245 bars, depth 38%, TV ₹15.6 cr — one of the three names research/161's fund pattern wrongly deleted |

Two things worth saying plainly. **The tie-break binds hard**: 86 refusals for want of a slot
means the ranking rule does real work most weeks, not just at the occasional clash — so
choosing traded value over a random draw is a live decision, not a formality. And **cash
binds too**: 21 refusals, each one a signal the book wanted and could not fund. Both
reproduce research/164's structural finding at live-sized capital, which is the best evidence
available that this executor implements the book the study measured.

### 8.3 Conversion dry-run — the eleven live positions

`scripts/dryrun165.py`, on the last official session **2026-09-11 (Friday)**. Read-only: no
Kite, no lock, no state write, run with `OA_RULESET='legacy'`.

**NAV ₹6,21,871 = positions ₹4,33,173 + cash ₹1,88,698. Capital ₹6,17,638. Slot at 6.25% =
₹38,867.**

| symbol | qty | buy | 11-Sep close | P&L % | ST(14,4) line | dir | to the line | 15-SMA | to SMA | **BASE AGE** | legacy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| INDSWFTLAB | 99 | 362.33 | 389.00 | +7.36% | 312.48 | +1 | **+24.5%** | 365.38 | +6.5% | **HOLD** | HOLD |
| SETL | 99 | 402.69 | 468.15 | +16.26% | 380.30 | +1 | **+23.1%** | 382.13 | +22.5% | **HOLD** | HOLD |
| WELCORP | 14 | 2,596.36 | 2,686.90 | +3.49% | 2,293.44 | +1 | **+17.2%** | 2,519.67 | +6.6% | **HOLD** | HOLD |
| SHILPAMED | 40 | 962.36 | 959.05 | −0.34% | 817.64 | +1 | **+17.3%** | 922.04 | +4.0% | **HOLD** | HOLD |
| SBCL | 34 | 1,115.48 | 1,088.00 | −2.46% | 924.16 | +1 | **+17.7%** | 1,076.40 | +1.1% | **HOLD** | HOLD |
| IRISDOREME | 634 | 62.39 | 61.97 | −0.67% | 54.13 | +1 | **+14.5%** | 59.77 | +3.7% | **HOLD** | HOLD |
| INOXINDIA | 17 | 2,236.50 | 2,242.50 | +0.27% | 1,896.64 | +1 | **+18.2%** | 2,137.79 | +4.9% | **HOLD** | HOLD |
| MANINDS | 47 | 800.19 | 870.00 | +8.72% | 688.06 | +1 | **+26.4%** | 779.40 | +11.6% | **HOLD** | HOLD |
| SSWL | 106 | 358.30 | 371.00 | +3.54% | 313.84 | +1 | **+18.2%** | 337.32 | +10.0% | **HOLD** | HOLD |
| ENTERO | 21 | 1,843.72 | 1,806.90 | −2.00% | 1,593.85 | +1 | **+13.4%** | 1,775.94 | +1.7% | **HOLD** | HOLD |
| NITINSPIN | 61 | 636.65 | 652.00 | +2.41% | 563.48 | +1 | **+15.7%** | 631.62 | +3.2% | **HOLD** | HOLD |

**Base Age would sell 0 of 11 at Monday's open. The legacy rule would also sell 0.**

**The conversion is therefore free on day one** — no position changes hands because of the
flip, and Arun is not being asked to approve a liquidation dressed up as a rule change. What
he *is* approving is the room each position gets from here: the ST(14,4) trail sits **13.4%
to 26.4%** below Friday's close, against a 15-SMA that sits **1.1% to 22.5%** below it. On
five of the eleven (SBCL +1.1%, ENTERO +1.7%, NITINSPIN +3.2%, IRISDOREME +3.7%, SHILPAMED
+4.0%) the legacy trail is within one ordinary down-day; the new trail is not. That looser
leash **is the change** — research/161 measured it at **+11.85pp of CAGR** over the
15-SMA-plus-8%-stop on identical entries — and it will feel wrong the first time a position
gives back 12% without being sold. It is supposed to.

### 8.4 Monday's candidate entries (signals from the 11-Sep close)

| symbol | close | prior ATH close | prior ATH date | base age | depth | 20-day TV | size | verdict |
|---|---|---|---|---|---|---|---|---|
| **PAYTM** | ₹1,817.00 | ₹1,798.75 | 2021-11-25 | **1,191 bars** | **82.4%** | ₹479.61 cr | 21 sh ≈ **₹38,157** | **ARM buy for the open** |

One signal, 5 free slots, cash ₹1,88,698 → cash left after it **₹1,50,541**. No tie-break was
needed and nothing was refused. PAYTM is a textbook instance of the adopted pattern: a first
close above a high set nearly five years earlier, after an 82% collapse in between — the
"aged deep base" research/161 found worth +2.6pp of CAGR and −6.8pp of drawdown over plain
new-ATH entries.

**This is the one event in the whole gate that research/164's list does not contain** (§8.1,
bucket 3), for a good reason: the study's event builder requires the *next* bar to exist so
it can book the entry open. Not a discrepancy — the live scanner doing the only thing a live
scanner can do.

### 8.5 The no-op proof — a `git pull` changes nothing that runs

```
$ venv/bin/python3 services/oa_real.py ruleset
legacy
$ venv/bin/python3 services/oa_real.py confirm --arm
OA_RULESET is 'legacy' - Base Age exit confirmation does not apply; nothing done.
$ venv/bin/python3 services/oa_baseage_entry.py
OA_RULESET is 'legacy' - the Base Age scanner is not the active ruleset; nothing scanned, nothing placed.
$ venv/bin/python3 -m py_compile services/oa_real.py services/oa_entry.py \
      services/oa_baseage.py services/oa_baseage_entry.py && echo COMPILE_OK
COMPILE_OK
```

- `check` and `mark` take the legacy branch: byte-for-byte the behaviour of 11-Sep.
- `confirm` is a **no-op** under `'legacy'`, which is why it is safe in cron before or after
  the flip.
- The two new modules are only ever reached through the switch.
- `backtest_data/oa_real_state.json` untouched: still **11 positions, cash ₹1,88,697.86,
  capital ₹6,17,637.68**. Its `M` in `git status` predates this session.
- **No gunicorn restart is required, and this was verified rather than assumed.** Nothing in
  `app.py` references `oa_real`; the only server-side importer is `services/sleeves_api.py`,
  which imports it lazily inside `_oa_book()` and uses only `deposit`, `withdraw`,
  `load_state` and `status` — none of which read `OA_RULESET`. `services/book_liveness.py`
  and `services/equity_executor.py` reference the state **file path**, not the module. Every
  job that reads the switch (`check`, `mark`, `confirm`, `oa_entry`) is a fresh cron process
  that re-imports on every run.

---

## 9. Runbook — the flip, in order

**Do not run any of this before 15:40 IST on a trading day**, and re-check the clock at the
moment of doing it — an approval given on Saturday is not valid on Monday afternoon. Nothing
here restarts a service, but steps 3 and 4 change what tomorrow's crons do.

```bash
# 0. THE CLOCK, EVERY TIME. Approvals do not survive a session gap.
ssh arun@94.136.185.54 'TZ=Asia/Kolkata date'

# 0b. Is anything of this book's resting at the broker right now?
ssh arun@94.136.185.54 "cd /home/arun/quantifyd && venv/bin/python3 -c \"
import sys; sys.path.insert(0,'.')
from services.oa_real import _kite
k=_kite()
[print(o['tradingsymbol'], o['transaction_type'], o['status'], o.get('tag')) for o in k.orders()
 if o.get('status') not in ('COMPLETE','REJECTED','CANCELLED')]\""
```

**STEP 1 — get the code onto the VPS.** It is already there and committed in the VPS working
tree (§7); the commit was **not pushed**, so there is nothing to pull. If the commit has since
been pushed and the tree reset, restore with `git -C /home/arun/quantifyd pull --ff-only` and
re-run the §8.5 no-op proof before going on.

**STEP 2 — SETTLE THE 09:20 TOP-UP FIRST (§8.0).** Do not skip this. The recommended,
code-free form is folded into step 4 below.

**STEP 3 — throw the switch.**

```bash
ssh arun@94.136.185.54 "cd /home/arun/quantifyd && \
  sed -i \"s/^OA_RULESET = 'legacy'/OA_RULESET = 'baseage'/\" services/oa_real.py && \
  venv/bin/python3 services/oa_real.py ruleset"
# must print: baseage
```

**STEP 4 — the crontab, following the crontab-safety rule.** Back up, transform into a temp
**file**, verify the line count, then install from that file. **Never pipe `sed` straight into
`crontab -`** — a failed filter wiped all 58 jobs on 2026-09-01. Use `~` as the `sed`
delimiter, never `|`.

```bash
ssh arun@94.136.185.54 'cd /home/arun/quantifyd && \
  B=/tmp/ct.bak.$(date +%Y%m%d-%H%M%S) && crontab -l > $B && wc -l $B && \
  sed -e "s~^#50 18 \* \* 1-5 cd /home/arun/quantifyd \&\& flock -n /tmp/oa_entry.lock~50 18 * * 1-5 cd /home/arun/quantifyd \&\& flock -n /tmp/oa_entry.lock~" \
      -e "s~services/equity_executor.py --arm~services/equity_executor.py --arm --book ipo-base~" \
      $B > /tmp/ct.new && \
  echo "before: $(wc -l < $B)  after: $(wc -l < /tmp/ct.new)" && \
  diff $B /tmp/ct.new'
```

Read the diff. It must show **exactly two changed lines**: line 112 losing its leading `#`,
and line 107 gaining `--book ipo-base`. Line counts must be **identical**. Line **114 (the
09:25 re-arm) must still be commented** — a next-open fill needs no intraday re-arm, and
re-placing an entry mid-session would be a different mechanic from the one measured (D6).
Only then:

```bash
ssh arun@94.136.185.54 'crontab /tmp/ct.new && crontab -l | grep -n "oa_entry\|equity_executor"'
```

**STEP 5 — no restart.** Verified in §8.5: no OA job runs inside gunicorn. Do **not** restart
`quantifyd` as part of this.

**STEP 6 — first-night watch (the evening of the flip, from 18:50).**

```bash
ssh arun@94.136.185.54 'tail -60 /tmp/oa_entry.log'
```

Look for, in this order:

1. `OA_RULESET=baseage - exits first, then the Base Age entry scan.`
2. the `confirm` block — either `no Base Age exit due`, or one `EXIT PLACED` line per name
   with the order id **and which order type went in** (`MARKET` or `LIMIT …`);
3. the scan line, the candidate list, the book line (held / free / NAV / cash / slot);
4. `ARM <SYM> BUY n for tomorrow's open` + `placed <id> as MARKET|LIMIT`.

**The one live unknown is the AMO order type.** The code tries **AMO MARKET** first and falls
back to **AMO LIMIT** at last close ±2%, logging which. It was deliberately **not test-fired**
— firing a test order is placing an order. Both branches are safe: MARKET is what the study
models; LIMIT fills at the open in any ordinary session and declines only a violent gap. If
the log shows the LIMIT fallback on every name, that is this account's RMS answer and nothing
needs fixing — but record it here so the next reader is not surprised.

**STEP 7 — next morning.**

```bash
ssh arun@94.136.185.54 'tail -40 /tmp/oa_reconcile.log; tail -20 /tmp/equity_executor.log'
```

`reconcile --arm` (09:35 / 11:35 / 13:35) applies the fills by tag. The executor log should
say it is running `ipo-base` only.

### Rollback

```bash
ssh arun@94.136.185.54 "cd /home/arun/quantifyd && \
  sed -i \"s/^OA_RULESET = 'baseage'/OA_RULESET = 'legacy'/\" services/oa_real.py && \
  venv/bin/python3 services/oa_real.py ruleset"      # -> legacy
# then re-install the crontab backup taken in step 4:
ssh arun@94.136.185.54 'crontab /tmp/ct.bak.<stamp> && crontab -l | grep -n oa_entry'
```

Cancel any resting AMO by hand in Kite. **Every position still carries its `stop` field**, so
the legacy −8% stop and 15-SMA trail resume immediately with no data loss.

### What could go wrong on day one

| Risk | Why it is or is not a problem here |
|---|---|
| **The 09:20 top-up spends the cash** | §8.0. Latent today by a ₹125 margin on SBCL; a 5–6% fall in any holding makes it live. **Settle it in step 4 or do not flip** |
| A position exits at Monday's open | Not on Friday's numbers — all 11 sit 13–26% above the ST line, and the nearest, ENTERO, needs a **13.4%** fall in one session |
| Gap risk on the AMO buy | An AMO MARKET takes the open whatever it is. The LIMIT fallback caps the buy at close +2% and simply does not fill on a violent gap up — which is the entry worth skipping anyway. Either way the exposure is one slot, **₹38,867**, 6.25% of NAV |
| A cash refusal on a day with several signals | Expected and logged as `refused for cash`, not swallowed. The walk saw 21 in 60 sessions. The book behaving as research/164 measured, not a bug |
| The 17:45 universe refresh fails | `confirm` refuses any symbol whose latest DB bar is not the latest session, alerts `N position(s) not evaluated`, and makes **no** exit decision. It never reads a partial candle |
| A split or bonus prints a −40% close | Held, not sold, plus an alert — `ipo_paper.py`'s precedent (D5). `market_data.db` is not split-adjusted, and selling into a data artefact is a real loss |
| A duplicate sell | `place_exit_amo` refuses a symbol that already has a live SELL, and on an unreadable order book it assumes one exists rather than risk a double |
| The book looks wrong on the page | `mark` now publishes `ruleset`, `trail_rule: 'ST(14,4)'`, `st_dir` and `stop_active: false`, so the dashboard shows the trail the book actually obeys rather than a 15-SMA it no longer uses |

---

## 10. Register, ops and TODO — TEXT STAGED, NOT YET APPLIED

At the time of writing, the concurrent idle-cash session's commit (idle cash → 5.2%) had
**not** appeared in `git log --oneline -10` — head was `60d30b36`. Under the no-collision
rule, `frontend/src/data/strategies.ts`,
`research/111_sensex_manual_mgmt/scripts/ops_center.py` and `TODO.md` were therefore **not
edited**. The exact copy to apply once that commit lands:

**`frontend/src/data/strategies.ts` — the Open Alpha row**

- name → **`Open Alpha · Base Age (converting)`**
- status → unchanged, **live** (real money, ₹6.18L, 11 positions)
- rule line → *"First close above an all-time-high close that is at least 60 trading bars old
  and at least 20% above the low in between; 20-day traded value ≥ ₹2 cr; bought at the next
  open. Exit: SuperTrend(14,4) on the close, sold at the next open. No stop, no time stop.
  16 slots at 6.25% of NAV; contested slots go to the most liquid name."*
- rules rows → replace the −8% stop and the 15-SMA trail with the two lines above; add
  "Entries paused 11-Sep-2026; Base Age entries staged 12-Sep-2026, switch OFF".
- studies → add **research/161** (`/app/backtest/ath-base-age-breakout-research161`) and
  **research/164** (the slots-and-sizing study).
- change-log → **`2026-09-12 — Open Alpha · Base Age conversion STAGED. Code shipped behind
  OA_RULESET in services/oa_real.py, defaulting to 'legacy'. Replication gate 100.00% against
  research/164's 3,619-event list; dry-run says all 11 live positions HOLD under
  SuperTrend(14,4). Switch OFF pending Arun's flip, and pending a decision on the 09:20
  equity_executor OA top-up.`**

Build-gated as always: `export PATH=$HOME/.nvm/versions/node/v20.20.2/bin:$PATH; cd frontend
&& npm run build`, then grep the emitted bundle for `Base Age (converting)`.

**`research/111_sensex_manual_mgmt/scripts/ops_center.py` — REVIEWS**

- **Repurpose** the existing 2026-09-26 Base Age paper-book review to: *"Open Alpha · Base
  Age LIVE conversion: first two weeks — did the live fills, exits and cash refusals track
  the study? Compare actual AMO fills against the next-open assumption, ST(14,4) exits
  against the dry-run lines, and the count of 'refused for cash' against research/164's
  1,955-in-3,619 base rate."* Status PENDING.
- **Add** `2026-09-15` — *"Open Alpha · Base Age day-one check after the flip: did the 18:50
  job run both legs, which AMO order type did Kite accept, did any exit or entry fill at the
  open, and did the 09:20 equity_executor stay off the OA book?"* Status PENDING.
- Mirror both in `docs/LABS_AND_JOBS_REFERENCE.md`.

**`TODO.md` — new entry at the top**

> **Open Alpha · Base Age live conversion — STAGED, SWITCH OFF (12-Sep-2026).** Code is on
> the VPS behind `OA_RULESET` in `services/oa_real.py` (default `'legacy'`, so nothing has
> changed). Replication gate 100.00% vs research/164's 3,619 events; all 11 live positions
> HOLD under SuperTrend(14,4); Monday's only candidate is PAYTM (21 sh ≈ ₹38,157).
> **BLOCKER before the flip:** `services/equity_executor.py` still tops up OA holdings at
> 09:20 with real orders — settle it (crontab `--book ipo-base`, or a guarded engine change
> with its own STATUS) or do not flip. Runbook:
> `research/165_oa_baseage_live_conversion/OA_BASE_AGE_LIVE_CONVERSION_DEPLOY_STATUS.md` §9.
