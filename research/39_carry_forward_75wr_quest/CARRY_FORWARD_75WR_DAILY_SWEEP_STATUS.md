# Carry-Forward 75% WR Quest — BTST + Multi-Day Swing + Weekly Continuation

STATUS: PATTERNS 1/2/3/4/6 DONE — NO 75% WALK-FORWARD WINNERS (2026-05-07)
> Pattern 1 (BTST), 2 (Swing breakout), 6 (Weekly continuation) findings:
> `results/RESULTS_BTST_SWING_WEEKLY.md`
> Patterns 3 (RSI mean-reversion) + 4 (Earnings drift) findings: see live event log.

> **Headline goal:** find one or more carry-forward systems (overnight to multi-week holds) on Indian equities with **walk-forward WR ≥ 75% AND RR > 1.1** (favorable risk-reward — TP > SL). The intraday 5-min hunt (research/37 + research/38) hit a structural ceiling at WR ≤ 60% with favorable RR. Carry-forward unlocks regimes where multi-day continuation, post-event drift, and weekly trend persistence have demonstrated 70-80% WR in academic literature.

---

## 1. The Ask (precise restatement)

**What Arun asked (verbatim):**
> "you can expand your universe to carry forward trades (long and shorts)
> - it can be BTST, 2 day - 1 week or as long as it takes...."

**What we're actually testing — cleaned-up:**

Across the **1,623-stock daily universe** (`market_data_unified` timeframe='day', 2000-2026) and supplementary **95-stock 60-min universe** (2018-2025), find **swing/positional systems** that hold positions overnight to multi-week, with:

| Metric | Required |
|---|---|
| Walk-forward WR (out-of-sample) | **≥ 75%** |
| Risk-reward (TP : SL) | **TP ≥ 1.1 × SL** (favorable) |
| Profit factor | **≥ 2.0** |
| Max drawdown (out-of-sample) | **≤ 15%** |
| Trades / year | **≥ 60** combined across cohort (≈ 1+/week) |

Hold range: 1 day (BTST) → 5 days (week) → as long as required by signal exit. Long AND short permitted.

---

## 2. The Base — what's available

### 2.1 Data inventory

| Timeframe | Stocks | Period | Notes |
|---|---|---|---|
| **day** | 1,623 | 2000-01-03 → 2026-03-19 | Full Nifty 500+ history |
| **60min** | 95 | 2018-01 → 2025-12 | F&O + liquid mid-caps |
| **30min** | 50 | 2020-01 → 2026-05 | Recent only |
| 5min | 380 | 2024-03 → 2026-03 | Already exhausted in research/37+38 |

For carry-forward research the **daily timeframe is the primary**. 60-min for finer entry timing on high-frequency setups.

### 2.2 F&O universe (recommended starting cohort)

86 stocks with F&O liquidity defined in `services/data_manager.py` (FNO_LOT_SIZES dict, lines 42-127). These are the most-liquid Indian equities — important for carry-forward because:
- Borrow availability (SLB) for short positions over multiple days
- Tighter spreads on overnight orders
- Lower gap risk vs micro-caps

### 2.3 Already-explored adjacencies (priors)

Use these as quick-rejects / starting hypotheses. **Don't re-test from scratch.**

| Research # | What was tested | Relevance |
|---|---|---|
| 02, 04, 09, 10 | MQ portfolio (multi-month rebalance) | Long-only, daily OHLC; baseline for swing universe |
| 03, 05 | Breakout V3 daily consolidation | Daily breakout edge known |
| 06 | Stop loss / exit rule sweeps | Wide-stop daily systems prior |
| 12, 15, 17 | Combined swing + EOD scans + 60-min | 60-min entry on daily setup |
| 21, 22, 23, 24, 25, 26, 27 | F&O daily breakout / RS / Donchian | F&O daily edge well-mapped |
| 33 | EMA20/50 + Stoch on 30-min | 30-min trend-following |

The 75% WR bar is **higher than every published result in those folders** — meaning the winning setup will likely **stack confluence** (e.g., daily breakout + sector trend + earnings catalyst).

### 2.4 Success criterion — single ranked metric

Same Adjusted Win-Rate Score (AWS) as research/37, recomputed for daily candidates:

```
AWS = WR × ln(1 + trades) × min(PF / 2, 1.5) × min(Sharpe / 1.5, 1.5)
```

A candidate is ELIGIBLE only if it ALSO clears all five hard gates above.

---

## 3. The Plan — 5 stages, parallelizable

### Stage A — Universe + tooling (in_progress)
- Audit daily universe (drop sub-Rs.100 / sub-Rs.1Cr-daily-volume / illiquid F&O names).
- Build daily-bar data loader/enrichment module mirroring research/37 `_engine.py`. Add daily-timeframe primitives:
  - SMA/EMA(5/10/20/50/200), RSI(14), ATR(14), MACD, Bollinger Bands
  - Daily VWAP (typical price weighted)
  - Donchian channels (10/20/55)
  - Per-stock prior-day high/low/close, gap %, range %
- Build trade simulator with **multi-day hold**, daily TP/SL on close-to-close, optional intraday SL trigger.

### Stage B — Pattern families (sub-agent dispatchable)

Six pattern families, each tested as a standalone agent track:

1. **BTST (overnight)** — exit at next-day close. Entry: last-15-min strong move + closing near high/low + sectoral confirmation. Hold: 1 night.
2. **2-3 day swing breakout** — daily Donchian-20 / 52-week high break + volume + sector strength. Hold: 2-5 days, trail or fixed exit.
3. **5-day mean reversion** — daily RSI(14) < 25 in long-term uptrend (above 200 SMA). Hold: 5-10 days, exit on RSI > 50 or fixed.
4. **Earnings post-drift** — public earnings dates; long if positive surprise > X%, hold 5-20 days. (Need earnings calendar — may pre-date some data.)
5. **Pair trading (stat-arb)** — long top sector laggard / short top leader, mean-revert toward sector mean. Hold: 3-15 days.
6. **Weekly trend continuation** — weekly close above prior 8-week high (or below 8-week low) → enter on next Monday open, hold 5-10 days.

Each pattern → per-stock screen → diamond cohort → param sweep → walk-forward.

### Stage C — Confluence stack
Layer NIFTY 50 daily regime (above/below 50-day EMA, ADX>20, weekly close direction) on Stage B survivors.

### Stage D — Walk-forward validation
- Train: 2015-01 to 2023-12-31 (9 years for daily — enough samples)
- Test: 2024-01-01 to 2026-03-19 (~2.25 years held out)
- Pass: train WR ≥ 75% AND test WR ≥ 70% AND test PF ≥ 1.8 AND test n ≥ 30 trades

### Stage E — Combined portfolio backtest
If 2+ patterns survive, run the same kind of combined-portfolio walker as research/37 Stage 13, with realistic concurrency, per-trade Rs.3K risk cap, multi-day position holds, and CNC-like cost model (different from MIS — see section below).

---

## 4. Cost model — different from intraday

CNC equity (delivery / multi-day hold):
- Brokerage Zerodha: **Rs.0** for delivery (CNC)
- STT: **0.1% on buy + 0.1% on sell** (delivery — much higher than intraday's 0.025% sell-only)
- Exchange + GST + Stamp: ~0.005% per side
- **Round-trip cost: ~0.21%** (dominated by STT) — much higher than intraday's 0.10%.

For F&O carry-forward (futures intraday → next-day):
- Brokerage Rs.20 / 0.03% per side, whichever lower
- STT: 0.0125% sell side
- Exchange + GST + Stamp: ~0.003% per side
- **Round-trip cost: ~0.06%** for futures (much lower than CNC)

Implication: **F&O futures carry-forward is materially cheaper than cash CNC delivery**. The 86-stock F&O universe is the natural starting point unless a CNC system delivers significantly better edge.

---

## 5. Status — live event log

State header:
- **Phase:** Stage A — universe inventory + tooling scaffolding
- **Started:** 2026-05-06

Event log:

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-05-06 — | Folder + STATUS doc created | research/39_carry_forward_75wr_quest/ scaffolded |
| 2026-05-06 — | Locked in research/37 + research/38 winners | "Intraday Final Live Setup" doc written |
| 2026-05-07 — | Patterns 3+4 sweep launched (sub-agent) | F&O 76 stocks, 6yr train + 2.25yr test |
| 2026-05-07 — | Patterns 3+4 walk-forward complete (sub-agent) | 0 survivors; near-miss on Pattern 4 LONG |
| 2026-05-07 — | Patterns 1+2+6 sweep launched (this agent) | F&O 78 stocks, _engine_daily.py |
| 2026-05-07 — | Pattern 6 (Weekly) sweep done | 22,400 rows; 0 survivors at gates |
| 2026-05-07 — | Pattern 1 (BTST) sweep done | 74,880 rows; 0 survivors at gates |
| 2026-05-07 — | Pattern 2 (Swing breakout) sweep done | 116,832 rows; 0 survivors at gates |
| 2026-05-07 — | Pattern 4b (PEAD with REAL earnings dates) | 9,600 cells; 0 walk-forward survivors |

### Live findings — accumulative log

- **2026-05-07: Patterns 3 (RSI mean-reversion) + 4 (Earnings drift) — both NO-GO** (sub-agent run).
  - Pattern 3 LONG best: rsi<25 + above SMA200 + tp=7%/sl=5%/hold=5d -> WR 58.1%, n=31. Far below 75%.
  - Pattern 3 SHORT best: rsi>80 + below SMA200 + tp=5%/sl=3% -> WR 54.5%, n=11. Far below 75%.
  - Pattern 4 LONG near-miss: gap=5%, vol>=1.5, results-months, tp=8%/sl=7%/hold=15d ->
    test WR 75%, PF 2.25, DD 7.1%, +19.3% — but only n=12 trades (fails n>=30 gate).
    Cost-robust at 0.20% RT (test WR 75%, PF 2.11).
  - Pattern 4 LONG high-N candidate: gap=3%, tp=8%/sl=7%/hold=10d ->
    test n=48 WR 62.5%, PF 1.68, DD 27.4%, +51% TotalRet. Misses WR + DD gates.
  - Pattern 4 SHORT: 0 cells pass gates; best test WR 58%.
  - Verdict: PEAD edge is real but capped at ~60-65% WR on daily F&O. 75% bar would need
    real earnings dates, surprise scores, or sector-confluence overlay.
  - Files: results/03_rsi_mr_*.csv, 04_earnings_*.csv, *_diamonds.txt.

- **2026-05-07: Pattern 4b (PEAD with REAL earnings dates) — NO-GO**.
  Full write-up: `results/RESULTS_PEAD_REAL_DATES.md`.
  - Real earnings dates from yfinance .NS: 3,666 announcements, 2,635 mapped
    in-period events (vs ~190 from gap+volume proxy in pattern 4 — 14x lift).
    Surprise % data available on 3,455 events (pre-2018 dropped).
  - **Verdict: real dates LIFTED N (n=31-36 in tests, vs n=12 proxy near-miss),
    but did NOT lift the WR ceiling. PEAD WR caps at ~65-70% on Indian F&O
    daily with all sensible filters (above SMA200, surprise>=0/+5%/+20%,
    vol>=1.5x).** 9,600 cells swept; 0 pass walk-forward gates.
  - Best train_wr: 69.3% (n=75, surprise>=20% LONG cell). Below 75% bar.
  - Best test cell: train_wr 52% / **test_wr 74.2%** PF 4.12 DD 14.8%
    +117% return (n=31, surprise>=5% cell). Beautiful test stats but
    train fails — regime-shift pattern, not persistent.
  - Cost-stress at 0.30% RT: top picks remain WR 72-74% PF 3.4-4.4. Not
    a cost problem, a structural WR ceiling.
  - Validated test cohort (100% WR n>=4 across top 3 candidates):
    HAL, TRENT, BANKBARODA, FEDERALBNK. Plus IOC at 83% (n=6).
  - Files: results/04b_pead_*.csv, RESULTS_PEAD_REAL_DATES.md.

- **2026-05-07: Patterns 1 (BTST) + 2 (Swing breakout) + 6 (Weekly continuation) — all NO-GO**.
  Full write-up: `results/RESULTS_BTST_SWING_WEEKLY.md`.
  - Pattern 1 BTST: 74,880 rows swept across 78 F&O stocks. Best test WR 68% (SIEMENS,
    n=44) but train was 49% — non-stationary. Walk-forward consistent best: TATAPOWER
    long ~60% train, 62% test; RELIANCE long 63%/59%. **MARUTI long is cost-resilient**:
    20/20 top setups still positive at 0.20% RT cost (54-60% test WR).
  - Pattern 2 Swing breakout (Donchian-20/55 + EMA50-rising + volume): 116,832 rows.
    Best test WR ~60% (M&M, BPCL, BEL n=20 each), train 46-52%. No setup survives
    0.20% stress cost in top-20.
  - Pattern 6 Weekly continuation (8/12-week Donchian breakout): 22,400 rows. Best
    consistent: MCX long W8 65% test / 58% train, +16.6% test ret PF 2.33. Weekly
    sample size is too low (often <30 train trades).
  - Verdict: 75% walk-forward bar not crossed by any single-signal carry-forward
    pattern on F&O daily. Cohorts that show edge but cap below the bar:
    BTST: SIEMENS, DRREDDY, BAJFINANCE, MARUTI, TATAPOWER, RELIANCE, ADANIENT, IRCTC, HAL.
    Swing: M&M, BPCL, BEL.
    Weekly: MCX, RELIANCE, INDUSINDBK (short), SIEMENS.

---

## 6. Crash recovery — how Arun resumes without me

If conversation dies mid-flight:

1. **Find where we are**
   ```powershell
   ls research/39_carry_forward_75wr_quest/results/
   Get-Content research/39_carry_forward_75wr_quest/CARRY_FORWARD_75WR_DAILY_SWEEP_STATUS.md | Select-String "Phase:"
   tail -30 research/39_carry_forward_75wr_quest/logs/*.log
   ```
2. **Resume scripts** (each script supports `--skip-done` resume):
   ```powershell
   python research/39_carry_forward_75wr_quest/scripts/00_universe_audit.py
   python research/39_carry_forward_75wr_quest/scripts/01_pattern_screen.py --pattern btst
   ```

---

## 7. Files (output map)

| File | Purpose | Committable? |
|---|---|---|
| `CARRY_FORWARD_75WR_DAILY_SWEEP_STATUS.md` | This live-status doc | yes |
| `scripts/_engine.py` | Daily-bar loader + simulator (extends research/37 _engine) | yes |
| `scripts/00_universe_audit.py` | Stage A daily-universe filter | yes |
| `scripts/01_btst.py` | BTST pattern (overnight) | yes |
| `scripts/02_swing_breakout.py` | Daily Donchian / 52-wk break | yes |
| `scripts/03_rsi_mean_reversion.py` | RSI<25 in uptrend | yes |
| `scripts/04_earnings_drift.py` | Post-earnings continuation (needs earnings dates) | yes |
| `scripts/05_pair_trading.py` | Sector pair stat-arb | yes |
| `scripts/06_weekly_trend.py` | Weekly close-based continuation | yes |
| `results/00_universe_audit.csv` | Liquid daily cohort | yes |
| `results/01..06_*_perstock.csv` | Per-stock screens | yes |
| `results/01..06_*_ranking.csv` | Sweep results | yes |
| `results/07_walk_forward.csv` | Walk-forward validated | yes |
| `results/08_portfolio_summary.txt` | Combined backtest | yes |
| `CARRY_FORWARD_75WR_DAILY_SWEEP_RESULTS.md` | Final write-up | yes (at end) |

---

## 8. Findings — accumulative

*(filled in as Stage B+ produces partial results)*

- (none yet)

---

## 9. Validation cases (real-world exemplars to pattern-match against)

These are concrete, observed setups that the patterns should detect when
the sub-agent backtests run on the relevant date. Use as sanity checks.

### Case 1 — BOSCHLTD, 2026-05-06 → 2026-05-07 (BTST)

User-flagged on 2026-05-07 at ~10:15 IST while live. Pattern: **late-day
volume spike + strong-close + prev-day-high break + overnight continuation**.

**On 2026-05-06 (entry day):**
- Sideways consolidation through morning at ~36,000-36,300
- ~14:30-15:25 IST: volume bars expand visibly above session average
- Price breaks day's high AND **prior-day high** in the last 30-60 min
- Closes near day's high at ~36,520

**On 2026-05-07 (exit day):**
- Gap up open above 37,000
- Continues higher through morning
- 10:15 IST printing **37,915 (+4.1% from prev close)**

**What the BTST detector must capture:**
1. Last 6 bars of session (~14:55-15:25 IST) average volume ≥ N × 20-day-avg of those bar positions (volume confirmation)
2. Session close within X% of day's high (strong-close)
3. Session close > prior-day's high (multi-day breakout)
4. Optional: stock above 50-day SMA (uptrend filter)
5. Entry: 15:25 IST close OR market-on-open next session
6. Exit: next-session close (or trailing stop if extending the hold)

When sub-agent 1 produces its BTST results, **verify BOSCHLTD 2026-05-06
fires this signal AND produces a winning trade on 2026-05-07.**

### Case 2 — GODREJCP, 2026-05-07 (first-30-min breakdown SHORT)

User-flagged on 2026-05-07 mid-morning while live. **Direction-mirror of
Case 1 with different timing**: first 30-min-of-today breakdown on volume,
not last 30-min-of-prev-day momentum.

**On 2026-05-07:**
- Stock had ranged 1075-1110 over prior multiple sessions
- First 30-min candle (09:15-09:45 IST): big volume spike
- Closes BELOW prev-day's high AND breaks DOWN through 1075 prior-range
  support
- RSI(14) on 30-min ≈ 28 (oversold from breakdown momentum)
- Now printing 1043 (down ~3% intraday) at observation time

**What the FIRST-30-MIN BREAKDOWN detector must capture (NEW pattern, not
the same as Case 1):**

1. First 30-min bar (or first 6 × 5-min bars) volume ≥ N × 20-day-avg of
   first-30-min volume on that stock
2. First 30-min bar closes **below prior-day's high** (failed range-up)
3. Optional: also closes below recent N-day low (range-breakdown
   confirmation)
4. Optional: RSI(14) on the close < 35 (oversold momentum)
5. Optional: NIFTY 50 also below own prev-day-high at 09:45 IST
6. Entry: 09:45 IST close OR 09:50 IST open (next bar)
7. Exit: TP/SL (favorable RR — TP > SL since this is a momentum-extension
   setup) OR session close at 15:25 IST

### Why these are different patterns from our locked systems

GODREJCP is **NOT in the 25-stock Diamond Short cohort** — it would need
its own per-stock screen + diamond cohort to qualify. Also the timing
(first-30-min) and reference level (prev-day-high break-down) differ
from Diamond Short's "stock weak at 09:45 + RSI<40 + below own VWAP +
NIFTY weak" mechanic.

**Implications for research/39:**
- Case 1 (BOSCHLTD): captured by sub-agent 1's BTST pattern (Pattern 1)
- Case 2 (GODREJCP): NOT captured by current pattern set — would require a
  new "first-30-min range-breakdown" pattern, OR widening the BTST
  sub-agent's signal to include first-30-min entries with prev-day-high
  as the reference level
- Track as a **deferred follow-up pattern** ("Pattern 7 — First-30-min
  range-breakdown") if sub-agent 1 misses it
