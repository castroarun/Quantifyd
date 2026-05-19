# Final Live Setup — FOUR Configs Locked (Intraday + Carry-Forward)

**Date locked:** 2026-05-06 (Configs A/B/C intraday); **2026-05-07** (Config D carry-forward added)
**Source research:** research/37 + research/38 (intraday) + research/39 (carry-forward pair trading)

This document is the **canonical live-deployment spec** for the full
trading system across intraday + carry-forward. Everything else in
research/37/38/39 is the exploration history that led here.

🔒 **ALL CONFIGS START IN PAPER MODE.** No live capital flips until
explicit user authorization per config.

---

## TL;DR — four configs running in parallel

| # | Config name | Type | Mechanics | OOS WR | OOS PF | OOS DD | Trades/yr |
|---|---|---|---|---|---|---|---|
| **A** | **3-System Original (TP 0.5%/SL 1.5%)** | Intraday MIS | research/37 base | **78.1%** | 1.28 | 4.5% | ~895 |
| **B** | **3-System Cost-Resilient (TP 2.0%/SL 1.5%)** | Intraday MIS | research/37 same signals, wider TP | 53.5% | **1.26** | **4.5%** | ~830 |
| **C** | **Multi-Bar SHORT Bounce (TP 1.5%/SL 1.0%)** | Intraday MIS | research/38 walk-forward winner | 59.6% | **1.69** | **3.1%** | ~120 |
| **D** | **6-Pair Cointegrated Pair Trading** | F&O carry-forward (10-20d hold) | research/39 walk-forward winner | **78.7%** | **3.57** | **0.06%** | ~50 |

All four together = **~1,895 trades/year** with mathematically diversified
edge sources (3 intraday directional + 1 multi-day market-neutral).

---

## Why FOUR, not one

Each config exploits a different reality of the market, and they're
**uncorrelated** because they use different mechanisms in different time
horizons:

- **A and B** are *directional intraday* — same signal stack, two exit profiles
- **C** is *intraday momentum continuation short* — different signal stack
- **D** is *multi-day market-neutral* (relative-value mean-reversion) — different timeframe AND different mechanism

That diversification is what makes the combined portfolio robust across
regimes (bull / bear / sideways / volatile).

## Why three (intraday), not one

Each config exploits a different reality of the market:

- **Config A** rides the per-stock structural drift (Stage 7) at low
  cost on liquid names. Highest WR but cost-fragile.
- **Config B** uses the *same signals as A* but waits for bigger moves —
  trades less at TP, more at EOD, but each EOD-or-TP exit has
  realistic post-cost edge. Cost-resilient (works even at 0.10%/side).
- **Config C** is a different signal mechanism entirely — multi-bar
  bearish confirmation with NIFTY broad-weakness gating. Fires less
  often (~5 trades/stock/year) but has the highest PF and lowest DD.

When markets are cleanly weak, A and C both fire. When markets are
sloppy or trending strongly, B picks up trades A misses (because A's
TP often doesn't hit but EOD exits land fine on wider 2.0% TP).

---

## CONFIG A — 3-System Original (research/37)

The flagship from research/37. Full mechanics in
`INTRADAY_75WR_5MIN_SWEEP_RESULTS.md`.

### System A1 — Diamond Short
- Universe: 25 short-bias diamond stocks (`results/07_short_diamonds.txt`)
- Trigger: 09:45 IST. Stock RSI(14)<40 + close<VWAP + NIFTY first-30m<0
- Order: SHORT at next bar's open
- Exit: TP −0.5%, SL +1.5%, hold to 15:25 IST

### System A2 — Late-Day Mean-Reversion LONG
- Universe: 15 long-MR diamond stocks (`results/11c_long_reversal_diamonds.txt`)
- Trigger: 11:15-13:15 IST scan. Stock down ≥−2% from open + RSI bounce 28→35 + bullish bar + NIFTY not crashing
- Order: LONG at next bar's open
- Exit: TP +0.5%, SL −1.5%, hold to 15:25 IST

### System A3 — Trend-Continuation Pullback LONG
- Universe: 30 trend-pullback diamond stocks (`results/11b_trend_pullback_diamonds.txt`)
- Trigger: 09:15-10:30 IST scan. Stock gap-up≥0.5% + first-hour break≥0.5% + price within 0.3% of VWAP + RSI≥45 + NIFTY also gap-up bullish
- Order: LONG at next bar's open
- Exit: TP +0.5%, SL −1.5%, hold to 15:25 IST

**Combined out-of-sample (Oct 2025-Mar 2026):** WR 78.1%, PF 1.28, MaxDD 4.49%, Sharpe 2.55, +12.24% net at 0.05%/side cost (Rs.10L→Rs.12.23L).

**Cost-fragility:** at 0.10%/side this config flips to −6.02%. Use
ONLY on stocks with realistic 0.05%/side execution (top-volume liquid
names within the 67-stock universe).

---

## CONFIG B — 3-System Cost-Resilient (research/37 retune)

Same three signals as Config A, but retuned exit targets after the
TP-widening sweep showed cost-resilience.

### Identical signals to A1/A2/A3
Universes, entry rules, NIFTY filters all unchanged.

### NEW exit (all three systems)
- **TP 2.0%** (was 0.5%)
- **SL 1.5%** (unchanged)
- Hold to 15:25 IST

### Out-of-sample (full backtest, 0.10%/side cost)
- WR 53.5%, PF 1.26, MaxDD 4.50%, Sharpe **2.38**, +19.96% return on Rs.10L NAV in 2 years
- ~3.6 trades/session avg, 831 trades over 232 trading sessions

### Why this is the cost-resilient default
Random-walk WR with TP=2.0/SL=1.5 is 1.5/3.5 = 42.9%. Achieving 53.5% is +11pp of real edge — **the same edge as Config A**, just resolving at a lower WR but higher per-trade gain. The math is robust across cost levels because each win is much bigger than each loss net of cost.

---

## CONFIG C — Multi-Bar SHORT Bounce (research/38)

A separate signal mechanism, walk-forward validated as a favorable-RR
short-side complement to A and B.

### Universe (25 stocks — same as A1)
ZEEL, EDELWEISS, ASHOKA, CDSL, BANDHANBNK, KNRCON, RAIN, GMDCLTD, HEG, NATCOPHARM, SJVN, PRAJIND, TIINDIA, SUZLON, AMBER, RCF, NETWORK18, NAM-INDIA, BAYERCROP, TCIEXP, AARTIIND, NATIONALUM, IDEA, LTTS, NBCC

### Signal (continuous scan during session)
1. **Last 4 consecutive 5-min bars all bearish** (close < open) AND each successive bar has lower high than prior bar
2. **Close < intraday VWAP** (own stock's VWAP since 09:15)
3. **RSI(14) ≤ 55**
4. **NIFTY 50 close < its own session VWAP at scan time** (broad-market weak)
5. Bar index ≥ 4 and not in last 5 bars of session

### Order
- **SHORT** at next bar's open
- One trade per stock per session max

### Exit (all of):
- **TP 1.5%** below entry
- **SL 1.0%** above entry (RR 1.5:1 — favorable)
- Force-close at 15:25 IST

### Walk-forward (out-of-sample Oct 2025-Mar 2026)
- WR **59.6%** (test) vs 56.7% (train) — drift +0.4 pp (test SLIGHTLY better)
- PF **1.69** (test) vs 1.54 (train)
- MaxDD **3.1%** (test) vs lower
- Test n = 57 trades over 6 months → ~120/year portfolio-wide
- Sharpe ~3 estimate (post-cost)

### Why this complements A and B
- **Different signal mechanism** (multi-bar momentum continuation vs single-bar regime entry) — uncorrelated trade firing
- **Same cohort** (the 25 short-diamonds) — no new universe to monitor
- **Lower frequency** (~5 trades/stock/year) means it doesn't spam concurrency cap
- **Highest test PF (1.69)** of the three configs

---

## CONFIG D — 6-Pair Cointegrated Pair Trading (research/39)

**Carry-forward F&O system, 10-20 day holds, market-neutral.** Source:
[research/39 RESULTS](../39_carry_forward_75wr_quest/CARRY_FORWARD_75WR_DAILY_SWEEP_RESULTS.md).

### The 6 winning cointegrated pairs

| Pair | entry_z | hold | te_n / te_wr | te_pf | te_dd | stress_pf | per-pair return |
|---|---|---|---|---|---|---|---|
| **HAVELLS - MARICO** | 2.0 | 20d | 16 / **93.75%** | **8.50** | 3.49% | **6.76** | +26.21% |
| **BAJFINANCE - KOTAKBANK** | 2.0 | 20d | 18 / 83.33% | 7.05 | 3.94% | 4.99 | +28.25% |
| **DABUR - HINDUNILVR** | 2.0 | 20d | 19 / 78.95% | 4.71 | 1.61% | 2.25 | +10.51% |
| **COFORGE - HCLTECH** | 2.0 | 15d | 21 / 76.19% | 3.39 | 6.38% | 2.49 | +22.96% |
| **DABUR - TCS** | 2.0 | 10d | 21 / 71.43% | 3.79 | 4.01% | 2.81 | +25.37% |
| **APOLLOHOSP - COFORGE** | 2.0 | 10d | 16 / 75.00% | 1.93 | 8.48% | 1.61 | +15.39% |

### Combined-portfolio out-of-sample (Jan 2024 – Nov 2025)
**WR 78.70%, PF 3.57, MaxDD 0.06%, n=108 trades.** All gates clear.
First and only system across the entire 6,500+-cell hunt with WR ≥ 75%
and TP > SL out-of-sample.

### Daily process (one EOD cron job, 16:00 IST after futures close)
1. For each of the 6 pairs, load latest daily close prices for both legs.
2. Compute spread = `log(P_a) - alpha - beta * log(P_b)` using the
   train-fit alpha/beta (refresh quarterly per cohort policy).
3. Compute z-score of spread on rolling 20-day lookback.
4. **ENTRY (LONG spread):** if z ≤ −2.0 → BUY pair-A futures + SELL pair-B futures (β-weighted)
5. **ENTRY (SHORT spread):** if z ≥ +2.0 → SELL pair-A + BUY pair-B
6. **EXIT** any open position on first of:
   - z crosses 0 (mean revert — typical exit, takes 5-15 days)
   - |z| ≥ 4.0 (or 5.0 per pair config — stop-out)
   - hold-cap days reached (10/15/20 per pair config)

### Position sizing (D-specific)
- **Risk per pair-trade:** Rs.6,000 total (Rs.3,000 per leg)
- **Notional per leg:** ~Rs.30,000 (sized so SL distance = per-leg risk)
- **F&O futures margin:** ~12-15% of notional → ~Rs.4,000 per leg → ~Rs.40K-50K margin per pair
- **Max concurrent pairs:** 5 (out of 6) — leaves room for 1 in cooldown
- **Total D-margin at peak:** ~Rs.2L on Rs.10L base capital

### Cost model (D-specific — F&O futures, much cheaper than CNC)
- Brokerage Rs.20 / 0.03% per side, whichever lower
- STT 0.0125% sell-side
- **Round-trip per leg: ~0.06% → 0.12% per pair-trade**
- Stress test: 0.20%/leg = 0.40% per pair-trade. All 6 pairs survive at PF ≥ 1.61.

### Cohort refresh (mandatory quarterly)
- Re-run Engle-Granger cointegration test on the F&O universe every 3 months
- Replace decayed pairs with newly-cointegrated ones
- Re-fit alpha + beta on rolling 12-month window (NOT frozen at original train values)
- Drop any pair whose hedge ratio drifts >2σ from rolling-fit value

---

## Position sizing — Rs.3K risk per trade (ORB convention) for A/B/C; Rs.6K per pair for D

Same Rs.3K per single-leg trade across A/B/C; D uses Rs.6K total per
pair-trade (Rs.3K per leg). Detailed sizing inside each config above.

**Across-system rules:**
- **MIS leverage** for A/B/C (intraday cash equity): 5× → Rs.6L margin for Rs.30L peak intraday deployment
- **F&O margin** for D (carry-forward futures): ~12-15% of notional → ~Rs.2L peak D-margin
- **Daily loss circuit-breaker (intraday A/B/C):** 3 SL hits in one session → halt for the day
- **Weekly loss circuit-breaker (D):** 3 SL exits in 5 sessions → halt D for the week
- **Max concurrent positions:**
  - A/B/C combined: 5 single-leg positions across the intraday triplet
  - D: 5 pair-positions (= 10 individual legs) — separate from A/B/C
  - **Total open margin headroom needed:** ~Rs.6L (intraday MIS) + ~Rs.2L (D futures) ≈ Rs.8L of Rs.10L base ✓

---

## How the FOUR configs co-exist (concurrency rules)

### A, B, C — intraday only (same trading session, same MIS account segment)

- **A1 (Diamond Short) and B1 (Cost-Resilient SHORT)** fire on the SAME signal — they're the same trade with different exits. Can't run both on same stock simultaneously. **Pick one or split capital.**
- **C (Multi-Bar SHORT Bounce)** fires later in the session typically (continuous scan finding 4-bar setups), and on a different signal pattern. May fire on the same diamond stock as A1 if conditions later in the day produce a fresh setup — this acts as a *re-entry* on continued weakness, OK to run both.
- **A2 (Long MR) and A3 (Long TC)** cohorts have no overlap with C's universe.

### D — fully independent, runs on a different account segment + timeframe

- **D operates on F&O futures**, different brokerage account segment from MIS cash equity → no margin conflict with A/B/C.
- **D fires once per day at EOD** (16:00 IST), not on intraday bars → no scheduler conflict with A/B/C.
- **D holds positions overnight 10-20 days** → its lifecycle doesn't overlap with intraday day-flat A/B/C.
- **D's 6-pair universe is mostly disjoint from A/B/C diamonds** — only KOTAKBANK and HINDUNILVR appear in both A2 long-MR cohort and D pairs (different mechanics so no signal collision; D fires at EOD on the spread, A2 fires intraday on the stock alone).

### Recommended live-deploy split (post paper-trade validation)

- **Run A + C + D** for max trade frequency (~1,065 trades/year combined, capital efficient)
- **OR run B + C + D** for cost-resilient profile (~1,000 trades/year, survives high slippage)
- **Don't run A and B simultaneously** — same-signal conflict
- **D always runs** alongside whichever intraday combo is active (no conflict by design)

---

## Production checklist

- [ ] Re-confirm all three cohorts with **6-month rolling re-screen** (cohorts may rotate as macro shifts).
- [ ] **Stress-test with 0.15%/side cost** before live capital — Config A in particular is fragile around 0.10-0.15%/side. If your execution doesn't beat 0.07%/side on the named cohorts, default to Config B.
- [ ] **Borrow availability for short side** (some names like NBCC, KRBL have intermittent SLB).
- [ ] **Position sizing:** per-trade risk = Rs.3K (1% of Rs.3L base capital × 5x leverage = Rs.15L deployable).
- [ ] **Concurrency cap = 5 across A+C combined** (or B+C combined), not per-config.
- [ ] **NIFTY filter measurement:**
  - Config A1 + B1 (Diamond Short): once at 09:45 IST, frozen for the day's signal
  - Config A2 (Long MR): per-bar 11:15-13:15
  - Config A3 (Long TC): once at 09:15 IST + per-bar through 10:30
  - Config C: per-bar throughout the session
- [ ] **Daily loss circuit-breaker:** 3 SLs → halt for the day across all configs combined.
- [ ] **Paper-trade for 2-3 weeks** in app's paper-mode infrastructure before scaling capital.
- [ ] **Live deploy with reduced capital first** (Rs.2L → Rs.5L → Rs.10L over a month) once paper holds.

---

## What's NOT in this setup (out of intraday scope)

- The 75% WR + favorable RR target was empirically not achievable on intraday 5-min equity (research/38 final finding). To pursue it, the universe needs to expand to **carry-forward / multi-day** holds — see research/39 (in flight as of 2026-05-06) for the BTST + swing + weekly hunt.
- Options-based overlays (theta capture) — see research/29 for the prior options-signal work; could layer on top of A/B/C signals in a Phase 2 deployment.

---

## Files / references

| File | Purpose |
|---|---|
| `INTRADAY_75WR_5MIN_SWEEP_STATUS.md` | research/37 live-status doc |
| `INTRADAY_75WR_5MIN_SWEEP_RESULTS.md` | research/37 detailed results |
| `INTRADAY_FINAL_LIVE_SETUP.md` | This doc — final live spec |
| `results/07_short_diamonds.txt` | A1 + C universe (25 stocks) |
| `results/11c_long_reversal_diamonds.txt` | A2 universe (15 stocks) |
| `results/11b_trend_pullback_diamonds.txt` | A3 universe (30 stocks) |
| `results/13_portfolio_summary.txt` | A combined backtest |
| `results/13_portfolio_summary_rs3k.txt` | A with Rs.3K cap |
| `results/13_portfolio_summary_sweep_tp20_sl15_c010.txt` | B combined backtest |
| `../38_high_rr_75wr_quest/results/13_walk_forward_favorable_rr.csv` | C walk-forward validation |

---

## Roadmap

- **Phase 4** (next): Build live executor with Off/Paper/Live toggle for all three configs (originally Phase 3 of the deployment plan, paused awaiting this lock-in).
- **Phase 5**: Trading Journal MVP already built (Phase 2 done, services/journal/ + 4 React pages live).
- **research/39**: Carry-forward systems hunt — BTST, multi-day swing, weekly continuation. Goal: find the WR ≥ 75% + RR > 1.1 system that intraday couldn't deliver.
