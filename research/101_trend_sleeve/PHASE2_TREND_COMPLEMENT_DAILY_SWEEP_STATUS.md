# Phase 2 — Trend Sleeve that Complements the Naked-Straddle Flow

STATUS: RUNNING (G1 gate done; dual-SuperTrend build = next)

## The Ask

**What Arun asked:** Build a trend-following sleeve (any underlying — NIFTY, BankNifty,
SENSEX, F&O stocks; any structure — futures, hedged futures (covered call / married put),
or debit option spreads) that **complements the naked short-straddle flow**. The straddle
is short-vol and dies in big moves; a long-vol / trend sleeve should earn during those
same tail events and cushion the straddle's overnight/crash risk. **Objective: maximize
the COMBINED (straddle + trend) portfolio Calmar.** Scoping (2026-08-05): universe =
indices + F&O stock basket; instruments = all three; objective = combined Calmar.

**What we're actually testing:** Is there a trend sleeve with (a) a real standalone edge
and (b) P&L that is negatively/low-correlated with the short straddle — especially firing
in the straddle's worst months — such that the blended book has a higher Calmar and a
smaller drawdown than the straddle alone?

## Candidate systems & findings so far

### C0 — Naive Donchian-20 long/short (G1 cheap gate) — **FAILS standalone**
NIFTY+BankNifty equal-weight, daily, causal 1-day lag, 2018-2026:
- Standalone: ann. Sharpe **−0.04**, ~−0.7%/yr, −69%-pts maxDD → **no edge** (Indian
  indices whipsaw a naive breakout).
- Monthly corr vs naked DTE-3 straddle = **−0.13** (weakly complementary — encouraging).
- Cushion test: only 2 of the straddle's 6 worst months were cushioned → **unreliable**.
- **Verdict:** a naive trend signal is not enough. Need a better-engineered trend system.

### C1 — Long BankNifty monthly options (long-vol cushion) — **no standalone edge, real cushion**
Buy ATM straddle → 5% strangles, ~30 DTE, carried to expiry, per lot, 2016-2026:
- ~break-even standalone (M/SD 0.03-0.09); **~100% of P&L is March-2020 (COVID)**.
- Loss capped at premium; big payoffs land in the straddle's worst months → valid tail
  cushion, but **negative carry** (pure insurance). Trend-following should beat it by
  capturing the same convexity with positive carry.

### C2 — Dual-SuperTrend master/child futures + option hedge (PRIMARY build — NEXT)
**Arun's spec (NIFTY daily, EOD-close actions, 7+ yrs; ST params to be OPTIMIZED):**
- Two SuperTrends, e.g. ST(7,2)=child (fast), ST(7,5)=master (slow). Optimize (ATR len,
  mult) for both.
- **Master signal** sets direction: price/close vs master ST → long or short NIFTY futures.
- **Entry:** take futures in the master's direction; reverse on master flip.
- **Hedge:** on a CHILD reverse signal against the position, sell an ATM option to cushion
  the pullback — short CE if long futures (covered call), short PE if short futures.
- **Pyramid:** on a new child signal back IN the master's direction, add the next futures
  lot (and lift/adjust the hedge); scale in as the trend resumes.
- **Hold** while the master signal holds; exit all on master flip.
- **Data reality:** only EOD open/close option premiums for 7+ yrs → run on DAILY candles,
  all action/management on EOD close.
- **Reference implementation for correct entry/exit/management:** `services/maruthi_strategy.py`
  (our existing master→child dual-SuperTrend). Read it before coding C2.
- Then measure complementarity with the naked straddle and the COMBINED Calmar.

## Plan (stage-gated)

1. **G1 (done):** cheap trend probe → naive trend fails; corr −0.13 says the *right* trend
   sleeve could still complement. Proceed to a real system (C2).
2. **G2:** build + optimize C2 (dual-ST) on NIFTY daily, 7 yrs, standalone first
   (net-of-cost, causal). Gate on a real standalone Calmar/Sharpe.
3. **G3:** if C2 has a standalone edge, blend with the naked-straddle monthly P&L →
   combined Calmar, drawdown, correlation. Optimize the blend weight.
4. **G4:** extend the winning structure across BankNifty/SENSEX + the F&O stock basket
   (the "correlated set"); pick the diversified sleeve that best lifts combined Calmar.
5. Futures vs hedged-futures vs debit-spread variants compared at G2/G3.

## Straddle reference series
Naked DTE-3 monthly P&L from `/tmp/realistic_trades.json` (dte3 key) — the flow to
complement. (Honest, look-ahead-free, per research/100.)

## Files
| File | Purpose |
|---|---|
| `scripts/phase2_g1.py` | G1 Donchian probe + straddle correlation (done) |
| `scripts/banknifty_long.py` | C1 long-options cushion test (done) |
| `scripts/dual_supertrend_daily.py` | C2 dual-ST build (TODO — next) |
| `results/RESULTS.md` | verdicts (pending) |

## Crash recovery
- G1/C1 are complete (numbers above). C2 not started.
- To build C2: read `services/maruthi_strategy.py` for the master/child logic, implement on
  NIFTY daily bars (market_data_unified NIFTY50) + nse_options_bhav for the ATM hedge
  premiums, sweep ST params, run net-of-cost, then blend with the straddle series.

---

## Findings (G2/G3 — 2026-08-05)

### C2 dual-SuperTrend futures core (NIFTY daily, 2018-2026) — WEAK standalone
Swept master/child ST params (period 7/10/14 × mult 3/4/5 × child 7/10 × 1.5/2/2.5).
Best = ST(14,3) master / ST(7,2.5) child: **Sharpe 0.22, Calmar 0.06**, +5,667 pts/8.3y
(~₹4.25L/lot) on a −11,294-pt maxDD. Most configs ≈0 or negative. The user's ST(7,5)/ST(7,2)
was not near the top. → Naive daily trend on NIFTY has no meaningful standalone edge, same as
the Donchian G1 probe (Sharpe −0.04).

### G3 blend with the naked straddle — DOES NOT HELP
Monthly corr(trend, straddle) = **+0.03** (not negative). Adding the trend sleeve monotonically
LOWERS combined Calmar (2.66 straddle-alone → 2.15 @1 lot → 1.00 @3 → 0.23 @10) and DEEPENS
drawdown. **Best blend = 0 trend lots.** The futures dual-ST is not a complement.
(Note: the 2.66 straddle-alone Calmar is monthly-bucketed and flatters the naked book, whose
true tail is unbounded — but the RELATIVE result stands: trend adds risk, not diversification.)

### Interim verdict
The first two trend candidates (Donchian L/S, dual-ST futures) FAIL both gates — no standalone
edge AND no complementarity (corr ~0). The only thing that reliably fired in the straddle's worst
months was naive LONG options (C1) — but that's negative-carry insurance. So the "trend cushions
the straddle" thesis is NOT yet supported.

### Next candidates to mine (Phase 2 continues — my own systems)
1. **Dual-ST + option hedge (stage 2 of C2):** add the short-ATM-option hedge on counter-trend
   child flips + pyramiding — adds premium carry + cushions futures pullbacks. May rescue C2.
2. **Momentum-ranked F&O stock basket:** cross-sectional momentum has shown real edge elsewhere
   in this project (momentum-30) — a long basket is convex-ish and may correlate better with the
   straddle's tail than index trend.
3. **Cheap systematic long-vol / tail hedge** sized small (the C1 payoff profile, minimized carry)
   — accept it as insurance if it lifts combined Calmar net of its bleed.
4. **Weekly / higher-timeframe trend** (daily whipsaws killed the edge) and **multi-asset** (gold)
   for genuine diversification.
The gate stays: a candidate must EITHER carry its own edge OR show a real negative correlation in
the straddle's worst months. Corr ~0 sleeves are rejected.
