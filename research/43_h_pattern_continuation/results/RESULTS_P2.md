# H-Pattern Continuation — Phase 2 Results (confluence-filter gating)

**Verdict: the filters work — the prev-day-low break flips the raw breakeven H
into a cost-surviving edge, but only SHORT-side and only thinly. Best config =
short-only breakout|MM with `pdl + daily-downtrend`: short-net +0.060R/trade,
PF ~1.03, +1,130 total R over 32.6k short trades (2018→2026). It's a real,
positive-after-cost signal but marginal — better as a confluence overlay than a
standalone system. Longs should be dropped.**

Same engine/universe/costs as P1 (375 stocks + NIFTY50, 5-min, 6 bps round-trip).
Entry fixed to the P1 winner (breakout), targets MM & 3R, both directions tallied,
each filled trade routed into every filter-combo it satisfies. Full tables:
`ranking_p2_stocks.csv`, `ranking_p2_nifty50.csv`.

## Stocks cohort — breakout|MM by filter (sorted by short net expectancy)

| Filters | Trades | Shorts | Gross R | Net R | Long net | **Short net** | Net total R | PF |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **pdl + trend** | 57,014 | 32,608 | +0.210 | +0.020 | −0.034 | **+0.060** | **+1,130** | 1.03 |
| pdl | 113,340 | 68,040 | +0.199 | +0.010 | −0.018 | +0.028 | +1,092 | 1.01 |
| cpr_pos + trend | 101,203 | 55,198 | +0.183 | −0.015 | −0.064 | +0.026 | −1,501 | 0.98 |
| cpr_pos | 203,146 | 119,262 | +0.183 | −0.014 | −0.042 | +0.006 | −2,757 | 0.98 |
| trend | 171,224 | 88,090 | +0.153 | −0.045 | −0.079 | −0.013 | −7,750 | 0.93 |
| **none (= P1)** | 348,321 | 195,009 | +0.163 | −0.035 | −0.058 | −0.016 | −12,084 | 0.95 |
| cpr_w | 207,925 | 116,691 | +0.166 | −0.049 | −0.081 | −0.024 | −10,133 | 0.93 |

(3R target behaves almost identically — within ~0.01R of MM. `pdl+cpr_pos` ≡ `pdl`
and `all` ≡ `pdl+trend`, because breaking the prior-day low almost always already
puts price below the CPR bottom, so cpr_pos adds nothing on top of pdl.)

## What the data says

1. **`none` reproduces P1 exactly** (gross +0.163R, net −0.035R, 348k trades) —
   the P2 harness is consistent; the filter effects below are real, not artefacts.

2. **Prev-day-low break is the whole story.** It is the *only* single filter that
   turns short-net positive (+0.028R) and total-R positive. Every other filter on
   its own leaves the system net-negative. Intuition: the H is a continuation
   pattern; demanding that the continuation also takes out the prior session's low
   selects for genuine trend days and discards intraday chop.

3. **Daily-downtrend stacks cleanly on top.** `pdl+trend` halves the trade count
   (113k→57k) but more than doubles short-net (+0.028→+0.060R) and is the only
   variant with both net>0 and a positive total-R *and* the highest short edge.

4. **The edge is entirely short-side.** Longs are net-negative in every single
   filter combo (best −0.018R). The bullish H simply doesn't pay after costs;
   trade this **short-only**.

5. **CPR adds nothing here.** `cpr_pos` is redundant with `pdl`; `cpr_w` (narrow
   CPR / trend-day) actually *hurts* — it admits more chop than it filters.

6. **NIFTY50 stays untradable** at cash-equity costs (best short-net +0.046R but
   net deeply negative — the index's tiny % bands make 6 bps ≈ 0.5R). Index would
   need a futures cost model (~1 bp) to be re-examined.

## Bidirectional detail — best config (breakout|MM + pdl+trend)

| Side | Trades | Win % | Net exp R | PF |
|---|--:|--:|--:|--:|
| **Short** | 32,608 | **39.5%** | **+0.060** | **1.092** |
| Long | 24,406 | 36.7% | −0.034 | 0.949 |

Both directions trigger at ~similar (low) win rates; the edge is a
**low-WR / let-winners-run** profile. The asymmetry is in *payoff*, not hit-rate:
short winners run far enough to clear costs, long winners don't. **Longs are
net-negative (PF < 1.0) in every variant → trade short-only, drop longs.**

## Honest read on the edge quality

- **Positive but thin.** Short-only breakout|MM+pdl+trend: **39.5% win rate,
  +0.060R/trade, PF 1.092** (≈ +1,956 R over 32.6k short trades). Clears the
  "net > 0" bar but **not the PF > 1.2 gate**. Confirmed with per-direction
  win/loss tracking (re-run 2026-05-29) — combined PF 1.03 understated it because
  losing longs dragged the book.
- **Cost-sensitive.** At 6 bps the short edge is +0.06R; at ~8 bps it compresses
  toward zero. The conclusion depends on achieving ≤6 bps round-trip execution.
- **Aggregate is meaningful, per-trade is small.** +1,130 R over 32.6k shorts is
  a lot of total R, but it's spread very thin per trade — fragile to slippage,
  regime, and the fixed detection thresholds (unswept).

## Recommendation

- **Trade it short-only, breakout|MM, gated on `pdl + daily-downtrend`** — and
  treat it as a **confluence overlay / candidate signal**, not a standalone book,
  given PF ~1.03.
- **Drop longs.**
- **Highest-value next step:** the **detection-threshold sweep** (POLE_ATR,
  RETR_MAX, FLAG_MAX, WIDTH) on the short-only `pdl+trend` config — tighter,
  higher-quality H definitions are the most likely way to push PF from ~1.03 to
  a comfortable >1.2. The **2-min entry refinement** (deferred; needs a VPS Kite
  download) is the other lever, since a finer entry would tighten R without the
  fade-style cost blow-up.
