# research/151 — BananaPatterns "VCP" screen

# VERDICT: **NO EDGE** (as a system) — REJECTED for the book

Three separable questions, three answers:

| Question | Verdict |
|---|---|
| **THE RULES** — can their VCP screen be reproduced? | **PARTIAL (62.2% joint trade match).** Their exit engine reproduces 31/32 exactly. Their "VCP" pivot is a **rolling closing high with no volatility-contraction structure whatsoever** |
| **THE CLAIM** — 25.99× / +72.1% CAGR / −14.8% worst fall | **REFUTED.** Honest replay of their own dials: 32.4% CAGR (seed range 6.5–61.6%) at −34.5% drawdown |
| **THE STRATEGY** — is it worth deploying? | **NO.** The breakout screen contributes **negative** value (null control), it correlates **0.75** with the live Open Alpha book, it fails every limb of the pre-registered blend bar, and it **loses to a plain cash sleeve** |

Window basis: signals from a 2,321-symbol NSE daily universe, 2006-04-03 → 2026-09-01.
All headline figures are **after tax** (20% STCG / 12.5% LTCG, Indian FY loss-netting with
carry-forward), **net of 25 bps per side**, with idle cash at **5% p.a.**, reported as
**30-seed medians [min..max]**. ~230 sweep cells were run; discount any single winner
accordingly.

---

## 1. The replication gate — PARTIAL PASS (62.2%)

Ground truth: 40 trades transcribed from a VCP-screen run of the site
(`data/vcp_trades_groundtruth.csv`); 37 usable (BONDADA and E2E are absent from our DB).

**The exit engine is theirs, exactly.** Replaying 12 stop × trail conventions:

| Stop | Trail booked at | Date hits | Date **and** price hits |
|---|---|---|---|
| **8%, on the CLOSE** | **the close that breaks the 50-SMA** | **32/32** | **31/32** |
| 7%, on the close | signal close | 29/32 | 28/32 |
| 8%, intraday touch | signal close | 27/32 | 25/32 |
| any | next open / next close | 4–6/32 | 3–4/32 |

Identical to the engine r/142 decoded for their Blue Sky screen. One site, one engine.
(Note the panel dial reads "cut a loser at 7%"; the ground-truth run used 8% — its own exit
labels say `stop_8pct`.)

**The entry pivot is a prior CLOSE — but their pattern is not identifiable.**

- The buy price is an **exact prior close** in 36/37 trades (within 0.15%).
- It has never been exceeded on a closing basis between the pivot bar and the break (37/37),
  and the **entry-day close is above it** in every case → close-basis pivot, close-basis trigger.
- It is **not** the all-time-high close: the median buy sits **6% below** the prior ATH close
  and only 16/37 are at or above it. VCP and Blue Sky are genuinely different screens with
  roughly 43% overlap.
- **There is no volatility contraction in their VCP trades.** Pivot ages run 1 → 157 bars, so
  there is no minimum base length; **11 of 37 bases contain zero measurable contractions**;
  the volume "dry-up" ratio spans 0.27–1.53 (median 0.85). The name promises a pattern the
  published trade list does not contain.
- **No fixed lookback can fit them**: N would have to be ≥ 157 (deepest pivot age) and ≤ 11
  (shortest run since a higher close) simultaneously — infeasible.

Across 68 candidate parameterizations in three families (rolling max close; structural base
high with a confirmed ≥X% contraction; zigzag last-peak), the best is a **30-day rolling
maximum CLOSE**: 25/37 exact pivot prices, 28/37 first-break dates, **23/37 = 62.2% joint**
— just over the pre-registered 60% bar. Plateau: N = 25–75 all score 20–23 joint, so the
choice is not a knife edge. Zigzag families scored 13–16/37, structural base-high families
15–20/37.

**So: gate PARTIAL PASS. Everything downstream is our best reconstruction, not a trade-exact
replica** the way r/142's Blue Sky work was.

---

## 2. The published claim — REFUTED

Their panel (stamped PROVISIONAL by the site, "under a methodology review"):
**₹10L → ₹2,59,86,848 = 25.99×, CAGR +72.1%, worst fall −14.8%, 48% won, 164 trades**,
2020–2025, with their own footnote *"85% of this period was a strong market"*.

| Arm — 2020-01-01 → 2025-12-31, their dials (5 positions, risk 2%, stop 7/8%, trail 50-day, gate off) | Terminal | CAGR median [min..max] | MaxDD | Trades |
|---|---|---|---|---|
| **Their published run** | 25.99× | **+72.1%** | **−14.8%** | 164 |
| Faithful replica — their optimistic pivot fills, **no costs, no tax**, 10 seeds | 7.64× | **40.0% [23.5..65.0]** | −26.8% | 124 |
| **Honest** — realistic `max(pivot, open)` fills, 25 bps, after tax, 5% idle cash, 30 seeds | 5.38× | **32.4% [6.5..61.6]** | **−34.5%** | 121 |

- **The mechanics are right; the number is not.** Our trade count (121–124) lands close to
  their 164, so we are running their machine — we simply do not get their return, and our
  drawdown is **more than twice** what they publish.
- **Their −14.8% "worst fall" is reachable only on the single luckiest path, and only with
  every optimism switched on.** Measured across 30 seeds (corrected 2026-09-05 — an earlier
  draft of this file asserted "−21%, unreachable" before the best-seed spread had been
  computed; that figure is retracted):

  | Arm | Daily-marked DD: best / median / worst | Monthly-marked DD: best / median |
  |---|---|---|
  | Honest (realistic fills, 25 bps, after tax) | −27.5% / −34.5% / −43.0% | −20.7% / −28.9% |
  | Their fills, no costs, no tax | −22.1% / −29.7% / −37.8% | **−14.7%** / −22.0% |

  So −14.8% requires their optimistic pivot fills, zero costs, zero tax, month-end marking
  **and** the best of 30 selection paths simultaneously. As an expectation it is meaningless;
  the honest median is −34.5% and the number a plan should use is the worst seed, −43.0%.
- **Path dependence is the whole story.** On their own dials the seed range spans
  **6.5% to 61.6% CAGR** — 55 points. Their single figure is one draw from that distribution.
  The cause is their sizing: risk 2% ÷ stop 7% = **28.6% of capital per position**, so with 5
  slots only ~3.5 positions fit and the book is cash-bound and wildly concentrated. Our
  fixed-weight sizing beats it on Calmar in every window **and halves the seed spread**:

| Window | Their dials (5 slots, risk-sized) | Fixed 16 × 6.25% |
|---|---|---|
| 2020–2025 | 32.4% [6.5..61.6], −34.5%, Calmar 0.96 | 36.5% [26.8..47.5], −29.4%, **1.23** |
| 2012–2026 | 31.6% [22.4..49.3], −44.0%, Calmar 0.71 | 31.0% [22.9..36.1], −34.2%, **0.89** |
| 2006–2026 | 25.5% [15.2..31.6], −54.9%, Calmar 0.45 | 27.5% [24.4..30.8], −51.6%, **0.53** |

---

## 3. The killer — the breakout screen does negative work

Pre-registered null control: shrink the pivot lookback toward "no breakout condition at all"
and see whether the screen was contributing anything. 2012–2026, fixed 16 × 6.25%, 15-SMA
trail, after tax, 25 bps, 10 seeds.

| Pivot lookback | CAGR | MaxDD | **Calmar** | Trades |
|---|---|---|---|---|
| **2 days (the null)** | **56.6%** | −22.1% | **2.63** | 6,319 |
| 3 days | 49.3% | −25.6% | 1.91 | 5,806 |
| 5 days | 46.3% | −26.3% | 1.70 | 5,015 |
| 7 days | 43.9% | −27.1% | 1.65 | 4,618 |
| 10 days | 43.2% | −26.9% | 1.59 | 4,321 |
| **30 days (their pattern)** | 37.5% | −29.5% | **1.28** | 4,008 |

**Monotone: the more of the "VCP pattern" you require, the worse the book gets.** Every
number the screen produces comes from the surrounding machinery — RS ≥ 70 momentum ranking,
the ₹5cr liquidity floor, the 15-SMA trailing exit and 16 equal-weight slots — none of which
is a VCP. The pattern is pure drag on frequency.

Two more axes say the same thing:
- **The stop is inert.** Stops of 6 / 8 / 10 / 15% and *no stop at all* (99%) all return
  ~43.2% CAGR at −26.7% DD, Calmar 1.59–1.64. Under a 15-SMA trail the "cut a loser at 7%"
  dial — the site's headline risk control — never fires first. It does nothing.
- **Proximity is inert** (near 10 / 20 / 50% identical), while **RS is the real filter**
  (RS 0 → 32.4%, RS 50 → 38.9%, RS 70 → 43.3%, RS 85 → 51.2% but at −35% DD). RS ≥ 70 is
  roughly the Calmar optimum — an independent validation of that one dial of theirs.
- The **weak-market gate hurts** (Calmar 1.59 → 1.27–1.38), consistent with Open Alpha's
  no-gate spec.

---

## 4. Standalone book (adopted spec) — a real signal, a poor system

**Adopted spec** (`results/vcp_adopted_spec.json`): pivot = 30-day rolling closing high
(the replication-anchored value, *not* the sweep peak — the lookback axis is flat/noisy and
picking its edge would be overfitting), RS ≥ 70, ₹5cr floor, ETFs excluded, buy-stop at the
pivot filled at `max(pivot, open)`, −8% close stop, 15-SMA close trail, 16 slots × 6.25% of
NAV, no gate. 2006-01-01 → 2026-09-01, 30 seeds, after tax, 25 bps, cash 5%.

**36.1% CAGR [31.5..38.3] · MaxDD −40.8% [−42.9..−39.7] · Calmar 0.89 · 5,247 trades ·
win 45.1% · avg win +11.1% · avg loss −3.9% · mean +2.89%/trade · longest losing streak 26 ·
254 trades/year.**

Tradeability and robustness:

| Test | Result | Read |
|---|---|---|
| **Cost ladder** (25 / 40 / 60 bps per side) | Calmar **0.90 / 0.71 / 0.51**; CAGR 37.0 / 30.9 / 23.4% | **≈ −6.8pp CAGR per +15 bps.** The book trades ~37× NAV/year; it is a cost machine |
| Two-window split | 2006–2015: 27.7%, Calmar 0.68 · 2016–2026: 43.4%, Calmar 1.50 | Both positive, but the edge is heavily recent-era |
| Delete the 10 best trades | 0.903 → **0.903** | Genuinely broad — not a lottery-ticket book |
| Cap every winner at +50% / +100% | 32.0% / 36.4% CAGR | Modest fat-tail dependence |
| Longest losing streak | **26 trades** | Psychologically brutal at 254 trades/year |
| Capacity | 254 trades/yr × 6.25% of NAV on ₹5cr-ADV smallcaps | On a ₹1cr book each position is ₹6.25L ≈ 12% of a qualifying name's daily traded value. **Hard capacity wall well below the deployed book's size** |

The 60 bps rung is the decisive one: a 14-day-hold smallcap breakout book will not achieve
25 bps all-in per side in size, and at 60 bps Calmar 0.51 is not deployable.

---

## 5. Portfolio fit — REJECTED on every limb of the pre-registered bar

Baseline: the deployed **True North + Open Alpha 50-50, monthly rebalanced**, after tax.
Paths: 10 OA seeds × 3 TN offsets, VCP seeds paired to OA seeds (paired comparison, never
unpaired medians). Common window 2006-04-03 → 2026-09-01.

**Correlation (daily / monthly returns):**

| Pair | Daily | Monthly |
|---|---|---|
| **VCP ↔ Open Alpha** | **0.749** | **0.759** |
| VCP ↔ True North | 0.480 | 0.546 |
| Open Alpha ↔ True North | 0.418 | 0.510 |

**0.75 to Open Alpha.** The pre-registered bar was < 0.4. This is not a third sleeve; it is
Open Alpha with a weaker entry filter, which is exactly what §3 predicts.

**Blend sweep and the paired test:**

| Blend | CAGR median [min..max] | MaxDD | Calmar | Paired ΔCalmar vs baseline | Calmar wins |
|---|---|---|---|---|---|
| **TN+OA 50-50 (baseline)** | **27.17 [24.8..28.5]** | **−16.42%** | **1.597** | — | — |
| + VCP 10% | 28.14 | −16.77% | 1.642 | **+0.033** | 27/30 |
| + VCP 15% | 28.62 | −17.86% | 1.584 | −0.029 | 12/30 |
| + VCP 20% | 29.10 | −18.96% | 1.520 | −0.089 | 6/30 |
| + VCP 25% | 29.66 | −20.05% | 1.463 | −0.146 | 1/30 |
| + VCP 33% | 30.48 | −21.82% | 1.384 | −0.223 | 0/30 |
| **+ CASH-NULL 10%** | 24.94 | **−14.48%** | **1.659** | — | — |
| + CASH-NULL 20% | 22.72 | −12.63% | **1.745** | — | — |

- Best case (10%) is **+0.033 Calmar**, a third of the required +0.10, and it makes the
  drawdown **worse**, not better.
- **A plain cash sleeve at the same weight beats it** (1.659 vs 1.642 at 10%, and cash keeps
  improving to 1.90 at 33%). By the doctrine's own test, the "diversifier" is worse than
  de-levering.
- Every heavier weight loses on the paired test — at 25% VCP wins 1 path in 30.

**Per-window (blend medians) — it adds pain in exactly the windows that matter:**

| Blend | 2008 crash ret / DD | 2018 grind ret / DD | 2022H1 grind ret / DD |
|---|---|---|---|
| TN+OA 50-50 | **+1.2% / −2.6%** | −9.9% / −11.2% | −5.9% / −9.1% |
| + VCP 20% | −5.0% / −7.0% | −11.3% / −12.3% | −7.7% / −10.5% |
| + CASH-NULL 20% | +2.1% / −1.9% | −7.2% / −8.6% | −4.3% / −7.1% |

The TN gate plus OA's stops have already stripped the crash tail (blend drawdown inside 2008
is just −2.6%); VCP puts it back (−7.0%) and is *also* worse in both grind windows. It is the
r/146 failure mode repeated with a different sleeve, and the r/145 failure mode repeated with
a different screen: **more smallcap momentum beta imported into a book that already harvests
it.**

**G-BLEND: FAIL on correlation, FAIL on Calmar uplift, FAIL on drawdown, FAIL vs the cash-null.**

---

## 6. YoY house-format table

Full table in `results/vcp_yoy_table.txt` (return with intra-year max drawdown in braces,
best-of columns, benchmarks excluded from the picks). Summary row, 2006-04 → 2026-09,
after tax, 25 bps, seed/offset medians:

| | VCP (r/151) | Open Alpha | True North | TN+OA 50-50 | TN+OA+VCP 40/40/20 | NIFTY 50 | Midcap 150 | Smallcap 250 |
|---|---|---|---|---|---|---|---|---|
| CAGR | 35.6% | 33.5% | 19.6% | **27.2%** | 29.1% | 9.1% | 14.7% | 12.2% |
| MaxDD | −41.1% | −24.9% | −24.1% | **−16.8%** | −19.0% | −38.4% | −44.2% | −60.8% |
| Calmar | 0.87 | 1.35 | 0.81 | **1.62** | 1.53 | 0.24 | 0.33 | 0.20 |

VCP buys +2.1pp of CAGR over Open Alpha for **+16pp of drawdown**. That is the r/145
signature — a brilliant-looking standalone that duplicates beta the book already owns.
Chart: `results/vcp_tearsheet.png` (growth of ₹100, log, with a drawdown panel).

---

## 7. Caveats, led not hidden

- **Survivorship on both sides.** Kite lists only current instruments; delisted names are
  absent from our universe (their backtest very likely shares this). 2006 coverage is ~528
  priced symbols, so the early window is survivorship-flattered.
- **The gate is only a 62% match.** Their VCP definition is unpublished and, from 37 trades,
  **not identifiable**. Our 30-day closing-high reconstruction is the best of 68 candidates,
  not their engine. A different reconstruction could score differently — though the null
  control in §3 makes it hard to see how *any* pattern definition rescues the family, since
  requiring less pattern always helped.
- **Split-adjustment defect.** `market_data.db` is not retroactively split-adjusted; r/142
  repaired 72 scale-broken symbols. This screen is a *high* screen, so residual unrepaired
  names would create phantom breakouts. No scale anomalies appeared in the 37 ground-truth
  trades (all buy prices sat inside the entry-day range for 34/37).
- **Capacity was estimated, not measured against real fills**; no paper soak was run, which is
  the honest consequence of a kill verdict.
- **Not tested:** intraday fills / circuit-limit behaviour on breakout days; a mcap floor
  (no shares-outstanding history); the site's "lock in no-loss" breakeven stop beyond the
  P6C grid (where it was neutral-to-negative); options or leveraged expressions.

---

## 8. What to carry forward

1. **One site, one engine.** bananapatterns runs the same stop + MA-trail machine behind every
   screen. r/142 decoded it; r/151 confirms it on a second screen. Any future screen from that
   site needs only a new pivot matrix — the engine is already in
   `research/151_vcp_breakout/scripts/vcp_replay.py`.
2. **Their published numbers are one lucky path with concentrated sizing.** Both studies show
   the same pattern: trade counts reproduce, returns do not, and the published "worst fall" is
   unreachable. Treat every panel figure on that site as an upper bound with no confidence
   interval.
3. **"VCP" as marketed by that screen is not a volatility-contraction pattern.** Their own
   trades contain no contraction structure. Do not re-test this family without new evidence
   of an actual, specified contraction rule.
4. **Add to the known-dead-ends register**: rolling-closing-high breakout screens on the NSE
   universe are dominated by the existing Open Alpha ATH construction (corr 0.75) and lose to
   a cash sleeve in the blend.
5. **Deliverables for study r/154 are in place**: `results/vcp_equity_seeds.csv` (30 daily
   after-tax equity curves, adopted spec, cash 5%) and `results/vcp_adopted_spec.json`.
