# research/64 — Calm-Day Entry Screen for Neutral / Iron-Fly Selling on NIFTY

**STATUS: PHASE 1 DONE (univariate)** · **VERDICT: SIGNAL — "calm" is strongly predictable from a single
family (volatility / range COMPRESSION). Trend, momentum, MA, Ichimoku, ADX, inside-candles add ~nothing.**
Not yet a net-₹ STRATEGY (premium tradeoff unmodeled — see caveat).

## The ask
Find the sweet-spot entry conditions for selling a neutral/iron fly on NIFTY — a "calm days" pattern.
Assess *all* indicators/combinations comprehensively (CPR multi-TF, prior day/week breaks, Bollinger,
HTF, MAs, Ichimoku, other bands, trend-then-halt, inside candles D/W, inverses, RSI, Stoch, ADX, ATR…).

## Critical framing (what "backtest" means here)
No historical NIFTH option premiums exist in the DB (recorder starts 2026-04-20; Kite intraday ~60d).
So true fly P&L over years is not computable. But a fly's edge IS *staying calm*, so the question is
fully answerable from price action: **outcome = the 2% move-stop is NOT hit within the next H trading
days (calm window = the fly wins).** Model-free, no look-ahead (features use data ≤ prior close).
calm-rate = the fly's win-rate proxy; each "stopped" window ≈ a ~₹34k loser.

## Method
NIFTY daily + India VIX from Kite, 2015-01-01→2026-06-12 (2,828 entry days). ~24 causal features across
every family requested. Outcome calm_H for H∈{3,5,8}. Univariate quintile screen on calm_5 with 3-era
stability check. `research/64_calm_day_ironfly_entry/scripts/calm_study.py`, ranking in `results/`.

## Base rates
calm (no 2% stop): **H3 = 77.3% · H5 = 59.4% · H8 = 39.4%.** (Headline = H5, ~1-week hold.)

## Findings — the predictors that matter (top-quintile vs bottom-quintile calm_5, all 3/3 eras consistent)
| Feature | calm in best quintile | worst | spread | calm when |
|---|---|---|---|---|
| **India VIX** | 81.0% | 32.4% | **0.486** | LOW |
| **ATR(14)/price** | 78.8% | 32.7% | **0.461** | LOW |
| **realized vol 20d** | 77.2% | 39.7% | 0.375 | LOW |
| realized vol 10d | 74.5% | 40.4% | 0.340 | LOW |
| **Donchian-20 width** | 71.5% | 39.9% | 0.317 | NARROW |
| range last 5d | 70.8% | 42.8% | 0.280 | NARROW |
| Stochastic %K | 73.0% | 46.8% | 0.262 | HIGH |
| **Bollinger width (squeeze)** | 68.7% | 44.7% | 0.240 | NARROW |
| **prior-day CPR width** | 66.3% | 42.6% | 0.237 | NARROW |
| prior-day gap | 64.7% | 44.9% | 0.198 | SMALL |

**One clean theme: volatility / range COMPRESSION predicts calm.** Low VIX, low ATR, low realized vol,
narrow Donchian/Bollinger/CPR/5-day range, small prior gap — these cluster and dominate. Vol clusters
(calm begets calm). Best single quintile reaches **~80% calm vs 59% base.**

## What DIED — stop testing these for calm-prediction
Near-zero spread / inconsistent across eras: **ADX (0.002!), Ichimoku cloud thickness (0.022),
MA20-50 compression (0.005), 200-DMA distance, 5-day momentum, RSI-distance, weekly CPR, MA20 slope,
20/50-DMA distance.** Binary: **inside-day 60.6% vs 59.3%** and **inside-week 61.0% vs 59.2%** — both
negligible (this challenges the inside-week filter currently in the live engine — it barely beats base
and is far weaker than any vol-compression feature). Day-of-week: Friday entries slightly calmer (62%
vs ~58%); otherwise flat.

## The decisive caveat — calm-rate ≠ net P&L
This screen ranks the **win-rate** axis only. **Low VIX = calmer but thinner premium**; high VIX = richer
premium but more stops. The net-₹ sweet spot is a *tradeoff* — you want calm AND enough credit — which
this price-only study cannot resolve. That's why the live strategy uses a VIX *floor* (≥13, for premium
richness), even though calm-rate keeps rising as VIX falls. Resolving the true optimum needs the option
premium (AlgoTest or the recorder forward), layering credit-collected × calm-rate − stop-cost.

## Next phases
- **P2 — combinations & composite:** AND-stacks of the survivors (e.g. low-ATR ∧ narrow-BB ∧ small-gap),
  a single "compression score", multiple-testing control, per-year P&L-proxy lift.
- **P3 — premium-aware net edge:** map calm-rate buckets to credit (AlgoTest / forward recorder) to find
  the net-₹ optimum (the real "sweet spot"), not just the calmest.
- **P4 — wire the winning compression-gate into the live V2 entry filter** (replacing/augmenting the weak
  inside-week leg), forward-validate via the shadow log.

---

## P2 — Combinations, composite & multivariate (DONE)

**VERDICT: the calm signal is ~ONE factor (vol/range compression). The only *independent* additions are
CPR-width and Stochastic. Best practical gate = low-VIX ∧ narrow-CPR ∧ high-Stoch → ~75% calm at ~17-22%
coverage (walk-forward), and it behaves as a true risk filter — strongly protective in volatile years,
neutral in already-calm years.** OOS AUC ≈ 0.65.

### A. Redundancy (Spearman) — the vol cluster is one thing
VIX / ATR% / realized-vol / Donchian / Bollinger / 5d-range are mutually correlated **0.55–0.93**
(Donchian≈Bollinger 0.93; ATR is the hub). **Independent axes:** `cpr_width_d` (corr ≤0.35 to vol),
`stoch_k` (slightly inverse), `gap_prev` (≤0.34). → stacking vol twins adds nothing; the orthogonal
information lives in CPR + Stoch (+gap).

### B. Conditional lift — what adds *beyond* low-ATR (base 72.1% in low-ATR half)
| add to low-ATR | calm | lift |
|---|---|---|
| + VIX | 77.6% | +5.5pp |
| + realized-vol 20d | 77.0% | +4.9pp |
| + **Stochastic** | 75.7% | +3.6pp |
| + **CPR width** | 75.2% | +3.1pp |
| + Donchian / BB / range / gap | ~71–72% | ~0 (redundant) |

### C. Best gates — walk-forward (thresholds from TRAIN, calm-rate on TEST; coverage shown so tiny gates can't cheat)
| Gate | TEST calm | coverage | lift vs base (62%) |
|---|---|---|---|
| ATR-low (half) | 70.8% | 60% | +9pp |
| ATR-low ∧ CPR-narrow | 70.9% | 38% | +9pp |
| ATR-low ∧ CPR-narrow ∧ Stoch-high | 70.3% | 22% | +9pp |
| **VIX-low ∧ CPR-narrow ∧ Stoch-high** | **74.6%** | 17% | **+13pp** |

Composite "compression score" (z on train → test quintiles): monotonic **38% → 57 → 65 → 70 → 76%**;
**OOS AUC 0.657** (naive 5-feat) / 0.632 (4-axis). Including VIX as the vol axis beats ATR.

### D. The chosen gate across horizons (low-ATR ∧ narrow-CPR ∧ high-Stoch, cov 21%, full sample)
| Hold | gated calm | base | lift |
|---|---|---|---|
| 3 days | **89.6%** | 77.5% | +12pp |
| 5 days | 74.9% | 59.6% | +15pp |
| 8 days | 58.8% | 39.5% | +19pp |

### E. Per-year — it's a risk filter (protective when it matters, neutral when calm)
Huge lift in volatile years — **2020 +36pp, 2022 +34pp, 2021 +26pp, 2016 +24pp, 2026 +28pp**; roughly
**neutral** in already-calm recent years (2023 −6, 2024 +2, 2025 −2). (2015 −14pp but n=16, ignore.)
Exactly the asymmetry you want: it removes the dangerous entries and barely touches the good ones.

### F. EV proxy (per 10-lot trade; W=premium ASSUMED, Lstop=−₹34k)
Gate ≈ **doubles** expected value vs base across premium assumptions (e.g. W=₹60k: base EV ₹22k →
gate EV ₹42k). **Caveat stands:** low-vol = thinner premium, so true W is lower in the gated subset →
P3 resolves the net tradeoff using VIX as a premium proxy.

---

## P3 — Premium-aware net edge (the real sweet spot) — DONE

Premium proxy: ATM-straddle credit ≈ VIX-implied (IV·√T·spot); fly keeps ~65%; win-capture ~55%; stop
calibrated to the **verified ₹34k** 2% MTM loss. So BOTH win-size (W, rises with VIX) and calm-rate (falls
with VIX) vary per regime. EV/trade (10 lots) = calm·W − (1−calm)·₹34k-ish.

### EV by VIX bucket — the calm-vs-premium tradeoff
| VIX | calm | W (win) | EV/trade |
|---|---|---|---|
| ≤11 | 80% | ₹56k | **+₹37.8k** |
| 11–12 | 80% | ₹51k | +₹35.7k |
| 12–13 | 78% | ₹55k | +₹36.3k |
| **13–14** | **63%** | ₹57k | **+₹25.8k ← local dip (the known weak band)** |
| 14–15 | 67% | ₹58k | +₹32.7k |
| 15–16 | 71% | ₹55k | +₹33.9k |
| 16–18 | 53% | ₹58k | +₹21.9k |
| 18–20 | 53% | ₹71k | +₹29.3k |
| 20–25 | 37% | ₹80k | +₹15.2k |
| **25+** | **16%** | ₹98k | **−₹1.5k ← only losing regime** |

Richer premium at high VIX does NOT compensate for the collapse in calm-rate. **VIX 25+ is the one
clearly negative regime; VIX 13–14 is a local trough** (matches the old AlgoTest "12–14 band is the loser").

### The compression gate stacks on the VIX floor
| Regime | n | calm | EV/trade |
|---|---|---|---|
| VIX≥13 | 2179 | 54% | +₹24.2k |
| **VIX≥13 + compression-gate** | 290 | **73%** | **+₹35.6k (+47%)** |
| VIX 13–18 + gate | 267 | 74% | +₹36.0k |

The gate (low-vol ∧ narrow-CPR ∧ high-Stoch) lifts EV/trade ~+47% **and** calm 54%→73% on top of the
existing VIX≥13 floor — i.e. far fewer painful stops at higher per-trade EV.

### The frequency nuance (decides the recommendation)
The strict gate qualifies only **~20% of days** → for a ~weekly positional book that's ~10 trades/yr vs
~50. Per-trade quality is best, but total annual ₹ is lower simply from trading less. So:
- **Risk-adjusted / smooth equity:** the strict compression gate (73–75% calm).
- **Total ₹ on a weekly book:** a *softer* filter that only removes the worst regimes (skip VIX>20–25 and
  the bottom compression-score quintile / the 13–14 dip) keeps most weeks while cutting the big losers.

**Caveats:** premium is a VIX proxy — the low-VIX buckets look great on calm but real low-VIX premium is
thin; the exact ₹ (esp. the low-VIX question and the VIX floor) needs AlgoTest's real premiums. Stop fixed
at the verified ₹34k MTM (varies with DTE).

## P4 — Recommendation (proposal; live change is your call)
1. **Add a compression filter to the live V2 entry, on top of VIX≥13** — skip entry when the prior-day
   compression score is weak (low-vol ∧ narrow-CPR ∧ high-Stoch failing). Tune the threshold for ~50–60%
   coverage (soft filter), not the strict 20% gate, so the weekly book keeps trading while skipping the
   volatile weeks.
2. **Retire / down-weight the inside-week leg** — P1 showed it barely beats base (61% vs 59%); the
   compression score dominates it.
3. **Hard-skip the disaster regime:** VIX > ~22–25 (calm collapses to 16%, EV negative).
4. **Forward-validate** via the engine's existing shadow-skip log (it already records would-skips), and
   **confirm exact ₹ on AlgoTest** (real premiums) before committing — especially the low-VIX premium
   question this price-only study can't settle.

---

## Final gate design + standard CONVICTION TABLE (locked)

**Gate = compression (ATR<1.1% & CPR_d<0.16% & Stoch>65, 2 of 3) AND VIX regime 13-22.** VIX is a
*separate* premium/regime control (floor 13, hard-skip >22), NOT a calm flag — fixing the earlier
double-counting where `VIX<15.4` was wrongly inside the compression score and skipped the premium-rich
16-20 zone. This is what the live shadow logger records as `would_enter`.

### CONVICTION TABLE (NIFTY 2015-26, N=2813) — include in EVERY report
| Hold | BASE | Compression only (48% cov) | Compression + VIX 13-22 (28% cov, = live gate) |
|---|---|---|---|
| 3 days | 77.5% | 88.3% | 86.2% |
| 4 days | 68.4% | 80.8% | 77.8% |
| 5 days | 59.6% | 72.6% | 68.8% |
| 8 days | 39.5% | 51.6% | 47.8% |

**Key insight:** the VIX>=13 floor *reduces* calm by ~4pp (it removes the calmest low-VIX days) — a
premium choice, not a calm choice. Shorter hold = higher conviction but more rolls (3d ~84 trades/yr,
4d ~63, 5d ~50). Coverage: VIX 13-22 alone 67%, compression alone 48%, both 28%.

---

## P5 — Directional-drift screen for SKEWED strategies (jade lizard / skewed condor / broken-wing) — first pass

**VERDICT: the mild-directional regime is frequent (~31% of weeks, bull-skewed) BUT its DIRECTION is
~unpredictable from standard trend/momentum/breakout features. → lean structurally bullish (match the
drift), don't time per-trade direction.**

### 5-day signed-move base rates (N=2630)
| Outcome | share |
|---|---|
| calm (±1.5%) | 57.1% |
| mild bull (+1.5-3%) | 19.0% |
| mild bear (-1.5 to -3%) | 11.7% |
| strong bull (>3%) | 6.3% |
| strong bear (<-3%) | 5.9% |

Moved ≥1.5% by close = 43%; **sustained-directional (breach 2% & hold same side) = 33%.** Upward skew
(mild-bull 19% vs mild-bear 12%).

### Direction is ~unpredictable (among movers, base P(up)=59.2% = the drift)
Quintile spreads in P(up): ADX +7pp (best), mom20/ma_align +3, di_diff/pweek_break +2, RSI/ma_slope/cpr
~0, stoch/mom5 −5, donch −4. **All weak/non-monotonic — no feature reliably picks the sign.** The mild
band (1.5-3%) is also not cleanly predictable (best: low-RSI +10pp, weak).

### Implication
A NIFTY skewed strategy should be a **structural bullish lean** (jade lizard = short put + short call
spread, zero upside risk, wins flat-to-up — fits the 59% up-drift), NOT a per-trade direction bet. Edge =
drift + premium, not prediction. Prior-week breakouts don't follow through (matches research/61 bear-side).

### Next (P5b/P5c — pending user pick)
- P5b: does direction become predictable AFTER a compression squeeze (vol expansion)?
- P5c: backtest the jade-lizard / broken-wing structure on the unconditional drift, sized for mild moves.
