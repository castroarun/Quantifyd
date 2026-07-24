# RESULTS — Weekly SuperTrend (10,3) Trend-Following Equity System (research/73)

## VERDICT: **SIGNAL → tradeable as a simple "beat-passive" book on Nifty 200, but NOT a new edge.**

Weekly SuperTrend(10,3), long-only, is a **real trend-following signal** (genuine cut-losers /
ride-winners asymmetry) that **beats buy-and-hold on every cap band at similar-or-lower drawdown,
net of cost and tax.** The best risk-adjusted, capacity-real version is **Nifty 200: 17.5% CAGR /
−31.7% MaxDD / Calmar 0.55 / Sharpe 1.03, beating NIFTYBEES (10.6% / −34% / 0.31) by +6.9pp CAGR and
in 11 of 17 years** — robust to parameters and *stronger out-of-sample*. **However** it does NOT beat
our existing regime-gated midcap-momentum book (research/41/62, Calmar ~1.7): weekly ST is too slow to
cut fast crashes, so drawdown ≈ market. Its merits are **simplicity (one indicator), low turnover
(cost/tax-robust), and high capacity** — a better *passive-plus* than a best-in-class active book.

**Two findings that contradict the source video's own advice** (both robust in the per-year table):
1. **The 40/40/20 profit-booking rule is a return-killer** (Nifty200 17.5% → 8.8% CAGR). The entire
   edge is the fat right tail — **top-10% of trades supply 73–82% of all return** — and booking caps
   exactly the winners that carry the system. It only "helps" as pure de-risking (Nifty50 + booking:
   Calmar 0.62 at −14.6% DD, but just 9% CAGR).
2. **A market-regime gate (NIFTYBEES > 200-DMA) is redundant here and mostly hurts** (Nifty200 17.5%
   → 11.0%, and it doesn't even improve DD). Weekly ST *already self-exits* on flip-down, so a slow
   index gate just misses the 2014/2020 re-entries — it double-counts risk control. (Opposite to
   research/41, where the gate rescued an **always-invested** book that had no built-in exit.)

---

## The system (as tested)

- **Signal:** weekly (W-FRI resampled) SuperTrend, ATR period 10, multiplier 3 — TradingView-accurate
  (`services/technical_indicators.calc_supertrend`). Long-only. Enter on flip-up, exit on flip-down.
- **Execution (causal):** signal read at the **Friday weekly close of week _t_** → trade fills at the
  **open of week _t+1_**. No look-ahead. ("Wait for Friday close" is automatic on weekly bars.)
- **Portfolio (G4):** max 16 concurrent names at 6.25% target weight; when signals > slots, take the
  strongest flip-week gain (a king-candle proxy); idle cash @ 6.5% (liquid fund).
- **Costs/tax:** 0.15%/side (0.30% round-trip) + STCG 15% (<1y) / LTCG 10% (>1y) on realized gains.
- **Universe:** CURRENT official Nifty 50 / 200 / Midcap 150 / Smallcap 250 (survivorship-biased).
- **Benchmark:** NIFTYBEES buy-and-hold. **Period:** 2010-01-01 → 2026-07-07 (16.5y).

## G1 — signal probe (per-trade, does timing beat just owning the names?)

Edge over a **random-duration placebo** (controls for beta/survivorship within the same names)
increases **monotonically down the cap curve**:

| Band | n | win% | mean net/trade | median | avg_win / avg_loss | edge vs placebo |
|---|---|---|---|---|---|---|
| Nifty50 | 555 | 56 | 32.5% | 3.5% | 68 / −13.6 | **−1.8pp (NONE)** |
| Nifty200 | 1713 | 55 | 44.0% | 3.5% | 94 / −16.0 | **+5.2pp** |
| Midcap150 | 1098 | 54 | 47.5% | 2.3% | 102 / −16.8 | **+7.5pp** |
| Smallcap250 | 1373 | 51 | 47.6% | 1.1% | 111 / −19.3 | **+12.2pp** |

- **Large caps have no timing alpha** — the flip is worse than random holding (efficient names).
- **Median trade ≈ 0** (Smallcap 2015+ median is *negative*); the edge is entirely right-tail.
- **Super-winner guard:** dropping the top-3 trades costs only 4–7pp (edge survives — not 3 trades),
  but the carrying names are textbook survivors (TARIL +5347%, KEI +3717%, PAGEIND, RVNL, ADANIENT).
  Nifty200 has the LOWEST top-3 concentration (10% of return) → least survivorship-fragile.
- **Cost-robust:** gross ≈ net everywhere (avg hold ~48 weeks, low turnover).

## G4 — portfolio (blind ST exit, no gate, no booking; net of cost+tax)

| Band | CAGR | MaxDD | Calmar | Sharpe | Beat index |
|---|---|---|---|---|---|
| Nifty50 | 12.2% | **−22.9%** | 0.53 | 0.90 | 10/17 |
| **Nifty200** | **17.5%** | −31.7% | **0.55** | **1.03** | **11/17** |
| Midcap150 | 15.2% | −30.3% | 0.50 | 0.93 | 10/17 |
| Smallcap250 | **19.3%** | **−40.5%** | 0.48 | 0.95 | 9/17 |
| NIFTYBEES B&H | 10.6% | −34.0% | 0.31 | — | — |

Every band beats passive. **Nifty200 = best risk-adjusted + real capacity.** Smallcap = highest
return but ugly DD and untradeable at size (research/62 capacity wall). Notably, **even Nifty50 — which
had no per-trade timing edge — still beats B&H at the portfolio level**, because concentration into 16
trending names + sitting in cash during downtrends adds value even when the flip timing does not.

## G3 — robustness (Nifty200 core)

- **Parameter ridge:** mult=3 is best at every ATR period (7/10/14); mult=2 whipsaws (2× trades),
  mult=4 too slow. All 9 cells beat the index. **The video's "3" is genuinely the sweet spot.**
  (14,3 slightly beats 10,3 — Calmar 0.69 vs 0.55 — a worthwhile refinement.)
- **OOS split:** train 2010–19 Calmar 0.48 → **test 2020–26 Calmar 1.03** (beats index 0.36 OOS and
  *better* than in-sample). Not overfit.

---

## Honest caveats

- **Survivorship bias (dominant).** Current index membership applied to the past; delisted/failed
  names are absent. The right tail that drives returns is exactly the survivorship-selected multibaggers.
  A point-in-time universe would materially cut the small-cap edge; the Nifty200 conclusion is the most
  robust because its returns are the least tail-concentrated.
- **Drawdown ≈ market.** Weekly ST is slow; in fast crashes (2011, Mar-2020) the flip comes after a big
  drop, so MaxDD (−32% Nifty200) is index-like. This is NOT a low-DD product. Do not confuse "beats
  passive" with "capital-preserving."
- **Regime-dependent.** A bull/momentum engine (monster 2014/2020/2023; bleeds 2018/2019, flat 2024/25).
- **Modeled sizing.** Idle-cash rate, cost (0.30% RT) and tax are modeled; slippage/impact not stress-
  tested at large AUM for lower caps (but Nifty200 is high-capacity).
- **t-stats inflated** by trade clustering in the same bull years — treated as directional, not decision-
  grade; the placebo edge and OOS split are the credible robustness evidence.
- **Not a new edge.** This converges on the existing regime-gated momentum book (research/41/62/72) with
  a cruder, simpler entry. It is a cleaner *passive-plus*, not alpha beyond what we already run.

## Next levers

1. **Nifty200, 14,3, blind-exit, no-booking** is the cleanest deployable spec (Calmar 0.69) — candidate
   for a G5 paper soak alongside the momentum book, IF a simple, low-maintenance "beat-Nifty" sleeve is
   wanted (its selling point is one indicator + low turnover, not risk-adjusted supremacy).
2. **PIT universe test** — re-run Nifty200 on point-in-time membership to quantify the survivorship
   haircut before any capital.
3. **DD control that doesn't kill return** — the gate/booking both fail; the only real lever is a faster
   crash overlay (e.g. a daily-ST or vol-spike hard exit layered on the weekly entry) — test if it cuts
   the −32% DD without capping the tail.
4. If the goal is best risk-adjusted, **improve the existing research/41/62 book** rather than deploy this.

---

**Reproducibility:** DB snapshot `backtest_data/market_data.db` (day tf, max date 2026-07-07).
Scripts: `research/73_weekly_supertrend_investing/scripts/{st_weekly_engine,g1_signal_probe,
g1_superwinner,g4_portfolio,g3_param_sens,make_tearsheet}.py`. ST = calc_supertrend(10,3).
Cost 0.30% RT, STCG 15%/LTCG 10%. Ran on VPS venv (numpy 2.4.4 / pandas 3.0.2), 2026-07-07.

---

## ⚠️ CORRECTION (2026-07-07, same day) — the headline was a BENCHMARK ARTIFACT. Verdict downgraded.

**The +6.9pp "beats passive" was almost entirely survivorship + breadth, NOT the SuperTrend timing.**
I benchmarked the ST book against NIFTYBEES (the Nifty **50** index) while the book trades today's Nifty
**200** names — two free edges before ST does anything: (1) survivorship (today's Nifty200 applied to 2010
includes the names that are in the index BECAUSE they 10–50×'d, e.g. RVNL/KEI, and excludes the drop-outs),
and (2) breadth (Nifty200's mid-tilt beat Nifty50 over 2010–26 regardless of timing).

**Fair, survivorship-matched test — hold the SAME 200 names, no timing (2010–2026):**

| Book (same universe) | CAGR | MaxDD | Calmar | Sharpe | Total |
|---|---|---|---|---|---|
| **ST-core (flip in / flip out)** | 17.5% | **−31.7%** | 0.55 | 1.03 | 14.3× |
| EW buy-&-hold, weekly-reb, same 200 | **21.0%** | −35.9% | 0.58 | 1.16 | 23.2× |
| B&H-drift, fixed-cap, start-names (least survivorship) | 20.4% | −36.3% | 0.56 | 1.17 | 21.2× |
| NIFTYBEES (Nifty 50 index) | 10.6% | −34.0% | 0.31 | 0.74 | 5.2× |

**The SuperTrend timing LOSES to simply holding the same basket by ~3–3.5pp/yr**, at essentially identical
Calmar (0.55 vs 0.56–0.58). All it buys is a marginally shallower MaxDD (−31.7 vs −36.3) — a tiny de-risk
paid for with ~3pp/yr of return. In a 16-year bull, being in cash / under-concentrated during pullbacks and
whipsaws (2010 −23pp, 2012 −21pp, 2021 −36pp vs the basket) costs more than the entry edge earns.

**Reconciling with G1 (+5.2pp vs placebo):** the per-trade ENTRY timing does have a small real edge (ST
entries beat random-duration entries of the same names). But that edge is swamped at the PORTFOLIO level by
the opportunity cost of not being fully invested through a secular bull. A real signal at the trade level is
not a strategy at the book level — exactly the playbook's SIGNAL≠STRATEGY warning, and the same lesson as
research/49 ("it's beta, not alpha").

**CORRECTED VERDICT: NO INVESTABLE TIMING EDGE.** Weekly ST(10,3) does not beat owning the same universe;
the attractive headline was the survivor basket vs the wrong (Nifty 50) index. It is *not* a critical
finding. If you want the ~20% CAGR, the honest way to get it is **own the basket** (with its real
survivorship caveat), or better, run the existing regime-gated momentum book (Calmar ~1.7). The only
salvageable angle: ST as a mild **de-risk overlay** on a basket you'd hold anyway (−4pp DD for −3pp CAGR —
a poor trade as-is; would need the fast crash-exit idea to be worth it).

| 2026-07-07 | Fair-benchmark control (`fair_bench.py`) | ST LOSES to same-200 B&H −3.5pp/yr → headline was survivorship+breadth; verdict downgraded to NO TIMING EDGE |

---

## PHASE 2 (2026-07-08) — ST as a MARKET-LEVEL CRASH OVERLAY: this is where the indicator actually works.

**The salvageable angle succeeds.** Own the same-basket EW book always; layer a MARKET-level (index)
trend filter that flattens the whole book in downtrends and re-enters. Unlike per-name weekly-ST timing
(which loses), a single daily-ST crash filter roughly **doubles pre-tax Calmar** by cutting drawdown for a
small CAGR give-up. (`scripts/crash_overlay.py`, daily resolution, causal signal→next-day, cost 0.30% RT
per switch, idle cash 6.5%.)

**Nifty 200 basket (2010–2026), overlays vs the plain always-in basket:**

| Book | CAGR | MaxDD | Calmar | Sharpe | Sw/yr | Net-tax CAGR | Net-tax Calmar |
|---|---|---|---|---|---|---|---|
| PLAIN basket (tax-deferred) | 21.8% | −39.2% | 0.56 | 1.25 | 0 | 21.8% | 0.56 |
| + **daily-ST(7,3)** | 19.6% | **−15.3%** | **1.28** | 1.64 | 5.0 | 16.8% | **1.01** |
| + daily-ST(10,3) | 19.4% | −15.5% | 1.25 | 1.64 | 5.1 | 16.5% | 0.96 |
| + daily-ST(20,3) | 19.8% | −17.3% | 1.14 | 1.61 | 5.2 | 16.7% | 0.94 |
| + 50-DMA gate | 20.3% | −16.6% | 1.23 | 1.71 | 13.6 | 16.6% | 0.93 |
| + 200-DMA gate | 15.1% | −33.3% | 0.45 | 1.17 | 9.5 | 13.2% | 0.38 |
| + vol-spike (panic) | 18.2% | −27.4% | 0.67 | 1.22 | 8.2 | 15.2% | 0.49 |

Consistent across bands (net-tax Calmar): Smallcap 0.40→0.72 (DD −56%→−27%), Nifty500 0.50→0.96 (DD −44%→−18%).

**Reads:**
1. **It works, and it's the DAILY ST (not weekly) that matters** — a fast market trend filter catches
   crashes (2011 plain −20% → overlay +3.7%; 2018 −5.8 → −2.7; 2022 +12 with the DD capped). Weekly ST is
   too slow.
2. **Robust, not overfit** — the whole fast family (daily-ST 7/10/20 + 50-DMA) clusters at Calmar 0.9–1.3;
   only the slow 200-DMA fails. Monotonic "faster gate helps." Economically grounded (trend overlay / tail hedge).
3. **200-DMA is a BAD regime gate here** (Calmar 0.45 < plain 0.56) — daily-ST(7,3) strictly dominates it.
   **Cross-pollination lever:** the live momentum book (research/62) gates on a NIFTYBEES moving-average →
   test a daily-ST(7,3) gate as a candidate upgrade.
4. **TAX is the real cost, not switching.** Selling the whole cash basket ~2.5×/yr realises STCG →
   net CAGR 19.6%→16.8%, net Calmar 1.28→1.01 (still ≫ plain 0.56, but a ~5pp CAGR give-up vs tax-deferred B&H).
5. **⇒ Correct implementation = hedge with NIFTY FUTURES / puts on the red signal, NOT liquidation.** Shorting
   the index against the book triggers no sale of the cash equities → no tax event, holdings untouched →
   recovers most of the pre-tax 1.28 Calmar. (Costs: roll/basis/margin + basket-vs-Nifty tracking error — owed a test.)

**PHASE-2 VERDICT: the SuperTrend’s real, robust edge is as a MARKET-LEVEL crash overlay (daily 7–10,3),
NOT per-name entry/exit.** As a cash-liquidation overlay it doubles pre-tax Calmar (0.56→1.28) at a real
tax cost (net 1.01); as a futures/puts hedge it should be far more tax-efficient. Best next step:
(a) implement the overlay as a Nifty-futures hedge and re-measure net; (b) swap the momentum book’s MA gate
for daily-ST(7,3) and re-test. This is a materially better use of the indicator than anything in Phase 1.

| 2026-07-08 | Phase 2 crash-overlay (`crash_overlay.py`) | daily-ST(7,3) doubles pre-tax Calmar 0.56→1.28 (DD −39→−15); net-tax 1.01; tax is the cost → hedge via futures; 200DMA HURTS |

---

## PHASE 3 (2026-07-08) — INDEX-LEVEL ST on the actual ETF: the cleanest tradeable version.

**User idea, tested + confirmed:** trade the index ETF ITSELF on a daily trend filter (hold when
green → to cash when red), vs buy-and-hold. INDEX-LEVEL (one instrument), so NO survivorship, infinite
capacity, dead simple. `scripts/etf_st.py` + `etf_st2.py`, net cost+tax.

**NIFTYBEES (2010–2026):**

| Signal on the ETF | CAGR | MaxDD | Calmar | Sharpe | Sw/yr | Net-tax CAGR | Net-tax Calmar |
|---|---|---|---|---|---|---|---|
| Buy & hold | 10.6% | −36.3% | 0.29 | 0.75 | 0 | 10.6% | 0.29 |
| **ST(7,3)** | 10.7% | **−14.2%** | 0.75 | 1.11 | 5.0 | **9.0%** | **0.53** |
| 100-DMA | 10.3% | −14.2% | 0.72 | 1.02 | 8.7 | 8.4% | 0.52 |
| 50-DMA | 10.1% | −14.0% | 0.72 | 1.05 | 13.6 | 8.0% | 0.49 |
| 200-DMA (too slow) | 5.5% | −20.7% | 0.27 | 0.55 | 9.5 | 4.6% | 0.20 |

**Reads:**
1. **Confirmed net-of-tax:** ~1.5pp CAGR give-up (10.6→9.0%), **drawdown MORE THAN HALVED (−36→−14%)**,
   Calmar 0.29→0.53 and Sharpe 0.75→1.11 (~doubled). PRE-tax the give-up is ~zero (10.7 vs 10.6).
2. **Not ST-specific:** ST(7,3) is marginally best (fewest switches → least tax) but 50/100-DMA are tied.
   It's "any fast-medium trend filter." The **200-DMA HURTS on the single ETF** (halves CAGR — exits/re-enters
   too late). (Note the contrast: on the momentum BOOK the 100-DMA beat ST because ST over-traded the book;
   on a single ETF ST slightly wins because it switches least. Context-dependent — always A/B on the actual asset.)
3. **Robust across equity ETFs:** NIFTYBEES −36→−14, JUNIORBEES −39→−18, BANKBEES −48→−23 (DD ~halved each,
   CAGR ~matched pre-tax). **Does NOT help GOLDBEES** (Calmar 0.62→0.50 — trend-timing suits crash-prone equity
   indices, not gold).
4. **It's a WELL-KNOWN result** (Faber-style tactical index timing), now cleanly confirmed on Indian ETFs.
   Value = clean (no survivorship) + infinitely scalable + one instrument, NOT novelty.
5. **Tax is the whole give-up.** ~5 cash-switches/yr realise STCG (10.7% pre → 9.0% net). Implement as a
   NIFTY-futures/puts hedge (no sale) to recover most of the 1.5pp — same fix as Phase 2.

**PHASE-3 VERDICT: the honest, tradeable takeaway of the whole study.** ST (or any fast-medium trend
filter) does NOT beat buy-and-hold on RETURN, but it HALVES the drawdown of an index ETF for ~1.5pp net-tax
CAGR — a clean, scalable, well-known risk-management overlay. Best expressed on the liquid index ETF (not
per-name, not the survivorship basket). Next: (a) futures-hedge implementation to kill the tax drag;
(b) multi-ETF sleeve (equity + gold, each trend-timed) — but note gold didn't respond to ST.

## GATE CROSS-CHECK (2026-07-08) — daily-ST gate on the LIVE momentum book: TESTED, REJECTED.
`research/62_momentum_etf_subselect/scripts/62i_st_gate.py` (winner rsblend N8 buf22 donch15, net STCG20%,
2014–26). Swapping the book's 100-DMA regime gate for a daily-ST gate is WORSE: net Calmar 100-DMA **1.71**
vs dST(7,3) 1.33 / dST(10,3) 1.25 / 50-DMA 0.99. ST gates are twitchier (de-risk 30–36× vs 23), whipsaw out
of recoveries, give up ~6pp CAGR for no DD benefit. **KEEP the live 100-DMA gate.** (Engine got a
backward-compatible `gate_roff` param; `.bak_stgate` backup kept. Live book services/momentum_paper.py untouched.)

| 2026-07-08 | Phase 3 ETF timing (`etf_st.py`) + gate cross-check (`62i_st_gate.py`) | ETF: DD halved for ~1.5pp net-tax (ST7≈100DMA, 200DMA bad); momentum-book gate: 100-DMA beats ST → keep live gate |

---

## PHASE 3b (2026-07-08) — REALISTIC FRICTIONS on the winner (liquid-fund idle + T+1 settlement)
User pushed for realism: idle cash parked in a LIQUID fund earns a rate NET of its expense + slab tax
(~6.5% gross -> ~4.5% net), and India ETF settlement is T+1 (exit -> proceeds in transit ~1 day at 0%;
re-entry ~1 day late). `scripts/settlement_liquid.py` on NIFTYBEES · ST(7,3):

| Scenario | CAGR | MaxDD | Calmar | net-all-tax CAGR | net-tax Calmar |
|---|---|---|---|---|---|
| Buy & hold | 10.6% | -36.3% | 0.29 | 10.6% | 0.29 |
| Idealized (6.5% cash, instant) | 10.7% | -14.2% | 0.76 | 9.1% | 0.54 |
| **REALISTIC (liquid 4.5% net + T+1 lag)** | **9.3%** | **-14.3%** | 0.65 | **7.8%** | 0.46 |
| Conservative (2-day lags) | 9.0% | -16.6% | 0.54 | 7.4% | 0.41 |
| Cash 0% (no liquid) | 7.5% | -16.1% | 0.47 | 6.0% | 0.32 |

**Reads:** (1) DRAWDOWN-halving is FRICTION-PROOF (-14 to -17% everywhere). (2) The CAGR give-up GROWS
with realism: idealized ~0pp -> realistic ~1.3pp pre-tax / **~2.8pp net of ALL tax** (earlier ~1.5pp was
too kind -- corrected). (3) The winner is ROUGHLY SHARPE-NEUTRAL (0.33 vs 0.34) -- a DRAWDOWN-reduction
overlay, NOT a return-enhancer. (4) The liquid fund is ESSENTIAL (worth ~1.8pp vs 0% cash); its tax/expense
costs ~0.8pp, the T+1 lag ~0.6pp. (5) => the tax-free NIFTY-FUTURES/PUTS HEDGE is the preferred build (no
ETF sale -> no equity tax, no settlement lag -> recovers most of the give-up).

**WINNER REFRAMED + PUBLISHED:** the study + HTML report now LEAD with the winner (NIFTYBEES · ST(7,3),
its own dark factsheet niftybees-st73-winner.png, realistic numbers) instead of the Phase-1 illusion.
Also answered the "Phase 2 vs 3 Calmar gap": same signal, same DD job; Phase 2's higher Calmar (1.28) only
reflects timing the survivorship-inflated basket (21.8% CAGR) -- a mirage; Phase 3 (real ETF) is the honest
number. Scripts: settlement_liquid.py, make_winner_tearsheet.py (realistic).

| 2026-07-08 | Phase 3b realistic frictions + winner reframe | net-all-tax 7.8% CAGR / DD -14% / Calmar 0.46 (Sharpe-neutral DD overlay); winner factsheet published; futures-hedge preferred |

---

## PHASE 3c (2026-07-08) — MODELED futures-hedge build: recovers the give-up.
`scripts/futures_hedge.py`. Keep NIFTYBEES (never sold -> NO equity CGT, deferred like B&H; NO T+1 lag;
margin funded by PLEDGING the ETF) and SHORT NIFTY futures on the red signal. While hedged, long-ETF +
short-future ~= a synthetic T-bill earning the carry (~ risk-free).

| Approach (NIFTYBEES · ST 7,3) | CAGR | MaxDD | Calmar | Sharpe |
|---|---|---|---|---|
| Buy & hold | 10.6% | -36.3% | 0.29 | 0.75 |
| Cash-rotation (realistic, net all tax) | 7.8% | -14.3% | 0.46 | ~0.9 |
| **Futures-hedge, ~4.6% net carry (MODELED)** | **10.6%** | **-14.4%** | **0.74** | 1.10 |

**Result: recovers essentially the WHOLE ~2.8pp give-up -> ~B&H return at HALF the drawdown** (Calmar
0.74 vs 0.29). This is the near-free-lunch the idealized numbers hinted at, achieved via the tax structure
(defer equity CGT, no settlement lag, similar out-of-market yield from carry).

**CRITICAL CAVEAT: MODELED, not backtested.** The DB has NO NIFTY futures series, so the ~4.6% net carry
is an ASSUMPTION (sensitivity 4.0/4.6/5.2% -> Calmar 0.71/0.74/0.76). The hidden risk: in crashes futures
often flip to BACKWARDATION (discount) -> the short-future carry can go NEGATIVE exactly when you are
hedged, which the model does not capture. Also: basis/tracking error (ETF vs future), monthly roll
cost/execution, lot-size granularity, slab tax on futures gains. **NEXT (top priority): validate on REAL
futures data before any capital** (acquire a NIFTY continuous-futures series; test crash-time carry).

Study + HTML report updated with the futures-hedge table (section 05, labelled MODELED) + the caveat;
verdict "best build" now cites the modeled recovery.

| 2026-07-08 | Phase 3c MODELED futures-hedge (`futures_hedge.py`) | ~B&H 10.6% CAGR / DD -14% / Calmar 0.74 (recovers give-up); MODELED, validate on real futures data; risk = crash backwardation |

---

## PHASE 3d (2026-07-08) — REAL-DATA VALIDATION of the futures-hedge (NSE bhavcopy basis).
`scripts/kite_futures_probe.py` (current basis) + `build_real_basis.py` (196 real NIFTY near-month
future basis points from NSE F&O bhavcopy, crash windows daily + normal months). Kite only serves the
current contract; NSE bhavcopy archives ARE reachable from the VPS (old + UDiFF formats) → real basis.

**Real annualised carry (futures premium the short captures):**

| Regime | pts | mean | median | backwardation days |
|---|---|---|---|---|
| Hedge-OFF (ST green) | 49 | +5.1% | +4.5% | 0% |
| Hedge-ON (ST red) | 147 | +3.1% | +1.1% | 36% |
| COVID 2020 Feb-May | 60 | +3.0% | -0.6% | **52%** |
| 2022 selloff | 59 | +2.3% | +1.6% | 27% |
| 2018 correction | 30 | +4.5% | +4.4% | 13% |

**Findings:** (1) The backwardation risk I flagged is REAL — during COVID 52% of days were in backwardation,
clustering when the hedge is ON. My +4.6% modeled carry was TOO OPTIMISTIC. (2) BUT it's BOUNDED — the
average hedge-on carry is still POSITIVE (+3.1% mean / +1.1% median); the scary -20..-46%/yr figures are
near-expiry annualisation artifacts of tiny (-0.4%) absolute basis moves. (3) Re-running the hedge with the
REAL carry (~+3%, which already includes the crash backwardation): **~9.9% CAGR (-0.6pp vs B&H 10.5%) /
-14.8% DD (halved) / Calmar 0.67 / Sharpe 1.03.** So the hedge recovers MOST of the give-up (down to ~0.6pp
from the cash version's 2.8pp), still halves the drawdown, and — unlike the Sharpe-neutral cash version —
now GENUINELY IMPROVES Sharpe (1.03 vs 0.75). Corrected the modeled 10.5% down to the real ~9.9%.

**Verdict on the winner (final, validated):** trend-time NIFTYBEES with daily ST(7,3), implemented as a
NIFTY-futures hedge = ~9.9% CAGR at HALF the drawdown (Calmar 0.67 vs B&H 0.29, Sharpe 1.03 vs 0.75), on
REAL futures-basis data including crash backwardation. Residual before capital: full daily basis series
(vs the 196-pt crash-window sample) for path-exact P&L + a paper-forward soak of the actual futures roll.
Study + HTML report updated with the validated numbers + the real-basis backwardation table. `real_basis.csv`.

| 2026-07-08 | Phase 3d REAL-DATA validation (NSE bhavcopy) | carry ~+3% real (not +4.6%); backwardation real but bounded (COVID 52% days); hedge ~9.9% CAGR / -15% DD / Calmar 0.67 / Sharpe 1.03 |

---

## PHASE 3e (2026-07-08) — bidirectional long/short? Tested (daily AND weekly ST) — NO, stay long-only.
User idea: instead of going FLAT when ST is red, go net SHORT to profit from downtrends. `scripts/
bidirectional_st.py` on NIFTYBEES, daily ST(7,3) AND weekly ST(10,3), 2010-2026, carry ~3.2% net.

| Book | signal | CAGR | MaxDD | Calmar | Sharpe |
|---|---|---|---|---|---|
| Buy & hold | — | 10.5% | -36.3% | 0.29 | 0.75 |
| Long-only (hedge, WINNER) | daily 7,3 | 9.9% | -14.8% | 0.67 | 1.03 |
| Bidirectional L/S | daily 7,3 | 6.6% | -25.3% | 0.26 | 0.51 |
| Short-only (diag) | daily 7,3 | 0.8% | -33.8% | 0.02 | 0.13 |
| Long-only (hedge) | weekly 10,3 | 6.3% | -31.1% | 0.20 | 0.61 |
| Bidirectional L/S | weekly 10,3 | 0.3% | -50.9% | 0.00 | 0.09 |
| Short-only (diag) | weekly 10,3 | -1.9% | -43.6% | neg | neg |

**Smoking gun:** during ST-RED periods the index STILL RISES — **+6%/yr (daily), +19%/yr (weekly)** — because
the slow filter flags red AFTER the drop and stays red THROUGH the recovery, so you short into the bounce.
Markets drift up + spend more time recovering than crashing. **Short side is a structural loser:** short-only
~0 (daily) / negative (weekly) at huge DD; bidirectional cuts CAGR and roughly DOUBLES drawdown (worse than
B&H). Weekly worse than daily throughout (too slow — consistent with the whole study). **STAY LONG-ONLY:
hold the ETF, hedge to flat in downtrends, never short.** Clean negative — the winner (long-only hedge) stands.
Study + HTML report updated (bidirectional table).

| 2026-07-08 | Phase 3e bidirectional L/S (`bidirectional_st.py`) | short side loses (red-period index +6%/+19% daily/weekly); bidir doubles DD, cuts CAGR; stay long-only; weekly worse than daily |

---

## PHASE 3f (2026-07-08) — apply the crash overlay to our BEST-CAGR book? Tested. The value is INVERSE to return.
User: "we did recent backtesting with the best CAGR, take that." Best-CAGR recent = research/75 Nifty-250
momentum (`combo__ret252` 46.5% gross / 43.5% net / 39.6% post-tax, but −42% DD, midcap+smallcap = the
capacity/survivorship mirage). Used the tradeable base config NAV (`nav_base.csv`, 31.9% CAGR / −31.6% DD,
2006-2026, already monthly-gated) and applied the research/73 NIFTY daily-ST(7,3) crash overlay.
`scripts/overlay_momentum.py`.

| Version | CAGR | MaxDD | Calmar | Sharpe |
|---|---|---|---|---|
| Base momentum book (no overlay) | 31.9% | -31.6% | 1.01 | 1.45 |
| + NIFTY-ST overlay -> cash (pre-tax) | 26.9% | -22.2% | 1.21 | 1.49 |
| + NIFTY-ST overlay -> cash (net STCG) | 21.9% | -23.5% | **0.93** | 1.22 |
| + NIFTY-ST overlay -> hedge-carry (proxy) | 25.4% | -22.2% | 1.14 | 1.42 |

**Findings:** (1) The overlay cuts DD (−32%→−22%) and PRE-TAX improves Calmar (1.01→1.17). (2) NET of STCG it
HURTS (0.88 < 1.01) — pulling a high-gain momentum book to cash ~5×/yr triggers heavy short-term tax, and
being out 39% of the time forgoes ~30%/yr (vs a 10% index). (3) Hedge version (1.14) avoids tax but NIFTY
futures don't cleanly hedge a midcap book (beta>1, idiosyncratic) → optimistic.

**KEY LESSON (generalises the whole study): the crash overlay's value is INVERSELY related to the
underlying's return.** Low-return index ETF (NIFTYBEES 10.6%) → sitting out costs ~nothing → overlay ~halves
DD for free. High-return momentum book (31.9%) → sitting out forgoes ~30%/yr + triggers tax → net-negative.
The overlay is a tool for LOW-return, high-DD INDEX ETFs; a high-Calmar momentum book is best de-risked by
its OWN regime gate (consistent with the Phase-3c gate cross-check: 100-DMA gate beats an ST overlay on the
momentum book). Also: the 46.5% "best CAGR" itself is the lower-cap gross-only mirage (research/62 capacity
wall) — not a real investable number. Study + HTML report updated.

| 2026-07-08 | Phase 3f overlay on best-CAGR book (`overlay_momentum.py`) | cuts DD −32→−22% pre-tax (Cal 1.01→1.17) but NET-of-tax HURTS (0.88); overlay value is inverse to return → index-ETF tool, not for momentum books |
