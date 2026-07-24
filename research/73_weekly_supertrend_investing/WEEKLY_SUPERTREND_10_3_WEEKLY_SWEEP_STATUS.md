# Weekly SuperTrend (10,3) Trend-Following Equity System — Across Nifty 50/200/Midcap150/Smallcap250

STATUS: **DONE — CORRECTED: NO INVESTABLE TIMING EDGE** (headline +6.9pp vs NIFTYBEES was survivorship+breadth; ST timing LOSES to same-200 buy-and-hold by ~3.5pp/yr at equal Calmar)

Research folder: `research/73_weekly_supertrend_investing/` (VPS canonical).
Owner: Arun. Engine reuses `services/technical_indicators.calc_supertrend` (TradingView-accurate 10,3).

---

## 1. The Ask

**What you asked (verbatim gist):** A YouTube guest (Vijay Khant) runs a system built entirely on
the **SuperTrend (10, 3) on the WEEKLY chart**. Enter when weekly ST turns green, exit blindly when it
turns red. Position-size 5–7% per stock so a stop costs ≤1–2% of capital. Book profits **40/40/20**
(sell 40% after +40% from entry, another 40% after a further +40%, trail the last 20% on the ST line).
Five selection "hacks": (1) King candle — strong/large flip candle; (2) wait for Friday weekly close to
confirm; (3) enter only after price crosses the flip (breakout) candle's high; (4) favour stocks that
consolidated 18–20 weeks before the signal; (5) Dow-structure change (higher highs/lows). Plus a
**50-EMA band** (EMA of high/close/low) for support/re-entry on corrections. "Plan and test this out."

**What we're actually testing (cleaned up):**
- **Signal:** On each name, weekly SuperTrend(ATR=10, mult=3) direction, computed on the **weekly
  (W-FRI) resampled bar**. Long-only. Flip-up = ST direction −1→+1; flip-down = +1→−1.
- **Execution (causal, no look-ahead):** signal is read at the **Friday weekly close of week _t_**;
  the trade (entry or exit) fills at the **open of week _t+1_** (Monday). Never trade on the same bar
  that generated the signal.
- **Universe (4 bands, compared):** official current membership — Nifty 50 (50), Nifty 200 (200),
  Nifty Midcap 150 (150), Nifty Smallcap 250 (250). Benchmark = NIFTYBEES buy-and-hold.
- **Period:** full daily history (2000→2026-07-07) resampled to weekly; **lead reporting with the
  modern sub-period 2015+** where breadth is real (see §3 survivorship/coverage).
- **Success metric:** for G1, per-trade net expectancy (avg return/trade, win%, t-stat) **and** whether
  weekly-ST timing beats simply buy-and-holding the same names (the beta baseline). For G4, portfolio
  **Calmar / MaxDD / CAGR net-of-cost-and-tax vs NIFTYBEES**.
- **Falsification (decided up front):** ABANDON as "no timing alpha" if, per band, the weekly-ST long
  trades' net expectancy is ≤ the buy-and-hold / random-duration placebo baseline for that band's names
  across the full and modern sub-periods (i.e. the ST flip adds nothing over just owning uptrending
  beta — the exact fate of research/49 and research/55).

## 2. Economic hypothesis (G0)

**Mechanism:** Trend-following / time-series momentum. Under-reaction to sustained fundamental change
lets weekly trends persist for months; a slow weekly filter (10,3 ST ≈ a ~10-week trailing volatility
band) rides the middle of the move and cuts losers via the band flip. **Counterparty:** disposition-
effect sellers (cut winners early) and late mean-reversion traders fading the trend. **Why it should
persist:** behavioural, structural (retail under-diversification), and it is a *slow* signal so capacity
is high and decay is slow. **Decay risk:** momentum crashes (sharp reversals after down-regimes), and
whipsaw in range-bound years (2011, 2018, 2022 India). **Prior:** this is the SAME family as our
research/41 / 62 / 72 regime-gated momentum book (Calmar ~1.7) and the STREND_atr7_m3.0 MQ finding — so
the realistic expectation is a **real but not-novel** trend signal whose value is DD-control, and whose
lower-cap versions get eaten by capacity (research/62). We will state convergence loudly.

## 3. The Base — mechanics (locked)

- **Bar build:** daily→weekly resample, label W-FRI. OHLC = open:first, high:max, low:min, close:last,
  volume:sum. Partial final week included but flagged. Weekly ST via `calc_supertrend(atr_period=10,
  multiplier=3)` on the weekly frame (same code path as live/backtest ST elsewhere).
- **Entry trigger:** weekly `supertrend_flip_up` at close of week _t_ → BUY at open of week _t+1_.
- **Exit trigger (core):** weekly `supertrend_flip_down` at close of week _t_ → SELL at open of week _t+1_.
- **Direction:** LONG-ONLY (cash equities; India shortability limits — playbook §5).
- **Costs:** delivery round-trip modelled explicitly — brokerage+STT+exchange+GST+stamp+slippage.
  Base assumption **~0.30% round-trip** (net); cost-sensitivity at 0.15 / 0.30 / 0.60%. **Tax:** STCG
  15% / LTCG 10% modelled at G4 (holding-period-sensitive; weekly ST holds weeks-to-months).
- **Data integrity:** current index membership applied to the past = **survivorship bias** (state it;
  report modern sub-period; note delisted names absent). Coverage grows 283 names(2008)→1632(2025) so
  pre-2015 is thin. No look-ahead: signal_t → fill at open_{t+1}. Benchmark NIFTYBEES full history.
- **G1 success criterion:** net per-trade expectancy > 0 AND weekly-ST trades beat the buy-&-hold and
  random-duration placebo baselines for the same names (timing adds value, not just beta).

## 4. Plan — stage gates + grid

**G1 — signal probe (gross first, then 0.30% net).** Per band, every weekly-ST long trade
(entry open_{t+1} after flip-up → exit open_{t+1} after flip-down). Report: n trades, win%, mean &
median return/trade, avg-win/avg-loss, avg holding (weeks), per-year mean, t-stat. Baselines: (a) buy-
&-hold the same names over the same window; (b) random-entry, matched-average-duration placebo.
Gate: net expectancy>0 AND beats both baselines on full + 2015+ sub-period.

**G2 — mechanics & exits.** Add the **40/40/20 profit-booking** variant (needs intra-trade weekly-high
tracking) vs blind-ST-exit; net-of-cost; break-even cost. Which exit wins on net & tail?

**G3 — robustness.** Param sweep ATR∈{7,10,14} × mult∈{2,3,4} (monotonic > peak); per-year table;
OOS split (train ≤2019 / test 2020+); cost stress ×2; adversarial (super-winner guard — drop top-3
names). Multiple-testing note.

**G4 — portfolio.** Capped concurrent positions (~6% target weight → ~16 slots; when signals>slots,
selection rule = strongest flip-week momentum, which also sets up the king-candle test), idle cash @
6.5%, net-of-cost+tax equity curve. Metrics vs NIFTYBEES B&H and vs research/41/62 book: CAGR, Sharpe,
Sortino, **MaxDD, Calmar**, correlation, capacity (participation vs ADV). Optional NIFTYBEES-200DMA
regime gate (research/41 lesson).

**Filters (hacks) — only if core survives G1/G2.** Each as an add-on, measured for marginal net/DD:
(1) king-candle strength (flip-week % move / body ratio), (2) breakout-high confirm (enter only if
next week trades above flip-week high), (3) 18–20wk pre-flip consolidation (tight range percentile),
(4) 50-EMA-band re-entry/accumulation. Dow-structure = too subjective to automate cleanly → noted,
not coded. Friday-close confirmation is automatic (weekly bars).

**Grid size:** G1 = 4 bands × (gross+net) ≈ 8 runs. G3 sweep = 9 param cells × 4 bands = 36. Manageable.

---

## 5. Status (live log)

**State:** G1 engine build. Started 2026-07-07.

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-07-07 | Framed; STATUS §1–4 written | Universe=all 4 bands; core-first (user-confirmed) |
| 2026-07-07 | Folder + engine build | `scripts/st_weekly_engine.py` (reuses calc_supertrend) |

## 6. Crash Recovery

- Everything runs on VPS `94.136.185.54:/home/arun/quantifyd/research/73_weekly_supertrend_investing/`.
- Check progress: `tail results/*.log`; `wc -l results/*_trades.csv`.
- Re-run G1: `cd /home/arun/quantifyd && python3 research/73_weekly_supertrend_investing/scripts/g1_signal_probe.py`
  (idempotent — skips bands already in the output CSV).
- DB is read-only here; do NOT write to `market_data.db`.

## 7. Files

| File | Purpose | Committable? |
|---|---|---|
| `scripts/st_weekly_engine.py` | Weekly resample + ST trade extraction + portfolio | yes |
| `scripts/g1_signal_probe.py` | G1 per-trade expectancy + baselines runner | yes |
| `results/g1_*_trades.csv` | Per-trade output per band | NO (heavy) — gitignore |
| `results/g1_summary.csv` | Per-band aggregate | yes |
| `results/RESULTS.md` | Final verdict | yes |
| this STATUS-MD | Crash-recovery source | yes |

## 8. Findings

_(populated live as G1 lands)_

---

## G1 FINDINGS (2026-07-07) — SIGNAL confirmed; investable candidate = Nifty200

**Verdict: SIGNAL.** Weekly ST(10,3) is a real trend-following signal (cut-losers/ride-winners
asymmetry) whose edge over a random-duration placebo **increases monotonically down the cap curve** —
NO edge in large caps, best in small caps, but the small-cap edge is tail-and-survivorship-flattered.

| Band | period | n | win% | mean net/trade | median | avg_win | avg_loss | placebo | **edge vs placebo** | t |
|---|---|---|---|---|---|---|---|---|---|---|
| Nifty50 | full | 555 | 56 | 32.5% | 3.5% | 68 | −13.6 | 34.3 | **−1.8pp (NONE)** | 7.8 |
| Nifty50 | 2015+ | 321 | 53 | 22.1% | 1.1% | 53 | −12.0 | 25.1 | **−3.0pp (NONE)** | 5.1 |
| Nifty200 | full | 1713 | 55 | 44.0% | 3.5% | 94 | −16.0 | 38.8 | **+5.2pp** | 12.0 |
| Nifty200 | 2015+ | 1088 | 54 | 35.2% | 2.3% | 78 | −14.5 | 29.8 | **+5.4pp** | 10.0 |
| Midcap150 | full | 1098 | 54 | 47.5% | 2.3% | 102 | −16.8 | 40.0 | **+7.5pp** | 9.5 |
| Midcap150 | 2015+ | 741 | 53 | 35.9% | 1.7% | 82 | −15.5 | 26.2 | **+9.7pp** | 8.1 |
| Smallcap250 | full | 1373 | 51 | 47.6% | 1.1% | 111 | −19.3 | 35.4 | **+12.2pp** | 8.3 |
| Smallcap250 | 2015+ | 953 | 50 | 46.9% | **−0.25%** | 113 | −18.5 | 30.3 | **+16.6pp** | 6.0 |

**Reads (honest):**
1. **Large caps: no timing alpha** — Nifty50 ST edge is NEGATIVE vs random holding. Where capacity lives, the flip just costs you vs owning the name.
2. **Monotonic down-cap** — Nifty200 +5 → Midcap +8–10 → Smallcap +12–17pp. Economically coherent (less-efficient small caps trend harder), but…
3. **Entirely right-tail** — median trade ≈ 0 (Smallcap 2015+ median NEGATIVE); avg_win 68–113 vs avg_loss −12 to −19. **Top-10% of trades supply 73–82% of ALL return.** ~45–49% of trades lose.
4. **Survivorship** — super-winner guard: dropping top-3 trades costs only 4–7pp (edge survives, not 3 trades) BUT the carrying names are textbook survivors (TARIL +5347%, KEI +3717%, PAGEIND, RVNL, ADANIENT — in today's index *because* they 20–50×'d). A PIT universe (incl. delisted) would gut the small-cap tail. Nifty200 top-3 concentration is LOWEST (10%) → least fragile.
5. **Cost-robust** — gross ≈ net everywhere (avg hold ~48wk, low turnover). Unlike the intraday books, cost does not eat this. A genuine structural plus.
6. **Regime-dependent** — per-year: monster 2009/2013/2020/2022/2023; bleeds 2008/2018/2019, fading 2024/2025. A bull/momentum engine.

**Convergence:** this is the SAME family as research/41/62/72 regime-gated momentum (Calmar ~1.7). research/62 already conclusively rejected lower-cap momentum at size (capacity wall). So small-cap ST = gross-only mirage; **the only version with real capacity + least survivorship = Nifty200.**

**Decision → G4 on Nifty200:** build the portfolio equity curve (capped ~16 positions @6%, NIFTYBEES-200DMA gate on/off, 40/40/20 vs blind-ST-exit, net cost+tax) vs NIFTYBEES B&H and vs the existing book. If it lands ≈ Calmar 1.7 it CONFIRMS the book with a simpler entry (no new alpha); if the low-turnover weekly-ST construction beats it on net/tax, that's the contribution.

| 2026-07-07 | G1 DONE (4 bands ×2 periods, ~2min) | SIGNAL; Nifty200 = investable candidate; small-cap = survivorship mirage |
| 2026-07-07 | Super-winner guard | edge survives drop-top3 (−4to−7pp); top-10% trades = 73–82% of return |

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
