# Neutral-Book Tail Diversifier — what to run ALONGSIDE a book that is entirely short-vol

**STATUS: DONE** · 2026-08-27 · research/134 · verdict **CONCLUDED — the diversifier is long equity, already owned**  
Full findings: [`results/RESULTS.md`](results/RESULTS.md)

---

## 1. The Ask

**What you asked:**

> "my only concern is, with this, all our systems are neutral bias and has more
> correlation, we shud hv some uncorrelated system which can be directional, need
> to be [not] fully successful in all metrics by itself, but by combination we shud
> be able to avert the trending moves results in our neutral systems into deep losses."
>
> (earlier, same thread) "Can we identify ways may be to short or long stocks — may
> be by putting on some opt strategies like debit spreads, skewed iron condors, jade
> lizard etc? Im ok to not make money out of it, win rate need not be the point, but
> profits during extremes will be good... even simple put options, or call options."

**Assumption I am making explicit** (the sentence reads either way): the sleeve
**need not** stand up on its own metrics. The prior message — "I'm ok to not make
money out of it, win rate need not be the point" — settles it. So the sleeve is
judged on the **combined** book, not on itself. If a candidate happens to also be
strong standalone, that is a bonus and will be reported separately.

**What we are actually testing:**

Given a book that is now almost entirely **short volatility and delta-neutral**
(45-DTE NIFTY straddle, C1 stock winged strangles, the NAS intraday straddle suite),
does adding a **directional / long-convex sleeve** improve the *combined* outcome —
specifically the worst month and the max drawdown — by **more than the honest
alternative of simply trading the neutral book smaller**?

That last clause is the whole study. Any hedge reduces drawdown; so does trading
fewer lots. The question is only ever whether the hedge does it **more cheaply**.

---

## 2. Economic hypothesis (G0)

**Mechanism.** A short-vol book's loss is a convex function of realised move: it
earns a bounded credit and pays an unbounded move. Its losses are therefore
concentrated in the few windows where |return| over the holding horizon exceeds the
premium sold. This is not random — it clusters, because large moves cluster.

**Why a directional sleeve can help.** Two distinct families, with different logic:

| Family | What it is | Why it should pay when we bleed | What it costs |
|---|---|---|---|
| **L — long convexity** | long puts/calls, debit spreads, backspreads | mechanically long the exact thing that hurts us (gamma/vega) | negative carry, every calm month |
| **T — trend following** | long/short directional on price, no vol sign | large moves ARE trends; a trend system is *long* persistence of the move that kills a straddle | whipsaw in choppy markets — i.e. exactly when the neutral book is *winning* |

**Family T is the more interesting claim**, because its cost shows up when the
neutral book is making money, and its payoff when the neutral book is losing. That
is the definition of a diversifier, and it is the classic "crisis alpha" argument
for managed futures. Family L pays for the same protection with certain, continuous
theta.

**Counterparty / why it persists.** Family L is not an edge at all — it is buying
insurance, and we should *expect* to lose money on it. Family T's premium is
behavioural (under-reaction to news, forced flows) and is the most-replicated
anomaly in the literature — but also the most crowded, so decay risk is real.

**Structures the ask names that do NOT solve the problem — stated up front:**
**jade lizards and skewed iron condors are themselves net short volatility.** They
would *raise* the correlation this study exists to lower. They can improve the shape
of a short-vol payoff, but they cannot be the uncorrelated sleeve. They are excluded
from the sleeve search and noted here so the exclusion is deliberate, not an
oversight.

**Prior art that constrains this study** (do not re-derive):

- **research/103 — NO EDGE.** Every naked long-convexity family tested (put/call
  backspread, Batman, broken-wing fly, iron fly) *loses* on NIFTY 2015–26: pure theta
  bleed. One survivor: **CALL_BACKSPREAD gated on ATR contraction**, +₹7.25L, 42% win,
  −₹0.90L DD. Crucially, r/103 judged everything **standalone vs NIFTY B&H** — it never
  asked whether a sleeve improves a short-vol book. That is the gap this study fills.
- **research/105 — SIGNAL, not STRATEGY.** Put-hedge overlay on the momentum book is
  tenor-dependent.
- **research/128 — REFUTED.** Index wings do not protect stock-strangle tails; those
  tails are idiosyncratic. Implication: an index-level sleeve may protect the NIFTY
  straddle and the NAS book but **cannot be assumed to protect the C1 stock book**.
  This must be measured per-leg, not assumed.

---

## 3. The base — what is being measured

### 3a. The neutral book (the thing to be protected)

Assembled from the studies that justified each live/paper book, at a **monthly**
frequency, equal-risk weighted unless stated:

| Sleeve | Source | Period | Cadence |
|---|---|---|---|
| C1 stock winged strangles | `research/127/results/phase_e_equity.csv` | 2016-03 → 2026 (87 mo) | monthly cycle |
| 45-DTE NIFTY straddle | `research/119/results/trades_daily.csv` (89 trades) | 2019 → 2026 | monthly |
| NAS intraday straddles | `nas_trading.db` (90 daily states) — **too short** | 2026 only | daily |

**Honest scoping decision:** NAS is excluded from the combined series and studied
separately if this reaches G4. Its 90 days of live history cannot support a
portfolio claim, and its failure horizon is **intraday** (one-way sessions), which
is a different problem from the multi-week trend that hurts the 45-DTE and C1 books.
A sleeve tuned to monthly moves is not evidence about NAS. Saying so now prevents a
false "the whole book is protected" conclusion later.

### 3b. Sleeve candidates

**Family L — long convexity on NIFTY** (monthly, sized as % of book):
1. Long ATM / 2% / 5% OTM put, 30d, held to expiry or rolled
2. Long call, same grid (the crash is not always down — 2020-11 and 2023-12 hurt short calls)
3. Put debit spread, call debit spread (cheaper carry, capped payoff)
4. Call backspread + ATR-contraction gate — the r/103 survivor, re-tested in *this* frame
5. Long strangle (both tails)

**Family T — directional trend, no vol sign**:
6. NIFTY long/short: 200DMA state, Donchian(20/55) breakout, 12-1 time-series momentum
7. Cross-sectional stock momentum long/short (universe already in `market_data.db`)
8. Long-only trend with a cash-exit gate (the "no shorting" variant)

**Costs.** Options: the study's own slippage model (0.5% of premium turnover, plus
the r/119 point-cost model for NIFTY). Futures/cash trend: 10 bps round trip.
Every result reported **gross and net**, with a cost sensitivity.

### 3c. Success criterion — and the controls that can kill it

Ranked on the **combined** book, not the sleeve:

- **Primary:** combined **Calmar** (CAGR / MaxDD) and **worst month**.
- **Secondary:** conditional capture — sleeve P&L in the neutral book's worst-decile
  months. A sleeve that pays in the *wrong* months is noise, however good its average.

**Pre-declared falsification (decided now, before any number is seen):**

| # | Control | This study is ABANDONED if… |
|---|---|---|
| **C1** | **SIZE-DOWN null** — shrink the neutral book until its worst month equals the hedged book's | …size-down delivers **equal or higher CAGR** at the same worst month. Then the sleeve is an expensive way to do what one fewer lot does for free. |
| **C2** | **Random-timing** — same structure, entry dates shuffled | …the gated version does not beat random timing. Then the "signal" is just the structure's beta. |
| **C3** | **Per-leg** — sleeve vs C1 stock book specifically | (not fatal, but) if the sleeve only protects the NIFTY leg, it must be reported as a partial hedge — r/128 says stock tails are idiosyncratic. |
| **C4** | **Era split** — 2016–20 vs 2021–26 | …the benefit exists in only one era. |

C1 is the one that matters. Most published "tail hedge" results die on it.

---

## 4. Plan — stages and cell counts

| Stage | What | Cells | Gate to pass |
|---|---|---|---|
| **A** | **Characterise the problem.** Build the combined neutral monthly series; measure pairwise correlation, joint drawdown, and *what NIFTY was doing* in the worst months. No sleeve yet. | — | Is the tail actually concentrated + trend-linked? If the bad months are idiosyncratic and uncorrelated to index moves, **no index sleeve can help** and the study redirects. |
| **B** | **Family T probe** — trend rules on NIFTY + cross-sectional, standalone stats & correlation to the neutral book | ~24 | corr ≤ 0 to neutral book, and positive net standalone expectancy |
| **C** | **Family L probe** — long-convexity structures priced on real option data where available, model where not | ~30 | conditional capture in worst-decile months > cost of carry |
| **D** | **Combination** — sweep sleeve weight 0–40%, combined Calmar / worst-month, **against the C1 size-down null** | ~40 | beats size-down on CAGR at matched worst month |
| **E** | Robustness — C2 random timing, C3 per-leg, C4 era split, cost sensitivity | — | survives all four |

Stage A is cheap and may kill or redirect everything. It runs first, alone.

---

## 5. Status

**Phase: A — characterising the joint loss profile.**

| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-08-27 19:4x | Study opened, sections 1–4 written before any run | prior art r/103, r/105, r/128 read first |
| 2026-08-27 19:4x | Data coverage confirmed | 127: 87 monthly rows; 119: 89 trades; NIFTY50 daily 2011→2026 (3,880); INDIAVIX 2015→2026 |
| 2026-08-27 19:4x | NAS excluded from the combined series, with reason | 90 days live, and an intraday failure horizon |
| 2026-08-27 20:0x | **Stage A DONE — gate passed, but the premise is inverted** | see findings below |
| 2026-08-27 20:0x | Robustness checks on the Stage A finding | dense era only, and the VIX-filtered live ruleset — both agree |
| 2026-08-27 20:2x | Stage B — sleeve build + combination sweep vs the size-down null | Donchian rows found BROKEN (never entered: close vs max-of-highs-incl-today) — discarded, not reported |
| 2026-08-27 20:3x | Stage C — Donchian fixed, era split, equity-premium strip | all controls pass; timing loses to plain B&H |
| 2026-08-27 20:3x | **DONE** — RESULTS.md written, verdict CONCLUDED | recommendation is portfolio weighting, not a new system |

**STAGE A HEADLINE: the book's enemy is the low-vol MELT-UP, not the crash.**

---

## 6. Crash recovery

Nothing long-running has been launched yet. When Stage A runs:

```bash
ssh arun@94.136.185.54
cd /home/arun/quantifyd/research/134_directional_diversifier
tail -40 results/stage_a.log          # progress
ls -la results/                       # what exists
./venv/bin/python3 scripts/stage_a_problem.py   # safe to re-run, idempotent
```

Nothing in this study writes to any trading DB, places any order, or touches any
live service. It reads `market_data.db`, `options_data.db` and prior research CSVs
only. It is safe to kill and restart at any point.

---

## 7. Files

| File | Purpose | Committable |
|---|---|---|
| `NEUTRAL_BOOK_TAIL_DIVERSIFIER_MONTHLY_SWEEP_STATUS.md` | this file | yes |
| `scripts/stage_a_problem.py` | build combined neutral series, joint loss profile | yes |
| `results/stage_a_*.csv` | monthly series + correlation matrix | yes (small) |
| `results/RESULTS.md` | final verdict | yes |

---

## 8. Findings

### Stage A — the joint loss profile (75 common months, 2019-08 → 2026)

**A1. The two neutral sleeves are less correlated than feared: corr = +0.32.**
Not one bet. The combined book (equal-risk, scaled to 4%/mo vol) runs +1.70%/mo
with a −10.4% max drawdown. The concern was reasonable but the diversification
between C1 and the 45-DTE straddle is already real.

**A2. The damage comes from UP-trends, not down-trends.** This is the study's
central finding and it inverts the instinct behind the request.

| 45-day NIFTY run | n | mean month | worst month |
|---|---|---|---|
| ≤ −5% (down trend) | 7 | **+3.62%** | −1.19% |
| \|run\| < 5% (chop) | 48 | +2.62% | −5.67% |
| ≥ +5% (**up trend**) | 20 | **−1.19%** | **−9.27%** |

corr(book, up-run) = **−0.532**; corr(book, down-run) = **+0.222**. Nine of the
ten worst months had NIFTY *rising*. The worst month in the sample, 2023-12
(−9.27%), came with a **+15.2%** 45-day run. 2020-12 (−5.27%) came with **+19.2%**.

**A3. The deepest down-run in the sample was profitable.** April 2020 — NIFTY's
45-day run at **−18.4%** — the book returned **+7.85%**, the 45-DTE leg alone
+14.13%. Mechanism: a sell-off expands implied vol, so the premium sold is rich and
mean-reverts; a melt-up is a *low-vol* grind that walks through the short call with
no vol spike to compensate. Short vol is not short the market — it is short
*surprise*, and India's surprises to the upside are the quiet ones.

**A4. It survives both robustness checks.**
- Dense era only (2021+, when C1 actually held 4–10 positions): down n=5 mean
  **+3.74%**, worst **+0.18%**; up n=17 mean −0.90%, worst −9.27%.
- The **VIX-rank>25 filtered** 45-DTE ruleset — i.e. the book actually being run
  live, 61 of 89 trades: down n=8 mean **+5.40%**, worst **+1.05%** (never lost);
  up n=19 mean +0.18%, worst −7.24%, **total contribution +3.35%** across all
  nineteen. The up-trend regime is where the book earns nothing and risks most.

**A5. What this kills, before it is built.** A long-put tail hedge — the reflex
answer, and what the ask reached for first — is the *wrong* hedge for this book.
It would pay carry every month to insure a state (down-trend) in which the book has
never lost money in 75 months. research/103 already showed naked long convexity
bleeds; this shows that for *this* book it would also be aimed at the wrong tail.

**A6. The honest limit of A2/A3.** Only 7 down-trend months (5 in the dense era).
The sample contains one fast V-shaped crash and no slow grinding bear market — no
2000–03, no repeated-leg 2008. The correct statement is: **the down-tail is
unmeasured, not proven safe.** A multi-quarter bear with sustained elevated vol
and no mean-reversion is a real risk this data cannot speak to, and any conclusion
here must carry that caveat.

### What Stage A implies for the sleeve search

The sleeve should be **long the upside grind**, not long the crash. That changes the
candidate list materially — and it points first at books that already exist
(Momentum-30, Breakout, HA-2green are all long-only equity trend). Stage B tests
whether they are the diversifier already sitting in the portfolio.
