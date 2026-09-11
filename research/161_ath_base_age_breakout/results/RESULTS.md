# research/161 — New-ATH-close breakout: does the AGE of the previous high, and volume, add value?

## VERDICT: **STRATEGY (candidate)** — base age YES, volume NO, saucer NO

Arun asked three things. The answers, in one line each:

| His question | Answer |
|---|---|
| *"looking at only ATH closes?"* | Yes — but the **exit matters far more than the entry**. A plain new-ATH-close book with Open Alpha's own exits makes **6.81%**; the same entries with a SuperTrend(14,4) trail make **18.66%**. |
| *"ATH closes where the last ATH is at least x candles before?"* | **Yes, this adds real value.** X ≥ 60 bars **plus** a ≥ 20% base depth lifts CAGR **18.66% → 21.26%** and cuts drawdown **−41.59% → −34.80%** (Calmar 0.449 → 0.618). |
| *"Breakout with volumes...?"* | **No. Do not add a volume filter.** It raises expectancy per trade (+12.43% → +13.04%) but cuts the event count, and the book gets **worse**: CAGR 21.26% → 20.51% and drawdown **−34.80% → −44.14%**. |

And the bonus answer, from running research/159's shape through the same engine:
**the saucer requirement is strongly negative** — 5.90% CAGR at 5.6 trades a year.

---

## 1. The adoption bar — all five criteria PASS

Pre-registered in the STATUS doc **before any cell ran**. Winner:
**X ≥ 60 bars · base depth ≥ 20% · no volume filter · no saucer · SuperTrend(14,4) close trail,
no hard stop · 16 slots @ 6.25% · ₹10L · 25 bps · after tax · liquidity ≥ ₹2 cr.**

| # | Criterion | Result | |
|---|---|---|---|
| 1 | **≥ 20% after-tax net CAGR** (30-seed median) | **21.26%** — worst seed **19.87%** | **PASS** (see caveat 1) |
| 2 | Beats NIFTYBEES on CAGR **and** drawdown in **both** windows | pre-2016 **19.33% / −32.45%** vs **12.68% / −59.71%**; 2016+ **23.24% / −32.43%** vs **11.88% / −36.34%** | **PASS** |
| 3 | Beats the honest in-engine plain-ATH **OA-proxy** cell | **21.26%** vs **6.81%** → **+14.45pp** | **PASS** |
| 4 | Beats a **date-matched random-entry** control | **21.26%** vs **12.11%** → **+9.15pp** | **PASS** |
| 5 | The winning X/K sits on a **plateau** | X=40 gives **21.23%**, X=60 **21.26%** — indistinguishable; K=none wins at every X | **PASS** |

**Headline (30 seeds, 25 bps, after tax, 03-Jan-2005 → 11-Sep-2026):**
**CAGR 21.26% [19.87 .. 21.89] · MaxDD −34.80% · Calmar 0.618 · win rate 49.2% ·
avg win +37.23% / avg loss −11.56% · expectancy +12.43%/trade · 31.7 trades/yr ·
max losing streak 14.**

---

## 2. The decision table (30 seeds each, identical engine and costs)

| Cell | CAGR med | worst seed | MaxDD | Calmar | WR | avg win | avg loss | expectancy | trades/yr |
|---|---|---|---|---|---|---|---|---|---|
| **WINNER** X≥60, depth≥20%, no vol | **21.26%** | 19.87% | **−34.80%** | **0.618** | **49.2%** | +37.23% | −11.56% | **+12.43%** | 31.7 |
| NEIGHBOUR X≥40, depth≥20% | 21.23% | 19.72% | −34.36% | 0.617 | 48.5% | +37.96% | −11.80% | +12.35% | 32.3 |
| WINNER at ₹5 cr liquidity | 20.62% | 19.26% | −36.20% | 0.583 | 49.2% | +35.85% | −11.31% | +11.90% | 30.7 |
| WIN + volume K≥2 | 20.51% | 19.76% | **−44.14%** | 0.465 | 47.1% | +41.69% | −12.48% | +13.04% | 29.6 |
| PLAIN ATH (X=0), ST(14,4) | 18.66% | 16.80% | −41.59% | 0.449 | 47.0% | +34.35% | −11.49% | +10.12% | 37.6 |
| **OA PROXY** (X=0, 15-SMA + −8%, ₹5 cr) | **6.81%** | 5.72% | −40.67% | 0.165 | 36.2% | +10.93% | −5.28% | +0.60% | 171.4 |
| WIN + saucer (r/159 shape) | 5.90% | 5.90% | −18.34% | 0.322 | 40.5% | +28.95% | −11.98% | +4.59% | **5.6** |
| Date-matched RANDOM control | 12.11% | 8.75% | −36.13% | 0.356 | 39.1% | — | — | +4.13% | — |
| **NIFTYBEES buy-and-hold** | **12.30%** | — | **−59.71%** | 0.206 | — | — | — | — | — |

### The decomposition that matters
- Open Alpha's own exits on plain ATH entries: **6.81%**
- Swap the exit to ST(14,4): **18.66%** → **the exit is worth +11.85pp**
- Add base age ≥ 60 bars and depth ≥ 20%: **21.26%** → **the entry filter is worth +2.60pp and −6.79pp of drawdown**

**So: yes, base age helps — but anyone reading this as "age is the big win" would be wrong.
The trail is the big win. Age is a real, smaller, second improvement that mostly buys
drawdown.** That is an honest answer to "how can we improve from here on": **change the exit
first, then add the age/depth filter.**

---

## 3. The X × K table Arun asked for

Best exit (ST(14,4), no stop), depth = any, saucer **OFF**, 10-seed scan:

**CAGR %**
| X \\ K | none | ≥2× | ≥3× | ≥5× |
|---|---|---|---|---|
| 0 | 18.88 | 18.36 | 19.55 | 19.00 |
| 20 | 18.07 | 19.71 | 18.04 | 18.12 |
| **40** | **20.68** | 18.66 | 18.84 | 17.81 |
| 60 | 18.85 | 18.10 | 17.79 | 16.00 |
| 120 | 18.03 | 18.09 | 17.34 | 15.29 |
| 250 | 19.02 | 17.68 | 17.20 | 14.77 |

**Win rate %** — essentially flat; volume does **not** buy accuracy
| X \\ K | none | ≥2× | ≥3× | ≥5× |
|---|---|---|---|---|
| 0 | 47.1 | 47.3 | 49.9 | 48.8 |
| 40 | 47.8 | 46.4 | 47.7 | 45.5 |
| 60 | **49.3** | 47.2 | 46.0 | 45.2 |
| 250 | 47.0 | 47.2 | 46.4 | 42.4 |

**Expectancy %/trade** — volume DOES improve each trade
| X \\ K | none | ≥2× | ≥3× | ≥5× |
|---|---|---|---|---|
| 0 | 10.25 | 10.28 | 11.15 | 10.88 |
| 40 | 11.53 | 10.67 | 11.13 | 11.87 |
| 120 | 10.36 | 11.16 | 11.56 | 11.73 |
| 250 | 12.37 | **14.01** | **15.12** | 14.53 |

**Events**
| X \\ K | none | ≥2× | ≥3× | ≥5× |
|---|---|---|---|---|
| 0 | 10,293 | 9,064 | 7,743 | 5,572 |
| 60 | 4,651 | 3,483 | 2,744 | 1,897 |
| 250 | 1,806 | 1,357 | 1,062 | 728 |

**How to read this.** Volume confirmation makes each trade **better** (expectancy rises
monotonically with K at long base ages — +12.37% → +15.12% at X=250) and makes the **book
worse** (CAGR falls, most sharply at long X: 19.02% → 14.77%). The reason is visible in the
event counts: K≥5 throws away **60%** of the opportunities. A 16-slot book that is already
starved of candidates cannot afford that trade. **Volume is a good filter for a discretionary
trader picking a handful of names, and a bad one for a mechanical book.** Worth saying to Arun
plainly, because both halves of that sentence are true at once.

With the **saucer requirement ON** the whole grid collapses to 3.6 trades a year and 6.08%
median CAGR — the shape is far too restrictive to fill a book.

---

## 4. Robustness

| Test | Result |
|---|---|
| **Cost ladder** (25 / 40 / 60 bps) | **21.26 / 20.34 / 19.35%** — shallow slope, low turnover |
| **Idle-cash contribution** | with 5.5% carry **21.26%**, with **no** carry **19.27%** → the cash yield contributes **~2.0pp**; the equity engine is doing the rest |
| **Two windows** | pre-2016 **19.33%** (−32.45%), 2016+ **23.24%** (−32.43%) — **positive and index-beating in both**, unlike research/159 |
| **Seed band (30)** | 19.87 .. 21.89%, median 21.26% |
| **Liquidity ₹5 cr** | 20.62% / −36.20% / Calmar 0.583 — survives the stricter floor |
| **Outlier dependence** | product of (1+r): all **1.36e21**, top-10 removed **1.02e16**, capped at +50% **5.67e11** — a **133,000×** collapse. **Still heavily tail-carried** |
| **Sweep** | 810 of 864 cells completed (54 skipped for < 20 events); **16 cells ≥ 20%**; 350 (43%) beat NIFTYBEES on both CAGR and DD; 361 (45%) beat the OA proxy |

---

## 5. Caveats — read before acting on any of this

1. **The worst of 30 seeds is 19.87%, a whisker below the 20% floor.** The bar was written as
   a 30-seed *median* with the worst seed stated, and on that reading it passes — but anyone
   who meant "every path must clear 20%" should read this as a **fail by 0.13pp**. Stated
   rather than rounded away.
2. **Outlier dependence is severe.** Removing ten trades out of 687 collapses compounded growth
   by a factor of 133,000. This is the same weakness research/159 had, and it is not fixed by
   the better entry. A book like this makes its money in a few names.
3. **Drawdown is deep in absolute terms** (−34.80%, worst seed deeper) even though it is far
   shallower than NIFTYBEES' −59.71%. A 14-trade losing streak on ~32 trades a year is roughly
   five months of nothing but losers.
4. **The OA-proxy comparison rests on another study's finding.** `research/159_oa_honest_reoptimization`
   established that Open Alpha's published ~34.9% rests on a same-bar look-ahead fill; their
   study was **still running** when this was written, so the 6.81% proxy here is *our own*
   in-engine number, not theirs. It happens to corroborate their ~9.5% range closely, but it is
   a proxy for OA's **entry and exit rules**, not a re-simulation of the live book.
5. **Survivorship.** The universe is symbols present in the DB today; delisted names never
   appear. Sharp for a pattern requiring a new all-time high.
6. **The DB is not retroactively split-adjusted.** ATH is computed only from bars after the last
   day-over-day fall worse than −35%; genuine highs on names that split are lost with the fakes.
7. **864 cells disclosed**; the winner is discounted for multiple testing. It is reported as a
   plateau (X=40 and X=60 are indistinguishable, and K=none wins at every X), not a spike.
8. **This is not a deployment recommendation.** No paper book, no live engine, no orders. It is
   a STRATEGY *candidate*: the next step, if Arun wants it, is a portfolio-fit test against an
   **honestly re-measured** Open Alpha — which cannot be done until that study lands.

---

## 6. What to tell Arun in one paragraph

Yes — you can improve on this, and the improvement is real. But the first and largest fix is
not the entry at all: **changing the exit from the 15-day-SMA-plus-8%-stop to a SuperTrend(14,4)
trail takes the same all-time-high entries from 6.8% to 18.7%.** On top of that, **requiring the
previous all-time high to be at least ~40-60 bars old and the stock to have fallen at least 20%
below it in between adds a further ~2.6 points of return and takes about 7 points off the
drawdown** — a genuine improvement, and it survives 30 seeds, both windows, the cost ladder and a
stricter liquidity floor. **Volume confirmation should not be added**: it makes each trade better
and the book worse, because it throws away 40-60% of the opportunities a 16-slot book needs. And
the semi-circle shape from the last study should be dropped entirely — it is far too rare to fill
a book and it costs about 15 points of CAGR.
