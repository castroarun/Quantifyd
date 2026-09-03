# RESULTS — why the Wed→Fri condor flipped sign: the engine, not the rules

**Verdict: THE PRICING ENGINE, alone. On 334 campaigns matched day-for-day — same
anchor, same period, same strikes, same no-stop rule — the simulated engine says
+₹1,029/campaign and real traded prices say −₹328. The gap is ₹1,357 and it is
positive in every one of the eight years. Every rule difference I suspected turns out
to be worth less than ₹100/campaign. research/140's REFUTED verdict stands, and stands
on firmer ground than when it was written.**

research/143 · 2026-09-03 · 2 lots (130 qty) throughout

---

## 1. Why this run exists

Arun: *"if our earlier backtest says very positive calmar 1.63, win so and so, how come
now we are saying loss making? ensure if both rules are same first."*

The right challenge, and I had not earned the claim. research/140 reported that the
engine "was wrong by ₹1,073 per campaign" — but research/140 did not run research/80's
strategy. It ran a **different** one and compared the results. Four things changed at
once and I attributed the whole gap to one of them.

| | research/80 (Calmar 1.63) | research/140 (REFUTED) |
|---|---|---|
| **Anchor** | expiry exactly **6 calendar days** out → exit at ≤4 | **Wednesday** → **Friday** |
| what that *is* | NIFTY expiry was **Thursday until Sep-2025**, so this is mostly **FRIDAY → MONDAY, over a weekend**, DTE 6 → DTE 3 | never over a weekend; DTE 8 → DTE 6 in the Thursday era |
| **Stop** | close if combined premium **doubles**, checked on 5-min bars | **none** (matching the paper book) |
| **Instrument** | a **synthetic weekly expiry every Thursday from 2015** | whatever expiry actually traded, DTE 4–11 |
| **Prices** | Black-Scholes off India VIX, per-DTE IV multiplier, **no skew** | real NSE bhavcopy closes |
| **Period** | 2015–2026 | 2011–2026 |
| Strikes | 0.8% shorts / wings 1.0% beyond | **same** |

The instrument line is a defect in its own right: **NIFTY weekly options did not exist
until Feb 2019.** Our own bhavcopy shows 14 distinct expiries a year through 2018, then
56 in 2019. Four of research/80's "11/12 positive years" traded a contract that was
never listed.

So: hold everything constant but one difference at a time.

## 2. The ladder

Each row changes exactly one thing from the row above.

| arm | n | mean | annual | win | maxDD | Calmar | t | +yrs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **A** BS · r80 anchor · stop ×2 · 2015-26 | 545 | **+921** | +41,815 | 75% | −23,591 | **1.77** | +8.34 | 12/12 |
| **B** BS · r80 anchor · **no stop** · 2015-26 | 545 | +915 | +41,572 | 75% | −26,503 | 1.57 | +8.20 | 12/12 |
| **B19** BS · r80 anchor · no stop · **2019-26** | 357 | +1,047 | +46,706 | 73% | −26,503 | 1.76 | +6.46 | 8/8 |
| **C** **REAL** · r80 anchor · no stop · 2019-26 | 370 | **−265** | −12,243 | 57% | −133,703 | −0.09 | −1.50 | 2/8 |
| **D** REAL · **Wed→Fri** · no stop · 2019-26 | 349 | −201 | −8,748 | 59% | −135,549 | −0.06 | −1.12 | 3/8 |
| **E** REAL · Wed→Fri · no stop · **2011-26** | 434 | −193 | −5,223 | 59% | −135,549 | −0.04 | −1.30 | 7/16 |

Arm A reproduces the published headline (+₹921 vs +₹880, Calmar 1.77 vs 1.63) — close
enough on an independent re-implementation to call it the same result.

**Reading the ladder, one difference at a time:**

| difference | worth |
|---|---:|
| the **stop** (A→B) | **₹6/campaign** — and it *never fired*: 0% of 545 |
| the **pre-weekly fiction** (B→B19) | **−₹132** — removing the fake years makes it look *better* |
| **the pricing engine** (B19→C) | **₹1,312** ← the entire flip |
| the **anchor** (C→D) | ₹64 |
| the **period** (D→E) | ₹8 |

## 3. The pricing gap, matched day for day

Arms B19 and C share 334 identical entry days. Same anchor, same strikes, same rule,
same fortnight of market — only the prices differ.

| | |
|---|---:|
| BS engine | **+₹1,029/campaign** |
| Real traded prices | **−₹328/campaign** |
| **gap** | **+₹1,357** |

| year | n | BS mean | REAL mean | gap |
|---|---:|---:|---:|---:|
| 2019 | 44 | +516 | −178 | +694 |
| 2020 | 46 | +173 | −545 | +718 |
| 2021 | 44 | +1,028 | −319 | +1,347 |
| 2022 | 49 | +1,334 | −354 | +1,689 |
| 2023 | 48 | +1,653 | −267 | +1,920 |
| 2024 | 41 | +1,716 | +55 | +1,661 |
| 2025 | 39 | +1,175 | −578 | +1,753 |
| 2026 | 23 | +297 | −531 | +828 |

**Positive in all eight years, and widening.** A bias that flips sign year to year
could be noise; one that points the same way every year, growing as index levels rise,
is structural.

The mechanism is the one research/80's own documentation warned about: the engine has
**no volatility skew**, only a per-DTE IV multiplier. Real OTM puts trade at a higher
implied vol than the ATM the multiplier is calibrated to — so the engine prices the
**long wings too cheaply** and hands the structure a credit the market never offered.
The wings are exactly where a condor's economics live, so the error lands squarely on
the P&L.

## 4. Two things I got wrong, corrected

1. **research/140 claimed the engine was wrong by ₹1,073/campaign by comparing two
   different strategies.** The comparison was invalid as constructed. Run properly it
   is **₹1,312–1,357** — the conclusion survived, but the reasoning that produced it
   did not, and I should not have stated it as a clean like-for-like.
2. **The `/app/backtest/fardte-rescue` page describes the entry as "Wednesday close
   (DTE6 against the Tuesday weekly expiry)."** That is what the rule became after
   Sep-2025. For 2015 → Sep-2025 — 92% of the sample — the same rule entered on a
   **Friday** and exited **Monday**, carrying a weekend. The page described the last
   9 months of a 11-year test.

Both are the same failure: reporting what a rule was *meant* to be rather than
checking what it *did* over the sample.

## 5. What this changes

Nothing about the verdict, and everything about the confidence in it.

- The condor is **REFUTED** — and now on matched rules, not on a comparison of two
  different constructions.
- The **stop is irrelevant**, so "but it was tested with a stop" is not a defence. In
  the simulated world it never triggered once in 545 campaigns.
- The **anchor is irrelevant** — Wed→Fri and Fri→Mon are both about −₹200/campaign on
  real prices. It is not a weekend-risk story either.
- **Every result in research/80 that leaned on this engine is suspect**, not only the
  condor. Four of its five ideas were *killed* by the engine, and a credit-inflating
  bias makes killing conservative — those stay dead. The one that **survived** is the
  one this bias would manufacture, and it did.

## 6. Guards

| deadly sin | control |
|---|---|
| like-for-like | one difference per rung; 334 campaigns matched day-for-day for the pricing test |
| survivorship / stale prices | `contracts > 0 AND close > 0` on every leg, both dates |
| look-ahead | strikes snapped to the entry day's own traded ladder; exit priced on the exit day |
| overlapping trades | one campaign per week per arm — independent observations |
| instrument reality | weekly-expiry availability checked against bhavcopy, not assumed |
| period effects | 2019-26 run separately so the pre-weekly era cannot carry the result |

**Limitation:** arm C cannot check a ×2 stop intraday, because bhavcopy is end-of-day.
It does not matter here — the stop never fired in the simulated world and the two
no-stop arms are what the paper book actually runs.

## 7. Actions

1. **Correct `/app/backtest/fardte-rescue`** — it still advertises Calmar 1.63,
   +₹38,944/yr and "~100% p.a. on margin". *(done, this commit)*
2. **Correct research/140's claim** that the engine was "wrong by ₹1,073" — the number
   is right, the comparison that produced it was not. *(done, this commit)*
3. Keep the paper book running at 10 lots as a forward record, still labelled refuted.
4. **Re-check the other engine-derived conclusions in research/80** — the kills are
   safe, the survivor was not.
