# RESULTS — the Wed→Fri iron condor on REAL option prices

**Verdict: REFUTED. 434 real campaigns over 15 years say −₹193/campaign at t −1.30.
research/80's +₹880/campaign and Calmar 1.63 were artefacts of a no-skew simulation.
Do NOT deploy. The paper book should be stopped or explicitly re-framed.**

research/140 · 2026-08-31

---

## 1. Why this run exists

Arun: *"but why we hv acces ot EOD options data for years from NSE bhavcopy right?"*

He was right and it was the obvious point I had missed. I had been repeating
research/80's caveat — *"never tested on a real chain"* — as though it were an
unavoidable limitation. It was not. `nse_options_bhav` holds **5,179,544 real NIFTY
option rows over 3,861 trading days, 2011-01-03 → 2026-08-28**, and this strategy
**enters at a Wednesday close and exits at a Friday close**. Daily closing prices are
not an approximation for it — they are precisely the two prices it transacts at.

So the engine could be replaced outright rather than caveated.

## 2. Construction — research/80's spec, unchanged

| | |
|---|---|
| Entry | Wednesday close |
| Structure | SELL ~0.8%-OTM strangle, BUY each wing **1.0% beyond its own short** |
| Expiry | nearest with DTE 4–11 (a genuine ~1-week expiry; mean obtained **7.7**) |
| Exit | Friday close |
| Stop | **none** — matching what `condor_paper.py` actually runs |
| Size | 2 lots (qty 130), ₹20/leg round-trip cost |

Strikes are **snapped to strikes that genuinely traded that day** (`contracts > 0`),
not to arithmetic ideals — a real book trades the strike the market quotes. Realised
mean vertical width **163 pts**, and on the most recent cycle the snap reproduces the
live book's own strikes exactly (short CE 24450 / long CE 24700 at spot 24,277).

## 3. The result

| | research/80 engine | **REAL bhavcopy** |
|---|---:|---:|
| sample | 11 yrs, simulated | **434 campaigns, 2011–2026** |
| mean / campaign | **+₹880** | **−₹193** |
| net (2 lots) | +₹38,944/yr | **−₹83,569 total** |
| t | — | **−1.30** |
| win rate | 75% | 59% |
| max drawdown | −₹23,820 | **−₹1,35,546** |
| worst campaign | — | −₹17,116 |

**The engine was wrong by ₹1,073 per campaign, and it was wrong in the direction its
own documentation warned about.** research/80's RESULTS.md says the engine has *no
volatility skew* and is weakest far-OTM — which is exactly where the long wings sit.
Without skew, the wings are priced too cheaply, so the simulated condor collects a
credit it could never have collected in the market.

**It is not a cost artefact.** At zero costs it is still **−₹48,849 (t −0.76)**. The
structure is negative gross.

## 4. It is getting worse, not better

| year | n | net | mean | win% |
|---|---:|---:|---:|---:|
| 2019 | 42 | +₹9,242 | +₹220 | 69% |
| 2020 | 48 | +₹6,195 | +₹129 | 67% |
| 2021 | 44 | −₹1,910 | −₹43 | 61% |
| 2022 | 46 | +₹15,533 | +₹338 | 63% |
| 2023 | 50 | −₹6,478 | −₹130 | 60% |
| **2024** | 42 | **−₹39,364** | **−₹937** | 50% |
| **2025** | 46 | **−₹43,482** | **−₹945** | 48% |
| **2026** | 31 | −₹9,721 | −₹314 | 55% |

**The three most recent years are the three worst.** Whatever edge the earlier period
had, it is not present now — which is the period any deployment would trade in.

Note the shape: **59% of campaigns win.** It wins often and small, and loses rarely and
large — the classic short-premium profile, except here the losses are not paid for.

## 5. What about the paper book's +₹1,351/cycle over 7 cycles?

It is **noise, not counter-evidence.** With sd ₹2,813 over 7 cycles the standard error
is ₹1,063, so +₹1,351 sits **1.45 standard errors** above the long-run −₹193 — entirely
consistent with drawing 7 lucky weeks from a distribution whose mean is slightly
negative. Seven cycles was never going to detect a −₹193 mean; it would need hundreds.

This is the trap I walked into earlier in this session when I reconciled the paper book
against the engine and reported "z = 0.44, CONSISTENT". It was consistent — **with a
number that was itself wrong.** Agreement with a bad benchmark is not validation.

## 6. Guards

| deadly sin | control |
|---|---|
| survivorship / stale prices | `contracts > 0 AND close > 0` on every leg, entry and exit (research/89 binding rule) |
| look-ahead | strikes chosen from the entry day's own traded ladder; exit priced on the exit day |
| overlapping trades | one campaign per week, Wed→Fri — 434 **independent** observations |
| cost neglect | costs charged, plus a 0/20/40/80 sensitivity — negative at every level |
| regime dependence | full per-year table; the decay is the headline, not a footnote |
| **data-integrity** | the spot symbol is `NIFTY50`, pinned and **asserted**. A first run used `'NIFTY 50'` (0 rows) and fell back to `LIKE 'NIFTY%50%'`, which also matches **NIFTY500** — a different index — silently corrupting every strike. It produced 6 campaigns instead of 434 and was caught by that absurdity, not by the numbers looking wrong. |

**Limitations, stated plainly:** EOD closes cannot see an intraday breach, so the
**stopped** variant (research/80's ×2 stop) is not evaluated here. That does not weaken
the verdict for the deployed book, because `condor_paper.py` runs **no stop at all** —
the unstopped variant tested here is exactly what is live on paper. A ×2 stop would
have to rescue −₹193/campaign to change the conclusion, and stops in this family have
repeatedly cost money rather than made it.

## 7. Actions

1. **Do not deploy the Wed→Fri condor.** The case for it was a simulation artefact.
2. **Stop the paper book, or re-label it** as a refuted structure being tracked for
   curiosity. Leaving it on `/app/straddles` presented as a promising SIGNAL is now
   misleading.
3. **Update research/80's published verdict** — `/app/backtest/fardte-rescue` currently
   advertises Calmar 1.63 and "~100% p.a. on margin". That page needs this correction.
4. **Re-check the other engine-derived conclusions in research/80.** The five ideas it
   tested were all valued on the same no-skew engine. Four were killed — killing is
   robust to a pricing bias that inflates credit. The one that *survived* is the one
   that bias would have created, and it did.
5. The 45-DTE straddle is unaffected — research/119 was built on this same bhavcopy
   data from the start, which is exactly why it is the stronger candidate.
