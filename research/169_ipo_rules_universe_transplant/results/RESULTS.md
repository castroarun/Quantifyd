# research/169 — Why is IPO Base IPO-specific? Its rules transplanted to Nifty-50 / 100 / 200 / Midcap / 500 / Smallcap / all-stock universes

## VERDICT: **NO EDGE** — the rules do not transplant to any universe. IPO Base is IPO-specific because the return lives in a stock's first months after listing, not in the breakout rule. Two corrections to the live IPO book come with it.

IPO — the answer first, then the evidence. Every figure is after tax (20% STCG / 12.5% LTCG with
Indian FY loss netting), net of 25 bps a side, idle cash credited at 5.2% post-tax, 8 slots at
18.75% of a ₹10,00,000 book, median of 30 paired seeds, window W2 2006-01-01 → 2026-09-04 unless
stated. WA = 2006–2015, WB = 2016–2026. "Null" = date-matched random entry: the same days, the
same number of entries, names drawn at random from THAT universe's eligible set whose day high
reached their own pivot, with the same fill, stop, target, trail and gate. The size universes are
a **causal traded-value PROXY**, not historical index membership (Q7). Nothing live was changed.

---

**Q1. Why is the IPO system IPO-specific?**

IPO — because the money is made by owning stocks in their first months after listing, and the
age band is the only rule that puts the book there. Remove it and the identical rules become a
25-day closing-high breakout on seasoned stocks, which earns 2–9% a year at −34% to −54%
drawdown and does not beat buying random names on the same days.

For a stock listed under six months, "the highest close of the last 25 bars" is close to its
all-time high since listing, and the base is its first base. For a seasoned stock the same words
describe a 25-day Donchian close breakout — a different signal, and under these exits not a signal
at all. Three measurements say where the return actually lives:

1. **Widen the age band and the book decays steeply and smoothly** (all-stock universe):
   ≤6 months 22.39% → ≤12 months 16.28% → ≤24 months 14.68% → any vetted listing 7.55% → no age
   limit 6.97% → seasoned >6 months 5.86% → seasoned >24 months 4.63%.
2. **Inside the six-month band, most of the return is the very first base, 25–60 sessions after
   listing.** Requiring 60 bars instead of 25 takes Spec A from 21.80% to 11.57% (research/167's own
   engine, Q6). Entries that early happen before a 50-day average exists, so for their first weeks
   they are +25% / −10% bracket trades on brand-new listings — and they are the best trades in the
   book: +8.4% per trade against +5.7% for trail-managed holds.
3. **The breakout is a smaller part of the story than research/167 reported** (Q5). From 2016 to
   2026, random young, liquid names bought on the same days with the same exits and gate earned
   what Spec A earned: 28.27% against 28.37%.

**Q2. What happens on Nifty 50, 100, 200, Midcap, 500, Smallcap and all stocks?**

IPO — every one fails, with the age band removed and with seasoned names only. Spec A's rules,
identical except for the universe, each against its own null:

| Universe (Spec A rules otherwise identical) | CAGR | worst seed | MaxDD (worst seed) | Calmar | null CAGR | W2 edge [range] | wins W2 / WA / WB | WA real / null | WB real / null | edge test |
|---|---|---|---|---|---|---|---|---|---|---|
| IPO Spec A — all stocks, listed ≤ 6 months | 22.39% | 21.09 | −24.79% (−35.08%) | 0.903 | 20.18% | +2.25pp [−0.40..+5.00] | 29 / 30 / 14 | 17.01 / 12.39 | 28.37 / 28.27 | FAIL |
| IPO live book as coded — ≤ 6 months, ≥ 60 bars | 12.39% | 11.30 | −36.35% (−41.72%) | 0.341 | 11.83% | +0.79pp [−0.83..+2.17] | 23 / 30 / 19 | 9.18 / 8.98 | 15.32 / 14.30 | FAIL |
| Nifty-50-like, no age limit | 3.04% | −0.00 | −43.62% (−61.89%) | 0.070 | 5.36% | −2.23pp [−6.33..+0.82] | 3 / 1 / 11 | 3.33 / 8.01 | 2.73 / 3.18 | FAIL |
| Nifty-50-like, seasoned > 6m | 2.27% | −0.25 | −45.42% (−65.39%) | 0.050 | 1.98% | +0.25pp [−3.58..+4.34] | 17 / 11 / 24 | 2.60 / 3.94 | 2.14 / 0.23 | FAIL |
| Nifty-100-like, no age limit | 4.39% | −2.57 | −43.54% (−70.93%) | 0.101 | 3.65% | +0.01pp [−3.46..+6.26] | 15 / 18 / 18 | 6.09 / 4.79 | 2.81 / 2.21 | FAIL |
| Nifty-100-like, seasoned > 6m | 4.00% | −1.70 | −45.41% (−64.62%) | 0.088 | 4.67% | −0.47pp [−4.50..+5.05] | 13 / 8 / 13 | 5.64 / 7.89 | 2.66 / 3.11 | FAIL |
| Nifty-200-like, no age limit | 5.51% | 1.58 | −48.17% (−65.77%) | 0.114 | 8.86% | −3.24pp [−8.87..+4.21] | 5 / 11 / 6 | 7.25 / 11.22 | 3.26 / 6.95 | FAIL |
| Nifty-200-like, seasoned > 6m | 5.79% | 1.60 | −48.85% (−60.88%) | 0.119 | 6.58% | −0.73pp [−6.20..+4.26] | 12 / 12 / 17 | 7.71 / 9.50 | 3.39 / 2.69 | FAIL |
| Nifty-500-like, no age limit | 5.53% | 1.46 | −49.71% (−62.75%) | 0.111 | 5.76% | −0.59pp [−6.96..+4.78] | 13 / 12 / 13 | 7.02 / 8.81 | 3.24 / 5.17 | FAIL |
| Nifty-500-like, seasoned > 6m | 5.24% | −1.28 | −48.74% (−74.00%) | 0.108 | 5.06% | −0.71pp [−10.19..+5.90] | 12 / 12 / 13 | 6.62 / 8.72 | 2.77 / 3.13 | FAIL |
| Midcap-like (rank 101-250), no age limit | 8.94% | 5.08 | −50.36% (−61.00%) | 0.177 | 7.81% | +1.52pp [−4.03..+5.49] | 20 / 23 / 15 | 13.45 / 9.90 | 4.40 / 4.70 | FAIL |
| Midcap-like, seasoned > 6m | 8.92% | 4.45 | −49.48% (−62.20%) | 0.180 | 11.00% | −1.70pp [−9.36..+3.28] | 8 / 6 / 12 | 13.53 / 16.61 | 4.45 / 6.13 | FAIL |
| Smallcap-like (rank 251-500), no age limit | 2.52% | −2.43 | −49.33% (−60.65%) | 0.051 | 3.03% | −0.84pp [−4.00..+3.12] | 13 / 0 / 25 | −1.08 / 2.47 | 5.45 / 2.73 | FAIL |
| Smallcap-like, seasoned > 6m | 2.46% | −1.80 | −53.57% (−69.12%) | 0.046 | 3.17% | −1.42pp [−6.05..+3.04] | 12 / 0 / 15 | −1.08 / 2.47 | 4.80 / 3.42 | FAIL |
| Beyond 500 (rank 501+), no age limit | 6.76% | 4.03 | −33.96% (−55.49%) | 0.199 | 8.21% | −1.39pp [−5.10..+1.80] | 8 / 0 / 9 | 4.86 / 5.78 | 7.95 / 10.62 | FAIL |
| Beyond 500, seasoned > 6m | 7.41% | 4.02 | −36.04% (−49.27%) | 0.206 | 6.80% | +0.69pp [−4.66..+4.42] | 20 / 0 / 19 | 5.11 / 5.11 | 9.39 / 8.26 | FAIL |
| All stocks, no age limit | 6.97% | 2.24 | −46.57% (−68.27%) | 0.150 | 7.18% | −0.04pp [−9.21..+5.27] | 15 / 4 / 17 | 8.98 / 13.33 | 5.06 / 2.42 | FAIL |
| All stocks, seasoned > 6m | 5.86% | 0.83 | −44.87% (−70.21%) | 0.131 | 6.87% | −1.50pp [−4.93..+6.53] | 11 / 10 / 10 | 6.29 / 8.96 | 3.63 / 4.25 | FAIL |

The pre-registered test for "the edge exists on universe X" was: W2 paired edge ≥ +1.0pp with the
real book beating its null on ≥ 25 of 30 seeds, AND ≥ 25/30 in both WA and WB. **0 of 16 transplant
cells pass, and none comes close**: the best W2 edge is +1.52pp (Midcap-like, no age limit) on 20
of 30 seeds, and 15 of 30 in WB.

Tradeability and capacity for the same cells (position size as a share of the name's 20-day
median traded value, at a ₹10 L book):

| Universe | trades / yr | median hold (days) | invested | win rate | avg win / avg loss | longest losing streak | net expectancy / trade (after 50 bps) | ten best trades’ share | median position, % of 20d traded value @ ₹10 L | p90 position @ ₹10 L | book size where p90 = 5% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| IPO Spec A — all stocks, listed ≤ 6 months | 20.3 | 29 | 37.1% | 48.3% | +21.4% / −7.4% | 11 | +6.00% | 14.5% | 1.48% | 9.86% | ₹5 L |
| IPO live book as coded — ≤ 6 months, ≥ 60 bars | 17.4 | 28 | 31.7% | 40.6% | +20.7% / −7.3% | 23 | +3.61% | 25.0% | 0.59% | 2.65% | ₹19 L |
| Nifty-50-like, no age limit | 33.3 | 31 | 67.7% | 34.6% | +14.1% / −5.8% | 26 | +0.58% | 44.8% | 0.02% | 0.07% | ₹742 L |
| Nifty-50-like, seasoned > 6m | 33.2 | 31 | 67.6% | 34.4% | +14.1% / −5.8% | 25 | +0.44% | 50.5% | 0.02% | 0.06% | ₹791 L |
| Nifty-100-like, no age limit | 34.5 | 32 | 68.9% | 35.5% | +14.6% / −5.9% | 22 | +0.83% | 35.5% | 0.04% | 0.16% | ₹305 L |
| Nifty-100-like, seasoned > 6m | 34.7 | 31 | 68.8% | 35.5% | +14.6% / −6.0% | 22 | +0.74% | 38.8% | 0.04% | 0.16% | ₹308 L |
| Nifty-200-like, no age limit | 35.8 | 31 | 69.5% | 35.6% | +15.6% / −6.1% | 21 | +1.06% | 30.9% | 0.07% | 0.38% | ₹133 L |
| Nifty-200-like, seasoned > 6m | 35.6 | 32 | 69.7% | 36.0% | +15.5% / −6.1% | 22 | +1.10% | 30.2% | 0.08% | 0.38% | ₹132 L |
| Nifty-500-like, no age limit | 36.8 | 30 | 69.5% | 35.6% | +15.9% / −6.5% | 19 | +1.04% | 31.1% | 0.12% | 0.55% | ₹92 L |
| Nifty-500-like, seasoned > 6m | 36.8 | 30 | 69.5% | 34.9% | +16.1% / −6.4% | 19 | +0.99% | 31.9% | 0.12% | 0.51% | ₹97 L |
| Midcap-like (rank 101-250), no age limit | 36.0 | 30 | 67.2% | 37.5% | +16.9% / −6.8% | 18 | +1.60% | 23.6% | 0.32% | 0.86% | ₹58 L |
| Midcap-like, seasoned > 6m | 35.9 | 30 | 67.3% | 37.6% | +16.7% / −6.7% | 17 | +1.57% | 22.9% | 0.32% | 0.83% | ₹60 L |
| Smallcap-like (rank 251-500), no age limit | 24.3 | 28 | 43.6% | 31.8% | +18.3% / −7.5% | 18 | +0.30% | 91.2% | 0.20% | 0.36% | ₹139 L |
| Smallcap-like, seasoned > 6m | 24.4 | 28 | 43.6% | 31.7% | +18.0% / −7.5% | 19 | +0.26% | 96.7% | 0.19% | 0.35% | ₹142 L |
| Beyond 500 (rank 501+), no age limit | 16.2 | 28 | 27.6% | 36.9% | +19.4% / −8.0% | 13 | +1.59% | 55.1% | 0.62% | 1.26% | ₹40 L |
| Beyond 500, seasoned > 6m | 16.2 | 28 | 27.6% | 37.6% | +19.4% / −7.9% | 13 | +1.82% | 48.9% | 0.64% | 1.32% | ₹38 L |
| All stocks, no age limit | 37.2 | 30 | 69.5% | 35.8% | +16.4% / −6.5% | 19 | +1.26% | 27.1% | 0.16% | 0.72% | ₹70 L |
| All stocks, seasoned > 6m | 36.9 | 30 | 69.5% | 36.2% | +16.0% / −6.5% | 18 | +1.06% | 29.3% | 0.14% | 0.62% | ₹81 L |

**Second check on true point-in-time market cap.** `fundamentals.db` `mcap_pit`, ranked on the
previous month's row, exists only from Aug-2018, so this runs 2018-09-01 → 2026-09-04:

| Universe, no age limit (2018-09 → 2026-09) | signals | CAGR | MaxDD | Calmar | null CAGR | edge (seeds real wins / 30) |
|---|---|---|---|---|---|---|
| IPO Spec A (same window) | 1,325 | 24.14% | −25.57% | 0.944 | 24.19% | −0.09pp (15) |
| Nifty-50-like — by PIT market cap | 5,789 | 1.19% | −37.47% | 0.032 | 2.26% | −1.50pp (11) |
| Nifty-50-like — by traded value | 5,623 | 0.72% | −41.26% | 0.017 | 0.77% | +0.02pp (15) |
| Nifty-100-like — by PIT market cap | 11,153 | −0.57% | −40.33% | −0.014 | 1.58% | −1.66pp (10) |
| Nifty-100-like — by traded value | 11,144 | 0.95% | −40.40% | 0.023 | 0.03% | +1.50pp (19) |
| Nifty-200-like — by PIT market cap | 21,180 | 1.64% | −39.26% | 0.042 | 1.34% | +0.19pp (15) |
| Nifty-200-like — by traded value | 21,902 | 2.13% | −45.73% | 0.046 | 4.71% | −2.89pp (12) |
| Nifty-500-like — by PIT market cap | 42,470 | 0.73% | −39.09% | 0.019 | 1.04% | −1.10pp (14) |
| Nifty-500-like — by traded value | 44,660 | 3.23% | −44.31% | 0.073 | 3.48% | −0.71pp (14) |
| Midcap-like — by PIT market cap | 14,316 | 3.50% | −38.89% | 0.090 | 5.97% | −2.72pp (11) |
| Midcap-like — by traded value | 15,518 | 3.75% | −41.02% | 0.091 | 1.72% | +1.85pp (18) |
| Smallcap-like — by PIT market cap | 17,001 | −0.72% | −45.71% | −0.016 | 6.47% | −5.43pp (4) |
| Smallcap-like — by traded value | 17,998 | 3.69% | −42.69% | 0.086 | 2.37% | +0.16pp (15) |

Market-cap universes are no better than the traded-value proxies — worse in the mid and small
bands — and Spec A itself ties its null in this window. **The verdict does not depend on how
"Nifty-50-like" is defined.**

Cost ladder (a transplant trades about 36 times a year against Spec A's 20):

| Spec | 25 bps a side | 40 bps | 60 bps | CAGR lost 25 → 60 |
|---|---|---|---|---|
| All vetted listings, any age | 7.55% / −44.20% | 6.32% / −47.97% | 3.69% / −54.49% | −3.86pp |
| Midcap-like (rank 101-250), no age limit | 8.94% / −50.36% | 7.19% / −54.47% | 4.30% / −61.19% | −4.64pp |
| Beyond 500, seasoned > 6m | 7.41% / −36.04% | 6.45% / −38.62% | 5.51% / −40.84% | −1.90pp |
| IPO Spec A — all stocks, listed ≤ 6 months | 22.39% / −24.79% | 20.83% / −26.49% | 19.40% / −28.47% | −2.99pp |
| IPO live book as coded — ≤ 6 months, ≥ 60 bars | 12.39% / −36.35% | 11.45% / −38.77% | 10.15% / −42.05% | −2.24pp |

**Q3. Is the edge in the young-stock condition, or in the breakout + 50-day-trail mechanics?**

IPO — not in the mechanics. The seasoned-only arm (listed more than 6 months ago) fails on all 8
size universes, with W2 edges from −1.70 to +0.69pp. The pre-registered "the mechanics are the
edge" needed a majority of universes to pass: **0 of 8 did.**

The return is in the young-stock condition, but the honest wording is narrower than "the age band
is the edge". That pre-registered label also required Spec A itself to pass the null test, and on
the clean panel it does not: +2.25pp in W2 (29/30), +4.63pp in WA (30/30), −0.18pp in WB (14/30).
What the six-month band holds is a **cohort** that pays when it is held with a slow trail and an
index gate. The breakout selection added about +4.6pp in 2006–2015 and nothing since 2016.

| Age band (all-stock universe) | signals | CAGR | MaxDD | Calmar | null CAGR | W2 edge (wins) | WA edge (wins) | WB edge (wins) | trades on stocks listed ≤ 6m |
|---|---|---|---|---|---|---|---|---|---|
| IPO Spec A — all stocks, listed ≤ 6 months | 1,707 | 22.39% | −24.79% | 0.903 | 20.18% | +2.25pp (29) | +4.63pp (30) | −0.18pp (14) | 99.1% |
| All stocks, listed ≤ 12 months | 3,635 | 16.28% | −31.52% | 0.516 | 18.04% | −1.93pp (9) | −0.33pp (9) | −3.34pp (5) | 49.6% |
| All stocks, listed ≤ 24 months | 6,502 | 14.68% | −38.73% | 0.379 | 16.29% | −1.29pp (10) | −3.23pp (0) | −0.37pp (14) | 30.2% |
| All vetted listings, any age | 32,974 | 7.55% | −44.20% | 0.171 | 6.87% | +1.53pp (22) | +4.80pp (29) | +0.36pp (16) | 8.8% |
| All stocks, no age limit | 89,619 | 6.97% | −46.57% | 0.150 | 7.18% | −0.04pp (15) | −4.33pp (4) | +0.96pp (17) | 2.0% |
| All stocks, seasoned > 6m | 87,610 | 5.86% | −44.87% | 0.131 | 6.87% | −1.50pp (11) | −2.16pp (10) | −1.67pp (10) | 0.0% |
| All stocks, seasoned > 24 months | 81,812 | 4.63% | −49.69% | 0.093 | 3.95% | +0.73pp (17) | −1.27pp (11) | +0.00pp (15) | 0.0% |

At ≤12 and ≤24 months the random null BEATS the breakout rule (18.04% vs 16.28%; 16.29% vs
14.68%): once the band is wider than six months, the rule picks worse young names than chance.

**Q4. Is a transplanted version just an existing book in disguise?**

IPO — partly, and a worse one. The broad transplants correlate 0.53–0.56 monthly with OA · Base
Age (the live Open Alpha book: all-time-high breakout after an aged, deep base), against IPO-A's
0.33, and every one of them makes the three-sleeve book worse on every path:

| Candidate | monthly corr TN / OA·BaseAge / IPO-A | replacing IPO-A at 25%: CAGR / MaxDD / Calmar | ΔCAGR (paths better) | ΔCalmar (paths better) | vs risk-matched cash (paths better) | 4th sleeve at 10%: ΔCAGR / ΔCalmar / vs cash | 4th sleeve at 25%: ΔCAGR / ΔCalmar / vs cash |
|---|---|---|---|---|---|---|---|
| All vetted listings, any age | 0.45 / 0.53 / 0.37 | 17.41% / −29.63% / 0.591 | −3.74pp (0/30) | −0.274 (0/30) | −2.74pp (0/30) | −1.19pp / −0.072 / −1.19pp (0/30) | −3.02pp / −0.189 / −3.02pp (0/30) |
| Midcap-like (rank 101-250), no age limit | 0.49 / 0.56 / 0.27 | 17.37% / −28.51% / 0.617 | −3.54pp (0/30) | −0.256 (0/30) | −2.75pp (0/30) | −1.16pp / −0.061 / −1.16pp (0/30) | −2.96pp / −0.188 / −2.96pp (0/30) |
| Beyond 500, seasoned > 6m | 0.23 / 0.32 / 0.25 | 17.06% / −25.42% / 0.683 | −3.84pp (0/30) | −0.201 (0/30) | −2.22pp (0/30) | −1.27pp / −0.036 / −0.81pp (0/30) | −3.21pp / −0.100 / −2.60pp (0/30) |
| IPO Spec A — all stocks, listed ≤ 6 months | 0.37 / 0.34 / 0.93 | 21.13% / −23.53% / 0.914 | +0.05pp (19/30) | +0.029 (30/30) | +2.86pp (30/30) | +0.25pp / +0.056 / +1.04pp (30/30) | +0.58pp / +0.146 / +2.47pp (30/30) |
| IPO live book as coded — ≤ 6 months, ≥ 60 bars | 0.25 / 0.34 / 0.78 | 18.80% / −22.84% / 0.827 | −2.46pp (0/30) | −0.061 (0/30) | +0.61pp (30/30) | −0.71pp / +0.021 / +0.19pp (30/30) | −1.83pp / +0.038 / −0.03pp (14/30) |

Base book: TN 37.5 / OA · Base Age 37.5 / IPO-A 25, monthly rebalanced, research/168's paths —
21.18% / −24.01% / Calmar 0.885. "vs cash" = the same book with cash in place of the candidate,
the cash weight solved to the same median drawdown (research/168's risk-matched null).

This converges with families already on file: research/71 (breakout exit bake-off — a trailing
stop beats a target), research/82 (10–15-day breakouts are real but cyclical and converge on the
breakout-paper book), research/142 (Blue-Sky ATH breakout), research/152 (multi-year breakout =
Open Alpha in disguise) and research/161 (OA · Base Age). A 25-bar breakout on seasoned stocks is
the high-turnover, shallow-base end of that family: it re-imports Open Alpha's breakout beta
without Base Age's base filter and without IPO Base's cohort. The pre-registered "worth a sleeve"
bar (+0.10 Calmar, or +2pp CAGR at no worse drawdown, on ≥ 20 of 30 paths; beats risk-matched cash
on ≥ 25 of 30; monthly correlation to OA · Base Age below 0.60) **is failed by every transplant, at
every weight, on 0 of 30 paths.**

**Q5. Does this change what we believed about IPO Base itself?**

IPO — yes, in two ways. Neither changes the blend value research/168 measured, because the real
curves it blended reproduce exactly.

**(a) research/167's "+4.8pp over a random null on 30 of 30 seeds" is about half a data artifact.**
research/167's panel computes rolling windows on a date index that is the union of every symbol's
dates. One missing row for a name — a partial-coverage special session, a phantom holiday row —
makes its 25-bar pivot and 50-bar trail NaN for weeks afterwards. Rebuilt with per-symbol windows
(the playbook's mandatory NaN-robust rule), the real book barely moves and the null rises:

| Panel | signals | W2 real / null | W2 edge (wins) | WA real / null | WA edge (wins) | WB real / null | WB edge (wins) |
|---|---|---|---|---|---|---|---|
| research/167’s panel (union-date windows) | 1,545 | 21.80 / 16.87 | +5.14pp (30) | 16.12 / 11.53 | +4.59pp (30) | 27.20 / 21.93 | +5.64pp (30) |
| + per-symbol NaN-robust windows | 1,707 | 22.24 / 20.03 | +2.25pp (30) | 16.83 / 12.21 | +4.62pp (30) | 28.26 / 28.16 | −0.18pp (13) |
| + phantom holiday rows dropped | 1,707 | 22.23 / 20.03 | +2.25pp (30) | 16.83 / 12.21 | +4.62pp (30) | 28.26 / 28.16 | −0.18pp (13) |
| + split back-adjustment (this study’s panel) | 1,707 | 22.23 / 20.03 | +2.25pp (30) | 16.83 / 12.21 | +4.62pp (30) | 28.26 / 28.16 | −0.18pp (13) |

research/167 ran its null on W2 only, so the WB collapse was never visible. The measured side
effect on the old panel: 32% of trades held through a stretch with no trail, and in those stretches
real breakouts earned +8.9% per trade against the null's +5.9% — the blackout turned breakouts into
bracket trades that suited them. **Read Spec A as a young-listing cohort harvest with a good exit
and gate, not as a stock-picking edge.**

**(b) research/167's capacity line quoted a median as a 90th percentile.** Its own
`stage9_adoption.csv`: at ₹10 L the **median** position is 1.56% of the name's 20-day traded value
and the **90th percentile is 9.05%** (9.64% on the clean panel). Its "~90% of a day's volume at
₹1 cr" was the p90 and is right. On a "p90 position ≤ 5% of traded value" yardstick Spec A is a
~₹5 L book; on a "median ≤ 5%" yardstick, ~₹32 L. The ₹20–25 L ceiling is optimistic on the tail.

**Q6. The Capital Desk: does anything here change whether IPO Spec A should go to 25%?**

IPO — it does not change **which** third sleeve: no transplant is a candidate, so IPO Base remains
the only third sleeve on file. It changes **what a 25% allocation buys today**, because
`services/ipo_paper.py` runs `MIN_BARS = 60` and Spec A was validated at 25.

The 6-Sep-2026 decision to run 60 rested on reading research/153's loader (`... where n >= 60`) as
"a stock is invisible until it has 60 bars". That count is taken over the WHOLE database today, not
at the signal date, so a name with 2,000 rows today is scanned from its 25th bar. Only listings in
the last ~60 sessions are invisible — exactly the window the 06-Sep reconciliation looked at, which
is why it saw 75% agreement. research/167's engine, unchanged, with its own null:

| Minimum bars since listing | signals | CAGR | worst seed | MaxDD | Calmar | WA / WB | null CAGR | edge (wins / 30) |
|---|---|---|---|---|---|---|---|---|
| 25 — the validated Spec A | 1,548 | 21.80% | 20.83 | −26.63% | 0.819 | 16.12 / 27.20 | 16.98% | +4.78pp (30) |
| 40 | 1,308 | 17.87% | 15.57 | −32.40% | 0.552 | 14.09 / 22.27 | 18.94% | −1.38pp (3) |
| 60 — what services/ipo_paper.py runs | 988 | 11.57% | 10.11 | −38.97% | 0.297 | 9.25 / 13.39 | 9.54% | +2.07pp (30) |

research/167 engine and panel, idle cash 5.0%, 30 seeds.

What each version is worth inside the book Arun is about to fund:

| Third sleeve at 25% (TN 37.5 / OA·BaseAge 37.5) | blend CAGR | MaxDD | Calmar | vs IPO-A at 25% | vs cash at equal drawdown | 40 bps / 60 bps CAGR |
|---|---|---|---|---|---|---|
| IPO-A as validated (min_bars 25) — research/168 | 21.18% | −24.01% | 0.885 | — | +2.52pp, 30/30 (research/168) | 19.47% / 17.92% |
| IPO live book as coded (min_bars 60) | 18.80% | −22.84% | 0.827 | −2.46pp CAGR (0/30 better); −0.061 Calmar (0/30) | +0.61pp (30/30) | 17.11% / 15.60% |
| Spec A on this study’s clean panel (min_bars 25), check | 21.13% | −23.53% | 0.914 | +0.05pp (19/30); +0.029 Calmar (30/30) | +2.86pp (30/30) | — |
| two-sleeve TN + OA·BaseAge 50:50 (no IPO) | 20.28% | −26.91% | 0.749 | — | — | — |

Standalone, the 60-bar book is 12.39% / −36.35% / Calmar 0.341, worst seed 11.30%, a 23-trade
losing streak, and it does not clear its own null (+0.79pp; WB 19 of 30). Inside the blend it has
the fingerprint that disqualified the old incumbent in research/168: it buys a little drawdown and
adds barely more than an arbitrage fund at equal risk.

**The line for the decision: 25% to IPO is supported only for the spec that was validated.** Until
the live book runs `MIN_BARS = 25`, the Capital Desk would be funding a ~12% sleeve that lowers the
blend's CAGR on every one of 30 paths. Changing `MIN_BARS` is a strategy change — its own STATUS
doc, the capacity question in Q5(b) answered for entries 25–60 sessions after listing (the
thinnest names the book will ever buy), and an after-15:40 deploy. **This study changed nothing
live.** A dated review is registered for 19-Sep-2026, ahead of the 26-Sep funding call.

**Q7. How honest are the universes?**

IPO — honest about survivorship, loose about exact membership. No point-in-time index constituent
history exists anywhere in the repo, only today's official lists, and using those back to 2006
hands the backtest the names that went on to become large. So on the first trading day of each
month every non-fund symbol with ≥ 60 bars was ranked by its trailing 126-bar median traded value
known at the prior close (the research/41 method) and held in that band for the month:

| Proxy band (traded-value rank) | overlap with the PIT market-cap band (median month, Aug-2018+) | overlap with today’s official index | official index |
|---|---|---|---|
| top 50 | 62% | 58% | Nifty 50 |
| top 100 | 69% | 66% | Nifty 50 + Next 50 |
| top 200 | 76% | 80% | Nifty 200 |
| top 500 | 80% | 79% | Nifty 500 (proxy list) |
| ranks 101-250 | 44% | 46% | Nifty Midcap 150 |
| ranks 251-500 | 44% | 42% | Nifty Smallcap 250 |

Spearman rank correlation, traded value vs PIT market cap, within the top 500 by market cap: 0.70 (median of 98 months).

The top-200 / top-500 proxies are good; the Nifty-50-like set is fair; the Midcap- and
Smallcap-like bands are rough. That is why the `mcap_pit` re-run in Q2 exists, and it agrees.

---

## Reproduction gate and data defenses

- **research/167's engine reproduces Spec A exactly**: 21.80% / −26.63% (delta 0.000pp); all 30
  seed paths identical to research/168's `A_25bps_y52` (max abs diff 0.0); null 16.98%, edge
  +4.78pp, 30/30.
- **The new full-universe panel, set to research/167's conventions, reproduces 21.80% / −26.63%**
  with the same 1,545 signals, before any defense is switched on.
- Defenses switched on one at a time (Spec A, 5.0% cash): per-symbol NaN-robust windows → 22.24%
  (+162 signals); phantom holiday rows dropped (24-Apr-2014: 121 symbols, 15-Oct-2014: 129, all
  zero volume) → no change; split back-adjustment at one-day close ratios ≤ 0.60 or ≥ 1.80 → no
  change for IPO. Full universe: 273 adjustment events, 38 on names at ≥ ₹5 cr (e.g. ADANIENT 2015,
  ARVIND 2018, INFIBEAM 2018).
- Two process incidents, both recovered: a stage was killed out-of-memory when two panels were in
  memory at once (re-run alone, no partial results kept), and a progress-print bug aborted the
  first market-cap run before any cell was written.

## Per-year house table

Each cell is the annual return with the intra-year max drawdown beneath it, measured from the
running peak of the FULL curve. Columns are the median-CAGR path of each ensemble; the summary row
carries the 30-path medians. Best-of columns exclude NIFTYBEES. Common window 2006-04-03 →
2026-09-03.

| Year | TN | OA · Base Age | IPO-A (validated) | IPO live (60 bars) | Midcap-like transplant | TN / OA / IPO-A 37.5/37.5/25 | TN / OA / IPO live 37.5/37.5/25 | TN / OA / Midcap transplant 37.5/37.5/25 | NIFTYBEES | BEST CAGR | LEAST DD | BEST OVERALL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2006 | +8.4<br><sub>(−11.9)</sub> | +8.3<br><sub>(−20.6)</sub> | +69.1<br><sub>(−10.1)</sub> | +37.1<br><sub>(−2.6)</sub> | +10.4<br><sub>(−15.7)</sub> | +22.2<br><sub>(−12.0)</sub> | +13.6<br><sub>(−10.5)</sub> | +9.6<br><sub>(−11.2)</sub> | +16.1<br><sub>(−29.9)</sub> | IPO-A (validated) | IPO live (60 bars) | IPO-A (validated) |
| 2007 | +81.6<br><sub>(−15.6)</sub> | +84.7<br><sub>(−10.8)</sub> | +51.6<br><sub>(−15.9)</sub> | +27.3<br><sub>(−12.5)</sub> | +36.3<br><sub>(−21.2)</sub> | +65.4<br><sub>(−12.4)</sub> | +60.7<br><sub>(−10.7)</sub> | +62.8<br><sub>(−11.6)</sub> | +53.0<br><sub>(−14.9)</sub> | OA · Base Age | TN / OA / IPO live 37.5/37.5/25 | OA · Base Age |
| 2008 | −15.6<br><sub>(−21.5)</sub> | −29.9<br><sub>(−32.6)</sub> | −10.1<br><sub>(−15.8)</sub> | −7.0<br><sub>(−12.8)</sub> | −17.8<br><sub>(−21.6)</sub> | −18.4<br><sub>(−21.4)</sub> | −18.4<br><sub>(−21.1)</sub> | −21.3<br><sub>(−24.0)</sub> | −52.1<br><sub>(−59.7)</sub> | IPO live (60 bars) | IPO live (60 bars) | IPO live (60 bars) |
| 2009 | +52.3<br><sub>(−22.4)</sub> | +62.8<br><sub>(−31.7)</sub> | +4.0<br><sub>(−12.5)</sub> | +4.2<br><sub>(−9.4)</sub> | +72.7<br><sub>(−18.5)</sub> | +54.6<br><sub>(−20.8)</sub> | +47.7<br><sub>(−20.8)</sub> | +71.3<br><sub>(−23.1)</sub> | +75.6<br><sub>(−59.1)</sub> | Midcap-like transplant | IPO live (60 bars) | Midcap-like transplant |
| 2010 | +5.5<br><sub>(−20.4)</sub> | +15.5<br><sub>(−15.7)</sub> | +21.1<br><sub>(−9.8)</sub> | +17.8<br><sub>(−10.2)</sub> | +7.9<br><sub>(−17.8)</sub> | +15.3<br><sub>(−10.3)</sub> | +15.5<br><sub>(−10.8)</sub> | +10.0<br><sub>(−14.1)</sub> | +18.6<br><sub>(−25.0)</sub> | IPO-A (validated) | IPO-A (validated) | IPO-A (validated) |
| 2011 | −12.4<br><sub>(−18.6)</sub> | −10.5<br><sub>(−18.6)</sub> | −2.4<br><sub>(−10.2)</sub> | −2.4<br><sub>(−8.2)</sub> | −23.1<br><sub>(−35.9)</sub> | −8.6<br><sub>(−14.7)</sub> | −10.3<br><sub>(−15.0)</sub> | −9.4<br><sub>(−21.1)</sub> | −24.1<br><sub>(−27.3)</sub> | IPO live (60 bars) | IPO live (60 bars) | IPO live (60 bars) |
| 2012 | +22.2<br><sub>(−16.3)</sub> | +28.2<br><sub>(−19.5)</sub> | +11.9<br><sub>(−9.8)</sub> | +11.9<br><sub>(−7.8)</sub> | +13.9<br><sub>(−35.5)</sub> | +25.0<br><sub>(−14.1)</sub> | +22.2<br><sub>(−14.4)</sub> | +22.8<br><sub>(−20.9)</sub> | +26.5<br><sub>(−26.0)</sub> | OA · Base Age | IPO live (60 bars) | TN / OA / IPO-A 37.5/37.5/25 |
| 2013 | +0.6<br><sub>(−12.1)</sub> | +4.5<br><sub>(−9.3)</sub> | +4.8<br><sub>(−0.4)</sub> | +6.1<br><sub>(−4.2)</sub> | −13.0<br><sub>(−44.0)</sub> | +1.4<br><sub>(−7.5)</sub> | +5.3<br><sub>(−5.0)</sub> | +5.3<br><sub>(−10.2)</sub> | +7.2<br><sub>(−16.0)</sub> | IPO live (60 bars) | IPO-A (validated) | IPO-A (validated) |
| 2014 | +38.0<br><sub>(−12.0)</sub> | +49.7<br><sub>(−7.3)</sub> | +5.2<br><sub>(0.0)</sub> | +0.4<br><sub>(−4.5)</sub> | +48.9<br><sub>(−42.4)</sub> | +39.8<br><sub>(−7.2)</sub> | +35.4<br><sub>(−6.5)</sub> | +51.0<br><sub>(−9.8)</sub> | +31.6<br><sub>(−6.2)</sub> | TN / OA / Midcap transplant 37.5/37.5/25 | IPO-A (validated) | OA · Base Age |
| 2015 | −3.3<br><sub>(−10.9)</sub> | −3.6<br><sub>(−23.5)</sub> | +27.7<br><sub>(−9.6)</sub> | +2.4<br><sub>(−3.9)</sub> | −5.3<br><sub>(−23.8)</sub> | +4.5<br><sub>(−9.9)</sub> | −2.4<br><sub>(−10.2)</sub> | −5.8<br><sub>(−12.7)</sub> | −4.3<br><sub>(−15.0)</sub> | IPO-A (validated) | IPO live (60 bars) | IPO-A (validated) |
| 2016 | +31.1<br><sub>(−11.8)</sub> | +7.3<br><sub>(−29.5)</sub> | +75.8<br><sub>(−10.6)</sub> | +20.2<br><sub>(−10.7)</sub> | −10.4<br><sub>(−30.6)</sub> | +27.8<br><sub>(−10.3)</sub> | +21.1<br><sub>(−12.1)</sub> | +10.2<br><sub>(−13.8)</sub> | +4.0<br><sub>(−21.6)</sub> | IPO-A (validated) | TN / OA / IPO-A 37.5/37.5/25 | IPO-A (validated) |
| 2017 | +31.7<br><sub>(−10.1)</sub> | +65.2<br><sub>(−13.2)</sub> | +72.3<br><sub>(−10.2)</sub> | +65.0<br><sub>(−13.2)</sub> | +56.9<br><sub>(−29.4)</sub> | +46.8<br><sub>(−6.7)</sub> | +50.0<br><sub>(−6.8)</sub> | +49.9<br><sub>(−8.1)</sub> | +29.9<br><sub>(−8.5)</sub> | IPO-A (validated) | TN / OA / IPO-A 37.5/37.5/25 | IPO-A (validated) |
| 2018 | −8.3<br><sub>(−23.1)</sub> | −30.5<br><sub>(−35.6)</sub> | −9.1<br><sub>(−19.2)</sub> | −1.4<br><sub>(−13.5)</sub> | −23.5<br><sub>(−30.8)</sub> | −12.9<br><sub>(−21.7)</sub> | −14.2<br><sub>(−23.3)</sub> | −18.0<br><sub>(−26.8)</sub> | +4.8<br><sub>(−14.1)</sub> | IPO live (60 bars) | IPO live (60 bars) | IPO live (60 bars) |
| 2019 | −2.7<br><sub>(−26.0)</sub> | +30.1<br><sub>(−35.9)</sub> | +5.4<br><sub>(−15.6)</sub> | +1.2<br><sub>(−8.9)</sub> | −13.1<br><sub>(−38.2)</sub> | +14.0<br><sub>(−22.2)</sub> | +12.4<br><sub>(−23.6)</sub> | +6.0<br><sub>(−30.2)</sub> | +13.6<br><sub>(−10.5)</sub> | OA · Base Age | IPO live (60 bars) | OA · Base Age |
| 2020 | +66.6<br><sub>(−25.9)</sub> | +48.8<br><sub>(−23.2)</sub> | +74.8<br><sub>(−13.2)</sub> | +50.5<br><sub>(−8.3)</sub> | +0.1<br><sub>(−44.0)</sub> | +62.0<br><sub>(−10.6)</sub> | +51.8<br><sub>(−13.4)</sub> | +37.7<br><sub>(−25.8)</sub> | +15.4<br><sub>(−36.3)</sub> | IPO-A (validated) | IPO live (60 bars) | IPO-A (validated) |
| 2021 | +61.8<br><sub>(−11.2)</sub> | +84.3<br><sub>(−10.9)</sub> | +52.5<br><sub>(−14.5)</sub> | +41.5<br><sub>(−21.1)</sub> | +55.3<br><sub>(−35.8)</sub> | +66.2<br><sub>(−8.2)</sub> | +69.7<br><sub>(−8.1)</sub> | +61.3<br><sub>(−7.8)</sub> | +26.0<br><sub>(−9.5)</sub> | OA · Base Age | TN / OA / Midcap transplant 37.5/37.5/25 | OA · Base Age |
| 2022 | +16.1<br><sub>(−14.1)</sub> | −0.8<br><sub>(−24.3)</sub> | +0.9<br><sub>(−21.2)</sub> | +1.7<br><sub>(−29.1)</sub> | −18.5<br><sub>(−26.9)</sub> | −0.3<br><sub>(−18.5)</sub> | +5.6<br><sub>(−18.4)</sub> | +0.3<br><sub>(−15.5)</sub> | +5.5<br><sub>(−16.1)</sub> | TN | TN | TN |
| 2023 | +52.3<br><sub>(−11.0)</sub> | +59.8<br><sub>(−15.4)</sub> | +46.5<br><sub>(−11.0)</sub> | +11.7<br><sub>(−22.2)</sub> | +47.1<br><sub>(−31.8)</sub> | +50.6<br><sub>(−10.0)</sub> | +35.4<br><sub>(−9.3)</sub> | +38.1<br><sub>(−10.2)</sub> | +21.0<br><sub>(−9.7)</sub> | OA · Base Age | TN / OA / IPO live 37.5/37.5/25 | OA · Base Age |
| 2024 | +24.2<br><sub>(−18.2)</sub> | −1.3<br><sub>(−26.1)</sub> | +13.6<br><sub>(−22.6)</sub> | +3.0<br><sub>(−20.7)</sub> | +16.3<br><sub>(−19.8)</sub> | +14.9<br><sub>(−16.3)</sub> | +18.9<br><sub>(−15.6)</sub> | +11.2<br><sub>(−14.6)</sub> | +10.4<br><sub>(−10.5)</sub> | TN | TN / OA / Midcap transplant 37.5/37.5/25 | TN |
| 2025 | +4.7<br><sub>(−17.3)</sub> | +2.0<br><sub>(−21.9)</sub> | +1.9<br><sub>(−17.9)</sub> | −13.5<br><sub>(−26.7)</sub> | +10.6<br><sub>(−21.9)</sub> | +4.3<br><sub>(−9.8)</sub> | −2.3<br><sub>(−13.6)</sub> | +7.6<br><sub>(−11.8)</sub> | +11.7<br><sub>(−15.2)</sub> | Midcap-like transplant | TN / OA / IPO-A 37.5/37.5/25 | TN / OA / Midcap transplant 37.5/37.5/25 |
| 2026 | +7.7<br><sub>(−8.7)</sub> | +27.6<br><sub>(−23.6)</sub> | −0.2<br><sub>(−24.0)</sub> | +7.3<br><sub>(−36.5)</sub> | −4.1<br><sub>(−12.4)</sub> | +13.1<br><sub>(−12.1)</sub> | +16.3<br><sub>(−19.5)</sub> | +13.5<br><sub>(−10.6)</sub> | −7.7<br><sub>(−14.8)</sub> | OA · Base Age | TN | OA · Base Age |
| **full** | **19.63**<br><sub>−26.0 / 0.76</sub> | **19.92**<br><sub>−34.0 / 0.58</sub> | **22.08**<br><sub>−26.6 / 0.83</sub> | **12.38**<br><sub>−36.3 / 0.34</sub> | **8.51**<br><sub>−50.4 / 0.17</sub> | **21.18**<br><sub>−24.0 / 0.89</sub> | **18.80**<br><sub>−22.8 / 0.83</sub> | **17.37**<br><sub>−28.5 / 0.62</sub> | **10.58**<br><sub>−59.7 / 0.18</sub> |  |  |  |

## Caveats — what would make this wrong

- **Proxy universes** (Q7): the Midcap- and Smallcap-like bands overlap the real bands only ~45%.
  The 2018+ market-cap re-run agrees, but it is eight years, not twenty.
- **Survivorship**: names never onboarded to Kite cannot be measured. Delisted names that are in the
  database are traded.
- **The ₹5 cr liquidity floor is nominal**, not inflation-adjusted, so it admits fewer names in 2006
  than in 2026 on every universe.
- **Split back-adjustment treats demerger drops as price adjustments** (the total-return
  convention). 38 events on ≥ ₹5 cr names in twenty years.
- **The re-fit of a survivor was pre-registered and therefore not run** — nothing survived. A
  seasoned-stock breakout with a different exit could behave differently; research/71 and
  research/161 already cover that ground, and OA · Base Age is that book.
- **Multiple testing**: ~100 backtest cells, each on 30 seeds. Nothing here is a discovery, so the
  concern cuts only against the corrections in Q5 and Q6 — and those rest on reproduced, paired,
  direct comparisons, not on a sweep peak.
- **Mixed panels in Q6**: TN, OA · Base Age and IPO-A are research/168's curves (research/167's
  panel); the 60-bar live book and the transplants are this study's clean panel. The clean-panel
  Spec A ties IPO-A inside the blend (+0.05pp CAGR, monthly correlation 0.925), so the two are
  commensurable for this purpose.
- **Not done**: an equity-curve chart pack (the family died at the null gate, before a tearsheet is
  owed); VIX gates; alternative entry mechanics for seasoned universes (the next-day buy-stop was
  held identical by design).

## Cells disclosed

S0 reproduction 3 checks · S1 panel equivalence 5 · S1b min_bars on research/167's engine 3 × (real
+ null) · S1c null attribution 4 × 2 · S1d mechanism 2 × 2 · S2 transplant 18 × 2 · S3 age axis 4 × 2
· S2b market-cap universes 13 × 2 · S4 re-fit 0 (no survivor) · S5 cost ladder 10 · S6 blends 5
candidates × (replace + 4 weights) each with a risk-matched cash null. About 100 backtest cells, 30
seeds each, most across three windows.

## Files

| File | What |
|---|---|
| `scripts/xpanel.py` | full-universe panel: causal traded-value ranks, split adjustment, NaN-robust windows, signal / null / runner |
| `scripts/run169.py` | stages repro, equiv, transplant, age, refit, costs, blend, proxy |
| `scripts/minbars_check.py`, `null_attrib.py`, `mechanism.py`, `mcap_check.py` | S1b, S1c, S1d, S2b |
| `scripts/close_loop.py` | builds this file's tables and the app entry from the result files |
| `results/s0_repro.json`, `s1_equiv.csv`, `s1b_minbars_r167engine.json`, `s1c_null_attribution.csv`, `s1d_mechanism.json` | reproduction, defenses, min_bars, null attribution, mechanism |
| `results/s2_transplant.csv`, `s3_age.csv`, `s2b_mcap_universes_2018.csv`, `s5_costs.csv` | cell tables |
| `results/s6_blend.json`, `s6_peryear.json`, `s6_peryear_table.md`, `carried.json` | portfolio fit |
| `results/s7_proxy.json`, `s1_split_events.csv` | proxy validation, adjustment events |
| `results/navs/*.npz` | 30-seed W2 NAV paths (gitignored) |

Reproduce: `venv/bin/python research/169_ipo_rules_universe_transplant/scripts/run169.py repro`,
then `equiv`, `grid`, `minbars_check.py`, `null_attrib.py`, `run169.py refit`, `costs`,
`mcap_check.py`, `run169.py blend`, `mechanism.py`, `close_loop.py`. One panel process at a time
(~3.5 GB each); about 40 minutes in total on the VPS.
