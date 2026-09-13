# research/173 — IPO Base Spec A: exiting at the NEXT OPEN instead of the SIGNAL CLOSE

## Verdict: **IMMATERIAL** — and the deviation runs in the book's favour, not against it

IPO (IPO Base, research/167 Spec A) — fidelity check of the live exit mechanic
The live book (commit 01bc4bce) sells at the next morning's open for an exit decided on
tonight's close; the backtest sold at the close. This measures what that one-open delay does.

Basis for every number: 2006-01-01 → 2026-09-04 unless stated, after tax (20% STCG /
12.5% LTCG, FY loss netting), 25 bps a side, ₹10L start, 8 slots @ 18.75%, 30 random-
selection seeds (1–30), idle cash 5.0% (research/167's basis) with 5.2% (house standard)
alongside. Medians across the 30 seeds unless labelled worst seed. Calmar = median CAGR ÷
|median max drawdown|; paired deltas use each seed's own Calmar.

---

**Q1. Did we reproduce the published Spec A before changing anything?**

Yes, exactly. The forked simulator in arm A is bit-identical to research/153's `simulate_ipo`
on seeds 1–3 (same trade count, 396/396 and 398/398; same final NAV to the rupee). At 5.0%
cash it returns **21.80% CAGR (worst seed 20.83%) / −26.63% median DD (worst seed −32.88%) /
Calmar 0.819**, the research/167 figures to the digit. The same holds on the committed and
on the refreshed (13-Sep 10:15) `listing_dates.csv`, so the refreshed file was used.

**Q2. What does each exit fill return?**

The arms, all on the same exit decisions and the same 30 seeds:
- **A**: sell at the signal close (the study as published).
- **B**: sell at the next open, no floor.
- **C**: sell at the next open only if the open is at least 98% of the latest close; if not, retry at the next open with a floor 2% under the most recent close.
- **C2** (sensitivity only): as C, but a day LIMIT order also fills at the floor if the day's high reaches it.

Idle cash 5.0%:

| window | arm | CAGR med | CAGR worst | DD med | DD worst | Calmar | mean/trade |
|---|---|---|---|---|---|---|---|
| 2006–2026 | A close | 21.80 | 20.83 | −26.63 | −32.88 | 0.819 | +6.74% |
| 2006–2026 | B next open | **22.97** | 20.87 | −26.10 | −34.19 | 0.880 | +7.08% |
| 2006–2026 | C next open, 2% floor | **23.42** | 21.24 | −26.16 | −32.74 | 0.895 | +7.20% |
| 2006–2026 | C2 floor + intraday fill | 23.16 | 21.17 | −26.15 | −33.68 | 0.886 | +7.15% |
| 2006–2015 | A | 16.12 | 16.12 | −15.94 | −15.94 | 1.011 | +9.04% |
| 2006–2015 | B | 17.01 | 17.01 | −17.30 | −17.30 | 0.983 | +9.65% |
| 2006–2015 | C | 17.56 | 17.56 | −16.21 | −16.21 | 1.083 | +10.00% |
| 2016–2026 | A | 27.20 | 24.14 | −26.56 | −36.58 | 1.024 | +6.04% |
| 2016–2026 | B | 28.51 | 25.89 | −25.11 | −36.85 | 1.135 | +6.25% |
| 2016–2026 | C | 28.97 | 26.17 | −25.82 | −34.53 | 1.122 | +6.31% |

At 5.2% cash, full period: A 21.96 / −26.56 / 0.827; B 23.12 / −26.02 / 0.889;
C 23.58 / −26.07 / 0.905 (every delta within 0.01pp of the 5.0% rows; `results/summary.csv`).

The 2006–2015 rows are identical on every seed: the book had fewer triggers than slots then,
so the random selection never binds. Its "30 of 30" counts below are one path, not thirty.

**Q3. What are the paired differences, and who wins on how many seeds?**

A positive number means the next-open arm is better. "A wins" counts seeds where the close exit beat it.

| cash | window | arm | ΔCAGR med [min..max] | A wins CAGR | ΔCalmar med | A wins Calmar | ΔDD med | A wins DD |
|---|---|---|---|---|---|---|---|---|
| 5.0% | 2006–2026 | B−A | **+1.18** [+0.04..+1.93] | **0/30** | +0.048 | 3/30 | +0.22 | 13/30 |
| 5.0% | 2006–2026 | C−A | **+1.59** [+0.41..+2.38] | **0/30** | +0.100 | 1/30 | +1.06 | 3/30 |
| 5.0% | 2006–2015 | B−A | +0.89 | 0/30 | −0.028 | 30/30* | −1.36 | 30/30* |
| 5.0% | 2006–2015 | C−A | +1.44 | 0/30 | +0.072 | 0/30 | −0.27 | 30/30* |
| 5.0% | 2016–2026 | B−A | +1.32 [+0.13..+3.11] | 0/30 | +0.066 | 4/30 | +0.57 | 10/30 |
| 5.0% | 2016–2026 | C−A | +1.70 [+0.40..+2.70] | 0/30 | +0.106 | 3/30 | +0.96 | 5/30 |
| 5.2% | 2006–2026 | B−A | +1.17 | 0/30 | +0.048 | 3/30 | +0.22 | 13/30 |
| 5.2% | 2006–2026 | C−A | +1.59 | 0/30 | +0.101 | 1/30 | +1.02 | 3/30 |

\* one deterministic path repeated (see Q2).

**Q4. What does the overnight gap look like on the exit signals?**

The gap is close[i] → open of the name's next trading bar, on arm A's exit decisions.
Unique events are 419 distinct (name, day, reason) exits across all seeds. Seed-weighted
counts each exit as often as a book actually took it (11,631).

| basis | reason | n | mean | median | p10 | p90 | share < −2% | share < −5% | worst | best |
|---|---|---|---|---|---|---|---|---|---|---|
| unique | all | 419 | +0.46% | +0.42% | −1.07% | +2.00% | 4.3% | 0.2% | −6.85% | +7.17% |
| unique | stop | 98 | +0.35% | +0.27% | −1.30% | +2.00% | 6.1% | 0.0% | −4.11% | +7.17% |
| unique | target | 130 | +0.56% | +0.43% | −1.07% | +2.37% | 4.6% | 0.8% | −6.85% | +6.88% |
| unique | trail | 191 | +0.46% | +0.49% | −0.79% | +1.74% | 3.1% | 0.0% | −3.58% | +4.02% |
| seed-weighted | stop | 1,886 | +0.23% | +0.15% | −2.03% | +2.02% | 10.5% | 0.0% | | |
| seed-weighted | target | 4,000 | +0.34% | +0.36% | −1.15% | +2.23% | 6.6% | 2.3% | | |
| seed-weighted | trail | 5,745 | +0.42% | +0.43% | −1.07% | +1.90% | 2.6% | 0.0% | | |

By half (unique events), the mean gap is positive for every reason in both: 2006–2015 stop
+0.65 / target +0.58 / trail +1.25; 2016–2026 stop +0.31 / target +0.55 / trail +0.33
(`results/gaps_by_reason_half.csv`). None of the exits gaps more than −7%, so no
unadjusted-split artefact is driving the result.

What the gap is worth in rupees (the gap × exit value, summed over 20 years on the
compounding ₹10L book, median seed): target +₹29.3L, trail +₹26.6L, stop +₹7.1L. Stops
are the only reason with a losing seed (min −₹0.4L).

**Q5. How often does the 2% LIMIT floor bite, and how many extra days are held?**

Arm C, all 30 seeds pooled:
- **613 of 11,648 exits (5.3%) hit the floor**, about 20 per seed over 20 years (roughly 1 a year).
- By reason: stop 207 exits (10.6%), target 267 (6.7%), trail 139 (2.4%).
- **Extra days held on a floor hit: mean 1.1, maximum 2.**
- No exit was left unfilled at the end of a window.

On those floor hits the eventual sale averaged +1.13% versus the signal close: stop +3.67%,
trail +1.42%, target −0.99%. The day after a gap-down below the floor tended to recover,
which is why C edges out B. C2 (a day LIMIT that fills at the floor intraday) almost never
misses: 10 exits across 30 seeds, and those sold on average −10.8% under the signal close.
That is the tail a floored order carries on a name that keeps falling, and it is rare.

**Q6. Against the pre-registered bar, is the deviation material?**

The bar was written before running: **material if B or C costs more than 1.0pp of CAGR or
more than 0.10 Calmar at the median paired difference, on at least 20 of 30 seeds**,
full period.

- **B does not cost; it adds +1.18pp** of CAGR, and A wins on 0/30 seeds. Calmar is +0.048, and A wins 3/30.
- **C does not cost; it adds +1.59pp**, and A wins on 0/30 seeds. Calmar is +0.100, and A wins 1/30.
- Every window and both cash rates point the same way.

**IMMATERIAL.** The live book's next-open exit is no worse than the study, and on this
history it is about a point of CAGR better.

What it gives up, stated honestly:
- **Worst-seed drawdown is 1.3pp deeper in B** (−34.19% vs −32.88%). C does not show this (−32.74%).
- **In 2006–2015 B's drawdown is 1.36pp deeper** (−17.30% vs −15.94%), with Calmar −0.028. That is one deterministic path, and it is well inside the bar.
- The overnight exposure is real: about 10% of stop exits (seed-weighted) open more than 2% lower.

**Q7. Which exit reason drives it, and is a same-day ~15:05 close-proxy exit worth building?**

The effect is favourable, and no reason drives a cost. The rupee gain splits target 46% /
trail 42% / stop 11%. Targets and trails dominate because they are most of the exits (34%
and 49% of them), and their mean gaps are +0.34% and +0.42%.

**A 15:05 close-proxy exit is NOT worth building.** It would move the exit toward A, the arm
that loses on 30 of 30 seeds. This is a mean-reversion pattern: a name that has just closed
under its stop or SMA-50 tends to open slightly higher; a name that has just closed through
+25% tends to carry on. Selling the same afternoon forfeits both.

---

## Caveats (leading, not hiding)

- **Don't turn this into a strategy.** The +1.2 to +1.6pp is a by-product of a fidelity check. Deliberately delaying exits further (two opens, VWAP, and so on) would be a new, untested rule with its own multiple-testing bill. The only claim here is that the live mechanic does not degrade the study.
- **The daily open is the official open** (pre-open auction price). A MARKET AMO fills at or very near it, and 25 bps a side already covers slippage. A LIMIT AMO refused at the floor is modelled as a full miss for the day (C). The real day order's intraday fill is C2. Both beat A.
- **Survivorship and panel.** These are research/167's vetted listings with name-based fund exclusion. The window ends 04-Sep-2026 as published.
- **The 2006–2015 half is a single path** (selection never binds), so its seed counts are not 30 independent observations.
- **Not tested**: 40/60 bps cost ladders (the delta is a fill-price effect and should not interact with cost), and exits delayed beyond the first open.

## Files

- `scripts/exit_timing.py`: fork of `simulate_ipo` with exit-fill arms (`check` / `run` / `report`)
- `results/seed_stats.csv`: 720 rows, one per (cash, window, arm, seed)
- `results/summary.csv`, `results/paired.csv`: medians and paired deltas
- `results/gaps_by_reason.csv`, `results/gaps_by_reason_half.csv`, `results/gap_rupees_by_reason_seed.csv`
- `results/floor_stats.csv`: floor hits, extra days, realised sale vs signal close
- `results/trades_w2_y050.csv`: every trade, full period, 5.0% cash, all arms and seeds
