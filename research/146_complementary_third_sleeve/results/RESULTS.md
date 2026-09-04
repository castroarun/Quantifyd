# Research 146 — Complementary Third Sleeve for TN+OA — RESULTS

**VERDICT: NO EDGE (as a complement) — NOTHING clears the pre-registered blend bar. The TN+OA
50-50 pair stands alone.** Four candidate families were tested (KC6 mean reversion, Arun's
pullback-reversal sketch, Connors RSI2/3 oversold-in-uptrend, N-day-low washout). Three died
at G1 with negative net expectancy — the pullback sketch catastrophically so. KC6, the one
survivor (and the one positive-expectancy cell with its crash filter on), then FAILED the
blend test: every KC6 variant at every weight LOWERS the 3-sleeve Calmar (1.65 → 1.21-1.53)
and is strictly dominated by a plain-cash third sleeve. The deep reason is structural, and it
is the study's real finding: **the TN gate + OA stops have already stripped the crash tail
out of the pair** (baseline drawdown inside the 2008 window: −2.4%; inside the 2020 crash:
−1.5%) — while mean reversion, which buys weakness, takes its losses EXACTLY in those windows
(KC6 2008: −17..−32%; 2020: −3..−17%). A weakness-buying sleeve adds back the tail the gate
removed. Low average correlation (0.06-0.15) hid this; the per-crash-window check caught it.

> Engine: `scripts/sleeve_engine.py` (OHLC panel 2005→2026-09-03, PIT top-500 traded-value
> universe, 10 slots × 10%, hard stops, no averaging, 25bps/side, FY-netted 20/12.5% tax,
> cash 5%); blends: `scripts/blend3.py` (TN deployed spec offsets 0/4/8 + OA adopted spec
> 10 seeds, all after-tax, monthly rebalanced). Baseline = TN+OA 50-50: **27.2% / −16.4% /
> Calmar 1.65** (reproduced in-run: 27.16/−16.38/1.65).

## G1 — candidate families (after-tax, 50bps RT, 2012→now; tradeability gate columns shown)

| Variant (best of family) | Trades | WR | Avg win/loss | **Expectancy/trade** | Max lose streak | CAGR | MaxDD | corr TN |
|---|---|---|---|---|---|---|---|---|
| **KC6 + crash filter** (kc_crash) | 1,229 | 60.5% | +4.3 / −6.3% | **+0.104%** | 20 | 4.6% | −16.0% | 0.11 |
| KC6 base (SL5/TP15/15d/mid-exit) | 1,336 | 59.1% | +4.5 / −6.6% | −0.016% | 28 | 4.5% | −21.0% | 0.12 |
| KC6 1.5×ATR (kc_m15) | 413 | 55.2% | +5.5 / −7.2% | −0.218% | 12 | 5.5% | −7.2% (Sharpe 1.27) | 0.06 |
| Pullback-reversal, best of 6 (Arun's sketch) | 4,101-7,818 | 30-39% | +5..7 / −3..5% | **−0.69..−0.91%** | 33-57 | −11..−19% | −85..−97% | 0.30 |
| Connors RSI2<10, best of 6 | 6,374-8,416 | ~60% | +2.8 / −4.6% | −0.13..−0.16% | 16-18 | −1.5..−6% | −46..−71% | 0.28 |
| 7/10-day-low washout, best of 4 | 6,741-7,788 | 56-60% | +3.2 / −4.8% | −0.25..−0.30% | 17-20 | −7..−11% | −74..−86% | 0.33 |

- **F2 pullback-reversal: KILLED, plainly.** The buy-stop above the green candle with the
  tight candle-low stop produces a 30-39% win rate whose 1.5-3R winners never pay for the
  churn: −0.7 to −0.9% per trade net across ALL six variants (SMA/EMA touch, 1.5/2/3R,
  time/SMA20 exits) — a losing family, not a mis-tuned one.
- **F3/F4: KILLED — the r/84 win-rate illusion on schedule.** ~60% WR, negative expectancy
  once the avg loss (−4.6 to −4.8%) and 50bps costs are counted.
- **F1 KC6: the honest survivor.** The crash filter (universe ATR ratio ≥1.3 blocks entries)
  is what turns expectancy positive (+0.104%/trade) — matching the parked live system's
  design. But net, slot-constrained, after-tax on a survivorship-honest PIT universe, the
  sleeve compounds at ~cash rate (4.6%): tradeable, not investable on its own.

## The blend test — the pre-registered bar (all after-tax, 10-OA-seed medians, TN offset 0)

Rule: best w3 ∈ {10..33}% must give **Calmar ≥ 1.75 with CAGR ≥ 25.2%** OR **DD ≤ −14.4%
(−2pp) with CAGR ≥ 27.2%**; corr < 0.4 vs both legs; robust across seeds and offsets; crash
windows not worsened by >2pp.

| Third sleeve | w3 | CAGR med | DD med | Calmar med | Verdict |
|---|---|---|---|---|---|
| (baseline, none) | 0% | 27.16 | −16.38 | 1.65 | — |
| KC6 crash-filtered | 10% | 24.84 | −16.17 | 1.53 | worse Calmar at −2.3pp CAGR — FAIL |
| KC6 crash-filtered | 20% | 22.52 | −16.09 | 1.39 | FAIL |
| KC6 base | 20% | 22.30 | −19.42 | 1.14 | DD WORSENS — FAIL |
| KC6 1.5×ATR | 20% | 22.44 | −17.85 | 1.25 | FAIL |
| **Cash null (5% p.a.)** | 10% | 24.91 | −14.48 | 1.71 | fails CAGR floor (25.2) by 0.3pp — the closest thing to a pass is PLAIN CASH |
| Cash null | 25% | 21.55 | −11.72 | 1.83 | Calmar up only by de-levering — not adoptable under the rule |

**Every KC6 blend is strictly dominated by the cash null at the same weight** (same or less
CAGR, more DD). Offsets 4/8 and worst-seed columns change nothing (blend3.csv). Correlation
condition passed (0.06-0.15 vs both legs) — and is exactly why the rule ALSO required the
crash-window check:

## Crash windows (the finding that matters)

| Window | Baseline blend DD | +25% KC6-crash blend DD | KC6 sleeve return in window |
|---|---|---|---|
| 2008 | **−2.4%** | −5.3% | −17.2% (base variant −31.9%) |
| 2015-16 | −6.8% | −4.0% | +6.7% |
| 2018 | −11.0% | −8.0% | +2.5% |
| 2020 crash | **−1.5%** | −3.2% | −11.9% (base −17.0%) |
| 2022 H1 | −8.8% | −8.3% | −8.1% |

Two structural facts fall out:

1. **The pair has almost no crash pain left to diversify.** TN liquidates on its weekly gate
   and OA stops out name-by-name — the baseline's 2008-window and 2020-crash drawdowns are
   −2.4% and −1.5%. A "crash diversifier" has nothing to earn there; it can only add risk
   back, and mean reversion does (−3.9 to −5.3% in 2008 at w3=25%, breaching the 2pp
   tolerance).
2. **The pair's real drawdowns live in the grinding phases** (2018-19-like: −11%; 2022H1:
   −8.8%) — where mean reversion does help a little (2018: −11.0 → −8.0) but not enough to
   pay for its crash-window and full-period damage.

## Recommendations

1. **ADOPT: nothing. REJECT all four families as a third sleeve.** The TN+OA 50-50 pair
   stands alone — a valid outcome, pre-declared.
2. **If Arun wants a smoother ride, the honest lever is allocation, not a new system:** a
   20-25% cash/liquid sleeve takes the pair to ~22%/−12% (Calmar ~1.8) — that is a sizing
   decision (the r/134 size-down null), not alpha; it deliberately gives up ~5pp CAGR.
3. **The direction worth exploring next is NOT more long-equity mean reversion.** Anything
   long NSE equities that buys weakness re-imports the tail. If a third sleeve is wanted, it
   must earn in the pair's actual loss regimes (grinding sideways/derating phases with the
   gate flickering) — r/134 already mapped this territory for the short-vol book and found
   plain long equity was the answer there; for THIS pair the open question is a
   carry/debt-plus or genuinely non-equity sleeve, which is a new study with its own G0.
4. **Separate, out of scope but flagged:** our net, slot-constrained KC6 rebuild compounds
   near cash rate — if the parked live KC6 system is ever considered for revival, re-derive
   its economics with this engine first (the advertised 20-year PF 1.70 does not survive
   costs+tax+slots on the PIT universe in sleeve form).

## What was NOT tested, and why

- **Non-equity sleeves** (gold, debt-plus-carry, short-vol overlays) — outside "NSE cash
  equities EOD" scope of the ask; the crash-window table above is the motivation to take
  them up properly (new G0, own study).
- Intraday variants of the families; short side (shortability constraints, prior kills).
- Averaging-down variants — banned by the r/84 prior, deliberately.
- Per-family deep re-tuning after G1 kill — the kills were monotone across every coarse
  variant (not near-misses); re-tuning dead families is data-mining.
- KC6 at other slot counts / sizing — its blend failure is structural (crash convergence),
  not a sizing artifact.

## Seven sins

Look-ahead: signals at close t, entries at close t or next-day buy-stops; all indicators
causal; monthly universe applied from the following day. **Survivorship: the dominant caveat
for dip-buying — delisted names are absent, so every standalone number above is
optimistic-end; the negative verdicts are therefore CONSERVATIVE (the true numbers are
worse).** Overfitting: coarse grids, blend metric + kill thresholds pre-registered, baseline
default-wins (and won); no re-tuning of killed families. Costs: 50bps RT base; the kills are
cost-robust (expectancy would need +0.25-0.9%/trade to flip — 5-18× the cost step). Regime:
per-crash-window table is the core exhibit. Correlation: measured daily/monthly AND
per-window — the average-vs-crash divergence is the study's central lesson. Capacity:
top-500 universe, w3 ≤ 33% of a ₹20L-class book — not binding.

## Reproducibility

VPS `/home/arun/quantifyd/research/146_complementary_third_sleeve/`: `scripts/sleeve_engine.py`
(G1, 44 runs), `scripts/blend3.py` (blend/crash stage), `results/g1_candidates.csv`,
`results/blend3.csv`, `results/crash_windows.csv`, sleeve/OA/TN NAV caches. Baseline legs:
r/144 TN engine + r/142 `bluesky_replay`. Data: market_data.db snapshot 2026-09-03.
Compute ~6 min total.
