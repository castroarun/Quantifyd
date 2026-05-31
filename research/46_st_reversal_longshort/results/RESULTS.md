# Short-Term Reversal Long-Short — Results

**Verdict: the reversal spread is real and strong GROSS, but it's a high-turnover
(~50%/day) signal whose edge is almost entirely eaten by realistic costs, and it
has clearly DECAYED in the last ~5 years. At a realistic 10 bps/side it's only
Sharpe 0.36 / +4%/yr (2018+) with −24% MaxDD; at 15 bps it's negative. Plus
shorting Indian cash stocks overnight is structurally hard. Conclusion: a genuine
anomaly but NOT an investable standalone net of costs — an institutional-execution
or diversifier-only proposition, not a retail book.**

Universe 378 liquid names, daily 2000–2026, dollar-neutral long losers / short
winners, daily rebalance with H-day overlapping holding. Cost = per-unit turnover.

## The decisive exhibit — cost sensitivity (best config N=5, decile, H=5, 2018+)

| Cost / side | Net Sharpe | Net ann% |
|---|--:|--:|
| 5 bps | 1.11 | +13.3% |
| **10 bps (realistic liquid-stock)** | **0.36** | **+4.3%** |
| 15 bps | −0.38 | −4.6% |
| 20 bps | −1.13 | −13.5% |

The gross signal is strong (≈Sharpe 1.1 at near-zero cost) but the **break-even is
~13 bps/side** — i.e. the whole edge is a turnover/cost story. Best configs turn
over 50–90% of the book *per day*; even the lowest-turnover variants (H=21) only
cut turnover by trading away most of the gross edge too.

## Edge has decayed (per-year net, best config, 10 bps)

| Era | Read |
|---|---|
| 2005–2011 | Sharpe 1.3–3.1 — but **survivorship-inflated** (today's liquid names in a young, inefficient market) |
| 2018–2020 | mixed-positive (2018 +37%, 2019 +11%, 2020 +3%) |
| **2021–2026** | **mostly negative**: 2021 −13%, 2023 −11%, 2024 +1%, 2026 −11% |

The classic signature of a well-known anomaly being arbitraged away as the market
matured and HFT/quant flow grew. The strong headline (full-sample) numbers lean on
the early, biased period.

## Why it's not investable as-is

1. **Turnover/cost bound.** Break-even ~13 bps/side; retail liquid-stock all-in cost
   (brokerage + STT + impact + slippage, both legs) is comfortably ≥10–15 bps. The
   margin is too thin and on the wrong side of realistic.
2. **Decay.** Negative in 3 of the last 5 years.
3. **Shorting constraint (India-specific).** Daily-rebalanced overnight shorts in
   cash equities aren't feasible without SLB; only the ~180 F&O names are shortable
   via futures — which shrinks the universe and adds cost. A naive 378-name short
   book is not executable.
4. **MaxDD −24%** even market-neutral — not the smooth diversifier we hoped.

## Honest silver lining

It IS market-neutral, so its (modest) return is ~uncorrelated to the long equity
books (MQ, etc.). As a *small* diversifying sleeve for an institution with ≤5 bps
execution it has some value (Sharpe ~1.1 gross-of-decay). For this project's
retail/SME execution reality, it doesn't clear the bar.

## Remaining levers (low expected payoff, honest)

- **F&O-shortable universe only** + futures cost model — more realistic but smaller,
  and the cost problem persists.
- **Long-tilt version** (buy losers vs universe mean, minimal/0 short) — sidesteps
  the shorting constraint but reintroduces market beta (the research/44 problem).
- **Turn-of-month overlay** (from research/45) — orthogonal timing; could lift the
  long side a touch but won't fix the turnover/cost core.

## Where this leaves the search (synthesis across research/43–46)

Four systems tested; a consistent pattern:
- **Directional intraday (H-pattern 43, breakout 44):** real micro-edges killed by
  cost-on-tight-stops and/or single-factor correlation (uninvestible DD).
- **Cross-sectional anomalies (45/46):** the robust ones are either **eaten by
  turnover costs** (reversal) or **already harvested** (momentum → the MQ book).

The implication: net-of-cost alpha in this universe is scarce at high frequency.
The best expected use of effort is **improving the one system that already clears
costs and makes money — the MQ momentum/quality portfolio (32–48% CAGR)** — e.g. a
market-regime overlay to cut its 27% MaxDD, or a modest long-short/quality tilt —
rather than continuing to hunt new high-frequency edges that costs erase.
