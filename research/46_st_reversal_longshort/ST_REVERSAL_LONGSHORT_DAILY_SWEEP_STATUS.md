# Short-Term Reversal — Market-Neutral Long-Short Portfolio (daily, 378 liquid names)

**STATUS: DONE** — real gross edge but **cost/turnover-bound and decaying**.
Break-even ~13 bps/side; at realistic 10 bps Sharpe only 0.36 / +4%/yr (2018+),
negative in 3 of last 5 yrs; shorting Indian cash overnight is structurally hard.
**Not investable standalone.** Verdict: `results/RESULTS.md`. Synthesis across
research/43–46: net-of-cost HF alpha is scarce here → best ROI is improving the
MQ winner (regime overlay to cut its 27% MaxDD), not hunting new HF edges.

Long recent losers / short recent winners, dollar-neutral, rebalanced daily with
H-day overlapping holding. Born from research/45's anomaly scan, where short-term
reversal was the standout (monotonic deciles, t≈−9.78) — and being market-neutral
it structurally avoids the correlated drawdowns that made research/43–44
uninvestible. The whole game here is **whether the gross reversal spread survives
turnover/costs** — so this build is all about portfolio construction + a realistic
cost model + an honest equity curve.

## 1. The Ask
Build the short-term mean-reversion long-short properly (research/45 recommendation):
test holding period, quantile width, weighting (turnover), realistic costs; produce
an equity curve with Sharpe/MaxDD; add a turn-of-month overlay.

## 2. The Base — mechanics
- **Universe:** 378 liquid F&O/N500 names (the 5-min-available set), daily 2000–2026.
  *Caveat: universe = today's liquid names → survivorship bias pre-~2012; report a
  2018+ subperiod as the honest modern read.*
- **Signal (causal):** past-N-day return `close_t/close_{t-N} − 1`, ranked
  cross-sectionally each day. Low = loser (LONG), high = winner (SHORT).
- **Construction (axes):**
  - lookback N ∈ {3, 5, 10}
  - quantile: decile (10/90), quintile (20/80), rank-weighted (all names, linear)
  - holding H ∈ {3, 5, 10, 21} via overlapping tranches (book = H-day mean of daily
    target weights → daily rebalance of ~1/H of the book; cuts turnover)
  - dollar-neutral: long weights sum +1, short sum −1 (gross 2×, net 0)
- **Return:** `r_t = Σ book_{t−1} · ret_t` (prior-day book earns today's return → no
  look-ahead). **Turnover** `= Σ|book_t − book_{t−1}|`; **cost** `= turnover × c`,
  `c = 10 bps one-way` (liquid-stock brokerage+impact+slippage; sensitivities shown).
- **Metrics:** annualized return, vol, Sharpe, MaxDD, avg daily turnover, gross vs
  net — full sample and 2018+ subperiod.

## 3. Plan
- Grid: N(3) × quantile(3) × H(4) = 36 configs, full + 2018+ subperiod.
- Pick best net-Sharpe config → save equity curve, per-year returns, cost sensitivity.
- Turn-of-month overlay: scale exposure on ToM days (≤3 / ≥28) as a variant.

## 4. Status
| Date/time (IST) | Event | Notes |
|---|---|---|
| 2026-05-29 | Folder + STATUS created | before launch |

## 5. Crash recovery
Single script `scripts/run_reversal.py` (fast, in-memory; no long run). Re-run to
regenerate. Outputs in `results/`.

## 6. Files
| File | Purpose |
|---|---|
| `scripts/run_reversal.py` | Panel load + long-short backtest grid + best-config detail |
| `results/grid.csv` | All configs: net Sharpe/return/DD/turnover |
| `results/equity_best.csv` | Equity curve of the chosen config |
| `results/RESULTS.md` | Verdict |

## 7. Findings
*(populated as results land)*
