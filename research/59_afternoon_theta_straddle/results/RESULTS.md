# Afternoon Theta Straddle (12:15→15:15) — RESULTS

**Verdict: SIGNAL (regime + DTE conditional), NOT a standalone STRATEGY.**
One narrow keeper survives per-year stability; the rest is overfit or net-negative.

## What was tested
Short ATM NIFTY straddle, sell 12:15 / cover 15:15, 10 lots (QTY 650), to harvest
afternoon theta on idle post-noon cash. Baseline (hold to close) and +0.4%
underlying move-stop. Datasets: user CSVs (598 days, 2024-01→2026-06, real fills +
costs) and our recorded chain (options_data.db, 30 days 2026-04→06, true expiries).

## Headline numbers (user CSVs, 598 days)
| | Baseline | +0.4% SL |
|---|---|---|
| Total | **−₹1.22 L** | **−₹2.16 L** |
| Win rate | 63.2% | 55.5% |
| Median day | +₹3,594 | +₹1,265 |
| Profit factor | 0.98 | 0.94 |
| Max DD | −₹6.39 L | −₹5.22 L |
| Worst day | −₹1.82 L | −₹1.01 L |

Neither is net-positive over 2.5 years. Theta is real (win most days) but the
high-vol left tail eats it. **The 0.4% stop makes total WORSE** (−₹94k) — it cuts the
tail (DD/worst-day/P5 all ~20% better) but whipsaws on 45% of days; worst in calm VIX.

## The two real levers

### 1. VIX gate (robust)
Baseline by VIX: <13 +₹55k, 13-16 +₹131k, 16-20 −₹82k, **≥20 −₹2.26 L (−₹5,650/day)**.
Gate at **VIX<16 → flips to +₹1.86 L / 459 days (+₹405/day)**. 2026's −₹6 L year = the
high-VIX stretch.

### 2. DTE = EXPIRY DAY (DTE 0), the only per-year-stable bucket
Baseline, VIX<16, total by year × DTE:
| Year | DTE0 | DTE1 | DTE2 | DTE3 | DTE4 |
|---|---|---|---|---|---|
| 2024 | +126k | −181k | +168k | +33k | −72k |
| 2025 | +88k | +73k | +40k | +38k | +85k |
| 2026 | +37k | −122k | −88k | +4k | −46k |

**Only DTE 0 is positive all three years** (+126/+88/+37k = +₹2,514/day over 100 days).
DTE 2 looked great in aggregate but flipped −88k in 2026 (overfit). DTE 1/4 unstable.
DTE 0 reconstruction validated (2024 Thu→DTE0 50/50; 2026 Tue→DTE0 19/19).

### Killed by stability (were overfit):
- "DTE 0/2/3 all work" → only DTE 0 (DTE 2 flipped 2026).
- "0.4% stop helps on expiry day" → **2024 artifact only** (DTE0 VIX<16: 2024 +326k w/stop
  vs +126k base; but 2025 +59k<+88k, 2026 +11k<+37k → stop hurts in both recent years).

## Cross-check (our recorded chain, 30 days, all high-VIX 2026)
Baseline −₹1.32 L, +SL −₹1.53 L (stop worse — consistent). **True-DTE: expiry-day (DTE0)
= −₹1.49 L / 6 days (−₹24,844/day)** — confirms DTE0 is the WORST place in high VIX (max
gamma). So the edge is **DTE0 AND VIX<16**, not DTE0 alone. (Tiny high-vol-only sample;
can't validate the low-VIX edge, but independently confirms the high-vol danger + stop-hurts.)

## Deployable rule the data supports (narrow, honest)
> Afternoon ATM straddle, **EXPIRY DAY ONLY (DTE 0), only VIX < 16, hold to 15:15, no stop.**
> +₹2,514/day, positive every year — though edge is shrinking (+126→+88→+37k).

## Caveats / open
- Expiry-day gamma tail is real (−₹1.82 L worst day on DTE0). Test **defined-risk wings**
  (~₹300-500 OTM longs) instead of the whipsaw-prone move-stop.
- DTE0 edge is shrinking YoY — is it VIX-level drift or genuine decay? (open).
- 2025 DTE labels assume Thu→Tue switch on 2025-09-01 (2024/2026 anchored & validated).
- In-sample bucket selection across DTE×VIX×stop → real multiple-testing risk; treat the
  DTE0+VIX<16 rule as a hypothesis to paper-forward, not a proven edge.
