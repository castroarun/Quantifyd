# RESULTS — NAS Portfolio Bracket (daily TP/SL on the combined 3-system 916 book)

**Verdict: SIGNAL — a wide daily STOP (~−8k) is a real, robust risk improver; a take-profit is
value-destructive. The 5k/5k the question started from is WORSE than no bracket.**

64 recorded days (2026-04-20 → 2026-07-22), 3 live 916 systems replayed at current configs, 2 lots
each, on the per-minute NIFTY chain. Combined intraday portfolio path per day; daily bracket = first
minute combined P&L breaches TP/SL → flatten (book breach value), else ride to 15:15.

## The headline

| Config | Total (64d) | Calmar | maxDD | Worst day | 1st-half | 2nd-half |
|---|--:|--:|--:|--:|--:|--:|
| Baseline (no bracket) | +17,530 | 0.28 | −61,938 | −25,582 | −15,240 | +32,770 |
| **Stop −8k, no target** | **+73,750** | **2.17** | **−34,037** | −12,310 | +22,713 | +51,036 |
| Stop −8k + target 15k | +83,882 | 2.30 | −36,545 | −12,310 | +32,590 | +51,292 |
| **5k / −5k (the original idea)** | **−5,455** | **−0.11** | −50,986 | −7,246 | −24,522 | +19,068 |

The edge is **entirely in the stop, and specifically near −8,000**. Adding it ~4× the net P&L and
roughly halves the drawdown (−62k → −34k), and it is positive in BOTH halves — it rescues the
otherwise-negative first half (−15,240 → +22,713). That two-regime consistency is the main evidence
it is real, not curve-fit.

## Why a take-profit hurts

Target-only curve (no stop): TP 4k = **−34,020**, 5k = −22,897, 6k = −6,834, 8k = +7,886, 15k =
+27,663. A low daily target **caps the fat right tail of winning (theta) days** that carries the
whole edge. At the best stop (−8k), varying the target is monotonic-bad: none = +73,750, 5k =
+31,301, 8k = +62,084, 15k = +83,882 — i.e. a target only stops hurting once it is so wide (≥12–15k)
it barely triggers. **Do not run a daily take-profit on this book.**

## Why −8k for the stop (and the honest caveat)

Stop-only curve (no target), by SL: −3k +6,288 · −4k +26,192 · −5k +8,842 · −6k +11,398 ·
**−8k +73,750** · −10k +52,368 · −12k +19,616 · −15k +7,400.

−8k is a **peak, not a smooth plateau** — the tight-stop region (−3k…−6k) is noisy and the jump
−6k→−8k (+11k→+73k) is large, so some of −8k's exact dominance is sample luck about where daily
paths happened to dip. The **trustworthy zone is a WIDE stop of −8k to −10k** (both Calmar > 1,
both halves positive); I would not bank on −8,000 being precisely optimal. This is the multiple-
testing caveat (best of 81 cells) stated plainly.

## Honesty / robustness

- **Optimism:** LTP fills, no slippage, 1-min resolution. The −8k stop actually books **−12,310**
  on the worst day (gap-through overshoot) — consistent with live leakage (a −5k live stop booked
  −8k). Real-world totals would be lower; direction robust.
- **Both-halves positive** for the wide-stop configs → not one-regime-driven.
- Replays the CURRENT config over all 64 days (isolates the bracket effect on today's system; not
  the historical realised P&L).
- n=64 is decent but not large; forward-test the exact magnitude before trusting it.

## Recommendation

Run a **daily combined-book stop around −8k to −10k, and NO take-profit.** Simplest effective rule:
flatten all three 916 legs if the combined intraday MTM hits ≈ −8,000; otherwise ride to 15:15.
Expected effect: remove the disaster days, keep the winning tail — ~2× Calmar, ~half the drawdown.
The 5k/5k idea, and low take-profits generally, should be dropped.
