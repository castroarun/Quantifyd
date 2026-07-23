# RESULTS — SENSEX Portfolio Bracket (own per-lot stop calibration)

**Verdict: the deployed −₹1,300/lot stop is CORRECT for SENSEX too (−8k on 6 lots = −1,333/lot is
the only positive stop). But SENSEX is more fragile than NIFTY — a knife-edge peak on a barely-
positive base — and, unlike NIFTY, a take-profit HELPS on SENSEX. Keep −1,300/lot; treat a SENSEX TP
as a hypothesis, not a deploy.**

64 recorded days, 3 SENSEX systems replayed at current live configs, 2 lots each (QTY 40), on the
per-minute SENSEX chain.

## Stop level — confirmed −1,300/lot, but a knife-edge

Stop-only curve (no target), by SL (per-lot = SL/6):

| SL | per-lot | total | Calmar | maxDD |
|--:|--:|--:|--:|--:|
| −2k | −333 | −67,574 | −0.65 | −103,234 |
| −4k | −667 | −51,320 | −0.53 | −97,178 |
| −6k | −1,000 | −23,456 | −0.35 | −67,398 |
| **−8k** | **−1,333** | **+17,376** | **0.39** | −45,090 |
| −10k | −1,667 | −2,094 | −0.03 | −64,726 |
| −12k | −2,000 | −20,508 | −0.25 | −82,134 |

**Only the −8k (−1,333/lot) stop is positive** — essentially identical to NIFTY's −1,333/lot, so the
deployed −₹1,300/lot needs no change. But note the sharp peak: −6k = −23k, −8k = +17k, −10k = −2k.
Tight stops are *catastrophic* on SENSEX (index is more volatile per point → a tight rupee stop is
hit on noise and locks losses before the theta recovery). This is far less robust than NIFTY's −8k/
−10k plateau — the margin for error is thin. Both halves positive at −8k (h1 +7,302 / h2 +10,074).

## Take-profit — HELPS on SENSEX (opposite of NIFTY)

| Config | total | Calmar | maxDD | h1 / h2 |
|---|--:|--:|--:|--:|
| Stop −8k, no target | +17,376 | 0.39 | −45,090 | +7,302 / +10,074 |
| **TP 5k + SL −8k** | +24,192 | **0.74** | **−32,536** | +2,936 / +21,256 |
| TP 10k + SL −8k | **+34,324** | 0.69 | −49,628 | +14,572 / +19,752 |

On SENSEX a 5k–10k daily target *improves* both total and Calmar over stop-only — the reverse of
NIFTY. **Mechanism:** SENSEX gives its intraday gains back by the close (median EOD **−396**), whereas
NIFTY grinds up into the close (median EOD **+798**). So locking a profit captures the SENSEX peak
before the fade, while on NIFTY it chopped the winner. This is a real structural difference, not noise.

## Honesty / robustness

- **Weak base:** SENSEX baseline (no bracket) is −2,610 over 64 days — barely negative. The book is
  marginal; the stop is what makes it viable. Temper any enthusiasm for SENSEX live sizing.
- **Fragile stop:** −8k is a knife-edge peak, not a plateau. Small level shifts flip the sign.
- **TP-helps is tentative:** best-of-grid on a weak, noisy base with only ~2 real SENSEX live days,
  and it contradicts NIFTY. Do NOT deploy a SENSEX TP yet — revisit after more live/OOS data.
- Same optimism caveat (LTP fills, no slippage, 1-min).

## Recommendation

1. **Keep −₹1,300/lot** as the SENSEX stop — now validated (was provisional). No code change.
2. **Do not add a SENSEX take-profit yet** — promising but thin. Log it as a hypothesis; re-test once
   more SENSEX live days accumulate.
3. **Flag SENSEX as marginal** — near-zero base + fragile stop argues for small size / caution live.
