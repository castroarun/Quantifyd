# NAS 09:16 ATM Straddle — Churn / Exit-Policy Sweep (reconstructed from the chain recorder)
STATUS: DONE — see RESULTS_P2.md (G1 probe, n=14: exit-15:00 + lull-hold promising, keep 0.4% move-stop, book-early NOT supported)

## The Ask
**What you asked:** you feel the trade is green in the first 30–60 min then "volatility seeps in" and
erodes it — proposing book-early/re-enter. Phase-1 (research/76 verify) showed the OPPOSITE for the
*idealized held* straddle (red early, builds to a LATE peak), and pointed at the **move-stop churn** as
the real driver of the live erosion. You said "yes" to the follow-up.

**What we're testing:** across every recorded day, simulate the 09:16 ATM short straddle under different
exit/adjustment policies and find which one keeps the most of the day's (back-loaded) theta net of cost —
i.e. does *reducing move-stop churn* (wider bands / hold-through-lull / no-re-enter) beat the current 0.4%
re-center? Confirm the late-peak shape on the fuller sample too.

## The Base (what's fixed)
- Signal/entry: SELL ATM straddle (CE+PE at round(spot/50)) at **09:16**, front weekly expiry.
- Exit: **15:15** EOD square-off (unless policy exits earlier).
- Universe/period: NIFTY, all recorded days in options_data.db (**2026-04-20 → 07-09, ~55 days**).
- Spot: underlying_spot TABLE (dense, ~1066/day). Premiums: option_chain, nearest snapshot ≤90s.
- Sizing: per 1 lot (65) for reporting; scale later.
- **Cost: 0.15%/leg per transaction** (each open & close of each leg) — a re-center = 4 leg-txns.
  Also report gross and a cost-sensitivity (0.10 / 0.15 / 0.20%).
- Success metric: **mean net daily P&L per lot**, ranked; plus win%, per-DTE, avg #re-centers (churn),
  worst day, std. A policy must beat HOLD *and* the current 0.4% net.

## Plan (policy grid)
| Policy | Rule |
|---|---|
| HOLD | no adjustment, hold to 15:15 (idealized decay baseline) |
| SL30_NORE | per-leg 30% SL, close that leg, keep the other, no re-enter |
| MOVE_b | move-stop re-center (close+re-open ATM) at |spot−entry_spot|≥ b%, cap 5, always re-center. b ∈ {0.3, 0.4(current), 0.5, 0.6, 0.8, 1.0} |
| MOVE_0.4_LULLHOLD | 0.4% band but suppress re-center 11:15–13:00 (research/75 lull) |
| MOVE_0.4_NOREENTER | 0.4% band closes on trigger, then FLAT (no re-open) |
| MOVE_0.4_EXIT1500 | 0.4% band, hard exit 15:00 (test late-peak: decay is back-loaded) |

Total ≈ 11 policies × ~55 days. Read-only sim, in-memory per day. Also re-runs the phase-1 late-peak
path on the fuller sample.

## Status (live log)
| Time | Event |
|---|---|
| 2026-07-09 ~16:1x | STATUS written; sim script being built |

## Crash Recovery
- Script: `research/76_early_peak_reentry/scripts/churn_exit_sweep.py` (VPS). Read-only on
  options_data.db. Re-run: `cd /home/arun/quantifyd && ./venv/bin/python3 research/76_early_peak_reentry/scripts/churn_exit_sweep.py`.
- Output: `research/76_early_peak_reentry/results/policy_summary.csv` + `per_day.csv`. Safe to re-run
  (idempotent, overwrites). Nothing live is touched.

## Files
| File | Purpose | Commit? |
|---|---|---|
| scripts/churn_exit_sweep.py | the simulator | yes |
| results/policy_summary.csv | per-policy aggregate | yes |
| results/per_day.csv | per-day per-policy P&L | yes if small |
| RESULTS.md | verdict | yes |

## Findings
(pending run)
