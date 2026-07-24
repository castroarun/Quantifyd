# NAS 09:16 Straddle — Per-Leg SL Tightening Sweep (does an intraday-tightened stop beat fixed 30%-off-entry?)
STATUS: DONE — VERDICT: NO VALIDATED EDGE. Re-anchor benefit FLIPS SIGN (Apr-May −8,831/d vs Jun-Jul +7,246/d; full 48d ≈ −457/d). Earlier +1,225/lot was a window artifact. Ex-ante gating (CPR/gap/VIX/OR) has zero predictive power. DO NOT deploy. Shadow cron running for forward evidence.

## The Ask
**What you asked:** we act on a 30% SL, but that's 30% off the MORNING premium — once the day's decay has
banked most of the gain, a stop 30% above the entry sits miles above the now-cheap premium (a leg at 80
with a 165 stop can rally 85 pts before stopping). Would bringing the SL closer at some point work better?
Assess comprehensively, trial-and-error.

**What we're testing:** on the reconstructed 09:16 ATM short straddle, does any intraday SL-tightening
policy beat the fixed `SL = entry×1.3` per leg, NET of cost, across all recorded days — and if so which
trigger (trail-to-breakeven / ratchet / time / profit-lock) and how tight.

## The Base (fixed)
- 09:16 ATM straddle (CE+PE), front weekly. Each leg managed by its own per-leg SL. Held to 15:15 unless a
  leg's premium rises to its (policy-defined) SL, which books that leg; the other leg runs on.
- Universe/period: NIFTY, all recorded days in options_data.db (2026-04-20 → 07-10).
- Spot = underlying_spot TABLE; premiums = option_chain (that strike, nearest ≤90s), full 09:16–15:15 series.
- Per 1 lot = 65. Cost = 0.15%/leg per transaction (open+close each leg). Cost sensitivity 0.10/0.15/0.20%.
- Success metric: mean NET daily P&L/lot vs BASELINE; plus win%, worst day, per-DTE, and the honest
  **premature-stopout cost vs give-back saved** split.
- SIMPLIFICATION: no re-enter after a stop (isolates the SL-timing effect); noted as a caveat.

## Plan (per-leg SL policies)
| Policy | Rule (per leg) |
|---|---|
| BASELINE | SL = entry×1.3 (current 30% off morning premium) |
| TRAIL_BE | once premium < entry (in profit), SL → entry (breakeven lock) |
| RATCHET_X | SL = min(entry×1.3, running_min_premium×(1+X)), X ∈ {0.20,0.30,0.40} |
| TIME_TIGHTEN_T | after T, SL → min(entry×1.3, premium@T ×1.3) [re-anchor to decayed premium], T ∈ {11:15,12:00,13:00} |
| PROFIT_LOCK_Y | once premium ≤ entry×(1−Y), SL → entry (breakeven lock), Y ∈ {0.40,0.50,0.60} |

≈ 11 policies × all days. Read-only reconstruction.

## Honest tension (what we're really testing)
research/76 showed straddle P&L is BACK-LOADED (peaks ~15:00). Tightening the SL protects banked gains BUT
risks whipsawing you out of a leg that spikes then decays back — forfeiting the late theta. The study
measures whether protection > premature-exit cost, per policy.

## Status
| Time | Event |
|---|---|
| 2026-07-10 15:3x | STATUS written; simulator building |

## Crash Recovery
Script: research/77_sl_tightening/scripts/sl_tighten_sweep.py (VPS). Read-only on options_data.db.
Re-run: cd /home/arun/quantifyd && ./venv/bin/python3 research/77_sl_tightening/scripts/sl_tighten_sweep.py.
Output: results/policy_summary.csv, per_day.csv, run.log. Safe to re-run (idempotent). Nothing live touched.

## Files
| File | Purpose | Commit? |
|---|---|---|
| scripts/sl_tighten_sweep.py | simulator | yes |
| results/policy_summary.csv | per-policy aggregate | yes |
| results/per_day.csv | per-day per-policy | yes if small |
| RESULTS.md | verdict | yes |

## Findings
(pending run)
