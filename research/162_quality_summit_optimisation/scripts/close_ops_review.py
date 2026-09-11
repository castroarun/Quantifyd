# -*- coding: utf-8 -*-
"""research/162 — close the 10-Oct-2026 overlay review in the Ops & Review Centre.

research/160 registered a dated review: "does a loose quality gate improve Open Alpha's OWN
entries as an overlay?", due 2026-10-10, with a pre-registered pass criterion. research/162
Part B ran it five weeks early against the honest Base Age book and it failed on every seed.
This rewrites that entry as a DONE record carrying the outcome (due = None, so the renderer
cannot flag a completed review as overdue), and registers the one dated obligation the
result actually creates.

Idempotent: re-running it is a no-op.
"""
import io
import sys
from pathlib import Path

ROOT = Path('/home/arun/quantifyd')
if not ROOT.exists():
    ROOT = Path(__file__).resolve().parents[3]
P = ROOT / 'research/111_sensex_manual_mgmt/scripts/ops_center.py'

START = '    ("research/160 quality-growth near ATH - the ONE open question'
END = 'quality-growth-near-ath-research160"),\n'

DONE = '''    ("research/160 quality-growth near ATH - quality as an OVERLAY inside Open Alpha's entries - DONE 12-Sep-2026, FAILED",
     None, "DONE",
     "CLOSED EARLY (due was 2026-10-10) by research/162 Part B, run against the HONEST book - "
     "Open Alpha V2.0 / Base Age (r/161), whose no-mask control reproduces its published "
     "21.26% / -34.80% / Calmar 0.618 exactly, so the r/159 look-ahead problem that blocked this "
     "test is no longer in the way. Each eligibility mask was applied to ENTRIES only: a candidate "
     "new-ATH close is dropped unless its symbol passes the screen on the SIGNAL day; exits, sizing, "
     "slot contention and the 60-bar re-arm untouched; 30 seeds; both missing-data policies; two "
     "windows. PASS CRITERION WAS >= +2pp after-tax CAGR or >= +0.15 Calmar on >= 8 of 12 paired "
     "paths. RESULT: NOT ONE SCREEN WINS ON A SINGLE SEED OUT OF THIRTY on return, in either window, "
     "under either policy. Control 26.57% CAGR / -26.57% DD / Calmar 0.999 (2018-08->2026-09, 87% "
     "invested); + b7 (the loose gate this review named) 16.79%, paired -9.71pp on 0/30 and -0.248 "
     "Calmar on 0/30; + quality-only 15.02% (-11.28pp); + growth-only 20.77% (-5.80pp, Calmar +0.014 "
     "on 19/30, still a fail); + the screen as Arun wrote it 8.80% (-17.77pp) at 18.8% invested. "
     "MECHANISM: starvation, not selection - the screen cuts qualifying signals from 3,619 to 468 "
     "(b7) or 76 (strict) and the invested fraction from 87% to 63% or 19%. THE QUALITY-SCREEN LINE "
     "IS CLOSED PERMANENTLY, as this review's own text instructed on a fail. Evidence: "
     "research/162_quality_summit_optimisation/results/RESULTS.md and "
     "/app/backtest/quality-summit-optimisation-research162"),
'''

NEW = '''    ("research/162 - re-open the Quality Summit optimisation ONLY when the holdout has grown a year",
     "2027-09-12", "PENDING",
     "r/162 Part A found a real-looking improvement to Quality Summit - keep the b7 screen, widen the "
     "near-ATH band from k=0.90 to 0.85, cut the book from 15 names to 10, size inverse-volatility - "
     "worth +6.06pp CAGR and +0.282 Calmar on 12 of 12 rebalance offsets in the FIT window "
     "(2018-08 -> 2022-06), on a verified 24-cell plateau, surviving the cost ladder. The "
     "pre-registered HOLDOUT (2022-07 -> 2026-09) returned -3.48pp on 3 of 12 offsets and -0.152 "
     "Calmar on 1 of 12; its holdout CAGR sits 9.22pp below its fit CAGR against a 4pp limit written "
     "down before the run, and deleting its ten best trades takes its compounding proxy BELOW ONE. "
     "NOT ADOPTED. The honest caveat is that the holdout is only four years and is dominated by the "
     "2023-25 smallcap boom, so the failure could in principle be regime rather than overfit. "
     "RE-CHECK WITH A YEAR MORE DATA - and only with a year more data; re-running it on the same "
     "window is holdout mining. PASS CRITERION, unchanged: >= +0.15 Calmar OR >= +2pp CAGR at no "
     "worse drawdown, on >= 8 of 12 offsets, in BOTH windows, with the extended holdout no more than "
     "4pp below the fit window. If it fails again, close the Quality Summit optimisation line "
     "permanently. Cost: about two hours - every derived cache, mask and grid is committed. "
     "Evidence: research/162_quality_summit_optimisation/results/RESULTS.md and "
     "/app/backtest/quality-summit-optimisation-research162"),
'''


def main():
    s = io.open(P, encoding='utf-8').read()
    if 'research/162 - re-open the Quality Summit optimisation' in s:
        print('already closed - nothing to do')
        return 0
    i = s.find(START)
    if i < 0:
        print('FAILED: could not find the r/160 overlay review entry')
        return 2
    j = s.find(END, i)
    if j < 0:
        print('FAILED: could not find the end of that entry')
        return 2
    j += len(END)
    s = s[:i] + DONE + NEW + s[j:]
    io.open(P, 'w', encoding='utf-8').write(s)
    print('ops_center.py: 10-Oct-2026 overlay review marked DONE (failed); '
          'one new dated review registered for 2027-09-12')
    return 0


if __name__ == '__main__':
    sys.exit(main())
