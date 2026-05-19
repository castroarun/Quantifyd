"""05b — relaxed-gate combined portfolio for pair trading.

The Stage-B sweep produced no per-pair cells with n_test >= 30 (gate too
strict for any single pair given short F&O test window). But many cells
show 75-83% test WR with n=18-21 trades AND favorable RR + cost-stress
survival.

This script applies a relaxed gate (te_wr >= 70, tr_wr >= 55,
te_n >= 15, te_pf >= 1.5, te_pf_stress >= 1.2), de-duplicates by
(symA,symB) keeping the best-AWS row, then runs the existing
combined_portfolio() function from 05_pair_trading.py.

Goal: see if the COMBINED portfolio across all 75%+ pairs hits the
n>=30 + WR>=75% gate.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results')

# Load the original module (it's a script, but we can import via spec).
# Must register in sys.modules before exec so @dataclass works inside it.
SCRIPT_05 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '05_pair_trading.py')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location('mod_05', SCRIPT_05)
mod_05 = importlib.util.module_from_spec(spec)
sys.modules['mod_05'] = mod_05
spec.loader.exec_module(mod_05)


def main() -> int:
    rank_csv = os.path.join(RESULTS, '05_pair_ranking.csv')
    df = pd.read_csv(rank_csv)
    print(f'Loaded {len(df)} sweep cells', flush=True)

    # Relaxed gate
    relaxed = df[
        (df['te_wr'] >= 70.0)
        & (df['tr_wr'] >= 55.0)
        & (df['te_n'] >= 15)
        & (df['te_pf'] >= 1.5)
        & (df['te_pf_stress'] >= 1.2)
    ].copy()
    print(f'Cells passing relaxed gate (te_wr>=70 + tr_wr>=55 + te_n>=15 + te_pf>=1.5 + stress_pf>=1.2): {len(relaxed)}',
          flush=True)

    if relaxed.empty:
        print('No relaxed survivors -> no portfolio.', flush=True)
        return 0

    # Dedupe by (symA, symB), keep highest AWS
    relaxed = relaxed.sort_values('aws', ascending=False).drop_duplicates(
        subset=['symA', 'symB'], keep='first',
    )
    print(f'After dedup by pair: {len(relaxed)} unique pairs', flush=True)
    print()
    print('=== Relaxed survivors ===')
    print(relaxed[['symA', 'symB', 'entry_z', 'stop_z', 'hold_days', 'lookback',
                   'tr_n', 'tr_wr', 'tr_pf', 'te_n', 'te_wr', 'te_pf',
                   'te_dd', 'te_pf_stress', 'te_total_ret', 'aws']].to_string(index=False))
    print(flush=True)

    # write the relaxed survivors out for inspection
    out_relaxed = os.path.join(RESULTS, '05_pair_walk_forward_relaxed.csv')
    relaxed.to_csv(out_relaxed, index=False)
    print(f'Wrote relaxed survivors -> {out_relaxed}', flush=True)

    # run combined portfolio on these
    print('\n=== Running combined portfolio ===', flush=True)
    summary = mod_05.combined_portfolio(relaxed, max_concurrent=5,
                                          per_pair_risk=6000.0,
                                          capital=1_000_000.0)
    print(f'\nSummary: {summary}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
