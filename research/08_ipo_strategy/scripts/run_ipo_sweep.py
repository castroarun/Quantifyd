"""
IPO Strategy Research — Full Parameter Sweep
Phase 1: All exit strategies with default entry (vol=1.5, ATH=5)
Phase 2: Top exits × volume multipliers
Phase 3: Top combos × ATH lookback periods
"""
import sys, os, csv, time, logging
import numpy as np

logging.disable(logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.ipo_strategy import IPOStrategyBacktester, IPOConfig

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ipo_sweep_results.csv')
FIELDNAMES = [
    'phase', 'label', 'exit_strategy', 'exit_params',
    'vol_multiplier', 'ath_lookback', 'vol_avg_period',
    'total_trades', 'winners', 'losers', 'win_rate',
    'avg_return_pct', 'avg_winner_pct', 'avg_loser_pct',
    'max_return_pct', 'max_loss_pct', 'profit_factor', 'expectancy',
    'total_pnl', 'final_capital', 'total_return_pct', 'cagr', 'sharpe', 'max_drawdown',
    'avg_hold_days', 'exit_breakdown',
]

# Skip already-done configs
done = set()
if os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV) as f:
        done = {row['label'] for row in csv.DictReader(f)}
    if done:
        print(f'Skipping {len(done)} already-completed configs')
else:
    with open(OUTPUT_CSV, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

def run_config(phase, label, params):
    """Run a single config and write to CSV."""
    if label in done:
        print(f'  [{label}] SKIPPED')
        return None

    t0 = time.time()
    cfg = IPOConfig(**params)
    bt = IPOStrategyBacktester(cfg)
    r = bt.run()
    elapsed = time.time() - t0

    # Exit reason breakdown
    exit_counts = {}
    for t in r.trades:
        exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1
    exit_breakdown = '; '.join(f'{k}={v}' for k, v in sorted(exit_counts.items()))

    row = {
        'phase': phase,
        'label': label,
        'exit_strategy': params.get('exit_strategy', 'supertrend'),
        'exit_params': _describe_exit_params(params),
        'vol_multiplier': params.get('vol_multiplier', 1.5),
        'ath_lookback': params.get('ath_lookback_days', 5),
        'vol_avg_period': params.get('vol_avg_period', 20),
        'total_trades': r.total_trades,
        'winners': r.winners,
        'losers': r.losers,
        'win_rate': r.win_rate,
        'avg_return_pct': r.avg_return_pct,
        'avg_winner_pct': r.avg_winner_pct,
        'avg_loser_pct': r.avg_loser_pct,
        'max_return_pct': r.max_return_pct,
        'max_loss_pct': r.max_loss_pct,
        'profit_factor': r.profit_factor,
        'expectancy': r.expectancy,
        'total_pnl': round(r.total_pnl),
        'final_capital': round(r.final_capital),
        'total_return_pct': r.total_return_pct,
        'cagr': r.cagr,
        'sharpe': r.sharpe,
        'max_drawdown': r.max_drawdown,
        'avg_hold_days': r.avg_hold_days,
        'exit_breakdown': exit_breakdown,
    }

    with open(OUTPUT_CSV, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

    print(f'  [{label}] {elapsed:.1f}s | Trades={r.total_trades} Win={r.win_rate}% PF={r.profit_factor} '
          f'Exp={r.expectancy} CAGR={r.cagr}% Sharpe={r.sharpe} MaxDD={r.max_drawdown}% '
          f'AvgHold={r.avg_hold_days}d')
    sys.stdout.flush()
    return r

def _describe_exit_params(params):
    """Human-readable exit params."""
    strat = params.get('exit_strategy', 'supertrend')
    if strat == 'supertrend':
        return f"ATR={params.get('st_atr_period',10)},M={params.get('st_multiplier',3.0)}"
    elif strat == 'trailing_sl':
        return f"Trail={params.get('trail_pct',15)}%"
    elif strat == 'fixed_target':
        return f"T={params.get('target_pct',30)}%,SL={params.get('stop_loss_pct',10)}%"
    elif strat == 'time_exit':
        return f"Days={params.get('max_hold_days',30)},SL={params.get('stop_loss_pct',10)}%"
    elif strat == 'ema_cross':
        return f"EMA={params.get('ema_fast',5)}/{params.get('ema_slow',20)}"
    elif strat == 'atr_trail':
        return f"ATR={params.get('atr_trail_period',14)},M={params.get('atr_trail_mult',2.0)}"
    elif strat == 'chandelier':
        return f"P={params.get('chandelier_period',22)},M={params.get('chandelier_mult',3.0)}"
    return strat

# ============================================================
# PHASE 1: All exit strategies (vol=1.5, ATH=5)
# ============================================================
print('='*60)
print('PHASE 1: Exit Strategy Sweep (vol=1.5, ATH lookback=5)')
print('='*60)

phase1_configs = []
base = dict(vol_multiplier=1.5, ath_lookback_days=5, vol_avg_period=20)

# 1a. Supertrend variants
for atr_p, mult in [(7, 2.0), (7, 3.0), (10, 2.0), (10, 3.0), (14, 2.0), (14, 3.0)]:
    phase1_configs.append((
        f'ST_{atr_p}_{mult}',
        {**base, 'exit_strategy': 'supertrend', 'st_atr_period': atr_p, 'st_multiplier': mult}
    ))

# 1b. Trailing SL variants
for trail in [8, 10, 12, 15, 20, 25, 30]:
    phase1_configs.append((
        f'TRAIL_{trail}PCT',
        {**base, 'exit_strategy': 'trailing_sl', 'trail_pct': trail}
    ))

# 1c. Fixed Target + SL variants
for target, sl in [(15, 7), (20, 10), (25, 10), (30, 10), (30, 15), (40, 15), (50, 15), (50, 20), (60, 20)]:
    phase1_configs.append((
        f'FIX_T{target}_SL{sl}',
        {**base, 'exit_strategy': 'fixed_target', 'target_pct': target, 'stop_loss_pct': sl}
    ))

# 1d. Time exit variants
for days, sl in [(10, 7), (15, 10), (20, 10), (30, 10), (45, 15), (60, 15), (90, 15)]:
    phase1_configs.append((
        f'TIME_{days}D_SL{sl}',
        {**base, 'exit_strategy': 'time_exit', 'max_hold_days': days, 'stop_loss_pct': sl}
    ))

# 1e. EMA cross variants
for fast, slow in [(5, 13), (5, 20), (8, 21), (10, 30), (13, 34)]:
    phase1_configs.append((
        f'EMA_{fast}_{slow}',
        {**base, 'exit_strategy': 'ema_cross', 'ema_fast': fast, 'ema_slow': slow}
    ))

# 1f. ATR trailing stop variants
for atr_p, mult in [(10, 2.0), (10, 3.0), (14, 2.0), (14, 3.0), (20, 2.0), (20, 3.0)]:
    phase1_configs.append((
        f'ATR_{atr_p}_{mult}',
        {**base, 'exit_strategy': 'atr_trail', 'atr_trail_period': atr_p, 'atr_trail_mult': mult}
    ))

# 1g. Chandelier exit variants
for period, mult in [(14, 2.0), (14, 3.0), (22, 2.0), (22, 3.0)]:
    phase1_configs.append((
        f'CHAN_{period}_{mult}',
        {**base, 'exit_strategy': 'chandelier', 'chandelier_period': period, 'chandelier_mult': mult}
    ))

print(f'Phase 1: {len(phase1_configs)} configs')
for label, params in phase1_configs:
    run_config('P1', label, params)

# ============================================================
# PHASE 2: Top exits × Volume Multipliers
# ============================================================
print()
print('='*60)
print('PHASE 2: Volume Multiplier Sweep (best exits × vol variations)')
print('='*60)

# Read Phase 1 results to find top exits
phase1_results = []
if os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV) as f:
        for row in csv.DictReader(f):
            if row['phase'] == 'P1' and int(row['total_trades']) >= 5:
                phase1_results.append(row)

# Sort by expectancy (risk-adjusted per-trade metric)
phase1_results.sort(key=lambda x: float(x['expectancy']), reverse=True)
top5_exits = phase1_results[:5]

if top5_exits:
    print(f'Top 5 exits by expectancy:')
    for r in top5_exits:
        print(f'  {r["label"]}: Exp={r["expectancy"]} Win={r["win_rate"]}% PF={r["profit_factor"]} Trades={r["total_trades"]}')
else:
    print('No Phase 1 results found with >= 5 trades, using defaults')
    top5_exits = [
        {'label': 'ST_10_3.0', 'exit_strategy': 'supertrend', 'exit_params': 'ATR=10,M=3.0'},
        {'label': 'TRAIL_15PCT', 'exit_strategy': 'trailing_sl', 'exit_params': 'Trail=15%'},
        {'label': 'FIX_T30_SL10', 'exit_strategy': 'fixed_target', 'exit_params': 'T=30%,SL=10%'},
        {'label': 'EMA_5_20', 'exit_strategy': 'ema_cross', 'exit_params': 'EMA=5/20'},
        {'label': 'ATR_14_2.0', 'exit_strategy': 'atr_trail', 'exit_params': 'ATR=14,M=2.0'},
    ]

# Reconstruct config params from top exit labels
def label_to_params(label):
    """Reconstruct config params from Phase 1 label."""
    parts = label.split('_')
    if label.startswith('ST_'):
        return {'exit_strategy': 'supertrend', 'st_atr_period': int(parts[1]), 'st_multiplier': float(parts[2])}
    elif label.startswith('TRAIL_'):
        pct = int(parts[1].replace('PCT', ''))
        return {'exit_strategy': 'trailing_sl', 'trail_pct': pct}
    elif label.startswith('FIX_'):
        target = int(parts[1][1:])  # T30 → 30
        sl = int(parts[2][2:])      # SL10 → 10
        return {'exit_strategy': 'fixed_target', 'target_pct': target, 'stop_loss_pct': sl}
    elif label.startswith('TIME_'):
        days = int(parts[1][:-1])  # 30D → 30
        sl = int(parts[2][2:])     # SL10 → 10
        return {'exit_strategy': 'time_exit', 'max_hold_days': days, 'stop_loss_pct': sl}
    elif label.startswith('EMA_'):
        return {'exit_strategy': 'ema_cross', 'ema_fast': int(parts[1]), 'ema_slow': int(parts[2])}
    elif label.startswith('ATR_'):
        return {'exit_strategy': 'atr_trail', 'atr_trail_period': int(parts[1]), 'atr_trail_mult': float(parts[2])}
    elif label.startswith('CHAN_'):
        return {'exit_strategy': 'chandelier', 'chandelier_period': int(parts[1]), 'chandelier_mult': float(parts[2])}
    return {}

phase2_configs = []
for exit_row in top5_exits:
    exit_label = exit_row['label']
    exit_params = label_to_params(exit_label)
    for vol_mult in [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]:
        lbl = f'{exit_label}_V{vol_mult}'
        phase2_configs.append((
            lbl,
            {**exit_params, 'vol_multiplier': vol_mult, 'ath_lookback_days': 5, 'vol_avg_period': 20}
        ))

print(f'Phase 2: {len(phase2_configs)} configs')
for label, params in phase2_configs:
    run_config('P2', label, params)

# ============================================================
# PHASE 3: Top combos × ATH Lookback
# ============================================================
print()
print('='*60)
print('PHASE 3: ATH Lookback Sweep (best combos × lookback variations)')
print('='*60)

# Read Phase 2 results to find top combos
phase2_results = []
if os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV) as f:
        for row in csv.DictReader(f):
            if row['phase'] == 'P2' and int(row['total_trades']) >= 5:
                phase2_results.append(row)

phase2_results.sort(key=lambda x: float(x['expectancy']), reverse=True)
top3_p2 = phase2_results[:3]

if top3_p2:
    print(f'Top 3 combos from Phase 2:')
    for r in top3_p2:
        print(f'  {r["label"]}: Exp={r["expectancy"]} Win={r["win_rate"]}% PF={r["profit_factor"]} Trades={r["total_trades"]}')

phase3_configs = []
for combo_row in top3_p2:
    combo_label = combo_row['label']
    # Extract exit + vol from label: e.g. ST_10_3.0_V1.5
    vol_idx = combo_label.rfind('_V')
    exit_label = combo_label[:vol_idx]
    vol_mult = float(combo_label[vol_idx+2:])
    exit_params = label_to_params(exit_label)

    for ath_lookback in [3, 5, 7, 10, 15]:
        for vol_avg in [10, 15, 20, 30]:
            lbl = f'{exit_label}_V{vol_mult}_ATH{ath_lookback}_VA{vol_avg}'
            phase3_configs.append((
                lbl,
                {**exit_params, 'vol_multiplier': vol_mult, 'ath_lookback_days': ath_lookback,
                 'vol_avg_period': vol_avg}
            ))

if phase3_configs:
    print(f'Phase 3: {len(phase3_configs)} configs')
    for label, params in phase3_configs:
        run_config('P3', label, params)
else:
    print('No Phase 2 results to build Phase 3 from')

# ============================================================
# FINAL: Age range variations on top systems
# ============================================================
print()
print('='*60)
print('PHASE 4: IPO Age Range Sweep (top systems × age windows)')
print('='*60)

# Read all results to find absolute top 3
all_results = []
if os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV) as f:
        for row in csv.DictReader(f):
            if int(row['total_trades']) >= 5:
                all_results.append(row)

all_results.sort(key=lambda x: float(x['expectancy']), reverse=True)
top3_all = all_results[:3]

phase4_configs = []
for best_row in top3_all:
    best_label = best_row['label']
    # Reconstruct full params
    full_params = {}
    # Parse label backwards for ATH and VA if present
    parts_left = best_label
    vol_avg = 20
    ath_lookback = 5
    vol_mult = 1.5

    if '_VA' in parts_left:
        va_idx = parts_left.rfind('_VA')
        vol_avg = int(parts_left[va_idx+3:])
        parts_left = parts_left[:va_idx]
    if '_ATH' in parts_left:
        ath_idx = parts_left.rfind('_ATH')
        ath_lookback = int(parts_left[ath_idx+4:])
        parts_left = parts_left[:ath_idx]
    if '_V' in parts_left:
        v_idx = parts_left.rfind('_V')
        try:
            vol_mult = float(parts_left[v_idx+2:])
            parts_left = parts_left[:v_idx]
        except ValueError:
            pass  # Not a vol suffix

    exit_params = label_to_params(parts_left)
    full_params = {**exit_params, 'vol_multiplier': vol_mult, 'ath_lookback_days': ath_lookback,
                   'vol_avg_period': vol_avg}

    for min_age, max_age in [(5, 30), (5, 50), (10, 30), (10, 50), (10, 75), (10, 100), (15, 60), (20, 75)]:
        lbl = f'{best_label}_AGE{min_age}_{max_age}'
        phase4_configs.append((
            lbl,
            {**full_params, 'min_ipo_age_days': min_age, 'max_ipo_age_days': max_age}
        ))

if phase4_configs:
    print(f'Phase 4: {len(phase4_configs)} configs')
    for label, params in phase4_configs:
        run_config('P4', label, params)
else:
    print('No results to build Phase 4 from')

# ============================================================
# SUMMARY
# ============================================================
print()
print('='*60)
print('FINAL RESULTS SUMMARY')
print('='*60)

final_results = []
if os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV) as f:
        for row in csv.DictReader(f):
            if int(row['total_trades']) >= 3:
                final_results.append(row)

# Sort by different criteria
print(f'\nTotal configs tested: {len(final_results)}')

print('\n--- TOP 10 by Expectancy ---')
by_exp = sorted(final_results, key=lambda x: float(x['expectancy']), reverse=True)[:10]
for i, r in enumerate(by_exp, 1):
    print(f'{i}. {r["label"]}: Exp={r["expectancy"]} Win={r["win_rate"]}% PF={r["profit_factor"]} '
          f'Trades={r["total_trades"]} CAGR={r["cagr"]}% AvgHold={r["avg_hold_days"]}d')

print('\n--- TOP 10 by Profit Factor (min 10 trades) ---')
pf_results = [r for r in final_results if int(r['total_trades']) >= 10]
by_pf = sorted(pf_results, key=lambda x: float(x['profit_factor']), reverse=True)[:10]
for i, r in enumerate(by_pf, 1):
    print(f'{i}. {r["label"]}: PF={r["profit_factor"]} Win={r["win_rate"]}% Exp={r["expectancy"]} '
          f'Trades={r["total_trades"]} AvgWin={r["avg_winner_pct"]}% AvgLoss={r["avg_loser_pct"]}%')

print('\n--- TOP 10 by Win Rate (min 10 trades) ---')
by_wr = sorted(pf_results, key=lambda x: float(x['win_rate']), reverse=True)[:10]
for i, r in enumerate(by_wr, 1):
    print(f'{i}. {r["label"]}: Win={r["win_rate"]}% Trades={r["total_trades"]} PF={r["profit_factor"]} '
          f'Exp={r["expectancy"]} AvgRet={r["avg_return_pct"]}%')

print(f'\nResults saved to: {OUTPUT_CSV}')
print('SWEEP COMPLETE.')
