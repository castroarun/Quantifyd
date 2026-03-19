"""
IPO Strategy Research — Practical Short-Term Sweep
Focus on strategies with avg hold < 200 days (real IPO breakout trading).
Sweep: Top 8 short-term exits × 5 volume multipliers × 5 ATH lookbacks = 200 configs
"""
import sys, os, csv, time, logging
import numpy as np

logging.disable(logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.ipo_strategy import IPOStrategyBacktester, IPOConfig

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ipo_sweep_practical.csv')
FIELDNAMES = [
    'label', 'exit_strategy', 'exit_params',
    'vol_multiplier', 'ath_lookback', 'vol_avg_period',
    'total_trades', 'winners', 'losers', 'win_rate',
    'avg_return_pct', 'avg_winner_pct', 'avg_loser_pct',
    'max_return_pct', 'max_loss_pct', 'profit_factor', 'expectancy',
    'total_pnl', 'final_capital', 'total_return_pct',
    'avg_hold_days', 'median_return_pct', 'exit_breakdown',
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


def run_config(label, params):
    """Run a single config and write to CSV."""
    if label in done:
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

    # Compute median return (more robust than avg with outliers)
    returns = [t.return_pct for t in r.trades]
    median_ret = float(np.median(returns)) if returns else 0

    row = {
        'label': label,
        'exit_strategy': params.get('exit_strategy', 'supertrend'),
        'exit_params': _describe_exit(params),
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
        'avg_hold_days': r.avg_hold_days,
        'median_return_pct': round(median_ret, 2),
        'exit_breakdown': exit_breakdown,
    }

    with open(OUTPUT_CSV, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

    print(f'  [{label}] {elapsed:.1f}s | T={r.total_trades} W={r.win_rate}% PF={r.profit_factor} '
          f'Exp={r.expectancy} Med={median_ret:.1f}% AvgHold={r.avg_hold_days}d')
    sys.stdout.flush()
    return r


def _describe_exit(params):
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
# Best short-term exit strategies from Phase 1
# (ranked by Sharpe with avg hold < 200 days)
# ============================================================
exit_strategies = [
    # (name, params) — all had 44 trades with vol=1.5, ATH=5
    ('FIX_T20_SL10',  {'exit_strategy': 'fixed_target', 'target_pct': 20, 'stop_loss_pct': 10}),  # Sharpe=3.63, W=59%
    ('FIX_T40_SL15',  {'exit_strategy': 'fixed_target', 'target_pct': 40, 'stop_loss_pct': 15}),  # Sharpe=3.54, W=55%
    ('FIX_T30_SL15',  {'exit_strategy': 'fixed_target', 'target_pct': 30, 'stop_loss_pct': 15}),  # Sharpe=3.48, W=59%
    ('FIX_T25_SL10',  {'exit_strategy': 'fixed_target', 'target_pct': 25, 'stop_loss_pct': 10}),  # Sharpe=3.42, W=55%
    ('FIX_T15_SL7',   {'exit_strategy': 'fixed_target', 'target_pct': 15, 'stop_loss_pct': 7}),   # Sharpe=3.31, W=57%
    ('FIX_T60_SL20',  {'exit_strategy': 'fixed_target', 'target_pct': 60, 'stop_loss_pct': 20}),  # Sharpe=3.08, W=48%
    ('TIME_60D_SL15', {'exit_strategy': 'time_exit', 'max_hold_days': 60, 'stop_loss_pct': 15}),  # Sharpe=3.03, W=61%
    ('TIME_90D_SL15', {'exit_strategy': 'time_exit', 'max_hold_days': 90, 'stop_loss_pct': 15}),  # Sharpe=3.06, W=61%
    ('EMA_8_21',      {'exit_strategy': 'ema_cross', 'ema_fast': 8, 'ema_slow': 21}),             # Sharpe=2.30, W=50%
    ('EMA_10_30',     {'exit_strategy': 'ema_cross', 'ema_fast': 10, 'ema_slow': 30}),            # Sharpe=2.50, W=50%
    ('ATR_10_3',      {'exit_strategy': 'atr_trail', 'atr_trail_period': 10, 'atr_trail_mult': 3.0}),  # Sharpe=2.30
    ('TRAIL_8PCT',    {'exit_strategy': 'trailing_sl', 'trail_pct': 8}),                           # Sharpe=2.27, W=55%
    ('ST_14_2',       {'exit_strategy': 'supertrend', 'st_atr_period': 14, 'st_multiplier': 2.0}), # Sharpe=2.56, tighter ST
]

vol_multipliers = [1.0, 1.25, 1.5, 2.0, 2.5]
ath_lookbacks = [3, 5, 7, 10]
vol_avg_periods = [15, 20, 30]

configs = []
for exit_name, exit_params in exit_strategies:
    for vol in vol_multipliers:
        for ath in ath_lookbacks:
            for va in vol_avg_periods:
                label = f'{exit_name}_V{vol}_ATH{ath}_VA{va}'
                params = {
                    **exit_params,
                    'vol_multiplier': vol,
                    'ath_lookback_days': ath,
                    'vol_avg_period': va,
                }
                configs.append((label, params))

print(f'=== IPO Practical Strategy Sweep: {len(configs)} configs ===')
print(f'Exits: {len(exit_strategies)}, Vol: {len(vol_multipliers)}, ATH: {len(ath_lookbacks)}, VA: {len(vol_avg_periods)}')

for i, (label, params) in enumerate(configs, 1):
    if i % 50 == 1:
        print(f'\n--- Progress: {i}/{len(configs)} ---')
    run_config(label, params)

# ============================================================
# ANALYSIS
# ============================================================
print('\n' + '='*70)
print('ANALYSIS — Practical IPO Strategies (short hold periods)')
print('='*70)

results = []
with open(OUTPUT_CSV) as f:
    for row in csv.DictReader(f):
        if int(row['total_trades']) >= 5:
            results.append(row)

# Filter: Practical — avg hold < 200 days, PF >= 1.5
practical = [r for r in results if float(r['avg_hold_days']) < 200 and float(r['profit_factor']) >= 1.5]

print(f'\nTotal configs: {len(results)}, Practical (hold<200d, PF>=1.5): {len(practical)}')

# Composite score: normalize and weight
# Win Rate × 0.3 + PF × 0.3 + Median Return × 0.2 + (negative) MaxLoss × 0.2
def score(r):
    wr = float(r['win_rate']) / 100  # 0-1
    pf = min(float(r['profit_factor']), 10) / 10  # cap at 10, normalize to 0-1
    med = max(float(r['median_return_pct']), -30) / 60  # normalize to ~0-1
    ml = 1 + float(r['max_loss_pct']) / 30  # max_loss_pct is negative, higher is better
    trades = min(int(r['total_trades']), 50) / 50  # more trades = more reliable
    return wr * 0.25 + pf * 0.25 + med * 0.2 + ml * 0.15 + trades * 0.15

for r in practical:
    r['_score'] = score(r)

practical.sort(key=lambda x: x['_score'], reverse=True)

print('\n--- TOP 20 by Composite Score (Win% + PF + Median Ret + Max Loss + Trade Count) ---')
print(f'{"Rank":<5} {"Label":<35} {"Trades":>6} {"Win%":>6} {"PF":>6} {"Exp":>8} {"MedRet%":>8} {"AvgHold":>8} {"MaxLoss%":>9}')
for i, r in enumerate(practical[:20], 1):
    print(f'{i:<5} {r["label"]:<35} {r["total_trades"]:>6} {r["win_rate"]:>6} {r["profit_factor"]:>6} '
          f'{r["expectancy"]:>8} {r["median_return_pct"]:>8} {r["avg_hold_days"]:>8} {r["max_loss_pct"]:>9}')

print('\n--- TOP 10 by Win Rate (min 10 trades) ---')
min10 = [r for r in practical if int(r['total_trades']) >= 10]
by_wr = sorted(min10, key=lambda x: float(x['win_rate']), reverse=True)[:10]
for i, r in enumerate(by_wr, 1):
    print(f'{i}. {r["label"]}: Win={r["win_rate"]}% T={r["total_trades"]} PF={r["profit_factor"]} '
          f'MedRet={r["median_return_pct"]}% AvgHold={r["avg_hold_days"]}d Exits=[{r["exit_breakdown"]}]')

print('\n--- TOP 10 by Profit Factor (min 15 trades) ---')
min15 = [r for r in practical if int(r['total_trades']) >= 15]
by_pf = sorted(min15, key=lambda x: float(x['profit_factor']), reverse=True)[:10]
for i, r in enumerate(by_pf, 1):
    print(f'{i}. {r["label"]}: PF={r["profit_factor"]} Win={r["win_rate"]}% T={r["total_trades"]} '
          f'Exp={r["expectancy"]} AvgWin={r["avg_winner_pct"]}% AvgLoss={r["avg_loser_pct"]}%')

# Best per exit strategy
print('\n--- BEST CONFIG per Exit Strategy ---')
exit_types = set(r['exit_strategy'] for r in practical)
for et in sorted(exit_types):
    sub = [r for r in practical if r['exit_strategy'] == et and int(r['total_trades']) >= 10]
    if sub:
        best = max(sub, key=lambda x: x['_score'])
        print(f'  {et}: {best["label"]} — Win={best["win_rate"]}% PF={best["profit_factor"]} '
              f'T={best["total_trades"]} MedRet={best["median_return_pct"]}% AvgHold={best["avg_hold_days"]}d')

# Volume multiplier analysis (aggregate across strategies)
print('\n--- VOLUME MULTIPLIER IMPACT (avg across all exits) ---')
for vol in vol_multipliers:
    sub = [r for r in practical if abs(float(r['vol_multiplier']) - vol) < 0.01]
    if sub:
        avg_wr = np.mean([float(r['win_rate']) for r in sub])
        avg_pf = np.mean([float(r['profit_factor']) for r in sub])
        avg_t = np.mean([int(r['total_trades']) for r in sub])
        avg_med = np.mean([float(r['median_return_pct']) for r in sub])
        print(f'  Vol {vol}x: AvgWinRate={avg_wr:.1f}% AvgPF={avg_pf:.2f} AvgTrades={avg_t:.0f} AvgMedRet={avg_med:.1f}%')

# ATH lookback analysis
print('\n--- ATH LOOKBACK IMPACT (avg across all exits) ---')
for ath in ath_lookbacks:
    sub = [r for r in practical if int(r['ath_lookback']) == ath]
    if sub:
        avg_wr = np.mean([float(r['win_rate']) for r in sub])
        avg_pf = np.mean([float(r['profit_factor']) for r in sub])
        avg_t = np.mean([int(r['total_trades']) for r in sub])
        avg_med = np.mean([float(r['median_return_pct']) for r in sub])
        print(f'  ATH {ath}d: AvgWinRate={avg_wr:.1f}% AvgPF={avg_pf:.2f} AvgTrades={avg_t:.0f} AvgMedRet={avg_med:.1f}%')

print(f'\nResults saved to: {OUTPUT_CSV}')
print('DONE.')
