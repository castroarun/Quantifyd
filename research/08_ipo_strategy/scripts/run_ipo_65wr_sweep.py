"""
IPO Strategy — 65%+ Win Rate Sweep
Tests entry quality filters, hybrid exits, and filter combinations
on the expanded 1151-stock universe to find systems with >= 65% win rate.

Phase A: Individual entry filters (on best base configs)
Phase B: Hybrid exit strategies
Phase C: Combined filter stacking
"""
import sys, os, csv, time, logging
import numpy as np

logging.disable(logging.WARNING)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from services.ipo_strategy import IPOStrategyBacktester, IPOConfig

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ipo_65wr_sweep.csv')
FIELDNAMES = [
    'label', 'exit_strategy', 'exit_params', 'entry_filters',
    'vol_multiplier', 'ath_lookback', 'vol_avg_period',
    'total_trades', 'winners', 'losers', 'win_rate',
    'avg_return_pct', 'avg_winner_pct', 'avg_loser_pct',
    'max_return_pct', 'max_loss_pct', 'profit_factor', 'expectancy',
    'total_pnl', 'final_capital', 'total_return_pct',
    'avg_hold_days', 'median_return_pct', 'exit_breakdown',
]

done = set()
if os.path.exists(OUTPUT_CSV):
    with open(OUTPUT_CSV) as f:
        done = {row['label'] for row in csv.DictReader(f)}
    if done:
        print(f'Skipping {len(done)} already-completed configs')
else:
    with open(OUTPUT_CSV, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

# Preload data ONCE (takes ~30s, saves ~40s per config)
print('Preloading IPO stock data...', end=' ', flush=True)
t_load = time.time()
PRELOADED = IPOStrategyBacktester.preload_data(ipo_start_year=2015)
print(f'{len(PRELOADED)} stocks loaded in {time.time()-t_load:.1f}s')


def run_config(label, params, filter_desc=''):
    if label in done:
        return None

    t0 = time.time()
    cfg = IPOConfig(**params)
    bt = IPOStrategyBacktester(cfg, preloaded_data=PRELOADED)
    r = bt.run()
    elapsed = time.time() - t0

    exit_counts = {}
    for t in r.trades:
        exit_counts[t.exit_reason] = exit_counts.get(t.exit_reason, 0) + 1
    exit_breakdown = '; '.join(f'{k}={v}' for k, v in sorted(exit_counts.items()))

    returns = [t.return_pct for t in r.trades]
    median_ret = float(np.median(returns)) if returns else 0

    row = {
        'label': label,
        'exit_strategy': params.get('exit_strategy', params.get('hybrid_exit', 'hybrid')),
        'exit_params': _describe_exit(params),
        'entry_filters': filter_desc,
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

    wr_marker = ' *** 65%+ ***' if r.win_rate >= 65 else ''
    print(f'  [{label}] {elapsed:.1f}s | T={r.total_trades} W={r.win_rate}% PF={r.profit_factor} '
          f'Exp={r.expectancy} Med={median_ret:.1f}% AvgHold={r.avg_hold_days}d{wr_marker}')
    sys.stdout.flush()
    return r


def _describe_exit(params):
    strat = params.get('exit_strategy', '')
    hybrid = params.get('hybrid_exit', 'none')

    if hybrid != 'none':
        return f"Hybrid={hybrid},T={params.get('hybrid_target_pct',15)}%,Trail={params.get('hybrid_trail_pct',10)}%,SL={params.get('hybrid_sl_pct',7)}%"

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
    return strat


# ============================================================
# BASE CONFIG: Best configs from Phase 3 (expanded universe)
# ============================================================
# FIX_T30_SL15 / V2.5 / ATH10 / VA30: 53.6% WR, PF 2.02 (highest WR)
# TRAIL_8PCT / V2.5 / ATH10 / VA30: 49.2% WR, PF 2.81 (highest PF)
# FIX_T15_SL7 / V2.5 / ATH7 / VA30: 47.1% WR, PF 1.79

base_best = dict(ipo_start_year=2015, vol_multiplier=2.5, ath_lookback_days=10, vol_avg_period=30)
base_best7 = dict(ipo_start_year=2015, vol_multiplier=2.5, ath_lookback_days=7, vol_avg_period=30)

configs = []

# ============================================================
# PHASE A: Individual Entry Filters
# Each filter tested on FIX_T30_SL15 (best WR base) and FIX_T15_SL7 (tighter target)
# ============================================================
print('=== PHASE A: Individual Entry Filters ===')

exit_bases = [
    ('FIX_T30_SL15', {'exit_strategy': 'fixed_target', 'target_pct': 30, 'stop_loss_pct': 15}),
    ('FIX_T15_SL7', {'exit_strategy': 'fixed_target', 'target_pct': 15, 'stop_loss_pct': 7}),
    ('FIX_T20_SL10', {'exit_strategy': 'fixed_target', 'target_pct': 20, 'stop_loss_pct': 10}),
    ('TRAIL_8PCT', {'exit_strategy': 'trailing_sl', 'trail_pct': 8}),
]

# A1: RSI filter (use period=7 for IPO stocks — enough data by day 10)
for exit_name, exit_p in exit_bases:
    for rsi_min in [50, 55, 60, 65]:
        for rsi_period in [7, 10]:
            label = f'A_RSI{rsi_period}p{rsi_min}_{exit_name}'
            params = {**base_best, **exit_p, 'use_rsi_filter': True, 'rsi_min': rsi_min, 'rsi_period': rsi_period}
            configs.append((label, params, f'RSI({rsi_period})>{rsi_min}'))

# A2: EMA filter
for exit_name, exit_p in exit_bases:
    for ema in [5, 7]:
        label = f'A_EMA{ema}_{exit_name}'
        params = {**base_best, **exit_p, 'use_ema_filter': True, 'ema_filter_period': ema}
        configs.append((label, params, f'Close>EMA{ema}'))

# A3: ADX filter (use period=7 for IPO)
for exit_name, exit_p in exit_bases:
    for adx_min in [15, 20, 25]:
        label = f'A_ADX7p{adx_min}_{exit_name}'
        params = {**base_best, **exit_p, 'use_adx_filter': True, 'adx_min': adx_min, 'adx_period': 7}
        configs.append((label, params, f'ADX(7)>{adx_min}'))

# A4: Min breakout %
for exit_name, exit_p in exit_bases:
    for bp in [1, 2, 3, 5]:
        label = f'A_BRK{bp}PCT_{exit_name}'
        params = {**base_best, **exit_p, 'min_breakout_pct': bp}
        configs.append((label, params, f'Breakout>{bp}%'))

# A5: Listing gain filter
for exit_name, exit_p in exit_bases:
    label = f'A_LISTGAIN_{exit_name}'
    params = {**base_best, **exit_p, 'require_listing_gain': True}
    configs.append((label, params, 'ListingGain'))

# A6: Gap-up filter
for exit_name, exit_p in exit_bases:
    label = f'A_GAPUP_{exit_name}'
    params = {**base_best, **exit_p, 'require_gap_up': True}
    configs.append((label, params, 'GapUp'))

# A7: Above avg price filter
for exit_name, exit_p in exit_bases:
    label = f'A_ABVAVG_{exit_name}'
    params = {**base_best, **exit_p, 'require_above_avg_price': True}
    configs.append((label, params, 'AboveAvgPrice'))

# A8: Consecutive close (2 days)
for exit_name, exit_p in exit_bases:
    label = f'A_CONSEC2_{exit_name}'
    params = {**base_best, **exit_p, 'require_consec_close': 2}
    configs.append((label, params, 'ConsecClose=2'))

# A9: MFI filter (period 7 for IPO)
for exit_name, exit_p in exit_bases:
    for mfi_min in [50, 60]:
        label = f'A_MFI7p{mfi_min}_{exit_name}'
        params = {**base_best, **exit_p, 'use_mfi_filter': True, 'mfi_min': mfi_min, 'mfi_period': 7}
        configs.append((label, params, f'MFI(7)>{mfi_min}'))

# ============================================================
# PHASE B: Hybrid Exit Strategies
# ============================================================
print('=== PHASE B: Hybrid Exit Strategies ===')

# B1: Target + Trail (best of both worlds)
for target in [15, 20, 25]:
    for trail in [8, 10, 12]:
        for sl in [5, 7, 10]:
            label = f'B_HYBRID_T{target}_TR{trail}_SL{sl}'
            params = {
                **base_best,
                'hybrid_exit': 'target_or_trail',
                'hybrid_target_pct': target,
                'hybrid_trail_pct': trail,
                'hybrid_sl_pct': sl,
                'max_hold_days': 0,  # No time cap
            }
            configs.append((label, params, 'Hybrid:Target+Trail'))

# B2: Target + Trail with time cap
for target in [15, 20]:
    for trail in [8, 10]:
        for days in [30, 45, 60]:
            label = f'B_HYBRID_T{target}_TR{trail}_D{days}'
            params = {
                **base_best,
                'hybrid_exit': 'target_or_trail',
                'hybrid_target_pct': target,
                'hybrid_trail_pct': trail,
                'hybrid_sl_pct': 7,
                'max_hold_days': days,
            }
            configs.append((label, params, f'Hybrid:T+TR+TimeCap{days}d'))

# B3: Target + Breakeven stop
for target in [15, 20, 25, 30]:
    for be_trigger in [3, 5, 7]:
        for sl in [5, 7, 10]:
            label = f'B_TGTBE_T{target}_BE{be_trigger}_SL{sl}'
            params = {
                **base_best,
                'hybrid_exit': 'target_with_breakeven',
                'hybrid_target_pct': target,
                'breakeven_trigger_pct': be_trigger,
                'hybrid_sl_pct': sl,
                'use_breakeven_stop': True,
                'max_hold_days': 0,
            }
            configs.append((label, params, f'Hybrid:Target+BE@{be_trigger}%'))

# B4: Breakeven stop on existing exits
for exit_name, exit_p in exit_bases[:2]:
    for be_trigger in [3, 5, 7]:
        label = f'B_BE{be_trigger}_{exit_name}'
        params = {**base_best, **exit_p, 'use_breakeven_stop': True, 'breakeven_trigger_pct': be_trigger}
        configs.append((label, params, f'BE@{be_trigger}%'))

# B5: Trail tightening
for trail_start in [12, 15]:
    for trail_tight in [6, 8]:
        for tight_days in [15, 20, 25]:
            label = f'B_TIGHTEN_T{trail_start}_TO{trail_tight}_D{tight_days}'
            params = {
                **base_best,
                'exit_strategy': 'trailing_sl',
                'trail_pct': trail_start,
                'use_time_tightening': True,
                'tighten_after_days': tight_days,
                'tightened_trail_pct': trail_tight,
            }
            configs.append((label, params, f'TrailTighten:{trail_start}%->{trail_tight}%@{tight_days}d'))

# ============================================================
# PHASE C: Tighter target configs (10-15% target, 5% SL)
# ============================================================
print('=== PHASE C: Tight Target Sweep ===')

for target in [8, 10, 12, 15]:
    for sl in [3, 5, 7]:
        for ath in [5, 7, 10]:
            label = f'C_TIGHT_T{target}_SL{sl}_ATH{ath}'
            params = {
                **base_best,
                'exit_strategy': 'fixed_target',
                'target_pct': target,
                'stop_loss_pct': sl,
                'ath_lookback_days': ath,
            }
            configs.append((label, params, f'TightTarget'))

# ============================================================
# PHASE D: Combined Filters (stack best individual filters)
# ============================================================
print('=== PHASE D: Combined Filter Stacking ===')

# D1: RSI + Listing gain
for rsi in [50, 55, 60]:
    for exit_name, exit_p in exit_bases:
        label = f'D_RSI7p{rsi}_LISTGAIN_{exit_name}'
        params = {**base_best, **exit_p, 'use_rsi_filter': True, 'rsi_min': rsi,
                  'rsi_period': 7, 'require_listing_gain': True}
        configs.append((label, params, f'RSI7>{rsi}+ListGain'))

# D2: RSI + Breakout %
for rsi in [50, 55, 60]:
    for bp in [2, 3]:
        for exit_name, exit_p in exit_bases[:2]:
            label = f'D_RSI7p{rsi}_BRK{bp}_{exit_name}'
            params = {**base_best, **exit_p, 'use_rsi_filter': True, 'rsi_min': rsi,
                      'rsi_period': 7, 'min_breakout_pct': bp}
            configs.append((label, params, f'RSI7>{rsi}+BRK>{bp}%'))

# D3: RSI + Above avg price
for rsi in [50, 55]:
    for exit_name, exit_p in exit_bases[:2]:
        label = f'D_RSI7p{rsi}_ABVAVG_{exit_name}'
        params = {**base_best, **exit_p, 'use_rsi_filter': True, 'rsi_min': rsi,
                  'rsi_period': 7, 'require_above_avg_price': True}
        configs.append((label, params, f'RSI7>{rsi}+AbvAvg'))

# D4: Listing gain + Breakout %
for bp in [2, 3, 5]:
    for exit_name, exit_p in exit_bases[:2]:
        label = f'D_LISTGAIN_BRK{bp}_{exit_name}'
        params = {**base_best, **exit_p, 'require_listing_gain': True, 'min_breakout_pct': bp}
        configs.append((label, params, f'ListGain+BRK>{bp}%'))

# D5: RSI + Listing gain + Hybrid exit
for rsi in [50, 55, 60]:
    for target in [15, 20]:
        for trail in [8, 10]:
            label = f'D_RSI7p{rsi}_LISTGAIN_HYB_T{target}_TR{trail}'
            params = {
                **base_best,
                'use_rsi_filter': True, 'rsi_min': rsi, 'rsi_period': 7,
                'require_listing_gain': True,
                'hybrid_exit': 'target_or_trail',
                'hybrid_target_pct': target,
                'hybrid_trail_pct': trail,
                'hybrid_sl_pct': 7,
                'max_hold_days': 0,
            }
            configs.append((label, params, f'RSI7>{rsi}+ListGain+Hybrid'))

# D6: Triple filter: RSI + Listing gain + Breakout %
for rsi in [50, 55, 60]:
    for bp in [2, 3]:
        for exit_name, exit_p in exit_bases[:2]:
            label = f'D_RSI7p{rsi}_LG_BRK{bp}_{exit_name}'
            params = {**base_best, **exit_p, 'use_rsi_filter': True, 'rsi_min': rsi,
                      'rsi_period': 7, 'require_listing_gain': True, 'min_breakout_pct': bp}
            configs.append((label, params, f'RSI7>{rsi}+LG+BRK>{bp}%'))

# D7: RSI + EMA + best exits
for rsi in [50, 55]:
    for exit_name, exit_p in exit_bases[:2]:
        label = f'D_RSI7p{rsi}_EMA5_{exit_name}'
        params = {**base_best, **exit_p, 'use_rsi_filter': True, 'rsi_min': rsi,
                  'rsi_period': 7, 'use_ema_filter': True, 'ema_filter_period': 5}
        configs.append((label, params, f'RSI7>{rsi}+EMA5'))

# D8: Listing gain + Hybrid + Breakeven
for bp in [0, 2]:
    for target in [15, 20]:
        for be in [3, 5]:
            label = f'D_LG_BRK{bp}_TGTBE_T{target}_BE{be}'
            params = {
                **base_best,
                'require_listing_gain': True,
                'min_breakout_pct': bp,
                'hybrid_exit': 'target_with_breakeven',
                'hybrid_target_pct': target,
                'breakeven_trigger_pct': be,
                'hybrid_sl_pct': 7,
                'use_breakeven_stop': True,
                'max_hold_days': 0,
            }
            configs.append((label, params, f'LG+BRK>{bp}%+TgtBE'))

# D9: ADX + RSI + best exits
for adx in [15, 20]:
    for rsi in [50, 55]:
        for exit_name, exit_p in exit_bases[:2]:
            label = f'D_ADX7p{adx}_RSI7p{rsi}_{exit_name}'
            params = {**base_best, **exit_p, 'use_adx_filter': True, 'adx_min': adx, 'adx_period': 7,
                      'use_rsi_filter': True, 'rsi_min': rsi, 'rsi_period': 7}
            configs.append((label, params, f'ADX7>{adx}+RSI7>{rsi}'))

# D10: Kitchen sink — all top filters together
for exit_name, exit_p in exit_bases[:2]:
    label = f'D_ALL_RSI7p55_LG_BRK2_EMA5_{exit_name}'
    params = {**base_best, **exit_p,
              'use_rsi_filter': True, 'rsi_min': 55, 'rsi_period': 7,
              'require_listing_gain': True,
              'min_breakout_pct': 2,
              'use_ema_filter': True, 'ema_filter_period': 5}
    configs.append((label, params, 'RSI7>55+LG+BRK2+EMA5'))

# D11: Narrower IPO window (max 30 days instead of 50)
for exit_name, exit_p in exit_bases[:2]:
    for max_age in [25, 30, 35]:
        label = f'D_MAXAGE{max_age}_{exit_name}'
        params = {**base_best, **exit_p, 'max_ipo_age_days': max_age}
        configs.append((label, params, f'MaxAge={max_age}d'))

# D12: Narrower window + filters
for max_age in [25, 30]:
    for exit_name, exit_p in exit_bases[:2]:
        label = f'D_MAXAGE{max_age}_RSI7p55_LG_{exit_name}'
        params = {**base_best, **exit_p, 'max_ipo_age_days': max_age,
                  'use_rsi_filter': True, 'rsi_min': 55, 'rsi_period': 7,
                  'require_listing_gain': True}
        configs.append((label, params, f'MaxAge{max_age}+RSI7>55+LG'))

# D13: Strong breakout + Listing gain + tight exit
for bp in [3, 5, 7]:
    for exit_name, exit_p in exit_bases[:3]:
        label = f'D_BRK{bp}_LG_{exit_name}'
        params = {**base_best, **exit_p, 'min_breakout_pct': bp, 'require_listing_gain': True}
        configs.append((label, params, f'BRK>{bp}%+LG'))

# D14: RSI + Listing gain + Breakout + Hybrid breakeven
for rsi in [55, 60]:
    for bp in [2, 3]:
        for be in [3, 5]:
            label = f'D_RSI7p{rsi}_LG_BRK{bp}_TGTBE20_BE{be}'
            params = {
                **base_best,
                'use_rsi_filter': True, 'rsi_min': rsi, 'rsi_period': 7,
                'require_listing_gain': True, 'min_breakout_pct': bp,
                'hybrid_exit': 'target_with_breakeven',
                'hybrid_target_pct': 20, 'breakeven_trigger_pct': be,
                'hybrid_sl_pct': 7, 'use_breakeven_stop': True, 'max_hold_days': 0,
            }
            configs.append((label, params, f'RSI7>{rsi}+LG+BRK>{bp}%+TgtBE'))

# D15: AboveAvgPrice + Listing gain + exits
for exit_name, exit_p in exit_bases[:3]:
    label = f'D_ABVAVG_LG_{exit_name}'
    params = {**base_best, **exit_p, 'require_above_avg_price': True, 'require_listing_gain': True}
    configs.append((label, params, 'AbvAvg+LG'))

# D16: ConsecClose + Listing gain
for exit_name, exit_p in exit_bases[:3]:
    label = f'D_CONSEC2_LG_{exit_name}'
    params = {**base_best, **exit_p, 'require_consec_close': 2, 'require_listing_gain': True}
    configs.append((label, params, 'Consec2+LG'))

# D17: GapUp + Listing gain
for exit_name, exit_p in exit_bases[:3]:
    label = f'D_GAPUP_LG_{exit_name}'
    params = {**base_best, **exit_p, 'require_gap_up': True, 'require_listing_gain': True}
    configs.append((label, params, 'GapUp+LG'))


# ============================================================
# RUN
# ============================================================
print(f'\n=== IPO 65%+ WR Sweep: {len(configs)} configs ===')
print(f'Phase A: Individual entry filters')
print(f'Phase B: Hybrid exit strategies')
print(f'Phase C: Tight target sweep')
print(f'Phase D: Combined filter stacking')

for i, (label, params, fdesc) in enumerate(configs, 1):
    if i % 25 == 1:
        print(f'\n--- Progress: {i}/{len(configs)} ---')
    run_config(label, params, fdesc)

# ============================================================
# ANALYSIS
# ============================================================
print('\n' + '=' * 70)
print('ANALYSIS — 65%+ Win Rate IPO Strategies')
print('=' * 70)

results = []
with open(OUTPUT_CSV) as f:
    for row in csv.DictReader(f):
        if int(row['total_trades']) >= 15:
            results.append(row)

wr65 = [r for r in results if float(r['win_rate']) >= 65]
wr60 = [r for r in results if float(r['win_rate']) >= 60]
practical = [r for r in results if float(r['profit_factor']) >= 1.0]

print(f'\nTotal configs with 15+ trades: {len(results)}')
print(f'Configs with WR >= 65%: {len(wr65)}')
print(f'Configs with WR >= 60%: {len(wr60)}')
print(f'Configs with PF >= 1.0: {len(practical)}')


def score_wr(r):
    """Score biased toward win rate."""
    wr = float(r['win_rate']) / 100
    pf = min(float(r['profit_factor']), 10) / 10
    exp = max(float(r['expectancy']), -10) / 20 + 0.5
    trades = min(int(r['total_trades']), 200) / 200
    return wr * 0.40 + pf * 0.25 + exp * 0.20 + trades * 0.15


for r in results:
    r['_score'] = score_wr(r)

# Top configs by win rate
if wr65:
    print('\n--- CONFIGS WITH 65%+ WIN RATE ---')
    wr65.sort(key=lambda x: float(x['win_rate']), reverse=True)
    print(f'{"Rank":<5} {"Label":<50} {"Trades":>6} {"Win%":>6} {"PF":>6} {"Exp":>8} {"MedRet%":>8} {"AvgHold":>8} {"Filters"}')
    for i, r in enumerate(wr65[:30], 1):
        print(f'{i:<5} {r["label"]:<50} {r["total_trades"]:>6} {r["win_rate"]:>6} {r["profit_factor"]:>6} '
              f'{r["expectancy"]:>8} {r["median_return_pct"]:>8} {r["avg_hold_days"]:>8} {r.get("entry_filters","")}')
else:
    print('\n--- NO CONFIGS HIT 65% WR ---')

if wr60:
    print(f'\n--- TOP 20 WITH 60%+ WIN RATE ---')
    wr60.sort(key=lambda x: x['_score'], reverse=True)
    print(f'{"Rank":<5} {"Label":<50} {"Trades":>6} {"Win%":>6} {"PF":>6} {"Exp":>8} {"Filters"}')
    for i, r in enumerate(wr60[:20], 1):
        print(f'{i:<5} {r["label"]:<50} {r["total_trades"]:>6} {r["win_rate"]:>6} {r["profit_factor"]:>6} '
              f'{r["expectancy"]:>8} {r.get("entry_filters","")}')

# Top by composite score
results.sort(key=lambda x: x['_score'], reverse=True)
print(f'\n--- TOP 25 by Composite Score (WR-biased) ---')
print(f'{"Rank":<5} {"Label":<50} {"Trades":>6} {"Win%":>6} {"PF":>6} {"Exp":>8} {"MedRet%":>8} {"Filters"}')
for i, r in enumerate(results[:25], 1):
    print(f'{i:<5} {r["label"]:<50} {r["total_trades"]:>6} {r["win_rate"]:>6} {r["profit_factor"]:>6} '
          f'{r["expectancy"]:>8} {r["median_return_pct"]:>8} {r.get("entry_filters","")}')

# Filter impact analysis
print('\n--- ENTRY FILTER IMPACT (avg WR delta from base ~47-53%) ---')
filter_types = {}
for r in results:
    filt = r.get('entry_filters', '')
    if filt and filt not in ('Hybrid:Target+Trail', 'TightTarget'):
        base_key = filt.split('+')[0] if '+' in filt else filt
        if base_key not in filter_types:
            filter_types[base_key] = []
        filter_types[base_key].append(float(r['win_rate']))

for filt, wrs in sorted(filter_types.items()):
    avg_wr = np.mean(wrs)
    max_wr = max(wrs)
    print(f'  {filt:<25}: AvgWR={avg_wr:.1f}% MaxWR={max_wr:.1f}% ({len(wrs)} configs)')

# Best hybrid exit
print('\n--- BEST HYBRID EXITS ---')
hybrids = [r for r in results if 'hybrid' in r.get('exit_strategy', '').lower()
           or 'hybrid' in r.get('entry_filters', '').lower()
           or r['label'].startswith('B_')]
if hybrids:
    hybrids.sort(key=lambda x: float(x['win_rate']), reverse=True)
    for i, r in enumerate(hybrids[:15], 1):
        print(f'{i}. {r["label"]}: WR={r["win_rate"]}% PF={r["profit_factor"]} '
              f'T={r["total_trades"]} Exp={r["expectancy"]} [{r["exit_breakdown"]}]')

print(f'\nResults saved to: {OUTPUT_CSV}')
print('DONE.')
