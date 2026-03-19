"""
Aggressive search for 65%+ win rate breakout strategies.
Tests tighter filter combinations than the initial study.
"""
import pandas as pd
import numpy as np
from itertools import combinations

df = pd.read_csv('breakout_analysis_full.csv')
df['date'] = pd.to_datetime(df['date'])
print(f"Loaded {len(df)} trades, {df['symbol'].nunique()} stocks")

def compute_calmar(subset):
    if len(subset) < 10:
        return None, None, None
    sorted_trades = subset.sort_values('date')
    dates = sorted_trades['date'].values
    returns = sorted_trades['trade_return'].values / 100
    equity = [100.0]
    for r in returns:
        equity.append(equity[-1] * (1 + r))
    equity = np.array(equity)
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max * 100
    max_dd = abs(drawdowns.min())
    years = (dates[-1] - dates[0]) / np.timedelta64(365, 'D')
    years = max(years, 0.5)
    cagr = ((equity[-1] / equity[0]) ** (1 / years) - 1) * 100
    calmar = round(cagr / max_dd, 2) if max_dd > 0.1 else 999
    return round(cagr, 1), round(max_dd, 1), calmar

def eval_filter(mask, label, min_trades=15):
    subset = df[mask].copy()
    n = len(subset)
    if n < min_trades:
        return None
    stopped = (subset['exit_reason'] == 'STOP').mean() * 100
    win_pct = (subset['trade_return'] > 0).mean() * 100
    avg_ret = subset['trade_return'].mean()
    med_ret = subset['trade_return'].median()
    wins = subset[subset['trade_return'] > 0]['trade_return']
    losses = subset[subset['trade_return'] <= 0]['trade_return']
    avg_gain = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    total_gains = wins.sum() if len(wins) > 0 else 0
    total_losses = abs(losses.sum()) if len(losses) > 0 else 0.01
    pf = round(total_gains / total_losses, 2)
    cagr, max_dd, calmar = compute_calmar(subset)
    return {
        'filter': label, 'trades': n, 'stop_pct': round(stopped, 1),
        'win_pct': round(win_pct, 1), 'avg_ret': round(avg_ret, 1),
        'med_ret': round(med_ret, 1), 'avg_gain': round(avg_gain, 1),
        'avg_loss': round(avg_loss, 1), 'profit_factor': pf,
        'cagr': cagr, 'max_dd': max_dd, 'calmar': calmar,
    }

# ============================================================
# PHASE 1: Systematic sweep of individual thresholds
# ============================================================
print("\n" + "="*80)
print("PHASE 1: Finding optimal individual thresholds")
print("="*80)

# RSI thresholds
print("\n--- RSI thresholds ---")
for thresh in [55, 60, 65, 70, 75, 80]:
    mask = df['rsi14'] >= thresh
    r = eval_filter(mask, f"RSI>={thresh}")
    if r:
        print(f"  RSI>={thresh}: {r['trades']:>5} trades, {r['win_pct']:>5.1f}% win, {r['avg_ret']:>+6.1f}%, PF={r['profit_factor']}")

# Volume thresholds
print("\n--- Volume ratio thresholds ---")
for thresh in [2, 3, 5, 7, 10, 15, 20]:
    mask = df['volume_ratio'] >= thresh
    r = eval_filter(mask, f"Vol>={thresh}x")
    if r:
        print(f"  Vol>={thresh}x: {r['trades']:>5} trades, {r['win_pct']:>5.1f}% win, {r['avg_ret']:>+6.1f}%, PF={r['profit_factor']}")

# Breakout % thresholds
print("\n--- Breakout % thresholds ---")
for thresh in [1, 2, 3, 5, 7, 10, 15]:
    mask = df['breakout_pct'] >= thresh
    r = eval_filter(mask, f"BO>={thresh}%")
    if r:
        print(f"  BO>={thresh}%: {r['trades']:>5} trades, {r['win_pct']:>5.1f}% win, {r['avg_ret']:>+6.1f}%, PF={r['profit_factor']}")

# ATH proximity thresholds
print("\n--- ATH proximity thresholds ---")
for thresh in [0.80, 0.85, 0.90, 0.95, 0.98]:
    mask = df['ath_proximity'] >= thresh
    r = eval_filter(mask, f"ATH>={thresh*100:.0f}%")
    if r:
        print(f"  ATH>={thresh*100:.0f}%: {r['trades']:>5} trades, {r['win_pct']:>5.1f}% win, {r['avg_ret']:>+6.1f}%, PF={r['profit_factor']}")

# Box height thresholds
print("\n--- Box height thresholds ---")
for lo, hi in [(5,10), (5,15), (10,20), (15,25), (5,12), (8,16)]:
    mask = df['box_height_pct'].between(lo, hi)
    r = eval_filter(mask, f"Box {lo}-{hi}%")
    if r:
        print(f"  Box {lo}-{hi}%: {r['trades']:>5} trades, {r['win_pct']:>5.1f}% win, {r['avg_ret']:>+6.1f}%, PF={r['profit_factor']}")

# Momentum thresholds
print("\n--- 20d Momentum thresholds ---")
for thresh in [0, 5, 10, 15, 20]:
    mask = df['mom_20d'] >= thresh
    r = eval_filter(mask, f"Mom20>={thresh}%")
    if r:
        print(f"  Mom20>={thresh}%: {r['trades']:>5} trades, {r['win_pct']:>5.1f}% win, {r['avg_ret']:>+6.1f}%, PF={r['profit_factor']}")

# Detector type
print("\n--- Detector type ---")
for det in ['darvas', 'flat']:
    mask = df['detector'] == det
    r = eval_filter(mask, det)
    if r:
        print(f"  {det}: {r['trades']:>5} trades, {r['win_pct']:>5.1f}% win, {r['avg_ret']:>+6.1f}%, PF={r['profit_factor']}")

# ============================================================
# PHASE 2: Aggressive 2-3 filter combos with tighter thresholds
# ============================================================
print("\n\n" + "="*80)
print("PHASE 2: Aggressive combinations targeting 65%+ win rate")
print("="*80)

# Define tighter filter building blocks
filters = {
    'RSI>=65': df['rsi14'] >= 65,
    'RSI>=70': df['rsi14'] >= 70,
    'RSI>=75': df['rsi14'] >= 75,
    'Vol>=3x': df['volume_ratio'] >= 3,
    'Vol>=5x': df['volume_ratio'] >= 5,
    'Vol>=7x': df['volume_ratio'] >= 7,
    'Vol>=10x': df['volume_ratio'] >= 10,
    'BO>=3%': df['breakout_pct'] >= 3,
    'BO>=5%': df['breakout_pct'] >= 5,
    'BO>=7%': df['breakout_pct'] >= 7,
    'BO>=10%': df['breakout_pct'] >= 10,
    'ATH>=85%': df['ath_proximity'] >= 0.85,
    'ATH>=90%': df['ath_proximity'] >= 0.90,
    'ATH>=95%': df['ath_proximity'] >= 0.95,
    'EMA20>50': df['ema20_above_50'] == True,
    'AbvEMA200': df['above_ema200'] == True,
    'Mom20>=10': df['mom_20d'] >= 10,
    'Mom20>=15': df['mom_20d'] >= 15,
    'Mom60>=10': df['mom_60d'] >= 10,
    'Mom60>=20': df['mom_60d'] >= 20,
    'Box5-15%': df['box_height_pct'].between(5, 15),
    'Box5-12%': df['box_height_pct'].between(5, 12),
    'Darvas': df['detector'] == 'darvas',
    'VolTrend>1.2': df['vol_trend'] >= 1.2,
}

# Test all 2-filter combinations
print("\n--- All 2-filter combos with win% >= 55% ---")
results_2 = []
filter_names = list(filters.keys())
for i in range(len(filter_names)):
    for j in range(i+1, len(filter_names)):
        name = f"{filter_names[i]} + {filter_names[j]}"
        mask = filters[filter_names[i]] & filters[filter_names[j]]
        r = eval_filter(mask, name, min_trades=15)
        if r and r['win_pct'] >= 55:
            results_2.append(r)

results_2.sort(key=lambda x: x['win_pct'], reverse=True)
print(f"Found {len(results_2)} combos with >=55% win rate")
for r in results_2[:30]:
    cal_str = f"Calmar={r['calmar']}" if r['calmar'] is not None else "Calmar=N/A"
    print(f"  {r['filter']:<45} n={r['trades']:>4}  win={r['win_pct']:>5.1f}%  avg={r['avg_ret']:>+6.1f}%  PF={r['profit_factor']:>5.2f}  {cal_str}")

# Test all 3-filter combinations
print("\n\n--- All 3-filter combos with win% >= 58% ---")
results_3 = []
for i in range(len(filter_names)):
    for j in range(i+1, len(filter_names)):
        for k in range(j+1, len(filter_names)):
            name = f"{filter_names[i]} + {filter_names[j]} + {filter_names[k]}"
            mask = filters[filter_names[i]] & filters[filter_names[j]] & filters[filter_names[k]]
            r = eval_filter(mask, name, min_trades=15)
            if r and r['win_pct'] >= 58:
                results_3.append(r)

results_3.sort(key=lambda x: x['win_pct'], reverse=True)
print(f"Found {len(results_3)} combos with >=58% win rate")
for r in results_3[:40]:
    cal_str = f"Calmar={r['calmar']}" if r['calmar'] is not None else "Calmar=N/A"
    print(f"  {r['filter']:<60} n={r['trades']:>4}  win={r['win_pct']:>5.1f}%  avg={r['avg_ret']:>+6.1f}%  PF={r['profit_factor']:>5.2f}  {cal_str}")

# Test 4-filter combos from the best building blocks
print("\n\n--- 4-filter combos with win% >= 60% ---")
# Use only the filters that appeared in top 3-filter results
best_filter_names = set()
for r in results_3[:20]:
    for f in r['filter'].split(' + '):
        best_filter_names.add(f.strip())
best_filter_names = [f for f in filter_names if f in best_filter_names]
print(f"Building 4-filter combos from {len(best_filter_names)} best filters: {best_filter_names}")

results_4 = []
for combo in combinations(best_filter_names, 4):
    name = ' + '.join(combo)
    mask = filters[combo[0]]
    for c in combo[1:]:
        mask = mask & filters[c]
    r = eval_filter(mask, name, min_trades=15)
    if r and r['win_pct'] >= 60:
        results_4.append(r)

results_4.sort(key=lambda x: x['win_pct'], reverse=True)
print(f"Found {len(results_4)} combos with >=60% win rate")
for r in results_4[:30]:
    cal_str = f"Calmar={r['calmar']}" if r['calmar'] is not None else "Calmar=N/A"
    print(f"  {r['filter']:<70} n={r['trades']:>4}  win={r['win_pct']:>5.1f}%  avg={r['avg_ret']:>+6.1f}%  PF={r['profit_factor']:>5.2f}  {cal_str}")

# 5-filter combos
print("\n\n--- 5-filter combos with win% >= 62% ---")
results_5 = []
if len(best_filter_names) >= 5:
    for combo in combinations(best_filter_names, 5):
        name = ' + '.join(combo)
        mask = filters[combo[0]]
        for c in combo[1:]:
            mask = mask & filters[c]
        r = eval_filter(mask, name, min_trades=15)
        if r and r['win_pct'] >= 62:
            results_5.append(r)

    results_5.sort(key=lambda x: x['win_pct'], reverse=True)
    print(f"Found {len(results_5)} combos with >=62% win rate")
    for r in results_5[:30]:
        cal_str = f"Calmar={r['calmar']}" if r['calmar'] is not None else "Calmar=N/A"
        print(f"  {r['filter']:<80} n={r['trades']:>4}  win={r['win_pct']:>5.1f}%  avg={r['avg_ret']:>+6.1f}%  PF={r['profit_factor']:>5.2f}  {cal_str}")

# 6-filter combos
print("\n\n--- 6-filter combos with win% >= 64% ---")
results_6 = []
if len(best_filter_names) >= 6:
    for combo in combinations(best_filter_names, 6):
        name = ' + '.join(combo)
        mask = filters[combo[0]]
        for c in combo[1:]:
            mask = mask & filters[c]
        r = eval_filter(mask, name, min_trades=15)
        if r and r['win_pct'] >= 64:
            results_6.append(r)

    results_6.sort(key=lambda x: x['win_pct'], reverse=True)
    print(f"Found {len(results_6)} combos with >=64% win rate")
    for r in results_6[:20]:
        cal_str = f"Calmar={r['calmar']}" if r['calmar'] is not None else "Calmar=N/A"
        print(f"  {r['filter']:<90} n={r['trades']:>4}  win={r['win_pct']:>5.1f}%  avg={r['avg_ret']:>+6.1f}%  PF={r['profit_factor']:>5.2f}  {cal_str}")

# ============================================================
# PHASE 3: Summary of best strategies at each win% tier
# ============================================================
print("\n\n" + "="*80)
print("PHASE 3: BEST STRATEGY AT EACH WIN RATE TIER")
print("="*80)

all_results = results_2 + results_3 + results_4 + results_5 + results_6
all_results.sort(key=lambda x: (-x['win_pct'], -x['profit_factor']))

# Find best for each win% tier
for target in [65, 64, 63, 62, 61, 60, 58, 55]:
    tier = [r for r in all_results if r['win_pct'] >= target]
    if tier:
        # Best by PF
        best_pf = max(tier, key=lambda x: x['profit_factor'])
        # Best by Calmar
        best_cal = max(tier, key=lambda x: x['calmar'] if x['calmar'] is not None else 0)
        # Most trades
        most = max(tier, key=lambda x: x['trades'])
        print(f"\n  >= {target}% WIN RATE ({len(tier)} strategies found):")
        print(f"    Best PF:     {best_pf['filter']:<60} n={best_pf['trades']:>4}  win={best_pf['win_pct']}%  PF={best_pf['profit_factor']}  Calmar={best_pf['calmar']}")
        print(f"    Best Calmar: {best_cal['filter']:<60} n={best_cal['trades']:>4}  win={best_cal['win_pct']}%  PF={best_cal['profit_factor']}  Calmar={best_cal['calmar']}")
        print(f"    Most trades: {most['filter']:<60} n={most['trades']:>4}  win={most['win_pct']}%  PF={most['profit_factor']}  Calmar={most['calmar']}")
    else:
        print(f"\n  >= {target}% WIN RATE: NO STRATEGIES FOUND with min 15 trades")

print("\n\nDone!")
