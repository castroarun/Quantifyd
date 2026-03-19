"""
Breakout Filter Optimization - Full Analysis with Calmar Ratio
Outputs structured data for the MD report.
"""
import pandas as pd
import numpy as np
import json

df = pd.read_csv('breakout_analysis_full.csv')
df['date'] = pd.to_datetime(df['date'])
print(f"Loaded {len(df)} trades, date range: {df['date'].min().date()} to {df['date'].max().date()}")

# ============================================================
# Helper: Evaluate a filter with Calmar ratio
# ============================================================
def eval_filter(mask, label):
    subset = df[mask].copy()
    n = len(subset)
    if n < 20:
        return None

    stopped = (subset['exit_reason'] == 'STOP').mean() * 100
    winners_pct = (subset['trade_return'] > 0).mean() * 100
    avg_ret = subset['trade_return'].mean()
    med_ret = subset['trade_return'].median()

    wins = subset[subset['trade_return'] > 0]['trade_return']
    losses = subset[subset['trade_return'] <= 0]['trade_return']

    avg_gain = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0

    # Profit Factor = Total Gains / |Total Losses|
    total_gains = wins.sum() if len(wins) > 0 else 0
    total_losses = abs(losses.sum()) if len(losses) > 0 else 0.01
    profit_factor = round(total_gains / total_losses, 2) if total_losses > 0 else 999

    # Calmar Ratio via equity curve simulation
    # Sort trades chronologically, simulate equal-weight allocation
    sorted_trades = subset.sort_values('date')
    returns_pct = sorted_trades['trade_return'].values  # already in %

    # Build equity curve: start at 100, each trade adds its % return
    # (simplified: assume sequential trades with equal capital allocation)
    equity = [100.0]
    for r in returns_pct:
        new_val = equity[-1] * (1 + r / 100)
        equity.append(new_val)
    equity = np.array(equity)

    # Max drawdown from equity curve
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max * 100  # in %
    max_dd = abs(drawdowns.min())

    # CAGR from equity curve
    dates_sorted = sorted_trades['date'].values
    if len(dates_sorted) >= 2:
        years = (dates_sorted[-1] - dates_sorted[0]) / np.timedelta64(365, 'D')
        years = max(years, 0.5)  # minimum half year to avoid infinity
        final_val = equity[-1]
        cagr = ((final_val / 100) ** (1 / years) - 1) * 100
    else:
        cagr = avg_ret
        years = 1

    calmar = round(cagr / max_dd, 2) if max_dd > 0 else 999

    return {
        'filter': label,
        'trades': n,
        'stop_pct': round(stopped, 1),
        'win_pct': round(winners_pct, 1),
        'avg_ret': round(avg_ret, 1),
        'med_ret': round(med_ret, 1),
        'avg_gain': round(avg_gain, 1),
        'avg_loss': round(avg_loss, 1),
        'profit_factor': profit_factor,
        'cagr': round(cagr, 1),
        'max_dd': round(max_dd, 1),
        'calmar': calmar,
    }

# ============================================================
# Define all filter masks
# ============================================================
rsi_60 = df['rsi14'] > 60
rsi_70 = df['rsi14'] > 70
vol_3x = df['volume_ratio'] >= 3.0
vol_5x = df['volume_ratio'] >= 5.0
vol_10x = df['volume_ratio'] >= 10.0
bo_3 = df['breakout_pct'] >= 3.0
bo_5 = df['breakout_pct'] >= 5.0
ema20_above = df['above_ema20'] == True
ema50_above = df['above_ema50'] == True
ema200_above = df['above_ema200'] == True
ema20_50 = df['ema20_above_50'] == True
mom20_pos = df['mom_20d'] > 0
mom60_pos = df['mom_60d'] > 0
ath_near = df['ath_proximity'] >= 0.85

combos = [
    ("BASELINE (all trades)", pd.Series([True]*len(df))),
    # 2-filter
    ("RSI>60 + Vol>3x", rsi_60 & vol_3x),
    ("RSI>60 + BO>3%", rsi_60 & bo_3),
    ("RSI>70 + Vol>3x", rsi_70 & vol_3x),
    ("RSI>70 + BO>5%", rsi_70 & bo_5),
    ("Vol>5x + BO>3%", vol_5x & bo_3),
    ("Vol>5x + BO>5%", vol_5x & bo_5),
    ("EMA20>50 + RSI>60", ema20_50 & rsi_60),
    ("EMA20>50 + Vol>3x", ema20_50 & vol_3x),
    ("Above EMA200 + RSI>60", ema200_above & rsi_60),
    ("Above EMA200 + BO>3%", ema200_above & bo_3),
    ("Near ATH + Vol>3x", ath_near & vol_3x),
    ("Near ATH + RSI>60", ath_near & rsi_60),
    ("Mom20>0 + Mom60>0", mom20_pos & mom60_pos),
    # 3-filter
    ("RSI>60 + Vol>3x + BO>3%", rsi_60 & vol_3x & bo_3),
    ("RSI>60 + Vol>3x + EMA20>50", rsi_60 & vol_3x & ema20_50),
    ("RSI>60 + BO>3% + EMA20>50", rsi_60 & bo_3 & ema20_50),
    ("RSI>70 + Vol>3x + BO>3%", rsi_70 & vol_3x & bo_3),
    ("RSI>70 + Vol>5x + BO>3%", rsi_70 & vol_5x & bo_3),
    ("Vol>5x + BO>5% + EMA20>50", vol_5x & bo_5 & ema20_50),
    ("Near ATH + Vol>3x + RSI>60", ath_near & vol_3x & rsi_60),
    ("Near ATH + Vol>3x + BO>3%", ath_near & vol_3x & bo_3),
    ("Mom20>0 + Mom60>0 + RSI>60", mom20_pos & mom60_pos & rsi_60),
    ("Mom20>0 + Mom60>0 + Vol>3x", mom20_pos & mom60_pos & vol_3x),
    ("Above EMA200 + RSI>60 + Vol>3x", ema200_above & rsi_60 & vol_3x),
    # 4-filter
    ("RSI>60 + Vol>3x + BO>3% + EMA20>50", rsi_60 & vol_3x & bo_3 & ema20_50),
    ("RSI>60 + Vol>3x + BO>3% + EMA200", rsi_60 & vol_3x & bo_3 & ema200_above),
    ("RSI>70 + Vol>3x + BO>3% + EMA20>50", rsi_70 & vol_3x & bo_3 & ema20_50),
    ("RSI>60 + Vol>5x + BO>3% + EMA20>50", rsi_60 & vol_5x & bo_3 & ema20_50),
    ("Near ATH + RSI>60 + Vol>3x + BO>3%", ath_near & rsi_60 & vol_3x & bo_3),
    ("Mom dual + RSI>60 + Vol>3x", mom20_pos & mom60_pos & rsi_60 & vol_3x),
    # 5-filter
    ("RSI>60 + Vol>3x + BO>3% + EMA20>50 + Mom20>0", rsi_60 & vol_3x & bo_3 & ema20_50 & mom20_pos),
    ("RSI>60 + Vol>3x + BO>3% + EMA200 + Near ATH", rsi_60 & vol_3x & bo_3 & ema200_above & ath_near),
]

# ============================================================
# Run all filters
# ============================================================
results = []
for label, mask in combos:
    r = eval_filter(mask, label)
    if r:
        results.append(r)
        print(f"  {r['filter']:<50} trades={r['trades']:>5}  win={r['win_pct']:>5.1f}%  avg={r['avg_ret']:>+6.1f}%  PF={r['profit_factor']:>5.2f}  CAGR={r['cagr']:>+6.1f}%  MaxDD={r['max_dd']:>5.1f}%  Calmar={r['calmar']:>5.2f}")

# ============================================================
# Darvas vs Flat breakdown for top filters
# ============================================================
print("\n\n=== DARVAS vs FLAT with top filters ===")
darvas_vs_flat = []
top_filter_names = [
    "RSI>70 + Vol>3x + BO>3% + EMA20>50",
    "RSI>70 + Vol>3x + BO>3%",
    "Vol>5x + BO>5%",
    "RSI>60 + Vol>5x + BO>3% + EMA20>50",
    "RSI>60 + Vol>3x + BO>3% + EMA20>50",
    "RSI>60 + Vol>3x + BO>3%",
    "Vol>5x + BO>3%",
    "Near ATH + Vol>3x + BO>3%",
]

for filt_name in top_filter_names:
    for label, mask in combos:
        if label == filt_name:
            row = {'filter': filt_name}
            for det_type in ['darvas', 'flat']:
                det_mask = mask & (df['detector'] == det_type)
                r = eval_filter(det_mask, f"{det_type}")
                if r:
                    row[f'{det_type}_trades'] = r['trades']
                    row[f'{det_type}_win'] = r['win_pct']
                    row[f'{det_type}_avg'] = r['avg_ret']
                    row[f'{det_type}_stop'] = r['stop_pct']
                    row[f'{det_type}_pf'] = r['profit_factor']
                    row[f'{det_type}_calmar'] = r['calmar']
                    row[f'{det_type}_maxdd'] = r['max_dd']
                else:
                    row[f'{det_type}_trades'] = '<20'
            darvas_vs_flat.append(row)
            break

for r in darvas_vs_flat:
    print(f"  {r['filter']}")
    for det in ['darvas', 'flat']:
        if f'{det}_trades' in r and r[f'{det}_trades'] != '<20':
            print(f"    {det.upper()}: {r[f'{det}_trades']} trades, {r[f'{det}_win']}% win, {r[f'{det}_avg']:+.1f}% avg, PF={r[f'{det}_pf']}, Calmar={r[f'{det}_calmar']}")
        else:
            print(f"    {det.upper()}: <20 trades")

# ============================================================
# Winner vs Loser profiling
# ============================================================
print("\n\n=== WINNER vs LOSER PROFILING ===")
best_mask = rsi_60 & vol_3x & bo_3
best = df[best_mask].copy()
winners = best[best['trade_return'] > 0]
losers = best[best['trade_return'] <= 0]

print(f"Filter: RSI>60 + Vol>3x + BO>3% | Total: {len(best)}, Winners: {len(winners)}, Losers: {len(losers)}")

metrics = ['rsi14', 'volume_ratio', 'breakout_pct', 'box_height_pct', 'ath_proximity',
           'mom_20d', 'mom_60d', 'vol_trend', 'risk_pct']

winner_loser_data = []
for m in metrics:
    if m in best.columns:
        w_val = winners[m].mean()
        l_val = losers[m].mean()
        diff = w_val - l_val
        overall_std = best[m].std()
        norm_diff = diff / overall_std if overall_std > 0 else 0
        discrim = "STRONG" if abs(norm_diff) > 0.3 else ("MODERATE" if abs(norm_diff) > 0.15 else "WEAK")
        winner_loser_data.append({
            'metric': m, 'winners': round(w_val, 2), 'losers': round(l_val, 2),
            'diff': round(diff, 2), 'strength': discrim
        })

ema_cols = ['above_ema20', 'above_ema50', 'above_ema200', 'ema20_above_50', 'ema50_above_200']
ema_data = []
for m in ema_cols:
    if m in best.columns:
        w_val = winners[m].mean() * 100
        l_val = losers[m].mean() * 100
        ema_data.append({'metric': m, 'winners': round(w_val, 1), 'losers': round(l_val, 1), 'diff': round(w_val - l_val, 1)})

# Return distributions
dist_data = {}
for label, sub in [("winners", winners), ("losers", losers)]:
    dist_data[label] = {
        'count': len(sub),
        'mean': round(sub['trade_return'].mean(), 1),
        'median': round(sub['trade_return'].median(), 1),
        'std': round(sub['trade_return'].std(), 1),
        'best': round(sub['trade_return'].max(), 1),
        'worst': round(sub['trade_return'].min(), 1),
        'avg_max_gain': round(sub['max_gain'].mean(), 1) if 'max_gain' in sub.columns else None,
        'avg_days_to_peak': round(sub['days_to_peak'].mean()) if 'days_to_peak' in sub.columns else None,
    }

# Exit reason breakdown for losers
exit_data = []
for reason in best['exit_reason'].unique():
    loser_subset = losers[losers['exit_reason'] == reason]
    count = len(loser_subset)
    pct = count / len(losers) * 100 if len(losers) > 0 else 0
    avg_ret = loser_subset['trade_return'].mean()
    exit_data.append({'reason': reason, 'count': count, 'pct': round(pct, 1), 'avg_ret': round(avg_ret, 1)})

# Full universe winner/loser profile
all_winners = df[df['trade_return'] > 0]
all_losers = df[df['trade_return'] <= 0]
full_wl_data = []
for m in metrics:
    if m in df.columns:
        w_val = all_winners[m].mean()
        l_val = all_losers[m].mean()
        diff = w_val - l_val
        overall_std = df[m].std()
        norm_diff = diff / overall_std if overall_std > 0 else 0
        discrim = "STRONG" if abs(norm_diff) > 0.3 else ("MODERATE" if abs(norm_diff) > 0.15 else "WEAK")
        full_wl_data.append({'metric': m, 'winners': round(w_val, 2), 'losers': round(l_val, 2), 'diff': round(diff, 2), 'strength': discrim})

# ============================================================
# Tier performance
# ============================================================
tiers = [
    ("TIER 1: RSI>70 + Vol>5x + BO>3%", rsi_70 & vol_5x & bo_3),
    ("TIER 2: RSI>60 + Vol>3x + BO>3% + EMA20>50", rsi_60 & vol_3x & bo_3 & ema20_50),
    ("TIER 2-ALT: Vol>5x + BO>5%", vol_5x & bo_5),
    ("TIER 3: RSI>60 + Vol>3x", rsi_60 & vol_3x),
    ("BASELINE: No filters", pd.Series([True]*len(df))),
]

tier_results = []
for label, mask in tiers:
    r = eval_filter(mask, label)
    if r:
        tier_results.append(r)

# ============================================================
# Save all results as JSON for report generation
# ============================================================
report_data = {
    'total_trades': len(df),
    'date_range': f"{df['date'].min().date()} to {df['date'].max().date()}",
    'stocks_count': df['symbol'].nunique(),
    'darvas_count': len(df[df['detector'] == 'darvas']),
    'flat_count': len(df[df['detector'] == 'flat']),
    'all_filters': results,
    'darvas_vs_flat': darvas_vs_flat,
    'winner_loser': winner_loser_data,
    'ema_alignment': ema_data,
    'return_distributions': dist_data,
    'exit_breakdown': exit_data,
    'full_universe_wl': full_wl_data,
    'tier_results': tier_results,
    'all_winners_count': len(all_winners),
    'all_losers_count': len(all_losers),
}

with open('filter_analysis_data.json', 'w') as f:
    json.dump(report_data, f, indent=2, default=str)

print(f"\nSaved analysis data to filter_analysis_data.json")
print("Done!")
