"""
Breakout Filter Optimization - Full Analysis with proper Calmar Ratio
Equal-dollar-per-trade portfolio simulation for each filter.
"""
import pandas as pd
import numpy as np
import json

df = pd.read_csv('breakout_analysis_full.csv')
df['date'] = pd.to_datetime(df['date'])
print(f"Loaded {len(df)} trades, {df['symbol'].nunique()} stocks")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

def compute_calmar(subset):
    """
    Compute Calmar ratio via equal-dollar portfolio simulation.

    Each trade gets $1 invested. P&L accumulates into an equity curve.
    Calmar = Annualized Return / Max Drawdown of the equity curve.
    """
    if len(subset) < 20:
        return None, None, None

    sorted_trades = subset.sort_values('date')
    dates = sorted_trades['date'].values
    returns = sorted_trades['trade_return'].values / 100  # convert % to decimal

    # Build equity curve: start with $100, each trade invests $1 (1% of initial)
    # P&L per trade = $1 * return
    # This is equivalent to equal-dollar allocation per trade
    initial_capital = 100.0
    equity = [initial_capital]
    trade_dates = [dates[0] - np.timedelta64(1, 'D')]  # day before first trade

    cumulative_pnl = 0
    for i, r in enumerate(returns):
        cumulative_pnl += r  # $1 per trade, so pnl = return as decimal
        equity.append(initial_capital + cumulative_pnl * initial_capital)
        trade_dates.append(dates[i])

    equity = np.array(equity)

    # Max drawdown from peak
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max * 100
    max_dd = abs(drawdowns.min())

    # Annualized return
    total_return = (equity[-1] / equity[0] - 1) * 100
    years = (dates[-1] - dates[0]) / np.timedelta64(365, 'D')
    years = max(years, 0.5)
    cagr = ((equity[-1] / equity[0]) ** (1 / years) - 1) * 100

    calmar = round(cagr / max_dd, 2) if max_dd > 0.1 else 999

    return round(cagr, 1), round(max_dd, 1), calmar


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
    profit_factor = round(total_gains / total_losses, 2)

    # Calmar ratio
    cagr, max_dd, calmar = compute_calmar(subset)

    return {
        'filter': label, 'trades': n, 'stop_pct': round(stopped, 1),
        'win_pct': round(winners_pct, 1), 'avg_ret': round(avg_ret, 1),
        'med_ret': round(med_ret, 1), 'avg_gain': round(avg_gain, 1),
        'avg_loss': round(avg_loss, 1), 'profit_factor': profit_factor,
        'cagr': cagr, 'max_dd': max_dd, 'calmar': calmar,
    }


# ============================================================
# Filter masks
# ============================================================
rsi_60 = df['rsi14'] > 60
rsi_70 = df['rsi14'] > 70
vol_3x = df['volume_ratio'] >= 3.0
vol_5x = df['volume_ratio'] >= 5.0
vol_10x = df['volume_ratio'] >= 10.0
bo_3 = df['breakout_pct'] >= 3.0
bo_5 = df['breakout_pct'] >= 5.0
ema20_50 = df['ema20_above_50'] == True
ema200_above = df['above_ema200'] == True
mom20_pos = df['mom_20d'] > 0
mom60_pos = df['mom_60d'] > 0
ath_near = df['ath_proximity'] >= 0.85

combos = [
    ("BASELINE (all trades)", pd.Series([True]*len(df))),
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
    ("RSI>60 + Vol>3x + BO>3% + EMA20>50", rsi_60 & vol_3x & bo_3 & ema20_50),
    ("RSI>60 + Vol>3x + BO>3% + EMA200", rsi_60 & vol_3x & bo_3 & ema200_above),
    ("RSI>70 + Vol>3x + BO>3% + EMA20>50", rsi_70 & vol_3x & bo_3 & ema20_50),
    ("RSI>60 + Vol>5x + BO>3% + EMA20>50", rsi_60 & vol_5x & bo_3 & ema20_50),
    ("Near ATH + RSI>60 + Vol>3x + BO>3%", ath_near & rsi_60 & vol_3x & bo_3),
    ("Mom dual + RSI>60 + Vol>3x", mom20_pos & mom60_pos & rsi_60 & vol_3x),
    ("RSI>60 + Vol>3x + BO>3% + EMA20>50 + Mom20>0", rsi_60 & vol_3x & bo_3 & ema20_50 & mom20_pos),
    ("RSI>60 + Vol>3x + BO>3% + EMA200 + Near ATH", rsi_60 & vol_3x & bo_3 & ema200_above & ath_near),
]

# ============================================================
# Run all filters
# ============================================================
print("\n=== ALL FILTER RESULTS ===")
results = []
for label, mask in combos:
    r = eval_filter(mask, label)
    if r:
        results.append(r)
        print(f"  {r['filter']:<50} n={r['trades']:>5}  win={r['win_pct']:>5.1f}%  avg={r['avg_ret']:>+6.1f}%  stop={r['stop_pct']:>5.1f}%  PF={r['profit_factor']:>5.2f}  CAGR={r['cagr']:>+7.1f}%  MaxDD={r['max_dd']:>5.1f}%  Calmar={r['calmar']:>5.2f}")

# ============================================================
# Darvas vs Flat
# ============================================================
print("\n=== DARVAS vs FLAT ===")
dvf_filters = [
    "RSI>70 + Vol>3x + BO>3% + EMA20>50",
    "RSI>70 + Vol>3x + BO>3%",
    "Vol>5x + BO>5%",
    "RSI>60 + Vol>5x + BO>3% + EMA20>50",
    "RSI>60 + Vol>3x + BO>3% + EMA20>50",
    "RSI>60 + Vol>3x + BO>3%",
    "Vol>5x + BO>3%",
    "Near ATH + Vol>3x + BO>3%",
]

darvas_vs_flat = []
for filt_name in dvf_filters:
    for label, mask in combos:
        if label == filt_name:
            row = {'filter': filt_name}
            for det in ['darvas', 'flat']:
                r = eval_filter(mask & (df['detector'] == det), det)
                if r:
                    for k in ['trades', 'win_pct', 'avg_ret', 'stop_pct', 'profit_factor', 'calmar', 'max_dd', 'cagr']:
                        row[f'{det}_{k}'] = r[k]
                else:
                    row[f'{det}_trades'] = '<20'
            darvas_vs_flat.append(row)
            print(f"  {filt_name}")
            for det in ['darvas', 'flat']:
                if f'{det}_win_pct' in row:
                    print(f"    {det.upper()}: {row[f'{det}_trades']} trades, {row[f'{det}_win_pct']}% win, {row[f'{det}_avg_ret']:+.1f}% avg, PF={row[f'{det}_profit_factor']}, Calmar={row[f'{det}_calmar']}")
                else:
                    print(f"    {det.upper()}: <20 trades")
            break

# ============================================================
# Winner vs Loser
# ============================================================
print("\n=== WINNER vs LOSER ===")
best_mask = rsi_60 & vol_3x & bo_3
best = df[best_mask].copy()
winners = best[best['trade_return'] > 0]
losers = best[best['trade_return'] <= 0]
print(f"Filter: RSI>60 + Vol>3x + BO>3% | {len(best)} total, {len(winners)} win, {len(losers)} lose")

metrics = ['rsi14', 'volume_ratio', 'breakout_pct', 'box_height_pct', 'ath_proximity',
           'mom_20d', 'mom_60d', 'vol_trend', 'risk_pct']

wl_data = []
for m in metrics:
    w = winners[m].mean()
    l = losers[m].mean()
    d = w - l
    s = best[m].std()
    nd = d / s if s > 0 else 0
    strength = "STRONG" if abs(nd) > 0.3 else ("MODERATE" if abs(nd) > 0.15 else "WEAK")
    wl_data.append({'metric': m, 'winners': round(w, 2), 'losers': round(l, 2), 'diff': round(d, 2), 'strength': strength})
    print(f"  {m:<20} W={w:>8.2f}  L={l:>8.2f}  D={d:>+8.2f}  {strength}")

ema_cols = ['above_ema20', 'above_ema50', 'above_ema200', 'ema20_above_50', 'ema50_above_200']
ema_data = []
for m in ema_cols:
    w = winners[m].mean() * 100
    l = losers[m].mean() * 100
    ema_data.append({'metric': m, 'winners': round(w, 1), 'losers': round(l, 1), 'diff': round(w - l, 1)})

dist = {}
for label, sub in [("winners", winners), ("losers", losers)]:
    dist[label] = {
        'count': len(sub), 'mean': round(sub['trade_return'].mean(), 1),
        'median': round(sub['trade_return'].median(), 1), 'std': round(sub['trade_return'].std(), 1),
        'best': round(sub['trade_return'].max(), 1), 'worst': round(sub['trade_return'].min(), 1),
        'avg_max_gain': round(sub['max_gain'].mean(), 1),
        'avg_days_peak': int(round(sub['days_to_peak'].mean())),
    }

exit_data = []
for reason in losers['exit_reason'].unique():
    sub = losers[losers['exit_reason'] == reason]
    exit_data.append({'reason': reason, 'count': len(sub), 'pct': round(len(sub)/len(losers)*100, 1), 'avg_ret': round(sub['trade_return'].mean(), 1)})

# Full universe
all_w = df[df['trade_return'] > 0]
all_l = df[df['trade_return'] <= 0]
full_wl = []
for m in metrics:
    w = all_w[m].mean(); l = all_l[m].mean(); d = w - l
    s = df[m].std(); nd = d / s if s > 0 else 0
    strength = "STRONG" if abs(nd) > 0.3 else ("MODERATE" if abs(nd) > 0.15 else "WEAK")
    full_wl.append({'metric': m, 'winners': round(w, 2), 'losers': round(l, 2), 'diff': round(d, 2), 'strength': strength})

# ============================================================
# Tier results
# ============================================================
print("\n=== TIER RESULTS ===")
tier_defs = [
    ("TIER 1: RSI>70 + Vol>5x + BO>3%", rsi_70 & vol_5x & bo_3),
    ("TIER 2: RSI>60 + Vol>3x + BO>3% + EMA20>50", rsi_60 & vol_3x & bo_3 & ema20_50),
    ("TIER 2-ALT: Vol>5x + BO>5%", vol_5x & bo_5),
    ("TIER 3: RSI>60 + Vol>3x", rsi_60 & vol_3x),
    ("BASELINE: No filters", pd.Series([True]*len(df))),
]

tier_results = []
for label, mask in tier_defs:
    r = eval_filter(mask, label)
    if r:
        tier_results.append(r)
        print(f"  {r['filter']:<50} n={r['trades']:>5}  win={r['win_pct']:>5.1f}%  avg={r['avg_ret']:>+6.1f}%  PF={r['profit_factor']:>5.2f}  Calmar={r['calmar']:>5.2f}")
    else:
        print(f"  {label}: <20 trades")

# ============================================================
# Save JSON
# ============================================================
report = {
    'total_trades': len(df), 'stocks': df['symbol'].nunique(),
    'date_range': f"{df['date'].min().date()} to {df['date'].max().date()}",
    'darvas_count': int((df['detector'] == 'darvas').sum()),
    'flat_count': int((df['detector'] == 'flat').sum()),
    'all_filters': results,
    'darvas_vs_flat': darvas_vs_flat,
    'winner_loser': wl_data,
    'ema_alignment': ema_data,
    'distributions': dist,
    'exit_breakdown': exit_data,
    'full_universe_wl': full_wl,
    'tier_results': tier_results,
    'winners_total': len(all_w), 'losers_total': len(all_l),
}

with open('filter_analysis_data.json', 'w') as f:
    json.dump(report, f, indent=2, default=str)

print(f"\nSaved to filter_analysis_data.json")
