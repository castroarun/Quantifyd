"""
Breakout Analysis - Combined Filter Strategies & Winner/Loser Profiling
Sections 3-6 of the comprehensive report
"""
import pandas as pd
import numpy as np

df = pd.read_csv('breakout_analysis_full.csv')
print(f"Loaded {len(df)} trades")
print(f"Columns: {list(df.columns)}")
print()

# Helper to evaluate a filter
def eval_filter(mask, label):
    subset = df[mask]
    n = len(subset)
    if n < 20:
        return None
    stopped = (subset['exit_reason'] == 'STOP').mean() * 100
    winners = (subset['trade_return'] > 0).mean() * 100
    avg_ret = subset['trade_return'].mean()
    med_ret = subset['trade_return'].median()
    avg_gain = subset[subset['trade_return'] > 0]['trade_return'].mean() if (subset['trade_return'] > 0).any() else 0
    avg_loss = subset[subset['trade_return'] <= 0]['trade_return'].mean() if (subset['trade_return'] <= 0).any() else 0
    return {
        'filter': label, 'trades': n, 'stop_pct': round(stopped, 1),
        'win_pct': round(winners, 1), 'avg_ret': round(avg_ret, 1),
        'med_ret': round(med_ret, 1), 'avg_gain': round(avg_gain, 1),
        'avg_loss': round(avg_loss, 1),
        'profit_factor': round(abs(avg_gain / avg_loss), 2) if avg_loss != 0 else 999
    }

# ============================================================
# SECTION 3: Combined Filter Strategies
# ============================================================
print("=" * 80)
print("SECTION 3: COMBINED FILTER STRATEGIES")
print("=" * 80)

# Define filters
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
box_narrow = df['box_height_pct'].between(5, 15)
box_mid = df['box_height_pct'].between(10, 20)
ath_near = df['ath_proximity'] >= 0.85

combos = [
    ("BASELINE (all trades)", pd.Series([True]*len(df))),
    # 2-filter combos
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
    # 3-filter combos
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
    # 4-filter combos (the kitchen sink)
    ("RSI>60 + Vol>3x + BO>3% + EMA20>50", rsi_60 & vol_3x & bo_3 & ema20_50),
    ("RSI>60 + Vol>3x + BO>3% + EMA200", rsi_60 & vol_3x & bo_3 & ema200_above),
    ("RSI>70 + Vol>3x + BO>3% + EMA20>50", rsi_70 & vol_3x & bo_3 & ema20_50),
    ("RSI>60 + Vol>5x + BO>3% + EMA20>50", rsi_60 & vol_5x & bo_3 & ema20_50),
    ("Near ATH + RSI>60 + Vol>3x + BO>3%", ath_near & rsi_60 & vol_3x & bo_3),
    ("Mom dual + RSI>60 + Vol>3x", mom20_pos & mom60_pos & rsi_60 & vol_3x),
    # 5-filter combos
    ("RSI>60 + Vol>3x + BO>3% + EMA20>50 + Mom20>0", rsi_60 & vol_3x & bo_3 & ema20_50 & mom20_pos),
    ("RSI>60 + Vol>3x + BO>3% + EMA200 + Near ATH", rsi_60 & vol_3x & bo_3 & ema200_above & ath_near),
]

results = []
for label, mask in combos:
    r = eval_filter(mask, label)
    if r:
        results.append(r)

res_df = pd.DataFrame(results)
res_df = res_df.sort_values('win_pct', ascending=False)

print("\nRanked by Win Rate:")
print(f"{'Filter':<48} {'Trades':>6} {'Stop%':>6} {'Win%':>6} {'AvgRet':>7} {'MedRet':>7} {'AvgGain':>8} {'AvgLoss':>8} {'PF':>5}")
print("-" * 110)
for _, r in res_df.iterrows():
    print(f"{r['filter']:<48} {r['trades']:>6} {r['stop_pct']:>5.1f}% {r['win_pct']:>5.1f}% {r['avg_ret']:>6.1f}% {r['med_ret']:>6.1f}% {r['avg_gain']:>7.1f}% {r['avg_loss']:>7.1f}% {r['profit_factor']:>5.2f}")

# Also rank by avg return
print("\n\nRanked by Average Return:")
res_df2 = pd.DataFrame(results).sort_values('avg_ret', ascending=False)
print(f"{'Filter':<48} {'Trades':>6} {'Stop%':>6} {'Win%':>6} {'AvgRet':>7} {'MedRet':>7} {'PF':>5}")
print("-" * 90)
for _, r in res_df2.head(15).iterrows():
    print(f"{r['filter']:<48} {r['trades']:>6} {r['stop_pct']:>5.1f}% {r['win_pct']:>5.1f}% {r['avg_ret']:>6.1f}% {r['med_ret']:>6.1f}% {r['profit_factor']:>5.2f}")

# Also rank by lowest stop rate
print("\n\nRanked by Lowest Stop Rate:")
res_df3 = pd.DataFrame(results).sort_values('stop_pct', ascending=True)
print(f"{'Filter':<48} {'Trades':>6} {'Stop%':>6} {'Win%':>6} {'AvgRet':>7} {'MedRet':>7} {'PF':>5}")
print("-" * 90)
for _, r in res_df3.head(15).iterrows():
    print(f"{r['filter']:<48} {r['trades']:>6} {r['stop_pct']:>5.1f}% {r['win_pct']:>5.1f}% {r['avg_ret']:>6.1f}% {r['med_ret']:>6.1f}% {r['profit_factor']:>5.2f}")


# ============================================================
# SECTION 4: DARVAS vs FLAT with best filters
# ============================================================
print("\n\n" + "=" * 80)
print("SECTION 4: BEST FILTERS - DARVAS vs FLAT BREAKDOWN")
print("=" * 80)

# Take top 5 filters by win rate
top_filters = res_df.head(10)['filter'].tolist()
for filt_name in top_filters:
    # Find the corresponding mask
    for label, mask in combos:
        if label == filt_name:
            for det_type in ['darvas', 'flat']:
                det_mask = mask & (df['detector'] == det_type)
                r = eval_filter(det_mask, f"  {det_type.upper()}")
                if r:
                    print(f"{filt_name} -> {r['filter']}: {r['trades']} trades, {r['stop_pct']}% stop, {r['win_pct']}% win, {r['avg_ret']}% avg")
            print()
            break


# ============================================================
# SECTION 5: WINNER vs LOSER PROFILING
# ============================================================
print("\n\n" + "=" * 80)
print("SECTION 5: WINNER vs LOSER PROFILING")
print("=" * 80)

# Use the best practical filter (enough trades + good win rate)
# Let's use RSI>60 + Vol>3x + BO>3% as a reasonable filter
best_mask = rsi_60 & vol_3x & bo_3
best = df[best_mask].copy()

winners = best[best['trade_return'] > 0]
losers = best[best['trade_return'] <= 0]

print(f"\nUsing filter: RSI>60 + Vol>3x + BO>3%")
print(f"Total trades: {len(best)}, Winners: {len(winners)}, Losers: {len(losers)}")

print("\n--- Average Indicator Values ---")
metrics = ['rsi14', 'volume_ratio', 'breakout_pct', 'box_height_pct', 'ath_proximity',
           'mom_20d', 'mom_60d', 'vol_trend', 'risk_pct']
print(f"{'Metric':<20} {'Winners':>10} {'Losers':>10} {'Diff':>10} {'Edge':>10}")
print("-" * 65)
for m in metrics:
    if m in best.columns:
        w_val = winners[m].mean()
        l_val = losers[m].mean()
        diff = w_val - l_val
        edge = "WINNER" if abs(diff) > 0.01 and diff > 0 else ("LOSER" if abs(diff) > 0.01 and diff < 0 else "NEUTRAL")
        print(f"{m:<20} {w_val:>10.2f} {l_val:>10.2f} {diff:>+10.2f} {edge:>10}")

# EMA alignment comparison
print("\n--- EMA Alignment (% True) ---")
ema_cols = ['above_ema20', 'above_ema50', 'above_ema200', 'ema20_above_50', 'ema50_above_200']
print(f"{'Metric':<20} {'Winners':>10} {'Losers':>10} {'Diff':>10}")
print("-" * 55)
for m in ema_cols:
    if m in best.columns:
        w_val = winners[m].mean() * 100
        l_val = losers[m].mean() * 100
        diff = w_val - l_val
        print(f"{m:<20} {w_val:>9.1f}% {l_val:>9.1f}% {diff:>+9.1f}%")

# Distribution of returns
print("\n--- Return Distribution ---")
for label, sub in [("Winners", winners), ("Losers", losers)]:
    print(f"\n{label} ({len(sub)} trades):")
    print(f"  Mean return: {sub['trade_return'].mean():+.1f}%")
    print(f"  Median return: {sub['trade_return'].median():+.1f}%")
    print(f"  Std dev: {sub['trade_return'].std():.1f}%")
    print(f"  Best trade: {sub['trade_return'].max():+.1f}%")
    print(f"  Worst trade: {sub['trade_return'].min():+.1f}%")
    if 'max_gain' in sub.columns:
        print(f"  Avg max gain before exit: {sub['max_gain'].mean():+.1f}%")
    if 'days_to_peak' in sub.columns:
        print(f"  Avg days to peak: {sub['days_to_peak'].mean():.0f}")

# What kills losers?
print("\n--- What Kills Losers? (exit reason breakdown) ---")
if 'exit_reason' in best.columns:
    for reason in best['exit_reason'].unique():
        count = (losers['exit_reason'] == reason).sum()
        pct = count / len(losers) * 100 if len(losers) > 0 else 0
        avg_ret = losers[losers['exit_reason'] == reason]['trade_return'].mean()
        print(f"  {reason}: {count} trades ({pct:.1f}%), avg return: {avg_ret:+.1f}%")


# ============================================================
# SECTION 5b: FULL UNIVERSE winner/loser profile (no filter)
# ============================================================
print("\n\n--- FULL UNIVERSE Winner vs Loser Profile (no filters) ---")

all_winners = df[df['trade_return'] > 0]
all_losers = df[df['trade_return'] <= 0]
print(f"Total: {len(df)}, Winners: {len(all_winners)} ({len(all_winners)/len(df)*100:.1f}%), Losers: {len(all_losers)} ({len(all_losers)/len(df)*100:.1f}%)")

print(f"\n{'Metric':<20} {'Winners':>10} {'Losers':>10} {'Diff':>10} {'Discriminating?':>15}")
print("-" * 70)
for m in metrics:
    if m in df.columns:
        w_val = all_winners[m].mean()
        l_val = all_losers[m].mean()
        diff = w_val - l_val
        # Normalized diff
        overall_std = df[m].std()
        norm_diff = diff / overall_std if overall_std > 0 else 0
        discrim = "STRONG" if abs(norm_diff) > 0.3 else ("MODERATE" if abs(norm_diff) > 0.15 else "WEAK")
        print(f"{m:<20} {w_val:>10.2f} {l_val:>10.2f} {diff:>+10.2f} {discrim:>15}")


# ============================================================
# SECTION 6: OPTIMAL ENTRY CRITERIA RECOMMENDATION
# ============================================================
print("\n\n" + "=" * 80)
print("SECTION 6: OPTIMAL ENTRY CRITERIA RECOMMENDATIONS")
print("=" * 80)

# Find Pareto-optimal filters (best trade-off between win rate and trade count)
print("\n--- Pareto Analysis: Win Rate vs Trade Count ---")
res_all = pd.DataFrame(results)
# Score = win_pct * log(trades) to balance both
res_all['score'] = res_all['win_pct'] * np.log10(res_all['trades'])
res_all = res_all.sort_values('score', ascending=False)

print(f"{'Rank':>4} {'Filter':<48} {'Trades':>6} {'Win%':>6} {'AvgRet':>7} {'Score':>7}")
print("-" * 85)
for i, (_, r) in enumerate(res_all.head(15).iterrows()):
    print(f"{i+1:>4} {r['filter']:<48} {r['trades']:>6} {r['win_pct']:>5.1f}% {r['avg_ret']:>6.1f}% {r['score']:>7.1f}")

# Final recommendations
print("\n\n" + "=" * 80)
print("FINAL RECOMMENDATIONS")
print("=" * 80)

# Tier system
print("""
RECOMMENDED ENTRY FILTER TIERS:

TIER 1 - HIGH CONVICTION (fewer trades, higher win rate):
  Criteria: RSI>70 + Volume>5x + Breakout>3%
  Expected: ~60%+ win rate, low stop rate, high avg return
  Use for: Concentrated positions, larger allocations

TIER 2 - BALANCED (good trade-off):
  Criteria: RSI>60 + Volume>3x + Breakout>3%
  Expected: ~55%+ win rate, moderate trade count
  Use for: Standard positions, regular screening

TIER 3 - BROAD NET (more signals, moderate win rate):
  Criteria: RSI>60 + Volume>3x
  Expected: ~50%+ win rate, many opportunities
  Use for: Watchlist generation, initial screening
""")

# Print the actual numbers for each tier
tiers = [
    ("TIER 1: RSI>70 + Vol>5x + BO>3%", rsi_70 & vol_5x & bo_3),
    ("TIER 2: RSI>60 + Vol>3x + BO>3%", rsi_60 & vol_3x & bo_3),
    ("TIER 3: RSI>60 + Vol>3x", rsi_60 & vol_3x),
    ("TIER 2+EMA: RSI>60 + Vol>3x + BO>3% + EMA20>50", rsi_60 & vol_3x & bo_3 & ema20_50),
]

print("\nActual Performance by Tier:")
print(f"{'Tier':<50} {'Trades':>6} {'Stop%':>6} {'Win%':>6} {'AvgRet':>7} {'AvgGain':>8} {'AvgLoss':>8} {'PF':>5}")
print("-" * 100)
for label, mask in tiers:
    r = eval_filter(mask, label)
    if r:
        print(f"{r['filter']:<50} {r['trades']:>6} {r['stop_pct']:>5.1f}% {r['win_pct']:>5.1f}% {r['avg_ret']:>6.1f}% {r['avg_gain']:>7.1f}% {r['avg_loss']:>7.1f}% {r['profit_factor']:>5.2f}")
    else:
        print(f"{label:<50} <20 trades - insufficient data")

print("\n\nDone!")
