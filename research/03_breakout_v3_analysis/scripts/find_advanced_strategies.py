"""
Advanced Strategy Research: New filter types beyond simple threshold combos.

New approaches tested:
  1. Risk-reward filters (tight stops = better R:R)
  2. Momentum divergence (short-term vs medium-term alignment)
  3. Composite scores (weighted rank across multiple indicators)
  4. ATH proximity bands (near-ATH vs recovering stocks)
  5. Interaction terms (RSI*Volume, BO*Momentum)
  6. Percentile-based adaptive thresholds
  7. Detector-specific optimized filters
  8. Exhaustive combo search with new building blocks
"""
import pandas as pd
import numpy as np
from itertools import combinations

df = pd.read_csv('breakout_analysis_full.csv')
df['date'] = pd.to_datetime(df['date'])
print(f"Loaded {len(df)} trades, {df['symbol'].nunique()} stocks")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Baseline win rate: {(df['trade_return'] > 0).mean()*100:.1f}%")

# --- Derived features ---
# Momentum alignment: short > medium (accelerating)
df['mom_accel'] = df['mom_20d'] - df['mom_60d'] / 3  # 20d vs annualized 60d rate
df['mom_ratio'] = df['mom_20d'] / df['mom_60d'].clip(lower=0.1)  # short/long momentum
# Risk-reward proxy: breakout strength vs risk (distance to stop)
df['rr_ratio'] = df['breakout_pct'] / df['risk_pct'].clip(lower=0.1)
# Volume-price composite
df['vol_rsi_product'] = df['volume_ratio'] * df['rsi14'] / 60  # normalized
# Tight consolidation indicator
df['tight_box'] = df['box_height_pct'] <= df['box_height_pct'].quantile(0.25)  # tightest 25%
# ATH breakout (very near ATH)
df['near_ath_95'] = df['ath_proximity'] >= 95
df['near_ath_90'] = df['ath_proximity'] >= 90
# Low risk entry (tighter stop)
df['low_risk'] = df['risk_pct'] <= df['risk_pct'].quantile(0.25)
df['med_risk'] = df['risk_pct'] <= df['risk_pct'].median()
# Momentum confirmation: both 20d and 60d positive AND 20d > some threshold
df['dual_mom_strong'] = (df['mom_20d'] >= 5) & (df['mom_60d'] >= 10)
# Volume trend categories
df['vol_trend_rising'] = df['vol_trend'] >= 1.2
df['vol_trend_strong'] = df['vol_trend'] >= 1.5
# EMA triple alignment
df['ema_triple'] = (df['above_ema20'] == 1) & (df['above_ema50'] == 1) & (df['above_ema200'] == 1)
# Momentum acceleration: 20d momentum > 60d/3 (annualized rate)
df['mom_accelerating'] = df['mom_20d'] > (df['mom_60d'] / 3)

# --- Composite score (rank-based) ---
# Rank each trade on multiple dimensions, higher = better
for col in ['rsi14', 'volume_ratio', 'breakout_pct', 'mom_60d', 'vol_trend', 'ath_proximity']:
    df[f'{col}_pctile'] = df[col].rank(pct=True) * 100

df['composite_score'] = (
    df['rsi14_pctile'] * 0.25 +
    df['volume_ratio_pctile'] * 0.20 +
    df['breakout_pct_pctile'] * 0.20 +
    df['mom_60d_pctile'] * 0.15 +
    df['vol_trend_pctile'] * 0.10 +
    df['ath_proximity_pctile'] * 0.10
)

# Alternative composite: momentum-heavy
df['composite_mom'] = (
    df['rsi14_pctile'] * 0.15 +
    df['mom_20d'].rank(pct=True) * 100 * 0.25 +
    df['mom_60d_pctile'] * 0.25 +
    df['vol_trend_pctile'] * 0.20 +
    df['breakout_pct_pctile'] * 0.15
)

# Alternative composite: volume-heavy
df['composite_vol'] = (
    df['volume_ratio_pctile'] * 0.30 +
    df['vol_trend_pctile'] * 0.25 +
    df['breakout_pct_pctile'] * 0.20 +
    df['rsi14_pctile'] * 0.15 +
    df['ath_proximity_pctile'] * 0.10
)


def compute_calmar(subset):
    if len(subset) < 10:
        return None, None, None
    sorted_trades = subset.sort_values('date')
    dates = sorted_trades['date'].values
    returns = sorted_trades['trade_return'].values / 100
    initial_capital = 100.0
    equity = [initial_capital]
    cumulative_pnl = 0
    for r in returns:
        cumulative_pnl += r
        equity.append(initial_capital + cumulative_pnl * initial_capital)
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
    win_pct = (subset['trade_return'] > 0).mean() * 100
    avg_ret = subset['trade_return'].mean()
    med_ret = subset['trade_return'].median()
    stopped = (subset['exit_reason'] == 'STOP').mean() * 100
    wins = subset[subset['trade_return'] > 0]['trade_return']
    losses = subset[subset['trade_return'] <= 0]['trade_return']
    total_gains = wins.sum() if len(wins) > 0 else 0
    total_losses = abs(losses.sum()) if len(losses) > 0 else 0.01
    pf = round(total_gains / total_losses, 2)
    cagr, max_dd, calmar = compute_calmar(subset)
    return {
        'filter': label, 'trades': n, 'stop_pct': round(stopped, 1),
        'win_pct': round(win_pct, 1), 'avg_ret': round(avg_ret, 1),
        'med_ret': round(med_ret, 1), 'profit_factor': pf,
        'cagr': cagr, 'max_dd': max_dd, 'calmar': calmar,
    }


# ============================================================
# PHASE 1: New individual filter ideas
# ============================================================
print("\n" + "="*80)
print("PHASE 1: New individual filter ideas")
print("="*80)

new_singles = [
    # Risk-reward based
    ("Low Risk (stop <= 25th pctile)", df['low_risk']),
    ("Medium Risk (stop <= median)", df['med_risk']),
    ("Risk% <= 8%", df['risk_pct'] <= 8),
    ("Risk% <= 10%", df['risk_pct'] <= 10),
    ("Risk% <= 6%", df['risk_pct'] <= 6),
    # R:R ratio
    ("R:R >= 0.3", df['rr_ratio'] >= 0.3),
    ("R:R >= 0.5", df['rr_ratio'] >= 0.5),
    ("R:R >= 0.7", df['rr_ratio'] >= 0.7),
    # Momentum divergence
    ("Mom accelerating", df['mom_accelerating']),
    ("Dual Mom Strong (20d>=5 & 60d>=10)", df['dual_mom_strong']),
    ("Mom ratio > 1.0 (20d > 60d/3)", df['mom_ratio'] > 1.0),
    # ATH bands
    ("ATH >= 95%", df['near_ath_95']),
    ("ATH >= 98%", df['ath_proximity'] >= 98),
    ("ATH 70-85% (recovery zone)", df['ath_proximity'].between(70, 85)),
    ("ATH < 70% (deep value)", df['ath_proximity'] < 70),
    # Tight consolidation
    ("Tight Box (25th pctile)", df['tight_box']),
    ("Box <= 8%", df['box_height_pct'] <= 8),
    ("Box 5-10%", df['box_height_pct'].between(5, 10)),
    # EMA alignment
    ("EMA Triple (above 20+50+200)", df['ema_triple']),
    ("EMA50 above 200", df['ema50_above_200'] == 1),
    # Volume trend
    ("VolTrend >= 1.5x", df['vol_trend_strong']),
    ("VolTrend >= 1.3x", df['vol_trend'] >= 1.3),
    # Volume-RSI product
    ("Vol*RSI product >= 5", df['vol_rsi_product'] >= 5),
    ("Vol*RSI product >= 7", df['vol_rsi_product'] >= 7),
    # Composite scores
    ("Composite Top 20%", df['composite_score'] >= df['composite_score'].quantile(0.80)),
    ("Composite Top 10%", df['composite_score'] >= df['composite_score'].quantile(0.90)),
    ("Composite Top 5%", df['composite_score'] >= df['composite_score'].quantile(0.95)),
    ("CompositeMom Top 20%", df['composite_mom'] >= df['composite_mom'].quantile(0.80)),
    ("CompositeMom Top 10%", df['composite_mom'] >= df['composite_mom'].quantile(0.90)),
    ("CompositeVol Top 20%", df['composite_vol'] >= df['composite_vol'].quantile(0.80)),
    ("CompositeVol Top 10%", df['composite_vol'] >= df['composite_vol'].quantile(0.90)),
    # Detector specific
    ("Darvas only", df['detector'] == 'darvas'),
    ("Flat only", df['detector'] == 'flat'),
]

print(f"\nTesting {len(new_singles)} new individual filters...")
singles_results = []
for label, mask in new_singles:
    r = eval_filter(mask, label)
    if r:
        singles_results.append(r)

singles_results.sort(key=lambda x: x['win_pct'], reverse=True)
print(f"\n{'Filter':<45} {'Trades':>6} {'Win%':>6} {'AvgRet':>7} {'PF':>6} {'Calmar':>7}")
print("-" * 80)
for r in singles_results:
    cal = f"{r['calmar']:>6.2f}" if r['calmar'] is not None else "   N/A"
    print(f"  {r['filter']:<43} {r['trades']:>6} {r['win_pct']:>5.1f}% {r['avg_ret']:>+6.1f}% {r['profit_factor']:>5.2f} {cal}")


# ============================================================
# PHASE 2: Composite score threshold sweep
# ============================================================
print("\n\n" + "="*80)
print("PHASE 2: Composite score thresholds")
print("="*80)

for score_name, score_col in [("Balanced", "composite_score"), ("Momentum", "composite_mom"), ("Volume", "composite_vol")]:
    print(f"\n--- {score_name} Composite ---")
    for pctile in [60, 70, 75, 80, 85, 90, 95]:
        thresh = df[score_col].quantile(pctile / 100)
        mask = df[score_col] >= thresh
        r = eval_filter(mask, f"{score_name} Top {100-pctile}%")
        if r:
            cal = f"{r['calmar']:.2f}" if r['calmar'] is not None else "N/A"
            print(f"  Top {100-pctile:>2}% (>={thresh:.1f}): {r['trades']:>5} trades, {r['win_pct']:>5.1f}% win, {r['avg_ret']:>+6.1f}%, PF={r['profit_factor']:>5.2f}, Calmar={cal}")


# ============================================================
# PHASE 3: Best composite scores + additional filters
# ============================================================
print("\n\n" + "="*80)
print("PHASE 3: Composite score + additional filter combos")
print("="*80)

# Take top composite quintiles and add extra filters
composite_bases = [
    ("Top20% Balanced", df['composite_score'] >= df['composite_score'].quantile(0.80)),
    ("Top10% Balanced", df['composite_score'] >= df['composite_score'].quantile(0.90)),
    ("Top20% Momentum", df['composite_mom'] >= df['composite_mom'].quantile(0.80)),
    ("Top10% Momentum", df['composite_mom'] >= df['composite_mom'].quantile(0.90)),
    ("Top20% Volume", df['composite_vol'] >= df['composite_vol'].quantile(0.80)),
]

extra_filters = [
    ("+ VolTrend>=1.2", df['vol_trend'] >= 1.2),
    ("+ VolTrend>=1.5", df['vol_trend'] >= 1.5),
    ("+ ATH>=90%", df['ath_proximity'] >= 90),
    ("+ ATH>=95%", df['ath_proximity'] >= 95),
    ("+ LowRisk", df['low_risk']),
    ("+ EMA Triple", df['ema_triple']),
    ("+ DualMomStrong", df['dual_mom_strong']),
    ("+ Darvas", df['detector'] == 'darvas'),
    ("+ Box<=10%", df['box_height_pct'] <= 10),
    ("+ Risk<=8%", df['risk_pct'] <= 8),
    ("+ BO>=3%", df['breakout_pct'] >= 3),
    ("+ BO>=5%", df['breakout_pct'] >= 5),
]

phase3_results = []
for base_name, base_mask in composite_bases:
    for extra_name, extra_mask in extra_filters:
        label = f"{base_name} {extra_name}"
        mask = base_mask & extra_mask
        r = eval_filter(mask, label)
        if r and r['win_pct'] >= 55:
            phase3_results.append(r)

# Also test 2 extra filters on top of composite
for base_name, base_mask in composite_bases[:3]:
    for i in range(len(extra_filters)):
        for j in range(i+1, len(extra_filters)):
            label = f"{base_name} {extra_filters[i][0]} {extra_filters[j][0]}"
            mask = base_mask & extra_filters[i][1] & extra_filters[j][1]
            r = eval_filter(mask, label)
            if r and r['win_pct'] >= 60:
                phase3_results.append(r)

phase3_results.sort(key=lambda x: x['win_pct'], reverse=True)
print(f"\nFound {len(phase3_results)} combos with >= 55% win rate")
for r in phase3_results[:40]:
    cal = f"{r['calmar']:.2f}" if r['calmar'] is not None else "N/A"
    print(f"  {r['filter']:<65} n={r['trades']:>4}  win={r['win_pct']:>5.1f}%  avg={r['avg_ret']:>+6.1f}%  PF={r['profit_factor']:>5.2f}  Calmar={cal}")


# ============================================================
# PHASE 4: Risk-reward focused strategies
# ============================================================
print("\n\n" + "="*80)
print("PHASE 4: Risk-reward focused strategies")
print("="*80)

rr_combos = [
    ("Low Risk + RSI>=65", (df['risk_pct'] <= 8) & (df['rsi14'] >= 65)),
    ("Low Risk + RSI>=70", (df['risk_pct'] <= 8) & (df['rsi14'] >= 70)),
    ("Low Risk + Vol>=3x", (df['risk_pct'] <= 8) & (df['volume_ratio'] >= 3)),
    ("Low Risk + Vol>=5x", (df['risk_pct'] <= 8) & (df['volume_ratio'] >= 5)),
    ("Low Risk + BO>=3%", (df['risk_pct'] <= 8) & (df['breakout_pct'] >= 3)),
    ("Low Risk + BO>=5%", (df['risk_pct'] <= 8) & (df['breakout_pct'] >= 5)),
    ("Low Risk + VolTrend>=1.2", (df['risk_pct'] <= 8) & (df['vol_trend'] >= 1.2)),
    ("Low Risk + Mom60>=10", (df['risk_pct'] <= 8) & (df['mom_60d'] >= 10)),
    ("Low Risk + ATH>=90%", (df['risk_pct'] <= 8) & (df['ath_proximity'] >= 90)),
    ("Low Risk + EMA Triple", (df['risk_pct'] <= 8) & df['ema_triple']),
    ("R:R>=0.5 + Vol>=3x", (df['rr_ratio'] >= 0.5) & (df['volume_ratio'] >= 3)),
    ("R:R>=0.5 + RSI>=65", (df['rr_ratio'] >= 0.5) & (df['rsi14'] >= 65)),
    ("R:R>=0.5 + Mom60>=10", (df['rr_ratio'] >= 0.5) & (df['mom_60d'] >= 10)),
    ("R:R>=0.5 + VolTrend>=1.2", (df['rr_ratio'] >= 0.5) & (df['vol_trend'] >= 1.2)),
    ("R:R>=0.7 + Vol>=3x", (df['rr_ratio'] >= 0.7) & (df['volume_ratio'] >= 3)),
    # 3-filter RR combos
    ("LowRisk + RSI>=65 + Vol>=3x", (df['risk_pct'] <= 8) & (df['rsi14'] >= 65) & (df['volume_ratio'] >= 3)),
    ("LowRisk + RSI>=65 + BO>=3%", (df['risk_pct'] <= 8) & (df['rsi14'] >= 65) & (df['breakout_pct'] >= 3)),
    ("LowRisk + RSI>=65 + VolTrend>=1.2", (df['risk_pct'] <= 8) & (df['rsi14'] >= 65) & (df['vol_trend'] >= 1.2)),
    ("LowRisk + Vol>=3x + BO>=3%", (df['risk_pct'] <= 8) & (df['volume_ratio'] >= 3) & (df['breakout_pct'] >= 3)),
    ("LowRisk + Vol>=3x + Mom60>=10", (df['risk_pct'] <= 8) & (df['volume_ratio'] >= 3) & (df['mom_60d'] >= 10)),
    ("LowRisk + VolTrend>=1.2 + Mom60>=10", (df['risk_pct'] <= 8) & (df['vol_trend'] >= 1.2) & (df['mom_60d'] >= 10)),
    ("LowRisk + ATH>=90% + Vol>=3x", (df['risk_pct'] <= 8) & (df['ath_proximity'] >= 90) & (df['volume_ratio'] >= 3)),
    ("R:R>=0.5 + Vol>=3x + RSI>=65", (df['rr_ratio'] >= 0.5) & (df['volume_ratio'] >= 3) & (df['rsi14'] >= 65)),
    ("R:R>=0.5 + Vol>=3x + BO>=3%", (df['rr_ratio'] >= 0.5) & (df['volume_ratio'] >= 3) & (df['breakout_pct'] >= 3)),
    ("R:R>=0.5 + VolTrend>=1.2 + Mom60>=10", (df['rr_ratio'] >= 0.5) & (df['vol_trend'] >= 1.2) & (df['mom_60d'] >= 10)),
]

rr_results = []
for label, mask in rr_combos:
    r = eval_filter(mask, label)
    if r:
        rr_results.append(r)

rr_results.sort(key=lambda x: x['win_pct'], reverse=True)
for r in rr_results:
    cal = f"{r['calmar']:.2f}" if r['calmar'] is not None else "N/A"
    print(f"  {r['filter']:<50} n={r['trades']:>4}  win={r['win_pct']:>5.1f}%  avg={r['avg_ret']:>+6.1f}%  PF={r['profit_factor']:>5.2f}  Calmar={cal}")


# ============================================================
# PHASE 5: Detector-specific optimization
# ============================================================
print("\n\n" + "="*80)
print("PHASE 5: Detector-specific optimized filters")
print("="*80)

for det_name in ['darvas', 'flat']:
    print(f"\n--- {det_name.upper()} only ---")
    det_mask = df['detector'] == det_name

    det_filters = {
        'RSI>=65': df['rsi14'] >= 65,
        'RSI>=70': df['rsi14'] >= 70,
        'RSI>=75': df['rsi14'] >= 75,
        'Vol>=3x': df['volume_ratio'] >= 3,
        'Vol>=5x': df['volume_ratio'] >= 5,
        'Vol>=7x': df['volume_ratio'] >= 7,
        'BO>=3%': df['breakout_pct'] >= 3,
        'BO>=5%': df['breakout_pct'] >= 5,
        'BO>=7%': df['breakout_pct'] >= 7,
        'ATH>=90%': df['ath_proximity'] >= 90,
        'ATH>=95%': df['ath_proximity'] >= 95,
        'EMA20>50': df['ema20_above_50'] == 1,
        'Mom60>=10': df['mom_60d'] >= 10,
        'Mom60>=15': df['mom_60d'] >= 15,
        'Mom60>=20': df['mom_60d'] >= 20,
        'VolTrend>=1.2': df['vol_trend'] >= 1.2,
        'VolTrend>=1.5': df['vol_trend'] >= 1.5,
        'Box<=10%': df['box_height_pct'] <= 10,
        'Risk<=8%': df['risk_pct'] <= 8,
        'DualMomStrong': df['dual_mom_strong'],
    }

    det_results = []
    fnames = list(det_filters.keys())

    # 2-filter combos
    for i in range(len(fnames)):
        for j in range(i+1, len(fnames)):
            mask = det_mask & det_filters[fnames[i]] & det_filters[fnames[j]]
            r = eval_filter(mask, f"{det_name}+{fnames[i]}+{fnames[j]}")
            if r and r['win_pct'] >= 58:
                det_results.append(r)

    # 3-filter combos
    for i in range(len(fnames)):
        for j in range(i+1, len(fnames)):
            for k in range(j+1, len(fnames)):
                mask = det_mask & det_filters[fnames[i]] & det_filters[fnames[j]] & det_filters[fnames[k]]
                r = eval_filter(mask, f"{det_name}+{fnames[i]}+{fnames[j]}+{fnames[k]}")
                if r and r['win_pct'] >= 62:
                    det_results.append(r)

    det_results.sort(key=lambda x: x['win_pct'], reverse=True)
    print(f"  Found {len(det_results)} strategies with high win rate")
    for r in det_results[:25]:
        cal = f"{r['calmar']:.2f}" if r['calmar'] is not None else "N/A"
        print(f"    {r['filter']:<60} n={r['trades']:>4}  win={r['win_pct']:>5.1f}%  avg={r['avg_ret']:>+6.1f}%  PF={r['profit_factor']:>5.2f}  Calmar={cal}")


# ============================================================
# PHASE 6: Full exhaustive search with ALL building blocks
# ============================================================
print("\n\n" + "="*80)
print("PHASE 6: Exhaustive search with ALL building blocks (old + new)")
print("="*80)

all_filters = {
    # Original proven filters
    'RSI>=65': df['rsi14'] >= 65,
    'RSI>=70': df['rsi14'] >= 70,
    'RSI>=75': df['rsi14'] >= 75,
    'Vol>=3x': df['volume_ratio'] >= 3,
    'Vol>=5x': df['volume_ratio'] >= 5,
    'Vol>=7x': df['volume_ratio'] >= 7,
    'BO>=3%': df['breakout_pct'] >= 3,
    'BO>=5%': df['breakout_pct'] >= 5,
    'BO>=7%': df['breakout_pct'] >= 7,
    'ATH>=90%': df['ath_proximity'] >= 90,
    'ATH>=95%': df['ath_proximity'] >= 95,
    'EMA20>50': df['ema20_above_50'] == 1,
    'AbvEMA200': df['above_ema200'] == 1,
    'Mom60>=10': df['mom_60d'] >= 10,
    'Mom60>=15': df['mom_60d'] >= 15,
    'Mom60>=20': df['mom_60d'] >= 20,
    'Mom20>=10': df['mom_20d'] >= 10,
    'VolTrend>=1.2': df['vol_trend'] >= 1.2,
    'VolTrend>=1.5': df['vol_trend'] >= 1.5,
    'Box5-12%': df['box_height_pct'].between(5, 12),
    'Box<=10%': df['box_height_pct'] <= 10,
    'Darvas': df['detector'] == 'darvas',
    # NEW filters
    'Risk<=8%': df['risk_pct'] <= 8,
    'Risk<=10%': df['risk_pct'] <= 10,
    'R:R>=0.3': df['rr_ratio'] >= 0.3,
    'R:R>=0.5': df['rr_ratio'] >= 0.5,
    'DualMomStrong': df['dual_mom_strong'],
    'MomAccel': df['mom_accelerating'],
    'EMATriple': df['ema_triple'],
    'ATH>=98%': df['ath_proximity'] >= 98,
    'VolRSI>=5': df['vol_rsi_product'] >= 5,
    'VolRSI>=7': df['vol_rsi_product'] >= 7,
    'EMA50>200': df['ema50_above_200'] == 1,
}

fnames = list(all_filters.keys())
print(f"Testing {len(fnames)} building blocks in 2/3/4-filter combos...")

# 2-filter combos
print("\n--- 2-filter combos (win% >= 60%) ---")
results_2 = []
for i in range(len(fnames)):
    for j in range(i+1, len(fnames)):
        mask = all_filters[fnames[i]] & all_filters[fnames[j]]
        r = eval_filter(mask, f"{fnames[i]} + {fnames[j]}")
        if r and r['win_pct'] >= 60:
            results_2.append(r)

results_2.sort(key=lambda x: (-x['win_pct'], -x['profit_factor']))
print(f"Found {len(results_2)} combos with >= 60% win rate")
for r in results_2[:30]:
    cal = f"{r['calmar']:.2f}" if r['calmar'] is not None else "N/A"
    print(f"  {r['filter']:<50} n={r['trades']:>4}  win={r['win_pct']:>5.1f}%  avg={r['avg_ret']:>+6.1f}%  PF={r['profit_factor']:>5.2f}  Calmar={cal}")

# 3-filter combos
print("\n--- 3-filter combos (win% >= 62%) ---")
results_3 = []
for i in range(len(fnames)):
    for j in range(i+1, len(fnames)):
        for k in range(j+1, len(fnames)):
            mask = all_filters[fnames[i]] & all_filters[fnames[j]] & all_filters[fnames[k]]
            r = eval_filter(mask, f"{fnames[i]} + {fnames[j]} + {fnames[k]}")
            if r and r['win_pct'] >= 62:
                results_3.append(r)

results_3.sort(key=lambda x: (-x['win_pct'], -x['profit_factor']))
print(f"Found {len(results_3)} combos with >= 62% win rate")
for r in results_3[:40]:
    cal = f"{r['calmar']:.2f}" if r['calmar'] is not None else "N/A"
    print(f"  {r['filter']:<65} n={r['trades']:>4}  win={r['win_pct']:>5.1f}%  avg={r['avg_ret']:>+6.1f}%  PF={r['profit_factor']:>5.2f}  Calmar={cal}")

# Identify best building blocks for 4-filter search
best_blocks = set()
for r in results_3[:30]:
    for f in r['filter'].split(' + '):
        best_blocks.add(f.strip())
best_fnames = [f for f in fnames if f in best_blocks]
print(f"\n--- 4-filter combos from {len(best_fnames)} best blocks (win% >= 65%) ---")

results_4 = []
for combo in combinations(best_fnames, 4):
    mask = all_filters[combo[0]]
    for c in combo[1:]:
        mask = mask & all_filters[c]
    r = eval_filter(mask, ' + '.join(combo))
    if r and r['win_pct'] >= 65:
        results_4.append(r)

results_4.sort(key=lambda x: (-x['win_pct'], -x['profit_factor']))
print(f"Found {len(results_4)} combos with >= 65% win rate")
for r in results_4[:40]:
    cal = f"{r['calmar']:.2f}" if r['calmar'] is not None else "N/A"
    print(f"  {r['filter']:<75} n={r['trades']:>4}  win={r['win_pct']:>5.1f}%  avg={r['avg_ret']:>+6.1f}%  PF={r['profit_factor']:>5.2f}  Calmar={cal}")

# 5-filter from best blocks
if len(best_fnames) >= 5:
    print(f"\n--- 5-filter combos from best blocks (win% >= 67%) ---")
    results_5 = []
    for combo in combinations(best_fnames, 5):
        mask = all_filters[combo[0]]
        for c in combo[1:]:
            mask = mask & all_filters[c]
        r = eval_filter(mask, ' + '.join(combo))
        if r and r['win_pct'] >= 67:
            results_5.append(r)

    results_5.sort(key=lambda x: (-x['win_pct'], -x['profit_factor']))
    print(f"Found {len(results_5)} combos with >= 67% win rate")
    for r in results_5[:30]:
        cal = f"{r['calmar']:.2f}" if r['calmar'] is not None else "N/A"
        print(f"  {r['filter']:<85} n={r['trades']:>4}  win={r['win_pct']:>5.1f}%  avg={r['avg_ret']:>+6.1f}%  PF={r['profit_factor']:>5.2f}  Calmar={cal}")
else:
    results_5 = []


# ============================================================
# PHASE 7: GRAND SUMMARY - Best strategies at each tier
# ============================================================
print("\n\n" + "="*80)
print("PHASE 7: GRAND SUMMARY")
print("="*80)

all_results = results_2 + results_3 + results_4 + results_5 + phase3_results + rr_results
# Deduplicate by filter name
seen = set()
unique_results = []
for r in all_results:
    if r['filter'] not in seen:
        seen.add(r['filter'])
        unique_results.append(r)

unique_results.sort(key=lambda x: (-x['win_pct'], -x['profit_factor']))

# Show best at each win% tier, optimized for different objectives
for target in [75, 70, 67, 65, 63, 60, 58, 55]:
    tier = [r for r in unique_results if r['win_pct'] >= target]
    if not tier:
        print(f"\n  >= {target}% WIN RATE: NO STRATEGIES FOUND")
        continue

    # Best by PF
    best_pf = max(tier, key=lambda x: x['profit_factor'])
    # Best by Calmar
    best_cal = max(tier, key=lambda x: x['calmar'] if x['calmar'] is not None else 0)
    # Most trades (liquidity)
    most = max(tier, key=lambda x: x['trades'])
    # Best composite: win% * PF * sqrt(trades)
    best_composite = max(tier, key=lambda x: x['win_pct'] * x['profit_factor'] * (x['trades']**0.5) / 1000)

    print(f"\n  >= {target}% WIN RATE ({len(tier)} strategies):")
    print(f"    Best PF:        {best_pf['filter']:<60} n={best_pf['trades']:>4}  win={best_pf['win_pct']}%  PF={best_pf['profit_factor']}  Calmar={best_pf['calmar']}")
    print(f"    Best Calmar:    {best_cal['filter']:<60} n={best_cal['trades']:>4}  win={best_cal['win_pct']}%  PF={best_cal['profit_factor']}  Calmar={best_cal['calmar']}")
    print(f"    Most trades:    {most['filter']:<60} n={most['trades']:>4}  win={most['win_pct']}%  PF={most['profit_factor']}  Calmar={most['calmar']}")
    print(f"    Best overall:   {best_composite['filter']:<60} n={best_composite['trades']:>4}  win={best_composite['win_pct']}%  PF={best_composite['profit_factor']}  Calmar={best_composite['calmar']}")


# ============================================================
# PHASE 8: THE SHORTLIST - Strategies with 65%+ win, 30+ trades, Calmar > 1
# ============================================================
print("\n\n" + "="*80)
print("PHASE 8: THE SHORTLIST (65%+ win, 30+ trades, Calmar > 1.0)")
print("="*80)

shortlist = [r for r in unique_results
             if r['win_pct'] >= 65
             and r['trades'] >= 30
             and r['calmar'] is not None
             and r['calmar'] >= 1.0]

shortlist.sort(key=lambda x: (-x['calmar'], -x['profit_factor']))
print(f"\nFound {len(shortlist)} strategies meeting all criteria")
print(f"\n{'#':>2} {'Strategy':<70} {'N':>4} {'Win%':>5} {'AvgRet':>6} {'PF':>5} {'Calmar':>6} {'CAGR':>6} {'MaxDD':>5}")
print("-" * 110)
for i, r in enumerate(shortlist[:20], 1):
    print(f"{i:>2} {r['filter']:<70} {r['trades']:>4} {r['win_pct']:>4.1f}% {r['avg_ret']:>+5.1f}% {r['profit_factor']:>5.2f} {r['calmar']:>5.2f} {r['cagr']:>+5.1f}% {r['max_dd']:>4.1f}%")

# Also show 65%+ with 20+ trades (more strategies visible)
print(f"\n\n--- Extended: 65%+ win, 20+ trades ---")
extended = [r for r in unique_results
            if r['win_pct'] >= 65
            and r['trades'] >= 20
            and r['calmar'] is not None
            and r['calmar'] >= 0.5]

extended.sort(key=lambda x: (-x['win_pct'], -x['profit_factor']))
print(f"Found {len(extended)} strategies")
for i, r in enumerate(extended[:30], 1):
    print(f"{i:>2} {r['filter']:<70} {r['trades']:>4} {r['win_pct']:>4.1f}% {r['avg_ret']:>+5.1f}% {r['profit_factor']:>5.2f} {r['calmar']:>5.2f}")


print("\n\nDone!")
