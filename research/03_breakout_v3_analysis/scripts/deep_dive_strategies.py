"""
Deep-dive strategy search: Consolidation duration, indicator bands, K-of-N
confirmation, OR combinations, and 60+ building blocks.

Target: maximize trades at 65%+ win rate (or find the true frontier).

New dimensions vs previous searches:
  1. Consolidation days (computed from price data)
  2. Indicator BANDS (not just thresholds) - e.g., RSI 65-75 vs RSI 75+
  3. K-of-N confirmation counting (require 5/8 bullish signals, not all 8)
  4. OR combinations (union of complementary strategies)
  5. Volatility context (ATR%, BB width bands)
  6. Stochastic/MFI/CCI granular bands
  7. Detector-specific with weekly overlay
"""
import pandas as pd
import numpy as np
import sqlite3
import time
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# STEP 1: Load enhanced data + compute new features
# =====================================================================
df = pd.read_csv('breakout_analysis_enhanced.csv')
df['date'] = pd.to_datetime(df['date'])
print(f"Loaded {len(df)} trades, {len(df.columns)} columns")
print(f"Baseline: {(df['trade_return']>0).mean()*100:.1f}% win rate")

# --- Compute consolidation days from price data ---
print("\nComputing consolidation days from price data...")
conn = sqlite3.connect('backtest_data/market_data.db')
t0 = time.time()

consol_days = np.full(len(df), np.nan)
symbols = df['symbol'].unique()
total_syms = len(symbols)

for sym_idx, symbol in enumerate(symbols):
    if sym_idx % 100 == 0:
        print(f"  [{sym_idx+1}/{total_syms}] {symbol} ({time.time()-t0:.0f}s)")

    query = ("SELECT date, high, low, close FROM market_data_unified "
             "WHERE symbol = ? AND timeframe = 'day' ORDER BY date")
    daily = pd.read_sql(query, conn, params=(symbol,), parse_dates=['date'])
    if len(daily) < 60:
        continue

    daily = daily.set_index('date')
    highs = daily['high'].values
    lows = daily['low'].values
    closes = daily['close'].values
    dates = daily.index

    sym_mask = df['symbol'] == symbol
    sym_trades = df[sym_mask]

    for idx, row in sym_trades.iterrows():
        trade_date = pd.Timestamp(row['date'])
        # Find position in daily data
        pos = dates.searchsorted(trade_date, side='right') - 1
        if pos < 20:
            continue

        # Look back up to 120 days to find consolidation start
        # Consolidation = price staying within box_height_pct of a range
        box_ht = row['box_height_pct']
        if pd.isna(box_ht) or box_ht <= 0:
            continue

        breakout_price = closes[pos]
        range_high = breakout_price / (1 + row['breakout_pct']/100) if row['breakout_pct'] > 0 else breakout_price
        range_low = range_high * (1 - box_ht/100)

        # Count days price stayed within this range going backward
        days_in_range = 0
        for lookback in range(1, min(pos, 121)):
            p = pos - lookback
            if lows[p] >= range_low * 0.98 and highs[p] <= range_high * 1.02:
                days_in_range += 1
            else:
                break

        consol_days[idx] = days_in_range

df['consol_days'] = consol_days
valid_cd = df['consol_days'].notna().sum()
print(f"Consolidation days: {valid_cd}/{len(df)} valid ({valid_cd/len(df)*100:.0f}%)")
print(f"  Mean: {df['consol_days'].mean():.1f}, Median: {df['consol_days'].median():.1f}")
print(f"  Quartiles: {df['consol_days'].quantile([0.25, 0.5, 0.75]).values}")
conn.close()

# --- Derived features ---
# Confirmation count: how many bullish indicators agree simultaneously
bullish_cols = [
    ('ema20_above_50', 1), ('w_ema20_gt_50', 1), ('w_macd_positive', 1),
    ('w_above_ema50', 1), ('macd_bullish', 1), ('adx_bullish', 1),
    ('obv_bullish', 1), ('above_ema100', 1), ('ema20_rising', 1),
]
df['bullish_count'] = sum(
    (df[col] == val).astype(int) for col, val in bullish_cols
)

# Oscillator confirmation count: RSI7>70, StochK>70, MFI>60, CCI>100, WillR>-20
df['osc_confirm'] = (
    (df['rsi7'] > 70).astype(int) +
    (df['stoch_k'] > 70).astype(int) +
    (df['mfi'] > 60).astype(int) +
    (df['cci'] > 100).astype(int) +
    (df['williams_r'] > -20).astype(int)
)

# Momentum composite: normalized sum of momentum indicators
df['mom_composite'] = (
    df['mom_10d'].rank(pct=True) * 100 * 0.30 +
    df['mom_20d'].rank(pct=True) * 100 * 0.30 +
    df['mom_60d'].rank(pct=True) * 100 * 0.20 +
    df['rsi14'].rank(pct=True) * 100 * 0.20
)

# ATR-normalized box height (tighter consolidation relative to volatility)
df['box_atr_ratio'] = df['box_height_pct'] / df['atr_pct'].clip(lower=0.5)

# BB width categories
df['bb_narrow'] = df['bb_width'] < df['bb_width'].quantile(0.25)
df['bb_medium'] = df['bb_width'].between(
    df['bb_width'].quantile(0.25), df['bb_width'].quantile(0.75)
)

# Weekly trend strength: count of weekly bullish signals
df['weekly_bullish_count'] = (
    (df['w_above_ema20'] == 1).astype(int) +
    (df['w_above_ema50'] == 1).astype(int) +
    (df['w_ema20_gt_50'] == 1).astype(int) +
    (df['w_macd_positive'] == 1).astype(int) +
    (df['w_macd_bullish'] == 1).astype(int) +
    (df['w_rsi'] > 55).astype(int)
)

print(f"\nNew features computed:")
print(f"  bullish_count: {df['bullish_count'].describe().to_dict()}")
print(f"  osc_confirm:   {df['osc_confirm'].describe().to_dict()}")
print(f"  weekly_bullish: {df['weekly_bullish_count'].describe().to_dict()}")
print(f"  mom_composite:  mean={df['mom_composite'].mean():.1f}")
print(f"  box_atr_ratio:  mean={df['box_atr_ratio'].mean():.1f}")

# =====================================================================
# Evaluation function
# =====================================================================
def compute_calmar(subset):
    if len(subset) < 15:
        return None, None, None
    s = subset.sort_values('date')
    dates = s['date'].values
    returns = s['trade_return'].values / 100
    eq = [100.0]
    pnl = 0
    for r in returns:
        pnl += r
        eq.append(100 + pnl * 100)
    eq = np.array(eq)
    rm = np.maximum.accumulate(eq)
    dd = (eq - rm) / rm * 100
    max_dd = abs(dd.min())
    yrs = max((dates[-1] - dates[0]) / np.timedelta64(365, 'D'), 0.5)
    if eq[-1] <= 0:
        return None, None, None
    cagr = ((eq[-1]/eq[0])**(1/yrs)-1)*100
    cal = round(cagr/max_dd, 2) if max_dd > 0.1 else 999
    return round(cagr,1), round(max_dd,1), cal

def ev(mask, label, min_trades=20):
    s = df[mask].copy()
    n = len(s)
    if n < min_trades:
        return None
    wp = (s['trade_return']>0).mean()*100
    ar = s['trade_return'].mean()
    wins = s[s['trade_return']>0]['trade_return']
    losses = s[s['trade_return']<=0]['trade_return']
    tg = wins.sum() if len(wins)>0 else 0
    tl = abs(losses.sum()) if len(losses)>0 else 0.01
    pf = round(tg/tl, 2)
    cagr, mdd, cal = compute_calmar(s)
    return {
        'filter': label, 'trades': n, 'win_pct': round(wp,1),
        'avg_ret': round(ar,1), 'profit_factor': pf,
        'cagr': cagr, 'max_dd': mdd, 'calmar': cal,
    }

# =====================================================================
# PHASE 1: Consolidation days analysis
# =====================================================================
print(f"\n{'='*80}")
print("PHASE 1: Consolidation Duration Analysis")
print("="*80)

cd_results = []
for days_min, days_max, label in [
    (5, 15, 'Consol 5-15d (short)'),
    (15, 30, 'Consol 15-30d (medium)'),
    (30, 60, 'Consol 30-60d (long)'),
    (60, 120, 'Consol 60-120d (very long)'),
    (20, 999, 'Consol >= 20d'),
    (30, 999, 'Consol >= 30d'),
    (40, 999, 'Consol >= 40d'),
    (50, 999, 'Consol >= 50d'),
    (10, 25, 'Consol 10-25d'),
    (25, 50, 'Consol 25-50d'),
]:
    mask = df['consol_days'].between(days_min, days_max)
    r = ev(mask, label)
    if r:
        cd_results.append(r)
        cal = f"{r['calmar']:.2f}" if r['calmar'] else "N/A"
        print(f"  {label:<30} n={r['trades']:>5}  win={r['win_pct']:>5.1f}%  PF={r['profit_factor']:>5.2f}  Calmar={cal}")

# Consolidation days + other filters
print("\n--- Consolidation days + best filters ---")
cd_combos = []
for cd_label, cd_mask in [
    ('CD>=20', df['consol_days'] >= 20),
    ('CD>=30', df['consol_days'] >= 30),
    ('CD>=40', df['consol_days'] >= 40),
    ('CD 15-30', df['consol_days'].between(15, 30)),
    ('CD 20-50', df['consol_days'].between(20, 50)),
]:
    for f_label, f_mask in [
        ('VolTr>=1.2 + ATH>=85%', (df['vol_trend']>=1.2) & (df['ath_proximity']>=85)),
        ('VolTr>=1.2 + ATH>=90%', (df['vol_trend']>=1.2) & (df['ath_proximity']>=90)),
        ('VolTr>=1.2 + wEMA20>50', (df['vol_trend']>=1.2) & (df['w_ema20_gt_50']==1)),
        ('Vol>=3x + ATH>=85%', (df['volume_ratio']>=3) & (df['ath_proximity']>=85)),
        ('Vol>=3x + ATH>=90%', (df['volume_ratio']>=3) & (df['ath_proximity']>=90)),
        ('BO>=3% + VolTr>=1.2', (df['breakout_pct']>=3) & (df['vol_trend']>=1.2)),
        ('RSI>=65 + VolTr>=1.2', (df['rsi14']>=65) & (df['vol_trend']>=1.2)),
        ('wEMA20>50 + ATH>=85%', (df['w_ema20_gt_50']==1) & (df['ath_proximity']>=85)),
        ('wEMA20>50 + RSI7>70', (df['w_ema20_gt_50']==1) & (df['rsi7']>70)),
        ('VolTr>=1.2 + RSI7>70', (df['vol_trend']>=1.2) & (df['rsi7']>70)),
        ('BO>=3% + wEMA20>50 + ATH>=85%', (df['breakout_pct']>=3) & (df['w_ema20_gt_50']==1) & (df['ath_proximity']>=85)),
        ('VolTr>=1.2 + wEMA20>50 + ATH>=85%', (df['vol_trend']>=1.2) & (df['w_ema20_gt_50']==1) & (df['ath_proximity']>=85)),
        ('VolTr>=1.2 + wEMA20>50 + RSI7>80', (df['vol_trend']>=1.2) & (df['w_ema20_gt_50']==1) & (df['rsi7']>80)),
    ]:
        label = f"{cd_label} + {f_label}"
        mask = cd_mask & f_mask
        r = ev(mask, label, min_trades=30)
        if r and r['win_pct'] >= 58:
            cd_combos.append(r)

cd_combos.sort(key=lambda x: (-x['win_pct'], -x['trades']))
print(f"Found {len(cd_combos)} combos with consol days + filters at 58%+")
for r in cd_combos[:30]:
    cal = f"{r['calmar']:.2f}" if r['calmar'] else "N/A"
    print(f"  {r['filter']:<65} n={r['trades']:>5}  win={r['win_pct']:>5.1f}%  PF={r['profit_factor']:>5.2f}  Calmar={cal}")

# =====================================================================
# PHASE 2: K-of-N Confirmation (require K bullish signals out of N)
# =====================================================================
print(f"\n{'='*80}")
print("PHASE 2: K-of-N Confirmation Counting")
print("="*80)

print("\n--- Trend confirmation count (9 signals) ---")
for k in range(5, 10):
    mask = df['bullish_count'] >= k
    r = ev(mask, f"Trend confirm >= {k}/9")
    if r:
        cal = f"{r['calmar']:.2f}" if r['calmar'] else "N/A"
        print(f"  {r['filter']:<35} n={r['trades']:>5}  win={r['win_pct']:>5.1f}%  PF={r['profit_factor']:>5.2f}  Calmar={cal}")

print("\n--- Oscillator confirmation count (5 signals) ---")
for k in range(2, 6):
    mask = df['osc_confirm'] >= k
    r = ev(mask, f"Osc confirm >= {k}/5")
    if r:
        cal = f"{r['calmar']:.2f}" if r['calmar'] else "N/A"
        print(f"  {r['filter']:<35} n={r['trades']:>5}  win={r['win_pct']:>5.1f}%  PF={r['profit_factor']:>5.2f}  Calmar={cal}")

print("\n--- Weekly bullish count (6 signals) ---")
for k in range(3, 7):
    mask = df['weekly_bullish_count'] >= k
    r = ev(mask, f"Weekly bullish >= {k}/6")
    if r:
        cal = f"{r['calmar']:.2f}" if r['calmar'] else "N/A"
        print(f"  {r['filter']:<35} n={r['trades']:>5}  win={r['win_pct']:>5.1f}%  PF={r['profit_factor']:>5.2f}  Calmar={cal}")

# Combined K-of-N with basic filters
print("\n--- K-of-N + basic filters ---")
kn_results = []
for kn_label, kn_mask in [
    ('Trend>=7', df['bullish_count'] >= 7),
    ('Trend>=8', df['bullish_count'] >= 8),
    ('Trend>=9', df['bullish_count'] >= 9),
    ('Osc>=4', df['osc_confirm'] >= 4),
    ('Osc>=5', df['osc_confirm'] >= 5),
    ('Weekly>=4', df['weekly_bullish_count'] >= 4),
    ('Weekly>=5', df['weekly_bullish_count'] >= 5),
    ('Weekly>=6', df['weekly_bullish_count'] >= 6),
]:
    for f_label, f_mask in [
        ('ATH>=85%', df['ath_proximity'] >= 85),
        ('ATH>=90%', df['ath_proximity'] >= 90),
        ('VolTr>=1.2', df['vol_trend'] >= 1.2),
        ('BO>=3%', df['breakout_pct'] >= 3),
        ('Vol>=3x', df['volume_ratio'] >= 3),
        ('RSI7>70', df['rsi7'] > 70),
        ('RSI7>80', df['rsi7'] > 80),
        ('ATH>=85% + VolTr>=1.2', (df['ath_proximity']>=85) & (df['vol_trend']>=1.2)),
        ('ATH>=90% + VolTr>=1.2', (df['ath_proximity']>=90) & (df['vol_trend']>=1.2)),
        ('ATH>=85% + BO>=3%', (df['ath_proximity']>=85) & (df['breakout_pct']>=3)),
        ('BO>=3% + VolTr>=1.2', (df['breakout_pct']>=3) & (df['vol_trend']>=1.2)),
        ('Vol>=3x + ATH>=85%', (df['volume_ratio']>=3) & (df['ath_proximity']>=85)),
    ]:
        label = f"{kn_label} + {f_label}"
        mask = kn_mask & f_mask
        r = ev(mask, label, min_trades=50)
        if r and r['win_pct'] >= 58:
            kn_results.append(r)

kn_results.sort(key=lambda x: (-x['trades'], -x['win_pct']))
print(f"Found {len(kn_results)} K-of-N combos at 58%+")
for r in kn_results[:40]:
    cal = f"{r['calmar']:.2f}" if r['calmar'] else "N/A"
    print(f"  {r['filter']:<55} n={r['trades']:>5}  win={r['win_pct']:>5.1f}%  PF={r['profit_factor']:>5.2f}  Calmar={cal}")

# =====================================================================
# PHASE 3: Indicator BANDS (not just single thresholds)
# =====================================================================
print(f"\n{'='*80}")
print("PHASE 3: Indicator Bands (granular ranges)")
print("="*80)

# Define bands for key indicators
band_filters = {}

# RSI14 bands
for lo, hi, label in [(55,65,'RSI 55-65'), (60,70,'RSI 60-70'), (65,75,'RSI 65-75'), (70,80,'RSI 70-80'), (75,90,'RSI 75+')]:
    band_filters[label] = df['rsi14'].between(lo, hi) if hi < 90 else (df['rsi14'] >= lo)

# RSI7 bands
for lo, hi, label in [(60,70,'RSI7 60-70'), (70,80,'RSI7 70-80'), (80,90,'RSI7 80-90'), (90,100,'RSI7 90+')]:
    band_filters[label] = df['rsi7'].between(lo, hi) if hi < 100 else (df['rsi7'] >= lo)

# Stochastic K bands
for lo, hi, label in [(70,80,'StK 70-80'), (80,90,'StK 80-90'), (90,100,'StK 90+')]:
    band_filters[label] = df['stoch_k'].between(lo, hi) if hi < 100 else (df['stoch_k'] >= lo)

# ADX bands
for lo, hi, label in [(20,30,'ADX 20-30'), (30,40,'ADX 30-40'), (40,100,'ADX 40+')]:
    band_filters[label] = df['adx'].between(lo, hi) if hi < 100 else (df['adx'] >= lo)

# MFI bands
for lo, hi, label in [(50,65,'MFI 50-65'), (65,80,'MFI 65-80'), (80,100,'MFI 80+')]:
    band_filters[label] = df['mfi'].between(lo, hi) if hi < 100 else (df['mfi'] >= lo)

# ATR% bands (volatility)
for lo, hi, label in [(0,2,'ATR <2%'), (2,3,'ATR 2-3%'), (3,5,'ATR 3-5%'), (5,15,'ATR 5%+')]:
    band_filters[label] = df['atr_pct'].between(lo, hi) if hi < 15 else (df['atr_pct'] >= lo)

# BB width bands
q25, q50, q75 = df['bb_width'].quantile([0.25, 0.50, 0.75]).values
band_filters['BBw narrow'] = df['bb_width'] < q25
band_filters['BBw medium'] = df['bb_width'].between(q25, q50)
band_filters['BBw wide'] = df['bb_width'] > q75
band_filters['BB squeeze'] = df['bb_squeeze'] == 1

# Box height bands
for lo, hi, label in [(2,6,'Box 2-6%'), (5,10,'Box 5-10%'), (6,12,'Box 6-12%'), (10,15,'Box 10-15%'), (15,20,'Box 15-20%')]:
    band_filters[label] = df['box_height_pct'].between(lo, hi)

# Breakout % bands
for lo, hi, label in [(1,3,'BO 1-3%'), (3,5,'BO 3-5%'), (5,10,'BO 5-10%'), (10,100,'BO 10%+')]:
    band_filters[label] = df['breakout_pct'].between(lo, hi) if hi < 100 else (df['breakout_pct'] >= lo)

# Volume bands
for lo, hi, label in [(1.5,3,'Vol 1.5-3x'), (3,5,'Vol 3-5x'), (5,7,'Vol 5-7x'), (7,10,'Vol 7-10x'), (10,35,'Vol 10x+')]:
    band_filters[label] = df['volume_ratio'].between(lo, hi) if hi < 35 else (df['volume_ratio'] >= lo)

# Vol trend bands
for lo, hi, label in [(0.8,1.0,'VT 0.8-1.0'), (1.0,1.2,'VT 1.0-1.2'), (1.2,1.5,'VT 1.2-1.5'), (1.5,4.5,'VT 1.5+')]:
    band_filters[label] = df['vol_trend'].between(lo, hi) if hi < 4.5 else (df['vol_trend'] >= lo)

# ATH bands
for lo, hi, label in [(70,80,'ATH 70-80%'), (80,90,'ATH 80-90%'), (85,95,'ATH 85-95%'), (90,100,'ATH 90-100%'), (95,100,'ATH 95-100%')]:
    band_filters[label] = df['ath_proximity'].between(lo, hi+0.01)

# CCI bands
for lo, label in [(100,'CCI>100'), (150,'CCI>150'), (200,'CCI>200')]:
    band_filters[label] = df['cci'] > lo

# Williams %R bands
for lo, label in [(-20,'WR>-20'), (-30,'WR>-30'), (-10,'WR>-10')]:
    band_filters[label] = df['williams_r'] > lo

# Weekly RSI bands
for lo, label in [(50,'wRSI>50'), (55,'wRSI>55'), (60,'wRSI>60'), (65,'wRSI>65')]:
    band_filters[label] = df['w_rsi'] > lo

# Weekly filters
band_filters['wEMA20>50'] = df['w_ema20_gt_50'] == 1
band_filters['wMACD+'] = df['w_macd_positive'] == 1
band_filters['wMACDbull'] = df['w_macd_bullish'] == 1
band_filters['wEMA50'] = df['w_above_ema50'] == 1

# EMA alignment
band_filters['EMA20>50'] = df['ema20_above_50'] == 1
band_filters['AbvEMA100'] = df['above_ema100'] == 1
band_filters['EMA20rise'] = df['ema20_rising'] == 1

# Consolidation days bands
band_filters['CD>=20'] = df['consol_days'] >= 20
band_filters['CD>=30'] = df['consol_days'] >= 30
band_filters['CD>=40'] = df['consol_days'] >= 40
band_filters['CD 15-30'] = df['consol_days'].between(15, 30)
band_filters['CD 20-50'] = df['consol_days'].between(20, 50)
band_filters['CD 30-60'] = df['consol_days'].between(30, 60)

# Momentum composites
band_filters['MomComp>=70'] = df['mom_composite'] >= 70
band_filters['MomComp>=75'] = df['mom_composite'] >= 75
band_filters['MomComp>=80'] = df['mom_composite'] >= 80

# Momentum thresholds
band_filters['Mom10>=3'] = df['mom_10d'] >= 3
band_filters['Mom10>=5'] = df['mom_10d'] >= 5
band_filters['Mom10>=8'] = df['mom_10d'] >= 8
band_filters['Mom20>=3'] = df['mom_20d'] >= 3
band_filters['Mom20>=5'] = df['mom_20d'] >= 5
band_filters['Mom60>=10'] = df['mom_60d'] >= 10
band_filters['Mom60>=15'] = df['mom_60d'] >= 15

# Detector specific
band_filters['Darvas'] = df['detector'] == 'darvas'
band_filters['Flat'] = df['detector'] == 'flat'

# K-of-N
band_filters['Trend>=7'] = df['bullish_count'] >= 7
band_filters['Trend>=8'] = df['bullish_count'] >= 8
band_filters['Osc>=4'] = df['osc_confirm'] >= 4
band_filters['Osc>=5'] = df['osc_confirm'] >= 5
band_filters['Weekly>=4'] = df['weekly_bullish_count'] >= 4
band_filters['Weekly>=5'] = df['weekly_bullish_count'] >= 5

fnames = list(band_filters.keys())
print(f"\n{len(fnames)} building blocks defined")

# Quick singles pass
print("\n--- Band filter singles (sorted by win%) ---")
singles = []
for fname in fnames:
    r = ev(band_filters[fname], fname, min_trades=50)
    if r and r['win_pct'] >= 50:
        singles.append(r)
singles.sort(key=lambda x: -x['win_pct'])
for r in singles[:40]:
    cal = f"{r['calmar']:.2f}" if r['calmar'] else "N/A"
    print(f"  {r['filter']:<25} n={r['trades']:>5}  win={r['win_pct']:>5.1f}%  PF={r['profit_factor']:>5.2f}  Calmar={cal}")

# =====================================================================
# PHASE 4: Systematic 2-filter search (wide net)
# =====================================================================
print(f"\n{'='*80}")
print(f"PHASE 4: 2-filter combos (55%+, 200+ trades) [{time.time()-t0:.0f}s]")
print("="*80)

results_2 = []
for i in range(len(fnames)):
    for j in range(i+1, len(fnames)):
        mask = band_filters[fnames[i]] & band_filters[fnames[j]]
        r = ev(mask, f"{fnames[i]} + {fnames[j]}", min_trades=200)
        if r and r['win_pct'] >= 55:
            results_2.append(r)

results_2.sort(key=lambda x: (-x['win_pct'], -x['trades']))
print(f"Found {len(results_2)} combos")
for r in results_2[:30]:
    cal = f"{r['calmar']:.2f}" if r['calmar'] else "N/A"
    print(f"  {r['filter']:<55} n={r['trades']:>5}  win={r['win_pct']:>5.1f}%  PF={r['profit_factor']:>5.2f}  Calmar={cal}")

# =====================================================================
# PHASE 5: 3-filter combos (targeting 250+ at 60%+)
# =====================================================================
print(f"\n{'='*80}")
print(f"PHASE 5: 3-filter combos (60%+, 150+ trades) [{time.time()-t0:.0f}s]")
print("="*80)

# Use blocks that appear in top 2-filter results + proven blocks
top_blocks = set()
for r in results_2[:60]:
    for f in r['filter'].split(' + '):
        top_blocks.add(f.strip())

# Always include strong blocks
for f in ['ATH 90-100%', 'ATH 85-95%', 'ATH>=85%' if 'ATH>=85%' in fnames else 'ATH 80-90%',
          'VolTr>=1.2', 'VT 1.2-1.5', 'VT 1.5+', 'BO>=3%', 'BO 3-5%', 'BO 5-10%',
          'Vol 3-5x', 'Vol 5-7x', 'Vol>=3x' if 'Vol>=3x' in fnames else 'Vol 3-5x',
          'wEMA20>50', 'wMACD+', 'wEMA50', 'wRSI>55', 'wRSI>60',
          'RSI7 70-80', 'RSI7 80-90', 'RSI7 90+', 'RSI 65-75', 'RSI 70-80', 'RSI 75+',
          'WR>-20', 'WR>-30', 'MFI 65-80', 'MFI 80+',
          'StK 80-90', 'StK 90+', 'CCI>100', 'CCI>150',
          'ADX 30-40', 'ADX 40+', 'BB squeeze',
          'Mom10>=5', 'Mom10>=3', 'Mom20>=5', 'Mom60>=10',
          'EMA20>50', 'AbvEMA100', 'EMA20rise',
          'CD>=20', 'CD>=30', 'CD 20-50',
          'Trend>=7', 'Trend>=8', 'Osc>=4', 'Osc>=5',
          'Weekly>=4', 'Weekly>=5',
          'MomComp>=70', 'MomComp>=75',
          'Box 5-10%', 'Box 6-12%',
          'Darvas', 'Flat',
          'BBw narrow', 'ATR <2%']:
    if f in fnames:
        top_blocks.add(f)

top_fnames = [f for f in fnames if f in top_blocks]
print(f"Using {len(top_fnames)} blocks for 3-filter search")

results_3 = []
for combo in combinations(top_fnames, 3):
    mask = band_filters[combo[0]]
    for c in combo[1:]:
        mask = mask & band_filters[c]
    r = ev(mask, ' + '.join(combo), min_trades=150)
    if r and r['win_pct'] >= 60:
        results_3.append(r)

results_3.sort(key=lambda x: (-x['win_pct'], -x['trades']))
print(f"Found {len(results_3)} combos [{time.time()-t0:.0f}s]")
for r in results_3[:40]:
    cal = f"{r['calmar']:.2f}" if r['calmar'] else "N/A"
    print(f"  {r['filter']:<65} n={r['trades']:>5}  win={r['win_pct']:>5.1f}%  PF={r['profit_factor']:>5.2f}  Calmar={cal}")

# =====================================================================
# PHASE 6: 4-filter combos (targeting 250+ at 62%+)
# =====================================================================
print(f"\n{'='*80}")
print(f"PHASE 6: 4-filter combos (62%+, 150+ trades) [{time.time()-t0:.0f}s]")
print("="*80)

top_blocks_3 = set()
for r in results_3[:50]:
    for f in r['filter'].split(' + '):
        top_blocks_3.add(f.strip())
top_fnames_3 = [f for f in top_fnames if f in top_blocks_3]
print(f"Using {len(top_fnames_3)} blocks from phase 5")

results_4 = []
for combo in combinations(top_fnames_3, 4):
    mask = band_filters[combo[0]]
    for c in combo[1:]:
        mask = mask & band_filters[c]
    r = ev(mask, ' + '.join(combo), min_trades=150)
    if r and r['win_pct'] >= 62:
        results_4.append(r)

results_4.sort(key=lambda x: (-x['win_pct'], -x['trades']))
print(f"Found {len(results_4)} combos [{time.time()-t0:.0f}s]")
for r in results_4[:40]:
    cal = f"{r['calmar']:.2f}" if r['calmar'] else "N/A"
    print(f"  {r['filter']:<75} n={r['trades']:>5}  win={r['win_pct']:>5.1f}%  PF={r['profit_factor']:>5.2f}  Calmar={cal}")

# =====================================================================
# PHASE 7: 5-filter combos (targeting 200+ at 64%+)
# =====================================================================
print(f"\n{'='*80}")
print(f"PHASE 7: 5-filter combos (64%+, 150+ trades) [{time.time()-t0:.0f}s]")
print("="*80)

top_blocks_4 = set()
for r in (results_4[:40] if results_4 else results_3[:40]):
    for f in r['filter'].split(' + '):
        top_blocks_4.add(f.strip())
top_fnames_4 = [f for f in top_fnames if f in top_blocks_4]
print(f"Using {len(top_fnames_4)} blocks")

results_5 = []
if len(top_fnames_4) >= 5:
    for combo in combinations(top_fnames_4, 5):
        mask = band_filters[combo[0]]
        for c in combo[1:]:
            mask = mask & band_filters[c]
        r = ev(mask, ' + '.join(combo), min_trades=150)
        if r and r['win_pct'] >= 64:
            results_5.append(r)

results_5.sort(key=lambda x: (-x['win_pct'], -x['trades']))
print(f"Found {len(results_5)} combos [{time.time()-t0:.0f}s]")
for r in results_5[:30]:
    cal = f"{r['calmar']:.2f}" if r['calmar'] else "N/A"
    print(f"  {r['filter']:<80} n={r['trades']:>5}  win={r['win_pct']:>5.1f}%  PF={r['profit_factor']:>5.2f}  Calmar={cal}")

# =====================================================================
# PHASE 8: 6-filter combos (targeting 200+ at 65%+)
# =====================================================================
print(f"\n{'='*80}")
print(f"PHASE 8: 6-filter combos (65%+, 150+ trades) [{time.time()-t0:.0f}s]")
print("="*80)

top_blocks_5 = set()
for r in (results_5[:30] if results_5 else results_4[:30]):
    for f in r['filter'].split(' + '):
        top_blocks_5.add(f.strip())
top_fnames_5 = [f for f in top_fnames if f in top_blocks_5]
print(f"Using {len(top_fnames_5)} blocks")

results_6 = []
if len(top_fnames_5) >= 6:
    for combo in combinations(top_fnames_5, 6):
        mask = band_filters[combo[0]]
        for c in combo[1:]:
            mask = mask & band_filters[c]
        r = ev(mask, ' + '.join(combo), min_trades=150)
        if r and r['win_pct'] >= 65:
            results_6.append(r)

results_6.sort(key=lambda x: (-x['win_pct'], -x['trades']))
print(f"Found {len(results_6)} combos [{time.time()-t0:.0f}s]")
for r in results_6[:25]:
    cal = f"{r['calmar']:.2f}" if r['calmar'] else "N/A"
    print(f"  {r['filter']:<90} n={r['trades']:>5}  win={r['win_pct']:>5.1f}%  PF={r['profit_factor']:>5.2f}  Calmar={cal}")

# =====================================================================
# PHASE 9: OR Combinations (Union of strategies for more trades)
# =====================================================================
print(f"\n{'='*80}")
print(f"PHASE 9: OR Combinations (Union of strategies) [{time.time()-t0:.0f}s]")
print("="*80)

# Define high-conviction strategies to combine with OR
strategy_masks = {
    'ALPHA': (df['rsi14']>=75) & (df['volume_ratio']>=3) & (df['ema20_above_50']==1),
    'T1A_ATH90': (df['breakout_pct']>=3) & (df['vol_trend']>=1.2) & (df['ath_proximity']>=90) & (df['w_ema20_gt_50']==1) & (df['rsi7']>80),
    'T1B_WillR': (df['breakout_pct']>=3) & (df['vol_trend']>=1.2) & (df['ath_proximity']>=85) & (df['ema20_above_50']==1) & (df['w_ema20_gt_50']==1) & (df['williams_r']>-20),
    'MomVol': (df['mom_60d']>=15) & (df['vol_trend']>=1.2) & (df['ath_proximity']>=90) & (df['volume_ratio']>=3),
    'Calmar': (df['volume_ratio']>=5) & (df['vol_trend']>=1.2) & (df['ema20_above_50']==1) & (df['rsi7']>80),
    'BBupper_Mom': (df['bb_pct_b']>1.0) & (df['mom_60d']>=15) & (df['vol_trend']>=1.2) & (df['ath_proximity']>=90),
    'StochHigh': (df['stoch_k']>90) & (df['vol_trend']>=1.2) & (df['ath_proximity']>=90) & (df['w_ema20_gt_50']==1),
    'MFI_ATH': (df['mfi']>80) & (df['ath_proximity']>=90) & (df['vol_trend']>=1.2) & (df['w_ema20_gt_50']==1),
    'RSI7_Vol': (df['rsi7']>85) & (df['volume_ratio']>=3) & (df['ath_proximity']>=85) & (df['w_ema20_gt_50']==1),
    'ADXstrong': (df['adx']>35) & (df['ath_proximity']>=90) & (df['vol_trend']>=1.2) & (df['w_ema20_gt_50']==1),
    'CCI_Mom': (df['cci']>200) & (df['vol_trend']>=1.2) & (df['ath_proximity']>=85) & (df['w_ema20_gt_50']==1),
    'Osc5_ATH': (df['osc_confirm']>=5) & (df['ath_proximity']>=90) & (df['vol_trend']>=1.2),
    'Trend9_BO': (df['bullish_count']>=9) & (df['breakout_pct']>=3) & (df['vol_trend']>=1.2),
    'WeeklyAll': (df['weekly_bullish_count']>=6) & (df['ath_proximity']>=85) & (df['vol_trend']>=1.2),
}

# Evaluate each individually first
print("\n--- Individual strategies ---")
strat_results = {}
for sname, smask in strategy_masks.items():
    r = ev(smask, sname, min_trades=10)
    if r:
        strat_results[sname] = r
        cal = f"{r['calmar']:.2f}" if r['calmar'] else "N/A"
        print(f"  {sname:<20} n={r['trades']:>5}  win={r['win_pct']:>5.1f}%  PF={r['profit_factor']:>5.2f}  Calmar={cal}")

# Test OR combinations of 2 strategies
print("\n--- OR combinations of 2 strategies ---")
or_results = []
snames = list(strategy_masks.keys())
for i in range(len(snames)):
    for j in range(i+1, len(snames)):
        union_mask = strategy_masks[snames[i]] | strategy_masks[snames[j]]
        label = f"{snames[i]} OR {snames[j]}"
        r = ev(union_mask, label, min_trades=100)
        if r and r['win_pct'] >= 60:
            or_results.append(r)

or_results.sort(key=lambda x: (-x['trades'], -x['win_pct']))
print(f"Found {len(or_results)} OR combos at 60%+")
for r in or_results[:30]:
    cal = f"{r['calmar']:.2f}" if r['calmar'] else "N/A"
    print(f"  {r['filter']:<50} n={r['trades']:>5}  win={r['win_pct']:>5.1f}%  PF={r['profit_factor']:>5.2f}  Calmar={cal}")

# Test OR combinations of 3 strategies
print("\n--- OR combinations of 3 strategies ---")
or3_results = []
for i in range(len(snames)):
    for j in range(i+1, len(snames)):
        for k in range(j+1, len(snames)):
            union_mask = strategy_masks[snames[i]] | strategy_masks[snames[j]] | strategy_masks[snames[k]]
            label = f"{snames[i]} OR {snames[j]} OR {snames[k]}"
            r = ev(union_mask, label, min_trades=200)
            if r and r['win_pct'] >= 60:
                or3_results.append(r)

or3_results.sort(key=lambda x: (-x['trades'], -x['win_pct']))
print(f"Found {len(or3_results)} triple-OR combos at 60%+")
for r in or3_results[:30]:
    cal = f"{r['calmar']:.2f}" if r['calmar'] else "N/A"
    print(f"  {r['filter']:<65} n={r['trades']:>5}  win={r['win_pct']:>5.1f}%  PF={r['profit_factor']:>5.2f}  Calmar={cal}")

# =====================================================================
# PHASE 10: Detector-specific + weekly + new indicators
# =====================================================================
print(f"\n{'='*80}")
print(f"PHASE 10: Detector-specific deep dive [{time.time()-t0:.0f}s]")
print("="*80)

for det_name in ['darvas', 'flat']:
    print(f"\n--- {det_name.upper()} ---")
    det_mask = df['detector'] == det_name
    det_filters = {}
    for fname in top_fnames:
        if fname not in ['Darvas', 'Flat']:
            det_filters[fname] = det_mask & band_filters[fname]

    det_fnames = list(det_filters.keys())
    det_results = []

    # 3-filter combos for detector
    for combo in combinations(det_fnames[:35], 3):
        mask = det_filters[combo[0]]
        for c in combo[1:]:
            mask = mask & band_filters[c]
        mask = mask & det_mask
        r = ev(mask, f"[{det_name}] {' + '.join(combo)}", min_trades=50)
        if r and r['win_pct'] >= 62:
            det_results.append(r)

    det_results.sort(key=lambda x: (-x['win_pct'], -x['trades']))
    print(f"  Found {len(det_results)} strategies at 62%+")
    for r in det_results[:20]:
        cal = f"{r['calmar']:.2f}" if r['calmar'] else "N/A"
        print(f"    {r['filter']:<70} n={r['trades']:>5}  win={r['win_pct']:>5.1f}%  PF={r['profit_factor']:>5.2f}  Calmar={cal}")

# =====================================================================
# GRAND SUMMARY
# =====================================================================
print(f"\n\n{'='*80}")
print(f"GRAND SUMMARY [{time.time()-t0:.0f}s]")
print("="*80)

all_r = results_2 + results_3 + results_4 + results_5 + results_6 + cd_combos + kn_results + or_results + or3_results
seen = set()
unique = []
for r in all_r:
    if r['filter'] not in seen:
        seen.add(r['filter'])
        unique.append(r)

# === Frontier: Best win% at each trade-count tier ===
print("\n--- FRONTIER: Best win% at each trade-count tier ---")
for min_n in [600, 500, 400, 350, 300, 250, 200, 150, 100]:
    tier = [r for r in unique if r['trades'] >= min_n]
    if tier:
        best = max(tier, key=lambda x: x['win_pct'])
        cal = f"{best['calmar']:.2f}" if best['calmar'] else "N/A"
        print(f"  {min_n:>4}+ trades: {best['filter']:<65} n={best['trades']:>5}  win={best['win_pct']:>5.1f}%  PF={best['profit_factor']:>5.2f}  Calmar={cal}")

# === Best Calmar at each tier ===
print("\n--- Best Calmar at each trade-count tier ---")
for min_n in [500, 400, 300, 250, 200, 150, 100]:
    tier = [r for r in unique if r['trades'] >= min_n and r['calmar'] and r['calmar'] > 0]
    if tier:
        best = max(tier, key=lambda x: x['calmar'])
        print(f"  {min_n:>4}+ trades: {best['filter']:<65} n={best['trades']:>5}  win={best['win_pct']:>5.1f}%  PF={best['profit_factor']:>5.2f}  Calmar={best['calmar']:.2f}")

# === 250+ trades target ===
print("\n--- TARGET: 250+ trades, 60%+ win ---")
target = [r for r in unique if r['trades'] >= 250 and r['win_pct'] >= 60]
target.sort(key=lambda x: (-x['win_pct'], -x['profit_factor']))
for r in target[:20]:
    cal = f"{r['calmar']:.2f}" if r['calmar'] else "N/A"
    print(f"  {r['filter']:<70} n={r['trades']:>5}  win={r['win_pct']:>5.1f}%  PF={r['profit_factor']:>5.2f}  Calmar={cal}")

# === 300+ trades target ===
print("\n--- TARGET: 300+ trades, 58%+ win ---")
target2 = [r for r in unique if r['trades'] >= 300 and r['win_pct'] >= 58]
target2.sort(key=lambda x: (-x['win_pct'], -x['profit_factor']))
for r in target2[:20]:
    cal = f"{r['calmar']:.2f}" if r['calmar'] else "N/A"
    print(f"  {r['filter']:<70} n={r['trades']:>5}  win={r['win_pct']:>5.1f}%  PF={r['profit_factor']:>5.2f}  Calmar={cal}")

# === OR combinations summary ===
print("\n--- Best OR combinations (most trades with decent win%) ---")
all_or = or_results + or3_results
all_or.sort(key=lambda x: (-x['trades'], -x['win_pct']))
for r in all_or[:15]:
    cal = f"{r['calmar']:.2f}" if r['calmar'] else "N/A"
    print(f"  {r['filter']:<65} n={r['trades']:>5}  win={r['win_pct']:>5.1f}%  PF={r['profit_factor']:>5.2f}  Calmar={cal}")

# === Consolidation days summary ===
print("\n--- Best with consolidation days filter ---")
cd_all = cd_combos
cd_all.sort(key=lambda x: (-x['win_pct'], -x['trades']))
for r in cd_all[:15]:
    cal = f"{r['calmar']:.2f}" if r['calmar'] else "N/A"
    print(f"  {r['filter']:<65} n={r['trades']:>5}  win={r['win_pct']:>5.1f}%  PF={r['profit_factor']:>5.2f}  Calmar={cal}")

# === Final mega shortlist ===
print(f"\n\n{'='*80}")
print("FINAL SHORTLIST: 100+ trades, 60%+, Calmar > 0.5, sorted by trades desc")
print("="*80)
shortlist = [r for r in unique if r['trades']>=100 and r['win_pct']>=60
             and r['calmar'] and r['calmar'] > 0.5]
shortlist.sort(key=lambda x: (-x['trades'], -x['win_pct']))
print(f"Found {len(shortlist)} strategies")
print(f"\n{'#':>3} {'Strategy':<75} {'N':>5} {'Win%':>5} {'PF':>5} {'Calmar':>6}")
print("-"*105)
for i, r in enumerate(shortlist[:50], 1):
    print(f"{i:>3} {r['filter']:<75} {r['trades']:>5} {r['win_pct']:>4.1f}% {r['profit_factor']:>5.2f} {r['calmar']:>5.2f}")

print(f"\n\nTotal time: {time.time()-t0:.0f}s")
print("Done!")
