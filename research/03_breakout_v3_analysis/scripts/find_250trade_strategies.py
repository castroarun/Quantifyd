"""
Exhaustive strategy search on enhanced data (53 columns, 9,739 trades).
Target: 65%+ win rate with 250+ trades, best possible PF and Calmar.

Approach:
  1. Define 50+ filter building blocks from all indicators
  2. Test 2/3/4 filter combos systematically
  3. Apply top-down: weekly filter -> daily filter layering
  4. Score by composite metric: win% * PF * sqrt(trades) / 100
"""
import pandas as pd
import numpy as np
from itertools import combinations
import time
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('breakout_analysis_enhanced.csv')
df['date'] = pd.to_datetime(df['date'])
print(f"Loaded {len(df)} trades, {len(df.columns)} columns")
print(f"Baseline: {(df['trade_return']>0).mean()*100:.1f}% win rate")


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
    yrs = max((dates[-1] - dates[0]) / np.timedelta64(365,'D'), 0.5)
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


# =================================================================
# BUILDING BLOCKS: 50+ filters from all indicators
# =================================================================

# --- Proven strong filters ---
filters = {
    # RSI family
    'RSI>=60': df['rsi14'] >= 60,
    'RSI>=65': df['rsi14'] >= 65,
    'RSI>=70': df['rsi14'] >= 70,
    # Volume
    'Vol>=3x': df['volume_ratio'] >= 3,
    'Vol>=5x': df['volume_ratio'] >= 5,
    # Breakout magnitude
    'BO>=3%': df['breakout_pct'] >= 3,
    'BO>=5%': df['breakout_pct'] >= 5,
    # Momentum
    'Mom60>=10': df['mom_60d'] >= 10,
    'Mom60>=15': df['mom_60d'] >= 15,
    'Mom20>=5': df['mom_20d'] >= 5,
    'Mom10>=5': df['mom_10d'] >= 5,
    # Volume trend
    'VolTr>=1.2': df['vol_trend'] >= 1.2,
    'VolTr>=1.5': df['vol_trend'] >= 1.5,
    # ATH proximity
    'ATH>=85%': df['ath_proximity'] >= 85,
    'ATH>=90%': df['ath_proximity'] >= 90,
    'ATH>=95%': df['ath_proximity'] >= 95,
    # EMA alignment
    'EMA20>50': df['ema20_above_50'] == 1,

    # --- NEW: Weekly top-down filters ---
    'wEMA20': df['w_above_ema20'] == 1,
    'wEMA50': df['w_above_ema50'] == 1,
    'wEMA20>50': df['w_ema20_gt_50'] == 1,
    'wMACD+': df['w_macd_positive'] == 1,
    'wMACDbull': df['w_macd_bullish'] == 1,
    'wRSI>55': df['w_rsi'] > 55,
    'wRSI>60': df['w_rsi'] > 60,

    # --- NEW: Daily momentum/oscillator filters ---
    'MACD+': df['macd_positive'] == 1,
    'MACDbull': df['macd_bullish'] == 1,
    'ADX>20': df['adx'] > 20,
    'ADX>25': df['adx'] > 25,
    'ADX>30': df['adx'] > 30,
    'ADXbull': df['adx_bullish'] == 1,
    'MFI>60': df['mfi'] > 60,
    'MFI>70': df['mfi'] > 70,
    'StochK>70': df['stoch_k'] > 70,
    'BB>upper': df['bb_pct_b'] > 1.0,
    'OBVbull': df['obv_bullish'] == 1,
    'EMA20rise': df['ema20_rising'] == 1,
    'EMA9>21': df['ema9_gt_21'] == 1,
    'AbvEMA100': df['above_ema100'] == 1,
    'CCI>100': df['cci'] > 100,
    'RSI7>70': df['rsi7'] > 70,
    'RSI7>80': df['rsi7'] > 80,
    'WillR>-20': df['williams_r'] > -20,
}

fnames = list(filters.keys())
print(f"\n{len(fnames)} building blocks defined")

# =================================================================
# PHASE 1: 2-filter combos targeting 250+ trades at 55%+
# =================================================================
t0 = time.time()
print("\n" + "="*80)
print("PHASE 1: 2-filter combos (55%+ win, 200+ trades)")
print("="*80)

results_2 = []
for i in range(len(fnames)):
    for j in range(i+1, len(fnames)):
        mask = filters[fnames[i]] & filters[fnames[j]]
        r = ev(mask, f"{fnames[i]} + {fnames[j]}", min_trades=200)
        if r and r['win_pct'] >= 55:
            results_2.append(r)

results_2.sort(key=lambda x: (-x['win_pct'], -x['profit_factor']))
print(f"Found {len(results_2)} combos")
for r in results_2[:25]:
    cal = f"{r['calmar']:.2f}" if r['calmar'] else "N/A"
    print(f"  {r['filter']:<45} n={r['trades']:>5}  win={r['win_pct']:>5.1f}%  PF={r['profit_factor']:>5.2f}  Calmar={cal}")

# =================================================================
# PHASE 2: 3-filter combos targeting 250+ trades at 58%+
# =================================================================
print(f"\n{'='*80}")
print(f"PHASE 2: 3-filter combos (58%+ win, 100+ trades) [{time.time()-t0:.0f}s]")
print("="*80)

results_3 = []
for i in range(len(fnames)):
    for j in range(i+1, len(fnames)):
        for k in range(j+1, len(fnames)):
            mask = filters[fnames[i]] & filters[fnames[j]] & filters[fnames[k]]
            r = ev(mask, f"{fnames[i]} + {fnames[j]} + {fnames[k]}", min_trades=100)
            if r and r['win_pct'] >= 58:
                results_3.append(r)

results_3.sort(key=lambda x: (-x['win_pct'], -x['profit_factor']))
print(f"Found {len(results_3)} combos [{time.time()-t0:.0f}s]")
for r in results_3[:30]:
    cal = f"{r['calmar']:.2f}" if r['calmar'] else "N/A"
    print(f"  {r['filter']:<60} n={r['trades']:>5}  win={r['win_pct']:>5.1f}%  PF={r['profit_factor']:>5.2f}  Calmar={cal}")

# =================================================================
# PHASE 3: 4-filter combos from best blocks targeting 250+ trades at 60%+
# =================================================================
print(f"\n{'='*80}")
print(f"PHASE 3: 4-filter combos (60%+ win, 100+ trades) [{time.time()-t0:.0f}s]")
print("="*80)

# Identify which filters appear in top 3-filter results
best_blocks = set()
for r in results_3[:50]:
    for f in r['filter'].split(' + '):
        best_blocks.add(f.strip())

# Also add the proven strong ones
for f in ['RSI>=60','RSI>=65','RSI>=70','Vol>=3x','Vol>=5x','BO>=3%','BO>=5%',
          'Mom60>=10','Mom60>=15','VolTr>=1.2','ATH>=90%','ATH>=95%','EMA20>50',
          'wEMA20','wEMA50','wMACD+','wRSI>60','MACD+','ADX>25','MFI>70']:
    best_blocks.add(f)

best_fnames = [f for f in fnames if f in best_blocks]
print(f"Using {len(best_fnames)} best building blocks: {best_fnames}")

results_4 = []
for combo in combinations(best_fnames, 4):
    mask = filters[combo[0]]
    for c in combo[1:]:
        mask = mask & filters[c]
    r = ev(mask, ' + '.join(combo), min_trades=100)
    if r and r['win_pct'] >= 60:
        results_4.append(r)

results_4.sort(key=lambda x: (-x['win_pct'], -x['profit_factor']))
print(f"Found {len(results_4)} combos [{time.time()-t0:.0f}s]")
for r in results_4[:30]:
    cal = f"{r['calmar']:.2f}" if r['calmar'] else "N/A"
    print(f"  {r['filter']:<70} n={r['trades']:>5}  win={r['win_pct']:>5.1f}%  PF={r['profit_factor']:>5.2f}  Calmar={cal}")

# =================================================================
# PHASE 4: 5-filter combos targeting 250+ trades at 62%+
# =================================================================
print(f"\n{'='*80}")
print(f"PHASE 4: 5-filter combos (62%+ win, 100+ trades) [{time.time()-t0:.0f}s]")
print("="*80)

# Narrow to best performing filters from phase 3
top_blocks_4 = set()
for r in results_4[:30]:
    for f in r['filter'].split(' + '):
        top_blocks_4.add(f.strip())
top_fnames_4 = [f for f in best_fnames if f in top_blocks_4]
print(f"Using {len(top_fnames_4)} top blocks from phase 3")

results_5 = []
if len(top_fnames_4) >= 5:
    for combo in combinations(top_fnames_4, 5):
        mask = filters[combo[0]]
        for c in combo[1:]:
            mask = mask & filters[c]
        r = ev(mask, ' + '.join(combo), min_trades=100)
        if r and r['win_pct'] >= 62:
            results_5.append(r)

results_5.sort(key=lambda x: (-x['win_pct'], -x['profit_factor']))
print(f"Found {len(results_5)} combos [{time.time()-t0:.0f}s]")
for r in results_5[:25]:
    cal = f"{r['calmar']:.2f}" if r['calmar'] else "N/A"
    print(f"  {r['filter']:<80} n={r['trades']:>5}  win={r['win_pct']:>5.1f}%  PF={r['profit_factor']:>5.2f}  Calmar={cal}")

# =================================================================
# PHASE 5: 6-filter combos targeting 250+ trades at 64%+
# =================================================================
print(f"\n{'='*80}")
print(f"PHASE 5: 6-filter combos (64%+ win, 100+ trades) [{time.time()-t0:.0f}s]")
print("="*80)

top_blocks_5 = set()
for r in (results_5[:20] if results_5 else results_4[:20]):
    for f in r['filter'].split(' + '):
        top_blocks_5.add(f.strip())
top_fnames_5 = [f for f in best_fnames if f in top_blocks_5]
print(f"Using {len(top_fnames_5)} top blocks")

results_6 = []
if len(top_fnames_5) >= 6:
    for combo in combinations(top_fnames_5, 6):
        mask = filters[combo[0]]
        for c in combo[1:]:
            mask = mask & filters[c]
        r = ev(mask, ' + '.join(combo), min_trades=100)
        if r and r['win_pct'] >= 64:
            results_6.append(r)

    results_6.sort(key=lambda x: (-x['win_pct'], -x['profit_factor']))
    print(f"Found {len(results_6)} combos [{time.time()-t0:.0f}s]")
    for r in results_6[:20]:
        cal = f"{r['calmar']:.2f}" if r['calmar'] else "N/A"
        print(f"  {r['filter']:<90} n={r['trades']:>5}  win={r['win_pct']:>5.1f}%  PF={r['profit_factor']:>5.2f}  Calmar={cal}")

# =================================================================
# PHASE 6: TOP-DOWN APPROACH - Weekly filter -> daily filter
# =================================================================
print(f"\n{'='*80}")
print(f"PHASE 6: TOP-DOWN (weekly + daily combos) [{time.time()-t0:.0f}s]")
print("="*80)

weekly_bases = [
    ("wEMA20 + wMACD+", (df['w_above_ema20']==1) & (df['w_macd_positive']==1)),
    ("wEMA50 + wMACD+", (df['w_above_ema50']==1) & (df['w_macd_positive']==1)),
    ("wEMA20>50 + wRSI>55", (df['w_ema20_gt_50']==1) & (df['w_rsi']>55)),
    ("wEMA20>50 + wMACD+", (df['w_ema20_gt_50']==1) & (df['w_macd_positive']==1)),
    ("wEMA20 + wEMA50 + wMACD+", (df['w_above_ema20']==1) & (df['w_above_ema50']==1) & (df['w_macd_positive']==1)),
    ("wEMA20>50 + wRSI>60", (df['w_ema20_gt_50']==1) & (df['w_rsi']>60)),
]

daily_overlays = [
    ("RSI>=60 + BO>=3%", (df['rsi14']>=60) & (df['breakout_pct']>=3)),
    ("RSI>=65 + BO>=3%", (df['rsi14']>=65) & (df['breakout_pct']>=3)),
    ("RSI>=60 + Vol>=3x", (df['rsi14']>=60) & (df['volume_ratio']>=3)),
    ("RSI>=65 + Vol>=3x", (df['rsi14']>=65) & (df['volume_ratio']>=3)),
    ("ATH>=90% + BO>=3%", (df['ath_proximity']>=90) & (df['breakout_pct']>=3)),
    ("ATH>=90% + Vol>=3x", (df['ath_proximity']>=90) & (df['volume_ratio']>=3)),
    ("ATH>=90% + Mom60>=10", (df['ath_proximity']>=90) & (df['mom_60d']>=10)),
    ("Mom60>=10 + VolTr>=1.2", (df['mom_60d']>=10) & (df['vol_trend']>=1.2)),
    ("BO>=5% + VolTr>=1.2", (df['breakout_pct']>=5) & (df['vol_trend']>=1.2)),
    ("RSI>=60 + ATH>=90%", (df['rsi14']>=60) & (df['ath_proximity']>=90)),
    ("RSI>=65 + ATH>=90%", (df['rsi14']>=65) & (df['ath_proximity']>=90)),
    ("RSI>=70 + ATH>=90%", (df['rsi14']>=70) & (df['ath_proximity']>=90)),
    ("RSI>=60 + Vol>=3x + BO>=3%", (df['rsi14']>=60) & (df['volume_ratio']>=3) & (df['breakout_pct']>=3)),
    ("ATH>=90% + Vol>=3x + BO>=3%", (df['ath_proximity']>=90) & (df['volume_ratio']>=3) & (df['breakout_pct']>=3)),
    ("ATH>=90% + Mom60>=10 + VolTr>=1.2", (df['ath_proximity']>=90) & (df['mom_60d']>=10) & (df['vol_trend']>=1.2)),
    ("RSI>=60 + ATH>=90% + BO>=3%", (df['rsi14']>=60) & (df['ath_proximity']>=90) & (df['breakout_pct']>=3)),
    ("RSI>=60 + ATH>=90% + Vol>=3x", (df['rsi14']>=60) & (df['ath_proximity']>=90) & (df['volume_ratio']>=3)),
    ("RSI>=65 + ATH>=90% + BO>=3%", (df['rsi14']>=65) & (df['ath_proximity']>=90) & (df['breakout_pct']>=3)),
    ("RSI>=65 + ATH>=90% + Vol>=3x", (df['rsi14']>=65) & (df['ath_proximity']>=90) & (df['volume_ratio']>=3)),
    ("RSI>=65 + ATH>=90% + Mom60>=10", (df['rsi14']>=65) & (df['ath_proximity']>=90) & (df['mom_60d']>=10)),
    ("RSI>=70 + ATH>=90% + Vol>=3x", (df['rsi14']>=70) & (df['ath_proximity']>=90) & (df['volume_ratio']>=3)),
    ("ATH>=85% + Mom60>=10 + VolTr>=1.2", (df['ath_proximity']>=85) & (df['mom_60d']>=10) & (df['vol_trend']>=1.2)),
    ("RSI>=60 + ATH>=85% + Mom60>=10", (df['rsi14']>=60) & (df['ath_proximity']>=85) & (df['mom_60d']>=10)),
]

topdown_results = []
for w_name, w_mask in weekly_bases:
    for d_name, d_mask in daily_overlays:
        label = f"[W]{w_name} + [D]{d_name}"
        mask = w_mask & d_mask
        r = ev(mask, label, min_trades=50)
        if r and r['win_pct'] >= 55:
            topdown_results.append(r)

topdown_results.sort(key=lambda x: (-x['win_pct'], -x['profit_factor']))
print(f"Found {len(topdown_results)} top-down combos with >= 55% win")
for r in topdown_results[:30]:
    cal = f"{r['calmar']:.2f}" if r['calmar'] else "N/A"
    print(f"  {r['filter']:<80} n={r['trades']:>5}  win={r['win_pct']:>5.1f}%  PF={r['profit_factor']:>5.2f}  Calmar={cal}")


# =================================================================
# GRAND SUMMARY
# =================================================================
print(f"\n\n{'='*80}")
print(f"GRAND SUMMARY: BEST STRATEGIES BY TRADE COUNT [{time.time()-t0:.0f}s]")
print("="*80)

all_r = results_2 + results_3 + results_4 + results_5 + results_6 + topdown_results
# Dedupe
seen = set()
unique = []
for r in all_r:
    if r['filter'] not in seen:
        seen.add(r['filter'])
        unique.append(r)

# === TARGET: 250+ trades at 65%+ ===
print("\n--- TARGET: 250+ trades, 65%+ win rate ---")
target = [r for r in unique if r['trades'] >= 250 and r['win_pct'] >= 65]
target.sort(key=lambda x: (-x['win_pct'], -x['profit_factor']))
if target:
    for r in target[:20]:
        cal = f"{r['calmar']:.2f}" if r['calmar'] else "N/A"
        print(f"  {r['filter']:<75} n={r['trades']:>5}  win={r['win_pct']:>5.1f}%  PF={r['profit_factor']:>5.2f}  Calmar={cal}")
else:
    print("  No strategies found at 250+/65%+")
    # Show what's close
    print("\n--- Closest: 200+ trades, 60%+ win ---")
    close = [r for r in unique if r['trades'] >= 200 and r['win_pct'] >= 60]
    close.sort(key=lambda x: (-x['win_pct'], -x['profit_factor']))
    for r in close[:20]:
        cal = f"{r['calmar']:.2f}" if r['calmar'] else "N/A"
        print(f"  {r['filter']:<75} n={r['trades']:>5}  win={r['win_pct']:>5.1f}%  PF={r['profit_factor']:>5.2f}  Calmar={cal}")

    print("\n--- Closest: 150+ trades, 62%+ win ---")
    close2 = [r for r in unique if r['trades'] >= 150 and r['win_pct'] >= 62]
    close2.sort(key=lambda x: (-x['win_pct'], -x['profit_factor']))
    for r in close2[:20]:
        cal = f"{r['calmar']:.2f}" if r['calmar'] else "N/A"
        print(f"  {r['filter']:<75} n={r['trades']:>5}  win={r['win_pct']:>5.1f}%  PF={r['profit_factor']:>5.2f}  Calmar={cal}")

# === Best at each trade-count tier ===
print("\n\n--- Best win% at each trade-count tier ---")
for min_n in [500, 400, 300, 250, 200, 150, 100]:
    tier = [r for r in unique if r['trades'] >= min_n]
    if tier:
        best = max(tier, key=lambda x: x['win_pct'])
        cal = f"{best['calmar']:.2f}" if best['calmar'] else "N/A"
        print(f"  {min_n:>4}+ trades: {best['filter']:<65} n={best['trades']:>5}  win={best['win_pct']:>5.1f}%  PF={best['profit_factor']:>5.2f}  Calmar={cal}")

# === Best Calmar at each trade-count tier ===
print("\n--- Best Calmar at each trade-count tier ---")
for min_n in [500, 400, 300, 250, 200, 150, 100]:
    tier = [r for r in unique if r['trades'] >= min_n and r['calmar'] is not None and r['calmar'] > 0]
    if tier:
        best = max(tier, key=lambda x: x['calmar'])
        print(f"  {min_n:>4}+ trades: {best['filter']:<65} n={best['trades']:>5}  win={best['win_pct']:>5.1f}%  PF={best['profit_factor']:>5.2f}  Calmar={best['calmar']:.2f}")

# === FINAL SHORTLIST: 100+ trades, 60%+ win, Calmar > 0.5 ===
print(f"\n\n{'='*80}")
print("FINAL SHORTLIST: 100+ trades, 60%+ win, sorted by trades desc")
print("="*80)

shortlist = [r for r in unique
             if r['trades'] >= 100 and r['win_pct'] >= 60
             and r['calmar'] is not None and r['calmar'] > 0.5]
shortlist.sort(key=lambda x: (-x['trades'], -x['win_pct']))
print(f"Found {len(shortlist)} strategies")
print(f"\n{'#':>2} {'Strategy':<75} {'N':>5} {'Win%':>5} {'AvgRet':>6} {'PF':>5} {'Calmar':>6}")
print("-"*110)
for i, r in enumerate(shortlist[:30], 1):
    print(f"{i:>2} {r['filter']:<75} {r['trades']:>5} {r['win_pct']:>4.1f}% {r['avg_ret']:>+5.1f}% {r['profit_factor']:>5.2f} {r['calmar']:>5.2f}")

print(f"\n\nTotal time: {time.time()-t0:.0f}s")
print("Done!")
