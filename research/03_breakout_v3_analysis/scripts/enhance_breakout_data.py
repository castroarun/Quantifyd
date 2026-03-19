"""
Enhance breakout_analysis_full.csv with 20+ new indicators.
Vectorized approach: compute all indicators per-symbol, then look up by date.
"""
import pandas as pd
import numpy as np
import sqlite3
import time
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('breakout_analysis_full.csv')
df['date'] = pd.to_datetime(df['date'])
print(f"Loaded {len(df)} trades, {df['symbol'].nunique()} symbols")

conn = sqlite3.connect('backtest_data/market_data.db')

def ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def rsi(c, p=14):
    d = c.diff()
    g = d.where(d>0,0).rolling(p).mean()
    l = (-d.where(d<0,0)).rolling(p).mean()
    return 100 - 100/(1 + g/l.replace(0,np.nan))

def compute_indicators_for_symbol(daily):
    """Compute ALL indicators as columns on the daily DataFrame. Fully vectorized."""
    d = daily.copy()
    h, l, c, v = d['high'], d['low'], d['close'], d['volume']

    # --- Stochastics ---
    ll = l.rolling(14).min()
    hh = h.rolling(14).max()
    d['stoch_k'] = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    d['stoch_d'] = d['stoch_k'].rolling(3).mean()

    # --- MACD ---
    ema12 = ema(c, 12); ema26 = ema(c, 26)
    macd_line = ema12 - ema26
    signal = ema(macd_line, 9)
    d['macd_hist'] = macd_line - signal
    d['macd_positive'] = (macd_line > 0).astype(int)
    d['macd_bullish'] = (macd_line > signal).astype(int)

    # --- ADX ---
    plus_dm = h.diff()
    minus_dm = -l.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)  # fixed: was overwriting
    tr = pd.concat([h-l, abs(h-c.shift(1)), abs(l-c.shift(1))], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    plus_di = 100 * plus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
    minus_di_val = 100 * minus_dm.rolling(14).mean() / atr14.replace(0, np.nan)
    dx = 100 * abs(plus_di - minus_di_val) / (plus_di + minus_di_val).replace(0, np.nan)
    d['adx'] = dx.rolling(14).mean()
    d['plus_di'] = plus_di
    d['minus_di'] = minus_di_val
    d['adx_bullish'] = (plus_di > minus_di_val).astype(int)
    d['atr_pct'] = atr14 / c * 100

    # --- Bollinger Bands ---
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    bb_upper = bb_mid + 2*bb_std
    bb_lower = bb_mid - 2*bb_std
    d['bb_pct_b'] = (c - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
    d['bb_width'] = (bb_upper - bb_lower) / bb_mid * 100
    width_q20 = d['bb_width'].rolling(120, min_periods=60).quantile(0.2)
    d['bb_squeeze'] = (d['bb_width'] < width_q20).astype(int)

    # --- MFI ---
    tp = (h + l + c) / 3
    rmf = tp * v
    delta_tp = tp.diff()
    pos = rmf.where(delta_tp > 0, 0).rolling(14).sum()
    neg = rmf.where(delta_tp <= 0, 0).rolling(14).sum()
    d['mfi'] = 100 - 100/(1 + pos/neg.replace(0, np.nan))

    # --- CCI ---
    tp_ma = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    d['cci'] = (tp - tp_ma) / (0.015 * tp_mad.replace(0, np.nan))

    # --- Williams %R ---
    d['williams_r'] = -100 * (h.rolling(14).max() - c) / (h.rolling(14).max() - l.rolling(14).min()).replace(0, np.nan)

    # --- OBV ---
    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    obv_ema20 = ema(obv, 20)
    d['obv_bullish'] = (obv > obv_ema20).astype(int)

    # --- Supertrend (vectorized approximation) ---
    hl2 = (h + l) / 2
    atr10 = tr.rolling(10).mean()
    st_upper = hl2 + 3.0 * atr10
    st_lower = hl2 - 3.0 * atr10
    # Simplified: bullish if close > lower band continuously
    d['supertrend_bull'] = (c > st_lower).astype(int)

    # --- Parabolic SAR (simplified: use EMA crossover proxy) ---
    ema5 = ema(c, 5)
    d['psar_bullish'] = (c > ema5).astype(int)

    # --- Additional EMAs ---
    ema9 = ema(c, 9); ema21 = ema(c, 21); ema100 = ema(c, 100)
    d['above_ema9'] = (c > ema9).astype(int)
    d['above_ema21'] = (c > ema21).astype(int)
    d['above_ema100'] = (c > ema100).astype(int)
    d['ema9_gt_21'] = (ema9 > ema21).astype(int)

    # --- EMA20 rising ---
    ema20 = ema(c, 20)
    d['ema20_rising'] = (ema20 > ema20.shift(5)).astype(int)

    # --- 10-day momentum ---
    d['mom_10d'] = (c / c.shift(10) - 1) * 100

    # --- RSI(7) shorter-term ---
    d['rsi7'] = rsi(c, 7)

    return d


def compute_weekly_for_symbol(daily):
    """Compute weekly indicators, return as daily-indexed Series."""
    weekly = daily.resample('W').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum'
    }).dropna()
    if len(weekly) < 30:
        return None

    wc = weekly['close']
    weekly['w_above_ema20'] = (wc > ema(wc, 20)).astype(int)
    weekly['w_above_ema50'] = (wc > ema(wc, 50)).astype(int)
    weekly['w_ema20_gt_50'] = (ema(wc, 20) > ema(wc, 50)).astype(int)
    weekly['w_rsi'] = rsi(wc, 14)
    w_macd = ema(wc, 12) - ema(wc, 26)
    w_signal = ema(w_macd, 9)
    weekly['w_macd_positive'] = (w_macd > 0).astype(int)
    weekly['w_macd_bullish'] = (w_macd > w_signal).astype(int)

    # Forward-fill weekly to daily index
    w_cols = ['w_above_ema20', 'w_above_ema50', 'w_ema20_gt_50',
              'w_rsi', 'w_macd_positive', 'w_macd_bullish']
    return weekly[w_cols].reindex(daily.index, method='ffill')


# New indicator columns we're adding
NEW_COLS = [
    'stoch_k', 'stoch_d', 'macd_hist', 'macd_positive', 'macd_bullish',
    'adx', 'plus_di', 'minus_di', 'adx_bullish', 'atr_pct',
    'bb_pct_b', 'bb_width', 'bb_squeeze', 'supertrend_bull',
    'mfi', 'cci', 'williams_r', 'obv_bullish', 'psar_bullish',
    'above_ema9', 'above_ema21', 'above_ema100', 'ema9_gt_21',
    'ema20_rising', 'mom_10d', 'rsi7',
    'w_above_ema20', 'w_above_ema50', 'w_ema20_gt_50',
    'w_rsi', 'w_macd_positive', 'w_macd_bullish'
]

# Initialize new columns with NaN
for col in NEW_COLS:
    df[col] = np.nan

symbols = df['symbol'].unique()
total = len(symbols)
start_time = time.time()
success = 0
errors = 0

for sym_idx, symbol in enumerate(symbols):
    if sym_idx % 100 == 0:
        elapsed = time.time() - start_time
        rate = (sym_idx + 1) / max(elapsed, 1)
        eta = (total - sym_idx) / max(rate, 0.01)
        print(f"  [{sym_idx+1}/{total}] {symbol} ... ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    # Load daily data
    query = "SELECT date, open, high, low, close, volume FROM market_data_unified WHERE symbol = ? AND timeframe = 'day' ORDER BY date"
    daily = pd.read_sql(query, conn, params=(symbol,), parse_dates=['date']).set_index('date')

    if len(daily) < 60:
        errors += 1
        continue

    try:
        # Compute all daily indicators
        enriched = compute_indicators_for_symbol(daily)

        # Compute weekly indicators
        weekly = compute_weekly_for_symbol(daily)
        if weekly is not None:
            for col in weekly.columns:
                enriched[col] = weekly[col]

        # Look up indicator values for each trade of this symbol
        sym_mask = df['symbol'] == symbol
        sym_trades = df[sym_mask]

        for idx, row in sym_trades.iterrows():
            trade_date = pd.Timestamp(row['date'])
            # Find exact or nearest prior date
            valid = enriched.index[enriched.index <= trade_date]
            if len(valid) == 0:
                continue
            lookup_date = valid[-1]

            for col in NEW_COLS:
                if col in enriched.columns:
                    val = enriched.loc[lookup_date, col]
                    if not pd.isna(val):
                        df.at[idx, col] = round(float(val), 2) if isinstance(val, (float, np.floating)) else int(val)

        success += 1
    except Exception as e:
        errors += 1
        if sym_idx < 5:
            print(f"    Error for {symbol}: {e}")

elapsed = time.time() - start_time
print(f"\nProcessed {success}/{total} symbols in {elapsed:.1f}s ({errors} errors)")

# Save enhanced CSV
df.to_csv('breakout_analysis_enhanced.csv', index=False)
print(f"Saved breakout_analysis_enhanced.csv: {len(df)} rows, {len(df.columns)} columns")

# Validation
print(f"\n=== VALIDATION ===")
for col in NEW_COLS:
    valid = df[col].notna().sum()
    print(f"  {col:<20} {valid:>6}/{len(df)} valid ({valid/len(df)*100:.1f}%)")

# Quick win rate check on new indicators
print(f"\n=== QUICK WIN RATE CHECK (new indicators) ===")
checks = [
    ("MACD Bullish", df['macd_bullish'] == 1),
    ("MACD Positive", df['macd_positive'] == 1),
    ("ADX > 25 (trending)", df['adx'] > 25),
    ("ADX Bullish (+DI > -DI)", df['adx_bullish'] == 1),
    ("Stoch K > 80 (overbought)", df['stoch_k'] > 80),
    ("Stoch K > 50", df['stoch_k'] > 50),
    ("BB %B > 1.0 (above upper)", df['bb_pct_b'] > 1.0),
    ("BB Squeeze then BO", df['bb_squeeze'] == 1),
    ("Supertrend Bullish", df['supertrend_bull'] == 1),
    ("MFI > 50", df['mfi'] > 50),
    ("MFI > 70", df['mfi'] > 70),
    ("CCI > 100", df['cci'] > 100),
    ("Williams %R > -20", df['williams_r'] > -20),
    ("OBV Bullish", df['obv_bullish'] == 1),
    ("PSAR Bullish", df['psar_bullish'] == 1),
    ("EMA9 > EMA21", df['ema9_gt_21'] == 1),
    ("Above EMA100", df['above_ema100'] == 1),
    ("EMA20 Rising", df['ema20_rising'] == 1),
    ("Weekly EMA20 Bullish", df['w_above_ema20'] == 1),
    ("Weekly EMA50 Bullish", df['w_above_ema50'] == 1),
    ("Weekly EMA20>50", df['w_ema20_gt_50'] == 1),
    ("Weekly MACD Positive", df['w_macd_positive'] == 1),
    ("Weekly MACD Bullish", df['w_macd_bullish'] == 1),
    ("Weekly RSI > 60", df['w_rsi'] > 60),
    ("Weekly RSI > 70", df['w_rsi'] > 70),
    ("RSI(7) > 70", df['rsi7'] > 70),
    ("RSI(7) > 80", df['rsi7'] > 80),
    ("Mom 10d > 5%", df['mom_10d'] > 5),
]

for label, mask in checks:
    subset = df[mask & mask.notna()]
    if len(subset) >= 20:
        wr = (subset['trade_return'] > 0).mean() * 100
        print(f"  {label:<35} n={len(subset):>5}  win={wr:.1f}%")

conn.close()
print("\nDone!")
