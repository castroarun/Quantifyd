"""
Comprehensive Strategy Research Scanner V2 - VECTORIZED
=========================================================

FAST vectorized backtesting - no row-by-row loops for signal evaluation.
Pre-computes all entry/exit signals as boolean columns, then uses vectorized
trade extraction.

Scans 120+ indicator combinations across 470+ stocks to find strategies with:
- 250+ trades, 65%+ win rate, 1+ Calmar ratio, 1.5+ profit factor
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import time
import warnings

warnings.filterwarnings('ignore')

DB_PATH = Path(__file__).parent / 'backtest_data' / 'market_data.db'
RESULTS_PATH = Path(__file__).parent / 'backtest_data' / 'strategy_research_results_v2.json'
CSV_RESULTS_PATH = Path(__file__).parent / 'strategy_research_results_v2.csv'

# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_daily_data():
    conn = sqlite3.connect(DB_PATH)
    symbols = pd.read_sql_query(
        "SELECT DISTINCT symbol FROM market_data_unified WHERE timeframe='day'", conn
    )['symbol'].tolist()
    print(f"Loading daily data for {len(symbols)} stocks...")
    all_data = {}
    for sym in symbols:
        df = pd.read_sql_query(
            "SELECT date,open,high,low,close,volume FROM market_data_unified WHERE symbol=? AND timeframe='day' ORDER BY date",
            conn, params=[sym])
        if len(df) < 200:
            continue
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df = df.astype(float)
        all_data[sym] = df
    conn.close()
    print(f"Loaded {len(all_data)} stocks with 200+ days")
    return all_data


# ============================================================================
# VECTORIZED INDICATORS (no loops)
# ============================================================================

def ema(s, p): return s.ewm(span=p, adjust=False).mean()
def sma(s, p): return s.rolling(window=p).mean()

def rsi(s, p=14):
    d = s.diff()
    g = d.where(d > 0, 0.0)
    l = (-d).where(d < 0, 0.0)
    ag = g.ewm(span=p, adjust=False).mean()
    al = l.ewm(span=p, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)

def stoch_k(df, p=14):
    ll = df['low'].rolling(p).min()
    hh = df['high'].rolling(p).max()
    return (((df['close'] - ll) / (hh - ll).replace(0, np.nan)) * 100).fillna(50)

def williams_r(df, p=14):
    hh = df['high'].rolling(p).max()
    ll = df['low'].rolling(p).min()
    return (-100 * (hh - df['close']) / (hh - ll).replace(0, np.nan)).fillna(-50)

def cci(df, p=20):
    tp = (df['high'] + df['low'] + df['close']) / 3
    s = tp.rolling(p).mean()
    mad = tp.rolling(p).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return ((tp - s) / (0.015 * mad)).fillna(0)

def mfi(df, p=14):
    tp = (df['high'] + df['low'] + df['close']) / 3
    mf = tp * df['volume']
    pos = mf.where(tp > tp.shift(1), 0)
    neg = mf.where(tp <= tp.shift(1), 0)
    ps = pos.rolling(p).sum()
    ns = neg.rolling(p).sum()
    return (100 - 100 / (1 + ps / ns.replace(0, np.nan))).fillna(50)

def atr(df, p=14):
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift(1)).abs()
    lc = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(span=p, adjust=False).mean()

def adx_calc(df, p=14):
    h, l, c = df['high'], df['low'], df['close']
    tr = pd.concat([h-l, (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    pdm = h.diff().where((h.diff() > -l.diff()) & (h.diff() > 0), 0)
    mdm = (-l.diff()).where((-l.diff() > h.diff()) & (-l.diff() > 0), 0)
    atr_v = tr.ewm(span=p, adjust=False).mean()
    pdi = 100 * pdm.ewm(span=p, adjust=False).mean() / atr_v
    mdi = 100 * mdm.ewm(span=p, adjust=False).mean() / atr_v
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(span=p, adjust=False).mean().fillna(0), pdi.fillna(0), mdi.fillna(0)

def macd_calc(s, f=12, sl=26, sig=9):
    ml = ema(s, f) - ema(s, sl)
    sigl = ema(ml, sig)
    return ml, sigl, ml - sigl

def bollinger(s, p=20, std=2.0):
    mid = s.rolling(p).mean()
    sd = s.rolling(p).std()
    return mid, mid + std*sd, mid - std*sd

def keltner(df, ep=20, ap=10, m=2.0):
    mid = ema(df['close'], ep)
    a = atr(df, ap)
    return mid, mid + m*a, mid - m*a

def obv_calc(df):
    sign = np.sign(df['close'].diff())
    return (sign * df['volume']).cumsum()

def cmf_calc(df, p=20):
    mfm = ((df['close']-df['low']) - (df['high']-df['close'])) / (df['high']-df['low']).replace(0, np.nan)
    mfv = mfm * df['volume']
    return (mfv.rolling(p).sum() / df['volume'].rolling(p).sum()).fillna(0)

def roc_calc(s, p=12):
    return ((s / s.shift(p)) - 1) * 100

def tsi_calc(s, lp=25, sp=13):
    d = s.diff()
    ds = ema(ema(d, lp), sp)
    dsa = ema(ema(d.abs(), lp), sp)
    return (100 * ds / dsa.replace(0, np.nan)).fillna(0)

def supertrend_vec(df, ap=10, mult=3.0):
    """Supertrend - must use loop but optimized with numpy."""
    a = atr(df, ap).values
    hl2 = ((df['high'] + df['low']) / 2).values
    close = df['close'].values
    n = len(df)
    ub = hl2 + mult * a
    lb = hl2 - mult * a
    direction = np.ones(n, dtype=int)
    st = np.full(n, lb[0])

    for i in range(1, n):
        if close[i] > st[i-1]:
            st[i] = lb[i]; direction[i] = 1
        elif close[i] < st[i-1]:
            st[i] = ub[i]; direction[i] = -1
        else:
            st[i] = st[i-1]; direction[i] = direction[i-1]
            if direction[i] == 1 and lb[i] > st[i]: st[i] = lb[i]
            elif direction[i] == -1 and ub[i] < st[i]: st[i] = ub[i]

    return pd.Series(direction, index=df.index)

def psar_vec(df, af_s=0.02, af_i=0.02, af_m=0.2):
    """Parabolic SAR direction - loop with numpy."""
    h, l = df['high'].values, df['low'].values
    n = len(df)
    sar = np.zeros(n); ep = np.zeros(n); af = np.zeros(n); trend = np.zeros(n)
    sar[0]=l[0]; ep[0]=h[0]; af[0]=af_s; trend[0]=1
    for i in range(1, n):
        if trend[i-1]==1:
            sar[i] = sar[i-1]+af[i-1]*(ep[i-1]-sar[i-1])
            sar[i] = min(sar[i], l[i-1], l[i-2] if i>1 else l[i-1])
            if l[i]<sar[i]:
                trend[i]=-1; sar[i]=ep[i-1]; ep[i]=l[i]; af[i]=af_s
            else:
                trend[i]=1
                if h[i]>ep[i-1]: ep[i]=h[i]; af[i]=min(af[i-1]+af_i,af_m)
                else: ep[i]=ep[i-1]; af[i]=af[i-1]
        else:
            sar[i]=sar[i-1]+af[i-1]*(ep[i-1]-sar[i-1])
            sar[i]=max(sar[i],h[i-1],h[i-2] if i>1 else h[i-1])
            if h[i]>sar[i]:
                trend[i]=1; sar[i]=ep[i-1]; ep[i]=h[i]; af[i]=af_s
            else:
                trend[i]=-1
                if l[i]<ep[i-1]: ep[i]=l[i]; af[i]=min(af[i-1]+af_i,af_m)
                else: ep[i]=ep[i-1]; af[i]=af[i-1]
    return pd.Series(trend, index=df.index)


# ============================================================================
# COMPUTE ALL INDICATORS FOR ONE STOCK
# ============================================================================

def enrich(df):
    d = df.copy()
    c = d['close']

    # MAs
    for p in [5, 9, 10, 13, 20, 21, 50, 100, 200]:
        d[f'ema_{p}'] = ema(c, p)
    for p in [5, 10, 20, 50, 200]:
        d[f'sma_{p}'] = sma(c, p)

    # Momentum
    d['rsi_2'] = rsi(c, 2); d['rsi_5'] = rsi(c, 5); d['rsi_14'] = rsi(c, 14); d['rsi_21'] = rsi(c, 21)
    d['stk14'] = stoch_k(d, 14); d['std14'] = d['stk14'].rolling(3).mean()
    d['stk5'] = stoch_k(d, 5); d['std5'] = d['stk5'].rolling(3).mean()
    d['stk21'] = stoch_k(d, 21); d['std21'] = d['stk21'].rolling(9).mean()
    d['wr2'] = williams_r(d, 2); d['wr5'] = williams_r(d, 5); d['wr14'] = williams_r(d, 14)
    d['cci20'] = cci(d, 20); d['cci14'] = cci(d, 14)
    d['mfi14'] = mfi(d, 14); d['mfi10'] = mfi(d, 10)
    d['roc12'] = roc_calc(c, 12); d['roc9'] = roc_calc(c, 9)
    d['tsi'] = tsi_calc(c)

    # Trend
    d['adx'], d['pdi'], d['mdi'] = adx_calc(d, 14)
    d['macd'], d['macd_sig'], d['macd_h'] = macd_calc(c)
    d['fmacd'], d['fmacd_sig'], d['fmacd_h'] = macd_calc(c, 8, 17, 9)
    d['st10'] = supertrend_vec(d, 10, 3.0)
    d['st7'] = supertrend_vec(d, 7, 2.0)
    d['psar'] = psar_vec(d)

    # Volatility
    d['atr14'] = atr(d, 14)
    d['bb_m'], d['bb_u'], d['bb_l'] = bollinger(c, 20, 2.0)
    d['bb_pctb'] = (c - d['bb_l']) / (d['bb_u'] - d['bb_l']).replace(0, np.nan)
    d['bb_w'] = (d['bb_u'] - d['bb_l']) / d['bb_m'] * 100
    d['bb10_m'], d['bb10_u'], d['bb10_l'] = bollinger(c, 10, 1.5)
    d['kc_m'], d['kc_u'], d['kc_l'] = keltner(d, 20, 10, 2.0)
    d['kc6_m'], d['kc6_u'], d['kc6_l'] = keltner(d, 6, 6, 1.3)
    d['dc20_u'] = d['high'].rolling(20).max()
    d['dc20_l'] = d['low'].rolling(20).min()
    d['dc20_m'] = (d['dc20_u'] + d['dc20_l']) / 2

    # Volume
    d['obv'] = obv_calc(d)
    d['obv_ema'] = ema(d['obv'], 20)
    d['cmf'] = cmf_calc(d, 20)
    d['vol_sma'] = sma(d['volume'], 20)
    d['vol_r'] = d['volume'] / d['vol_sma'].replace(0, np.nan)

    # BB squeeze
    d['squeeze'] = (d['bb_u'] < d['kc_u']) & (d['bb_l'] > d['kc_l'])

    # Derived booleans
    d['above200s'] = c > d['sma_200']
    d['above200e'] = c > d['ema_200']
    d['above50e'] = c > d['ema_50']
    d['above20e'] = c > d['ema_20']
    d['e9_21'] = d['ema_9'] > d['ema_21']
    d['e20_50'] = d['ema_20'] > d['ema_50']
    d['estack'] = (d['ema_9'] > d['ema_21']) & (d['ema_21'] > d['ema_50']) & (d['ema_50'] > d['ema_200'])

    # Crossovers (vectorized)
    d['e9_21_xup'] = d['e9_21'] & ~d['e9_21'].shift(1).fillna(False)
    d['e9_21_xdn'] = ~d['e9_21'] & d['e9_21'].shift(1).fillna(True)
    d['e20_50_xup'] = d['e20_50'] & ~d['e20_50'].shift(1).fillna(False)
    macd_bull = d['macd'] > d['macd_sig']
    d['macd_xup'] = macd_bull & ~macd_bull.shift(1).fillna(False)
    d['macd_xdn'] = ~macd_bull & macd_bull.shift(1).fillna(True)
    d['macd_pos'] = d['macd'] > 0
    d['stk14_xup'] = (d['stk14'] > d['std14']) & ~(d['stk14'].shift(1) > d['std14'].shift(1)).fillna(False)
    d['st10_fup'] = (d['st10'] == 1) & (d['st10'].shift(1) == -1)
    d['st10_fdn'] = (d['st10'] == -1) & (d['st10'].shift(1) == 1)
    d['psar_fup'] = (d['psar'] == 1) & (d['psar'].shift(1) == -1)
    d['psar_fdn'] = (d['psar'] == -1) & (d['psar'].shift(1) == 1)
    d['dc20_bup'] = c > d['dc20_u'].shift(1)

    # ATH
    d['ath'] = d['high'].cummax()
    d['ath_pct'] = (d['ath'] - c) / d['ath'] * 100

    # Candle patterns
    d['bull_eng'] = (c > d['open']) & (c.shift(1) < d['open'].shift(1)) & (c > d['open'].shift(1)) & (d['open'] < c.shift(1))

    return d


def weekly_flags(daily_df):
    """Compute weekly indicator flags, forward-filled to daily index."""
    w = daily_df.resample('W').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
    if len(w) < 52:
        return None
    c = w['close']
    flags = pd.DataFrame(index=w.index)
    flags['wk_above50e'] = c > ema(c, 50)
    flags['wk_e10_20'] = ema(c, 10) > ema(c, 20)
    r14 = rsi(c, 14)
    flags['wk_rsi_bull'] = r14 > 50
    flags['wk_rsi_nob'] = r14 < 70
    ml, sl_, _ = macd_calc(c)
    flags['wk_macd_bull'] = ml > sl_
    adx_v, _, _ = adx_calc(w, 14)
    flags['wk_adx_trend'] = adx_v > 20
    flags['wk_st_bull'] = supertrend_vec(w, 10, 3.0) == 1
    return flags.reindex(daily_df.index, method='ffill')


# ============================================================================
# VECTORIZED BACKTEST ENGINE
# ============================================================================

def vectorized_backtest(entry_signals: pd.Series, exit_signals: pd.Series,
                        close: pd.Series, sl_pct=10, tp_pct=20, max_hold=30,
                        start_idx=200):
    """
    Fast vectorized trade extraction from boolean entry/exit signal series.
    Returns list of (entry_idx, exit_idx, pnl_pct, hold_days, exit_reason).
    """
    close_vals = close.values
    entry_vals = entry_signals.values
    exit_vals = exit_signals.values
    n = len(close_vals)
    trades = []
    i = start_idx
    while i < n:
        # Find next entry
        if not entry_vals[i]:
            i += 1
            continue
        entry_price = close_vals[i]
        entry_i = i
        i += 1  # Move to next day after entry

        # Find exit
        while i < n:
            pnl = (close_vals[i] / entry_price - 1) * 100
            hold = i - entry_i
            if pnl <= -sl_pct:
                trades.append((entry_i, i, pnl, hold, 'sl'))
                i += 1; break
            elif pnl >= tp_pct:
                trades.append((entry_i, i, pnl, hold, 'tp'))
                i += 1; break
            elif hold >= max_hold:
                trades.append((entry_i, i, pnl, hold, 'mh'))
                i += 1; break
            elif exit_vals[i]:
                trades.append((entry_i, i, pnl, hold, 'sig'))
                i += 1; break
            i += 1
        else:
            # Still in trade at end of data
            if i > entry_i:
                pnl = (close_vals[min(i-1, n-1)] / entry_price - 1) * 100
                trades.append((entry_i, min(i-1, n-1), pnl, min(i-1, n-1) - entry_i, 'eod'))
    return trades


def run_strategy_all_stocks(all_enriched, weekly_maps, entry_col_fn, exit_col_fn,
                            sl=10, tp=20, hold=30, wk_filter_fn=None):
    """
    Run a strategy across all stocks using vectorized backtesting.

    entry_col_fn(df, wk) -> pd.Series of booleans (entry signals)
    exit_col_fn(df, wk) -> pd.Series of booleans (exit signals)
    wk_filter_fn(wk) -> pd.Series of booleans (weekly filter)
    """
    all_trades = []
    syms_traded = set()

    for sym, df in all_enriched.items():
        try:
            wk = weekly_maps.get(sym)

            # Compute entry/exit signals
            entry_sig = entry_col_fn(df, wk)
            exit_sig = exit_col_fn(df, wk)

            # Apply weekly filter
            if wk_filter_fn is not None and wk is not None:
                wk_ok = wk_filter_fn(wk)
                entry_sig = entry_sig & wk_ok

            trades = vectorized_backtest(entry_sig, exit_sig, df['close'], sl, tp, hold)
            if trades:
                syms_traded.add(sym)
                all_trades.extend(trades)
        except Exception:
            continue

    return _compute_metrics(all_trades, syms_traded)


def _compute_metrics(trades, syms_traded):
    if not trades:
        return {'total_trades':0, 'win_rate':0, 'profit_factor':0, 'calmar_ratio':0,
                'max_drawdown_pct':0, 'total_return_pct':0, 'avg_hold_days':0,
                'sharpe_ratio':0, 'symbols_traded':0, 'trades_per_year':0,
                'avg_win_pct':0, 'avg_loss_pct':0, 'wins':0, 'losses':0}

    pnls = np.array([t[2] for t in trades])
    holds = np.array([t[3] for t in trades])

    total = len(pnls)
    wins = pnls > 0
    n_wins = wins.sum()
    n_losses = total - n_wins
    wr = n_wins / total * 100

    avg_win = pnls[wins].mean() if n_wins > 0 else 0
    avg_loss = pnls[~wins].mean() if n_losses > 0 else 0

    gp = pnls[wins].sum() if n_wins > 0 else 0
    gl = abs(pnls[~wins].sum()) if n_losses > 0 else 0
    pf = gp / gl if gl > 0 else (99.99 if gp > 0 else 0)

    total_ret = pnls.sum()

    # Equity curve
    cum = np.cumprod(1 + pnls / 100)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak * 100
    max_dd = abs(dd.min()) if len(dd) > 0 else 0

    # Time-based metrics (approximate)
    avg_hold_per_trade = holds.mean()
    # Approximate total calendar days
    total_trade_days = holds.sum()
    years = max(total_trade_days / 252, 0.5)  # Approx trading days to years
    annual_ret = total_ret / years
    calmar = annual_ret / max_dd if max_dd > 0 else 0
    tpy = total / years

    # Sharpe
    sharpe = pnls.mean() / pnls.std() * np.sqrt(tpy) if pnls.std() > 0 else 0

    return {
        'total_trades': total, 'wins': int(n_wins), 'losses': int(n_losses),
        'win_rate': round(wr, 2), 'avg_win_pct': round(float(avg_win), 2),
        'avg_loss_pct': round(float(avg_loss), 2), 'total_return_pct': round(float(total_ret), 2),
        'max_drawdown_pct': round(float(max_dd), 2), 'calmar_ratio': round(float(calmar), 2),
        'profit_factor': round(float(min(pf, 99.99)), 2), 'avg_hold_days': round(float(holds.mean()), 1),
        'sharpe_ratio': round(float(sharpe), 2), 'symbols_traded': len(syms_traded),
        'trades_per_year': round(float(tpy), 1)
    }


# ============================================================================
# STRATEGY DEFINITIONS (vectorized)
# ============================================================================

def define_strategies():
    """Define 120+ strategies using vectorized column operations."""
    S = []

    def add(name, entry_fn, exit_fn, sl=10, tp=20, hold=30, wk_fn=None):
        S.append({'name': name, 'entry': entry_fn, 'exit': exit_fn,
                  'sl': sl, 'tp': tp, 'hold': hold, 'wk': wk_fn})

    # ==== MEAN REVERSION: RSI(2) / Connors ====
    add('MR_RSI2_lt5_SMA200',
        lambda d,w: (d['rsi_2'] < 5) & d['above200s'],
        lambda d,w: d['close'] > d['sma_5'],
        sl=8, tp=15, hold=10)

    add('MR_RSI2_lt10_EMA200',
        lambda d,w: (d['rsi_2'] < 10) & d['above200e'],
        lambda d,w: d['close'] > d['sma_5'],
        sl=8, tp=15, hold=10)

    add('MR_RSI2_lt5_EMAstack',
        lambda d,w: (d['rsi_2'] < 5) & d['estack'],
        lambda d,w: d['close'] > d['sma_5'],
        sl=8, tp=15, hold=10)

    add('MR_RSI2_lt5_MACD_pos',
        lambda d,w: (d['rsi_2'] < 5) & d['macd_pos'],
        lambda d,w: d['close'] > d['sma_5'],
        sl=8, tp=12, hold=10)

    add('MR_RSI2_lt10_ADX_EMA200',
        lambda d,w: (d['rsi_2'] < 10) & (d['adx'] > 25) & d['above200e'],
        lambda d,w: d['close'] > d['sma_5'],
        sl=8, tp=15, hold=10)

    add('MR_RSI2_lt3_SMA200',
        lambda d,w: (d['rsi_2'] < 3) & d['above200s'],
        lambda d,w: d['close'] > d['sma_5'],
        sl=8, tp=15, hold=10)

    add('MR_RSI2_lt5_exit_SMA10',
        lambda d,w: (d['rsi_2'] < 5) & d['above200s'],
        lambda d,w: d['close'] > d['sma_10'],
        sl=8, tp=15, hold=12)

    add('MR_RSI2_lt8_SMA200',
        lambda d,w: (d['rsi_2'] < 8) & d['above200s'],
        lambda d,w: d['close'] > d['sma_5'],
        sl=8, tp=15, hold=10)

    add('MR_RSI2_lt15_SMA200',
        lambda d,w: (d['rsi_2'] < 15) & d['above200s'],
        lambda d,w: d['close'] > d['sma_5'],
        sl=8, tp=15, hold=12)

    add('MR_RSI5_lt20_EMA50',
        lambda d,w: (d['rsi_5'] < 20) & d['above50e'],
        lambda d,w: d['rsi_5'] > 70,
        sl=8, tp=15, hold=15)

    add('MR_RSI14_lt30_SMA200',
        lambda d,w: (d['rsi_14'] < 30) & d['above200s'],
        lambda d,w: d['rsi_14'] > 65,
        sl=10, tp=20, hold=25)

    add('MR_RSI14_lt35_EMAstack_ADX',
        lambda d,w: (d['rsi_14'] < 35) & d['estack'] & (d['adx'] > 20),
        lambda d,w: d['rsi_14'] > 60,
        sl=8, tp=15, hold=15)

    # ==== MEAN REVERSION: Williams %R ====
    add('MR_WR2_lt95_SMA200',
        lambda d,w: (d['wr2'] < -95) & d['above200s'],
        lambda d,w: d['wr2'] > -20,
        sl=8, tp=12, hold=10)

    add('MR_WR5_lt90_EMA50',
        lambda d,w: (d['wr5'] < -90) & d['above50e'],
        lambda d,w: d['wr5'] > -20,
        sl=8, tp=15, hold=15)

    add('MR_WR2_lt95_EMAstack',
        lambda d,w: (d['wr2'] < -95) & d['estack'],
        lambda d,w: d['wr2'] > -30,
        sl=7, tp=12, hold=10)

    add('MR_WR2_lt90_SMA200',
        lambda d,w: (d['wr2'] < -90) & d['above200s'],
        lambda d,w: d['wr2'] > -20,
        sl=8, tp=12, hold=10)

    add('MR_WR2_lt98_SMA200',
        lambda d,w: (d['wr2'] < -98) & d['above200s'],
        lambda d,w: d['wr2'] > -20,
        sl=8, tp=12, hold=10)

    # ==== MEAN REVERSION: Keltner Channels ====
    add('MR_KC6_lower_SMA200',
        lambda d,w: (d['close'] < d['kc6_l']) & d['above200s'],
        lambda d,w: d['close'] > d['kc6_m'],
        sl=8, tp=15, hold=15)

    add('MR_KC20_lower_RSI40',
        lambda d,w: (d['close'] < d['kc_l']) & (d['rsi_14'] < 40),
        lambda d,w: d['close'] > d['kc_m'],
        sl=10, tp=15, hold=20)

    add('MR_KC6_lower_EMA50',
        lambda d,w: (d['close'] < d['kc6_l']) & d['above50e'],
        lambda d,w: d['close'] > d['kc6_m'],
        sl=8, tp=12, hold=15)

    # ==== MEAN REVERSION: Bollinger Bands ====
    add('MR_BB_lower_RSI30',
        lambda d,w: (d['bb_pctb'] < 0.05) & (d['rsi_14'] < 30),
        lambda d,w: d['bb_pctb'] > 0.5,
        sl=10, tp=15, hold=20)

    add('MR_BB_lower_SMA200',
        lambda d,w: (d['close'] < d['bb_l']) & d['above200s'],
        lambda d,w: d['close'] > d['bb_m'],
        sl=8, tp=15, hold=15)

    add('MR_BB10_lower_EMA50',
        lambda d,w: (d['close'] < d['bb10_l']) & d['above50e'],
        lambda d,w: d['close'] > d['bb10_m'],
        sl=8, tp=12, hold=10)

    add('MR_BB_lower_EMAstack_vol',
        lambda d,w: (d['close'] < d['bb_l']) & d['e9_21'] & (d['vol_r'] > 1.5),
        lambda d,w: d['close'] > d['bb_m'],
        sl=8, tp=12, hold=15)

    # ==== MEAN REVERSION: CCI ====
    add('MR_CCI20_lt100_EMA200',
        lambda d,w: (d['cci20'] < -100) & d['above200e'],
        lambda d,w: d['cci20'] > 100,
        sl=10, tp=15, hold=25)

    add('MR_CCI14_lt100_EMA50',
        lambda d,w: (d['cci14'] < -100) & d['above50e'],
        lambda d,w: d['cci14'] > 50,
        sl=10, tp=15, hold=20)

    # ==== MEAN REVERSION: MFI ====
    add('MR_MFI14_lt20_EMA200',
        lambda d,w: (d['mfi14'] < 20) & d['above200e'],
        lambda d,w: d['mfi14'] > 80,
        sl=10, tp=15, hold=25)

    add('MR_MFI10_RSI_double',
        lambda d,w: (d['mfi10'] < 25) & (d['rsi_14'] < 35) & d['above200s'],
        lambda d,w: (d['mfi10'] > 75) | (d['rsi_14'] > 65),
        sl=10, tp=15, hold=20)

    # ==== MEAN REVERSION: Stochastic ====
    add('MR_Stoch14_oversold_EMA200',
        lambda d,w: d['stk14_xup'] & (d['stk14'] < 30) & d['above200e'],
        lambda d,w: d['stk14'] > 80,
        sl=8, tp=15, hold=20)

    add('MR_Stoch14_RSI_double',
        lambda d,w: (d['stk14'] < 20) & (d['rsi_14'] < 35) & d['above200s'],
        lambda d,w: (d['stk14'] > 70) | (d['rsi_14'] > 65),
        sl=10, tp=15, hold=20)

    add('MR_Stoch_oversold_EMAstack',
        lambda d,w: (d['stk14'] < 20) & d['e9_21'] & d['above200e'],
        lambda d,w: d['stk14'] > 75,
        sl=8, tp=15, hold=20)

    # ==== MEAN REVERSION: Volume spike ====
    add('MR_VolSpike_RSI_oversold',
        lambda d,w: (d['vol_r'] > 2.0) & (d['rsi_14'] < 35) & d['above200e'],
        lambda d,w: d['rsi_14'] > 60,
        sl=10, tp=15, hold=20)

    # ==== MEAN REVERSION: ROC ====
    add('MR_ROC9_neg_SMA200',
        lambda d,w: (d['roc9'] < -8) & d['above200s'],
        lambda d,w: d['roc9'] > 5,
        sl=10, tp=15, hold=20)

    # ==== TREND: EMA Crossovers ====
    add('TR_EMA9_21_cross_EMA200',
        lambda d,w: d['e9_21_xup'] & d['above200e'],
        lambda d,w: d['e9_21_xdn'],
        sl=10, tp=20, hold=30)

    add('TR_EMA9_21_cross_RSI',
        lambda d,w: d['e9_21_xup'] & (d['rsi_14'] > 45) & (d['rsi_14'] < 70),
        lambda d,w: d['e9_21_xdn'] | (d['rsi_14'] > 75),
        sl=8, tp=15, hold=25)

    add('TR_EMA9_21_cross_vol',
        lambda d,w: d['e9_21_xup'] & (d['vol_r'] > 1.3),
        lambda d,w: d['e9_21_xdn'],
        sl=10, tp=20, hold=30)

    add('TR_EMA20_50_cross_ADX',
        lambda d,w: d['e20_50_xup'] & (d['adx'] > 25),
        lambda d,w: d['close'] < d['ema_20'],
        sl=12, tp=25, hold=30)

    # ==== TREND: MACD ====
    add('TR_MACD_cross_RSI60',
        lambda d,w: d['macd_xup'] & (d['rsi_14'] < 60),
        lambda d,w: d['macd_xdn'] | (d['rsi_14'] > 75),
        sl=10, tp=20, hold=25)

    add('TR_MACD_cross_EMA200_ADX',
        lambda d,w: d['macd_xup'] & d['above200e'] & (d['adx'] > 20),
        lambda d,w: d['macd_xdn'],
        sl=10, tp=20, hold=30)

    add('TR_MACD_cross_OBV',
        lambda d,w: d['macd_xup'] & (d['obv'] > d['obv_ema']),
        lambda d,w: d['macd_xdn'],
        sl=10, tp=20, hold=30)

    add('TR_Triple_MACD_RSI_EMA',
        lambda d,w: d['macd_xup'] & (d['rsi_14'] > 50) & (d['rsi_14'] < 70) & d['e9_21'],
        lambda d,w: d['macd_xdn'] | (d['rsi_14'] > 80),
        sl=10, tp=20, hold=25)

    add('TR_MACD_ST_ADX',
        lambda d,w: d['macd_xup'] & (d['st10'] == 1) & (d['adx'] > 25),
        lambda d,w: (d['st10'] == -1) | d['macd_xdn'],
        sl=12, tp=25, hold=30)

    add('TR_MACD_PSAR_EMA200',
        lambda d,w: d['macd_xup'] & (d['psar'] == 1) & d['above200e'],
        lambda d,w: d['psar_fdn'] | d['macd_xdn'],
        sl=10, tp=20, hold=30)

    add('TR_FastMACD_RSI',
        lambda d,w: (d['fmacd_h'] > 0) & (d['rsi_14'] < 65) & (d['rsi_14'] > 40),
        lambda d,w: (d['fmacd_h'] < 0) & (d['rsi_14'] > 70),
        sl=8, tp=15, hold=20)

    # ==== TREND: Supertrend ====
    add('TR_ST_flip_RSI',
        lambda d,w: d['st10_fup'] & (d['rsi_14'] > 40) & (d['rsi_14'] < 65),
        lambda d,w: d['st10_fdn'],
        sl=10, tp=20, hold=30)

    add('TR_ST_EMA_aligned',
        lambda d,w: d['st10_fup'] & d['e9_21'] & d['above50e'],
        lambda d,w: d['st10_fdn'],
        sl=10, tp=20, hold=30)

    add('TR_ST_ADX',
        lambda d,w: d['st10_fup'] & (d['adx'] > 25) & (d['pdi'] > d['mdi']),
        lambda d,w: d['st10_fdn'],
        sl=12, tp=25, hold=30)

    add('TR_ST_RSI_EMA_research',
        lambda d,w: (d['st10'] == 1) & (d['rsi_14'] > 50) & d['above50e'],
        lambda d,w: (d['st10'] == -1) | (d['rsi_14'] > 80),
        sl=12, tp=25, hold=30)

    # ==== TREND: PSAR ====
    add('TR_PSAR_ADX',
        lambda d,w: d['psar_fup'] & (d['adx'] > 25),
        lambda d,w: d['psar_fdn'],
        sl=10, tp=20, hold=30)

    add('TR_PSAR_EMA200_vol',
        lambda d,w: d['psar_fup'] & d['above200e'] & (d['vol_r'] > 1.2),
        lambda d,w: d['psar_fdn'],
        sl=10, tp=20, hold=30)

    # ==== TREND: ADX ====
    add('TR_ADX_DI_EMA50',
        lambda d,w: (d['adx'] > 25) & (d['pdi'] > d['mdi']) & d['above50e'],
        lambda d,w: (d['pdi'] < d['mdi']) | (d['adx'] < 20),
        sl=10, tp=20, hold=30)

    add('TR_ADX_MACD_EMA',
        lambda d,w: (d['adx'] > 20) & d['macd_pos'] & d['e9_21'],
        lambda d,w: (d['adx'] < 15) | d['macd_xdn'],
        sl=10, tp=20, hold=30)

    # ==== BREAKOUT ====
    add('BO_BB_squeeze_DC',
        lambda d,w: d['squeeze'] & d['dc20_bup'],
        lambda d,w: d['close'] < d['ema_20'],
        sl=8, tp=20, hold=25)

    add('BO_DC20_vol_EMA50',
        lambda d,w: d['dc20_bup'] & (d['vol_r'] > 1.5) & d['above50e'],
        lambda d,w: d['close'] < d['dc20_m'],
        sl=10, tp=20, hold=30)

    add('BO_ATH_vol',
        lambda d,w: (d['ath_pct'] < 1) & (d['vol_r'] > 1.5),
        lambda d,w: d['close'] < d['ema_20'],
        sl=8, tp=20, hold=25)

    add('BO_NearATH_ADX_EMAstack',
        lambda d,w: (d['ath_pct'] < 5) & (d['adx'] > 25) & d['estack'],
        lambda d,w: (d['close'] < d['ema_21']) | (d['ath_pct'] > 15),
        sl=8, tp=15, hold=25)

    # ==== VOLUME + MOMENTUM ====
    add('VM_OBV_EMA_cross',
        lambda d,w: (d['obv'] > d['obv_ema']) & d['e9_21_xup'],
        lambda d,w: (d['obv'] < d['obv_ema']) & d['e9_21_xdn'],
        sl=10, tp=20, hold=30)

    add('VM_CMF_pos_EMA50',
        lambda d,w: (d['cmf'] > 0.1) & d['above50e'] & (d['rsi_14'] < 70),
        lambda d,w: d['cmf'] < -0.1,
        sl=10, tp=20, hold=30)

    add('VM_ROC12_ADX_EMA',
        lambda d,w: (d['roc12'] > 3) & (d['adx'] > 25) & d['e9_21'] & d['above50e'],
        lambda d,w: (d['roc12'] < -3) | (d['adx'] < 20),
        sl=10, tp=20, hold=25)

    add('VM_TSI_ST_vol',
        lambda d,w: (d['tsi'] > 5) & (d['st10'] == 1) & (d['vol_r'] > 1.2),
        lambda d,w: (d['tsi'] < -5) | (d['st10'] == -1),
        sl=10, tp=20, hold=30)

    # ==== PULLBACK ====
    add('PB_EMA21_uptrend',
        lambda d,w: (((d['close'] - d['ema_21']).abs() / d['ema_21'] * 100) < 1.5) & d['estack'] & (d['rsi_14'] < 55),
        lambda d,w: (d['close'] < d['ema_50']) | (d['rsi_14'] > 75),
        sl=8, tp=15, hold=20)

    add('PB_EMA50_strong',
        lambda d,w: (((d['close'] - d['ema_50']).abs() / d['ema_50'] * 100) < 2) & d['above200e'] & (d['adx'] > 25),
        lambda d,w: (d['close'] > d['ema_20']) & (d['rsi_14'] > 65),
        sl=10, tp=20, hold=25)

    add('PB_EMA9_lowvol',
        lambda d,w: (((d['close'] - d['ema_9']).abs() / d['ema_9'] * 100) < 1) & d['estack'] & (d['vol_r'] < 0.7),
        lambda d,w: (d['vol_r'] > 1.5) & (d['close'] > d['ema_9']),
        sl=5, tp=10, hold=10)

    # ==== CANDLE PATTERNS ====
    add('CP_BullEngulf_EMA200_RSI',
        lambda d,w: d['bull_eng'] & d['above200e'] & (d['rsi_14'] < 50),
        lambda d,w: (d['rsi_14'] > 70) | (d['close'] < d['ema_21']),
        sl=8, tp=15, hold=20)

    # ==== MULTI-INDICATOR COMBOS ====
    add('MC_Triple_RSI_MACD_OBV',
        lambda d,w: (d['rsi_14'] > 50) & d['macd_pos'] & (d['obv'] > d['obv_ema']) & d['above200e'],
        lambda d,w: (d['rsi_14'] > 80) | (d['macd_xdn'] & (d['obv'] < d['obv_ema'])),
        sl=10, tp=20, hold=30)

    add('MC_MACD_RSI_ST',
        lambda d,w: d['macd_pos'] & (d['rsi_14'] > 50) & (d['rsi_14'] < 70) & (d['st10'] == 1),
        lambda d,w: (d['st10'] == -1) | (d['rsi_14'] > 80),
        sl=10, tp=20, hold=30)

    add('MC_Quad_EMA_RSI_ADX_vol',
        lambda d,w: d['e9_21_xup'] & (d['rsi_14'] > 45) & (d['rsi_14'] < 65) & (d['adx'] > 20) & (d['vol_r'] > 1.2),
        lambda d,w: d['e9_21_xdn'] | (d['rsi_14'] > 80),
        sl=10, tp=20, hold=25)

    add('MC_KC_RSI_MACD',
        lambda d,w: (d['close'] < d['kc_l']) & (d['rsi_14'] < 35),
        lambda d,w: (d['close'] > d['kc_m']) | (d['rsi_14'] > 70),
        sl=10, tp=15, hold=20)

    # ==== TOP-DOWN: WEEKLY FILTER + DAILY ====
    add('TD_wEMA50_dRSI2_lt10',
        lambda d,w: (d['rsi_2'] < 10) & d['above200e'],
        lambda d,w: d['close'] > d['sma_5'],
        sl=8, tp=12, hold=10,
        wk_fn=lambda wk: wk['wk_above50e'].fillna(False))

    add('TD_wMACD_dEMA_cross',
        lambda d,w: d['e9_21_xup'] & d['above50e'],
        lambda d,w: d['e9_21_xdn'],
        sl=10, tp=20, hold=30,
        wk_fn=lambda wk: wk['wk_macd_bull'].fillna(False))

    add('TD_wADX_dMACD_cross',
        lambda d,w: d['macd_xup'] & d['above50e'],
        lambda d,w: d['macd_xdn'],
        sl=10, tp=20, hold=30,
        wk_fn=lambda wk: wk['wk_adx_trend'].fillna(False))

    add('TD_wST_dStoch_oversold',
        lambda d,w: (d['stk14'] < 25) & d['above50e'],
        lambda d,w: d['stk14'] > 80,
        sl=8, tp=15, hold=20,
        wk_fn=lambda wk: wk['wk_st_bull'].fillna(False))

    add('TD_wRSI_dBB_lower',
        lambda d,w: (d['close'] < d['bb_l']) & d['above200s'],
        lambda d,w: d['close'] > d['bb_m'],
        sl=8, tp=12, hold=15,
        wk_fn=lambda wk: wk['wk_rsi_bull'].fillna(False) & wk['wk_rsi_nob'].fillna(False))

    add('TD_wEMA_dRSI_pullback',
        lambda d,w: (d['rsi_14'] < 40) & d['above200e'] & d['e20_50'],
        lambda d,w: d['rsi_14'] > 65,
        sl=10, tp=15, hold=20,
        wk_fn=lambda wk: wk['wk_e10_20'].fillna(False) & wk['wk_above50e'].fillna(False))

    add('TD_wST_ADX_dEMA_vol',
        lambda d,w: d['e9_21_xup'] & (d['vol_r'] > 1.3) & d['above200e'],
        lambda d,w: d['e9_21_xdn'],
        sl=10, tp=20, hold=30,
        wk_fn=lambda wk: wk['wk_st_bull'].fillna(False) & wk['wk_adx_trend'].fillna(False))

    add('TD_wAllBull_dMACD_ST',
        lambda d,w: d['macd_xup'] & (d['st10'] == 1),
        lambda d,w: d['st10'] == -1,
        sl=12, tp=25, hold=30,
        wk_fn=lambda wk: wk['wk_above50e'].fillna(False) & wk['wk_rsi_bull'].fillna(False) & wk['wk_macd_bull'].fillna(False))

    add('TD_wAllBull_dPullback',
        lambda d,w: (d['rsi_14'] < 45) & d['above50e'] & d['e20_50'],
        lambda d,w: d['rsi_14'] > 70,
        sl=10, tp=18, hold=25,
        wk_fn=lambda wk: wk['wk_above50e'].fillna(False) & wk['wk_st_bull'].fillna(False) & wk['wk_rsi_nob'].fillna(False))

    add('TD_wDouble_dRSI2_lt10',
        lambda d,w: (d['rsi_2'] < 10) & d['above200e'],
        lambda d,w: d['close'] > d['sma_5'],
        sl=8, tp=12, hold=10,
        wk_fn=lambda wk: wk['wk_rsi_bull'].fillna(False) & wk['wk_st_bull'].fillna(False))

    add('TD_wTriple_dBB_lower',
        lambda d,w: (d['close'] < d['bb_l']) & d['above200s'],
        lambda d,w: d['close'] > d['bb_m'],
        sl=8, tp=12, hold=15,
        wk_fn=lambda wk: wk['wk_rsi_bull'].fillna(False) & wk['wk_above50e'].fillna(False) & wk['wk_macd_bull'].fillna(False))

    add('TD_wMACD_ADX_dStoch_RSI',
        lambda d,w: (d['stk14'] < 20) & (d['rsi_14'] < 40) & d['above50e'],
        lambda d,w: (d['stk14'] > 70) | (d['rsi_14'] > 60),
        sl=8, tp=15, hold=20,
        wk_fn=lambda wk: wk['wk_macd_bull'].fillna(False) & wk['wk_adx_trend'].fillna(False))

    add('TD_wAllBull_dWR2',
        lambda d,w: (d['wr2'] < -95) & d['above200s'],
        lambda d,w: d['wr2'] > -30,
        sl=8, tp=12, hold=10,
        wk_fn=lambda wk: wk['wk_rsi_bull'].fillna(False) & wk['wk_st_bull'].fillna(False) & wk['wk_macd_bull'].fillna(False))

    add('TD_wEMA_RSI_dCCI',
        lambda d,w: (d['cci20'] < -100) & d['above200e'],
        lambda d,w: d['cci20'] > 50,
        sl=10, tp=15, hold=20,
        wk_fn=lambda wk: wk['wk_e10_20'].fillna(False) & wk['wk_rsi_nob'].fillna(False))

    # ==== PARAMETER VARIATIONS (SL/TP combos) ====
    for sl, tp in [(5,10), (6,12), (8,15), (10,20), (12,25), (15,30)]:
        add(f'MR_RSI2_lt5_SMA200_SL{sl}_TP{tp}',
            lambda d,w: (d['rsi_2'] < 5) & d['above200s'],
            lambda d,w: d['close'] > d['sma_5'],
            sl=sl, tp=tp, hold=10)

    # Hold period variations
    for hold in [5, 8, 10, 15, 20, 25, 30]:
        add(f'MR_RSI2_lt5_SMA200_H{hold}',
            lambda d,w: (d['rsi_2'] < 5) & d['above200s'],
            lambda d,w: d['close'] > d['sma_5'],
            sl=8, tp=15, hold=hold)

    # Weekly filter variations for top strategy
    add('MR_RSI2_lt5_wMACD',
        lambda d,w: (d['rsi_2'] < 5) & d['above200s'],
        lambda d,w: d['close'] > d['sma_5'],
        sl=8, tp=12, hold=10,
        wk_fn=lambda wk: wk['wk_macd_bull'].fillna(False))

    add('MR_RSI2_lt5_wST',
        lambda d,w: (d['rsi_2'] < 5) & d['above200s'],
        lambda d,w: d['close'] > d['sma_5'],
        sl=8, tp=12, hold=10,
        wk_fn=lambda wk: wk['wk_st_bull'].fillna(False))

    add('MR_RSI2_lt5_wADX',
        lambda d,w: (d['rsi_2'] < 5) & d['above200s'],
        lambda d,w: d['close'] > d['sma_5'],
        sl=8, tp=12, hold=10,
        wk_fn=lambda wk: wk['wk_adx_trend'].fillna(False))

    add('MR_RSI2_lt5_wRSI',
        lambda d,w: (d['rsi_2'] < 5) & d['above200s'],
        lambda d,w: d['close'] > d['sma_5'],
        sl=8, tp=12, hold=10,
        wk_fn=lambda wk: wk['wk_rsi_bull'].fillna(False))

    # ==== HYBRID: Mean reversion entry, trend exit ====
    add('HY_RSI2_entry_EMA_exit',
        lambda d,w: (d['rsi_2'] < 5) & d['above200s'],
        lambda d,w: d['e9_21_xdn'],
        sl=10, tp=25, hold=30)

    add('HY_BB_entry_ST_exit',
        lambda d,w: (d['close'] < d['bb_l']) & d['above200s'],
        lambda d,w: d['st10'] == -1,
        sl=10, tp=25, hold=30)

    add('HY_WR2_entry_MACD_exit',
        lambda d,w: (d['wr2'] < -95) & d['above200s'],
        lambda d,w: d['macd_xdn'],
        sl=10, tp=20, hold=25)

    add('HY_KC_entry_RSI_exit',
        lambda d,w: (d['close'] < d['kc6_l']) & d['above200s'],
        lambda d,w: d['rsi_14'] > 65,
        sl=8, tp=15, hold=20)

    # ==== EXTREME OVERSOLD (multi-indicator confirmation) ====
    add('EO_RSI_WR_Stoch_triple',
        lambda d,w: (d['rsi_2'] < 10) & (d['wr2'] < -90) & (d['stk14'] < 20) & d['above200s'],
        lambda d,w: d['close'] > d['sma_5'],
        sl=8, tp=12, hold=10)

    add('EO_RSI_BB_MFI_triple',
        lambda d,w: (d['rsi_14'] < 30) & (d['close'] < d['bb_l']) & (d['mfi14'] < 25) & d['above200e'],
        lambda d,w: (d['rsi_14'] > 55) | (d['close'] > d['bb_m']),
        sl=10, tp=15, hold=20)

    add('EO_RSI_CCI_WR_quad',
        lambda d,w: (d['rsi_2'] < 8) & (d['cci20'] < -100) & (d['wr14'] < -80) & d['above200s'],
        lambda d,w: d['close'] > d['sma_5'],
        sl=8, tp=15, hold=12)

    add('EO_all_oversold',
        lambda d,w: (d['rsi_2'] < 5) & (d['wr2'] < -95) & (d['stk14'] < 15) & (d['cci20'] < -150) & d['above200s'],
        lambda d,w: d['close'] > d['sma_5'],
        sl=8, tp=12, hold=10)

    print(f"Total strategies defined: {len(S)}")
    return S


# ============================================================================
# MAIN
# ============================================================================

def main():
    t0 = time.time()
    print("=" * 80)
    print("STRATEGY RESEARCH SCANNER V2 (Vectorized)")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load data
    print("[1/4] Loading data...")
    all_data = load_all_daily_data()
    print()

    # Enrich
    print("[2/4] Computing indicators...")
    enriched = {}
    wk_maps = {}
    total = len(all_data)
    for i, (sym, df) in enumerate(all_data.items(), 1):
        if i % 50 == 0 or i == 1:
            print(f"  {i}/{total}: {sym}...")
        try:
            enriched[sym] = enrich(df)
            wf = weekly_flags(df)
            if wf is not None:
                wk_maps[sym] = wf
        except Exception as e:
            continue
    print(f"  Done: {len(enriched)} stocks enriched, {len(wk_maps)} with weekly")
    print(f"  Time: {time.time()-t0:.0f}s")
    print()

    # Strategies
    print("[3/4] Defining strategies...")
    strategies = define_strategies()
    print()

    # Run
    print("[4/4] Running strategies...")
    results = []
    for i, s in enumerate(strategies, 1):
        name = s['name']
        try:
            r = run_strategy_all_stocks(
                enriched, wk_maps,
                s['entry'], s['exit'],
                s['sl'], s['tp'], s['hold'],
                s.get('wk'))
            r['name'] = name
            results.append(r)
            t = r['total_trades']; wr = r['win_rate']; pf = r['profit_factor']; cal = r['calmar_ratio']
            flag = " ***" if t >= 250 and wr >= 65 and cal >= 1 and pf >= 1.5 else ""
            print(f"  [{i}/{len(strategies)}] {name}: trades={t}, WR={wr}%, PF={pf}, Calmar={cal}{flag}")
        except Exception as e:
            print(f"  [{i}/{len(strategies)}] {name}: ERROR {e}")
            results.append({'name': name, 'error': str(e), 'total_trades': 0})

        if i % 20 == 0:
            _interim_save(results)

    # Results
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)

    df = pd.DataFrame([r for r in results if r.get('total_trades', 0) > 0])
    df.to_csv(CSV_RESULTS_PATH, index=False)

    # Qualifying
    q = df[(df['total_trades'] >= 250) & (df['win_rate'] >= 65) & (df['calmar_ratio'] >= 1.0) & (df['profit_factor'] >= 1.5)]
    q = q.sort_values('calmar_ratio', ascending=False)
    print(f"\nFULL CRITERIA (250+ trades, 65%+ WR, 1+ Calmar, 1.5+ PF): {len(q)} strategies")
    if len(q) > 0:
        print(q[['name','total_trades','win_rate','calmar_ratio','profit_factor','sharpe_ratio','total_return_pct','max_drawdown_pct','avg_hold_days','symbols_traded']].to_string())

    # Near miss
    nm = df[(df['total_trades'] >= 150) & (df['win_rate'] >= 60) & (df['profit_factor'] >= 1.2)]
    nm = nm.sort_values('win_rate', ascending=False)
    print(f"\nNEAR MISS (150+ trades, 60%+ WR, 1.2+ PF): {len(nm)} strategies")
    if len(nm) > 0:
        print(nm[['name','total_trades','win_rate','calmar_ratio','profit_factor','sharpe_ratio','total_return_pct','avg_hold_days']].head(30).to_string())

    # By category
    print("\n" + "=" * 80)
    print("TOP 10 BY WIN RATE (100+ trades):")
    v = df[df['total_trades'] >= 100].sort_values('win_rate', ascending=False)
    print(v[['name','total_trades','win_rate','profit_factor','calmar_ratio','avg_hold_days']].head(10).to_string())

    print("\nTOP 10 BY CALMAR (100+ trades):")
    v = df[df['total_trades'] >= 100].sort_values('calmar_ratio', ascending=False)
    print(v[['name','total_trades','win_rate','profit_factor','calmar_ratio','total_return_pct']].head(10).to_string())

    print("\nTOP 10 BY PROFIT FACTOR (100+ trades):")
    v = df[df['total_trades'] >= 100].sort_values('profit_factor', ascending=False)
    print(v[['name','total_trades','win_rate','profit_factor','calmar_ratio','avg_hold_days']].head(10).to_string())

    print("\nTOP 10 BY SHARPE (100+ trades):")
    v = df[df['total_trades'] >= 100].sort_values('sharpe_ratio', ascending=False)
    print(v[['name','total_trades','win_rate','profit_factor','sharpe_ratio','calmar_ratio']].head(10).to_string())

    # Save JSON
    out = {
        'run_date': datetime.now().isoformat(),
        'stocks': len(enriched),
        'strategies_tested': len(strategies),
        'qualifying': q.to_dict('records') if len(q) > 0 else [],
        'near_miss': nm.to_dict('records') if len(nm) > 0 else [],
        'all': df.to_dict('records')
    }
    with open(RESULTS_PATH, 'w') as f:
        json.dump(out, f, indent=2, default=str)

    print(f"\nTotal time: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)")
    print(f"CSV: {CSV_RESULTS_PATH}")
    print(f"JSON: {RESULTS_PATH}")


def _interim_save(results):
    valid = [r for r in results if r.get('total_trades', 0) > 0]
    if valid:
        try:
            pd.DataFrame(valid).to_csv(CSV_RESULTS_PATH, index=False)
        except: pass


if __name__ == '__main__':
    main()
