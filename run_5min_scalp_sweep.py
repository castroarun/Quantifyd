"""
5-Minute Scalping Strategy Sweep (Vectorized)
===============================================
Tests 6 intraday scalping strategies on 10 premium stocks (5-min data, 2018-2025).
Processes each stock independently with vectorized signal generation for speed.

Strategies: ORB, VWAP+EMA, EMA Crossover, SuperTrend, First Candle, MACD Reversal
"""

import os, sys, csv, time, logging
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

logging.disable(logging.WARNING)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from services.intraday_backtest_engine import load_data_from_db

SYMBOLS = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
           'SBIN', 'BHARTIARTL', 'ITC', 'KOTAKBANK', 'HINDUNILVR']
INITIAL_CAPITAL = 10_000_000
POSITION_SIZE_PCT = 0.10
MAX_POSITIONS = 5

OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'scalp_5min_results.csv')
FIELDNAMES = [
    'label', 'strategy', 'cost_mode', 'total_trades', 'win_rate', 'profit_factor',
    'total_pnl', 'total_pnl_pct', 'cagr', 'max_drawdown', 'sharpe', 'sortino',
    'calmar', 'avg_win', 'avg_loss', 'avg_rr', 'avg_bars_held',
    'long_trades', 'short_trades', 'long_wr', 'short_wr',
    'exit_sl', 'exit_tp', 'exit_eod', 'exit_other',
    'y2018_trades', 'y2018_wr', 'y2018_pnl_pct',
    'y2019_trades', 'y2019_wr', 'y2019_pnl_pct',
    'y2020_trades', 'y2020_wr', 'y2020_pnl_pct',
    'y2021_trades', 'y2021_wr', 'y2021_pnl_pct',
    'y2022_trades', 'y2022_wr', 'y2022_pnl_pct',
    'y2023_trades', 'y2023_wr', 'y2023_pnl_pct',
    'y2024_trades', 'y2024_wr', 'y2024_pnl_pct',
    'y2025_trades', 'y2025_wr', 'y2025_pnl_pct',
]


# ============================================================================
# Vectorized indicator helpers (avoid Python loops)
# ============================================================================

def fast_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def fast_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)

def fast_atr(df, period=14):
    h_l = df['high'] - df['low']
    h_pc = (df['high'] - df['close'].shift(1)).abs()
    l_pc = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def fast_macd(series, fast=12, slow=26, signal=9):
    ema_f = fast_ema(series, fast)
    ema_s = fast_ema(series, slow)
    macd = ema_f - ema_s
    sig = fast_ema(macd, signal)
    hist = macd - sig
    return macd, sig, hist

def fast_supertrend(df, atr_period=10, multiplier=3.0):
    """Vectorized SuperTrend using numpy for the loop."""
    atr = fast_atr(df, atr_period)
    hl2 = (df['high'].values + df['low'].values) / 2
    upper = hl2 + multiplier * atr.values
    lower = hl2 - multiplier * atr.values
    close = df['close'].values
    n = len(df)

    st = np.empty(n)
    d = np.empty(n, dtype=np.int32)
    st[0] = upper[0]
    d[0] = 1

    for i in range(1, n):
        if close[i] > st[i-1]:
            st[i] = lower[i]
            d[i] = 1
        elif close[i] < st[i-1]:
            st[i] = upper[i]
            d[i] = -1
        else:
            st[i] = st[i-1]
            d[i] = d[i-1]
            if d[i] == 1 and lower[i] > st[i]:
                st[i] = lower[i]
            elif d[i] == -1 and upper[i] < st[i]:
                st[i] = upper[i]

    return pd.Series(st, index=df.index), pd.Series(d, index=df.index)


# ============================================================================
# Vectorized trade simulation per stock
# ============================================================================

def simulate_trades_vectorized(
    trades_df: pd.DataFrame,
    initial_capital: float,
    position_size_pct: float,
    commission_pct: float,
    slippage_pct: float,
) -> dict:
    """
    Given a DataFrame of trade signals with columns:
    [entry_price, sl, tp, direction (1=long, -1=short), entry_date,
     day_high_after, day_low_after, eod_close, year]

    Simulate the trades and compute stats.

    For each trade:
    - Check if SL or TP is hit within the day's remaining range
    - If neither, exit at EOD close
    """
    if len(trades_df) == 0:
        return _empty_result()

    trades = trades_df.copy()
    capital = initial_capital
    pos_size = initial_capital * position_size_pct  # Fixed sizing

    results = []
    equity_by_date = {}
    peak = initial_capital
    max_dd = 0.0

    for _, t in trades.iterrows():
        entry = t['entry_price']
        sl = t['sl']
        tp = t['tp']
        direction = t['direction']  # 1 or -1
        day_high = t['day_high_after']
        day_low = t['day_low_after']
        eod_close = t['eod_close']
        year = t['year']
        entry_date = t['entry_date']

        # Apply slippage to entry
        if direction == 1:
            entry_adj = entry * (1 + slippage_pct)
        else:
            entry_adj = entry * (1 - slippage_pct)

        qty = max(int(pos_size / entry_adj), 1)

        # Determine exit
        exit_price = None
        exit_type = None

        if direction == 1:  # LONG
            # Check SL first (conservative: SL before TP if both could hit)
            if day_low <= sl:
                exit_price = sl
                exit_type = 'fixed_sl'
            elif tp is not None and day_high >= tp:
                exit_price = tp
                exit_type = 'fixed_tp'
            else:
                exit_price = eod_close
                exit_type = 'eod'
        else:  # SHORT
            if day_high >= sl:
                exit_price = sl
                exit_type = 'fixed_sl'
            elif tp is not None and day_low <= tp:
                exit_price = tp
                exit_type = 'fixed_tp'
            else:
                exit_price = eod_close
                exit_type = 'eod'

        # Apply slippage to exit
        if direction == 1:
            exit_adj = exit_price * (1 - slippage_pct)
        else:
            exit_adj = exit_price * (1 + slippage_pct)

        # PnL
        if direction == 1:
            pnl_per_share = exit_adj - entry_adj
        else:
            pnl_per_share = entry_adj - exit_adj

        gross_pnl = pnl_per_share * qty
        commission = (entry_adj + exit_adj) * qty * commission_pct
        net_pnl = gross_pnl - commission
        pnl_pct = (pnl_per_share / entry_adj) * 100

        capital += net_pnl

        results.append({
            'pnl': net_pnl,
            'pnl_pct': pnl_pct,
            'direction': direction,
            'exit_type': exit_type,
            'year': year,
            'entry_date': entry_date,
        })

        # Track equity by date
        date_key = str(entry_date)[:10]
        equity_by_date[date_key] = capital

        if capital > peak:
            peak = capital
        dd = (peak - capital) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    return _compute_stats(results, equity_by_date, initial_capital, max_dd, capital)


def _empty_result():
    row = {k: 0 for k in FIELDNAMES if k not in ('label', 'strategy', 'cost_mode')}
    return row


def _compute_stats(results, equity_by_date, initial_capital, max_dd, final_capital):
    total = len(results)
    if total == 0:
        return _empty_result()

    wins = [r for r in results if r['pnl'] > 0]
    losses = [r for r in results if r['pnl'] <= 0]
    longs = [r for r in results if r['direction'] == 1]
    shorts = [r for r in results if r['direction'] == -1]

    total_pnl = sum(r['pnl'] for r in results)
    gross_profit = sum(r['pnl'] for r in wins) if wins else 0
    gross_loss = abs(sum(r['pnl'] for r in losses)) if losses else 1

    # Exit reason counts
    exit_counts = {}
    for r in results:
        exit_counts[r['exit_type']] = exit_counts.get(r['exit_type'], 0) + 1

    # CAGR from equity curve
    if equity_by_date:
        dates = sorted(equity_by_date.keys())
        first_d = pd.Timestamp(dates[0])
        last_d = pd.Timestamp(dates[-1])
        years = max((last_d - first_d).days / 365.25, 0.1)
        if final_capital > 0:
            cagr = ((final_capital / initial_capital) ** (1 / years) - 1) * 100
        else:
            cagr = -100
    else:
        cagr = 0
        years = 1

    # Sharpe from daily returns
    sharpe = 0
    sortino = 0
    if len(equity_by_date) >= 20:
        eq_series = pd.Series(equity_by_date).sort_index()
        rets = eq_series.pct_change().dropna()
        if len(rets) > 0 and rets.std() > 0:
            sharpe = (rets.mean() / rets.std()) * np.sqrt(252)
            downside = rets[rets < 0]
            if len(downside) > 0 and downside.std() > 0:
                sortino = (rets.mean() / downside.std()) * np.sqrt(252)

    calmar = cagr / (max_dd * 100) if max_dd > 0 else 0

    # Year-by-year
    yearly = {}
    for r in results:
        y = r['year']
        if y not in yearly:
            yearly[y] = {'trades': 0, 'wins': 0, 'pnl': 0}
        yearly[y]['trades'] += 1
        if r['pnl'] > 0:
            yearly[y]['wins'] += 1
        yearly[y]['pnl'] += r['pnl']

    avg_win_pct = np.mean([r['pnl_pct'] for r in wins]) if wins else 0
    avg_loss_pct = np.mean([r['pnl_pct'] for r in losses]) if losses else 0

    stats = {
        'total_trades': total,
        'win_rate': round(len(wins) / total * 100, 2),
        'profit_factor': round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0,
        'total_pnl': round(total_pnl, 0),
        'total_pnl_pct': round(total_pnl / initial_capital * 100, 2),
        'cagr': round(cagr, 2),
        'max_drawdown': round(max_dd * 100, 2),
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'calmar': round(calmar, 2),
        'avg_win': round(avg_win_pct, 2),
        'avg_loss': round(avg_loss_pct, 2),
        'avg_rr': round(abs(avg_win_pct / avg_loss_pct), 2) if avg_loss_pct != 0 else 0,
        'avg_bars_held': 0,  # Not applicable for vectorized
        'long_trades': len(longs),
        'short_trades': len(shorts),
        'long_wr': round(len([r for r in longs if r['pnl'] > 0]) / max(len(longs), 1) * 100, 2),
        'short_wr': round(len([r for r in shorts if r['pnl'] > 0]) / max(len(shorts), 1) * 100, 2),
        'exit_sl': exit_counts.get('fixed_sl', 0),
        'exit_tp': exit_counts.get('fixed_tp', 0),
        'exit_eod': exit_counts.get('eod', 0),
        'exit_other': sum(v for k, v in exit_counts.items() if k not in ('fixed_sl', 'fixed_tp', 'eod')),
    }

    for y in range(2018, 2026):
        ys = yearly.get(y, {'trades': 0, 'wins': 0, 'pnl': 0})
        stats[f'y{y}_trades'] = ys['trades']
        stats[f'y{y}_wr'] = round(ys['wins'] / ys['trades'] * 100, 1) if ys['trades'] > 0 else 0
        stats[f'y{y}_pnl_pct'] = round(ys['pnl'] / initial_capital * 100, 2)

    return stats


# ============================================================================
# Strategy signal generators (vectorized per stock)
# ============================================================================

def gen_orb_signals(stock_data: Dict[str, pd.DataFrame], tp_mult: float = 1.5) -> pd.DataFrame:
    """
    ORB: First 30 min (6 bars) = opening range.
    Buy stop at range high, sell stop at low.
    Returns DataFrame of trade signals.
    """
    all_trades = []

    for sym, df in stock_data.items():
        df = df.copy()
        df['trading_date'] = df.index.date
        df['bar_in_day'] = df.groupby('trading_date').cumcount()
        df['time'] = df.index.time

        for date, day_df in df.groupby('trading_date'):
            if len(day_df) < 8:
                continue

            # Opening range: first 6 bars (0-5)
            or_bars = day_df.iloc[:6]
            or_high = or_bars['high'].max()
            or_low = or_bars['low'].min()
            or_range = or_high - or_low

            if or_range <= 0 or or_range / or_low < 0.001:
                continue

            # Trading bars: after opening range, before EOD
            trade_bars = day_df.iloc[6:]
            eod_mask = trade_bars.index.map(lambda dt: dt.hour > 15 or (dt.hour == 15 and dt.minute >= 15))
            trade_bars_valid = trade_bars[~eod_mask]

            if len(trade_bars_valid) == 0:
                continue

            # Find first bar that breaks OR high or low
            traded = False
            for idx in range(len(trade_bars_valid)):
                bar = trade_bars_valid.iloc[idx]

                long_trigger = bar['high'] > or_high and bar['open'] <= or_high
                short_trigger = bar['low'] < or_low and bar['open'] >= or_low

                if long_trigger and not short_trigger and not traded:
                    risk = or_range
                    entry_price = or_high
                    sl = or_low
                    tp = entry_price + tp_mult * risk

                    # Remaining bars high/low for exit simulation
                    remaining = trade_bars.iloc[trade_bars.index.get_indexer([trade_bars_valid.index[idx]])[0]:]
                    eod_bar = day_df.iloc[-1]
                    # Check if any remaining bar (including entry bar) hits SL or TP
                    # Use remaining day range
                    day_high_after = remaining['high'].max()
                    day_low_after = remaining['low'].min()
                    eod_close = eod_bar['close']

                    all_trades.append({
                        'symbol': sym,
                        'entry_price': entry_price,
                        'sl': sl,
                        'tp': tp,
                        'direction': 1,
                        'entry_date': trade_bars_valid.index[idx],
                        'day_high_after': day_high_after,
                        'day_low_after': day_low_after,
                        'eod_close': eod_close,
                        'year': date.year,
                    })
                    traded = True
                    break

                elif short_trigger and not long_trigger and not traded:
                    risk = or_range
                    entry_price = or_low
                    sl = or_high
                    tp = entry_price - tp_mult * risk

                    remaining = trade_bars.iloc[trade_bars.index.get_indexer([trade_bars_valid.index[idx]])[0]:]
                    eod_bar = day_df.iloc[-1]
                    day_high_after = remaining['high'].max()
                    day_low_after = remaining['low'].min()
                    eod_close = eod_bar['close']

                    all_trades.append({
                        'symbol': sym,
                        'entry_price': entry_price,
                        'sl': sl,
                        'tp': tp,
                        'direction': -1,
                        'entry_date': trade_bars_valid.index[idx],
                        'day_high_after': day_high_after,
                        'day_low_after': day_low_after,
                        'eod_close': eod_close,
                        'year': date.year,
                    })
                    traded = True
                    break

    return pd.DataFrame(all_trades) if all_trades else pd.DataFrame()


def gen_first_candle_signals(stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """First 5-min candle breakout. Similar to ORB but first candle only."""
    all_trades = []

    for sym, df in stock_data.items():
        df = df.copy()
        df['trading_date'] = df.index.date

        for date, day_df in df.groupby('trading_date'):
            if len(day_df) < 3:
                continue

            first = day_df.iloc[0]
            fc_high = first['high'] * 1.001
            fc_low = first['low'] * 0.999
            fc_range = fc_high - fc_low
            if fc_range <= 0:
                continue

            trade_bars = day_df.iloc[1:]
            traded = False

            for idx in range(len(trade_bars)):
                bar = trade_bars.iloc[idx]
                bar_dt = trade_bars.index[idx]
                if bar_dt.hour > 15 or (bar_dt.hour == 15 and bar_dt.minute >= 15):
                    break

                long_trigger = bar['high'] > fc_high and bar['open'] <= fc_high
                short_trigger = bar['low'] < fc_low and bar['open'] >= fc_low

                if long_trigger and not short_trigger and not traded:
                    remaining = day_df.iloc[day_df.index.get_indexer([bar_dt])[0]:]
                    all_trades.append({
                        'symbol': sym,
                        'entry_price': fc_high,
                        'sl': fc_low,
                        'tp': fc_high + 2 * fc_range,
                        'direction': 1,
                        'entry_date': bar_dt,
                        'day_high_after': remaining['high'].max(),
                        'day_low_after': remaining['low'].min(),
                        'eod_close': day_df.iloc[-1]['close'],
                        'year': date.year,
                    })
                    traded = True
                    break
                elif short_trigger and not long_trigger and not traded:
                    remaining = day_df.iloc[day_df.index.get_indexer([bar_dt])[0]:]
                    all_trades.append({
                        'symbol': sym,
                        'entry_price': fc_low,
                        'sl': fc_high,
                        'tp': fc_low - 2 * fc_range,
                        'direction': -1,
                        'entry_date': bar_dt,
                        'day_high_after': remaining['high'].max(),
                        'day_low_after': remaining['low'].min(),
                        'eod_close': day_df.iloc[-1]['close'],
                        'year': date.year,
                    })
                    traded = True
                    break

    return pd.DataFrame(all_trades) if all_trades else pd.DataFrame()


def gen_vwap_ema_signals(stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """VWAP + EMA(20) trend following. Vectorized signal detection."""
    all_trades = []

    for sym, df in stock_data.items():
        df = df.copy()
        df['trading_date'] = df.index.date
        df['ema20'] = fast_ema(df['close'], 20)
        df['atr14'] = fast_atr(df, 14)

        for date, day_df in df.groupby('trading_date'):
            if len(day_df) < 10:
                continue

            # Compute intraday VWAP
            tp = (day_df['high'] + day_df['low'] + day_df['close']) / 3
            cum_tp_vol = (tp * day_df['volume']).cumsum()
            cum_vol = day_df['volume'].cumsum().replace(0, np.nan)
            vwap = cum_tp_vol / cum_vol

            trades_today = 0

            for i in range(4, len(day_df)):
                if trades_today >= 2:
                    break
                bar = day_df.iloc[i]
                bar_dt = day_df.index[i]
                if bar_dt.hour > 15 or (bar_dt.hour == 15 and bar_dt.minute >= 15):
                    break
                if i + 1 >= len(day_df):
                    break

                prev = day_df.iloc[i-1]
                atr = bar['atr14']
                ema20 = bar['ema20']

                if pd.isna(atr) or atr <= 0 or pd.isna(ema20):
                    continue

                cur_vwap = vwap.iloc[i]
                prev_vwap = vwap.iloc[i-1]
                if pd.isna(cur_vwap) or pd.isna(prev_vwap):
                    continue

                # EMA rising/falling
                if i >= 3:
                    ema20_3ago = day_df.iloc[i-3]['ema20']
                    if pd.isna(ema20_3ago):
                        continue
                else:
                    continue

                cross_above = prev['close'] <= prev_vwap and bar['close'] > cur_vwap
                cross_below = prev['close'] >= prev_vwap and bar['close'] < cur_vwap
                ema_rising = ema20 > ema20_3ago
                ema_falling = ema20 < ema20_3ago

                next_bar = day_df.iloc[i+1]
                remaining = day_df.iloc[i+1:]
                eod_close = day_df.iloc[-1]['close']

                if cross_above and ema_rising and bar['close'] > ema20:
                    entry = next_bar['open']
                    risk = 1.5 * atr
                    all_trades.append({
                        'symbol': sym,
                        'entry_price': entry,
                        'sl': entry - risk,
                        'tp': entry + 2 * risk,
                        'direction': 1,
                        'entry_date': day_df.index[i+1],
                        'day_high_after': remaining['high'].max(),
                        'day_low_after': remaining['low'].min(),
                        'eod_close': eod_close,
                        'year': date.year,
                    })
                    trades_today += 1

                elif cross_below and ema_falling and bar['close'] < ema20:
                    entry = next_bar['open']
                    risk = 1.5 * atr
                    all_trades.append({
                        'symbol': sym,
                        'entry_price': entry,
                        'sl': entry + risk,
                        'tp': entry - 2 * risk,
                        'direction': -1,
                        'entry_date': day_df.index[i+1],
                        'day_high_after': remaining['high'].max(),
                        'day_low_after': remaining['low'].min(),
                        'eod_close': eod_close,
                        'year': date.year,
                    })
                    trades_today += 1

    return pd.DataFrame(all_trades) if all_trades else pd.DataFrame()


def gen_ema_cross_signals(stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """EMA(9)/EMA(21) crossover + RSI(14) > 50 filter."""
    all_trades = []

    for sym, df in stock_data.items():
        df = df.copy()
        df['trading_date'] = df.index.date
        df['ema9'] = fast_ema(df['close'], 9)
        df['ema21'] = fast_ema(df['close'], 21)
        df['rsi14'] = fast_rsi(df['close'], 14)
        df['atr14'] = fast_atr(df, 14)

        for date, day_df in df.groupby('trading_date'):
            if len(day_df) < 10:
                continue

            trades_today = 0
            for i in range(1, len(day_df)):
                if trades_today >= 3:
                    break
                bar = day_df.iloc[i]
                bar_dt = day_df.index[i]
                if bar_dt.hour > 15 or (bar_dt.hour == 15 and bar_dt.minute >= 15):
                    break
                if i + 1 >= len(day_df):
                    break

                prev = day_df.iloc[i-1]
                atr = bar['atr14']
                if pd.isna(atr) or atr <= 0:
                    continue

                ema9 = bar['ema9']
                ema21 = bar['ema21']
                rsi = bar['rsi14']
                prev_ema9 = prev['ema9']
                prev_ema21 = prev['ema21']

                if any(pd.isna(x) for x in [ema9, ema21, rsi, prev_ema9, prev_ema21]):
                    continue

                cross_up = prev_ema9 <= prev_ema21 and ema9 > ema21
                cross_down = prev_ema9 >= prev_ema21 and ema9 < ema21

                next_bar = day_df.iloc[i+1]
                remaining = day_df.iloc[i+1:]
                eod_close = day_df.iloc[-1]['close']

                if cross_up and bar['close'] > ema9 and bar['close'] > ema21 and rsi > 50:
                    entry = next_bar['open']
                    risk = 1.0 * atr
                    all_trades.append({
                        'symbol': sym, 'entry_price': entry,
                        'sl': entry - risk, 'tp': entry + 1.5 * risk,
                        'direction': 1, 'entry_date': day_df.index[i+1],
                        'day_high_after': remaining['high'].max(),
                        'day_low_after': remaining['low'].min(),
                        'eod_close': eod_close, 'year': date.year,
                    })
                    trades_today += 1

                elif cross_down and bar['close'] < ema9 and bar['close'] < ema21 and rsi < 50:
                    entry = next_bar['open']
                    risk = 1.0 * atr
                    all_trades.append({
                        'symbol': sym, 'entry_price': entry,
                        'sl': entry + risk, 'tp': entry - 1.5 * risk,
                        'direction': -1, 'entry_date': day_df.index[i+1],
                        'day_high_after': remaining['high'].max(),
                        'day_low_after': remaining['low'].min(),
                        'eod_close': eod_close, 'year': date.year,
                    })
                    trades_today += 1

    return pd.DataFrame(all_trades) if all_trades else pd.DataFrame()


def gen_supertrend_signals(stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """SuperTrend(10,3) flip signals. Exit on next flip or EOD."""
    all_trades = []

    for sym, df in stock_data.items():
        df = df.copy()
        df['trading_date'] = df.index.date
        df['st'], df['st_dir'] = fast_supertrend(df, 10, 3.0)

        for date, day_df in df.groupby('trading_date'):
            if len(day_df) < 5:
                continue

            trades_today = 0
            in_trade = False

            for i in range(1, len(day_df)):
                if trades_today >= 4:
                    break
                bar = day_df.iloc[i]
                bar_dt = day_df.index[i]
                prev = day_df.iloc[i-1]

                st_dir = bar['st_dir']
                prev_st_dir = prev['st_dir']
                if pd.isna(st_dir) or pd.isna(prev_st_dir):
                    continue

                # On flip, close existing conceptual trade and open new one
                flip_up = prev_st_dir == -1 and st_dir == 1
                flip_down = prev_st_dir == 1 and st_dir == -1

                if (flip_up or flip_down) and not (bar_dt.hour > 15 or (bar_dt.hour == 15 and bar_dt.minute >= 15)):
                    if i + 1 >= len(day_df):
                        continue

                    next_bar = day_df.iloc[i+1]
                    entry = next_bar['open']

                    # Find the next flip or EOD
                    exit_price = day_df.iloc[-1]['close']  # Default EOD
                    exit_found = False

                    # Search remaining bars for next flip
                    remaining_after = day_df.iloc[i+1:]
                    day_high_after = remaining_after['high'].max()
                    day_low_after = remaining_after['low'].min()

                    for j in range(i+2, len(day_df)):
                        future_bar = day_df.iloc[j]
                        future_dir = future_bar['st_dir']
                        future_prev_dir = day_df.iloc[j-1]['st_dir']
                        if pd.isna(future_dir) or pd.isna(future_prev_dir):
                            continue
                        next_flip = (future_prev_dir != future_dir)
                        if next_flip:
                            exit_price = future_bar['close']
                            # Adjust high/low to only include bars up to exit
                            subset = day_df.iloc[i+1:j+1]
                            day_high_after = subset['high'].max()
                            day_low_after = subset['low'].min()
                            exit_found = True
                            break

                    direction = 1 if flip_up else -1
                    # For SuperTrend, no fixed SL/TP — we use the exit_price directly
                    # Encode: set SL very far, TP = None, eod_close = exit_price
                    if direction == 1:
                        sl = entry * 0.90  # Won't trigger
                        tp_val = entry * 1.50  # Won't trigger
                    else:
                        sl = entry * 1.10
                        tp_val = entry * 0.50

                    all_trades.append({
                        'symbol': sym, 'entry_price': entry,
                        'sl': sl, 'tp': None,
                        'direction': direction,
                        'entry_date': day_df.index[i+1],
                        'day_high_after': day_high_after,
                        'day_low_after': day_low_after,
                        'eod_close': exit_price,  # This IS the exit price
                        'year': date.year,
                    })
                    trades_today += 1

    return pd.DataFrame(all_trades) if all_trades else pd.DataFrame()


def gen_macd_rev_signals(stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """MACD histogram zero-cross + EMA(50) filter."""
    all_trades = []

    for sym, df in stock_data.items():
        df = df.copy()
        df['trading_date'] = df.index.date
        df['ema50'] = fast_ema(df['close'], 50)
        df['atr14'] = fast_atr(df, 14)
        df['macd_line'], df['macd_sig'], df['macd_hist'] = fast_macd(df['close'])

        for date, day_df in df.groupby('trading_date'):
            if len(day_df) < 5:
                continue

            trades_today = 0
            for i in range(1, len(day_df)):
                if trades_today >= 2:
                    break
                bar = day_df.iloc[i]
                bar_dt = day_df.index[i]
                if bar_dt.hour > 15 or (bar_dt.hour == 15 and bar_dt.minute >= 15):
                    break
                if i + 1 >= len(day_df):
                    break

                prev = day_df.iloc[i-1]
                atr = bar['atr14']
                ema50 = bar['ema50']
                mh = bar['macd_hist']
                pmh = prev['macd_hist']

                if any(pd.isna(x) for x in [atr, ema50, mh, pmh]) or atr <= 0:
                    continue

                hist_cross_up = pmh < 0 and mh >= 0
                hist_cross_down = pmh > 0 and mh <= 0

                next_bar = day_df.iloc[i+1]
                remaining = day_df.iloc[i+1:]
                eod_close = day_df.iloc[-1]['close']

                if hist_cross_up and bar['close'] > ema50:
                    entry = next_bar['open']
                    risk = 1.5 * atr
                    all_trades.append({
                        'symbol': sym, 'entry_price': entry,
                        'sl': entry - risk, 'tp': entry + 2 * risk,
                        'direction': 1, 'entry_date': day_df.index[i+1],
                        'day_high_after': remaining['high'].max(),
                        'day_low_after': remaining['low'].min(),
                        'eod_close': eod_close, 'year': date.year,
                    })
                    trades_today += 1

                elif hist_cross_down and bar['close'] < ema50:
                    entry = next_bar['open']
                    risk = 1.5 * atr
                    all_trades.append({
                        'symbol': sym, 'entry_price': entry,
                        'sl': entry + risk, 'tp': entry - 2 * risk,
                        'direction': -1, 'entry_date': day_df.index[i+1],
                        'day_high_after': remaining['high'].max(),
                        'day_low_after': remaining['low'].min(),
                        'eod_close': eod_close, 'year': date.year,
                    })
                    trades_today += 1

    return pd.DataFrame(all_trades) if all_trades else pd.DataFrame()


# ============================================================================
# Main
# ============================================================================

def write_row(row):
    with open(OUTPUT_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(row)


def main():
    start_time = time.time()

    done = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV) as f:
            done = {row['label'] for row in csv.DictReader(f)}
        if done:
            print(f'Found {len(done)} already-completed configs, skipping those.')
        else:
            pass  # Header only, no need to rewrite
    else:
        with open(OUTPUT_CSV, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    # Load data
    print(f'Loading 5-min data for {len(SYMBOLS)} symbols...', flush=True)
    t0 = time.time()
    raw_data = load_data_from_db(SYMBOLS, '5minute', '2018-01-01', '2025-12-31')
    print(f'  Loaded {len(raw_data)} symbols in {time.time()-t0:.1f}s', flush=True)
    for sym, df in raw_data.items():
        print(f'  {sym}: {len(df)} bars, {df.index.min()} to {df.index.max()}')

    cost_configs = {
        'cash': {'commission_pct': 0.0005, 'slippage_pct': 0.0005},
        'futures': {'commission_pct': 0.0001, 'slippage_pct': 0.0003},
    }

    # Strategy generators (name, function, extra_params)
    strategies = [
        ('ORB_TP1.0x', 'ORB', lambda: gen_orb_signals(raw_data, tp_mult=1.0)),
        ('ORB_TP1.5x', 'ORB', lambda: gen_orb_signals(raw_data, tp_mult=1.5)),
        ('ORB_TP2.0x', 'ORB', lambda: gen_orb_signals(raw_data, tp_mult=2.0)),
        ('VWAP_EMA', 'VWAP_EMA', lambda: gen_vwap_ema_signals(raw_data)),
        ('EMA_CROSS', 'EMA_CROSS', lambda: gen_ema_cross_signals(raw_data)),
        ('SUPERTREND', 'SUPERTREND', lambda: gen_supertrend_signals(raw_data)),
        ('FIRST_CANDLE', 'FIRST_CANDLE', lambda: gen_first_candle_signals(raw_data)),
        ('MACD_REV', 'MACD_REV', lambda: gen_macd_rev_signals(raw_data)),
    ]

    total_configs = len(strategies) * len(cost_configs)
    config_idx = 0

    for strat_name, strat_type, gen_func in strategies:
        # Generate signals once per strategy (reuse for cash/futures)
        needs_generation = any(
            f'{strat_name}_{mode}' not in done
            for mode in cost_configs
        )

        signals_df = None
        if needs_generation:
            print(f'\nGenerating signals for {strat_name}...', end='', flush=True)
            t0 = time.time()
            signals_df = gen_func()
            n_signals = len(signals_df) if signals_df is not None and len(signals_df) > 0 else 0
            print(f' {time.time()-t0:.0f}s, {n_signals} raw signals', flush=True)

        for mode, costs in cost_configs.items():
            config_idx += 1
            label = f'{strat_name}_{mode}'

            if label in done:
                print(f'[{config_idx}/{total_configs}] {label} — SKIPPED', flush=True)
                continue

            print(f'[{config_idx}/{total_configs}] {label} ...', end='', flush=True)
            t0 = time.time()

            try:
                if signals_df is None or len(signals_df) == 0:
                    stats = _empty_result()
                else:
                    stats = simulate_trades_vectorized(
                        signals_df,
                        INITIAL_CAPITAL,
                        POSITION_SIZE_PCT,
                        costs['commission_pct'],
                        costs['slippage_pct'],
                    )

                row = {'label': label, 'strategy': strat_type, 'cost_mode': mode}
                row.update(stats)
                write_row(row)
                elapsed = time.time() - t0

                print(f' {elapsed:.0f}s | Trades={stats["total_trades"]} WR={stats["win_rate"]}% '
                      f'CAGR={stats["cagr"]}% MaxDD={stats["max_drawdown"]}% '
                      f'PF={stats["profit_factor"]} Sharpe={stats["sharpe"]}', flush=True)

            except Exception as e:
                elapsed = time.time() - t0
                print(f' {elapsed:.0f}s | ERROR: {e}', flush=True)
                import traceback
                traceback.print_exc()

    total_time = time.time() - start_time
    print(f'\n{"=" * 80}')
    print(f'SWEEP COMPLETE in {total_time:.0f}s ({total_time/60:.1f} min)')
    print(f'Results: {OUTPUT_CSV}')
    print(f'{"=" * 80}')

    # Print summary
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV) as f:
            rows = list(csv.DictReader(f))
        if rows:
            print(f'\n{"SUMMARY - 5-Min Scalping Strategies":^80}')
            print(f'{"Label":<25} {"Trades":>7} {"WR%":>6} {"CAGR%":>8} {"MaxDD%":>8} {"PF":>6} {"Sharpe":>7}')
            print('-' * 80)
            for r in sorted(rows, key=lambda x: float(x.get('cagr', 0) or 0), reverse=True):
                print(f'{r["label"]:<25} {r["total_trades"]:>7} {r["win_rate"]:>6} '
                      f'{r["cagr"]:>8} {r["max_drawdown"]:>8} {r["profit_factor"]:>6} {r["sharpe"]:>7}')

            print(f'\n{"YEAR-BY-YEAR BREAKDOWN":^80}')
            for r in sorted(rows, key=lambda x: float(x.get('cagr', 0) or 0), reverse=True):
                print(f'\n  {r["label"]}:')
                for y in range(2018, 2026):
                    trades = r.get(f'y{y}_trades', 0)
                    wr = r.get(f'y{y}_wr', 0)
                    pnl = r.get(f'y{y}_pnl_pct', 0)
                    if int(trades) > 0:
                        print(f'    {y}: {trades:>5} trades, WR={wr}%, PnL={pnl}%')


if __name__ == '__main__':
    main()
