"""
Part 2: Continue trend following research — remaining SuperTrend, MACD, Donchian, Combo variants.
Optimized SuperTrend using numpy arrays for speed.
"""

import sqlite3
import pandas as pd
import numpy as np
import csv
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backtest_data', 'market_data.db')
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'research_results_trend.csv')

SLIPPAGE_PCT = 0.0005
BROKERAGE_PER_RT = 40
INITIAL_CAPITAL = 100_000

FIELDNAMES = [
    'strategy', 'variant', 'total_trades', 'win_rate', 'profit_factor',
    'avg_win_pct', 'avg_loss_pct', 'win_loss_ratio', 'avg_trade_pct',
    'total_return_pct', 'max_drawdown_pct', 'avg_holding_bars',
    'best_stock', 'worst_stock', 'median_trade_pct',
    'trades_per_stock', 'sharpe_approx'
]


def load_all_60min_data():
    print("Loading 60-min data from DB...", flush=True)
    t0 = time.time()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT symbol, date, open, high, low, close, volume "
        "FROM market_data_unified WHERE timeframe='60minute' ORDER BY symbol, date",
        conn
    )
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    data = {}
    for sym, gdf in df.groupby('symbol'):
        sdf = gdf.set_index('date').sort_index()
        sdf = sdf[['open', 'high', 'low', 'close', 'volume']].astype(float)
        if len(sdf) >= 500:
            data[sym] = sdf
    print(f"Loaded {len(data)} stocks in {time.time()-t0:.1f}s", flush=True)
    return data


def compute_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def compute_sma(series, period):
    return series.rolling(period).mean()

def compute_adx(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=close.index).ewm(alpha=1/period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=close.index).ewm(alpha=1/period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx

def compute_supertrend_fast(high_arr, low_arr, close_arr, period=10, multiplier=3.0):
    """Optimized SuperTrend using pure numpy arrays."""
    n = len(close_arr)
    hl2 = (high_arr + low_arr) / 2.0

    # True Range
    tr = np.empty(n)
    tr[0] = high_arr[0] - low_arr[0]
    for i in range(1, n):
        tr[i] = max(high_arr[i] - low_arr[i],
                     abs(high_arr[i] - close_arr[i-1]),
                     abs(low_arr[i] - close_arr[i-1]))

    # ATR using EMA
    alpha = 1.0 / period
    atr = np.empty(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i-1]

    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    direction = np.ones(n)
    final_upper = upper.copy()
    final_lower = lower.copy()

    for i in range(1, n):
        if not (final_upper[i] < final_upper[i-1] or close_arr[i-1] > final_upper[i-1]):
            final_upper[i] = final_upper[i-1]
        if not (final_lower[i] > final_lower[i-1] or close_arr[i-1] < final_lower[i-1]):
            final_lower[i] = final_lower[i-1]

        if direction[i-1] == 1:
            direction[i] = -1 if close_arr[i] < final_lower[i] else 1
        else:
            direction[i] = 1 if close_arr[i] > final_upper[i] else -1

    return direction


def signals_supertrend(df, period, multiplier):
    direction = compute_supertrend_fast(
        df['high'].values, df['low'].values, df['close'].values, period, multiplier
    )
    dir_s = pd.Series(direction, index=df.index)
    flip_up = (dir_s == 1) & (dir_s.shift(1) == -1)
    flip_dn = (dir_s == -1) & (dir_s.shift(1) == 1)
    sig = pd.Series(0, index=df.index)
    sig[flip_up] = 1
    sig[flip_dn] = -1
    return sig

def signals_supertrend_sma(df, st_period, st_mult, sma_period=200):
    direction = compute_supertrend_fast(
        df['high'].values, df['low'].values, df['close'].values, st_period, st_mult
    )
    dir_s = pd.Series(direction, index=df.index)
    sma = compute_sma(df['close'], sma_period)
    flip_up = (dir_s == 1) & (dir_s.shift(1) == -1)
    flip_dn = (dir_s == -1) & (dir_s.shift(1) == 1)
    sig = pd.Series(0, index=df.index)
    sig[flip_up & (df['close'] > sma)] = 1
    sig[flip_dn] = -1
    return sig

def signals_macd(df, fast, slow, signal_period):
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    cross_up = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
    cross_dn = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
    sig = pd.Series(0, index=df.index)
    sig[cross_up] = 1
    sig[cross_dn] = -1
    return sig

def signals_donchian(df, entry_period, exit_period):
    upper = df['high'].rolling(entry_period).max()
    lower = df['low'].rolling(exit_period).min()
    buy = df['close'] > upper.shift(1)
    sell = df['close'] < lower.shift(1)
    sig = pd.Series(0, index=df.index)
    sig[buy] = 1
    sig[sell] = -1
    return sig


def backtest_signals(df, signals):
    trades = []
    in_trade = False
    entry_price = 0
    entry_idx = 0
    close = df['close'].values
    dates = df.index
    sig = signals.values
    for i in range(len(sig)):
        if not in_trade:
            if sig[i] == 1:
                entry_price = close[i] * (1 + SLIPPAGE_PCT)
                entry_idx = i
                in_trade = True
        else:
            if sig[i] == -1:
                exit_price = close[i] * (1 - SLIPPAGE_PCT)
                pnl_pct = (exit_price / entry_price - 1) * 100
                brokerage_pct = BROKERAGE_PER_RT / INITIAL_CAPITAL * 100
                pnl_pct -= brokerage_pct
                trades.append({
                    'entry_date': dates[entry_idx],
                    'exit_date': dates[i],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': pnl_pct,
                    'holding_bars': i - entry_idx,
                })
                in_trade = False
    return trades


def compute_metrics(all_trades, per_stock_trades):
    if not all_trades:
        return None
    pnls = [t['pnl_pct'] for t in all_trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    holdings = [t['holding_bars'] for t in all_trades]
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001
    cum_pnl = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdowns = running_max - cum_pnl
    max_dd = drawdowns.max() if len(drawdowns) > 0 else 0
    stock_returns = {}
    for sym, tlist in per_stock_trades.items():
        stock_returns[sym] = sum(t['pnl_pct'] for t in tlist)
    best_stock = max(stock_returns, key=stock_returns.get) if stock_returns else 'N/A'
    worst_stock = min(stock_returns, key=stock_returns.get) if stock_returns else 'N/A'
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0.001
    sharpe = np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(252) if len(pnls) > 1 else 0
    return {
        'total_trades': len(all_trades),
        'win_rate': round(len(wins) / len(all_trades) * 100, 2),
        'profit_factor': round(gross_profit / gross_loss, 3),
        'avg_win_pct': round(avg_win, 3),
        'avg_loss_pct': round(avg_loss, 3),
        'win_loss_ratio': round(avg_win / avg_loss, 3) if avg_loss > 0 else 0,
        'avg_trade_pct': round(np.mean(pnls), 3),
        'total_return_pct': round(sum(pnls), 2),
        'max_drawdown_pct': round(max_dd, 2),
        'avg_holding_bars': round(np.mean(holdings), 1),
        'best_stock': f"{best_stock}({stock_returns.get(best_stock,0):.1f}%)",
        'worst_stock': f"{worst_stock}({stock_returns.get(worst_stock,0):.1f}%)",
        'median_trade_pct': round(np.median(pnls), 3),
        'trades_per_stock': round(len(all_trades) / max(len(per_stock_trades), 1), 1),
        'sharpe_approx': round(sharpe, 3),
    }


def run_strategy(data, strategy_name, variant_name, signal_func, **kwargs):
    all_trades = []
    per_stock = {}
    for sym, df in data.items():
        try:
            signals = signal_func(df, **kwargs)
            trades = backtest_signals(df, signals)
            if trades:
                all_trades.extend(trades)
                per_stock[sym] = trades
        except:
            pass
    metrics = compute_metrics(all_trades, per_stock)
    if metrics is None:
        return None
    metrics['strategy'] = strategy_name
    metrics['variant'] = variant_name
    return metrics


def write_result(row):
    with open(OUTPUT_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(row)


def main():
    t_start = time.time()
    data = load_all_60min_data()

    # Check already done
    done = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV) as f:
            for row in csv.DictReader(f):
                done.add(row.get('variant', ''))
    print(f"Already completed: {len(done)} variants", flush=True)

    strategies = []

    # Remaining SuperTrend variants (ST10 and ST14)
    for period in [10, 14]:
        for mult in [2.0, 2.5, 3.0, 3.5]:
            strategies.append((
                'SuperTrend', f'ST{period}_m{mult}',
                signals_supertrend, {'period': period, 'multiplier': mult}
            ))

    # MACD
    strategies.append(('MACD', 'MACD_12_26_9', signals_macd, {'fast': 12, 'slow': 26, 'signal_period': 9}))
    strategies.append(('MACD', 'MACD_8_21_5', signals_macd, {'fast': 8, 'slow': 21, 'signal_period': 5}))

    # Donchian
    strategies.append(('Donchian', 'DCH20_exit10', signals_donchian, {'entry_period': 20, 'exit_period': 10}))
    strategies.append(('Donchian', 'DCH40_exit10', signals_donchian, {'entry_period': 40, 'exit_period': 10}))
    strategies.append(('Donchian', 'DCH40_exit20', signals_donchian, {'entry_period': 40, 'exit_period': 20}))

    # SuperTrend + SMA200
    for period in [7, 10, 14]:
        for mult in [2.0, 3.0]:
            strategies.append((
                'ST_SMA200', f'ST{period}_m{mult}_SMA200',
                signals_supertrend_sma, {'st_period': period, 'st_mult': mult}
            ))

    # Filter out done
    remaining = [(s, v, f, k) for s, v, f, k in strategies if v not in done]
    total = len(remaining)
    print(f"Remaining to run: {total}\n", flush=True)

    for idx, (strat_name, variant, sig_func, kwargs) in enumerate(remaining, 1):
        t0 = time.time()
        result = run_strategy(data, strat_name, variant, sig_func, **kwargs)
        elapsed = time.time() - t0

        if result:
            write_result(result)
            print(f"[{idx}/{total}] {variant:25s} | {elapsed:5.1f}s | "
                  f"Trades={result['total_trades']:5d} WR={result['win_rate']:5.1f}% "
                  f"PF={result['profit_factor']:5.3f} AvgTrade={result['avg_trade_pct']:+.3f}%",
                  flush=True)
        else:
            print(f"[{idx}/{total}] {variant:25s} | {elapsed:5.1f}s | NO TRADES", flush=True)

    # ── Final Summary ──
    print(f"\n{'='*90}")
    print(f"ALL DONE — {time.time()-t_start:.0f}s total")
    print(f"{'='*90}\n")

    results_df = pd.read_csv(OUTPUT_CSV)
    viable = results_df[results_df['total_trades'] >= 100].sort_values('profit_factor', ascending=False)

    print(f"All variants: {len(results_df)}")
    print(f"Viable (100+ trades): {len(viable)}")
    print()

    print("TOP STRATEGIES BY PROFIT FACTOR (100+ trades):")
    print("-" * 120)
    print(f"{'Variant':30s} {'Trades':>7s} {'WR%':>6s} {'PF':>7s} {'AvgTrade':>9s} {'W/L':>6s} "
          f"{'MaxDD':>7s} {'HoldBars':>9s} {'Sharpe':>7s}")
    print("-" * 120)
    for _, row in viable.head(25).iterrows():
        print(f"{row['variant']:30s} {row['total_trades']:7.0f} {row['win_rate']:6.1f} "
              f"{row['profit_factor']:7.3f} {row['avg_trade_pct']:+8.3f}% {row['win_loss_ratio']:6.3f} "
              f"{row['max_drawdown_pct']:6.1f}% {row['avg_holding_bars']:9.1f} {row['sharpe_approx']:7.3f}")

    profitable = viable[viable['profit_factor'] > 1.0]
    strong = viable[viable['profit_factor'] > 1.2]
    print(f"\nProfitable (PF > 1.0): {len(profitable)} / {len(viable)}")
    print(f"Strong (PF > 1.2): {len(strong)} / {len(viable)}")

    print(f"\n{'='*90}")
    print("BEST VARIANT PER STRATEGY TYPE:")
    print(f"{'='*90}")
    for stype in results_df['strategy'].unique():
        subset = viable[viable['strategy'] == stype]
        if len(subset) == 0:
            print(f"\n{stype}: No viable variants (< 100 trades)")
            continue
        best = subset.sort_values('profit_factor', ascending=False).iloc[0]
        print(f"\n{stype}: {best['variant']}")
        print(f"  Trades={best['total_trades']:.0f} WR={best['win_rate']:.1f}% "
              f"PF={best['profit_factor']:.3f} AvgTrade={best['avg_trade_pct']:+.3f}% "
              f"Sharpe={best['sharpe_approx']:.3f} MaxDD={best['max_drawdown_pct']:.1f}%")


if __name__ == '__main__':
    main()
