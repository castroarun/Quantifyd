"""
Trend Following Strategy Research on 60-minute F&O Data
=======================================================
Tests EMA crossovers, SuperTrend, MACD, Donchian, and combo strategies
on 93 F&O stocks from 2018-2025 using 60-minute candles.

Costs: 0.05% slippage per side + Rs 40 brokerage per round trip
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

SLIPPAGE_PCT = 0.0005  # 0.05% per side
BROKERAGE_PER_RT = 40  # Rs 40 per round trip
INITIAL_CAPITAL = 100_000  # Rs 1 lakh per stock for PnL calc

FIELDNAMES = [
    'strategy', 'variant', 'total_trades', 'win_rate', 'profit_factor',
    'avg_win_pct', 'avg_loss_pct', 'win_loss_ratio', 'avg_trade_pct',
    'total_return_pct', 'max_drawdown_pct', 'avg_holding_bars',
    'best_stock', 'worst_stock', 'median_trade_pct',
    'trades_per_stock', 'sharpe_approx'
]


def load_all_60min_data():
    """Load all 60-min data from DB, return dict of symbol -> DataFrame."""
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
        if len(sdf) >= 500:  # need enough data
            data[sym] = sdf

    print(f"Loaded {len(data)} stocks in {time.time()-t0:.1f}s ({len(df):,} total rows)", flush=True)
    return data


# ── Indicator computations (vectorized) ──────────────────────────────────

def compute_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def compute_sma(series, period):
    return series.rolling(period).mean()

def compute_adx(high, low, close, period=14):
    """Compute ADX using vectorized operations."""
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

def compute_supertrend(high, low, close, period=10, multiplier=3.0):
    """Compute SuperTrend indicator. Returns (supertrend_line, direction: 1=up, -1=down)."""
    hl2 = (high + low) / 2
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()

    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    n = len(close)
    direction = np.ones(n)  # 1 = uptrend
    final_upper = upper.values.copy()
    final_lower = lower.values.copy()

    for i in range(1, n):
        # Final upper band
        if final_upper[i] < final_upper[i-1] or close.iloc[i-1] > final_upper[i-1]:
            pass  # keep current
        else:
            final_upper[i] = final_upper[i-1]

        # Final lower band
        if final_lower[i] > final_lower[i-1] or close.iloc[i-1] < final_lower[i-1]:
            pass
        else:
            final_lower[i] = final_lower[i-1]

        # Direction
        if direction[i-1] == 1:  # was uptrend
            if close.iloc[i] < final_lower[i]:
                direction[i] = -1
            else:
                direction[i] = 1
        else:  # was downtrend
            if close.iloc[i] > final_upper[i]:
                direction[i] = 1
            else:
                direction[i] = -1

    return pd.Series(direction, index=close.index)

def compute_macd(close, fast=12, slow=26, signal=9):
    """Returns macd_line, signal_line."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def compute_donchian(high, low, period):
    """Returns upper channel, lower channel."""
    upper = high.rolling(period).max()
    lower = low.rolling(period).min()
    return upper, lower


# ── Signal generators ──────────────────────────────────────────────────

def signals_ema_cross(df, fast, slow, adx_filter=None):
    """EMA crossover signals. Returns Series of 1 (buy), -1 (sell), 0 (no signal)."""
    ema_f = compute_ema(df['close'], fast)
    ema_s = compute_ema(df['close'], slow)

    cross_up = (ema_f > ema_s) & (ema_f.shift(1) <= ema_s.shift(1))
    cross_dn = (ema_f < ema_s) & (ema_f.shift(1) >= ema_s.shift(1))

    sig = pd.Series(0, index=df.index)
    sig[cross_up] = 1
    sig[cross_dn] = -1

    if adx_filter is not None:
        adx = compute_adx(df['high'], df['low'], df['close'])
        sig[adx < adx_filter] = 0

    return sig

def signals_supertrend(df, period, multiplier):
    """SuperTrend signals. Buy on flip to uptrend, sell on flip to downtrend."""
    direction = compute_supertrend(df['high'], df['low'], df['close'], period, multiplier)

    flip_up = (direction == 1) & (direction.shift(1) == -1)
    flip_dn = (direction == -1) & (direction.shift(1) == 1)

    sig = pd.Series(0, index=df.index)
    sig[flip_up] = 1
    sig[flip_dn] = -1
    return sig

def signals_macd(df, fast, slow, signal_period):
    """MACD crossover signals."""
    macd_line, signal_line = compute_macd(df['close'], fast, slow, signal_period)

    cross_up = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
    cross_dn = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))

    sig = pd.Series(0, index=df.index)
    sig[cross_up] = 1
    sig[cross_dn] = -1
    return sig

def signals_donchian(df, entry_period, exit_period):
    """Donchian breakout: buy when close > upper(entry_period), sell when close < lower(exit_period)."""
    upper, _ = compute_donchian(df['high'], df['low'], entry_period)
    _, lower = compute_donchian(df['high'], df['low'], exit_period)

    # Shift by 1 to avoid lookahead — breakout above PREVIOUS period high
    buy = df['close'] > upper.shift(1)
    sell = df['close'] < lower.shift(1)

    sig = pd.Series(0, index=df.index)
    sig[buy] = 1
    sig[sell] = -1
    return sig

def signals_supertrend_sma(df, st_period, st_mult, sma_period=200):
    """SuperTrend + SMA(200) combo. Longs only above SMA, shorts only below."""
    direction = compute_supertrend(df['high'], df['low'], df['close'], st_period, st_mult)
    sma = compute_sma(df['close'], sma_period)

    flip_up = (direction == 1) & (direction.shift(1) == -1)
    flip_dn = (direction == -1) & (direction.shift(1) == 1)

    sig = pd.Series(0, index=df.index)
    # Only long signals when above SMA
    sig[flip_up & (df['close'] > sma)] = 1
    sig[flip_dn] = -1  # exit on any flip down
    return sig


# ── Backtester (vectorized where possible) ───────────────────────────

def backtest_signals(df, signals, long_only=True):
    """
    Given a DataFrame and signal series (1=buy, -1=sell, 0=hold),
    simulate trades with slippage and brokerage.
    Returns list of trade dicts.
    """
    trades = []
    in_trade = False
    entry_price = 0
    entry_idx = 0
    entry_bar = 0

    close = df['close'].values
    dates = df.index
    sig = signals.values

    bar = 0
    for i in range(len(sig)):
        if not in_trade:
            if sig[i] == 1:  # buy signal
                entry_price = close[i] * (1 + SLIPPAGE_PCT)  # slippage on entry
                entry_idx = i
                entry_bar = bar
                in_trade = True
        else:
            if sig[i] == -1:  # sell signal
                exit_price = close[i] * (1 - SLIPPAGE_PCT)  # slippage on exit
                pnl_pct = (exit_price / entry_price - 1) * 100
                # Subtract brokerage as pct of trade value (approximate)
                brokerage_pct = BROKERAGE_PER_RT / (entry_price * 1) * 100  # 1 share approx
                # For realistic calc, assume position size = INITIAL_CAPITAL
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
        bar += 1

    return trades


def compute_metrics(all_trades, per_stock_trades):
    """Compute strategy-level metrics from trade list."""
    if not all_trades:
        return None

    pnls = [t['pnl_pct'] for t in all_trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    holdings = [t['holding_bars'] for t in all_trades]

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001

    # Max drawdown from cumulative PnL
    cum_pnl = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdowns = running_max - cum_pnl
    max_dd = drawdowns.max() if len(drawdowns) > 0 else 0

    # Per-stock stats
    stock_returns = {}
    for sym, tlist in per_stock_trades.items():
        stock_returns[sym] = sum(t['pnl_pct'] for t in tlist)

    best_stock = max(stock_returns, key=stock_returns.get) if stock_returns else 'N/A'
    worst_stock = min(stock_returns, key=stock_returns.get) if stock_returns else 'N/A'

    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0.001

    # Sharpe approximation (annualized from trade-level returns)
    if len(pnls) > 1:
        sharpe = np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(252)
    else:
        sharpe = 0

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
    """Run a strategy across all stocks and return metrics."""
    all_trades = []
    per_stock = {}

    for sym, df in data.items():
        try:
            signals = signal_func(df, **kwargs)
            trades = backtest_signals(df, signals)
            if trades:
                all_trades.extend(trades)
                per_stock[sym] = trades
        except Exception as e:
            pass  # skip stocks with insufficient data

    metrics = compute_metrics(all_trades, per_stock)
    if metrics is None:
        return None

    metrics['strategy'] = strategy_name
    metrics['variant'] = variant_name
    return metrics


def write_result(row):
    """Append one result row to CSV."""
    file_exists = os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0
    with open(OUTPUT_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    t_start = time.time()

    # Load data
    data = load_all_60min_data()

    # Initialize CSV only if it doesn't exist
    if not os.path.exists(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0:
        with open(OUTPUT_CSV, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    # ── Define all strategy variants ────────────────────────────────
    strategies = []

    # 1. EMA Crossovers (4 pairs x 3 ADX variants = 12)
    for fast, slow in [(5,13), (9,21), (13,34), (20,50)]:
        strategies.append((
            'EMA_Cross', f'EMA{fast}_{slow}',
            signals_ema_cross, {'fast': fast, 'slow': slow, 'adx_filter': None}
        ))
        strategies.append((
            'EMA_Cross', f'EMA{fast}_{slow}_ADX20',
            signals_ema_cross, {'fast': fast, 'slow': slow, 'adx_filter': 20}
        ))
        strategies.append((
            'EMA_Cross', f'EMA{fast}_{slow}_ADX25',
            signals_ema_cross, {'fast': fast, 'slow': slow, 'adx_filter': 25}
        ))

    # 2. SuperTrend (4 periods x 4 multipliers = 16)
    for period in [7, 10, 14]:
        for mult in [2.0, 2.5, 3.0, 3.5]:
            strategies.append((
                'SuperTrend', f'ST{period}_m{mult}',
                signals_supertrend, {'period': period, 'multiplier': mult}
            ))

    # 3. MACD (2 variants)
    strategies.append((
        'MACD', 'MACD_12_26_9',
        signals_macd, {'fast': 12, 'slow': 26, 'signal_period': 9}
    ))
    strategies.append((
        'MACD', 'MACD_8_21_5',
        signals_macd, {'fast': 8, 'slow': 21, 'signal_period': 5}
    ))

    # 4. Donchian Channel Breakout (2 variants)
    strategies.append((
        'Donchian', 'DCH20_exit10',
        signals_donchian, {'entry_period': 20, 'exit_period': 10}
    ))
    strategies.append((
        'Donchian', 'DCH40_exit10',
        signals_donchian, {'entry_period': 40, 'exit_period': 10}
    ))
    strategies.append((
        'Donchian', 'DCH40_exit20',
        signals_donchian, {'entry_period': 40, 'exit_period': 20}
    ))

    # 5. SuperTrend + SMA(200) combo (selected variants)
    for period in [7, 10, 14]:
        for mult in [2.0, 3.0]:
            strategies.append((
                'ST_SMA200', f'ST{period}_m{mult}_SMA200',
                signals_supertrend_sma, {'st_period': period, 'st_mult': mult}
            ))

    total = len(strategies)
    print(f"\nRunning {total} strategy variants across {len(data)} stocks...\n", flush=True)

    completed = 0
    # Check already done
    done = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV) as f:
            reader = csv.DictReader(f)
            for row in reader:
                done.add(row.get('variant', ''))

    for strat_name, variant, sig_func, kwargs in strategies:
        if variant in done:
            completed += 1
            continue

        t0 = time.time()
        result = run_strategy(data, strat_name, variant, sig_func, **kwargs)
        elapsed = time.time() - t0
        completed += 1

        if result:
            write_result(result)
            print(f"[{completed}/{total}] {variant:25s} | {elapsed:5.1f}s | "
                  f"Trades={result['total_trades']:5d} WR={result['win_rate']:5.1f}% "
                  f"PF={result['profit_factor']:5.3f} AvgTrade={result['avg_trade_pct']:+.3f}%",
                  flush=True)
        else:
            print(f"[{completed}/{total}] {variant:25s} | {elapsed:5.1f}s | NO TRADES", flush=True)

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"RESEARCH COMPLETE — {time.time()-t_start:.0f}s total")
    print(f"{'='*90}\n")

    # Read results and rank
    results_df = pd.read_csv(OUTPUT_CSV)
    if len(results_df) == 0:
        print("No results!")
        return

    # Filter: 100+ trades
    viable = results_df[results_df['total_trades'] >= 100].copy()
    viable = viable.sort_values('profit_factor', ascending=False)

    print(f"All variants tested: {len(results_df)}")
    print(f"Viable (100+ trades): {len(viable)}")
    print()

    # Top strategies by profit factor
    print("TOP STRATEGIES BY PROFIT FACTOR (100+ trades):")
    print("-" * 110)
    print(f"{'Variant':30s} {'Trades':>7s} {'WR%':>6s} {'PF':>7s} {'AvgTrade':>9s} {'W/L':>6s} "
          f"{'MaxDD':>7s} {'HoldBars':>9s} {'Sharpe':>7s}")
    print("-" * 110)
    for _, row in viable.head(20).iterrows():
        print(f"{row['variant']:30s} {row['total_trades']:7.0f} {row['win_rate']:6.1f} "
              f"{row['profit_factor']:7.3f} {row['avg_trade_pct']:+8.3f}% {row['win_loss_ratio']:6.3f} "
              f"{row['max_drawdown_pct']:6.1f}% {row['avg_holding_bars']:9.1f} {row['sharpe_approx']:7.3f}")

    # Profitable strategies (PF > 1.0)
    profitable = viable[viable['profit_factor'] > 1.0]
    print(f"\nProfitable strategies (PF > 1.0): {len(profitable)} / {len(viable)}")

    strong = viable[viable['profit_factor'] > 1.2]
    print(f"Strong strategies (PF > 1.2): {len(strong)} / {len(viable)}")

    # Best per strategy type
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
              f"Sharpe={best['sharpe_approx']:.3f}")


if __name__ == '__main__':
    main()
