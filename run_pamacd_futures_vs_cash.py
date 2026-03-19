#!/usr/bin/env python3
"""
PA_MACD: Futures (L+S) vs Cash (LO) comparison
================================================
- Cash: 0.1% commission + 0.1% slippage, LONGS ONLY
- Futures: 0.01% commission + 0.05% slippage, LONG + SHORT
"""
import csv, os, sys, time, json, logging
import numpy as np
import pandas as pd

logging.disable(logging.WARNING)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from services.intraday_backtest_engine import (
    IntradayBacktestEngine, TradeSignal, Direction, ExitType,
    load_data_from_db, BacktestResult,
)
from services.technical_indicators import calc_macd, calc_atr

OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'pamacd_futures_vs_cash.csv')
FIELDNAMES = [
    'label', 'mode', 'commission', 'slippage', 'direction',
    'total_trades', 'long_trades', 'short_trades',
    'win_rate', 'long_wr', 'short_wr',
    'total_pnl', 'pnl_pct', 'profit_factor', 'cagr',
    'max_drawdown', 'sharpe', 'sortino', 'calmar',
    'avg_win', 'avg_loss', 'avg_rr', 'avg_bars_held',
    'exit_reasons',
    'y2018_trades', 'y2018_wr', 'y2018_pf', 'y2018_pnl_pct',
    'y2019_trades', 'y2019_wr', 'y2019_pf', 'y2019_pnl_pct',
    'y2020_trades', 'y2020_wr', 'y2020_pf', 'y2020_pnl_pct',
    'y2021_trades', 'y2021_wr', 'y2021_pf', 'y2021_pnl_pct',
    'y2022_trades', 'y2022_wr', 'y2022_pf', 'y2022_pnl_pct',
    'y2023_trades', 'y2023_wr', 'y2023_pf', 'y2023_pnl_pct',
    'y2024_trades', 'y2024_wr', 'y2024_pf', 'y2024_pnl_pct',
    'y2025_trades', 'y2025_wr', 'y2025_pf', 'y2025_pnl_pct',
]

# Top 50 F&O stocks
FNO_STOCKS = [
    'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'SBIN', 'BHARTIARTL',
    'ITC', 'KOTAKBANK', 'HINDUNILVR', 'LT', 'AXISBANK', 'BAJFINANCE', 'MARUTI',
    'HCLTECH', 'SUNPHARMA', 'TATAMOTORS', 'NTPC', 'POWERGRID', 'TITAN',
    'WIPRO', 'ULTRACEMCO', 'ADANIENT', 'ADANIPORTS', 'NESTLEIND', 'JSWSTEEL',
    'TATASTEEL', 'TECHM', 'M&M', 'COALINDIA', 'ONGC', 'GRASIM', 'BAJAJFINSV',
    'APOLLOHOSP', 'DIVISLAB', 'DRREDDY', 'CIPLA', 'HEROMOTOCO', 'EICHERMOT',
    'BPCL', 'INDUSINDBK', 'TATACONSUM', 'SHRIRAMFIN', 'BRITANNIA', 'HINDALCO',
    'ASIANPAINT', 'SBILIFE', 'HDFCLIFE', 'PIDILITIND', 'DABUR',
]


def prepare_indicators(df):
    df = df.copy()
    df['date_str'] = df.index.strftime('%Y-%m-%d')
    df['macd'], df['macd_signal_line'], df['macd_hist'] = calc_macd(df['close'])
    df['atr'] = calc_atr(df, 14)
    return df


def strategy_pa_macd(df, i, rr, max_hold, longs_only=False):
    if i < 2:
        return None
    row = df.iloc[i]
    prev = df.iloc[i - 1]

    is_green = row['close'] > row['open']
    is_red = row['close'] < row['open']
    prev_green = prev['close'] > prev['open']
    prev_red = prev['close'] < prev['open']

    macd_pos = row.get('macd_hist', 0) > 0
    macd_neg = row.get('macd_hist', 0) < 0

    if is_green and prev_red and row['close'] > prev['high'] and macd_pos:
        entry = prev['high']
        sl = prev['low']
        risk = entry - sl
        if risk <= 0:
            return None
        tp = entry + rr * risk
        return TradeSignal(Direction.LONG, entry, sl, tp, None, max_hold)

    if not longs_only:
        if is_red and prev_green and row['close'] < prev['low'] and macd_neg:
            entry = prev['low']
            sl = prev['high']
            risk = sl - entry
            if risk <= 0:
                return None
            tp = entry - rr * risk
            return TradeSignal(Direction.SHORT, entry, sl, tp, None, max_hold)

    return None


def run_backtest(symbols_data, rr, max_hold, longs_only, commission, slippage):
    engine = IntradayBacktestEngine(
        initial_capital=10_000_000,
        position_size_pct=0.10,
        max_positions=10,
        commission_pct=commission,
        slippage_pct=slippage,
        fixed_sizing=True,
    )

    prepared = {}
    for symbol, df in symbols_data.items():
        try:
            prepared[symbol] = prepare_indicators(df)
        except Exception:
            continue

    for symbol, df in prepared.items():
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        date_strs = df['date_str'].values

        for i in range(len(df)):
            if symbol in engine.positions:
                atr_v = df['atr'].values[i]
                engine.check_exits(
                    symbol=symbol, bar_idx=i, bar_date=date_strs[i],
                    high=highs[i], low=lows[i], close=closes[i],
                    atr=float(atr_v) if not np.isnan(atr_v) else None,
                )
            if symbol not in engine.positions:
                signal = strategy_pa_macd(df, i, rr, max_hold, longs_only)
                if signal:
                    engine.open_position(symbol, signal, i, date_strs[i])

        for i in range(len(df)):
            engine.update_equity(date_strs[i], {symbol: closes[i]})

    return engine.get_results()


def write_row(label, mode, commission, slippage, direction, result):
    row = {
        'label': label, 'mode': mode,
        'commission': commission, 'slippage': slippage, 'direction': direction,
        'total_trades': result.total_trades,
        'long_trades': result.long_trades,
        'short_trades': result.short_trades,
        'win_rate': round(result.win_rate, 2),
        'long_wr': round(result.long_win_rate, 2),
        'short_wr': round(result.short_win_rate, 2),
        'total_pnl': round(result.total_pnl, 0),
        'pnl_pct': round(result.total_pnl_pct, 2),
        'profit_factor': round(result.profit_factor, 4),
        'cagr': round(result.cagr, 2),
        'max_drawdown': round(result.max_drawdown, 2),
        'sharpe': round(result.sharpe_ratio, 4),
        'sortino': round(result.sortino_ratio, 4),
        'calmar': round(result.calmar_ratio, 4),
        'avg_win': round(result.avg_win, 2),
        'avg_loss': round(result.avg_loss, 2),
        'avg_rr': round(result.avg_rr, 2),
        'avg_bars_held': round(result.avg_bars_held, 1),
        'exit_reasons': json.dumps(result.exit_reasons),
    }
    for year in range(2018, 2026):
        ys = result.yearly_stats.get(year, {})
        row[f'y{year}_trades'] = ys.get('trades', 0)
        row[f'y{year}_wr'] = round(ys.get('win_rate', 0), 2)
        row[f'y{year}_pf'] = round(ys.get('profit_factor', 0), 4)
        row[f'y{year}_pnl_pct'] = round(ys.get('pnl_pct', 0), 2)

    with open(OUTPUT_CSV, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)


if __name__ == '__main__':
    print('Loading data for 50 F&O stocks...')
    t0 = time.time()
    symbols_data = load_data_from_db(FNO_STOCKS, 'day', '2018-01-01', '2025-12-31')
    print(f'Loaded {len(symbols_data)} symbols in {time.time()-t0:.0f}s')

    # Write header
    with open(OUTPUT_CSV, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    configs = [
        # (label, longs_only, commission, slippage, mode)
        ('CASH_LO',     True,  0.001, 0.001, 'cash'),      # Cash, longs only (baseline)
        ('CASH_LS',     False, 0.001, 0.001, 'cash'),      # Cash, long+short
        ('FUT_LS',      False, 0.0001, 0.0005, 'futures'),  # Futures, long+short (0.01% comm + 0.05% slip)
        ('FUT_LO',      True,  0.0001, 0.0005, 'futures'),  # Futures, longs only
        ('FUT_LS_REAL', False, 0.0003, 0.001, 'futures'),   # Futures realistic (0.03% comm + 0.1% slip)
    ]

    for i, (label, longs_only, comm, slip, mode) in enumerate(configs):
        t1 = time.time()
        print(f'[{i+1}/{len(configs)}] {label} (comm={comm*100:.2f}% slip={slip*100:.2f}% {"LO" if longs_only else "L+S"})...', end='', flush=True)

        result = run_backtest(symbols_data, rr=3.0, max_hold=10, longs_only=longs_only,
                              commission=comm, slippage=slip)

        direction = 'LO' if longs_only else 'LS'
        write_row(label, mode, comm, slip, direction, result)

        elapsed = time.time() - t1
        print(f' {elapsed:.0f}s | CAGR={result.cagr:.2f}% PF={result.profit_factor:.2f} MaxDD={result.max_drawdown:.2f}% '
              f'Trades={result.total_trades} WR={result.win_rate:.1f}%')

        # Print year-by-year
        print(f'         Year-by-year: ', end='')
        for year in range(2018, 2026):
            ys = result.yearly_stats.get(year, {})
            pnl = ys.get('pnl_pct', 0)
            wr = ys.get('win_rate', 0)
            pf = ys.get('profit_factor', 0)
            print(f'{year}:{pnl:+.1f}%', end=' ')
        print()
        print()
        sys.stdout.flush()

    print(f'\nAll done! Results: {OUTPUT_CSV}')
