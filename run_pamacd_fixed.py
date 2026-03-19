#!/usr/bin/env python3
"""
PA_MACD FIXED — Enter on NEXT BAR OPEN (no look-ahead bias)
=============================================================
Signal detected at Day N close → Entry at Day N+1 open
SL and TP still based on Day N-1 (the red candle) range

Configs:
1. CASH_LO_FIXED  — Cash, longs only, next-bar entry
2. CASH_LS_FIXED  — Cash, long+short, next-bar entry
3. FUT_LO_FIXED   — Futures, longs only, next-bar entry
4. FUT_LS_FIXED   — Futures, long+short, next-bar entry
5. FUT_LS_REAL_FIXED — Futures realistic costs, long+short
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

OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'pamacd_fixed_results.csv')
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


def run_backtest(symbols_data, rr, max_hold, longs_only, commission, slippage):
    """
    FIXED version: Signal detected at bar i, entry at bar i+1 open.

    Logic:
    - At bar i: check if pattern formed (green close > prev red high + MACD pos)
    - Store pending signal with SL/TP calculated from the pattern bar
    - At bar i+1: enter at OPEN price, keeping the same SL/TP levels
    """
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
        opens = df['open'].values
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        date_strs = df['date_str'].values
        macd_hist = df['macd_hist'].values
        atr_vals = df['atr'].values
        n = len(df)

        pending_signal = None  # Signal detected but not yet entered

        for i in range(n):
            # --- 1. Check exits on open positions ---
            if symbol in engine.positions:
                atr_v = atr_vals[i]
                engine.check_exits(
                    symbol=symbol, bar_idx=i, bar_date=date_strs[i],
                    high=highs[i], low=lows[i], close=closes[i],
                    atr=float(atr_v) if not np.isnan(atr_v) else None,
                )

            # --- 2. Enter pending signal at TODAY'S OPEN ---
            if pending_signal is not None and symbol not in engine.positions:
                # Modify signal to use today's open as entry price
                sig_dir, sig_sl, sig_rr_mult, sig_max_hold = pending_signal
                entry_price = opens[i]

                # Recalculate target based on actual entry and original SL
                if sig_dir == Direction.LONG:
                    risk = entry_price - sig_sl
                    if risk > 0:
                        tp = entry_price + sig_rr_mult * risk
                        signal = TradeSignal(sig_dir, entry_price, sig_sl, tp, None, sig_max_hold)
                        engine.open_position(symbol, signal, i, date_strs[i])
                else:  # SHORT
                    risk = sig_sl - entry_price
                    if risk > 0:
                        tp = entry_price - sig_rr_mult * risk
                        signal = TradeSignal(sig_dir, entry_price, sig_sl, tp, None, sig_max_hold)
                        engine.open_position(symbol, signal, i, date_strs[i])

                pending_signal = None

            # --- 3. Detect pattern at TODAY'S CLOSE (for tomorrow's entry) ---
            if i < 2 or i >= n - 1:  # Need prev bar and next bar
                pending_signal = None
                continue

            if symbol in engine.positions:
                pending_signal = None
                continue

            row_close = closes[i]
            row_open = opens[i]
            row_high = highs[i]
            prev_close = closes[i - 1]
            prev_open = opens[i - 1]
            prev_high = highs[i - 1]
            prev_low = lows[i - 1]
            mhist = macd_hist[i]

            is_green = row_close > row_open
            is_red = row_close < row_open
            prev_green = prev_close > prev_open
            prev_red = prev_close < prev_open
            macd_pos = (not np.isnan(mhist)) and mhist > 0
            macd_neg = (not np.isnan(mhist)) and mhist < 0

            pending_signal = None

            # LONG: green candle closes above prev red high + MACD positive
            if is_green and prev_red and row_close > prev_high and macd_pos:
                # SL = prev red candle's low (the pattern bar)
                sl = prev_low
                pending_signal = (Direction.LONG, sl, rr, max_hold)

            # SHORT: red candle closes below prev green low + MACD negative
            elif not longs_only:
                if is_red and prev_green and row_close < prev_low and macd_neg:
                    sl = prev_high
                    pending_signal = (Direction.SHORT, sl, rr, max_hold)

        # Update equity curve
        for i in range(n):
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


def print_results(label, result):
    print(f' CAGR={result.cagr:.2f}% PF={result.profit_factor:.2f} MaxDD={result.max_drawdown:.2f}% '
          f'Trades={result.total_trades} (L:{result.long_trades} S:{result.short_trades}) '
          f'WR={result.win_rate:.1f}% (L:{result.long_win_rate:.1f}% S:{result.short_win_rate:.1f}%)')
    print(f'         Avg Win={result.avg_win:.2f}% Avg Loss={result.avg_loss:.2f}% RR={result.avg_rr:.2f} Hold={result.avg_bars_held:.1f}d')
    print(f'         Exits: {result.exit_reasons}')
    print(f'         Year-by-year:')
    for year in range(2018, 2026):
        ys = result.yearly_stats.get(year, {})
        t = ys.get('trades', 0)
        wr = ys.get('win_rate', 0)
        pf = ys.get('profit_factor', 0)
        pnl = ys.get('pnl_pct', 0)
        lt = ys.get('long_trades', 0)
        st = ys.get('short_trades', 0)
        print(f'           {year}: Trades={t:>4d} WR={wr:>5.1f}% PF={pf:>5.2f} PnL={pnl:>+8.1f}% (L:{lt} S:{st})')
    print()


if __name__ == '__main__':
    print('='*70)
    print('PA_MACD FIXED — Next-Bar Entry (No Look-Ahead Bias)')
    print('Signal at Day N close -> Entry at Day N+1 open')
    print('='*70)
    print()

    print('Loading data for 50 F&O stocks...')
    t0 = time.time()
    symbols_data = load_data_from_db(FNO_STOCKS, 'day', '2018-01-01', '2025-12-31')
    print(f'Loaded {len(symbols_data)} symbols in {time.time()-t0:.0f}s')
    print()

    with open(OUTPUT_CSV, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    configs = [
        ('CASH_LO_FIXED',     True,  0.001,  0.001,  'cash'),
        ('CASH_LS_FIXED',     False, 0.001,  0.001,  'cash'),
        ('FUT_LO_FIXED',      True,  0.0001, 0.0005, 'futures'),
        ('FUT_LS_FIXED',      False, 0.0001, 0.0005, 'futures'),
        ('FUT_LS_REAL_FIXED', False, 0.0003, 0.001,  'futures'),
    ]

    for i, (label, longs_only, comm, slip, mode) in enumerate(configs):
        t1 = time.time()
        dir_str = 'LO' if longs_only else 'L+S'
        print(f'[{i+1}/{len(configs)}] {label} (comm={comm*100:.2f}% slip={slip*100:.2f}% {dir_str})...', flush=True)

        result = run_backtest(symbols_data, rr=3.0, max_hold=10, longs_only=longs_only,
                              commission=comm, slippage=slip)

        write_row(label, mode, comm, slip, 'LO' if longs_only else 'LS', result)

        elapsed = time.time() - t1
        print(f'  Done in {elapsed:.0f}s')
        print_results(label, result)
        sys.stdout.flush()

    print(f'Results saved: {OUTPUT_CSV}')
