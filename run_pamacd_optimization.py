#!/usr/bin/env python3
"""
PA_MACD Optimization Sweep
===========================
Tests concentration, risk-reward, hold period, and direction variants
of the PA_MACD (Price Action + MACD) strategy.

Target: Push CAGR from 20% to 24%+ while keeping MaxDD < 20%.
"""
import csv, os, sys, time, json, logging
import numpy as np
import pandas as pd

logging.disable(logging.WARNING)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from services.intraday_backtest_engine import (
    IntradayBacktestEngine, TradeSignal, Direction, ExitType,
    load_data_from_db, get_all_symbols_for_timeframe, BacktestResult,
)
from services.technical_indicators import calc_macd, calc_atr

OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'pamacd_optimization_results.csv')
FIELDNAMES = [
    'label', 'n_stocks', 'rr', 'max_hold', 'direction', 'pos_size_pct', 'max_positions',
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

# --- Stock universes by liquidity ---
TOP_10 = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
          'SBIN', 'BHARTIARTL', 'ITC', 'KOTAKBANK', 'HINDUNILVR']

TOP_20 = TOP_10 + ['LT', 'AXISBANK', 'BAJFINANCE', 'MARUTI', 'HCLTECH',
                    'SUNPHARMA', 'TATAMOTORS', 'NTPC', 'POWERGRID', 'TITAN']

TOP_30 = TOP_20 + ['WIPRO', 'ULTRACEMCO', 'ADANIENT', 'ADANIPORTS', 'JSWSTEEL',
                    'TATASTEEL', 'TECHM', 'M&M', 'COALINDIA', 'ONGC']

# --- Strategy functions ---

def prepare_indicators(df):
    """Add MACD and ATR to daily dataframe."""
    df = df.copy()
    df['date_str'] = df.index.strftime('%Y-%m-%d')
    df['macd'], df['macd_signal_line'], df['macd_hist'] = calc_macd(df['close'])
    df['atr'] = calc_atr(df, 14)
    return df


def strategy_pa_macd(df, i, rr, max_hold, longs_only=False):
    """PA_MACD strategy. Returns TradeSignal or None."""
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

    # LONG: green candle closes above previous red candle's high + MACD positive
    if is_green and prev_red and row['close'] > prev['high'] and macd_pos:
        entry = prev['high']
        sl = prev['low']
        risk = entry - sl
        if risk <= 0:
            return None
        tp = entry + rr * risk
        return TradeSignal(Direction.LONG, entry, sl, tp, None, max_hold)

    # SHORT: red candle closes below previous green candle's low + MACD negative
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


def run_config(label, symbols_data, rr, max_hold, longs_only, pos_size_pct, max_positions):
    """Run a single PA_MACD config across given symbols."""
    engine = IntradayBacktestEngine(
        initial_capital=10_000_000,
        position_size_pct=pos_size_pct,
        max_positions=max_positions,
        commission_pct=0.001,
        slippage_pct=0.001,
        fixed_sizing=True,
    )

    prepared = {}
    for symbol, df in symbols_data.items():
        try:
            prepared[symbol] = prepare_indicators(df)
        except Exception:
            continue

    if not prepared:
        return None

    for symbol, df in prepared.items():
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        date_strs = df['date_str'].values

        for i in range(len(df)):
            # Check exits first
            if symbol in engine.positions:
                atr_v = df['atr'].values[i]
                engine.check_exits(
                    symbol=symbol, bar_idx=i, bar_date=date_strs[i],
                    high=highs[i], low=lows[i], close=closes[i],
                    atr=float(atr_v) if not np.isnan(atr_v) else None,
                )

            # Generate signals
            if symbol not in engine.positions:
                signal = strategy_pa_macd(df, i, rr, max_hold, longs_only)
                if signal:
                    engine.open_position(symbol, signal, i, date_strs[i])

        # Update equity curve per symbol
        for i in range(len(df)):
            engine.update_equity(date_strs[i], {symbol: closes[i]})

    return engine.get_results()


def write_row(label, cfg, result):
    """Write a single result row to CSV."""
    row = {
        'label': label,
        'n_stocks': cfg['n_stocks'],
        'rr': cfg['rr'],
        'max_hold': cfg['max_hold'],
        'direction': cfg['direction'],
        'pos_size_pct': cfg['pos_size_pct'],
        'max_positions': cfg['max_positions'],
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


# --- Configuration grid ---
CONFIGS = [
    # (label, n_stocks, rr, max_hold, direction, pos_size_pct, max_positions, symbol_list)
    ('CONC10_RR3_MH10_LS',   10, 3.0, 10, 'LS', 0.20, 5,  TOP_10),
    ('CONC10_RR3_MH10_LO',   10, 3.0, 10, 'LO', 0.20, 5,  TOP_10),
    ('CONC10_RR2.5_MH7_LS',  10, 2.5,  7, 'LS', 0.20, 5,  TOP_10),
    ('CONC10_RR3.5_MH10_LS', 10, 3.5, 10, 'LS', 0.20, 5,  TOP_10),
    ('CONC10_RR4_MH15_LS',   10, 4.0, 15, 'LS', 0.20, 5,  TOP_10),
    ('CONC10_RR3_MH5_LS',    10, 3.0,  5, 'LS', 0.20, 5,  TOP_10),
    ('CONC20_RR3_MH10_LS',   20, 3.0, 10, 'LS', 0.15, 7,  TOP_20),
    ('CONC20_RR3_MH10_LO',   20, 3.0, 10, 'LO', 0.15, 7,  TOP_20),
    ('CONC20_RR2.5_MH7_LS',  20, 2.5,  7, 'LS', 0.15, 7,  TOP_20),
    ('CONC20_RR3.5_MH10_LS', 20, 3.5, 10, 'LS', 0.15, 7,  TOP_20),
    ('CONC20_RR4_MH15_LS',   20, 4.0, 15, 'LS', 0.15, 7,  TOP_20),
    ('CONC20_RR3_MH5_LS',    20, 3.0,  5, 'LS', 0.15, 7,  TOP_20),
    ('CONC30_RR3_MH10_LS',   30, 3.0, 10, 'LS', 0.12, 8,  TOP_30),
    ('CONC30_RR3_MH10_LO',   30, 3.0, 10, 'LO', 0.12, 8,  TOP_30),
    ('CONC30_RR2.5_MH7_LS',  30, 2.5,  7, 'LS', 0.12, 8,  TOP_30),
    ('CONC30_RR3.5_MH10_LS', 30, 3.5, 10, 'LS', 0.12, 8,  TOP_30),
    ('CONC50_RR2_MH5_LS',    50, 2.0,  5, 'LS', 0.10, 10, None),  # None = use FNO universe
    ('CONC50_RR3_MH10_LS',   50, 3.0, 10, 'LS', 0.10, 10, None),
    ('CONC50_RR3_MH10_LO',   50, 3.0, 10, 'LO', 0.10, 10, None),
    ('CONC50_RR3.5_MH15_LS', 50, 3.5, 15, 'LS', 0.10, 10, None),
]


def main():
    print('=== PA_MACD Optimization Sweep ===')
    print(f'Total configs: {len(CONFIGS)}')

    # Write header if needed
    if not os.path.exists(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0:
        with open(OUTPUT_CSV, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    # Load already completed
    done = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV) as f:
            done = {row['label'] for row in csv.DictReader(f)}
    if done:
        print(f'Skipping {len(done)} already-completed configs')

    # Load ALL data once (superset — we need up to ~86 FNO stocks)
    print('\nLoading daily data...', flush=True)
    t0 = time.time()

    all_symbols = get_all_symbols_for_timeframe('day', min_rows=1500)
    try:
        from services.data_manager import FNO_LOT_SIZES
        fno_symbols = [s for s in FNO_LOT_SIZES.keys() if s in all_symbols]
    except ImportError:
        fno_symbols = all_symbols[:100]

    # Make sure all our concentrated universe stocks are included
    all_needed = set(fno_symbols) | set(TOP_30)
    all_needed_list = [s for s in all_needed if s in all_symbols]

    all_data = load_data_from_db(all_needed_list, 'day', '2018-01-01', '2025-12-31')
    print(f'Loaded {len(all_data)} stocks in {time.time()-t0:.0f}s\n', flush=True)

    # FNO subset for CONC50 configs (limit to ~50 most liquid)
    fno_data_50 = {s: all_data[s] for s in fno_symbols[:50] if s in all_data}

    # Pre-build symbol data subsets
    data_by_universe = {
        10: {s: all_data[s] for s in TOP_10 if s in all_data},
        20: {s: all_data[s] for s in TOP_20 if s in all_data},
        30: {s: all_data[s] for s in TOP_30 if s in all_data},
        50: fno_data_50,
    }

    total = len(CONFIGS)
    for idx, (label, n_stocks, rr, max_hold, direction, pos_size_pct, max_pos, sym_list) in enumerate(CONFIGS, 1):
        if label in done:
            print(f'[{idx}/{total}] {label} — SKIP (done)')
            continue

        longs_only = (direction == 'LO')
        symbols_data = data_by_universe[n_stocks]

        print(f'[{idx}/{total}] {label} ({len(symbols_data)} stocks, RR={rr}, MH={max_hold}, '
              f'{"LONG only" if longs_only else "L+S"}) ...', end='', flush=True)

        t1 = time.time()
        try:
            result = run_config(label, symbols_data, rr, max_hold, longs_only, pos_size_pct, max_pos)

            if result is None or result.total_trades == 0:
                print(f' NO TRADES', flush=True)
                continue

            elapsed = time.time() - t1
            print(f' {elapsed:.0f}s | Trades={result.total_trades} '
                  f'WR={result.win_rate:.1f}% PF={result.profit_factor:.2f} '
                  f'CAGR={result.cagr:.1f}% MaxDD={result.max_drawdown:.1f}%', flush=True)

            # Year-by-year summary
            if result.yearly_stats:
                years_str = ' | '.join(
                    f'{y}:{s["trades"]}t/{s["win_rate"]:.0f}%/{s["pnl_pct"]:+.1f}%'
                    for y, s in sorted(result.yearly_stats.items())
                )
                print(f'  Years: {years_str}', flush=True)

            cfg = {
                'n_stocks': n_stocks, 'rr': rr, 'max_hold': max_hold,
                'direction': direction, 'pos_size_pct': pos_size_pct,
                'max_positions': max_pos,
            }
            write_row(label, cfg, result)

        except Exception as e:
            elapsed = time.time() - t1
            print(f' ERROR ({elapsed:.0f}s): {e}', flush=True)

    print('\n=== SWEEP COMPLETE ===')
    # Print summary
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV) as f:
            rows = list(csv.DictReader(f))
        if rows:
            print(f'\n{"Label":<30} {"CAGR":>6} {"MaxDD":>6} {"PF":>6} {"WR":>6} {"Trades":>6} {"Sharpe":>6} {"Calmar":>6}')
            print('-' * 100)
            for r in sorted(rows, key=lambda x: float(x['cagr']), reverse=True):
                print(f'{r["label"]:<30} {float(r["cagr"]):>6.1f} {float(r["max_drawdown"]):>6.1f} '
                      f'{float(r["profit_factor"]):>6.2f} {float(r["win_rate"]):>6.1f} '
                      f'{int(r["total_trades"]):>6} {float(r["sharpe"]):>6.2f} {float(r["calmar"]):>6.2f}')


if __name__ == '__main__':
    main()
