"""
Daily Strategies with NO Look-Ahead Bias
==========================================
6 strategies x 4 variants = 24 configs

All entry prices are determinable BEFORE or AT the time of entry.
No strategy uses today's close to decide today's entry.

Strategies:
1. PA_MACD BuyStop - Buy stop at yesterday's high when red candle + MACD positive
2. Gap + Momentum - Gap up > 0.5% with MACD positive, enter at open
3. Previous Day Range Breakout - Buy stop at 5-bar highest high
4. EMA Pullback - Limit order at EMA(20) when trending up
5. Keltner Channel Mean Reversion - Close < KC lower, enter next open
6. Inside Day Breakout - Buy/sell stop at yesterday's range after inside day
"""

import os
import sys
import csv
import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

# Suppress logging
logging.disable(logging.WARNING)

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from services.intraday_backtest_engine import (
    IntradayBacktestEngine, TradeSignal, Direction, ExitType,
    BacktestResult, load_data_from_db
)
from services.technical_indicators import (
    calc_ema, calc_rsi, calc_atr, calc_macd, calc_keltner_channels, calc_adx
)

# ============================================================================
# Config
# ============================================================================
OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'daily_no_lookahead_results.csv')

FNO_STOCKS = [
    'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'SBIN', 'BHARTIARTL',
    'ITC', 'KOTAKBANK', 'HINDUNILVR', 'LT', 'AXISBANK', 'BAJFINANCE', 'MARUTI',
    'HCLTECH', 'SUNPHARMA', 'TATAMOTORS', 'NTPC', 'POWERGRID', 'TITAN',
    'WIPRO', 'ULTRACEMCO', 'ADANIENT', 'ADANIPORTS', 'NESTLEIND', 'JSWSTEEL',
    'TATASTEEL', 'TECHM', 'M&M', 'COALINDIA', 'ONGC', 'GRASIM', 'BAJAJFINSV',
    'APOLLOHOSP', 'DIVISLAB', 'DRREDDY', 'CIPLA', 'HEROMOTOCO', 'EICHERMOT',
    'BPCL', 'INDUSINDBK', 'TATACONSUM', 'SHRIRAMFIN', 'BRITANNIA', 'HINDALCO',
    'ASIANPAINT', 'SBILIFE', 'HDFCLIFE', 'PIDILITIND', 'DABUR'
]

INITIAL_CAPITAL = 10_000_000  # Rs 1 Crore
POSITION_SIZE_PCT = 0.10      # 10% per position
MAX_POSITIONS = 10

FIELDNAMES = [
    'label', 'strategy', 'variant', 'mode', 'direction_filter',
    'total_trades', 'win_rate', 'profit_factor', 'cagr', 'sharpe',
    'sortino', 'calmar', 'max_drawdown', 'total_pnl_pct', 'avg_win',
    'avg_loss', 'avg_rr', 'avg_bars_held', 'long_trades', 'short_trades',
    'long_wr', 'short_wr', 'exit_reasons',
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
# Indicator Precomputation
# ============================================================================

def precompute_indicators(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Precompute all needed indicators for all symbols."""
    result = {}
    for sym, df in data.items():
        df = df.copy()
        if len(df) < 60:
            continue

        # MACD
        macd_line, macd_signal, macd_hist = calc_macd(df['close'])
        df['macd_hist'] = macd_hist

        # EMA 20, 50
        df['ema20'] = calc_ema(df['close'], 20)
        df['ema50'] = calc_ema(df['close'], 50)

        # RSI
        df['rsi'] = calc_rsi(df['close'], 14)

        # ATR
        df['atr'] = calc_atr(df, 14)

        # ADX
        adx, plus_di, minus_di = calc_adx(df, 14)
        df['adx'] = adx

        # Keltner Channels (20, 10, 2.0)
        kc_mid, kc_upper, kc_lower = calc_keltner_channels(df, 20, 10, 2.0)
        df['kc_mid'] = kc_mid
        df['kc_upper'] = kc_upper
        df['kc_lower'] = kc_lower

        # SMA 200
        df['sma200'] = df['close'].rolling(200).mean()

        # Previous bar values (shifted)
        df['prev_open'] = df['open'].shift(1)
        df['prev_high'] = df['high'].shift(1)
        df['prev_low'] = df['low'].shift(1)
        df['prev_close'] = df['close'].shift(1)
        df['prev_macd_hist'] = df['macd_hist'].shift(1)

        # Is previous candle red/green?
        df['prev_red'] = df['prev_close'] < df['prev_open']
        df['prev_green'] = df['prev_close'] > df['prev_open']

        # For Inside Day: bar i-2 values
        df['prev2_high'] = df['high'].shift(2)
        df['prev2_low'] = df['low'].shift(2)

        # 5-bar lookback for range breakout (bars i-5 to i-1)
        df['highest_5'] = df['high'].rolling(5).max().shift(1)  # highest of last 5 bars (shifted = i-5 to i-1)
        df['lowest_5'] = df['low'].rolling(5).min().shift(1)

        result[sym] = df
    return result


# ============================================================================
# Strategy Functions
# Each returns a list of (bar_index, symbol, TradeSignal) for the whole dataset
# ============================================================================

def strategy_pa_macd_buystop(data: Dict[str, pd.DataFrame], longs_only: bool = False) -> List[Tuple]:
    """
    Strategy 1: PA_MACD BuyStop
    - At bar i: check bar i-1 was red AND MACD hist at i-1 positive
    - Buy stop level = bar i-1 high. If bar i high >= that level, triggered.
    - Entry price = bar i-1 high (buy stop fills there)
    - SL = bar i-1 low, TP = entry + 3*(entry - SL), max hold 10 days

    For shorts: bar i-1 green AND MACD hist at i-1 negative
    - Sell stop at bar i-1 low. If bar i low <= that level, triggered.
    - Entry price = bar i-1 low, SL = bar i-1 high
    """
    signals = []
    for sym, df in data.items():
        for i in range(2, len(df)):
            row = df.iloc[i]
            date_str = row['date_str']

            # LONG: prev candle red + MACD hist positive at prev bar
            if row['prev_red'] and row['prev_macd_hist'] > 0:
                buy_stop = row['prev_high']
                if row['high'] >= buy_stop and buy_stop > 0:
                    sl = row['prev_low']
                    risk = buy_stop - sl
                    if risk > 0:
                        tp = buy_stop + 3 * risk
                        signals.append((i, sym, date_str, TradeSignal(
                            direction=Direction.LONG,
                            entry_price=buy_stop,
                            stop_loss=sl,
                            target=tp,
                            max_hold_bars=10,
                        )))

            # SHORT: prev candle green + MACD hist negative at prev bar
            if not longs_only and row['prev_green'] and row['prev_macd_hist'] < 0:
                sell_stop = row['prev_low']
                if row['low'] <= sell_stop and sell_stop > 0:
                    sl = row['prev_high']
                    risk = sl - sell_stop
                    if risk > 0:
                        tp = sell_stop - 3 * risk
                        signals.append((i, sym, date_str, TradeSignal(
                            direction=Direction.SHORT,
                            entry_price=sell_stop,
                            stop_loss=sl,
                            target=tp,
                            max_hold_bars=10,
                        )))
    return signals


def strategy_gap_momentum(data: Dict[str, pd.DataFrame], longs_only: bool = False) -> List[Tuple]:
    """
    Strategy 2: Gap + Momentum
    - Today opens > yesterday close by > 0.5% AND yesterday MACD hist positive
    - Enter at today's open (known at open)
    - SL = yesterday's low, TP = 3x risk, max hold 10 days

    Shorts: gap down > 0.5% + MACD hist negative
    """
    signals = []
    for sym, df in data.items():
        for i in range(2, len(df)):
            row = df.iloc[i]
            date_str = row['date_str']

            # LONG: gap up > 0.5%
            if row['prev_close'] > 0:
                gap_pct = (row['open'] - row['prev_close']) / row['prev_close']
                if gap_pct > 0.005 and row['prev_macd_hist'] > 0:
                    entry = row['open']
                    sl = row['prev_low']
                    risk = entry - sl
                    if risk > 0:
                        tp = entry + 3 * risk
                        signals.append((i, sym, date_str, TradeSignal(
                            direction=Direction.LONG,
                            entry_price=entry,
                            stop_loss=sl,
                            target=tp,
                            max_hold_bars=10,
                        )))

            # SHORT: gap down > 0.5%
            if not longs_only and row['prev_close'] > 0:
                gap_pct = (row['prev_close'] - row['open']) / row['prev_close']
                if gap_pct > 0.005 and row['prev_macd_hist'] < 0:
                    entry = row['open']
                    sl = row['prev_high']
                    risk = sl - entry
                    if risk > 0:
                        tp = entry - 3 * risk
                        signals.append((i, sym, date_str, TradeSignal(
                            direction=Direction.SHORT,
                            entry_price=entry,
                            stop_loss=sl,
                            target=tp,
                            max_hold_bars=10,
                        )))
    return signals


def strategy_range_breakout(data: Dict[str, pd.DataFrame], longs_only: bool = False) -> List[Tuple]:
    """
    Strategy 3: Previous Day Range Breakout (5-bar)
    - Bar i high > highest high of bars i-5 to i-1 (breakout)
    - Entry = that highest high (buy stop placed pre-market)
    - SL = lowest low of last 5 bars, TP = entry + 2x risk, max hold 10 days

    Shorts: bar i low < lowest low of last 5 bars
    """
    signals = []
    for sym, df in data.items():
        for i in range(6, len(df)):
            row = df.iloc[i]
            date_str = row['date_str']
            hh5 = row['highest_5']
            ll5 = row['lowest_5']

            if pd.isna(hh5) or pd.isna(ll5):
                continue

            # LONG breakout
            if row['high'] >= hh5 and hh5 > 0:
                entry = hh5
                sl = ll5
                risk = entry - sl
                if risk > 0:
                    tp = entry + 2 * risk
                    signals.append((i, sym, date_str, TradeSignal(
                        direction=Direction.LONG,
                        entry_price=entry,
                        stop_loss=sl,
                        target=tp,
                        max_hold_bars=10,
                    )))

            # SHORT breakdown
            if not longs_only and row['low'] <= ll5 and ll5 > 0:
                entry = ll5
                sl = hh5
                risk = sl - entry
                if risk > 0:
                    tp = entry - 2 * risk
                    signals.append((i, sym, date_str, TradeSignal(
                        direction=Direction.SHORT,
                        entry_price=entry,
                        stop_loss=sl,
                        target=tp,
                        max_hold_bars=10,
                    )))
    return signals


def strategy_ema_pullback(data: Dict[str, pd.DataFrame], longs_only: bool = False) -> List[Tuple]:
    """
    Strategy 4: EMA Pullback
    - Conditions at bar i-1 close: close > EMA(20) > EMA(50), ADX > 20, RSI < 60
    - At bar i: low touches or goes below EMA(20) at bar i → buy at EMA(20) level (limit order)
    - EMA(20) at bar i is computable from bar i-1's close (approx; we use bar i-1's EMA value
      as the limit order level, set pre-market)
    - Actually: we use the EMA(20) value at bar i-1 as the limit price (known pre-market)
    - SL = 2*ATR below EMA(20), TP = 3x risk, max hold 15 days

    No shorts for this strategy (trend pullback is inherently directional).
    """
    signals = []
    for sym, df in data.items():
        for i in range(2, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i - 1]
            date_str = row['date_str']

            # Conditions at bar i-1
            if (pd.isna(prev['ema20']) or pd.isna(prev['ema50']) or
                    pd.isna(prev['adx']) or pd.isna(prev['rsi']) or pd.isna(prev['atr'])):
                continue

            if (prev['close'] > prev['ema20'] > prev['ema50'] and
                    prev['adx'] > 20 and prev['rsi'] < 60):
                # Limit order at prev EMA20 level
                limit_price = prev['ema20']
                if row['low'] <= limit_price and limit_price > 0:
                    # Order filled
                    entry = limit_price
                    sl = entry - 2 * prev['atr']
                    risk = entry - sl
                    if risk > 0 and sl > 0:
                        tp = entry + 3 * risk
                        signals.append((i, sym, date_str, TradeSignal(
                            direction=Direction.LONG,
                            entry_price=entry,
                            stop_loss=sl,
                            target=tp,
                            max_hold_bars=15,
                        )))
    return signals


def strategy_kc_mean_reversion(data: Dict[str, pd.DataFrame], longs_only: bool = False) -> List[Tuple]:
    """
    Strategy 5: Keltner Channel Mean Reversion (KC6-style)
    - At bar i-1 close: close < KC(20,2.0) lower AND close > SMA(200)
    - Entry at bar i open (next bar open)
    - Exit at KC mid (sell limit each morning) — we approximate by checking if bar i high >= KC mid at i-1
    - SL = 5% below entry, max hold 15 days
    - LONGS only
    """
    signals = []
    for sym, df in data.items():
        for i in range(2, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i - 1]
            date_str = row['date_str']

            if (pd.isna(prev['kc_lower']) or pd.isna(prev['sma200']) or
                    pd.isna(prev['kc_mid'])):
                continue

            # Entry condition at bar i-1
            if prev['close'] < prev['kc_lower'] and prev['close'] > prev['sma200']:
                entry = row['open']
                if entry > 0:
                    sl = entry * 0.95  # 5% SL
                    # Target = KC mid at bar i-1 (the level we'd set the sell limit at)
                    tp = prev['kc_mid']
                    if tp > entry:
                        signals.append((i, sym, date_str, TradeSignal(
                            direction=Direction.LONG,
                            entry_price=entry,
                            stop_loss=sl,
                            target=tp,
                            max_hold_bars=15,
                        )))
    return signals


def strategy_inside_day_breakout(data: Dict[str, pd.DataFrame], longs_only: bool = False) -> List[Tuple]:
    """
    Strategy 6: Inside Day Breakout
    - Bar i-1 was Inside Day: bar i-1 high < bar i-2 high AND bar i-1 low > bar i-2 low
    - Place buy stop at bar i-1 high, sell stop at bar i-1 low
    - Whichever triggers first on bar i
    - SL = opposite side of bar i-1 range, TP = 2x risk, max hold 7 days

    Priority: if both triggered in same bar, take the long (arbitrary).
    """
    signals = []
    for sym, df in data.items():
        for i in range(3, len(df)):
            row = df.iloc[i]
            date_str = row['date_str']

            if pd.isna(row['prev_high']) or pd.isna(row['prev2_high']):
                continue

            # Check if bar i-1 was inside day
            inside = (row['prev_high'] < row['prev2_high']) and (row['prev_low'] > row['prev2_low'])
            if not inside:
                continue

            buy_stop = row['prev_high']
            sell_stop = row['prev_low']

            # Check long trigger
            long_triggered = row['high'] >= buy_stop
            # Check short trigger
            short_triggered = not longs_only and row['low'] <= sell_stop

            if long_triggered:
                entry = buy_stop
                sl = sell_stop
                risk = entry - sl
                if risk > 0:
                    tp = entry + 2 * risk
                    signals.append((i, sym, date_str, TradeSignal(
                        direction=Direction.LONG,
                        entry_price=entry,
                        stop_loss=sl,
                        target=tp,
                        max_hold_bars=7,
                    )))
            elif short_triggered:
                entry = sell_stop
                sl = buy_stop
                risk = sl - entry
                if risk > 0:
                    tp = entry - 2 * risk
                    signals.append((i, sym, date_str, TradeSignal(
                        direction=Direction.SHORT,
                        entry_price=entry,
                        stop_loss=sl,
                        target=tp,
                        max_hold_bars=7,
                    )))
    return signals


# ============================================================================
# Multi-Symbol Backtest Runner
# ============================================================================

def run_multi_symbol_backtest(
    data: Dict[str, pd.DataFrame],
    signals: List[Tuple],
    commission_pct: float,
    slippage_pct: float,
) -> BacktestResult:
    """
    Run backtest across multiple symbols with shared capital.
    Signals: list of (bar_index, symbol, date_str, TradeSignal)

    Optimized: pre-builds date->row index maps, avoids DataFrame lookups in hot loop.
    """
    engine = IntradayBacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        position_size_pct=POSITION_SIZE_PCT,
        max_positions=MAX_POSITIONS,
        commission_pct=commission_pct,
        slippage_pct=slippage_pct,
        mode='cash',
        fixed_sizing=True,
    )

    if not signals:
        return engine.get_results()

    # Pre-build fast lookup: sym -> {date_str: (iloc_idx, high, low, close)}
    sym_date_data = {}
    all_dates_set = set()
    for sym, df in data.items():
        date_strs = df['date_str'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        lookup = {}
        for j in range(len(df)):
            ds = date_strs[j]
            lookup[ds] = (j, highs[j], lows[j], closes[j])
            all_dates_set.add(ds)
        sym_date_data[sym] = lookup

    all_dates = sorted(all_dates_set)

    # Index signals by (date_str, symbol) - keep only first signal per (date, sym)
    signal_map = {}
    for bar_idx, sym, date_str, sig in signals:
        key = (date_str, sym)
        if key not in signal_map:
            signal_map[key] = (bar_idx, sig)

    # Process day by day
    for date_str in all_dates:
        # First: check exits for all open positions
        for sym in list(engine.positions.keys()):
            if sym not in sym_date_data:
                continue
            lookup = sym_date_data[sym]
            if date_str not in lookup:
                continue
            iloc_idx, high, low, close = lookup[date_str]

            engine.check_exits(
                symbol=sym,
                bar_idx=iloc_idx,
                bar_date=date_str,
                high=high,
                low=low,
                close=close,
            )

        # Then: try to open new positions from signals
        for sym in sorted(data.keys()):
            key = (date_str, sym)
            if key in signal_map:
                bar_idx, sig = signal_map[key]
                engine.open_position(sym, sig, bar_idx, date_str)

        # Update equity (only if there are open positions)
        if engine.positions:
            prices = {}
            for sym in engine.positions:
                if sym in sym_date_data:
                    lookup = sym_date_data[sym]
                    if date_str in lookup:
                        prices[sym] = lookup[date_str][3]  # close
            engine.update_equity(date_str, prices)
        else:
            engine.update_equity(date_str)

    return engine.get_results()


# ============================================================================
# Build Result Row
# ============================================================================

def build_row(label: str, strategy: str, variant: str, mode: str,
              direction_filter: str, result: BacktestResult) -> dict:
    row = {
        'label': label,
        'strategy': strategy,
        'variant': variant,
        'mode': mode,
        'direction_filter': direction_filter,
        'total_trades': result.total_trades,
        'win_rate': round(result.win_rate, 2),
        'profit_factor': round(result.profit_factor, 2),
        'cagr': round(result.cagr, 2),
        'sharpe': round(result.sharpe_ratio, 2),
        'sortino': round(result.sortino_ratio, 2),
        'calmar': round(result.calmar_ratio, 2),
        'max_drawdown': round(result.max_drawdown, 2),
        'total_pnl_pct': round(result.total_pnl_pct, 2),
        'avg_win': round(result.avg_win, 2),
        'avg_loss': round(result.avg_loss, 2),
        'avg_rr': round(result.avg_rr, 2),
        'avg_bars_held': round(result.avg_bars_held, 1) if result.total_trades > 0 else 0,
        'long_trades': result.long_trades,
        'short_trades': result.short_trades,
        'long_wr': round(result.long_win_rate, 2),
        'short_wr': round(result.short_win_rate, 2),
        'exit_reasons': str(result.exit_reasons),
    }

    for y in range(2018, 2026):
        ys = result.yearly_stats.get(y, {})
        row[f'y{y}_trades'] = ys.get('trades', 0)
        row[f'y{y}_wr'] = round(ys.get('win_rate', 0), 1)
        row[f'y{y}_pnl_pct'] = round(ys.get('pnl_pct', 0), 2)

    return row


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()

    # Load done configs
    done = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV) as f:
            done = {row['label'] for row in csv.DictReader(f)}
        if done:
            print(f'Skipping {len(done)} already-completed configs')
    else:
        with open(OUTPUT_CSV, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    # Load data
    print('Loading data...', flush=True)
    data = load_data_from_db(FNO_STOCKS, 'day', '2018-01-01', '2025-12-31')
    print(f'  Loaded {len(data)} symbols in {time.time()-t0:.1f}s', flush=True)

    # Precompute indicators
    print('Computing indicators...', flush=True)
    data = precompute_indicators(data)
    print(f'  Indicators done in {time.time()-t0:.1f}s for {len(data)} symbols', flush=True)

    # Define all configs
    # Variants: (suffix, mode, commission_pct, slippage_pct, longs_only)
    variants = [
        ('Cash_LO', 'cash', 0.001, 0.001, True),      # 0.1% comm + 0.1% slip
        ('Cash_LS', 'cash', 0.001, 0.001, False),      # both directions
        ('Fut_LO', 'futures', 0.0001, 0.0005, True),   # 0.01% comm + 0.05% slip
        ('Fut_LS', 'futures', 0.0001, 0.0005, False),  # both directions
    ]

    strategies = [
        ('PA_MACD_BuyStop', strategy_pa_macd_buystop),
        ('Gap_Momentum', strategy_gap_momentum),
        ('Range_Breakout_5d', strategy_range_breakout),
        ('EMA_Pullback', strategy_ema_pullback),
        ('KC_MeanRevert', strategy_kc_mean_reversion),
        ('InsideDay_Breakout', strategy_inside_day_breakout),
    ]

    configs = []
    for strat_name, strat_fn in strategies:
        for var_suffix, mode, comm, slip, lo in variants:
            # EMA_Pullback and KC_MeanRevert are longs-only by design
            if strat_name in ('EMA_Pullback', 'KC_MeanRevert') and not lo:
                # Still run with lo=True even for LS variant (same result)
                actual_lo = True
                dir_filter = 'longs_only'
            else:
                actual_lo = lo
                dir_filter = 'longs_only' if lo else 'long_short'
            label = f'{strat_name}_{var_suffix}'
            configs.append((label, strat_name, var_suffix, mode, comm, slip, actual_lo, dir_filter, strat_fn))

    total = len(configs)
    completed = 0

    for label, strat_name, var_suffix, mode, comm, slip, lo, dir_filter, strat_fn in configs:
        if label in done:
            completed += 1
            continue

        t1 = time.time()
        print(f'[{completed+1}/{total}] {label} ...', end='', flush=True)

        # Generate signals
        signals = strat_fn(data, longs_only=lo)

        # Run backtest
        result = run_multi_symbol_backtest(data, signals, comm, slip)

        elapsed = time.time() - t1
        print(f' {elapsed:.0f}s | Trades={result.total_trades} WR={result.win_rate:.1f}% '
              f'CAGR={result.cagr:.2f}% PF={result.profit_factor:.2f} '
              f'MaxDD={result.max_drawdown:.1f}%', flush=True)

        row = build_row(label, strat_name, var_suffix, mode, dir_filter, result)

        # Write incrementally
        with open(OUTPUT_CSV, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

        completed += 1

    print(f'\nDone! {completed}/{total} configs in {time.time()-t0:.1f}s')
    print(f'Results: {OUTPUT_CSV}')


if __name__ == '__main__':
    main()
