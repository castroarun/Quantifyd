#!/usr/bin/env python3
"""
Strategy Sweep — Test Multiple Intraday Strategies
====================================================
Tests 8+ strategies on 60-min data (93 stocks, 2018-2025) for year-by-year consistency.
Target: 24%+ CAGR, <20% MaxDD, both long & short.

Usage:
    python run_strategy_sweep.py                    # Run all pending
    python run_strategy_sweep.py --strategy ORB     # Run specific strategy
    python run_strategy_sweep.py --list             # List all strategies
"""
import csv, os, sys, time, json, logging, argparse
import numpy as np
import pandas as pd

logging.disable(logging.WARNING)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from typing import Dict
from services.intraday_backtest_engine import (
    IntradayBacktestEngine, TradeSignal, Direction, ExitType,
    load_data_from_db, get_all_symbols_for_timeframe, BacktestResult,
)
from services.technical_indicators import (
    calc_ema, calc_rsi, calc_atr, calc_supertrend, calc_macd,
    calc_bollinger_bands, calc_keltner_channels, calc_donchian_channels,
    calc_adx, calc_stochastics,
    add_supertrend_signals, add_ema_signals, add_macd_signals,
    add_bollinger_signals, add_rsi_signals, add_adx_signals,
)

OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'strategy_sweep_results.csv')
FIELDNAMES = [
    'strategy', 'params', 'timeframe', 'n_symbols', 'period',
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


# ============================================================================
# STRATEGY IMPLEMENTATIONS
# ============================================================================

def prepare_indicators(df, indicators_needed):
    """Add required indicators to dataframe. Returns modified df."""
    df = df.copy()

    if 'ema' in indicators_needed:
        fast, slow = indicators_needed.get('ema_params', (9, 21))
        df['ema_fast'] = calc_ema(df['close'], fast)
        df['ema_slow'] = calc_ema(df['close'], slow)

    if 'ema200' in indicators_needed:
        df['ema200'] = calc_ema(df['close'], 200)

    if 'rsi' in indicators_needed:
        period = indicators_needed.get('rsi_period', 14)
        df['rsi'] = calc_rsi(df['close'], period)

    if 'atr' in indicators_needed:
        period = indicators_needed.get('atr_period', 14)
        df['atr'] = calc_atr(df, period)

    if 'supertrend' in indicators_needed:
        atr_p = indicators_needed.get('st_atr', 10)
        mult = indicators_needed.get('st_mult', 3.0)
        df['supertrend'], df['st_direction'] = calc_supertrend(df, atr_p, mult)

    if 'macd' in indicators_needed:
        fast_p = indicators_needed.get('macd_fast', 12)
        slow_p = indicators_needed.get('macd_slow', 26)
        sig_p = indicators_needed.get('macd_signal', 9)
        df['macd'], df['macd_signal_line'], df['macd_hist'] = calc_macd(
            df['close'], fast_p, slow_p, sig_p)

    if 'bollinger' in indicators_needed:
        period = indicators_needed.get('bb_period', 20)
        std = indicators_needed.get('bb_std', 2.0)
        df['bb_mid'], df['bb_upper'], df['bb_lower'] = calc_bollinger_bands(df, period, std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100

    if 'keltner' in indicators_needed:
        ema_p = indicators_needed.get('kc_ema', 20)
        atr_p = indicators_needed.get('kc_atr', 10)
        mult = indicators_needed.get('kc_mult', 1.5)
        df['kc_mid'], df['kc_upper'], df['kc_lower'] = calc_keltner_channels(
            df, ema_p, atr_p, mult)

    if 'donchian' in indicators_needed:
        period = indicators_needed.get('dc_period', 20)
        df['dc_mid'], df['dc_upper'], df['dc_lower'] = calc_donchian_channels(df, period)

    if 'adx' in indicators_needed:
        period = indicators_needed.get('adx_period', 14)
        df['adx'], df['plus_di'], df['minus_di'] = calc_adx(df, period)

    if 'stoch' in indicators_needed:
        k_p = indicators_needed.get('stoch_k', 14)
        d_p = indicators_needed.get('stoch_d', 3)
        df['stoch_k'], df['stoch_d'] = calc_stochastics(df, k_p, d_p)

    if 'prev_day' in indicators_needed:
        # For 60-min bars, compute previous day's OHLC
        df['trading_date'] = df.index.date
        daily = df.groupby('trading_date').agg(
            day_open=('open', 'first'),
            day_high=('high', 'max'),
            day_low=('low', 'min'),
            day_close=('close', 'last'),
        )
        daily['prev_high'] = daily['day_high'].shift(1)
        daily['prev_low'] = daily['day_low'].shift(1)
        daily['prev_close'] = daily['day_close'].shift(1)
        daily['prev_range'] = daily['prev_high'] - daily['prev_low']

        # NR4: today's range < min of last 3 ranges
        daily['range'] = daily['day_high'] - daily['day_low']
        daily['min_range_3'] = daily['range'].rolling(3).min().shift(1)
        daily['is_nr4'] = daily['range'] < daily['min_range_3']

        df = df.join(daily[['prev_high', 'prev_low', 'prev_close', 'prev_range',
                            'is_nr4']], on='trading_date')

    return df


# --- Strategy 1: EMA Crossover + MACD Confirmation ---
def strategy_ema_macd(df, i, params):
    """
    LONG: EMA fast crosses above slow, MACD histogram positive
    SHORT: EMA fast crosses below slow, MACD histogram negative
    Exit: ATR trailing stop
    """
    if i < 2:
        return None

    row = df.iloc[i]
    prev = df.iloc[i - 1]
    atr = row.get('atr', 0)
    if atr <= 0 or pd.isna(atr):
        return None

    sl_mult = params.get('sl_atr_mult', 2.0)
    tp_mult = params.get('tp_atr_mult', 4.0)
    trail_mult = params.get('trail_atr_mult', 2.5)

    # EMA crossover
    ema_bull_now = row['ema_fast'] > row['ema_slow']
    ema_bull_prev = prev['ema_fast'] > prev['ema_slow']

    # MACD confirmation
    macd_pos = row['macd_hist'] > 0
    macd_neg = row['macd_hist'] < 0

    if ema_bull_now and not ema_bull_prev and macd_pos:
        # LONG signal
        entry = row['close']
        sl = entry - sl_mult * atr
        tp = entry + tp_mult * atr
        return TradeSignal(Direction.LONG, entry, sl, tp, trail_mult)

    if not ema_bull_now and ema_bull_prev and macd_neg:
        # SHORT signal
        entry = row['close']
        sl = entry + sl_mult * atr
        tp = entry - tp_mult * atr
        return TradeSignal(Direction.SHORT, entry, sl, tp, trail_mult)

    return None


# --- Strategy 2: SuperTrend Momentum ---
def strategy_supertrend(df, i, params):
    """
    LONG: SuperTrend flips to uptrend
    SHORT: SuperTrend flips to downtrend
    Exit: SuperTrend flip or trailing ATR
    """
    if i < 2:
        return None

    row = df.iloc[i]
    prev = df.iloc[i - 1]
    atr = row.get('atr', 0)
    if atr <= 0 or pd.isna(atr):
        return None

    sl_mult = params.get('sl_atr_mult', 2.0)
    tp_mult = params.get('tp_atr_mult', 0)  # 0 = no fixed target

    st_now = row['st_direction']
    st_prev = prev['st_direction']

    if st_now == 1 and st_prev == -1:
        entry = row['close']
        sl = entry - sl_mult * atr
        tp = entry + tp_mult * atr if tp_mult > 0 else None
        return TradeSignal(Direction.LONG, entry, sl, tp)

    if st_now == -1 and st_prev == 1:
        entry = row['close']
        sl = entry + sl_mult * atr
        tp = entry - tp_mult * atr if tp_mult > 0 else None
        return TradeSignal(Direction.SHORT, entry, sl, tp)

    return None


# --- Strategy 3: RSI Mean Reversion ---
def strategy_rsi_reversion(df, i, params):
    """
    LONG: RSI crosses above oversold (30) from below, close > EMA200
    SHORT: RSI crosses below overbought (70) from above, close < EMA200
    Exit: Fixed SL/TP (risk:reward based)
    """
    if i < 2:
        return None

    row = df.iloc[i]
    prev = df.iloc[i - 1]
    atr = row.get('atr', 0)
    if atr <= 0 or pd.isna(atr):
        return None

    oversold = params.get('rsi_oversold', 30)
    overbought = params.get('rsi_overbought', 70)
    sl_mult = params.get('sl_atr_mult', 1.5)
    rr = params.get('risk_reward', 2.0)

    rsi_now = row['rsi']
    rsi_prev = prev['rsi']

    # Long: RSI was below oversold, now above
    if rsi_prev < oversold and rsi_now >= oversold:
        if 'ema200' in df.columns and row['close'] > row['ema200']:
            entry = row['close']
            sl = entry - sl_mult * atr
            risk = entry - sl
            tp = entry + rr * risk
            return TradeSignal(Direction.LONG, entry, sl, tp)

    # Short: RSI was above overbought, now below
    if rsi_prev > overbought and rsi_now <= overbought:
        if 'ema200' in df.columns and row['close'] < row['ema200']:
            entry = row['close']
            sl = entry + sl_mult * atr
            risk = sl - entry
            tp = entry - rr * risk
            return TradeSignal(Direction.SHORT, entry, sl, tp)

    return None


# --- Strategy 4: Bollinger Squeeze Breakout ---
def strategy_bb_squeeze(df, i, params):
    """
    Setup: BB width contracts (squeeze) then expands
    LONG: Close breaks above upper BB after squeeze, MACD positive
    SHORT: Close breaks below lower BB after squeeze, MACD negative
    Exit: ATR trailing stop
    """
    if i < 5:
        return None

    row = df.iloc[i]
    prev = df.iloc[i - 1]
    atr = row.get('atr', 0)
    if atr <= 0 or pd.isna(atr):
        return None

    trail_mult = params.get('trail_atr_mult', 2.5)
    sl_mult = params.get('sl_atr_mult', 2.0)
    squeeze_lookback = params.get('squeeze_lookback', 5)

    # Check if BB was inside KC (squeeze) in recent bars
    in_squeeze = False
    if 'kc_upper' in df.columns:
        for j in range(max(0, i - squeeze_lookback), i):
            r = df.iloc[j]
            if r['bb_upper'] < r['kc_upper'] and r['bb_lower'] > r['kc_lower']:
                in_squeeze = True
                break

    if not in_squeeze:
        return None

    # Now check for breakout
    macd_pos = row.get('macd_hist', 0) > 0
    macd_neg = row.get('macd_hist', 0) < 0

    if row['close'] > row['bb_upper'] and prev['close'] <= prev['bb_upper'] and macd_pos:
        entry = row['close']
        sl = entry - sl_mult * atr
        return TradeSignal(Direction.LONG, entry, sl, None, trail_mult)

    if row['close'] < row['bb_lower'] and prev['close'] >= prev['bb_lower'] and macd_neg:
        entry = row['close']
        sl = entry + sl_mult * atr
        return TradeSignal(Direction.SHORT, entry, sl, None, trail_mult)

    return None


# --- Strategy 5: Donchian Breakout (Turtle) ---
def strategy_donchian(df, i, params):
    """
    LONG: Close breaks above 20-period Donchian upper
    SHORT: Close breaks below 20-period Donchian lower
    Exit: ATR trailing or opposite Donchian (10-period for exit)
    """
    if i < 2:
        return None

    row = df.iloc[i]
    prev = df.iloc[i - 1]
    atr = row.get('atr', 0)
    if atr <= 0 or pd.isna(atr):
        return None

    sl_mult = params.get('sl_atr_mult', 2.0)
    trail_mult = params.get('trail_atr_mult', 2.0)

    # Breakout above upper channel
    if row['close'] > prev['dc_upper'] and prev['close'] <= prev['dc_upper']:
        entry = row['close']
        sl = entry - sl_mult * atr
        return TradeSignal(Direction.LONG, entry, sl, None, trail_mult)

    # Breakdown below lower channel
    if row['close'] < prev['dc_lower'] and prev['close'] >= prev['dc_lower']:
        entry = row['close']
        sl = entry + sl_mult * atr
        return TradeSignal(Direction.SHORT, entry, sl, None, trail_mult)

    return None


# --- Strategy 6: Price Action + MACD (User's example pattern) ---
def strategy_price_action_macd(df, i, params):
    """
    LONG: Green candle closes above previous red candle's high, MACD positive
          Entry: above the high, SL: below red candle low, Target: 2x risk
    SHORT: Red candle closes below previous green candle's low, MACD negative
           Entry: below the low, SL: above green candle high, Target: 2x risk
    """
    if i < 2:
        return None

    row = df.iloc[i]
    prev = df.iloc[i - 1]
    rr = params.get('risk_reward', 2.0)

    is_green = row['close'] > row['open']
    is_red = row['close'] < row['open']
    prev_green = prev['close'] > prev['open']
    prev_red = prev['close'] < prev['open']

    macd_pos = row.get('macd_hist', 0) > 0
    macd_neg = row.get('macd_hist', 0) < 0

    # LONG: green candle closes above prev red candle's high
    if is_green and prev_red and row['close'] > prev['high'] and macd_pos:
        entry = prev['high']  # Entry at prev high
        sl = prev['low']  # SL at prev low
        risk = entry - sl
        if risk <= 0:
            return None
        tp = entry + rr * risk
        return TradeSignal(Direction.LONG, entry, sl, tp)

    # SHORT: red candle closes below prev green candle's low
    if is_red and prev_green and row['close'] < prev['low'] and macd_neg:
        entry = prev['low']
        sl = prev['high']
        risk = sl - entry
        if risk <= 0:
            return None
        tp = entry - rr * risk
        return TradeSignal(Direction.SHORT, entry, sl, tp)

    return None


# --- Strategy 7: ADX Trend + EMA ---
def strategy_adx_trend(df, i, params):
    """
    LONG: ADX > 25 (trending), +DI > -DI (uptrend), close > EMA21
    SHORT: ADX > 25, -DI > +DI (downtrend), close < EMA21
    Trigger: EMA fast cross as entry timing
    Exit: ATR trailing
    """
    if i < 2:
        return None

    row = df.iloc[i]
    prev = df.iloc[i - 1]
    atr = row.get('atr', 0)
    if atr <= 0 or pd.isna(atr):
        return None

    adx_thresh = params.get('adx_threshold', 25)
    sl_mult = params.get('sl_atr_mult', 2.0)
    trail_mult = params.get('trail_atr_mult', 2.5)

    if row['adx'] < adx_thresh:
        return None

    ema_bull_now = row['ema_fast'] > row['ema_slow']
    ema_bull_prev = prev['ema_fast'] > prev['ema_slow']

    # LONG: ADX trending + uptrend + EMA cross up
    if row['plus_di'] > row['minus_di'] and ema_bull_now and not ema_bull_prev:
        entry = row['close']
        sl = entry - sl_mult * atr
        return TradeSignal(Direction.LONG, entry, sl, None, trail_mult)

    # SHORT: ADX trending + downtrend + EMA cross down
    if row['minus_di'] > row['plus_di'] and not ema_bull_now and ema_bull_prev:
        entry = row['close']
        sl = entry + sl_mult * atr
        return TradeSignal(Direction.SHORT, entry, sl, None, trail_mult)

    return None


# --- Strategy 8: Heikin-Ashi Trend ---
def strategy_heikin_ashi(df, i, params):
    """
    Convert to Heikin-Ashi candles.
    LONG: First green HA after 2+ red HA candles, RSI > 40
    SHORT: First red HA after 2+ green HA candles, RSI < 60
    Exit: HA color change or ATR trail
    """
    if i < 4:
        return None

    atr = df.iloc[i].get('atr', 0)
    if atr <= 0 or pd.isna(atr):
        return None

    sl_mult = params.get('sl_atr_mult', 2.0)
    trail_mult = params.get('trail_atr_mult', 2.0)
    min_streak = params.get('min_streak', 2)

    # Compute HA candles
    row = df.iloc[i]
    ha_close = (row['open'] + row['high'] + row['low'] + row['close']) / 4

    # Get previous HA values (simplified — use regular OHLC for HA open approximation)
    prev = df.iloc[i - 1]
    ha_open_prev = (prev['open'] + prev['close']) / 2
    ha_open = (ha_open_prev + ha_close) / 2

    ha_green_now = ha_close > ha_open

    # Check streak of opposite color
    streak = 0
    for j in range(i - 1, max(i - 6, 0), -1):
        r = df.iloc[j]
        ha_c = (r['open'] + r['high'] + r['low'] + r['close']) / 4
        r_prev = df.iloc[j - 1]
        ha_o = ((r_prev['open'] + r_prev['close']) / 2 + ha_c) / 2
        if ha_green_now and ha_c < ha_o:
            streak += 1
        elif not ha_green_now and ha_c > ha_o:
            streak += 1
        else:
            break

    if streak < min_streak:
        return None

    rsi = row.get('rsi', 50)
    entry = row['close']

    if ha_green_now and rsi > 40:
        sl = entry - sl_mult * atr
        return TradeSignal(Direction.LONG, entry, sl, None, trail_mult)

    if not ha_green_now and rsi < 60:
        sl = entry + sl_mult * atr
        return TradeSignal(Direction.SHORT, entry, sl, None, trail_mult)

    return None


# ============================================================================
# STRATEGY REGISTRY
# ============================================================================

STRATEGIES = {
    'EMA_MACD': {
        'fn': strategy_ema_macd,
        'indicators': {
            'ema': True, 'ema_params': (9, 21),
            'macd': True,
            'atr': True, 'atr_period': 14,
        },
        'variants': [
            {'label': 'EMA9_21_MACD_SL2_TP4_TR2.5', 'sl_atr_mult': 2.0, 'tp_atr_mult': 4.0, 'trail_atr_mult': 2.5},
            {'label': 'EMA9_21_MACD_SL1.5_TP3_TR2', 'sl_atr_mult': 1.5, 'tp_atr_mult': 3.0, 'trail_atr_mult': 2.0},
            {'label': 'EMA5_13_MACD_SL2_TP4_TR2.5', 'sl_atr_mult': 2.0, 'tp_atr_mult': 4.0, 'trail_atr_mult': 2.5,
             '_ema_params': (5, 13)},
        ],
    },
    'SUPERTREND': {
        'fn': strategy_supertrend,
        'indicators': {
            'supertrend': True, 'st_atr': 10, 'st_mult': 3.0,
            'atr': True, 'atr_period': 14,
        },
        'variants': [
            {'label': 'ST10_M3_SL2', 'sl_atr_mult': 2.0},
            {'label': 'ST7_M3.5_SL2', 'sl_atr_mult': 2.0, '_st_atr': 7, '_st_mult': 3.5},
            {'label': 'ST7_M2.5_SL1.5', 'sl_atr_mult': 1.5, '_st_atr': 7, '_st_mult': 2.5},
            {'label': 'ST14_M3_SL2.5', 'sl_atr_mult': 2.5, '_st_atr': 14, '_st_mult': 3.0},
        ],
    },
    'RSI_REVERSION': {
        'fn': strategy_rsi_reversion,
        'indicators': {
            'rsi': True, 'rsi_period': 14,
            'ema200': True,
            'atr': True, 'atr_period': 14,
        },
        'variants': [
            {'label': 'RSI30_70_RR2_SL1.5', 'rsi_oversold': 30, 'rsi_overbought': 70, 'risk_reward': 2.0, 'sl_atr_mult': 1.5},
            {'label': 'RSI25_75_RR2.5_SL1.5', 'rsi_oversold': 25, 'rsi_overbought': 75, 'risk_reward': 2.5, 'sl_atr_mult': 1.5},
            {'label': 'RSI20_80_RR3_SL2', 'rsi_oversold': 20, 'rsi_overbought': 80, 'risk_reward': 3.0, 'sl_atr_mult': 2.0},
        ],
    },
    'BB_SQUEEZE': {
        'fn': strategy_bb_squeeze,
        'indicators': {
            'bollinger': True, 'bb_period': 20, 'bb_std': 2.0,
            'keltner': True, 'kc_ema': 20, 'kc_atr': 10, 'kc_mult': 1.5,
            'macd': True,
            'atr': True, 'atr_period': 14,
        },
        'variants': [
            {'label': 'BBSQ_SL2_TR2.5', 'sl_atr_mult': 2.0, 'trail_atr_mult': 2.5, 'squeeze_lookback': 5},
            {'label': 'BBSQ_SL1.5_TR2', 'sl_atr_mult': 1.5, 'trail_atr_mult': 2.0, 'squeeze_lookback': 3},
        ],
    },
    'DONCHIAN': {
        'fn': strategy_donchian,
        'indicators': {
            'donchian': True, 'dc_period': 20,
            'atr': True, 'atr_period': 14,
        },
        'variants': [
            {'label': 'DC20_SL2_TR2', 'sl_atr_mult': 2.0, 'trail_atr_mult': 2.0},
            {'label': 'DC10_SL1.5_TR1.5', 'sl_atr_mult': 1.5, 'trail_atr_mult': 1.5, '_dc_period': 10},
            {'label': 'DC30_SL2.5_TR2.5', 'sl_atr_mult': 2.5, 'trail_atr_mult': 2.5, '_dc_period': 30},
        ],
    },
    'PRICE_ACTION_MACD': {
        'fn': strategy_price_action_macd,
        'indicators': {
            'macd': True,
        },
        'variants': [
            {'label': 'PA_MACD_RR2', 'risk_reward': 2.0},
            {'label': 'PA_MACD_RR3', 'risk_reward': 3.0},
            {'label': 'PA_MACD_RR1.5', 'risk_reward': 1.5},
        ],
    },
    'ADX_TREND': {
        'fn': strategy_adx_trend,
        'indicators': {
            'adx': True, 'adx_period': 14,
            'ema': True, 'ema_params': (9, 21),
            'atr': True, 'atr_period': 14,
        },
        'variants': [
            {'label': 'ADX25_SL2_TR2.5', 'adx_threshold': 25, 'sl_atr_mult': 2.0, 'trail_atr_mult': 2.5},
            {'label': 'ADX30_SL2_TR2.5', 'adx_threshold': 30, 'sl_atr_mult': 2.0, 'trail_atr_mult': 2.5},
            {'label': 'ADX20_SL1.5_TR2', 'adx_threshold': 20, 'sl_atr_mult': 1.5, 'trail_atr_mult': 2.0},
        ],
    },
    'HEIKIN_ASHI': {
        'fn': strategy_heikin_ashi,
        'indicators': {
            'rsi': True, 'rsi_period': 14,
            'atr': True, 'atr_period': 14,
        },
        'variants': [
            {'label': 'HA_STREAK2_SL2_TR2', 'min_streak': 2, 'sl_atr_mult': 2.0, 'trail_atr_mult': 2.0},
            {'label': 'HA_STREAK3_SL2_TR2.5', 'min_streak': 3, 'sl_atr_mult': 2.0, 'trail_atr_mult': 2.5},
        ],
    },
}


# ============================================================================
# BACKTEST RUNNER
# ============================================================================

def run_single_strategy(
    strategy_name: str,
    strategy_fn,
    params: dict,
    data: Dict[str, pd.DataFrame],
    indicators: dict,
    initial_capital: float = 10_000_000,
    max_positions: int = 10,
    timeframe: str = '60minute',
) -> BacktestResult:
    """Run a single strategy across all symbols — process per-symbol for speed."""

    engine = IntradayBacktestEngine(
        initial_capital=initial_capital,
        position_size_pct=1.0 / max_positions,
        max_positions=max_positions,
        commission_pct=0.0005,
        slippage_pct=0.0005,
        fixed_sizing=True,
    )

    # Pre-compute indicators for each symbol
    prepared_data = {}
    for symbol, df in data.items():
        try:
            ind = dict(indicators)
            for k, v in params.items():
                if k.startswith('_'):
                    ind[k[1:]] = v
            prepared_data[symbol] = prepare_indicators(df, ind)
        except Exception:
            continue

    if not prepared_data:
        return BacktestResult()

    # Process each symbol independently through all bars
    # This is MUCH faster than day-by-day interleaving
    for symbol, df in prepared_data.items():
        dates = df['date_str'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        opens = df['open'].values

        atr_vals = df['atr'].values if 'atr' in df.columns else [None] * len(df)
        st_dirs = df['st_direction'].values if 'st_direction' in df.columns else [None] * len(df)

        prev_date = None
        bars_in_day = 0
        day_bar_count = 0

        # Pre-count bars per day for EOD detection
        date_counts = {}
        for d in dates:
            date_counts[d] = date_counts.get(d, 0) + 1

        bar_in_day_counter = {}

        for i in range(len(df)):
            date_str = dates[i]
            bar_in_day_counter[date_str] = bar_in_day_counter.get(date_str, 0) + 1
            is_last_bar = (bar_in_day_counter[date_str] == date_counts[date_str])

            # Check exits first
            if symbol in engine.positions:
                atr_v = atr_vals[i]
                st_v = st_dirs[i]

                engine.check_exits(
                    symbol=symbol,
                    bar_idx=i,
                    bar_date=str(df.index[i]),
                    high=highs[i],
                    low=lows[i],
                    close=closes[i],
                    atr=float(atr_v) if atr_v is not None and not np.isnan(atr_v) else None,
                    supertrend_dir=int(st_v) if st_v is not None and not np.isnan(st_v) else None,
                    is_eod=is_last_bar,
                )

            # Generate signals (not on last bar of day)
            if not is_last_bar and symbol not in engine.positions:
                signal = strategy_fn(df, i, params)
                if signal:
                    engine.open_position(symbol, signal, i, str(df.index[i]))

        # Update equity using last close of each trading date for this symbol
        last_closes = df.groupby('date_str')['close'].last()
        for date_str, close_price in last_closes.items():
            engine.update_equity(date_str, {symbol: close_price})

    return engine.get_results()


def load_done():
    done = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, newline='') as f:
            done = {row['strategy'] for row in csv.DictReader(f)}
    return done


def ensure_header():
    if not os.path.exists(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0:
        with open(OUTPUT_CSV, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def write_result(strategy_label, params_str, result, timeframe, n_symbols, period):
    row = {
        'strategy': strategy_label,
        'params': params_str,
        'timeframe': timeframe,
        'n_symbols': n_symbols,
        'period': period,
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

    # Year-by-year
    for year in range(2018, 2026):
        ys = result.yearly_stats.get(year, {})
        row[f'y{year}_trades'] = ys.get('trades', 0)
        row[f'y{year}_wr'] = round(ys.get('win_rate', 0), 2)
        row[f'y{year}_pf'] = round(ys.get('profit_factor', 0), 4)
        row[f'y{year}_pnl_pct'] = round(ys.get('pnl_pct', 0), 2)

    with open(OUTPUT_CSV, 'a', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', type=str, help='Run specific strategy')
    parser.add_argument('--list', action='store_true', help='List strategies')
    parser.add_argument('--symbols', type=int, default=0, help='Limit symbols (0=all)')
    args = parser.parse_args()

    if args.list:
        for name, cfg in STRATEGIES.items():
            print(f'{name}: {len(cfg["variants"])} variants')
            for v in cfg['variants']:
                print(f'  - {v["label"]}')
        return

    # Build config list
    configs = []
    for strat_name, strat_cfg in STRATEGIES.items():
        if args.strategy and strat_name != args.strategy:
            continue
        for variant in strat_cfg['variants']:
            label = f'{strat_name}_{variant["label"]}'
            configs.append((label, strat_name, variant))

    ensure_header()
    done = load_done()
    pending = [(l, s, v) for l, s, v in configs if l not in done]

    total_variants = sum(len(s['variants']) for s in STRATEGIES.values())
    print(f'=== Strategy Sweep ===')
    print(f'Total strategies: {len(STRATEGIES)} | Variants: {total_variants}')
    print(f'Done: {len(done)} | Pending: {len(pending)}')

    if not pending:
        print('All strategies completed!')
        return

    # Load data once
    timeframe = '60minute'
    print(f'\nLoading {timeframe} data...', flush=True)
    t0 = time.time()

    symbols = get_all_symbols_for_timeframe(timeframe, min_rows=5000)
    if args.symbols > 0:
        symbols = symbols[:args.symbols]

    data = load_data_from_db(symbols, timeframe, '2018-01-01', '2025-11-30')
    loaded_symbols = list(data.keys())
    print(f'Loaded {len(loaded_symbols)} symbols in {time.time()-t0:.0f}s', flush=True)

    # Run each pending config
    for i, (label, strat_name, variant) in enumerate(pending, 1):
        strat_cfg = STRATEGIES[strat_name]
        print(f'\n[{i}/{len(pending)}] {label}', flush=True)
        t1 = time.time()

        try:
            result = run_single_strategy(
                strategy_name=label,
                strategy_fn=strat_cfg['fn'],
                params=variant,
                data=data,
                indicators=strat_cfg['indicators'],
            )

            elapsed = time.time() - t1
            print(f'  {elapsed:.0f}s | Trades={result.total_trades} '
                  f'WR={result.win_rate:.1f}% PF={result.profit_factor:.2f} '
                  f'CAGR={result.cagr:.1f}% MaxDD={result.max_drawdown:.1f}%',
                  flush=True)

            # Print year-by-year
            if result.yearly_stats:
                years_str = ' | '.join(
                    f'{y}:{s["trades"]}t/{s["win_rate"]:.0f}%/{s["pnl_pct"]:+.1f}%'
                    for y, s in sorted(result.yearly_stats.items())
                )
                print(f'  Years: {years_str}', flush=True)

            write_result(label, json.dumps(variant), result, timeframe,
                        len(loaded_symbols), '2018-01 to 2025-11')

            IntradayBacktestEngine.print_results(result, label)

        except Exception as e:
            import traceback
            print(f'  ERROR: {e}', flush=True)
            traceback.print_exc()

    print(f'\nResults: {OUTPUT_CSV}')


if __name__ == '__main__':
    main()
