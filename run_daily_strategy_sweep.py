#!/usr/bin/env python3
"""
Daily Strategy Sweep — Swing Trading Systems
=============================================
Tests multiple swing strategies on DAILY data (2005-2025, Nifty 500).
Holding period: 1-20 days. Both long & short on futures-eligible stocks.
Target: 24%+ CAGR, <20% MaxDD.

Usage:
    python run_daily_strategy_sweep.py              # Run all pending
    python run_daily_strategy_sweep.py --list       # List strategies
    python run_daily_strategy_sweep.py --symbols 50 # Limit symbols
"""
import csv, os, sys, time, json, logging, argparse
import numpy as np
import pandas as pd
from typing import Dict, List

logging.disable(logging.WARNING)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from services.intraday_backtest_engine import (
    IntradayBacktestEngine, TradeSignal, Direction, ExitType,
    load_data_from_db, get_all_symbols_for_timeframe, BacktestResult,
)
from services.technical_indicators import (
    calc_ema, calc_rsi, calc_atr, calc_supertrend, calc_macd,
    calc_bollinger_bands, calc_keltner_channels, calc_donchian_channels,
    calc_adx, calc_stochastics,
)

OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'daily_strategy_sweep_results.csv')
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


def prepare_daily_indicators(df, indicators_needed):
    """Add required indicators to daily dataframe."""
    df = df.copy()
    df['date_str'] = df.index.strftime('%Y-%m-%d')

    if 'ema' in indicators_needed:
        fast, slow = indicators_needed.get('ema_params', (9, 21))
        df['ema_fast'] = calc_ema(df['close'], fast)
        df['ema_slow'] = calc_ema(df['close'], slow)

    if 'ema200' in indicators_needed:
        df['ema200'] = calc_ema(df['close'], 200)

    if 'sma200' in indicators_needed:
        df['sma200'] = df['close'].rolling(200).mean()

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
        df['macd'], df['macd_signal_line'], df['macd_hist'] = calc_macd(df['close'])

    if 'bollinger' in indicators_needed:
        period = indicators_needed.get('bb_period', 20)
        std = indicators_needed.get('bb_std', 2.0)
        df['bb_mid'], df['bb_upper'], df['bb_lower'] = calc_bollinger_bands(df, period, std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100
        # BB width percentile for squeeze detection
        df['bb_width_pctile'] = df['bb_width'].rolling(120).rank(pct=True)

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
        df['stoch_k'], df['stoch_d'] = calc_stochastics(df)

    if 'nr' in indicators_needed:
        df['range'] = df['high'] - df['low']
        df['min_range_3'] = df['range'].rolling(3).min().shift(1)
        df['min_range_6'] = df['range'].rolling(6).min().shift(1)
        df['is_nr4'] = df['range'] < df['min_range_3']
        df['is_nr7'] = df['range'] < df['min_range_6']
        df['prev_high'] = df['high'].shift(1)
        df['prev_low'] = df['low'].shift(1)

    return df


# ============================================================================
# DAILY/SWING STRATEGIES
# ============================================================================

def strategy_supertrend_daily(df, i, params):
    """SuperTrend on daily. Enter on flip, trail with SuperTrend."""
    if i < 2:
        return None
    row = df.iloc[i]
    prev = df.iloc[i - 1]
    atr = row.get('atr', 0)
    if atr <= 0 or pd.isna(atr):
        return None

    sl_mult = params.get('sl_atr_mult', 2.0)
    max_hold = params.get('max_hold', 20)

    st_now = row['st_direction']
    st_prev = prev['st_direction']

    if st_now == 1 and st_prev == -1:
        entry = row['close']
        sl = entry - sl_mult * atr
        return TradeSignal(Direction.LONG, entry, sl, None, params.get('trail_atr_mult'), max_hold)

    if st_now == -1 and st_prev == 1:
        entry = row['close']
        sl = entry + sl_mult * atr
        return TradeSignal(Direction.SHORT, entry, sl, None, params.get('trail_atr_mult'), max_hold)

    return None


def strategy_ema_crossover_daily(df, i, params):
    """EMA crossover with ADX trending filter."""
    if i < 2:
        return None
    row = df.iloc[i]
    prev = df.iloc[i - 1]
    atr = row.get('atr', 0)
    if atr <= 0 or pd.isna(atr):
        return None

    adx_thresh = params.get('adx_threshold', 20)
    sl_mult = params.get('sl_atr_mult', 2.0)
    trail_mult = params.get('trail_atr_mult', 2.5)
    max_hold = params.get('max_hold', 15)

    if 'adx' in df.columns and row['adx'] < adx_thresh:
        return None

    ema_bull = row['ema_fast'] > row['ema_slow']
    prev_bull = prev['ema_fast'] > prev['ema_slow']

    if ema_bull and not prev_bull:
        entry = row['close']
        sl = entry - sl_mult * atr
        return TradeSignal(Direction.LONG, entry, sl, None, trail_mult, max_hold)

    if not ema_bull and prev_bull:
        entry = row['close']
        sl = entry + sl_mult * atr
        return TradeSignal(Direction.SHORT, entry, sl, None, trail_mult, max_hold)

    return None


def strategy_rsi_mean_reversion_daily(df, i, params):
    """RSI mean reversion on daily. Enter on oversold/overbought reversal."""
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
    max_hold = params.get('max_hold', 10)

    # Long: RSI crosses above oversold
    if prev['rsi'] < oversold and row['rsi'] >= oversold:
        entry = row['close']
        sl = entry - sl_mult * atr
        risk = entry - sl
        tp = entry + rr * risk
        return TradeSignal(Direction.LONG, entry, sl, tp, None, max_hold)

    # Short: RSI crosses below overbought
    if prev['rsi'] > overbought and row['rsi'] <= overbought:
        entry = row['close']
        sl = entry + sl_mult * atr
        risk = sl - entry
        tp = entry - rr * risk
        return TradeSignal(Direction.SHORT, entry, sl, tp, None, max_hold)

    return None


def strategy_bb_squeeze_daily(df, i, params):
    """Bollinger Squeeze on daily — enter when BB expands after contraction."""
    if i < 5:
        return None
    row = df.iloc[i]
    prev = df.iloc[i - 1]
    atr = row.get('atr', 0)
    if atr <= 0 or pd.isna(atr):
        return None

    sl_mult = params.get('sl_atr_mult', 2.0)
    trail_mult = params.get('trail_atr_mult', 2.5)
    max_hold = params.get('max_hold', 15)

    # Check for squeeze release: BB was inside KC, now expanding
    was_squeezed = False
    for j in range(max(0, i - 5), i):
        r = df.iloc[j]
        if 'kc_upper' in df.columns:
            if r['bb_upper'] < r['kc_upper'] and r['bb_lower'] > r['kc_lower']:
                was_squeezed = True
                break

    if not was_squeezed:
        return None

    # Direction from MACD histogram
    macd_pos = row.get('macd_hist', 0) > 0
    macd_neg = row.get('macd_hist', 0) < 0

    if row['close'] > row['bb_upper'] and macd_pos:
        entry = row['close']
        sl = entry - sl_mult * atr
        return TradeSignal(Direction.LONG, entry, sl, None, trail_mult, max_hold)

    if row['close'] < row['bb_lower'] and macd_neg:
        entry = row['close']
        sl = entry + sl_mult * atr
        return TradeSignal(Direction.SHORT, entry, sl, None, trail_mult, max_hold)

    return None


def strategy_donchian_daily(df, i, params):
    """Donchian channel breakout on daily."""
    if i < 2:
        return None
    row = df.iloc[i]
    prev = df.iloc[i - 1]
    atr = row.get('atr', 0)
    if atr <= 0 or pd.isna(atr):
        return None

    sl_mult = params.get('sl_atr_mult', 2.0)
    trail_mult = params.get('trail_atr_mult', 2.0)
    max_hold = params.get('max_hold', 20)

    if row['close'] > prev['dc_upper'] and prev['close'] <= prev['dc_upper']:
        entry = row['close']
        sl = entry - sl_mult * atr
        return TradeSignal(Direction.LONG, entry, sl, None, trail_mult, max_hold)

    if row['close'] < prev['dc_lower'] and prev['close'] >= prev['dc_lower']:
        entry = row['close']
        sl = entry + sl_mult * atr
        return TradeSignal(Direction.SHORT, entry, sl, None, trail_mult, max_hold)

    return None


def strategy_nr_breakout_daily(df, i, params):
    """NR4/NR7 narrow range breakout. Enter on breakout of previous range."""
    if i < 2:
        return None
    row = df.iloc[i]
    atr = row.get('atr', 0)
    if atr <= 0 or pd.isna(atr):
        return None

    use_nr7 = params.get('use_nr7', False)
    rr = params.get('risk_reward', 2.0)
    max_hold = params.get('max_hold', 5)

    is_nr = row.get('is_nr7' if use_nr7 else 'is_nr4', False)
    if not is_nr:
        return None

    prev_high = row.get('prev_high', None)
    prev_low = row.get('prev_low', None)
    if prev_high is None or prev_low is None or pd.isna(prev_high):
        return None

    # Breakout above previous high
    if row['close'] > prev_high:
        entry = row['close']
        sl = prev_low
        risk = entry - sl
        if risk <= 0:
            return None
        tp = entry + rr * risk
        return TradeSignal(Direction.LONG, entry, sl, tp, None, max_hold)

    # Breakdown below previous low
    if row['close'] < prev_low:
        entry = row['close']
        sl = prev_high
        risk = sl - entry
        if risk <= 0:
            return None
        tp = entry - rr * risk
        return TradeSignal(Direction.SHORT, entry, sl, tp, None, max_hold)

    return None


def strategy_price_action_macd_daily(df, i, params):
    """Green candle above prev red high / Red candle below prev green low."""
    if i < 2:
        return None
    row = df.iloc[i]
    prev = df.iloc[i - 1]
    atr = row.get('atr', 0)
    rr = params.get('risk_reward', 2.0)
    max_hold = params.get('max_hold', 10)

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

    if is_red and prev_green and row['close'] < prev['low'] and macd_neg:
        entry = prev['low']
        sl = prev['high']
        risk = sl - entry
        if risk <= 0:
            return None
        tp = entry - rr * risk
        return TradeSignal(Direction.SHORT, entry, sl, tp, None, max_hold)

    return None


def strategy_macd_histogram_daily(df, i, params):
    """MACD histogram reversal — enter when histogram changes direction."""
    if i < 3:
        return None
    row = df.iloc[i]
    prev = df.iloc[i - 1]
    prev2 = df.iloc[i - 2]
    atr = row.get('atr', 0)
    if atr <= 0 or pd.isna(atr):
        return None

    sl_mult = params.get('sl_atr_mult', 2.0)
    trail_mult = params.get('trail_atr_mult', 2.5)
    max_hold = params.get('max_hold', 15)

    hist = row['macd_hist']
    hist_prev = prev['macd_hist']
    hist_prev2 = prev2['macd_hist']

    # Long: histogram was declining (getting more negative), now turns up
    if hist > hist_prev and hist_prev < hist_prev2 and hist_prev < 0:
        entry = row['close']
        sl = entry - sl_mult * atr
        return TradeSignal(Direction.LONG, entry, sl, None, trail_mult, max_hold)

    # Short: histogram was rising (getting more positive), now turns down
    if hist < hist_prev and hist_prev > hist_prev2 and hist_prev > 0:
        entry = row['close']
        sl = entry + sl_mult * atr
        return TradeSignal(Direction.SHORT, entry, sl, None, trail_mult, max_hold)

    return None


def strategy_stoch_rsi_daily(df, i, params):
    """Stochastic + RSI combined mean reversion."""
    if i < 2:
        return None
    row = df.iloc[i]
    prev = df.iloc[i - 1]
    atr = row.get('atr', 0)
    if atr <= 0 or pd.isna(atr):
        return None

    sl_mult = params.get('sl_atr_mult', 1.5)
    rr = params.get('risk_reward', 2.0)
    max_hold = params.get('max_hold', 10)

    # Long: Stoch K crosses above D from oversold + RSI > 30
    stoch_cross_up = (row['stoch_k'] > row['stoch_d'] and
                      prev['stoch_k'] <= prev['stoch_d'] and
                      row['stoch_k'] < 40)
    if stoch_cross_up and row['rsi'] > 30 and row['rsi'] < 50:
        entry = row['close']
        sl = entry - sl_mult * atr
        risk = entry - sl
        tp = entry + rr * risk
        return TradeSignal(Direction.LONG, entry, sl, tp, None, max_hold)

    # Short: Stoch K crosses below D from overbought + RSI < 70
    stoch_cross_down = (row['stoch_k'] < row['stoch_d'] and
                        prev['stoch_k'] >= prev['stoch_d'] and
                        row['stoch_k'] > 60)
    if stoch_cross_down and row['rsi'] < 70 and row['rsi'] > 50:
        entry = row['close']
        sl = entry + sl_mult * atr
        risk = sl - entry
        tp = entry - rr * risk
        return TradeSignal(Direction.SHORT, entry, sl, tp, None, max_hold)

    return None


# ============================================================================
# STRATEGY REGISTRY
# ============================================================================

DAILY_STRATEGIES = {
    'D_SUPERTREND': {
        'fn': strategy_supertrend_daily,
        'indicators': {'supertrend': True, 'st_atr': 10, 'st_mult': 3.0, 'atr': True},
        'variants': [
            {'label': 'ST10_M3_SL2_MH20', 'sl_atr_mult': 2.0, 'max_hold': 20},
            {'label': 'ST10_M3_TR2_MH20', 'sl_atr_mult': 2.0, 'trail_atr_mult': 2.0, 'max_hold': 20},
            {'label': 'ST7_M3.5_SL2_MH15', 'sl_atr_mult': 2.0, 'max_hold': 15, '_st_atr': 7, '_st_mult': 3.5},
            {'label': 'ST7_M2.5_TR2_MH10', 'sl_atr_mult': 1.5, 'trail_atr_mult': 2.0, 'max_hold': 10, '_st_atr': 7, '_st_mult': 2.5},
        ],
    },
    'D_EMA_CROSS': {
        'fn': strategy_ema_crossover_daily,
        'indicators': {'ema': True, 'ema_params': (9, 21), 'adx': True, 'atr': True},
        'variants': [
            {'label': 'EMA9_21_ADX25_TR2.5_MH15', 'adx_threshold': 25, 'sl_atr_mult': 2.0, 'trail_atr_mult': 2.5, 'max_hold': 15},
            {'label': 'EMA5_13_ADX20_TR2_MH10', 'adx_threshold': 20, 'sl_atr_mult': 1.5, 'trail_atr_mult': 2.0, 'max_hold': 10, '_ema_params': (5, 13)},
            {'label': 'EMA13_34_ADX25_TR3_MH20', 'adx_threshold': 25, 'sl_atr_mult': 2.0, 'trail_atr_mult': 3.0, 'max_hold': 20, '_ema_params': (13, 34)},
        ],
    },
    'D_RSI_REVERT': {
        'fn': strategy_rsi_mean_reversion_daily,
        'indicators': {'rsi': True, 'atr': True},
        'variants': [
            {'label': 'RSI30_70_RR2_MH10', 'rsi_oversold': 30, 'rsi_overbought': 70, 'risk_reward': 2.0, 'sl_atr_mult': 1.5, 'max_hold': 10},
            {'label': 'RSI25_75_RR2.5_MH10', 'rsi_oversold': 25, 'rsi_overbought': 75, 'risk_reward': 2.5, 'sl_atr_mult': 1.5, 'max_hold': 10},
            {'label': 'RSI20_80_RR3_MH15', 'rsi_oversold': 20, 'rsi_overbought': 80, 'risk_reward': 3.0, 'sl_atr_mult': 2.0, 'max_hold': 15},
        ],
    },
    'D_BB_SQUEEZE': {
        'fn': strategy_bb_squeeze_daily,
        'indicators': {'bollinger': True, 'keltner': True, 'macd': True, 'atr': True},
        'variants': [
            {'label': 'BBSQ_SL2_TR2.5_MH15', 'sl_atr_mult': 2.0, 'trail_atr_mult': 2.5, 'max_hold': 15},
            {'label': 'BBSQ_SL1.5_TR2_MH10', 'sl_atr_mult': 1.5, 'trail_atr_mult': 2.0, 'max_hold': 10},
        ],
    },
    'D_DONCHIAN': {
        'fn': strategy_donchian_daily,
        'indicators': {'donchian': True, 'dc_period': 20, 'atr': True},
        'variants': [
            {'label': 'DC20_TR2_MH20', 'sl_atr_mult': 2.0, 'trail_atr_mult': 2.0, 'max_hold': 20},
            {'label': 'DC10_TR1.5_MH10', 'sl_atr_mult': 1.5, 'trail_atr_mult': 1.5, 'max_hold': 10, '_dc_period': 10},
            {'label': 'DC30_TR2.5_MH30', 'sl_atr_mult': 2.5, 'trail_atr_mult': 2.5, 'max_hold': 30, '_dc_period': 30},
        ],
    },
    'D_NR_BREAKOUT': {
        'fn': strategy_nr_breakout_daily,
        'indicators': {'nr': True, 'atr': True},
        'variants': [
            {'label': 'NR4_RR2_MH5', 'use_nr7': False, 'risk_reward': 2.0, 'max_hold': 5},
            {'label': 'NR4_RR3_MH5', 'use_nr7': False, 'risk_reward': 3.0, 'max_hold': 5},
            {'label': 'NR7_RR2_MH7', 'use_nr7': True, 'risk_reward': 2.0, 'max_hold': 7},
        ],
    },
    'D_PA_MACD': {
        'fn': strategy_price_action_macd_daily,
        'indicators': {'macd': True, 'atr': True},
        'variants': [
            {'label': 'PA_MACD_RR2_MH10', 'risk_reward': 2.0, 'max_hold': 10},
            {'label': 'PA_MACD_RR3_MH10', 'risk_reward': 3.0, 'max_hold': 10},
        ],
    },
    'D_MACD_HIST': {
        'fn': strategy_macd_histogram_daily,
        'indicators': {'macd': True, 'atr': True},
        'variants': [
            {'label': 'MACDH_SL2_TR2.5_MH15', 'sl_atr_mult': 2.0, 'trail_atr_mult': 2.5, 'max_hold': 15},
            {'label': 'MACDH_SL1.5_TR2_MH10', 'sl_atr_mult': 1.5, 'trail_atr_mult': 2.0, 'max_hold': 10},
        ],
    },
    'D_STOCH_RSI': {
        'fn': strategy_stoch_rsi_daily,
        'indicators': {'stoch': True, 'rsi': True, 'atr': True},
        'variants': [
            {'label': 'STOCHRSI_RR2_MH10', 'risk_reward': 2.0, 'sl_atr_mult': 1.5, 'max_hold': 10},
            {'label': 'STOCHRSI_RR3_MH15', 'risk_reward': 3.0, 'sl_atr_mult': 2.0, 'max_hold': 15},
        ],
    },
}


def run_single_strategy(
    strategy_name: str,
    strategy_fn,
    params: dict,
    data: Dict[str, pd.DataFrame],
    indicators: dict,
    initial_capital: float = 10_000_000,
    max_positions: int = 10,
) -> BacktestResult:
    """Run a single strategy across all symbols on daily data."""

    engine = IntradayBacktestEngine(
        initial_capital=initial_capital,
        position_size_pct=1.0 / max_positions,
        max_positions=max_positions,
        commission_pct=0.001,   # 0.1% for cash, higher than intraday
        slippage_pct=0.001,     # 0.1% for daily bars
        fixed_sizing=True,
    )

    prepared_data = {}
    for symbol, df in data.items():
        try:
            ind = dict(indicators)
            for k, v in params.items():
                if k.startswith('_'):
                    ind[k[1:]] = v
            prepared_data[symbol] = prepare_daily_indicators(df, ind)
        except Exception:
            continue

    if not prepared_data:
        return BacktestResult()

    # Process each symbol through all bars
    for symbol, df in prepared_data.items():
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
        atr_vals = df['atr'].values if 'atr' in df.columns else [None] * len(df)
        st_dirs = df['st_direction'].values if 'st_direction' in df.columns else [None] * len(df)
        date_strs = df['date_str'].values

        for i in range(len(df)):
            # Check exits
            if symbol in engine.positions:
                atr_v = atr_vals[i]
                st_v = st_dirs[i]

                engine.check_exits(
                    symbol=symbol,
                    bar_idx=i,
                    bar_date=date_strs[i],
                    high=highs[i],
                    low=lows[i],
                    close=closes[i],
                    atr=float(atr_v) if atr_v is not None and not np.isnan(atr_v) else None,
                    supertrend_dir=int(st_v) if st_v is not None and not np.isnan(st_v) else None,
                    is_eod=False,  # No EOD exit for swing trades
                )

            # Signals
            if symbol not in engine.positions:
                signal = strategy_fn(df, i, params)
                if signal:
                    engine.open_position(symbol, signal, i, date_strs[i])

        # Update equity on each trading day
        for i in range(len(df)):
            engine.update_equity(date_strs[i], {symbol: closes[i]})

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


def write_result(strategy_label, params_str, result, n_symbols, period):
    row = {
        'strategy': strategy_label,
        'params': params_str,
        'timeframe': 'day',
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
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--symbols', type=int, default=0, help='Limit symbols')
    args = parser.parse_args()

    if args.list:
        for name, cfg in DAILY_STRATEGIES.items():
            print(f'{name}: {len(cfg["variants"])} variants')
            for v in cfg['variants']:
                print(f'  - {v["label"]}')
        return

    configs = []
    for strat_name, strat_cfg in DAILY_STRATEGIES.items():
        if args.strategy and strat_name != args.strategy:
            continue
        for variant in strat_cfg['variants']:
            label = f'{strat_name}_{variant["label"]}'
            configs.append((label, strat_name, variant))

    ensure_header()
    done = load_done()
    pending = [(l, s, v) for l, s, v in configs if l not in done]

    print(f'=== Daily Strategy Sweep ===')
    print(f'Strategies: {len(DAILY_STRATEGIES)} | Variants: {len(configs)} | Pending: {len(pending)}')

    if not pending:
        print('All completed!')
        return

    # Load F&O stocks with daily data (2018-2025)
    # Use the FNO universe for futures-eligible stocks
    print('\nLoading daily data for F&O stocks...', flush=True)
    t0 = time.time()

    # Get F&O symbols from database (ones with sufficient daily data)
    all_symbols = get_all_symbols_for_timeframe('day', min_rows=1500)

    # Filter to a reasonable set — use top liquid stocks
    # Read FNO lot sizes from data_manager for F&O eligible stocks
    try:
        from services.data_manager import FNO_LOT_SIZES
        fno_symbols = [s for s in FNO_LOT_SIZES.keys() if s in all_symbols]
    except ImportError:
        fno_symbols = all_symbols[:100]

    if args.symbols > 0:
        fno_symbols = fno_symbols[:args.symbols]

    data = load_data_from_db(fno_symbols, 'day', '2018-01-01', '2025-12-31')
    loaded = list(data.keys())
    print(f'Loaded {len(loaded)} F&O stocks in {time.time()-t0:.0f}s', flush=True)

    for i, (label, strat_name, variant) in enumerate(pending, 1):
        strat_cfg = DAILY_STRATEGIES[strat_name]
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

            if result.yearly_stats:
                years_str = ' | '.join(
                    f'{y}:{s["trades"]}t/{s["win_rate"]:.0f}%/{s["pnl_pct"]:+.1f}%'
                    for y, s in sorted(result.yearly_stats.items())
                )
                print(f'  Years: {years_str}', flush=True)

            write_result(label, json.dumps(variant), result, len(loaded), '2018-01 to 2025-12')

            IntradayBacktestEngine.print_results(result, label)

        except Exception as e:
            import traceback
            print(f'  ERROR: {e}', flush=True)
            traceback.print_exc()

    print(f'\nResults: {OUTPUT_CSV}')


if __name__ == '__main__':
    main()
