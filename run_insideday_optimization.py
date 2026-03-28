#!/usr/bin/env python3
"""
InsideDay V2 Optimization
===========================
Sweeps InsideDay Breakout filter combinations to find higher win-rate configs.
All runs use the shared-pool 20-position Trident setup.

Filters tested:
1. TP multiplier: 2x vs 3x risk
2. MaxHold: 3 vs 5 bars
3. EMA trend filter: EMA(20) direction
4. Volume filter: inside bar volume < outer bar volume
5. Inner bar quality: min ratio of inner/outer range (0.3, 0.4, 0.5)
"""
import csv, os, sys, time, logging
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

OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'optimization_insideday_v2.csv')
FIELDNAMES = [
    'label', 'total_trades', 'id_trades', 'pamacd_trades', 'rb_trades',
    'win_rate', 'profit_factor', 'cagr', 'max_drawdown',
    'sharpe', 'sortino', 'calmar',
    'total_pnl', 'pnl_pct', 'avg_win', 'avg_loss', 'avg_rr',
    'exit_reasons',
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

INITIAL_CAPITAL = 10_000_000
COMMISSION = 0.0001
SLIPPAGE = 0.0005


# ─── InsideDay V2 with configurable filters ─────────────────────────────────

def detect_inside_day_v2(sym, df, i, tp_mult=3.0, sl_mult=2.0, max_hold=5,
                          use_ema=False, ema_period=20, use_vol=False, min_ratio=0.0):
    """InsideDay with optional V2 filters."""
    if i < max(15, ema_period + 1):
        return []

    h_curr = df['high'].iloc[i]
    l_curr = df['low'].iloc[i]
    h_prev = df['high'].iloc[i - 1]
    l_prev = df['low'].iloc[i - 1]

    # Inside day check
    if not (h_curr < h_prev and l_curr > l_prev):
        return []

    # Inner bar quality filter
    outer_range = h_prev - l_prev
    inner_range = h_curr - l_curr
    if outer_range <= 0:
        return []
    if inner_range / outer_range < min_ratio:
        return []

    # Volume filter: inside bar volume < outer bar volume
    if use_vol and df['volume'].iloc[i] >= df['volume'].iloc[i - 1]:
        return []

    atr_val = df['atr14'].iloc[i]
    if atr_val <= 0 or np.isnan(atr_val):
        return []

    close = df['close'].iloc[i]
    signals = []

    # EMA trend filter
    if use_ema:
        ema_val = df[f'ema{ema_period}'].iloc[i]
        long_ok = close > ema_val
        short_ok = close < ema_val
    else:
        long_ok = True
        short_ok = True

    # LONG
    if long_ok:
        entry_stop = h_prev
        sl = entry_stop - sl_mult * atr_val
        risk = entry_stop - sl
        tp = entry_stop + tp_mult * risk
        signals.append(('long_stop', entry_stop, sl, tp, max_hold, 'InsideDay'))

    # SHORT
    if short_ok:
        entry_stop_s = l_prev
        sl_s = entry_stop_s + sl_mult * atr_val
        risk_s = sl_s - entry_stop_s
        tp_s = entry_stop_s - tp_mult * risk_s
        signals.append(('short_stop', entry_stop_s, sl_s, tp_s, max_hold, 'InsideDay'))

    return signals


# ─── Unchanged PA_MACD and RangeBreakout ─────────────────────────────────────

def detect_pamacd_signals(sym, df, i):
    if i < 2:
        return []
    o_today = df['open'].iloc[i]
    c_today = df['close'].iloc[i]
    o_prev = df['open'].iloc[i - 1]
    c_prev = df['close'].iloc[i - 1]
    h_prev = df['high'].iloc[i - 1]
    l_prev = df['low'].iloc[i - 1]
    macd_hist = df['macd_hist'].iloc[i]
    if np.isnan(macd_hist):
        return []
    signals = []
    today_green = c_today > o_today
    today_red = c_today < o_today
    prev_green = c_prev > o_prev
    prev_red = c_prev < o_prev
    if today_green and prev_red and c_today > h_prev and macd_hist > 0:
        entry_stop = h_prev
        sl = l_prev
        risk = entry_stop - sl
        if risk > 0:
            tp = entry_stop + 3 * risk
            signals.append(('long_stop', entry_stop, sl, tp, 10, 'PA_MACD'))
    if today_red and prev_green and c_today < l_prev and macd_hist < 0:
        entry_stop = l_prev
        sl = h_prev
        risk = sl - entry_stop
        if risk > 0:
            tp = entry_stop - 3 * risk
            signals.append(('short_stop', entry_stop, sl, tp, 10, 'PA_MACD'))
    return signals


def detect_range_breakout_signals(sym, df, i):
    if i < 5:
        return []
    highs_5d = df['high'].iloc[i - 5:i].max()
    lows_5d = df['low'].iloc[i - 5:i].min()
    h_today = df['high'].iloc[i]
    l_today = df['low'].iloc[i]
    signals = []
    if h_today >= highs_5d:
        entry = highs_5d
        sl = lows_5d
        risk = entry - sl
        if risk > 0:
            tp = entry + 3 * risk
            signals.append(('long_immediate', entry, sl, tp, 15, 'RangeBreak5d'))
    if l_today <= lows_5d:
        entry = lows_5d
        sl = highs_5d
        risk = sl - entry
        if risk > 0:
            tp = entry - 3 * risk
            signals.append(('short_immediate', entry, sl, tp, 15, 'RangeBreak5d'))
    return signals


# ─── Shared pool runner (same as original, but uses configurable InsideDay) ──

def run_shared_pool_v2(all_data, id_params):
    """Run Trident with configurable InsideDay filters, 20 positions shared pool."""
    engine = IntradayBacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        position_size_pct=0.10,
        max_positions=20,
        commission_pct=COMMISSION,
        slippage_pct=SLIPPAGE,
        mode='cash',
        fixed_sizing=True,
    )

    all_dates = set()
    for sym, df in all_data.items():
        all_dates.update(df['date_str'].tolist())
    all_dates = sorted(all_dates)

    strat_counts = {'InsideDay': 0, 'PA_MACD': 0, 'RangeBreak5d': 0}
    pending = {}
    pos_strategy = {}

    for date_str in all_dates:
        # Phase 1: Exits
        for sym in list(engine.positions.keys()):
            if sym not in all_data:
                continue
            df = all_data[sym]
            mask = df['date_str'] == date_str
            if not mask.any():
                continue
            idx = df.index.get_loc(df[mask].index[0])
            h = df['high'].iloc[idx]
            l = df['low'].iloc[idx]
            c = df['close'].iloc[idx]
            trade = engine.check_exits(sym, idx, date_str, h, l, c)
            if trade:
                strat_name = pos_strategy.pop(sym, 'unknown')
                strat_counts[strat_name] = strat_counts.get(strat_name, 0) + 1

        # Phase 2: Check pending stops
        for key, pend in list(pending.items()):
            sym, direction = key
            if sym in engine.positions:
                continue
            if sym not in all_data:
                continue
            df = all_data[sym]
            mask = df['date_str'] == date_str
            if not mask.any():
                continue
            idx = df.index.get_loc(df[mask].index[0])
            h = df['high'].iloc[idx]
            l = df['low'].iloc[idx]

            if direction == 'long' and h >= pend['stop_level']:
                sig = TradeSignal(Direction.LONG, pend['stop_level'], pend['sl'],
                                  pend['tp'], max_hold_bars=pend['max_hold'])
                if engine.open_position(sym, sig, idx, date_str):
                    pos_strategy[sym] = pend['strategy']
            elif direction == 'short' and l <= pend['stop_level']:
                sig = TradeSignal(Direction.SHORT, pend['stop_level'], pend['sl'],
                                  pend['tp'], max_hold_bars=pend['max_hold'])
                if engine.open_position(sym, sig, idx, date_str):
                    pos_strategy[sym] = pend['strategy']

        pending.clear()

        # Phase 3: Generate signals
        for sym, df in all_data.items():
            mask = df['date_str'] == date_str
            if not mask.any():
                continue
            idx_loc = df.index.get_loc(df[mask].index[0])

            # InsideDay V2 with filters
            for sig_info in detect_inside_day_v2(sym, df, idx_loc, **id_params):
                sig_type, stop_level, sl, tp, max_hold, strat = sig_info
                direction = 'long' if 'long' in sig_type else 'short'
                key = (sym, direction)
                if key not in pending:
                    pending[key] = {'stop_level': stop_level, 'sl': sl, 'tp': tp,
                                    'max_hold': max_hold, 'strategy': strat}

            # PA_MACD (unchanged)
            for sig_info in detect_pamacd_signals(sym, df, idx_loc):
                sig_type, stop_level, sl, tp, max_hold, strat = sig_info
                direction = 'long' if 'long' in sig_type else 'short'
                key = (sym, direction)
                if key not in pending:
                    pending[key] = {'stop_level': stop_level, 'sl': sl, 'tp': tp,
                                    'max_hold': max_hold, 'strategy': strat}

            # RangeBreakout (unchanged, immediate)
            for sig_info in detect_range_breakout_signals(sym, df, idx_loc):
                sig_type, entry, sl, tp, max_hold, strat = sig_info
                if sym in engine.positions:
                    continue
                if 'long' in sig_type:
                    sig = TradeSignal(Direction.LONG, entry, sl, tp, max_hold_bars=max_hold)
                    if engine.open_position(sym, sig, idx_loc, date_str):
                        pos_strategy[sym] = strat
                elif 'short' in sig_type:
                    sig = TradeSignal(Direction.SHORT, entry, sl, tp, max_hold_bars=max_hold)
                    if engine.open_position(sym, sig, idx_loc, date_str):
                        pos_strategy[sym] = strat

        # Update equity
        prices = {}
        for sym in engine.positions:
            if sym in all_data:
                df_s = all_data[sym]
                mask = df_s['date_str'] == date_str
                if mask.any():
                    prices[sym] = df_s.loc[mask, 'close'].iloc[0]
        engine.update_equity(date_str, prices)

    # Force close remaining
    for sym in list(engine.positions.keys()):
        strat_name = pos_strategy.get(sym, 'unknown')
        if sym in all_data:
            df_s = all_data[sym]
            last_idx = len(df_s) - 1
            last_date = df_s['date_str'].iloc[last_idx]
            c = df_s['close'].iloc[last_idx]
            trade = engine.close_position(sym, c, last_idx, last_date, ExitType.EOD)
            if trade:
                strat_counts[strat_name] = strat_counts.get(strat_name, 0) + 1

    result = engine.get_results()
    return result, strat_counts


def result_to_row(label, result, strat_counts):
    row = {
        'label': label,
        'total_trades': result.total_trades,
        'id_trades': strat_counts.get('InsideDay', 0),
        'pamacd_trades': strat_counts.get('PA_MACD', 0),
        'rb_trades': strat_counts.get('RangeBreak5d', 0),
        'win_rate': round(result.win_rate, 2),
        'profit_factor': round(result.profit_factor, 2),
        'cagr': round(result.cagr, 2),
        'max_drawdown': round(result.max_drawdown, 2),
        'sharpe': round(getattr(result, 'sharpe_ratio', 0), 2),
        'sortino': round(getattr(result, 'sortino_ratio', 0), 2),
        'calmar': round(getattr(result, 'calmar_ratio', 0), 2),
        'total_pnl': round(result.total_pnl, 0),
        'pnl_pct': round(result.total_pnl_pct, 2),
        'avg_win': round(result.avg_win, 2),
        'avg_loss': round(result.avg_loss, 2),
        'avg_rr': round(result.avg_rr, 2),
        'exit_reasons': str(result.exit_reasons),
    }
    return row


def main():
    print('Loading data for 50 F&O stocks...', flush=True)
    t0 = time.time()
    all_data = load_data_from_db(FNO_STOCKS, 'day', '2018-01-01', '2025-12-31')
    print(f'Loaded {len(all_data)} symbols in {time.time() - t0:.1f}s', flush=True)

    # Precompute indicators
    print('Computing indicators (ATR14, MACD, EMAs)...', flush=True)
    for sym, df in all_data.items():
        df['atr14'] = calc_atr(df, 14)
        macd_line, macd_signal, macd_hist = calc_macd(df['close'])
        df['macd_hist'] = macd_hist
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    print('Indicators ready.', flush=True)

    # Write CSV header
    with open(OUTPUT_CSV, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    # Skip already-done configs
    done = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV) as f:
            done = {row['label'] for row in csv.DictReader(f)}

    # ─── Configuration sweep ───
    configs = []

    # Baseline (original V1)
    configs.append(('ID_BASELINE_TP3_MH5', {
        'tp_mult': 3.0, 'sl_mult': 2.0, 'max_hold': 5,
        'use_ema': False, 'ema_period': 20, 'use_vol': False, 'min_ratio': 0.0
    }))

    # TP reduction only
    configs.append(('ID_TP2_MH5', {
        'tp_mult': 2.0, 'sl_mult': 2.0, 'max_hold': 5,
        'use_ema': False, 'ema_period': 20, 'use_vol': False, 'min_ratio': 0.0
    }))
    configs.append(('ID_TP1.5_MH5', {
        'tp_mult': 1.5, 'sl_mult': 2.0, 'max_hold': 5,
        'use_ema': False, 'ema_period': 20, 'use_vol': False, 'min_ratio': 0.0
    }))

    # MaxHold reduction
    configs.append(('ID_TP2_MH3', {
        'tp_mult': 2.0, 'sl_mult': 2.0, 'max_hold': 3,
        'use_ema': False, 'ema_period': 20, 'use_vol': False, 'min_ratio': 0.0
    }))

    # EMA trend filter
    configs.append(('ID_TP2_MH5_EMA20', {
        'tp_mult': 2.0, 'sl_mult': 2.0, 'max_hold': 5,
        'use_ema': True, 'ema_period': 20, 'use_vol': False, 'min_ratio': 0.0
    }))
    configs.append(('ID_TP2_MH3_EMA20', {
        'tp_mult': 2.0, 'sl_mult': 2.0, 'max_hold': 3,
        'use_ema': True, 'ema_period': 20, 'use_vol': False, 'min_ratio': 0.0
    }))
    configs.append(('ID_TP3_MH5_EMA20', {
        'tp_mult': 3.0, 'sl_mult': 2.0, 'max_hold': 5,
        'use_ema': True, 'ema_period': 20, 'use_vol': False, 'min_ratio': 0.0
    }))

    # Volume filter
    configs.append(('ID_TP2_MH5_VOL', {
        'tp_mult': 2.0, 'sl_mult': 2.0, 'max_hold': 5,
        'use_ema': False, 'ema_period': 20, 'use_vol': True, 'min_ratio': 0.0
    }))

    # EMA + Volume combined
    configs.append(('ID_TP2_MH5_EMA20_VOL', {
        'tp_mult': 2.0, 'sl_mult': 2.0, 'max_hold': 5,
        'use_ema': True, 'ema_period': 20, 'use_vol': True, 'min_ratio': 0.0
    }))
    configs.append(('ID_TP2_MH3_EMA20_VOL', {
        'tp_mult': 2.0, 'sl_mult': 2.0, 'max_hold': 3,
        'use_ema': True, 'ema_period': 20, 'use_vol': True, 'min_ratio': 0.0
    }))

    # Min bar ratio filter
    configs.append(('ID_TP2_MH5_R40', {
        'tp_mult': 2.0, 'sl_mult': 2.0, 'max_hold': 5,
        'use_ema': False, 'ema_period': 20, 'use_vol': False, 'min_ratio': 0.4
    }))
    configs.append(('ID_TP2_MH5_R50', {
        'tp_mult': 2.0, 'sl_mult': 2.0, 'max_hold': 5,
        'use_ema': False, 'ema_period': 20, 'use_vol': False, 'min_ratio': 0.5
    }))

    # Full V2 combos (best candidates)
    configs.append(('ID_TP2_MH3_EMA20_VOL_R40', {
        'tp_mult': 2.0, 'sl_mult': 2.0, 'max_hold': 3,
        'use_ema': True, 'ema_period': 20, 'use_vol': True, 'min_ratio': 0.4
    }))
    configs.append(('ID_TP2_MH5_EMA20_VOL_R40', {
        'tp_mult': 2.0, 'sl_mult': 2.0, 'max_hold': 5,
        'use_ema': True, 'ema_period': 20, 'use_vol': True, 'min_ratio': 0.4
    }))

    # SL multiplier variants
    configs.append(('ID_TP2_SL1.5_MH5', {
        'tp_mult': 2.0, 'sl_mult': 1.5, 'max_hold': 5,
        'use_ema': False, 'ema_period': 20, 'use_vol': False, 'min_ratio': 0.0
    }))
    configs.append(('ID_TP2_SL1.5_MH5_EMA20', {
        'tp_mult': 2.0, 'sl_mult': 1.5, 'max_hold': 5,
        'use_ema': True, 'ema_period': 20, 'use_vol': False, 'min_ratio': 0.0
    }))

    total = len(configs)
    print(f'\n=== Running {total} InsideDay V2 configs (20-pos shared pool) ===\n', flush=True)

    for i, (label, id_params) in enumerate(configs):
        if label in done:
            print(f'[{i+1}/{total}] {label} — SKIP', flush=True)
            continue

        print(f'[{i+1}/{total}] {label} ...', end='', flush=True)
        t1 = time.time()

        result, strat_counts = run_shared_pool_v2(all_data, id_params)

        elapsed = time.time() - t1
        print(f' {elapsed:.0f}s | CAGR={result.cagr:.2f}% WR={result.win_rate:.1f}% '
              f'MaxDD={result.max_drawdown:.2f}% Sharpe={result.sharpe_ratio:.2f} '
              f'ID={strat_counts.get("InsideDay", 0)} trades={result.total_trades}',
              flush=True)
        sys.stdout.flush()

        row = result_to_row(label, result, strat_counts)
        with open(OUTPUT_CSV, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

    print(f'\nDone. Results: {OUTPUT_CSV}', flush=True)


if __name__ == '__main__':
    main()
