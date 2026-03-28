#!/usr/bin/env python3
"""
BB Expansion Directional Breakout — Backtest & Optimization
=============================================================
Tests BB squeeze-to-expansion breakout as:
1. Standalone strategy
2. 4th component in Trident (combined with InsideDay, PA_MACD, RangeBreakout)

Signal logic:
- SQUEEZE: BB width < BB width MA for N consecutive bars
- FIRE: BB width crosses above BB width MA (expansion starts)
- LONG:  Fire + close > SMA(20) → immediate entry at close
- SHORT: Fire + close < SMA(20) → immediate entry at close
- SL: ATR-based, TP: risk multiplier, MaxHold configurable
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

OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'optimization_bb_expansion.csv')
FIELDNAMES = [
    'label', 'total_trades', 'id_trades', 'pamacd_trades', 'rb_trades', 'bb_trades',
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


# ─── BB Expansion signal detector ────────────────────────────────────────────

def detect_bb_expansion_signals(sym, df, i, bb_period=20, sq_min_bars=3,
                                 sl_atr_mult=2.0, tp_mult=2.0, max_hold=10,
                                 sma_period=20):
    """BB Squeeze→Expansion directional breakout.
    Returns list of immediate signals: (type, entry, sl, tp, max_hold, strategy_name)
    """
    warmup = max(bb_period + sq_min_bars + 5, sma_period + 5, 20)
    if i < warmup:
        return []

    # Check for fire: expanding now, was squeezing for N bars
    bb_expanding_now = df['bb_expanding'].iloc[i]
    if not bb_expanding_now:
        return []

    # Count consecutive squeeze bars before this bar
    squeeze_count = 0
    for j in range(i - 1, max(i - 50, -1), -1):
        if df['bb_squeeze'].iloc[j]:
            squeeze_count += 1
        else:
            break

    if squeeze_count < sq_min_bars:
        return []

    # First fire only — prev bar should still be squeezing
    if i > 0 and df['bb_expanding'].iloc[i - 1]:
        return []

    close = df['close'].iloc[i]
    sma_val = df[f'sma{sma_period}'].iloc[i]
    atr_val = df['atr14'].iloc[i]

    if atr_val <= 0 or np.isnan(atr_val) or np.isnan(sma_val):
        return []

    signals = []

    # LONG: close > SMA
    if close > sma_val:
        entry = close
        sl = entry - sl_atr_mult * atr_val
        risk = entry - sl
        if risk > 0:
            tp = entry + tp_mult * risk
            signals.append(('long_immediate', entry, sl, tp, max_hold, 'BBExpansion'))

    # SHORT: close < SMA
    if close < sma_val:
        entry = close
        sl = entry + sl_atr_mult * atr_val
        risk = sl - entry
        if risk > 0:
            tp = entry - tp_mult * risk
            signals.append(('short_immediate', entry, sl, tp, max_hold, 'BBExpansion'))

    return signals


# ─── Existing Trident strategies (unchanged) ─────────────────────────────────

def detect_inside_day_signals(sym, df, i):
    if i < 15:
        return []
    h_curr, l_curr = df['high'].iloc[i], df['low'].iloc[i]
    h_prev, l_prev = df['high'].iloc[i - 1], df['low'].iloc[i - 1]
    if not (h_curr < h_prev and l_curr > l_prev):
        return []
    atr_val = df['atr14'].iloc[i]
    if atr_val <= 0 or np.isnan(atr_val):
        return []
    signals = []
    entry_stop = h_prev
    sl = entry_stop - 2.0 * atr_val
    risk = entry_stop - sl
    tp = entry_stop + 3.0 * risk
    signals.append(('long_stop', entry_stop, sl, tp, 5, 'InsideDay'))
    entry_stop_s = l_prev
    sl_s = entry_stop_s + 2.0 * atr_val
    risk_s = sl_s - entry_stop_s
    tp_s = entry_stop_s - 3.0 * risk_s
    signals.append(('short_stop', entry_stop_s, sl_s, tp_s, 5, 'InsideDay'))
    return signals


def detect_pamacd_signals(sym, df, i):
    if i < 2:
        return []
    o_today, c_today = df['open'].iloc[i], df['close'].iloc[i]
    o_prev, c_prev = df['open'].iloc[i - 1], df['close'].iloc[i - 1]
    h_prev, l_prev = df['high'].iloc[i - 1], df['low'].iloc[i - 1]
    macd_hist = df['macd_hist'].iloc[i]
    if np.isnan(macd_hist):
        return []
    signals = []
    if c_today > o_today and c_prev < o_prev and c_today > h_prev and macd_hist > 0:
        entry_stop, sl = h_prev, l_prev
        risk = entry_stop - sl
        if risk > 0:
            signals.append(('long_stop', entry_stop, sl, entry_stop + 3 * risk, 10, 'PA_MACD'))
    if c_today < o_today and c_prev > o_prev and c_today < l_prev and macd_hist < 0:
        entry_stop, sl = l_prev, h_prev
        risk = sl - entry_stop
        if risk > 0:
            signals.append(('short_stop', entry_stop, sl, entry_stop - 3 * risk, 10, 'PA_MACD'))
    return signals


def detect_range_breakout_signals(sym, df, i):
    if i < 5:
        return []
    highs_5d = df['high'].iloc[i - 5:i].max()
    lows_5d = df['low'].iloc[i - 5:i].min()
    h_today, l_today = df['high'].iloc[i], df['low'].iloc[i]
    signals = []
    if h_today >= highs_5d:
        entry, sl = highs_5d, lows_5d
        risk = entry - sl
        if risk > 0:
            signals.append(('long_immediate', entry, sl, entry + 3 * risk, 15, 'RangeBreak5d'))
    if l_today <= lows_5d:
        entry, sl = lows_5d, highs_5d
        risk = sl - entry
        if risk > 0:
            signals.append(('short_immediate', entry, sl, entry - 3 * risk, 15, 'RangeBreak5d'))
    return signals


# ─── Pool runners ────────────────────────────────────────────────────────────

def run_standalone_bb(all_data, bb_params, max_positions=20):
    """Run BB Expansion as standalone strategy."""
    engine = IntradayBacktestEngine(
        initial_capital=INITIAL_CAPITAL, position_size_pct=0.10,
        max_positions=max_positions, commission_pct=COMMISSION,
        slippage_pct=SLIPPAGE, mode='cash', fixed_sizing=True,
    )
    all_dates = sorted({d for df in all_data.values() for d in df['date_str'].tolist()})
    trade_count = 0

    for date_str in all_dates:
        for sym in list(engine.positions.keys()):
            if sym not in all_data: continue
            df = all_data[sym]
            mask = df['date_str'] == date_str
            if not mask.any(): continue
            idx = df.index.get_loc(df[mask].index[0])
            engine.check_exits(sym, idx, date_str, df['high'].iloc[idx], df['low'].iloc[idx], df['close'].iloc[idx])

        for sym, df in all_data.items():
            mask = df['date_str'] == date_str
            if not mask.any(): continue
            idx_loc = df.index.get_loc(df[mask].index[0])
            for sig_info in detect_bb_expansion_signals(sym, df, idx_loc, **bb_params):
                sig_type, entry, sl, tp, max_hold, strat = sig_info
                if sym in engine.positions: continue
                if 'long' in sig_type:
                    sig = TradeSignal(Direction.LONG, entry, sl, tp, max_hold_bars=max_hold)
                    if engine.open_position(sym, sig, idx_loc, date_str):
                        trade_count += 1
                elif 'short' in sig_type:
                    sig = TradeSignal(Direction.SHORT, entry, sl, tp, max_hold_bars=max_hold)
                    if engine.open_position(sym, sig, idx_loc, date_str):
                        trade_count += 1

        prices = {}
        for sym in engine.positions:
            if sym in all_data:
                df_s = all_data[sym]
                mask = df_s['date_str'] == date_str
                if mask.any():
                    prices[sym] = df_s.loc[mask, 'close'].iloc[0]
        engine.update_equity(date_str, prices)

    for sym in list(engine.positions.keys()):
        if sym in all_data:
            df_s = all_data[sym]
            engine.close_position(sym, df_s['close'].iloc[-1], len(df_s) - 1, df_s['date_str'].iloc[-1], ExitType.EOD)

    return engine.get_results(), {'BBExpansion': trade_count}


def run_combined_4strat(all_data, bb_params, max_positions=20):
    """Run all 4 strategies on a shared capital pool."""
    engine = IntradayBacktestEngine(
        initial_capital=INITIAL_CAPITAL, position_size_pct=0.10,
        max_positions=max_positions, commission_pct=COMMISSION,
        slippage_pct=SLIPPAGE, mode='cash', fixed_sizing=True,
    )
    all_dates = sorted({d for df in all_data.values() for d in df['date_str'].tolist()})
    strat_counts = {'InsideDay': 0, 'PA_MACD': 0, 'RangeBreak5d': 0, 'BBExpansion': 0}
    pending, pos_strategy = {}, {}

    for date_str in all_dates:
        # Phase 1: Exits
        for sym in list(engine.positions.keys()):
            if sym not in all_data: continue
            df = all_data[sym]
            mask = df['date_str'] == date_str
            if not mask.any(): continue
            idx = df.index.get_loc(df[mask].index[0])
            trade = engine.check_exits(sym, idx, date_str, df['high'].iloc[idx], df['low'].iloc[idx], df['close'].iloc[idx])
            if trade:
                strat_name = pos_strategy.pop(sym, 'unknown')
                strat_counts[strat_name] = strat_counts.get(strat_name, 0) + 1

        # Phase 2: Pending stops
        for key, pend in list(pending.items()):
            sym, direction = key
            if sym in engine.positions or sym not in all_data: continue
            df = all_data[sym]
            mask = df['date_str'] == date_str
            if not mask.any(): continue
            idx = df.index.get_loc(df[mask].index[0])
            h, l = df['high'].iloc[idx], df['low'].iloc[idx]
            if direction == 'long' and h >= pend['stop_level']:
                sig = TradeSignal(Direction.LONG, pend['stop_level'], pend['sl'], pend['tp'], max_hold_bars=pend['max_hold'])
                if engine.open_position(sym, sig, idx, date_str):
                    pos_strategy[sym] = pend['strategy']
            elif direction == 'short' and l <= pend['stop_level']:
                sig = TradeSignal(Direction.SHORT, pend['stop_level'], pend['sl'], pend['tp'], max_hold_bars=pend['max_hold'])
                if engine.open_position(sym, sig, idx, date_str):
                    pos_strategy[sym] = pend['strategy']
        pending.clear()

        # Phase 3: Generate signals
        for sym, df in all_data.items():
            mask = df['date_str'] == date_str
            if not mask.any(): continue
            idx_loc = df.index.get_loc(df[mask].index[0])

            # InsideDay (stop orders)
            for sig_info in detect_inside_day_signals(sym, df, idx_loc):
                sig_type, stop_level, sl, tp, max_hold, strat = sig_info
                direction = 'long' if 'long' in sig_type else 'short'
                key = (sym, direction)
                if key not in pending:
                    pending[key] = {'stop_level': stop_level, 'sl': sl, 'tp': tp, 'max_hold': max_hold, 'strategy': strat}

            # PA_MACD (stop orders)
            for sig_info in detect_pamacd_signals(sym, df, idx_loc):
                sig_type, stop_level, sl, tp, max_hold, strat = sig_info
                direction = 'long' if 'long' in sig_type else 'short'
                key = (sym, direction)
                if key not in pending:
                    pending[key] = {'stop_level': stop_level, 'sl': sl, 'tp': tp, 'max_hold': max_hold, 'strategy': strat}

            # RangeBreakout (immediate)
            for sig_info in detect_range_breakout_signals(sym, df, idx_loc):
                sig_type, entry, sl, tp, max_hold, strat = sig_info
                if sym in engine.positions: continue
                if 'long' in sig_type:
                    sig = TradeSignal(Direction.LONG, entry, sl, tp, max_hold_bars=max_hold)
                    if engine.open_position(sym, sig, idx_loc, date_str):
                        pos_strategy[sym] = strat
                elif 'short' in sig_type:
                    sig = TradeSignal(Direction.SHORT, entry, sl, tp, max_hold_bars=max_hold)
                    if engine.open_position(sym, sig, idx_loc, date_str):
                        pos_strategy[sym] = strat

            # BB Expansion (immediate)
            for sig_info in detect_bb_expansion_signals(sym, df, idx_loc, **bb_params):
                sig_type, entry, sl, tp, max_hold, strat = sig_info
                if sym in engine.positions: continue
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

    # Force close
    for sym in list(engine.positions.keys()):
        strat_name = pos_strategy.get(sym, 'unknown')
        if sym in all_data:
            df_s = all_data[sym]
            trade = engine.close_position(sym, df_s['close'].iloc[-1], len(df_s) - 1, df_s['date_str'].iloc[-1], ExitType.EOD)
            if trade:
                strat_counts[strat_name] = strat_counts.get(strat_name, 0) + 1

    return engine.get_results(), strat_counts


def result_to_row(label, result, strat_counts):
    return {
        'label': label,
        'total_trades': result.total_trades,
        'id_trades': strat_counts.get('InsideDay', 0),
        'pamacd_trades': strat_counts.get('PA_MACD', 0),
        'rb_trades': strat_counts.get('RangeBreak5d', 0),
        'bb_trades': strat_counts.get('BBExpansion', 0),
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


def main():
    # Read already-done configs
    done = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV) as f:
            done = {row['label'] for row in csv.DictReader(f)}
        print(f'Found {len(done)} already-completed configs', flush=True)

    # ─── Configuration sweep ───
    configs = []

    # --- Standalone BB Expansion parameter sweep ---
    # Base: bb_period=20, sq_min_bars=3, sl_atr_mult=2.0, tp_mult=2.0, max_hold=10
    bb_base = {'bb_period': 20, 'sq_min_bars': 3, 'sl_atr_mult': 2.0, 'tp_mult': 2.0, 'max_hold': 10, 'sma_period': 20}

    configs.append(('BB_STANDALONE_BASE', 'standalone', dict(bb_base)))

    # Squeeze min bars sweep
    for sq in [2, 4, 5]:
        p = dict(bb_base, sq_min_bars=sq)
        configs.append((f'BB_STANDALONE_SQ{sq}', 'standalone', p))

    # TP sweep
    for tp in [1.5, 3.0]:
        p = dict(bb_base, tp_mult=tp)
        configs.append((f'BB_STANDALONE_TP{tp}', 'standalone', p))

    # SL ATR sweep
    for sl in [1.5, 2.5]:
        p = dict(bb_base, sl_atr_mult=sl)
        configs.append((f'BB_STANDALONE_SL{sl}', 'standalone', p))

    # MaxHold sweep
    for mh in [5, 15]:
        p = dict(bb_base, max_hold=mh)
        configs.append((f'BB_STANDALONE_MH{mh}', 'standalone', p))

    # BB period sweep
    for bp in [10, 15]:
        p = dict(bb_base, bb_period=bp)
        configs.append((f'BB_STANDALONE_BP{bp}', 'standalone', p))

    # --- Combined 4-strategy tests ---
    configs.append(('QUAD_20POS_BASE', 'combined', dict(bb_base)))
    configs.append(('QUAD_25POS_BASE', 'combined_25', dict(bb_base)))

    # Best standalone params combined
    configs.append(('QUAD_20POS_SQ2', 'combined', dict(bb_base, sq_min_bars=2)))
    configs.append(('QUAD_20POS_TP1.5', 'combined', dict(bb_base, tp_mult=1.5)))
    configs.append(('QUAD_20POS_MH5', 'combined', dict(bb_base, max_hold=5)))

    remaining = [(l, t, p) for l, t, p in configs if l not in done]
    if not remaining:
        print('All configs already completed!', flush=True)
        return

    print(f'\n=== Running {len(remaining)} BB Expansion configs ===\n', flush=True)

    print('Loading data for 50 F&O stocks...', flush=True)
    t0 = time.time()
    all_data = load_data_from_db(FNO_STOCKS, 'day', '2018-01-01', '2025-12-31')
    print(f'Loaded {len(all_data)} symbols in {time.time() - t0:.1f}s', flush=True)

    # Precompute indicators
    print('Computing indicators (ATR14, MACD, BB, SMAs)...', flush=True)
    for sym, df in all_data.items():
        df['atr14'] = calc_atr(df, 14)
        macd_line, macd_signal, macd_hist = calc_macd(df['close'])
        df['macd_hist'] = macd_hist

        # BB indicators for multiple periods
        for bp in [10, 15, 20]:
            sma_bb = df['close'].rolling(bp).mean()
            std_bb = df['close'].rolling(bp).std()
            bb_width = (2 * 2.0 * std_bb) / sma_bb * 100
            bb_width_ma = bb_width.rolling(bp).mean()
            if bp == 20:  # default columns
                df['bb_squeeze'] = (bb_width < bb_width_ma).astype(int)
                df['bb_expanding'] = (bb_width > bb_width_ma).astype(int)
            df[f'bb_squeeze_{bp}'] = (bb_width < bb_width_ma).astype(int)
            df[f'bb_expanding_{bp}'] = (bb_width > bb_width_ma).astype(int)

        # SMAs
        df['sma20'] = df['close'].rolling(20).mean()
        df['sma15'] = df['close'].rolling(15).mean()
        df['sma10'] = df['close'].rolling(10).mean()
    print('Indicators ready.', flush=True)

    # Ensure CSV exists
    if not os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    for i, (label, cfg_type, bb_params) in enumerate(remaining):
        print(f'[{i+1}/{len(remaining)}] {label} ...', end='', flush=True)
        t1 = time.time()

        # For non-default BB periods, swap the columns
        bp = bb_params.get('bb_period', 20)
        if bp != 20:
            for sym, df in all_data.items():
                df['bb_squeeze'] = df[f'bb_squeeze_{bp}']
                df['bb_expanding'] = df[f'bb_expanding_{bp}']
                df[f'sma{bb_params.get("sma_period", 20)}'] = df['close'].rolling(bb_params.get('sma_period', 20)).mean()

        if cfg_type == 'standalone':
            result, strat_counts = run_standalone_bb(all_data, bb_params)
        elif cfg_type == 'combined_25':
            result, strat_counts = run_combined_4strat(all_data, bb_params, max_positions=25)
        else:
            result, strat_counts = run_combined_4strat(all_data, bb_params, max_positions=20)

        # Restore default BB columns
        if bp != 20:
            for sym, df in all_data.items():
                df['bb_squeeze'] = df['bb_squeeze_20']
                df['bb_expanding'] = df['bb_expanding_20']

        elapsed = time.time() - t1
        print(f' {elapsed:.0f}s | CAGR={result.cagr:.2f}% WR={result.win_rate:.1f}% '
              f'MaxDD={result.max_drawdown:.2f}% Sharpe={getattr(result, "sharpe_ratio", 0):.2f} '
              f'BB={strat_counts.get("BBExpansion", 0)} trades={result.total_trades}', flush=True)

        row = result_to_row(label, result, strat_counts)
        with open(OUTPUT_CSV, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

    print(f'\nAll configs done. Results in {OUTPUT_CSV}', flush=True)


if __name__ == '__main__':
    main()
