#!/usr/bin/env python3
"""
Multi-Strategy Portfolio Backtest
==================================
Combines 3 no-lookahead strategies on a single capital pool:
1. InsideDay Breakout (L+S)
2. PA_MACD BuyStop (L+S)
3. RangeBreakout 5d (L+S)

Tests equal weighting, tilted weighting, and shared-pool configs.
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

OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'multi_strategy_portfolio_results.csv')
FIELDNAMES = [
    'label', 'total_trades', 'id_trades', 'pamacd_trades', 'rb_trades',
    'win_rate', 'profit_factor', 'cagr', 'max_drawdown',
    'sharpe', 'sortino', 'calmar',
    'total_pnl', 'pnl_pct', 'avg_win', 'avg_loss', 'avg_rr',
    'exit_reasons',
    'y2018_pnl_pct', 'y2019_pnl_pct', 'y2020_pnl_pct', 'y2021_pnl_pct',
    'y2022_pnl_pct', 'y2023_pnl_pct', 'y2024_pnl_pct', 'y2025_pnl_pct',
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
COMMISSION = 0.0001   # 0.01%
SLIPPAGE = 0.0005     # 0.05%


# ─── Strategy signal generators ───────────────────────────────────────────────

def detect_inside_day_signals(sym, df, i):
    """InsideDay Breakout: detect at bar i, place stops for bar i+1.
    Returns list of pending signals (direction, stop_level, sl, tp, max_hold, strategy_name)
    """
    if i < 15:  # need ATR(14) warmup
        return []
    h_prev = df['high'].iloc[i]
    l_prev = df['low'].iloc[i]
    h_prev2 = df['high'].iloc[i - 1]
    l_prev2 = df['low'].iloc[i - 1]

    # Inside day: current bar's range is inside previous bar's range
    if h_prev < h_prev2 and l_prev > l_prev2:
        atr_val = df['atr14'].iloc[i]
        if atr_val <= 0 or np.isnan(atr_val):
            return []
        signals = []
        # LONG: buy-stop at prev bar high (the outer bar)
        entry_stop = h_prev2
        sl = entry_stop - 2 * atr_val
        risk = entry_stop - sl
        tp = entry_stop + 3 * risk
        signals.append(('long_stop', entry_stop, sl, tp, 5, 'InsideDay'))

        # SHORT: sell-stop at prev bar low (the outer bar)
        entry_stop_s = l_prev2
        sl_s = entry_stop_s + 2 * atr_val
        risk_s = sl_s - entry_stop_s
        tp_s = entry_stop_s - 3 * risk_s
        signals.append(('short_stop', entry_stop_s, sl_s, tp_s, 5, 'InsideDay'))
        return signals
    return []


def detect_pamacd_signals(sym, df, i):
    """PA_MACD BuyStop: detect at bar i, place stops for bar i+1.
    LONG: today green, prev red, close > prev high, MACD hist > 0
    SHORT: today red, prev green, close < prev low, MACD hist < 0
    """
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

    # LONG: today green, prev red, close > prev high, MACD hist > 0
    if today_green and prev_red and c_today > h_prev and macd_hist > 0:
        entry_stop = h_prev  # buy-stop at prev candle high
        sl = l_prev  # SL at prev candle low
        risk = entry_stop - sl
        if risk > 0:
            tp = entry_stop + 3 * risk
            signals.append(('long_stop', entry_stop, sl, tp, 10, 'PA_MACD'))

    # SHORT: today red, prev green, close < prev low, MACD hist < 0
    if today_red and prev_green and c_today < l_prev and macd_hist < 0:
        entry_stop = l_prev  # sell-stop at prev candle low  (actually today's prev = the green candle)
        # Wait — the spec says: "Sell-stop at prev low next day" and SL = prev candle high
        # "prev" here means the green candle (bar i-1)
        sl = h_prev  # SL at prev candle high
        risk = sl - entry_stop
        if risk > 0:
            tp = entry_stop - 3 * risk
            signals.append(('short_stop', entry_stop, sl, tp, 10, 'PA_MACD'))

    return signals


def detect_range_breakout_signals(sym, df, i):
    """Range Breakout 5d: at bar i, compute 5d high/low from [i-5..i-1].
    Place buy-stop at 5d high, sell-stop at 5d low.
    Check at bar i itself if triggered (levels are known before bar i opens).
    Returns triggered signals (not pending — these fire on current bar).
    """
    if i < 5:
        return []
    highs_5d = df['high'].iloc[i - 5:i].max()
    lows_5d = df['low'].iloc[i - 5:i].min()

    h_today = df['high'].iloc[i]
    l_today = df['low'].iloc[i]

    signals = []
    # LONG: price breaks above 5d high
    if h_today >= highs_5d:
        entry = highs_5d
        sl = lows_5d
        risk = entry - sl
        if risk > 0:
            tp = entry + 3 * risk
            signals.append(('long_immediate', entry, sl, tp, 15, 'RangeBreak5d'))

    # SHORT: price breaks below 5d low
    if l_today <= lows_5d:
        entry = lows_5d
        sl = highs_5d
        risk = sl - entry
        if risk > 0:
            tp = entry - 3 * risk
            signals.append(('short_immediate', entry, sl, tp, 15, 'RangeBreak5d'))

    return signals


# ─── Shared-pool multi-strategy backtest ──────────────────────────────────────

def run_shared_pool(all_data, max_positions, label):
    """Run all 3 strategies on a shared capital pool with a single engine."""
    engine = IntradayBacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        position_size_pct=0.10,
        max_positions=max_positions,
        commission_pct=COMMISSION,
        slippage_pct=SLIPPAGE,
        mode='cash',
        fixed_sizing=True,
    )

    # Build a unified date index across all symbols
    all_dates = set()
    for sym, df in all_data.items():
        all_dates.update(df['date_str'].tolist())
    all_dates = sorted(all_dates)

    # Strategy trade counters
    strat_counts = {'InsideDay': 0, 'PA_MACD': 0, 'RangeBreak5d': 0}

    # Pending stop orders: dict of (sym, direction) -> {stop_level, sl, tp, max_hold, strategy, signal_date}
    pending = {}

    # Track which positions belong to which strategy
    pos_strategy = {}  # sym -> strategy_name

    for date_str in all_dates:
        # --- Phase 1: Check exits for open positions ---
        closed_syms = []
        for sym in list(engine.positions.keys()):
            if sym not in all_data:
                continue
            df = all_data[sym]
            mask = df['date_str'] == date_str
            if not mask.any():
                continue
            idx = df.index.get_loc(df[mask].index[0])
            bar_date = date_str
            h = df['high'].iloc[idx]
            l = df['low'].iloc[idx]
            c = df['close'].iloc[idx]
            trade = engine.check_exits(sym, idx, bar_date, h, l, c)
            if trade:
                closed_syms.append(sym)
                strat_name = pos_strategy.pop(sym, 'unknown')
                strat_counts[strat_name] = strat_counts.get(strat_name, 0) + 1

        # --- Phase 2: Check pending stop orders ---
        triggered_keys = []
        for key, pend in list(pending.items()):
            sym, direction = key
            if sym in engine.positions:
                # Already have a position in this symbol
                triggered_keys.append(key)
                continue
            if sym not in all_data:
                triggered_keys.append(key)
                continue
            df = all_data[sym]
            mask = df['date_str'] == date_str
            if not mask.any():
                triggered_keys.append(key)
                continue
            idx = df.index.get_loc(df[mask].index[0])
            h = df['high'].iloc[idx]
            l = df['low'].iloc[idx]

            triggered = False
            if direction == 'long' and h >= pend['stop_level']:
                sig = TradeSignal(
                    direction=Direction.LONG,
                    entry_price=pend['stop_level'],
                    stop_loss=pend['sl'],
                    target=pend['tp'],
                    max_hold_bars=pend['max_hold'],
                )
                if engine.open_position(sym, sig, idx, date_str):
                    pos_strategy[sym] = pend['strategy']
                triggered = True
            elif direction == 'short' and l <= pend['stop_level']:
                sig = TradeSignal(
                    direction=Direction.SHORT,
                    entry_price=pend['stop_level'],
                    stop_loss=pend['sl'],
                    target=pend['tp'],
                    max_hold_bars=pend['max_hold'],
                )
                if engine.open_position(sym, sig, idx, date_str):
                    pos_strategy[sym] = pend['strategy']
                triggered = True

            if triggered:
                triggered_keys.append(key)

        # Remove triggered/expired pending orders (all pending are 1-day only)
        # Clear ALL pending at end of day — they are single-day stop orders
        pending.clear()

        # --- Phase 3: Generate new pending signals for next bar ---
        for sym, df in all_data.items():
            mask = df['date_str'] == date_str
            if not mask.any():
                continue
            idx_loc = df.index.get_loc(df[mask].index[0])

            # InsideDay signals (pending for next bar)
            for sig_info in detect_inside_day_signals(sym, df, idx_loc):
                sig_type, stop_level, sl, tp, max_hold, strat = sig_info
                if sig_type == 'long_stop':
                    key = (sym, 'long')
                    if key not in pending:
                        pending[key] = {'stop_level': stop_level, 'sl': sl, 'tp': tp,
                                        'max_hold': max_hold, 'strategy': strat}
                elif sig_type == 'short_stop':
                    key = (sym, 'short')
                    if key not in pending:
                        pending[key] = {'stop_level': stop_level, 'sl': sl, 'tp': tp,
                                        'max_hold': max_hold, 'strategy': strat}

            # PA_MACD signals (pending for next bar)
            for sig_info in detect_pamacd_signals(sym, df, idx_loc):
                sig_type, stop_level, sl, tp, max_hold, strat = sig_info
                if sig_type == 'long_stop':
                    key = (sym, 'long')
                    if key not in pending:
                        pending[key] = {'stop_level': stop_level, 'sl': sl, 'tp': tp,
                                        'max_hold': max_hold, 'strategy': strat}
                elif sig_type == 'short_stop':
                    key = (sym, 'short')
                    if key not in pending:
                        pending[key] = {'stop_level': stop_level, 'sl': sl, 'tp': tp,
                                        'max_hold': max_hold, 'strategy': strat}

            # RangeBreakout signals (immediate — triggers on current bar)
            for sig_info in detect_range_breakout_signals(sym, df, idx_loc):
                sig_type, entry, sl, tp, max_hold, strat = sig_info
                if sym in engine.positions:
                    continue
                if sig_type == 'long_immediate':
                    sig = TradeSignal(
                        direction=Direction.LONG,
                        entry_price=entry,
                        stop_loss=sl,
                        target=tp,
                        max_hold_bars=max_hold,
                    )
                    if engine.open_position(sym, sig, idx_loc, date_str):
                        pos_strategy[sym] = strat
                elif sig_type == 'short_immediate':
                    sig = TradeSignal(
                        direction=Direction.SHORT,
                        entry_price=entry,
                        stop_loss=sl,
                        target=tp,
                        max_hold_bars=max_hold,
                    )
                    if engine.open_position(sym, sig, idx_loc, date_str):
                        pos_strategy[sym] = strat

        # Update equity at end of day
        prices = {}
        for sym in engine.positions:
            if sym in all_data:
                df = all_data[sym]
                mask = df['date_str'] == date_str
                if mask.any():
                    prices[sym] = df.loc[mask, 'close'].iloc[0]
        engine.update_equity(date_str, prices)

    # Count remaining open positions by strategy
    for sym in list(engine.positions.keys()):
        strat_name = pos_strategy.get(sym, 'unknown')
        # Force close at last available price
        if sym in all_data:
            df = all_data[sym]
            last_idx = len(df) - 1
            last_date = df['date_str'].iloc[last_idx]
            c = df['close'].iloc[last_idx]
            trade = engine.close_position(sym, c, last_idx, last_date, ExitType.EOD)
            if trade:
                strat_counts[strat_name] = strat_counts.get(strat_name, 0) + 1

    result = engine.get_results()
    return result, strat_counts


def run_split_pool(all_data, allocations, label):
    """Run strategies on separate engines with split capital, then merge results."""
    # allocations = {'InsideDay': (pct, max_pos), 'PA_MACD': (pct, max_pos), 'RangeBreak5d': (pct, max_pos)}
    engines = {}
    pos_strategies = {}
    pendings = {}

    for strat_name, (pct, max_pos) in allocations.items():
        cap = INITIAL_CAPITAL * pct
        engines[strat_name] = IntradayBacktestEngine(
            initial_capital=cap,
            position_size_pct=0.10,
            max_positions=max_pos,
            commission_pct=COMMISSION,
            slippage_pct=SLIPPAGE,
            mode='cash',
            fixed_sizing=True,
        )
        pendings[strat_name] = {}

    all_dates = set()
    for sym, df in all_data.items():
        all_dates.update(df['date_str'].tolist())
    all_dates = sorted(all_dates)

    strat_counts = {'InsideDay': 0, 'PA_MACD': 0, 'RangeBreak5d': 0}

    for date_str in all_dates:
        # Phase 1: Check exits for all engines
        for strat_name, eng in engines.items():
            for sym in list(eng.positions.keys()):
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
                trade = eng.check_exits(sym, idx, date_str, h, l, c)
                if trade:
                    strat_counts[strat_name] += 1

        # Phase 2: Check pending stop orders for each strategy
        for strat_name, eng in engines.items():
            pend_dict = pendings[strat_name]
            to_remove = []
            for key, pend in list(pend_dict.items()):
                sym, direction = key
                if sym in eng.positions:
                    to_remove.append(key)
                    continue
                if sym not in all_data:
                    to_remove.append(key)
                    continue
                df = all_data[sym]
                mask = df['date_str'] == date_str
                if not mask.any():
                    to_remove.append(key)
                    continue
                idx = df.index.get_loc(df[mask].index[0])
                h = df['high'].iloc[idx]
                l = df['low'].iloc[idx]

                triggered = False
                if direction == 'long' and h >= pend['stop_level']:
                    sig = TradeSignal(Direction.LONG, pend['stop_level'], pend['sl'],
                                      pend['tp'], max_hold_bars=pend['max_hold'])
                    if eng.open_position(sym, sig, idx, date_str):
                        pass
                    triggered = True
                elif direction == 'short' and l <= pend['stop_level']:
                    sig = TradeSignal(Direction.SHORT, pend['stop_level'], pend['sl'],
                                      pend['tp'], max_hold_bars=pend['max_hold'])
                    if eng.open_position(sym, sig, idx, date_str):
                        pass
                    triggered = True
                if triggered:
                    to_remove.append(key)
            # Clear all pending (1-day stop orders)
            pend_dict.clear()

        # Phase 3: Generate new pending signals
        for sym, df_sym in all_data.items():
            mask = df_sym['date_str'] == date_str
            if not mask.any():
                continue
            idx_loc = df_sym.index.get_loc(df_sym[mask].index[0])

            # InsideDay → InsideDay engine
            if 'InsideDay' in engines:
                for sig_info in detect_inside_day_signals(sym, df_sym, idx_loc):
                    sig_type, stop_level, sl, tp, max_hold, strat = sig_info
                    direction = 'long' if 'long' in sig_type else 'short'
                    key = (sym, direction)
                    if key not in pendings['InsideDay']:
                        pendings['InsideDay'][key] = {
                            'stop_level': stop_level, 'sl': sl, 'tp': tp,
                            'max_hold': max_hold, 'strategy': strat
                        }

            # PA_MACD → PA_MACD engine
            if 'PA_MACD' in engines:
                for sig_info in detect_pamacd_signals(sym, df_sym, idx_loc):
                    sig_type, stop_level, sl, tp, max_hold, strat = sig_info
                    direction = 'long' if 'long' in sig_type else 'short'
                    key = (sym, direction)
                    if key not in pendings['PA_MACD']:
                        pendings['PA_MACD'][key] = {
                            'stop_level': stop_level, 'sl': sl, 'tp': tp,
                            'max_hold': max_hold, 'strategy': strat
                        }

            # RangeBreakout → immediate entry on RangeBreak5d engine
            if 'RangeBreak5d' in engines:
                eng_rb = engines['RangeBreak5d']
                for sig_info in detect_range_breakout_signals(sym, df_sym, idx_loc):
                    sig_type, entry, sl, tp, max_hold, strat = sig_info
                    if sym in eng_rb.positions:
                        continue
                    if 'long' in sig_type:
                        sig = TradeSignal(Direction.LONG, entry, sl, tp, max_hold_bars=max_hold)
                        eng_rb.open_position(sym, sig, idx_loc, date_str)
                    elif 'short' in sig_type:
                        sig = TradeSignal(Direction.SHORT, entry, sl, tp, max_hold_bars=max_hold)
                        eng_rb.open_position(sym, sig, idx_loc, date_str)

        # Update equity for all engines
        for strat_name, eng in engines.items():
            prices = {}
            for sym in eng.positions:
                if sym in all_data:
                    df_s = all_data[sym]
                    mask = df_s['date_str'] == date_str
                    if mask.any():
                        prices[sym] = df_s.loc[mask, 'close'].iloc[0]
            eng.update_equity(date_str, prices)

    # Force close remaining positions
    for strat_name, eng in engines.items():
        for sym in list(eng.positions.keys()):
            if sym in all_data:
                df_s = all_data[sym]
                last_idx = len(df_s) - 1
                last_date = df_s['date_str'].iloc[last_idx]
                c = df_s['close'].iloc[last_idx]
                trade = eng.close_position(sym, c, last_idx, last_date, ExitType.EOD)
                if trade:
                    strat_counts[strat_name] += 1

    # Merge results: combine equity curves and trades
    # Build combined equity curve by summing all engines' equity at each date
    combined_eq = {}
    for strat_name, eng in engines.items():
        for d, eq in eng.equity_curve.items():
            combined_eq[d] = combined_eq.get(d, 0) + eq

    # Merge trades
    all_trades = []
    all_exit_reasons = {}
    for strat_name, eng in engines.items():
        res = eng.get_results()
        all_trades.extend(res.trades)
        for k, v in res.exit_reasons.items():
            all_exit_reasons[k] = all_exit_reasons.get(k, 0) + v

    # Build a combined result manually
    result = BacktestResult()
    result.trades = all_trades
    result.total_trades = len(all_trades)
    result.equity_curve = dict(sorted(combined_eq.items()))
    result.exit_reasons = all_exit_reasons

    if all_trades:
        wins = [t for t in all_trades if t.pnl > 0]
        losses = [t for t in all_trades if t.pnl <= 0]
        result.winning_trades = len(wins)
        result.losing_trades = len(losses)
        result.win_rate = len(wins) / len(all_trades) * 100

        result.total_pnl = sum(t.pnl for t in all_trades)
        result.total_pnl_pct = result.total_pnl / INITIAL_CAPITAL * 100

        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 1
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        result.avg_win = np.mean([t.pnl_pct for t in wins]) if wins else 0
        result.avg_loss = np.mean([t.pnl_pct for t in losses]) if losses else 0
        result.avg_rr = abs(result.avg_win / result.avg_loss) if result.avg_loss != 0 else 0
        result.avg_bars_held = np.mean([t.bars_held for t in all_trades])

        long_trades = [t for t in all_trades if t.direction == Direction.LONG]
        short_trades = [t for t in all_trades if t.direction == Direction.SHORT]
        result.long_trades = len(long_trades)
        result.short_trades = len(short_trades)
        result.long_win_rate = len([t for t in long_trades if t.pnl > 0]) / max(len(long_trades), 1) * 100
        result.short_win_rate = len([t for t in short_trades if t.pnl > 0]) / max(len(short_trades), 1) * 100

    # CAGR, Sharpe, Sortino, MaxDD from combined equity curve
    if len(combined_eq) >= 2:
        eq_dates = sorted(combined_eq.keys())
        eq_values = [combined_eq[d] for d in eq_dates]
        first_date = pd.Timestamp(eq_dates[0])
        last_date = pd.Timestamp(eq_dates[-1])
        years = max((last_date - first_date).days / 365.25, 0.1)
        final_val = eq_values[-1]
        if final_val > 0:
            result.cagr = ((final_val / INITIAL_CAPITAL) ** (1 / years) - 1) * 100

        eq_series = pd.Series(eq_values, index=pd.to_datetime(eq_dates))
        returns = eq_series.pct_change().dropna()

        # Max drawdown
        peak = eq_series.expanding().max()
        dd = (peak - eq_series) / peak
        result.max_drawdown = dd.max() * 100
        result.max_drawdown_pct = result.max_drawdown

        if len(returns) > 0 and returns.std() > 0:
            result.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)
            downside = returns[returns < 0]
            if len(downside) > 0 and downside.std() > 0:
                result.sortino_ratio = (returns.mean() / downside.std()) * np.sqrt(252)

        if result.max_drawdown > 0:
            result.calmar_ratio = result.cagr / result.max_drawdown

    # Yearly stats
    yearly_trades = {}
    for t in all_trades:
        yearly_trades.setdefault(t.year, []).append(t)
    for year, trades in sorted(yearly_trades.items()):
        y_wins = [t for t in trades if t.pnl > 0]
        y_pnl = sum(t.pnl for t in trades)
        y_gross_profit = sum(t.pnl for t in y_wins)
        y_gross_loss = abs(sum(t.pnl for t in trades if t.pnl <= 0))
        result.yearly_stats[year] = {
            'trades': len(trades),
            'wins': len(y_wins),
            'win_rate': len(y_wins) / len(trades) * 100 if trades else 0,
            'pnl': y_pnl,
            'pnl_pct': y_pnl / INITIAL_CAPITAL * 100,
            'profit_factor': y_gross_profit / y_gross_loss if y_gross_loss > 0 else float('inf'),
            'long_trades': len([t for t in trades if t.direction == Direction.LONG]),
            'short_trades': len([t for t in trades if t.direction == Direction.SHORT]),
        }

    return result, strat_counts


def result_to_row(label, result, strat_counts):
    """Convert result to CSV row dict."""
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
        'sharpe': round(result.sharpe_ratio, 2),
        'sortino': round(result.sortino_ratio, 2),
        'calmar': round(result.calmar_ratio, 2),
        'total_pnl': round(result.total_pnl, 0),
        'pnl_pct': round(result.total_pnl_pct, 2),
        'avg_win': round(result.avg_win, 2),
        'avg_loss': round(result.avg_loss, 2),
        'avg_rr': round(result.avg_rr, 2),
        'exit_reasons': str(result.exit_reasons),
    }
    for y in range(2018, 2026):
        if y in result.yearly_stats:
            row[f'y{y}_pnl_pct'] = round(result.yearly_stats[y]['pnl_pct'], 2)
        else:
            row[f'y{y}_pnl_pct'] = 0.0
    return row


def main():
    print('Loading data for 50 F&O stocks...', flush=True)
    t0 = time.time()
    all_data = load_data_from_db(FNO_STOCKS, 'day', '2018-01-01', '2025-12-31')
    print(f'Loaded {len(all_data)} symbols in {time.time() - t0:.1f}s', flush=True)

    # Precompute indicators
    print('Computing indicators (ATR14, MACD)...', flush=True)
    for sym, df in all_data.items():
        df['atr14'] = calc_atr(df, 14)
        macd_line, macd_signal, macd_hist = calc_macd(df['close'])
        df['macd_hist'] = macd_hist
    print('Indicators ready.', flush=True)

    # Write CSV header
    with open(OUTPUT_CSV, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    # Skip already-done configs
    done = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV) as f:
            done = {row['label'] for row in csv.DictReader(f)}

    configs = [
        # (label, type, params)
        ('COMBINED_EQUAL', 'split', {
            'InsideDay': (0.333, 3),
            'PA_MACD': (0.333, 3),
            'RangeBreak5d': (0.334, 3),
        }),
        ('COMBINED_WEIGHT_ID', 'split', {
            'InsideDay': (0.50, 5),
            'PA_MACD': (0.25, 2),
            'RangeBreak5d': (0.25, 3),
        }),
        ('COMBINED_10POS', 'shared', 10),
        ('COMBINED_15POS', 'shared', 15),
        ('COMBINED_20POS', 'shared', 20),
    ]

    for i, (label, cfg_type, params) in enumerate(configs):
        if label in done:
            print(f'[{i+1}/{len(configs)}] {label} — SKIP (already done)', flush=True)
            continue

        print(f'[{i+1}/{len(configs)}] {label} ...', end='', flush=True)
        t1 = time.time()

        if cfg_type == 'split':
            result, strat_counts = run_split_pool(all_data, params, label)
        else:
            result, strat_counts = run_shared_pool(all_data, params, label)

        elapsed = time.time() - t1
        print(f' {elapsed:.0f}s | CAGR={result.cagr:.2f}% MaxDD={result.max_drawdown:.2f}% '
              f'Sharpe={result.sharpe_ratio:.2f} Trades={result.total_trades}', flush=True)

        # Print detailed results
        IntradayBacktestEngine.print_results(result, label)
        print(f'  Strategy breakdown: ID={strat_counts.get("InsideDay", 0)} '
              f'PAMACD={strat_counts.get("PA_MACD", 0)} '
              f'RB={strat_counts.get("RangeBreak5d", 0)}', flush=True)

        # Write CSV row
        row = result_to_row(label, result, strat_counts)
        with open(OUTPUT_CSV, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

    print(f'\nAll configs done. Results saved to {OUTPUT_CSV}', flush=True)


if __name__ == '__main__':
    main()
