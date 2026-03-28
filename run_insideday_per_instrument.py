#!/usr/bin/env python3
"""
InsideDay Per-Instrument Analysis
===================================
Runs ID_TP2_MH3 (best config) and tracks per-symbol win rate, P&L, trade count
to identify which instruments work best with this strategy.
"""
import csv, os, sys, time, logging
import numpy as np
import pandas as pd
from collections import defaultdict

logging.disable(logging.WARNING)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from services.intraday_backtest_engine import (
    IntradayBacktestEngine, TradeSignal, Direction, ExitType,
    load_data_from_db, BacktestResult,
)
from services.technical_indicators import calc_macd, calc_atr

OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'insideday_per_instrument.csv')

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

# Best InsideDay config
ID_PARAMS = {'tp_mult': 2.0, 'sl_mult': 2.0, 'max_hold': 3}


def detect_inside_day_v2(sym, df, i, tp_mult=2.0, sl_mult=2.0, max_hold=3):
    """InsideDay with TP2 MH3 params."""
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
    # LONG
    entry_stop = h_prev
    sl = entry_stop - sl_mult * atr_val
    risk = entry_stop - sl
    tp = entry_stop + tp_mult * risk
    signals.append(('long_stop', entry_stop, sl, tp, max_hold, 'InsideDay'))
    # SHORT
    entry_stop_s = l_prev
    sl_s = entry_stop_s + sl_mult * atr_val
    risk_s = sl_s - entry_stop_s
    tp_s = entry_stop_s - tp_mult * risk_s
    signals.append(('short_stop', entry_stop_s, sl_s, tp_s, max_hold, 'InsideDay'))
    return signals


def run_per_instrument(all_data):
    """Run InsideDay ID_TP2_MH3 standalone and track per-instrument stats."""
    engine = IntradayBacktestEngine(
        initial_capital=INITIAL_CAPITAL, position_size_pct=0.10,
        max_positions=50,  # No position limit — we want to see all instruments
        commission_pct=COMMISSION, slippage_pct=SLIPPAGE,
        mode='cash', fixed_sizing=True,
    )
    all_dates = sorted({d for df in all_data.values() for d in df['date_str'].tolist()})
    pending = {}

    # Per-instrument tracking
    sym_trades = defaultdict(list)  # sym -> list of trade objects

    for date_str in all_dates:
        # Phase 1: Exits
        for sym in list(engine.positions.keys()):
            if sym not in all_data: continue
            df = all_data[sym]
            mask = df['date_str'] == date_str
            if not mask.any(): continue
            idx = df.index.get_loc(df[mask].index[0])
            trade = engine.check_exits(sym, idx, date_str,
                                        df['high'].iloc[idx], df['low'].iloc[idx], df['close'].iloc[idx])
            if trade:
                sym_trades[sym].append(trade)

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
                sig = TradeSignal(Direction.LONG, pend['stop_level'], pend['sl'], pend['tp'],
                                  max_hold_bars=pend['max_hold'])
                engine.open_position(sym, sig, idx, date_str)
            elif direction == 'short' and l <= pend['stop_level']:
                sig = TradeSignal(Direction.SHORT, pend['stop_level'], pend['sl'], pend['tp'],
                                  max_hold_bars=pend['max_hold'])
                engine.open_position(sym, sig, idx, date_str)
        pending.clear()

        # Phase 3: Generate signals
        for sym, df in all_data.items():
            mask = df['date_str'] == date_str
            if not mask.any(): continue
            idx_loc = df.index.get_loc(df[mask].index[0])
            for sig_info in detect_inside_day_v2(sym, df, idx_loc, **ID_PARAMS):
                sig_type, stop_level, sl, tp, max_hold, strat = sig_info
                direction = 'long' if 'long' in sig_type else 'short'
                key = (sym, direction)
                if key not in pending:
                    pending[key] = {'stop_level': stop_level, 'sl': sl, 'tp': tp,
                                    'max_hold': max_hold, 'strategy': strat}

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
        if sym in all_data:
            df_s = all_data[sym]
            trade = engine.close_position(sym, df_s['close'].iloc[-1], len(df_s) - 1,
                                           df_s['date_str'].iloc[-1], ExitType.EOD)
            if trade:
                sym_trades[sym].append(trade)

    return sym_trades


def main():
    print('Loading data for 50 F&O stocks...', flush=True)
    t0 = time.time()
    all_data = load_data_from_db(FNO_STOCKS, 'day', '2018-01-01', '2025-12-31')
    print(f'Loaded {len(all_data)} symbols in {time.time() - t0:.1f}s', flush=True)

    print('Computing indicators (ATR14)...', flush=True)
    for sym, df in all_data.items():
        df['atr14'] = calc_atr(df, 14)
    print('Indicators ready.\n', flush=True)

    print('Running InsideDay ID_TP2_MH3 per-instrument analysis...', flush=True)
    t1 = time.time()
    sym_trades = run_per_instrument(all_data)
    elapsed = time.time() - t1
    print(f'Done in {elapsed:.0f}s\n', flush=True)

    # Build per-instrument stats
    rows = []
    for sym in sorted(sym_trades.keys()):
        trades = sym_trades[sym]
        if not trades:
            continue
        total = len(trades)
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        win_rate = len(wins) / total * 100

        total_pnl = sum(t.pnl for t in trades)
        avg_win_pct = np.mean([t.pnl_pct for t in wins]) if wins else 0
        avg_loss_pct = np.mean([t.pnl_pct for t in losses]) if losses else 0
        avg_rr = abs(avg_win_pct / avg_loss_pct) if avg_loss_pct != 0 else 0

        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 1
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        long_trades = [t for t in trades if t.direction == Direction.LONG]
        short_trades = [t for t in trades if t.direction == Direction.SHORT]
        long_wins = len([t for t in long_trades if t.pnl > 0])
        short_wins = len([t for t in short_trades if t.pnl > 0])
        long_wr = long_wins / len(long_trades) * 100 if long_trades else 0
        short_wr = short_wins / len(short_trades) * 100 if short_trades else 0

        # Exit reason breakdown
        exit_reasons = defaultdict(int)
        for t in trades:
            exit_reasons[str(t.exit_type).split('.')[-1]] += 1

        tp_hits = exit_reasons.get('FIXED_TP', 0)
        sl_hits = exit_reasons.get('FIXED_SL', 0)
        mh_exits = exit_reasons.get('MAX_HOLD', 0)

        rows.append({
            'symbol': sym,
            'total_trades': total,
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': round(win_rate, 1),
            'profit_factor': round(pf, 2),
            'total_pnl': round(total_pnl, 0),
            'avg_win_pct': round(avg_win_pct, 2),
            'avg_loss_pct': round(avg_loss_pct, 2),
            'avg_rr': round(avg_rr, 2),
            'long_trades': len(long_trades),
            'long_wr': round(long_wr, 1),
            'short_trades': len(short_trades),
            'short_wr': round(short_wr, 1),
            'tp_hits': tp_hits,
            'sl_hits': sl_hits,
            'max_hold_exits': mh_exits,
        })

    # Sort by profit factor descending
    rows.sort(key=lambda r: r['profit_factor'], reverse=True)

    # Write CSV
    fieldnames = list(rows[0].keys())
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Print summary
    print(f'{"Symbol":<15} {"Trades":>6} {"WR%":>6} {"PF":>6} {"PnL":>12} {"AvgWin%":>8} {"AvgLoss%":>9} {"RR":>5} {"LongWR":>7} {"ShortWR":>8} {"TP":>4} {"SL":>4} {"MH":>4}')
    print('─' * 120)

    for r in rows:
        pnl_str = f'{r["total_pnl"]:>12,.0f}'
        print(f'{r["symbol"]:<15} {r["total_trades"]:>6} {r["win_rate"]:>5.1f}% {r["profit_factor"]:>6.2f} '
              f'{pnl_str} {r["avg_win_pct"]:>7.2f}% {r["avg_loss_pct"]:>8.2f}% {r["avg_rr"]:>5.2f} '
              f'{r["long_wr"]:>6.1f}% {r["short_wr"]:>7.1f}% {r["tp_hits"]:>4} {r["sl_hits"]:>4} {r["max_hold_exits"]:>4}')

    print(f'\n─── TOP 15 by Profit Factor (min 20 trades) ───')
    top = [r for r in rows if r['total_trades'] >= 20][:15]
    for i, r in enumerate(top):
        print(f'{i+1:>2}. {r["symbol"]:<15} PF={r["profit_factor"]:.2f}  WR={r["win_rate"]:.1f}%  '
              f'Trades={r["total_trades"]}  PnL={r["total_pnl"]:>10,.0f}  RR={r["avg_rr"]:.2f}')

    print(f'\n─── BOTTOM 10 (worst performers) ───')
    bottom = [r for r in rows if r['total_trades'] >= 20]
    bottom.sort(key=lambda r: r['profit_factor'])
    for i, r in enumerate(bottom[:10]):
        print(f'{i+1:>2}. {r["symbol"]:<15} PF={r["profit_factor"]:.2f}  WR={r["win_rate"]:.1f}%  '
              f'Trades={r["total_trades"]}  PnL={r["total_pnl"]:>10,.0f}  RR={r["avg_rr"]:.2f}')

    # Overall stats
    total_trades = sum(r['total_trades'] for r in rows)
    total_wins = sum(r['wins'] for r in rows)
    total_pnl = sum(r['total_pnl'] for r in rows)
    print(f'\n─── OVERALL ───')
    print(f'Total trades: {total_trades}  Wins: {total_wins}  WR: {total_wins/total_trades*100:.1f}%')
    print(f'Total PnL: {total_pnl:,.0f}')
    print(f'Instruments with PF > 1.5: {len([r for r in rows if r["profit_factor"] > 1.5 and r["total_trades"] >= 20])}')
    print(f'Instruments with PF < 0.8: {len([r for r in rows if r["profit_factor"] < 0.8 and r["total_trades"] >= 20])}')

    print(f'\nResults saved to {OUTPUT_CSV}', flush=True)


if __name__ == '__main__':
    main()
