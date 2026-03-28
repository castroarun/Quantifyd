#!/usr/bin/env python3
"""
InsideDay Per-Instrument Analysis — BASELINE (TP3, MH5)
=========================================================
Original InsideDay strategy: TP=3x, SL=2xATR, MaxHold=5.
Finds which F&O instruments have the best win rates and profit factors.
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
from services.technical_indicators import calc_atr

OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'insideday_per_instrument_baseline.csv')

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


def detect_inside_day(sym, df, i):
    """Original InsideDay: TP=3x, SL=2xATR, MaxHold=5."""
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
    sl = entry_stop - 2.0 * atr_val
    risk = entry_stop - sl
    tp = entry_stop + 3.0 * risk
    signals.append(('long_stop', entry_stop, sl, tp, 5, 'InsideDay'))
    # SHORT
    entry_stop_s = l_prev
    sl_s = entry_stop_s + 2.0 * atr_val
    risk_s = sl_s - entry_stop_s
    tp_s = entry_stop_s - 3.0 * risk_s
    signals.append(('short_stop', entry_stop_s, sl_s, tp_s, 5, 'InsideDay'))
    return signals


def run_per_instrument(all_data):
    engine = IntradayBacktestEngine(
        initial_capital=INITIAL_CAPITAL, position_size_pct=0.10,
        max_positions=50, commission_pct=COMMISSION,
        slippage_pct=SLIPPAGE, mode='cash', fixed_sizing=True,
    )
    all_dates = sorted({d for df in all_data.values() for d in df['date_str'].tolist()})
    pending = {}
    sym_trades = defaultdict(list)

    for date_str in all_dates:
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

        for sym, df in all_data.items():
            mask = df['date_str'] == date_str
            if not mask.any(): continue
            idx_loc = df.index.get_loc(df[mask].index[0])
            for sig_info in detect_inside_day(sym, df, idx_loc):
                sig_type, stop_level, sl, tp, max_hold, strat = sig_info
                direction = 'long' if 'long' in sig_type else 'short'
                key = (sym, direction)
                if key not in pending:
                    pending[key] = {'stop_level': stop_level, 'sl': sl, 'tp': tp,
                                    'max_hold': max_hold, 'strategy': strat}

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

    print('Computing ATR14...', flush=True)
    for sym, df in all_data.items():
        df['atr14'] = calc_atr(df, 14)
    print('Ready.\n', flush=True)

    print('Running InsideDay BASELINE (TP3 MH5) per-instrument...', flush=True)
    t1 = time.time()
    sym_trades = run_per_instrument(all_data)
    print(f'Done in {time.time() - t1:.0f}s\n', flush=True)

    rows = []
    for sym in sorted(sym_trades.keys()):
        trades = sym_trades[sym]
        if not trades: continue
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
        long_wr = len([t for t in long_trades if t.pnl > 0]) / max(len(long_trades), 1) * 100
        short_wr = len([t for t in short_trades if t.pnl > 0]) / max(len(short_trades), 1) * 100

        exit_reasons = defaultdict(int)
        for t in trades:
            exit_reasons[str(t.exit_type).split('.')[-1]] += 1

        rows.append({
            'symbol': sym, 'total_trades': total, 'wins': len(wins), 'losses': len(losses),
            'win_rate': round(win_rate, 1), 'profit_factor': round(pf, 2),
            'total_pnl': round(total_pnl, 0),
            'avg_win_pct': round(avg_win_pct, 2), 'avg_loss_pct': round(avg_loss_pct, 2),
            'avg_rr': round(avg_rr, 2),
            'long_trades': len(long_trades), 'long_wr': round(long_wr, 1),
            'short_trades': len(short_trades), 'short_wr': round(short_wr, 1),
            'tp_hits': exit_reasons.get('FIXED_TP', 0),
            'sl_hits': exit_reasons.get('FIXED_SL', 0),
            'max_hold_exits': exit_reasons.get('MAX_HOLD', 0),
        })

    rows.sort(key=lambda r: r['profit_factor'], reverse=True)

    fieldnames = list(rows[0].keys())
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Print results
    print(f'{"Symbol":<15} {"Trades":>6} {"WR%":>6} {"PF":>6} {"PnL":>12} {"LongWR":>7} {"ShortWR":>8} {"TP":>4} {"SL":>4} {"MH":>4}')
    print('-' * 100)
    for r in rows:
        print(f'{r["symbol"]:<15} {r["total_trades"]:>6} {r["win_rate"]:>5.1f}% {r["profit_factor"]:>6.2f} '
              f'{r["total_pnl"]:>12,.0f} {r["long_wr"]:>6.1f}% {r["short_wr"]:>7.1f}% '
              f'{r["tp_hits"]:>4} {r["sl_hits"]:>4} {r["max_hold_exits"]:>4}')

    print('\n=== TOP 15 by Profit Factor (min 20 trades) ===')
    top = [r for r in rows if r['total_trades'] >= 20][:15]
    for i, r in enumerate(top):
        print(f'{i+1:>2}. {r["symbol"]:<15} PF={r["profit_factor"]:.2f}  WR={r["win_rate"]:.1f}%  '
              f'Trades={r["total_trades"]}  PnL={r["total_pnl"]:>10,.0f}  RR={r["avg_rr"]:.2f}  '
              f'TP={r["tp_hits"]} SL={r["sl_hits"]} MH={r["max_hold_exits"]}')

    print('\n=== BOTTOM 10 (worst performers) ===')
    bottom = sorted([r for r in rows if r['total_trades'] >= 20], key=lambda r: r['profit_factor'])
    for i, r in enumerate(bottom[:10]):
        print(f'{i+1:>2}. {r["symbol"]:<15} PF={r["profit_factor"]:.2f}  WR={r["win_rate"]:.1f}%  '
              f'Trades={r["total_trades"]}  PnL={r["total_pnl"]:>10,.0f}')

    total_t = sum(r['total_trades'] for r in rows)
    total_w = sum(r['wins'] for r in rows)
    profitable = len([r for r in rows if r['profit_factor'] >= 1.0 and r['total_trades'] >= 20])
    strong = len([r for r in rows if r['profit_factor'] >= 1.5 and r['total_trades'] >= 20])
    weak = len([r for r in rows if r['profit_factor'] < 0.8 and r['total_trades'] >= 20])
    print(f'\n=== SUMMARY ===')
    print(f'Total: {total_t} trades, {total_w} wins ({total_w/total_t*100:.1f}% WR)')
    print(f'Profitable (PF>=1.0): {profitable}/49 | Strong (PF>=1.5): {strong} | Weak (PF<0.8): {weak}')
    print(f'\nSaved to {OUTPUT_CSV}', flush=True)


if __name__ == '__main__':
    main()
