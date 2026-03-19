#!/usr/bin/env python3
"""
CPR Breakout Research Lab - V3 (Signal-first approach)
======================================================
Key insight: CPR is always below breakout price for longs (pivot < prev_high always).
So for LONGS, the real filter is: narrow CPR day + close > prev_high.
For SHORTS: narrow CPR + close < prev_low + CPR above price.

Approach:
1. Find ALL breakout events (first candle per symbol per day that breaks prev H/L)
2. Find the EOD close for that day (exit price)
3. Find intraday high/low after entry (for SL/TP/trail calculations)
4. Filter/score by CPR width, buffer, time, volume
5. Apply exit strategies analytically (no tick-by-tick simulation needed for most exits)

For trailing stops we need tick simulation, but for SL/TP/EOD we can vectorize.
"""

import sys, logging, csv, os, time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List

sys.path.insert(0, str(Path(__file__).parent))
logging.disable(logging.WARNING)

import sqlite3

DB_PATH = Path(__file__).parent / 'backtest_data' / 'market_data.db'

SYMBOLS = [
    'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
    'SBIN', 'BHARTIARTL', 'ITC', 'KOTAKBANK', 'HINDUNILVR'
]

START_DATE = '2024-01-01'
END_DATE = '2025-10-27'
INITIAL_CAPITAL = 1_000_000
SLIPPAGE_PCT = 0.05
BROKERAGE = 40

LOT_SIZES = {
    'RELIANCE': 250, 'TCS': 150, 'HDFCBANK': 550, 'INFY': 300,
    'ICICIBANK': 700, 'HINDUNILVR': 300, 'ITC': 1600, 'SBIN': 750,
    'BHARTIARTL': 475, 'KOTAKBANK': 400,
}


def build_signal_table():
    """
    For each symbol, for each day:
    - Compute daily CPR
    - Scan 5-min candles for FIRST close above prev_high (LONG) or below prev_low (SHORT)
    - Record entry time, entry price, and post-entry price trajectory to EOD

    Returns a DataFrame with one row per signal event.
    """
    conn = sqlite3.connect(DB_PATH)
    all_signals = []

    for sym in SYMBOLS:
        print(f"  {sym}...", end='', flush=True)

        # Daily data
        df_d = pd.read_sql_query(
            "SELECT date, open, high, low, close FROM market_data_unified "
            "WHERE symbol=? AND timeframe='day' AND date>=? AND date<=? ORDER BY date",
            conn, params=(sym, '2023-12-01', END_DATE))
        df_d['date'] = pd.to_datetime(df_d['date']).dt.date
        df_d['prev_high'] = df_d['high'].shift(1)
        df_d['prev_low'] = df_d['low'].shift(1)
        df_d['prev_close'] = df_d['close'].shift(1)
        df_d['pivot'] = (df_d['prev_high'] + df_d['prev_low'] + df_d['prev_close']) / 3
        df_d['bc'] = (df_d['prev_high'] + df_d['prev_low']) / 2
        df_d['tc'] = 2 * df_d['pivot'] - df_d['bc']
        df_d['cpr_width'] = abs(df_d['tc'] - df_d['bc']) / df_d['pivot'] * 100
        df_d = df_d.dropna()
        cpr_lookup = df_d.set_index('date').to_dict('index')

        # 5-min data
        df_5 = pd.read_sql_query(
            "SELECT date, open, high, low, close, volume FROM market_data_unified "
            "WHERE symbol=? AND timeframe='5minute' AND date>=? AND date<=? ORDER BY date",
            conn, params=(sym, START_DATE, END_DATE + ' 23:59:59'))
        df_5['datetime'] = pd.to_datetime(df_5['date'])
        df_5['trade_date'] = df_5['datetime'].dt.date
        df_5['time_str'] = df_5['datetime'].dt.strftime('%H:%M')

        # Volume rolling average
        df_5['vol_avg'] = df_5['volume'].rolling(20, min_periods=5).mean()
        df_5['vol_ratio'] = df_5['volume'] / df_5['vol_avg'].replace(0, np.nan)

        # Group by trade date
        for td, day_df in df_5.groupby('trade_date'):
            if td not in cpr_lookup:
                continue

            cpr = cpr_lookup[td]
            day_df = day_df.sort_values('datetime')

            # Find first LONG breakout (close > prev_high)
            for _, candle in day_df.iterrows():
                ts = candle['time_str']
                if ts < "09:20" or ts >= "14:45":
                    continue

                close = candle['close']

                # LONG signal
                if close > cpr['prev_high']:
                    # Get all remaining candles after entry until EOD
                    remaining = day_df[day_df['datetime'] > candle['datetime']]
                    eod_candles = remaining[remaining['time_str'] >= '15:20']
                    eod_price = eod_candles.iloc[0]['close'] if len(eod_candles) > 0 else (
                        remaining.iloc[-1]['close'] if len(remaining) > 0 else close)

                    # Post-entry max high and min low (for SL/TP checking)
                    post_entry_high = remaining['high'].max() if len(remaining) > 0 else close
                    post_entry_low = remaining['low'].min() if len(remaining) > 0 else close

                    # High watermark progression (for trailing stops) — sample at key points
                    # Get running max high after entry
                    if len(remaining) > 0:
                        running_max = remaining['high'].cummax()
                        # Find first point where low drops below trail from running max
                        # for various trail pcts
                        trail_info = {}
                        for trail_pct in [0.3, 0.5, 0.75, 1.0]:
                            trail_levels = running_max * (1 - trail_pct / 100)
                            triggered = remaining[remaining['low'] <= trail_levels]
                            if len(triggered) > 0:
                                first_trigger = triggered.iloc[0]
                                trail_info[trail_pct] = {
                                    'time': first_trigger['time_str'],
                                    'price': float(trail_levels.loc[triggered.index[0]]),
                                    'datetime': first_trigger['datetime'],
                                }
                    else:
                        trail_info = {}

                    all_signals.append({
                        'symbol': sym,
                        'trade_date': td,
                        'direction': 'LONG',
                        'entry_time': candle['datetime'],
                        'entry_time_str': ts,
                        'entry_close': close,
                        'entry_volume': candle['volume'],
                        'entry_vol_ratio': candle['vol_ratio'] if not pd.isna(candle['vol_ratio']) else 0,
                        'prev_high': cpr['prev_high'],
                        'prev_low': cpr['prev_low'],
                        'cpr_width': cpr['cpr_width'],
                        'pivot': cpr['pivot'],
                        'tc': cpr['tc'],
                        'bc': cpr['bc'],
                        'eod_price': eod_price,
                        'post_high': post_entry_high,
                        'post_low': post_entry_low,
                        'trail_info': trail_info,
                    })
                    break  # First breakout only

            # Find first SHORT breakout (close < prev_low)
            for _, candle in day_df.iterrows():
                ts = candle['time_str']
                if ts < "09:20" or ts >= "14:45":
                    continue

                close = candle['close']

                if close < cpr['prev_low'] and min(cpr['pivot'], cpr['tc'], cpr['bc']) > close:
                    remaining = day_df[day_df['datetime'] > candle['datetime']]
                    eod_candles = remaining[remaining['time_str'] >= '15:20']
                    eod_price = eod_candles.iloc[0]['close'] if len(eod_candles) > 0 else (
                        remaining.iloc[-1]['close'] if len(remaining) > 0 else close)

                    post_entry_high = remaining['high'].max() if len(remaining) > 0 else close
                    post_entry_low = remaining['low'].min() if len(remaining) > 0 else close

                    trail_info = {}
                    if len(remaining) > 0:
                        running_min = remaining['low'].cummin()
                        for trail_pct in [0.3, 0.5, 0.75, 1.0]:
                            trail_levels = running_min * (1 + trail_pct / 100)
                            triggered = remaining[remaining['high'] >= trail_levels]
                            if len(triggered) > 0:
                                first_trigger = triggered.iloc[0]
                                trail_info[trail_pct] = {
                                    'time': first_trigger['time_str'],
                                    'price': float(trail_levels.loc[triggered.index[0]]),
                                    'datetime': first_trigger['datetime'],
                                }

                    all_signals.append({
                        'symbol': sym,
                        'trade_date': td,
                        'direction': 'SHORT',
                        'entry_time': candle['datetime'],
                        'entry_time_str': ts,
                        'entry_close': close,
                        'entry_volume': candle['volume'],
                        'entry_vol_ratio': candle['vol_ratio'] if not pd.isna(candle['vol_ratio']) else 0,
                        'prev_high': cpr['prev_high'],
                        'prev_low': cpr['prev_low'],
                        'cpr_width': cpr['cpr_width'],
                        'pivot': cpr['pivot'],
                        'tc': cpr['tc'],
                        'bc': cpr['bc'],
                        'eod_price': eod_price,
                        'post_high': post_entry_high,
                        'post_low': post_entry_low,
                        'trail_info': trail_info,
                    })
                    break

        print(f" {len([s for s in all_signals if s['symbol']==sym])} signals")

    conn.close()
    return all_signals


def evaluate_config(signals, cpr_thresh, buffer_pct, entry_end, require_vol, vol_mult, exit_strat):
    """
    Evaluate a config by filtering signals and computing PnL.
    This is FAST because we're not iterating 5-min candles — just filtering a small signal list.
    """
    filtered = []
    for s in signals:
        # CPR width filter
        if s['cpr_width'] >= cpr_thresh:
            continue
        # Time window filter
        if s['entry_time_str'] >= entry_end:
            continue
        # Buffer filter
        if s['direction'] == 'LONG':
            trigger = s['prev_high'] * (1 + buffer_pct / 100)
            if s['entry_close'] <= trigger:
                continue
        else:
            trigger = s['prev_low'] * (1 - buffer_pct / 100)
            if s['entry_close'] >= trigger:
                continue
        # Volume filter
        if require_vol and s['entry_vol_ratio'] < vol_mult:
            continue
        filtered.append(s)

    if not filtered:
        return {'total_trades': 0, 'win_rate': 0, 'total_pnl': 0, 'profit_factor': 0,
                'avg_pnl_pct': 0, 'max_dd': 0, 'exit_reasons': {}}

    # Compute PnL for each signal
    sl_pct = exit_strat.get('sl_pct', 999)
    tp_pct = exit_strat.get('tp_pct', 999)
    trail_pct = exit_strat.get('trail_pct', None)

    results = []
    exit_reasons = {}

    for s in filtered:
        entry_px = s['entry_close']
        direction = s['direction']

        # Apply slippage to entry
        if direction == 'LONG':
            entry_px *= (1 + SLIPPAGE_PCT / 100)
        else:
            entry_px *= (1 - SLIPPAGE_PCT / 100)

        # Determine exit
        exit_px = None
        exit_reason = 'EOD'

        if direction == 'LONG':
            sl_price = entry_px * (1 - sl_pct / 100)
            tp_price = entry_px * (1 + tp_pct / 100)

            # Check SL first (worst case)
            if s['post_low'] <= sl_price:
                exit_px = sl_price
                exit_reason = 'SL'
            # Check TP
            elif s['post_high'] >= tp_price:
                # But did SL hit first? Conservative: if both could hit, assume SL
                # In reality we'd need tick data, but approximate: if SL level wasn't breached, TP hit
                exit_px = tp_price
                exit_reason = 'TARGET'

            # Trailing stop check
            if trail_pct and exit_reason == 'EOD':
                if trail_pct in s.get('trail_info', {}):
                    trail_data = s['trail_info'][trail_pct]
                    exit_px = trail_data['price']
                    exit_reason = 'TRAIL'

            # EOD exit
            if exit_reason == 'EOD':
                exit_px = s['eod_price'] * (1 - SLIPPAGE_PCT / 100)

            # Apply slippage on non-limit exits
            if exit_reason in ('SL', 'TRAIL'):
                exit_px *= (1 - SLIPPAGE_PCT / 100)

            pnl_pct = (exit_px - entry_px) / entry_px * 100

        else:  # SHORT
            sl_price = entry_px * (1 + sl_pct / 100)
            tp_price = entry_px * (1 - tp_pct / 100)

            if s['post_high'] >= sl_price:
                exit_px = sl_price
                exit_reason = 'SL'
            elif s['post_low'] <= tp_price:
                exit_px = tp_price
                exit_reason = 'TARGET'

            if trail_pct and exit_reason == 'EOD':
                if trail_pct in s.get('trail_info', {}):
                    trail_data = s['trail_info'][trail_pct]
                    exit_px = trail_data['price']
                    exit_reason = 'TRAIL'

            if exit_reason == 'EOD':
                exit_px = s['eod_price'] * (1 + SLIPPAGE_PCT / 100)

            if exit_reason in ('SL', 'TRAIL'):
                exit_px *= (1 + SLIPPAGE_PCT / 100)

            pnl_pct = (entry_px - exit_px) / entry_px * 100

        # Approximate PnL in Rs (using notional ~ 5% of 10L = 50K)
        notional = 50000
        pnl_rs = pnl_pct / 100 * notional - BROKERAGE

        results.append({
            'pnl_pct': pnl_pct,
            'pnl_rs': pnl_rs,
            'exit_reason': exit_reason,
            'symbol': s['symbol'],
            'entry_time': s['entry_time'],
            'entry_price': entry_px,
            'exit_price': exit_px,
            'direction': direction,
        })

        exit_reasons[exit_reason] = exit_reasons.get(exit_reason, 0) + 1

    winners = [r for r in results if r['pnl_rs'] > 0]
    losers = [r for r in results if r['pnl_rs'] <= 0]
    gw = sum(r['pnl_rs'] for r in winners) if winners else 0
    gl = abs(sum(r['pnl_rs'] for r in losers)) if losers else 1

    return {
        'total_trades': len(results),
        'winners': len(winners),
        'losers': len(losers),
        'win_rate': len(winners) / len(results) * 100 if results else 0,
        'total_pnl': sum(r['pnl_rs'] for r in results),
        'avg_pnl_pct': np.mean([r['pnl_pct'] for r in results]) if results else 0,
        'profit_factor': gw / gl if gl > 0 else 999,
        'max_winner': max([r['pnl_pct'] for r in results], default=0),
        'max_loser': min([r['pnl_pct'] for r in results], default=0),
        'exit_reasons': exit_reasons,
        '_results': results,
    }


def main():
    print("=" * 100)
    print("CPR BREAKOUT RESEARCH LAB - V3 (Signal-First)")
    print("=" * 100)

    # Step 1: Build signal table
    print("\nBuilding signal table...")
    t0 = time.time()
    signals = build_signal_table()
    print(f"  Total signals: {len(signals)} ({time.time()-t0:.1f}s)")

    longs = [s for s in signals if s['direction'] == 'LONG']
    shorts = [s for s in signals if s['direction'] == 'SHORT']
    print(f"  LONG: {len(longs)} | SHORT: {len(shorts)}")

    # Distribution of CPR widths for signals
    widths = [s['cpr_width'] for s in signals]
    print(f"  CPR width range: {min(widths):.3f}% - {max(widths):.3f}%")
    print(f"  CPR < 0.5%: {sum(1 for w in widths if w < 0.5)} signals")
    print(f"  CPR < 1.0%: {sum(1 for w in widths if w < 1.0)} signals")

    # Step 2: Phase 1 — Entry parameter sweep with EOD exit
    print("\n" + "=" * 100)
    print("PHASE 1: Entry Sweep (EOD-only exit)")
    print("=" * 100)

    eod = {'sl_pct': 999, 'tp_pct': 999}

    phase1 = []
    for cpr in [0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]:
        for buf in [0.0, 0.1, 0.2, 0.3, 0.5]:
            for end_label, ee in [('11:15', '11:15'), ('12:30', '12:30'), ('14:45', '14:45')]:
                for vol in [False, True]:
                    summary = evaluate_config(signals, cpr, buf, ee, vol, 1.5, eod)
                    label = f"CPR{cpr}_BUF{buf}_T{end_label}_V{'Y' if vol else 'N'}"
                    summary['label'] = label
                    summary['cpr'] = cpr
                    summary['buf'] = buf
                    summary['end'] = ee
                    summary['vol'] = vol
                    phase1.append(summary)

    # Rank
    p1 = pd.DataFrame(phase1)
    p1_valid = p1[p1['total_trades'] >= 10].sort_values('profit_factor', ascending=False)

    print(f"\nTOP 25 ENTRY COMBOS (min 10 trades, EOD exit)")
    print(f"{'Label':45s} {'T':>5} {'WR%':>6} {'PnL':>10} {'PF':>6} {'AvgPnL%':>8}")
    print("-" * 85)
    for _, r in p1_valid.head(25).iterrows():
        pf = r['profit_factor']
        pf_str = f"{pf:5.2f}" if pf < 100 else "  inf"
        print(f"{r['label']:45s} {r['total_trades']:5.0f} {r['win_rate']:5.1f}% "
              f"{r['total_pnl']:10,.0f} {pf_str} {r['avg_pnl_pct']:7.3f}%")

    # Step 3: Phase 2 — Exit sweep on top entry combos
    print("\n" + "=" * 100)
    print("PHASE 2: Exit Strategy Sweep on Top 10 Entry Combos")
    print("=" * 100)

    EXIT_STRATEGIES = [
        ('EOD_ONLY', {'sl_pct': 999, 'tp_pct': 999}),
        ('SL0.3', {'sl_pct': 0.3}),
        ('SL0.5', {'sl_pct': 0.5}),
        ('SL0.75', {'sl_pct': 0.75}),
        ('SL1.0', {'sl_pct': 1.0}),
        ('SL0.3_TP0.5', {'sl_pct': 0.3, 'tp_pct': 0.5}),
        ('SL0.3_TP0.75', {'sl_pct': 0.3, 'tp_pct': 0.75}),
        ('SL0.3_TP1.0', {'sl_pct': 0.3, 'tp_pct': 1.0}),
        ('SL0.5_TP0.75', {'sl_pct': 0.5, 'tp_pct': 0.75}),
        ('SL0.5_TP1.0', {'sl_pct': 0.5, 'tp_pct': 1.0}),
        ('SL0.5_TP1.5', {'sl_pct': 0.5, 'tp_pct': 1.5}),
        ('SL0.5_TP2.0', {'sl_pct': 0.5, 'tp_pct': 2.0}),
        ('SL0.75_TP1.0', {'sl_pct': 0.75, 'tp_pct': 1.0}),
        ('SL0.75_TP1.5', {'sl_pct': 0.75, 'tp_pct': 1.5}),
        ('SL0.75_TP2.0', {'sl_pct': 0.75, 'tp_pct': 2.0}),
        ('SL1.0_TP1.5', {'sl_pct': 1.0, 'tp_pct': 1.5}),
        ('SL1.0_TP2.0', {'sl_pct': 1.0, 'tp_pct': 2.0}),
        ('SL1.0_TP3.0', {'sl_pct': 1.0, 'tp_pct': 3.0}),
        ('TRAIL0.3', {'sl_pct': 0.5, 'trail_pct': 0.3}),
        ('TRAIL0.5', {'sl_pct': 0.75, 'trail_pct': 0.5}),
        ('TRAIL0.75', {'sl_pct': 1.0, 'trail_pct': 0.75}),
        ('TRAIL1.0', {'sl_pct': 1.5, 'trail_pct': 1.0}),
    ]

    top_entries = p1_valid.head(10)
    phase2 = []
    best_pf = 0
    best_result = None

    for _, entry in top_entries.iterrows():
        cpr = entry['cpr']
        buf = entry['buf']
        ee = entry['end']
        vol = entry['vol']

        entry_label = f"CPR{cpr}_BUF{buf}_T{ee}_V{'Y' if vol else 'N'}"
        print(f"\n  {entry_label} ({int(entry['total_trades'])} base trades)")

        combo_results = []
        for exit_name, exit_params in EXIT_STRATEGIES:
            summary = evaluate_config(signals, cpr, buf, ee, vol, 1.5, exit_params)
            label = f"{entry_label}_{exit_name}"
            summary['label'] = label
            summary['exit_name'] = exit_name
            combo_results.append(summary)
            phase2.append(summary)

            if summary['total_trades'] >= 10 and summary['profit_factor'] > best_pf:
                best_pf = summary['profit_factor']
                best_result = summary

        combo_results.sort(key=lambda x: x['profit_factor'], reverse=True)
        for r in combo_results[:5]:
            pf = r['profit_factor']
            pf_str = f"{pf:5.2f}" if pf < 100 else "  inf"
            exits = ', '.join(f"{k}:{v}" for k, v in r['exit_reasons'].items())
            print(f"    {r['exit_name']:20s} T={r['total_trades']:4d} WR={r['win_rate']:5.1f}% "
                  f"PnL={r['total_pnl']:>10,.0f} PF={pf_str} | {exits}")

    # Final leaderboard
    print("\n" + "=" * 100)
    print("FINAL LEADERBOARD - Top 30 (min 10 trades)")
    print("=" * 100)

    all_valid = [r for r in phase2 if r['total_trades'] >= 10]
    all_valid.sort(key=lambda x: x['profit_factor'], reverse=True)

    print(f"\n{'Label':60s} {'T':>4} {'WR%':>6} {'PnL':>10} {'PF':>6} {'AvgPnL':>7}")
    print("-" * 100)
    for r in all_valid[:30]:
        pf = r['profit_factor']
        pf_str = f"{pf:5.2f}" if pf < 100 else "  inf"
        print(f"{r['label']:60s} {r['total_trades']:4d} {r['win_rate']:5.1f}% "
              f"{r['total_pnl']:10,.0f} {pf_str} {r['avg_pnl_pct']:6.3f}%")

    # Trade log for best config
    if best_result and '_results' in best_result:
        print(f"\n{'='*120}")
        print(f"TRADE LOG: {best_result['label']} | PF={best_pf:.2f}")
        print(f"{'='*120}")

        trades = sorted(best_result['_results'], key=lambda t: t['entry_time'])

        print(f"\n{'#':>3} {'Symbol':12} {'Dir':6} {'Date':12} {'Time':>6} "
              f"{'EntPx':>10} {'ExPx':>10} {'Reason':10} {'PnL%':>8}")
        print("-" * 90)

        for i, t in enumerate(trades, 1):
            print(f"{i:3d} {t['symbol']:12} {t['direction']:6} "
                  f"{t['entry_time'].strftime('%Y-%m-%d'):12} "
                  f"{t['entry_time'].strftime('%H:%M'):>6} "
                  f"{t['entry_price']:10.2f} {t['exit_price']:10.2f} "
                  f"{t['exit_reason']:10} {t['pnl_pct']:7.3f}%")

        print(f"\nExit reasons: {best_result['exit_reasons']}")

    # Save CSV
    OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cpr_breakout_research.csv')
    fields = ['label', 'total_trades', 'winners', 'losers', 'win_rate',
              'total_pnl', 'avg_pnl_pct', 'profit_factor']
    with open(OUTPUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in phase2:
            w.writerow({k: r.get(k, '') for k in fields})
    print(f"\nSaved: {OUTPUT}")


if __name__ == '__main__':
    main()
