#!/usr/bin/env python3
"""
Daily CPR Breakout Strategy Backtester - V2 (Tightened Filters)
================================================================
Entry LONG:  5-min candle closes above (prev day high + buffer%) + daily CPR below price
Entry SHORT: 5-min candle closes below (prev day low - buffer%)  + daily CPR above price

Filters:
- Narrow daily CPR (tight thresholds)
- Only FIRST breakout per symbol per day (no re-entries)
- Volume confirmation: breakout candle vol > 1.5x rolling avg
- Entry window: 9:20 to 11:15 only (morning momentum)
- No entries after 14:45, all exits by 15:20

Sweeps across CPR thresholds, buffer sizes, and exit strategies.
"""

import sys, logging, csv, os, time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))
logging.disable(logging.WARNING)

import sqlite3

# ============================================================================
# Configuration
# ============================================================================
DB_PATH = Path(__file__).parent / 'backtest_data' / 'market_data.db'

SYMBOLS = [
    'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
    'SBIN', 'BHARTIARTL', 'ITC', 'KOTAKBANK', 'HINDUNILVR'
]

START_DATE = '2024-01-01'
END_DATE = '2025-10-27'
INITIAL_CAPITAL = 1_000_000

# Position sizing - FIXED: cap at position_size_pct of capital
POSITION_SIZE_PCT = 5.0
MAX_POSITIONS = 3

# Costs
SLIPPAGE_PCT = 0.05
BROKERAGE_PER_SIDE = 20

# Time rules
ENTRY_START = "09:20"        # Skip first candle (gap noise)
ENTRY_END_MORNING = "11:15"  # Only morning breakouts
NO_ENTRY_AFTER = "14:45"
EOD_EXIT = "15:20"

# Volume confirmation
VOL_LOOKBACK = 20  # 20-candle rolling average for volume comparison
VOL_MULTIPLIER = 1.5  # Breakout candle must be 1.5x avg volume

LOT_SIZES = {
    'RELIANCE': 250, 'TCS': 150, 'HDFCBANK': 550, 'INFY': 300,
    'ICICIBANK': 700, 'HINDUNILVR': 300, 'ITC': 1600, 'SBIN': 750,
    'BHARTIARTL': 475, 'KOTAKBANK': 400,
}


# ============================================================================
# Data loading
# ============================================================================
def load_data():
    conn = sqlite3.connect(DB_PATH)
    daily_data = {}
    five_min_data = {}

    for sym in SYMBOLS:
        df_d = pd.read_sql_query(
            "SELECT date, open, high, low, close, volume FROM market_data_unified "
            "WHERE symbol=? AND timeframe='day' AND date>=? AND date<=? ORDER BY date",
            conn, params=(sym, '2023-11-01', END_DATE)  # extra history for vol avg
        )
        df_d['date'] = pd.to_datetime(df_d['date'])
        daily_data[sym] = df_d

        df_5 = pd.read_sql_query(
            "SELECT date, open, high, low, close, volume FROM market_data_unified "
            "WHERE symbol=? AND timeframe='5minute' AND date>=? AND date<=? ORDER BY date",
            conn, params=(sym, START_DATE, END_DATE + ' 23:59:59')
        )
        df_5['date'] = pd.to_datetime(df_5['date'])
        # Pre-compute rolling volume average
        df_5['vol_avg'] = df_5['volume'].rolling(VOL_LOOKBACK, min_periods=5).mean()
        five_min_data[sym] = df_5

    conn.close()
    return daily_data, five_min_data


# ============================================================================
# Daily CPR
# ============================================================================
def compute_daily_cpr(daily_df: pd.DataFrame) -> dict:
    rows = {}
    for i in range(1, len(daily_df)):
        prev = daily_df.iloc[i - 1]
        curr = daily_df.iloc[i]

        ph, pl, pc = prev['high'], prev['low'], prev['close']
        pivot = (ph + pl + pc) / 3.0
        bc = (ph + pl) / 2.0
        tc = (pivot - bc) + pivot
        width_pct = abs(tc - bc) / pivot * 100.0

        rows[curr['date'].date()] = {
            'prev_high': ph,
            'prev_low': pl,
            'prev_close': pc,
            'pivot': pivot,
            'tc': tc,
            'bc': bc,
            'cpr_width_pct': width_pct,
        }
    return rows


# ============================================================================
# Trade
# ============================================================================
@dataclass
class Trade:
    symbol: str
    direction: str
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    quantity: int = 1
    pnl: float = 0.0
    pnl_pct: float = 0.0
    hwm: float = 0.0   # high watermark for trailing
    lwm: float = 0.0   # low watermark for trailing


# ============================================================================
# Exit logic
# ============================================================================
def check_exit(trade: Trade, candle, exit_strategy: dict, cpr: dict) -> Optional[Tuple[float, str]]:
    high = candle['high']
    low = candle['low']
    price = candle['close']

    if trade.direction == 'LONG':
        # SL
        sl_pct = exit_strategy.get('sl_pct', 999)
        sl_price = trade.entry_price * (1 - sl_pct / 100)
        if low <= sl_price:
            return (sl_price, 'SL')

        # Target
        tp_pct = exit_strategy.get('tp_pct', 999)
        tp_price = trade.entry_price * (1 + tp_pct / 100)
        if high >= tp_price:
            return (tp_price, 'TARGET')

        # Trailing
        if 'trail_pct' in exit_strategy:
            trade.hwm = max(trade.hwm, high)
            trail_price = trade.hwm * (1 - exit_strategy['trail_pct'] / 100)
            if low <= trail_price and trade.hwm > trade.entry_price:
                return (trail_price, 'TRAIL')

        # Exit below pivot
        if exit_strategy.get('exit_below_pivot') and price < cpr.get('pivot', 0):
            return (price, 'BELOW_PIVOT')

    else:  # SHORT
        sl_pct = exit_strategy.get('sl_pct', 999)
        sl_price = trade.entry_price * (1 + sl_pct / 100)
        if high >= sl_price:
            return (sl_price, 'SL')

        tp_pct = exit_strategy.get('tp_pct', 999)
        tp_price = trade.entry_price * (1 - tp_pct / 100)
        if low <= tp_price:
            return (tp_price, 'TARGET')

        if 'trail_pct' in exit_strategy:
            trade.lwm = min(trade.lwm, low)
            trail_price = trade.lwm * (1 + exit_strategy['trail_pct'] / 100)
            if high >= trail_price and trade.lwm < trade.entry_price:
                return (trail_price, 'TRAIL')

        if exit_strategy.get('exit_above_pivot') and price > cpr.get('pivot', 0):
            return (price, 'ABOVE_PIVOT')

    return None


# ============================================================================
# Core backtest
# ============================================================================
def run_backtest(
    daily_data: dict,
    five_min_data: dict,
    narrow_cpr_threshold: float,
    breakout_buffer_pct: float,
    exit_strategy: dict,
    entry_window: str,       # 'morning' or 'full_day'
    require_volume: bool,
) -> Tuple[List[Trade], dict]:

    capital = INITIAL_CAPITAL
    trades: List[Trade] = []
    open_trades: List[Trade] = []

    # Pre-compute CPR
    cpr_by_sym = {sym: compute_daily_cpr(daily_data[sym]) for sym in SYMBOLS}

    # Group 5-min by date
    fivemin_by_sym_date = {}
    for sym in SYMBOLS:
        df = five_min_data[sym].copy()
        if len(df) == 0:
            continue
        df['trade_date'] = df['date'].dt.date
        fivemin_by_sym_date[sym] = {d: g.sort_values('date') for d, g in df.groupby('trade_date')}

    all_dates = set()
    for sym in SYMBOLS:
        if sym in fivemin_by_sym_date:
            all_dates.update(fivemin_by_sym_date[sym].keys())
    all_dates = sorted(all_dates)

    # Entry end time based on window
    entry_end = ENTRY_END_MORNING if entry_window == 'morning' else NO_ENTRY_AFTER

    peak_capital = capital
    max_dd = 0.0

    for trade_date in all_dates:
        # Track which symbols already triggered today (first breakout only)
        triggered_today = set()

        for sym in SYMBOLS:
            if sym not in fivemin_by_sym_date or trade_date not in fivemin_by_sym_date[sym]:
                continue
            if trade_date not in cpr_by_sym.get(sym, {}):
                continue

            day_candles = fivemin_by_sym_date[sym][trade_date]
            cpr = cpr_by_sym[sym][trade_date]

            if cpr['cpr_width_pct'] >= narrow_cpr_threshold:
                continue

            prev_high = cpr['prev_high']
            prev_low = cpr['prev_low']
            pivot = cpr['pivot']
            tc = cpr['tc']
            bc = cpr['bc']

            # Breakout levels with buffer
            long_trigger = prev_high * (1 + breakout_buffer_pct / 100)
            short_trigger = prev_low * (1 - breakout_buffer_pct / 100)

            for _, candle in day_candles.iterrows():
                candle_time = candle['date']
                time_str = candle_time.strftime('%H:%M')

                # --- EXIT CHECK ---
                for t in open_trades:
                    if t.symbol != sym or t.exit_time is not None:
                        continue

                    if time_str >= EOD_EXIT:
                        exit_px = candle['close']
                        if t.direction == 'LONG':
                            exit_px *= (1 - SLIPPAGE_PCT / 100)
                        else:
                            exit_px *= (1 + SLIPPAGE_PCT / 100)
                        t.exit_time = candle_time
                        t.exit_price = exit_px
                        t.exit_reason = 'EOD'
                        if t.direction == 'LONG':
                            t.pnl = (t.exit_price - t.entry_price) * t.quantity - BROKERAGE_PER_SIDE * 2
                        else:
                            t.pnl = (t.entry_price - t.exit_price) * t.quantity - BROKERAGE_PER_SIDE * 2
                        t.pnl_pct = t.pnl / (t.entry_price * t.quantity) * 100
                        capital += t.pnl
                        continue

                    result = check_exit(t, candle, exit_strategy, cpr)
                    if result:
                        exit_px, reason = result
                        if t.direction == 'LONG':
                            exit_px *= (1 - SLIPPAGE_PCT / 100)
                        else:
                            exit_px *= (1 + SLIPPAGE_PCT / 100)
                        t.exit_time = candle_time
                        t.exit_price = exit_px
                        t.exit_reason = reason
                        if t.direction == 'LONG':
                            t.pnl = (t.exit_price - t.entry_price) * t.quantity - BROKERAGE_PER_SIDE * 2
                        else:
                            t.pnl = (t.entry_price - t.exit_price) * t.quantity - BROKERAGE_PER_SIDE * 2
                        t.pnl_pct = t.pnl / (t.entry_price * t.quantity) * 100
                        capital += t.pnl

                open_trades = [t for t in open_trades if t.exit_time is None]

                # --- ENTRY CHECK ---
                if time_str < ENTRY_START or time_str >= entry_end:
                    continue
                if len(open_trades) >= MAX_POSITIONS:
                    continue
                if any(t.symbol == sym for t in open_trades):
                    continue
                # First breakout only
                if sym in triggered_today:
                    continue

                # Volume check
                if require_volume:
                    vol_avg = candle.get('vol_avg', 0)
                    if vol_avg > 0 and candle['volume'] < vol_avg * VOL_MULTIPLIER:
                        continue

                close = candle['close']

                # LONG: close above prev high + buffer, CPR entirely below price
                if close > long_trigger and max(pivot, tc, bc) < close:
                    entry_px = close * (1 + SLIPPAGE_PCT / 100)
                    notional = capital * POSITION_SIZE_PCT / 100
                    lot = LOT_SIZES.get(sym, 1)
                    qty = int(notional / entry_px / lot) * lot
                    if qty < lot:
                        qty = lot
                    # Cap position at POSITION_SIZE_PCT of capital
                    if entry_px * qty > capital * POSITION_SIZE_PCT * 2 / 100:
                        qty = int(capital * POSITION_SIZE_PCT / 100 / entry_px / lot) * lot
                        if qty < lot and capital >= entry_px * lot * 0.15:
                            qty = lot
                        elif qty < lot:
                            continue  # Can't afford even 1 lot

                    t = Trade(symbol=sym, direction='LONG',
                              entry_time=candle_time, entry_price=entry_px,
                              quantity=qty, hwm=entry_px)
                    open_trades.append(t)
                    trades.append(t)
                    triggered_today.add(sym)

                # SHORT: close below prev low - buffer, CPR entirely above price
                elif close < short_trigger and min(pivot, tc, bc) > close:
                    entry_px = close * (1 - SLIPPAGE_PCT / 100)
                    notional = capital * POSITION_SIZE_PCT / 100
                    lot = LOT_SIZES.get(sym, 1)
                    qty = int(notional / entry_px / lot) * lot
                    if qty < lot:
                        qty = lot
                    if entry_px * qty > capital * POSITION_SIZE_PCT * 2 / 100:
                        qty = int(capital * POSITION_SIZE_PCT / 100 / entry_px / lot) * lot
                        if qty < lot and capital >= entry_px * lot * 0.15:
                            qty = lot
                        elif qty < lot:
                            continue

                    t = Trade(symbol=sym, direction='SHORT',
                              entry_time=candle_time, entry_price=entry_px,
                              quantity=qty, lwm=entry_px)
                    open_trades.append(t)
                    trades.append(t)
                    triggered_today.add(sym)

                peak_capital = max(peak_capital, capital)
                dd = (peak_capital - capital) / peak_capital * 100
                max_dd = max(max_dd, dd)

    # Force close remaining
    for t in open_trades:
        if t.exit_time is None:
            t.exit_time = t.entry_time
            t.exit_price = t.entry_price
            t.exit_reason = 'FORCE_CLOSE'
            t.pnl = -BROKERAGE_PER_SIDE * 2
            capital += t.pnl

    completed = [t for t in trades if t.exit_time is not None]
    winners = [t for t in completed if t.pnl > 0]
    losers = [t for t in completed if t.pnl <= 0]

    gross_win = sum(t.pnl for t in winners) if winners else 0
    gross_loss = abs(sum(t.pnl for t in losers)) if losers else 1

    # Exit reason breakdown
    exit_reasons = {}
    for t in completed:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    return trades, {
        'total_trades': len(completed),
        'winners': len(winners),
        'losers': len(losers),
        'win_rate': len(winners) / len(completed) * 100 if completed else 0,
        'total_pnl': sum(t.pnl for t in completed),
        'total_pnl_pct': (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100,
        'profit_factor': gross_win / gross_loss if gross_loss > 0 else 999,
        'avg_win': gross_win / len(winners) if winners else 0,
        'avg_loss': -gross_loss / len(losers) if losers else 0,
        'max_dd': max_dd,
        'final_capital': capital,
        'exit_reasons': str(exit_reasons),
    }


# ============================================================================
# Sweep configs
# ============================================================================
EXIT_STRATEGIES = [
    {'name': 'EOD_ONLY', 'sl_pct': 999, 'tp_pct': 999},

    {'name': 'SL_0.5', 'sl_pct': 0.5, 'tp_pct': 999},
    {'name': 'SL_0.75', 'sl_pct': 0.75, 'tp_pct': 999},
    {'name': 'SL_1.0', 'sl_pct': 1.0, 'tp_pct': 999},

    {'name': 'SL0.5_TP1.0', 'sl_pct': 0.5, 'tp_pct': 1.0},
    {'name': 'SL0.5_TP1.5', 'sl_pct': 0.5, 'tp_pct': 1.5},
    {'name': 'SL0.75_TP1.5', 'sl_pct': 0.75, 'tp_pct': 1.5},
    {'name': 'SL0.75_TP2.0', 'sl_pct': 0.75, 'tp_pct': 2.0},
    {'name': 'SL1.0_TP1.5', 'sl_pct': 1.0, 'tp_pct': 1.5},
    {'name': 'SL1.0_TP2.0', 'sl_pct': 1.0, 'tp_pct': 2.0},
    {'name': 'SL1.0_TP3.0', 'sl_pct': 1.0, 'tp_pct': 3.0},

    {'name': 'TRAIL_0.3', 'sl_pct': 0.75, 'tp_pct': 999, 'trail_pct': 0.3},
    {'name': 'TRAIL_0.5', 'sl_pct': 0.75, 'tp_pct': 999, 'trail_pct': 0.5},
    {'name': 'TRAIL_0.75', 'sl_pct': 1.0, 'tp_pct': 999, 'trail_pct': 0.75},

    {'name': 'CPR_PIVOT', 'sl_pct': 1.0, 'tp_pct': 999, 'exit_below_pivot': True, 'exit_above_pivot': True},

    {'name': 'TRAIL0.5_TP1.5', 'sl_pct': 0.75, 'tp_pct': 1.5, 'trail_pct': 0.5},
    {'name': 'TRAIL0.5_TP2.0', 'sl_pct': 0.75, 'tp_pct': 2.0, 'trail_pct': 0.5},
]

CPR_THRESHOLDS = [0.1, 0.15, 0.2, 0.3, 0.5]
BUFFER_PCTS = [0.0, 0.1, 0.2, 0.3]
ENTRY_WINDOWS = ['morning', 'full_day']
VOLUME_OPTIONS = [False, True]


def main():
    total_configs = len(CPR_THRESHOLDS) * len(BUFFER_PCTS) * len(EXIT_STRATEGIES) * len(ENTRY_WINDOWS) * len(VOLUME_OPTIONS)

    print("=" * 100)
    print("DAILY CPR BREAKOUT V2 - TIGHTENED FILTERS")
    print(f"Period: {START_DATE} to {END_DATE} | Symbols: {len(SYMBOLS)}")
    print(f"Filters: First breakout only | Entry {ENTRY_START}-{ENTRY_END_MORNING} (morning) or -{NO_ENTRY_AFTER}")
    print(f"Volume confirmation: candle vol > {VOL_MULTIPLIER}x {VOL_LOOKBACK}-candle avg")
    print(f"CPR thresholds: {CPR_THRESHOLDS}")
    print(f"Breakout buffers: {BUFFER_PCTS}%")
    print(f"Exit strategies: {len(EXIT_STRATEGIES)} | Windows: {ENTRY_WINDOWS} | Vol: {VOLUME_OPTIONS}")
    print(f"Total configs: {total_configs}")
    print("=" * 100)

    print("\nLoading data...", end='', flush=True)
    t0 = time.time()
    daily_data, five_min_data = load_data()
    print(f" {time.time()-t0:.1f}s")

    OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cpr_breakout_v2_sweep.csv')
    FIELDNAMES = [
        'cpr_threshold', 'buffer_pct', 'entry_window', 'volume_filter',
        'exit_strategy', 'total_trades', 'winners', 'losers',
        'win_rate', 'total_pnl', 'total_pnl_pct', 'profit_factor',
        'avg_win', 'avg_loss', 'max_dd', 'final_capital', 'exit_reasons',
    ]

    with open(OUTPUT_CSV, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    run_idx = 0

    for cpr_thresh in CPR_THRESHOLDS:
        for buffer in BUFFER_PCTS:
            for window in ENTRY_WINDOWS:
                for vol_filter in VOLUME_OPTIONS:
                    for exit_strat in EXIT_STRATEGIES:
                        run_idx += 1
                        label = (f"CPR{cpr_thresh}_BUF{buffer}_{window[:3].upper()}"
                                 f"_VOL{'Y' if vol_filter else 'N'}_{exit_strat['name']}")

                        print(f"[{run_idx}/{total_configs}] {label:55s}", end='', flush=True)

                        t1 = time.time()
                        _, summary = run_backtest(
                            daily_data, five_min_data,
                            cpr_thresh, buffer, exit_strat, window, vol_filter
                        )
                        elapsed = time.time() - t1

                        row = {
                            'cpr_threshold': cpr_thresh,
                            'buffer_pct': buffer,
                            'entry_window': window,
                            'volume_filter': vol_filter,
                            'exit_strategy': exit_strat['name'],
                            **summary,
                        }

                        with open(OUTPUT_CSV, 'a', newline='') as f:
                            csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

                        pf_str = f"{summary['profit_factor']:5.2f}" if summary['profit_factor'] < 100 else " inf "
                        print(f" {elapsed:4.1f}s | T={summary['total_trades']:4d} "
                              f"WR={summary['win_rate']:5.1f}% "
                              f"PnL={summary['total_pnl']:>10,.0f} "
                              f"PF={pf_str} DD={summary['max_dd']:.1f}%")

    # Results
    print("\n" + "=" * 100)
    results = pd.read_csv(OUTPUT_CSV)

    # Filter: min 10 trades, PF > 1
    good = results[(results['total_trades'] >= 10) & (results['profit_factor'] > 1.0)]
    good = good.sort_values('profit_factor', ascending=False)

    if len(good) > 0:
        print(f"PROFITABLE CONFIGS (PF > 1.0, min 10 trades): {len(good)}")
        print("=" * 100)
        print(f"{'CPR':>4} {'Buf':>4} {'Win':>4} {'Vol':>3} {'Exit':20} {'Trades':>6} "
              f"{'WR%':>6} {'PnL':>12} {'PF':>6} {'DD':>6} {'AvgW':>10} {'AvgL':>10}")
        print("-" * 100)

        for _, r in good.head(25).iterrows():
            print(f"{r['cpr_threshold']:4.2f} {r['buffer_pct']:4.1f} "
                  f"{r['entry_window'][:3]:>4} {'Y' if r['volume_filter'] else 'N':>3} "
                  f"{r['exit_strategy']:20} {r['total_trades']:6.0f} "
                  f"{r['win_rate']:5.1f}% {r['total_pnl']:12,.0f} "
                  f"{r['profit_factor']:6.2f} {r['max_dd']:5.1f}% "
                  f"{r['avg_win']:10,.0f} {r['avg_loss']:10,.0f}")
    else:
        print("NO PROFITABLE CONFIGS FOUND (PF > 1.0 with 10+ trades)")
        # Show best attempts
        print("\nTop 15 by PF (any trade count):")
        best = results[results['total_trades'] >= 3].sort_values('profit_factor', ascending=False)
        print(f"{'CPR':>4} {'Buf':>4} {'Win':>4} {'Vol':>3} {'Exit':20} {'Trades':>6} "
              f"{'WR%':>6} {'PnL':>12} {'PF':>6} {'DD':>6}")
        print("-" * 90)
        for _, r in best.head(15).iterrows():
            print(f"{r['cpr_threshold']:4.2f} {r['buffer_pct']:4.1f} "
                  f"{r['entry_window'][:3]:>4} {'Y' if r['volume_filter'] else 'N':>3} "
                  f"{r['exit_strategy']:20} {r['total_trades']:6.0f} "
                  f"{r['win_rate']:5.1f}% {r['total_pnl']:12,.0f} "
                  f"{r['profit_factor']:6.2f} {r['max_dd']:5.1f}%")

    print(f"\nFull results: {OUTPUT_CSV}")
    print(f"Total configs: {total_configs} | With trades: {len(results[results['total_trades']>0])}")


if __name__ == '__main__':
    main()
