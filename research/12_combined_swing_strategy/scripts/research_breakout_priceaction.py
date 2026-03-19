"""
Breakout & Price Action Strategy Research on Indian F&O Stocks (Daily)
Tests 6 strategy families with multiple variants on 2018-2025 daily data.
Outputs: research_results_breakout.csv + printed summary
"""

import sqlite3
import pandas as pd
import numpy as np
import csv
import os
import sys
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backtest_data', 'market_data.db')
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'research_results_breakout.csv')
START_DATE = '2018-01-01'
END_DATE = '2025-11-07'
SLIPPAGE_PCT = 0.0005  # 0.05% per side
BROKERAGE_RS = 40      # per round trip (ignored in % calc, added as flat)
CAPITAL_PER_TRADE = 200_000  # Rs 2L per trade for brokerage impact

FNO_STOCKS = [
    'RELIANCE','TCS','HDFCBANK','INFY','ICICIBANK','HINDUNILVR','ITC','SBIN',
    'BHARTIARTL','KOTAKBANK','LT','AXISBANK','ASIANPAINT','MARUTI','TITAN',
    'BAJFINANCE','HCLTECH','SUNPHARMA','ULTRACEMCO','NESTLEIND','WIPRO','ONGC',
    'NTPC','POWERGRID','M&M','TATAMOTORS','TECHM','JSWSTEEL','INDUSINDBK',
    'ADANIPORTS','BAJAJFINSV','HINDALCO','COALINDIA','DIVISLAB','GRASIM',
    'TATACONSUM','DRREDDY','CIPLA','EICHERMOT','BRITANNIA','APOLLOHOSP',
    'HEROMOTOCO','SBILIFE','BPCL','TATASTEEL','BAJAJ-AUTO','HDFCLIFE','SHREECEM',
    'ADANIENT','VEDL','TATAPOWER','GAIL','JINDALSTEL','DLF','GODREJPROP',
    'SIEMENS','HAVELLS','PIDILITIND','DABUR','MARICO','COLPAL','MUTHOOTFIN',
    'CHOLAFIN','BEL','HAL','IOC','IRCTC','COFORGE','PERSISTENT','MCX',
    'CUMMINSIND','VOLTAS','AMBUJACEM','TRENT','ZOMATO','PAYTM','DELHIVERY',
    'PNB','BANKBARODA','FEDERALBNK','IDFCFIRSTB'
]

FIELDNAMES = [
    'strategy','variant','direction','total_trades','wins','losses',
    'win_rate','avg_win_pct','avg_loss_pct','profit_factor',
    'avg_pnl_pct','median_pnl_pct','max_win_pct','max_loss_pct',
    'max_consec_losses','expectancy_pct','sharpe_per_trade',
    'best_stocks','worst_stocks','avg_hold_days'
]

# ─── DATA LOADING ─────────────────────────────────────────────────────────────
def load_data():
    """Load daily OHLCV for all F&O stocks, return dict of DataFrames."""
    print(f"Loading data from {DB_PATH}...")
    t0 = time.time()
    conn = sqlite3.connect(DB_PATH)
    placeholders = ','.join(['?' for _ in FNO_STOCKS])
    query = f"""
        SELECT symbol, date, open, high, low, close, volume
        FROM market_data_unified
        WHERE timeframe='day' AND symbol IN ({placeholders})
          AND date >= ? AND date <= ?
        ORDER BY symbol, date
    """
    df = pd.read_sql_query(query, conn, params=FNO_STOCKS + [START_DATE, END_DATE])
    conn.close()
    df['date'] = pd.to_datetime(df['date'])

    stock_data = {}
    for sym, grp in df.groupby('symbol'):
        g = grp.set_index('date').sort_index().copy()
        g = g[~g.index.duplicated(keep='first')]
        # Precompute common indicators
        g['atr14'] = compute_atr(g, 14)
        g['range'] = g['high'] - g['low']
        g['prev_high'] = g['high'].shift(1)
        g['prev_low'] = g['low'].shift(1)
        g['prev_close'] = g['close'].shift(1)
        stock_data[sym] = g

    elapsed = time.time() - t0
    print(f"Loaded {len(stock_data)} stocks, {len(df)} rows in {elapsed:.1f}s")
    return stock_data


def compute_atr(df, period=14):
    """Compute ATR."""
    h = df['high']
    l = df['low']
    c = df['close'].shift(1)
    tr = pd.concat([h - l, (h - c).abs(), (l - c).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ─── TRADE EVALUATION ─────────────────────────────────────────────────────────
def evaluate_trades(trades_list):
    """
    Given list of dicts with keys: symbol, entry_date, exit_date, entry_price,
    exit_price, direction ('long'/'short'), compute metrics.
    Returns dict of metrics or None if < 5 trades.
    """
    if len(trades_list) < 5:
        return None

    df = pd.DataFrame(trades_list)

    # PnL calculation with slippage
    if 'pnl_pct' not in df.columns:
        long_mask = df['direction'] == 'long'
        df.loc[long_mask, 'pnl_pct'] = (
            df.loc[long_mask, 'exit_price'] / df.loc[long_mask, 'entry_price'] - 1
        ) - 2 * SLIPPAGE_PCT
        df.loc[~long_mask, 'pnl_pct'] = (
            1 - df.loc[~long_mask, 'exit_price'] / df.loc[~long_mask, 'entry_price']
        ) - 2 * SLIPPAGE_PCT

    # Brokerage as % of trade
    brok_pct = BROKERAGE_RS / CAPITAL_PER_TRADE
    df['pnl_pct'] -= brok_pct

    # Hold days
    if 'hold_days' not in df.columns:
        df['hold_days'] = (pd.to_datetime(df['exit_date']) - pd.to_datetime(df['entry_date'])).dt.days

    wins = df[df['pnl_pct'] > 0]
    losses = df[df['pnl_pct'] <= 0]

    n_wins = len(wins)
    n_losses = len(losses)
    avg_win = wins['pnl_pct'].mean() * 100 if n_wins > 0 else 0
    avg_loss = losses['pnl_pct'].mean() * 100 if n_losses > 0 else 0

    gross_profit = wins['pnl_pct'].sum() if n_wins > 0 else 0
    gross_loss = abs(losses['pnl_pct'].sum()) if n_losses > 0 else 0.0001
    pf = gross_profit / gross_loss if gross_loss > 0 else 99.0

    # Max consecutive losses
    is_loss = (df['pnl_pct'] <= 0).astype(int)
    consec = is_loss * (is_loss.groupby((is_loss != is_loss.shift()).cumsum()).cumcount() + 1)
    max_consec_loss = int(consec.max())

    # Per-stock PnL
    stock_pnl = df.groupby('symbol')['pnl_pct'].sum().sort_values()
    worst_3 = ','.join(stock_pnl.head(3).index.tolist())
    best_3 = ','.join(stock_pnl.tail(3).index.tolist())

    pnl_arr = df['pnl_pct'].values
    sharpe = (pnl_arr.mean() / pnl_arr.std()) if pnl_arr.std() > 0 else 0

    return {
        'total_trades': len(df),
        'wins': n_wins,
        'losses': n_losses,
        'win_rate': round(n_wins / len(df) * 100, 2),
        'avg_win_pct': round(avg_win, 3),
        'avg_loss_pct': round(avg_loss, 3),
        'profit_factor': round(pf, 3),
        'avg_pnl_pct': round(df['pnl_pct'].mean() * 100, 3),
        'median_pnl_pct': round(df['pnl_pct'].median() * 100, 3),
        'max_win_pct': round(df['pnl_pct'].max() * 100, 2),
        'max_loss_pct': round(df['pnl_pct'].min() * 100, 2),
        'max_consec_losses': max_consec_loss,
        'expectancy_pct': round(df['pnl_pct'].mean() * 100, 3),
        'sharpe_per_trade': round(sharpe, 3),
        'best_stocks': best_3,
        'worst_stocks': worst_3,
        'avg_hold_days': round(df['hold_days'].mean(), 1),
    }


def write_result(row):
    """Append one result row to CSV."""
    file_exists = os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0
    with open(OUTPUT_CSV, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            w.writeheader()
        w.writerow(row)


def report(strategy, variant, direction, trades):
    """Evaluate trades and write to CSV + print."""
    metrics = evaluate_trades(trades)
    if metrics is None:
        print(f"  {variant} ({direction}): < 5 trades, skipping")
        return None
    row = {'strategy': strategy, 'variant': variant, 'direction': direction, **metrics}
    write_result(row)
    pf = metrics['profit_factor']
    wr = metrics['win_rate']
    nt = metrics['total_trades']
    avgpnl = metrics['avg_pnl_pct']
    flag = " <<<" if pf > 1.2 and nt >= 200 else ""
    print(f"  {variant:40s} {direction:5s} | N={nt:5d} WR={wr:5.1f}% PF={pf:5.2f} Avg={avgpnl:+6.3f}%{flag}")
    return row


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 1: PREVIOUS WEEK HIGH/LOW BREAKOUT
# ═══════════════════════════════════════════════════════════════════════════════
def strategy_weekly_breakout(stock_data):
    print("\n" + "="*80)
    print("STRATEGY 1: Previous Week High/Low Breakout")
    print("="*80)

    for hold_days in [3, 5, 10]:
        for use_atr_trail in [False, True]:
            variant = f"WeekHL_hold{hold_days}" + ("_ATRtrail" if use_atr_trail else "")
            all_long_trades = []
            all_short_trades = []

            for sym, df in stock_data.items():
                if len(df) < 100:
                    continue

                d = df.copy()
                # Assign week number (ISO)
                d['week'] = d.index.isocalendar().week.values
                d['year'] = d.index.year
                d['yw'] = d['year'] * 100 + d['week']

                # Weekly high/low
                weekly_hl = d.groupby('yw').agg(w_high=('high', 'max'), w_low=('low', 'min'))
                d['prev_w_high'] = d['yw'].map(weekly_hl['w_high'].shift(1).to_dict())
                d['prev_w_low'] = d['yw'].map(weekly_hl['w_low'].shift(1).to_dict())

                # But we need the previous COMPLETED week's high/low
                # Shift by mapping: for each row, get the week before its week
                unique_weeks = sorted(d['yw'].unique())
                week_to_prev = {w: unique_weeks[i-1] if i > 0 else None for i, w in enumerate(unique_weeks)}
                d['prev_yw'] = d['yw'].map(week_to_prev)
                d['prev_w_high'] = d['prev_yw'].map(weekly_hl['w_high'].to_dict())
                d['prev_w_low'] = d['prev_yw'].map(weekly_hl['w_low'].to_dict())

                d.dropna(subset=['prev_w_high', 'prev_w_low', 'atr14'], inplace=True)
                if len(d) < 50:
                    continue

                # Signals
                long_signal = d['close'] > d['prev_w_high']
                short_signal = d['close'] < d['prev_w_low']

                # Process trades — vectorized exit after N days
                dates = d.index.values
                closes = d['close'].values
                highs = d['high'].values
                lows = d['low'].values
                atr = d['atr14'].values

                def collect_trades(signal_mask, direction):
                    trades = []
                    idxs = np.where(signal_mask.values)[0]
                    i = 0
                    while i < len(idxs):
                        entry_idx = idxs[i]
                        if entry_idx + 1 >= len(dates):
                            i += 1
                            continue
                        # Enter next day open (use close as proxy for daily data)
                        entry_bar = entry_idx + 1
                        if entry_bar >= len(closes):
                            i += 1
                            continue
                        entry_price = closes[entry_bar]  # next day close as entry
                        entry_atr = atr[entry_bar] if not np.isnan(atr[entry_bar]) else entry_price * 0.02

                        # Hold for N days or ATR trail
                        exit_bar = min(entry_bar + hold_days, len(closes) - 1)

                        if use_atr_trail and direction == 'long':
                            trail_stop = entry_price - 2 * entry_atr
                            for b in range(entry_bar + 1, min(entry_bar + hold_days + 1, len(closes))):
                                trail_stop = max(trail_stop, closes[b] - 2 * atr[b] if not np.isnan(atr[b]) else trail_stop)
                                if lows[b] < trail_stop:
                                    exit_bar = b
                                    break
                            else:
                                exit_bar = min(entry_bar + hold_days, len(closes) - 1)
                        elif use_atr_trail and direction == 'short':
                            trail_stop = entry_price + 2 * entry_atr
                            for b in range(entry_bar + 1, min(entry_bar + hold_days + 1, len(closes))):
                                trail_stop = min(trail_stop, closes[b] + 2 * atr[b] if not np.isnan(atr[b]) else trail_stop)
                                if highs[b] > trail_stop:
                                    exit_bar = b
                                    break
                            else:
                                exit_bar = min(entry_bar + hold_days, len(closes) - 1)

                        exit_price = closes[exit_bar]
                        trades.append({
                            'symbol': sym,
                            'entry_date': str(pd.Timestamp(dates[entry_bar]).date()),
                            'exit_date': str(pd.Timestamp(dates[exit_bar]).date()),
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'direction': direction,
                            'hold_days': exit_bar - entry_bar,
                        })
                        # Skip ahead past this trade
                        i_next = np.searchsorted(idxs, exit_bar)
                        i = max(i + 1, i_next)
                    return trades

                all_long_trades.extend(collect_trades(long_signal, 'long'))
                all_short_trades.extend(collect_trades(short_signal, 'short'))

            report("WeekHL_Breakout", variant, "long", all_long_trades)
            report("WeekHL_Breakout", variant, "short", all_short_trades)
    sys.stdout.flush()


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 2: NARROW RANGE DAY BREAKOUT (NR4 / NR7)
# ═══════════════════════════════════════════════════════════════════════════════
def strategy_narrow_range(stock_data):
    print("\n" + "="*80)
    print("STRATEGY 2: Narrow Range Day Breakout (NR4/NR7)")
    print("="*80)

    for nr_period in [4, 7]:
        for exit_mode in ['atr_target', 'hold5', 'hold3']:
            variant = f"NR{nr_period}_{exit_mode}"
            all_long = []
            all_short = []

            for sym, df in stock_data.items():
                if len(df) < 100:
                    continue
                d = df.copy()

                # NR detection: today's range is minimum of last N days
                d['min_range_N'] = d['range'].rolling(nr_period).min()
                d['is_nr'] = (d['range'] == d['min_range_N']) & (d['range'] > 0)

                nr_days = np.where(d['is_nr'].values)[0]

                closes = d['close'].values
                highs = d['high'].values
                lows = d['low'].values
                opens = d['open'].values if 'open' in d.columns else closes
                atr = d['atr14'].values
                dates = d.index.values
                nr_highs = d['high'].values
                nr_lows = d['low'].values

                i = 0
                while i < len(nr_days):
                    nd = nr_days[i]
                    if nd + 1 >= len(closes):
                        i += 1
                        continue

                    nr_high = highs[nd]
                    nr_low = lows[nd]
                    next_bar = nd + 1

                    # Check if next day breaks above or below
                    broke_high = highs[next_bar] > nr_high
                    broke_low = lows[next_bar] < nr_low

                    direction = None
                    entry_price = None
                    if broke_high and not broke_low:
                        direction = 'long'
                        entry_price = nr_high  # breakout price
                    elif broke_low and not broke_high:
                        direction = 'short'
                        entry_price = nr_low
                    elif broke_high and broke_low:
                        # Both broken — skip (outside day)
                        i += 1
                        continue
                    else:
                        i += 1
                        continue

                    entry_atr = atr[nd] if not np.isnan(atr[nd]) else (nr_high - nr_low) * 2

                    if exit_mode == 'atr_target':
                        target = entry_price + 2 * entry_atr if direction == 'long' else entry_price - 2 * entry_atr
                        stop = entry_price - 1 * entry_atr if direction == 'long' else entry_price + 1 * entry_atr
                        max_bars = 10
                    elif exit_mode == 'hold5':
                        max_bars = 5
                        target = None
                        stop = None
                    else:
                        max_bars = 3
                        target = None
                        stop = None

                    exit_bar = min(next_bar + max_bars, len(closes) - 1)
                    exit_price = closes[exit_bar]

                    if target is not None:
                        for b in range(next_bar + 1, min(next_bar + max_bars + 1, len(closes))):
                            if direction == 'long':
                                if lows[b] <= stop:
                                    exit_bar = b; exit_price = stop; break
                                if highs[b] >= target:
                                    exit_bar = b; exit_price = target; break
                            else:
                                if highs[b] >= stop:
                                    exit_bar = b; exit_price = stop; break
                                if lows[b] <= target:
                                    exit_bar = b; exit_price = target; break
                        else:
                            exit_bar = min(next_bar + max_bars, len(closes) - 1)
                            exit_price = closes[exit_bar]

                    trade = {
                        'symbol': sym,
                        'entry_date': str(pd.Timestamp(dates[next_bar]).date()),
                        'exit_date': str(pd.Timestamp(dates[exit_bar]).date()),
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'direction': direction,
                        'hold_days': exit_bar - next_bar,
                    }
                    if direction == 'long':
                        all_long.append(trade)
                    else:
                        all_short.append(trade)

                    # Skip to after exit
                    i_next = np.searchsorted(nr_days, exit_bar)
                    i = max(i + 1, i_next)

            report("NarrowRange", variant, "long", all_long)
            report("NarrowRange", variant, "short", all_short)
    sys.stdout.flush()


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 3: INSIDE DAY BREAKOUT
# ═══════════════════════════════════════════════════════════════════════════════
def strategy_inside_day(stock_data):
    print("\n" + "="*80)
    print("STRATEGY 3: Inside Day Breakout")
    print("="*80)

    for exit_mode in ['atr_1r2r', 'atr_1r3r', 'hold5', 'hold10']:
        variant = f"InsideDay_{exit_mode}"
        all_long = []
        all_short = []

        for sym, df in stock_data.items():
            if len(df) < 100:
                continue
            d = df.copy()

            # Inside day: high < prev high AND low > prev low
            d['inside'] = (d['high'] < d['prev_high']) & (d['low'] > d['prev_low'])

            inside_days = np.where(d['inside'].values)[0]
            closes = d['close'].values
            highs = d['high'].values
            lows = d['low'].values
            atr = d['atr14'].values
            dates = d.index.values

            i = 0
            while i < len(inside_days):
                nd = inside_days[i]
                if nd + 1 >= len(closes):
                    i += 1
                    continue

                id_high = highs[nd]
                id_low = lows[nd]
                next_bar = nd + 1

                broke_high = highs[next_bar] > id_high
                broke_low = lows[next_bar] < id_low

                if broke_high and not broke_low:
                    direction = 'long'
                    entry_price = id_high
                elif broke_low and not broke_high:
                    direction = 'short'
                    entry_price = id_low
                elif broke_high and broke_low:
                    # Use close direction
                    if closes[next_bar] > id_high:
                        direction = 'long'
                        entry_price = id_high
                    elif closes[next_bar] < id_low:
                        direction = 'short'
                        entry_price = id_low
                    else:
                        i += 1
                        continue
                else:
                    i += 1
                    continue

                entry_atr = atr[nd] if not np.isnan(atr[nd]) else (id_high - id_low) * 2

                if exit_mode == 'atr_1r2r':
                    r_mult_target, r_mult_stop, max_bars = 2, 1, 15
                elif exit_mode == 'atr_1r3r':
                    r_mult_target, r_mult_stop, max_bars = 3, 1, 20
                elif exit_mode == 'hold5':
                    r_mult_target, r_mult_stop, max_bars = None, None, 5
                else:
                    r_mult_target, r_mult_stop, max_bars = None, None, 10

                if r_mult_target is not None:
                    if direction == 'long':
                        target = entry_price + r_mult_target * entry_atr
                        stop = entry_price - r_mult_stop * entry_atr
                    else:
                        target = entry_price - r_mult_target * entry_atr
                        stop = entry_price + r_mult_stop * entry_atr
                else:
                    target = stop = None

                exit_bar = min(next_bar + max_bars, len(closes) - 1)
                exit_price = closes[exit_bar]

                if target is not None:
                    for b in range(next_bar + 1, min(next_bar + max_bars + 1, len(closes))):
                        if direction == 'long':
                            if lows[b] <= stop:
                                exit_bar = b; exit_price = stop; break
                            if highs[b] >= target:
                                exit_bar = b; exit_price = target; break
                        else:
                            if highs[b] >= stop:
                                exit_bar = b; exit_price = stop; break
                            if lows[b] <= target:
                                exit_bar = b; exit_price = target; break
                    else:
                        exit_bar = min(next_bar + max_bars, len(closes) - 1)
                        exit_price = closes[exit_bar]

                trade = {
                    'symbol': sym,
                    'entry_date': str(pd.Timestamp(dates[next_bar]).date()),
                    'exit_date': str(pd.Timestamp(dates[exit_bar]).date()),
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'direction': direction,
                    'hold_days': exit_bar - next_bar,
                }
                if direction == 'long':
                    all_long.append(trade)
                else:
                    all_short.append(trade)

                i_next = np.searchsorted(inside_days, exit_bar)
                i = max(i + 1, i_next)

        report("InsideDay", variant, "long", all_long)
        report("InsideDay", variant, "short", all_short)
    sys.stdout.flush()


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 4: GAP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
def strategy_gap(stock_data):
    print("\n" + "="*80)
    print("STRATEGY 4: Gap Analysis")
    print("="*80)

    for gap_type in ['full_gap_up', 'full_gap_down']:
        for trade_mode in ['momentum', 'fade']:
            for hold in [1, 3, 5]:
                variant = f"{gap_type}_{trade_mode}_hold{hold}"
                all_trades = []

                for sym, df in stock_data.items():
                    if len(df) < 100:
                        continue
                    d = df.copy()

                    if gap_type == 'full_gap_up':
                        # Full gap up: open > prev high
                        d['gap'] = d['open'] > d['prev_high']
                    else:
                        # Full gap down: open < prev low
                        d['gap'] = d['open'] < d['prev_low']

                    gap_days = np.where(d['gap'].values)[0]
                    closes = d['close'].values
                    opens = d['open'].values
                    highs = d['high'].values
                    lows = d['low'].values
                    dates = d.index.values
                    prev_h = d['prev_high'].values
                    prev_l = d['prev_low'].values

                    i = 0
                    while i < len(gap_days):
                        gd = gap_days[i]

                        if gap_type == 'full_gap_up':
                            if trade_mode == 'momentum':
                                # Gap up momentum: if close > open (gap holds), go long
                                if closes[gd] <= opens[gd]:
                                    i += 1
                                    continue
                                direction = 'long'
                            else:
                                # Fade gap up: short expecting fill
                                direction = 'short'
                        else:  # gap down
                            if trade_mode == 'momentum':
                                if closes[gd] >= opens[gd]:
                                    i += 1
                                    continue
                                direction = 'short'
                            else:
                                direction = 'long'

                        entry_price = closes[gd]
                        exit_bar = min(gd + hold, len(closes) - 1)
                        exit_price = closes[exit_bar]

                        all_trades.append({
                            'symbol': sym,
                            'entry_date': str(pd.Timestamp(dates[gd]).date()),
                            'exit_date': str(pd.Timestamp(dates[exit_bar]).date()),
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'direction': direction,
                            'hold_days': exit_bar - gd,
                        })

                        i_next = np.searchsorted(gap_days, exit_bar)
                        i = max(i + 1, i_next)

                report("Gap", variant, all_trades[0]['direction'] if all_trades else 'long', all_trades)
    sys.stdout.flush()


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 5: ATR EXPANSION
# ═══════════════════════════════════════════════════════════════════════════════
def strategy_atr_expansion(stock_data):
    print("\n" + "="*80)
    print("STRATEGY 5: ATR Expansion")
    print("="*80)

    for atr_mult in [1.5, 2.0]:
        for use_vol_confirm in [False, True]:
            for mode in ['continuation', 'reversal']:
                for hold in [3, 5]:
                    vol_tag = "_volconf" if use_vol_confirm else ""
                    variant = f"ATRexp{atr_mult}x_{mode}_hold{hold}{vol_tag}"
                    all_trades = []

                    for sym, df in stock_data.items():
                        if len(df) < 100:
                            continue
                        d = df.copy()

                        d['range_ratio'] = d['range'] / d['atr14']
                        d['bullish'] = d['close'] > d['open']
                        d['vol_sma20'] = d['volume'].rolling(20).mean()
                        d['vol_confirm'] = d['volume'] > 1.5 * d['vol_sma20']

                        expansion = d['range_ratio'] > atr_mult
                        if use_vol_confirm:
                            expansion = expansion & d['vol_confirm']

                        exp_days = np.where(expansion.values)[0]
                        closes = d['close'].values
                        opens = d['open'].values
                        highs = d['high'].values
                        lows = d['low'].values
                        dates = d.index.values
                        bullish = d['bullish'].values

                        i = 0
                        while i < len(exp_days):
                            ed = exp_days[i]
                            if ed + 1 >= len(closes):
                                i += 1
                                continue

                            is_bull = bullish[ed]

                            if mode == 'continuation':
                                direction = 'long' if is_bull else 'short'
                            else:  # reversal
                                direction = 'short' if is_bull else 'long'

                            entry_price = closes[ed]
                            exit_bar = min(ed + hold, len(closes) - 1)
                            exit_price = closes[exit_bar]

                            all_trades.append({
                                'symbol': sym,
                                'entry_date': str(pd.Timestamp(dates[ed]).date()),
                                'exit_date': str(pd.Timestamp(dates[exit_bar]).date()),
                                'entry_price': entry_price,
                                'exit_price': exit_price,
                                'direction': direction,
                                'hold_days': exit_bar - ed,
                            })

                            i_next = np.searchsorted(exp_days, exit_bar)
                            i = max(i + 1, i_next)

                    report("ATR_Expansion", variant, all_trades[0]['direction'] if all_trades else 'long', all_trades)
    sys.stdout.flush()


# ═══════════════════════════════════════════════════════════════════════════════
# STRATEGY 6: N-DAY HIGH BREAKOUT (Turtle-style)
# ═══════════════════════════════════════════════════════════════════════════════
def strategy_nday_breakout(stock_data):
    print("\n" + "="*80)
    print("STRATEGY 6: N-Day High Breakout")
    print("="*80)

    for entry_n in [20, 50, 100]:
        for exit_mode in ['10d_low', 'atr_trail_2x', 'atr_trail_3x', 'hold20']:
            variant = f"{entry_n}D_high_{exit_mode}"
            all_long = []

            for sym, df in stock_data.items():
                if len(df) < entry_n + 50:
                    continue
                d = df.copy()

                d['n_high'] = d['high'].rolling(entry_n).max()
                d['prev_n_high'] = d['n_high'].shift(1)
                d['10d_low'] = d['low'].rolling(10).min().shift(1)

                # Entry: close > N-day high
                signal = d['close'] > d['prev_n_high']

                sig_days = np.where(signal.values)[0]
                closes = d['close'].values
                highs = d['high'].values
                lows = d['low'].values
                atr = d['atr14'].values
                dates = d.index.values
                low10 = d['10d_low'].values

                i = 0
                while i < len(sig_days):
                    sd = sig_days[i]
                    if sd + 1 >= len(closes):
                        i += 1
                        continue

                    entry_price = closes[sd]
                    max_hold = 60  # max hold for turtle

                    if exit_mode == '10d_low':
                        # Exit when close < 10-day low
                        exit_bar = min(sd + max_hold, len(closes) - 1)
                        for b in range(sd + 1, min(sd + max_hold + 1, len(closes))):
                            if not np.isnan(low10[b]) and closes[b] < low10[b]:
                                exit_bar = b
                                break
                    elif exit_mode.startswith('atr_trail'):
                        mult = 2.0 if '2x' in exit_mode else 3.0
                        trail_stop = entry_price - mult * atr[sd] if not np.isnan(atr[sd]) else entry_price * 0.95
                        exit_bar = min(sd + max_hold, len(closes) - 1)
                        for b in range(sd + 1, min(sd + max_hold + 1, len(closes))):
                            if not np.isnan(atr[b]):
                                trail_stop = max(trail_stop, closes[b] - mult * atr[b])
                            if lows[b] < trail_stop:
                                exit_bar = b
                                break
                    else:  # hold20
                        exit_bar = min(sd + 20, len(closes) - 1)

                    exit_price = closes[exit_bar]

                    all_long.append({
                        'symbol': sym,
                        'entry_date': str(pd.Timestamp(dates[sd]).date()),
                        'exit_date': str(pd.Timestamp(dates[exit_bar]).date()),
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'direction': 'long',
                        'hold_days': exit_bar - sd,
                    })

                    i_next = np.searchsorted(sig_days, exit_bar)
                    i = max(i + 1, i_next)

            report("NDayBreakout", variant, "long", all_long)
    sys.stdout.flush()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    t_start = time.time()

    # Clear output file
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)

    stock_data = load_data()

    strategy_weekly_breakout(stock_data)
    strategy_narrow_range(stock_data)
    strategy_inside_day(stock_data)
    strategy_gap(stock_data)
    strategy_atr_expansion(stock_data)
    strategy_nday_breakout(stock_data)

    # ─── FINAL SUMMARY ───────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("FINAL SUMMARY — Ranked by Profit Factor (PF > 1.0, N >= 50)")
    print("="*80)

    results = pd.read_csv(OUTPUT_CSV)
    good = results[(results['profit_factor'] > 1.0) & (results['total_trades'] >= 50)].sort_values('profit_factor', ascending=False)

    print(f"\n{'Strategy':<20} {'Variant':<45} {'Dir':>5} {'Trades':>6} {'WR%':>6} {'PF':>6} {'AvgPnL%':>8} {'MaxCL':>5}")
    print("-" * 110)
    for _, r in good.head(40).iterrows():
        flag = " <<<" if r['profit_factor'] > 1.2 and r['total_trades'] >= 200 else ""
        print(f"{r['strategy']:<20} {r['variant']:<45} {r['direction']:>5} {r['total_trades']:>6} {r['win_rate']:>6.1f} {r['profit_factor']:>6.2f} {r['avg_pnl_pct']:>+8.3f} {r['max_consec_losses']:>5}{flag}")

    print(f"\n--- HIGHLIGHT: PF > 1.2 AND trades >= 200 ---")
    highlight = results[(results['profit_factor'] > 1.2) & (results['total_trades'] >= 200)].sort_values('profit_factor', ascending=False)
    if len(highlight) > 0:
        for _, r in highlight.iterrows():
            print(f"  {r['strategy']:20s} {r['variant']:45s} {r['direction']:5s} N={r['total_trades']:5d} WR={r['win_rate']:.1f}% PF={r['profit_factor']:.2f} Avg={r['avg_pnl_pct']:+.3f}%")
    else:
        print("  None found — consider relaxing filters or testing more variants.")

    elapsed = time.time() - t_start
    print(f"\nTotal runtime: {elapsed:.0f}s")
    print(f"Results saved to: {OUTPUT_CSV}")
