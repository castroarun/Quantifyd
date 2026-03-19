"""
Mean Reversion Strategy Research on 60-minute F&O Data
=====================================================
Tests 5 strategy families with multiple parameter variants on 93 F&O stocks.
Realistic costs: 0.05% slippage per side + Rs 40 brokerage per round trip.
"""

import sqlite3
import pandas as pd
import numpy as np
import csv
import os
import sys
import time
from datetime import datetime

# ── Config ──────────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backtest_data', 'market_data.db')
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'research_results_meanrev.csv')
SLIPPAGE_PCT = 0.0005  # 0.05% per side
BROKERAGE_PER_RT = 40  # Rs per round trip
CAPITAL_PER_TRADE = 100_000  # Rs 1 lakh per trade for PnL calculation

FIELDNAMES = [
    'strategy', 'params', 'total_trades', 'win_rate', 'profit_factor',
    'avg_win_pct', 'avg_loss_pct', 'avg_trade_pct', 'total_pnl_pct',
    'max_consec_loss', 'avg_bars_held', 'best_stock', 'worst_stock',
    'yearly_breakdown', 'calmar_approx'
]

# ── Data Loading ────────────────────────────────────────────────────────────
def load_all_60min_data():
    """Load all 60-min data into a dict of DataFrames keyed by symbol."""
    print("Loading 60-min data from DB...", flush=True)
    t0 = time.time()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT symbol, date, open, high, low, close, volume "
        "FROM market_data_unified WHERE timeframe='60minute' ORDER BY symbol, date",
        conn
    )
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    df['trade_date'] = df['date'].dt.date

    stock_data = {}
    for sym, grp in df.groupby('symbol'):
        g = grp.copy().reset_index(drop=True)
        stock_data[sym] = g

    print(f"  Loaded {len(df)} candles for {len(stock_data)} stocks in {time.time()-t0:.1f}s", flush=True)
    return stock_data


# ── Indicator Functions (vectorized) ────────────────────────────────────────
def calc_ema(s, period):
    return s.ewm(span=period, adjust=False).mean()

def calc_sma(s, period):
    return s.rolling(period).mean()

def calc_atr(high, low, close, period):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def calc_rsi(close, period):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calc_bb(close, period, std_mult):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    width = (upper - lower) / mid
    return mid, upper, lower, width

def calc_kc(close, high, low, ema_period, atr_period, atr_mult):
    mid = calc_ema(close, ema_period)
    atr = calc_atr(high, low, close, atr_period)
    upper = mid + atr_mult * atr
    lower = mid - atr_mult * atr
    return mid, upper, lower


# ── Vectorized Backtester ───────────────────────────────────────────────────
def backtest_signals(df, entry_mask, exit_mask, direction='long', max_hold_bars=None):
    """
    Vectorized signal-based backtest.
    entry_mask: boolean Series, True on bars where we enter
    exit_mask: boolean Series, True on bars where we exit
    Returns list of trade dicts.
    """
    trades = []
    in_trade = False
    entry_price = 0
    entry_bar = 0
    entry_date = None
    entry_symbol = None

    close = df['close'].values
    dates = df['date'].values
    n = len(df)
    entry_flags = entry_mask.values if hasattr(entry_mask, 'values') else entry_mask
    exit_flags = exit_mask.values if hasattr(exit_mask, 'values') else exit_mask

    for i in range(n):
        if not in_trade:
            if entry_flags[i]:
                in_trade = True
                # Apply slippage on entry
                entry_price = close[i] * (1 + SLIPPAGE_PCT) if direction == 'long' else close[i] * (1 - SLIPPAGE_PCT)
                entry_bar = i
                entry_date = dates[i]
        else:
            should_exit = exit_flags[i]
            if max_hold_bars and (i - entry_bar) >= max_hold_bars:
                should_exit = True

            if should_exit or i == n - 1:
                # Apply slippage on exit
                exit_price = close[i] * (1 - SLIPPAGE_PCT) if direction == 'long' else close[i] * (1 + SLIPPAGE_PCT)

                if direction == 'long':
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                else:
                    pnl_pct = (entry_price - exit_price) / entry_price * 100

                # Deduct brokerage as % of capital
                brokerage_pct = BROKERAGE_PER_RT / CAPITAL_PER_TRADE * 100
                pnl_pct -= brokerage_pct

                trades.append({
                    'entry_date': entry_date,
                    'exit_date': dates[i],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': pnl_pct,
                    'bars_held': i - entry_bar,
                    'year': pd.Timestamp(entry_date).year
                })
                in_trade = False

    return trades


def compute_metrics(all_trades, per_stock_trades):
    """Compute aggregate metrics from trade list."""
    if not all_trades:
        return None

    pnls = [t['pnl_pct'] for t in all_trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    bars = [t['bars_held'] for t in all_trades]

    win_rate = len(wins) / len(pnls) * 100
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001
    profit_factor = gross_profit / gross_loss

    # Max consecutive losses
    max_consec = 0
    cur_consec = 0
    for p in pnls:
        if p <= 0:
            cur_consec += 1
            max_consec = max(max_consec, cur_consec)
        else:
            cur_consec = 0

    # Best/worst stock by avg PnL
    stock_pnls = {}
    for sym, trades in per_stock_trades.items():
        if trades:
            stock_pnls[sym] = np.mean([t['pnl_pct'] for t in trades])

    best_stock = max(stock_pnls, key=stock_pnls.get) if stock_pnls else 'N/A'
    worst_stock = min(stock_pnls, key=stock_pnls.get) if stock_pnls else 'N/A'

    # Yearly breakdown
    yearly = {}
    for t in all_trades:
        yr = t['year']
        if yr not in yearly:
            yearly[yr] = []
        yearly[yr].append(t['pnl_pct'])
    yearly_str = "; ".join(f"{yr}:{np.mean(v):.2f}%({len(v)}t)" for yr, v in sorted(yearly.items()))

    # Approx calmar: total return / max drawdown of cumulative curve
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = dd.max() if len(dd) > 0 else 1
    years_span = max(1, (max(t['year'] for t in all_trades) - min(t['year'] for t in all_trades) + 1))
    ann_return = sum(pnls) / years_span
    calmar = ann_return / max_dd if max_dd > 0 else 0

    return {
        'total_trades': len(pnls),
        'win_rate': round(win_rate, 2),
        'profit_factor': round(profit_factor, 3),
        'avg_win_pct': round(avg_win, 3),
        'avg_loss_pct': round(avg_loss, 3),
        'avg_trade_pct': round(np.mean(pnls), 4),
        'total_pnl_pct': round(sum(pnls), 2),
        'max_consec_loss': max_consec,
        'avg_bars_held': round(np.mean(bars), 1),
        'best_stock': f"{best_stock}({stock_pnls.get(best_stock,0):.2f}%)",
        'worst_stock': f"{worst_stock}({stock_pnls.get(worst_stock,0):.2f}%)",
        'yearly_breakdown': yearly_str,
        'calmar_approx': round(calmar, 3)
    }


# ── Strategy Runners ────────────────────────────────────────────────────────

def run_kc_meanrev(stock_data, kc_period, atr_mult, use_sma200, max_hold=None):
    """Keltner Channel mean reversion: buy below lower, exit at mid."""
    all_trades = []
    per_stock = {}

    for sym, df in stock_data.items():
        if len(df) < 250:
            continue
        mid, upper, lower = calc_kc(df['close'], df['high'], df['low'], kc_period, kc_period, atr_mult)

        entry = df['close'] < lower
        exit_sig = df['close'] >= mid

        if use_sma200:
            sma200 = calc_sma(df['close'], 200)
            entry = entry & (df['close'] > sma200)

        # Don't enter on consecutive bars (require exit first)
        trades = backtest_signals(df, entry, exit_sig, 'long', max_hold)
        all_trades.extend(trades)
        per_stock[sym] = trades

    return all_trades, per_stock


def run_bb_squeeze(stock_data, bb_period, std_mult, squeeze_pctile=20, lookback=100):
    """Bollinger Band squeeze + expansion: buy on breakout after squeeze."""
    all_trades = []
    per_stock = {}

    for sym, df in stock_data.items():
        if len(df) < 250:
            continue
        mid, upper, lower, width = calc_bb(df['close'], bb_period, std_mult)

        # Squeeze: width < Nth percentile of rolling lookback
        width_pctile = width.rolling(lookback).quantile(squeeze_pctile / 100)
        in_squeeze = width < width_pctile

        # Expansion: was in squeeze recently (within last 5 bars), now price breaks out
        was_squeezed = in_squeeze.rolling(5).max().fillna(0).astype(bool)

        # Long entry: close above upper after squeeze
        entry_long = was_squeezed & (df['close'] > upper)
        # Exit: return to mid
        exit_sig = df['close'] <= mid

        trades = backtest_signals(df, entry_long, exit_sig, 'long', max_hold_bars=50)
        all_trades.extend(trades)
        per_stock[sym] = trades

    # Also test mean reversion side: close below lower after squeeze -> buy, exit at mid
    all_trades_mr = []
    per_stock_mr = {}
    for sym, df in stock_data.items():
        if len(df) < 250:
            continue
        mid, upper, lower, width = calc_bb(df['close'], bb_period, std_mult)
        width_pctile = width.rolling(lookback).quantile(squeeze_pctile / 100)
        in_squeeze = width < width_pctile
        was_squeezed = in_squeeze.rolling(5).max().fillna(0).astype(bool)

        entry = was_squeezed & (df['close'] < lower)
        exit_sig = df['close'] >= mid

        trades = backtest_signals(df, entry, exit_sig, 'long', max_hold_bars=50)
        all_trades_mr.extend(trades)
        per_stock_mr[sym] = trades

    return (all_trades, per_stock), (all_trades_mr, per_stock_mr)


def run_rsi_meanrev(stock_data, rsi_period, entry_threshold, exit_threshold=50, use_sma200=False, max_hold=None):
    """RSI mean reversion: buy when oversold, exit when RSI crosses exit_threshold."""
    all_trades = []
    per_stock = {}

    for sym, df in stock_data.items():
        if len(df) < 250:
            continue
        rsi = calc_rsi(df['close'], rsi_period)

        entry = rsi < entry_threshold
        exit_sig = rsi >= exit_threshold

        if use_sma200:
            sma200 = calc_sma(df['close'], 200)
            entry = entry & (df['close'] > sma200)

        trades = backtest_signals(df, entry, exit_sig, 'long', max_hold)
        all_trades.extend(trades)
        per_stock[sym] = trades

    return all_trades, per_stock


def run_ema_deviation(stock_data, ema_period, atr_mult_entry, max_hold=None):
    """Price deviation from EMA: buy when price < EMA - N*ATR, exit at EMA."""
    all_trades = []
    per_stock = {}

    for sym, df in stock_data.items():
        if len(df) < 250:
            continue
        ema = calc_ema(df['close'], ema_period)
        atr = calc_atr(df['high'], df['low'], df['close'], 14)

        entry = df['close'] < (ema - atr_mult_entry * atr)
        exit_sig = df['close'] >= ema

        trades = backtest_signals(df, entry, exit_sig, 'long', max_hold)
        all_trades.extend(trades)
        per_stock[sym] = trades

    return all_trades, per_stock


def run_inside_bar(stock_data, atr_target_mult=2.0, max_hold_bars=7):
    """Inside bar breakout: detect inside hourly bars, enter on breakout."""
    all_trades = []
    per_stock = {}

    for sym, df in stock_data.items():
        if len(df) < 250:
            continue

        # Inside bar: high < prev high AND low > prev low
        prev_high = df['high'].shift(1)
        prev_low = df['low'].shift(1)
        is_inside = (df['high'] < prev_high) & (df['low'] > prev_low)

        # Breakout above inside bar high on next bar
        inside_high = df['high'].shift(1)  # the inside bar's high
        was_inside = is_inside.shift(1).fillna(False)

        entry = was_inside & (df['close'] > inside_high)

        # Exit: ATR-based target or max hold
        atr = calc_atr(df['high'], df['low'], df['close'], 14)
        # We'll use a simple target: exit when gain > atr_target_mult * ATR as %
        # For simplicity, just use max hold as primary exit
        exit_sig = pd.Series(False, index=df.index)  # rely on max_hold

        trades = backtest_signals(df, entry, exit_sig, 'long', max_hold_bars)
        all_trades.extend(trades)
        per_stock[sym] = trades

    return all_trades, per_stock


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    stock_data = load_all_60min_data()

    # Initialize CSV
    with open(OUTPUT_CSV, 'w', newline='') as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    results = []
    strategy_count = 0
    t_start = time.time()

    def save_result(strategy_name, params_str, all_trades, per_stock):
        nonlocal strategy_count
        strategy_count += 1
        metrics = compute_metrics(all_trades, per_stock)
        if metrics is None:
            print(f"  [{strategy_count}] {strategy_name} ({params_str}) — NO TRADES", flush=True)
            return

        row = {'strategy': strategy_name, 'params': params_str, **metrics}

        # Write incrementally
        with open(OUTPUT_CSV, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

        pf = metrics['profit_factor']
        wr = metrics['win_rate']
        nt = metrics['total_trades']
        marker = " ***" if pf > 1.2 and nt >= 100 else ""
        print(f"  [{strategy_count}] {strategy_name} ({params_str}): "
              f"PF={pf:.2f} WR={wr:.1f}% Trades={nt} AvgTrade={metrics['avg_trade_pct']:.3f}%{marker}",
              flush=True)
        results.append(row)

    # ═══════════════════════════════════════════════════════════════════════
    # 1. KELTNER CHANNEL MEAN REVERSION
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("STRATEGY 1: Keltner Channel Mean Reversion")
    print("="*70, flush=True)

    kc_configs = [
        # (kc_period, atr_mult, use_sma200, max_hold)
        (20, 1.5, False, None),
        (20, 2.0, False, None),
        (20, 2.5, False, None),
        (20, 1.5, True, None),
        (20, 2.0, True, None),
        (20, 2.5, True, None),
        (6, 1.3, False, None),    # KC6 from daily, test on 60min
        (6, 1.3, True, None),
        (6, 1.3, True, 50),       # KC6 + SMA200 + max hold 50 bars (~7 days)
        (10, 1.5, False, None),
        (10, 2.0, False, None),
        (10, 1.5, True, None),
        (10, 2.0, True, None),
        (20, 1.5, False, 35),     # With max hold ~5 days
        (20, 2.0, False, 35),
        (20, 2.5, False, 35),
    ]

    for kc_p, atr_m, sma, mh in kc_configs:
        params = f"KC({kc_p},{atr_m})"
        if sma: params += "+SMA200"
        if mh: params += f"+MH{mh}"
        trades, per_stock = run_kc_meanrev(stock_data, kc_p, atr_m, sma, mh)
        save_result("KC_MeanRev", params, trades, per_stock)

    # ═══════════════════════════════════════════════════════════════════════
    # 2. BOLLINGER BAND SQUEEZE + EXPANSION
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("STRATEGY 2: Bollinger Band Squeeze + Expansion")
    print("="*70, flush=True)

    bb_configs = [
        # (period, std_mult)
        (20, 1.5),
        (20, 2.0),
        (20, 2.5),
        (30, 1.5),
        (30, 2.0),
        (30, 2.5),
    ]

    for bb_p, std_m in bb_configs:
        (trades_bo, ps_bo), (trades_mr, ps_mr) = run_bb_squeeze(stock_data, bb_p, std_m)
        save_result("BB_Squeeze_Breakout", f"BB({bb_p},{std_m})", trades_bo, ps_bo)
        save_result("BB_Squeeze_MeanRev", f"BB({bb_p},{std_m})", trades_mr, ps_mr)

    # ═══════════════════════════════════════════════════════════════════════
    # 3. RSI MEAN REVERSION
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("STRATEGY 3: RSI Mean Reversion")
    print("="*70, flush=True)

    rsi_configs = [
        # (rsi_period, entry_thresh, exit_thresh, use_sma200, max_hold)
        (14, 30, 50, False, None),
        (14, 25, 50, False, None),
        (14, 20, 50, False, None),
        (14, 30, 45, False, None),
        (14, 30, 50, True, None),
        (14, 25, 50, True, None),
        (7, 30, 50, False, None),
        (7, 25, 50, False, None),
        (7, 20, 50, False, None),
        (7, 30, 50, True, None),
        (21, 30, 50, False, None),
        (21, 25, 50, False, None),
        (21, 30, 50, True, None),
        (14, 30, 50, False, 35),   # Max hold 5 days
        (14, 30, 50, True, 35),
        (7, 30, 50, False, 35),
        (7, 30, 50, True, 35),
    ]

    for rsi_p, entry_t, exit_t, sma, mh in rsi_configs:
        params = f"RSI({rsi_p})<{entry_t},exit>{exit_t}"
        if sma: params += "+SMA200"
        if mh: params += f"+MH{mh}"
        trades, per_stock = run_rsi_meanrev(stock_data, rsi_p, entry_t, exit_t, sma, mh)
        save_result("RSI_MeanRev", params, trades, per_stock)

    # ═══════════════════════════════════════════════════════════════════════
    # 4. PRICE DEVIATION FROM EMA
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("STRATEGY 4: Price Deviation from EMA")
    print("="*70, flush=True)

    ema_configs = [
        # (ema_period, atr_mult, max_hold)
        (20, 1.5, None),
        (20, 2.0, None),
        (20, 2.5, None),
        (20, 3.0, None),
        (50, 1.5, None),
        (50, 2.0, None),
        (50, 2.5, None),
        (20, 1.5, 35),
        (20, 2.0, 35),
        (20, 2.5, 35),
        (10, 1.5, None),
        (10, 2.0, None),
        (10, 2.5, None),
    ]

    for ema_p, atr_m, mh in ema_configs:
        params = f"EMA({ema_p})-{atr_m}ATR"
        if mh: params += f"+MH{mh}"
        trades, per_stock = run_ema_deviation(stock_data, ema_p, atr_m, mh)
        save_result("EMA_Deviation", params, trades, per_stock)

    # ═══════════════════════════════════════════════════════════════════════
    # 5. INSIDE BAR BREAKOUT
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("STRATEGY 5: Inside Bar Breakout")
    print("="*70, flush=True)

    ib_configs = [
        # (atr_target_mult, max_hold_bars)
        (2.0, 7),    # 1 day
        (2.0, 14),   # 2 days
        (2.0, 35),   # 5 days
        (3.0, 7),
        (3.0, 14),
        (1.5, 7),
        (1.5, 14),
    ]

    for atr_m, mh in ib_configs:
        params = f"ATR_tgt={atr_m},MH={mh}"
        trades, per_stock = run_inside_bar(stock_data, atr_m, mh)
        save_result("InsideBar_Breakout", params, trades, per_stock)

    # ═══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"COMPLETED: {strategy_count} strategies in {elapsed:.0f}s")
    print(f"Results saved to: {OUTPUT_CSV}")
    print(f"{'='*70}\n")

    # Rank by profit factor (only those with 100+ trades)
    viable = [r for r in results if r['total_trades'] >= 100 and r['profit_factor'] > 1.0]
    viable.sort(key=lambda x: x['profit_factor'], reverse=True)

    print(f"TOP STRATEGIES (PF > 1.0, 100+ trades):")
    print(f"{'Rank':<5} {'Strategy':<25} {'Params':<35} {'PF':>6} {'WR%':>6} {'Trades':>7} {'AvgTr%':>8} {'TotPnL%':>9}")
    print("-" * 105)
    for i, r in enumerate(viable[:30], 1):
        print(f"{i:<5} {r['strategy']:<25} {r['params']:<35} {r['profit_factor']:>6.2f} {r['win_rate']:>6.1f} {r['total_trades']:>7} {r['avg_trade_pct']:>8.3f} {r['total_pnl_pct']:>9.1f}")

    print(f"\n\nSTRATEGIES WITH PF > 1.2 AND 100+ TRADES:")
    strong = [r for r in results if r['total_trades'] >= 100 and r['profit_factor'] > 1.2]
    strong.sort(key=lambda x: x['profit_factor'], reverse=True)
    if strong:
        for r in strong:
            print(f"  {r['strategy']} ({r['params']}): PF={r['profit_factor']:.2f}, WR={r['win_rate']}%, "
                  f"Trades={r['total_trades']}, AvgTrade={r['avg_trade_pct']:.3f}%, Yearly={r['yearly_breakdown']}")
    else:
        print("  None found.")

    # Also show best by total PnL
    print(f"\nTOP 10 BY TOTAL PnL% (100+ trades):")
    by_pnl = sorted(viable, key=lambda x: x['total_pnl_pct'], reverse=True)
    for r in by_pnl[:10]:
        print(f"  {r['strategy']} ({r['params']}): TotalPnL={r['total_pnl_pct']:.1f}%, PF={r['profit_factor']:.2f}, Trades={r['total_trades']}")


if __name__ == '__main__':
    main()
