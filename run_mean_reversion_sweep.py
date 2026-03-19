"""
Mean Reversion Strategy Sweep — Daily Data, F&O Stocks (2018-2025)
===================================================================
Tests 6 mean reversion variants on 50 F&O stocks using the universal backtest engine.

Strategies:
1. KC6-style: Close < KC(6,1.3) lower AND Close > SMA(200). Exit KC mid. SL 5%, TP 15%, MaxHold 15d.
2. RSI Extreme + Volume: RSI(2)<10, volume>2x avg. Fixed 2:1 RR. MaxHold 5d.
3. BB Bounce: Close < BB(20,2) lower AND RSI(14)<30. Exit BB mid. SL 1.5x ATR. MaxHold 10d.
4. Keltner Wide: Close < KC(20,2.5) lower AND close > SMA(100). Exit KC mid. MaxHold 15d.
5. Oversold + Trend: RSI(14)<30 AND close > EMA(200) AND ADX(14)>20. Exit EMA(9). MaxHold 10d.
6. Gap Down Reversal: Open < prev low by >1% AND close > open. SL below low, target 2x risk. MaxHold 5d.
"""

import sys
import os
import csv
import time
import logging
import warnings

# Suppress noise
logging.disable(logging.WARNING)
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from services.intraday_backtest_engine import (
    IntradayBacktestEngine, TradeSignal, Direction, ExitType,
    load_data_from_db, BacktestResult
)
from services.technical_indicators import (
    calc_ema, calc_rsi, calc_atr, calc_bollinger_bands,
    calc_keltner_channels, calc_adx
)

# ─── Config ──────────────────────────────────────────────────────────────────
FNO_50 = [
    'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'SBIN', 'BHARTIARTL',
    'ITC', 'KOTAKBANK', 'HINDUNILVR', 'LT', 'AXISBANK', 'BAJFINANCE', 'MARUTI',
    'HCLTECH', 'SUNPHARMA', 'TATAMOTORS', 'NTPC', 'POWERGRID', 'TITAN',
    'WIPRO', 'ULTRACEMCO', 'ADANIENT', 'ADANIPORTS', 'NESTLEIND', 'JSWSTEEL',
    'TATASTEEL', 'TECHM', 'M&M', 'COALINDIA', 'ONGC', 'GRASIM', 'BAJAJFINSV',
    'APOLLOHOSP', 'DIVISLAB', 'DRREDDY', 'CIPLA', 'HEROMOTOCO', 'EICHERMOT',
    'BPCL', 'INDUSINDBK', 'TATACONSUM', 'SHRIRAMFIN', 'BRITANNIA', 'HINDALCO',
    'ASIANPAINT', 'SBILIFE', 'HDFCLIFE', 'PIDILITIND', 'DABUR',
]

INITIAL_CAPITAL = 10_000_000  # Rs 1 Crore
POSITION_SIZE_PCT = 0.10       # 10% per position
MAX_POSITIONS = 10
COMMISSION_PCT = 0.001         # 0.1%
SLIPPAGE_PCT = 0.001           # 0.1%

START_DATE = '2018-01-01'
END_DATE = '2025-12-31'

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mean_reversion_results.csv')
FIELDNAMES = [
    'strategy', 'total_trades', 'win_rate', 'profit_factor', 'cagr', 'sharpe',
    'sortino', 'calmar', 'max_drawdown', 'total_pnl', 'total_pnl_pct',
    'avg_win', 'avg_loss', 'avg_rr', 'avg_bars_held',
    'exit_sl', 'exit_tp', 'exit_maxhold', 'exit_signal',
    # Year-by-year
    'y2018_trades', 'y2018_wr', 'y2018_pf', 'y2018_pnl_pct',
    'y2019_trades', 'y2019_wr', 'y2019_pf', 'y2019_pnl_pct',
    'y2020_trades', 'y2020_wr', 'y2020_pf', 'y2020_pnl_pct',
    'y2021_trades', 'y2021_wr', 'y2021_pf', 'y2021_pnl_pct',
    'y2022_trades', 'y2022_wr', 'y2022_pf', 'y2022_pnl_pct',
    'y2023_trades', 'y2023_wr', 'y2023_pf', 'y2023_pnl_pct',
    'y2024_trades', 'y2024_wr', 'y2024_pf', 'y2024_pnl_pct',
    'y2025_trades', 'y2025_wr', 'y2025_pf', 'y2025_pnl_pct',
]


# ─── Indicator Pre-computation ──────────────────────────────────────────────

def precompute_indicators(data: dict) -> dict:
    """Pre-compute all indicators needed for all 6 strategies on each symbol's df."""
    enriched = {}
    for sym, df in data.items():
        if len(df) < 200:
            continue
        d = df.copy()

        # SMA(200), SMA(100)
        d['sma200'] = d['close'].rolling(200).mean()
        d['sma100'] = d['close'].rolling(100).mean()

        # EMA(200), EMA(9)
        d['ema200'] = calc_ema(d['close'], 200)
        d['ema9'] = calc_ema(d['close'], 9)

        # ATR(14)
        d['atr14'] = calc_atr(d, 14)

        # RSI(2), RSI(14)
        d['rsi2'] = calc_rsi(d['close'], 2)
        d['rsi14'] = calc_rsi(d['close'], 14)

        # Volume avg (20-day)
        d['vol_avg20'] = d['volume'].rolling(20).mean()

        # Keltner Channel (6, 1.3 ATR) — KC6 style
        d['kc6_mid'], d['kc6_upper'], d['kc6_lower'] = calc_keltner_channels(
            d, ema_period=6, atr_period=6, multiplier=1.3
        )

        # Keltner Channel (20, 2.5 ATR) — Wide
        d['kc20_mid'], d['kc20_upper'], d['kc20_lower'] = calc_keltner_channels(
            d, ema_period=20, atr_period=10, multiplier=2.5
        )

        # Bollinger Bands (20, 2.0)
        d['bb_mid'], d['bb_upper'], d['bb_lower'] = calc_bollinger_bands(d, 20, 2.0)

        # ADX(14)
        d['adx14'], d['plus_di'], d['minus_di'] = calc_adx(d, 14)

        # Previous day's low (for gap down)
        d['prev_low'] = d['low'].shift(1)

        enriched[sym] = d
    return enriched


# ─── Strategy Signal Generators ─────────────────────────────────────────────
# Each returns (entry_signal: bool, TradeSignal or None) for bar i

def strategy_kc6(df, i):
    """KC6-style: Close < KC(6,1.3) lower AND Close > SMA(200). Exit KC mid. SL 5%, TP 15%, MaxHold 15."""
    if i < 200:
        return False, None
    row = df.iloc[i]
    if pd.isna(row['kc6_lower']) or pd.isna(row['sma200']):
        return False, None
    if row['close'] < row['kc6_lower'] and row['close'] > row['sma200']:
        entry = row['close']
        sl = entry * 0.95       # 5% SL
        tp = row['kc6_mid']     # Exit at KC mid
        if tp <= entry:
            tp = entry * 1.15   # Fallback 15% TP
        return True, TradeSignal(
            direction=Direction.LONG,
            entry_price=entry,
            stop_loss=sl,
            target=tp,
            max_hold_bars=15,
        )
    return False, None


def strategy_rsi_extreme(df, i):
    """RSI(2)<10, volume > 2x avg. Fixed 2:1 RR. MaxHold 5d."""
    if i < 200:
        return False, None
    row = df.iloc[i]
    if pd.isna(row['rsi2']) or pd.isna(row['vol_avg20']) or pd.isna(row['atr14']):
        return False, None
    if row['rsi2'] < 10 and row['volume'] > 2 * row['vol_avg20']:
        entry = row['close']
        risk = row['atr14'] * 1.5
        sl = entry - risk
        tp = entry + risk * 2  # 2:1 RR
        return True, TradeSignal(
            direction=Direction.LONG,
            entry_price=entry,
            stop_loss=sl,
            target=tp,
            max_hold_bars=5,
        )
    return False, None


def strategy_bb_bounce(df, i):
    """Close < BB(20,2) lower AND RSI(14)<30. Exit BB mid. SL 1.5x ATR. MaxHold 10d."""
    if i < 200:
        return False, None
    row = df.iloc[i]
    if pd.isna(row['bb_lower']) or pd.isna(row['rsi14']) or pd.isna(row['atr14']):
        return False, None
    if row['close'] < row['bb_lower'] and row['rsi14'] < 30:
        entry = row['close']
        sl = entry - 1.5 * row['atr14']
        tp = row['bb_mid']
        if tp <= entry:
            tp = entry + 1.5 * row['atr14']  # Fallback
        return True, TradeSignal(
            direction=Direction.LONG,
            entry_price=entry,
            stop_loss=sl,
            target=tp,
            max_hold_bars=10,
        )
    return False, None


def strategy_keltner_wide(df, i):
    """Close < KC(20,2.5) lower AND close > SMA(100). Exit KC mid. SL 5%. MaxHold 15d."""
    if i < 200:
        return False, None
    row = df.iloc[i]
    if pd.isna(row['kc20_lower']) or pd.isna(row['sma100']):
        return False, None
    if row['close'] < row['kc20_lower'] and row['close'] > row['sma100']:
        entry = row['close']
        sl = entry * 0.95
        tp = row['kc20_mid']
        if tp <= entry:
            tp = entry * 1.10
        return True, TradeSignal(
            direction=Direction.LONG,
            entry_price=entry,
            stop_loss=sl,
            target=tp,
            max_hold_bars=15,
        )
    return False, None


def strategy_oversold_trend(df, i):
    """RSI(14)<30 AND close > EMA(200) AND ADX(14)>20. Exit at EMA(9). MaxHold 10d."""
    if i < 200:
        return False, None
    row = df.iloc[i]
    if pd.isna(row['rsi14']) or pd.isna(row['ema200']) or pd.isna(row['adx14']):
        return False, None
    if row['rsi14'] < 30 and row['close'] > row['ema200'] and row['adx14'] > 20:
        entry = row['close']
        sl = entry - 1.5 * row['atr14']
        tp = row['ema9']
        if tp <= entry:
            tp = entry + row['atr14']  # Fallback
        return True, TradeSignal(
            direction=Direction.LONG,
            entry_price=entry,
            stop_loss=sl,
            target=tp,
            max_hold_bars=10,
        )
    return False, None


def strategy_gap_down_reversal(df, i):
    """Open < prev low by >1% AND close > open. SL below low, target 2x risk. MaxHold 5d."""
    if i < 200:
        return False, None
    row = df.iloc[i]
    if pd.isna(row['prev_low']):
        return False, None
    gap_pct = (row['prev_low'] - row['open']) / row['prev_low']
    if gap_pct > 0.01 and row['close'] > row['open']:
        entry = row['close']
        sl = row['low'] * 0.999  # Just below today's low
        risk = entry - sl
        if risk <= 0:
            return False, None
        tp = entry + 2 * risk
        return True, TradeSignal(
            direction=Direction.LONG,
            entry_price=entry,
            stop_loss=sl,
            target=tp,
            max_hold_bars=5,
        )
    return False, None


# Strategy 5 needs dynamic exit at EMA(9) — we handle it in the check_exits override
STRATEGY_FUNCS = {
    '1_KC6_Style': strategy_kc6,
    '2_RSI_Extreme_Vol': strategy_rsi_extreme,
    '3_BB_Bounce': strategy_bb_bounce,
    '4_Keltner_Wide': strategy_keltner_wide,
    '5_Oversold_Trend': strategy_oversold_trend,
    '6_Gap_Down_Rev': strategy_gap_down_reversal,
}


# ─── Run a single strategy across all symbols ───────────────────────────────

def run_strategy(strategy_name, strategy_func, enriched_data):
    """Run one strategy across all symbols, return BacktestResult."""
    engine = IntradayBacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        position_size_pct=POSITION_SIZE_PCT,
        max_positions=MAX_POSITIONS,
        commission_pct=COMMISSION_PCT,
        slippage_pct=SLIPPAGE_PCT,
        mode='cash',
        fixed_sizing=True,
    )

    # Collect all symbols' dates into a sorted master timeline
    all_dates = set()
    for sym, df in enriched_data.items():
        for ds in df['date_str'].values:
            all_dates.add(ds)
    sorted_dates = sorted(all_dates)

    # Build per-symbol date index map for quick lookup
    sym_date_idx = {}
    for sym, df in enriched_data.items():
        date_to_idx = {ds: idx for idx, ds in enumerate(df['date_str'].values)}
        sym_date_idx[sym] = date_to_idx

    for date_str in sorted_dates:
        # Check exits first for all open positions
        for sym in list(engine.positions.keys()):
            if sym not in enriched_data or date_str not in sym_date_idx[sym]:
                continue
            df = enriched_data[sym]
            idx = sym_date_idx[sym][date_str]
            row = df.iloc[idx]

            # For strategy 5, dynamic exit at EMA(9)
            target_override = None
            if strategy_name == '5_Oversold_Trend' and not pd.isna(row.get('ema9', np.nan)):
                pos = engine.positions[sym]
                # Update target to current EMA(9) each bar
                new_target = row['ema9']
                if new_target > pos.entry_price:
                    pos.target = new_target

            engine.check_exits(
                symbol=sym,
                bar_idx=idx,
                bar_date=date_str,
                high=row['high'],
                low=row['low'],
                close=row['close'],
                atr=row.get('atr14', None),
            )

        # Generate entry signals for all symbols
        for sym, df in enriched_data.items():
            if date_str not in sym_date_idx[sym]:
                continue
            if sym in engine.positions:
                continue
            idx = sym_date_idx[sym][date_str]
            entry, signal = strategy_func(df, idx)
            if entry and signal:
                engine.open_position(sym, signal, idx, date_str)

        # Update equity
        prices = {}
        for sym in engine.positions:
            if sym in enriched_data and date_str in sym_date_idx[sym]:
                idx = sym_date_idx[sym][date_str]
                prices[sym] = enriched_data[sym].iloc[idx]['close']
        engine.update_equity(date_str, prices)

    result = engine.get_results()
    return result


def result_to_row(strategy_name, result):
    """Convert BacktestResult to CSV row dict."""
    row = {
        'strategy': strategy_name,
        'total_trades': result.total_trades,
        'win_rate': round(result.win_rate, 2),
        'profit_factor': round(result.profit_factor, 2),
        'cagr': round(result.cagr, 2),
        'sharpe': round(result.sharpe_ratio, 2),
        'sortino': round(result.sortino_ratio, 2),
        'calmar': round(result.calmar_ratio, 2),
        'max_drawdown': round(result.max_drawdown, 2),
        'total_pnl': round(result.total_pnl, 0),
        'total_pnl_pct': round(result.total_pnl_pct, 2),
        'avg_win': round(result.avg_win, 2),
        'avg_loss': round(result.avg_loss, 2),
        'avg_rr': round(result.avg_rr, 2),
        'avg_bars_held': round(result.avg_bars_held, 1) if result.total_trades > 0 else 0,
        'exit_sl': result.exit_reasons.get('fixed_sl', 0),
        'exit_tp': result.exit_reasons.get('fixed_tp', 0),
        'exit_maxhold': result.exit_reasons.get('max_hold', 0),
        'exit_signal': result.exit_reasons.get('signal_exit', 0),
    }
    for y in range(2018, 2026):
        ys = result.yearly_stats.get(y, {})
        row[f'y{y}_trades'] = ys.get('trades', 0)
        row[f'y{y}_wr'] = round(ys.get('win_rate', 0), 1)
        row[f'y{y}_pf'] = round(ys.get('profit_factor', 0), 2)
        row[f'y{y}_pnl_pct'] = round(ys.get('pnl_pct', 0), 2)
    return row


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print(f'Mean Reversion Sweep — {len(FNO_50)} F&O stocks, daily {START_DATE} to {END_DATE}')
    print(f'Capital: Rs {INITIAL_CAPITAL:,.0f}, Position: {POSITION_SIZE_PCT*100:.0f}%, Max positions: {MAX_POSITIONS}')
    print()

    # Load data
    print('Loading daily data...', end='', flush=True)
    data = load_data_from_db(FNO_50, 'day', START_DATE, END_DATE)
    loaded = [s for s in FNO_50 if s in data]
    print(f' {len(loaded)}/{len(FNO_50)} symbols loaded ({time.time()-t0:.0f}s)')

    # Pre-compute indicators
    print('Computing indicators...', end='', flush=True)
    enriched = precompute_indicators(data)
    print(f' done for {len(enriched)} symbols ({time.time()-t0:.0f}s)')
    print()

    # Check already completed
    done = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV) as f:
            done = {r['strategy'] for r in csv.DictReader(f)}
        if done:
            print(f'Skipping {len(done)} already-completed strategies')

    # Write header if new file
    if not done:
        with open(OUTPUT_CSV, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    # Run strategies
    total = len(STRATEGY_FUNCS)
    for idx, (name, func) in enumerate(STRATEGY_FUNCS.items(), 1):
        if name in done:
            print(f'[{idx}/{total}] {name} — SKIPPED (already done)')
            continue

        print(f'[{idx}/{total}] {name} ...', end='', flush=True)
        t1 = time.time()
        result = run_strategy(name, func, enriched)
        elapsed = time.time() - t1

        row = result_to_row(name, result)
        with open(OUTPUT_CSV, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

        print(f' {elapsed:.0f}s | Trades={result.total_trades} WR={result.win_rate:.1f}% '
              f'PF={result.profit_factor:.2f} CAGR={result.cagr:.2f}% MaxDD={result.max_drawdown:.1f}%')

        # Print year-by-year
        if result.yearly_stats:
            for y in sorted(result.yearly_stats.keys()):
                ys = result.yearly_stats[y]
                print(f'        {y}: {ys["trades"]:>3} trades  WR={ys["win_rate"]:5.1f}%  '
                      f'PF={ys["profit_factor"]:5.2f}  PnL={ys["pnl_pct"]:+7.2f}%')
        print()

    print(f'\nAll done in {time.time()-t0:.0f}s. Results saved to {OUTPUT_CSV}')

    # Summary table
    print(f'\n{"="*90}')
    print(f'{"Strategy":<22} {"Trades":>6} {"WR%":>6} {"PF":>6} {"CAGR%":>7} {"MaxDD%":>7} {"Sharpe":>7} {"AvgBars":>7}')
    print(f'{"-"*90}')
    with open(OUTPUT_CSV) as f:
        for r in csv.DictReader(f):
            print(f'{r["strategy"]:<22} {r["total_trades"]:>6} {r["win_rate"]:>6} '
                  f'{r["profit_factor"]:>6} {r["cagr"]:>7} {r["max_drawdown"]:>7} '
                  f'{r["sharpe"]:>7} {r["avg_bars_held"]:>7}')
    print(f'{"="*90}')


if __name__ == '__main__':
    main()
