#!/usr/bin/env python3
"""
BB Expansion V2 — Candle-based SL + Confirmation Indicators
=============================================================
Changes from V1:
- SL = entry candle's low (long) / high (short) with ATR buffer
- TP = 2x risk (SL distance)
- Confirmation indicators: RSI, ADX, Volume expansion

Sweep: SL buffer, TP mult, max_hold, squeeze min bars, confirmations
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

OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'optimization_bb_expansion_v2.csv')
FIELDNAMES = [
    'label', 'total_trades', 'bb_trades',
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


def detect_bb_expansion_v2(sym, df, i, bb_period=20, sq_min_bars=3,
                            sl_buffer_atr=0.2, tp_mult=2.0, max_hold=10,
                            sma_period=20, use_rsi=False, use_adx=False,
                            use_vol=False):
    """BB Squeeze->Expansion V2: candle-based SL, confirmation indicators.

    SL logic:
    - LONG:  SL = candle low - (sl_buffer_atr * ATR)
    - SHORT: SL = candle high + (sl_buffer_atr * ATR)
    - TP = entry +/- tp_mult * risk

    Confirmations (optional):
    - RSI: long if RSI > 50, short if RSI < 50
    - ADX: ADX > 20 (trending market)
    - Volume: fire bar volume > 20-bar avg volume (volume expansion)
    """
    warmup = max(bb_period + sq_min_bars + 5, sma_period + 5, 30)
    if i < warmup:
        return []

    # Check fire
    if not df['bb_expanding'].iloc[i]:
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

    # First fire only
    if i > 0 and df['bb_expanding'].iloc[i - 1]:
        return []

    close = df['close'].iloc[i]
    low = df['low'].iloc[i]
    high = df['high'].iloc[i]
    sma_val = df[f'sma{sma_period}'].iloc[i]
    atr_val = df['atr14'].iloc[i]

    if atr_val <= 0 or np.isnan(atr_val) or np.isnan(sma_val):
        return []

    # Confirmation filters
    if use_rsi:
        rsi = df['rsi14'].iloc[i]
        if np.isnan(rsi):
            return []
    if use_adx:
        adx = df['adx14'].iloc[i]
        if np.isnan(adx) or adx < 20:
            return []
    if use_vol:
        vol = df['volume'].iloc[i]
        vol_ma = df['vol_ma20'].iloc[i]
        if np.isnan(vol_ma) or vol <= vol_ma:
            return []

    signals = []
    buffer = sl_buffer_atr * atr_val

    # LONG: close > SMA
    if close > sma_val:
        if use_rsi and rsi < 50:
            pass  # RSI says bearish, skip long
        else:
            entry = close
            sl = low - buffer
            risk = entry - sl
            if risk > 0:
                tp = entry + tp_mult * risk
                signals.append(('long_immediate', entry, sl, tp, max_hold, 'BBExpansion'))

    # SHORT: close < SMA
    if close < sma_val:
        if use_rsi and rsi > 50:
            pass  # RSI says bullish, skip short
        else:
            entry = close
            sl = high + buffer
            risk = sl - entry
            if risk > 0:
                tp = entry - tp_mult * risk
                signals.append(('short_immediate', entry, sl, tp, max_hold, 'BBExpansion'))

    return signals


def run_standalone_bb(all_data, bb_params, max_positions=20):
    """Run BB Expansion V2 as standalone strategy."""
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
            for sig_info in detect_bb_expansion_v2(sym, df, idx_loc, **bb_params):
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


def result_to_row(label, result, strat_counts):
    return {
        'label': label,
        'total_trades': result.total_trades,
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
    done = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV) as f:
            done = {row['label'] for row in csv.DictReader(f)}
        print(f'Found {len(done)} already-completed configs', flush=True)

    # ─── Configuration sweep ───
    configs = []
    base = {'bb_period': 20, 'sq_min_bars': 3, 'sl_buffer_atr': 0.2, 'tp_mult': 2.0,
            'max_hold': 10, 'sma_period': 20, 'use_rsi': False, 'use_adx': False, 'use_vol': False}

    # Baseline: candle SL with 0.2 ATR buffer, TP 2x
    configs.append(('BBv2_BASE', dict(base)))

    # SL buffer sweep
    for buf in [0.0, 0.1, 0.3, 0.5]:
        configs.append((f'BBv2_BUF{buf}', dict(base, sl_buffer_atr=buf)))

    # TP sweep
    for tp in [1.5, 2.5, 3.0]:
        configs.append((f'BBv2_TP{tp}', dict(base, tp_mult=tp)))

    # MaxHold sweep
    for mh in [5, 7, 15]:
        configs.append((f'BBv2_MH{mh}', dict(base, max_hold=mh)))

    # Squeeze min bars
    for sq in [2, 4, 5]:
        configs.append((f'BBv2_SQ{sq}', dict(base, sq_min_bars=sq)))

    # Confirmation: RSI only
    configs.append(('BBv2_RSI', dict(base, use_rsi=True)))

    # Confirmation: ADX only
    configs.append(('BBv2_ADX', dict(base, use_adx=True)))

    # Confirmation: Volume expansion only
    configs.append(('BBv2_VOL', dict(base, use_vol=True)))

    # Combined confirmations
    configs.append(('BBv2_RSI_ADX', dict(base, use_rsi=True, use_adx=True)))
    configs.append(('BBv2_RSI_VOL', dict(base, use_rsi=True, use_vol=True)))
    configs.append(('BBv2_ADX_VOL', dict(base, use_adx=True, use_vol=True)))
    configs.append(('BBv2_ALL_CONFIRM', dict(base, use_rsi=True, use_adx=True, use_vol=True)))

    # Best combo candidates
    configs.append(('BBv2_MH7_RSI', dict(base, max_hold=7, use_rsi=True)))
    configs.append(('BBv2_MH7_ADX_VOL', dict(base, max_hold=7, use_adx=True, use_vol=True)))
    configs.append(('BBv2_SQ4_RSI_VOL', dict(base, sq_min_bars=4, use_rsi=True, use_vol=True)))
    configs.append(('BBv2_TP1.5_MH7_RSI', dict(base, tp_mult=1.5, max_hold=7, use_rsi=True)))
    configs.append(('BBv2_BUF0.1_TP1.5_MH7', dict(base, sl_buffer_atr=0.1, tp_mult=1.5, max_hold=7)))

    remaining = [(l, p) for l, p in configs if l not in done]
    if not remaining:
        print('All configs already completed!', flush=True)
        return

    print(f'\n=== Running {len(remaining)} BB Expansion V2 configs ===\n', flush=True)

    print('Loading data for 50 F&O stocks...', flush=True)
    t0 = time.time()
    all_data = load_data_from_db(FNO_STOCKS, 'day', '2018-01-01', '2025-12-31')
    print(f'Loaded {len(all_data)} symbols in {time.time() - t0:.1f}s', flush=True)

    print('Computing indicators...', flush=True)
    for sym, df in all_data.items():
        df['atr14'] = calc_atr(df, 14)

        # BB
        sma_bb = df['close'].rolling(20).mean()
        std_bb = df['close'].rolling(20).std()
        bb_width = (2 * 2.0 * std_bb) / sma_bb * 100
        bb_width_ma = bb_width.rolling(20).mean()
        df['bb_squeeze'] = (bb_width < bb_width_ma).astype(int)
        df['bb_expanding'] = (bb_width > bb_width_ma).astype(int)

        # SMA
        df['sma20'] = df['close'].rolling(20).mean()

        # RSI(14)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi14'] = 100 - (100 / (1 + rs))

        # ADX(14)
        high, low, close_s = df['high'], df['low'], df['close']
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        tr = pd.concat([high - low, (high - close_s.shift(1)).abs(), (low - close_s.shift(1)).abs()], axis=1).max(axis=1)
        atr_adx = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr_adx)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr_adx)
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di) * 100
        df['adx14'] = dx.rolling(14).mean()

        # Volume MA
        df['vol_ma20'] = df['volume'].rolling(20).mean()
    print('Indicators ready.', flush=True)

    # Ensure CSV exists
    if not os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    for i, (label, bb_params) in enumerate(remaining):
        print(f'[{i+1}/{len(remaining)}] {label} ...', end='', flush=True)
        t1 = time.time()

        result, strat_counts = run_standalone_bb(all_data, bb_params)

        elapsed = time.time() - t1
        print(f' {elapsed:.0f}s | CAGR={result.cagr:.2f}% WR={result.win_rate:.1f}% '
              f'MaxDD={result.max_drawdown:.2f}% PF={result.profit_factor:.2f} '
              f'Trades={result.total_trades}', flush=True)

        row = result_to_row(label, result, strat_counts)
        with open(OUTPUT_CSV, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

    print(f'\nAll configs done. Results in {OUTPUT_CSV}', flush=True)


if __name__ == '__main__':
    main()
