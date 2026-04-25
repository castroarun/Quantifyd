"""Intraday MA Crossover sweep — one signal timeframe per run.

Usage:
    python run_intraday_ma_crossover.py --tf 5min
    python run_intraday_ma_crossover.py --tf 10min
    python run_intraday_ma_crossover.py --tf 15min
    python run_intraday_ma_crossover.py --tf 30min

For each TF, tests 8 variants (baseline + 6 filter combos + fast pair) across
15 non-ORB liquid F&O stocks, 2024-03-18 to 2026-03-12.

Writes incrementally to results/tf_{XXmin}/:
  - trades.csv        (per-trade log)
  - daily_pnl.csv     (per-day series per variant)
  - summary.csv       (portfolio metrics per variant)
  - progress.txt      (heartbeat for crash recovery)
"""
from __future__ import annotations

import argparse
import csv
import logging
import sqlite3
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, time as dtime
from pathlib import Path

import numpy as np
import pandas as pd

logging.disable(logging.WARNING)
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / 'backtest_data' / 'market_data.db'
OUT_ROOT = Path(__file__).resolve().parents[1] / 'results'

UNIVERSE = [
    'INDUSINDBK', 'PNB', 'BANKBARODA',
    'WIPRO', 'HCLTECH',
    'MARUTI', 'HEROMOTOCO', 'EICHERMOT',
    'TITAN', 'ASIANPAINT',
    'JSWSTEEL', 'HINDALCO', 'JINDALSTEL',
    'LT', 'ADANIPORTS',
]

START_DATE = '2024-03-18'
END_DATE   = '2026-03-12'

# Per-TF EMA periods (bars), plus max bars-per-day for context
TF_CONFIG = {
    '5min':  {'freq': '5min',  'fast': 20, 'slow': 50, 'bars_per_day': 75, 'time_stop': 24, 'eod_bars_before_close': 3},
    '10min': {'freq': '10min', 'fast': 10, 'slow': 30, 'bars_per_day': 38, 'time_stop': 12, 'eod_bars_before_close': 2},
    '15min': {'freq': '15min', 'fast':  8, 'slow': 21, 'bars_per_day': 25, 'time_stop':  8, 'eod_bars_before_close': 2},
    '30min': {'freq': '30min', 'fast':  5, 'slow': 13, 'bars_per_day': 13, 'time_stop':  5, 'eod_bars_before_close': 1},
}

# Windows + shared defaults
ATR_PERIOD        = 14
RSI_PERIOD        = 14
BB_PERIOD         = 20
BB_K              = 2
HTF_EMA           = 21            # always on 60-min

ENTRY_WINDOW_START = dtime(9, 45)
ENTRY_WINDOW_END   = dtime(14, 0)
EOD_EXIT           = dtime(15, 15)

STOP_ATR_MULT     = 1.0
R_MULTIPLE        = 1.5

RISK_PER_TRADE    = 2_500
MAX_NOTIONAL      = 300_000
COST_PCT          = 0.0015

VARIANTS = {
    'baseline':   {'vwap': False, 'htf': False, 'rsi50': False, 'cpr': False, 'bb': False, 'fast_pair': False},
    'vwap':       {'vwap': True,  'htf': False, 'rsi50': False, 'cpr': False, 'bb': False, 'fast_pair': False},
    'htf':        {'vwap': False, 'htf': True,  'rsi50': False, 'cpr': False, 'bb': False, 'fast_pair': False},
    'rsi50':      {'vwap': False, 'htf': False, 'rsi50': True,  'cpr': False, 'bb': False, 'fast_pair': False},
    'cpr':        {'vwap': False, 'htf': False, 'rsi50': False, 'cpr': True,  'bb': False, 'fast_pair': False},
    'bb':         {'vwap': False, 'htf': False, 'rsi50': False, 'cpr': False, 'bb': True,  'fast_pair': False},
    'confluence': {'vwap': True,  'htf': True,  'rsi50': True,  'cpr': False, 'bb': False, 'fast_pair': False},
    'fast_pair':  {'vwap': False, 'htf': False, 'rsi50': False, 'cpr': False, 'bb': False, 'fast_pair': True},
}


@dataclass
class Trade:
    variant: str; symbol: str; date: str; direction: str
    entry_time: str; entry: float; stop: float; target: float; qty: int
    atr_at_entry: float
    exit_time: str = ''; exit_price: float = 0.0; exit_reason: str = ''
    gross_pnl: float = 0.0; net_pnl: float = 0.0

    def close(self, exit_time, exit_price, reason):
        self.exit_time, self.exit_price, self.exit_reason = exit_time, exit_price, reason
        sign = 1 if self.direction == 'LONG' else -1
        self.gross_pnl = sign * (exit_price - self.entry) * self.qty
        cost = COST_PCT * (self.entry + exit_price) * self.qty / 2.0
        self.net_pnl = self.gross_pnl - cost


def load_5min(symbol: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query(
        """SELECT date, open, high, low, close, volume FROM market_data_unified
            WHERE symbol=? AND timeframe='5minute' AND date>=? AND date<=?
         ORDER BY date""",
        conn, params=(symbol, START_DATE, END_DATE + ' 23:59:59'),
    )
    conn.close()
    if df.empty: return df
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df


def resample_bars(df5: pd.DataFrame, freq: str) -> pd.DataFrame:
    if df5.empty: return df5
    if freq == '5min':
        out = df5.copy()
    else:
        out = df5.resample(freq, label='right', closed='right').agg(
            {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
        ).dropna()
    out['day'] = out.index.date
    return out


def add_indicators(df: pd.DataFrame, df5: pd.DataFrame, fast: int, slow: int, fast_pair_periods) -> pd.DataFrame:
    """Compute all indicators. fast_pair_periods = (f, s) for the fast_pair variant."""
    if df.empty: return df
    # EMAs (primary pair)
    df['ema_fast'] = df['close'].ewm(span=fast, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=slow, adjust=False).mean()
    # Fast pair (for fast_pair variant) - 9/21 style
    fp, sp = fast_pair_periods
    df['ema_fp_fast'] = df['close'].ewm(span=fp, adjust=False).mean()
    df['ema_fp_slow'] = df['close'].ewm(span=sp, adjust=False).mean()

    # ATR
    pc = df['close'].shift(1)
    tr = pd.concat([df['high'] - df['low'],
                    (df['high'] - pc).abs(),
                    (df['low']  - pc).abs()], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1 / ATR_PERIOD, adjust=False).mean()

    # RSI
    ch = df['close'].diff()
    g = ch.clip(lower=0); l = -ch.clip(upper=0)
    ag = g.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    al = l.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    df['rsi'] = 100 - 100 / (1 + rs)

    # BB
    df['bb_mid'] = df['close'].rolling(BB_PERIOD).mean()
    bb_std = df['close'].rolling(BB_PERIOD).std()
    df['bb_upper'] = df['bb_mid'] + BB_K * bb_std
    df['bb_lower'] = df['bb_mid'] - BB_K * bb_std

    # Intraday VWAP (from 5-min — reset each day)
    typ5 = (df5['high'] + df5['low'] + df5['close']) / 3.0
    day5 = df5.index.date
    cum_pv = (typ5 * df5['volume']).groupby(day5).cumsum()
    cum_v  = df5['volume'].groupby(day5).cumsum()
    vwap5  = cum_pv / cum_v
    # Reindex to our TF using as-of (last known)
    df['vwap'] = vwap5.reindex(df.index, method='ffill')

    # HTF filter — 60-min EMA(21) from 5-min resample
    sixty = df5[['open','high','low','close','volume']].resample('60min', label='right', closed='right').agg(
        {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}
    ).dropna()
    sixty['htf_ema'] = sixty['close'].ewm(span=HTF_EMA, adjust=False).mean()
    sixty['htf_slope'] = sixty['htf_ema'].diff()
    df['htf_slope'] = sixty['htf_slope'].reindex(df.index, method='ffill')

    # CPR — previous day's pivot
    daily = df.groupby('day').agg(dh=('high','max'), dl=('low','min'), dc=('close','last'))
    daily['pivot'] = (daily['dh'] + daily['dl'] + daily['dc']) / 3.0
    daily_prev = daily[['pivot']].shift(1)
    df = df.join(daily_prev, on='day')

    # Cross flags
    df['cross_up']   = (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1)) & (df['ema_fast'] > df['ema_slow'])
    df['cross_down'] = (df['ema_fast'].shift(1) >= df['ema_slow'].shift(1)) & (df['ema_fast'] < df['ema_slow'])
    df['fp_cross_up']   = (df['ema_fp_fast'].shift(1) <= df['ema_fp_slow'].shift(1)) & (df['ema_fp_fast'] > df['ema_fp_slow'])
    df['fp_cross_down'] = (df['ema_fp_fast'].shift(1) >= df['ema_fp_slow'].shift(1)) & (df['ema_fp_fast'] < df['ema_fp_slow'])

    return df


def signal_ok(row, direction: str, cfg: dict) -> bool:
    if cfg['vwap']:
        if pd.isna(row.vwap): return False
        if direction == 'LONG' and not (row.close > row.vwap): return False
        if direction == 'SHORT' and not (row.close < row.vwap): return False
    if cfg['htf']:
        if pd.isna(row.htf_slope): return False
        if direction == 'LONG' and not (row.htf_slope > 0): return False
        if direction == 'SHORT' and not (row.htf_slope < 0): return False
    if cfg['rsi50']:
        if pd.isna(row.rsi): return False
        if direction == 'LONG' and not (row.rsi > 50): return False
        if direction == 'SHORT' and not (row.rsi < 50): return False
    if cfg['cpr']:
        if pd.isna(row.pivot): return False
        if direction == 'LONG' and not (row.close > row.pivot): return False
        if direction == 'SHORT' and not (row.close < row.pivot): return False
    if cfg['bb']:
        if pd.isna(row.bb_mid): return False
        if direction == 'LONG' and not (row.close > row.bb_mid): return False
        if direction == 'SHORT' and not (row.close < row.bb_mid): return False
    return True


def run_stock_variant(df: pd.DataFrame, symbol: str, variant: str, cfg: dict, tf_cfg: dict):
    if df.empty: return [], {}
    trades = []
    daily_pnl = {}
    time_stop_bars = tf_cfg['time_stop']

    for day, day_df in df.groupby('day'):
        active = None; bars_held = 0; traded_today = False; pending = None
        day_iter = list(day_df.itertuples())

        for i, row in enumerate(day_iter):
            ts = row.Index; t = ts.time()

            # Fill pending at this bar's open
            if pending is not None and active is None and not traded_today:
                direction, atr_e = pending
                entry_px = row.open
                if pd.notna(entry_px) and pd.notna(atr_e) and atr_e > 0:
                    if direction == 'LONG':
                        stop = entry_px - STOP_ATR_MULT * atr_e
                        target = entry_px + R_MULTIPLE * STOP_ATR_MULT * atr_e
                        rps = entry_px - stop
                    else:
                        stop = entry_px + STOP_ATR_MULT * atr_e
                        target = entry_px - R_MULTIPLE * STOP_ATR_MULT * atr_e
                        rps = stop - entry_px
                    if rps > 0:
                        qty = int(RISK_PER_TRADE // rps)
                        if qty * entry_px > MAX_NOTIONAL:
                            qty = int(MAX_NOTIONAL // entry_px)
                        if qty > 0:
                            active = Trade(variant, symbol, day.isoformat(), direction, ts.isoformat(),
                                           entry_px, stop, target, qty, atr_e)
                            bars_held = 0; traded_today = True
                pending = None

            # Exit logic
            if active is not None:
                bars_held += 1
                hit_stop = (active.direction == 'LONG'  and row.low  <= active.stop) \
                        or (active.direction == 'SHORT' and row.high >= active.stop)
                hit_tgt  = (active.direction == 'LONG'  and row.high >= active.target) \
                        or (active.direction == 'SHORT' and row.low  <= active.target)
                # Reverse signal using SAME cross we entered on
                if cfg['fast_pair']:
                    rev = (active.direction == 'LONG'  and bool(row.fp_cross_down)) \
                       or (active.direction == 'SHORT' and bool(row.fp_cross_up))
                else:
                    rev = (active.direction == 'LONG'  and bool(row.cross_down)) \
                       or (active.direction == 'SHORT' and bool(row.cross_up))

                if hit_stop:
                    active.close(ts.isoformat(), active.stop, 'STOP')
                    trades.append(active); daily_pnl[day.isoformat()] = daily_pnl.get(day.isoformat(), 0) + active.net_pnl
                    active = None
                elif hit_tgt:
                    active.close(ts.isoformat(), active.target, 'TARGET')
                    trades.append(active); daily_pnl[day.isoformat()] = daily_pnl.get(day.isoformat(), 0) + active.net_pnl
                    active = None
                elif rev:
                    active.close(ts.isoformat(), row.close, 'REVERSE')
                    trades.append(active); daily_pnl[day.isoformat()] = daily_pnl.get(day.isoformat(), 0) + active.net_pnl
                    active = None
                elif bars_held >= time_stop_bars:
                    active.close(ts.isoformat(), row.close, 'TIME_STOP')
                    trades.append(active); daily_pnl[day.isoformat()] = daily_pnl.get(day.isoformat(), 0) + active.net_pnl
                    active = None
                elif t >= EOD_EXIT:
                    active.close(ts.isoformat(), row.close, 'EOD')
                    trades.append(active); daily_pnl[day.isoformat()] = daily_pnl.get(day.isoformat(), 0) + active.net_pnl
                    active = None

            # Entry detection on cross close
            if (active is None and pending is None and not traded_today
                    and ENTRY_WINDOW_START <= t <= ENTRY_WINDOW_END
                    and pd.notna(row.atr) and row.atr > 0):
                direction = None
                if cfg['fast_pair']:
                    if bool(row.fp_cross_up) and signal_ok(row, 'LONG', cfg): direction = 'LONG'
                    elif bool(row.fp_cross_down) and signal_ok(row, 'SHORT', cfg): direction = 'SHORT'
                else:
                    if bool(row.cross_up) and signal_ok(row, 'LONG', cfg): direction = 'LONG'
                    elif bool(row.cross_down) and signal_ok(row, 'SHORT', cfg): direction = 'SHORT'
                if direction:
                    pending = (direction, row.atr)

        if active is not None and day_iter:
            last = day_iter[-1]
            active.close(last.Index.isoformat(), last.close, 'EOD_FORCED')
            trades.append(active); daily_pnl[day.isoformat()] = daily_pnl.get(day.isoformat(), 0) + active.net_pnl

    return trades, daily_pnl


def compute_metrics(trades, daily):
    if not trades:
        return {k: 0 for k in ['trades','wins','losses','win_rate_pct','avg_win','avg_loss',
                               'profit_factor','net_pnl','gross_pnl','costs','days_traded',
                               'cagr_pct','sharpe','max_dd','max_dd_pct','calmar']}
    net = [t.net_pnl for t in trades]
    wins = [p for p in net if p > 0]; losses = [p for p in net if p < 0]
    total = sum(net)
    sd = sorted(daily.keys())
    series = [daily[d] for d in sd]
    r = peak = mdd = 0.0
    for p in series:
        r += p
        if r > peak: peak = r
        mdd = max(mdd, peak - r)
    n = len(series)
    sharpe = 0.0
    if n > 1:
        m = sum(series) / n
        s = (sum((x-m)**2 for x in series) / (n-1))**0.5
        if s > 0: sharpe = (m/s) * (252**0.5)
    cap = 300_000
    yrs = n / 252 if n else 1
    end = cap + total
    cagr = ((end/cap)**(1/yrs) - 1) * 100 if yrs > 0 and end > 0 else 0.0
    return {
        'trades': len(trades), 'wins': len(wins), 'losses': len(losses),
        'win_rate_pct': round(100*len(wins)/len(trades), 2),
        'avg_win': round(sum(wins)/len(wins), 0) if wins else 0,
        'avg_loss': round(sum(losses)/len(losses), 0) if losses else 0,
        'profit_factor': round(sum(wins)/abs(sum(losses)), 2) if losses else 0,
        'net_pnl': round(total, 0),
        'gross_pnl': round(sum(t.gross_pnl for t in trades), 0),
        'costs': round(sum(t.gross_pnl - t.net_pnl for t in trades), 0),
        'days_traded': n, 'cagr_pct': round(cagr, 2), 'sharpe': round(sharpe, 2),
        'max_dd': round(mdd, 0), 'max_dd_pct': round(100*mdd/cap, 2),
        'calmar': round(cagr / (100*mdd/cap), 2) if mdd > 0 else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tf', required=True, choices=list(TF_CONFIG.keys()))
    args = ap.parse_args()

    tf = args.tf
    tf_cfg = TF_CONFIG[tf]
    out_dir = OUT_ROOT / f'tf_{tf}'
    out_dir.mkdir(parents=True, exist_ok=True)

    progress_path = out_dir / 'progress.txt'
    t_start = time.time()
    def heartbeat(msg: str):
        with progress_path.open('a') as f:
            f.write(f'{datetime.now().isoformat()}  {msg}\n')

    heartbeat(f'START tf={tf} universe={len(UNIVERSE)} variants={len(VARIANTS)} fast/slow={tf_cfg["fast"]}/{tf_cfg["slow"]}')

    all_trades = {v: [] for v in VARIANTS}
    daily_tot = {v: {} for v in VARIANTS}

    # Fast pair for fast_pair variant — tuned per TF
    fp_periods = {'5min': (9, 21), '10min': (5, 13), '15min': (4, 10), '30min': (3, 8)}[tf]

    for i, sym in enumerate(UNIVERSE, 1):
        t0 = time.time()
        df5 = load_5min(sym)
        if df5.empty:
            heartbeat(f'[{i}/{len(UNIVERSE)}] {sym} NO DATA')
            continue
        df = resample_bars(df5, tf_cfg['freq'])
        df = add_indicators(df, df5, tf_cfg['fast'], tf_cfg['slow'], fp_periods)

        stock_summary_parts = [f'[{i:2d}/{len(UNIVERSE)}] {sym:12s} bars={len(df):>5}']
        for vname, vcfg in VARIANTS.items():
            trades, daily = run_stock_variant(df, sym, vname, vcfg, tf_cfg)
            all_trades[vname].extend(trades)
            for d, p in daily.items(): daily_tot[vname][d] = daily_tot[vname].get(d, 0) + p
            net = sum(t.net_pnl for t in trades)
            stock_summary_parts.append(f'{vname[:9]:<9}={len(trades):>3}/Rs{net:>+7,.0f}')
        elapsed = time.time() - t0
        line = '  '.join(stock_summary_parts) + f'  ({elapsed:.1f}s)'
        print(line, flush=True)
        heartbeat(line)

        # Incremental CSV writes after each stock (crash safety)
        keys = ['trades','wins','losses','win_rate_pct','avg_win','avg_loss','profit_factor',
                'net_pnl','gross_pnl','costs','days_traded','cagr_pct','sharpe',
                'max_dd','max_dd_pct','calmar']
        with (out_dir / 'summary.csv').open('w', newline='') as f:
            w = csv.writer(f); w.writerow(['variant'] + keys)
            for vname in VARIANTS:
                m = compute_metrics(all_trades[vname], daily_tot[vname])
                w.writerow([vname] + [m.get(k, '') for k in keys])

    # Final trade log + daily
    with (out_dir / 'trades.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['variant','symbol','date','direction','entry_time','entry','stop','target',
                    'qty','atr_at_entry','exit_time','exit_price','exit_reason','gross_pnl','net_pnl'])
        for v, tr in all_trades.items():
            for t in tr:
                w.writerow([t.variant, t.symbol, t.date, t.direction, t.entry_time,
                            f'{t.entry:.2f}', f'{t.stop:.2f}', f'{t.target:.2f}', t.qty,
                            f'{t.atr_at_entry:.2f}',
                            t.exit_time, f'{t.exit_price:.2f}', t.exit_reason,
                            f'{t.gross_pnl:.2f}', f'{t.net_pnl:.2f}'])

    all_days = sorted({d for v in daily_tot.values() for d in v.keys()})
    with (out_dir / 'daily_pnl.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['date'] + list(VARIANTS.keys()))
        for d in all_days:
            w.writerow([d] + [f'{daily_tot[v].get(d, 0):.2f}' for v in VARIANTS])

    dur = time.time() - t_start
    heartbeat(f'DONE in {dur:.1f}s')
    print()
    print('=' * 105)
    print(f'TF={tf} EMA {tf_cfg["fast"]}/{tf_cfg["slow"]} — {len(UNIVERSE)} stocks, {START_DATE} to {END_DATE}')
    print('=' * 105)
    print(f'{"Variant":14s} {"Trades":>7} {"WR%":>6} {"PF":>6} {"Net P&L":>13} {"CAGR%":>7} {"Sharpe":>7} {"MaxDD":>10} {"Calmar":>7}')
    print('-' * 105)
    for vname in VARIANTS:
        m = compute_metrics(all_trades[vname], daily_tot[vname])
        print(f'{vname:14s} {m["trades"]:>7} {m["win_rate_pct"]:>6.1f} {m["profit_factor"]:>6.2f} '
              f'Rs {m["net_pnl"]:>+10,.0f} {m["cagr_pct"]:>+6.2f} {m["sharpe"]:>7.2f} '
              f'{m["max_dd"]:>+9,.0f} {m["calmar"]:>7.2f}')
    print(f'\nTotal runtime: {dur:.1f}s')


if __name__ == '__main__':
    sys.exit(main())
