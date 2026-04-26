"""3-bar reversal — bidirectional intraday system with MACD confirmation.

Pattern (LONG):
  Bar 1: strong GREEN (body >= 60% of range, close > open)
  Bar 2: strong RED   (body >= 60% of range, close < open)
  Bar 3: strong GREEN AND close > max(bar1.high, bar2.high)

Pattern (SHORT, mirrored):
  Bar 1: strong RED · Bar 2: strong GREEN · Bar 3: strong RED, close < min(bar1.low, bar2.low)

Phases:
  --phase 1   Signal sweep: 3 TFs × 2 MACD conditions = 6 cells
  --phase 2   Confluence filters on best Phase 1 variant
  --phase 3   SL/target tweaks on best Phase 2 variant

Args & state passed via written files between phases.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sqlite3
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime, time as dtime
from pathlib import Path

import numpy as np
import pandas as pd

logging.disable(logging.WARNING); warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / 'backtest_data' / 'market_data.db'
OUT = Path(__file__).resolve().parents[1] / 'results'
OUT.mkdir(exist_ok=True)

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

# Pattern thresholds
BODY_RATIO_THRESHOLD = 0.6     # body must be >= 60% of total range to qualify

# Trade rules
ENTRY_WINDOW_START = dtime(9, 45)
ENTRY_WINDOW_END   = dtime(14, 0)
EOD_EXIT           = dtime(15, 15)
TIME_STOP_BARS     = 24

# Sizing + costs
RISK_PER_TRADE = 2_500
MAX_NOTIONAL   = 300_000
COST_PCT       = 0.0015

# Indicator params
ATR_PERIOD = 14
RSI_PERIOD = 14
HTF_EMA    = 21
BB_PERIOD  = 20
BB_K       = 2
STOCH_K, STOCH_SMOOTH, STOCH_D = 14, 3, 3
MACD_FAST, MACD_SLOW, MACD_SIG = 12, 26, 9


@dataclass
class Trade:
    variant: str; symbol: str; date: str; direction: str
    fire_time: str; entry_time: str; entry: float; stop: float; target: float; qty: int
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


def resample(df5: pd.DataFrame, freq: str) -> pd.DataFrame:
    if freq == '5min':
        out = df5.copy()
    else:
        out = df5.resample(freq, label='right', closed='right').agg(
            {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
        ).dropna()
    out['day'] = out.index.date
    return out


def add_indicators(df: pd.DataFrame, df5: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df

    # MACD on the trade TF
    ema_f = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
    ema_s = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    df['macd_line']   = ema_f - ema_s
    df['macd_signal'] = df['macd_line'].ewm(span=MACD_SIG, adjust=False).mean()
    df['macd_hist']   = df['macd_line'] - df['macd_signal']
    df['macd_cross_up']   = (df['macd_line'].shift(1) <= df['macd_signal'].shift(1)) & (df['macd_line'] > df['macd_signal'])
    df['macd_cross_down'] = (df['macd_line'].shift(1) >= df['macd_signal'].shift(1)) & (df['macd_line'] < df['macd_signal'])

    # ATR
    pc = df['close'].shift(1)
    tr = pd.concat([df['high']-df['low'], (df['high']-pc).abs(), (df['low']-pc).abs()], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/ATR_PERIOD, adjust=False).mean()

    # RSI (Wilder)
    ch = df['close'].diff()
    g = ch.clip(lower=0); l = -ch.clip(upper=0)
    ag = g.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    al = l.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    df['rsi'] = 100 - 100 / (1 + rs)

    # BB
    df['bb_mid'] = df['close'].rolling(BB_PERIOD).mean()
    bb_std = df['close'].rolling(BB_PERIOD).std()
    df['bb_upper'] = df['bb_mid'] + BB_K * bb_std
    df['bb_lower'] = df['bb_mid'] - BB_K * bb_std

    # Stochastic
    hh = df['high'].rolling(STOCH_K).max()
    ll = df['low'].rolling(STOCH_K).min()
    k_raw = 100 * (df['close'] - ll) / (hh - ll).replace(0, np.nan)
    df['stoch_k'] = k_raw.rolling(STOCH_SMOOTH).mean()
    df['stoch_d'] = df['stoch_k'].rolling(STOCH_D).mean()
    df['stoch_cross_up']   = (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1)) & (df['stoch_k'] > df['stoch_d']) & (df['stoch_k'] < 30)
    df['stoch_cross_down'] = (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1)) & (df['stoch_k'] < df['stoch_d']) & (df['stoch_k'] > 70)

    # CPR — prev day pivot
    daily = df.groupby('day').agg(dh=('high', 'max'), dl=('low', 'min'), dc=('close', 'last'))
    daily['pivot'] = (daily['dh'] + daily['dl'] + daily['dc']) / 3.0
    df = df.join(daily['pivot'].shift(1).rename('prev_pivot'), on='day')

    # HTF — 60-min EMA(21) slope from 5-min resample
    sixty = df5[['open','high','low','close','volume']].resample('60min', label='right', closed='right').agg(
        {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}
    ).dropna()
    sixty['htf_ema'] = sixty['close'].ewm(span=HTF_EMA, adjust=False).mean()
    sixty['htf_slope'] = sixty['htf_ema'].diff()
    df['htf_slope'] = sixty['htf_slope'].reindex(df.index, method='ffill')

    # Body strength flags
    rng = (df['high'] - df['low']).replace(0, np.nan)
    body_abs = (df['close'] - df['open']).abs()
    df['body_ratio'] = body_abs / rng
    df['strong_bull'] = (df['close'] > df['open']) & (df['body_ratio'] >= BODY_RATIO_THRESHOLD)
    df['strong_bear'] = (df['close'] < df['open']) & (df['body_ratio'] >= BODY_RATIO_THRESHOLD)

    # 3-bar pattern detection (vectorized) — bar 3 = current bar
    # LONG: bar1 strong_bull, bar2 strong_bear, bar3 strong_bull AND close > max(bar1.high, bar2.high)
    sb = df['strong_bull']; sd = df['strong_bear']; close = df['close']; high = df['high']; low = df['low']
    df['pattern_long']  = sb.shift(2) & sd.shift(1) & sb & (close > pd.concat([high.shift(2), high.shift(1)], axis=1).max(axis=1))
    df['pattern_short'] = sd.shift(2) & sb.shift(1) & sd & (close < pd.concat([low.shift(2),  low.shift(1)], axis=1).min(axis=1))

    # Pre-compute reference levels for stops (bar1 low / bar1 high)
    df['bar1_low']  = low.shift(2)
    df['bar1_high'] = high.shift(2)
    df['bar2_low']  = low.shift(1)
    df['bar2_high'] = high.shift(1)

    return df


def macd_ok_long(row, kind: str) -> bool:
    if pd.isna(row.macd_line) or pd.isna(row.macd_signal): return False
    if kind == 'macd_cross':
        return bool(row.macd_cross_up)
    elif kind == 'macd_zero':
        return float(row.macd_line) > 0
    return False


def macd_ok_short(row, kind: str) -> bool:
    if pd.isna(row.macd_line) or pd.isna(row.macd_signal): return False
    if kind == 'macd_cross':
        return bool(row.macd_cross_down)
    elif kind == 'macd_zero':
        return float(row.macd_line) < 0
    return False


def filter_ok(row, direction: str, filt: str) -> bool:
    """Apply a confluence filter check at signal bar."""
    if filt == 'none': return True
    if filt == 'cpr':
        if pd.isna(row.prev_pivot): return False
        return (row.close > row.prev_pivot) if direction == 'LONG' else (row.close < row.prev_pivot)
    if filt == 'rsi50':
        if pd.isna(row.rsi): return False
        return (row.rsi > 50) if direction == 'LONG' else (row.rsi < 50)
    if filt == 'stoch':
        return bool(row.stoch_cross_up) if direction == 'LONG' else bool(row.stoch_cross_down)
    if filt == 'htf':
        if pd.isna(row.htf_slope): return False
        return (row.htf_slope > 0) if direction == 'LONG' else (row.htf_slope < 0)
    if filt == 'bb':
        if pd.isna(row.bb_mid): return False
        return (row.close > row.bb_mid) if direction == 'LONG' else (row.close < row.bb_mid)
    if filt == 'confluence':
        ok_cpr = (row.close > row.prev_pivot) if direction == 'LONG' else (row.close < row.prev_pivot)
        ok_rsi = (row.rsi > 50) if direction == 'LONG' else (row.rsi < 50)
        ok_htf = (row.htf_slope > 0) if direction == 'LONG' else (row.htf_slope < 0)
        if pd.isna(row.prev_pivot) or pd.isna(row.rsi) or pd.isna(row.htf_slope): return False
        return ok_cpr and ok_rsi and ok_htf
    return True


def run_variant(df: pd.DataFrame, symbol: str, variant: str, cfg: dict) -> tuple[list[Trade], dict]:
    """cfg keys: macd_kind, filter, sl_basis ('bar1'|'bar2'), r_mult, allow_long, allow_short."""
    if df.empty: return [], {}
    trades = []
    daily = {}

    for day, day_df in df.groupby('day'):
        active = None; bars_held = 0; traded_today = False; pending = None
        rows = list(day_df.itertuples())

        for idx, row in enumerate(rows):
            ts = row.Index; t = ts.time()

            # Fill pending entry at this bar's open
            if pending is not None and active is None and not traded_today:
                direction, stop, target = pending
                entry_px = row.open
                if pd.notna(entry_px) and entry_px > 0:
                    rps = abs(entry_px - stop)
                    if rps > 0:
                        # Recompute target relative to actual entry price
                        if direction == 'LONG':
                            target = entry_px + cfg['r_mult'] * rps
                        else:
                            target = entry_px - cfg['r_mult'] * rps
                        qty = int(RISK_PER_TRADE // rps)
                        if qty * entry_px > MAX_NOTIONAL: qty = int(MAX_NOTIONAL // entry_px)
                        if qty > 0:
                            active = Trade(variant, symbol, day.isoformat(), direction,
                                           rows[idx-1].Index.isoformat() if idx > 0 else ts.isoformat(),
                                           ts.isoformat(), entry_px, stop, target, qty)
                            bars_held = 0; traded_today = True
                pending = None

            # Exit logic
            if active is not None:
                bars_held += 1
                hit_stop = (active.direction == 'LONG'  and row.low  <= active.stop) \
                        or (active.direction == 'SHORT' and row.high >= active.stop)
                hit_tgt  = (active.direction == 'LONG'  and row.high >= active.target) \
                        or (active.direction == 'SHORT' and row.low  <= active.target)
                if hit_stop:
                    active.close(ts.isoformat(), active.stop, 'STOP')
                    trades.append(active); daily[day.isoformat()] = daily.get(day.isoformat(), 0) + active.net_pnl
                    active = None
                elif hit_tgt:
                    active.close(ts.isoformat(), active.target, 'TARGET')
                    trades.append(active); daily[day.isoformat()] = daily.get(day.isoformat(), 0) + active.net_pnl
                    active = None
                elif bars_held >= TIME_STOP_BARS:
                    active.close(ts.isoformat(), row.close, 'TIME_STOP')
                    trades.append(active); daily[day.isoformat()] = daily.get(day.isoformat(), 0) + active.net_pnl
                    active = None
                elif t >= EOD_EXIT:
                    active.close(ts.isoformat(), row.close, 'EOD')
                    trades.append(active); daily[day.isoformat()] = daily.get(day.isoformat(), 0) + active.net_pnl
                    active = None

            # Pattern detection on this bar (= bar 3 of pattern)
            if (active is None and pending is None and not traded_today
                    and ENTRY_WINDOW_START <= t <= ENTRY_WINDOW_END):
                # LONG
                if cfg['allow_long'] and bool(row.pattern_long) and macd_ok_long(row, cfg['macd_kind']):
                    if filter_ok(row, 'LONG', cfg['filter']):
                        if cfg['sl_basis'] == 'bar1':
                            stop = float(row.bar1_low) if pd.notna(row.bar1_low) else None
                        else:
                            stop = float(row.bar2_low) if pd.notna(row.bar2_low) else None
                        if stop is not None and stop < row.close:
                            pending = ('LONG', stop, None)
                # SHORT
                elif cfg['allow_short'] and bool(row.pattern_short) and macd_ok_short(row, cfg['macd_kind']):
                    if filter_ok(row, 'SHORT', cfg['filter']):
                        if cfg['sl_basis'] == 'bar1':
                            stop = float(row.bar1_high) if pd.notna(row.bar1_high) else None
                        else:
                            stop = float(row.bar2_high) if pd.notna(row.bar2_high) else None
                        if stop is not None and stop > row.close:
                            pending = ('SHORT', stop, None)

        if active is not None and rows:
            last = rows[-1]
            active.close(last.Index.isoformat(), last.close, 'EOD_FORCED')
            trades.append(active); daily[day.isoformat()] = daily.get(day.isoformat(), 0) + active.net_pnl

    return trades, daily


def compute_metrics(trades, daily):
    if not trades:
        return {k: 0 for k in ['trades','wins','losses','win_rate_pct','avg_win','avg_loss',
                               'profit_factor','net_pnl','sharpe','max_dd_pct','cagr_pct','calmar']}
    net = [t.net_pnl for t in trades]
    wins = [p for p in net if p > 0]; losses = [p for p in net if p < 0]
    sd = sorted(daily.keys()); series = [daily[d] for d in sd]
    r = peak = mdd = 0.0
    for p in series:
        r += p
        if r > peak: peak = r
        mdd = max(mdd, peak - r)
    n = len(series); sharpe = 0
    if n > 1:
        m = sum(series)/n; s = (sum((x-m)**2 for x in series)/(n-1))**0.5
        if s > 0: sharpe = (m/s) * (252**0.5)
    cap = 300_000
    yrs = n / 252 if n else 1; end = cap + sum(net)
    cagr = ((end/cap)**(1/yrs) - 1) * 100 if yrs > 0 and end > 0 else 0.0
    return {
        'trades': len(trades), 'wins': len(wins), 'losses': len(losses),
        'win_rate_pct': round(100*len(wins)/len(trades), 2),
        'avg_win': round(sum(wins)/len(wins), 0) if wins else 0,
        'avg_loss': round(sum(losses)/len(losses), 0) if losses else 0,
        'profit_factor': round(sum(wins)/abs(sum(losses)), 2) if losses else 0,
        'net_pnl': round(sum(net), 0),
        'sharpe': round(sharpe, 2),
        'max_dd_pct': round(100*mdd/cap, 2),
        'cagr_pct': round(cagr, 2),
        'calmar': round(cagr/(100*mdd/cap), 2) if mdd > 0 else 0,
    }


def heartbeat(msg: str):
    with (OUT / 'progress.txt').open('a') as f:
        f.write(f'{datetime.now().isoformat()}  {msg}\n')


def run_phase(variants: list[dict], phase_label: str):
    """variants: list of (name, cfg) dicts. Each cfg has tf, macd_kind, filter, sl_basis, r_mult, etc."""
    t_start = time.time()
    heartbeat(f'PHASE {phase_label} START — {len(variants)} variants × {len(UNIVERSE)} stocks')

    # Cache 5-min bars per stock
    bars_cache = {}
    for sym in UNIVERSE:
        df5 = load_5min(sym)
        bars_cache[sym] = df5

    # Group variants by TF for efficient resampling
    cells = {v['name']: {'cfg': v, 'trades': [], 'daily': {}} for v in variants}

    by_tf = {}
    for v in variants:
        by_tf.setdefault(v['tf'], []).append(v)

    for tf, tf_variants in by_tf.items():
        for sym in UNIVERSE:
            df5 = bars_cache[sym]
            if df5.empty: continue
            df_tf = resample(df5, tf)
            df_tf = add_indicators(df_tf, df5)

            for v in tf_variants:
                trades, daily = run_variant(df_tf, sym, v['name'], v)
                cells[v['name']]['trades'].extend(trades)
                for d, p in daily.items():
                    cells[v['name']]['daily'][d] = cells[v['name']]['daily'].get(d, 0) + p
            heartbeat(f'  tf={tf} sym={sym} done')

    # Output summary
    keys = ['trades','wins','losses','win_rate_pct','avg_win','avg_loss',
            'profit_factor','net_pnl','sharpe','max_dd_pct','cagr_pct','calmar']
    summary_path = OUT / f'phase{phase_label}_summary.csv'
    with summary_path.open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['variant'] + keys)
        for name, data in cells.items():
            m = compute_metrics(data['trades'], data['daily'])
            w.writerow([name] + [m.get(k, '') for k in keys])

    # Trade log
    with (OUT / f'phase{phase_label}_trades.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['variant','symbol','date','direction','entry_time','entry','stop','target',
                    'qty','exit_time','exit_price','exit_reason','gross_pnl','net_pnl'])
        for name, data in cells.items():
            for t in data['trades']:
                w.writerow([t.variant, t.symbol, t.date, t.direction,
                            t.entry_time, f'{t.entry:.2f}', f'{t.stop:.2f}', f'{t.target:.2f}',
                            t.qty, t.exit_time, f'{t.exit_price:.2f}', t.exit_reason,
                            f'{t.gross_pnl:.2f}', f'{t.net_pnl:.2f}'])

    # Print summary
    print()
    print('=' * 105)
    print(f'PHASE {phase_label} — {len(variants)} variants, {len(UNIVERSE)} stocks, {START_DATE} to {END_DATE}')
    print('=' * 105)
    print(f'{"Variant":>30} {"Trades":>7} {"WR%":>6} {"PF":>6} {"AvgWin":>8} {"AvgLoss":>9} {"Net P&L":>13} {"Sharpe":>7} {"MaxDD%":>7} {"Calmar":>7}')
    print('-' * 105)
    for name, data in cells.items():
        m = compute_metrics(data['trades'], data['daily'])
        print(f'{name:>30} {m["trades"]:>7} {m["win_rate_pct"]:>6.1f} {m["profit_factor"]:>6.2f} '
              f'Rs{m["avg_win"]:>+6,.0f} Rs{m["avg_loss"]:>+7,.0f} Rs{m["net_pnl"]:>+11,.0f} '
              f'{m["sharpe"]:>7.2f} {m["max_dd_pct"]:>7.2f} {m["calmar"]:>7.2f}')
    print(f'\nPhase runtime: {time.time()-t_start:.1f}s')
    heartbeat(f'PHASE {phase_label} DONE in {time.time()-t_start:.1f}s')


def phase1_variants():
    """3 TFs × 2 MACD = 6 cells. SL=bar1, target=2R, no extra filter."""
    variants = []
    for tf in ['5min', '10min', '15min']:
        for macd in ['macd_cross', 'macd_zero']:
            variants.append({
                'name': f'{tf}_{macd}',
                'tf': tf, 'macd_kind': macd, 'filter': 'none',
                'sl_basis': 'bar1', 'r_mult': 2.0,
                'allow_long': True, 'allow_short': True,
            })
    return variants


def phase2_variants(best_tf: str, best_macd: str):
    """Best from Phase 1 + each confluence filter."""
    variants = []
    for filt in ['none', 'cpr', 'rsi50', 'stoch', 'htf', 'bb', 'confluence']:
        variants.append({
            'name': f'{best_tf}_{best_macd}_{filt}',
            'tf': best_tf, 'macd_kind': best_macd, 'filter': filt,
            'sl_basis': 'bar1', 'r_mult': 2.0,
            'allow_long': True, 'allow_short': True,
        })
    return variants


def phase3_variants(best_tf: str, best_macd: str, best_filter: str):
    """SL/target tweaks on best Phase 2."""
    variants = [
        {'name': f'{best_tf}_{best_macd}_{best_filter}_bar1_2R', 'tf': best_tf, 'macd_kind': best_macd,
         'filter': best_filter, 'sl_basis': 'bar1', 'r_mult': 2.0, 'allow_long': True, 'allow_short': True},
        {'name': f'{best_tf}_{best_macd}_{best_filter}_bar2_2R', 'tf': best_tf, 'macd_kind': best_macd,
         'filter': best_filter, 'sl_basis': 'bar2', 'r_mult': 2.0, 'allow_long': True, 'allow_short': True},
        {'name': f'{best_tf}_{best_macd}_{best_filter}_bar1_15R', 'tf': best_tf, 'macd_kind': best_macd,
         'filter': best_filter, 'sl_basis': 'bar1', 'r_mult': 1.5, 'allow_long': True, 'allow_short': True},
        {'name': f'{best_tf}_{best_macd}_{best_filter}_bar1_3R', 'tf': best_tf, 'macd_kind': best_macd,
         'filter': best_filter, 'sl_basis': 'bar1', 'r_mult': 3.0, 'allow_long': True, 'allow_short': True},
    ]
    return variants


def find_best(phase_label: str, sort_key: str = 'sharpe'):
    p = OUT / f'phase{phase_label}_summary.csv'
    if not p.exists(): return None
    rows = list(csv.DictReader(p.open()))
    if not rows: return None
    rows = [r for r in rows if int(r['trades']) >= 50]   # need min sample
    if not rows: return None
    rows.sort(key=lambda r: -float(r[sort_key]))
    return rows[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', type=int, choices=[1, 2, 3], required=True)
    ap.add_argument('--best-tf')
    ap.add_argument('--best-macd')
    ap.add_argument('--best-filter')
    args = ap.parse_args()

    if args.phase == 1:
        run_phase(phase1_variants(), '1')
        # Save best to disk for next phase
        best = find_best('1')
        if best:
            with (OUT / 'best_phase1.json').open('w') as f:
                json.dump({'variant': best['variant'], 'sharpe': best['sharpe'], 'pf': best['profit_factor']}, f)

    elif args.phase == 2:
        if not args.best_tf or not args.best_macd:
            # Auto-load from phase1 best
            try:
                d = json.load((OUT / 'best_phase1.json').open())
                parts = d['variant'].split('_')  # e.g., '5min_macd_zero' or '15min_macd_cross'
                args.best_tf = parts[0]
                args.best_macd = '_'.join(parts[1:])  # macd_zero / macd_cross
            except Exception as e:
                print(f'Need --best-tf and --best-macd, or run phase 1 first. ({e})')
                return 1
        run_phase(phase2_variants(args.best_tf, args.best_macd), '2')
        best = find_best('2')
        if best:
            with (OUT / 'best_phase2.json').open('w') as f:
                json.dump({'variant': best['variant'], 'sharpe': best['sharpe'], 'pf': best['profit_factor']}, f)

    elif args.phase == 3:
        if not all([args.best_tf, args.best_macd, args.best_filter]):
            try:
                d = json.load((OUT / 'best_phase2.json').open())
                parts = d['variant'].split('_')
                # variant: '5min_macd_zero_cpr' → tf=5min, macd=macd_zero, filter=cpr
                args.best_tf = parts[0]
                args.best_filter = parts[-1]
                args.best_macd = '_'.join(parts[1:-1])
            except Exception as e:
                print(f'Need --best-tf, --best-macd, --best-filter (or run phase 2 first). ({e})')
                return 1
        run_phase(phase3_variants(args.best_tf, args.best_macd, args.best_filter), '3')


if __name__ == '__main__':
    sys.exit(main())
