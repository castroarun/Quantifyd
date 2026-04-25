"""Bollinger Squeeze CONTRA — failed-breakout / bull-trap fade.

Original (research/16 v1) had 33% WR going WITH the breakout direction.
The signal IS information — just inverted. This v2 takes the OPPOSITE side:
  - squeeze fires UP (close > BB upper) -> SHORT next bar (fade trap)
  - squeeze fires DOWN (close < BB lower) -> LONG next bar (fade trap)

If v1's 33% WR was clean, v2 should be ~50-67% WR (depending on time-stop /
EOD share of v1 exits). At 1:2 R:R that would flip a -0.01R/trade EV into
+0.5-1.0R/trade EV. Classic "bull/bear trap" pattern.

Tests both directions side-by-side: long_* are the v1 spec, contra_* flip.
"""
from __future__ import annotations

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
OUT = Path(__file__).resolve().parents[1] / 'results'
OUT.mkdir(exist_ok=True)

# 15 non-ORB liquid F&O — same set used in research 14 + 15 for cross-compare.
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

# Indicator periods (5-min bars)
BB_PERIOD     = 20
BB_K          = 2.0
KC_PERIOD     = 20
KC_K          = 1.5
ATR_PERIOD    = 14
RSI_PERIOD    = 14
HTF_EMA       = 21
VOL_LOOKBACK  = 20
MIN_SQZ_BARS  = 6      # require compression for >=30min before fire

# Trade rules
ENTRY_WINDOW_START = dtime(9, 45)
ENTRY_WINDOW_END   = dtime(14, 0)
EOD_EXIT           = dtime(15, 15)
TIME_STOP_BARS     = 18
STOP_ATR_MULT      = 1.0
R_MULTIPLE         = 2.0   # Squeeze breakouts target larger moves than MA cross

RISK_PER_TRADE     = 2_500
MAX_NOTIONAL       = 300_000
COST_PCT           = 0.0015

# Filter variants — long_* = v1 spec, contra_* = inverted direction
VARIANTS = {
    'long_baseline':   {'contra': False, 'rsi50': False, 'vwap': False, 'volspike': False, 'htf': False},
    'long_volspike':   {'contra': False, 'rsi50': False, 'vwap': False, 'volspike': True,  'htf': False},
    'long_htf':        {'contra': False, 'rsi50': False, 'vwap': False, 'volspike': False, 'htf': True},
    'long_confluence': {'contra': False, 'rsi50': False, 'vwap': True,  'volspike': True,  'htf': True},

    'contra_baseline':   {'contra': True, 'rsi50': False, 'vwap': False, 'volspike': False, 'htf': False},
    'contra_volspike':   {'contra': True, 'rsi50': False, 'vwap': False, 'volspike': True,  'htf': False},
    'contra_htf':        {'contra': True, 'rsi50': False, 'vwap': False, 'volspike': False, 'htf': True},
    'contra_confluence': {'contra': True, 'rsi50': False, 'vwap': True,  'volspike': True,  'htf': True},
}


@dataclass
class Trade:
    variant: str; symbol: str; date: str; direction: str
    entry_time: str; entry: float; stop: float; target: float; qty: int
    atr_at_entry: float; sqz_bars: int
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
    df['day'] = df.index.date
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df

    # Bollinger Bands (SMA-based)
    df['bb_mid'] = df['close'].rolling(BB_PERIOD).mean()
    bb_std = df['close'].rolling(BB_PERIOD).std()
    df['bb_upper'] = df['bb_mid'] + BB_K * bb_std
    df['bb_lower'] = df['bb_mid'] - BB_K * bb_std

    # Keltner Channels (EMA-based + ATR)
    df['kc_mid']  = df['close'].ewm(span=KC_PERIOD, adjust=False).mean()
    pc = df['close'].shift(1)
    tr = pd.concat([df['high'] - df['low'],
                    (df['high'] - pc).abs(),
                    (df['low']  - pc).abs()], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1 / ATR_PERIOD, adjust=False).mean()
    atr_kc = tr.rolling(KC_PERIOD).mean()
    df['kc_upper'] = df['kc_mid'] + KC_K * atr_kc
    df['kc_lower'] = df['kc_mid'] - KC_K * atr_kc

    # Squeeze flag — BB inside KC
    df['in_squeeze'] = (df['bb_upper'] < df['kc_upper']) & (df['bb_lower'] > df['kc_lower'])

    # Count consecutive squeeze bars (resets when not in squeeze)
    sq = df['in_squeeze'].astype(int)
    df['sqz_bars'] = sq * (sq.groupby((sq != sq.shift()).cumsum()).cumcount() + 1)

    # Squeeze fire = was in squeeze previous bar, not in squeeze this bar,
    # AND prior squeeze duration met the minimum
    df['fire'] = (~df['in_squeeze']) & (df['in_squeeze'].shift(1).fillna(False)) \
                 & (df['sqz_bars'].shift(1).fillna(0) >= MIN_SQZ_BARS)

    # Direction of breakout = position relative to BB
    df['break_up']   = df['fire'] & (df['close'] > df['bb_upper'])
    df['break_down'] = df['fire'] & (df['close'] < df['bb_lower'])

    # RSI
    ch = df['close'].diff()
    g = ch.clip(lower=0); l = -ch.clip(upper=0)
    ag = g.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    al = l.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    df['rsi'] = 100 - 100 / (1 + rs)

    # Intraday VWAP (resets daily)
    typ = (df['high'] + df['low'] + df['close']) / 3.0
    df['cum_pv'] = (typ * df['volume']).groupby(df['day']).cumsum()
    df['cum_v']  = df['volume'].groupby(df['day']).cumsum()
    df['vwap']   = df['cum_pv'] / df['cum_v']

    # Volume spike — current vs 20-bar prior avg
    df['vol_avg'] = df['volume'].rolling(VOL_LOOKBACK).mean().shift(1)
    df['vol_spike'] = df['volume'] / df['vol_avg']

    # HTF — 60-min EMA21 slope (resampled)
    sixty = df[['open','high','low','close','volume']].resample('60min', label='right', closed='right').agg(
        {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}
    ).dropna()
    sixty['htf_ema'] = sixty['close'].ewm(span=HTF_EMA, adjust=False).mean()
    sixty['htf_slope'] = sixty['htf_ema'].diff()
    df['htf_slope'] = sixty['htf_slope'].reindex(df.index, method='ffill')

    return df


def signal_ok(row, direction: str, cfg: dict) -> bool:
    if cfg['rsi50']:
        if pd.isna(row.rsi): return False
        if direction == 'LONG' and not (row.rsi > 50): return False
        if direction == 'SHORT' and not (row.rsi < 50): return False
    if cfg['vwap']:
        if pd.isna(row.vwap): return False
        if direction == 'LONG' and not (row.close > row.vwap): return False
        if direction == 'SHORT' and not (row.close < row.vwap): return False
    if cfg['volspike']:
        if pd.isna(row.vol_spike): return False
        if not (row.vol_spike >= 1.5): return False
    if cfg['htf']:
        if pd.isna(row.htf_slope): return False
        if direction == 'LONG' and not (row.htf_slope > 0): return False
        if direction == 'SHORT' and not (row.htf_slope < 0): return False
    return True


def run_stock_variant(df: pd.DataFrame, symbol: str, variant: str, cfg: dict):
    if df.empty: return [], {}
    trades = []
    daily_pnl = {}

    for day, day_df in df.groupby('day'):
        active = None; bars_held = 0; traded_today = False; pending = None
        day_iter = list(day_df.itertuples())

        for row in day_iter:
            ts = row.Index; t = ts.time()

            # Fill pending entry at this bar's open
            if pending is not None and active is None and not traded_today:
                direction, atr_e, sqz = pending
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
                                           entry_px, stop, target, qty, atr_e, sqz)
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
                    trades.append(active); daily_pnl[day.isoformat()] = daily_pnl.get(day.isoformat(), 0) + active.net_pnl
                    active = None
                elif hit_tgt:
                    active.close(ts.isoformat(), active.target, 'TARGET')
                    trades.append(active); daily_pnl[day.isoformat()] = daily_pnl.get(day.isoformat(), 0) + active.net_pnl
                    active = None
                elif bars_held >= TIME_STOP_BARS:
                    active.close(ts.isoformat(), row.close, 'TIME_STOP')
                    trades.append(active); daily_pnl[day.isoformat()] = daily_pnl.get(day.isoformat(), 0) + active.net_pnl
                    active = None
                elif t >= EOD_EXIT:
                    active.close(ts.isoformat(), row.close, 'EOD')
                    trades.append(active); daily_pnl[day.isoformat()] = daily_pnl.get(day.isoformat(), 0) + active.net_pnl
                    active = None

            # Entry signal (squeeze fire on this bar) — contra flips direction
            if (active is None and pending is None and not traded_today
                    and ENTRY_WINDOW_START <= t <= ENTRY_WINDOW_END
                    and pd.notna(row.atr) and row.atr > 0):
                direction = None
                if bool(row.break_up):
                    candidate = 'SHORT' if cfg.get('contra') else 'LONG'
                    if signal_ok(row, candidate, cfg):
                        direction = candidate
                elif bool(row.break_down):
                    candidate = 'LONG' if cfg.get('contra') else 'SHORT'
                    if signal_ok(row, candidate, cfg):
                        direction = candidate
                if direction:
                    sqz_count = int(row.sqz_bars) if pd.notna(row.sqz_bars) else 0
                    pending = (direction, row.atr, sqz_count)

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
    progress_path = OUT / 'progress_contra.txt'
    t_start = time.time()
    def heartbeat(msg: str):
        with progress_path.open('a') as f:
            f.write(f'{datetime.now().isoformat()}  {msg}\n')

    heartbeat(f'START universe={len(UNIVERSE)} variants={len(VARIANTS)} BB({BB_PERIOD},{BB_K}) KC({KC_PERIOD},{KC_K}) min_sqz={MIN_SQZ_BARS}')

    all_trades = {v: [] for v in VARIANTS}
    daily_tot = {v: {} for v in VARIANTS}

    for i, sym in enumerate(UNIVERSE, 1):
        t0 = time.time()
        df = load_5min(sym)
        if df.empty:
            heartbeat(f'[{i}/{len(UNIVERSE)}] {sym} NO DATA')
            continue
        df = add_indicators(df)
        n_squeezes = int(df['fire'].sum())

        parts = [f'[{i:2d}/{len(UNIVERSE)}] {sym:12s} bars={len(df):>5} squeezes={n_squeezes:>3}']
        for vname, vcfg in VARIANTS.items():
            trades, daily = run_stock_variant(df, sym, vname, vcfg)
            all_trades[vname].extend(trades)
            for d, p in daily.items(): daily_tot[vname][d] = daily_tot[vname].get(d, 0) + p
            net = sum(t.net_pnl for t in trades)
            parts.append(f'{vname[:9]:<9}={len(trades):>3}/Rs{net:>+7,.0f}')
        elapsed = time.time() - t0
        line = '  '.join(parts) + f'  ({elapsed:.1f}s)'
        print(line, flush=True)
        heartbeat(line)

        # Incremental write
        keys = ['trades','wins','losses','win_rate_pct','avg_win','avg_loss','profit_factor',
                'net_pnl','gross_pnl','costs','days_traded','cagr_pct','sharpe',
                'max_dd','max_dd_pct','calmar']
        with (OUT / 'summary_contra.csv').open('w', newline='') as f:
            w = csv.writer(f); w.writerow(['variant'] + keys)
            for vname in VARIANTS:
                m = compute_metrics(all_trades[vname], daily_tot[vname])
                w.writerow([vname] + [m.get(k, '') for k in keys])

    # Final dumps
    with (OUT / 'trades_contra.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['variant','symbol','date','direction','entry_time','entry','stop','target',
                    'qty','atr_at_entry','sqz_bars','exit_time','exit_price','exit_reason','gross_pnl','net_pnl'])
        for v, tr in all_trades.items():
            for t in tr:
                w.writerow([t.variant, t.symbol, t.date, t.direction, t.entry_time,
                            f'{t.entry:.2f}', f'{t.stop:.2f}', f'{t.target:.2f}', t.qty,
                            f'{t.atr_at_entry:.2f}', t.sqz_bars,
                            t.exit_time, f'{t.exit_price:.2f}', t.exit_reason,
                            f'{t.gross_pnl:.2f}', f'{t.net_pnl:.2f}'])

    all_days = sorted({d for v in daily_tot.values() for d in v.keys()})
    with (OUT / 'daily_pnl_contra.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['date'] + list(VARIANTS.keys()))
        for d in all_days:
            w.writerow([d] + [f'{daily_tot[v].get(d, 0):.2f}' for v in VARIANTS])

    dur = time.time() - t_start
    heartbeat(f'DONE in {dur:.1f}s')
    print()
    print('=' * 105)
    print(f'BB SQUEEZE BREAKOUT — {len(UNIVERSE)} stocks, 5-min, {START_DATE} to {END_DATE}, BB({BB_PERIOD},{BB_K}) inside KC({KC_PERIOD},{KC_K}) for >={MIN_SQZ_BARS} bars')
    print('=' * 105)
    print(f'{"Variant":12s} {"Trades":>7} {"WR%":>6} {"PF":>6} {"Net P&L":>13} {"CAGR%":>7} {"Sharpe":>7} {"MaxDD":>10} {"Calmar":>7}')
    print('-' * 105)
    for vname in VARIANTS:
        m = compute_metrics(all_trades[vname], daily_tot[vname])
        print(f'{vname:12s} {m["trades"]:>7} {m["win_rate_pct"]:>6.1f} {m["profit_factor"]:>6.2f} '
              f'Rs {m["net_pnl"]:>+10,.0f} {m["cagr_pct"]:>+6.2f} {m["sharpe"]:>7.2f} '
              f'{m["max_dd"]:>+9,.0f} {m["calmar"]:>7.2f}')
    print(f'\nRuntime: {dur:.1f}s')


if __name__ == '__main__':
    sys.exit(main())
