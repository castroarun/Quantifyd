"""Direction-prediction analysis on BB squeeze fires.

Goal: identify which features at fire-time predict whether price extends
≥2xATR UP first vs DOWN first (within next 18 bars).

For each fire across 15 stocks + NIFTY:
  - Compute fire-time features (vwap pos, RSI, HTF slope, vol spike,
    bar body, time-of-day, day-of-week, days-to-monthly-expiry, gap, etc.)
  - Compute outcome: which direction reaches 2xATR first (UP / DOWN / NEITHER)

Univariate analysis: bucket fires by each feature, report % UP / % DOWN /
% NEITHER per bucket. Identify features with strong directional skew.

Output: per-feature tables + overall feature ranking by predictive power.
"""
from __future__ import annotations

import logging
import sqlite3
import sys
import time
import warnings
from datetime import datetime, time as dtime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logging.disable(logging.WARNING); warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / 'backtest_data' / 'market_data.db'
OUT = Path(__file__).resolve().parents[1] / 'results' / 'direction_analysis'
OUT.mkdir(parents=True, exist_ok=True)

UNIVERSE = ['NIFTY50',
            'INDUSINDBK','PNB','BANKBARODA','WIPRO','HCLTECH','MARUTI','HEROMOTOCO',
            'EICHERMOT','TITAN','ASIANPAINT','JSWSTEEL','HINDALCO','JINDALSTEL','LT','ADANIPORTS']

START_DATE = '2024-03-18'
END_DATE   = '2026-03-25'

BB_PERIOD, BB_K       = 20, 2.0
KC_PERIOD, KC_K       = 20, 1.5
ATR_PERIOD            = 14
RSI_PERIOD            = 14
MIN_SQZ_BARS          = 6
HTF_EMA               = 21      # 60-min
VOL_LOOKBACK          = 20
LOOKAHEAD_BARS        = 18
TARGET_ATR_MULT       = 2.0


def last_thursday(year, month):
    if month == 12: first_next = datetime(year+1, 1, 1)
    else: first_next = datetime(year, month+1, 1)
    last = first_next - timedelta(days=1)
    while last.weekday() != 3: last -= timedelta(days=1)
    return last.date()


def days_to_monthly_expiry(d):
    exp = last_thursday(d.year, d.month)
    if d > exp:
        if d.month == 12: exp = last_thursday(d.year+1, 1)
        else:             exp = last_thursday(d.year, d.month+1)
    return (exp - d).days


def load_5min(symbol):
    conn = sqlite3.connect(DB)
    df = pd.read_sql_query(
        """SELECT date, open, high, low, close, volume FROM market_data_unified
            WHERE symbol=? AND timeframe='5minute' AND date>=? AND date<=?
         ORDER BY date""",
        conn, params=(symbol, START_DATE, END_DATE + ' 23:59:59'),
    )
    conn.close()
    if df.empty: return df
    df['date'] = pd.to_datetime(df['date']); df.set_index('date', inplace=True)
    df['day'] = df.index.date
    return df


def add_features(df):
    if df.empty: return df

    df['bb_mid']   = df['close'].rolling(BB_PERIOD).mean()
    bb_std         = df['close'].rolling(BB_PERIOD).std()
    df['bb_upper'] = df['bb_mid'] + BB_K * bb_std
    df['bb_lower'] = df['bb_mid'] - BB_K * bb_std

    pc = df['close'].shift(1)
    tr = pd.concat([df['high']-df['low'], (df['high']-pc).abs(), (df['low']-pc).abs()], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/ATR_PERIOD, adjust=False).mean()
    df['atr_pct'] = df['atr'] / df['close'] * 100

    df['kc_mid']   = df['close'].ewm(span=KC_PERIOD, adjust=False).mean()
    atr_kc         = tr.rolling(KC_PERIOD).mean()
    df['kc_upper'] = df['kc_mid'] + KC_K * atr_kc
    df['kc_lower'] = df['kc_mid'] - KC_K * atr_kc
    df['in_squeeze'] = (df['bb_upper'] < df['kc_upper']) & (df['bb_lower'] > df['kc_lower'])
    sq = df['in_squeeze'].astype(int)
    df['sqz_bars'] = sq * (sq.groupby((sq != sq.shift()).cumsum()).cumcount() + 1)
    df['fire'] = (~df['in_squeeze']) & (df['in_squeeze'].shift(1).fillna(False)) \
                 & (df['sqz_bars'].shift(1).fillna(0) >= MIN_SQZ_BARS)
    df['break_up']   = df['fire'] & (df['close'] > df['bb_upper'])
    df['break_down'] = df['fire'] & (df['close'] < df['bb_lower'])

    # RSI
    ch = df['close'].diff()
    g = ch.clip(lower=0); l = -ch.clip(upper=0)
    ag = g.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    al = l.ewm(alpha=1/RSI_PERIOD, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    df['rsi'] = 100 - 100 / (1 + rs)

    # Intraday VWAP
    typ = (df['high'] + df['low'] + df['close']) / 3.0
    cum_pv = (typ * df['volume']).groupby(df['day']).cumsum()
    cum_v  = df['volume'].groupby(df['day']).cumsum()
    df['vwap'] = cum_pv / cum_v
    df['vwap_dist_atr'] = (df['close'] - df['vwap']) / df['atr']

    # 60-min EMA21 slope
    sixty = df[['open','high','low','close','volume']].resample('60min', label='right', closed='right').agg(
        {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}
    ).dropna()
    sixty['htf_ema'] = sixty['close'].ewm(span=HTF_EMA, adjust=False).mean()
    sixty['htf_slope'] = sixty['htf_ema'].diff()
    df['htf_slope'] = sixty['htf_slope'].reindex(df.index, method='ffill')

    # Volume spike
    df['vol_avg'] = df['volume'].rolling(VOL_LOOKBACK).mean().shift(1)
    df['vol_spike'] = df['volume'] / df['vol_avg']

    # Bar body / closing position in range
    rng = (df['high'] - df['low']).replace(0, np.nan)
    df['body_ratio']     = (df['close'] - df['open']) / rng       # -1 (full bear) to +1 (full bull)
    df['close_in_range'] = (df['close'] - df['low']) / rng        # 0 (closed at low) to 1 (at high)

    # Day's open vs prev day's pivot (CPR) — bullish / bearish bias
    daily = df.groupby('day').agg(dh=('high','max'), dl=('low','min'),
                                  dc=('close','last'), do=('open','first'))
    daily['pivot'] = (daily['dh'] + daily['dl'] + daily['dc']) / 3.0
    daily['gap_pct'] = ((daily['do'] - daily['dc'].shift(1)) / daily['dc'].shift(1)) * 100
    daily_prev_pivot = daily['pivot'].shift(1)
    daily['cpr_bias'] = np.where(daily['do'] > daily_prev_pivot, 1, -1)
    df = df.join(daily[['cpr_bias','gap_pct']], on='day')

    # Time of day bucket
    def tod_bucket(t):
        if t < dtime(11, 0):  return 'morning'
        if t < dtime(13, 0):  return 'midday'
        return 'afternoon'
    df['tod'] = pd.Series(df.index.time, index=df.index).apply(tod_bucket)

    # Day of week
    df['dow'] = df.index.dayofweek

    return df


def extract_fire_features_and_outcomes(df, symbol):
    """Per fire row: fire-time features + outcome label."""
    in_window = (df.index.time >= dtime(9, 45)) & (df.index.time <= dtime(14, 0))
    fire_mask = df['fire'].values & in_window
    fire_pos = np.where(fire_mask)[0]

    arr_close = df['close'].values
    arr_high  = df['high'].values
    arr_low   = df['low'].values
    arr_atr   = df['atr'].values

    rows = []
    for i in fire_pos:
        if i + LOOKAHEAD_BARS >= len(df): continue
        atr = arr_atr[i]
        if not (atr > 0): continue
        entry = arr_close[i]
        # Forward bidirectional excursion bar-by-bar
        bars_to_up = None; bars_to_down = None
        for j in range(1, LOOKAHEAD_BARS + 1):
            up_d = (arr_high[i+j] - entry) / atr
            dn_d = (entry - arr_low[i+j]) / atr
            if up_d >= TARGET_ATR_MULT and bars_to_up is None:
                bars_to_up = j
            if dn_d >= TARGET_ATR_MULT and bars_to_down is None:
                bars_to_down = j
            if bars_to_up is not None and bars_to_down is not None: break

        if bars_to_up is None and bars_to_down is None:
            outcome = 'NEITHER'
        elif bars_to_up is None:
            outcome = 'DOWN'
        elif bars_to_down is None:
            outcome = 'UP'
        elif bars_to_up < bars_to_down:
            outcome = 'UP'
        elif bars_to_down < bars_to_up:
            outcome = 'DOWN'
        else:
            outcome = 'TIE'

        ts = df.index[i]
        rows.append({
            'symbol': symbol,
            'ts': ts,
            'date': ts.date(),
            'dow': int(ts.dayofweek),
            'tod': df['tod'].iloc[i],
            'days_to_expiry': days_to_monthly_expiry(ts.date()),

            'break_up':   bool(df['break_up'].iloc[i]),
            'break_down': bool(df['break_down'].iloc[i]),

            'rsi':            float(df['rsi'].iloc[i]) if pd.notna(df['rsi'].iloc[i]) else np.nan,
            'vwap_dist_atr':  float(df['vwap_dist_atr'].iloc[i]) if pd.notna(df['vwap_dist_atr'].iloc[i]) else np.nan,
            'htf_slope':      float(df['htf_slope'].iloc[i]) if pd.notna(df['htf_slope'].iloc[i]) else np.nan,
            'vol_spike':      float(df['vol_spike'].iloc[i]) if pd.notna(df['vol_spike'].iloc[i]) else np.nan,
            'body_ratio':     float(df['body_ratio'].iloc[i]) if pd.notna(df['body_ratio'].iloc[i]) else np.nan,
            'close_in_range': float(df['close_in_range'].iloc[i]) if pd.notna(df['close_in_range'].iloc[i]) else np.nan,
            'cpr_bias':       int(df['cpr_bias'].iloc[i]) if pd.notna(df['cpr_bias'].iloc[i]) else 0,
            'gap_pct':        float(df['gap_pct'].iloc[i]) if pd.notna(df['gap_pct'].iloc[i]) else np.nan,
            'atr_pct':        float(df['atr_pct'].iloc[i]) if pd.notna(df['atr_pct'].iloc[i]) else np.nan,
            'sqz_duration':   int(df['sqz_bars'].shift(1).iloc[i]) if pd.notna(df['sqz_bars'].shift(1).iloc[i]) else 0,

            'outcome': outcome,
        })
    return rows


def bucket_analysis(df, feature, buckets):
    """Bucket fires by feature value, show outcome distribution per bucket."""
    out = []
    for label, condition in buckets:
        sub = df[condition]
        n = len(sub)
        if n == 0:
            out.append((label, 0, 0, 0, 0, 0, 0))
            continue
        n_up   = (sub['outcome']=='UP').sum()
        n_down = (sub['outcome']=='DOWN').sum()
        n_either = n_up + n_down
        if n_either == 0:
            up_rate = 0; down_rate = 0; up_dom = 0
        else:
            up_rate = 100 * n_up / n_either
            down_rate = 100 * n_down / n_either
            # "Up dominance" — how lopsided
            up_dom = abs(up_rate - 50)
        n_neither = (sub['outcome']=='NEITHER').sum()
        out.append((label, n, n_up, n_down, n_neither, round(up_rate, 1), round(down_rate, 1)))
    return out


def main():
    t_start = time.time()
    all_rows = []
    for i, sym in enumerate(UNIVERSE, 1):
        t0 = time.time()
        df = load_5min(sym)
        if df.empty: continue
        df = add_features(df)
        rows = extract_fire_features_and_outcomes(df, sym)
        all_rows.extend(rows)
        print(f'[{i:2d}/{len(UNIVERSE)}] {sym:12s} fires={len(rows):>4} ({time.time()-t0:.1f}s)', flush=True)

    full = pd.DataFrame(all_rows)
    full.to_csv(OUT / 'fires_features.csv', index=False)
    n_total = len(full)
    n_either = ((full['outcome']=='UP') | (full['outcome']=='DOWN')).sum()
    n_up = (full['outcome']=='UP').sum()
    n_down = (full['outcome']=='DOWN').sum()
    n_neither = (full['outcome']=='NEITHER').sum()
    base_up_rate = 100 * n_up / n_either if n_either else 0

    print()
    print('=' * 95)
    print(f'BASELINE — {n_total} fires across {len(UNIVERSE)} symbols')
    print('=' * 95)
    print(f'  UP first:      {n_up}  ({100*n_up/n_total:.1f}% of all,  {base_up_rate:.1f}% of UP+DOWN)')
    print(f'  DOWN first:    {n_down}  ({100*n_down/n_total:.1f}% of all)')
    print(f'  NEITHER:       {n_neither}  ({100*n_neither/n_total:.1f}% of all)')
    print()
    print(f'>>> Baseline UP-rate (when extension happens): {base_up_rate:.1f}% — ANY feature lifting this materially')
    print(f'    above ~55% on a meaningful subset is a useful directional filter.')

    def print_table(name, rows):
        print()
        print('-' * 95)
        print(f'BY {name}')
        print(f'{"Bucket":>22} {"N":>6} {"UP":>5} {"DOWN":>6} {"NEITHER":>9} {"UP%":>7} {"DOWN%":>7}')
        print('-' * 95)
        for r in rows:
            label, n, u, d, ne, up_pct, dn_pct = r
            print(f'{str(label):>22} {n:>6} {u:>5} {d:>6} {ne:>9} {up_pct:>7.1f} {dn_pct:>7.1f}')

    # ---- BREAK direction (the original BB upper/lower side) ----
    rows_bd = bucket_analysis(full, 'break', [
        ('break_up=True',   full['break_up']==True),
        ('break_down=True', full['break_down']==True),
    ])
    print_table('break direction (the original BB upper/lower side)', rows_bd)

    # ---- RSI buckets ----
    rows = bucket_analysis(full, 'rsi', [
        ('rsi < 30',       full['rsi']<30),
        ('30 <= rsi < 45', (full['rsi']>=30) & (full['rsi']<45)),
        ('45 <= rsi < 55', (full['rsi']>=45) & (full['rsi']<55)),
        ('55 <= rsi < 70', (full['rsi']>=55) & (full['rsi']<70)),
        ('rsi >= 70',      full['rsi']>=70),
    ])
    print_table('RSI(14)', rows)

    # ---- VWAP distance ----
    rows = bucket_analysis(full, 'vwap_dist', [
        ('< -1 ATR (deep below)',     full['vwap_dist_atr']<-1.0),
        ('-1 to -0.3 ATR (below)',    (full['vwap_dist_atr']>=-1.0) & (full['vwap_dist_atr']<-0.3)),
        ('-0.3 to +0.3 ATR (at)',     (full['vwap_dist_atr']>=-0.3) & (full['vwap_dist_atr']<=0.3)),
        ('+0.3 to +1 ATR (above)',    (full['vwap_dist_atr']>0.3) & (full['vwap_dist_atr']<=1.0)),
        ('> +1 ATR (deep above)',     full['vwap_dist_atr']>1.0),
    ])
    print_table('VWAP distance (close vs intraday VWAP, in ATR units)', rows)

    # ---- HTF slope ----
    rows = bucket_analysis(full, 'htf', [
        ('htf_slope < 0 (down)',  full['htf_slope']<0),
        ('htf_slope >= 0 (up)',   full['htf_slope']>=0),
    ])
    print_table('60-min EMA(21) slope', rows)

    # ---- HTF + break direction interaction ----
    rows = bucket_analysis(full, 'htf+break', [
        ('htf up + break up',      (full['htf_slope']>=0) & (full['break_up']==True)),
        ('htf up + break down',    (full['htf_slope']>=0) & (full['break_down']==True)),
        ('htf down + break up',    (full['htf_slope']<0) & (full['break_up']==True)),
        ('htf down + break down',  (full['htf_slope']<0) & (full['break_down']==True)),
    ])
    print_table('HTF slope x break direction', rows)

    # ---- Bar body / close position ----
    rows = bucket_analysis(full, 'body', [
        ('body < -0.5 (strong bear)', full['body_ratio']<-0.5),
        ('body -0.5 to 0',            (full['body_ratio']>=-0.5) & (full['body_ratio']<0)),
        ('body 0 to +0.5',            (full['body_ratio']>=0) & (full['body_ratio']<0.5)),
        ('body >= 0.5 (strong bull)', full['body_ratio']>=0.5),
    ])
    print_table('Fire bar body ratio (close-open)/(high-low)', rows)

    # ---- Vol spike ----
    rows = bucket_analysis(full, 'volspike', [
        ('vol < 1x',     full['vol_spike']<1.0),
        ('1-2x',         (full['vol_spike']>=1.0) & (full['vol_spike']<2.0)),
        ('2-3x',         (full['vol_spike']>=2.0) & (full['vol_spike']<3.0)),
        ('>= 3x',        full['vol_spike']>=3.0),
    ])
    print_table('Volume spike (fire bar vs 20-bar avg)', rows)

    # ---- CPR bias ----
    rows = bucket_analysis(full, 'cpr', [
        ('cpr_bias = +1 (above prev pivot)', full['cpr_bias']==1),
        ('cpr_bias = -1 (below prev pivot)', full['cpr_bias']==-1),
    ])
    print_table('CPR bias — day open vs prev day pivot', rows)

    # ---- Time of day ----
    rows = bucket_analysis(full, 'tod', [
        ('morning (9:45-11:00)',  full['tod']=='morning'),
        ('midday (11:00-13:00)',  full['tod']=='midday'),
        ('afternoon (13:00-14:00)', full['tod']=='afternoon'),
    ])
    print_table('Time of day', rows)

    # ---- Day of week ----
    dow_names = ['Mon','Tue','Wed','Thu','Fri']
    rows = bucket_analysis(full, 'dow', [
        (n, full['dow']==i) for i, n in enumerate(dow_names)
    ])
    print_table('Day of week', rows)

    # ---- ATR pct ----
    rows = bucket_analysis(full, 'atr_pct', [
        ('atr% < 0.3',         full['atr_pct']<0.3),
        ('0.3 to 0.6',         (full['atr_pct']>=0.3) & (full['atr_pct']<0.6)),
        ('0.6 to 1.0',         (full['atr_pct']>=0.6) & (full['atr_pct']<1.0)),
        ('>= 1.0',             full['atr_pct']>=1.0),
    ])
    print_table('ATR as % of price (volatility regime)', rows)

    # ---- High-confluence combo: best directional filter ----
    print()
    print('=' * 95)
    print('SEARCH FOR HIGH-DIRECTIONAL-LIFT COMBINATIONS')
    print('=' * 95)
    combos = [
        ('LONG: rsi>60 + body>0.3 + htf>0 + break_up',
         (full['rsi']>60) & (full['body_ratio']>0.3) & (full['htf_slope']>0) & (full['break_up']==True),
         'UP'),
        ('LONG: rsi>55 + close_in_range>0.7 + htf>0',
         (full['rsi']>55) & (full['close_in_range']>0.7) & (full['htf_slope']>0),
         'UP'),
        ('LONG: vwap_dist>0.5 + htf>0 + body>0',
         (full['vwap_dist_atr']>0.5) & (full['htf_slope']>0) & (full['body_ratio']>0),
         'UP'),
        ('LONG: rsi>65 + vol>=1.5 + htf>0',
         (full['rsi']>65) & (full['vol_spike']>=1.5) & (full['htf_slope']>0),
         'UP'),
        ('SHORT: rsi<40 + body<-0.3 + htf<0 + break_down',
         (full['rsi']<40) & (full['body_ratio']<-0.3) & (full['htf_slope']<0) & (full['break_down']==True),
         'DOWN'),
        ('SHORT: rsi<45 + close_in_range<0.3 + htf<0',
         (full['rsi']<45) & (full['close_in_range']<0.3) & (full['htf_slope']<0),
         'DOWN'),
        ('SHORT: vwap_dist<-0.5 + htf<0 + body<0',
         (full['vwap_dist_atr']<-0.5) & (full['htf_slope']<0) & (full['body_ratio']<0),
         'DOWN'),
        ('SHORT: rsi<35 + vol>=1.5 + htf<0',
         (full['rsi']<35) & (full['vol_spike']>=1.5) & (full['htf_slope']<0),
         'DOWN'),
    ]
    print(f'{"Combination":>60} {"N":>6} {"UP%":>6} {"DOWN%":>7}  Expected dir')
    print('-' * 95)
    for label, cond, expected in combos:
        sub = full[cond]
        n = len(sub)
        n_up = (sub['outcome']=='UP').sum()
        n_down = (sub['outcome']=='DOWN').sum()
        n_either = n_up + n_down
        up_pct = 100 * n_up / n_either if n_either else 0
        down_pct = 100 * n_down / n_either if n_either else 0
        marker = '<-- HIT' if (expected == 'UP' and up_pct >= 60) or (expected == 'DOWN' and down_pct >= 60) else ''
        print(f'{label:>60} {n:>6} {up_pct:>6.1f} {down_pct:>7.1f}  {expected:>4} {marker}')

    print(f'\nRuntime: {time.time()-t_start:.1f}s')


if __name__ == '__main__':
    sys.exit(main())
