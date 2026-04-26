"""
Nifty ORB + CPR direction filter test.

CPR (Central Pivot Range) computed from PREVIOUS DAY's daily HLC:
  P  = (H + L + C) / 3
  BC = (H + L) / 2
  TC = 2 * P - BC
  CPR_low  = min(BC, TC)
  CPR_high = max(BC, TC)

Filter rule:
  LONG break  : entry candle close > CPR_high
  SHORT break : entry candle close < CPR_low
  Else: skip the signal.

Sweep:
  OR window: 5, 10, 15, 30, 60
  Entry filter: K=0 strict, RSI5m > 60 (long) / < 40 (short)
  SL: OR-width × 1.0 (wick-based)
  Compare WR/wins/year with vs without CPR filter.
"""

import sqlite3
from datetime import time as dtime, date as ddate

import numpy as np
import pandas as pd

DB = 'backtest_data/market_data.db'
SYMBOL = 'NIFTY50'
SESSION_OPEN = dtime(9, 15)
SESSION_CLOSE = dtime(15, 30)
NO_ENTRY_AFTER = dtime(14, 0)
YEARS_IN_SAMPLE = 455 / 252


def rsi(s, p=14):
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1/p, adjust=False).mean()
    al = loss.ewm(alpha=1/p, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def load_5min():
    conn = sqlite3.connect(DB)
    df = pd.read_sql(
        "SELECT date, open, high, low, close FROM market_data_unified "
        "WHERE symbol=? AND timeframe='5minute' ORDER BY date",
        conn, params=[SYMBOL])
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    for c in ['open', 'high', 'low', 'close']:
        df[c] = df[c].astype(float)
    df = df.between_time(SESSION_OPEN, SESSION_CLOSE)
    df['rsi5m'] = rsi(df['close'], 14)
    return df


def load_daily():
    """Load NIFTY daily HLC, return dict[date_obj] -> (H, L, C)."""
    conn = sqlite3.connect(DB)
    df = pd.read_sql(
        "SELECT date, high, low, close FROM market_data_unified "
        "WHERE symbol=? AND timeframe='day' ORDER BY date",
        conn, params=[SYMBOL])
    conn.close()
    df['date'] = pd.to_datetime(df['date']).dt.date
    return {row['date']: (float(row['high']), float(row['low']), float(row['close']))
            for _, row in df.iterrows()}


def compute_cpr_map(daily_hlc):
    """For each date, compute CPR using prior trading day's HLC.
    Returns dict[curr_date] -> (P, BC, TC, CPR_low, CPR_high)
    """
    sorted_dates = sorted(daily_hlc.keys())
    cpr = {}
    for i in range(1, len(sorted_dates)):
        prev = sorted_dates[i - 1]
        curr = sorted_dates[i]
        h, l, c = daily_hlc[prev]
        p = (h + l + c) / 3
        bc = (h + l) / 2
        tc = 2 * p - bc
        cpr[curr] = {
            'P': p, 'BC': bc, 'TC': tc,
            'CPR_low': min(bc, tc),
            'CPR_high': max(bc, tc),
            'CPR_width_pct': abs(tc - bc) / p * 100,
        }
    return cpr


def run_variant(df, cpr_map, *, or_min, rsi_long=60, rsi_short=40,
                cpr_mode='none'):
    """cpr_mode: 'none' | 'aligned' | 'against'
    aligned : LONG requires close > CPR_high; SHORT requires close < CPR_low
    against : LONG requires close <= CPR_high; SHORT requires close >= CPR_low
    """
    total_min = 9 * 60 + 15 + or_min
    or_end_time = dtime(total_min // 60, total_min % 60)

    trades = []
    raw_breaks = 0
    skipped_no_cpr = 0
    skipped_cpr_filter = 0
    skipped_rsi = 0
    days = df.index.normalize().unique()

    for day in days:
        day_d = day.date()
        sess = df[df.index.normalize() == day]
        if len(sess) < 10:
            continue

        cpr = cpr_map.get(day_d)
        if cpr is None:
            skipped_no_cpr += 1
            continue

        or_bars = sess.between_time(SESSION_OPEN, or_end_time)
        if len(or_bars) < 1:
            continue
        or_high = or_bars['high'].max()
        or_low = or_bars['low'].min()

        post = sess.between_time(or_end_time, SESSION_CLOSE).iloc[1:]
        if len(post) < 2:
            continue

        # Find first break
        break_i, break_dir = None, None
        for i, (_, row) in enumerate(post.iterrows()):
            if row['close'] > or_high:
                break_i, break_dir = i, 'LONG'
                break
            if row['close'] < or_low:
                break_i, break_dir = i, 'SHORT'
                break
        if break_i is None or post.index[break_i].time() >= NO_ENTRY_AFTER:
            continue

        raw_breaks += 1
        entry_row = post.iloc[break_i]
        entry_close = float(entry_row['close'])

        # K=0 strict RSI confirmation
        r = entry_row['rsi5m']
        if pd.isna(r):
            skipped_rsi += 1
            continue
        if break_dir == 'LONG' and r <= rsi_long:
            skipped_rsi += 1
            continue
        if break_dir == 'SHORT' and r >= rsi_short:
            skipped_rsi += 1
            continue

        # CPR direction filter
        if cpr_mode == 'aligned':
            if break_dir == 'LONG' and entry_close <= cpr['CPR_high']:
                skipped_cpr_filter += 1
                continue
            if break_dir == 'SHORT' and entry_close >= cpr['CPR_low']:
                skipped_cpr_filter += 1
                continue
        elif cpr_mode == 'against':
            if break_dir == 'LONG' and entry_close > cpr['CPR_high']:
                skipped_cpr_filter += 1
                continue
            if break_dir == 'SHORT' and entry_close < cpr['CPR_low']:
                skipped_cpr_filter += 1
                continue

        entry_ts = post.index[break_i]
        entry_price = entry_close

        # OR-width × 1.0 SL (wick-based)
        sl_price = or_low if break_dir == 'LONG' else or_high

        rest = post.iloc[break_i + 1:]
        sl_hit = False
        sl_time = None
        for rts, rrow in rest.iterrows():
            if break_dir == 'LONG' and rrow['low'] <= sl_price:
                sl_hit = True
                sl_time = rts
                break
            if break_dir == 'SHORT' and rrow['high'] >= sl_price:
                sl_hit = True
                sl_time = rts
                break

        trades.append({
            'date': day_d,
            'direction': break_dir,
            'entry_ts': entry_ts,
            'entry_price': entry_price,
            'cpr_width_pct': round(cpr['CPR_width_pct'], 3),
            'sl_hit': sl_hit,
            'mins_to_sl': (sl_time - entry_ts).total_seconds() / 60 if sl_hit else None,
        })

    n = len(trades)
    if n == 0:
        return None
    tdf = pd.DataFrame(trades)
    wins = tdf[~tdf['sl_hit']]
    losses = tdf[tdf['sl_hit']]
    return {
        'or_min': or_min,
        'cpr_mode': cpr_mode,
        'N': n,
        'WR_pct': round(len(wins) / n * 100, 1),
        'Tr_per_yr': round(n / YEARS_IN_SAMPLE, 1),
        'Wins_per_yr': round(len(wins) / YEARS_IN_SAMPLE, 1),
        'Losses_per_yr': round(len(losses) / YEARS_IN_SAMPLE, 1),
        'raw_breaks': raw_breaks,
        'skipped_rsi': skipped_rsi,
        'skipped_cpr': skipped_cpr_filter,
        'skipped_no_cpr_data': skipped_no_cpr,
        'med_min_to_sl': round(float(losses['mins_to_sl'].median()), 0) if len(losses) else None,
    }


def main():
    print("Loading NIFTY 5-min and daily...")
    df = load_5min()
    print(f"  5-min: {len(df)} bars, {df.index.normalize().nunique()} sessions")
    daily = load_daily()
    print(f"  Daily: {len(daily)} bars")
    cpr_map = compute_cpr_map(daily)
    print(f"  CPR computed for {len(cpr_map)} dates")

    rows = []
    for or_min in [5, 10, 15, 30, 60]:
        for mode in ['none', 'aligned', 'against']:
            r = run_variant(df, cpr_map, or_min=or_min, cpr_mode=mode)
            if r:
                label = {
                    'none': 'no CPR filter',
                    'aligned': 'CPR-aligned (must clear CPR)',
                    'against': 'CPR-against (still inside/wrong-side CPR)',
                }[mode]
                r['variant'] = f'OR{or_min}m  {label}'
                rows.append(r)

    res = pd.DataFrame(rows)
    pd.set_option('display.max_rows', 60)
    pd.set_option('display.width', 220)
    pd.set_option('display.max_colwidth', 50)

    print("\n" + "=" * 150)
    print("CPR-DIRECTION FILTER vs NO FILTER  (entry: K=0 strict, RSI5m>60/<40, SL = OR-width × 1.0)")
    print("=" * 150)
    cols = ['variant', 'N', 'WR_pct', 'Tr_per_yr', 'Wins_per_yr', 'Losses_per_yr',
            'raw_breaks', 'skipped_rsi', 'skipped_cpr', 'med_min_to_sl']
    print(res[cols].to_string(index=False))

    # Side-by-side three-way comparison
    print("\n" + "-" * 110)
    print("COMPARISON  (none / aligned / against)")
    print("-" * 110)
    for or_min in [5, 10, 15, 30, 60]:
        none = res[(res['or_min'] == or_min) & (res['cpr_mode'] == 'none')]
        aligned = res[(res['or_min'] == or_min) & (res['cpr_mode'] == 'aligned')]
        against = res[(res['or_min'] == or_min) & (res['cpr_mode'] == 'against')]
        if len(none) and len(aligned) and len(against):
            print(f"OR{or_min:3d}m:")
            for label, sub in [('  none   ', none), ('  aligned', aligned), ('  against', against)]:
                n, wr, wins = sub.iloc[0][['N', 'WR_pct', 'Wins_per_yr']]
                print(f"    {label}: N={n:4d}  WR={wr:5.1f}%  Wins/yr={wins:6.1f}")

    res.to_csv('nifty_orb_cpr_filter_results.csv', index=False)
    print(f"\nSaved: nifty_orb_cpr_filter_results.csv")

    # Top 5 by Wins/yr across all configs
    print("\n" + "=" * 80)
    print("TOP 5 by Wins/yr across all configs")
    print("=" * 80)
    top = res.sort_values('Wins_per_yr', ascending=False).head(5)
    print(top[['variant', 'N', 'WR_pct', 'Wins_per_yr']].to_string(index=False))


if __name__ == '__main__':
    main()
