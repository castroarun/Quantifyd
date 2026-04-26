"""
Phase 1d: Normalized SL test — % retracement from entry instead of OR breach.

Methodological fix: wider OR mechanically widens "opposite OR breach" distance,
inflating WR. Use a fixed % adverse move from entry as SL across ALL OR widths
to compare them fairly.

Sweep:
  OR window: 5, 15, 30, 45, 60, 90, 120 (minutes)
  SL %:     0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00 (% retracement from entry)
  Entry:    RSI5m > 60 (long) / < 40 (short), K=12 wait, lenient abort

Win = price never adversely moved more than X% from entry by EOD.
Loss = price's wick (low for long, high for short) reached entry * (1 ± X) at any point.
"""

import sqlite3
from datetime import time as dtime

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


def load_df():
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


def run_variant(df, *, or_min, sl_pct, rsi_long=60, rsi_short=40, wait_k=12):
    """
    SL = entry * (1 - sl_pct/100) for long, entry * (1 + sl_pct/100) for short.
    SL hit if low/high (wick) reaches that level on any subsequent bar.
    """
    total_min = 9 * 60 + 15 + or_min
    or_end_time = dtime(total_min // 60, total_min % 60)

    trades = []
    raw_breaks = 0
    days = df.index.normalize().unique()

    for day in days:
        sess = df[df.index.normalize() == day]
        if len(sess) < 10:
            continue

        or_bars = sess.between_time(SESSION_OPEN, or_end_time)
        if len(or_bars) < 1:
            continue
        or_high = or_bars['high'].max()
        or_low = or_bars['low'].min()
        or_width_pct = (or_high - or_low) / or_low * 100

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

        # Lenient wait for RSI confirm
        rsi_thresh = rsi_long if break_dir == 'LONG' else rsi_short

        def confirms(rval):
            if pd.isna(rval):
                return False
            return rval > rsi_thresh if break_dir == 'LONG' else rval < rsi_thresh

        def outside(close_p):
            return close_p > or_high if break_dir == 'LONG' else close_p < or_low

        entry_i = None
        for k in range(wait_k + 1):
            si = break_i + k
            if si >= len(post):
                break
            row = post.iloc[si]
            if k > 0 and not outside(row['close']):
                continue  # lenient: keep watching
            if outside(row['close']) and confirms(row['rsi5m']):
                entry_i = si
                break
        if entry_i is None:
            continue
        if post.index[entry_i].time() >= NO_ENTRY_AFTER:
            continue

        entry_ts = post.index[entry_i]
        entry_price = float(post.iloc[entry_i]['close'])
        sl_price = entry_price * (1 - sl_pct / 100) if break_dir == 'LONG' \
                   else entry_price * (1 + sl_pct / 100)

        # Outcome: did price (wick) reach SL by EOD?
        rest = post.iloc[entry_i + 1:]
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

        eod_close = float(sess.iloc[-1]['close'])
        favorable_move = (eod_close / entry_price - 1) * 100 if break_dir == 'LONG' \
                         else (entry_price / eod_close - 1) * 100

        # Also track if opposite OR was breached (for diagnostic, not used as SL here)
        opp_or_breached = False
        for _, rrow in rest.iterrows():
            if break_dir == 'LONG' and rrow['close'] < or_low:
                opp_or_breached = True
                break
            if break_dir == 'SHORT' and rrow['close'] > or_high:
                opp_or_breached = True
                break

        trades.append({
            'date': day.date(),
            'direction': break_dir,
            'or_width_pct': or_width_pct,
            'entry_ts': entry_ts,
            'entry_price': entry_price,
            'sl_hit': sl_hit,
            'sl_time': sl_time,
            'favorable_eod_pct': favorable_move,
            'opp_or_breached_eod': opp_or_breached,
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
        'sl_pct': sl_pct,
        'N': n,
        'WR_pct': round(len(wins) / n * 100, 1),
        'Tr_per_yr': round(n / YEARS_IN_SAMPLE, 1),
        'Wins_per_yr': round(len(wins) / YEARS_IN_SAMPLE, 1),
        'Losses_per_yr': round(len(losses) / YEARS_IN_SAMPLE, 1),
        'Med_OR_width_pct': round(float(tdf['or_width_pct'].median()), 3),
        'Med_min_to_SL': round(float(losses['mins_to_sl'].median()), 0) if len(losses) else None,
        'Avg_favorable_eod_pct': round(float(wins['favorable_eod_pct'].mean()), 3) if len(wins) else None,
        # How often did opposite OR breach happen on losing trades vs winning ones?
        'OppOR_brc_on_wins_pct': round(float(wins['opp_or_breached_eod'].mean()) * 100, 1) if len(wins) else None,
    }


def main():
    print("Loading NIFTY50 5-min...")
    df = load_df()
    print(f"  {len(df)} bars, {df.index.normalize().nunique()} sessions, ~{YEARS_IN_SAMPLE:.2f} years\n")

    or_widths = [5, 15, 30, 45, 60, 90, 120]
    sl_pcts = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00]

    rows = []
    for w in or_widths:
        for x in sl_pcts:
            r = run_variant(df, or_min=w, sl_pct=x)
            if r:
                rows.append(r)

    res = pd.DataFrame(rows)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.width', 220)

    print("=" * 180)
    print("NORMALIZED SL: % RETRACEMENT FROM ENTRY (not OR breach)")
    print("Entry: RSI5m>60 L / <40 S, K=12 lenient. Win = price never moved more than SL% adversely from entry.")
    print("=" * 180)
    print(res.to_string(index=False))

    # Pivot table: WR by OR x SL%
    print("\n" + "=" * 100)
    print("WR % PIVOT  (rows=OR min, cols=SL %)")
    print("=" * 100)
    pivot_wr = res.pivot(index='or_min', columns='sl_pct', values='WR_pct')
    print(pivot_wr.to_string())

    print("\n" + "=" * 100)
    print("WINS PER YEAR PIVOT  (rows=OR min, cols=SL %)")
    print("=" * 100)
    pivot_wy = res.pivot(index='or_min', columns='sl_pct', values='Wins_per_yr')
    print(pivot_wy.to_string())

    print("\n" + "=" * 100)
    print("MEDIAN OR WIDTH (% of spot) BY OR WINDOW  — diagnostic")
    print("=" * 100)
    diag = res.groupby('or_min')['Med_OR_width_pct'].first().to_frame()
    print(diag.to_string())

    res.to_csv('nifty_orb_pct_sl_sweep.csv', index=False)
    print(f"\nSaved: nifty_orb_pct_sl_sweep.csv")

    # Top 15 by Wins/year (with WR floor for sanity)
    print("\n" + "=" * 100)
    print("TOP 15 by Wins/year (WR >= 60%)")
    print("=" * 100)
    top = res[res['WR_pct'] >= 60].sort_values('Wins_per_yr', ascending=False).head(15)
    print(top[['or_min', 'sl_pct', 'N', 'WR_pct', 'Tr_per_yr', 'Wins_per_yr',
              'Losses_per_yr', 'Med_OR_width_pct', 'Med_min_to_SL']].to_string(index=False))


if __name__ == '__main__':
    main()
