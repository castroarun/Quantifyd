"""P1b — nail the VCP pivot definition.

P1 showed the buy price equals a prior CLOSE maximum for 36/37 trades, over short
windows (5-30 bars), i.e. the pivot is a *pattern/base high on a close basis*, not
the all-time-high close (median buy is 6% BELOW the prior ATH close).

This probe asks the precise question: is the pivot simply "the highest prior close
that has not been exceeded since" (a running max close over the base)?  If so, for
every trade there is a unique pivot bar P with close == buy, no close above it
between P and the entry day, and the entry day is the first bar whose CLOSE (or
HIGH) exceeds it.

Outputs per trade:
  pivot_age        bars from the pivot bar to the entry bar
  n_closes_above   closes strictly above buy between the pivot bar and entry (0 expected)
  trig_close       entry-day close > buy?
  trig_high        entry-day high >= buy?
  prev_close_gap%  (buy - prev close)/buy  -> "near the trigger" proximity
  base_depth%      (buy - min low between pivot bar and entry)/buy
  n_swings         alternating k=3 fractal swing highs between pivot bar and entry
  d1..d4           successive contraction depths measured from the swing structure
  vol_ratio        mean volume of the final contraction / mean volume of the base
"""
import csv
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parents[1]
DB = ROOT / 'backtest_data' / 'market_data.db'
GT = STUDY / 'data' / 'vcp_trades_groundtruth.csv'
TOL = 0.0015


def load_bars(db, sym):
    df = pd.read_sql_query(
        "select date, open, high, low, close, volume from market_data_unified "
        "where symbol=? and timeframe='day' order by date", db, params=(sym,))
    if df.empty:
        return df
    df['date'] = pd.to_datetime(df['date'].str[:10])
    return df.drop_duplicates('date').set_index('date').sort_index()


def swings(high, low, k=3):
    """Alternating confirmed fractal swing sequence over the given slice."""
    n = len(high)
    sh, sl = [], []
    for i in range(k, n - k):
        if high[i] >= high[i - k:i + k + 1].max():
            sh.append(i)
        if low[i] <= low[i - k:i + k + 1].min():
            sl.append(i)
    return sh, sl


def main():
    db = sqlite3.connect(str(DB))
    gt = list(csv.DictReader(open(GT)))
    rows = []
    for t in gt:
        s = t['symbol']
        df = load_bars(db, s)
        if df.empty:
            continue
        ed = pd.Timestamp(t['entry_date'])
        if ed not in df.index:
            continue
        i = df.index.get_loc(ed)
        buy = float(t['buy'])
        hist = df.iloc[:i]
        if len(hist) < 60:
            continue
        C = hist['close'].values
        H = hist['high'].values
        L = hist['low'].values
        V = hist['volume'].values.astype(float)
        eday = df.iloc[i]

        # last bar (searching backwards) whose close equals buy within tolerance
        pidx = None
        for j in range(len(C) - 1, -1, -1):
            if abs(C[j] - buy) / buy <= TOL:
                pidx = j
                break
        if pidx is None:
            # fall back to the closest close
            pidx = int(np.argmin(np.abs(C - buy)))
            exact = False
        else:
            exact = True
        pivot_age = len(C) - pidx           # bars from the pivot bar to the entry bar
        seg_c = C[pidx + 1:]
        n_above = int((seg_c > buy * (1 + TOL)).sum())
        seg_h = H[pidx + 1:]
        n_high_above = int((seg_h > buy * (1 + TOL)).sum())
        base_low = float(L[pidx + 1:].min()) if len(L) > pidx + 1 else np.nan
        depth = (buy - base_low) / buy * 100 if base_low == base_low else np.nan

        # swing structure inside the base (pivot bar -> entry)
        sub_h, sub_l = H[pidx:], L[pidx:]
        sh, sl = swings(sub_h, sub_l, 3) if len(sub_h) >= 9 else ([], [])
        # contraction depths: from each local high to the following local low
        marks = sorted([(j, 'H') for j in sh] + [(j, 'L') for j in sl])
        seqd = []
        cur_high = float(sub_h[0])
        for j, kind in marks:
            if kind == 'H':
                cur_high = max(cur_high, float(sub_h[j]))
            else:
                dd = (cur_high - float(sub_l[j])) / cur_high * 100
                if dd > 1.0:
                    seqd.append(round(dd, 2))
                    cur_high = float(sub_h[j]) if j < len(sub_h) else cur_high
        # volume dry-up: last 20% of the base vs whole base
        vb = V[pidx:]
        vtail = vb[max(1, int(len(vb) * 0.8)):]
        vratio = float(vtail.mean() / vb.mean()) if len(vb) and vb.mean() > 0 else np.nan

        rows.append(dict(
            symbol=s, entry=t['entry_date'], buy=buy, exact_close_pivot=exact,
            pivot_age=pivot_age, n_closes_above=n_above, n_highs_above=n_high_above,
            trig_close=float(eday['close']) > buy, trig_high=float(eday['high']) >= buy,
            prev_close_gap_pct=round((buy - float(C[-1])) / buy * 100, 2),
            base_depth_pct=round(depth, 2) if depth == depth else '',
            n_contractions=len(seqd), d_seq=';'.join(map(str, seqd[:5])),
            d1=seqd[0] if len(seqd) > 0 else '', d2=seqd[1] if len(seqd) > 1 else '',
            d3=seqd[2] if len(seqd) > 2 else '', d4=seqd[3] if len(seqd) > 3 else '',
            vol_tail_ratio=round(vratio, 2) if vratio == vratio else ''))
    db.close()
    d = pd.DataFrame(rows)
    d.to_csv(STUDY / 'results' / 'p1b_pivot_probe.csv', index=False)
    n = len(d)
    print(f'=== P1b PIVOT PROBE — {n} trades ===')
    print(f'buy == an exact prior CLOSE (<=0.15%): {int(d.exact_close_pivot.sum())}/{n}')
    print(f'no higher close between pivot bar and entry: {int((d.n_closes_above==0).sum())}/{n}')
    print(f'no higher HIGH between pivot bar and entry: {int((d.n_highs_above==0).sum())}/{n}')
    print(f'entry-day close > pivot: {int(d.trig_close.sum())}/{n}   '
          f'entry-day high >= pivot: {int(d.trig_high.sum())}/{n}')
    print('\npivot_age (bars from base high to breakout):')
    print(d.pivot_age.describe(percentiles=[.1, .25, .5, .75, .9]).round(1).to_string())
    print('\nbase_depth% (pivot -> lowest low in the base):')
    print(pd.to_numeric(d.base_depth_pct, errors='coerce').describe(
        percentiles=[.1, .25, .5, .75, .9]).round(1).to_string())
    print('\nprev_close_gap% (how far below the pivot the stock closed the day before):')
    print(d.prev_close_gap_pct.describe(percentiles=[.1, .5, .9]).round(2).to_string())
    print('\nn_contractions (k=3 fractal, >1% pullbacks) inside the base:')
    print(d.n_contractions.value_counts().sort_index().to_string())
    print('\nvol tail ratio (last 20% of base vs base):')
    print(pd.to_numeric(d.vol_tail_ratio, errors='coerce').describe(
        percentiles=[.25, .5, .75]).round(2).to_string())
    print('\nper-trade detail:')
    print(d[['symbol', 'entry', 'pivot_age', 'n_closes_above', 'base_depth_pct',
             'prev_close_gap_pct', 'n_contractions', 'd_seq', 'vol_tail_ratio']].to_string(index=False))


if __name__ == '__main__':
    main()
