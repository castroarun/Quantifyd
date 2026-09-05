"""P1d — score three candidate pivot families against the 37 usable ground-truth trades.

Established so far (P1/P1b/P1c):
  * the buy price is an exact PRIOR CLOSE (36/37 within 0.15%), never the ATH close
    (median 6% below it), never exceeded on a closing basis between the pivot bar and
    the break, and the entry day's CLOSE is above it (37/37)
  * NO fixed lookback N can produce it: N would have to be >= 157 (deepest pivot age)
    and <= 11 (shortest run since a higher close) at the same time

Families scored here (each gives pivot(i) usable at bar i from data through i-1):
  F1  rolling: pivot = max close over [i-N, i)
  F2  base-high: highest close in [i-L, i) that was subsequently followed by a close
      >= X% below it (a confirmed contraction) and has not been exceeded since
  F3  zigzag: last confirmed swing-peak close using a z% close-basis reversal

Score per cell on the 37 trades:
  price_hit  pivot(entry_bar) == their buy within 0.15%
  first_hit  close[i] > pivot(i) and close[i-1] <= pivot(i-1)  (a first break that day)
  both       price_hit AND first_hit
Written incrementally to results/p1d_family_scan.csv.
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
OUT = STUDY / 'results' / 'p1d_family_scan.csv'
TOL = 0.0015


def piv_rolling(C, N):
    s = pd.Series(C)
    return s.shift(1).rolling(N, min_periods=5).max().values


def piv_basehigh(C, L, X):
    """Highest close in the trailing L bars that has since been followed by a close
    >= X% below it, and has not been exceeded on a closing basis since."""
    n = len(C)
    out = np.full(n, np.nan)
    for i in range(5, n):
        lo = max(0, i - L)
        seg = C[lo:i]
        if len(seg) < 5:
            continue
        best = np.nan
        # walk candidate peaks from highest downward
        order = np.argsort(-seg)
        for j in order[:40]:
            p = seg[j]
            after = seg[j + 1:]
            if len(after) == 0:
                continue
            if after.min() <= p * (1 - X) and after.max() <= p * (1 + TOL):
                best = p
                break
        out[i] = best
    return out


def piv_zigzag(C, z):
    """Last confirmed swing-peak close; a peak confirms when close falls z% below it."""
    n = len(C)
    out = np.full(n, np.nan)
    run_max = C[0]
    run_min = C[0]
    up = True
    last_peak = np.nan
    for i in range(1, n):
        out[i] = last_peak            # value usable at bar i (from data through i-1)
        c = C[i - 1]
        if up:
            if c > run_max:
                run_max = c
            elif c <= run_max * (1 - z):
                last_peak = run_max
                up = False
                run_min = c
        else:
            if c < run_min:
                run_min = c
            elif c >= run_min * (1 + z):
                up = True
                run_max = c
    return out


def main():
    db = sqlite3.connect(str(DB))
    gt = [t for t in csv.DictReader(open(GT))]
    data = {}
    for t in gt:
        s = t['symbol']
        if s in data:
            continue
        df = pd.read_sql_query(
            "select date, close from market_data_unified where symbol=? and "
            "timeframe='day' order by date", db, params=(s,))
        if df.empty:
            continue
        df['date'] = pd.to_datetime(df['date'].str[:10])
        df = df.drop_duplicates('date').set_index('date').sort_index()
        data[s] = df['close']
    db.close()

    trades = []
    for t in gt:
        s = t['symbol']
        if s not in data:
            continue
        ser = data[s]
        ed = pd.Timestamp(t['entry_date'])
        if ed not in ser.index:
            continue
        trades.append((s, ser.index.get_loc(ed), float(t['buy']), t['entry_date']))
    print(f'{len(trades)} scorable trades')

    cells = []
    for N in (10, 15, 20, 25, 30, 35, 40, 50, 60, 75, 100, 150, 252):
        cells.append(('F1_rolling', dict(N=N)))
    for L in (30, 45, 60, 90, 120, 180, 252):
        for X in (0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20):
            cells.append(('F2_basehigh', dict(L=L, X=X)))
    for z in (0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10):
        cells.append(('F3_zigzag', dict(z=z)))

    rows = []
    cache = {}
    for fam, p in cells:
        key = (fam, tuple(sorted(p.items())))
        price_hit = first_hit = both = 0
        detail = []
        for s, i, buy, ed in trades:
            C = data[s].values
            ck = (s,) + key
            if ck not in cache:
                if fam == 'F1_rolling':
                    cache[ck] = piv_rolling(C, p['N'])
                elif fam == 'F2_basehigh':
                    cache[ck] = piv_basehigh(C, p['L'], p['X'])
                else:
                    cache[ck] = piv_zigzag(C, p['z'])
            pv = cache[ck]
            v = pv[i]
            ph = (v == v) and abs(v - buy) / buy <= TOL
            fh = False
            if v == v and i >= 1 and pv[i - 1] == pv[i - 1]:
                fh = bool(C[i] > v and C[i - 1] <= pv[i - 1])
            price_hit += ph
            first_hit += fh
            both += (ph and fh)
            detail.append(f'{s}:{"P" if ph else "."}{"F" if fh else "."}')
        rows.append(dict(family=fam, params=';'.join(f'{k}={v}' for k, v in p.items()),
                         n=len(trades), price_hit=price_hit, first_hit=first_hit,
                         both=both, price_pct=round(price_hit / len(trades) * 100, 1),
                         both_pct=round(both / len(trades) * 100, 1)))
        print(f"{fam:12s} {rows[-1]['params']:14s} price {price_hit:2d}/{len(trades)} "
              f"first {first_hit:2d} both {both:2d}", flush=True)
        # keep the cache small
        if len(cache) > 4000:
            cache = {}
    d = pd.DataFrame(rows).sort_values(['both', 'price_hit'], ascending=False)
    d.to_csv(OUT, index=False)
    print('\n=== TOP 15 by joint (price + first-break) match ===')
    print(d.head(15).to_string(index=False))


if __name__ == '__main__':
    main()
