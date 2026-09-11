# -*- coding: utf-8 -*-
"""Stage C: a proper market-gate bake-off, on an entry that can actually be placed.

research/142 ran a 144-cell gate bake-off and RETIRED the gate. Every cell of it was
evaluated on the look-ahead entry, where each trade began in profit by construction - a
regime in which standing aside can only ever cost money, so a gate was guaranteed to lose.
On an honest entry, where roughly half the trades fail and the drawdown is -45%, a gate that
keeps the book out of falling markets has real work to do.

Arun asked for all of it rather than three: simple and exponential averages solo, crossovers,
drawdown-from-high, momentum, and VIX both absolute and relative.

TWO WINDOWS, because the data forces it. NIFTYBEES runs from 2005 so price gates get the
full 2006-2026. INDIA VIX only starts 2015-01-01, so VIX gates get 2016-2026 and are scored
against their OWN no-gate baseline on that same window. Comparing a VIX gate on 2016-2026
against a price gate on 2006-2026 would be comparing the regimes, not the gates.

ADOPTION BAR, pre-registered, because 30 more cells on top of 112 is where overfitting
lives:
  1. beats the no-gate baseline on the SAME window, on Calmar AND on drawdown;
  2. its NEIGHBOURS in the same family also beat no-gate - a family that works, not a cell;
  3. it replicates on the second entry mechanic.
A cell that clears only (1) is reported as noise.
"""
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
sys.path.insert(0, str(ROOT / 'research/158_oa_arming_width/scripts'))
sys.path.insert(0, str(ROOT))
import oa_entry_mechanics as em          # noqa: E402

OUT = ROOT / 'research/159_oa_honest_reoptimization/results/stageC_gates.csv'
SEEDS = list(range(1, 31))
COST, SLOTS, SIZE, STOP, TRAIL = 0.0025, 16, 0.0625, 0.08, 75
CASH_Y = 0.05
W_PRICE = ('2006-01-01', '2026-08-31')
W_VIX = ('2016-01-01', '2026-08-31')     # VIX starts 2015; a year of warm-up for its own MAs

PRICE_GATES = (['none']
               + ['sma%d' % n for n in (50, 100, 150, 200)]
               + ['ema%d' % n for n in (50, 100, 150, 200)]
               + ['xo20_50', 'xo20_100', 'xo50_100', 'xo50_200']
               + ['dd8', 'dd10', 'dd12', 'dd15']
               + ['mom63', 'mom126', 'mom252'])
VIX_GATES = (['none']
             + ['vix15', 'vix18', 'vix20', 'vix25']
             + ['vixsma20', 'vixsma50']
             + ['vixpct70', 'vixpct80', 'vixpct90'])

FIELDS = ['cell', 'entry', 'gate', 'window', 'cagr_med', 'cagr_min', 'cagr_max',
          'dd_med', 'calmar_med', 'trades_med', 'blocked_pct', 'invested_pct', 'secs']


def series_from_db(sym):
    import sqlite3
    con = sqlite3.connect('file:%s?mode=ro' % (ROOT / 'backtest_data/market_data.db'),
                          uri=True)
    df = pd.read_sql_query(
        "select date, close from market_data_unified where symbol=? and timeframe='day' "
        "order by date", con, params=(sym,))
    con.close()
    df['date'] = pd.to_datetime(df['date'].str[:10])
    return df.drop_duplicates('date').set_index('date')['close']


def gate_arr(kind, index, nb, vix):
    """True on days the book stands aside.

    NaN-robust throughout: every statistic is computed on the DROPPED series and only then
    reindexed and forward-filled. A phantom holiday row silently disabled r/142's gate for
    months, which is the single cheapest bug to re-introduce here.
    """
    if kind == 'none':
        return np.zeros(len(index), bool)
    if kind.startswith('vix'):
        s = vix.dropna()
        if kind in ('vix15', 'vix18', 'vix20', 'vix25'):
            off = s > float(kind[3:])
        elif kind.startswith('vixsma'):
            n = int(kind[6:])
            off = s > s.rolling(n, min_periods=n).mean()
        elif kind.startswith('vixpct'):
            p = float(kind[6:]) / 100.0
            off = s > s.rolling(252, min_periods=252).quantile(p)
        else:
            raise ValueError(kind)
    else:
        s = nb.dropna()
        if kind.startswith('sma'):
            n = int(kind[3:]); off = s < s.rolling(n, min_periods=n).mean()
        elif kind.startswith('ema'):
            n = int(kind[3:]); off = s < s.ewm(span=n, adjust=False).mean()
        elif kind.startswith('xo'):
            f, sl = (int(x) for x in kind[2:].split('_'))
            off = (s.rolling(f, min_periods=f).mean()
                   < s.rolling(sl, min_periods=sl).mean())
        elif kind.startswith('dd'):
            x = float(kind[2:]) / 100.0
            off = s < s.rolling(252, min_periods=252).max() * (1 - x)
        elif kind.startswith('mom'):
            n = int(kind[3:]); off = s < s.shift(n)
        else:
            raise ValueError(kind)
    off = off.shift(1)                      # decided on yesterday's close, acted on today
    return (off.reindex(index).ffill().fillna(False).astype(bool).to_numpy())


def build(w, entry):
    close, high, athcp, tv20 = w['close'], w['high'], w['athcp'], w['tv20']
    etf = [c for c in close.columns if em.is_etf(c)]
    tv_prev, prev_close = tv20.shift(1), close.shift(1)
    elig = tv_prev >= em.TV_FLOOR
    elig[etf] = False
    r = {n: close / close.shift(n) - 1 for n in (63, 126, 189, 252)}
    rs = ((2 * r[63] + r[126] + r[189] + r[252]).where(elig)
          .rank(axis=1, pct=True) * 100).shift(1)
    setup = (prev_close < athcp) & (prev_close >= 0.8 * athcp) & elig & (rs >= 70.0)
    trig = setup & (close > athcp) & athcp.notna()
    piv = athcp
    if entry == 'stop_above_candle':
        hp = high.shift(1)
        trig = trig.shift(1).fillna(False).astype(bool) & (high > hp)
        piv = hp
    return trig, piv, rs, tv_prev


def main():
    have = set()
    if OUT.exists():
        with open(OUT) as f:
            have = {r['cell'] for r in csv.DictReader(f)}
    else:
        with open(OUT, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    print('loading frames (trail-%d, the plateau) ...' % TRAIL, flush=True)
    w = em.load_frames('2005-01-01', trail_sma=TRAIL)
    close, sma, dates = w['close'], w['sma50'], w['close'].index
    nb = close.get('NIFTYBEES')
    vix = series_from_db('INDIAVIX').reindex(dates)
    print('NIFTYBEES %d days, INDIA VIX %d days (from %s)'
          % (nb.notna().sum(), vix.notna().sum(),
             str(vix.dropna().index[0].date())), flush=True)

    built = {}
    plan = ([('close_same', g, W_PRICE) for g in PRICE_GATES]
            + [('close_same', g, W_VIX) for g in VIX_GATES])
    for entry, gate, win in plan:
        key = 'g|%s|%s|%s' % (entry, gate, win[0][:4])
        if key in have:
            continue
        if entry not in built:
            built[entry] = build(w, entry)
        trig, piv, rs, tvp = built[entry]
        days = np.array([i for i, d in enumerate(dates)
                         if win[0] <= str(d.date()) <= win[1]])
        garr = gate_arr(gate, dates, nb, vix)
        t0, stats, inv = time.time(), [], []
        for sd in SEEDS:
            eq, trd, _, ifrac = em.simulate(
                sd, 'random', days, dates, close.values, w['high'].values,
                w['open'].values, piv.values, sma.values, rs.values, tvp.values,
                trig.fillna(False).values, garr, True, COST, stop=STOP, slots=SLOTS,
                size_pct=SIZE, fill_close=(entry == 'close_same'), cash_yield=CASH_Y)
            st, _ = em.stats_from(eq, dates[days], trd, em.CAPITAL)
            stats.append(st)
            inv.append(100.0 * ifrac)
        d = pd.DataFrame(stats)
        row = dict(cell=key, entry=entry, gate=gate, window=win[0][:4] + '-2026',
                   cagr_med=round(d.cagr.median(), 2), cagr_min=round(d.cagr.min(), 2),
                   cagr_max=round(d.cagr.max(), 2), dd_med=round(d.dd.median(), 2),
                   calmar_med=round((d.cagr / d.dd.abs()).median(), 3),
                   trades_med=int(d.n.median()),
                   blocked_pct=round(100.0 * garr[days].mean(), 1),
                   invested_pct=round(float(np.mean(inv)), 1),
                   secs=round(time.time() - t0))
        with open(OUT, 'a', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
        print('  %-10s %-10s %s  CAGR %6.2f [%6.2f..%6.2f]  DD %6.2f  Cal %5.3f  '
              'blocked %4.1f%%  inv %4.1f%%  (%ds)'
              % (gate, row['window'], entry[:12], row['cagr_med'], row['cagr_min'],
                 row['cagr_max'], row['dd_med'], row['calmar_med'], row['blocked_pct'],
                 row['invested_pct'], row['secs']), flush=True)
    print('STAGE_C_DONE', flush=True)


if __name__ == '__main__':
    main()
