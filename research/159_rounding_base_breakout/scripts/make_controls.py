"""
research/159 — null controls (STATUS 11.4). These decide the study.

The v3 pattern is a SUBSET of Open Alpha's "buy strength near the all-time high".
So the question is not "does it make money" but "does the saucer + shelf shape add
anything to simply buying a volume thrust near the ATH".

(a) NEAR-ATH CONTROL — `control_ath_events.csv`
    close >= 0.90 x causal ATH  AND  volume >= 3x prior-20-bar median  AND  up-candle,
    with the SAME liquidity floor, the SAME split-guarded ATH and the SAME 60-bar
    re-arm as v3 -- but NO saucer, NO shelf. If v3 does not beat this after tax, the
    verdict is "no incremental edge over Open Alpha".

(b) RANDOM CONTROL — `liquid_matrix.npz`
    a boolean symbol x day matrix of "liquid and tradeable", so the book runner can draw
    date-matched random entries per seed.

Date-matching is done in the runner (per seed), not here: the pooled population is
written once and sampled against the v3 trigger dates inside each simulation.
"""
import csv
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / 'backtest_data' / 'market_data.db'
RES = Path(__file__).resolve().parents[1] / 'results'

ATH_MIN_FRAC = 0.90
TRIG_K = 3.0
VOL_MED = 20
REARM = 60
LIQ_MIN_TV = 2e7
SPLIT_DOWN = -0.35
SPLIT_UP = 0.50
ETF_PAT = ('BEES', 'ETF', 'IETF', 'GOLD', 'SILVER', 'LIQUID', 'NIFTY', 'SENSEX',
           'BANKNIFTY', 'MIDCAP', 'SMALLCAP', 'INDEX', 'MAFANG', 'HNGSNGBEES')


def main():
    t0 = time.time()
    con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    cal = [d for d in pd.read_sql_query(
        "SELECT DISTINCT date FROM market_data_unified WHERE timeframe='day' "
        "AND symbol='NIFTYBEES' ORDER BY date", con)['date'].tolist() if d >= '2005-01-03']
    pos = {d: i for i, d in enumerate(cal)}
    n = len(cal)
    syms = [r[0] for r in con.execute(
        "SELECT symbol FROM market_data_unified WHERE timeframe='day' "
        "GROUP BY symbol HAVING COUNT(*)>=90 ORDER BY symbol")]
    syms = [s for s in syms if not any(p in s.upper() for p in ETF_PAT) and s != 'SILLYMONKS']
    print('calendar %d days, scanning %d symbols' % (n, len(syms)), flush=True)

    out = open(RES / 'control_ath_events.csv', 'w', newline='', encoding='utf-8')
    w = csv.DictWriter(out, fieldnames=['symbol', 'trigger_date', 'trigger_close',
                                        'ath_before', 'dist_to_ath_pct', 'vol_multiple',
                                        'entry_date', 'entry_open'])
    w.writeheader()
    liquid = np.zeros((len(syms), n), dtype=bool)
    nev = 0
    for si, sym in enumerate(syms, 1):
        d = pd.read_sql_query(
            "SELECT date,open,close,volume FROM market_data_unified WHERE symbol=? "
            "AND timeframe='day' ORDER BY date", con, params=(sym,))
        d = d[(d['volume'] > 0) & (d['close'] > 0)].drop_duplicates(subset='date', keep='last')
        if len(d) < 90:
            continue
        r = d['close'].pct_change()
        hits = r.index[r < SPLIT_DOWN]
        if len(hits):
            d = d.loc[hits.max():]
        if len(d) < 90:
            continue
        d = d[d['date'].isin(pos)].reset_index(drop=True)
        if len(d) < 90:
            continue
        idx = d['date'].map(pos).to_numpy()
        c = d['close'].to_numpy(float); v = d['volume'].to_numpy(float)
        o = d['open'].to_numpy(float)
        cs = pd.Series(c)
        ath = cs.cummax().shift(1).to_numpy()
        med = pd.Series(v).rolling(VOL_MED, min_periods=VOL_MED).median().shift(1).to_numpy()
        with np.errstate(divide='ignore', invalid='ignore'):
            volx = np.where(med > 0, v / med, np.nan)
        tv20 = pd.Series(c * v).rolling(20, min_periods=10).median().to_numpy()
        ret1 = np.empty(len(c)); ret1[0] = 0.0; ret1[1:] = c[1:] / c[:-1] - 1.0
        liq = np.isfinite(tv20) & (tv20 >= LIQ_MIN_TV)
        liquid[si - 1, idx[liq]] = True

        block = -1
        for j in range(VOL_MED + 1, len(c)):
            if j < block or not liq[j]:
                continue
            if (np.isfinite(ath[j]) and c[j] >= ATH_MIN_FRAC * ath[j]
                    and np.isfinite(volx[j]) and volx[j] >= TRIG_K
                    and c[j] > c[j - 1] and abs(ret1[j]) <= SPLIT_UP and j + 1 < len(c)):
                w.writerow(dict(symbol=sym, trigger_date=d['date'].iloc[j],
                                trigger_close=round(c[j], 2), ath_before=round(ath[j], 2),
                                dist_to_ath_pct=round(100 * (c[j] / ath[j] - 1), 2),
                                vol_multiple=round(volx[j], 2),
                                entry_date=d['date'].iloc[j + 1], entry_open=round(o[j + 1], 2)))
                nev += 1
                block = j + REARM
        if si % 400 == 0:
            print('  %d/%d symbols, %d control events, %.0fs' % (si, len(syms), nev, time.time() - t0), flush=True)
    out.close()
    con.close()
    np.savez_compressed(RES / 'liquid_matrix.npz', liquid=liquid,
                        symbols=np.array(syms), dates=np.array(cal))
    print('DONE %.0fs: %d near-ATH control events, liquid matrix %s'
          % (time.time() - t0, nev, liquid.shape), flush=True)


if __name__ == '__main__':
    main()
