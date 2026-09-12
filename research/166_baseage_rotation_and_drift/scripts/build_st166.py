# -*- coding: utf-8 -*-
"""research/166 -- build the SuperTrend(14,4) LINE VALUES for every symbol in the panel.

research/161's panel stores only the exit SIGNAL (direction == -1).  The rotation axis needs
the distance of a holding's close ABOVE its trailing line -- "how close is this position to
being stopped out" -- so the line itself has to be reconstructed.  It is rebuilt here from
exactly the same rows research/159's `bt_core.Panel` used (volume > 0, close > 0, duplicate
dates dropped keeping the last, at least 60 bars, aligned to the master calendar), so the
line lines up bar-for-bar with the signal already in the panel.

Output: results/st166.pkl  ->  {symbol: float32 array over the master calendar}
Runtime: about four minutes.  Verifies itself against panel.sig['ST_14_4'].
"""
import pickle
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / 'results'
ROOT = HERE.parent.parent.parent                      # /home/arun/quantifyd
PANEL_PKL = ROOT / 'research' / '164_baseage_slots_sizing' / 'results' / 'panel164.pkl'
DB = ROOT / 'backtest_data' / 'market_data.db'
sys.path.insert(0, str(HERE))

PERIOD, MULT = 14, 4.0


def supertrend_line(high, low, close, period, mult):
    """Returns (line, direction).  `line` is the ACTIVE band: the lower (trailing-stop) band
    while the trend is up, the upper band while it is down.  Identical arithmetic to
    bt_core.supertrend_dir -- only the band is additionally returned."""
    n = len(close)
    if n <= period + 2:
        return np.full(n, np.nan), np.ones(n, dtype=np.int8)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    tr[1:] = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - close[:-1]),
                                   np.abs(low[1:] - close[:-1])))
    atr = np.full(n, np.nan)
    atr[period - 1] = np.nanmean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    hl2 = (high + low) / 2.0
    up, dn = hl2 + mult * atr, hl2 - mult * atr
    fu, fl = np.copy(up), np.copy(dn)
    d = np.ones(n, dtype=np.int8)
    for i in range(period + 1, n):
        fu[i] = up[i] if (up[i] < fu[i - 1] or close[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = dn[i] if (dn[i] > fl[i - 1] or close[i - 1] < fl[i - 1]) else fl[i - 1]
        d[i] = (-1 if close[i] < fl[i] else 1) if d[i - 1] == 1 else (1 if close[i] > fu[i] else -1)
    d[:period + 1] = 1
    line = np.where(d == 1, fl, fu)
    line[:period + 1] = np.nan
    return line, d


def main():
    t0 = time.time()
    sys.path.insert(0, str(HERE))
    panel = pickle.load(open(PANEL_PKL, 'rb'))
    print('panel: %d symbols, %d days (%s .. %s)'
          % (len(panel.close), panel.n, panel.cal[0], panel.cal[-1]), flush=True)
    con = sqlite3.connect('file:%s?mode=ro' % DB, uri=True)

    out = {}
    bad = 0
    checked = agree = 0
    syms = sorted(panel.close)
    for k, sym in enumerate(syms):
        d = pd.read_sql_query(
            "SELECT date,open,high,low,close,volume FROM market_data_unified "
            "WHERE symbol=? AND timeframe='day' ORDER BY date", con, params=(sym,))
        d = d[(d['volume'] > 0) & (d['close'] > 0)]
        d = d.drop_duplicates(subset='date', keep='last')
        idx = d['date'].map(panel.pos)
        keep = idx.notna()
        d, idx = d[keep], idx[keep].astype(int).to_numpy()
        if len(d) < 60:
            bad += 1
            out[sym] = np.full(panel.n, np.nan, np.float32)
            continue
        h = d['high'].to_numpy(np.float64)
        l = d['low'].to_numpy(np.float64)
        c = d['close'].to_numpy(np.float64)
        line, dirn = supertrend_line(h, l, c, PERIOD, MULT)
        L = np.full(panel.n, np.nan, np.float32)
        L[idx] = line
        # forward-fill so a holiday / missing bar reads the last real line value, exactly as
        # the panel forward-fills the close
        out[sym] = pd.Series(L).ffill().to_numpy(np.float32)
        # self-check: the reconstructed direction must equal the panel's stored exit signal
        sig = panel.sig[sym]['ST_14_4'][idx]
        checked += len(sig)
        agree += int(((dirn == -1) == sig).sum())
        if (k + 1) % 300 == 0:
            print('  %d/%d  (%.0fs)' % (k + 1, len(syms), time.time() - t0), flush=True)

    print('symbols with too few bars: %d' % bad)
    print('direction self-check: %d/%d bars agree (%.4f%%)'
          % (agree, checked, 100.0 * agree / max(checked, 1)), flush=True)
    RES.mkdir(exist_ok=True)
    with open(RES / 'st166.pkl', 'wb') as f:
        pickle.dump(out, f, protocol=4)
    print('wrote %s in %.0fs' % (RES / 'st166.pkl', time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
