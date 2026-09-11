# -*- coding: utf-8 -*-
"""OA: a two-year trade ledger you can check by hand, for two entry mechanics side by side.

Arun asked to publish the trades so he can verify them himself rather than take my word.
That is exactly right, and it is also the artefact that makes the entry finding checkable:
the published mechanic and the best placeable one are generated from the SAME signal, the
SAME exits and the SAME book, differing only in when and at what price the entry happens.

Columns are chosen so the defect is visible without any commentary:

    pivot            the level the order sits at
    entry price      what the mechanic pays
    day open/high/close on the entry day

For the published mechanic, entry price is at or near the OPEN while the CLOSE is far
above the pivot - and that gap, repeated over every trade, is the edge that cannot be
captured by a real order. Seeing the three numbers in one row is the whole argument.

ON TIMES. The study runs on daily bars, so there is no intraday time in it. Rather than
invent timestamps, each row carries the time the mechanic implies: the open for a
resting-stop fill, and about 15:20 for a close-based decision. Stated as such.

DRAWDOWN is taken from the book equity curve at the exit date, not from the trade, because
a per-trade drawdown would mean something different and less useful.

Output: static/app/oa_ledger.json, read by the page as a static file.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
sys.path.insert(0, str(ROOT / 'research/158_oa_arming_width/scripts'))
sys.path.insert(0, str(ROOT))
import oa_entry_mechanics as em          # noqa: E402

WIN = ('2024-09-01', '2026-09-04')
SEED = 7                                  # one deterministic path; stated on the page
OUT = ROOT / 'static/app/oa_ledger.json'

ARMS = [
    dict(key='published', label='As published (not placeable)', trail=15, mode='open_same',
         entry_time='09:15 open', note='Counts a trade only when the CLOSE finished above '
                                       'the pivot, but pays the OPEN. Needs the closing '
                                       'price to be known at the open, so no order can do '
                                       'this. Shown so the gap is visible.'),
    dict(key='placeable', label='Best placeable (stop above the breakout candle)', trail=20,
         mode='stop_above_candle', entry_time='intraday, on the stop',
         note='The breakout closed above the pivot yesterday; a stop rests above that '
              'candle high today and fills only if the price exceeds it.'),
]


def build(arm, w):
    close, high, open_, athcp, sma, tv20 = (w[k] for k in
                                            ('close', 'high', 'open', 'athcp',
                                             'sma50', 'tv20'))
    etf = [c for c in close.columns if em.ETF_RE.search(c)]
    tv_prev, prev_close = tv20.shift(1), close.shift(1)
    elig = tv_prev >= em.TV_FLOOR
    elig[etf] = False
    r = {n: close / close.shift(n) - 1 for n in (63, 126, 189, 252)}
    rs = ((2 * r[63] + r[126] + r[189] + r[252]).where(elig)
          .rank(axis=1, pct=True) * 100).shift(1)
    setup = (prev_close < athcp) & (prev_close >= 0.8 * athcp) & elig & (rs >= 70.0)
    trig = setup & (close > athcp) & athcp.notna()
    piv = athcp
    if arm['mode'] == 'stop_above_candle':
        hp = high.shift(1)
        trig = trig.shift(1).fillna(False).astype(bool) & (high > hp)
        piv = hp
    dates = close.index
    days = np.array([i for i, d in enumerate(dates)
                     if WIN[0] <= str(d.date()) <= WIN[1]])
    eq, trades, _ = em.simulate(
        SEED, 'rs', days, dates, close.values, high.values, open_.values,
        piv.values, sma.values, rs.values, tv_prev.values,
        trig.fillna(False).values, np.zeros(len(dates), bool),
        True, 0.0025, slots=16, size_pct=0.0625)
    ec = pd.Series(eq, index=dates[days])
    dd = (ec / ec.cummax() - 1) * 100
    cols = list(close.columns)
    rows, cum = [], 0.0
    for tr in sorted(trades, key=lambda x: x[2]):
        c, ei, xi, b, s, reason = tr[0], tr[1], tr[2], tr[3], tr[4], tr[5]
        q = tr[6] if len(tr) > 6 else 0
        sym = cols[c]
        pnl = q * (s - b) - 0.0025 * q * (b + s)
        cum += pnl
        ed, xd = dates[ei], dates[xi]
        rows.append(dict(
            symbol=sym,
            entry_date=str(ed.date()), exit_date=str(xd.date()),
            pivot=round(float(piv.values[ei, c]), 2)
            if np.isfinite(piv.values[ei, c]) else None,
            entry=round(float(b), 2), exit=round(float(s), 2), qty=int(q),
            day_open=round(float(open_.values[ei, c]), 2),
            day_high=round(float(high.values[ei, c]), 2),
            day_close=round(float(close.values[ei, c]), 2),
            held_days=int((xd - ed).days),
            reason=reason,
            ret_pct=round(100.0 * (s / b - 1), 2),
            pnl=round(pnl), cum_pnl=round(cum),
            dd_pct=round(float(dd.iloc[max(0, list(days).index(xi))]), 2)
            if xi in days else None))
    st, _ = em.stats_from(eq, dates[days], trades, em.CAPITAL)
    return rows, st


def main():
    print('loading frames (full history, so the pivot is a true all-time high) ...',
          flush=True)
    out = dict(generated=pd.Timestamp.now().strftime('%Y-%m-%d %H:%M IST'),
               window=list(WIN), seed=SEED, arms=[])
    for arm in ARMS:
        w = em.load_frames('2005-01-01', trail_sma=arm['trail'])
        rows, st = build(arm, w)
        print('  %-12s %d trades, CAGR %.1f%%, DD %.1f%%'
              % (arm['key'], len(rows), st['cagr'], st['dd']), flush=True)
        out['arms'].append(dict(key=arm['key'], label=arm['label'], trail=arm['trail'],
                                entry_time=arm['entry_time'], note=arm['note'],
                                cagr=round(st['cagr'], 2), dd=round(st['dd'], 2),
                                n=len(rows), win=round(st['win'], 1), trades=rows))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, 'w'))
    print('wrote %s (%.0f KB)' % (OUT, OUT.stat().st_size / 1024))


if __name__ == '__main__':
    main()
