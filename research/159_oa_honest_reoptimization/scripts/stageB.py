# -*- coding: utf-8 -*-
"""Stage B: tax, idle-cash yield, market gate and invested fraction, at the plateau.

Four things stage A left unmeasured, all of which change the headline.

TAX. Every figure so far is pre-tax. Turnover at the plateau is much lower than at the live
trail (1,150 trades against 4,339) but the average hold is still ~105 days, so most gains
remain short-term at 20% rather than long-term at 12.5%.

IDLE CASH. The engine accrues nothing on cash by default and stage A never passed a yield.
The live book sweeps to CASHIETF at 5-6%. Every number so far therefore UNDERSTATES the
book by whatever fraction of it sits idle - which nothing currently reports, so this also
adds the invested fraction to the output.

THE GATE. research/142 tested a 200-day market gate and RETIRED it. That decision was taken
on the look-ahead entry, where every trade began in profit by construction and a gate could
only ever cost money. On an honest entry, where half the trades fail, a gate that keeps the
book out of falling markets has real work to do. The trail inverted when the entry was
fixed; the gate plausibly does too. Tested at none / 100-day / 200-day.

Grid: trail {50,75,100} x entry {close_same, stop_above_candle} x gate {none,100,200}
      x tax {off,on}, all with cash yield at 5%.
= 36 cells, disclosed. The plateau, not the peak, is what gets read.
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

OUT = ROOT / 'research/159_oa_honest_reoptimization/results/stageB.csv'
WIN = ('2006-01-01', '2026-08-31')
SEEDS = list(range(1, 31))
COST, SLOTS, SIZE, STOP = 0.0025, 16, 0.0625, 0.08
CASH_Y = 0.05
TRAILS = [50, 75, 100]
ENTRIES = ['close_same', 'stop_above_candle']
GATES = ['none', 'sma100', 'sma200']
FIELDS = ['cell', 'entry', 'trail', 'gate', 'tax', 'cagr_med', 'cagr_min', 'cagr_max',
          'dd_med', 'calmar_med', 'trades_med', 'invested_pct', 'secs']


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


def gate_arr(w, kind):
    """True on days the book must stand aside. NaN-robust: computed on the traded series
    then reindexed - a phantom holiday row silently disabled r/142's gate for months."""
    n = len(w['close'].index)
    if kind == 'none':
        return np.zeros(n, bool)
    nb = w['close'].get('NIFTYBEES')
    if nb is None:
        raise SystemExit('NIFTYBEES missing; cannot build the gate')
    L = int(kind[3:])
    s = nb.dropna()
    weak = (s < s.rolling(L, min_periods=L).mean()).shift(1)
    return (weak.reindex(w['close'].index).ffill().fillna(False)
            .astype(bool).to_numpy())


def done():
    if not OUT.exists():
        return set()
    with open(OUT) as f:
        return {r['cell'] for r in csv.DictReader(f)}


def main():
    have = done()
    if not OUT.exists():
        with open(OUT, 'w', newline='') as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()
    for trail in TRAILS:
        todo = [(e, g, tx) for e in ENTRIES for g in GATES for tx in (0, 1)
                if 'b|%s|%d|%s|%d' % (e, trail, g, tx) not in have]
        if not todo:
            continue
        print('trail-%d: %d cells' % (trail, len(todo)), flush=True)
        w = em.load_frames('2005-01-01', trail_sma=trail)
        close, sma, dates = w['close'], w['sma50'], w['close'].index
        days = np.array([i for i, d in enumerate(dates)
                         if WIN[0] <= str(d.date()) <= WIN[1]])
        built, gates = {}, {}
        for entry, gate, tx in todo:
            if entry not in built:
                built[entry] = build(w, entry)
            if gate not in gates:
                gates[gate] = gate_arr(w, gate)
            trig, piv, rs, tvp = built[entry]
            t0, stats, inv = time.time(), [], []
            for sd in SEEDS:
                eq, trd, _, ifrac = em.simulate(
                    sd, 'random', days, dates, close.values, w['high'].values,
                    w['open'].values, piv.values, sma.values, rs.values, tvp.values,
                    trig.fillna(False).values, gates[gate], True, COST, stop=STOP,
                    slots=SLOTS, size_pct=SIZE, fill_close=(entry == 'close_same'),
                    cash_yield=CASH_Y,
                    stcg=(0.20 if tx else 0.0), ltcg=(0.125 if tx else 0.125))
                st, e = em.stats_from(eq, dates[days], trd, em.CAPITAL)
                stats.append(st)
                inv.append(100.0 * ifrac)
            d = pd.DataFrame(stats)
            row = dict(cell='b|%s|%d|%s|%d' % (entry, trail, gate, tx), entry=entry,
                       trail=trail, gate=gate, tax=('after-tax' if tx else 'pre-tax'),
                       cagr_med=round(d.cagr.median(), 2), cagr_min=round(d.cagr.min(), 2),
                       cagr_max=round(d.cagr.max(), 2), dd_med=round(d.dd.median(), 2),
                       calmar_med=round((d.cagr / d.dd.abs()).median(), 3),
                       trades_med=int(d.n.median()),
                       invested_pct=round(float(np.mean(inv)), 1),
                       secs=round(time.time() - t0))
            with open(OUT, 'a', newline='') as f:
                csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
            print('  %-18s t%-4d %-7s %-10s CAGR %6.2f [%6.2f..%6.2f] DD %6.2f Cal %5.3f '
                  'n%5d inv%4.0f%% (%ds)'
                  % (entry, trail, gate, row['tax'], row['cagr_med'], row['cagr_min'],
                     row['cagr_max'], row['dd_med'], row['calmar_med'],
                     row['trades_med'], row['invested_pct'], row['secs']), flush=True)
    print('STAGE_B_DONE', flush=True)


if __name__ == '__main__':
    main()
