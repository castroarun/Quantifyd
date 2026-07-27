"""
Phase-2 stage 1 (research/93): per-trade exit variants. Same entries (triple-aligned
signal), same initial SL; compare exits:
  target       : 26w swing-high target (phase-1 base)
  trail_hma44  : no target; exit at weekly close < HMA44 (ride the trend)
  trail_donch10: no target; exit at weekly close < prior-10wk low (Donchian trail)
Initial SL active throughout; 52w time-stop; 25 bps. Saves one trade CSV per mode
(consumed by run_g7_portfolio_opt.py) + pooled summary.
"""
import os, sys, csv, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logging; logging.disable(logging.WARNING)
import numpy as np
from collections import Counter
from hma_weekly_engine import (list_universe, load_daily, to_weekly, add_indicators,
                               find_signals, confirmed_pivot_low, summarize)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, '..', 'results'))
NEG, RWIN, TLB, MAXH, COST = 8, 4, 26, 52, 25.0
MODES = ('target', 'trail_hma44', 'trail_donch10')
FIELDS = ['symbol', 'entry_date', 'exit_date', 'entry', 'exit', 'stop', 'target',
          'reason', 'hold_wks', 'gross_pct', 'net_pct', 'risk_pct']


def sim(w, i, mode):
    o = w['open'].values; h = w['high'].values; l = w['low'].values; c = w['close'].values
    hs = w['hma_s'].values
    dates = w['date'].values
    n = len(w)
    if i + 1 >= n:
        return None
    ent = i + 1
    entry = o[ent]
    pj = confirmed_pivot_low(l, i)
    stop = l[pj] * 0.999 if pj is not None else None
    if stop is None or stop >= entry:
        stop = l[max(0, i - 9):i + 1].min() * 0.999
    if stop >= entry:
        return None
    tgt = h[max(0, i - TLB + 1):i + 1].max()
    if mode == 'target' and tgt <= entry:
        return None
    exit_price = None; exj = None; reason = None
    for k in range(ent, n):
        if l[k] <= stop:
            exit_price = min(o[k], stop); exj = k; reason = 'sl'; break
        if mode == 'target' and h[k] >= tgt:
            exit_price = max(o[k], tgt); exj = k; reason = 'target'; break
        if mode == 'trail_hma44' and k > ent and not np.isnan(hs[k]) and c[k] < hs[k]:
            exit_price = c[k]; exj = k; reason = 'trail'; break
        if mode == 'trail_donch10' and k > ent + 1:
            dl = l[max(0, k - 10):k].min()
            if c[k] < dl:
                exit_price = c[k]; exj = k; reason = 'trail'; break
        if (k - ent) >= MAXH:
            exit_price = c[k]; exj = k; reason = 'time'; break
    if exit_price is None:
        exj = n - 1; exit_price = c[exj]; reason = 'end'
    gross = (exit_price - entry) / entry
    risk = (entry - stop) / entry
    return dict(entry_date=str(dates[ent])[:10], exit_date=str(dates[exj])[:10],
                entry=round(float(entry), 3), exit=round(float(exit_price), 3),
                stop=round(float(stop), 3), target=round(float(tgt), 3),
                reason=reason, hold_wks=int(exj - ent),
                gross_pct=round(gross * 100, 4),
                net_pct=round((gross - COST / 1e4) * 100, 4),
                risk_pct=round(risk * 100, 4), exj=exj)


syms, _ = list_universe()
writers = {}
files = {}
for m in MODES:
    p = os.path.join(RESULTS, f'g6_trades_{m}.csv')
    files[m] = open(p, 'w', newline='')
    writers[m] = csv.DictWriter(files[m], fieldnames=FIELDS)
    writers[m].writeheader()

pooled = {m: [] for m in MODES}
t0 = time.time(); n_done = 0
for k, sym in enumerate(syms, 1):
    try:
        df = load_daily(sym)
        if df.empty:
            continue
        w = to_weekly(df)
        if len(w) < 200 or df['close'].median() < 20 or \
           (df['close'] * df['volume']).median() < 1e7:
            continue
        w = add_indicators(w)
        sigb = find_signals(w, neg_bars=NEG, recent_win=RWIN)
        for m in MODES:
            i = 0
            while i < len(w) - 1:
                if not sigb[i]:
                    i += 1; continue
                t = sim(w, i, m)
                if t is None:
                    i += 1; continue
                exj = t.pop('exj')
                writers[m].writerow({'symbol': sym, **t})
                t['R'] = None
                pooled[m].append(t)
                i = exj + 1
    except Exception as e:
        print(f'[{k}] {sym} ERROR {e}', flush=True)
        continue
    n_done += 1
    if n_done % 150 == 0:
        print(f'[{k}/{len(syms)}] {n_done} done, {time.time()-t0:.0f}s', flush=True)
        for f in files.values():
            f.flush()
for f in files.values():
    f.close()

print('\n=== PER-TRADE EXIT VARIANTS (net 25bps) ===', flush=True)
for m in MODES:
    s = summarize(pooled[m])
    rc = Counter(t['reason'] for t in pooled[m]); n = max(1, s['n'])
    print(f"{m:14s} n={s['n']:5d} gross={s['gross_exp_pct']:6.3f}% net={s['net_exp_pct']:6.3f}% "
          f"win={s['win_pct']:4.1f}% PF={s['pf']:4.2f} t={s['t_stat']:5.1f} hold={s['avg_hold_wks']:4.1f}wk"
          f" | sl {100*rc['sl']/n:4.1f}% trail/tgt {100*(rc['trail']+rc['target'])/n:4.1f}% "
          f"time {100*rc['time']/n:4.1f}%", flush=True)
print('\ntrade lists saved to results/g6_trades_<mode>.csv', flush=True)
