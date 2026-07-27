"""
Target-rule variants (research/93 follow-up from Arun's ASIANPAINT chart review):
the base target (26-wk max high, exact-touch fill) missed the Jul-2023 peak by Rs15
and turned a +27% winner into a +1.3% time-stop. Quantify across the full universe:

  A base     : target = 26w max high, fill at 100% of target (as shipped)
  B buffer99 : same target level, but exit fills when high >= 99% of target
  C fractal  : target = last CONFIRMED 2/2 fractal swing high before signal
               (fallback = 26w max) — the 'chart-reader' swing high

Same entries, same stops, same 52w time stop, 25 bps. Pooled per-trade stats per mode.
"""
import os, sys, csv, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import logging; logging.disable(logging.WARNING)
import numpy as np
from hma_weekly_engine import (list_universe, load_daily, to_weekly, add_indicators,
                               find_signals, confirmed_pivot_low, summarize)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, '..', 'results'))
OUT = os.path.join(RESULTS, 'g5_target_variants.csv')

NEG, RWIN, TLB, MAXH, COST = 8, 4, 26, 52, 25.0


def confirmed_pivot_high(h, i, wing=2):
    for j in range(i - wing, wing - 1, -1):
        hi = h[j]
        if all(hi > h[j - k] for k in range(1, wing + 1)) and \
           all(hi > h[j + k] for k in range(1, wing + 1)):
            return j
    return None


def sim(w, i, mode):
    """Entry at open[i+1]; returns trade dict or None. Signal bar = i."""
    o = w['open'].values; h = w['high'].values; l = w['low'].values; c = w['close'].values
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
    if mode == 'fractal':
        fj = confirmed_pivot_high(h, i)
        if fj is not None and h[fj] > entry:
            tgt = h[fj]
    if tgt <= entry:
        return None
    fill_lvl = tgt * 0.99 if mode == 'buffer99' else tgt
    exit_price = None; exj = None; reason = None
    for k in range(ent, n):
        if l[k] <= stop:
            exit_price = min(o[k], stop); exj = k; reason = 'sl'; break
        if h[k] >= fill_lvl:
            exit_price = max(o[k], fill_lvl); exj = k; reason = 'target'; break
        if (k - ent) >= MAXH:
            exit_price = c[k]; exj = k; reason = 'time'; break
    if exit_price is None:
        exj = n - 1; exit_price = c[exj]; reason = 'end'
    gross = (exit_price - entry) / entry
    return dict(net_pct=(gross - COST / 1e4) * 100, gross_pct=gross * 100,
                hold_wks=exj - ent, R=None, reason=reason, exj=exj)


MODES = ('base', 'buffer99', 'fractal')
syms, _ = list_universe()
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
                pooled[m].append(t)
                i = t['exj'] + 1
    except Exception as e:
        print(f'[{k}] {sym} ERROR {e}', flush=True)
        continue
    n_done += 1
    if n_done % 150 == 0:
        print(f'[{k}/{len(syms)}] {n_done} done, {time.time()-t0:.0f}s', flush=True)

print('\n=== TARGET-RULE VARIANTS (pooled, net 25bps) ===', flush=True)
with open(OUT, 'w', newline='') as f:
    wtr = csv.writer(f)
    wtr.writerow(['mode', 'n', 'gross_exp', 'net_exp', 'win_pct', 'pf', 't', 'avg_hold',
                  'pct_target', 'pct_sl', 'pct_time'])
    for m in MODES:
        s = summarize(pooled[m])
        from collections import Counter
        rc = Counter(t['reason'] for t in pooled[m])
        n = max(1, s['n'])
        row = [m, s['n'], s['gross_exp_pct'], s['net_exp_pct'], s['win_pct'], s['pf'],
               s['t_stat'], s['avg_hold_wks'],
               round(100 * rc['target'] / n, 1), round(100 * rc['sl'] / n, 1),
               round(100 * rc['time'] / n, 1)]
        wtr.writerow(row)
        print(f"{m:9s} n={s['n']:5d} gross={s['gross_exp_pct']:6.3f}% net={s['net_exp_pct']:6.3f}% "
              f"win={s['win_pct']:4.1f}% PF={s['pf']:4.2f} t={s['t_stat']:5.1f} "
              f"hold={s['avg_hold_wks']:4.1f}wk | target {100*rc['target']/n:4.1f}% "
              f"sl {100*rc['sl']/n:4.1f}% time {100*rc['time']/n:4.1f}%", flush=True)
print('\nsaved:', OUT, flush=True)
