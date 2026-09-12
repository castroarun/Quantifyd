# -*- coding: utf-8 -*-
"""research/163 — IPO Base (HONEST next-day entry) re-run at 5.2% idle cash.

research/159's `scripts/ipo_curve.py` built the curve the Momentum Portfolio report reads
(`research/159 .../ipo_honest_curve.csv`) at 5.0%. This is that script, verbatim in its
mechanics, with the idle-cash yield parameterised — 5.2% being the ARBITRAGE-FUND rate after
20% short-term tax at 2025-26 cash-futures spreads.

The spec, the seeds (1..30), the window (W2 = 2006-01-01 .. 2026-09-04), the engine
(research/153 `ipo_replay.simulate_ipo` via `ipo_g3.run`) and the next-day entry transform

    TRIG[i] <- TRIG[i-1]  AND  high[i] >= PIV[i-1]
    PIV[i]  <- PIV[i-1]

are all unchanged. The drawn path is the MEDIAN-CAGR seed, never the mean of thirty paths.

STEP 1 is a REPRODUCTION GATE against research/159's own 5.0% file.
Nothing in research/153 or research/159 is written to.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
sys.path.insert(0, str(ROOT / 'research/153_ipo_base/scripts'))
sys.path.insert(0, str(ROOT))
import ipo_replay as ir          # noqa: E402
import ipo_g3 as g3              # noqa: E402

R159 = ROOT / 'research/159_oa_honest_reoptimization/results'
OUT = ROOT / 'research/163_mpf_cash_yield_harmonisation/results/cash052'
OUT.mkdir(parents=True, exist_ok=True)
SPEC = json.load(open(ROOT / 'research/153_ipo_base/results/ipo_adopted_spec.json'))
SEEDS = list(range(1, 31))
OLD_Y, NEW_Y = 0.05, 0.052


def drawn(navs, stats):
    cagrs = stats['cagr'].tolist()
    k = int(np.argsort(cagrs)[len(cagrs) // 2])
    s = navs[k]
    return (s / s.iloc[0]), k


def main():
    t0 = time.time()
    print('building the IPO panel ...', flush=True)
    ctx = ir.Ctx()
    trig, piv, lo, sma = g3.build(ctx, SPEC)

    pivn = np.full_like(piv, np.nan)
    pivn[1:] = piv[:-1]
    tn = np.zeros_like(trig)
    tn[1:] = trig[:-1]
    with np.errstate(invalid='ignore'):
        reached = ctx.H >= pivn
    trign = tn & reached & np.isfinite(pivn)
    lon = np.full_like(lo, np.nan)
    lon[1:] = lo[:-1]
    print('same-day signals %d -> next-day reachable %d (%.1f%%)'
          % (int(trig.sum()), int(trign.sum()), 100.0 * trign.sum() / max(trig.sum(), 1)),
          flush=True)

    out = {}
    for y in (OLD_Y, NEW_Y):
        navs, _t, st = g3.run(ctx, SPEC, SEEDS, g3.W2, trign, pivn, lon, sma, cash_yield=y)
        s, k = drawn(navs, st)
        out[y] = dict(curve=s, stats=st, seed=SEEDS[k])
        print('idle %.1f%%  CAGR med %.2f%% [%.2f .. %.2f]  DD med %.2f%%  Calmar %.3f  '
              'invested med %.2f%%  drawn seed %d'
              % (y * 100, st.cagr.median(), st.cagr.min(), st.cagr.max(), st.dd.median(),
                 (st.cagr / st.dd.abs()).median(), st.invested_pct.median(), SEEDS[k]),
              flush=True)

    # ---------------- reproduction gate -----------------------------------------------
    print('\n--- reproduction gate at 5.0%% vs research/159 ipo_honest_curve.csv ---',
          flush=True)
    pub = pd.read_csv(R159 / 'ipo_honest_curve.csv', index_col=0,
                      parse_dates=True).iloc[:, 0].dropna()
    mine = out[OLD_Y]['curve']
    same_index = bool(mine.index.equals(pub.index))
    rel = (mine - pub.reindex(mine.index)).abs() / pub.reindex(mine.index).abs()
    bad = rel[rel > 1e-12]
    print('index identical     : %s (%d rows vs %d)' % (same_index, len(mine), len(pub)))
    print('rows differing      : %d of %d' % (len(bad), len(rel)))
    print('max rel difference  : %.3e' % rel.max())
    if not same_index:
        print('!! REPRODUCTION FAILED (index) — stopping.')
        sys.exit(2)
    if len(bad):
        print('first divergence    : %s (rel %.2e)' % (bad.index[0].date(), bad.iloc[0]))
        if bad.index[0] < pd.Timestamp('2026-08-01'):
            print('!! divergence starts too early to be a market_data.db refresh — stopping.')
            sys.exit(2)
    print('REPRODUCTION EXACT over %.2f%% of the history.'
          % (100.0 * (len(rel) - len(bad)) / len(rel)))
    mine.to_csv(OUT / 'ipo_honest_curve_cash05_reproduced.csv', header=['nav'])

    # ---------------- outputs ----------------------------------------------------------
    new = out[NEW_Y]['curve']
    new.to_csv(OUT / 'ipo_honest_curve_cash052.csv', header=['nav'])
    for y, tag in ((OLD_Y, '050'), (NEW_Y, '052')):
        out[y]['stats'].to_csv(OUT / ('ipo_seed_stats_%s.csv' % tag), index=False)

    def m(s):
        yrs = (s.index[-1] - s.index[0]).days / 365.25
        c = (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1
        d = (s / s.cummax() - 1).min()
        return c * 100, d * 100, c / abs(d)

    c50, d50, k50 = m(out[OLD_Y]['curve'])
    c52, d52, k52 = m(new)
    piv_ = pd.DataFrame({0.05: out[OLD_Y]['stats'].cagr.to_numpy(),
                         0.052: out[NEW_Y]['stats'].cagr.to_numpy()})
    paired = (piv_[0.052] - piv_[0.05])
    inv = float(out[OLD_Y]['stats'].invested_pct.median())
    rot = (1 - inv / 100.0) * 0.2
    print('\nDRAWN-PATH YIELD EFFECT 5.0%% -> 5.2%%: CAGR %+.3f pp  MaxDD %+.3f pp  '
          'Calmar %+.4f' % (c52 - c50, d52 - d50, k52 - k50))
    print('CONSISTENCY: paired per-seed CAGR delta median %+.3f pp [%+.3f .. %+.3f]; '
          'rule of thumb at inv %.1f%% = %+.3f pp'
          % (paired.median(), paired.min(), paired.max(), inv, rot))
    json.dump(dict(
        drawn_seed_050=out[OLD_Y]['seed'], drawn_seed_052=out[NEW_Y]['seed'],
        cagr_med_050=round(float(out[OLD_Y]['stats'].cagr.median()), 2),
        cagr_med_052=round(float(out[NEW_Y]['stats'].cagr.median()), 2),
        dd_med_050=round(float(out[OLD_Y]['stats'].dd.median()), 2),
        dd_med_052=round(float(out[NEW_Y]['stats'].dd.median()), 2),
        invested_median_pct=round(inv, 2),
        drawn_cagr_050=round(c50, 3), drawn_cagr_052=round(c52, 3),
        drawn_dd_050=round(d50, 3), drawn_dd_052=round(d52, 3),
        cagr_delta_paired_med_pp=round(float(paired.median()), 3),
        cagr_delta_rule_of_thumb_pp=round(rot, 3),
        reproduction_rows_differing=int(len(bad)),
        reproduction_max_rel=float(rel.max()),
    ), open(OUT / 'ipo_cash052_summary.json', 'w'), indent=1)
    print('\nwrote %s   (%.0fs)' % (OUT / 'ipo_honest_curve_cash052.csv', time.time() - t0))


if __name__ == '__main__':
    main()
