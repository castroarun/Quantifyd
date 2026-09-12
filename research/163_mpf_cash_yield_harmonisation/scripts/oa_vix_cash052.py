# -*- coding: utf-8 -*-
"""research/163 — Open Alpha · ATH + VIX re-run at 5.2% idle cash.

This is the variant the Momentum Portfolio report prints as ONE summary row inside the Open
Alpha section (19.23% / -34.15% / 0.56 at 5.0%), sourced from research/159's
`all_systems_summary.json`. Its curve is research/159 `scripts/build_curves.py` ->
`oa_curve(w, gated=True, win=('2016-01-01','2026-08-31'))`:

    buy at the breakout close, 75-day SMA trail, 8% stop (inert but kept), 16 slots at
    6.25% of NAV, funds excluded, traded-value floor, RS >= 70, 25 bps a side, after tax,
    INDIA VIX above its own 252-day 70th percentile blocks new entries, 30 seeds,
    drawn path = the MEDIAN-CAGR seed.

Only `cash_yield` moves, 0.05 -> 0.052 (the arbitrage-fund rate after 20% short-term tax).

The published row is measured on `compare_all.py`'s ALIGNED five-series frame, not on the
raw curve, so the alignment is reproduced exactly by reindexing onto the index of
research/159's own `all_systems_after_tax.csv` — the file that frame was saved as.

TWO REPRODUCTION GATES:
  1. the 0.05 curve must reproduce `curves_after_tax.csv['OA gated (VIX p70)']`;
  2. re-measuring that reproduced curve on the aligned index must return research/159's
     published 19.23 / -34.15 / 0.56.

Nothing in research/158 or research/159 is written to.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
sys.path.insert(0, str(ROOT / 'research/158_oa_arming_width/scripts'))
sys.path.insert(0, str(ROOT))
import oa_entry_mechanics as em          # noqa: E402

R159 = ROOT / 'research/159_oa_honest_reoptimization/results'
OUT = ROOT / 'research/163_mpf_cash_yield_harmonisation/results/cash052'
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = list(range(1, 31))
COST, SLOTS, SIZE, STOP, TRAIL = 0.0025, 16, 0.0625, 0.08, 75
STCG, LTCG = 0.20, 0.125
WIN = ('2016-01-01', '2026-08-31')
OLD_Y, NEW_Y = 0.05, 0.052
PUBLISHED = dict(cagr=19.23, maxdd=-34.15, calmar=0.56)


def vix_gate(index, pct=0.70):
    import sqlite3
    con = sqlite3.connect('file:%s?mode=ro' % (ROOT / 'backtest_data/market_data.db'),
                          uri=True)
    d = pd.read_sql_query("select date, close from market_data_unified where "
                          "symbol='INDIAVIX' and timeframe='day' order by date", con)
    con.close()
    d['date'] = pd.to_datetime(d['date'].str[:10])
    s = d.drop_duplicates('date').set_index('date')['close'].dropna()
    off = (s > s.rolling(252, min_periods=252).quantile(pct)).shift(1)
    return off.reindex(index).ffill().fillna(False).astype(bool).to_numpy()


def oa_curve(w, gated, win, cash_y):
    """research/159 build_curves.oa_curve, verbatim, with the yield as an argument and the
    mean invested fraction also returned (the 4th value em.simulate already hands back)."""
    close, high, athcp, tv20 = w['close'], w['high'], w['athcp'], w['tv20']
    etf = [c for c in close.columns if em.is_etf(c)]
    tvp, prev = tv20.shift(1), close.shift(1)
    elig = tvp >= em.TV_FLOOR
    elig[etf] = False
    r = {n: close / close.shift(n) - 1 for n in (63, 126, 189, 252)}
    rs = ((2 * r[63] + r[126] + r[189] + r[252]).where(elig)
          .rank(axis=1, pct=True) * 100).shift(1)
    setup = (prev < athcp) & (prev >= 0.8 * athcp) & elig & (rs >= 70.0)
    trig = setup & (close > athcp) & athcp.notna()
    dates = close.index
    days = np.array([i for i, d in enumerate(dates)
                     if win[0] <= str(d.date()) <= win[1]])
    garr = vix_gate(dates) if gated else np.zeros(len(dates), bool)
    curves, cagrs, invs = [], [], []
    for sd in SEEDS:
        eq, trd, _, inv = em.simulate(
            sd, 'random', days, dates, close.values, high.values, w['open'].values,
            athcp.values, w['sma50'].values, rs.values, tvp.values,
            trig.fillna(False).values, garr, True, COST, stop=STOP, slots=SLOTS,
            size_pct=SIZE, fill_close=True, cash_yield=cash_y, stcg=STCG, ltcg=LTCG)
        s = pd.Series(eq, index=dates[days])
        curves.append(s / s.iloc[0])
        st, _ = em.stats_from(eq, dates[days], trd, em.CAPITAL)
        cagrs.append(st['cagr'])
        invs.append(inv)
    k = int(np.argsort(cagrs)[len(cagrs) // 2])
    return curves, np.array(cagrs), np.array(invs), k


def aligned_stats(curve, idx):
    """Measure the way compare_all.py measures: reindex onto the aligned frame's own index,
    forward fill, rebase to 1.0 at its first row."""
    s = curve.reindex(idx.union(curve.index)).ffill().reindex(idx)
    s = s / s.iloc[0]
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    c = (s.iloc[-1] ** (1 / yrs) - 1) * 100
    d = (s / s.cummax() - 1).min() * 100
    return s, c, d, c / abs(d)


def main():
    t0 = time.time()
    print('loading frames (trail-%d) ...' % TRAIL, flush=True)
    w = em.load_frames('2005-01-01', trail_sma=TRAIL)
    print('frames loaded in %.0fs' % (time.time() - t0), flush=True)

    res = {}
    for y in (OLD_Y, NEW_Y):
        t1 = time.time()
        curves, cg, iv, k = oa_curve(w, True, WIN, y)
        res[y] = dict(curves=curves, cagrs=cg, invs=iv, med_i=k)
        print('idle %.1f%%  CAGR med %.2f%% [%.2f .. %.2f]  invested med %.2f%%  '
              'own median-CAGR seed %d  (%.0fs)'
              % (y * 100, np.median(cg), cg.min(), cg.max(), 100 * np.median(iv), SEEDS[k],
                 time.time() - t1), flush=True)

    # THE DRAWN PATH IS HELD AT THE SEED THE 5.0% PAGE DREW — same reasoning as Base Age:
    # the per-seed spread of this book is several points wide, so re-picking "this run's
    # median-CAGR seed" would publish path noise as a cash-rate result.
    frozen_i = res[OLD_Y]['med_i']
    print('\nDRAWN PATH: seed %d, frozen from the 5.0%% run (the 5.2%% run\'s own '
          'median-CAGR seed would have been %d)'
          % (SEEDS[frozen_i], SEEDS[res[NEW_Y]['med_i']]), flush=True)

    # ---------------- gate 1: the raw curve ------------------------------------------
    print('\n--- gate 1: reproduce curves_after_tax.csv[OA gated (VIX p70)] at 5.0%% ---',
          flush=True)
    # build_curves.py saved that column on the UNION index of four series and forward filled
    # it, so the published column has more rows than the book has trading days. The proof is
    # therefore run on this curve's own index, after asserting it is a subset of the
    # published one.
    pub = pd.read_csv(R159 / 'curves_after_tax.csv', index_col=0,
                      parse_dates=True)['OA gated (VIX p70)'].dropna()
    mine = res[OLD_Y]['curves'][frozen_i]
    subset = bool(mine.index.isin(pub.index).all())
    rel = (mine - pub.reindex(mine.index)).abs() / pub.reindex(mine.index).abs()
    bad = rel[rel > 1e-12]
    print('index is a subset   : %s (%d rows of the published %d; the published column is '
          'ffilled onto a union index)' % (subset, len(mine), len(pub)))
    print('rows differing      : %d of %d   max rel %.3e' % (len(bad), len(rel), rel.max()))
    if not subset:
        print('!! REPRODUCTION FAILED (index not a subset) — stopping.')
        sys.exit(2)
    if len(bad):
        print('first divergence    : %s (rel %.2e)' % (bad.index[0].date(), bad.iloc[0]))
        if bad.index[0] < pd.Timestamp('2026-08-01'):
            print('!! divergence starts too early to be a market_data.db refresh — stopping.')
            sys.exit(2)
    print('GATE 1 PASS — exact over %.2f%% of the history.'
          % (100.0 * (len(rel) - len(bad)) / len(rel)))

    # ---------------- gate 2: the published summary row -------------------------------
    idx = pd.read_csv(R159 / 'all_systems_after_tax.csv', index_col=0,
                      parse_dates=True).index
    print('\n--- gate 2: re-measure on compare_all.py\'s aligned index %s .. %s ---'
          % (idx[0].date(), idx[-1].date()), flush=True)
    _, c50, d50, k50 = aligned_stats(mine, idx)
    print('published  CAGR %.2f%%  MaxDD %.2f%%  Calmar %.2f'
          % (PUBLISHED['cagr'], PUBLISHED['maxdd'], PUBLISHED['calmar']))
    print('reproduced CAGR %.2f%%  MaxDD %.2f%%  Calmar %.2f' % (c50, d50, k50))
    ok2 = (abs(round(c50, 2) - PUBLISHED['cagr']) <= 0.01
           and abs(round(d50, 2) - PUBLISHED['maxdd']) <= 0.01
           and abs(round(k50, 2) - PUBLISHED['calmar']) <= 0.01)
    print('GATE 2 %s' % ('PASS' if ok2 else 'FAIL — investigate before publishing'))
    if not ok2:
        sys.exit(2)

    # ---------------- outputs ----------------------------------------------------------
    new_curve = res[NEW_Y]['curves'][frozen_i]
    new_aligned, c52, d52, k52 = aligned_stats(new_curve, idx)
    mine.to_csv(OUT / 'oa_gated_cash05_reproduced.csv', header=['nav'])
    new_curve.to_csv(OUT / 'oa_gated_cash052.csv', header=['nav'])
    new_aligned.to_csv(OUT / 'oa_gated_cash052_aligned.csv', header=['nav'])

    # ---- the ensemble, measured the SAME way the row is, so the re-draw noise is visible.
    # This book's path is unusually sensitive to the yield: a 20 bps change moves integer
    # share counts, which changes whether a buy is affordable, which changes the RNG
    # consumption order, which re-draws every later selection. The PAIRED per-seed spread is
    # several points wide, so one path's movement is NOT a cash-rate result and the row must
    # carry the band.
    band = {}
    for y in (OLD_Y, NEW_Y):
        cc, dd, kk = [], [], []
        for cur in res[y]['curves']:
            _, c_, d_, k_ = aligned_stats(cur, idx)
            cc.append(c_); dd.append(d_); kk.append(k_)
        band[y] = dict(cagr_med=round(float(np.median(cc)), 2),
                       cagr_min=round(float(np.min(cc)), 2),
                       cagr_max=round(float(np.max(cc)), 2),
                       maxdd_med=round(float(np.median(dd)), 2),
                       calmar_med=round(float(np.median(kk)), 2),
                       cagr_all=[round(float(v), 3) for v in cc])
        print('aligned 30-seed at %.1f%%: CAGR median %.2f%% [%.2f .. %.2f]  '
              'MaxDD median %.2f%%  Calmar median %.2f'
              % (y * 100, band[y]['cagr_med'], band[y]['cagr_min'], band[y]['cagr_max'],
                 band[y]['maxdd_med'], band[y]['calmar_med']))
    pa = np.array(band[NEW_Y]['cagr_all']) - np.array(band[OLD_Y]['cagr_all'])
    print('aligned PAIRED per-seed CAGR delta: median %+.3f pp, [%+.2f .. %+.2f]; the drawn '
          'seed %d alone moved %+.3f pp — that is the RE-DRAW, not the cash rate.'
          % (np.median(pa), pa.min(), pa.max(), SEEDS[frozen_i], pa[frozen_i]))

    inv = float(np.median(res[OLD_Y]['invs'])) * 100
    rot = (1 - inv / 100.0) * 0.2
    paired = res[NEW_Y]['cagrs'] - res[OLD_Y]['cagrs']
    print('\nALIGNED-ROW YIELD EFFECT 5.0%% -> 5.2%%: CAGR %.2f -> %.2f (%+.3f pp)  '
          'MaxDD %.2f -> %.2f  Calmar %.2f -> %.2f' % (c50, c52, c52 - c50, d50, d52, k50, k52))
    print('CONSISTENCY: paired per-seed CAGR delta median %+.3f pp [%+.3f .. %+.3f]; '
          'rule of thumb at inv %.1f%% = %+.3f pp'
          % (np.median(paired), paired.min(), paired.max(), inv, rot))
    json.dump(dict(
        label='Open Alpha · ATH + VIX (research/159)',
        window='%s to %s' % (idx[0].date(), idx[-1].date()),
        cagr_050=round(c50, 2), maxdd_050=round(d50, 2), calmar_050=round(k50, 2),
        cagr_052=round(c52, 2), maxdd_052=round(d52, 2), calmar_052=round(k52, 2),
        seed_median_cagr_050=round(float(np.median(res[OLD_Y]['cagrs'])), 2),
        seed_median_cagr_052=round(float(np.median(res[NEW_Y]['cagrs'])), 2),
        drawn_seed=SEEDS[frozen_i], drawn_seed_frozen_from_050=True,
        own_median_seed_052=SEEDS[res[NEW_Y]['med_i']],
        invested_median_pct=round(inv, 2),
        cagr_delta_paired_med_pp=round(float(np.median(paired)), 3),
        cagr_delta_rule_of_thumb_pp=round(rot, 3),
        gate1_rows_differing=int(len(bad)), gate2_pass=bool(ok2),
        aligned_band_050=band[OLD_Y], aligned_band_052=band[NEW_Y],
        aligned_paired_delta_med_pp=round(float(np.median(pa)), 3),
        aligned_paired_delta_min_pp=round(float(pa.min()), 3),
        aligned_paired_delta_max_pp=round(float(pa.max()), 3),
        drawn_seed_redraw_pp=round(float(pa[frozen_i]), 3),
    ), open(OUT / 'athvix_summary_cash052.json', 'w'), indent=1)
    print('\nwrote %s   (%.0fs)' % (OUT / 'athvix_summary_cash052.json', time.time() - t0))


if __name__ == '__main__':
    main()
