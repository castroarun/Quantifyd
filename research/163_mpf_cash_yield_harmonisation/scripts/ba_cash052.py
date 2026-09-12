# -*- coding: utf-8 -*-
"""research/163 — Open Alpha · Base Age re-run at 5.2% idle cash (the page has carried 5.0%
since 12-Sep-2026; research/161's own study used 5.5%).

5.2% is the ARBITRAGE-FUND rate after 20% short-term tax at 2025-26 cash-futures spreads.

The engine is NOT re-copied here: `simulate_inv` and `rearm` are imported from this folder's
own `ba_cash05.py`, which is research/161's `bt_core.simulate` verbatim plus a daily
invested-fraction array. One copy of the engine, one place to audit.

STEP 1 is a REPRODUCTION GATE: the 0.05 run must reproduce, BIT-EXACTLY,
    results/ba_nav_winner_cash05.csv  and  results/ba_navs_30seed_cash05.npz
(both of which are themselves proven bit-exact against research/161's published WINNER at
5.5%). Only then is the yield moved to 0.052.

Nothing in research/161 is written to. Outputs land in results/cash052/.
"""
import json
import pickle
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ba_cash05 as BA5                   # noqa: E402  (the engine copy + rearm live here)
import bt_core as B                       # noqa: E402  (put on the path by ba_cash05)

ROOT = Path('/home/arun/quantifyd')
R161 = ROOT / 'research/161_ath_base_age_breakout/results'
R163 = ROOT / 'research/163_mpf_cash_yield_harmonisation/results'
OUT = R163 / 'cash052'
OUT.mkdir(parents=True, exist_ok=True)
DB = ROOT / 'backtest_data/market_data.db'
PANEL = R163 / 'panel163.pkl'

SEEDS = list(range(1, 31))
CFG = dict(X=60, dep=20.0, K=0.0, sau=0, ex='ST_14_4', hard=False, liq=2.0)
OLD_Y, NEW_Y = 0.05, 0.052


def main():
    t0 = time.time()
    con = sqlite3.connect('file:%s?mode=ro' % DB, uri=True)
    cal = [d for d in pd.read_sql_query(
        "SELECT DISTINCT date FROM market_data_unified WHERE timeframe='day' "
        "AND symbol='NIFTYBEES' ORDER BY date", con)['date'].tolist() if d >= '2005-01-03']
    zz = np.load(R161 / 'curves161.npz', allow_pickle=True)
    CUT = str(zz['dates'][-1])[:10]
    cal = [d for d in cal if d <= CUT]
    print('calendar cut at research/161 last date %s -> %d days (%s .. %s)'
          % (CUT, len(cal), cal[0], cal[-1]), flush=True)
    raw = pd.read_csv(R161 / 'ath_events.csv')
    raw_liq = raw[raw['tv20_cr'] >= CFG['liq']]

    if PANEL.exists():
        panel = pickle.load(open(PANEL, 'rb'))
        print('panel cache: %d symbols' % len(panel.close), flush=True)
    else:
        print('building panel for %d symbols ...' % raw_liq['symbol'].nunique(), flush=True)
        panel = B.Panel(con, sorted(raw_liq['symbol'].unique()), cal)
        pickle.dump(panel, open(PANEL, 'wb'), protocol=4)
        print('panel built in %.0fs' % (time.time() - t0), flush=True)
    con.close()

    ev_df = raw[raw['symbol'].isin(panel.close) & raw['entry_date'].isin(panel.pos)].copy()
    ev_df['entry_i'] = ev_df['entry_date'].map(panel.pos)
    s = ev_df[(ev_df['tv20_cr'] >= CFG['liq']) & (ev_df['x_bars'] >= CFG['X'])
              & (ev_df['depth_pct'] >= CFG['dep'])]
    s = BA5.rearm(s)
    events = [dict(symbol=a, entry_i=int(b)) for a, b in zip(s['symbol'], s['entry_i'])]
    print('WINNER events: %d' % len(events), flush=True)

    cal_idx = pd.to_datetime(cal)
    rows, store = [], {}
    for iy in (OLD_Y, NEW_Y):
        cfg = dict(exit=CFG['ex'], hard_stop=CFG['hard'], time_stop=0, cost_bps=25.0,
                   gate_ok=None, idle_yield=iy)
        navs, invs, ms = [], [], []
        for sd in SEEDS:
            nav, tr, inv = BA5.simulate_inv(events, panel, cfg, sd)
            m = B.metrics(nav, cal, tr)
            navs.append(nav); invs.append(inv); ms.append(m)
            rows.append(dict(idle_yield=iy, seed=sd, cagr=m['cagr'], maxdd=m['maxdd'],
                             calmar=m['calmar'], sharpe=m.get('sharpe'),
                             trades=m.get('trades'), win_rate=m.get('win_rate'),
                             invested_pct=round(float(np.nanmean(inv)) * 100, 2)))
        cg = np.array([m['cagr'] for m in ms])
        med_i = int(np.argsort(cg)[len(cg) // 2])
        store[iy] = dict(navs=navs, invs=invs, ms=ms, med_i=med_i)
        print('\nidle %.1f%%  CAGR med %.2f%% (worst %.2f%%, best %.2f%%)  MaxDD med %.2f%%  '
              'Calmar med %.3f  invested med %.2f%% [%.2f .. %.2f]  drawn seed %d'
              % (iy * 100, float(np.median(cg)), cg.min(), cg.max(),
                 float(np.median([m['maxdd'] for m in ms])),
                 float(np.median([m['calmar'] for m in ms])),
                 100 * float(np.median([np.nanmean(v) for v in invs])),
                 100 * min(np.nanmean(v) for v in invs),
                 100 * max(np.nanmean(v) for v in invs), SEEDS[med_i]), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'ba_seed_stats_052.csv', index=False)

    # ---------------- step 1: reproduction gate at 5.0% -------------------------------
    print('\n--- reproduction gate at 5.0%% vs the file the page reads today ---', flush=True)
    pub = pd.read_csv(R163 / 'ba_nav_winner_cash05.csv',
                      index_col=0, parse_dates=True).iloc[:, 0]
    mine = pd.Series(store[OLD_Y]['navs'][store[OLD_Y]['med_i']], index=cal_idx)
    ok_idx = bool(mine.index.equals(pub.index))
    relmax = float(((mine - pub.reindex(mine.index)).abs()
                    / pub.reindex(mine.index).abs()).max())
    zold = np.load(R163 / 'ba_navs_30seed_cash05.npz', allow_pickle=True)
    allmax = float(np.abs(np.vstack(store[OLD_Y]['navs']) - zold['navs']).max())
    print('index identical           : %s (%d rows)' % (ok_idx, len(mine)))
    print('max rel diff, drawn path  : %.3e' % relmax)
    print('max abs diff, all 30 paths: %.3e' % allmax)
    # The npz holds the raw float64 paths, so it is the real proof and must be 0.0 exactly.
    # The CSV went through a decimal round trip, so its last ULP can differ; anything under
    # 1e-12 relative is that round trip and nothing else.
    repro_ok = bool(ok_idx and allmax == 0.0 and relmax < 1e-12)
    print('REPRODUCTION %s (30/30 paths bit-identical in the npz; the CSV differs only by '
          'its decimal round trip)' % ('EXACT' if repro_ok else 'MISMATCH'))
    if not repro_ok:
        print('!! the 5.0%% run does not reproduce the published 5.0%% curve — stopping.')
        sys.exit(2)

    # ---------------- outputs ----------------------------------------------------------
    # THE DRAWN PATH IS HELD AT THE SEED THE 5.0% PAGE DREW.
    # The page shows ONE path per ensemble book. Moving the yield by 20 bps perturbs integer
    # share counts, which changes which names win slot contention, which re-draws the path:
    # the per-seed spread below is -1.4 to +0.8 points, SEVEN TIMES the 0.05 pp effect being
    # measured. Re-picking "the median-CAGR seed of this run" would therefore publish path
    # noise as a cash-rate result. Freezing the seed makes the cash rate the only difference
    # between the 5.0% and 5.2% tables. The 5.2% ensemble's own median seed and full band are
    # still reported, and the frozen seed sits inside that band.
    st = store[NEW_Y]
    frozen_i = store[OLD_Y]['med_i']
    print('\nDRAWN PATH: seed %d, frozen from the 5.0%% run (this run\'s own median-CAGR '
          'seed would have been %d)' % (SEEDS[frozen_i], SEEDS[st['med_i']]))
    pd.Series(st['navs'][frozen_i], index=cal_idx, name='nav').to_csv(
        OUT / 'ba_nav_winner_cash052.csv')
    pd.Series(st['navs'][st['med_i']], index=cal_idx, name='nav').to_csv(
        OUT / 'ba_nav_winner_cash052_ownmedianseed.csv')
    np.savez_compressed(OUT / 'ba_navs_30seed_cash052.npz',
                        dates=np.array(cal), navs=np.vstack(st['navs']),
                        seeds=np.array(SEEDS), med_seed=SEEDS[st['med_i']],
                        drawn_seed=SEEDS[frozen_i])
    pd.Series(st['invs'][frozen_i] * 100, index=cal_idx,
              name='invested_pct').round(4).to_csv(OUT / 'baseage_invested_daily_052.csv',
                                                   index_label='date')

    per_seed_inv = np.array([np.nanmean(v) for v in st['invs']]) * 100
    g50 = df[df.idle_yield == OLD_Y]
    g52 = df[df.idle_yield == NEW_Y]

    # ---- consistency check: PAIRED, seed by seed -------------------------------------
    piv = df.pivot(index='seed', columns='idle_yield', values='cagr')
    paired = piv[NEW_Y] - piv[OLD_Y]
    y50 = (1.0 + OLD_Y) ** (1 / B.TRADING_DAYS) - 1
    y52 = (1.0 + NEW_Y) ** (1 / B.TRADING_DAYS) - 1
    cash_t = 1.0 - np.nan_to_num(st['invs'][frozen_i], nan=0.0)
    yrs = (cal_idx[-1] - cal_idx[0]).days / 365.25
    predicted = (np.exp(float(np.sum(cash_t) * (y52 - y50)) / yrs) - 1) * 100
    inv_med = float(np.median(per_seed_inv))
    rule_of_thumb = (1 - inv_med / 100.0) * 0.2
    dc = float(paired.median())
    print('\nCONSISTENCY CHECK (5.0%% -> 5.2%%)')
    print('  paired per-seed CAGR delta : median %+.3f pp   [%+.2f .. %+.2f] over 30 seeds'
          % (dc, paired.min(), paired.max()))
    print('  exact sum over the measured daily invested series : %+.3f pp' % predicted)
    print('  rule of thumb (1-inv) x 0.2pp at inv %.2f%%        : %+.3f pp'
          % (inv_med, rule_of_thumb))

    summary = dict(
        events=len(events), seeds=len(SEEDS), reproduction_bit_exact=repro_ok,
        cagr_med_050=round(float(g50.cagr.median()), 2),
        cagr_worst_050=round(float(g50.cagr.min()), 2),
        maxdd_med_050=round(float(g50.maxdd.median()), 2),
        calmar_med_050=round(float(g50.calmar.median()), 3),
        cagr_med_052=round(float(g52.cagr.median()), 2),
        cagr_worst_052=round(float(g52.cagr.min()), 2),
        maxdd_med_052=round(float(g52.maxdd.median()), 2),
        calmar_med_052=round(float(g52.calmar.median()), 3),
        invested_median_pct=round(inv_med, 2),
        invested_min_pct=round(float(per_seed_inv.min()), 2),
        invested_max_pct=round(float(per_seed_inv.max()), 2),
        drawn_seed=int(SEEDS[frozen_i]),
        own_median_seed_052=int(SEEDS[st['med_i']]),
        drawn_seed_frozen_from_050=True,
        cagr_delta_paired_med_pp=round(dc, 3),
        cagr_delta_paired_min_pp=round(float(paired.min()), 3),
        cagr_delta_paired_max_pp=round(float(paired.max()), 3),
        cagr_delta_exact_pp=round(float(predicted), 3),
        cagr_delta_rule_of_thumb_pp=round(float(rule_of_thumb), 3),
    )
    json.dump(summary, open(OUT / 'ba_cash052_summary.json', 'w'), indent=1)
    print('\nINVESTED (30-seed, 5.2%% run): median %.2f%%  [%.2f .. %.2f]'
          % (summary['invested_median_pct'], summary['invested_min_pct'],
             summary['invested_max_pct']))
    print('\ndone in %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
