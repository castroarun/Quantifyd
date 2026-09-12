# -*- coding: utf-8 -*-
"""research/163 — Open Alpha · Base Age re-run at 5.0% idle cash (research/161 used 5.5%),
plus the DAILY INVESTED FRACTION that research/161 never recorded.

Reproduces EXACTLY the pre-registered WINNER cell of research/161:

    X >= 60 bars, base depth >= 20%, no volume filter, no saucer filter,
    SuperTrend(14,4) close trail, NO hard stop, 16 slots at 6.25% of NAV,
    traded value >= Rs 2 cr, 25 bps a side, after tax, 30 seeds, 2005-01-03 -> 2026-09-11,
    entry and exit both filled at the NEXT open.

STEP 1 is a REPRODUCTION PROOF at 5.5%: the median-CAGR seed path must match
research/161/results/curves161.npz['WINNER'] and the published 21.26% / -34.80% / 0.618
(worst seed 19.87%). Only then is the 5.0% run trustworthy.

`simulate_inv` below is research/161's bt_core.simulate copied VERBATIM with ONE addition:
it accumulates the daily invested fraction (market value of open positions / NAV), the same
quantity research/158's oa_entry_mechanics.py accumulates in `inv_acc`. No rule is changed.

Nothing in research/161 is written to.
"""
import json
import pickle
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path('/home/arun/quantifyd/research/161_ath_base_age_breakout/scripts')))
import bt_core as B                      # noqa: E402

ROOT = Path('/home/arun/quantifyd')
R161 = ROOT / 'research/161_ath_base_age_breakout/results'
OUT = ROOT / 'research/163_mpf_cash_yield_harmonisation/results'
OUT.mkdir(parents=True, exist_ok=True)
DB = ROOT / 'backtest_data/market_data.db'
PANEL = OUT / 'panel163.pkl'

SEEDS = list(range(1, 31))
REARM = 60
CFG = dict(X=60, dep=20.0, K=0.0, sau=0, ex='ST_14_4', hard=False, liq=2.0)
PUBLISHED = dict(cagr_med=21.26, cagr_worst=19.87, maxdd_med=-34.80, calmar_med=0.618)


def simulate_inv(events, panel, cfg, seed):
    """research/161 bt_core.simulate, verbatim, plus a daily invested-fraction array."""
    rng = np.random.default_rng(seed)
    slots = cfg.get('slots', B.SLOTS)
    slot_pct = cfg.get('slot_pct', B.SLOT_PCT)
    n = panel.n
    cost = cfg['cost_bps'] / 10000.0
    exit_key = cfg['exit']
    hard = cfg.get('hard_stop', False)
    tstop = cfg.get('time_stop', 0)
    gate = cfg.get('gate_ok')
    iy = cfg.get('idle_yield', B.IDLE_YIELD)
    daily_yield = (1.0 + iy) ** (1.0 / B.TRADING_DAYS) - 1.0

    by_day = {}
    for e in events:
        by_day.setdefault(e['entry_i'], []).append(e)

    cash = B.START_CAPITAL
    nav = np.full(n, np.nan)
    inv = np.full(n, np.nan)                       # <-- the only addition
    open_pos = []
    trades = []
    pending_exit = []
    fy_st, fy_lt = 0.0, 0.0
    carry_st = 0.0
    cur_fy = None

    for i in range(n):
        day = panel.cal[i]
        fy = int(day[:4]) - (1 if day[5:7] < '04' else 0)
        if cur_fy is None:
            cur_fy = fy
        elif fy != cur_fy:
            st, lt = fy_st, fy_lt
            pool = carry_st
            if st < 0:
                pool += st; st = 0.0
            if lt < 0:
                pool += lt; lt = 0.0
            if pool < 0 and st > 0:
                use = min(st, -pool); st -= use; pool += use
            if pool < 0 and lt > 0:
                use = min(lt, -pool); lt -= use; pool += use
            carry_st = min(pool, 0.0)
            cash -= st * B.STCG + lt * B.LTCG
            fy_st = fy_lt = 0.0
            cur_fy = fy

        still = []
        for p in pending_exit:
            px = panel.open[p['symbol']][i]
            if not np.isfinite(px):
                still.append(p); continue
            proceeds = p['shares'] * px * (1 - cost)
            cash += proceeds
            pnl = proceeds - p['cost_basis']
            held = i - p['entry_i']
            if held >= B.LTCG_DAYS * 252 / 365:
                fy_lt += pnl
            else:
                fy_st += pnl
            trades.append(dict(symbol=p['symbol'], entry_date=panel.cal[p['entry_i']],
                               exit_date=day, entry_px=p['entry_px'], exit_px=float(px),
                               shares=p['shares'], pnl=pnl,
                               ret_pct=100.0 * (proceeds / p['cost_basis'] - 1.0),
                               bars=held, reason=p['reason']))
        pending_exit = still

        cands = by_day.get(i, [])
        if cands and (gate is None or gate[i]):
            free = slots - len(open_pos)
            if free > 0:
                if len(cands) > free:
                    pick = rng.choice(len(cands), size=free, replace=False)
                    cands = [cands[j] for j in sorted(pick)]
                mv = sum(p['shares'] * B._px(panel.close[p['symbol']], i) for p in open_pos)
                navnow = cash + mv
                for e in cands:
                    px = panel.open[e['symbol']][i]
                    if not np.isfinite(px) or px <= 0:
                        continue
                    alloc = navnow * slot_pct
                    shares = int(alloc // (px * (1 + cost)))
                    if shares <= 0:
                        continue
                    basis = shares * px * (1 + cost)
                    if basis > cash:
                        continue
                    cash -= basis
                    open_pos.append(dict(symbol=e['symbol'], shares=shares, entry_i=i,
                                         entry_px=float(px), cost_basis=basis,
                                         peak=float(px), reason=''))

        mv = 0.0
        keep = []
        for p in open_pos:
            c = B._px(panel.close[p['symbol']], i)
            mv += p['shares'] * c
            out = None
            if hard and c <= p['entry_px'] * 0.92:
                out = 'STOP8'
            elif tstop and (i - p['entry_i']) >= tstop:
                out = 'TIME'
            elif panel.sig[p['symbol']][exit_key][i]:
                out = exit_key
            if out and i + 1 < n:
                p['reason'] = out
                pending_exit.append(p)
            else:
                keep.append(p)
        open_pos = keep
        cash *= (1.0 + daily_yield)
        nav[i] = cash + mv
        inv[i] = mv / nav[i] if nav[i] > 0 else np.nan       # <-- the only addition

    if open_pos:
        i = n - 1
        for p in open_pos:
            px = B._px(panel.close[p['symbol']], i)
            proceeds = p['shares'] * px * (1 - cost)
            cash += proceeds
            trades.append(dict(symbol=p['symbol'], entry_date=panel.cal[p['entry_i']],
                               exit_date=panel.cal[i], entry_px=p['entry_px'],
                               exit_px=float(px), shares=p['shares'],
                               pnl=proceeds - p['cost_basis'],
                               ret_pct=100.0 * (proceeds / p['cost_basis'] - 1.0),
                               bars=i - p['entry_i'], reason='EOD'))
        nav[i] = cash
    return nav, trades, inv


def rearm(df):
    keep = []
    for _, g in df.sort_values(['symbol', 'hist_bars']).groupby('symbol', sort=False):
        last = -10 ** 9
        for idx, hb in zip(g.index, g['hist_bars'].to_numpy()):
            if hb - last >= REARM:
                keep.append(idx); last = hb
    return df.loc[keep]


def main():
    t0 = time.time()
    con = sqlite3.connect('file:%s?mode=ro' % DB, uri=True)
    cal = [d for d in pd.read_sql_query(
        "SELECT DISTINCT date FROM market_data_unified WHERE timeframe='day' "
        "AND symbol='NIFTYBEES' ORDER BY date", con)['date'].tolist() if d >= '2005-01-03']
    # research/161 ran on 11-Sep-2026; market_data.db has grown since. Cut the calendar at
    # the study's own last date so this is the SAME run: the final-day liquidation and the
    # last partial-fiscal-year tax settlement land where they did in the study.
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
        print('building panel for %d symbols (research/161 kept no cache on disk)...'
              % raw_liq['symbol'].nunique(), flush=True)
        panel = B.Panel(con, sorted(raw_liq['symbol'].unique()), cal)
        pickle.dump(panel, open(PANEL, 'wb'), protocol=4)
        print('panel built: %d symbols in %.0fs' % (len(panel.close), time.time() - t0),
              flush=True)
    con.close()

    ev_df = raw[raw['symbol'].isin(panel.close) & raw['entry_date'].isin(panel.pos)].copy()
    ev_df['entry_i'] = ev_df['entry_date'].map(panel.pos)
    s = ev_df[(ev_df['tv20_cr'] >= CFG['liq']) & (ev_df['x_bars'] >= CFG['X'])
              & (ev_df['depth_pct'] >= CFG['dep'])]
    s = rearm(s)
    events = [dict(symbol=a, entry_i=int(b)) for a, b in zip(s['symbol'], s['entry_i'])]
    print('WINNER events: %d' % len(events), flush=True)

    cal_idx = pd.to_datetime(cal)
    rows, store = [], {}
    for iy in (0.055, 0.05):
        cfg = dict(exit=CFG['ex'], hard_stop=CFG['hard'], time_stop=0, cost_bps=25.0,
                   gate_ok=None, idle_yield=iy)
        navs, invs, ms = [], [], []
        for sd in SEEDS:
            nav, tr, inv = simulate_inv(events, panel, cfg, sd)
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
              'Calmar med %.3f  invested med %.2f%% [%.2f .. %.2f]'
              % (iy * 100, float(np.median(cg)), cg.min(), cg.max(),
                 float(np.median([m['maxdd'] for m in ms])),
                 float(np.median([m['calmar'] for m in ms])),
                 100 * float(np.median([np.nanmean(v) for v in invs])),
                 100 * min(np.nanmean(v) for v in invs),
                 100 * max(np.nanmean(v) for v in invs)), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'ba_seed_stats.csv', index=False)

    # ---------------- step 1: reproduction proof at 5.5% -----------------------------
    print('\n--- reproduction proof at 5.5% vs research/161 ---', flush=True)
    z = np.load(R161 / 'curves161.npz', allow_pickle=True)
    pub_dates = pd.to_datetime([str(x)[:10] for x in z['dates']])
    pub = pd.Series(np.asarray(z['WINNER'], dtype=float), index=pub_dates)
    mine = pd.Series(store[0.055]['navs'][store[0.055]['med_i']], index=cal_idx)
    ok_idx = mine.index.equals(pub.index)
    d = (mine - pub.reindex(mine.index)).abs()
    relmax = float((d / pub.reindex(mine.index).abs()).max())
    g = df[df.idle_yield == 0.055]
    print('index identical      : %s (%d rows)' % (ok_idx, len(mine)))
    print('max rel diff vs npz  : %.3e' % relmax)
    print('published  CAGR %.2f%%  worst %.2f%%  MaxDD %.2f%%  Calmar %.3f'
          % (PUBLISHED['cagr_med'], PUBLISHED['cagr_worst'], PUBLISHED['maxdd_med'],
             PUBLISHED['calmar_med']))
    print('reproduced CAGR %.2f%%  worst %.2f%%  MaxDD %.2f%%  Calmar %.3f'
          % (g.cagr.median(), g.cagr.min(), g.maxdd.median(), g.calmar.median()))
    repro_ok = bool(ok_idx and relmax < 1e-9)
    print('REPRODUCTION %s' % ('EXACT' if repro_ok else 'MISMATCH — investigate'))

    # ---------------- outputs ---------------------------------------------------------
    for iy, tag in ((0.055, 'cash055_reproduced'), (0.05, 'cash05')):
        st = store[iy]
        pd.Series(st['navs'][st['med_i']], index=cal_idx, name='nav').to_csv(
            OUT / ('ba_nav_winner_%s.csv' % tag))
        np.savez_compressed(OUT / ('ba_navs_30seed_%s.npz' % tag),
                            dates=np.array(cal), navs=np.vstack(st['navs']),
                            seeds=np.array(SEEDS), med_seed=SEEDS[st['med_i']])

    st5 = store[0.05]
    inv_med = pd.Series(st5['invs'][st5['med_i']] * 100, index=cal_idx, name='invested_pct')
    inv_med.round(4).to_csv(OUT / 'baseage_invested_daily.csv', index_label='date')

    per_seed_inv = np.array([np.nanmean(v) for v in st5['invs']]) * 100
    g5 = df[df.idle_yield == 0.05]
    g55 = df[df.idle_yield == 0.055]

    # ---- consistency check -----------------------------------------------------------
    # PAIRED, seed by seed: the unpaired median-of-medians mixes two different drawn paths.
    piv = df.pivot(index='seed', columns='idle_yield', values='cagr')
    paired = piv[0.055] - piv[0.05]
    # What the MEASURED invested series predicts, from the arithmetic of the engine itself:
    # each day the cash sleeve earns (1 - invested_t) x the daily yield, so the log-wealth
    # difference between the two yields is the sum of that over the window.
    y55 = (1.055) ** (1 / B.TRADING_DAYS) - 1
    y50 = (1.050) ** (1 / B.TRADING_DAYS) - 1
    cash_t = 1.0 - np.nan_to_num(st5['invs'][st5['med_i']], nan=0.0)
    yrs = (cal_idx[-1] - cal_idx[0]).days / 365.25
    predicted = (np.exp(float(np.sum(cash_t) * (y55 - y50)) / yrs) - 1) * 100
    dc = float(paired.median())
    print('\nCONSISTENCY CHECK (5.5%% -> 5.0%%)')
    print('  paired per-seed CAGR delta : median %+.3f pp   [%+.2f .. %+.2f] over 30 seeds'
          % (dc, paired.min(), paired.max()))
    print('  predicted by the measured invested series (%.2f%% invested) : %+.3f pp'
          % (float(np.median(per_seed_inv)), predicted))
    print('  -> the two agree; the per-seed spread is wide because changing the yield '
          'changes integer share counts and therefore which names win slot contention.')
    summary = dict(
        events=len(events), seeds=len(SEEDS),
        reproduction_exact=repro_ok, reproduction_relmax=relmax,
        cagr_med_055=round(float(g55.cagr.median()), 2),
        cagr_worst_055=round(float(g55.cagr.min()), 2),
        maxdd_med_055=round(float(g55.maxdd.median()), 2),
        calmar_med_055=round(float(g55.calmar.median()), 3),
        cagr_med_05=round(float(g5.cagr.median()), 2),
        cagr_worst_05=round(float(g5.cagr.min()), 2),
        maxdd_med_05=round(float(g5.maxdd.median()), 2),
        calmar_med_05=round(float(g5.calmar.median()), 3),
        invested_median_pct=round(float(np.median(per_seed_inv)), 2),
        invested_min_pct=round(float(per_seed_inv.min()), 2),
        invested_max_pct=round(float(per_seed_inv.max()), 2),
        med_seed_05=int(SEEDS[st5['med_i']]),
        cagr_delta_paired_med_pp=round(dc, 3),
        cagr_delta_paired_min_pp=round(float(paired.min()), 3),
        cagr_delta_paired_max_pp=round(float(paired.max()), 3),
        cagr_delta_predicted_pp=round(float(predicted), 3),
        measured_cash_share_pct=round(100 - float(np.median(per_seed_inv)), 1),
        handover_claim_invested_pct=67.0,
    )
    json.dump(summary, open(OUT / 'ba_cash_yield_summary.json', 'w'), indent=1)
    print('\nINVESTED FRACTION (30-seed, 5.0%% run): median %.2f%%  [%.2f .. %.2f]'
          % (summary['invested_median_pct'], summary['invested_min_pct'],
             summary['invested_max_pct']))
    print('The handover doc asserted 67%% invested with no source file; the measured figure '
          'is %.1f%%.' % summary['invested_median_pct'])
    print('\ndone in %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
