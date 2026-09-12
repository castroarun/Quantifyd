# -*- coding: utf-8 -*-
"""research/166 — paired tests, windows, cost ladder, conversion accounting, tradeability,
the house YoY table and the figures. Reads only research/166's own outputs (plus NIFTYBEES
from market_data.db, read-only).

Writes results/tables166.md, results/paired166.csv, results/final166.json,
frontend-ready PNGs into results/.
"""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                       # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE.parent / 'results'
ROOT = HERE.parent.parent.parent
DB = ROOT / 'backtest_data' / 'market_data.db'
sys.path.insert(0, str(HERE))
import sim166 as S                                                    # noqa: E402

BASE = 'BASE_rand'
W1 = ('2005-01-03', '2015-12-31')
W2 = ('2016-01-01', '2026-12-31')
SCORE_NAME = {
    'cushion': 'cushion above its SuperTrend line',
    'rs': '12-month relative strength',
    'athdist': 'distance below its own running high',
    'unreal': 'unrealised return since entry',
    'held': 'bars held',
    'rand': 'a random holding (NULL)',
}


def load(stage):
    c = pd.read_csv(RES / ('cells_%s.csv' % stage))
    s = pd.read_csv(RES / ('seedstats_%s.csv' % stage))
    if stage == 'full' and (RES / 'cells_ctrl.csv').exists():
        # the CONTROL cells (tie-break-only, sell-without-buying, unconditional hard stop,
        # rate-matched nulls) live in their own stage but belong in the same tables
        c = pd.concat([c, pd.read_csv(RES / 'cells_ctrl.csv')], ignore_index=True)
        s = pd.concat([s, pd.read_csv(RES / 'seedstats_ctrl.csv')], ignore_index=True)
        c = c.drop_duplicates(subset='label', keep='first')
    return c, s


def paired(sd, cell, base, col):
    a = sd[sd.label == cell].sort_values('seed')[col].to_numpy(float)
    b = sd[sd.label == base].sort_values('seed')[col].to_numpy(float)
    if not len(a) or not len(b):
        return 0, 0, np.nan
    if len(a) == 1 or np.allclose(a, a[0]):
        return int((a[0] > b).sum()), len(b), float(a[0] - np.median(b))
    n = min(len(a), len(b))
    return int((a[:n] > b[:n]).sum()), n, float(np.median(a[:n] - b[:n]))


def yoy_cells(nav, cal):
    s = pd.Series(np.asarray(nav, float), index=pd.to_datetime(cal)).dropna()
    dd = s / s.cummax() - 1.0
    out = {}
    for y, g in s.groupby(s.index.year):
        prev = s[s.index < g.index[0]]
        base = prev.iloc[-1] if len(prev) else g.iloc[0]
        out[int(y)] = (round(100 * (g.iloc[-1] / base - 1), 1),
                       round(100 * dd[dd.index.year == y].min(), 1))
    return out


def med_nav(stage, label):
    z = np.load(RES / ('navs_%s' % stage) / ('%s.npz' % label), allow_pickle=True)
    navs, seeds, ms = z['navs'], list(z['seeds']), int(z['med_seed'])
    return np.asarray(navs[seeds.index(ms)], float), [str(x) for x in z['dates']], ms


def nifty_bh(cal):
    con = sqlite3.connect('file:%s?mode=ro' % DB, uri=True)
    nb = pd.read_sql_query("SELECT date,close FROM market_data_unified WHERE timeframe='day' "
                           "AND symbol='NIFTYBEES' ORDER BY date", con)
    con.close()
    s = nb[nb['date'].isin(set(cal))].set_index('date')['close'].reindex(cal).ffill()
    return (s / s.iloc[0] * 1_000_000.0).to_numpy()


def main():
    cells, sd = load('full')
    cells = cells.set_index('label', drop=False)
    base = cells.loc[BASE]
    rep = {'baseline': BASE, 'baseline_stats': {k: base[k] for k in
                                                ('cagr_med', 'cagr_min', 'maxdd_med',
                                                 'calmar_med', 'invested_pct_med',
                                                 'cash_refused_med', 'turned_away_med',
                                                 'entries_taken_med', 'tax_paid_med',
                                                 'turnover_x_med')}}
    lines = []

    # ------------------------------------------------ harness proof
    try:
        pc, _ = load('proof')
        pc = pc.set_index('label')
        lines += ['## OA — Base Age · harness proof (run before any selection cell)', '',
                  '| idle cash | CAGR median | worst seed | MaxDD | Calmar | invested |',
                  '|---|---|---|---|---|---|']
        for lab, nm in (('PROOF_050', '5.0% — research/164 published 20.94 / −35.50 / 0.601'),
                        ('PROOF_052', '**5.2% — THE BASELINE**'),
                        ('PROOF_055', '5.5% — research/161 published 21.26 / −34.80 / 0.618')):
            if lab in pc.index:
                r = pc.loc[lab]
                lines.append('| %s | %.2f%% | %.2f%% | %.2f%% | %.3f | %.1f%% |'
                             % (nm, r.cagr_med, r.cagr_min, r.maxdd_med, r.calmar_med,
                                r.invested_pct_med))
        lines += ['']
        rep['proof'] = {l: {k: float(pc.loc[l, k]) for k in
                            ('cagr_med', 'cagr_min', 'maxdd_med', 'calmar_med')}
                        for l in pc.index}
    except Exception as exc:
        lines += ['*(proof stage not found: %s)*' % exc, '']

    # -------------------------- the deterministic tie-break is a SINGLE PATH -------------
    if (RES / 'cells_tiebreak.csv').exists():
        tb = pd.read_csv(RES / 'cells_tiebreak.csv')
        tb['y'] = tb['idle']
        lines += ['## OA — Base Age · WARNING: the deterministic contested-slot tie-break has '
                  'no ensemble, and its drawdown is one coin flip', '',
                  'research/164 recommended ranking a contested slot by the largest 20-day '
                  'traded value (tv20), on the strength of a −32.67% drawdown and 0.665 '
                  'Calmar at 5.0% idle cash. A ranked rule is DETERMINISTIC: it consumes no '
                  'randomness, so it has exactly ONE path, and a change as small as the idle-'
                  'cash rate re-orders which entries are affordable and hands it a different '
                  'path. Here is the same rule at five cash rates, 30 seeds for the random '
                  'draw and one path for each ranked rule.', '',
                  '| idle cash | random draw (incumbent) | tie-break = tv20 | tie-break = '
                  'rs252 | tie-break = base age |', '|---|---|---|---|---|']
        for y in sorted(tb['y'].unique()):
            g = tb[tb['y'] == y].set_index(tb[tb['y'] == y].label.str.split('_y').str[0])
            cs = []
            for lab in ('BASE_rand', 'BASE_tv', 'CTRL_sel_rs', 'CTRL_sel_age'):
                if lab in g.index:
                    r = g.loc[lab]
                    cs.append('%.2f%% / %.2f%% / %.3f' % (r.cagr_med, r.maxdd_med,
                                                          r.calmar_med))
                else:
                    cs.append('—')
            lines.append('| %s%.1f%%%s | %s |'
                         % ('**' if abs(y - 0.052) < 1e-9 else '', 100 * y,
                            '**' if abs(y - 0.052) < 1e-9 else '', ' | '.join(cs)))
        lines += ['', '*CAGR / MaxDD / Calmar. The tv20 rule\'s CAGR is stable across every '
                  'rate (21.40–21.76%, always above the random draw). Its DRAWDOWN is not: '
                  '−32.67, −32.63, **−35.93**, −32.54, −32.45. The rs252 rule moves the '
                  'opposite way at exactly the same rate. Nothing about the market changed — '
                  'only which entries the book could afford on a handful of days. Read the '
                  'CAGR column; do not plan on the Calmar.*', '']

    # ------------------------------------------------ axis A: rotation
    A = cells[cells.axis == 'A'].copy()
    A['sc'] = A['rot_score']
    lines += ['## OA — Base Age · axis A: ROTATION — swap the weakest holding out for a '
              'signal the book could not take', '',
              'Full window 2005-01-03 → 2026-09-11, after tax, 25 bps a side, 5.2% post-tax '
              'idle cash, 16 slots @ 6.25%, 30 seeds, medians. Every figure is paired against '
              'the incumbent on the same seed.', '',
              '| weakest = | margin | CAGR | worst seed | MaxDD | Calmar | ΔCalmar | Calmar '
              'seeds won | ΔCAGR | CAGR seeds won | swaps/yr | turnover ×NAV | tax paid ₹ | '
              'cash refusals | slot refusals | invested |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    lines.append('| **never swap (incumbent)** | — | **%.2f%%** | %.2f%% | %.2f%% | **%.3f** | '
                 '— | — | — | — | 0.0 | %.2f | %s | %d | %d | %.1f%% |'
                 % (base.cagr_med, base.cagr_min, base.maxdd_med, base.calmar_med,
                    base.turnover_x_med, format(int(base.tax_paid_med), ','),
                    base.cash_refused_med, base.turned_away_med, base.invested_pct_med))
    order = ['cushion', 'rs', 'athdist', 'unreal', 'held']
    for sc in order:
        G = A[A.sc == sc].sort_values('rot_margin')
        for _, r in G.iterrows():
            wc, nc, dc = paired(sd, r.label, BASE, 'cagr')
            wk, nk, dk = paired(sd, r.label, BASE, 'calmar')
            lines.append('| %s | %s | %.2f%% | %.2f%% | %.2f%% | %.3f | %+.3f | %d/%d | '
                         '%+.2f pp | %d/%d | %.1f | %.2f | %s | %d | %d | %.1f%% |'
                         % (SCORE_NAME[sc], _marg(sc, r.rot_margin), r.cagr_med, r.cagr_min,
                            r.maxdd_med, r.calmar_med, dk, wk, nk, dc, wc, nc,
                            r.swaps_per_yr_med, r.turnover_x_med,
                            format(int(r.tax_paid_med), ','), r.cash_refused_med,
                            r.turned_away_med, r.invested_pct_med))
    lines += ['']

    # ------------------------------------------------ the null
    N = cells[cells.axis == 'A_null'].sort_values('rot_margin')
    lines += ['## OA — Base Age · the NULL: swap out a RANDOM holding at the same rate', '',
              '| swap probability | CAGR | MaxDD | Calmar | ΔCalmar vs incumbent | Calmar '
              'seeds won | swaps/yr | turnover ×NAV | tax paid ₹ |',
              '|---|---|---|---|---|---|---|---|---|']
    for _, r in N.iterrows():
        wk, nk, dk = paired(sd, r.label, BASE, 'calmar')
        lines.append('| p = %.2f | %.2f%% | %.2f%% | %.3f | %+.3f | %d/%d | %.1f | %.2f | %s |'
                     % (r.rot_margin, r.cagr_med, r.maxdd_med, r.calmar_med, dk, wk, nk,
                        r.swaps_per_yr_med, r.turnover_x_med,
                        format(int(r.tax_paid_med), ',')))
    lines += ['', '*The null is the honest test of Arun\'s question: if a random swap does as '
              'well as a ranked one, the ranking is not what is working.*', '']

    # ------------------------------------------------ the controls that decide the mechanism
    C = cells[cells.axis == 'CTRL']
    if len(C):
        lines += ['## OA — Base Age · the CONTROLS — is the swap doing the work, or is it a '
                  'stop-loss / a better tie-break in disguise?', '',
                  '| control | what it isolates | CAGR | MaxDD | Calmar | ΔCalmar vs '
                  'incumbent | Calmar seeds won | swaps or stops /yr | invested |',
                  '|---|---|---|---|---|---|---|---|---|']
        WHAT = {
            'CTRL_sel_rs': 'contested slots to the strongest 12-month mover; NO rotation',
            'CTRL_sel_age': 'contested slots to the oldest base; NO rotation',
            'CTRL_sel_ext': 'contested slots to the least extended; NO rotation',
            'CTRL_sellonly_unre_m010': 'SELL the ≥10%-under-water holding on the same '
                                       'trigger but do NOT buy the entrant',
            'CTRL_hardstop08': 'an UNCONDITIONAL −8% hard stop, no rotation',
            'CTRL_hardstop10': 'an UNCONDITIONAL −10% hard stop, no rotation',
            'CTRL_hardstop15': 'an UNCONDITIONAL −15% hard stop, no rotation',
        }
        for lab in [l for l in cells.index if str(cells.loc[l, 'axis']) == 'CTRL']:
            r = cells.loc[lab]
            wk, nk, dk = paired(sd, lab, BASE, 'calmar')
            lines.append('| %s | %s | %.2f%% | %.2f%% | %.3f | %+.3f | %d/%d | %.1f | %.1f%% |'
                         % (lab, WHAT.get(lab, ''), r.cagr_med, r.maxdd_med, r.calmar_med,
                            dk, wk, nk, r.swaps_per_yr_med, r.invested_pct_med))
        lines += ['']

    # ------------------------------------------------ axis B: drift
    B = cells[cells.axis == 'B']
    lines += ['## OA — Base Age · axis B: DRIFT — trim a bloated winner, or size the entry to '
              'the cash there is', '',
              '| rule | CAGR | worst seed | MaxDD | Calmar | ΔCalmar | Calmar seeds won | '
              'ΔCAGR | trims/yr | partial fills | cash refusals (base %d) | refusals '
              'CONVERTED | entries taken (base %d) | turnover ×NAV | tax paid ₹ | invested |'
              % (base.cash_refused_med, base.entries_taken_med),
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for _, r in B.iterrows():
        wc, nc, dc = paired(sd, r.label, BASE, 'cagr')
        wk, nk, dk = paired(sd, r.label, BASE, 'calmar')
        lines.append('| %s | %.2f%% | %.2f%% | %.2f%% | %.3f | %+.3f | %d/%d | %+.2f pp | '
                     '%.1f | %d | %d | **%+d** | %d | %.2f | %s | %.1f%% |'
                     % (_bname(r), r.cagr_med, r.cagr_min, r.maxdd_med, r.calmar_med, dk, wk,
                        nk, dc, r.trims_per_yr_med, r.partial_fills_med, r.cash_refused_med,
                        int(base.cash_refused_med - r.cash_refused_med), r.entries_taken_med,
                        r.turnover_x_med, format(int(r.tax_paid_med), ','),
                        r.invested_pct_med))
    lines += ['', '*"refusals CONVERTED" is how many of the incumbent\'s %d cash refusals this '
              'rule turned into an actual entry.*' % int(base.cash_refused_med), '']

    # ------------------------------------------------ extra / interaction cells
    for ax, title in (('X', 'axis A follow-ups: swaps per day, entrant priority, margin '
                            'plateau'),
                      ('C', 'axis C: INTERACTION — best rotation × best drift')):
        G = cells[cells.axis == ax]
        if not len(G):
            continue
        lines += ['## OA — Base Age · %s' % title, '',
                  '| cell | CAGR | worst seed | MaxDD | Calmar | ΔCalmar | Calmar seeds won | '
                  'ΔCAGR | swaps/yr | trims/yr | cash refusals | turnover ×NAV | tax paid ₹ |',
                  '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for _, r in G.iterrows():
            wc, nc, dc = paired(sd, r.label, BASE, 'cagr')
            wk, nk, dk = paired(sd, r.label, BASE, 'calmar')
            lines.append('| %s | %.2f%% | %.2f%% | %.2f%% | %.3f | %+.3f | %d/%d | %+.2f pp | '
                         '%.1f | %.1f | %d | %.2f | %s |'
                         % (r.label, r.cagr_med, r.cagr_min, r.maxdd_med, r.calmar_med, dk,
                            wk, nk, dc, r.swaps_per_yr_med, r.trims_per_yr_med,
                            r.cash_refused_med, r.turnover_x_med,
                            format(int(r.tax_paid_med), ',')))
        lines += ['']

    # ------------------------------------------------ the pre-registered ranking
    rows = []
    for lab, r in cells.iterrows():
        wc, nc, dc = paired(sd, lab, BASE, 'cagr')
        wk, nk, dk = paired(sd, lab, BASE, 'calmar')
        g = sd[sd.label == lab]
        w1, w2 = float(g.w1_cagr.median()), float(g.w2_cagr.median())
        rows.append(dict(label=lab, axis=r.axis, rot_score=r.rot_score,
                         rot_margin=float(r.rot_margin), trim_mult=float(r.trim_mult),
                         trim_when=r.trim_when, min_fill=float(r.min_fill_frac),
                         cagr=float(r.cagr_med), cagr_worst=float(r.cagr_min),
                         maxdd=float(r.maxdd_med), calmar=float(r.calmar_med),
                         invested=float(r.invested_pct_med),
                         streak=float(r.max_loss_streak_med),
                         tr_yr=float(r.trades_per_yr_med), swaps_yr=float(r.swaps_per_yr_med),
                         trims_yr=float(r.trims_per_yr_med),
                         cash_refused=float(r.cash_refused_med),
                         slot_refused=float(r.turned_away_med),
                         entries=float(r.entries_taken_med), tax=float(r.tax_paid_med),
                         turnover=float(r.turnover_x_med), w1_cagr=w1,
                         w1_dd=float(g.w1_dd.median()), w2_cagr=w2,
                         w2_dd=float(g.w2_dd.median()), w1_w2_drop=round(w1 - w2, 2),
                         robust_w2=bool(w2 >= w1 - 4.0), cagr_wins=wc, cagr_delta=round(dc, 2),
                         calmar_wins=wk, calmar_delta=round(dk, 3),
                         w1_wins=paired(sd, lab, BASE, 'w1_cagr')[0],
                         w2_wins=paired(sd, lab, BASE, 'w2_cagr')[0],
                         top10=float(r.top10_share_med),
                         top10_all=float(r.top10_share_all_med),
                         cap_med=float(r.cap_pct_med_med), cap_p95=float(r.cap_pct_p95_med),
                         cap_over1=float(r.cap_over1pct_med),
                         pos_rs=float(r.pos_rs_med_med)))
    P = pd.DataFrame(rows).set_index('label', drop=False)
    P.to_csv(RES / 'paired166.csv', index=False)

    elig = P[(P.cagr >= base.cagr_med) & (~P.index.isin([BASE, 'BASE_tv']))
             & (P.axis != 'A_null')]
    short = elig.sort_values('calmar', ascending=False).head(5)
    rep['eligible_n'] = int(len(elig))
    rep['shortlist'] = list(short.label)
    lines += ['## OA — Base Age · the pre-registered ranking: after-tax Calmar subject to '
              'CAGR ≥ the 5.2% baseline, paired on the same 30 seeds', '',
              '| rank | cell | CAGR | MaxDD | Calmar | ΔCalmar | Calmar seeds won | ΔCAGR | '
              'CAGR seeds won | W1 CAGR (won) | W2 CAGR (won) | W1→W2 | robust? | clears the '
              'bar? |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    lines.append('| — | **%s (incumbent)** | %.2f%% | %.2f%% | %.3f | — | — | — | — | %.2f%% | '
                 '%.2f%% | %+.2f | — | — |'
                 % (BASE, base.cagr_med, base.maxdd_med, base.calmar_med,
                    P.loc[BASE, 'w1_cagr'], P.loc[BASE, 'w2_cagr'],
                    -P.loc[BASE, 'w1_w2_drop']))
    for i, (_, r) in enumerate(elig.sort_values('calmar', ascending=False).head(10).iterrows(),
                               1):
        bar = (r.calmar_delta >= 0.10) or (r.cagr_delta >= 2.0 and r.maxdd >= base.maxdd_med)
        bar = bar and r.calmar_wins >= 20 and r.w1_wins >= 20 and r.w2_wins >= 20 and r.robust_w2
        why = 'YES' if bar else ('no: ΔCalmar %+.3f < 0.10 and ΔCAGR %+.2f < 2pp'
                                 % (r.calmar_delta, r.cagr_delta))
        if not bar and (r.calmar_delta >= 0.10 or r.cagr_delta >= 2.0):
            why = 'no: fails a seed-win, window or drawdown clause'
        lines.append('| %d | %s | %.2f%% | %.2f%% | %.3f | %+.3f | %d/30 | %+.2f pp | %d/30 | '
                     '%.2f%% (%d/30) | %.2f%% (%d/30) | %+.2f | %s | **%s** |'
                     % (i, r.label, r.cagr, r.maxdd, r.calmar, r.calmar_delta, r.calmar_wins,
                        r.cagr_delta, r.cagr_wins, r.w1_cagr, r.w1_wins, r.w2_cagr, r.w2_wins,
                        -r.w1_w2_drop, 'YES' if r.robust_w2 else 'NO', why))
    if not len(elig):
        lines.append('| — | *no cell reached the baseline CAGR* | | | | | | | | | | | | |')
    lines += ['']

    # ------------------------------------------------ cost ladder
    lad = {}
    for stage, bps in (('cost40', 40), ('cost60', 60)):
        try:
            c2, _ = load(stage)
            lad[bps] = c2.set_index(c2.label.str.replace('_b%d' % bps, '', regex=False))
        except Exception:
            pass
    if lad:
        lines += ['## OA — Base Age · cost ladder (after-tax CAGR / Calmar, 30 seeds)', '',
                  '| cell | 25 bps | 40 bps | 60 bps |', '|---|---|---|---|']
        for lab in [BASE] + list(short.label):
            cr = cells.loc[lab]
            cs = ['%.2f%% / %.3f' % (cr.cagr_med, cr.calmar_med)]
            for bps in (40, 60):
                if bps in lad and lab in lad[bps].index:
                    q = lad[bps].loc[lab]
                    cs.append('%.2f%% / %.3f' % (q.cagr_med, q.calmar_med))
                else:
                    cs.append('—')
            lines.append('| %s | %s |' % (lab, ' | '.join(cs)))
        lines += ['']

    # ------------------------------------------------ tradeability + capacity
    lines += ['## OA — Base Age · tradeability gate, churn and capacity', '',
              '| cell | win rate | avg win | avg loss | max losing streak | trades/yr | '
              'swaps/yr | trims/yr | turnover ×NAV | tax paid ₹ | top-10 share of profit | '
              'median position ₹ | % of the name\'s 20-day traded value | p95 | trades > 1% |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for lab in [BASE] + list(short.label):
        r, q = cells.loc[lab], P.loc[lab]
        lines.append('| %s | %.1f%% | %+.2f%% | %.2f%% | %d | %.1f | %.1f | %.1f | %.2f | %s | '
                     '%.1f%% | ₹%s | %.3f%% | %.3f%% | %.1f%% |'
                     % (lab, r.win_rate_med, r.avg_win_med, r.avg_loss_med,
                        r.max_loss_streak_med, r.trades_per_yr_med, r.swaps_per_yr_med,
                        r.trims_per_yr_med, r.turnover_x_med,
                        format(int(r.tax_paid_med), ','), q.top10_all,
                        format(int(q.pos_rs), ','), q.cap_med, q.cap_p95, q.cap_over1))
    lines += ['', '*Capacity is measured on a ₹10,00,000 book; at ₹1 crore every percentage '
              'here is 10× larger. "top-10 share of profit" counts every realisation, trims '
              'included.*', '']

    # ------------------------------------------------ the conversion accounting
    lines += ['## OA — Base Age · does it fix the cash-refusal problem? (30-seed medians, '
              '3,619 qualifying events)', '',
              '| cell | entries taken | refused: no free slot | refused: no cash | swaps | '
              'trims | average invested |', '|---|---|---|---|---|---|---|']
    for lab in [BASE] + list(short.label) + [l for l in cells.index
                                             if cells.loc[l, 'axis'] == 'B'
                                             and l not in list(short.label)]:
        if lab not in cells.index:
            continue
        r = cells.loc[lab]
        lines.append('| %s | %d | %d | %d | %d | %d | %.1f%% |'
                     % (lab, r.entries_taken_med, r.turned_away_med, r.cash_refused_med,
                        r.swaps_med, r.trims_med, r.invested_pct_med))
    lines += ['']

    # ------------------------------------------------ YoY + figures
    nav_b, cal, ms_b = med_nav('full', BASE)
    picks = [(BASE, 'OA · Base Age 16 slots (incumbent)', nav_b, ms_b)]
    bestA = _best(P, 'rot')                      # best PRE-REGISTERED rotation (tv entrant)
    bestX = _best(P, 'rotx')                     # best rotation incl. the post-hoc entrant
    bestB = _best(P, 'drift')
    for lab, tag in ((bestA, 'rotation, pre-registered entrant'),
                     (bestX, 'rotation, best entrant — post-hoc'),
                     (bestB, 'best DRIFT')):
        if lab and lab not in [p[0] for p in picks]:
            v, _, ms = med_nav('full', lab)
            picks.append((lab, 'OA · %s (%s)' % (lab, tag), v, ms))
    bh = nifty_bh(cal)
    cols = [(nm, v) for (_, nm, v, _) in picks] + [('NIFTYBEES (benchmark)', bh)]
    yy = {nm: yoy_cells(v, cal) for nm, v in cols}
    short_nm = {nm: (lab if i else '16 slots')
                for i, (lab, nm, _, _) in enumerate(picks)}
    lines += ['## OA — Base Age · house YoY table: incumbent vs best rotation vs best drift '
              'vs NIFTYBEES', '',
              'Each cell is the calendar-year return with that year\'s max drawdown beneath '
              'it, measured from the running peak of the FULL curve. After tax, net of 25 bps '
              'a side, 5.2%% idle cash, median-seed path (%s). Benchmarks are excluded from '
              'the best-of picks.'
              % ', '.join('%s seed %d' % (l, m) for (l, _, _, m) in picks), '',
              '| year | %s | BEST CAGR | LEAST DD | BEST OVERALL |'
              % ' | '.join(n for n, _ in cols),
              '|---|%s---|---|---|' % ('---|' * len(cols))]
    for y in sorted(yy[cols[0][0]]):
        cs, pk = [], {}
        for nm, _ in cols:
            r_, d_ = yy[nm][y]
            cs.append('%+.1f<br><sub>(%.1f)</sub>' % (r_, d_))
            if 'benchmark' not in nm:
                pk[nm] = (r_, d_)
        bc = max(pk, key=lambda k: pk[k][0])
        ld = max(pk, key=lambda k: pk[k][1])
        bo = max(pk, key=lambda k: pk[k][0] + pk[k][1])
        lines.append('| %d | %s | %s | %s | %s |'
                     % (y, ' | '.join(cs), short_nm[bc], short_nm[ld], short_nm[bo]))
    sm = []
    for nm, v in cols:
        m = S.metrics(v, cal)
        sm.append('**%.2f%% / %.2f%% / %.2f**' % (m['cagr'], m['maxdd'], m['calmar']))
    lines += ['| **CAGR / MaxDD / Calmar** | %s | | | |' % ' | '.join(sm), '',
              '*All columns span the same window, 2005-01-03 → 2026-09-11.*', '']

    _figure(cal, cols, RES / 'r166_curves.png')
    rep['yoy'] = {nm: yy[nm] for nm, _ in cols}
    rep['best_rotation'] = bestA
    rep['best_drift'] = bestB
    rep['cells_run'] = int(len(cells))
    open(RES / 'tables166.md', 'w', encoding='utf-8').write('\n'.join(lines))
    json.dump(rep, open(RES / 'final166.json', 'w'), indent=1, default=str)
    print('\n'.join(lines))
    print('\nwrote tables166.md, paired166.csv, final166.json, r166_curves.png')


def _marg(sc, m):
    if sc == 'held':
        return '≥ %d bars held' % m
    if sc in ('unreal', 'athdist'):
        return '≥ %g pp below' % m
    return '≥ %g pp better' % m


def _bname(r):
    if r.trim_mult and r.trim_when == 'month':
        return 'trim above %g× target, month-end' % r.trim_mult
    if r.trim_mult and r.trim_when == 'demand':
        return 'trim above %g× target, on demand' % r.trim_mult
    if r.min_fill_frac:
        return 'partial fill, min %.0f%% of a slot' % (100 * r.min_fill_frac)
    return r.label


def _best(P, kind):
    """Highest-Calmar cell of a family, subject to CAGR >= the baseline.
    rot   = pure rotation on the PRE-REGISTERED entrant priority (largest tv20)
    rotx  = pure rotation, any entrant priority (includes the post-hoc rs-entrant variant)
    drift = trimming / partial fills only, no rotation"""
    base_cagr = float(P.loc[BASE, 'cagr'])
    rot = P.rot_score.notna() & (P.rot_score != 'rand') & (P.trim_mult == 0) \
        & (P.min_fill == 0) & (P.axis != 'CTRL')
    if kind == 'rot':
        G = P[rot & (P.axis == 'A')]
    elif kind == 'rotx':
        G = P[rot]
    else:
        # a drift rule that never fires is not a drift rule -- require real activity
        G = P[P.rot_score.isna() & ((P.trims_yr >= 1.0) | (P.min_fill > 0))
              & (P.axis != 'CTRL')]
    G = G[G.cagr >= base_cagr]
    if not len(G):
        return None
    return str(G.sort_values('calmar', ascending=False).index[0])


def _figure(cal, cols, path):
    idx = pd.to_datetime(cal)
    fig, ax = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                           gridspec_kw=dict(height_ratios=[2.4, 1]))
    colors = ['#e0e0e0', '#4fc3f7', '#ffb74d', '#8d8d8d']
    for k, (nm, v) in enumerate(cols):
        s = pd.Series(np.asarray(v, float), index=idx).dropna()
        ax[0].plot(s.index, s / s.iloc[0] * 100.0, lw=1.6 if k < 3 else 1.1,
                   color=colors[k % len(colors)],
                   ls='--' if 'benchmark' in nm else '-', label=nm)
        ax[1].plot(s.index, 100 * (s / s.cummax() - 1.0), lw=1.1,
                   color=colors[k % len(colors)],
                   ls='--' if 'benchmark' in nm else '-')
    ax[0].set_yscale('log')
    ax[0].set_ylabel('growth of ₹100 (log)')
    ax[0].legend(loc='upper left', fontsize=9, framealpha=0.2)
    ax[0].set_title('Open Alpha · Base Age — rotation and drift vs the incumbent '
                    '(after tax, 25 bps, 5.2% idle cash, median seed)', fontsize=12)
    ax[1].set_ylabel('drawdown %')
    for a in ax:
        a.grid(alpha=0.25)
        a.set_facecolor('#14161a')
    fig.patch.set_facecolor('#0f1115')
    for a in ax:
        a.tick_params(colors='#cfd3d8')
        for sp in a.spines.values():
            sp.set_color('#3a3f47')
        a.yaxis.label.set_color('#cfd3d8')
        a.xaxis.label.set_color('#cfd3d8')
    ax[0].title.set_color('#e8eaed')
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == '__main__':
    main()
