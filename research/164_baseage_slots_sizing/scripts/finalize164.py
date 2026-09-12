# -*- coding: utf-8 -*-
"""research/164 — paired tests, windows, cost ladder, cash-sleeve attribution, capacity,
contention and the house YoY table. Reads only research/164's own outputs.

Writes results/final164.json and results/tables164.md (the tables that go into RESULTS.md).
"""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / 'results'
sys.path.insert(0, str(HERE))
import sim164 as S                                                   # noqa: E402

DB = Path('/home/arun/quantifyd/backtest_data/market_data.db')
BASE = 'A_s16_eq'
W1 = ('2005-01-03', '2015-12-31')
W2 = ('2016-01-01', '2026-12-31')


def load(stage):
    c = pd.read_csv(RES / ('cells_%s.csv' % stage))
    s = pd.read_csv(RES / ('seedstats_%s.csv' % stage))
    return c, s


def paired(sd, cell, base, col):
    """Seed-paired win count. Deterministic cells (one value) are compared to every
    baseline seed, which is the honest test: a fixed rule must beat the draw's luck."""
    a = sd[sd.label == cell].sort_values('seed')[col].to_numpy(float)
    b = sd[sd.label == base].sort_values('seed')[col].to_numpy(float)
    if len(a) == 1 or np.allclose(a, a[0]):
        return int((a[0] > b).sum()), len(b), float(a[0] - np.median(b))
    n = min(len(a), len(b))
    return int((a[:n] > b[:n]).sum()), n, float(np.median(a[:n] - b[:n]))


def yoy_cells(nav, cal):
    s = pd.Series(np.asarray(nav, float), index=pd.to_datetime(cal)).dropna()
    dd = s / s.cummax() - 1.0                 # running peak of the FULL curve
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


def main():
    cells, sd = load('full')
    cells = cells.set_index('label', drop=False)
    base = cells.loc[BASE]
    rep = {'baseline': BASE, 'baseline_stats': base.to_dict()}

    # ---------------- the slot curve (axis A) ------------------------------------
    A = cells[cells.axis == 'A'].sort_values('slots')
    lines = ['## OA — Base Age · axis A: concentration, fully invested (slot_pct = 1/slots)',
             '', 'Full window 2005-01-03 → 2026-09-11, after tax, 25 bps a side, '
             '5.0% post-tax idle cash, 30 seeds, medians.', '',
             '| slots | size/slot | CAGR med | [worst..best] | MaxDD | Calmar | invested | '
             'tr/yr | max loss streak | entries taken | slot-blocked | cash-blocked |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for _, r in A.iterrows():
        lines.append('| %s%d | %.2f%% | %s%.2f%%%s | %.2f .. %.2f | %.2f%% | %s%.3f%s | %.1f%% '
                     '| %.1f | %d | %d | %d | %d |'
                     % ('**' if r.slots == 16 else '', r.slots, 100 * r.slot_pct,
                        '**' if r.slots == 16 else '', r.cagr_med,
                        '**' if r.slots == 16 else '', r.cagr_min, r.cagr_max, r.maxdd_med,
                        '**' if r.slots == 16 else '', r.calmar_med,
                        '**' if r.slots == 16 else '',
                        r.invested_pct_med, r.trades_per_yr_med, r.max_loss_streak_med,
                        r.entries_taken_med, r.turned_away_med, r.cash_refused_med))
    lines += ['', '*16 slots in bold is the incumbent. "slot-blocked" = qualifying signals '
              'refused because no slot was free; "cash-blocked" = signals whose slot WAS '
              'free but the book had no cash to fill it.*', '']

    # ---------------- axes B and C ------------------------------------------------
    try:
        i0c, _ = load('idle0')
        i0c = i0c.set_index(i0c.label.str.replace('_idle000', '', regex=False))
    except Exception:
        i0c = None
    for ax, title in (('B', 'axis B: fixed 6.25% size, deliberate cash buffer'),
                      ('C', 'axis C: size independent of slot count')):
        G = cells[cells.axis == ax].sort_values(['slots', 'slot_pct'])
        lines += ['## OA — Base Age · %s' % title, '',
                  '| cell | slots | size/slot | max invested | CAGR | MaxDD | Calmar | '
                  'avg invested | cash sleeve pp | tr/yr |', '|---|---|---|---|---|---|---|---|---|---|']
        for _, r in G.iterrows():
            cs = ''
            if i0c is not None and r.label in i0c.index:
                cs = '%.2f' % (r.cagr_med - float(i0c.loc[r.label, 'cagr_med']))
            lines.append('| %s | %d | %.2f%% | %.0f%% | %.2f%% | %.2f%% | %.3f | %.1f%% | %s | %.1f |'
                         % (r.label, r.slots, 100 * r.slot_pct, 100 * r.slots * r.slot_pct,
                            r.cagr_med, r.maxdd_med, r.calmar_med, r.invested_pct_med, cs,
                            r.trades_per_yr_med))
        lines += ['']
    # the incumbent's own cash-sleeve contribution
    if i0c is not None and BASE in i0c.index:
        rep['base_cash_sleeve_pp'] = round(float(base.cagr_med - i0c.loc[BASE, 'cagr_med']), 3)

    # ---------------- axis D: contested slot vs the random null --------------------
    lines += ['## OA — Base Age · axis D: who wins a contested slot (vs the random draw null)',
              '',
              '| rule | slots | CAGR | MaxDD | Calmar | vs random CAGR | seeds beaten (CAGR) | '
              'seeds beaten (Calmar) |', '|---|---|---|---|---|---|---|---|']
    for s_ in (8, 16):
        ref = cells.loc['A_s%02d_eq' % s_]
        lines.append('| random draw (null) | %d | %.2f%% | %.2f%% | %.3f | — | — | — |'
                     % (s_, ref.cagr_med, ref.maxdd_med, ref.calmar_med))
        for sel in ('rs', 'ext', 'tv', 'age'):
            lab = 'D_%s_s%02d' % (sel, s_)
            if lab not in cells.index:
                continue
            r = cells.loc[lab]
            wc, nc, dc = paired(sd, lab, 'A_s%02d_eq' % s_, 'cagr')
            wk, nk, _ = paired(sd, lab, 'A_s%02d_eq' % s_, 'calmar')
            lines.append('| %s | %d | %.2f%% | %.2f%% | %.3f | %+.2f pp | %d/%d | %d/%d |'
                         % (sel, s_, r.cagr_med, r.maxdd_med, r.calmar_med, dc, wc, nc, wk, nk))
    lines += ['', '*The ranked rules are DETERMINISTIC — they consume no randomness, so each '
              'has a single path. "seeds beaten" is that one path against all 30 random-draw '
              'paths, which is the honest test of a fixed rule against the draw\'s luck.*', '']

    # ---------------- paired test + windows for every cell -------------------------
    rows = []
    for lab, r in cells.iterrows():
        wc, nc, dc = paired(sd, lab, BASE, 'cagr')
        wk, nk, dk = paired(sd, lab, BASE, 'calmar')
        g = sd[sd.label == lab]
        w1, w2 = float(g.w1_cagr.median()), float(g.w2_cagr.median())
        rows.append(dict(label=lab, axis=r.axis, slots=int(r.slots), slot_pct=float(r.slot_pct),
                         select=r.select, cagr=float(r.cagr_med), cagr_worst=float(r.cagr_min),
                         maxdd=float(r.maxdd_med), calmar=float(r.calmar_med),
                         invested=float(r.invested_pct_med),
                         streak=float(r.max_loss_streak_med), tr_yr=float(r.trades_per_yr_med),
                         w1_cagr=w1, w1_dd=float(g.w1_dd.median()), w2_cagr=w2,
                         w2_dd=float(g.w2_dd.median()), w1_w2_drop=round(w1 - w2, 2),
                         robust_w2=bool(w2 >= w1 - 4.0),
                         cagr_wins=wc, cagr_delta=round(dc, 2),
                         calmar_wins=wk, calmar_delta=round(dk, 3),
                         w1_wins=paired(sd, lab, BASE, 'w1_cagr')[0],
                         w2_wins=paired(sd, lab, BASE, 'w2_cagr')[0],
                         cap_med=float(r.cap_pct_med_med), cap_p95=float(r.cap_pct_p95_med),
                         cap_over1=float(r.cap_over1pct_med),
                         pos_rs=float(r.pos_rs_med_med),
                         mult_all=float(r.mult_all_med), mult_drop10=float(r.mult_drop10_med)))
    P = pd.DataFrame(rows).set_index('label', drop=False)
    P.to_csv(RES / 'paired164.csv', index=False)

    # ---------------- shortlist: top 3 by Calmar subject to CAGR >= baseline -------
    elig = P[(P.cagr >= base.cagr_med) & (P.label != BASE)]
    short = elig.sort_values('calmar', ascending=False).head(3)
    rep['eligible_n'] = int(len(elig))
    rep['shortlist'] = list(short.label)
    lines += ['## OA — Base Age · pre-registered ranking: Calmar subject to CAGR >= the '
              '16-slot baseline, paired on the same 30 seeds', '',
              '| rank | cell | CAGR | MaxDD | Calmar | ΔCalmar | Calmar seeds won | ΔCAGR | '
              'CAGR seeds won | W1 CAGR (seeds won) | W2 CAGR (seeds won) | W1→W2 | robust? | '
              'clears the bar? |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    lines.append('| — | **%s (incumbent)** | %.2f%% | %.2f%% | %.3f | — | — | — | — | %.2f%% | '
                 '%.2f%% | %+.2f | — | — |'
                 % (BASE, base.cagr_med, base.maxdd_med, base.calmar_med,
                    P.loc[BASE, 'w1_cagr'], P.loc[BASE, 'w2_cagr'],
                    -P.loc[BASE, 'w1_w2_drop']))
    for i, (_, r) in enumerate(elig.sort_values('calmar', ascending=False).head(8).iterrows(), 1):
        bar = (r.calmar_delta >= 0.10) or (r.cagr_delta >= 2.0 and r.maxdd >= base.maxdd_med)
        bar = bar and r.calmar_wins >= 20 and r.w1_wins >= 20 and r.w2_wins >= 20 and r.robust_w2
        why = 'YES' if bar else ('no: ΔCalmar %+.3f < 0.10 and ΔCAGR %+.2f < 2pp'
                                 % (r.calmar_delta, r.cagr_delta))
        if not bar and (r.calmar_delta >= 0.10 or r.cagr_delta >= 2.0):
            why = 'no: fails a seed-win or drawdown clause'
        lines.append('| %d | %s | %.2f%% | %.2f%% | %.3f | %+.3f | %d/30 | %+.2f pp | %d/30 | '
                     '%.2f%% (%d/30) | %.2f%% (%d/30) | %+.2f | %s | **%s** |'
                     % (i, r.label, r.cagr, r.maxdd, r.calmar, r.calmar_delta, r.calmar_wins,
                        r.cagr_delta, r.cagr_wins, r.w1_cagr, r.w1_wins, r.w2_cagr, r.w2_wins,
                        -r.w1_w2_drop, 'YES' if r.robust_w2 else 'NO', why))
    lines += ['']

    # ---------------- cost ladder on the shortlist ---------------------------------
    lad = {}
    for stage, bps in (('cost40', 40), ('cost60', 60)):
        try:
            c2, _ = load(stage)
            lad[bps] = c2.set_index(c2.label.str.replace('_bps%d' % bps, '', regex=False))
        except Exception:
            pass
    if lad:
        lines += ['## OA — Base Age · cost ladder (after-tax CAGR / Calmar, 30 seeds)', '',
                  '| cell | 25 bps | 40 bps | 60 bps |', '|---|---|---|---|']
        for lab in [BASE] + list(short.label):
            cellrow = cells.loc[lab]
            cs = ['%.2f%% / %.3f' % (cellrow.cagr_med, cellrow.calmar_med)]
            for bps in (40, 60):
                if bps in lad and lab in lad[bps].index:
                    q = lad[bps].loc[lab]
                    cs.append('%.2f%% / %.3f' % (q.cagr_med, q.calmar_med))
                else:
                    cs.append('—')
            lines.append('| %s | %s |' % (lab, ' | '.join(cs)))
        lines += ['']

    # ---------------- tradeability + capacity --------------------------------------
    lines += ['## OA — Base Age · tradeability gate and capacity', '',
              '| cell | win rate | avg win | avg loss | max losing streak | trades/yr | '
              'median position ₹ | as % of the name\'s 20-day traded value | p95 | '
              'trades above 1% | multiple all | multiple drop-10 |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for lab in [BASE] + list(short.label):
        r, q = cells.loc[lab], P.loc[lab]
        lines.append('| %s | %.1f%% | %+.2f%% | %.2f%% | %d | %.1f | ₹%s | %.3f%% | %.3f%% | '
                     '%.1f%% |'
                     % (lab, r.win_rate_med, r.avg_win_med, r.avg_loss_med,
                        r.max_loss_streak_med, r.trades_per_yr_med,
                        format(int(q.pos_rs), ','), q.cap_med, q.cap_p95, q.cap_over1))
    lines += ['', '*Capacity is measured on a ₹10,00,000 book. The percentages scale '
              'linearly with capital: at ₹1 crore every figure here is 10× larger, which is '
              'the number that matters for sizing this book up.*', '']

    # ---------------- contention: does the slot limit actually bind? ---------------
    lines += ['## OA — Base Age · does slot contention actually bind? (30-seed medians, '
              '3,619 qualifying events over 1,880 signal days)', '',
              '| slots | entries actually taken | refused: no free slot | refused: free slot '
              'but no cash | days the slot cap bound | days the book was completely full |',
              '|---|---|---|---|---|---|']
    for _, r in A.iterrows():
        lines.append('| %d | %d | %d | %d | %d | %d |'
                     % (r.slots, r.entries_taken_med, r.turned_away_med, r.cash_refused_med,
                        r.days_bind_med, r.days_full_med))
    lines += ['', '*Read the third column first. At EVERY slot count the commonest reason a '
              'qualifying signal is not taken is that the book has no CASH, not that it has no '
              'SLOT — because the book never trims a winner, so a handful of bloated positions '
              'can absorb 95% of NAV while slots sit nominally free.*', '']

    # ---------------- house YoY table ----------------------------------------------
    con = sqlite3.connect('file:%s?mode=ro' % DB, uri=True)
    nb = pd.read_sql_query("SELECT date,close FROM market_data_unified WHERE timeframe='day' "
                           "AND symbol='NIFTYBEES' ORDER BY date", con)
    con.close()
    nav_b, cal, ms_b = med_nav('full', BASE)
    win_lab = short.index[0] if len(short) else BASE
    nav_w, _, ms_w = med_nav('full', win_lab)
    nbs = nb[nb['date'].isin(set(cal))].set_index('date')['close'].reindex(cal).ffill()
    bh = (nbs / nbs.iloc[0] * 1_000_000.0).to_numpy()

    cols = [('OA · Base Age 16 slots (incumbent)', nav_b),
            ('OA · Base Age %s' % win_lab, nav_w),
            ('NIFTYBEES (benchmark)', bh)]
    yy = {nm: yoy_cells(v, cal) for nm, v in cols}
    years = sorted(yy[cols[0][0]])
    lines += ['## OA — Base Age · house YoY table: incumbent vs best cell vs NIFTYBEES', '',
              'Each cell is the calendar-year return with that year\'s max drawdown beneath it, '
              'measured from the running peak of the FULL curve. After tax, net of 25 bps a '
              'side, median-seed path (incumbent seed %d, challenger seed %d). Benchmarks are '
              'excluded from the best-of picks.' % (ms_b, ms_w), '',
              '| year | %s | BEST CAGR | LEAST DD | BEST OVERALL |'
              % ' | '.join(n for n, _ in cols),
              '|---|%s---|---|---|' % ('---|' * len(cols))]
    for y in years:
        cs, picks = [], {}
        for nm, _ in cols:
            r_, d_ = yy[nm][y]
            cs.append('%+.1f<br><sub>(%.1f)</sub>' % (r_, d_))
            if 'benchmark' not in nm:
                picks[nm] = (r_, d_)
        bc = max(picks, key=lambda k: picks[k][0])
        ld = max(picks, key=lambda k: picks[k][1])
        bo = max(picks, key=lambda k: picks[k][0] + picks[k][1])
        short_ = lambda n: '16 slots' if 'incumbent' in n else win_lab
        lines.append('| %d | %s | %s | %s | %s |' % (y, ' | '.join(cs), short_(bc),
                                                     short_(ld), short_(bo)))
    sm = []
    for nm, v in cols:
        m = S.metrics(v, cal)
        sm.append('**%.2f%% / %.2f%% / %.2f**' % (m['cagr'], m['maxdd'], m['calmar']))
    lines += ['| **CAGR / MaxDD / Calmar** | %s | | | |' % ' | '.join(sm), '',
              '*All three columns span the same window, 2005-01-03 → 2026-09-11.*', '']

    rep['yoy'] = {nm: yy[nm] for nm, _ in cols}
    rep['winner'] = win_lab
    rep['cells_run'] = int(len(cells))
    open(RES / 'tables164.md', 'w', encoding='utf-8').write('\n'.join(lines))
    json.dump(rep, open(RES / 'final164.json', 'w'), indent=1, default=str)
    print('\n'.join(lines))
    print('\nwrote tables164.md, paired164.csv, final164.json')


if __name__ == '__main__':
    main()
