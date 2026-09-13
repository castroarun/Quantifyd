# -*- coding: utf-8 -*-
"""research/173 - IPO Base Spec A: exit at the NEXT OPEN instead of the SIGNAL CLOSE.

Fork of research/153 `ipo_replay.simulate_ipo` (read, not modified) with an exit-fill arm:
  A   sell at close[i]                                   (as published)
  B   sell at open[i+1], no floor
  C   sell at open[j] only if open[j] >= 0.98 * latest close; else retry next open
  C2  as C, but a day LIMIT also fills AT the floor if high[j] reaches it (sensitivity)
In B/C/C2 the slot is freed on day i (available to day i+1's entries, as in A), the position
stays marked to the close until sold, and the sale cash lands on day j before entries.
An exit firing on the window's last bar sells at that close.

usage:  exit_timing.py check [head]   |   run [head]   |   report [head]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/arun/quantifyd')
HERE = ROOT / 'research/173_ipo_exit_next_open'
RES = HERE / 'results'
RES.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / 'research/167_ipo_base_honest_reopt/scripts'))
import ipo_honest as ih                                     # noqa: E402

SPEC_A = {**ih.INCUMBENT, 'trail': 50, 'stop': 0.10, 'target': 0.25}
GATE_N = 150
YIELDS = (0.05, 0.052)
WINS = ('w2', 'wa', 'wb')
ARMS = ('A', 'B', 'C', 'C2')
SEEDS = ih.SEEDS
CAPITAL = 1_000_000.0
FLOOR = 0.98
SEED_CSV = RES / 'seed_stats.csv'
TRADES_CSV = RES / 'trades_w2_y050.csv'


def load(head=False):
    import ipo_replay as ir
    if head:
        ir.RES = HERE / 'listing_head'
        print('[ctx] using COMMITTED listing_dates.csv (HEAD copy)', flush=True)
    ctx, ir = ih.load_ctx(clean=True)
    cfg = dict(SPEC_A)
    nb = ctx.close.get('NIFTYBEES').dropna()
    w = (nb < nb.rolling(GATE_N).mean()).shift(1)
    cfg['gate'] = 'custom'
    cfg['weak_series'] = w.reindex(ctx.dates).ffill().fillna(False).to_numpy(bool)
    setup, piv, lo0 = ih.build_setup(ctx, cfg)
    trig, lvl, lo, fc = ih.apply_mechanic(ctx, setup, piv, lo0, 'nextday_pivot')
    assert fc is False
    return ctx, ir, cfg, trig, lvl, lo


def sim(seed, days_idx, dates, C, O, H, PIV, SMA, TVp, TRIG, weak, *, arm, cost, stop,
        slots, size_pct, target, cash_yield, stcg=0.20, ltcg=0.125, capital=CAPITAL):
    rng = np.random.default_rng(seed)
    cash = float(capital)
    positions = []          # (col, ei, buy, qty, stop_px, tv)
    pending = []            # exiting positions awaiting a fill
    trades = []
    n = len(days_idx)
    equity = np.empty(n)
    invested = np.empty(n)
    y_day = 1.0 + cash_yield / 252.0
    st_acc = [0.0, 0.0]     # fy short-term, long-term
    carry = 0.0

    def fy_of(d):
        return d.year if d.month >= 4 else d.year - 1

    cur_fy = fy_of(dates[days_idx[0]])

    def book(c, ei, b, q, tv, px, i, reason, extra):
        pnl = q * (px * (1 - cost) - b * (1 + cost))
        held = (dates[i] - dates[ei]).days
        if held > 365:
            st_acc[1] += pnl
        else:
            st_acc[0] += pnl
        trades.append(dict(col=int(c), ei=int(ei), xi=int(i), buy=b, sell=float(px), qty=q,
                           reason=reason, held=held, tv=tv, ret=float(px) / b - 1.0,
                           notional=q * b, **extra))
        return q * px * (1 - cost)

    def mark(i):
        m = sum(q * (C[i, c] if not np.isnan(C[i, c]) else b)
                for c, _, b, q, _, _ in positions)
        m += sum(p['q'] * (C[i, p['c']] if not np.isnan(C[i, p['c']]) else p['last'])
                 for p in pending)
        return m

    for k, i in enumerate(days_idx):
        if cash_yield and cash > 0:
            cash *= y_day
        d = dates[i]
        if stcg and fy_of(d) != cur_fy:
            st, lt, cf = st_acc[0], st_acc[1], carry
            if cf < 0:
                u = min(-cf, max(st, 0.0)); st -= u; cf += u
                u = min(-cf, max(lt, 0.0)); lt -= u; cf += u
            if st < 0:
                lt += st; st = 0.0
            if lt < 0:
                cf += lt; lt = 0.0
            cash -= stcg * max(st, 0.0) + ltcg * max(lt, 0.0)
            carry = cf
            st_acc[0] = st_acc[1] = 0.0
            cur_fy = fy_of(d)

        # pending exits fill at this morning's open, before entries
        if pending:
            keep = []
            for p in pending:
                c = p['c']
                o = O[i, c]
                if not (np.isfinite(o) and o > 0):
                    if np.isfinite(C[i, c]):
                        p['last'] = float(C[i, c])
                    keep.append(p)
                    continue
                if p['first_open'] is None:
                    p['first_open'] = float(o)
                px = None
                fl = FLOOR * p['last']
                if arm == 'B' or o >= fl:
                    px = float(o)
                elif arm == 'C2' and np.isfinite(H[i, c]) and H[i, c] >= fl:
                    px = fl
                if px is None:
                    p['floor_misses'] += 1
                    if np.isfinite(C[i, c]):
                        p['last'] = float(C[i, c])
                    keep.append(p)
                    continue
                cash += book(c, p['ei'], p['b'], p['q'], p['tv'], px, i, p['reason'],
                             dict(sig_xi=p['sig_i'], sig_close=p['sig_close'],
                                  gap=p['first_open'] / p['sig_close'] - 1.0,
                                  floor_misses=p['floor_misses'],
                                  extra_days=k - p['sig_k'] - 1,
                                  cost_vs_close=px / p['sig_close'] - 1.0))
            pending = keep

        # entries: identical to simulate_ipo (pct stop, realistic fill, fixed size)
        if not weak[i]:
            cand = np.nonzero(TRIG[i])[0]
            if len(cand):
                eq = cash + mark(i)
                cand = rng.permutation(cand)
                for c in cand:
                    if len(positions) >= slots:
                        continue
                    pv = float(PIV[i, c])
                    fill = max(pv, float(O[i, c]))
                    if not np.isfinite(fill) or fill <= 0:
                        continue
                    sp = fill * (1 - stop)
                    size = min(size_pct, 0.30) * eq
                    qty = int(size / fill)
                    if qty < 1 or cash < qty * fill * (1 + cost):
                        continue
                    cash -= qty * fill * (1 + cost)
                    positions.append((c, i, fill, qty, sp, float(TVp[i, c])))

        # exit decisions at the close
        still = []
        last_bar = (k == n - 1)
        for c, ei, b, q, sp, tv in positions:
            cl = C[i, c]
            if np.isnan(cl):
                still.append((c, ei, b, q, sp, tv))
                continue
            reason = None
            if cl <= sp:
                reason = 'stop'
            elif target is not None and cl >= b * (1 + target):
                reason = 'target'
            elif i > ei and not np.isnan(SMA[i, c]) and cl < SMA[i, c]:
                reason = 'trail'
            if not reason:
                still.append((c, ei, b, q, sp, tv))
            elif arm == 'A' or last_bar:
                cash += book(c, ei, b, q, tv, float(cl), i, reason,
                             dict(sig_xi=int(i), sig_close=float(cl), gap=np.nan,
                                  floor_misses=0, extra_days=-1, cost_vs_close=0.0))
            else:
                pending.append(dict(c=c, ei=ei, b=b, q=q, tv=tv, reason=reason,
                                    sig_i=int(i), sig_k=k, sig_close=float(cl),
                                    last=float(cl), first_open=None, floor_misses=0))
        positions = still
        mtm = mark(i)
        equity[k] = cash + mtm
        invested[k] = mtm

    last = days_idx[-1]
    for c, ei, b, q, sp, tv in positions:
        cl = C[last, c]
        px = float(cl) if not np.isnan(cl) else b
        trades.append(dict(col=int(c), ei=int(ei), xi=int(last), buy=b, sell=px, qty=q,
                           reason='open_marked', held=(dates[last] - dates[ei]).days, tv=tv,
                           ret=px / b - 1.0, notional=q * b, sig_xi=-1, sig_close=np.nan,
                           gap=np.nan, floor_misses=0, extra_days=-1, cost_vs_close=0.0))
    for p in pending:       # never filled inside the window: marked at the last close
        trades.append(dict(col=int(p['c']), ei=int(p['ei']), xi=int(last), buy=p['b'],
                           sell=p['last'], qty=p['q'], reason='pending_marked',
                           held=(dates[last] - dates[p['ei']]).days, tv=p['tv'],
                           ret=p['last'] / p['b'] - 1.0, notional=p['q'] * p['b'],
                           sig_xi=p['sig_i'], sig_close=p['sig_close'], gap=np.nan,
                           floor_misses=p['floor_misses'], extra_days=n - 1 - p['sig_k'],
                           cost_vs_close=p['last'] / p['sig_close'] - 1.0))
    return equity, trades, invested


def one(ctx, ir, cfg, trig, lvl, win, seed, arm, y):
    days = ih.days_cached(ctx, win)
    eq, trd, inv = sim(seed, days, ctx.dates, ctx.C, ctx.O, ctx.H, lvl, ctx.sma(cfg['trail']),
                       ctx.TVp, trig, cfg['weak_series'], arm=arm, cost=cfg['cost'],
                       stop=cfg['stop'], slots=cfg['slots'], size_pct=cfg['size_pct'],
                       target=cfg['target'], cash_yield=y)
    st, _ = ir.stats_from(eq, ctx.dates[days], trd, invested=inv)
    return eq, trd, st


def check(head):
    ctx, ir, cfg, trig, lvl, lo = load(head)
    days = ih.days_cached(ctx, 'w2')
    for sd in (1, 2, 3):
        e0, t0, _, _ = ir.simulate_ipo(sd, days, ctx.dates, ctx.C, ctx.O, lvl, lo,
                                       ctx.sma(cfg['trail']), ctx.RSF, ctx.TVp, trig,
                                       cfg['weak_series'], cost=cfg['cost'], stop=cfg['stop'],
                                       slots=cfg['slots'], size_pct=cfg['size_pct'],
                                       target=cfg['target'], cash_yield=0.05)
        e1, t1, _ = one(ctx, ir, cfg, trig, lvl, 'w2', sd, 'A', 0.05)
        print('seed %d: max |eq diff| %.6f  trades %d vs %d  final %.0f vs %.0f'
              % (sd, np.max(np.abs(e0 - e1)), len(t0), len(t1), e0[-1], e1[-1]), flush=True)
    o, _ = ih.run_cell(ctx, ir, cfg, trig=trig, level=lvl, lo=lo, windows=('w2',),
                       fill_close=False, cash_yield=0.05)
    print('ih.run_cell Spec A @5.0%%: CAGR %s [worst %s] DD %s (worst %s) Cal %s '
          '-- published 21.80 / -26.63 / 0.819'
          % (o['w2_cagr'], o['w2_cagr_lo'], o['w2_dd'], o['w2_dd_worst'], o['w2_calmar']),
          flush=True)


SEED_FIELDS = ['cash_y', 'window', 'arm', 'seed', 'cagr', 'dd', 'calmar', 'mean', 'median',
               'n', 'n_closed', 'win', 'avg_win', 'avg_loss', 'hold', 'invested_pct',
               'max_loss_streak', 'final', 'n_exits', 'n_deferred', 'n_floor_hit',
               'floor_miss_days', 'extra_days_mean', 'pending_marked', 'gap_mean',
               'cost_vs_close_mean']


def run(head):
    ctx, ir, cfg, trig, lvl, lo = load(head)
    done = set()
    if SEED_CSV.exists():
        d = pd.read_csv(SEED_CSV)
        done = set(zip(d.cash_y.round(4), d.window, d.arm))
    t0 = time.time()
    blocks = [(y, w, a) for y in YIELDS for w in WINS for a in ARMS]
    for bi, (y, w, a) in enumerate(blocks, 1):
        if (round(y, 4), w, a) in done:
            continue
        rows, tr_all = [], []
        for sd in SEEDS:
            _, trd, st = one(ctx, ir, cfg, trig, lvl, w, sd, a, y)
            ex = [t for t in trd if t['reason'] not in ('open_marked', 'pending_marked')]
            dfr = [t for t in ex if t['extra_days'] >= 0]
            rows.append(dict(
                cash_y=y, window=w, arm=a, seed=sd, cagr=st['cagr'], dd=st['dd'],
                calmar=st['cagr'] / abs(st['dd']),
                **{kk: st[kk] for kk in ('mean', 'median', 'n', 'n_closed', 'win', 'avg_win',
                                         'avg_loss', 'hold', 'invested_pct',
                                         'max_loss_streak', 'final')},
                n_exits=len(ex), n_deferred=len(dfr),
                n_floor_hit=sum(1 for t in ex if t['floor_misses'] > 0),
                floor_miss_days=sum(t['floor_misses'] for t in ex),
                extra_days_mean=float(np.mean([t['extra_days'] for t in dfr])) if dfr else 0.0,
                pending_marked=sum(1 for t in trd if t['reason'] == 'pending_marked'),
                gap_mean=float(np.nanmean([t['gap'] for t in dfr])) if dfr else np.nan,
                cost_vs_close_mean=float(np.mean([t['cost_vs_close'] for t in ex]))
                if ex else 0.0))
            if w == 'w2' and y == 0.05:
                for t in trd:
                    t['seed'] = sd
                    t['arm'] = a
                tr_all.extend(trd)
        pd.DataFrame(rows)[SEED_FIELDS].to_csv(SEED_CSV, mode='a',
                                               header=not SEED_CSV.exists(), index=False)
        if tr_all:
            pd.DataFrame(tr_all).to_csv(TRADES_CSV, mode='a',
                                        header=not TRADES_CSV.exists(), index=False)
        dd = pd.DataFrame(rows)
        print('[%d/%d] cash %.1f%% %s arm %-2s: CAGR med %6.2f worst %6.2f | DD med %7.2f '
              'worst %7.2f | floor-hit exits/seed %.0f | %.1f min'
              % (bi, len(blocks), y * 100, w, a, dd.cagr.median(), dd.cagr.min(),
                 dd.dd.median(), dd.dd.min(), dd.n_floor_hit.median(),
                 (time.time() - t0) / 60), flush=True)
    print('RUN DONE', flush=True)


def report(head):
    d = pd.read_csv(SEED_CSV)
    summ, pair = [], []
    for (y, w), g in d.groupby(['cash_y', 'window']):
        a = g[g.arm == 'A'].set_index('seed')
        for arm, h in g.groupby('arm'):
            h = h.set_index('seed')
            summ.append(dict(
                cash_y=y, window=w, arm=arm, cagr_med=h.cagr.median(), cagr_worst=h.cagr.min(),
                dd_med=h.dd.median(), dd_worst=h.dd.min(),
                calmar_med=h.cagr.median() / abs(h.dd.median()),
                calmar_worst_seed=h.calmar.min(), mean_tr=h['mean'].median(),
                win=h.win.median(), n_exits=h.n_exits.median(),
                n_floor_hit=h.n_floor_hit.median(), floor_miss_days=h.floor_miss_days.median(),
                extra_days_mean=h.extra_days_mean.median(),
                pending_marked_total=h.pending_marked.sum(),
                sell_vs_close_pct=100 * h.cost_vs_close_mean.median()))
            if arm == 'A':
                continue
            dc = h.cagr - a.cagr
            dk = h.calmar - a.calmar
            ddd = h.dd - a.dd
            dm = h['mean'] - a['mean']
            mat_c = (dc.median() < -1.0) and ((dc < 0).sum() >= 20)
            mat_k = (dk.median() < -0.10) and ((dk < 0).sum() >= 20)
            pair.append(dict(
                cash_y=y, window=w, arm=arm, d_cagr_med=dc.median(), d_cagr_min=dc.min(),
                d_cagr_max=dc.max(), A_wins_cagr=int((dc < 0).sum()),
                d_calmar_med=dk.median(), A_wins_calmar=int((dk < 0).sum()),
                d_dd_med=ddd.median(), A_wins_dd=int((ddd < 0).sum()),
                d_mean_tr_med=dm.median(), material=bool(mat_c or mat_k)))
    s = pd.DataFrame(summ).round(3)
    p = pd.DataFrame(pair).round(3)
    s.to_csv(RES / 'summary.csv', index=False)
    p.to_csv(RES / 'paired.csv', index=False)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 40)
    print(s.to_string(index=False))
    print()
    print(p.to_string(index=False))

    # gaps on the exit signals: arm A decisions, next available open vs the signal close
    t = pd.read_csv(TRADES_CSV)
    ctx, ir, cfg, trig, lvl, lo = load(head)
    A = t[(t.arm == 'A') & t.reason.isin(['stop', 'target', 'trail'])].copy()
    ev = A.drop_duplicates(['col', 'xi', 'reason'])[['col', 'xi', 'reason']].copy()
    O, C = ctx.O, ctx.C
    T = len(ctx.dates)
    gaps = []
    for c, xi in zip(ev.col.values, ev.xi.values):
        j = xi + 1
        while j < T and not (np.isfinite(O[j, c]) and O[j, c] > 0):
            j += 1
        gaps.append(O[j, c] / C[xi, c] - 1 if j < T else np.nan)
    ev['gap'] = gaps
    ev = ev.dropna()
    A2 = A.drop(columns=['gap']).merge(ev[['col', 'xi', 'reason', 'gap']], on=['col', 'xi', 'reason'])

    def dist(x):
        x = np.asarray(x) * 100
        return dict(n=len(x), mean=x.mean(), median=np.median(x), p10=np.percentile(x, 10),
                    p90=np.percentile(x, 90), share_below_m2=(x < -2).mean() * 100,
                    share_below_m5=(x < -5).mean() * 100, share_above_p2=(x > 2).mean() * 100,
                    worst=x.min(), best=x.max())
    rows = []
    for basis, frame in (('unique_events', ev), ('seed_trades', A2)):
        rows.append(dict(basis=basis, reason='all', **dist(frame.gap)))
        for r, g in frame.groupby('reason'):
            rows.append(dict(basis=basis, reason=r, **dist(g.gap)))
    gd = pd.DataFrame(rows).round(3)
    gd.to_csv(RES / 'gaps_by_reason.csv', index=False)
    print()
    print(gd.to_string(index=False))
    # by half
    ev['half'] = np.where(ctx.dates[ev.xi.values] < pd.Timestamp('2016-01-01'), '2006-2015',
                          '2016-2026')
    hh = [dict(half=hname, reason=r, **dist(g.gap)) for (hname, r), g in
          ev.groupby(['half', 'reason'])]
    print(pd.DataFrame(hh).round(3).to_string(index=False))
    pd.DataFrame(hh).round(3).to_csv(RES / 'gaps_by_reason_half.csv', index=False)
    # rupee attribution: overnight gap x exit value, summed per seed, median seed
    A2['rs'] = A2.qty * A2.sell * A2.gap
    att = A2.groupby(['seed', 'reason']).rs.sum().unstack()
    print('\nrupee effect of the overnight gap by reason, Rs over 20y, on a Rs10L start '
          '(median / min / max over 30 seeds):')
    print(pd.DataFrame(dict(median=att.median(), min=att.min(), max=att.max())).round(0)
          .to_string())
    att.to_csv(RES / 'gap_rupees_by_reason_seed.csv')
    # realised sell vs signal close, and floor behaviour
    fl_rows = []
    for arm in ('B', 'C', 'C2'):
        X = t[(t.arm == arm) & t.reason.isin(['stop', 'target', 'trail'])]
        X = X[X.extra_days >= 0]
        for r, g in list(X.groupby('reason')) + [('all', X)]:
            hit = g[g.floor_misses > 0]
            fl_rows.append(dict(
                arm=arm, reason=r, exits_all_seeds=len(g),
                sell_vs_close_mean_pct=100 * g.cost_vs_close.mean(),
                sell_vs_close_median_pct=100 * g.cost_vs_close.median(),
                floor_hit_exits=len(hit), floor_hit_share_pct=100 * len(hit) / max(len(g), 1),
                extra_days_mean_on_hits=hit.extra_days.mean() if len(hit) else 0.0,
                extra_days_max=int(g.extra_days.max()) if len(g) else 0,
                floor_hit_sell_vs_close_mean_pct=100 * hit.cost_vs_close.mean()
                if len(hit) else np.nan))
    fr = pd.DataFrame(fl_rows).round(3)
    fr.to_csv(RES / 'floor_stats.csv', index=False)
    print()
    print(fr.to_string(index=False))


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'run'
    use_head = 'head' in sys.argv[2:]
    {'check': check, 'run': run, 'report': report}[what](use_head)
