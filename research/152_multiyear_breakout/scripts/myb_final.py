"""research/152 — Phase D/E: 30-seed adoption run, robustness battery, OA trade overlap,
correlation + 3-sleeve blend vs the deployed TN+OA book, and the r/154 hand-off files.

Emits (all after-tax, net of costs, cash_yield 5%):
  results/myb_equity_seeds.csv    daily equity, index=date, one column per seed  [for r/154]
  results/myb_adopted_spec.json   the adopted spec                               [for r/154]
  results/final_robustness.csv    cost ladder / outlier deletion / per-window rows
  results/final_yoy.csv           per-year median return + intra-year max drawdown
  results/oa_overlap.csv          signal-date AND holding overlap with Open Alpha
  results/blend152.csv            3-sleeve blend sweep vs TN+OA 50-50 + cash-null
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import myb_replay as M                       # noqa: E402
import myb_sweep as S                        # noqa: E402

RES = Path(__file__).resolve().parents[1] / 'results'
R146 = Path('/home/arun/quantifyd/research/146_complementary_third_sleeve/results')
CRASH = {'2008': ('2008-01-01', '2009-03-31'), '2015-16': ('2015-08-01', '2016-02-29'),
         '2018': ('2018-01-01', '2018-10-31'), '2020crash': ('2020-02-01', '2020-04-30'),
         '2022H1': ('2022-01-01', '2022-06-30')}
W3 = [0.0, 0.10, 0.15, 0.20, 0.25, 0.33]


def nav_stats(nav):
    y = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / y) - 1
    dd = float((nav / nav.cummax() - 1).min())
    return round(cagr * 100, 2), round(dd * 100, 2), (round(cagr / abs(dd), 2) if dd < 0 else np.nan)


def intra_year_dd(nav):
    out = {}
    for yr, seg in nav.groupby(nav.index.year):
        out[int(yr)] = round(float((seg / seg.cummax() - 1).min() * 100), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', required=True)
    ap.add_argument('--stop', type=float, default=0.08)
    ap.add_argument('--trail', type=int, default=50)
    ap.add_argument('--take-profit', type=float, default=None)
    ap.add_argument('--slots', type=int, default=16)
    ap.add_argument('--size', type=float, default=0.0625)
    ap.add_argument('--risk', type=float, default=None)
    ap.add_argument('--gate', action='store_true')
    ap.add_argument('--fill-close', action='store_true')
    ap.add_argument('--maxdist', type=float, default=0.20)
    ap.add_argument('--tight', type=float, default=None)
    ap.add_argument('--seeds', type=int, default=30)
    ap.add_argument('--window', default='W2')
    ap.add_argument('--skip-blend', action='store_true')
    a = ap.parse_args()

    ctx = S.Ctx()
    n, level, athvar, age = S.parse_family(a.family)
    piv, olds, bo = S.signal_cache(ctx, n, level)
    ctx.PIV_CUR = piv.values.astype('float32')
    nrows_ok = (ctx.cm['nrows'] >= int(n * M.YR)).values
    trig = S.build_trig(ctx, piv, olds, bo, athvar, age, maxdist=a.maxdist,
                        tight=a.tight) & nrows_ok
    win = {'W1': S.W1, 'W2': S.W2, 'W2B': S.W2B}[a.window]
    idx = ctx.d(win)
    dates_used = ctx.dates[idx]
    weak = ctx.weak_on if a.gate else ctx.weak_off
    trail = ctx.trail(a.trail) if a.trail else None

    spec = dict(study='research/152_multiyear_breakout', family=a.family, n_years=n,
                level=level, ath_variant=athvar, pivot_age_months=age,
                maxdist=a.maxdist, tightness=a.tight, stop_pct=a.stop, trail_sma=a.trail,
                take_profit=a.take_profit, slots=a.slots, size_pct=a.size,
                risk_pct=a.risk, cap_pct=M.CAP_PCT, market_gate=a.gate,
                fill='signal_close' if a.fill_close else 'buy_stop_at_pivot',
                cost_bps_per_side=25, tax='20% STCG / 12.5% LTCG, Indian FY netting',
                cash_yield=0.05, capital=M.CAPITAL, window=list(win), seeds=a.seeds,
                rs_min=70, liquidity_floor_inr=5e7,
                data_snapshot='market_data.db 2026-09-04', built=time.strftime('%Y-%m-%d'))
    (RES / 'myb_adopted_spec.json').write_text(json.dumps(spec, indent=2))

    # ── 30-seed ensemble ──
    navs, allstats, all_trades = {}, [], {}
    for seed in range(1, a.seeds + 1):
        eq, tr, _ = M.simulate_ext(seed, idx, ctx.dates, ctx.C, ctx.O, ctx.PIV_CUR,
                                   trail, trig, weak, cost=0.0025, stop=a.stop,
                                   slots=a.slots, size_pct=a.size, risk_pct=a.risk,
                                   take_profit=a.take_profit, fill_close=a.fill_close)
        st, e = M.stats_from(eq, dates_used, tr, dates=ctx.dates)
        navs[f's{seed}'] = e
        allstats.append(st)
        all_trades[seed] = tr
        print(f'seed {seed:2d}: CAGR {st["cagr"]:6.2f}%  DD {st["dd"]:6.1f}%  '
              f'n={st["n"]:4d} win {st["win"]:.0f}%', flush=True)
    eqdf = pd.DataFrame(navs)
    eqdf.index.name = 'date'
    eqdf.to_csv(RES / 'myb_equity_seeds.csv')
    cg = [s['cagr'] for s in allstats]
    dd = [s['dd'] for s in allstats]
    print(f'\nENSEMBLE {a.seeds} seeds: CAGR median {np.median(cg):.2f} '
          f'[{min(cg):.2f}..{max(cg):.2f}]  DD median {np.median(dd):.1f} '
          f'worst {min(dd):.1f}  Calmar {np.median(cg)/abs(np.median(dd)):.2f}', flush=True)

    rob = []
    rob.append(dict(test='adopted 25bps', cagr_med=round(float(np.median(cg)), 2),
                    cagr_min=round(min(cg), 2), cagr_max=round(max(cg), 2),
                    dd_med=round(float(np.median(dd)), 2), dd_worst=round(min(dd), 2),
                    calmar=round(float(np.median(cg)) / abs(float(np.median(dd))), 2),
                    trades=int(np.median([s['n'] for s in allstats])),
                    win=round(float(np.median([s['win'] for s in allstats])), 1),
                    mean_tr=round(float(np.median([s['mean'] for s in allstats])), 3),
                    streak=int(np.median([s['max_lose_streak'] for s in allstats]))))

    # ── cost ladder (10 seeds each) ──
    for c in (0.0040, 0.0060):
        cs, ds = [], []
        for seed in range(1, 11):
            eq, tr, _ = M.simulate_ext(seed, idx, ctx.dates, ctx.C, ctx.O, ctx.PIV_CUR,
                                       trail, trig, weak, cost=c, stop=a.stop,
                                       slots=a.slots, size_pct=a.size, risk_pct=a.risk,
                                       take_profit=a.take_profit, fill_close=a.fill_close)
            st, _e = M.stats_from(eq, dates_used, tr, dates=ctx.dates)
            cs.append(st['cagr']); ds.append(st['dd'])
        rob.append(dict(test=f'cost {int(c*10000)}bps', cagr_med=round(float(np.median(cs)), 2),
                        cagr_min=round(min(cs), 2), cagr_max=round(max(cs), 2),
                        dd_med=round(float(np.median(ds)), 2), dd_worst=round(min(ds), 2),
                        calmar=round(float(np.median(cs)) / abs(float(np.median(ds))), 2)))
        print(f'cost {int(c*10000)}bps: CAGR {np.median(cs):.2f}%', flush=True)

    # ── outlier dependence: drop top-10 trades / cap winners ──
    def replay_capped(cap=None, drop_top=0):
        outs = []
        for seed in range(1, 11):
            tr = all_trades[seed]
            r = np.array([t[4] / t[3] - 1 for t in tr])
            if drop_top:
                keep = np.argsort(-r)[drop_top:]
                r = r[keep]
            if cap is not None:
                r = np.minimum(r, cap)
            outs.append(r)
        return outs

    for label, kw in (('drop top-10 trades', dict(drop_top=10)),
                      ('cap winners +50%', dict(cap=0.50)),
                      ('cap winners +100%', dict(cap=1.00))):
        rs_ = replay_capped(**kw)
        base = replay_capped()
        d_mean = float(np.median([r.mean() for r in rs_])) * 100
        b_mean = float(np.median([r.mean() for r in base])) * 100
        rob.append(dict(test=label, mean_tr=round(d_mean, 3),
                        note=f'base mean/trade {b_mean:.3f}%'))
        print(f'{label}: mean/trade {d_mean:.3f}% vs base {b_mean:.3f}%', flush=True)

    # ── per-window behaviour ──
    for wname, (s_, e_) in CRASH.items():
        vals = []
        for c in eqdf.columns:
            seg = eqdf[c][(eqdf.index >= s_) & (eqdf.index <= e_)]
            if len(seg) > 2:
                vals.append((float(seg.iloc[-1] / seg.iloc[0] - 1) * 100,
                             float((seg / seg.cummax() - 1).min() * 100)))
        if vals:
            rob.append(dict(test=f'window {wname}',
                            ret_med=round(float(np.median([v[0] for v in vals])), 1),
                            dd_med=round(float(np.median([v[1] for v in vals])), 1)))
            print(f'window {wname}: ret {np.median([v[0] for v in vals]):.1f}% '
                  f'dd {np.median([v[1] for v in vals]):.1f}%', flush=True)
    pd.DataFrame(rob).to_csv(RES / 'final_robustness.csv', index=False)

    # ── YoY: median annual return + median intra-year max drawdown ──
    yrs = sorted(set(eqdf.index.year))
    rows = []
    for y in yrs:
        rets, dds = [], []
        for c in eqdf.columns:
            s_ = eqdf[c][eqdf.index.year == y]
            if len(s_) < 5:
                continue
            prev = eqdf[c][eqdf.index.year < y]
            start = prev.iloc[-1] if len(prev) else M.CAPITAL
            rets.append(float(s_.iloc[-1] / start - 1) * 100)
            run = pd.concat([pd.Series([start], index=[s_.index[0]]), s_])
            dds.append(float((run / run.cummax() - 1).min() * 100))
        rows.append(dict(year=int(y), ret_med=round(float(np.median(rets)), 1),
                         dd_med=round(float(np.median(dds)), 1)))
    pd.DataFrame(rows).to_csv(RES / 'final_yoy.csv', index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)

    # ── Open Alpha overlap: signal-date AND holding overlap ──
    oa_trig = ctx.OA_TRIG
    sub = trig[idx]
    oas = oa_trig[idx]
    sig_overlap = 100 * float((sub & oas).sum()) / max(1, int(sub.sum()))
    # holding overlap: run OA's adopted spec on the same seeds and compare held (sym, day)
    oa_pivot = ctx.cm['athp'].values.astype('float32')
    trail15 = ctx.trail(15)
    hold_rows = []
    for seed in (1, 2, 3, 4, 5):
        myb_hold = held_matrix(ctx, idx, trig, ctx.PIV_CUR, trail, seed, a)
        save_piv = ctx.PIV_CUR
        oa_hold = held_matrix(ctx, idx, oa_trig, oa_pivot, trail15, seed,
                              argparse.Namespace(stop=0.08, slots=16, size=0.0625,
                                                 risk=None, take_profit=None,
                                                 fill_close=False, gate=False))
        ctx.PIV_CUR = save_piv
        inter = int((myb_hold & oa_hold).sum())
        hold_rows.append(dict(seed=seed, myb_holddays=int(myb_hold.sum()),
                              oa_holddays=int(oa_hold.sum()), both=inter,
                              pct_of_myb=round(100 * inter / max(1, int(myb_hold.sum())), 1)))
        print(f'holding overlap seed {seed}: {hold_rows[-1]}', flush=True)
    pd.DataFrame([dict(metric='signal_date_overlap_pct_of_myb', value=round(sig_overlap, 1))]
                 + hold_rows).to_csv(RES / 'oa_overlap.csv', index=False)

    if not a.skip_blend:
        blend_stage(eqdf)
    print('FINAL DONE', flush=True)


def held_matrix(ctx, idx, trig, pivot, trail, seed, a):
    """Boolean (day x symbol) matrix of positions HELD, for the overlap measurement."""
    ctx.PIV_CUR = pivot
    eq, tr, _ = M.simulate_ext(seed, idx, ctx.dates, ctx.C, ctx.O, pivot, trail, trig,
                               ctx.weak_on if a.gate else ctx.weak_off, cost=0.0025,
                               stop=a.stop, slots=a.slots, size_pct=a.size,
                               risk_pct=a.risk, take_profit=a.take_profit,
                               fill_close=a.fill_close)
    hold = np.zeros((len(ctx.dates), ctx.C.shape[1]), dtype=bool)
    for c, ei, xi, _b, _s, _r in tr:
        hold[ei:xi + 1, c] = True
    return hold[idx]


def blend_stage(eqdf):
    """Correlation + 3-sleeve blend vs the deployed TN+OA 50-50 book, plus a cash-null."""
    oa = pd.read_csv(R146 / 'oa_navs.csv', index_col=0, parse_dates=True)
    tns = {o: pd.read_csv(R146 / f'tn_nav_off{o}.csv', index_col=0,
                          parse_dates=True).iloc[:, 0] for o in (0, 4, 8)
           if (R146 / f'tn_nav_off{o}.csv').exists()}
    cand = eqdf.median(axis=1)
    cash = pd.Series((1 + 0.05 / 252) ** np.arange(len(cand)), index=cand.index)

    cr = []
    for name, other in [('OA', oa.median(axis=1))] + [(f'TN_off{o}', v) for o, v in tns.items()]:
        i = cand.index.intersection(other.index)
        cr.append(dict(vs=name,
                       corr_daily=round(float(cand.loc[i].pct_change().corr(
                           other.loc[i].pct_change())), 3),
                       corr_monthly=round(float(
                           cand.loc[i].resample('ME').last().pct_change().corr(
                               other.loc[i].resample('ME').last().pct_change())), 3)))
        print(cr[-1], flush=True)

    def blend(o_nav, t_nav, c_nav, w3):
        i = o_nav.index.intersection(t_nav.index)
        if c_nav is not None:
            i = i.intersection(c_nav.index)
        legs = [o_nav.loc[i], t_nav.loc[i]] + ([c_nav.loc[i]] if c_nav is not None else [])
        m = [x.resample('ME').last().pct_change().fillna(0) for x in legs]
        wl = (1 - w3) / 2
        r = wl * m[0] + wl * m[1] + (w3 * m[2] if c_nav is not None else 0)
        return (1 + r).cumprod()

    rows = []
    for third, label in ((eqdf, 'MYB'), (None, 'cashnull')):
        for off, tn in tns.items():
            for w3 in W3:
                cs, ds, ks = [], [], []
                for j, col in enumerate(oa.columns):
                    cn = (eqdf.iloc[:, j % eqdf.shape[1]] if label == 'MYB' else cash) \
                        if w3 > 0 else None
                    b = blend(oa[col], tn, cn, w3)
                    c_, d_, k_ = nav_stats(b)
                    cs.append(c_); ds.append(d_); ks.append(k_)
                rows.append(dict(cand=label, offset=off, w3=w3,
                                 cagr_med=round(float(np.median(cs)), 2),
                                 cagr_min=round(min(cs), 2),
                                 dd_med=round(float(np.median(ds)), 2),
                                 dd_worst=round(min(ds), 2),
                                 calmar_med=round(float(np.median(ks)), 2)))
                if off == 0:
                    print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(RES / 'blend152.csv', index=False)
    pd.DataFrame(cr).to_csv(RES / 'corr152.csv', index=False)


if __name__ == '__main__':
    main()
