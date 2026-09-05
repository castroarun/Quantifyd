"""P8 — portfolio fit: correlation to the deployed legs and the 3-sleeve blend test.

Baseline to beat = the DEPLOYED pair: True North + Open Alpha, 50-50, monthly rebalanced,
after tax. Uses the same cached after-tax NAVs research/146 and research/147 used, so the
comparison is like-for-like:
  research/146_.../results/oa_navs.csv        10 Open Alpha seeds  (adopted OA spec)
  research/146_.../results/tn_nav_off{0,4,8}.csv   3 True North rebalance-day offsets
Candidate = research/151 results/vcp_equity_seeds.csv (30 seeds; the first 10 are paired
with the 10 OA seeds so every comparison is PAIRED, never an unpaired median).

Pre-registered adoption bar (STATUS §2.5): +0.10 Calmar or -2pp drawdown at >= equal CAGR,
correlation < 0.4 to both legs, and it must beat the CASH-NULL at the same weight.
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[1]
RES = STUDY / 'results'
R146 = Path('/home/arun/quantifyd/research/146_complementary_third_sleeve/results')
WEIGHTS = [0.10, 0.15, 0.20, 0.25, 0.33]
WINDOWS = {'2008crash': ('2008-01-01', '2009-03-31'), '2018grind': ('2018-01-01', '2018-10-31'),
           '2020crash': ('2020-02-01', '2020-04-30'), '2022H1grind': ('2022-01-01', '2022-06-30')}
CASH_Y = 0.05


def st(nav):
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1
    dd = float((nav / nav.cummax() - 1).min())
    return cagr * 100, dd * 100, (cagr * 100) / abs(dd * 100) if dd < 0 else np.nan


def wstats(nav, a, b):
    s = nav[(nav.index >= a) & (nav.index <= b)]
    if len(s) < 5:
        return np.nan, np.nan
    return float(s.iloc[-1] / s.iloc[0] - 1) * 100, float((s / s.cummax() - 1).min()) * 100


def main():
    oa = pd.read_csv(R146 / 'oa_navs.csv', index_col=0, parse_dates=True)
    tn = {o: pd.read_csv(R146 / f'tn_nav_off{o}.csv', index_col=0,
                         parse_dates=True).iloc[:, 0].dropna() for o in (0, 4, 8)}
    vc = pd.read_csv(RES / 'vcp_equity_seeds.csv', index_col=0, parse_dates=True)
    idx = oa.index
    for s in list(tn.values()) + [vc]:
        idx = idx.intersection(s.index)
    idx = idx.sort_values()
    print(f'common window {idx[0].date()} .. {idx[-1].date()}  ({len(idx)} days)')

    oa_n = [oa[c].loc[idx] for c in oa.columns]
    tn_n = [tn[o].loc[idx] for o in (0, 4, 8)]
    vc_n = [vc[c].loc[idx] for c in vc.columns]

    # ---------- correlations (daily and monthly returns), paired seed-by-seed
    def corr(a, b, monthly=False):
        if monthly:
            a = a.resample('ME').last().pct_change().dropna()
            b = b.resample('ME').last().pct_change().dropna()
        else:
            a, b = a.pct_change().dropna(), b.pct_change().dropna()
        j = a.index.intersection(b.index)
        return float(a.loc[j].corr(b.loc[j]))

    cor = []
    for i in range(min(len(oa_n), len(vc_n))):
        cor.append(dict(pair='VCP~OA', seed=i + 1, daily=corr(vc_n[i], oa_n[i]),
                        monthly=corr(vc_n[i], oa_n[i], True)))
    for i, t in enumerate(tn_n):
        cor.append(dict(pair='VCP~TN', seed=i, daily=corr(vc_n[0], t),
                        monthly=corr(vc_n[0], t, True)))
    cor.append(dict(pair='OA~TN', seed=0, daily=corr(oa_n[0], tn_n[0]),
                    monthly=corr(oa_n[0], tn_n[0], True)))
    cdf = pd.DataFrame(cor)
    cdf.to_csv(RES / 'p8_correlations.csv', index=False)
    print('\n=== CORRELATIONS (median across pairings) ===')
    print(cdf.groupby('pair')[['daily', 'monthly']].median().round(3).to_string())

    # ---------- blends on monthly returns, monthly rebalanced
    mo = {k: [n.resample('ME').last().pct_change().fillna(0) for n in v]
          for k, v in (('oa', oa_n), ('tn', tn_n), ('vc', vc_n))}
    midx = mo['oa'][0].index
    cash_m = pd.Series((1 + CASH_Y) ** (1 / 12) - 1, index=midx)

    rows = []
    combos = list(itertools.product(range(len(mo['oa'])), range(len(mo['tn']))))
    for label, w, third in ([('baseline_TN+OA_50-50', 0.0, None)]
                            + [(f'+VCP_{int(w*100)}%', w, 'vcp') for w in WEIGHTS]
                            + [(f'+CASHNULL_{int(w*100)}%', w, 'cash') for w in WEIGHTS]):
        cs, ds, ks, wr = [], [], [], {k: [] for k in WINDOWS}
        for oi, ti in combos:
            leg = (1 - w) / 2
            r = leg * mo['oa'][oi] + leg * mo['tn'][ti]
            if third == 'vcp':
                r = r + w * mo['vc'][oi % len(mo['vc'])]
            elif third == 'cash':
                r = r + w * cash_m
            nav = (1 + r).cumprod()
            c_, d_, k_ = st(nav)
            cs.append(c_)
            ds.append(d_)
            ks.append(k_)
            for wk, (a, b) in WINDOWS.items():
                wr[wk].append(wstats(nav, a, b))
        row = dict(blend=label, weight=w, n_paths=len(cs),
                   cagr_med=round(np.median(cs), 2), cagr_min=round(min(cs), 2),
                   cagr_max=round(max(cs), 2), dd_med=round(np.median(ds), 2),
                   dd_worst=round(min(ds), 2), calmar_med=round(np.median(ks), 3))
        for wk in WINDOWS:
            arr = np.array(wr[wk], dtype=float)
            row[f'{wk}_ret'] = round(float(np.nanmedian(arr[:, 0])), 1)
            row[f'{wk}_dd'] = round(float(np.nanmedian(arr[:, 1])), 1)
        rows.append(row)
        print(f"{label:24s} CAGR {row['cagr_med']:6.2f} [{row['cagr_min']:.1f}..{row['cagr_max']:.1f}] "
              f"DD {row['dd_med']:7.2f} Calmar {row['calmar_med']:.3f}", flush=True)
    bdf = pd.DataFrame(rows)
    bdf.to_csv(RES / 'p8_blend.csv', index=False)

    # ---------- paired deltas vs the baseline (never unpaired medians)
    base = {}
    for oi, ti in combos:
        r = 0.5 * mo['oa'][oi] + 0.5 * mo['tn'][ti]
        base[(oi, ti)] = st((1 + r).cumprod())
    print('\n=== PAIRED delta vs TN+OA 50-50 (per path) ===')
    prs = []
    for w in WEIGHTS:
        dc, dk, dd_, wins = [], [], [], 0
        for oi, ti in combos:
            leg = (1 - w) / 2
            r = leg * mo['oa'][oi] + leg * mo['tn'][ti] + w * mo['vc'][oi % len(mo['vc'])]
            c_, d_, k_ = st((1 + r).cumprod())
            bc, bd, bk = base[(oi, ti)]
            dc.append(c_ - bc)
            dd_.append(d_ - bd)
            dk.append(k_ - bk)
            wins += (k_ > bk)
        prs.append(dict(weight=w, d_cagr_med=round(np.median(dc), 2),
                        d_dd_med=round(np.median(dd_), 2), d_calmar_med=round(np.median(dk), 3),
                        calmar_wins=f'{wins}/{len(combos)}'))
        print(f"w={w:.2f}  dCAGR {prs[-1]['d_cagr_med']:+6.2f}pp  dDD {prs[-1]['d_dd_med']:+6.2f}pp  "
              f"dCalmar {prs[-1]['d_calmar_med']:+.3f}  Calmar wins {prs[-1]['calmar_wins']}")
    pd.DataFrame(prs).to_csv(RES / 'p8_paired.csv', index=False)

    print('\n=== PER-WINDOW (blend medians) ===')
    cols = ['blend'] + [f'{k}_{s}' for k in WINDOWS for s in ('ret', 'dd')]
    print(bdf[cols].to_string(index=False))
    print('\nAdoption bar: +0.10 Calmar OR -2pp DD at >= equal CAGR, corr < 0.4 to both '
          'legs, and beats the cash-null at the same weight.')


if __name__ == '__main__':
    main()
