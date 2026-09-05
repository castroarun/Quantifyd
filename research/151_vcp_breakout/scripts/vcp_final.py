"""P9 — run the ADOPTED spec, 30 seeds, and write the required deliverables.

Outputs (consumed by study r/154 and by the report):
  results/vcp_equity_seeds.csv   daily equity curves, index=date, one column per seed
                                 (30 seeds, adopted spec, AFTER TAX, cash_yield 0.05)
  results/vcp_adopted_spec.json  the spec in words and parameters
  results/vcp_yearly.csv         per-seed yearly returns + intra-year max drawdowns
  results/vcp_cost_ladder.csv    25 / 40 / 60 bps per side
  results/vcp_robustness.csv     outlier deletion + winner caps + two-window split
"""
import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vcp_frames import load                                        # noqa: E402
from vcp_replay import Cfg, build_signal, weak_array, simulate, stats, CAPITAL  # noqa: E402

STUDY = Path(__file__).resolve().parents[1]
RES = STUDY / 'results'


def run(F, dates, symbols, meta, cfg, seeds):
    TRIG, PIV, TRAIL, RS = build_signal(F, dates, symbols, meta, cfg)
    weak = weak_array(F, dates, symbols, cfg)
    C, H, O = F['close'], F['high'], F['open']
    di = np.array([i for i, d in enumerate(dates)
                   if pd.Timestamp(cfg.start) <= d <= pd.Timestamp(cfg.end)])
    du = dates[di]
    curves, allst, alltr = {}, [], {}
    for s in seeds:
        eq, tr, _ = simulate(s, cfg, di, dates, C, H, O, PIV, TRAIL, TRIG, RS, weak)
        st, e = stats(eq, du, tr)
        st['seed'] = s
        curves[f'seed{s}'] = e
        allst.append(st)
        alltr[s] = tr
    return pd.DataFrame(curves), pd.DataFrame(allst), alltr, du


def band(d, col):
    return f'{d[col].median():.2f} [{d[col].min():.2f}..{d[col].max():.2f}]'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pivot-n', type=int, default=30)
    ap.add_argument('--exit', default='sma50')
    ap.add_argument('--stop', type=float, default=0.08)
    ap.add_argument('--slots', type=int, default=16)
    ap.add_argument('--sizing', default='fixed')
    ap.add_argument('--size-pct', type=float, default=0.0625)
    ap.add_argument('--risk-pct', type=float, default=0.02)
    ap.add_argument('--near', type=float, default=0.20)
    ap.add_argument('--gate', default='')
    ap.add_argument('--start', default='2006-01-01')
    ap.add_argument('--end', default='2026-09-01')
    ap.add_argument('--seeds', type=int, default=30)
    a = ap.parse_args()

    F, dates, symbols, meta = load()
    cfg = Cfg(pivot_n=a.pivot_n, exit_kind=a.exit, stop_pct=a.stop, slots=a.slots,
              sizing=a.sizing, size_pct=a.size_pct, risk_pct=a.risk_pct,
              near_pct=a.near, gate=a.gate, cost_bps=25.0, fill='realistic',
              tax=True, cash_yield=0.05, start=a.start, end=a.end)
    seeds = list(range(1, a.seeds + 1))
    print('ADOPTED SPEC:', json.dumps(asdict(cfg)), flush=True)

    curves, sdf, alltr, du = run(F, dates, symbols, meta, cfg, seeds)
    curves.to_csv(RES / 'vcp_equity_seeds.csv')
    print(f'wrote vcp_equity_seeds.csv {curves.shape}', flush=True)
    print(f"CAGR {band(sdf,'cagr')}  DD {band(sdf,'dd')}  Calmar {band(sdf,'calmar')}  "
          f"trades {int(sdf.n.median())}  win {sdf.win.median():.1f}%  "
          f"avgwin {sdf.avg_win.median():.1f}%  avgloss {sdf.avg_loss.median():.1f}%  "
          f"mean/tr {sdf['mean'].median():.2f}%  streak {int(sdf.streak.median())}  "
          f"tpy {sdf.tpy.median():.0f}", flush=True)

    spec = dict(
        study='research/151_vcp_breakout',
        name='VCP breakout (bananapatterns VCP screen, decoded approximation)',
        words=("Long-only NSE cash book. A stock qualifies when its CLOSE breaks above the "
               f"highest close of the prior {cfg.pivot_n} trading days (the screen's pattern "
               "pivot), while the previous close sat below that pivot and within "
               f"{cfg.near_pct*100:.0f}% of it, the 20-day median traded value is at least "
               "Rs 5 crore as of the previous day, ETFs are excluded, and the IBD-weighted "
               "relative-strength percentile (2*r63 + r126 + r189 + r252, ranked across "
               f"eligible names) is at least {cfg.rs_min:.0f}. The order is a buy-stop at the "
               "pivot, filled at max(pivot, open). A position is cut when the CLOSE falls "
               f"{cfg.stop_pct*100:.0f}% below the fill, and winners are trailed out at the "
               f"first CLOSE below the {cfg.exit_kind.replace('sma','')}-day simple moving "
               f"average. {cfg.slots} slots at "
               + (f"{cfg.size_pct*100:.2f}% of NAV each" if cfg.sizing == 'fixed'
                  else f"risk {cfg.risk_pct*100:.1f}% / stop {cfg.stop_pct*100:.0f}% of NAV, "
                       f"capped at {cfg.cap_pct*100:.0f}%")
               + ", cash-constrained, no leverage, no market-regime gate."),
        params=asdict(cfg),
        basis=dict(costs_bps_per_side=cfg.cost_bps, tax='20% STCG / 12.5% LTCG, Indian FY '
                                                        'loss-netting with carry-forward',
                   idle_cash_yield=cfg.cash_yield, fills='realistic max(pivot, open)',
                   seeds=len(seeds), window=f'{cfg.start}..{cfg.end}',
                   selection='random among same-day candidates when slots are scarce'),
        results=dict(cagr_median=round(float(sdf.cagr.median()), 2),
                     cagr_min=round(float(sdf.cagr.min()), 2),
                     cagr_max=round(float(sdf.cagr.max()), 2),
                     maxdd_median=round(float(sdf.dd.median()), 2),
                     maxdd_worst=round(float(sdf.dd.min()), 2),
                     calmar_median=round(float(sdf.calmar.median()), 3),
                     trades_median=int(sdf.n.median()),
                     win_rate_median=round(float(sdf.win.median()), 1)),
        caveats=["Survivorship: Kite lists only current instruments; delisted names absent.",
                 "The site's VCP definition is unpublished; this is the best-matching "
                 "approximation (62% joint trade match on their 40-trade ground truth).",
                 "Pre-2015 coverage is thin (2006 ~528 priced symbols) so the early window "
                 "is survivorship-flattered."])
    json.dump(spec, open(RES / 'vcp_adopted_spec.json', 'w'), indent=2)
    print('wrote vcp_adopted_spec.json', flush=True)

    # yearly + intra-year drawdowns
    yr = pd.DataFrame([s['yearly'] for s in sdf.to_dict('records')])
    iy = pd.DataFrame([s['intra_dd'] for s in sdf.to_dict('records')])
    out = pd.DataFrame({'ret_median': yr.median().round(1), 'dd_median': iy.median().round(1),
                        'ret_min': yr.min().round(1), 'ret_max': yr.max().round(1)})
    out.to_csv(RES / 'vcp_yearly.csv')
    print('\nyearly medians (return / intra-year DD):')
    print(out.to_string())

    # cost ladder
    rows = []
    for bps in (25.0, 40.0, 60.0):
        _, s2, _, _ = run(F, dates, symbols, meta, replace(cfg, cost_bps=bps), seeds[:10])
        rows.append(dict(cost_bps=bps, cagr_med=round(s2.cagr.median(), 2),
                         dd_med=round(s2.dd.median(), 2),
                         calmar=round(s2.calmar.median(), 3)))
        print('cost', bps, rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(RES / 'vcp_cost_ladder.csv', index=False)

    # robustness: winner caps, top-trade deletion, two windows
    rob = []
    for tag, kw in (('base', {}), ('cap_win_50', dict(capwin=0.50)),
                    ('cap_win_100', dict(capwin=1.00)), ('drop_top10', dict(drop_topn=10)),
                    ('win_2006_2015', dict(end='2015-12-31')),
                    ('win_2016_2026', dict(start='2016-01-01'))):
        _, s2, _, _ = run(F, dates, symbols, meta, replace(cfg, **kw), seeds[:10])
        rob.append(dict(arm=tag, cagr_med=round(s2.cagr.median(), 2),
                        cagr_min=round(s2.cagr.min(), 2),
                        dd_med=round(s2.dd.median(), 2),
                        calmar=round(s2.calmar.median(), 3), n=int(s2.n.median())))
        print('robust', tag, rob[-1], flush=True)
    pd.DataFrame(rob).to_csv(RES / 'vcp_robustness.csv', index=False)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
