"""Is their -14.8% 'worst fall' reachable on ANY honest path? Report the full DD spread."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vcp_frames import load                                        # noqa: E402
from vcp_replay import Cfg, build_signal, weak_array, simulate, stats  # noqa: E402

F, dates, symbols, meta = load()
for tag, cfg in (
        ('their dials, honest', Cfg(pivot_n=30, exit_kind='sma50', stop_pct=0.07, slots=5,
                                    sizing='risk', risk_pct=0.02, cost_bps=25.0,
                                    fill='realistic', tax=True, cash_yield=0.05,
                                    start='2020-01-01', end='2025-12-31')),
        ('their dials, their fills, no cost/tax',
         Cfg(pivot_n=30, exit_kind='sma50', stop_pct=0.07, slots=5, sizing='risk',
             risk_pct=0.02, cost_bps=0.0, fill='pivot', tax=False, cash_yield=0.0,
             start='2020-01-01', end='2025-12-31'))):
    TRIG, PIV, TRAIL, RS = build_signal(F, dates, symbols, meta, cfg)
    weak = weak_array(F, dates, symbols, cfg)
    di = np.array([i for i, d in enumerate(dates)
                   if pd.Timestamp(cfg.start) <= d <= pd.Timestamp(cfg.end)])
    dd, mdd, cg = [], [], []
    for s in range(1, 31):
        eq, tr, _ = simulate(s, cfg, di, dates, F['close'], F['high'], F['open'],
                             PIV, TRAIL, TRIG, RS, weak)
        st, e = stats(eq, dates[di], tr)
        dd.append(st['dd'])
        cg.append(st['cagr'])
        m = e.resample('ME').last()
        mdd.append(float((m / m.cummax() - 1).min()) * 100)
    print(f'{tag}: daily DD best {max(dd):.1f}% median {np.median(dd):.1f}% worst {min(dd):.1f}% | '
          f'MONTHLY-marked DD best {max(mdd):.1f}% median {np.median(mdd):.1f}% | '
          f'CAGR median {np.median(cg):.1f}%', flush=True)
