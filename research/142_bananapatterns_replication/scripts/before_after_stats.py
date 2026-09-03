"""BEFORE vs AFTER the 03-Sep-2026 spec revision, both with 5% p.a. on idle cash.

BEFORE (as first went live 02-Sep): trail-20, -8% stop, 8 slots @18.75%,
NIFTYBEES SMA200 gate ON (computed correctly - the intended spec).
AFTER  (03-Sep adopted):            trail-15, -8% stop, 16 slots @6.25%, NO gate.

30 paired seeds, 2006->now, pre-tax and after-tax rows. Metrics: CAGR, maxDD,
Calmar, win ratio, trades/yr, % days with no trade activity, % days fully in
cash. Idle cash earns 5% p.a. in BOTH configs (new cash_yield model).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / 'scripts'))
import bluesky_replay as br

print('loading frames ...', flush=True)
w = br.load_frames('2004-06-01', trail_sma=20)
close, high, open_, athcp, sma20f, tv20 = (w[k] for k in
    ('close', 'high', 'open', 'athcp', 'sma50', 'tv20'))
etf = [c for c in close.columns if br.ETF_RE.search(c)]
tv_prev = tv20.shift(1)
prev_close = close.shift(1)
elig = tv_prev >= br.TV_FLOOR
elig[etf] = False
score = 2*(close/close.shift(63)-1) + (close/close.shift(126)-1) \
    + (close/close.shift(189)-1) + (close/close.shift(252)-1)
rs = (score.where(elig).rank(axis=1, pct=True)*100).shift(1)
setup = (prev_close < athcp) & (prev_close >= 0.8*athcp) & elig & (rs >= 70.0)
trig = (setup & (close > athcp) & athcp.notna()).fillna(False).values
dates = close.index
C, H, O, ATH = close.values, high.values, open_.values, athcp.values
RSv, TVv = rs.values, tv_prev.values
days = np.array([i for i, d in enumerate(dates) if str(d.date()) >= '2006-01-01'])
n_yrs = len(days) / 247.0
nb = close['NIFTYBEES'].dropna()
weak_on = (nb < nb.rolling(200).mean()).shift(1).reindex(dates)\
    .ffill().fillna(False).astype(bool).values
weak_off = np.zeros(len(dates), dtype=bool)
S15 = close.rolling(15).mean().values
S20 = sma20f.values

CONFIGS = {
    'BEFORE (trail20/8slots/gate)': dict(S=S20, weak=weak_on, slots=8, size=0.1875),
    'AFTER (trail15/16slots/nogate)': dict(S=S15, weak=weak_off, slots=16, size=0.0625),
}

rows = []
for tax in (0.0, 0.20):
    for name, cfg in CONFIGS.items():
        cagrs, dds, wins, ntr, noact, allcash = [], [], [], [], [], []
        for seed in range(1, 31):
            eq, trades, _ = br.simulate(seed, 'random', days, dates, C, H, O, ATH,
                                        cfg['S'], RSv, TVv, trig, cfg['weak'], True,
                                        0.0025, stop=0.08, slots=cfg['slots'],
                                        size_pct=cfg['size'], stcg=tax,
                                        cash_yield=0.05)
            eq = np.asarray(eq, dtype=float)
            cagrs.append(((eq[-1]/eq[0])**(1/n_yrs)-1)*100)
            dds.append(float((eq/np.maximum.accumulate(eq)-1).min()*100))
            closed = [t for t in trades if t[5] != 'open_marked']
            rets = [t[4]/t[3]-1 for t in closed]
            wins.append(100*np.mean([r > 0 for r in rets]))
            ntr.append(len(closed))
            # activity + exposure day counts
            act = set()
            pos_days = np.zeros(len(days), dtype=int)
            di = {i: k for k, i in enumerate(days)}
            for c_, ei, xi, *_ in [(t[0], t[1], t[2]) for t in trades]:
                act.add(ei); act.add(xi)
                pos_days[di[ei]:di.get(xi, len(days)-1)+1] += 1
            noact.append(100*(1 - len(act & set(days))/len(days)))
            allcash.append(100*(pos_days == 0).mean())
        rows.append(dict(tax='after-tax' if tax else 'pre-tax', config=name,
                         cagr=np.median(cagrs), cagr_worst=min(cagrs),
                         dd=np.median(dds),
                         calmar=np.median(cagrs)/abs(np.median(dds)),
                         win_pct=np.median(wins), trades=int(np.median(ntr)),
                         trades_yr=round(np.median(ntr)/n_yrs, 1),
                         days_no_trade_pct=round(float(np.median(noact)), 1),
                         days_all_cash_pct=round(float(np.median(allcash)), 1)))
        r = rows[-1]
        print(f"{r['tax']:9s} {name:32s} CAGR {r['cagr']:.1f}% (worst {r['cagr_worst']:.1f}) "
              f"dd {r['dd']:.1f}% Calmar {r['calmar']:.2f} win {r['win_pct']:.1f}% "
              f"n {r['trades']} ({r['trades_yr']}/yr) no-trade-days {r['days_no_trade_pct']}% "
              f"all-cash-days {r['days_all_cash_pct']}%", flush=True)

pd.DataFrame(rows).round(2).to_csv(STUDY / 'results' / 'before_after_stats.csv', index=False)
print('DONE', flush=True)
