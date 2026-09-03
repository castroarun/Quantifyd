"""Seed bluesky_paper_state.json from a backtest replay of the adopted spec
(trail-20, no mcap floor, gate ON, real fills, 25bps), 2020-01-01 -> last DB day,
Rs 10L, median-terminal seed of 10 — so /app/bluesky-paper opens as a living model
portfolio. Every backfilled trade carries src='backtest'; live runs append on top.

Provenance is stated in the state and the UI feed (live-first/backfill convention:
sources and date ranges are never blended silently).
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / 'scripts'))
import bluesky_replay as br

START = '2020-01-01'
CAPITAL = 1_000_000
TRAIL = 20
COST = 0.0025
STATE = ROOT / 'backtest_data' / 'bluesky_paper_state.json'
UI_JSON = ROOT / 'static' / 'app' / 'bluesky_paper.json'

w = br.load_frames('2018-06-01', trail_sma=TRAIL)
close, high, open_, athcp, sma, tv20 = (w[k] for k in
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
trig_df = (setup & (close > athcp) & athcp.notna()).fillna(False)
trig = trig_df.values
nb = close['NIFTYBEES'].dropna()   # NaN-robust: phantom holiday rows in the union
weak_s = (nb < nb.rolling(200).mean()).shift(1)   # index must not poison the SMA
weak = weak_s.reindex(close.index).ffill().fillna(False).astype(bool).values
dates = close.index
last_day = dates[-1]
days = np.array([i for i, d in enumerate(dates) if str(d.date()) >= START])
C, H, O, ATH, S = close.values, high.values, open_.values, athcp.values, sma.values

finals = {}
for seed in range(1, 11):
    eq, _, _ = br.simulate(seed, 'random', days, dates, C, H, O, ATH, S, rs.values,
                           tv_prev.values, trig, weak, True, COST, stop=0.08, slots=8)
    finals[seed] = eq[-1]
med = min(finals, key=lambda s: abs(finals[s] - np.median(list(finals.values()))))
print(f'window {START}->{last_day.date()}; median seed {med} (NAV {finals[med]:,.0f}; '
      f'range {min(finals.values()):,.0f}-{max(finals.values()):,.0f})')

# re-run median seed capturing everything; recompute cash/positions by replaying trades
eq, trades, _ = br.simulate(med, 'random', days, dates, C, H, O, ATH, S, rs.values,
                            tv_prev.values, trig, weak, True, COST, stop=0.08, slots=8)
syms = list(close.columns)

closed, open_pos = [], []
for c, ei, xi, buy, sell, reason in trades:
    if reason == 'open_marked':
        open_pos.append((c, ei, buy))
    else:
        closed.append(dict(symbol=syms[c], entry_date=str(dates[ei].date()),
                           exit_date=str(dates[xi].date()), buy=round(buy, 2),
                           sell=round(sell, 2), qty=None,
                           ret_pct=round((sell/buy - 1)*100, 2),
                           reason=('stop_8pct' if reason == 'stop_8pct' else 'trail_sma20'),
                           src='backtest'))

# rebuild qty for open positions from the sim's mechanics is not exposed; size them
# off the final NAV proportionally to the engine's sizing (18.75% at entry). For the
# model portfolio we assign qty = round(0.1875 * NAV_at_entry / buy) approximated with
# final-eq scaling: use the nav curve.
nav_curve = pd.Series(eq, index=dates[days])
positions = []
mtm = 0.0
for c, ei, buy in open_pos:
    d_entry = dates[ei]
    nav_at = float(nav_curve.loc[:d_entry].iloc[-1]) if len(nav_curve.loc[:d_entry]) else CAPITAL
    qty = max(1, int(0.1875 * nav_at / buy))
    ltp = float(close.iloc[-1][syms[c]])
    positions.append(dict(symbol=syms[c], qty=qty, buy=round(buy, 2),
                          entry_date=str(d_entry.date()),
                          pivot=round(float(ATH[ei, syms.index(syms[c])]), 2),
                          signal_date=str(d_entry.date()), src='backtest'))
    mtm += qty * ltp
final_nav = float(eq[-1])
cash = max(0.0, final_nav - mtm)

# pending = last day's gate-permitted signals (they enter live tomorrow)
pending = []
if not weak[-1]:
    row = trig_df.iloc[-1]
    for s in trig_df.columns[row.values]:
        pending.append(dict(symbol=s, pivot=round(float(athcp.iloc[-1][s]), 2),
                            rs=round(float(rs.iloc[-1][s]), 1),
                            signal_date=str(last_day.date())))

nav_hist = [dict(date=str(d.date()), nav=round(float(v), 0))
            for d, v in nav_curve.items()]
prov = (f'History BACKFILLED from the adopted-spec backtest (trail-20, median seed {med}, '
        f'{START} -> {last_day.date()}, {len(closed)} trades); LIVE paper from 2026-09-02. '
        f'Backfill uses backtest fills (at-pivot) - live entries use the stricter '
        f'next-day buy-stop.')
st = dict(capital=CAPITAL, cash=round(cash, 0), positions=positions, pending=pending,
          nav=nav_hist, trades=closed, missed=[], started=START,
          seeded_from=prov, last_run=None)
json.dump(st, open(STATE, 'w'), indent=1, default=str)

navs = pd.Series({r['date']: r['nav'] for r in nav_hist}).astype(float)
dd = float((navs/navs.cummax()-1).min()*100)
wins = [t for t in closed if t['ret_pct'] > 0]
ltp_map = close.ffill().iloc[-1]   # last VALID close per symbol (no NaN in JSON)
ui_pos = []
for p in positions:
    lp = float(ltp_map.get(p['symbol'], np.nan))
    ui_pos.append(dict(**p, ltp=(round(lp, 2) if not np.isnan(lp) else None),
                       pnl_pct=(round((lp/p['buy']-1)*100, 1) if not np.isnan(lp) else None)))
ui = dict(updated=str(datetime.now()), nav=round(final_nav, 0), capital=CAPITAL,
          ret_pct=round((final_nav/CAPITAL-1)*100, 2), max_dd_pct=round(dd, 2),
          gate_weak=bool(weak[-1]), positions=ui_pos, pending=pending,
          trades=closed[-60:], n_trades=len(closed),
          win_pct=round(100*len(wins)/len(closed), 1) if closed else None,
          nav_curve=nav_hist[-500:], missed_tail=[],
          spec='trail-20 taxable pick; no mcap floor; gate 200DMA; 25bps; Rs 10L model',
          provenance=prov,
          study='/app/backtest/bluesky-ath-breakout-research142',
          log=[f'seeded from backtest replay ({len(closed)} trades, {len(positions)} open)'])
UI_JSON.parent.mkdir(parents=True, exist_ok=True)
json.dump(ui, open(UI_JSON, 'w'), indent=1, default=str)
print(f'SEEDED: NAV {final_nav:,.0f} ({(final_nav/CAPITAL-1)*100:+.1f}%) cash {cash:,.0f} '
      f'open {len(positions)} closed {len(closed)} pending {len(pending)} dd {dd:.1f}%')
