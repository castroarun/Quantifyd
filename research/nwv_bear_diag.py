#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Why does the BEARISH view underperform?  (spot-only, robust, no option model)
=============================================================================
Reuses the engine replay from nwv_phase1_regime. For each directional week
measures the UNDERLYING behaviour (no options, so no pricing error):

  - forward weekly return (Mon 09:45 close -> Fri close)
  - directional hit-rate (did price move the predicted way?)
  - drift headwind (mean fwd return of bearish vs bullish vs all weeks)
  - intra-week MFE / MAE in the predicted direction (favorable vs adverse)
  - did price reach the target pivot (S1 bear / R1 bull)?
  - how often a favorable excursion >= ~debit (rough breakeven) occurred
    INTRA-week but reverted by Friday  -> argues for a profit-target exit
"""
import sys, statistics
from datetime import date
sys.path.insert(0, '/home/arun/quantifyd')
sys.path.insert(0, '/home/arun/quantifyd/research')
from nwv_phase1_regime import (
    build_view, daily_close, m30_by_day, mondays, week_sessions, intraday,
    VIEW_BEARISH, VIEW_NTB, VIEW_BULLISH, VIEW_NTBULL,
)

BREAKEVEN_PTS = 80      # rough net debit of a 200-wide ATM spread (~40%)

def pct(x): return f"{100*x:5.1f}%"

def run():
    rec = {'BEAR': [], 'BULL': []}
    allret = []
    for mon in mondays(date(2020, 2, 3), date(2026, 3, 23)):
        v = build_view(mon)
        if not v:
            continue
        sess = week_sessions(v['ms'])
        if len(sess) < 2:
            continue
        exit_d = sess[-1]
        entry = v['spot']; fri = daily_close[exit_d]
        ib = intraday(v['ms'], exit_d)
        if not ib:
            continue
        wk_hi = max(b['high'] for b in ib); wk_lo = min(b['low'] for b in ib)
        fwd = (fri - entry) / entry
        allret.append(fwd)
        if v['view'] in (VIEW_BEARISH, VIEW_NTB):
            side = 'BEAR'
            hit = fri < entry
            mfe = (entry - wk_lo) / entry          # favorable (down)
            mae = (wk_hi - entry) / entry          # adverse (up)
            reached = wk_lo <= v['s1']
            fav_pts = entry - wk_lo                 # max favorable points
            be_hit = fav_pts >= BREAKEVEN_PTS       # intra-week reached rough breakeven
            be_but_flat = be_hit and fri >= entry - BREAKEVEN_PTS  # reverted by Fri
        elif v['view'] in (VIEW_BULLISH, VIEW_NTBULL):
            side = 'BULL'
            hit = fri > entry
            mfe = (wk_hi - entry) / entry
            mae = (entry - wk_lo) / entry
            reached = wk_hi >= v['r1']
            fav_pts = wk_hi - entry
            be_hit = fav_pts >= BREAKEVEN_PTS
            be_but_flat = be_hit and fri <= entry + BREAKEVEN_PTS
        else:
            continue
        rec[side].append({'year': v['ms'][:4], 'fwd': fwd, 'hit': hit, 'mfe': mfe,
                          'mae': mae, 'reached': reached, 'be_hit': be_hit,
                          'be_but_flat': be_but_flat})

    base_up = sum(1 for r in allret if r > 0) / len(allret)
    print(f"unconditional weekly drift: mean={pct(statistics.mean(allret))}  "
          f"%up={pct(base_up)}  n={len(allret)}")
    for side in ('BEAR', 'BULL'):
        rs = rec[side]
        n = len(rs)
        fwd = [r['fwd'] for r in rs]
        print(f"\n=== {side} views (n={n}) ===")
        print(f"  fwd weekly return : mean={pct(statistics.mean(fwd))}  "
              f"median={pct(statistics.median(fwd))}")
        print(f"  directional hit   : {pct(sum(r['hit'] for r in rs)/n)} "
              f"(price moved predicted way by Fri)")
        print(f"  reached target piv: {pct(sum(r['reached'] for r in rs)/n)} "
              f"({'S1' if side=='BEAR' else 'R1'} touched intra-week)")
        print(f"  mean MFE (favorable): {pct(statistics.mean(r['mfe'] for r in rs))}   "
              f"mean MAE (adverse): {pct(statistics.mean(r['mae'] for r in rs))}")
        print(f"  intra-wk reached ~breakeven ({BREAKEVEN_PTS}pt): "
              f"{pct(sum(r['be_hit'] for r in rs)/n)}")
        print(f"  ...of which REVERTED to flat/loss by Fri: "
              f"{pct(sum(r['be_but_flat'] for r in rs)/max(1,sum(r['be_hit'] for r in rs)))} "
              f"(=> profit-target exit would have banked these)")
        # by year hit-rate
        yrs = sorted(set(r['year'] for r in rs))
        line = "  hit% by year      : " + "  ".join(
            f"{y}:{pct(sum(r['hit'] for r in rs if r['year']==y)/max(1,sum(1 for r in rs if r['year']==y)))}"
            for y in yrs)
        print(line)

if __name__ == '__main__':
    run()
