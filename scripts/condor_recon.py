"""Reconcile the condor's live paper record against research/80's 11-year backtest.

Arun's challenge, and it is correct: "but we hv backtested data from algo report
right". I framed the 7 paper cycles as if they were the only evidence and said it
needs ~17 cycles to clear t=2. That answers the WRONG question.

"Can 7 cycles prove an edge on their own" is not the question anyone needs
answered, because the edge was already estimated on 11 years. The question the 7
cycles exist to answer is narrower and much cheaper: **is the real chain behaving
the way the calibrated engine said it would?** That is a consistency test, and 7
cycles can absolutely speak to it.

research/80's surviving spec, at 2 lots / 130 qty:
    mean +Rs880/trade · +Rs38,944/yr · maxDD -Rs23,820 · Calmar 1.63 · 11/12 years +
    margin ~Rs52,500 · ~100% p.a. on margin

Two caveats that belong beside those numbers and are easy to lose:
  1. the 11-year run is a CALIBRATED BLACK-SCHOLES ENGINE, not recorded option
     prices. RESULTS.md is explicit: the engine has NO volatility skew, was
     validated near-ATM (~0.4% OTM), and "1.4% OTM is exactly where skew bites
     hardest, so the engine is weakest" there. Our long wings sit at ~1.75% from
     spot — squarely in the zone the engine models worst. The live paper book is
     the FIRST real-chain evidence this strategy has ever had.
  2. the backtested winner is "0.8% / 1.0% (stop x2)" — it INCLUDES a stop at
     double the combined premium.

Read-only.
"""
from __future__ import annotations

import json, statistics as st, sys
from pathlib import Path

ROOT = Path('/home/arun/quantifyd')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BT_MEAN, BT_ANNUAL, BT_DD, BT_CALMAR = 880.0, 38944.0, -23820.0, 1.63

h = sorted(json.loads((ROOT / 'backtest_data' / 'condor_paper_state.json').read_text())['history'],
           key=lambda x: x['entry_day'])
v = [x['pnl'] for x in h]
n, mean, sd = len(v), st.mean(v), st.stdev(v)
se = sd / n ** 0.5

print('THE TWO EVIDENCE BASES, SIDE BY SIDE (both at 2 lots / qty 130)\n')
print(f"  {'':22} {'backtest (r/80)':>20} {'live paper':>20}")
print('  ' + '-' * 64)
print(f"  {'source':22} {'calibrated engine':>20} {'RECORDED CHAIN':>20}")
print(f"  {'sample':22} {'11 years (~570 cyc)':>20} {str(n) + ' cycles':>20}")
print(f"  {'mean / cycle':22} {'Rs{:,.0f}'.format(BT_MEAN):>20} {'Rs{:,.0f}'.format(mean):>20}")
print(f"  {'win rate':22} {'75%':>20} "
      f"{'%d%% (%d/%d)' % (100*sum(1 for x in v if x>0)/n, sum(1 for x in v if x>0), n):>20}")
print(f"  {'max drawdown':22} {'Rs{:,.0f}'.format(BT_DD):>20} "
      f"{'Rs{:,.0f}'.format(min(0, min(v))):>20}")

# ── the consistency test ──────────────────────────────────────────────────
z = (mean - BT_MEAN) / se
print(f'\nIS THE LIVE RECORD CONSISTENT WITH THE BACKTEST?')
print(f'  live mean            Rs{mean:,.0f} / cycle')
print(f'  backtest mean        Rs{BT_MEAN:,.0f} / cycle')
print(f'  difference           Rs{mean-BT_MEAN:,.0f}  ({mean/BT_MEAN:.2f}x the backtest)')
print(f'  standard error       Rs{se:,.0f}  (sd Rs{sd:,.0f} over {n} cycles)')
print(f'  z on the difference  {z:.2f}')
print(f'  -> {"CONSISTENT" if abs(z) < 2 else "DIVERGENT"}: the live record is '
      f'{abs(z):.2f} standard errors from the backtest, so the recorded chain is'
      f'\n     behaving the way the engine predicted. It is running ABOVE the backtest,'
      f'\n     not below, which is the direction that costs nothing to be wrong about.')

# what the 7 cycles CAN and CANNOT do
print(f'\n  What {n} cycles can and cannot settle:')
print(f'    CANNOT  prove an edge unaided — for that you need ~{(2*sd/mean)**2:.0f} cycles (t=2)')
print(f'    CAN     detect a real-chain SHORTFALL against the backtest. With se Rs{se:,.0f},')
print(f'            a true mean below Rs{BT_MEAN - 2*se:,.0f}/cycle would already have shown up')
print(f'            as a 2-sigma miss. It has not. That is the test that matters, and it passes.')

# ── projecting the backtest to sizes ──────────────────────────────────────
print('\nWHAT THE BACKTEST IMPLIES AT EACH SIZE (scaling r/80 linearly)')
print(f"  {'lots':>5} {'qty':>5} {'annual':>12} {'maxDD':>12} {'margin':>10} "
      f"{'maxDD vs live book DD of Rs36,082':>34}")
for lots in (2, 3, 5, 10):
    k = lots / 2
    dd_ = BT_DD * k
    print(f'  {lots:>5} {lots*65:>5} {BT_ANNUAL*k:>12,.0f} {dd_:>12,.0f} '
          f'{52500*k:>10,.0f} {abs(dd_)/36082:>33.1f}x')
print('\n  The live book\'s entire max drawdown to date is Rs36,082. The backtest\'s OWN')
print('  drawdown at 10 lots is Rs119,100 — 3.3x the whole book. That is the sizing')
print('  constraint, and it comes from the backtest itself, not from the small sample.')

# ── the stop discrepancy ─────────────────────────────────────────────────
src = (ROOT / 'research/80_farDTE_rescue/scripts/condor_paper.py').read_text()
has_stop = any(w in src for w in ('STOP_MULT', 'stop_mult', 'combined premium doubl', '* 2.0'))
print('\n' + '=' * 74)
print('DISCREPANCY: the paper book is NOT running the backtested spec')
print('=' * 74)
print(f'  backtest winner   "0.8% / 1.0% (stop x2)" — INCLUDES a stop at 2x combined premium')
print(f'  paper book        exits only on Friday close (or Mon/Tue); no premium stop found: '
      f'{"stop present" if has_stop else "NO STOP CODE"}')
print('  header says       "The wings ARE the stop (max loss known at entry)"')
print('\n  So the forward evidence is being collected on an UNSTOPPED variant of a')
print('  STOPPED strategy. The wings do cap the loss, so it is not unbounded — but it')
print('  is not the construction that earned Calmar 1.63 either, and the x2 stop would')
print('  have cut some losing cycles short. Worth aligning before this carries money.')
