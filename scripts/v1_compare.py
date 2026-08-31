"""Is the 30% CSL variant better because of its STOP, or because of its DAYS?

The two look like one system with a different stop. They are not:

  V1 one-and-done   0.4% underlying-move stop, and only trades 0/1-DTE days -> 34 days
  V1 + 30% CSL      30% combined-premium stop, and trades EVERY day          -> 92 days

So the headline gap mixes a day-selection effect with a stop effect, and there is
a third arm that separates them: V1 daily re-enter is the same 0.4% stop over the
same every-day universe as the CSL variant.

  same stop, different days  ->  daily re-enter  vs  one-and-done
  same days, different stop  ->  daily re-enter  vs  30% CSL
  and on the 34 shared days  ->  one-and-done    vs  30% CSL, paired

Read-only.
"""
import json, statistics as st, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
P = Path('/home/arun/quantifyd/frontend/public/straddles')


def per_day(fname):
    d = json.loads((P / fname).read_text())
    blob = d.get('per_day') or {}
    return {str(k)[:10]: v['series'][-1][1]
            for k, v in blob.items() if isinstance(v, dict) and v.get('series')}


oad = per_day('v1.json')                       # 0/1-DTE only, 0.4% stop
daily = per_day('v1_daily.json')               # every day, 0.4% stop
csl = {t['day'][:10]: t['final']
       for t in json.loads((P / 'v1_sl30.json').read_text())['trades']
       if t.get('final') is not None}          # every day, 30% combined-premium stop


def stats(x, label):
    v = list(x.values())
    print(f'  {label:26} {len(v):>3} days · net {sum(v):>10,.0f} · '
          f'mean/day {sum(v)/len(v):>8,.0f} · worst {min(v):>10,.0f}')
    return v


print('THE THREE ARMS, AS PUBLISHED')
stats(oad, 'one-and-done (0/1-DTE)')
stats(daily, 'daily re-enter (all days)')
stats(csl, '30% CSL (all days)')

print('\nEFFECT 1 — DAY SELECTION (same 0.4% stop, different day universe)')
mo, md = st.mean(list(oad.values())), st.mean(list(daily.values()))
print(f'  all days {md:>8,.0f}/day  ->  0/1-DTE only {mo:>8,.0f}/day   '
      f'= {mo/md:.2f}x from choosing days alone')

print('\nEFFECT 2 — THE STOP (same every-day universe, different stop)')
mc = st.mean(list(csl.values()))
print(f'  0.4% move {md:>8,.0f}/day  ->  30% premium {mc:>8,.0f}/day   '
      f'= {mc/md:.2f}x from changing the stop alone')

print('\nPAIRED, on the days BOTH one-and-done and the CSL variant traded')
common = sorted(set(oad) & set(csl))
if len(common) >= 8:
    diff = [csl[d] - oad[d] for d in common]
    m, sd = st.mean(diff), st.stdev(diff)
    t = m / (sd / len(diff) ** 0.5)
    print(f'  {len(common)} shared days · one-and-done {sum(oad[d] for d in common):>10,.0f} · '
          f'CSL {sum(csl[d] for d in common):>10,.0f}')
    print(f'  mean difference {m:>+9,.0f}/day · sd {sd:>9,.0f} · t {t:>5.2f}'
          f'   {"significant" if abs(t) >= 2 else "NOT significant"}')
    wins = sum(1 for x in diff if x > 0)
    print(f'  CSL better on {wins}/{len(diff)} of those days')
    top = max(diff)
    print(f'  largest single day {top:>+9,.0f} = {100*top/sum(diff):.0f}% of the total gap'
          if sum(diff) else '')
else:
    print(f'  only {len(common)} shared days — too few to pair')

print('\nTHE UNTESTED CELL')
print('  0/1-DTE days AND the 30% premium stop has never been run. If both effects')
print('  are real and independent, that is the cell worth testing — not either alone.')
