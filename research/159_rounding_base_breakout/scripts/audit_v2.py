"""research/159 v2 — causality self-audit + cost of the D4 vertex widening."""
import csv
from collections import Counter
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / 'results'
rows = list(csv.DictReader(open(RES / 'rounding_base_events_v2.csv', encoding='utf-8')))

bad = Counter()
for r in rows:
    if r['left_rim_date'] >= r['trough_date']:            bad['rim>=trough'] += 1
    if r['trough_date'] > r['base_qualify_date']:         bad['trough>q'] += 1
    if r['base_qualify_date'] > r['trigger_date']:        bad['q>trigger'] += 1
    if r['entry_date'] and r['entry_date'] <= r['trigger_date']: bad['entry<=trigger'] += 1
    if float(r['trigger_close']) <= float(r['trigger_level']):   bad['close<=ceiling'] += 1
    if float(r['vol_multiple']) < 3.0:                    bad['volx<K'] += 1
    if r['fired_n60'] != '1':                             bad['n60_flag_off'] += 1
    if not (20.0 <= float(r['depth_pct']) <= 70.0):       bad['depth_out'] += 1
    if float(r['fit_r2']) < 0.70:                         bad['r2_low'] += 1
    if float(r['flat_frac']) < 0.40:                      bad['flat_low'] += 1
    if not (0.30 <= float(r['trough_pos']) <= 0.70):      bad['trough_pos_out'] += 1
    if not (0.30 <= float(r['vertex_frac']) <= 0.70):     bad['vertex_out'] += 1
    if int(r['days_q_to_trigger']) < 0:                   bad['negative_wait'] += 1
    if int(r['days_q_to_trigger']) > 120:                 bad['wait>max'] += 1

print('v2 events audited: %d' % len(rows))
if not bad:
    print('  ALL INVARIANTS CLEAN — 0 violations across 14 checks')
for k, v in bad.items():
    print('  VIOLATION %-18s %d' % (k, v))

# cost of D4: events only admitted because the vertex bound was widened to 0.30-0.70
d4 = [r for r in rows if not (0.33 <= float(r['vertex_frac']) <= 0.67)]
print('\n--- D4 cost (vertex bound widened from 0.33-0.67 to 0.30-0.70) ---')
print('  events admitted ONLY by D4: %d of %d raw (%.1f%%)'
      % (len(d4), len(rows), 100.0 * len(d4) / len(rows)))
ded = {}
for r in rows:
    ded.setdefault((r['symbol'], r['trigger_date']), r)
d4d = [k for k, r in ded.items() if not (0.33 <= float(r['vertex_frac']) <= 0.67)]
print('  de-duplicated: %d of %d (%.1f%%)' % (len(d4d), len(ded), 100.0 * len(d4d) / len(ded)))
print('  NOTE: SKFINDIA 16-May-2025 (vertex 0.674) is one of these — it is Arun\'s own date.')

print('\n--- days from base recognition to trigger ---')
w = sorted(int(r['days_q_to_trigger']) for r in rows)
print('  median %d bars   p10 %d   p90 %d   (0 bars: %d events)'
      % (w[len(w) // 2], w[int(.1 * len(w))], w[int(.9 * len(w))], sum(1 for x in w if x == 0)))

print('\n--- entry vs trigger close (fill slippage on the next open) ---')
g = sorted(100.0 * (float(r['entry_open']) / float(r['trigger_close']) - 1.0)
           for r in rows if r['entry_open'])
print('  median %+.2f%%  p10 %+.2f%%  p90 %+.2f%%' % (g[len(g) // 2], g[int(.1 * len(g))], g[int(.9 * len(g))]))
