"""research/161 — house-format YoY table + factsheet."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, '/home/arun/quantifyd/research/_utilities')
RES = Path(__file__).resolve().parents[1] / 'results'
r = json.load(open(RES / 'final161.json'))

COLS = [('BH', 'NIFTYBEES'), ('R159', 'r/159 saucer'), ('OA_PROXY', 'plain ATH (OA proxy)'),
        ('PLAIN_ATH', 'plain ATH + ST(14,4)'), ('WINNER', 'X60 + depth20'),
        ('WIN_SAUCER', 'X60 + depth20 + saucer')]

# r/159 v3 book YoY, pulled from its own saved report
r159 = {}
p159 = (RES.parents[1] / '159_rounding_base_breakout' / 'results'
        / 'final_report_ST_14_4_h0_s15_k3_a0.90_sl16.json')
if p159.exists():
    d = json.load(open(p159))
    r159 = {int(k): v for k, v in d.get('yoy_sys', {}).items()}
    r['R159'] = dict(yoy=r159, cagr_med=14.60, maxdd_med=-24.94, calmar_med=0.585)

years = sorted(r['BH']['yoy'].keys(), key=int)
BENCH = {'BH'}

def cell(key, y):
    d = r.get(key, {}).get('yoy', {})
    v = d.get(y, d.get(str(y)))
    if v is None:
        return None
    if isinstance(v, str):
        v = eval(v)
    return (float(v[0]), float(v[1]))

lines = []
hdr = '| Year | ' + ' | '.join(lab for _, lab in COLS) + ' | BEST CAGR | LEAST DD | BEST OVERALL |'
lines.append(hdr)
lines.append('|' + '---|' * (len(COLS) + 4))
for y in years:
    yi = int(y)
    vals = {k: cell(k, yi) for k, _ in COLS}
    cand = {k: v for k, v in vals.items() if v and k not in BENCH}
    bc = max(cand, key=lambda k: cand[k][0]) if cand else '-'
    ld = max(cand, key=lambda k: cand[k][1]) if cand else '-'
    bo = max(cand, key=lambda k: cand[k][0] + cand[k][1]) if cand else '-'
    lab = dict(COLS)
    row = ['%d' % yi]
    for k, _ in COLS:
        v = vals[k]
        row.append('%+.1f%%<br><sub>(%.1f%%)</sub>' % v if v else '—')
    row += [lab.get(bc, '-'), lab.get(ld, '-'), lab.get(bo, '-')]
    lines.append('| ' + ' | '.join(row) + ' |')
summ = ['**FULL**']
for k, _ in COLS:
    d = r.get(k, {})
    if 'cagr_med' in d or 'cagr' in d:
        c = d.get('cagr_med', d.get('cagr'))
        m = d.get('maxdd_med', d.get('maxdd'))
        cal = d.get('calmar_med', d.get('calmar'))
        summ.append('**%.2f%% / %.2f%%**<br><sub>Calmar %.3f</sub>' % (float(c), float(m), float(cal)))
    else:
        summ.append('—')
summ += ['—', '—', '—']
lines.append('| ' + ' | '.join(summ) + ' |')
out = '\n'.join(lines)
(RES / 'yoy161.md').write_text(out, encoding='utf-8')
print(out)

# ---------------- factsheet ----------------
try:
    from tearsheet import generate_tearsheet
    z = np.load(RES / 'curves161.npz', allow_pickle=True)
    idx = pd.to_datetime([str(d) for d in z['dates']])
    nav = pd.Series(z['WINNER'], index=idx).dropna()
    bh = pd.Series(z['bh'], index=idx).dropna()
    extra = pd.Series(z['OA_PROXY'], index=idx).dropna().reindex(nav.index).ffill()
    meta = {
        'Strategy': 'New-ATH-close breakout, previous high >= 60 bars old, base >= 20% deep (research/161)',
        'Universe': 'NSE cash daily, 20d median traded value >= Rs2 cr, ETFs excluded',
        'Entry': 'first close above the prior all-time-high close, where that prior ATH is at '
                 'least 60 trading bars old AND price fell at least 20% below it in between; '
                 'NO volume filter (volume confirmation was tested and rejected)',
        'Fill': 'next-day open, both legs',
        'Exit': 'SuperTrend(14,4) close trail, no hard stop',
        'Book': '16 slots @ 6.25% of NAV, Rs10L, idle cash 5.5% p.a.',
        'Costs': '25 bps per side; after-tax 20% STCG / 12.5% LTCG, Indian FY loss netting',
        'Robustness': '30-seed median; 03-Jan-2005 -> 11-Sep-2026; 864-cell sweep disclosed',
        'Verdict': 'STRATEGY candidate - clears all five pre-registered criteria',
    }
    generate_tearsheet(nav, bh, 'ATH Base-Age Breakout (research/161)', meta,
                       out_dir=str(RES), extra_nav=extra,
                       extra_label='plain ATH + OA exits (honest)')
    print('\nfactsheet written')
except Exception as e:
    print('tearsheet failed:', repr(e))
