"""Banana-style trade book for the ADOPTED taxable spec (trail-20, no floor, gate ON,
real fills, 25bps): median-terminal seed of the 10-seed ensemble, every trade listed
newest-first with still-open rows — like the site's "The trades taken" table.

Output: results/bluesky_tradebook.html (self-contained, dark/light safe) + trades CSV.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY / 'scripts'))
import bluesky_replay as br

START, END = '2006-01-01', '2026-08-31'
TRAIL = 20
COST = 0.0025

w = br.load_frames('2004-06-29', trail_sma=TRAIL)
close, high, open_, athcp, sma, tv20 = (w[k] for k in
    ('close', 'high', 'open', 'athcp', 'sma50', 'tv20'))
etf = [c for c in close.columns if br.ETF_RE.search(c)]
tv_prev = tv20.shift(1)
prev_close = close.shift(1)
elig = tv_prev >= br.TV_FLOOR
elig[etf] = False
c1 = close.shift(1)
score = 2*(close/close.shift(63)-1) + (close/close.shift(126)-1) \
    + (close/close.shift(189)-1) + (close/close.shift(252)-1)
rs = (score.where(elig).rank(axis=1, pct=True)*100).shift(1)
setup = (prev_close < athcp) & (prev_close >= 0.8*athcp) & elig & (rs >= 70.0)
trig = (setup & (close > athcp) & athcp.notna()).fillna(False).values
nb = close['NIFTYBEES']
weak = (nb < nb.rolling(200).mean()).shift(1).fillna(False).values
dates = close.index
days = np.array([i for i, d in enumerate(dates) if START <= str(d.date()) <= END])
C, H, O, ATH, S = close.values, high.values, open_.values, athcp.values, sma.values

finals = {}
for seed in range(1, 11):
    eq, _, _ = br.simulate(seed, 'random', days, dates, C, H, O, ATH, S, rs.values,
                           tv_prev.values, trig, weak, True, COST, stop=0.08, slots=8)
    finals[seed] = eq[-1]
med = min(finals, key=lambda s: abs(finals[s] - np.median(list(finals.values()))))
print(f'median seed {med}: {finals[med]:,.0f}')
eq, trades, _ = br.simulate(med, 'random', days, dates, C, H, O, ATH, S, rs.values,
                            tv_prev.values, trig, weak, True, COST, stop=0.08, slots=8)

syms = list(close.columns)
rows = []
for c, ei, xi, buy, sell, reason in trades:
    still = reason == 'open_marked'
    held = (dates[xi] - dates[ei]).days
    rows.append(dict(symbol=syms[c], entry=dates[ei].strftime('%d-%b-%Y'),
                     exit='' if still else dates[xi].strftime('%d-%b-%Y'), held=held,
                     buy=round(buy, 2), sell='' if still else round(sell, 2),
                     ret=round((sell/buy - 1)*100, 2),
                     why=('still open · marked to Aug 2026 close' if still else
                          ('−8% stop' if reason == 'stop_8pct' else 'closed below the 20-SMA'))))
df = pd.DataFrame(rows)
df['_k'] = pd.to_datetime(df['entry'], format='%d-%b-%Y')
df = df.sort_values('_k', ascending=False).drop(columns=['_k'])
df.to_csv(STUDY / 'results' / 'bluesky_tradebook.csv', index=False)

body = []
for _, r in df.iterrows():
    cls = 'pos' if r['ret'] >= 0 else 'neg'
    ecls = ' class="open"' if not r['exit'] else ''
    body.append(
        f"<tr{ecls}><td class=sym>{r['symbol']}</td><td>{r['entry']}</td>"
        f"<td>{r['exit'] or '<i>still open</i>'}</td><td class=r>{r['held']}d</td>"
        f"<td class=r>{r['buy']}</td><td class=r>{r['sell'] or '—'}</td>"
        f"<td class='r {cls}'>{'+' if r['ret'] >= 0 else ''}{r['ret']}%</td>"
        f"<td class=why>{r['why']}</td></tr>")

wins = (df.ret > 0).mean()*100
html = f"""<!doctype html><meta charset=utf-8>
<style>
 body{{margin:0;font:13px/1.5 -apple-system,Segoe UI,sans-serif;color:#c8cdd4;background:#0f1418}}
 .wrap{{padding:12px 14px}} h3{{margin:0 0 2px;font-size:14px;color:#e8ecf1}}
 .sub{{color:#7d8590;font-size:11px;margin-bottom:10px}}
 table{{width:100%;border-collapse:collapse}}
 th{{text-align:left;font-size:10px;text-transform:uppercase;letter-spacing:.05em;color:#7d8590;
    padding:6px 8px;border-bottom:1px solid #2a3138;position:sticky;top:0;background:#0f1418}}
 td{{padding:6px 8px;border-bottom:1px solid #1c2228}}
 .sym{{font-weight:600;color:#e8ecf1}} .r{{text-align:right;font-variant-numeric:tabular-nums}}
 .pos{{color:#3fb27f}} .neg{{color:#e5484d}} .why{{color:#7d8590}}
 tr.open td{{background:rgba(63,178,127,.05)}} tr.open i{{color:#3fb27f;font-style:italic}}
 .scroll{{max-height:640px;overflow-y:auto;border:1px solid #2a3138;border-radius:8px}}
</style>
<div class=wrap>
<h3>The trades taken — adopted spec (trail-20, median seed {med})</h3>
<div class=sub>{len(df)} trades · 2006 → Aug 2026 · net of 25bps/side · win {wins:.0f}% ·
newest first · one representative selection path of the 10-seed ensemble</div>
<div class=scroll><table>
<thead><tr><th>Stock</th><th>Entry</th><th>Exit</th><th>Held</th><th>Buy</th><th>Sell</th>
<th>Return</th><th>Why exited</th></tr></thead>
<tbody>{''.join(body)}</tbody></table></div></div>
"""
(STUDY / 'results' / 'bluesky_tradebook.html').write_text(html, encoding='utf-8')
print(f'{len(df)} trades written; win {wins:.0f}%')
