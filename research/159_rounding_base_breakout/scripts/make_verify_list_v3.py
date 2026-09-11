"""
research/159 v3 — verification list + duplicate-series detection.

Duplicate series: the DB carries the same price history under more than one symbol
(JSWDULUX and AKZOINDIA are the known pair -- the company was renamed and both tickers
kept the identical close series). Counting both would double-count one real trade and
inflate every event statistic, so v3 keeps ONE symbol per identical (date, close)
series and lists the dropped pairs explicitly.

The detector writes `series_hash` (md5 of the full date+close series) so this needs no
extra DB pass. Where two symbols collide we keep the one with more history, breaking
ties alphabetically.
"""
from collections import defaultdict
from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parents[1] / 'results'
ev = pd.read_csv(RES / 'rounding_base_events_v3.csv')
print('raw v3 events (symbol x window-mode): %d' % len(ev))

# ---- duplicate series -------------------------------------------------------------
groups = defaultdict(set)
for hsh, sym in zip(ev['series_hash'], ev['symbol']):
    groups[hsh].add(sym)
dups = {h: sorted(s) for h, s in groups.items() if len(s) > 1}
drop = set()
if dups:
    print('\nDUPLICATE SERIES FOUND (identical date+close history under >1 symbol):')
    hist = ev.groupby('symbol')['hist_bars'].max().to_dict()
    for h, syms in sorted(dups.items()):
        keep = sorted(syms, key=lambda s: (-hist.get(s, 0), s))[0]
        drop.update(set(syms) - {keep})
        print('  %s  ->  keep %s, drop %s' % (', '.join(syms), keep, ', '.join(sorted(set(syms) - {keep}))))
else:
    print('\nno duplicate series detected among symbols with v3 events')
if drop:
    ev = ev[~ev['symbol'].isin(drop)]
    print('dropped %d symbol(s), %d events remain' % (len(drop), len(ev)))

# ---- one row per symbol + trigger day ---------------------------------------------
ev = ev.sort_values('pattern_quality', ascending=False).drop_duplicates(['symbol', 'trigger_date'])
print('de-duplicated events (symbol x trigger date): %d' % len(ev))

cols = {
    'symbol': 'Symbol', 'trough_date': 'Trough_Date', 'trough_close': 'Trough_Close',
    'depth_pct': 'Depth_%', 'shelf_start_date': 'Shelf_Start_Date',
    'shelf_high': 'Shelf_High', 'shelf_range_pct': 'Shelf_Range_%',
    'ath_before_breakout': 'ATH_Before_Breakout', 'dist_to_ath_pct': 'DistToATH_%',
    'trigger_date': 'Breakout_Close_Date', 'trigger_close': 'Breakout_Close',
    'vol_multiple': 'VolMultiple', 'entry_date': 'Entry_Date',
    'entry_open': 'Entry_Price_NextOpen', 'hist_bars': 'History_Bars',
    'pattern_quality': 'Pattern_Score',
    'info_fwd250_pct': 'Info_Fwd250_%', 'info_st73_ret_pct': 'Info_ST73_Ret_%',
}
out = ev[list(cols)].rename(columns=cols)
for c in ['Trough_Date', 'Shelf_Start_Date', 'Breakout_Close_Date', 'Entry_Date']:
    out[c] = pd.to_datetime(out[c], errors='coerce').dt.strftime('%d-%b-%Y')
out = out.sort_values('Pattern_Score', ascending=False).round(2)
out.to_csv(RES / 'verify_list_v3.csv', index=False)
print('\nwrote %s  (%d rows)' % (RES / 'verify_list_v3.csv', len(out)))
print(out.head(10).to_string(index=False))
