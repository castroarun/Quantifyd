"""
research/159 v2 — simplified verification list for Arun.

Same column set and dd-Mon-yyyy formatting as make_verify_list.py (v1), plus the three
v2 columns: VolMultiple, Trigger_N and DistToLeftRim_%.

One extra column beyond the brief, BaseCeiling_Level, is included because without it a
row cannot be checked on a chart: the v2 entry is "close above the base ceiling", and
the ceiling is the level being cleared. Breakout_Close vs BaseCeiling_Level is the
whole trigger.

DistToLeftRim_% is how far BELOW the old v1 left-rim level the entry close sits, i.e.
how much overhead supply the trade still has to chew through. Positive = below the rim.
"""
import pandas as pd
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / 'results'
df = pd.read_csv(RES / 'rounding_base_events_v2.csv')

# one row per symbol + entry day (the four window modes often fire the same trigger)
df = df.sort_values('pattern_quality', ascending=False).drop_duplicates(['symbol', 'trigger_date'])

cols = {
    'symbol': 'Symbol',
    'left_rim_date': 'LeftRim_Date', 'left_rim_level': 'LeftRim_Level',
    'trough_date': 'Trough_Date', 'trough_close': 'Trough_Close', 'depth_pct': 'Depth_%',
    'base_qualify_date': 'Base_Complete_Date',
    'trigger_level': 'BaseCeiling_Level',
    'trigger_date': 'Breakout_Close_Date', 'trigger_close': 'Breakout_Close',
    'entry_date': 'Entry_Date', 'entry_open': 'Entry_Price_NextOpen',
    'vol_multiple': 'VolMultiple', 'trigger_N': 'Trigger_N',
    'dist_to_left_rim_pct': 'DistToLeftRim_%',
    'vol_ratio': 'Vol_RightHalf_vs_Left', 'pattern_quality': 'Pattern_Score',
}
out = df[list(cols)].rename(columns=cols)
for c in ['LeftRim_Date', 'Trough_Date', 'Base_Complete_Date', 'Breakout_Close_Date', 'Entry_Date']:
    out[c] = pd.to_datetime(out[c], errors='coerce').dt.strftime('%d-%b-%Y')
out = out.sort_values('Pattern_Score', ascending=False).round(2)
out.to_csv(RES / 'verify_list_v2.csv', index=False)
print('%d events -> %s' % (len(out), RES / 'verify_list_v2.csv'))
print(out.head(15).to_string(index=False))
