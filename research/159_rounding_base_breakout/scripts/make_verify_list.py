import pandas as pd
df = pd.read_csv('rounding_base_events.csv')
# one row per symbol+breakout day (the three window modes often fire together); keep best pattern quality
df = df.sort_values('pattern_quality', ascending=False).drop_duplicates(['symbol','breakout_date'])
cols = {
 'symbol':'Symbol','rim_date':'LeftRim_Date','rim_level':'LeftRim_Level',
 'trough_date':'Trough_Date','trough_close':'Trough_Close','depth_pct':'Depth_%',
 'base_qualify_date':'Base_Complete_Date','breakout_date':'Breakout_Close_Date','breakout_close':'Breakout_Close',
 'fill_a_nextopen_date':'Entry_Date','fill_a_nextopen':'Entry_Price_NextOpen',
 'vol_ratio':'Vol_RightHalf_vs_Left','pattern_quality':'Pattern_Score'}
out = df[list(cols)].rename(columns=cols)
for c in ['LeftRim_Date','Trough_Date','Base_Complete_Date','Breakout_Close_Date','Entry_Date']:
    out[c] = pd.to_datetime(out[c]).dt.strftime('%d-%b-%Y')
out = out.sort_values('Pattern_Score', ascending=False).round(2)
out.to_csv('verify_list.csv', index=False)
print(len(out), 'events'); print(out.head(12).to_string(index=False))
