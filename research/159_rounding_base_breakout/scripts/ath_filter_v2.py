"""ATH filter over v2 events: keep saucer breakouts whose trigger close is at / near the all-time-high close
(ATH computed causally = max close strictly before the trigger date, full DB history, split-guarded).
Writes results/verify_list_v2_ath.csv (dd-Mon-yyyy). Forward returns are information only."""
import sqlite3, pandas as pd, numpy as np
R = '/home/arun/quantifyd/research/159_rounding_base_breakout/results/'
c = sqlite3.connect('/home/arun/quantifyd/backtest_data/market_data.db')
ev = pd.read_csv(R + 'rounding_base_events_v2.csv')
ev = ev.sort_values('pattern_quality', ascending=False).drop_duplicates(['symbol', 'trigger_date'])
rows = []
for sym, g in ev.groupby('symbol'):
    d = pd.read_sql(f"select date,close,volume from market_data_unified where symbol='{sym}' and timeframe='day' order by date", c)
    d = d[d.volume > 0].reset_index(drop=True)
    # split guard on the history used for ATH: if any day-over-day move < -35% (split signature), ATH only from after it
    r = d.close.pct_change()
    cut = r[r < -0.35].index.max()
    d = d.loc[cut:] if pd.notna(cut) else d
    d['ath_prev'] = d.close.cummax().shift(1)
    m = d.set_index('date')
    for e in g.itertuples():
        if e.trigger_date not in m.index: continue
        ath = m.at[e.trigger_date, 'ath_prev']
        if pd.isna(ath): continue
        hist_bars = m.index.get_loc(e.trigger_date)
        rows.append(dict(e._asdict(), ath_prev=round(ath, 2), dist_to_ath_pct=round((e.trigger_close / ath - 1) * 100, 2),
                         hist_bars=hist_bars, split_cut=pd.notna(cut)))
a = pd.DataFrame(rows)
a.to_csv(R + 'rounding_base_events_v2_ath.csv', index=False)
print('events with ATH:', len(a))
for th in (-20, -15, -10, -5, 0):
    print(f'  trigger close >= ATH {th:+d}% :', (a.dist_to_ath_pct >= th).sum())
near = a[a.dist_to_ath_pct >= -10].copy()
near['year'] = near.trigger_date.str[:4]
print('per year (>= -10%):'); print(near.year.value_counts().sort_index().to_string())
cols = {'symbol':'Symbol','left_rim_date':'PriorHigh_Date','left_rim_level':'PriorHigh_Level','trough_date':'Trough_Date',
        'trough_close':'Trough_Close','depth_pct':'Depth_%','trigger_level':'BaseCeiling_Level','trigger_date':'Breakout_Close_Date',
        'trigger_close':'Breakout_Close','ath_prev':'ATH_Before_Breakout','dist_to_ath_pct':'DistToATH_%','entry_date':'Entry_Date',
        'entry_open':'Entry_Price_NextOpen','vol_multiple':'VolMultiple','hist_bars':'History_Bars','pattern_quality':'Pattern_Score',
        'info_fwd250_pct':'Info_Fwd250_%','info_st73_ret_pct':'Info_ST73_Ret_%'}
out = near[list(cols)].rename(columns=cols)
for k in ['PriorHigh_Date','Trough_Date','Breakout_Close_Date','Entry_Date']:
    out[k] = pd.to_datetime(out[k]).dt.strftime('%d-%b-%Y')
out = out.sort_values('Pattern_Score', ascending=False).round(2)
out.to_csv(R + 'verify_list_v2_ath.csv', index=False)
print('\nKMEW / SKFINDIA rows:'); print(a[a.symbol.isin(['KMEW','SKFINDIA'])][['symbol','trigger_date','trigger_close','ath_prev','dist_to_ath_pct']].to_string(index=False))
print('\nTop 25 by pattern score, trigger within 10% of ATH (fwd cols are info only):')
print(out.drop(columns=['PriorHigh_Date']).head(25).to_string(index=False))
