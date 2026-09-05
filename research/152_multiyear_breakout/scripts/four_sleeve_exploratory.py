"""EXPLORATORY (NOT pre-registered): 4-sleeve TN/OA/GOLD/MYB on the 2015+ common window."""
import sqlite3, numpy as np, pandas as pd
from pathlib import Path
RES=Path('/home/arun/quantifyd/research/152_multiyear_breakout/results')
R146=Path('/home/arun/quantifyd/research/146_complementary_third_sleeve/results')
con=sqlite3.connect('/home/arun/quantifyd/backtest_data/market_data.db')
g=pd.read_sql("SELECT substr(date,1,10) d, close FROM market_data_unified WHERE symbol='GOLDBEES' AND timeframe='day' AND close IS NOT NULL ORDER BY d",con); con.close()
g['d']=pd.to_datetime(g['d']); gold=g.set_index('d')['close']
myb=pd.read_csv(RES/'myb_equity_seeds.csv',index_col=0,parse_dates=True)
oa=pd.read_csv(R146/'oa_navs.csv',index_col=0,parse_dates=True)
tn={o:pd.read_csv(R146/f'tn_nav_off{o}.csv',index_col=0,parse_dates=True).iloc[:,0] for o in (0,4,8)}
idx=myb.index.intersection(oa.index).intersection(gold.index)
for v in tn.values(): idx=idx.intersection(v.index)
def mr(s): return s.loc[idx].resample('ME').last().pct_change().fillna(0)
m_oa={c:mr(oa[c]) for c in oa.columns}; m_tn={o:mr(v) for o,v in tn.items()}
m_myb={c:mr(myb[c]) for c in myb.columns}; m_g=mr(gold)
cols=list(oa.columns); mcols=list(myb.columns)
def stats(n):
    y=(n.index[-1]-n.index[0]).days/365.25; c=(n.iloc[-1]/n.iloc[0])**(1/y)-1
    d=float((n/n.cummax()-1).min()); return c*100,d*100,(c/abs(d) if d<0 else np.nan)
print(f'window {idx[0].date()} -> {idx[-1].date()}  (EXPLORATORY, not pre-registered)')
rows=[]
for wg,wm in [(0,0),(0.10,0),(0,0.10),(0.10,0.10),(0.10,0.15),(0.15,0.10),(0.15,0.15),(0.10,0.20),(0.20,0.10)]:
    wl=(1-wg-wm)/2; cs,ds,ks,dk=[],[],[],[]
    for off in tn:
        for j,c in enumerate(cols):
            b=(1+wl*m_oa[c]+wl*m_tn[off]+wg*m_g+wm*m_myb[mcols[j%len(mcols)]]).cumprod()
            base=(1+.5*m_oa[c]+.5*m_tn[off]).cumprod()
            x,bs=stats(b),stats(base); cs.append(x[0]);ds.append(x[1]);ks.append(x[2]);dk.append(x[2]-bs[2])
    rows.append(dict(gold=wg,myb=wm,tn_oa=round(wl*2,2),cagr=round(float(np.median(cs)),2),
        dd=round(float(np.median(ds)),2),calmar=round(float(np.median(ks)),2),
        calmar_worst=round(min(ks),2),dCalmar=round(float(np.median(dk)),3),
        wins=f"{int(np.sum(np.array(dk)>0))}/{len(dk)}"))
    print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv(RES/'four_sleeve_exploratory.csv',index=False)
