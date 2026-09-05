"""research/152 - literal 50-50 pair checks: MYB with Open Alpha, MYB with True North,
and the three-way, all on the common window, monthly rebalanced, paired across paths."""
import numpy as np, pandas as pd
from pathlib import Path
RES=Path('/home/arun/quantifyd/research/152_multiyear_breakout/results')
R146=Path('/home/arun/quantifyd/research/146_complementary_third_sleeve/results')
myb=pd.read_csv(RES/'myb_equity_seeds.csv',index_col=0,parse_dates=True)
oa=pd.read_csv(R146/'oa_navs.csv',index_col=0,parse_dates=True)
tn={o:pd.read_csv(R146/f'tn_nav_off{o}.csv',index_col=0,parse_dates=True).iloc[:,0] for o in (0,4,8)}
idx=myb.index.intersection(oa.index)
for v in tn.values(): idx=idx.intersection(v.index)
def mr(s): return s.loc[idx].resample('ME').last().pct_change().fillna(0)
m_oa={c:mr(oa[c]) for c in oa.columns}; m_tn={o:mr(v) for o,v in tn.items()}
m_my={c:mr(myb[c]) for c in myb.columns}
oc=list(oa.columns); mc=list(myb.columns)
def st(n):
    y=(n.index[-1]-n.index[0]).days/365.25; c=(n.iloc[-1]/n.iloc[0])**(1/y)-1
    d=float((n/n.cummax()-1).min()); return c*100,d*100,(c/abs(d) if d<0 else np.nan)
print(f'common window {idx[0].date()} -> {idx[-1].date()}  (after tax, 25bps, cash 5%, monthly rebalance)')
rows=[]
def rec(name, fn, npaths):
    cs,ds,ks=[],[],[]
    for k in range(npaths):
        n=fn(k); a=st(n); cs.append(a[0]); ds.append(a[1]); ks.append(a[2])
    rows.append(dict(book=name, cagr_med=round(float(np.median(cs)),2), cagr_min=round(min(cs),2),
        cagr_max=round(max(cs),2), dd_med=round(float(np.median(ds)),2), dd_worst=round(min(ds),2),
        calmar_med=round(float(np.median(ks)),2), calmar_worst=round(min(ks),2)))
    print(rows[-1],flush=True)
rec('MYB alone',            lambda k:(1+m_my[mc[k]]).cumprod(), len(mc))
rec('Open Alpha alone',     lambda k:(1+m_oa[oc[k]]).cumprod(), len(oc))
rec('True North alone',     lambda k:(1+m_tn[[0,4,8][k]]).cumprod(), 3)
rec('MYB + OA 50-50',       lambda k:(1+.5*m_my[mc[k%len(mc)]]+.5*m_oa[oc[k%len(oc)]]).cumprod(), 30)
rec('MYB + TN 50-50',       lambda k:(1+.5*m_my[mc[k%len(mc)]]+.5*m_tn[[0,4,8][k%3]]).cumprod(), 30)
rec('TN + OA 50-50 (deployed)', lambda k:(1+.5*m_oa[oc[k%len(oc)]]+.5*m_tn[[0,4,8][k%3]]).cumprod(), 30)
rec('MYB+OA+TN 1/3 each',   lambda k:(1+(m_my[mc[k%len(mc)]]+m_oa[oc[k%len(oc)]]+m_tn[[0,4,8][k%3]])/3).cumprod(), 30)
pd.DataFrame(rows).to_csv(RES/'pair_5050.csv',index=False)
print('\nPAIRED: MYB+OA vs OA alone, and MYB+TN vs TN alone, same path')
for lab,f_,g_,n_ in [('MYB+OA vs OA alone', lambda k:(1+.5*m_my[mc[k]]+.5*m_oa[oc[k%len(oc)]]).cumprod(), lambda k:(1+m_oa[oc[k%len(oc)]]).cumprod(), len(mc)),
                     ('MYB+TN vs TN alone', lambda k:(1+.5*m_my[mc[k]]+.5*m_tn[[0,4,8][k%3]]).cumprod(), lambda k:(1+m_tn[[0,4,8][k%3]]).cumprod(), len(mc))]:
    d=[np.array(st(f_(k)))-np.array(st(g_(k))) for k in range(n_)]
    d=np.array(d)
    print(f'  {lab}: dCAGR {np.median(d[:,0]):+.2f}pp  dDD {np.median(d[:,1]):+.2f}pp  '
          f'dCalmar {np.median(d[:,2]):+.3f}  Calmar-wins {int((d[:,2]>0).sum())}/{n_}')
