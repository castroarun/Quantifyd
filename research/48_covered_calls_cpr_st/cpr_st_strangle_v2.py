#!/usr/bin/env python3
"""
System B v2 — gap-day short strangle priced with REAL India VIX.

Replaces the RV proxy + deterministic crush with the actual intraday VIX path
(symbol INDIAVIX, 5minute). IV at every bar = VIX/100, so real crush AND real
spikes are captured. Also reports the empirical question that decides System B:
do gap-day VIX levels actually crush intraday? Pure stdlib, runs on VPS.

Caveat: India VIX is NIFTY's ~30-day implied vol; weekly-option ATM IV differs
by the term-structure basis (usually < VIX in calm, > VIX on event days). VIX is
the best real proxy available without per-strike option history. IV_BASIS sweeps
a multiplier on VIX to bracket that basis.
"""
import sqlite3, math
from datetime import datetime, timedelta
from collections import defaultdict
DB="/home/arun/quantifyd/backtest_data/market_data.db"; SYM="NIFTY50"
R=0.065; COST=1.5; EXPIRY_WD=3; EXIT=(15,15); TRAIN_END=datetime(2025,7,1).date()
GAP_THRS=[0.005,0.007,0.010]; OFFSETS=["0.4pct","0.6pct"]
TGT_STOP=[(0.25,0.40),(0.30,0.30),(0.35,0.50)]; IV_BASIS=[1.0,1.15]

def N(x): return 0.5*(1+math.erf(x/math.sqrt(2)))
def bs(S,K,T,s,k):
    if T<=1e-9 or s<=1e-9: return max(0.0,(S-K) if k=="C" else (K-S))
    srt=s*math.sqrt(T); d1=(math.log(S/K)+(R+0.5*s*s)*T)/srt; d2=d1-srt
    return (S*N(d1)-K*math.exp(-R*T)*N(d2)) if k=="C" else (K*math.exp(-R*T)*N(-d2)-S*N(-d1))

con=sqlite3.connect(DB)
bars=[(datetime.strptime(d,"%Y-%m-%d %H:%M:%S"),float(o),float(h),float(l),float(c))
      for d,o,h,l,c in con.execute("SELECT date,open,high,low,close FROM market_data_unified "
      "WHERE symbol=? AND timeframe='5minute' ORDER BY date",(SYM,))]
dclose={}
for d,c in con.execute("SELECT date,close FROM market_data_unified WHERE symbol=? AND timeframe='day' ORDER BY date",(SYM,)):
    dclose[datetime.strptime(d.split()[0],"%Y-%m-%d").date()]=float(c)
vix={}
for d,c in con.execute("SELECT date,close FROM market_data_unified WHERE symbol='INDIAVIX' AND timeframe='5minute' ORDER BY date"):
    vix[datetime.strptime(d,"%Y-%m-%d %H:%M:%S")]=float(c)
con.close()

# iv path aligned to NIFTY bars (forward-fill VIX)
iv_at={}; last=0.13
for dt,o,h,l,c in bars:
    if dt in vix: last=vix[dt]/100.0
    iv_at[dt]=last

byday=defaultdict(list)
for i,b in enumerate(bars): byday[b[0].date()].append(i)
prevclose={}
sd=sorted(byday)
for j in range(1,len(sd)):
    pc=dclose.get(sd[j-1]) or bars[byday[sd[j-1]][-1]][4]
    prevclose[sd[j]]=pc

def expiry_dt(d):
    off=(EXPIRY_WD-d.weekday())%7; e=d+timedelta(days=off)
    return datetime(e.year,e.month,e.day,15,30),off

def replay(d,gap_thr,offset,tgt,stop,basis):
    idx=byday[d]
    if len(idx)<2 or d not in prevclose: return None
    op=bars[idx[0]][1]; gap=(op-prevclose[d])/prevclose[d]
    if abs(gap)<gap_thr: return None
    edt,dte=expiry_dt(d); ei=idx[1]; S0=bars[ei][1]
    frac=0.006 if offset=="0.6pct" else 0.004
    Kc=round(S0*(1+frac)/50)*50; Kp=round(S0*(1-frac)/50)*50
    def T(b): return max((edt-bars[b][0]).total_seconds()/(365*86400),1e-6)
    s0=iv_at[bars[ei][0]]*basis
    C0=bs(S0,Kc,T(ei),s0,"C")+bs(S0,Kp,T(ei),s0,"P")
    if C0<=1.0: return None
    # find 15:15 bar + capture VIX open/close-of-window for crush stats
    vix_open=iv_at[bars[ei][0]]*100; vix_exit=vix_open
    reason="time"; Cexit=C0
    for i in idx[1:]:
        _,o,h,l,cl=bars[i]; sig=iv_at[bars[i][0]]*basis; Ti=T(i)
        comb=lambda S: bs(S,Kc,Ti,sig,"C")+bs(S,Kp,Ti,sig,"P")
        Cw=max(comb(h),comb(l)); Cc=comb(cl)
        vix_exit=iv_at[bars[i][0]]*100
        if Cw>=C0*(1+stop): Cexit=C0*(1+stop); reason="stop"; break
        if Cc<=C0*(1-tgt): Cexit=C0*(1-tgt); reason="target"; break
        if (bars[i][0].hour,bars[i][0].minute)>=EXIT: Cexit=Cc; reason="time"; break
    pnl=(C0-Cexit)-COST
    return dict(date=d,dte=dte,bkt=("0DTE" if dte==0 else str(dte) if dte<3 else "3+"),
                gapdir=("up" if gap>0 else "dn"),reason=reason,pnl=pnl,
                vix_open=vix_open,vix_exit=vix_exit,move=abs((bars[idx[-1]][4]-S0)/S0))

def expect(rows):
    if not rows: return 0,0,0,0
    n=len(rows);tot=sum(r["pnl"] for r in rows)
    return tot/n,tot,100*sum(1 for r in rows if r["pnl"]>0)/n,n
def maxdd(rows):
    eq=peak=dd=0
    for r in sorted(rows,key=lambda x:x["date"]):
        eq+=r["pnl"];peak=max(peak,eq);dd=min(dd,eq-peak)
    return dd

res=[]
for gt in GAP_THRS:
    for off in OFFSETS:
        for (tgt,stop) in TGT_STOP:
            for bsf in IV_BASIS:
                tr=[r for r in (replay(d,gt,off,tgt,stop,bsf) for d in sd) if r]
                train=[r for r in tr if r["date"]<TRAIN_END]; test=[r for r in tr if r["date"]>=TRAIN_END]
                eTr,_,wTr,nTr=expect(train); eTe,tTe,wTe,nTe=expect(test)
                res.append(dict(gap=gt,off=off,tgt=tgt,stop=stop,basis=bsf,n=len(tr),
                    exp_tr=eTr,win_tr=wTr,n_tr=nTr,exp_te=eTe,tot_te=tTe,win_te=wTe,n_te=nTe,
                    dd=maxdd(test),rows=tr))
L=[];P=L.append
P("="*86);P("SYSTEM B v2 — GAP STRANGLE priced with REAL India VIX (pts/unit, after %.1f cost)"%COST);P("="*86)

# EMPIRICAL CRUSH TEST (the decisive question)
allgap=[r for r in res if r["gap"]==0.005 and r["off"]=="0.4pct" and r["tgt"]==0.30 and r["basis"]==1.0][0]["rows"]
P("\n[CRUSH] Did gap-day VIX actually fall intraday? (VIX open -> 15:15, gap>=0.5%)")
for lab,rows in (("ALL",allgap),("gap-up",[r for r in allgap if r["gapdir"]=="up"]),
                 ("gap-dn",[r for r in allgap if r["gapdir"]=="dn"])):
    if not rows: continue
    n=len(rows); vo=sum(r["vix_open"] for r in rows)/n; ve=sum(r["vix_exit"] for r in rows)/n
    fell=100*sum(1 for r in rows if r["vix_exit"]<r["vix_open"])/n
    P(f"  {lab:7s} n={n:3d}  VIX {vo:5.2f} -> {ve:5.2f}  ({100*(ve-vo)/vo:+5.1f}%)  fell on {fell:.0f}% of days")

P("\n[A] Configs ranked by OUT-OF-SAMPLE expectancy (n_test>=10):")
P("    exp_te win_te tot_te  maxDD | exp_tr win_tr | gap  off    t/s    basis n")
cc=[r for r in res if r["n_te"]>=10]; cc.sort(key=lambda x:x["exp_te"],reverse=True)
for r in cc[:14]:
    P(f"    {r['exp_te']:+6.2f} {r['win_te']:5.1f} {r['tot_te']:+7.1f} {r['dd']:+7.1f} | "
      f"{r['exp_tr']:+5.2f} {r['win_tr']:5.1f} | {r['gap']*100:.1f}% {r['off']:7s} "
      f"{int(r['tgt']*100)}/{int(r['stop']*100):<3d} x{r['basis']:.2f} {r['n']}")
pos=[r for r in cc if r["exp_tr"]>0 and r["exp_te"]>0]
P(f"\n[B] Profitable in BOTH train AND test: {len(pos)} of {len(cc)}")
if pos:
    best=max(pos,key=lambda x:x["exp_te"])
    P(f"    BEST: gap>{best['gap']*100:.1f}% {best['off']} {int(best['tgt']*100)}/{int(best['stop']*100)} basis x{best['basis']:.2f}"
      f"  exp_te={best['exp_te']:+.2f} win={best['win_te']:.0f}% n={best['n']} maxDD={best['dd']:+.0f}")
    for b in ("0DTE","1","2","3+"):
        rr=[x for x in best["rows"] if x["bkt"]==b]; e,t,w,n=expect(rr)
        if n: P(f"      DTE {b:4s} n={n:3d} exp={e:+6.2f} win={w:4.0f}% tot={t:+7.1f}")
    for gd in ("up","dn"):
        rr=[x for x in best["rows"] if x["gapdir"]==gd]; e,t,w,n=expect(rr)
        if n: P(f"      gap-{gd} n={n:3d} exp={e:+6.2f} win={w:4.0f}% tot={t:+7.1f}")
    rc=defaultdict(int)
    for x in best["rows"]: rc[x["reason"]]+=1
    P("      exit reasons: "+", ".join(f"{k}={v}" for k,v in sorted(rc.items())))
print("\n".join(L))
