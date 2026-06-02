#!/usr/bin/env python3
"""
CPR-ST Morning Scalp — System B: gap-day short strangle (Black-Scholes synthetic).

On big-gap mornings, sell an OTM call + OTM put at the 09:20 bar; exit on
+target% / -stop% of COMBINED premium, or 15:15 time-stop. Thesis = IV crush +
theta as the post-gap move stalls. Models constant-IV AND an intraday IV-crush
path (IV decays from markup*RV at open back to RV by EOD), since the crush is
the edge. Sweeps gap threshold, strikes, target/stop pairs, markup, crush on/off.
Walk-forward OOS, grouped by DTE and gap direction.

Pure stdlib. Reads market_data.db read-only. Runs on the VPS. Same IV-proxy
caveat as Phase 2 (no historical options data).
"""
import sqlite3, math
from datetime import datetime, timedelta
from collections import defaultdict

DB="/home/arun/quantifyd/backtest_data/market_data.db"; SYM="NIFTY50"
R=0.065; COST_PTS=1.5; EXPIRY_WD=3            # strangle = 2 legs, slightly higher cost
TRAIN_END=datetime(2025,7,1).date(); EXIT_HHMM=(15,15)
SESS_OPEN=(9,20); SESS_CLOSE=(15,15)

GAP_THRS  =[0.005,0.007,0.010]                # |open-prevclose|/prevclose
OFFSETS   =["0.4pct","0.6pct"]
TGT_STOP  =[(0.25,0.40),(0.30,0.30),(0.35,0.50)]
IV_MARKUPS=[1.0,1.3]
CRUSH     =[False,True]

def N(x): return 0.5*(1.0+math.erf(x/math.sqrt(2.0)))
def bs(S,K,T,sig,kind):
    if T<=1e-9 or sig<=1e-9: return max(0.0,(S-K) if kind=="C" else (K-S))
    srt=sig*math.sqrt(T); d1=(math.log(S/K)+(R+0.5*sig*sig)*T)/srt; d2=d1-srt
    return (S*N(d1)-K*math.exp(-R*T)*N(d2)) if kind=="C" else (K*math.exp(-R*T)*N(-d2)-S*N(-d1))

def load(con):
    b=[]
    for d,o,h,l,c in con.execute("SELECT date,open,high,low,close FROM market_data_unified "
        "WHERE symbol=? AND timeframe='5minute' ORDER BY date",(SYM,)):
        b.append((datetime.strptime(d,"%Y-%m-%d %H:%M:%S"),float(o),float(h),float(l),float(c)))
    days=[]
    for d,h,l,c in con.execute("SELECT date,high,low,close FROM market_data_unified "
        "WHERE symbol=? AND timeframe='day' ORDER BY date",(SYM,)):
        days.append((datetime.strptime(d.split()[0],"%Y-%m-%d").date(),float(h),float(l),float(c)))
    return b,days

def daily_iv_prevclose(days):
    iv={}; pc={}
    for i in range(1,len(days)):
        pc[days[i][0]]=days[i-1][3]
        if i>=10:
            cl=[days[j][3] for j in range(i-10,i)]
            lr=[math.log(cl[k+1]/cl[k]) for k in range(len(cl)-1)]
            m=sum(lr)/len(lr); var=sum((x-m)**2 for x in lr)/(len(lr)-1)
            iv[days[i][0]]=max(math.sqrt(var*252),0.08)
        else: iv[days[i][0]]=0.12
    return iv,pc

def expiry_dt(d):
    off=(EXPIRY_WD-d.weekday())%7; e=d+timedelta(days=off)
    return datetime(e.year,e.month,e.day,15,30),off

def sess_frac(t):
    a=SESS_OPEN[0]*60+SESS_OPEN[1]; b=SESS_CLOSE[0]*60+SESS_CLOSE[1]
    x=t.hour*60+t.minute
    return min(1.0,max(0.0,(x-a)/(b-a)))

def build_gap_days(bars,iv,pc):
    byday=defaultdict(list)
    for i,b in enumerate(bars): byday[b[0].date()].append(i)
    out=[]
    for d in sorted(byday):
        idx=byday[d]
        if len(idx)<2 or d not in pc or d not in iv: continue
        op=bars[idx[0]][1]; gap=(op-pc[d])/pc[d]
        edt,dte=expiry_dt(d)
        out.append(dict(date=d,idx=idx,gap=gap,iv=iv[d],edt=edt,dte=dte))
    return out

def replay(g,gap_thr,offset,tgt,stop,markup,crush,bars):
    if abs(g["gap"])<gap_thr: return None
    idx=g["idx"]; edt=g["edt"]; rv=g["iv"]
    ei=idx[1]; S0=bars[ei][1]
    frac = 0.006 if offset=="0.6pct" else 0.004
    Kc=round(S0*(1+frac)/50)*50; Kp=round(S0*(1-frac)/50)*50
    def ivt(t):
        if not crush: return rv*markup
        return rv*(markup-(markup-1.0)*sess_frac(t))
    T0=max((edt-bars[ei][0]).total_seconds()/(365*86400),1e-6)
    s0=ivt(bars[ei][0])
    C0=bs(S0,Kc,T0,s0,"C")+bs(S0,Kp,T0,s0,"P")
    if C0<=1.0: return None
    reason="time"; Cexit=C0
    for i in idx[1:]:
        _,o,h,l,cl=bars[i]
        T=max((edt-bars[i][0]).total_seconds()/(365*86400),1e-6); sig=ivt(bars[i][0])
        comb_at=lambda S: bs(S,Kc,T,sig,"C")+bs(S,Kp,T,sig,"P")
        Cclose=comb_at(cl); Cworst=max(comb_at(h),comb_at(l))
        if Cworst>=C0*(1+stop):  Cexit=C0*(1+stop); reason="stop"; break
        if Cclose<=C0*(1-tgt):   Cexit=C0*(1-tgt); reason="target"; break
        if (bars[i][0].hour,bars[i][0].minute)>=EXIT_HHMM: Cexit=Cclose; reason="time"; break
    pnl=(C0-Cexit)-COST_PTS
    return dict(date=g["date"],dte=g["dte"],
                bkt=("0DTE" if g["dte"]==0 else str(g["dte"]) if g["dte"]<3 else "3+"),
                gapdir=("up" if g["gap"]>0 else "dn"),reason=reason,C0=C0,pnl=pnl)

def expect(rows):
    if not rows: return 0,0,0,0
    n=len(rows); tot=sum(r["pnl"] for r in rows)
    return tot/n,tot,100*sum(1 for r in rows if r["pnl"]>0)/n,n
def maxdd(rows):
    eq=peak=dd=0
    for r in sorted(rows,key=lambda x:x["date"]):
        eq+=r["pnl"]; peak=max(peak,eq); dd=min(dd,eq-peak)
    return dd

def main():
    con=sqlite3.connect(DB); bars,days=load(con); con.close()
    iv,pc=daily_iv_prevclose(days); gaps=build_gap_days(bars,iv,pc)
    res=[]
    for gt in GAP_THRS:
        for off in OFFSETS:
            for (tgt,stop) in TGT_STOP:
                for mk in IV_MARKUPS:
                    for cr in CRUSH:
                        tr=[r for r in (replay(g,gt,off,tgt,stop,mk,cr,bars) for g in gaps) if r]
                        train=[r for r in tr if r["date"]<TRAIN_END]
                        test =[r for r in tr if r["date"]>=TRAIN_END]
                        eTr,_,wTr,nTr=expect(train); eTe,tTe,wTe,nTe=expect(test)
                        res.append(dict(gap=gt,off=off,tgt=tgt,stop=stop,mk=mk,crush=cr,
                            n=len(tr),exp_tr=eTr,win_tr=wTr,n_tr=nTr,
                            exp_te=eTe,tot_te=tTe,win_te=wTe,n_te=nTe,dd=maxdd(test),rows=tr))
    L=[];P=L.append
    P("="*82);P("SYSTEM B — GAP-DAY SHORT STRANGLE  (points/unit, after %.1f pt cost)"%COST_PTS);P("="*82)
    ng=len(set(r["date"] for r in res[0]["rows"])) if res and res[0]["rows"] else 0
    P("\nGap-day counts by threshold:")
    for gt in GAP_THRS:
        nn=sum(1 for g in gaps if abs(g["gap"])>=gt)
        P(f"  |gap|>= {gt*100:.1f}%  ->  {nn} days")

    P("\n[A] All configs ranked by OUT-OF-SAMPLE expectancy (n_test>=10):")
    P("    exp_te win_te tot_te  maxDD | exp_tr win_tr | gap  off    tgt/stop  mk   crush  n")
    cc=[r for r in res if r["n_te"]>=10]; cc.sort(key=lambda x:x["exp_te"],reverse=True)
    for r in cc[:18]:
        P(f"    {r['exp_te']:+6.2f} {r['win_te']:5.1f} {r['tot_te']:+7.1f} {r['dd']:+7.1f} | "
          f"{r['exp_tr']:+5.2f} {r['win_tr']:5.1f} | {r['gap']*100:.1f}% {r['off']:7s} "
          f"{int(r['tgt']*100)}/{int(r['stop']*100):<3d}  x{r['mk']:.1f}  {str(r['crush']):5s} {r['n']}")

    pos=[r for r in cc if r["exp_tr"]>0 and r["exp_te"]>0]
    P(f"\n[B] Configs profitable in BOTH train AND test: {len(pos)} of {len(cc)}")
    if pos:
        best=max(pos,key=lambda x:x["exp_te"])
        P(f"    BEST robust: gap>{best['gap']*100:.1f}% {best['off']} {int(best['tgt']*100)}/{int(best['stop']*100)} "
          f"mk x{best['mk']:.1f} crush={best['crush']}  exp_te={best['exp_te']:+.2f} win={best['win_te']:.0f}% n={best['n']}")
        P("    DTE / gap-dir breakdown:")
        for b in ("0DTE","1","2","3+"):
            rr=[x for x in best["rows"] if x["bkt"]==b]; e,t,w,n=expect(rr)
            if n: P(f"      DTE {b:4s} n={n:3d} exp={e:+6.2f} win={w:4.0f}% tot={t:+7.1f}")
        for gd in ("up","dn"):
            rr=[x for x in best["rows"] if x["gapdir"]==gd]; e,t,w,n=expect(rr)
            if n: P(f"      gap-{gd} n={n:3d} exp={e:+6.2f} win={w:4.0f}% tot={t:+7.1f}")

    P("\n[C] Crush effect (mean test expectancy):")
    for cr in (False,True):
        s=[r["exp_te"] for r in cc if r["crush"]==cr]
        if s: P(f"    crush={str(cr):5s} mean exp/trade={sum(s)/len(s):+.2f}")
    print("\n".join(L))

if __name__=="__main__": main()
