#!/usr/bin/env python3
"""
CPR-ST Morning Scalp — Phase 2: Black-Scholes synthetic option P&L.

Phase 1 showed the directional signal is a ~coin flip on the underlying.
This layer asks the real question: does theta (short-option decay) turn it
profitable? Prices the OTM leg via Black-Scholes at each day's DTE, replays
delta+theta+vega bar-by-bar, and applies the %-prem / Supertrend / target /
time exits. Sweeps ST params, stop level, OTM offset, and the entry variant
(pure-CPR vs ST-aligned). Walk-forward OOS split. Grouped by DTE.

Pure stdlib. Reads market_data.db read-only. Runs on the VPS.

CAVEATS (honest):
- No historical IV. IV proxied from 10-day realized vol (floored). Real
  option IV is usually higher and mean-reverts; results are indicative.
- Weekly expiry assumed Thursday; NSE shifted the weekday over 2024-25.
- P&L in NIFTY premium POINTS per unit (multiply by lot size for rupees).
- COST_PTS round-trip cost/slippage haircut applied per trade.
"""
import sqlite3, math, csv
from datetime import datetime, timedelta
from collections import defaultdict

DB  = "/home/arun/quantifyd/backtest_data/market_data.db"
SYM = "NIFTY50"
R   = 0.065
COST_PTS   = 1.0       # round-trip cost+slippage on the option, in premium points
EXPIRY_WD  = 3         # Thursday
TRAIN_END  = datetime(2025, 7, 1).date()
EXIT_HHMM  = (15, 15)

# fixed classification thresholds for Phase 2 (narrow recalibrated lower)
THR_INSIDE = True
THR_FAR    = 0.0050

# ---- sweep dimensions ----
ST_PERIODS = [7, 10, 14, 21]
ST_MULTS   = [1.5, 2.0, 2.5, 3.0, 3.5]
STOPS      = [0.30, 0.50, 0.75, 1.00, None]   # None = Supertrend-only
OFFSETS    = ["0.4pct", "0.6pct", "0.30delta"]
NARROWS    = [0.0005, 0.0008]                 # recalibrated far lower
ENTRY_VAR  = ["cpr", "st_aligned"]            # base vs ST-entry-filter variant
IV_MARKUPS = [1.0, 1.15, 1.30, 1.50]          # sell at RV x markup (volatility risk premium)

# ---------------------------------------------------------------------------
def N(x): return 0.5*(1.0 + math.erf(x/math.sqrt(2.0)))

def bs(S, K, T, sig, kind):
    if T <= 1e-9 or sig <= 1e-9:
        return max(0.0, (S-K) if kind=="C" else (K-S))
    srt = sig*math.sqrt(T)
    d1 = (math.log(S/K) + (R + 0.5*sig*sig)*T)/srt
    d2 = d1 - srt
    if kind == "C":
        return S*N(d1) - K*math.exp(-R*T)*N(d2)
    return K*math.exp(-R*T)*N(-d2) - S*N(-d1)

def put_delta(S, K, T, sig):
    if T <= 1e-9: return -1.0 if K > S else 0.0
    d1 = (math.log(S/K) + (R + 0.5*sig*sig)*T)/(sig*math.sqrt(T))
    return N(d1) - 1.0

# ---------------------------------------------------------------------------
def load(con):
    b = []
    for d,o,h,l,c in con.execute(
        "SELECT date,open,high,low,close FROM market_data_unified "
        "WHERE symbol=? AND timeframe='5minute' ORDER BY date",(SYM,)):
        b.append((datetime.strptime(d,"%Y-%m-%d %H:%M:%S"),float(o),float(h),float(l),float(c)))
    days=[]
    for d,h,l,c in con.execute(
        "SELECT date,high,low,close FROM market_data_unified "
        "WHERE symbol=? AND timeframe='day' ORDER BY date",(SYM,)):
        days.append((datetime.strptime(d.split()[0],"%Y-%m-%d").date(),float(h),float(l),float(c)))
    return b, days

def build_cpr_and_iv(days):
    cpr, iv = {}, {}
    rets=[]
    for i in range(1,len(days)):
        _,ph,pl,pc = days[i-1]
        P=(ph+pl+pc)/3; BC=(ph+pl)/2; TC=2*P-BC
        cpr[days[i][0]] = dict(P=P,BC=min(BC,TC),TC=max(BC,TC),
                               R1=2*P-pl,S1=2*P-ph,width=abs(TC-BC))
        # realized-vol IV proxy from last 10 daily closes
        if i>=10:
            cl=[days[j][3] for j in range(i-10,i)]
            lr=[math.log(cl[k+1]/cl[k]) for k in range(len(cl)-1)]
            m=sum(lr)/len(lr); var=sum((x-m)**2 for x in lr)/(len(lr)-1)
            iv[days[i][0]]=max(math.sqrt(var*252),0.08)
        else:
            iv[days[i][0]]=0.12
    return cpr, iv

def supertrend_dir(bars, period, mult):
    n=len(bars); tr=[0.0]*n
    for i in range(n):
        _,o,h,l,c=bars[i]
        tr[i]= h-l if i==0 else max(h-l,abs(h-bars[i-1][4]),abs(l-bars[i-1][4]))
    atr=[0.0]*n; atr[period-1]=sum(tr[:period])/period
    for i in range(period,n): atr[i]=(atr[i-1]*(period-1)+tr[i])/period
    d=[0]*n; fub=flb=0.0
    for i in range(n):
        _,o,h,l,c=bars[i]; mid=(h+l)/2
        if i<period: d[i]=1; continue
        bub=mid+mult*atr[i]; blb=mid-mult*atr[i]
        fub=bub if (bub<fub or bars[i-1][4]>fub or fub==0) else fub
        flb=blb if (blb>flb or bars[i-1][4]<flb or flb==0) else flb
        prev=d[i-1] or 1
        d[i]=(-1 if c<flb else 1) if prev==1 else (1 if c>fub else -1)
    return d

def expiry_dt(d):
    off=(EXPIRY_WD-d.weekday())%7
    e=d+timedelta(days=off)
    return datetime(e.year,e.month,e.day,15,30), off

def pick_strike(S, offset, T, sig):
    if offset=="0.4pct":  return round(S*0.996/50)*50, round(S*1.004/50)*50
    if offset=="0.6pct":  return round(S*0.994/50)*50, round(S*1.006/50)*50
    # 0.30 delta: search strikes for put delta ~ -0.30 (and symmetric call)
    K=round(S/50)*50
    while put_delta(S,K,T,sig) < -0.30 and K>S*0.95: K-=50
    pk=K
    Kc=round(S/50)*50
    while (1+put_delta(S,Kc,T,sig)) < 0.30 and Kc<S*1.05: Kc+=50   # call delta=1+putdelta(same K)? approximate via symmetry
    # simpler: call strike symmetric distance
    ck=round((S+(S-pk))/50)*50
    return pk, ck

# ---------------------------------------------------------------------------
def build_candidates(bars, cpr, iv, narrow):
    byday=defaultdict(list)
    for i,b in enumerate(bars): byday[b[0].date()].append(i)
    cands=[]
    for d in sorted(byday):
        idx=byday[d]
        if len(idx)<2: continue
        lv=cpr.get(d);  iV=iv.get(d)
        if lv is None or iV is None: continue
        spot=bars[idx[0]][1]; fclose=bars[idx[0]][4]
        wpct=lv["width"]/spot
        if wpct<narrow: continue
        if lv["BC"]<=fclose<=lv["TC"]: continue
        if fclose>lv["TC"]:
            dist=(fclose-lv["TC"])/spot; side="bull"
        else:
            dist=(lv["BC"]-fclose)/spot; side="bear"
        if dist>THR_FAR: continue       # SYS-B gap day, handled separately
        edt,dte=expiry_dt(d)
        cands.append(dict(date=d,side=side,idx=idx,entry_i=idx[1],
                          spot=spot,lv=lv,iv=iV,edt=edt,dte=dte))
    return cands

def replay(c, direction, offset, stop, entry_var, bars, markup=1.0):
    side=c["side"]; idx=c["idx"]; iv=c["iv"]*markup; edt=c["edt"]
    ei=c["entry_i"]; S0=bars[ei][1]
    want = 1 if side=="bull" else -1
    if entry_var=="st_aligned" and direction[ei]!=want:
        return None                       # variant filter: skip misaligned
    T0=max((edt-bars[ei][0]).total_seconds()/(365*86400),1e-6)
    pk,ck=pick_strike(S0,offset,T0,iv)
    K = pk if side=="bull" else ck
    kind = "P" if side=="bull" else "C"
    P0 = bs(S0,K,T0,iv,kind)
    if P0<=0.5: return None               # too cheap to be real
    target=c["lv"]["R1"] if side=="bull" else c["lv"]["S1"]
    reason="time"; Pexit=P0
    for i in idx[1:]:
        _,o,h,l,cl=bars[i]
        T=max((edt-bars[i][0]).total_seconds()/(365*86400),1e-6)
        # underlying-driven exits checked on this bar; price option at close
        if side=="bull" and h>=target:
            Pexit=bs(target,K,T,iv,kind); reason="target"; break
        if side=="bear" and l<=target:
            Pexit=bs(target,K,T,iv,kind); reason="target"; break
        # %-prem stop: use adverse extreme of the bar (high spot hurts call, low spot hurts put)
        adverse = l if side=="bull" else h
        Padv=bs(adverse,K,T,iv,kind)
        if stop is not None and Padv>=P0*(1+stop):
            Pexit=P0*(1+stop); reason="stop"; break
        if direction[i]!=want and direction[i]!=0:
            Pexit=bs(cl,K,T,iv,kind); reason="st_flip"; break
        if (bars[i][0].hour,bars[i][0].minute)>=EXIT_HHMM:
            Pexit=bs(cl,K,T,iv,kind); reason="time"; break
    pnl=(P0-Pexit)-COST_PTS               # short: collect P0, buy back Pexit
    return dict(date=c["date"],side=side,dte=c["dte"],
                bkt=("0DTE" if c["dte"]==0 else str(c["dte"]) if c["dte"]<3 else "3+"),
                K=K,P0=P0,Pexit=Pexit,reason=reason,pnl=pnl)

def expectancy(rows):
    if not rows: return 0,0,0,0
    n=len(rows); tot=sum(r["pnl"] for r in rows)
    win=sum(1 for r in rows if r["pnl"]>0)
    return tot/n, tot, 100*win/n, n

def maxdd(rows):
    eq=0; peak=0; dd=0
    for r in sorted(rows,key=lambda x:x["date"]):
        eq+=r["pnl"]; peak=max(peak,eq); dd=min(dd,eq-peak)
    return dd

# ---------------------------------------------------------------------------
def main():
    con=sqlite3.connect(DB); bars,days=load(con); con.close()
    cpr,iv=build_cpr_and_iv(days)
    # candidates depend only on narrow threshold
    cand_by_narrow={nw:build_candidates(bars,cpr,iv,nw) for nw in NARROWS}
    # precompute ST direction per (period,mult)
    dir_cache={}
    for p in ST_PERIODS:
        for m in ST_MULTS:
            dir_cache[(p,m)]=supertrend_dir(bars,p,m)

    results=[]  # one row per config
    for nw in NARROWS:
        cands=cand_by_narrow[nw]
        for (p,m),direction in dir_cache.items():
            for off in OFFSETS:
                for stop in STOPS:
                    for ev in ENTRY_VAR:
                        for mk in IV_MARKUPS:
                            tr=[]
                            for c in cands:
                                r=replay(c,direction,off,stop,ev,bars,mk)
                                if r: tr.append(r)
                            train=[r for r in tr if r["date"]<TRAIN_END]
                            test =[r for r in tr if r["date"]>=TRAIN_END]
                            eTr,_,wTr,nTr=expectancy(train)
                            eTe,tTe,wTe,nTe=expectancy(test)
                            results.append(dict(narrow=nw,period=p,mult=m,offset=off,
                                stop=("ST" if stop is None else f"{int(stop*100)}%"),
                                entry=ev,markup=mk,n=len(tr),
                                exp_train=eTr,win_train=wTr,n_train=nTr,
                                exp_test=eTe,tot_test=tTe,win_test=wTe,n_test=nTe,
                                dd_test=maxdd(test),rows=tr))
    rank(results)

def rank(results):
    L=[];P=L.append
    P("="*78);P("PHASE 2 — BLACK-SCHOLES SYNTHETIC P&L  (points/unit, after %g pt cost)"%COST_PTS);P("="*78)

    # 1) headline: base config (pure-CPR entry, ST 10/3, 0.4pct) across stops
    P("\n[A] Base config (entry=cpr, ST 10/3.0, offset 0.4pct, narrow 0.0008) by STOP:")
    P("    stop   n   exp/trade  win%   tot(test)   maxDD(test)")
    base=[r for r in results if r["entry"]=="cpr" and r["period"]==10 and r["mult"]==3.0
          and r["offset"]=="0.4pct" and r["narrow"]==0.0008 and r["markup"]==1.0]
    for r in sorted(base,key=lambda x:x["stop"]):
        P(f"    {r['stop']:>4s} {r['n']:4d}  {r['exp_test']:+7.2f}   {r['win_test']:4.1f}  "
          f"{r['tot_test']:+9.1f}   {r['dd_test']:+8.1f}")

    # E) IV-markup sensitivity on the base config (the decisive test)
    P("\n[E] IV-MARKUP SENSITIVITY (base entry=cpr ST 10/3.0 0.4pct narrow .0008, stop=ST-only):")
    P("    markup   exp/trade  win%   tot(test)   maxDD     <- captures volatility risk premium")
    bm=[r for r in results if r["entry"]=="cpr" and r["period"]==10 and r["mult"]==3.0
        and r["offset"]=="0.4pct" and r["narrow"]==0.0008 and r["stop"]=="ST"]
    for r in sorted(bm,key=lambda x:x["markup"]):
        P(f"    x{r['markup']:.2f}    {r['exp_test']:+7.2f}   {r['win_test']:4.1f}  "
          f"{r['tot_test']:+9.1f}   {r['dd_test']:+8.1f}")

    # 2) overall ranking by OOS test expectancy (require enough test trades + train agreement)
    cand=[r for r in results if r["n_test"]>=20 and r["exp_train"]>0]
    cand.sort(key=lambda x:x["exp_test"],reverse=True)
    P("\n[B] Top 15 by OUT-OF-SAMPLE expectancy (train>0 filter, n_test>=20):")
    P("    exp_te  win_te  tot_te   maxDD   | exp_tr win_tr | mk    entry  ST     off      stop  narrow")
    for r in cand[:15]:
        P(f"    {r['exp_test']:+6.2f}  {r['win_test']:4.1f}  {r['tot_test']:+7.1f}  {r['dd_test']:+7.1f}  "
          f"| {r['exp_train']:+5.2f} {r['win_train']:4.1f} | x{r['markup']:.2f} {r['entry']:10s} {r['period']:>2d}/{r['mult']:.1f} "
          f"{r['offset']:9s} {r['stop']:>4s} {r['narrow']}")

    # 3) entry-variant comparison (avg test expectancy across all configs)
    P("\n[C] Entry variant comparison (mean test expectancy over all configs):")
    for ev in ("cpr","st_aligned"):
        s=[r["exp_test"] for r in results if r["entry"]==ev and r["n_test"]>=20]
        if s: P(f"    {ev:10s} mean exp/trade={sum(s)/len(s):+.2f}  (configs={len(s)})")

    # 4) best config DTE breakdown
    if cand:
        best=cand[0]
        P(f"\n[D] DTE breakdown of best OOS config "
          f"(entry={best['entry']} ST {best['period']}/{best['mult']} {best['offset']} stop {best['stop']}):")
        for b in ("0DTE","1","2","3+"):
            rr=[x for x in best["rows"] if x["bkt"]==b]
            e,t,w,n=expectancy(rr)
            if n: P(f"    DTE {b:4s} n={n:3d}  exp={e:+6.2f}  win={w:4.1f}%  tot={t:+7.1f}")
        # reason mix
        rc=defaultdict(int)
        for x in best["rows"]: rc[x["reason"]]+=1
        P("    exit reasons: "+", ".join(f"{k}={v}" for k,v in sorted(rc.items())))
    print("\n".join(L))

if __name__=="__main__":
    main()
