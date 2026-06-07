#!/usr/bin/env python3
"""
G2c — clean position-book engine for the dual-ST credit-structure book on the
REAL 6-week NIFTY chain. Fixes the g2b bugs (flat Rs20/order fees, missing-strike
fallback, mark-to-transaction leg P&L) and adds the user's LAYERING variants.

Modes (only CLOSE-ALL on MST flip in the layered modes — CST just adjusts):
  V0  as-specced baseline: full swap on every MST or CST change.
  V1  MST-only: one directional spread, flip only on MST.
  V3S stack  : MST flip -> reset to fresh dir. CST-against -> ADD hedge (->condor).
               CST-realign -> ADD a NEW dir on top (retain everything; pyramid).
  V3C convert: MST flip -> reset to fresh dir. CST-against -> ADD hedge (->condor).
               CST-realign -> REMOVE the hedge (convert back to dir). Retain core.

Structures (credit, defined-risk):
  dir+  = bull put spread  [-1 (f-50)PE, +1 (f-350)PE]
  dir-  = bear call spread [-1 (f+50)CE, +1 (f+350)CE]
  hedge = the opposite spread (turns a dir into an iron condor)

Leg P&L = qty*(exit_px - entry_px)*LOT. Conservative fills: open long@ask /
short@bid; close long@bid / short@ask. Missing strike -> LTP -> intrinsic (+flag).
Pure stdlib, reads options_data.db read-only on VPS.
"""
import sqlite3, os, bisect
from datetime import datetime
from collections import defaultdict

HERE=os.path.dirname(os.path.abspath(__file__))
RESDIR=os.path.join(os.path.dirname(HERE),"results"); os.makedirs(RESDIR,exist_ok=True)
DB="/home/arun/quantifyd/backtest_data/options_data.db"
SYM="NIFTY"; LOT=75
PERIOD,MST_MULT,CST_MULT=14,5.0,2.5
SHORT_OTM=50; WING=300; STEP=50
MIN_DTE=2   # avoid expiry-day pin; use nearest weekly with >=2 days

missing_legs=0

FEE_MULT=1.0   # cost-stress multiplier for robustness
def order_fee(premium, side):  # flat Rs20/order F&O + statutory, per leg, 1 lot
    to=premium*LOT
    brk=20.0
    stt=0.000625*to if side=="S" else 0.0
    exch=0.0003503*to; sebi=0.000001*to
    stamp=0.00003*to if side=="B" else 0.0
    gst=0.18*(brk+exch+sebi)
    return (brk+stt+exch+sebi+stamp+gst)*FEE_MULT

def parse_ts(s):
    s=str(s).replace("T"," ")
    try: return datetime.strptime(s[:19],"%Y-%m-%d %H:%M:%S")
    except ValueError: return datetime.strptime(s,"%Y-%m-%d %H:%M:%S.%f")

def load_spot_30min(con):
    rows=con.execute("SELECT snapshot_time,spot_price FROM underlying_spot "
        "WHERE symbol=? AND spot_price IS NOT NULL ORDER BY snapshot_time",(SYM,)).fetchall()
    b=defaultdict(list)
    for ts,sp in rows:
        dt=parse_ts(ts); m=dt.hour*60+dt.minute
        if m<9*60+15 or m>15*60+30: continue
        b[(dt.date(),(m-(9*60+15))//30)].append((dt,float(sp)))
    return [ (sorted(b[k])[-1][0],sorted(b[k])[0][1],max(x[1] for x in b[k]),
              min(x[1] for x in b[k]),sorted(b[k])[-1][1]) for k in sorted(b) ]

def st_dir(bars,period,mult):
    n=len(bars)
    if n<period+1: return [1]*n
    tr=[0.0]*n
    for i in range(n):
        _,o,h,l,c=bars[i]
        tr[i]=(h-l) if i==0 else max(h-l,abs(h-bars[i-1][4]),abs(l-bars[i-1][4]))
    atr=[0.0]*n; atr[period-1]=sum(tr[:period])/period
    for i in range(period,n): atr[i]=(atr[i-1]*(period-1)+tr[i])/period
    fub=[0.0]*n; flb=[0.0]*n; stt=[0.0]*n; d=[1]*n
    for i in range(n):
        _,o,h,l,c=bars[i]
        if i<period: continue
        mid=(h+l)/2; bub=mid+mult*atr[i]; blb=mid-mult*atr[i]
        pub=fub[i-1] if fub[i-1] else bub; plb=flb[i-1] if flb[i-1] else blb
        fub[i]=bub if (bub<pub or bars[i-1][4]>pub) else pub
        flb[i]=blb if (blb>plb or bars[i-1][4]<plb) else plb
        pst=stt[i-1] if stt[i-1] else fub[i]
        if pst==(fub[i-1] if fub[i-1] else fub[i]): stt[i]=fub[i] if c<=fub[i] else flb[i]
        else: stt[i]=flb[i] if c>=flb[i] else fub[i]
        d[i]=1 if c>=stt[i] else -1
    return d

def chain_snaps(con):
    r=con.execute("SELECT DISTINCT snapshot_time FROM option_chain WHERE symbol=? "
                  "ORDER BY snapshot_time",(SYM,)).fetchall()
    return [parse_ts(x[0]) for x in r],[x[0] for x in r]

def nearest(dts,t):
    i=bisect.bisect_left(dts,t); c=[j for j in (i-1,i) if 0<=j<len(dts)]
    return min(c,key=lambda j:abs((dts[j]-t).total_seconds())) if c else None

def get_chain(con,raw):
    rows=con.execute("SELECT expiry_date,strike,instrument_type,bid,ask,ltp,underlying_spot "
        "FROM option_chain WHERE symbol=? AND snapshot_time=?",(SYM,raw)).fetchall()
    book={}; spot=None; exps=set()
    for ed,k,it,bid,ask,ltp,us in rows:
        if us: spot=float(us)
        exps.add(ed); bb=float(bid or 0); aa=float(ask or 0); lp=float(ltp or 0)
        book[(ed,int(float(k)),it)]=(bb,aa,lp)
    return book,spot,exps

def pick_exp(exps,asof):
    """BI-WEEKLY rule: trade the 2nd-nearest expiry (>= today), i.e. SKIP the front
    weekly. Reproduces the user's Mon-Tue->next-Tue, Wed-Fri->Tue-after cadence."""
    v=[]
    for ed in exps:
        try: d=datetime.strptime(str(ed)[:10],"%Y-%m-%d").date()
        except ValueError: continue
        if d>=asof: v.append((d,ed))
    v.sort()
    if not v: return (None,None)
    return v[1] if len(v)>=2 else v[0]   # 2nd-nearest (skip front weekly)

def rnd(x): return int(round(x/STEP)*STEP)

def fill_px(book,ed,strike,it,action,spot):
    """action: open_long/open_short/close_long/close_short. Returns px/share."""
    global missing_legs
    q=book.get((ed,int(strike),it))
    buy = action in ("open_long","close_short")
    if q:
        b,a,lp=q
        px = a if buy else b
        if px<=0: px = lp if lp>0 else (a if a>0 else b)
        if px>0: return px
    # missing/zero -> intrinsic from spot
    missing_legs+=1
    return max(0.0, spot-strike) if it=="CE" else max(0.0, strike-spot)

# spread leg templates: list of (qty, strike, it)
def dir_legs(spot,dr):
    f=rnd(spot)
    return [(-1,f-SHORT_OTM,"PE"),(+1,f-SHORT_OTM-WING,"PE")] if dr>0 else \
           [(-1,f+SHORT_OTM,"CE"),(+1,f+SHORT_OTM+WING,"CE")]
def hedge_legs(spot,dr):   # opposite of core dr
    return dir_legs(spot,-dr)

# ---------------------------------------------------------------------------
class Engine:
    def __init__(self,con,dts,raws):
        self.con=con; self.dts=dts; self.raws=raws
        self.fees=0.0; self.realized=0.0; self.book=[]  # list of spreads
        self.opens=0; self.closes=0; self.structs=0
        self.worst=0.0  # worst realized per spread
        self.max_spreads=0  # peak concurrent spreads (for margin/capital)

    def snap(self,t):
        j=nearest(self.dts,t); bk,spot,exps=get_chain(self.con,self.raws[j])
        return bk,spot,exps

    def open_spread(self,t,role,kind_dr):
        bk,spot,exps=self.snap(t)
        if spot is None: return
        ed_d,ed=pick_exp(exps,t.date())
        if ed is None: return
        # kind_dr = the CORE direction; hedge = opposite spread (-> iron condor)
        legs = hedge_legs(spot,kind_dr) if role=="hedge" else dir_legs(spot,kind_dr)
        rec=[]
        for qty,k,it in legs:
            act="open_long" if qty>0 else "open_short"
            px=fill_px(bk,ed,k,it,act,spot)
            self.fees+=order_fee(px,"B" if qty>0 else "S"); self.opens+=1
            rec.append([qty,k,it,px])
        self.book.append(dict(role=role,ed=ed,ed_d=ed_d,legs=rec))
        self.structs+=1
        self.max_spreads=max(self.max_spreads,len(self.book))

    def close_spread(self,t,spread):
        bk,spot,exps=self.snap(t)
        settle = spot if (spot is not None and t.date()>spread["ed_d"]) else None
        pnl=0.0
        for qty,k,it,entry in spread["legs"]:
            if settle is not None:
                exit_px=max(0.0,settle-k) if it=="CE" else max(0.0,k-settle)
                self.fees+=order_fee(exit_px,"S" if qty>0 else "B")
            else:
                act="close_long" if qty>0 else "close_short"
                exit_px=fill_px(bk,ed=spread["ed"],strike=k,it=it,action=act,spot=spot)
                self.fees+=order_fee(exit_px,"S" if qty>0 else "B")
            pnl += qty*(exit_px-entry)*LOT
            self.closes+=1
        self.realized+=pnl; self.worst=min(self.worst,pnl)
        return pnl

    def close_all(self,t):
        for s in list(self.book): self.close_spread(t,s)
        self.book=[]

    def roll_expired(self,t):
        for s in list(self.book):
            if t.date()>=s["ed_d"]:
                role=s["role"]; legs=s["legs"]
                # infer dr from legs (PE short=bull, CE short=bear) for re-open
                dr = 1 if any(it=="PE" and q<0 for q,_,it,_ in legs) and \
                          not any(it=="CE" and q<0 for q,_,it,_ in legs) else \
                     (-1 if any(it=="CE" and q<0 for q,_,it,_ in legs) and \
                          not any(it=="PE" and q<0 for q,_,it,_ in legs) else 0)
                self.close_spread(t,s); self.book.remove(s)
                if dr!=0: self.open_spread(t,role,dr)

    def net(self): return self.realized - self.fees

# ---------------------------------------------------------------------------
TIMEOUT=4   # V4: enter anyway this many bars after MST flip if no pullback-resume

def run(con,bars,dts,raws,mode):
    global missing_legs; missing_legs=0
    mst=st_dir(bars,PERIOD,MST_MULT); cst=st_dir(bars,PERIOD,CST_MULT)
    warm=PERIOD+1
    E=Engine(con,dts,raws)
    cur_mst=0; hedged=False
    armed=False; entered=False; pullback=False; since=0   # V4 state
    for i in range(warm,len(bars)):
        t=bars[i][0]; m=mst[i]; c=cst[i]
        E.roll_expired(t)
        if mode=="V4":
            # PULLBACK ENTRY: on MST flip, DON'T enter; wait for first
            # pullback (CST against) then resume (CST realign), else TIMEOUT.
            if m!=cur_mst:
                E.close_all(t); cur_mst=m
                armed=True; entered=False; pullback=False; hedged=False; since=0
                continue
            since+=1
            if not entered:
                if c!=m: pullback=True
                if (pullback and c==m) or since>=TIMEOUT:
                    E.open_spread(t,"core",m); entered=True; hedged=False
                continue
            # entered -> manage with stack-layering (same as V3S)
            if c!=m and not hedged: E.open_spread(t,"hedge",m); hedged=True
            elif c==m and hedged: E.open_spread(t,"core",m); hedged=False
            continue
        if mode=="V0":
            tgt = ("dir",m) if m==c else ("condor",0)
            key=(tgt,)
            if key!=getattr(E,"_k",None):
                E.close_all(t)
                if tgt[0]=="dir": E.open_spread(t,"core",m)
                else: E.open_spread(t,"core",1); E.open_spread(t,"hedge",1)
                E._k=key
            continue
        if mode=="V1":
            if m!=cur_mst:
                E.close_all(t); E.open_spread(t,"core",m); cur_mst=m
            continue
        # layered modes V3S / V3C
        if m!=cur_mst:
            E.close_all(t); E.open_spread(t,"core",m); cur_mst=m; hedged=False
            continue
        if c!=m and not hedged:           # CST against core -> add hedge (go neutral)
            E.open_spread(t,"hedge",m); hedged=True
        elif c==m and hedged:             # CST realigned
            if mode=="V3S":               # STACK: add a new dir on top, keep hedge
                E.open_spread(t,"core",m); hedged=False
            elif mode=="V3C":             # CONVERT: lift the hedge, back to dir
                for s in [x for x in E.book if x["role"]=="hedge"]:
                    E.close_spread(t,s); E.book.remove(s)
                hedged=False
    E.close_all(bars[-1][0])
    return E, missing_legs

def main():
    global TIMEOUT
    print(f"DB: {DB}")
    con=sqlite3.connect(DB)
    bars=load_spot_30min(con); dts,raws=chain_snaps(con)
    days=max(1,(bars[-1][0]-bars[14][0]).days)
    print(f"NIFTY 30-min bars {len(bars)} over {days} days; chain snaps {len(dts)}")
    print("BI-WEEKLY EXPIRY (2nd-nearest Tue) + ENTRY-TIMING TEST. 1 lot, ~6wk real chain.\n")
    print(f"{'mode':>6} {'opens':>6} {'closes':>7} {'gross':>9} {'fees':>8} "
          f"{'NET':>9} {'worst':>9} {'miss':>5}")
    # baselines
    for mode in ("V0","V3S"):
        E,miss=run(con,bars,dts,raws,mode)
        print(f"{mode:>6} {E.opens:>6} {E.closes:>7} {E.realized:>9.0f} {E.fees:>8.0f} "
              f"{E.net():>9.0f} {E.worst:>9.0f} {miss:>5}")
    # V4 pullback-entry, sweep the timeout (in 30-min bars; 999 = pure pullback)
    for to in (2,4,8,999):
        TIMEOUT=to
        E,miss=run(con,bars,dts,raws,"V4")
        lbl=f"V4/{'inf' if to==999 else to}"
        print(f"{lbl:>6} {E.opens:>6} {E.closes:>7} {E.realized:>9.0f} {E.fees:>8.0f} "
              f"{E.net():>9.0f} {E.worst:>9.0f} {miss:>5}")
    con.close()
    print("\nV4 = enter on first pullback-and-resume after MST flip (else after "
          "TIMEOUT bars). Does delaying entry to a better price flip GROSS positive?")
    print("\n1 lot, ~6 weeks real chain. NET = realized gross - full fees "
          "(flat Rs20/order + STT+exch+GST+stamp). 'miss'=legs priced off intrinsic "
          "fallback (data gaps; lower is better). Annualize NET x ~8.8 for 1 lot/yr.")

def robust():
    """Robustness sweep of the WINNER: V4 pure-pullback (TIMEOUT=inf), bi-weekly
    expiry, stack-layering. Perturb each knob; the +net must survive (monotonic/
    robust, not a lone peak)."""
    global TIMEOUT,CST_MULT,MST_MULT,PERIOD,SHORT_OTM,WING,FEE_MULT
    TIMEOUT=999
    con=sqlite3.connect(DB)
    bars=load_spot_30min(con); dts,raws=chain_snaps(con)
    base=dict(PERIOD=PERIOD,MST_MULT=MST_MULT,CST_MULT=CST_MULT,
              SHORT_OTM=SHORT_OTM,WING=WING,FEE_MULT=FEE_MULT)
    def one(label):
        E,miss=run(con,bars,dts,raws,"V4")
        margin=E.max_spreads*WING*LOT
        print(f"  {label:<22} net {E.net():>8.0f}  gross {E.realized:>8.0f}  "
              f"structs {E.structs:>3}  worst {E.worst:>7.0f}  peakMargin {margin:>7.0f}")
    print("ROBUSTNESS — V4 pure-pullback, bi-weekly, stack. Base then perturbations:\n")
    one("BASE (14/5/2.5)")
    print(" -- pullback sensitivity (CST mult):")
    for cm in (2.0,2.5,3.0):
        CST_MULT=cm; one(f"CST_MULT={cm}"); CST_MULT=base["CST_MULT"]
    print(" -- MST mult:")
    for mm in (4.0,5.0,6.0):
        MST_MULT=mm; one(f"MST_MULT={mm}"); MST_MULT=base["MST_MULT"]
    print(" -- ST period:")
    for p in (10,14,18):
        PERIOD=p; one(f"PERIOD={p}"); PERIOD=base["PERIOD"]
    print(" -- short strike OTM:")
    for s in (0,50,100):
        SHORT_OTM=s; one(f"SHORT_OTM={s}"); SHORT_OTM=base["SHORT_OTM"]
    print(" -- wing width (defined risk):")
    for w in (200,300,400):
        WING=w; one(f"WING={w}"); WING=base["WING"]
    print(" -- COST STRESS:")
    for fm in (1.0,1.5,2.0):
        FEE_MULT=fm; one(f"FEE x{fm}"); FEE_MULT=base["FEE_MULT"]
    con.close()
    print("\nVerdict: if net stays >0 across most perturbations -> robust SIGNAL; "
          "if it flips on small changes -> fragile peak, do NOT wire live.")

if __name__=="__main__":
    import sys
    if len(sys.argv)>1 and sys.argv[1]=="robust": robust()
    else: main()
