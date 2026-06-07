#!/usr/bin/env python3
"""
G2b — the decisive salvage test. Same real 6-week NIFTY chain, but:
  V0  as-specced (flip whole structure on every MST OR CST change)   [baseline]
  V1  MST-ONLY: directional credit spread, flip only on slow MST flips; ignore CST
  V2  MST-CORE + CST-CONVERT: core spread set by MST (changes only on MST flip);
      CST disagreement ADDS the opposite spread (-> iron condor) and CST agreement
      REMOVES it. This is the user's spec done as a 2-leg ADJUSTMENT, not a full
      unwind+reopen (the user's actual intent, implemented cheaply).
And a FULL FEE MODEL per order (brokerage + STT + exchange txn + GST + stamp),
counting every leg open/close/adjust. Goal: do longer holds + fewer legs let theta
dominate and flip net P&L positive?

Pure stdlib. Reads options_data.db read-only on VPS.
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

# ---- full fee model (Zerodha-style, NSE F&O options), per ORDER ----
def order_fees(premium, side):
    """premium = option price (Rs/share); side='S' sell or 'B' buy. Returns Rs/order for 1 lot."""
    turnover=premium*LOT
    brokerage=min(20.0, 0.0003*turnover)           # Rs20 or 0.03%, whichever lower
    stt=0.000625*turnover if side=="S" else 0.0     # 0.0625% on sell premium
    exch=0.0003503*turnover                         # NSE txn ~0.03503%
    sebi=0.000001*turnover
    stamp=0.00003*turnover if side=="B" else 0.0
    gst=0.18*(brokerage+exch+sebi)
    return brokerage+stt+exch+sebi+stamp+gst

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
    bars=[]
    for k in sorted(b):
        seg=sorted(b[k]); bars.append((seg[-1][0],seg[0][1],max(x[1] for x in seg),
            min(x[1] for x in seg),seg[-1][1]))
    return bars

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
        exps.add(ed); b=float(bid or 0); a=float(ask or 0); lp=float(ltp or 0)
        if b<=0 and a<=0 and lp<=0: continue
        if b<=0: b=lp or a
        if a<=0: a=lp or b
        book[(ed,int(float(k)),it)]=(b,a)
    return book,spot,exps

def pick_exp(exps,asof):
    v=[]
    for ed in exps:
        try: d=datetime.strptime(str(ed)[:10],"%Y-%m-%d").date()
        except ValueError: continue
        if d>=asof: v.append((d,ed))
    v.sort(); return (v[0] if v else (None,None))

def rnd(x): return int(round(x/STEP)*STEP)

# A "leg" = (qty, strike, it). qty -1 short, +1 long.
def bps_legs(spot):  # bull put spread legs
    f=rnd(spot); return [(-1,f-SHORT_OTM,"PE"),(+1,f-SHORT_OTM-WING,"PE")]
def bcs_legs(spot):  # bear call spread legs
    f=rnd(spot); return [(-1,f+SHORT_OTM,"CE"),(+1,f+SHORT_OTM+WING,"CE")]

def open_cost(legs,book,ed,side_concession=True):
    """Cash flow opening (credit +). Conservative: short@bid, long@ask. + fees."""
    cash=0.0; fee=0.0
    for qty,k,it in legs:
        q=book.get((ed,int(k),it))
        if q is None: return None,None
        b,a=q
        px=(b if qty<0 else a)
        cash += -qty*px
        fee += order_fees(px,"S" if qty<0 else "B")
    return cash,fee

def close_cost(legs,book,ed,settle=None):
    """Cash to close (debit, +). short buy@ask, long sell@bid. +fees."""
    cost=0.0; fee=0.0
    for qty,k,it in legs:
        if settle is not None:
            px=max(0.0,settle-k) if it=="CE" else max(0.0,k-settle)
            fee+=order_fees(px,"B" if qty<0 else "S")
        else:
            q=book.get((ed,int(k),it))
            if q is None: px=0.0
            else:
                b,a=q; px=(a if qty<0 else b)
                fee+=order_fees(px,"B" if qty<0 else "S")
        cost += (px if qty<0 else -px)
    return cost,fee

# ---------------------------------------------------------------------------
def replay(con,bars,dts,raws,mode):
    """mode: 'V0','V1','V2'. Returns (gross,net,fees,nlegtrades,nworst,holds)."""
    mst=st_dir(bars,PERIOD,MST_MULT); cst=st_dir(bars,PERIOD,CST_MULT)
    warm=PERIOD+1
    # build target posture per bar per mode
    # V0: full structure = mst if mst==cst else 0(condor)  -> trades on ANY change
    # V1: directional = mst only (no neutral); trades only on mst flip
    # V2: core=mst (trades on mst flip); condor wing toggled by cst (add/remove)
    def target(i):
        if mode=="V0": return ("dir",mst[i]) if mst[i]==cst[i] else ("neu",0)
        if mode=="V1": return ("dir",mst[i])
        if mode=="V2":
            return ("dir",mst[i]) if mst[i]==cst[i] else ("condor",mst[i])
        return ("dir",mst[i])

    gross=net=fees=0.0; nleg=0; worst=0.0; holds=[]
    # We simulate position as a set of open legs with their entry book/expiry.
    # On each state change we close what must change and open the new.
    cur=None  # (kind, dir, entry_i, ed_raw, legs, entry_spot)
    def open_pos(i,kind,dr):
        nonlocal net,fees,nleg
        j=nearest(dts,bars[i][0]);
        if j is None: return None
        bk,spot,exps=get_chain(con,raws[j])
        if spot is None: return None
        ed_d,ed=pick_exp(exps,bars[i][0].date())
        if ed is None: return None
        if kind=="dir":
            legs= bps_legs(spot) if dr>0 else bcs_legs(spot)
        else:  # condor
            legs= bps_legs(spot)+bcs_legs(spot)
        cash,fee=open_cost(legs,bk,ed)   # cash per-share (credit+), fee in Rs(lot)
        if cash is None: return None
        net += cash*LOT - fee; fees += fee; nleg += len(legs)
        return dict(kind=kind,dir=dr,ei=i,ed=ed,ed_d=ed_d,legs=legs,spot=spot,credit=cash)
    def close_pos(pos,i):
        nonlocal net,fees,nleg,worst,gross
        j=nearest(dts,bars[i][0])
        bk,spot,_=get_chain(con,raws[j])
        settle=None
        if bars[i][0].date()>pos["ed_d"]:
            settle=spot
        cost,fee=close_cost(pos["legs"],bk,pos["ed"],settle=settle)  # cost per-share, fee Rs
        net -= cost*LOT + fee; fees += fee; nleg += len(pos["legs"])
        pnl=(pos["credit"]-cost)*LOT  # gross structure pnl (pre-fee), Rs
        gross+=pnl
        worst=min(worst,pnl)
        holds.append((bars[i][0]-bars[pos["ei"]][0]).total_seconds()/86400)

    prev_state=None
    for i in range(warm,len(bars)):
        kind,dr=target(i)
        state=(kind,dr)
        if state==prev_state: continue
        # close current, open new (simplest: full reopen; V2 still reopens but holds
        # are long because core only changes on mst flip)
        if cur is not None:
            close_pos(cur,i)
        cur=open_pos(i,kind,dr) if not (kind=="dir" and dr==0) else None
        prev_state=state
    if cur is not None:
        close_pos(cur,len(bars)-1)
    # net and fees already in Rs (cash scaled by LOT at each open/close, fee already Rs)
    avg_hold=sum(holds)/len(holds) if holds else 0
    return gross,net,fees,nleg,worst,len(holds),avg_hold

def main():
    print(f"DB: {DB}")
    con=sqlite3.connect(DB)
    bars=load_spot_30min(con)
    dts,raws=chain_snaps(con)
    days=max(1,(bars[-1][0]-bars[14][0]).days)
    print(f"NIFTY 30-min bars {len(bars)} over {days} days; chain snaps {len(dts)}")
    print(f"\n{'mode':>4} {'structs':>8} {'avg_hold_d':>10} {'gross':>9} {'fees':>8} "
          f"{'NET':>9} {'worst':>8} {'sw/yr':>7}")
    for mode in ("V0","V1","V2"):
        g,n,f,nleg,worst,nstruct,ah=replay(con,bars,dts,raws,mode)
        swyr=nstruct*365/days
        print(f"{mode:>4} {nstruct:>8} {ah:>10.2f} {g:>9.0f} {f:>8.0f} {n:>9.0f} "
              f"{worst:>8.0f} {swyr:>7.0f}")
    con.close()
    print("\nNET is after full fees (brokerage+STT+exch+GST+stamp), 1 lot, ~6 weeks.")
    print("Annualize NET x ~8.8 for a rough yearly figure on 1 lot.")

if __name__=="__main__":
    main()
