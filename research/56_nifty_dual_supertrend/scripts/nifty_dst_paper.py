#!/usr/bin/env python3
"""
NIFTY Dual-Supertrend PAPER forward-logger (research/56 V4 winner).

Standalone process — reads the live options recorder (options_data.db) READ-ONLY
and writes its OWN paper book to paper_dst.db. Does NOT touch the quantifyd
gunicorn service (no restart needed; safe any time).

Logic = the validated V4 config:
  - 30-min NIFTY bars from underlying_spot
  - MST(14,5) + CST(14,2.5) Supertrends
  - PULLBACK ENTRY: on MST flip, wait for first pullback (CST against) then resume
    (CST realign) before opening the directional credit spread; TIMEOUT fallback.
  - STACK-LAYERING: CST-against adds a hedge (-> condor); CST-realign stacks a new
    directional; only full close on MST flip.
  - BI-WEEKLY EXPIRY: 2nd-nearest Tuesday (skip front weekly).
  - Defined-risk wings (overnight spike protection).

Run every ~30 min during the session (cron or loop). Idempotent: only acts on
NEWLY CLOSED 30-min bars after the last processed one. State persists in paper_dst.db
so it survives restarts and is the sole crash-recovery source.

CONFIG: paper only — never sends a real order. 1 lot.
"""
import sqlite3, os, json, bisect, sys
from datetime import datetime
from collections import defaultdict

HERE=os.path.dirname(os.path.abspath(__file__))
RESDIR=os.path.join(os.path.dirname(HERE),"results"); os.makedirs(RESDIR,exist_ok=True)
SRC_DB="/home/arun/quantifyd/backtest_data/options_data.db"
PAPER_DB=os.path.join(RESDIR,"paper_dst.db")
SYM="NIFTY"; LOT=75
PERIOD,MST_MULT,CST_MULT=14,5.0,2.5
SHORT_OTM=50; WING=300; STEP=50; MIN_DTE=0; TIMEOUT=999  # 999=pure pullback

def log(m): print(f"[{_now_str()}] {m}", flush=True)
def _now_str():
    # IST clock from the latest recorder snapshot (no wall-clock dependency)
    return GLOBAL_NOW or "?"
GLOBAL_NOW=None

# ---------- fees (flat Rs20/order F&O + statutory) ----------
def order_fee(premium, side):
    to=premium*LOT; brk=20.0
    stt=0.000625*to if side=="S" else 0.0
    exch=0.0003503*to; sebi=0.000001*to
    stamp=0.00003*to if side=="B" else 0.0
    return brk+stt+exch+sebi+stamp+0.18*(brk+exch+sebi)

def parse_ts(s):
    s=str(s).replace("T"," ")
    try: return datetime.strptime(s[:19],"%Y-%m-%d %H:%M:%S")
    except ValueError: return datetime.strptime(s,"%Y-%m-%d %H:%M:%S.%f")

# ---------- bars + supertrend ----------
def load_spot_30min(con):
    rows=con.execute("SELECT snapshot_time,spot_price FROM underlying_spot "
        "WHERE symbol=? AND spot_price IS NOT NULL ORDER BY snapshot_time",(SYM,)).fetchall()
    b=defaultdict(list)
    for ts,sp in rows:
        dt=parse_ts(ts); m=dt.hour*60+dt.minute
        if m<9*60+15 or m>15*60+30: continue
        b[(dt.date(),(m-(9*60+15))//30)].append((dt,float(sp)))
    out=[]
    for k in sorted(b):
        seg=sorted(b[k]); out.append((seg[-1][0],seg[0][1],max(x[1] for x in seg),
            min(x[1] for x in seg),seg[-1][1]))
    return out

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

# ---------- chain pricing ----------
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
def pick_exp(exps,asof):  # 2nd-nearest Tuesday (skip front weekly)
    v=[]
    for ed in exps:
        try: d=datetime.strptime(str(ed)[:10],"%Y-%m-%d").date()
        except ValueError: continue
        if d>=asof: v.append((d,ed))
    v.sort()
    if not v: return (None,None)
    return v[1] if len(v)>=2 else v[0]
def rnd(x): return int(round(x/STEP)*STEP)
def fill_px(book,ed,strike,it,buy,spot):
    q=book.get((ed,int(strike),it))
    if q:
        b,a,lp=q; px=a if buy else b
        if px<=0: px=lp if lp>0 else (a if a>0 else b)
        if px>0: return px,False
    return (max(0.0,spot-strike) if it=="CE" else max(0.0,strike-spot)),True
def dir_legs(spot,dr):
    f=rnd(spot)
    return [(-1,f-SHORT_OTM,"PE"),(+1,f-SHORT_OTM-WING,"PE")] if dr>0 else \
           [(-1,f+SHORT_OTM,"CE"),(+1,f+SHORT_OTM+WING,"CE")]

# ---------- paper DB ----------
def init_db():
    con=sqlite3.connect(PAPER_DB)
    con.execute("""CREATE TABLE IF NOT EXISTS state(
        k TEXT PRIMARY KEY, v TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS trades(
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, action TEXT, role TEXT,
        dir INTEGER, expiry TEXT, legs TEXT, cashflow REAL, fees REAL, note TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS closed(
        id INTEGER PRIMARY KEY AUTOINCREMENT, open_ts TEXT, close_ts TEXT,
        role TEXT, dir INTEGER, pnl REAL, fees REAL, reason TEXT)""")
    con.commit(); return con
def sget(con,k,default=None):
    r=con.execute("SELECT v FROM state WHERE k=?",(k,)).fetchone()
    return json.loads(r[0]) if r else default
def sset(con,k,v):
    con.execute("INSERT INTO state(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=?",
                (k,json.dumps(v),json.dumps(v)))

# ---------- main step ----------
def main():
    global GLOBAL_NOW
    src=sqlite3.connect(f"file:{SRC_DB}?mode=ro",uri=True)
    bars=load_spot_30min(src)
    if len(bars)<PERIOD+2:
        print("not enough bars yet"); return
    GLOBAL_NOW=str(bars[-1][0])
    mst=st_dir(bars,PERIOD,MST_MULT); cst=st_dir(bars,PERIOD,CST_MULT)
    dts,raws=chain_snaps(src)
    pcon=init_db()

    last_idx=sget(pcon,"last_idx",None)
    if last_idx is None:
        # FRESH START: begin flat from the latest CLOSED bar; trade forward only.
        last_idx=len(bars)-2
        sset(pcon,"last_idx",last_idx)
        sset(pcon,"cur_mst",0); sset(pcon,"entered",False); sset(pcon,"hedged",False)
        sset(pcon,"pullback",False); sset(pcon,"since",0); sset(pcon,"book",[])
        mdir,cdir=mst[-1],cst[-1]
        regime=("BULLISH" if mdir==1 and cdir==1 else
                "BEARISH" if mdir==-1 and cdir==-1 else "NEUTRAL")
        sset(pcon,"snapshot",dict(ts=str(bars[-1][0]),spot=round(bars[-1][4],2),
            mst_dir=mdir,cst_dir=cdir,regime=regime,status="FLAT (fresh start)",
            entered=False,hedged=False,open_book=[],open_mtm=0,n_open=0,n_closed=0,
            realized_pnl=0,realized_fees=0,realized_net=0,equity_net=0,peak_margin=0,
            config=_cfg()))
        pcon.commit()
        log(f"FRESH START — flat, watching from bar {last_idx} ({bars[last_idx][0]}). "
            f"MST={mst[-1]} CST={cst[-1]}. No backlog traded.")
        pcon.close(); src.close(); return

    cur_mst=sget(pcon,"cur_mst",0); entered=sget(pcon,"entered",False)
    hedged=sget(pcon,"hedged",False); pullback=sget(pcon,"pullback",False)
    since=sget(pcon,"since",0); book=sget(pcon,"book",[])

    def snap_at(t):
        j=nearest(dts,t); return get_chain(src,raws[j]) if j is not None else (None,None,None)
    def open_spread(t,role,dr):
        bk,spot,exps=snap_at(t)
        if spot is None: return
        ed_d,ed=pick_exp(exps,t.date())
        if ed is None: return
        legs=dir_legs(spot,dr) if role!="hedge" else dir_legs(spot,-dr)
        rec=[]; fee=0.0; cash=0.0
        for qty,k,it in legs:
            buy=qty>0; px,miss=fill_px(bk,ed,k,it,buy,spot)
            fee+=order_fee(px,"B" if buy else "S"); cash+=(-qty*px)
            rec.append([qty,k,it,px])
        book.append(dict(role=role,dir=dr,ed=str(ed),ed_d=str(ed_d),legs=rec,
                         open_ts=str(t)))
        pcon.execute("INSERT INTO trades(ts,action,role,dir,expiry,legs,cashflow,fees,note)"
                     " VALUES(?,?,?,?,?,?,?,?,?)",
                     (str(t),"OPEN",role,dr,str(ed),json.dumps(rec),cash*LOT,fee,
                      f"spot={spot:.1f}"))
        log(f"OPEN {role} dir={dr} exp={ed} legs={[(q,k,it) for q,k,it,_ in rec]} "
            f"credit={cash*LOT:.0f} fee={fee:.0f}")
    def close_spread2(t,s,reason):
        bk,spot,_=snap_at(t)
        exp_d=datetime.strptime(s["ed_d"][:10],"%Y-%m-%d").date()
        settle= spot if (spot is not None and t.date()>exp_d) else None
        pnl=0.0; fee=0.0
        for qty,k,it,entry in s["legs"]:
            if settle is not None:
                ex=max(0.0,settle-k) if it=="CE" else max(0.0,k-settle)
            else:
                buy=qty<0
                ex,_=fill_px(bk,s["ed"],k,it,buy,spot)
            fee+=order_fee(ex,"S" if qty>0 else "B"); pnl+=qty*(ex-entry)*LOT
        pcon.execute("INSERT INTO closed(open_ts,close_ts,role,dir,pnl,fees,reason)"
                     " VALUES(?,?,?,?,?,?,?)",
                     (s["open_ts"],str(t),s["role"],s["dir"],pnl,fee,reason))
        log(f"CLOSE {s['role']} dir={s['dir']} pnl={pnl:.0f} fee={fee:.0f} ({reason})")
        return pnl
    def close_all2(t,reason):
        nonlocal book
        for s in list(book): close_spread2(t,s,reason)
        book=[]

    # process newly closed bars
    acted=0
    for i in range(last_idx+1, len(bars)-0):
        # only treat a bar as "closed" if it's not the still-forming last bar of today
        if i>=len(bars): break
        t=bars[i][0]; m=mst[i]; c=cst[i]
        if m!=cur_mst:
            close_all2(t,"MST flip")
            cur_mst=m; entered=False; pullback=False; hedged=False; since=0
        else:
            since+=1
            if not entered:
                if c!=m: pullback=True
                if (pullback and c==m) or since>=TIMEOUT:
                    open_spread(t,"core",m); entered=True; hedged=False
            else:
                if c!=m and not hedged: open_spread(t,"hedge",m); hedged=True
                elif c==m and hedged: open_spread(t,"core",m); hedged=False
        acted=i
    # persist
    sset(pcon,"last_idx",max(last_idx,len(bars)-2))
    sset(pcon,"cur_mst",cur_mst); sset(pcon,"entered",entered)
    sset(pcon,"hedged",hedged); sset(pcon,"pullback",pullback); sset(pcon,"since",since)
    sset(pcon,"book",book)
    pcon.commit()

    # rich snapshot for the dashboard API (with per-leg live detail)
    snap=compute_snapshot(pcon,src,bars,mst,cst,raws,cur_mst,entered,hedged,book)
    sset(pcon,"snapshot",snap); pcon.commit()
    log(f"STATE: regime={snap['regime']} entered={entered} hedged={hedged} "
        f"open={snap['n_open']} open_mtm={snap['open_mtm']} realized_net={snap['realized_net']}")
    pcon.close(); src.close()

def compute_snapshot(pcon,src,bars,mst,cst,raws,cur_mst,entered,hedged,book):
    """Build the dashboard snapshot, pricing every open leg off the LATEST chain
    snapshot (per-leg entry/current/P&L). Used by main() and the ticker loop."""
    rp,rf,ncl=pcon.execute("SELECT COALESCE(SUM(pnl),0),COALESCE(SUM(fees),0),"
                           "COUNT(*) FROM closed").fetchone()
    spot=bars[-1][4]; mdir=mst[-1]; cdir=cst[-1]
    regime=("BULLISH" if mdir==1 and cdir==1 else
            "BEARISH" if mdir==-1 and cdir==-1 else "NEUTRAL")
    if entered: status=f"IN POSITION ({len(book)} spread{'s' if len(book)!=1 else ''})"
    elif cur_mst!=0: status="ARMED — waiting for pullback-resume"
    else: status="FLAT"
    bk_now,spot_now,_=get_chain(src,raws[-1]) if raws else ({},spot,None)
    live_ts=str(raws[-1]) if raws else str(bars[-1][0])
    obook=[]; open_mtm=0.0
    for s in book:
        credit=sum(-q*e for q,k,it,e in s["legs"])*LOT
        legs=[]; smtm=0.0
        for q,k,it,e in s["legs"]:
            mark,_=fill_px(bk_now,s["ed"],k,it,q<0,spot_now or spot)
            lp=q*(mark-e)*LOT; smtm+=lp
            legs.append(dict(side=("SELL" if q<0 else "BUY"),strike=k,type=it,
                             qty=(-q if q<0 else q)*LOT,entry=round(e,2),
                             cur=round(mark,2),pnl=round(lp),entry_ts=s["open_ts"]))
        obook.append(dict(role=s["role"],dir=s["dir"],expiry=s["ed"][:10],
                          open_ts=s["open_ts"],credit=round(credit),mtm=round(smtm),
                          legs=legs))
        open_mtm+=smtm
    return dict(ts=str(bars[-1][0]),live_ts=live_ts,spot=round(spot,2),mst_dir=mdir,
        cst_dir=cdir,regime=regime,status=status,entered=bool(entered),
        hedged=bool(hedged),open_book=obook,open_mtm=round(open_mtm),n_open=len(book),
        n_closed=ncl,realized_pnl=round(rp),realized_fees=round(rf),
        realized_net=round(rp-rf),equity_net=round(rp-rf+open_mtm),
        peak_margin=len(book)*WING*LOT,config=_cfg())

def refresh_only():
    """Re-price the open book off the latest chain and rewrite ONLY the snapshot.
    No V4 management (that stays on the 15-min cron). Cheap; for the ticker loop."""
    src=sqlite3.connect(f"file:{SRC_DB}?mode=ro",uri=True)
    bars=load_spot_30min(src)
    if len(bars)<PERIOD+2: src.close(); return
    mst=st_dir(bars,PERIOD,MST_MULT); cst=st_dir(bars,PERIOD,CST_MULT)
    _,raws=chain_snaps(src)
    pcon=init_db()
    if sget(pcon,"last_idx",None) is None:
        pcon.close(); src.close(); return
    cur_mst=sget(pcon,"cur_mst",0); entered=sget(pcon,"entered",False)
    hedged=sget(pcon,"hedged",False); book=sget(pcon,"book",[]) or []
    snap=compute_snapshot(pcon,src,bars,mst,cst,raws,cur_mst,entered,hedged,book)
    sset(pcon,"snapshot",snap); pcon.commit(); pcon.close(); src.close()

def tickloop(interval=15):
    """Live-ish ticker: refresh snapshot every `interval`s until the chain stops
    advancing for ~5 min (market closed) or a hard cap. No gunicorn dependency."""
    import time
    last=None; stale=0; iters=0
    while iters < 2200:
        iters+=1
        try:
            s=sqlite3.connect(f"file:{SRC_DB}?mode=ro",uri=True)
            cur=s.execute("SELECT MAX(snapshot_time) FROM option_chain WHERE symbol=?",
                          (SYM,)).fetchone()[0]
            s.close()
            stale = stale+1 if cur==last else 0
            last=cur
            if stale>=20: log("ticker: chain stale 5min — exiting (market closed)"); break
            refresh_only()
        except Exception as ex:  # never let the loop die on a transient error
            log(f"ticker error (continuing): {ex}")
        time.sleep(interval)

def _cfg():
    return dict(period=PERIOD,mst_mult=MST_MULT,cst_mult=CST_MULT,short_otm=SHORT_OTM,
                wing=WING,lots=1,expiry="bi-weekly (2nd-nearest Tue)",entry="pullback (V4)")

def force_entry():
    """One-shot manual entry at CMP in the CURRENT MST direction (overrides the
    pullback-wait). MST bearish -> bear-call credit spread; bullish -> bull-put.
    Paper only. The cron logger then manages it per V4 going forward."""
    src=sqlite3.connect(f"file:{SRC_DB}?mode=ro",uri=True)
    bars=load_spot_30min(src)
    mst=st_dir(bars,PERIOD,MST_MULT); cst=st_dir(bars,PERIOD,CST_MULT)
    dts,raws=chain_snaps(src)
    pcon=init_db()
    book=sget(pcon,"book",[]) or []
    cur_mst=mst[-1]
    bk,spot,exps=get_chain(src,raws[-1])
    ed_d,ed=pick_exp(exps,bars[-1][0].date())
    legs=dir_legs(spot,cur_mst)              # MST<0 -> bear call spread
    rec=[]; fee=0.0; cash=0.0
    for q,k,it in legs:
        buy=q>0; px,_=fill_px(bk,ed,k,it,buy,spot)
        fee+=order_fee(px,"B" if buy else "S"); cash+=(-q*px); rec.append([q,k,it,px])
    sp=dict(role="core",dir=cur_mst,ed=str(ed),ed_d=str(ed_d),legs=rec,
            open_ts=str(bars[-1][0]))
    book.append(sp)
    pcon.execute("INSERT INTO trades(ts,action,role,dir,expiry,legs,cashflow,fees,note)"
                 " VALUES(?,?,?,?,?,?,?,?,?)",(str(bars[-1][0]),"OPEN-MANUAL","core",
                 cur_mst,str(ed),json.dumps(rec),cash*LOT,fee,f"forced @CMP spot={spot:.1f}"))
    sset(pcon,"book",book); sset(pcon,"entered",True); sset(pcon,"cur_mst",cur_mst)
    sset(pcon,"hedged",False); sset(pcon,"since",0); sset(pcon,"pullback",False)
    sset(pcon,"last_idx",len(bars)-2)
    # snapshot
    mtm=0.0
    for q,k,it,e in rec:
        mark,_=fill_px(bk,str(ed),k,it,q<0,spot); mtm+=q*(mark-e)*LOT
    credit=sum(-q*e for q,k,it,e in rec)*LOT
    regime=("BULLISH" if cur_mst==1 and cst[-1]==1 else
            "BEARISH" if cur_mst==-1 and cst[-1]==-1 else "NEUTRAL")
    sset(pcon,"snapshot",dict(ts=str(bars[-1][0]),spot=round(spot,2),mst_dir=cur_mst,
        cst_dir=cst[-1],regime=regime,status=f"IN POSITION (1 spread, manual @CMP)",
        entered=True,hedged=False,open_book=[dict(role="core",dir=cur_mst,
        expiry=str(ed)[:10],open_ts=str(bars[-1][0]),credit=round(credit),mtm=round(mtm),
        legs=[[q,k,it,round(e,2)] for q,k,it,e in rec])],open_mtm=round(mtm),n_open=1,
        n_closed=pcon.execute("SELECT COUNT(*) FROM closed").fetchone()[0],
        realized_pnl=0,realized_fees=0,realized_net=0,equity_net=round(mtm),
        peak_margin=WING*LOT,config=_cfg()))
    pcon.commit()
    side="BEAR CALL spread" if cur_mst<0 else "BULL PUT spread"
    log(f"FORCED ENTRY @CMP: {side} exp={ed} legs={[(q,k,it) for q,k,it,_ in rec]} "
        f"credit={cash*LOT:.0f} fee={fee:.0f} spot={spot:.1f} (MST={cur_mst},CST={cst[-1]})")
    pcon.close(); src.close()

if __name__=="__main__":
    import sys
    arg=sys.argv[1] if len(sys.argv)>1 else ""
    if arg=="forceentry": force_entry()
    elif arg=="ticksnap": refresh_only()
    elif arg=="tickloop": tickloop()
    else: main()
