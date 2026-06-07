#!/usr/bin/env python3
"""
G2a — REAL 6-week chain replay of the dual-Supertrend regime with CREDIT
(theta-positive) defined-risk structures. Decision-grade transaction cost.

Pipeline:
 1. Reconstruct NIFTY 30-min bars from options_data.db `underlying_spot` table
    (the 5-min series doesn't reach the options window).
 2. Dual-ST regime (period 14 / MST 5 / CST 2.5 from G1):
        posture = mst_dir if mst_dir==cst_dir else 0  (+1 bull, -1 bear, 0 neutral)
 3. Switch counts for as-specced vs hysteresis (confirm 1/2/3 bars) variant.
 4. For the AS-SPECCED timeline, replay credit structures on the REAL chain:
        BULLISH -> bull put spread (sell put ~OTM, buy put WING lower)
        BEARISH -> bear call spread (sell call ~OTM, buy call WING higher)
        NEUTRAL -> iron condor (both)
    Held from switch to next switch (overnight OK = defined risk). Priced both
    conservatively (sell@bid / buy@ask) and at mid; the gap = true round-trip
    transaction cost. Reports net P&L, theta carry, worst loss (spike proxy),
    and avg cost per switch vs G1 break-even.

Pure stdlib. Reads options_data.db read-only on the VPS.
"""
import sqlite3, os, math, bisect
from datetime import datetime
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESDIR = os.path.join(os.path.dirname(HERE), "results")
os.makedirs(RESDIR, exist_ok=True)
DB = "/home/arun/quantifyd/backtest_data/options_data.db"

SYM = "NIFTY"
LOT = 75
PERIOD, MST_MULT, CST_MULT = 14, 5.0, 2.5   # G1 winner
SHORT_OTM = 50       # short strike this many pts OTM from spot
WING = 300           # long strike this many pts beyond short (defined risk width)
STRIKE_STEP = 50

def parse_ts(s):
    s = str(s).replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try: return datetime.strptime(s, fmt)
        except ValueError: continue
    return datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")

# ---------------------------------------------------------------------------
def load_spot_30min(con):
    rows = con.execute(
        "SELECT snapshot_time, spot_price FROM underlying_spot "
        "WHERE symbol=? AND spot_price IS NOT NULL ORDER BY snapshot_time",
        (SYM,)).fetchall()
    buckets = defaultdict(list)
    for ts, sp in rows:
        dt = parse_ts(ts)
        mins = dt.hour*60 + dt.minute
        if mins < 9*60+15 or mins > 15*60+30: continue
        bidx = (mins - (9*60+15)) // 30
        buckets[(dt.date(), bidx)].append((dt, float(sp)))
    bars = []
    for key in sorted(buckets):
        seg = sorted(buckets[key])
        o = seg[0][1]; c = seg[-1][1]
        h = max(x[1] for x in seg); l = min(x[1] for x in seg)
        bars.append((seg[-1][0], o, h, l, c))
    return bars

def supertrend_dir(bars, period, mult):
    n = len(bars)
    if n < period+1: return [1]*n
    tr=[0.0]*n
    for i in range(n):
        _,o,h,l,c=bars[i]
        tr[i]=(h-l) if i==0 else max(h-l,abs(h-bars[i-1][4]),abs(l-bars[i-1][4]))
    atr=[0.0]*n; atr[period-1]=sum(tr[:period])/period
    for i in range(period,n): atr[i]=(atr[i-1]*(period-1)+tr[i])/period
    fub=[0.0]*n; flb=[0.0]*n; st=[0.0]*n; d=[1]*n
    for i in range(n):
        _,o,h,l,c=bars[i]
        if i<period: continue
        mid=(h+l)/2; bub=mid+mult*atr[i]; blb=mid-mult*atr[i]
        pub=fub[i-1] if fub[i-1] else bub; plb=flb[i-1] if flb[i-1] else blb
        fub[i]=bub if (bub<pub or bars[i-1][4]>pub) else pub
        flb[i]=blb if (blb>plb or bars[i-1][4]<plb) else plb
        pst=st[i-1] if st[i-1] else fub[i]
        if pst==(fub[i-1] if fub[i-1] else fub[i]):
            st[i]=fub[i] if c<=fub[i] else flb[i]
        else:
            st[i]=flb[i] if c>=flb[i] else fub[i]
        d[i]=1 if c>=st[i] else -1
    return d

def regime(bars):
    mst=supertrend_dir(bars,PERIOD,MST_MULT)
    cst=supertrend_dir(bars,PERIOD,CST_MULT)
    warm=PERIOD+1
    post=[0]*len(bars)
    for i in range(len(bars)):
        post[i]=0 if i<warm else (mst[i] if mst[i]==cst[i] else 0)
    return post, warm

def hysteresis(post, warm, confirm):
    """Only change posture after `confirm` consecutive bars agree."""
    out=list(post); cur=0; run=0; pend=post[warm]
    for i in range(warm,len(post)):
        if post[i]==pend: run+=1
        else: pend=post[i]; run=1
        if run>=confirm: cur=pend
        out[i]=cur
    return out

def count_switches(post, warm):
    sw=0; prev=0
    for i in range(warm,len(post)):
        if post[i]!=prev: sw+=1; prev=post[i]
    return sw

# ---------------------------------------------------------------------------
def chain_snapshots(con):
    rows=con.execute("SELECT DISTINCT snapshot_time FROM option_chain WHERE symbol=? "
                     "ORDER BY snapshot_time",(SYM,)).fetchall()
    return [parse_ts(r[0]) for r in rows], [r[0] for r in rows]

def nearest_snap(dts, target):
    i=bisect.bisect_left(dts,target)
    cands=[j for j in (i-1,i) if 0<=j<len(dts)]
    if not cands: return None
    return min(cands,key=lambda j:abs((dts[j]-target).total_seconds()))

def get_chain(con, snap_raw):
    rows=con.execute(
        "SELECT expiry_date,strike,instrument_type,bid,ask,ltp,underlying_spot "
        "FROM option_chain WHERE symbol=? AND snapshot_time=?",(SYM,snap_raw)).fetchall()
    book={}; spot=None; expiries=set()
    for ed,k,it,bid,ask,ltp,us in rows:
        if us: spot=float(us)
        expiries.add(ed)
        b=float(bid) if bid else 0.0; a=float(ask) if ask else 0.0
        lp=float(ltp) if ltp else 0.0
        if b<=0 and a<=0 and lp<=0: continue
        # fallbacks when one side missing
        if b<=0: b=lp if lp>0 else a
        if a<=0: a=lp if lp>0 else b
        book[(ed,int(float(k)),it)]=(b,a,lp)
    return book, spot, expiries

def pick_expiry(expiries, asof_date):
    valid=[]
    for ed in expiries:
        try: d=datetime.strptime(str(ed)[:10],"%Y-%m-%d").date()
        except ValueError: continue
        if d>=asof_date: valid.append((d,ed))
    if not valid: return None,None
    valid.sort(); return valid[0]

def rnd(x): return int(round(x/STRIKE_STEP)*STRIKE_STEP)

def price_legs(book, expiry_raw, spot, state):
    """Return list of (qty, leg_price_bid, leg_price_ask) and structure label.
    qty +1 = long (pay ask to open), -1 = short (receive bid to open)."""
    legs=[]; lab=""
    sp_floor=rnd(spot)
    def L(strike,it):  # fetch leg
        return book.get((expiry_raw,int(strike),it))
    if state==1 or state==0:   # bull put spread (also part of condor)
        sk=sp_floor-SHORT_OTM; lk=sk-WING
        s=L(sk,"PE"); l=L(lk,"PE")
        if s and l: legs += [(-1,)+s[:2], (+1,)+l[:2]]; lab+="BPS "
    if state==-1 or state==0:  # bear call spread
        sk=sp_floor+SHORT_OTM; lk=sk+WING
        s=L(sk,"CE"); l=L(lk,"CE")
        if s and l: legs += [(-1,)+s[:2], (+1,)+l[:2]]; lab+="BCS "
    return legs, lab.strip()

def open_value(legs, mode):
    """Cash received opening (credit positive). mode: 'cons' or 'mid'."""
    cash=0.0
    for qty,bid,ask in legs:
        px = (bid if qty<0 else ask) if mode=="cons" else (bid+ask)/2
        cash += -qty*px   # short(-1): +px received; long(+1): -px paid
    return cash

def close_value(legs_struct, book_close, expiry_raw, mode, settle_spot=None):
    """Cost to close (debit positive reduces P&L). legs_struct = original legs w/ strikes."""
    cost=0.0
    for qty,strike,it in legs_struct:
        if settle_spot is not None:  # expiry intrinsic settlement
            intr = max(0.0, settle_spot-strike) if it=="CE" else max(0.0, strike-settle_spot)
            px=intr
        else:
            q=book_close.get((expiry_raw,int(strike),it))
            if q is None: px=0.0
            else:
                b,a,lp=q
                px=(ask if qty<0 else b) if mode=="cons" and False else 0
                # to CLOSE: short(-1) buy back at ask; long(+1) sell at bid
                px=(a if qty<0 else b) if mode=="cons" else (b+a)/2
        cost += qty*px*-1 if False else (px if qty<0 else -px)
    return cost

# ---------------------------------------------------------------------------
def main():
    print(f"DB: {DB}")
    con=sqlite3.connect(DB)
    bars=load_spot_30min(con)
    if not bars:
        print("NO underlying_spot data"); return
    print(f"NIFTY 30-min bars from options window: {len(bars)} "
          f"[{bars[0][0]} .. {bars[-1][0]}]")
    post,warm=regime(bars)
    base_sw=count_switches(post,warm)
    print(f"\nSwitch counts over window (~{(bars[-1][0]-bars[warm][0]).days} days):")
    print(f"  as-specced (no hysteresis): {base_sw} switches")
    for cf in (2,3):
        h=hysteresis(post,warm,cf)
        print(f"  hysteresis confirm={cf} bars : {count_switches(h,warm)} switches")
    days=max(1,(bars[-1][0]-bars[warm][0]).days)
    print(f"  -> as-specced annualized ~{base_sw*365/days:.0f} switches/yr "
          f"(G1 said ~136; break-even ~10bps/switch)")

    # Build switch segments (entry bar -> exit bar) on as-specced timeline
    segs=[]; prev=0; entry_i=None
    for i in range(warm,len(bars)):
        if post[i]!=prev:
            if entry_i is not None and prev!=0:
                segs.append((entry_i,i,prev))
            elif entry_i is not None and prev==0:
                segs.append((entry_i,i,0))
            entry_i=i; prev=post[i]
    if entry_i is not None:
        segs.append((entry_i,len(bars)-1,prev))

    dts,raws=chain_snapshots(con)
    print(f"\nChain snapshots available: {len(dts)} "
          f"[{dts[0]} .. {dts[-1]}]")

    print(f"\n=== CREDIT-STRUCTURE REPLAY (real bid/ask) ===")
    print(f"{'seg':>3} {'state':>6} {'entry':>16} {'hold_d':>6} {'struct':>8} "
          f"{'cons_PnL':>9} {'mid_PnL':>9} {'cost':>7}")
    tot_cons=tot_mid=tot_cost=0.0; worst=0.0; nseg=0
    for k,(ei,xi,state) in enumerate(segs):
        t_entry=bars[ei][0]; t_exit=bars[xi][0]
        je=nearest_snap(dts,t_entry); jx=nearest_snap(dts,t_exit)
        if je is None or jx is None: continue
        bk_e,spot_e,exps_e=get_chain(con,raws[je])
        if spot_e is None: continue
        ed_date,ed_raw=pick_expiry(exps_e,t_entry.date())
        if ed_raw is None: continue
        legs,lab=price_legs(bk_e,ed_raw,spot_e,state)
        if not legs: continue
        legs_struct=[]   # (qty,strike,it) for closing
        sp_floor=rnd(spot_e)
        if state in (1,0):
            legs_struct += [(-1,sp_floor-SHORT_OTM,"PE"),(+1,sp_floor-SHORT_OTM-WING,"PE")]
        if state in (-1,0):
            legs_struct += [(-1,sp_floor+SHORT_OTM,"CE"),(+1,sp_floor+SHORT_OTM+WING,"CE")]
        cred_cons=open_value(legs,"cons"); cred_mid=open_value(legs,"mid")
        # close
        exp_d=ed_date
        if t_exit.date()>exp_d:   # expiry hit before next switch: settle intrinsic
            spot_settle=None
            # find spot near expiry close
            for j in range(len(dts)):
                if dts[j].date()==exp_d:
                    _,ss,_=get_chain(con,raws[j])
                    if ss: spot_settle=ss
            if spot_settle is None: spot_settle=spot_e
            close_cons=close_value(legs_struct,None,ed_raw,"cons",settle_spot=spot_settle)
            close_mid=close_cons
        else:
            bk_x,_,_=get_chain(con,raws[jx])
            close_cons=close_value(legs_struct,bk_x,ed_raw,"cons")
            close_mid=close_value(legs_struct,bk_x,ed_raw,"mid")
        pnl_cons=(cred_cons-close_cons)*LOT
        pnl_mid=(cred_mid-close_mid)*LOT
        cost=pnl_mid-pnl_cons   # transaction cost = mid minus conservative
        hold_d=(t_exit-t_entry).total_seconds()/86400
        sname={1:"BULL",-1:"BEAR",0:"NEUT"}[state]
        print(f"{k:>3} {sname:>6} {str(t_entry):>16} {hold_d:>6.1f} {lab:>8} "
              f"{pnl_cons:>9.0f} {pnl_mid:>9.0f} {cost:>7.0f}")
        tot_cons+=pnl_cons; tot_mid+=pnl_mid; tot_cost+=cost
        worst=min(worst,pnl_cons); nseg+=1
    print("-"*80)
    print(f"Segments traded: {nseg}")
    print(f"TOTAL net P&L  conservative (real bid/ask fills): Rs {tot_cons:,.0f}")
    print(f"TOTAL net P&L  mid-price (zero-cost ideal)      : Rs {tot_mid:,.0f}")
    print(f"TOTAL transaction cost (bid/ask drag)          : Rs {tot_cost:,.0f}")
    print(f"Worst single structure (spike proxy)           : Rs {worst:,.0f}")
    if nseg:
        avg_cost=tot_cost/nseg
        # cost per switch in NIFTY points & bps of notional
        notional=bars[-1][4]*LOT
        print(f"Avg transaction cost / switch                  : Rs {avg_cost:,.0f} "
              f"= {avg_cost/LOT:.1f} pts = {1e4*avg_cost/notional:.1f} bps of 1-lot notional")
    con.close()

if __name__=="__main__":
    main()
