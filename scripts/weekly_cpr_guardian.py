#!/usr/bin/env python3
"""Weekly CPR Guardian harness (research/67) — TOKEN-INDEPENDENT, with GRADED trip-wire + R3/S3.
Data: NIFTY HLC from the options recorder (underlying_spot); live spot from /api/nas/ticker/status.
  --assess [--now]              -> classify the week -> verdict + levels (incl R3/S3), save state
  --monitor [--belo X --behi Y] -> live price vs levels/breakevens -> CALM / graded WATCH-ALERT-ACT lines
Graded downside trip-wire (research/67 validated): intraday poke below CPR = WATCH (S1 holds only 32%
intraday); a 30-min CLOSE below CPR = ALERT (~55% the week flips, base 20%); a 30-min close below S1 =
ACT (88% continuation)."""
import sys, json, sqlite3, urllib.request
from datetime import date, datetime, timedelta, timezone
OPT="backtest_data/options_data.db"; STATE="/tmp/cpr_guardian_state.json"
def load():
    try: return json.load(open(STATE))
    except Exception: return {}
def save(s): json.dump(s, open(STATE,"w"))
def ist(): return datetime.now(timezone.utc).replace(tzinfo=None)+timedelta(hours=5,minutes=30)
def cpr(h,l,c):
    p=(h+l+c)/3.0; bc=(h+l)/2.0; tc=2*p-bc; lo,hi=min(bc,tc),max(bc,tc)
    return dict(p=p,lo=lo,hi=hi,r1=2*p-l,s1=2*p-h,r2=p+(h-l),s2=p-(h-l),r3=h+2*(p-l),s3=l-2*(h-p),width=abs(tc-bc)/c*100)
def sd(px,lo,hi): return "ABOVE" if px>hi else ("BELOW" if px<lo else "INSIDE")
def live_spot():
    try:
        d=json.load(urllib.request.urlopen("http://127.0.0.1:5000/api/nas/ticker/status",timeout=4))
        if d.get("last_ltp"): return d["last_ltp"]
    except Exception: pass
    con=sqlite3.connect(OPT); r=con.execute("SELECT spot_price FROM underlying_spot WHERE symbol='NIFTY' AND spot_price>0 ORDER BY snapshot_time DESC LIMIT 1").fetchone(); con.close()
    return r[0] if r else None
def last_30m_close(con):
    """Close of the most recently COMPLETED 30-min candle (aligned :15/:45), from the recorder."""
    n=ist(); m=n.minute
    if m>=45: bnd=n.replace(minute=45,second=0,microsecond=0)
    elif m>=15: bnd=n.replace(minute=15,second=0,microsecond=0)
    else: bnd=(n-timedelta(hours=1)).replace(minute=45,second=0,microsecond=0)
    t=bnd.strftime("%Y-%m-%dT%H:%M"); day=bnd.strftime("%Y-%m-%d")
    r=con.execute("SELECT spot_price FROM underlying_spot WHERE symbol='NIFTY' AND substr(snapshot_time,1,10)=? AND snapshot_time<=? AND spot_price>0 ORDER BY snapshot_time DESC LIMIT 1",(day,t+":59")).fetchone()
    return r[0] if r else None
def hlc_range(con,d1,d2):
    r=con.execute("SELECT MAX(spot_price),MIN(spot_price) FROM underlying_spot WHERE symbol='NIFTY' AND substr(snapshot_time,1,10) BETWEEN ? AND ? AND spot_price>0",(d1,d2)).fetchone()
    cc=con.execute("SELECT spot_price FROM underlying_spot WHERE symbol='NIFTY' AND substr(snapshot_time,1,10) BETWEEN ? AND ? AND spot_price>0 ORDER BY snapshot_time DESC LIMIT 1",(d1,d2)).fetchone()
    return (r[0],r[1],cc[0]) if r and r[0] and cc else None
def day_open(con,day):
    r=con.execute("SELECT spot_price FROM underlying_spot WHERE symbol='NIFTY' AND substr(snapshot_time,1,10)=? AND spot_price>0 ORDER BY snapshot_time LIMIT 1",(day,)).fetchone()
    return r[0] if r else None

def assess(now_as_close):
    con=sqlite3.connect(OPT); today=date.today(); ws=today-timedelta(days=today.weekday())
    pw=hlc_range(con,(ws-timedelta(days=7)).isoformat(),(ws-timedelta(days=1)).isoformat())
    pdays=[r[0] for r in con.execute("SELECT DISTINCT substr(snapshot_time,1,10) d FROM underlying_spot WHERE symbol='NIFTY' AND substr(snapshot_time,1,10)<? ORDER BY d DESC LIMIT 1",(today.isoformat(),))]
    pd=hlc_range(con,pdays[0],pdays[0]) if pdays else None
    sp=live_spot(); op=day_open(con,today.isoformat())
    if not(pw and pd): print("DATA NOT READY (recorder has no prior week/day NIFTY spots)"); return
    W=cpr(*pw); D=cpr(*pd)
    ref=sp if (sp and now_as_close) else (sp or (op or pw[2])); o=op or ref; col="GREEN" if ref>=o else "RED"
    fri_close=pd[2]; gap=(o-fri_close)/fri_close*100 if fri_close else 0
    wpos=sd(ref,W["lo"],W["hi"]); dpos=sd(ref,D["lo"],D["hi"])
    agree=("AGREE-UP" if wpos==dpos=="ABOVE" else "AGREE-DOWN" if wpos==dpos=="BELOW" else "INSIDE" if (wpos=="INSIDE" and dpos=="INSIDE") else "DISAGREE")
    if agree=="AGREE-UP" and col=="GREEN": rec="BULL tilt (72% hold,+0.36%) -> bull-put spread / bullish jade lizard, normal size"
    elif agree=="AGREE-UP" and col=="RED": rec="Holds up, ~0 net (NEUTRAL) -> iron fly/condor, sell the range"
    elif agree=="AGREE-DOWN" and col=="RED": rec="BEAR tilt (61%,-0.40%) -> bear-call/put-debit (defined), SMALLER size"
    elif wpos=="BELOW" and col=="GREEN": rec="Reversal-UP trap -> do NOT go bear; neutral or mild bull"
    elif agree=="DISAGREE": rec="Coin-flip (52%) -> NEUTRAL only: iron condor, short ~S1/R1, long ~S2/R2"
    else: rec="Inside/unclear -> small neutral premium"
    wsz=" | WIDE CPR: sideways-friendly, can size up" if W["width"]>=0.8 else (" | NARROW CPR: trend/whip risk, size DOWN or skip" if W["width"]<=0.4 else "")
    save(dict(week=str(ws),W={k:round(v,1) for k,v in W.items()},ref=round(ref),wpos=wpos,dpos=dpos,agree=agree,col=col,last=round(ref)))
    print(f"=== WEEKLY CPR PLAYBOOK — week {ws} @ {ist().strftime('%H:%M')} IST ===")
    print(f"NIFTY {ref:.0f}  gap {gap:+.2f}%  1st-30m {col} (open {o:.0f})")
    print(f"WEEKLY CPR {W['lo']:.0f}-{W['hi']:.0f} ({W['width']:.2f}%) -> price {wpos} | DAILY CPR {D['lo']:.0f}-{D['hi']:.0f} -> price {dpos}")
    print(f"Pivots: S3 {W['s3']:.0f} S2 {W['s2']:.0f} S1 {W['s1']:.0f} PP {W['p']:.0f} R1 {W['r1']:.0f} R2 {W['r2']:.0f} R3 {W['r3']:.0f}")
    print(f">>> STATE: {agree} + {col}\n>>> DEPLOY: {rec}{wsz}")

def monitor(belo,behi):
    s=load(); sp=live_spot(); con=sqlite3.connect(OPT)
    if not s.get("W"): print("NO STATE — run --assess first"); return
    if not sp: print("NO SPOT"); return
    W=s["W"]; last=s.get("last",s["ref"]); cur=sd(sp,W["lo"],W["hi"]); flags=[]; t30=last_30m_close(con)
    # --- graded DOWNSIDE trip-wire (for an above/AGREE-UP entry; the put-leg risk) ---
    if s["wpos"]=="ABOVE":
        if t30 is not None and t30<W["s1"]:
            flags.append(f"ACT: 30m close {t30:.0f} BELOW S1 {W['s1']:.0f} -> 88% continuation down (research/67). Defend/exit the put spread NOW.")
        elif t30 is not None and t30<W["lo"]:
            flags.append(f"ALERT: 30m close {t30:.0f} BELOW weekly CPR {W['lo']:.0f} -> ~55% the week flips down (base 20%). Go on alert + ready to defend the put leg; CONFIRM on a daily close or an S1 break before acting hard.")
        elif sp<W["lo"]:
            flags.append(f"WATCH: intraday poke {sp:.0f} below weekly CPR {W['lo']:.0f} (no 30m close below yet) -> S1 holds only 32% intraday; do NOT act on the wick.")
    elif s["wpos"]=="BELOW" and cur=="ABOVE":
        flags.append(f"CPR CROSS UP {sp:.0f} above band {W['hi']:.0f} -> bias may be flipping up (mirror of the down trip-wire)")
    # --- UPSIDE levels (call-side risk / far-OTM call selling) ---
    if sp>=W.get("r3",1e12): flags.append(f"AT/ABOVE R3 {W['r3']:.0f} -> far-OTM/trend territory (only ~11-29% of weeks reach here); call side under real pressure")
    elif sp>=W["r2"]: flags.append(f"AT/ABOVE R2 {W['r2']:.0f} (next R3 {W.get('r3',0):.0f})")
    # --- breakevens / spike / day-close ---
    if belo and sp<=belo*1.004: flags.append(f"BREAKEVEN THREAT (low): {sp:.0f} near put BE {belo:.0f}")
    if behi and sp>=behi*0.996: flags.append(f"BREAKEVEN THREAT (high): {sp:.0f} near call BE {behi:.0f}")
    mv=(sp-last)/last*100
    if abs(mv)>=0.4: flags.append(f"PRICE SPIKE {mv:+.2f}% ({sp-last:+.0f} pts) since last check")
    if ist().strftime("%H:%M")>="15:10": flags.append("DAY-CLOSE WINDOW (>=15:10) -> review PT/roll/exit; do NOT add size into expiry gamma")
    s["last"]=round(sp); save(s)
    if flags:
        print("ALERT "+ist().strftime("%H:%M"))
        for f in flags: print("  ! "+f)
        print(f"  ctx NIFTY {sp:.0f} | band {W['lo']:.0f}-{W['hi']:.0f} | S1 {W['s1']:.0f} R1 {W['r1']:.0f} R2 {W['r2']:.0f} R3 {W.get('r3',0):.0f} | {s['agree']}")
    else:
        print(f"CALM {ist().strftime('%H:%M')} | NIFTY {sp:.0f} {cur} band | {s['agree']} intact | S1 {W['s1']:.0f} / R1 {W['r1']:.0f} / R3 {W.get('r3',0):.0f}")

if __name__=="__main__":
    a=sys.argv
    if "--assess" in a: assess("--now" in a)
    elif "--monitor" in a:
        belo=float(a[a.index("--belo")+1]) if "--belo" in a else None
        behi=float(a[a.index("--behi")+1]) if "--behi" in a else None
        monitor(belo,behi)
    else: print("usage: --assess [--now] | --monitor [--belo X --behi Y]")
