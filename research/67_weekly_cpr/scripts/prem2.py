"""combo_skip narrow-CPR premium check — FIXED: daily CPR built from the recorder's OWN intraday spot
(market_data.db NIFTY ends 2026-03, recorder is 04-20..06-16). Per day: prior-day CPR width, 09:20 ATM
straddle credit (nearest weekly), day's actual max move. Q: are narrow-CPR (skipped) days thin on premium?"""
import sqlite3, numpy as np, pandas as pd
oc=sqlite3.connect("backtest_data/options_data.db")
sp=pd.read_sql("SELECT snapshot_time,spot_price FROM underlying_spot WHERE symbol='NIFTY' AND spot_price>0 ORDER BY snapshot_time",oc,parse_dates=["snapshot_time"])
sp["d"]=sp["snapshot_time"].dt.strftime("%Y-%m-%d")
dd=sp.groupby("d")["spot_price"].agg(["first","max","min","last"]).rename(columns={"first":"o","max":"h","min":"l","last":"c"})
pH,pL,pC=dd["h"].shift(1),dd["l"].shift(1),dd["c"].shift(1)
dd["cprw"]=((2*pC-pH-pL).abs()/3)/pC*100
cpr={d:w for d,w in dd["cprw"].items()}

days=[r[0] for r in oc.execute("SELECT DISTINCT substr(snapshot_time,1,10) FROM option_chain WHERE symbol='NIFTY' ORDER BY 1")]
rows=[]
for day in days:
    r=oc.execute("SELECT spot_price FROM underlying_spot WHERE symbol='NIFTY' AND substr(snapshot_time,1,10)=? AND spot_price>0 AND snapshot_time<=? ORDER BY snapshot_time DESC LIMIT 1",(day,day+"T09:20:59")).fetchone()
    if not r: continue
    s0=r[0]; K=round(s0/50)*50
    exps=sorted({x[0] for x in oc.execute("SELECT DISTINCT expiry_date FROM option_chain WHERE symbol='NIFTY' AND substr(snapshot_time,1,10)=? AND expiry_date>=?",(day,day))})
    if not exps: continue
    E=exps[0]
    def lp(ot):
        q=oc.execute("SELECT ltp FROM option_chain WHERE expiry_date=? AND strike=? AND instrument_type=? AND substr(snapshot_time,1,10)=? AND snapshot_time<=? AND ltp>0 ORDER BY snapshot_time DESC LIMIT 1",(E,K,ot,day,day+"T09:20:59")).fetchone()
        return q[0] if q else None
    ce,pe=lp("CE"),lp("PE")
    if not(ce and pe): continue
    dh=oc.execute("SELECT MAX(spot_price),MIN(spot_price) FROM underlying_spot WHERE symbol='NIFTY' AND substr(snapshot_time,1,10)=?",(day,)).fetchone()
    move=max(dh[0]-s0,s0-dh[1])/s0*100 if dh and dh[0] else np.nan
    rows.append(dict(day=day,cprw=cpr.get(day,np.nan),spot=round(s0),K=K,credit=round(ce+pe,1),
                     credit_pct=round((ce+pe)/s0*100,2),dte=(pd.Timestamp(E)-pd.Timestamp(day)).days,maxmove=round(move,2)))
df=pd.DataFrame(rows).dropna(subset=["cprw"])
print(f"recorder days with prior-day CPR + 09:20 straddle: {len(df)}  ({df['day'].min()}..{df['day'].max()})")
narrow=df[df["cprw"]<0.10]; norm=df[df["cprw"]>=0.10]
print(f"\n{'group':24s} {'n':>3} {'CPRw avg':>8} {'credit pts':>10} {'credit %spot':>12} {'maxmove% avg/med':>17} {'calm<2%':>8}")
for nm,g in [("NARROW (<0.10, SKIPPED)",narrow),("NORMAL (>=0.10, TAKEN)",norm)]:
    if len(g):
        print(f"{nm:24s} {len(g):>3} {g['cprw'].mean():>8.3f} {g['credit'].mean():>10.1f} {g['credit_pct'].mean():>11.2f}% {g['maxmove'].mean():>7.2f}/{g['maxmove'].median():>8.2f} {(g['maxmove']<2).mean()*100:>7.0f}%")
print("\nEach skipped (narrow-CPR) day:")
print(f"  {'day':10s} {'CPRw%':>6} {'credit':>7} {'cred%':>6} {'dte':>3} {'maxmove%':>8}")
for _,r in narrow.sort_values("day").iterrows():
    print(f"  {r['day']:10s} {r['cprw']:>6.3f} {r['credit']:>7.1f} {r['credit_pct']:>5.2f}% {int(r['dte']):>3} {r['maxmove']:>8.2f}")
