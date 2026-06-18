"""Intra-week BREACH / cross-back as an adjustment trigger. You entered ABOVE the weekly CPR; a later day
CLOSES BELOW the weekly CPR (cross-down) or BELOW S1 (deep breach) -> does the view change (does the week
then END below = confirmed turn, or recover = fakeout)? Vice versa for entries below (cross-up / R1).
NIFTY 5min->daily, 11y 2015-26. Causal within the week."""
import sqlite3, numpy as np, pandas as pd
c=sqlite3.connect("backtest_data/market_data.db")
df=pd.read_sql("SELECT date,high,low,close FROM market_data_unified WHERE symbol='NIFTY50' AND timeframe='5minute' ORDER BY date",c,parse_dates=["date"]).set_index("date")
df["d"]=df.index.normalize()
dd=df.groupby("d").agg(h=("high","max"),l=("low","min"),cl=("close","last"))
dd["wk"]=dd.index.to_period("W-FRI")
wkb=dd.groupby("wk").agg(H=("h","max"),L=("l","min"),C=("cl","last"))
pH,pL,pC=wkb["H"].shift(1),wkb["L"].shift(1),wkb["C"].shift(1)
P=(pH+pL+pC)/3;BC=(pH+pL)/2;TC=2*P-BC
wkb["wlo"]=np.minimum(BC,TC);wkb["whi"]=np.maximum(BC,TC)
wkb["R1"]=2*P-pL; wkb["S1"]=2*P-pH
wkb=wkb.dropna(subset=["wlo","S1"])
dd=dd.join(wkb[["wlo","whi","R1","S1","C"]].rename(columns={"C":"wkclose"}),on="wk")
dd["didx"]=dd.groupby("wk").cumcount()
D=dd.dropna(subset=["wlo"]).copy()

rows=[]
for wk,g in D.groupby("wk"):
    g=g.sort_values("didx")
    if len(g)<3: continue
    e=g.iloc[0]; lo,hi,S1,R1,wc=e["wlo"],e["whi"],e["S1"],e["R1"],e["wkclose"]
    entry="above" if e["cl"]>hi else ("below" if e["cl"]<lo else "inside")
    later=g.iloc[1:]
    rec={"entry":entry,"wkclose":wc,"lo":lo,"hi":hi}
    # cross to the other side of the band, and pivot breach, on a LATER day close
    cd=later[later["cl"]<lo]; cu=later[later["cl"]>hi]
    bs1=later[later["cl"]<S1]; br1=later[later["cl"]>R1]
    rec["crossdn"]=(None if cd.empty else cd.iloc[0]["cl"])
    rec["crossup"]=(None if cu.empty else cu.iloc[0]["cl"])
    rec["bS1"]=(None if bs1.empty else bs1.iloc[0]["cl"])
    rec["bR1"]=(None if br1.empty else br1.iloc[0]["cl"])
    rows.append(rec)
T=pd.DataFrame(rows)

def report(side, crosscol, breachcol, end_test, lab_cross, lab_breach):
    base=T[T["entry"]==side]
    print(f"\n===== entered {side.upper()} the weekly CPR (n={len(base)}) — baseline: week ends {end_test} {ended(base,end_test)*100:.0f}% =====")
    for col,lab in [(crosscol,lab_cross),(breachcol,lab_breach)]:
        ev=base[base[col].notna()]
        if len(ev)<6: print(f"  {lab}: n={len(ev)} (thin)"); continue
        # after the event: rest move from event close to week close; did week end on the NEW side?
        rest=(ev["wkclose"]-ev[col])/ev[col]*100
        endnew=ended(ev,end_test)
        print(f"  {lab}: occurs {len(ev)/len(base)*100:>3.0f}% of weeks (n={len(ev)}) -> week ENDS {end_test} {endnew*100:>3.0f}% | rest-move after event {rest.mean():>+5.2f}% (med {rest.median():+.2f})")

def ended(d,test):
    if test=="below band": return (d["wkclose"]<d["lo"]).mean()
    if test=="above band": return (d["wkclose"]>d["hi"]).mean()

# for 'above' the 'change of view' test is whether it ends BELOW the band
T2=T.copy()
def report2(side,col,lab,endtest):
    base=T2[T2["entry"]==side]; ev=base[base[col].notna()]
    if len(ev)<6: print(f"  {lab}: thin"); return
    rest=(ev["wkclose"]-ev[col])/ev[col]*100
    e=(ev["wkclose"]<ev["lo"]).mean() if endtest=="below" else (ev["wkclose"]>ev["hi"]).mean()
    print(f"  {lab}: occurs {len(ev)/len(base)*100:.0f}% (n={len(ev)}) -> ends {endtest} band {e*100:.0f}% | rest-move {rest.mean():+.2f}%")
print("\n===== ABOVE entries — does a downside breach flip the view? (end BELOW = view changed) =====")
report2("above","crossdn","cross BELOW weekly CPR","below")
report2("above","bS1","breach BELOW S1","below")
print("\n===== BELOW entries — does an upside breach flip the view? (end ABOVE = view changed) =====")
report2("below","crossup","cross ABOVE weekly CPR","above")
report2("below","bR1","breach ABOVE R1","above")
print("\nbaselines: ABOVE entries end below band %=", round((T[T.entry=='above']['wkclose']<T[T.entry=='above']['lo']).mean()*100),
      "| BELOW entries end above band %=", round((T[T.entry=='below']['wkclose']>T[T.entry=='below']['hi']).mean()*100))
