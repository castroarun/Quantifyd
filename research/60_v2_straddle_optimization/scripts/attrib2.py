import io, contextlib, datetime as dt, statistics
from collections import defaultdict
from kiteconnect import KiteConnect
import config
_b=io.StringIO()
with contextlib.redirect_stdout(_b):
    exec(open("/tmp/cd_data.py").read())
TR=[r for r in C if r[0] not in ("2020-03-13","2020-03-20")]
tok=__import__("json").load(open("backtest_data/access_token.json"))
k=KiteConnect(api_key=config.KITE_API_KEY); k.set_access_token(tok["access_token"])
bars={}
for a,b in [("2018-09-01","2022-06-30"),("2022-07-01","2026-06-08")]:
    for c in k.historical_data(256265,a,b,"day"):
        d=c["date"].strftime("%Y-%m-%d"); bars[d]=(c["open"],c["high"],c["low"],c["close"])
dates=sorted(bars); idx={d:i for i,d in enumerate(dates)}; closes=[bars[d][3] for d in dates]
def feat(d):
    if d not in idx: return None
    i=idx[d]
    if i<200: return None
    spot=bars[d][0]
    rets=[closes[j]/closes[j-1]-1 for j in range(i-20,i)]; rv=statistics.pstdev(rets)*(252**0.5)*100
    ph,pl,pc=bars[dates[i-1]][1],bars[dates[i-1]][2],bars[dates[i-1]][3]
    piv=(ph+pl+pc)/3; bc=(ph+pl)/2; tc=2*piv-bc; cprw=abs(tc-bc)/spot*100
    return dict(cprw=cprw,rv=rv,month=int(d[5:7]))
MARGIN=958020.0; YEARS=7.3
def metr(rows):
    eq=0;peak=-1e18;mdd=0
    for _,_,p in rows: eq+=p;peak=max(peak,eq);mdd=min(mdd,eq-peak)
    tot=sum(p for _,_,p in rows); cal=(tot/YEARS)/abs(mdd) if mdd else 0
    yt=defaultdict(float)
    for d,_,p in rows: yt[d[:4]]+=p
    neg=[y for y in sorted(yt) if yt[y]<0]
    return tot,cal,mdd,len(rows),neg,yt
def show(name,rows):
    tot,cal,mdd,n,neg,yt=metr(rows)
    print("{:<34} n={:<4} tot={:>9,.0f} Calmar={:>4.2f} MaxDD={:>9,.0f} neg={}".format(name,n,tot,cal,mdd,neg))

V=[(d,v,p,feat(d)) for d,v,p in TR if v>=13 and feat(d)]
print("VIX>=13 with features:",len(V),"\n--- FILTER TESTS on the locked VIX>=13 book ---")
show("baseline (VIX>=13)",[(d,v,p) for d,v,p,f in V])
show("+ skip CPRwidth<0.10",[(d,v,p) for d,v,p,f in V if f["cprw"]>=0.10])
show("+ skip CPRwidth<0.12",[(d,v,p) for d,v,p,f in V if f["cprw"]>=0.12])
show("+ skip RV<9",[(d,v,p) for d,v,p,f in V if f["rv"]>=9])
show("+ skip Jan/Aug/Sep",[(d,v,p) for d,v,p,f in V if f["month"] not in (1,8,9)])
show("+ skip CPR<0.10 & Jan/Aug/Sep",[(d,v,p) for d,v,p,f in V if f["cprw"]>=0.10 and f["month"] not in (1,8,9)])
show("+ skip CPR<0.12 & RV<9",[(d,v,p) for d,v,p,f in V if f["cprw"]>=0.12 and f["rv"]>=9])

# per-year stability of the headline buckets (on full 258, ex-COVID, all VIX for sample size)
A=[(d,v,p,feat(d)) for d,v,p in TR if feat(d)]
def yearwise_bucket(name,pred):
    yt=defaultdict(float);yn=defaultdict(int)
    for d,v,p,f in A:
        if pred(f,v): yt[d[:4]]+=p;yn[d[:4]]+=1
    tot=sum(yt.values());n=sum(yn.values())
    pos=sum(1 for y in yt if yt[y]>0)
    print("\n{} (n={}, tot={:,.0f}, pos-years {}/{}):".format(name,n,tot,pos,len(yt)))
    print("   "+" ".join("{}:{:+,.0f}".format(y,yt[y]) for y in sorted(yt)))
print("\n--- PER-YEAR STABILITY of the bleeder buckets ---")
yearwise_bucket("CPR width bottom (cprw<0.10)",lambda f,v:f["cprw"]<0.10)
yearwise_bucket("Realized vol bottom (rv<9)",lambda f,v:f["rv"]<9)
yearwise_bucket("Months Jan/Aug/Sep",lambda f,v:f["month"] in (1,8,9))
yearwise_bucket("KEEP set: CPR>=0.10 (the survivors)",lambda f,v:f["cprw"]>=0.10)
