import io, contextlib, datetime as dt, statistics
from collections import defaultdict
from kiteconnect import KiteConnect
import config
_b=io.StringIO()
with contextlib.redirect_stdout(_b):
    exec(open("/tmp/cd_data.py").read())   # C = SL 2.0% (date,vix,pnl)
TR=[r for r in C if r[0] not in ("2020-03-13","2020-03-20")]   # 271 ex-COVID

tok=__import__("json").load(open("backtest_data/access_token.json"))
k=KiteConnect(api_key=config.KITE_API_KEY); k.set_access_token(tok["access_token"])
NIFTY=256265
bars={}
for a,b in [("2018-09-01","2022-06-30"),("2022-07-01","2026-06-08")]:
    for c in k.historical_data(NIFTY,a,b,"day"):
        d=c["date"].strftime("%Y-%m-%d"); bars[d]=(c["open"],c["high"],c["low"],c["close"])
dates=sorted(bars); idx={d:i for i,d in enumerate(dates)}; closes=[bars[d][3] for d in dates]

def feat(d):
    if d not in idx: return None
    i=idx[d]
    if i<200: return None
    o,h,l,c=bars[d]; spot=o
    sma20=sum(closes[i-20:i])/20; sma50=sum(closes[i-50:i])/50; sma200=sum(closes[i-200:i])/200
    ret20=closes[i-1]/closes[i-21]-1
    rets=[closes[j]/closes[j-1]-1 for j in range(i-20,i)]; rv=statistics.pstdev(rets)*(252**0.5)
    ph,pl,pc=bars[dates[i-1]][1],bars[dates[i-1]][2],bars[dates[i-1]][3]
    piv=(ph+pl+pc)/3; bc=(ph+pl)/2; tc=2*piv-bc; cprw=abs(tc-bc)/spot*100
    y,m=int(d[:4]),int(d[5:7]); pm=m-1 or 12; py=y if m>1 else y-1; pmt="%04d-%02d"%(py,pm)
    hs=[bars[x][1] for x in dates if x[:7]==pmt]; ls=[bars[x][2] for x in dates if x[:7]==pmt]
    pmpos=(spot-min(ls))/(max(hs)-min(ls)) if hs and max(hs)>min(ls) else None
    return dict(month=m,dow=dt.date.fromisoformat(d).weekday(),d20=(spot/sma20-1)*100,
                d50=(spot/sma50-1)*100,above200=spot>sma200,ret20=ret20*100,rv=rv*100,
                cprw=cprw,pmpos=pmpos)

rows=[]
for d,v,p in TR:
    f=feat(d)
    if f: rows.append((d,v,p,f))
print("trades with features:",len(rows),"of",len(TR))

def line(lbl,sub):
    if not sub: return
    tot=sum(p for p in sub); n=len(sub); avg=tot/n; win=100*sum(1 for p in sub if p>0)/n
    print("   {:<22} n={:<4} tot={:>10,.0f} avg={:>8,.0f} win={:>4.0f}%".format(lbl,n,tot,avg,win))

def cat(name,keyfn,order=None):
    print("\n## "+name)
    g=defaultdict(list)
    for d,v,p,f in rows:
        kk=keyfn(f,v)
        if kk is not None: g[kk].append(p)
    keys=order if order else sorted(g)
    for kk in keys:
        if kk in g: line(str(kk),g[kk])

def quart(name,valfn,vix13=False):
    print("\n## "+name+(" [VIX>=13 only]" if vix13 else ""))
    data=[(valfn(f,v),p) for d,v,p,f in rows if valfn(f,v) is not None and (v>=13 or not vix13)]
    data.sort()
    n=len(data);
    for qi in range(4):
        lo=qi*n//4; hi=(qi+1)*n//4
        seg=data[lo:hi]; vals=[x[0] for x in seg]; ps=[x[1] for x in seg]
        line("Q{} [{:.2f}..{:.2f}]".format(qi+1,vals[0],vals[-1]),ps)

MON=["","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
cat("Month of year (entry)",lambda f,v:MON[f["month"]],order=MON[1:])
cat("Day of week (entry)",lambda f,v:["Mon","Tue","Wed","Thu","Fri"][f["dow"]] if f["dow"]<5 else None,order=["Mon","Tue","Wed","Thu","Fri"])
cat("Spot vs 200DMA",lambda f,v:"above200" if f["above200"] else "below200")
quart("Distance from 20DMA (%)",lambda f,v:f["d20"])
quart("20-day momentum ret20 (%)",lambda f,v:f["ret20"])
quart("CPR width (% of spot)",lambda f,v:f["cprw"])
quart("Prior-month range position (0=low,1=high)",lambda f,v:f["pmpos"])
quart("Realized vol 20d (% ann)",lambda f,v:f["rv"])
quart("Entry VIX",lambda f,v:v)
# directional: abs momentum (trend strength) — the fly-killer hypothesis
quart("|20-day momentum| (trend strength)",lambda f,v:abs(f["ret20"]))
quart("|20-day momentum| (trend strength)",lambda f,v:abs(f["ret20"]),vix13=True)
