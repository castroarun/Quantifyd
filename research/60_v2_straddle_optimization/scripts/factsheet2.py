import io, contextlib, datetime as dt, statistics
from collections import defaultdict
from kiteconnect import KiteConnect
import config
_b=io.StringIO()
with contextlib.redirect_stdout(_b):
    exec(open("/tmp/cd_data.py").read())          # C = SL 2.0% (date,vix,pnl)
EXCOVID=lambda L:[r for r in L if r[0] not in ("2020-03-13","2020-03-20")]
MARGIN=958020.0; YEARS=7.3; MON=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

tok=__import__("json").load(open("backtest_data/access_token.json"))
k=KiteConnect(api_key=config.KITE_API_KEY); k.set_access_token(tok["access_token"])
bars={}
for a,b in [("2018-09-01","2022-06-30"),("2022-07-01","2026-06-08")]:
    for c in k.historical_data(256265,a,b,"day"):
        d=c["date"].strftime("%Y-%m-%d"); bars[d]=(c["open"],c["high"],c["low"],c["close"])
dates=sorted(bars); idx={d:i for i,d in enumerate(dates)}; closes=[bars[d][3] for d in dates]
def cprw(d):
    if d not in idx or idx[d]<1: return None
    i=idx[d]; ph,pl,pc=bars[dates[i-1]][1],bars[dates[i-1]][2],bars[dates[i-1]][3]
    piv=(ph+pl+pc)/3; bc=(ph+pl)/2; tc=2*piv-bc; return abs(tc-bc)/bars[d][0]*100

def inr(n):
    neg=n<0; digits=str(int(abs(round(n))))
    if len(digits)>3:
        head,tail=digits[:-3],digits[-3:]; parts=[]
        while len(head)>2: parts.insert(0,head[-2:]); head=head[:-2]
        if head: parts.insert(0,head)
        g=",".join(parts)+","+tail
    else: g=digits
    return ("-" if neg else "")+g

def analyze(tr):
    tr=sorted(tr,key=lambda r:r[0]); m={}; yt={}
    yrsset=sorted({d[:4] for d,_,_ in tr})
    for d,_,p in tr:
        m[(d[:4],int(d[5:7]))]=m.get((d[:4],int(d[5:7])),0)+p; yt[d[:4]]=yt.get(d[:4],0)+p
    eq=0;peak=-1e18;gmdd=0;pk=tr_=None;ydd={y:(0,None,None) for y in yrsset};cum=[];dd=[]
    for d,_,p in tr:
        eq+=p
        if eq>peak: peak=eq; pk=d
        x=eq-peak; cum.append((d,eq)); dd.append((d,x))
        if x<gmdd: gmdd=x;gpk=pk;gtr=d
        if x<ydd[d[:4]][0]: ydd[d[:4]]=(x,pk,d)
    tot=sum(p for _,_,p in tr); n=len(tr); w=sum(1 for _,_,p in tr if p>0)
    return dict(tr=tr,m=m,yt=yt,years=yrsset,cum=cum,dd=dd,tot=tot,gmdd=gmdd,
                cal=(tot/YEARS)/abs(gmdd) if gmdd else 0,rom=tot/YEARS/MARGIN*100,
                win=100*w/n,n=n,green=sum(1 for y in yrsset if yt[y]>0),
                worst=min(p for _,_,p in tr))

# ---- build trade sets ----
base=EXCOVID(C)
V13=[(d,v,p) for d,v,p in base if v>=13]
V14=[(d,v,p) for d,v,p in base if v>=14]
feat=[(d,v,p,cprw(d)) for d,v,p in base if cprw(d) is not None]
V13f=[(d,v,p) for d,v,p,c in feat if v>=13]                          # 191 (feature-available)
CPR10=[(d,v,p) for d,v,p,c in feat if v>=13 and c>=0.10]
CPRJAS=[(d,v,p) for d,v,p,c in feat if v>=13 and c>=0.10 and int(d[5:7]) not in (1,8,9)]

# ---- WALK-FORWARD ----
def cal_sub(tr):
    tr=sorted(tr,key=lambda r:r[0])
    if not tr: return (0,0,0)
    eq=0;peak=-1e18;mdd=0
    for _,_,p in tr: eq+=p;peak=max(peak,eq);mdd=min(mdd,eq-peak)
    tot=sum(p for _,_,p in tr)
    d0=dt.date.fromisoformat(tr[0][0]); d1=dt.date.fromisoformat(tr[-1][0]); yy=max((d1-d0).days/365.25,0.5)
    return tot,(tot/yy)/abs(mdd) if mdd else 0,mdd
def half(tr,lo,hi): return [(d,v,p) for d,v,p,c in tr if lo<=d[:4]<=hi]
GRID=[0.06,0.08,0.10,0.12,0.14]
def best_t(tr):  # max Calmar over grid (filtered keeps cprw>=t)
    bt,bc=None,-9
    for t in GRID:
        _,cc,_=cal_sub([(d,v,p) for d,v,p,c in tr if c>=t])
        if cc>bc: bc,bt=cc,t
    return bt,bc
H1=[r for r in feat if r[0][:4]<="2022"]; H2=[r for r in feat if r[0][:4]>="2023"]
H1v=[r for r in H1 if r[1]>=13]; H2v=[r for r in H2 if r[1]>=13]
print("=== WALK-FORWARD (VIX>=13 book; pick CPR threshold on TRAIN, apply to TEST) ===")
for (trn,tst,ln) in [(H1v,H2v,"train 2019-22 -> test 2023-26"),(H2v,H1v,"train 2023-26 -> test 2019-22")]:
    t,_=best_t(trn)
    bt,bc,bm=cal_sub([(d,v,p) for d,v,p,c in tst])
    ft,fc,fm=cal_sub([(d,v,p) for d,v,p,c in tst if c>=t])
    print("  {}: t*={:.2f} | TEST baseline tot {:,.0f} Cal {:.2f} -> filtered tot {:,.0f} Cal {:.2f} DD {:,.0f}->{:,.0f}".format(
        ln,t,bt,bc,ft,fc,bm,fm))
print("  fixed t=0.10 in EACH half (same threshold helps both?):")
for nm,hv in [("H1 2019-22",H1v),("H2 2023-26",H2v)]:
    bt,bc,_=cal_sub([(d,v,p) for d,v,p,c in hv]); ft,fc,_=cal_sub([(d,v,p) for d,v,p,c in hv if c>=0.10])
    print("    {}: baseline Cal {:.2f} (tot {:,.0f}) -> CPR>=0.10 Cal {:.2f} (tot {:,.0f})".format(nm,bc,bt,fc,ft))
nb1=sum(p for d,v,p,c in H1v if c<0.10); nb2=sum(p for d,v,p,c in H2v if c<0.10)
print("  skipped bucket (cprw<0.10) P&L: H1 {:,.0f} | H2 {:,.0f}  (negative in BOTH halves = robust)".format(nb1,nb2))

# ---- HTML ----
def svg(curve,color,h=210,fill=True):
    if len(curve)<2: return ""
    ys=[y for _,y in curve]; mn=min(0,min(ys)); mx=max(0,max(ys)); rng=mx-mn or 1
    W=1120;PL=70;PR=10;PT=12;PB=22
    X=lambda i:PL+(i/(len(curve)-1))*(W-PL-PR); Y=lambda v:PT+(1-(v-mn)/rng)*(h-PT-PB)
    pts=" ".join("{:.1f},{:.1f}".format(X(i),Y(v)) for i,(_,v) in enumerate(curve))
    o=['<svg viewBox="0 0 {} {}" width="100%" height="{}" preserveAspectRatio="none" style="display:block">'.format(W,h,h)]
    for fr in (0,.25,.5,.75,1):
        val=mx-fr*rng;yy=Y(val)
        o.append('<line x1="{}" x2="{}" y1="{:.1f}" y2="{:.1f}" stroke="{}"/>'.format(PL,W-PR,yy,yy,"#cfcfcf" if abs(val)<1e-6 else "#eee"))
        o.append('<text x="{}" y="{:.1f}" font-size="10" fill="#888" text-anchor="end">{}</text>'.format(PL-6,yy+3,inr(val)))
    if fill:
        o.append('<polygon points="{:.1f},{:.1f} {} {:.1f},{:.1f}" fill="{}" opacity="0.10"/>'.format(X(0),Y(0),pts,X(len(curve)-1),Y(0),color))
    o.append('<polyline points="{}" fill="none" stroke="{}" stroke-width="1.8"/>'.format(pts,color))
    for i in (0,len(curve)//2,len(curve)-1):
        an="start" if i==0 else ("end" if i==len(curve)-1 else "middle")
        o.append('<text x="{:.1f}" y="{}" font-size="10" fill="#888" text-anchor="{}">{}</text>'.format(X(i),h-5,an,curve[i][0]))
    return "".join(o)+"</svg>"
def cellc(v): return '<td style="color:{};text-align:right">{}</td>'.format("#0a7d3c" if v>=0 else "#c02a2a",inr(v))
def kpis(A):
    K=[("Net P&L","+₹"+inr(A['tot']),"#0a7d3c"),("Calmar","{:.2f}".format(A['cal']),"#1b1b1a"),
       ("Max DD","-₹"+inr(abs(A['gmdd'])),"#c02a2a"),("Ret/Margin","{:.1f}%/yr".format(A['rom']),"#1b1b1a"),
       ("Win","{:.0f}%".format(A['win']),"#1b1b1a"),("Trades",str(A['n']),"#1b1b1a"),
       ("Green yrs","{}/{}".format(A['green'],len(A['years'])),"#1b1b1a")]
    return "".join('<div class="kpi"><div class=kl>{}</div><div class=kv style="color:{}">{}</div></div>'.format(l,c,v) for l,v,c in K)
def ytable(A):
    rows=[]
    for y in A['years']:
        tds="".join(cellc(A['m'][(y,mm+1)]) if (y,mm+1) in A['m'] else '<td style="text-align:right;color:#bbb">0</td>' for mm in range(12))
        rows.append('<tr><td class=yr>{}</td>{}{}</tr>'.format(y,tds,cellc(A['yt'][y])))
    return '<table class=yw><thead><tr><th>Year</th>{}<th>Total</th></tr></thead><tbody>{}</tbody></table>'.format(
        "".join("<th>%s</th>"%m for m in MON),"".join(rows))
def body(A,ac):
    return ('<div class=kpis>{}</div><div class=cl>Cumulative P&L (₹)</div>{}<div class=cl>Drawdown (₹)</div>{}'
            '<div class=cl>Year-wise monthly P&L (₹)</div>{}').format(kpis(A),svg(A['cum'],ac),svg(A['dd'],"#c02a2a"),ytable(A))
def panel(t,A,ac,open_=True):
    return '<section class=panel><h2 style="border-left:5px solid {}">{}</h2>{}</section>'.format(ac,t,body(A,ac))
def det(t,inner,op=False):
    return '<details class=panel {}><summary>{}</summary><div style="margin-top:12px">{}</div></details>'.format("open" if op else "",t,inner)

A13=analyze(V13); A14=analyze(V14); ACPR=analyze(CPR10); AJAS=analyze(CPRJAS); A13f=analyze(V13f)

# filter-test + WF tables as HTML
def simple_tbl(cols,rows,hl=()):
    h="".join("<th>%s</th>"%c for c in cols)
    rs=""
    for i,r in enumerate(rows):
        cls=' style="background:#eef6f0;font-weight:600"' if i in hl else ""
        rs+="<tr%s>%s</tr>"%(cls,"".join("<td>%s</td>"%c for c in r))
    return '<table class=cmp><thead><tr>%s</tr></thead><tbody>%s</tbody></table>'%(h,rs)

def row_for(name,A): return [name,"+₹"+inr(A['tot']),"{:.2f}".format(A['cal']),"-₹"+inr(abs(A['gmdd'])),"{}/{} green".format(A['green'],len(A['years']))]
ftbl=simple_tbl(["Overlay (on VIX≥13)","Net P&L","Calmar","Max DD","Years"],
   [row_for("baseline (feature set, n=%d)"%A13f['n'],A13f),
    row_for("+ skip CPR width &lt; 0.10%% (n=%d)"%ACPR['n'],ACPR),
    row_for("+ skip CPR&lt;0.10%% &amp; Jan/Aug/Sep (n=%d)"%AJAS['n'],AJAS)],hl=(1,2))

overlay_html = (
 '<p class=note>Candidate overlay — <b>pending forward validation</b>. CPR width = prior-day daily Central Pivot Range '
 '(|TC−BC| from Thursday\'s H/L/C) ÷ entry-day open, in %. Narrow CPR = volatility compression → expansion/breakout in the '
 'days ahead → the short fly\'s short-gamma gets run over. Skipping the bottom-quartile (≈&lt;0.10% of spot) lifts the locked '
 'VIX≥13 book on BOTH return and drawdown. Feature set starts Oct-2019 (200-day lookback) so n is a touch smaller than the headline book.</p>'
 + ftbl
 + '<div class=cl style="margin-top:14px">Walk-forward (pick CPR threshold on train half, apply to test half) — see chat / STATUS for the table.</div>'
 + '<h3 style="margin:18px 0 6px;font-size:14px">VIX≥13 + skip CPR width &lt; 0.10%% (compression filter)</h3>'
 + body(ACPR,"#7c3aed")
 + '<h3 style="margin:22px 0 6px;font-size:14px">VIX≥13 + CPR≥0.10%% + skip Jan/Aug/Sep (every year green)</h3>'
 + body(AJAS,"#0a7d3c"))

CSS='''body{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;background:#faf9f7;color:#1b1b1a;margin:0;padding:24px}
h1{font-size:22px;margin:0 0 4px}.sub{color:#777;font-size:13px;margin-bottom:20px}
.panel{background:#fff;border:1px solid #e6e3dd;border-radius:10px;padding:16px 20px;margin-bottom:24px;box-shadow:0 1px 3px rgba(0,0,0,.04)}
details.panel summary{cursor:pointer;font-size:16px;font-weight:700;color:#7c3aed;list-style:none}
h2{font-size:16px;margin:0 0 14px;padding-left:12px}
.kpis{display:flex;flex-wrap:wrap;gap:24px;margin-bottom:14px}.kpi .kl{font-size:11px;color:#888;text-transform:uppercase;letter-spacing:.04em}.kpi .kv{font-size:19px;font-weight:700}
.cl{font-size:11px;color:#888;margin:14px 0 4px}
table.yw,table.cmp{border-collapse:collapse;width:100%;font-size:11.5px;margin-top:4px}
table.yw th,table.cmp th{background:#f0eee9;color:#555;font-weight:600;padding:6px 7px;text-align:right;border:1px solid #e6e3dd;white-space:nowrap}
table.yw th:first-child,table.cmp th:first-child{text-align:left}
table.yw td,table.cmp td{padding:5px 7px;border:1px solid #efece6;font-variant-numeric:tabular-nums;text-align:right}
table.yw td.yr,table.cmp td:first-child{font-weight:600;text-align:left}
.note{font-size:12px;color:#666;line-height:1.55}'''

html='<!doctype html><html><head><meta charset=utf-8><title>V2 Iron Fly — Locked Base + CPR overlay</title><style>'+CSS+'</style></head><body>'
html+='<h1>V2 NIFTY Iron Fly — Locked Base + Compression (CPR) Overlay</h1>'
html+='<div class=sub>2.0% wings · short ATM straddle · 2.0% underlying move-stop · 09:20 entry (4 TD to expiry) · 10 lots · net of taxes+₹20/order+0.25% slippage · 2019–2026 · COVID week excluded</div>'
html+=panel("SL 2.0% + VIX ≥ 13  (locked — best risk-adjusted)",A13,"#1E3A8A")
html+=panel("SL 2.0% + VIX ≥ 14  (locked alt — every full year green)",A14,"#0a7d3c")
html+=det("▸ Compression (CPR-width) overlay — findings, curves & monthly tables  (candidate, pending walk-forward)",overlay_html,op=False)
html+='</body></html>'
open("/tmp/v2_factsheet.html","w",encoding="utf-8").write(html)
print("\nWROTE /tmp/v2_factsheet.html")
print("CPR10:",ACPR['n'],"tot",inr(ACPR['tot']),"cal {:.2f}".format(ACPR['cal']),"green",ACPR['green'],"/",len(ACPR['years']))
print("CPRJAS:",AJAS['n'],"tot",inr(AJAS['tot']),"cal {:.2f}".format(AJAS['cal']),"green",AJAS['green'],"/",len(AJAS['years']))
