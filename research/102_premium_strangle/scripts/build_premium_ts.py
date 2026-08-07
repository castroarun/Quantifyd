# -*- coding: utf-8 -*-
"""Premium-strangle study tearsheet with a 6%-liquid-yield toggle. 4 books (progression), all sections."""
import json
d = json.load(open("/home/arun/quantifyd/research/102_premium_strangle/results/premium_curves.json"))
DATA = json.dumps(d)
HTML = r"""<title>Premium-Based Weekly Strangle (10 lots)</title>
<style>
:root{--sans:'Segoe UI',system-ui,sans-serif;--bg:#0d1117;--panel:#161b22;--panel2:#1c232c;--bd:#293039;--bd2:#3a434e;--ink:#e6edf3;--mut:#9aa4af;--faint:#6b7581;
 --straddle:#58a6ff;--bnf3:#a78bfa;--bnf8:#2f9152;--sbi:#e0a13a;--neg:#e0564f;--gl:#222a33;}
@media(prefers-color-scheme:light){:root{--bg:#f5f7fa;--panel:#fff;--panel2:#f0f3f7;--bd:#dde3ea;--bd2:#c6cfd8;--ink:#111820;--mut:#586170;--faint:#8a94a1;--straddle:#1f6feb;--bnf3:#7c3aed;--bnf8:#2f9152;--sbi:#bd7a12;--gl:#eaeef3;}}
:root[data-theme="dark"]{--bg:#0d1117;--panel:#161b22;--panel2:#1c232c;--bd:#293039;--bd2:#3a434e;--ink:#e6edf3;--mut:#9aa4af;--faint:#6b7581;--straddle:#58a6ff;--bnf3:#a78bfa;--bnf8:#2f9152;--sbi:#e0a13a;--gl:#222a33;}
:root[data-theme="light"]{--bg:#f5f7fa;--panel:#fff;--panel2:#f0f3f7;--bd:#dde3ea;--bd2:#c6cfd8;--ink:#111820;--mut:#586170;--faint:#8a94a1;--straddle:#1f6feb;--bnf3:#7c3aed;--bnf8:#2f9152;--sbi:#bd7a12;--gl:#eaeef3;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.5}
.wrap{max-width:1040px;margin:0 auto;padding:26px 18px 60px}
h1{font-size:22px;margin:0 0 3px}.sub{color:var(--mut);font-size:13.5px;margin-bottom:8px}
.eyebrow{text-transform:uppercase;letter-spacing:.09em;font-size:11px;color:var(--faint);font-weight:700;margin:24px 0 9px;display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.panel{background:var(--panel);border:1px solid var(--bd);border-radius:12px;padding:16px 18px}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}@media(max-width:720px){.kpis{grid-template-columns:1fr 1fr}}
.kpi{background:var(--panel);border:1px solid var(--bd);border-radius:11px;padding:13px 14px;position:relative;overflow:hidden}
.kpi .rail{position:absolute;left:0;top:0;bottom:0;width:4px}.kpi h3{margin:0 0 7px;font-size:12.5px;display:flex;align-items:center;gap:6px;min-height:32px}
.kpi .dot{width:9px;height:9px;border-radius:50%;flex:none}.kpi .big{font-size:20px;font-weight:750}
.kpi .row{display:flex;justify-content:space-between;font-size:12px;padding:2px 0;border-top:1px solid var(--bd);margin-top:4px}
.kpi .row span{color:var(--mut)}.kpi .row b{font-variant-numeric:tabular-nums}.neg{color:var(--neg)}.pos{color:var(--bnf8)}
h2{font-size:15px;margin:0 0 3px}.note{color:var(--mut);font-size:12.5px;margin:0 0 10px}
.legend{display:flex;gap:15px;flex-wrap:wrap;font-size:12px;color:var(--mut);margin:8px 0}.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:5px;vertical-align:-1px}
svg{width:100%;height:auto;display:block}.gl{stroke:var(--gl);stroke-width:1}.axt{fill:var(--faint);font-size:10.5px}
table{width:100%;border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:6px 8px;border-bottom:1px solid var(--bd);white-space:nowrap}th:first-child,td:first-child{text-align:left}
th{color:var(--faint);font-weight:600;font-size:10.5px}
.seg{display:inline-flex;border:1px solid var(--bd);border-radius:7px;overflow:hidden;vertical-align:middle}
.seg button{background:transparent;border:0;color:var(--mut);font:600 10.5px var(--sans);padding:3px 9px;cursor:pointer}
.seg button.on{background:var(--panel2);color:var(--ink)}
.yseg button.on{background:var(--bnf8);color:#fff}
.scroll{max-height:460px;overflow:auto;border:1px solid var(--bd);border-radius:8px}
.scroll th{position:sticky;top:0;background:var(--panel);z-index:1}
.foot{color:var(--mut);font-size:12.5px;line-height:1.65}.foot b{color:var(--ink)}
.tip{position:fixed;pointer-events:none;background:var(--panel2);border:1px solid var(--bd2);border-radius:7px;padding:6px 9px;font-size:11.5px;opacity:0;transition:opacity .1s;z-index:9}
</style>
<div class="wrap">
  <h1>Premium-Based Weekly NIFTY Strangle</h1>
  <div class="sub">Strikes chosen by <b>target premium (~₹20/leg), not %OTM</b>. Enter ~4 trading days before weekly expiry, 10 lots (qty 750), 2019–2026, net of costs, on ₹20L capital. Book progression: raw → + VIX≥15 gate → + ATR&lt;1.2% gate → <b>final</b> (live-realistic <b>intraday</b> 3× combined-premium stop + re-deploy).</div>
  <div class="sub"><b>The edge:</b> sell rich implied vol (VIX≥15) only when realized vol is calm (ATR&lt;1.2%) — the IV&gt;RV vol-risk-premium. The ATR threshold is <b>walk-forward validated</b> (both halves independently pick 1.2%, holds out-of-sample). <b>Liquid-yield toggle:</b> the book is idle ~85% of weeks; parked ₹20L earns 6% p.a. (₹10k/mo).</div>

  <div class="eyebrow">The four books · headline <span class="seg yseg" id="yseg"><button data-y="0" class="on">Base</button><button data-y="1">+ 6% liquid yield</button></span></div>
  <div class="kpis" id="kpis"></div>

  <div class="eyebrow">Cumulative P&amp;L · 10 lots</div>
  <div class="panel"><div class="legend" id="leg1"></div><svg id="eq" viewBox="0 0 1000 320"></svg></div>
  <div class="panel" style="margin-top:14px"><h2>Drawdown (underwater) · 10 lots</h2>
    <p class="note" id="ddnote"></p><div class="legend" id="leg2"></div><svg id="dd" viewBox="0 0 1000 220"></svg></div>

  <div class="eyebrow">Year-by-year P&amp;L · 10 lots</div>
  <div class="panel" style="overflow-x:auto"><table id="ytab"></table></div>

  <div class="eyebrow">Monthly P&amp;L heatmap <span class="seg" id="hmseg"></span></div>
  <div class="panel" style="overflow-x:auto"><table id="hmap"></table>
    <div class="foot" style="margin-top:8px">Each cell = that month's P&amp;L in ₹ (k = thousands; hover for exact), 10 lots. With the yield toggle on, flat (no-trade) months show the ₹10k liquid interest instead of blank.</div></div>

  <div class="eyebrow">Month-on-month P&amp;L + running drawdown <span class="seg" id="mnseg"></span></div>
  <div class="panel"><div class="scroll"><table id="mntab"></table></div></div>

  <div class="eyebrow">Full trade blotter · final book, every leg</div>
  <div class="panel"><p class="note">The final book's 57 positions (VIX≥15 + ATR&lt;1.2% weeks). Each shaded row: month, entry→exit, entry VIX &amp; ATR, exit reason; indented legs = <b style="color:var(--bnf8)">NIFTY short CE + PE (~₹20 each)</b>. Group total is exact (net of costs + any re-deploys); leg P&amp;L is indicative — re-deployed trades re-struck mid-flight, so leg in→out can span different strikes.</p>
    <details><summary id="blsum"></summary><div class="scroll"><table id="bltab"></table></div></details></div>

  <div class="eyebrow">Takeaway</div>
  <div class="panel foot"><b>Sell rich IV into calm RV, hold through the move, and don't leave idle capital idle.</b> The VIX≥15 gate alone earns the most rupees (+₹14.2L) but with a −₹3.6L drawdown; adding the <b>ATR&lt;1.2%</b> gate trades far less (57 vs 204 weeks) for a much smoother ride (Calmar 1.0→2.4 with liquid, DD −₹1.05L). Honest live sizing uses an <b>intraday</b> 3× stop — that doubles the drawdown vs the daily-close backtest (−₹0.8L→−₹1.8L, Calmar +liq ~1.2), the number to trust. Four independent tests agree: <b>reacting to the move (ATR-breach exit, underlying-move stops, rolling the winning side in) all hurt</b> — hold, and if you must act, re-deploy premium rather than defend directionally.</div>
</div>
<div class="tip" id="tip"></div>
<script>
const D=""" + DATA + r""";
const $=s=>document.querySelector(s),tip=$("#tip");
const names=Object.keys(D.curves);
const COL={};[["Weekly ₹20 · all weeks","--straddle"],["+ VIX≥15","--bnf3"],["+ VIX≥15 + ATR<1.2%","--sbi"],["Final · intraday stop + redeploy","--bnf8"]].forEach(([n,v])=>COL[n]="var("+v+")");
const inr=v=>{const a=Math.abs(v);let s;if(a>=1e7)s="₹"+(v/1e7).toFixed(2)+" Cr";else if(a>=1e5)s="₹"+(v/1e5).toFixed(2)+" L";else s="₹"+Math.round(v).toLocaleString("en-IN");return s;};
const inrL=v=>(v/1e5).toFixed(0);
const Y0=+D.months[0].slice(0,4),Y1=+D.months[D.months.length-1].slice(0,4);
let YON=false, CUR={curves:D.curves,kpi:D.kpi}, RAW={};
function deriveRAW(){RAW={};names.forEach(n=>{const c=CUR.curves[n];RAW[n]={};c.forEach((p,i)=>RAW[n][p[0]]=i?p[1]-c[i-1][1]:p[1]);});}
const T0=+new Date(D.months[0]+"-01"),T1=+new Date(D.months[D.months.length-1]+"-01");
const xp=(m,W,L,R)=>L+(+new Date(m+"-01")-T0)/(T1-T0)*(W-L-R);
function renderKPI(){$("#kpis").innerHTML=names.map(n=>{const k=CUR.kpi[n],c=COL[n];return `<div class="kpi"><div class="rail" style="background:${c}"></div>
 <h3><span class="dot" style="background:${c}"></span>${n}</h3>
 <div class="big" style="color:var(--bnf8)">${k.cagr}%<span style="font-size:11px;color:var(--mut);font-weight:400"> CAGR</span></div>
 <div class="row"><span>Total (10 lots)</span><b>${inr(k.total)}</b></div>
 <div class="row"><span>Win rate</span><b>${k.win}%</b></div>
 <div class="row"><span>Calmar</span><b>${k.calmar}</b></div>
 <div class="row"><span>Max DD</span><b class="neg">${inr(k.mdd)}</b></div></div>`;}).join("");
 $("#leg1").innerHTML=$("#leg2").innerHTML=names.map(n=>`<span><i style="background:${COL[n]}"></i>${n}</span>`).join("");}
function drawEq(){const W=1000,H=320,L=64,R=14,T=12,B=24;let vmax=0,vmin=0;
 names.forEach(n=>CUR.curves[n].forEach(p=>{vmax=Math.max(vmax,p[1]);vmin=Math.min(vmin,p[1]);}));
 const y=v=>T+(vmax-v)/(vmax-vmin)*(H-T-B);let g="";
 for(let i=0;i<=5;i++){const v=vmin+(vmax-vmin)*i/5,yy=y(v);g+=`<line class="gl" x1="${L}" x2="${W-R}" y1="${yy}" y2="${yy}"/><text class="axt" x="${L-7}" y="${yy+3}" text-anchor="end">${inrL(v)}L</text>`;}
 for(let yr=Y0;yr<=Y1;yr++){const xx=xp(yr+"-01",W,L,R);if(xx>L&&xx<W-R)g+=`<text class="axt" x="${xx}" y="${H-6}" text-anchor="middle">${yr}</text>`;}
 g+=`<line x1="${L}" x2="${W-R}" y1="${y(0)}" y2="${y(0)}" style="stroke:var(--bd2);stroke-dasharray:3 3"/>`;
 names.forEach(n=>{const pth=CUR.curves[n].map((p,i)=>(i?"L":"M")+xp(p[0],W,L,R).toFixed(1)+" "+y(p[1]).toFixed(1)).join(" ");
   const bold=n.indexOf("Final")>=0;g+=`<path d="${pth}" style="fill:none;stroke:${COL[n]};stroke-width:${bold?2.6:1.8}"/>`;});
 g+=`<rect id="hit" x="${L}" y="0" width="${W-L-R}" height="${H}" fill="transparent"/>`;$("#eq").innerHTML=g;
 $("#hit").onmousemove=e=>{const r=$("#eq").getBoundingClientRect();const px=(e.clientX-r.left)/r.width*W;let bi=0,bd=1e15;
   D.months.forEach((m,i)=>{const dx=Math.abs(xp(m,W,L,R)-px);if(dx<bd){bd=dx;bi=i;}});
   tip.style.opacity=1;tip.style.left=(e.clientX+12)+"px";tip.style.top=(e.clientY-10)+"px";
   tip.innerHTML=`<b>${D.months[bi]}</b><br>`+names.map(n=>`<span style="color:${COL[n]}">■</span> ${inr(CUR.curves[n][bi][1])}`).join("<br>");};
 $("#hit").onmouseleave=()=>tip.style.opacity=0;}
function drawDD(){const W=1000,H=220,L=64,R=14,T=10,B=22;const DD={};let mn=-1;
 names.forEach(n=>{let pk=-1e18;DD[n]=CUR.curves[n].map(p=>{pk=Math.max(pk,p[1]);const v=p[1]-pk;mn=Math.min(mn,v);return[p[0],v];});});
 const y=v=>T+(0-v)/(0-mn)*(H-T-B);let g="";
 for(let i=0;i<=3;i++){const v=mn*i/3,yy=y(v);g+=`<line class="gl" x1="${L}" x2="${W-R}" y1="${yy}" y2="${yy}"/><text class="axt" x="${L-7}" y="${yy+3}" text-anchor="end">${inrL(v)}L</text>`;}
 for(let yr=Y0;yr<=Y1;yr++){const xx=xp(yr+"-01",W,L,R);if(xx>L&&xx<W-R)g+=`<text class="axt" x="${xx}" y="${H-5}" text-anchor="middle">${yr}</text>`;}
 names.forEach(n=>{const pth="M"+DD[n].map((p,i)=>(i?"L":"")+xp(p[0],W,L,R).toFixed(1)+" "+y(p[1]).toFixed(1)).join(" ");
   const bold=n.indexOf("Final")>=0||n.indexOf("all weeks")>=0;g+=`<path d="${pth}" style="fill:none;stroke:${COL[n]};stroke-width:${bold?2.2:1.4};opacity:${bold?1:.8}"/>`;});
 $("#dd").innerHTML=g;
 $("#ddnote").innerHTML=`Raw all-weeks (blue) digs to <b>${inr(CUR.kpi["Weekly ₹20 · all weeks"].mdd)}</b>; the final live-realistic book (green) holds <b>${inr(CUR.kpi["Final · intraday stop + redeploy"].mdd)}</b>.`;}
const short=n=>n.replace("Weekly ₹20 · all weeks","Raw").replace("+ VIX≥15 + ATR<1.2%","VIX+ATR").replace("+ VIX≥15","VIX").replace("Final · intraday stop + redeploy","Final");
function drawYear(){const YEARS=[...new Set(D.months.map(m=>m.slice(0,4)))];
 let h=`<thead><tr><th>Year</th>`+names.map(n=>`<th style="color:${COL[n]}">${short(n)}</th>`).join("")+`</tr></thead><tbody>`;
 YEARS.forEach(y=>{h+=`<tr><td><b>${y}</b></td>`+names.map(n=>{const v=D.months.filter(m=>m.startsWith(y)).reduce((a,m)=>a+(RAW[n][m]||0),0);return `<td class="${v>=0?'pos':'neg'}">${inr(v)}</td>`;}).join("")+`</tr>`;});
 h+=`<tr style="border-top:2px solid var(--bd2)"><td><b>TOTAL</b></td>`+names.map(n=>`<td class="pos"><b>${inr(CUR.kpi[n].total)}</b></td>`).join("")+`</tr></tbody>`;
 $("#ytab").innerHTML=h;}
let HMV=names[3];
function drawHM(){const YEARS=[...new Set(D.months.map(m=>m.slice(0,4)))];const MON=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
 const vals=Object.values(RAW[HMV]).filter(v=>v!=null);const mx=Math.max(...vals.map(Math.abs))||1;
 let h=`<thead><tr><th>Year</th>`+MON.map(m=>`<th>${m}</th>`).join("")+`<th>Total</th></tr></thead><tbody>`;
 YEARS.forEach(y=>{h+=`<tr><td><b>${y}</b></td>`;let yt=0;
   for(let mo=1;mo<=12;mo++){const key=y+"-"+String(mo).padStart(2,"0");const v=RAW[HMV][key];
     if(v==null){h+=`<td style="color:var(--faint)">·</td>`;}else{yt+=v;const t=Math.min(.85,Math.abs(v/mx));
       const col=v>=0?`rgba(63,176,99,${t})`:`rgba(224,86,79,${t})`;h+=`<td style="background:${col};font-size:10px" title="${inr(v)}">${(v>0?"+":"")+Math.round(v/1000)}k</td>`;}}
   h+=`<td class="${yt>=0?'pos':'neg'}" title="${inr(yt)}"><b>${(yt>0?"+":"")+Math.round(yt/1000)}k</b></td></tr>`;});
 $("#hmap").innerHTML=h+`</tbody>`;}
let MNV=names[3];
function drawMN(){const c=CUR.curves[MNV];let pk=-1e18;
 let h=`<thead><tr><th>Month</th><th>Monthly P&L</th><th>Cumulative</th><th>Peak</th><th>Drawdown</th></tr></thead><tbody>`;
 c.forEach(p=>{const r=RAW[MNV][p[0]];pk=Math.max(pk,p[1]);const dd=p[1]-pk;
   h+=`<tr><td>${p[0]}</td><td class="${r>=0?'pos':'neg'}">${inr(r)}</td><td><b>${inr(p[1])}</b></td><td style="color:var(--faint)">${inr(pk)}</td><td class="neg">${dd<-1?inr(dd):"—"}</td></tr>`;});
 $("#mntab").innerHTML=h+`</tbody>`;}
function drawBlot(){const T=D.trades;$("#blsum").textContent=`Show the ${T.length}-trade blotter (final book)`;const cl=v=>v>=0?"pos":"neg";let gt=0;
 const legrow=(lbl,K,ip,op)=>{const pl=Math.round((ip-op)*750);return `<tr><td style="padding-left:20px;color:var(--bnf8)">${lbl}</td><td>${K}</td><td>${ip} → ${op}</td><td class="${cl(pl)}">${inr(pl)}</td></tr>`;};
 let h=`<thead><tr><th>Position / leg</th><th>Strike</th><th>Premium in → out</th><th>P&L</th></tr></thead><tbody>`;
 T.forEach(t=>{gt+=t.total;
   h+=`<tr style="background:var(--panel2)"><td><b>${t.m}</b> &nbsp;<span style="color:var(--faint);font-size:11px">${t.entry}→${t.exit} · VIX ${t.vix} · ATR ${t.atr}% · ${t.reason}</span></td><td></td><td></td><td class="${cl(t.total)}"><b>${inr(t.total)}</b></td></tr>`;
   h+=legrow("NIFTY short CE · 10 lot",t.Kc,t.ceIn,t.ceOut);
   h+=legrow("NIFTY short PE · 10 lot",t.Kp,t.peIn,t.peOut);});
 h+=`</tbody><tfoot><tr style="border-top:2px solid var(--bd2)"><td><b>GROUP TOTAL · ${T.length} positions</b></td><td></td><td></td><td class="pos"><b>${inr(gt)}</b></td></tr></tfoot>`;
 $("#bltab").innerHTML=h;}
function renderAll(){deriveRAW();renderKPI();drawEq();drawDD();drawYear();drawHM();drawMN();}
$("#hmseg").innerHTML=names.map((n,i)=>`<button data-h="${i}"${i===3?' class="on"':''}>${short(n)}</button>`).join("");
$("#mnseg").innerHTML=names.map((n,i)=>`<button data-m="${i}"${i===3?' class="on"':''} style="color:${COL[n]}">${short(n)}</button>`).join("");
document.querySelectorAll("#hmseg button").forEach(b=>b.onclick=()=>{document.querySelectorAll("#hmseg button").forEach(x=>x.classList.remove("on"));b.classList.add("on");HMV=names[+b.dataset.h];drawHM();});
document.querySelectorAll("#mnseg button").forEach(b=>b.onclick=()=>{document.querySelectorAll("#mnseg button").forEach(x=>x.classList.remove("on"));b.classList.add("on");MNV=names[+b.dataset.m];drawMN();});
document.querySelectorAll("#yseg button").forEach(b=>b.onclick=()=>{document.querySelectorAll("#yseg button").forEach(x=>x.classList.remove("on"));b.classList.add("on");
  YON=b.dataset.y==="1";CUR=YON?{curves:D.curvesY,kpi:D.kpiY}:{curves:D.curves,kpi:D.kpi};renderAll();});
renderAll();drawBlot();
</script>"""
open("/home/arun/quantifyd/frontend/public/premium_strangle_tearsheet.html","w",encoding="utf-8").write(HTML)
print("wrote premium_strangle_tearsheet.html", len(HTML), "chars")
