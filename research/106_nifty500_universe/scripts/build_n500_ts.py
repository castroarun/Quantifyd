# -*- coding: utf-8 -*-
"""Nifty 200 vs Nifty 500 universe — interactive tearsheet with a selectable date range."""
import json
d = json.load(open("/home/arun/quantifyd/research/106_nifty500_universe/results/n200_n500_curves.json"))
DATA = json.dumps(d)
HTML = r"""<title>Momentum Book — Nifty 200 vs Nifty 500 universe</title>
<style>
:root{--sans:'Segoe UI',system-ui,sans-serif;--bg:#0d1117;--panel:#161b22;--panel2:#1c232c;--bd:#293039;--bd2:#3a434e;--ink:#e6edf3;--mut:#9aa4af;--faint:#6b7581;--c0:#58a6ff;--c1:#e0a13a;--bh:#8a94a1;--neg:#e0564f;--pos:#2f9152;--gl:#222a33;}
@media(prefers-color-scheme:light){:root{--bg:#f5f7fa;--panel:#fff;--panel2:#f0f3f7;--bd:#dde3ea;--bd2:#c6cfd8;--ink:#111820;--mut:#586170;--faint:#8a94a1;--c0:#1f6feb;--c1:#bd7a12;--bh:#8a94a1;--gl:#eaeef3;}}
:root[data-theme="dark"]{--bg:#0d1117;--panel:#161b22;--panel2:#1c232c;--bd:#293039;--bd2:#3a434e;--ink:#e6edf3;--mut:#9aa4af;--faint:#6b7581;--c0:#58a6ff;--c1:#e0a13a;--bh:#8a94a1;--gl:#222a33;}
:root[data-theme="light"]{--bg:#f5f7fa;--panel:#fff;--panel2:#f0f3f7;--bd:#dde3ea;--bd2:#c6cfd8;--ink:#111820;--mut:#586170;--faint:#8a94a1;--c0:#1f6feb;--c1:#bd7a12;--bh:#8a94a1;--gl:#eaeef3;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.5}
.wrap{max-width:1040px;margin:0 auto;padding:26px 18px 60px}
h1{font-size:22px;margin:0 0 3px}.sub{color:var(--mut);font-size:13.5px;margin-bottom:10px}
.eyebrow{text-transform:uppercase;letter-spacing:.09em;font-size:11px;color:var(--faint);font-weight:700;margin:22px 0 9px}
.panel{background:var(--panel);border:1px solid var(--bd);border-radius:12px;padding:16px 18px}
.ctl{display:flex;gap:10px;align-items:center;flex-wrap:wrap;background:var(--panel);border:1px solid var(--bd);border-radius:10px;padding:12px 14px;margin-bottom:14px}
.ctl label{font-size:12px;color:var(--mut)}
.ctl input,.ctl select{background:var(--panel2);color:inherit;border:1px solid var(--bd2);border-radius:6px;padding:6px 9px;font:inherit;font-size:13px}
.ctl button{background:var(--panel2);color:inherit;border:1px solid var(--bd2);border-radius:6px;padding:6px 11px;font-size:12px;cursor:pointer}
.ctl button:hover{border-color:var(--c0)}
.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}@media(max-width:760px){.kpis{grid-template-columns:1fr}}
.kpi{background:var(--panel);border:1px solid var(--bd);border-radius:11px;padding:13px 14px;position:relative;overflow:hidden}
.kpi .rail{position:absolute;left:0;top:0;bottom:0;width:4px}.kpi h3{margin:0 0 7px;font-size:12.5px;display:flex;align-items:center;gap:6px}
.kpi .dot{width:9px;height:9px;border-radius:50%}.kpi .big{font-size:20px;font-weight:750}
.kpi .row{display:flex;justify-content:space-between;font-size:12px;padding:2px 0;border-top:1px solid var(--bd);margin-top:4px}
.kpi .row span{color:var(--mut)}.kpi .row b{font-variant-numeric:tabular-nums}
svg{width:100%;height:auto;display:block}.gl{stroke:var(--gl);stroke-width:1}.axt{fill:var(--faint);font-size:10.5px}
.legend{display:flex;gap:15px;flex-wrap:wrap;font-size:12px;color:var(--mut);margin:6px 0}.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:5px;vertical-align:-1px}
table{width:100%;border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:6px 9px;border-bottom:1px solid var(--bd);white-space:nowrap}th:first-child,td:first-child{text-align:left}
th{color:var(--faint);font-weight:600;font-size:10.5px}
.pos{color:var(--pos)}.neg{color:var(--neg)}
.foot{color:var(--mut);font-size:12.5px;line-height:1.6}.foot b{color:var(--ink)}
.tip{position:fixed;pointer-events:none;background:var(--panel2);border:1px solid var(--bd2);border-radius:7px;padding:6px 9px;font-size:11.5px;opacity:0;transition:opacity .1s;z-index:9}
</style>
<div class="wrap">
  <h1>Momentum Book — Nifty 200 vs Nifty 500 universe</h1>
  <div class="sub">Identical rules (top-8, top-22 buffer, 15-day Donchian stop, weekly NIFTYBEES&gt;100-SMA cash gate, monthly rotate-only). Only the <b>selection universe</b> differs: top-200 vs top-500 by traded value (survivorship-free PIT proxy). Daily-marked, net of 0.15%/leg, cash 6.5%. <b>Drag the date range to compare any period.</b></div>

  <div class="ctl">
    <label>From <input type="date" id="d0"></label>
    <label>To <input type="date" id="d1"></label>
    <label>P&amp;L basis
      <select id="basis"><option value="1">Pre-tax</option><option value="2">Net of STCG</option></select>
    </label>
    <button data-q="all">Full period</button>
    <button data-q="2011">2011–2015</button>
    <button data-q="2016">2016–2019</button>
    <button data-q="2020">2020–2026</button>
    <button data-q="5y">Last 5y</button>
  </div>

  <div class="kpis" id="kpis"></div>

  <div class="eyebrow">Growth of ₹1 (rebased to the selected start) — log scale</div>
  <div class="panel"><div class="legend" id="leg"></div><svg id="eq" viewBox="0 0 1000 330"></svg></div>

  <div class="panel" style="margin-top:14px"><b style="font-size:14px">Drawdown (within the selected range)</b>
    <svg id="dd" viewBox="0 0 1000 210" style="margin-top:8px"></svg></div>

  <div class="eyebrow">Year-by-year returns (%)</div>
  <div class="panel" style="overflow-x:auto"><table id="ytab"></table>
    <div class="foot" style="margin-top:8px">Calendar-year returns computed inside the selected range (partial years at the edges are partial-period returns).</div></div>

  <div class="eyebrow">Read</div>
  <div class="panel foot"><b>Widening the universe does not help.</b> Over the full 2011–2026 cycle the Nifty-500 book earns essentially the same headline return (31.7% vs 31.4% CAGR) but with a deeper drawdown (−18.4% vs −17.5%) and worse after-tax risk-adjustment (net Calmar <b>0.86 vs 0.99</b>) — and it degrades faster as costs rise (at 0.50%/leg: 0.47 vs 0.56). The extra 300 names add churn and small-cap volatility without adding edge. Use the date-range control to check regime dependence: the gap is not uniform through time.</div>
</div>
<div class="tip" id="tip"></div>
<script>
const D=""" + DATA + r""";
const $=s=>document.querySelector(s),tip=$("#tip");
const NAMES=Object.keys(D.series);
const COL={"Nifty 200":"var(--c0)","Nifty 500":"var(--c1)","NIFTY B&H":"var(--bh)"};
const ALL=D.series[NAMES[0]].map(r=>r[0]);
const MIN=ALL[0], MAX=ALL[ALL.length-1];
let basis=1;
$("#d0").min=MIN;$("#d0").max=MAX;$("#d1").min=MIN;$("#d1").max=MAX;
$("#d0").value=MIN;$("#d1").value=MAX;
const fmt=v=>(v>=0?"+":"")+v.toFixed(1)+"%";
function slice(name){
  const a=$("#d0").value,b=$("#d1").value;
  return D.series[name].filter(r=>r[0]>=a&&r[0]<=b).map(r=>[r[0],r[basis]]);
}
function stats(s){
  if(s.length<3)return null;
  const v0=s[0][1],last=s[s.length-1][1];
  const yrs=(new Date(s[s.length-1][0])-new Date(s[0][0]))/(365.25*864e5);
  const cagr=yrs>0?(Math.pow(last/v0,1/yrs)-1)*100:0;
  let pk=-1e18,mdd=0;const rets=[];
  for(let i=0;i<s.length;i++){pk=Math.max(pk,s[i][1]);mdd=Math.min(mdd,s[i][1]/pk-1);
    if(i)rets.push(s[i][1]/s[i-1][1]-1);}
  const m=rets.reduce((a,b)=>a+b,0)/rets.length;
  const sd=Math.sqrt(rets.reduce((a,b)=>a+(b-m)*(b-m),0)/rets.length);
  const sh=sd>0?m/sd*Math.sqrt(252):0;
  return {cagr,mdd:mdd*100,sh,mult:last/v0,cal:mdd<0?cagr/Math.abs(mdd*100):0};
}
function render(){
  basis=+$("#basis").value;
  const S={},K={};
  NAMES.forEach(n=>{S[n]=slice(n);K[n]=stats(S[n]);});
  $("#kpis").innerHTML=NAMES.map(n=>{const k=K[n];if(!k)return"";return `<div class="kpi"><div class="rail" style="background:${COL[n]}"></div>
    <h3><span class="dot" style="background:${COL[n]}"></span>${n}</h3>
    <div class="big" style="color:var(--pos)">${k.cagr.toFixed(1)}%<span style="font-size:11px;color:var(--mut);font-weight:400"> CAGR</span></div>
    <div class="row"><span>Max DD</span><b class="neg">${k.mdd.toFixed(1)}%</b></div>
    <div class="row"><span>Calmar</span><b>${k.cal.toFixed(2)}</b></div>
    <div class="row"><span>Sharpe</span><b>${k.sh.toFixed(2)}</b></div>
    <div class="row"><span>Growth</span><b>${k.mult.toFixed(2)}×</b></div></div>`;}).join("");
  $("#leg").innerHTML=NAMES.map(n=>`<span><i style="background:${COL[n]}"></i>${n}</span>`).join("");
  drawEq(S);drawDD(S);drawYear(S);
}
function xpos(dt,s,W,L,R){const t0=+new Date(s[0][0]),t1=+new Date(s[s.length-1][0]);
  return L+(+new Date(dt)-t0)/Math.max(1,t1-t0)*(W-L-R);}
function drawEq(S){
  const W=1000,H=330,L=54,R=14,T=12,B=24;const base=S[NAMES[0]];
  if(!base||base.length<3){$("#eq").innerHTML="";return;}
  let lo=1e9,hi=-1e9;const N={};
  NAMES.forEach(n=>{const s=S[n];if(!s.length)return;const v0=s[0][1];
    N[n]=s.map(r=>[r[0],r[1]/v0]);N[n].forEach(p=>{lo=Math.min(lo,p[1]);hi=Math.max(hi,p[1]);});});
  lo=Math.max(0.2,lo);const l0=Math.log10(lo),l1=Math.log10(hi);
  const y=v=>T+(l1-Math.log10(Math.max(v,lo)))/Math.max(1e-9,l1-l0)*(H-T-B);
  let g="";
  for(let e=Math.floor(l0*2)/2;e<=l1;e+=0.5){const v=Math.pow(10,e);const yy=y(v);
    if(yy>T&&yy<H-B)g+=`<line class="gl" x1="${L}" x2="${W-R}" y1="${yy}" y2="${yy}"/><text class="axt" x="${L-6}" y="${yy+3}" text-anchor="end">${v>=10?v.toFixed(0):v.toFixed(1)}×</text>`;}
  const yrs=[...new Set(base.map(r=>r[0].slice(0,4)))];
  yrs.forEach(yr=>{const xx=xpos(yr+"-01-01",base,W,L,R);
    if(xx>L&&xx<W-R)g+=`<text class="axt" x="${xx}" y="${H-6}" text-anchor="middle">${yr}</text>`;});
  NAMES.forEach(n=>{const s=N[n];if(!s)return;
    const p=s.map((r,i)=>(i?"L":"M")+xpos(r[0],base,W,L,R).toFixed(1)+" "+y(r[1]).toFixed(1)).join(" ");
    const bh=n==="NIFTY B&H";
    g+=`<path d="${p}" style="fill:none;stroke:${COL[n]};stroke-width:${bh?1.3:2};opacity:${bh?.75:1};${bh?"stroke-dasharray:4 3":""}"/>`;});
  g+=`<rect id="hit" x="${L}" y="0" width="${W-L-R}" height="${H}" fill="transparent"/>`;
  $("#eq").innerHTML=g;
  $("#hit").onmousemove=e=>{const r=$("#eq").getBoundingClientRect();const px=(e.clientX-r.left)/r.width*W;
    let bi=0,bd=1e15;base.forEach((p,i)=>{const dx=Math.abs(xpos(p[0],base,W,L,R)-px);if(dx<bd){bd=dx;bi=i;}});
    tip.style.opacity=1;tip.style.left=(e.clientX+12)+"px";tip.style.top=(e.clientY-10)+"px";
    tip.innerHTML=`<b>${base[bi][0]}</b><br>`+NAMES.map(n=>N[n]&&N[n][bi]?`<span style="color:${COL[n]}">■</span> ${N[n][bi][1].toFixed(2)}×`:"").join("<br>");};
  $("#hit").onmouseleave=()=>tip.style.opacity=0;
}
function drawDD(S){
  const W=1000,H=210,L=54,R=14,T=10,B=22;const base=S[NAMES[0]];
  if(!base||base.length<3){$("#dd").innerHTML="";return;}
  const DD={};let mn=-0.01;
  NAMES.forEach(n=>{const s=S[n];if(!s.length)return;let pk=-1e18;
    DD[n]=s.map(r=>{pk=Math.max(pk,r[1]);const v=r[1]/pk-1;mn=Math.min(mn,v);return[r[0],v];});});
  const y=v=>T+(0-v)/(0-mn)*(H-T-B);let g="";
  for(let i=0;i<=4;i++){const v=mn*i/4,yy=y(v);
    g+=`<line class="gl" x1="${L}" x2="${W-R}" y1="${yy}" y2="${yy}"/><text class="axt" x="${L-6}" y="${yy+3}" text-anchor="end">${(v*100).toFixed(0)}%</text>`;}
  const yrs=[...new Set(base.map(r=>r[0].slice(0,4)))];
  yrs.forEach(yr=>{const xx=xpos(yr+"-01-01",base,W,L,R);
    if(xx>L&&xx<W-R)g+=`<text class="axt" x="${xx}" y="${H-5}" text-anchor="middle">${yr}</text>`;});
  NAMES.forEach(n=>{const s=DD[n];if(!s)return;
    const p="M"+s.map((r,i)=>(i?"L":"")+xpos(r[0],base,W,L,R).toFixed(1)+" "+y(r[1]).toFixed(1)).join(" ");
    const bh=n==="NIFTY B&H";
    g+=`<path d="${p}" style="fill:none;stroke:${COL[n]};stroke-width:${bh?1.3:1.8};opacity:${bh?.7:.95};${bh?"stroke-dasharray:4 3":""}"/>`;});
  $("#dd").innerHTML=g;
}
function drawYear(S){
  const base=S[NAMES[0]];if(!base||!base.length){$("#ytab").innerHTML="";return;}
  const yrs=[...new Set(base.map(r=>r[0].slice(0,4)))];
  let h=`<thead><tr><th>Year</th>`+NAMES.map(n=>`<th style="color:${COL[n]}">${n}</th>`).join("")+`<th>200 − 500</th></tr></thead><tbody>`;
  yrs.forEach(y=>{
    const vals=NAMES.map(n=>{const s=S[n].filter(r=>r[0].slice(0,4)===y);
      return s.length>1?(s[s.length-1][1]/s[0][1]-1)*100:null;});
    h+=`<tr><td><b>${y}</b></td>`+vals.map(v=>v===null?`<td>·</td>`:`<td class="${v>=0?'pos':'neg'}">${fmt(v)}</td>`).join("");
    const df=(vals[0]!==null&&vals[1]!==null)?vals[0]-vals[1]:null;
    h+=df===null?`<td>·</td>`:`<td class="${df>=0?'pos':'neg'}">${fmt(df)}</td></tr>`;});
  $("#ytab").innerHTML=h+`</tbody>`;
}
document.querySelectorAll(".ctl button").forEach(b=>b.onclick=()=>{
  const q=b.dataset.q;
  if(q==="all"){$("#d0").value=MIN;$("#d1").value=MAX;}
  else if(q==="5y"){const d=new Date(MAX);d.setFullYear(d.getFullYear()-5);$("#d0").value=d.toISOString().slice(0,10);$("#d1").value=MAX;}
  else if(q==="2011"){$("#d0").value="2011-01-01";$("#d1").value="2015-12-31";}
  else if(q==="2016"){$("#d0").value="2016-01-01";$("#d1").value="2019-12-31";}
  else if(q==="2020"){$("#d0").value="2020-01-01";$("#d1").value=MAX;}
  render();});
["d0","d1","basis"].forEach(i=>$("#"+i).onchange=render);
render();
</script>"""
open("/home/arun/quantifyd/frontend/public/n200_vs_n500_tearsheet.html", "w", encoding="utf-8").write(HTML)
print("wrote n200_vs_n500_tearsheet.html", len(HTML), "chars")
