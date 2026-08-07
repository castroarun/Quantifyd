# -*- coding: utf-8 -*-
"""Momentum-leverage frontier tearsheet — log-scale equity curves, drawdown, per-year, static tables."""
import json
d = json.load(open("/home/arun/quantifyd/research/104_momentum_leverage/results/lev_curves.json"))
DATA = json.dumps(d)
HTML = r"""<title>Momentum-250 Leverage Frontier (radj_z N8)</title>
<style>
:root{--sans:'Segoe UI',system-ui,sans-serif;--bg:#0d1117;--panel:#161b22;--panel2:#1c232c;--bd:#293039;--bd2:#3a434e;--ink:#e6edf3;--mut:#9aa4af;--faint:#6b7581;
 --c0:#58a6ff;--c1:#2f9152;--c2:#e0a13a;--c3:#c76fe0;--bh:#8a94a1;--neg:#e0564f;--gl:#222a33;}
@media(prefers-color-scheme:light){:root{--bg:#f5f7fa;--panel:#fff;--panel2:#f0f3f7;--bd:#dde3ea;--bd2:#c6cfd8;--ink:#111820;--mut:#586170;--faint:#8a94a1;--c0:#1f6feb;--c1:#2f9152;--c2:#bd7a12;--c3:#9333ea;--bh:#8a94a1;--gl:#eaeef3;}}
:root[data-theme="dark"]{--bg:#0d1117;--panel:#161b22;--panel2:#1c232c;--bd:#293039;--bd2:#3a434e;--ink:#e6edf3;--mut:#9aa4af;--faint:#6b7581;--c0:#58a6ff;--c1:#2f9152;--c2:#e0a13a;--c3:#c76fe0;--bh:#8a94a1;--gl:#222a33;}
:root[data-theme="light"]{--bg:#f5f7fa;--panel:#fff;--panel2:#f0f3f7;--bd:#dde3ea;--bd2:#c6cfd8;--ink:#111820;--mut:#586170;--faint:#8a94a1;--c0:#1f6feb;--c1:#2f9152;--c2:#bd7a12;--c3:#9333ea;--bh:#8a94a1;--gl:#eaeef3;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.5}
.wrap{max-width:1040px;margin:0 auto;padding:26px 18px 60px}
h1{font-size:22px;margin:0 0 3px}.sub{color:var(--mut);font-size:13.5px;margin-bottom:8px}
.eyebrow{text-transform:uppercase;letter-spacing:.09em;font-size:11px;color:var(--faint);font-weight:700;margin:24px 0 9px}
.panel{background:var(--panel);border:1px solid var(--bd);border-radius:12px;padding:16px 18px}
.kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}@media(max-width:820px){.kpis{grid-template-columns:1fr 1fr}}
.kpi{background:var(--panel);border:1px solid var(--bd);border-radius:11px;padding:12px 13px;position:relative;overflow:hidden}
.kpi .rail{position:absolute;left:0;top:0;bottom:0;width:4px}.kpi h3{margin:0 0 6px;font-size:11.5px;display:flex;align-items:center;gap:6px;min-height:28px}
.kpi .dot{width:9px;height:9px;border-radius:50%;flex:none}.kpi .big{font-size:18px;font-weight:750}
.kpi .row{display:flex;justify-content:space-between;font-size:11.5px;padding:2px 0;border-top:1px solid var(--bd);margin-top:4px}
.kpi .row span{color:var(--mut)}.neg{color:var(--neg)}.pos{color:var(--c1)}
h2{font-size:15px;margin:0 0 3px}.note{color:var(--mut);font-size:12.5px;margin:0 0 10px}
.legend{display:flex;gap:15px;flex-wrap:wrap;font-size:12px;color:var(--mut);margin:8px 0}.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:5px;vertical-align:-1px}
svg{width:100%;height:auto;display:block}.gl{stroke:var(--gl);stroke-width:1}.axt{fill:var(--faint);font-size:10.5px}
table{width:100%;border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:6px 8px;border-bottom:1px solid var(--bd);white-space:nowrap}th:first-child,td:first-child{text-align:left}
th{color:var(--faint);font-weight:600;font-size:10.5px}
tr.hl td{background:var(--panel2)}
.foot{color:var(--mut);font-size:12.5px;line-height:1.65}.foot b{color:var(--ink)}
.tip{position:fixed;pointer-events:none;background:var(--panel2);border:1px solid var(--bd2);border-radius:7px;padding:6px 9px;font-size:11.5px;opacity:0;transition:opacity .1s;z-index:9}
</style>
<div class="wrap">
  <h1>Momentum-250 — Leverage Frontier</h1>
  <div class="sub">Risk-adjusted momentum, <b>top-8 equal-weight</b>, monthly rebalance, <b>NIFTYBEES&gt;EMA100 index cash gate</b>. Daily-marked, net of 0.3% cost, survivorship-free top-250 proxy, 2006–2026. Leverage applied only while the gate is risk-on; <b>financing 10.5% p.a. (MTF)</b>; 0 margin calls in 20 years.</div>
  <div class="sub"><b>The idea:</b> the index gate moves the book to cash in downtrends, so leverage is only ever on during uptrends — that's what makes it survivable. Leverage buys raw CAGR at a proportional drawdown cost (Calmar drifts down); vol-targeting does <i>not</i> help (it de-levers into momentum's high-vol melt-ups).</div>

  <div class="eyebrow">Headline · 10 lots-equivalent, ₹ grows from 1× base</div>
  <div class="kpis" id="kpis"></div>

  <div class="eyebrow">Growth of ₹1 — log scale (2006–2026)</div>
  <div class="panel"><div class="legend" id="leg1"></div><svg id="eq" viewBox="0 0 1000 340"></svg></div>
  <div class="panel" style="margin-top:14px"><h2>Drawdown (underwater)</h2>
    <p class="note" id="ddnote"></p><div class="legend" id="leg2"></div><svg id="dd" viewBox="0 0 1000 220"></svg></div>

  <div class="eyebrow">Year-by-year returns (%)</div>
  <div class="panel" style="overflow-x:auto"><table id="ytab"></table></div>

  <div class="eyebrow">The leverage frontier — pick your point (radj_z N8, 10.5% MTF)</div>
  <div class="panel" style="overflow-x:auto"><table id="frontier"></table>
    <div class="foot" style="margin-top:8px">1.0× uses <b>your own capital only</b> (no borrowing). 1.3–1.6× is the return sweet spot. Above 1.6× the Calmar dips below 1 and the drawdown exceeds −55%.</div></div>

  <div class="eyebrow">Borrow-rate sensitivity — 8% (futures basis) vs 10.5% (MTF)</div>
  <div class="panel" style="overflow-x:auto"><table id="borrow"></table>
    <div class="foot" style="margin-top:8px">Financing costs little because leverage is only on ~76% of months (gate-on) and only on the borrowed slice — 10.5% vs 8% costs just 0.7–2.4% CAGR.</div></div>

  <div class="eyebrow">Why vol-targeting does NOT help</div>
  <div class="panel" style="overflow-x:auto"><table id="vt"></table>
    <div class="foot" style="margin-top:8px">Dynamic leverage from trailing vol lands <b>below</b> the static frontier at every matched average leverage. Two reasons: (1) the index gate already handles the downside, so vol-targeting duplicates it with lag + cost; (2) momentum's biggest up-years (2014 +108%, 2021 +101%) are <b>high-vol rallies</b> — vol-targeting de-levers straight into the melt-ups, amputating the fat right tail that is the edge.</div></div>

  <div class="eyebrow">Takeaway</div>
  <div class="panel foot"><b>The index gate makes leverage survivable; keep leverage simple and static.</b> On the gated top-8 risk-adjusted momentum book, static <b>1.3–1.6× leverage → 40.7–47.3% CAGR at −39 to −48% drawdown, Calmar ~1.0, zero margin calls across 2006–2026 (incl. 2008 &amp; 2020)</b>. At 1.0× (own capital, no borrowing) it's still <b>33.8% CAGR / −30% DD</b> vs NIFTYBEES buy-hold's 11.7% / −60%. Leverage is a proportional trade of drawdown for return — not a free lunch, but survivable <i>because</i> of the gate. Vol-targeting and deeper concentration (N&lt;8) both reduce risk-adjusted return. This is the same momentum edge magnified, not new alpha.</div>
</div>
<div class="tip" id="tip"></div>
<script>
const D=""" + DATA + r""";
const $=s=>document.querySelector(s),tip=$("#tip");
const names=Object.keys(D.curves);
const COL={"Own capital · 1.0×":"var(--c0)","Levered · 1.3×":"var(--c1)","Levered · 1.6×":"var(--c2)","Levered · 2.0×":"var(--c3)","NIFTY B&H":"var(--bh)"};
const mult=v=>v>=1000?(v/1000).toFixed(1)+"k×":v>=1?v.toFixed(0)+"×":v.toFixed(2)+"×";
const Y0=+D.months[0].slice(0,4),Y1=+D.months[D.months.length-1].slice(0,4);
const T0=+new Date(D.months[0]+"-01"),T1=+new Date(D.months[D.months.length-1]+"-01");
const xp=(m,W,L,R)=>L+(+new Date(m+"-01")-T0)/(T1-T0)*(W-L-R);
function renderKPI(){$("#kpis").innerHTML=names.map(n=>{const k=D.kpi[n],c=COL[n];return `<div class="kpi"><div class="rail" style="background:${c}"></div>
 <h3><span class="dot" style="background:${c}"></span>${n}</h3>
 <div class="big" style="color:var(--c1)">${k.cagr}%<span style="font-size:10px;color:var(--mut);font-weight:400"> CAGR</span></div>
 <div class="row"><span>Max DD</span><b class="neg">${k.dd}%</b></div>
 <div class="row"><span>Calmar</span><b>${k.calmar}</b></div>
 <div class="row"><span>Sharpe</span><b>${k.sharpe}</b></div>
 <div class="row"><span>Growth</span><b>${k.mult}×</b></div></div>`;}).join("");
 $("#leg1").innerHTML=$("#leg2").innerHTML=names.map(n=>`<span><i style="background:${COL[n]}"></i>${n}</span>`).join("");}
function drawEq(){const W=1000,H=340,L=52,R=14,T=12,B=24;let lo=0,hi=0;
 names.forEach(n=>D.curves[n].forEach(p=>{const l=Math.log10(Math.max(p[1],0.2));lo=Math.min(lo,l);hi=Math.max(hi,l);}));
 const y=v=>T+(hi-Math.log10(Math.max(v,0.2)))/(hi-lo)*(H-T-B);let g="";
 for(let e=Math.ceil(lo);e<=Math.floor(hi);e++){const yy=y(Math.pow(10,e));g+=`<line class="gl" x1="${L}" x2="${W-R}" y1="${yy}" y2="${yy}"/><text class="axt" x="${L-6}" y="${yy+3}" text-anchor="end">${e<3?Math.pow(10,e)+"×":Math.pow(10,e-3)+"k×"}</text>`;}
 for(let yr=Y0;yr<=Y1;yr++){const xx=xp(yr+"-01",W,L,R);if(xx>L&&xx<W-R)g+=`<text class="axt" x="${xx}" y="${H-6}" text-anchor="middle">${yr}</text>`;}
 names.forEach(n=>{const pth=D.curves[n].map((p,i)=>(i?"L":"M")+xp(p[0],W,L,R).toFixed(1)+" "+y(p[1]).toFixed(1)).join(" ");
   const bh=n==="NIFTY B&H";g+=`<path d="${pth}" style="fill:none;stroke:${COL[n]};stroke-width:${bh?1.4:2};opacity:${bh?.7:1};${bh?'stroke-dasharray:4 3':''}"/>`;});
 g+=`<rect id="hit" x="${L}" y="0" width="${W-L-R}" height="${H}" fill="transparent"/>`;$("#eq").innerHTML=g;
 $("#hit").onmousemove=e=>{const r=$("#eq").getBoundingClientRect();const px=(e.clientX-r.left)/r.width*W;let bi=0,bd=1e15;
   D.months.forEach((m,i)=>{const dx=Math.abs(xp(m,W,L,R)-px);if(dx<bd){bd=dx;bi=i;}});
   tip.style.opacity=1;tip.style.left=(e.clientX+12)+"px";tip.style.top=(e.clientY-10)+"px";
   tip.innerHTML=`<b>${D.months[bi]}</b><br>`+names.map(n=>{const c=D.curves[n][bi];return c?`<span style="color:${COL[n]}">■</span> ${mult(c[1])}`:"";}).join("<br>");};
 $("#hit").onmouseleave=()=>tip.style.opacity=0;}
function drawDD(){const W=1000,H=220,L=52,R=14,T=10,B=22;const DD={};let mn=-1;
 names.forEach(n=>{let pk=-1e18;DD[n]=D.curves[n].map(p=>{pk=Math.max(pk,p[1]);const v=p[1]/pk-1;mn=Math.min(mn,v);return[p[0],v];});});
 const y=v=>T+(0-v)/(0-mn)*(H-T-B);let g="";
 for(let i=0;i<=4;i++){const v=mn*i/4,yy=y(v);g+=`<line class="gl" x1="${L}" x2="${W-R}" y1="${yy}" y2="${yy}"/><text class="axt" x="${L-6}" y="${yy+3}" text-anchor="end">${(v*100).toFixed(0)}%</text>`;}
 for(let yr=Y0;yr<=Y1;yr++){const xx=xp(yr+"-01",W,L,R);if(xx>L&&xx<W-R)g+=`<text class="axt" x="${xx}" y="${H-5}" text-anchor="middle">${yr}</text>`;}
 names.forEach(n=>{const pth="M"+DD[n].map((p,i)=>(i?"L":"")+xp(p[0],W,L,R).toFixed(1)+" "+y(p[1]).toFixed(1)).join(" ");
   const bh=n==="NIFTY B&H";g+=`<path d="${pth}" style="fill:none;stroke:${COL[n]};stroke-width:${bh?1.4:1.8};opacity:${bh?.7:.9};${bh?'stroke-dasharray:4 3':''}"/>`;});
 $("#dd").innerHTML=g;
 $("#ddnote").innerHTML=`The gate keeps every leverage far shallower than NIFTYBEES' <b>−60%</b>. 1.0× bottoms at <b>−30%</b>; 1.6× at <b>−48%</b> — real and brutal, but never a margin call.`;}
function drawYear(){const yrs=[];for(let y=Y0;y<=Y1;y++)yrs.push(y);
 let h=`<thead><tr><th>Year</th>`+names.map(n=>`<th style="color:${COL[n]}">${n.replace("Own capital · ","").replace("Levered · ","").replace("NIFTY B&H","B&H")}</th>`).join("")+`</tr></thead><tbody>`;
 yrs.forEach(y=>{h+=`<tr><td><b>${y}</b></td>`+names.map(n=>{const v=D.peryear[n][y];return v==null?`<td class="mut">·</td>`:`<td class="${v>=0?'pos':'neg'}">${v>0?"+":""}${v}%</td>`;}).join("")+`</tr>`;});
 h+=`<tr class="hl"><td><b>CAGR</b></td>`+names.map(n=>`<td class="pos"><b>${D.kpi[n].cagr}%</b></td>`).join("")+`</tr></tbody>`;
 $("#ytab").innerHTML=h;}
function tbl(id,head,rows,hlrow){let h=`<thead><tr>`+head.map(x=>`<th>${x}</th>`).join("")+`</tr></thead><tbody>`;
 rows.forEach((r,i)=>{h+=`<tr class="${i===hlrow?'hl':''}">`+r.map((c,j)=>`<td${j===0?'':''}>${c}</td>`).join("")+`</tr>`;});
 $(id).innerHTML=h+`</tbody>`;}
renderKPI();drawEq();drawDD();drawYear();
tbl("#frontier",["Leverage","CAGR","Max DD","Sharpe","Calmar","Margin calls / 20y"],
 [["1.0× (own capital)","33.8%","−29.9%","1.49","1.13","0"],["1.3×","40.7%","−38.9%","1.40","1.05","0"],
  ["1.6×","47.3%","−47.9%","1.35","0.99","0"],["2.0×","55.4%","−59.7%","1.29","0.93","0"]],1);
tbl("#borrow",["Leverage","CAGR @8%","CAGR @10.5%","Max DD","Calmar @10.5%"],
 [["1.0×","33.8%","33.8%","−30%","1.13"],["1.3×","41.4%","40.7%","−39%","1.05"],
  ["1.6×","48.7%","47.3%","−48%","0.99"],["2.0×","57.8%","55.4%","−60%","0.93"]],2);
tbl("#vt",["Approach","Avg lev","CAGR","Max DD","Calmar"],
 [["Static leverage 1.3×","1.30","41.4%","−38.9%","1.06"],["Vol-target (vt25/lb60)","1.14","33.2%","−39.6%","0.84"],
  ["Static leverage 1.6×","1.60","48.7%","−47.8%","1.02"],["Vol-target (vt35/lb20)","1.60","43.2%","−52.3%","0.83"]],0);
</script>"""
open("/home/arun/quantifyd/frontend/public/momentum_leverage_tearsheet.html","w",encoding="utf-8").write(HTML)
print("wrote momentum_leverage_tearsheet.html", len(HTML), "chars")
