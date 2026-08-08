# -*- coding: utf-8 -*-
"""Visual: where puts were bought along the curve, what each cost/returned, and the regime periods."""
import json
d = json.load(open("/home/arun/quantifyd/research/105_momentum_put_hedge/results/hedge_viz.json"))
DATA = json.dumps(d)
HTML = r"""<title>Put-Hedge Overlay — where the puts were bought, and what they cost</title>
<style>
:root{--sans:'Segoe UI',system-ui,sans-serif;--bg:#0d1117;--panel:#161b22;--panel2:#1c232c;--bd:#293039;--bd2:#3a434e;--ink:#e6edf3;--mut:#9aa4af;--faint:#6b7581;--hedged:#e0a13a;--base:#58a6ff;--bh:#8a94a1;--neg:#e0564f;--pos:#2f9152;--gl:#222a33;--shade:rgba(224,86,79,.13);}
@media(prefers-color-scheme:light){:root{--bg:#f5f7fa;--panel:#fff;--panel2:#f0f3f7;--bd:#dde3ea;--bd2:#c6cfd8;--ink:#111820;--mut:#586170;--faint:#8a94a1;--hedged:#bd7a12;--base:#1f6feb;--gl:#eaeef3;--shade:rgba(224,86,79,.10);}}
:root[data-theme="dark"]{--bg:#0d1117;--panel:#161b22;--panel2:#1c232c;--bd:#293039;--bd2:#3a434e;--ink:#e6edf3;--mut:#9aa4af;--faint:#6b7581;--hedged:#e0a13a;--base:#58a6ff;--gl:#222a33;--shade:rgba(224,86,79,.13);}
:root[data-theme="light"]{--bg:#f5f7fa;--panel:#fff;--panel2:#f0f3f7;--bd:#dde3ea;--bd2:#c6cfd8;--ink:#111820;--mut:#586170;--faint:#8a94a1;--hedged:#bd7a12;--base:#1f6feb;--gl:#eaeef3;--shade:rgba(224,86,79,.10);}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.5}
.wrap{max-width:1060px;margin:0 auto;padding:26px 18px 60px}
h1{font-size:22px;margin:0 0 3px}.sub{color:var(--mut);font-size:13.5px;margin-bottom:8px}
.eyebrow{text-transform:uppercase;letter-spacing:.09em;font-size:11px;color:var(--faint);font-weight:700;margin:22px 0 9px}
.panel{background:var(--panel);border:1px solid var(--bd);border-radius:12px;padding:16px 18px}
.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}@media(max-width:820px){.kpis{grid-template-columns:1fr 1fr}}
.kpi{background:var(--panel);border:1px solid var(--bd);border-radius:11px;padding:13px 14px}
.kpi .big{font-size:19px;font-weight:750}.kpi .lab{font-size:11.5px;color:var(--mut);margin-top:3px}
svg{width:100%;height:auto;display:block}.gl{stroke:var(--gl);stroke-width:1}.axt{fill:var(--faint);font-size:10.5px}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--mut);margin:6px 0 10px}
.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:5px;vertical-align:-1px}
table{width:100%;border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:6px 9px;border-bottom:1px solid var(--bd);white-space:nowrap}th:first-child,td:first-child{text-align:left}
th{color:var(--faint);font-weight:600;font-size:10.5px;position:sticky;top:0;background:var(--panel)}
.pos{color:var(--pos)}.neg{color:var(--neg)}
.scroll{max-height:460px;overflow:auto;border:1px solid var(--bd);border-radius:8px}
.foot{color:var(--mut);font-size:12.5px;line-height:1.65}.foot b{color:var(--ink)}
.tip{position:fixed;pointer-events:none;background:var(--panel2);border:1px solid var(--bd2);border-radius:7px;padding:7px 10px;font-size:11.5px;opacity:0;transition:opacity .1s;z-index:9;max-width:260px}
.callout{background:var(--panel2);border-left:3px solid var(--neg);border-radius:0 8px 8px 0;padding:12px 15px;font-size:13px;margin:12px 0}
</style>
<div class="wrap">
  <h1>Put-Hedge Overlay — where the puts were bought, and what they cost</h1>
  <div class="sub">The tested arm: at a weekly risk-off gate, <b>hold the 8 stocks and buy ATM NIFTY weekly puts at 2× equity</b> instead of selling to cash. Re-sized daily, rolled at expiry, closed when the gate turns risk-on. 2019–2026 (weekly options don't exist earlier). <b>Red bands = gate risk-off (hedge active).</b></div>

  <div class="panel" style="padding:11px 14px;margin-bottom:12px">
    <label style="font-size:12.5px;color:var(--mut)">Period / tenor &nbsp;
      <select id="dsel" style="background:var(--panel2);color:inherit;border:1px solid var(--bd2);border-radius:6px;padding:6px 10px;font:inherit;font-size:13px"></select>
    </label>
    <span id="dnote" style="font-size:12px;color:var(--faint);margin-left:10px"></span>
  </div>

  <div class="kpis" id="kpis"></div>

  <div class="callout" id="callout"></div>

  <div class="eyebrow">Growth of ₹1 — with every put purchase marked, risk-off regimes shaded</div>
  <div class="panel">
    <div class="legend">
      <span><i style="background:var(--hedged)"></i>Hedged book (hold + puts)</span>
      <span><i style="background:var(--base)"></i>Cash-exit baseline (live rule)</span>
      <span><i style="background:var(--bh)"></i>NIFTY</span>
      <span><i style="background:var(--shade);border:1px solid var(--neg)"></i>Gate risk-off (hedge on)</span>
      <span><span style="color:var(--pos)">▲</span> profitable put &nbsp; <span style="color:var(--neg)">▼</span> losing put</span>
    </div>
    <svg id="eq" viewBox="0 0 1000 360"></svg>
  </div>

  <div class="panel" style="margin-top:14px"><b style="font-size:14px">Hedge value held (% of starting capital) — the premium at work</b>
    <svg id="hv" viewBox="0 0 1000 150" style="margin-top:8px"></svg></div>

  <div class="eyebrow">Cumulative premium spent vs cumulative hedge P&amp;L</div>
  <div class="panel"><svg id="cum" viewBox="0 0 1000 220"></svg>
    <div class="foot" style="margin-top:8px">The gap between the two lines is the net cost of carrying the hedge across the whole test.</div></div>

  <div class="eyebrow">Every put purchase (<span id="nEp"></span>)</div>
  <div class="panel"><div class="scroll"><table id="blot"></table></div>
    <div class="foot" style="margin-top:8px">Premium in/out are per-unit index points. Cost and P&amp;L are % of the starting capital. "Why" is what closed the position.</div></div>

  <div class="eyebrow">What this shows</div>
  <div class="panel foot" id="read"></div>
</div>
<div class="tip" id="tip"></div>
<script>
const D=""" + DATA + r""";
const $=s=>document.querySelector(s),tip=$("#tip");
const KEYS=Object.keys(D);
$("#dsel").innerHTML=KEYS.map(k=>`<option value="${k}">${D[k].label}</option>`).join("");
let CUR=KEYS[0];
let H,B,G,EP,BM,dates,T0,T1;
function bind(){const d=D[CUR];H=d.hedged;B=d.baseline;G=d.gate;EP=d.episodes;BM=d.bench;
  dates=H.map(r=>r[0]);T0=+new Date(dates[0]);T1=+new Date(dates[dates.length-1]);}
bind();
const xp=(ds,W,L,R)=>L+(+new Date(ds)-T0)/(T1-T0)*(W-L-R);
function renderAll(){
const totCost=EP.reduce((a,e)=>a+e.cost_pct,0), totPnl=EP.reduce((a,e)=>a+e.pnl_pct,0);
const wins=EP.filter(e=>e.pnl_pct>0).length;
const offDays=G.filter(g=>g[1]===1).length;
$("#dnote").textContent=`${dates[0]} to ${dates[dates.length-1]} · ${EP.length} put purchases · hedge active on ${offDays} of ${G.length} days (${Math.round(100*offDays/G.length)}%)`;
$("#kpis").innerHTML=[
 [EP.length,"put purchases"],
 [wins+" ("+Math.round(100*wins/EP.length)+"%)","were profitable"],
 [(totCost>=0?"":"")+totCost.toFixed(0)+"%","total premium spent"],
 [(totPnl>=0?"+":"")+totPnl.toFixed(0)+"%","NET hedge P&L"]
].map((k,i)=>`<div class="kpi"><div class="big" ${i===3?`style="color:var(--${totPnl>=0?'pos':'neg'})"`:""}>${k[0]}</div><div class="lab">${k[1]}</div></div>`).join("");
$("#callout").innerHTML=`<b>The puts themselves LOST money.</b> Across ${EP.length} purchases only ${wins} (${Math.round(100*wins/EP.length)}%) paid off; the hedge cost <b>${totCost.toFixed(0)}%</b> of starting capital in premium and returned <b style="color:var(--neg)">${totPnl.toFixed(0)}%</b> net. The hedged book still edged the baseline — but that came from <b>staying invested</b> through the risk-off periods, not from the hedge working. That is why it was not deployed.`;
// ── equity chart ──
function drawEq(){
 const W=1000,Hh=360,L=54,R=14,T=14,Bm=26;
 const hs=H.map(r=>[r[0],r[1]]),bs=B.map(r=>[r[0],r[1]]);
 const bm=BM.map(r=>[r[0],r[1]]);
 let lo=1e9,hi=-1e9;[hs,bs,bm].forEach(s=>s.forEach(p=>{lo=Math.min(lo,p[1]);hi=Math.max(hi,p[1]);}));
 lo=Math.max(0.5,lo);const l0=Math.log10(lo),l1=Math.log10(hi);
 const y=v=>T+(l1-Math.log10(Math.max(v,lo)))/Math.max(1e-9,l1-l0)*(Hh-T-Bm);
 let g="";
 // regime shading
 let st=null;
 G.forEach((r,i)=>{if(r[1]===1&&st===null)st=r[0];
   if((r[1]===0||i===G.length-1)&&st!==null){const x1=xp(st,W,L,R),x2=xp(r[0],W,L,R);
     g+=`<rect x="${x1.toFixed(1)}" y="${T}" width="${Math.max(1,x2-x1).toFixed(1)}" height="${Hh-T-Bm}" fill="var(--shade)"/>`;st=null;}});
 for(let e=Math.floor(l0*2)/2;e<=l1;e+=0.5){const v=Math.pow(10,e),yy=y(v);
   if(yy>T&&yy<Hh-Bm)g+=`<line class="gl" x1="${L}" x2="${W-R}" y1="${yy}" y2="${yy}"/><text class="axt" x="${L-6}" y="${yy+3}" text-anchor="end">${v.toFixed(v<10?1:0)}×</text>`;}
 [...new Set(dates.map(d=>d.slice(0,4)))].forEach(yr=>{const xx=xp(yr+"-01-01",W,L,R);
   if(xx>L&&xx<W-R)g+=`<text class="axt" x="${xx}" y="${Hh-8}" text-anchor="middle">${yr}</text>`;});
 const path=s=>s.map((p,i)=>(i?"L":"M")+xp(p[0],W,L,R).toFixed(1)+" "+y(p[1]).toFixed(1)).join(" ");
 g+=`<path d="${path(bm)}" style="fill:none;stroke:var(--bh);stroke-width:1.2;opacity:.65;stroke-dasharray:4 3"/>`;
 g+=`<path d="${path(bs)}" style="fill:none;stroke:var(--base);stroke-width:1.9"/>`;
 g+=`<path d="${path(hs)}" style="fill:none;stroke:var(--hedged);stroke-width:2.2"/>`;
 // put markers on the hedged curve
 const navAt={};hs.forEach(p=>navAt[p[0]]=p[1]);
 EP.forEach((e,i)=>{const x=xp(e.entry,W,L,R),v=navAt[e.entry];if(v===undefined)return;
   const yy=y(v),win=e.pnl_pct>0;
   g+=`<path class="mk" data-i="${i}" d="${win?`M${x} ${yy-9} l4.5 8 l-9 0 z`:`M${x} ${yy+9} l4.5 -8 l-9 0 z`}" fill="var(--${win?'pos':'neg'})" opacity=".9" style="cursor:pointer"/>`;});
 g+=`<rect id="hit" x="${L}" y="0" width="${W-L-R}" height="${Hh}" fill="transparent"/>`;
 $("#eq").innerHTML=g;
 document.querySelectorAll("#eq .mk").forEach(m=>{
   m.onmouseenter=ev=>{const e=EP[+m.dataset.i];
     tip.style.opacity=1;tip.style.left=(ev.clientX+12)+"px";tip.style.top=(ev.clientY-10)+"px";
     tip.innerHTML=`<b>${e.entry} → ${e.exit}</b><br>Bought ${e.strike} PE (spot ${e.spot})<br>
       Premium ${e.prem_in} → ${e.prem_out}<br>Cost ${e.cost_pct.toFixed(2)}% · P&L <b style="color:var(--${e.pnl_pct>=0?'pos':'neg'})">${e.pnl_pct>=0?"+":""}${e.pnl_pct.toFixed(2)}%</b><br>
       <span style="color:var(--mut)">${e.days}d · ${e.why}</span>`;};
   m.onmouseleave=()=>tip.style.opacity=0;});
}
function drawHV(){
 const W=1000,Hh=150,L=54,R=14,T=10,Bm=20;
 const mx=Math.max(...G.map(g=>g[2]),0.1);
 const y=v=>T+(mx-v)/mx*(Hh-T-Bm);
 let g="";
 for(let i=0;i<=2;i++){const v=mx*i/2,yy=y(v);
   g+=`<line class="gl" x1="${L}" x2="${W-R}" y1="${yy}" y2="${yy}"/><text class="axt" x="${L-6}" y="${yy+3}" text-anchor="end">${v.toFixed(1)}%</text>`;}
 g+=`<path d="${G.map((r,i)=>(i?"L":"M")+xp(r[0],W,L,R).toFixed(1)+" "+y(r[2]).toFixed(1)).join(" ")}" style="fill:none;stroke:var(--hedged);stroke-width:1.5"/>`;
 [...new Set(dates.map(d=>d.slice(0,4)))].forEach(yr=>{const xx=xp(yr+"-01-01",W,L,R);
   if(xx>L&&xx<W-R)g+=`<text class="axt" x="${xx}" y="${Hh-5}" text-anchor="middle">${yr}</text>`;});
 $("#hv").innerHTML=g;
}
function drawCum(){
 const W=1000,Hh=220,L=58,R=14,T=12,Bm=24;
 let c=0,p=0;const cc=[],pp=[];
 EP.forEach(e=>{c+=e.cost_pct;p+=e.pnl_pct;cc.push([e.exit,c]);pp.push([e.exit,p]);});
 const all=cc.concat(pp);let lo=Math.min(...all.map(a=>a[1]),0),hi=Math.max(...all.map(a=>a[1]),1);
 const y=v=>T+(hi-v)/(hi-lo)*(Hh-T-Bm);
 let g="";
 for(let i=0;i<=4;i++){const v=lo+(hi-lo)*i/4,yy=y(v);
   g+=`<line class="gl" x1="${L}" x2="${W-R}" y1="${yy}" y2="${yy}"/><text class="axt" x="${L-6}" y="${yy+3}" text-anchor="end">${v.toFixed(0)}%</text>`;}
 g+=`<line x1="${L}" x2="${W-R}" y1="${y(0)}" y2="${y(0)}" style="stroke:var(--bd2);stroke-dasharray:3 3"/>`;
 const pth=s=>s.map((q,i)=>(i?"L":"M")+xp(q[0],W,L,R).toFixed(1)+" "+y(q[1]).toFixed(1)).join(" ");
 g+=`<path d="${pth(cc)}" style="fill:none;stroke:var(--neg);stroke-width:2"/>`;
 g+=`<path d="${pth(pp)}" style="fill:none;stroke:var(--pos);stroke-width:2"/>`;
 g+=`<text class="axt" x="${W-R}" y="${y(cc[cc.length-1][1])-6}" text-anchor="end" style="fill:var(--neg)">premium spent ${c.toFixed(0)}%</text>`;
 g+=`<text class="axt" x="${W-R}" y="${y(pp[pp.length-1][1])+14}" text-anchor="end" style="fill:var(--pos)">net hedge P&L ${p.toFixed(0)}%</text>`;
 [...new Set(dates.map(d=>d.slice(0,4)))].forEach(yr=>{const xx=xp(yr+"-01-01",W,L,R);
   if(xx>L&&xx<W-R)g+=`<text class="axt" x="${xx}" y="${Hh-6}" text-anchor="middle">${yr}</text>`;});
 $("#cum").innerHTML=g;
}
function drawBlot(){
 $("#nEp").textContent=EP.length+" trades";
 let h=`<thead><tr><th>Entry → Exit</th><th>Strike</th><th>Spot</th><th>Prem in→out</th><th>Cost %</th><th>P&L %</th><th>Return</th><th>Days</th><th>Closed by</th></tr></thead><tbody>`;
 EP.forEach(e=>{h+=`<tr><td>${e.entry} → ${e.exit}</td><td>${e.strike} PE</td><td>${e.spot}</td>
   <td>${e.prem_in} → ${e.prem_out}</td><td>${e.cost_pct.toFixed(2)}</td>
   <td class="${e.pnl_pct>=0?'pos':'neg'}">${e.pnl_pct>=0?"+":""}${e.pnl_pct.toFixed(2)}</td>
   <td class="${e.ret>=0?'pos':'neg'}">${e.ret>=0?"+":""}${e.ret}%</td><td>${e.days}</td>
   <td style="color:var(--mut)">${e.why}</td></tr>`;});
 $("#blot").innerHTML=h+`</tbody>`;
}
$("#read").innerHTML=`<b>Read the red bands first.</b> They mark every period the NIFTYBEES 100-day gate was risk-off — the only times a put was held. There are ${offDays} such days out of ${G.length} (${Math.round(100*offDays/G.length)}% of the test), so the hedge is idle most of the time and the premium clock only runs inside the bands.<br><br>
<b>The triangles are the individual put purchases</b> — green pointing up where the put made money, red pointing down where it expired or was closed at a loss. Hover any of them for the strike, premium in/out and P&L. The visual pattern is stark: a dense run of small red losses punctuated by a handful of large green wins around genuine selloffs (notably early 2020).<br><br>
<b>The bottom chart is the verdict.</b> Cumulative premium spent climbs relentlessly while cumulative hedge P&L does not keep up — the puts finished <b style="color:var(--neg)">${totPnl.toFixed(0)}%</b> net across ${EP.length} purchases with only a ${Math.round(100*wins/EP.length)}% hit rate. The hedged book still edged the cash-exit baseline on risk-adjusted return, but that advantage came from <b>remaining invested in the 8 momentum stocks</b> through the risk-off windows — not from the hedge. Paying a losing insurance premium to justify staying invested is a far weaker thesis than it first appears, and it is the reason this was recorded as a SIGNAL and left out of the live book.`;
drawEq();drawHV();drawCum();drawBlot();
}
renderAll();
$("#dsel").onchange=e=>{CUR=e.target.value;bind();renderAll();};
</script>"""
open("/home/arun/quantifyd/frontend/public/hedge_viz_tearsheet.html", "w", encoding="utf-8").write(HTML)
print("wrote hedge_viz_tearsheet.html", len(HTML), "chars")
