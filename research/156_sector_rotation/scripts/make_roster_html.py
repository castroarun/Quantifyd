import json, io
from pathlib import Path

D = Path(__file__).parent
data = json.loads((D / "roster.json").read_text(encoding="utf-8"))

GROUPS = [
    ("Live · real money", ["Deployed blend TN40/OA40/IPO20", "Open Alpha", "True North"]),
    ("Paper · validating", ["IPO Base"]),
    ("Tested · not in the book", ["Gold (GOLDBEES)", "Multi-year breakout", "VCP breakout",
                                      "Sector-gated stocks", "Sector rotation (best)"]),
    ("Benchmarks", ["NIFTY 500", "Midcap 150"]),
]
COLORS = {
    "Deployed blend TN40/OA40/IPO20": "#0d6b5c",
    "Open Alpha": "#1f7a5a",
    "True North": "#4f9d86",
    "IPO Base": "#c68a1e",
    "Gold (GOLDBEES)": "#8a6d3b",
    "Multi-year breakout": "#5d6f9e",
    "VCP breakout": "#8b6ea8",
    "Sector-gated stocks": "#b3543f",
    "Sector rotation (best)": "#a83a2c",
    "NIFTY 500": "#8c948f",
    "Midcap 150": "#b0b7b2",
}
CHIP = {
    "LIVE (the book)": "live", "LIVE": "live", "PAPER": "paper",
    "CANDIDATE - not adopted": "cand", "SIGNAL - not adopted": "cand",
    "NO EDGE": "dead", "NO ADDED VALUE": "dead", "BENCHMARK": "bench",
}
NEW = {"Sector rotation (best)", "Sector-gated stocks"}

HTML = """<title>Nine Systems, One Book</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{
  --paper:#f1f3f2; --panel:#fbfcfb; --ink:#161b19; --muted:#69746f; --faint:#96a09b;
  --rule:#d7dcd9; --rule-soft:#e5e9e7; --accent:#0d6b5c; --accent-soft:#0d6b5c1a;
  --pos:#146b4a; --neg:#a83a2c;
  --live:#0d6b5c; --paper-chip:#a5761a; --cand:#5d6f9e; --dead:#9a4437; --bench:#7d867f;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  --paper:#101413; --panel:#171d1b; --ink:#e6ebe8; --muted:#96a19b; --faint:#6d7873;
  --rule:#28312e; --rule-soft:#1f2724; --accent:#46b8a3; --accent-soft:#46b8a31f;
  --pos:#4cbb8a; --neg:#e0705c;
  --live:#46b8a3; --paper-chip:#d8a63f; --cand:#8ea1d0; --dead:#e0705c; --bench:#8b948e;
}}
:root[data-theme="dark"]{
  --paper:#101413; --panel:#171d1b; --ink:#e6ebe8; --muted:#96a19b; --faint:#6d7873;
  --rule:#28312e; --rule-soft:#1f2724; --accent:#46b8a3; --accent-soft:#46b8a31f;
  --pos:#4cbb8a; --neg:#e0705c;
  --live:#46b8a3; --paper-chip:#d8a63f; --cand:#8ea1d0; --dead:#e0705c; --bench:#8b948e;
}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);
  font:400 16px/1.6 "IBM Plex Sans",system-ui,-apple-system,Segoe UI,sans-serif;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1180px;margin:0 auto;padding:56px 24px 96px;display:flex;flex-direction:column;gap:56px}
h1,h2,h3{margin:0;text-wrap:balance}
h1{font:600 clamp(34px,5.2vw,54px)/1.06 Spectral,Georgia,serif;letter-spacing:-.015em}
h2{font:600 24px/1.25 Spectral,Georgia,serif;letter-spacing:-.005em}
h3{font:600 13px/1.3 "IBM Plex Sans",sans-serif;text-transform:uppercase;letter-spacing:.11em;color:var(--muted)}
p{margin:0}
.lede{font:400 18.5px/1.62 Spectral,Georgia,serif;color:var(--ink);max-width:66ch}
.sub{color:var(--muted);font-size:14.5px;max-width:78ch}
.eyebrow{font:500 12px/1 "IBM Plex Mono",monospace;text-transform:uppercase;letter-spacing:.16em;color:var(--accent)}
header{display:flex;flex-direction:column;gap:16px;border-bottom:2px solid var(--ink);padding-bottom:28px}
section{display:flex;flex-direction:column;gap:18px}
.num{font-family:"IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums}
.pos{color:var(--pos)} .neg{color:var(--neg)}

/* ---- roster board ---- */
.board{display:flex;flex-direction:column;gap:26px}
.grp{display:flex;flex-direction:column;gap:9px}
.rows{display:flex;flex-direction:column}
.row{display:grid;grid-template-columns:14px minmax(190px,1.5fr) repeat(3,minmax(78px,.62fr)) minmax(150px,1fr) minmax(128px,.9fr);
  gap:14px;align-items:baseline;padding:11px 4px;border-bottom:1px solid var(--rule-soft)}
.row.head{border-bottom:1px solid var(--rule);padding-bottom:7px}
.row.head span{font:500 11px/1 "IBM Plex Sans",sans-serif;text-transform:uppercase;letter-spacing:.09em;color:var(--faint)}
.swatch{width:11px;height:11px;border-radius:2px;align-self:center}
.name{font-weight:500}
.name em{font-style:normal;color:var(--accent);font:500 10.5px/1 "IBM Plex Mono",monospace;
  letter-spacing:.08em;margin-left:7px;padding:2px 5px;border:1px solid var(--accent);border-radius:3px;vertical-align:1px}
.row .num{text-align:right}
.span{font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--faint);text-align:right}
.chip{justify-self:start;font:500 11px/1 "IBM Plex Sans",sans-serif;text-transform:uppercase;
  letter-spacing:.07em;padding:4px 8px;border-radius:3px;white-space:nowrap;
  color:var(--c);background:color-mix(in srgb,var(--c) 13%,transparent);border:1px solid color-mix(in srgb,var(--c) 34%,transparent)}
.chip.live{--c:var(--live)} .chip.paper{--c:var(--paper-chip)} .chip.cand{--c:var(--cand)}
.chip.dead{--c:var(--dead)} .chip.bench{--c:var(--bench)}

/* ---- chart ---- */
.chartbox{background:var(--panel);border:1px solid var(--rule);border-radius:5px;padding:20px 18px 14px}
.legend{display:flex;flex-wrap:wrap;gap:7px;margin-bottom:14px}
.lg{display:inline-flex;align-items:center;gap:7px;font-size:12.5px;padding:4px 9px 4px 7px;
  border:1px solid var(--rule);border-radius:99px;background:transparent;color:var(--ink);cursor:pointer;font-family:inherit}
.lg:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.lg .dot{width:9px;height:9px;border-radius:99px}
.lg[aria-pressed="false"]{opacity:.38}
svg{display:block;width:100%;height:auto}
.axis{font-family:"IBM Plex Mono",monospace;font-size:10.5px;fill:var(--faint)}
.gridline{stroke:var(--rule-soft);stroke-width:1}
.zero{stroke:var(--rule);stroke-width:1}

/* ---- yoy table ---- */
.scroll{overflow-x:auto;border:1px solid var(--rule);border-radius:5px;background:var(--panel)}
table{border-collapse:collapse;font-size:12.5px;min-width:1120px;width:100%}
th,td{padding:8px 9px;text-align:right;border-bottom:1px solid var(--rule-soft);white-space:nowrap}
thead th{position:sticky;top:0;background:var(--panel);z-index:1;font:500 10.5px/1.35 "IBM Plex Sans",sans-serif;
  text-transform:uppercase;letter-spacing:.06em;color:var(--muted);vertical-align:bottom;border-bottom:1px solid var(--rule)}
tbody th{text-align:left;font-family:"IBM Plex Mono",monospace;font-weight:500;color:var(--ink)}
td .r{display:block;font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}
td .d{display:block;font-family:"IBM Plex Mono",monospace;font-size:10.5px;color:var(--faint);margin-top:1px}
td.bo{font-size:11px;text-align:left;color:var(--muted);white-space:normal;min-width:112px}
tr.summary td,tr.summary th{border-top:2px solid var(--ink);border-bottom:none;padding-top:11px;font-size:11.5px}
tr.summary td .r{font-weight:500}
.newcol{background:var(--accent-soft)}

/* ---- notes ---- */
.notes{display:grid;grid-template-columns:repeat(auto-fit,minmax(268px,1fr));gap:20px}
.note{border-top:2px solid var(--rule);padding-top:13px;display:flex;flex-direction:column;gap:7px}
.note p{font-size:14px;color:var(--muted)}
.note strong{color:var(--ink);font-weight:600}
footer{border-top:1px solid var(--rule);padding-top:18px;color:var(--faint);font-size:12.5px;
  display:flex;flex-direction:column;gap:6px}
a{color:var(--accent)}
@media (max-width:760px){
  .row{grid-template-columns:12px 1fr repeat(3,minmax(60px,.6fr));row-gap:4px}
  .row .span,.row .chip{grid-column:2/-1;justify-self:start;text-align:left}
  .row.head span:nth-child(6),.row.head span:nth-child(7){display:none}
}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>

<div class="wrap">
<header>
  <span class="eyebrow">The book after research/156 &middot; 07-Sep-2026</span>
  <h1>Nine systems, one book</h1>
  <p class="lede">Every system we run, have run on paper, or have finished testing &mdash; side by
  side on one pair of axes, each from its own start date. Two of them are new today, and both are
  dead.</p>
  <p class="sub">__NOTE__</p>
</header>

<section>
  <h2>What is in the book, and what is not</h2>
  <p class="sub">Grouped by whose money is at risk, which is the question this page exists to
  answer. Windows differ &mdash; each system's own is stated &mdash; so read the CAGR column across
  a row, not down the page.</p>
  <div class="board">__BOARD__</div>
</section>

<section>
  <h2>Growth of &#8377;100, and the holes along the way</h2>
  <p class="sub">Month-end marks on a log scale, each series rebased to 100 at its own first month,
  with drawdown from the running peak beneath. Click a name to hide or show it.</p>
  <div class="chartbox">
    <div class="legend" id="legend"></div>
    <svg id="chart" viewBox="0 0 1000 560" role="img"
         aria-label="Growth of 100 rupees on a log scale for every system, with a drawdown panel beneath"></svg>
  </div>
</section>

<section>
  <h2>Year by year</h2>
  <p class="sub">Each cell is the calendar-year return with that year's deepest drawdown beneath it,
  measured from the running peak of the <em>full</em> curve rather than the year's first bar. The
  three columns on the right pick the best system of that year on return, on shallowest drawdown,
  and on the two together; benchmarks are excluded from the picks. Blank cells are years the system
  did not yet exist.</p>
  <div class="scroll">__TABLE__</div>
</section>

<section>
  <h2>What research/156 changes</h2>
  <div class="notes">
    <div class="note"><h3>Sector rotation</h3><p><strong>NO EDGE.</strong> Zero of 1,440
    configurations on the nine real NSE sector indices clear a 20&#8239;% CAGR bar after tax. The
    best returns 16.3&#8239;% at &minus;36.9&#8239;% &mdash; less than simply holding the Midcap 150
    index, which returned 18.0&#8239;% on the same window.</p></div>
    <div class="note"><h3>Sector-gated stocks</h3><p><strong>NO ADDED VALUE.</strong> Picking the
    leaders inside the leading sectors returns 32.5&#8239;%. Running the identical stock rule with
    <em>no sector filter at all</em> returns 32.9&#8239;% and wins 11 of 16 paired offsets. The
    return is stock momentum; the sector layer is a round trip to nowhere.</p></div>
    <div class="note"><h3>Neither one complements</h3><p>Correlation to the live legs runs
    0.41&ndash;0.54 against a 0.40 ceiling, the best blend improvement is +0.04 Calmar at slightly
    lower return, and a plain cash sleeve at the same weight beats every candidate. Sector-flavoured
    Indian equity is Indian equity.</p></div>
    <div class="note"><h3>A warning worth keeping</h3><p>Sector proxies built from today's index
    membership out-drift the real indices by <strong>+4 to +14 percentage points of CAGR a
    year</strong>, and shuffling the industry labels reproduces most of the apparent &ldquo;sector
    momentum&rdquo;. Any future sector work runs that null first.</p></div>
  </div>
</section>

<footer>
  <span>Full study: <a href="http://94.136.185.54:5000/app/backtest/sector-trend-rotation-research156">/app/backtest/sector-trend-rotation-research156</a>
  &middot; sources research/142, /144, /147, /151, /152, /153, /154 and /156.</span>
  <span>Research figures, not statements of account. Nothing on this page was deployed; no live or
  paper engine was touched.</span>
</footer>
</div>

<script>
const DATA = __DATA__;
const COLORS = __COLORS__;
const ORDER = __ORDER__;
const hidden = new Set(["VCP breakout"]);

const months = (()=>{const s=new Set();for(const k in DATA.curves)for(const m in DATA.curves[k])s.add(m);
  return [...s].sort();})();
const mi = new Map(months.map((m,i)=>[m,i]));

const W=1000,H=560,L=52,R=14,T=12,GAP=26,H1=352,H2=112;
function build(){
  const svg=document.getElementById('chart');
  const vis=ORDER.filter(k=>!hidden.has(k));
  let lo=Infinity,hi=-Infinity,ddlo=0;
  for(const k of vis){for(const m in DATA.curves[k]){const v=DATA.curves[k][m];if(v>0){lo=Math.min(lo,v);hi=Math.max(hi,v);}}
    for(const m in DATA.dd[k])ddlo=Math.min(ddlo,DATA.dd[k][m]);}
  if(!isFinite(lo)){lo=100;hi=1000;}
  const l0=Math.log10(lo*0.85),l1=Math.log10(hi*1.15);
  const x=i=>L+(W-L-R)*(i/(months.length-1));
  const y=v=>T+H1-(H1)*((Math.log10(v)-l0)/(l1-l0));
  const y2=v=>T+H1+GAP+ (H2)*(v/ (ddlo||-1));
  let s='';
  // y grid (powers of ten and halves)
  const ticks=[];
  for(let e=1;e<=7;e++){for(const m of [1,2,5]){const v=m*Math.pow(10,e);
    if(Math.log10(v)>=l0&&Math.log10(v)<=l1)ticks.push(v);}}
  for(const t of ticks){const yy=y(t).toFixed(1);
    s+=`<line class="gridline" x1="${L}" x2="${W-R}" y1="${yy}" y2="${yy}"/>`+
       `<text class="axis" x="${L-7}" y="${(+yy+3.4).toFixed(1)}" text-anchor="end">${t>=1000?(t/1000)+'k':t}</text>`;}
  for(const d of [0,-10,-20,-30,-40,-50,-60]){ if(d<ddlo-6) continue; const yy=y2(d).toFixed(1);
    s+=`<line class="${d===0?'zero':'gridline'}" x1="${L}" x2="${W-R}" y1="${yy}" y2="${yy}"/>`+
       `<text class="axis" x="${L-7}" y="${(+yy+3.4).toFixed(1)}" text-anchor="end">${d}%</text>`;}
  // x ticks
  for(const m of months){ if(!m.endsWith('-01')||(+m.slice(0,4))%3) continue;
    const xx=x(mi.get(m)).toFixed(1);
    s+=`<line class="gridline" x1="${xx}" x2="${xx}" y1="${T}" y2="${T+H1}"/>`+
       `<text class="axis" x="${xx}" y="${T+H1+GAP+H2+15}" text-anchor="middle">${m.slice(0,4)}</text>`;}
  s+=`<text class="axis" x="${L-7}" y="${T+9}" text-anchor="end">&#8377;</text>`;
  const emph=new Set(["Deployed blend TN40/OA40/IPO20","Open Alpha","True North"]);
  for(const k of vis){
    const c=COLORS[k], dash=(k==='NIFTY 500'||k==='Midcap 150')?' stroke-dasharray="4 3"':'';
    const w=emph.has(k)?2.3:1.45;
    let p='',q='',first=true;
    for(const m of months){const v=DATA.curves[k][m];if(v==null){continue;}
      p+=(first?'M':'L')+x(mi.get(m)).toFixed(1)+' '+y(v).toFixed(1)+' ';first=false;}
    first=true;
    for(const m of months){const v=DATA.dd[k][m];if(v==null){continue;}
      q+=(first?'M':'L')+x(mi.get(m)).toFixed(1)+' '+y2(v).toFixed(1)+' ';first=false;}
    s+=`<path d="${p}" fill="none" stroke="${c}" stroke-width="${w}"${dash} stroke-linejoin="round"/>`;
    s+=`<path d="${q}" fill="none" stroke="${c}" stroke-width="${(w*0.62).toFixed(2)}" opacity=".8"${dash}/>`;
  }
  s+=`<text class="axis" x="${L}" y="${T+H1+GAP-8}" text-anchor="start">DRAWDOWN FROM RUNNING PEAK</text>`;
  svg.innerHTML=s;
}
const lg=document.getElementById('legend');
for(const k of ORDER){
  const b=document.createElement('button');
  b.className='lg';b.type='button';b.setAttribute('aria-pressed',String(!hidden.has(k)));
  b.innerHTML=`<span class="dot" style="background:${COLORS[k]}"></span>${k}`;
  b.onclick=()=>{hidden.has(k)?hidden.delete(k):hidden.add(k);
    b.setAttribute('aria-pressed',String(!hidden.has(k)));build();};
  lg.appendChild(b);
}
build();
</script>
"""


def fmt(v):
    return f"{v:+.1f}" if v is not None else ""


def board_html():
    out = io.StringIO()
    for gname, keys in GROUPS:
        keys = [k for k in keys if k in data["summary"]]
        out.write('<div class="grp"><h3>' + gname + "</h3><div class=\"rows\">")
        out.write('<div class="row head"><span></span><span>System</span><span '
                  'style="text-align:right">CAGR</span><span style="text-align:right">Max DD</span>'
                  '<span style="text-align:right">Calmar</span><span style="text-align:right">'
                  'Window</span><span>Status</span></div>')
        for k in keys:
            s = data["summary"][k]
            sp = data["spans"][k]
            tag = ' <em>new</em>' if k in NEW else ""
            out.write(
                f'<div class="row"><span class="swatch" style="background:{COLORS[k]}"></span>'
                f'<span class="name">{k}{tag}</span>'
                f'<span class="num">{s["cagr"]:.2f}%</span>'
                f'<span class="num neg">{s["maxdd"]:.1f}%</span>'
                f'<span class="num">{s["calmar"]:.2f}</span>'
                f'<span class="span">{sp[0][:7]} &ndash; {sp[1][:7]}</span>'
                f'<span class="chip {CHIP[s["status"]]}">{s["status"]}</span></div>')
        out.write("</div></div>")
    return out.getvalue()


def table_html():
    cols = [k for _, ks in GROUPS for k in ks if k in data["summary"]]
    short = {"Deployed blend TN40/OA40/IPO20": "Deployed blend<br>TN40/OA40/IPO20",
             "Sector rotation (best)": "Sector<br>rotation",
             "Sector-gated stocks": "Sector-gated<br>stocks",
             "Multi-year breakout": "Multi-year<br>breakout",
             "Gold (GOLDBEES)": "Gold<br>(GOLDBEES)",
             "VCP breakout": "VCP<br>breakout"}
    o = io.StringIO()
    o.write("<table><thead><tr><th>Year</th>")
    for k in cols:
        cls = ' class="newcol"' if k in NEW else ""
        o.write(f"<th{cls}>{short.get(k, k)}</th>")
    o.write("<th>Best CAGR</th><th>Least DD</th><th>Best overall</th></tr></thead><tbody>")
    for row in data["yoy"]:
        o.write(f'<tr><th>{row["year"]}</th>')
        for k in cols:
            c = row["cells"].get(k)
            cls = ' class="newcol"' if k in NEW else ""
            if not c:
                o.write(f"<td{cls}></td>")
                continue
            r, d = c
            tone = "pos" if r >= 0 else "neg"
            o.write(f'<td{cls}><span class="r {tone}">{fmt(r)}</span>'
                    f'<span class="d">{d:.1f}%</span></td>')
        for key in ("best_cagr", "least_dd", "best_overall"):
            o.write(f'<td class="bo">{row.get(key, "")}</td>')
        o.write("</tr>")
    o.write('<tr class="summary"><th>Full period<br>CAGR / DD / Calmar</th>')
    for k in cols:
        s = data["summary"][k]
        cls = ' class="newcol"' if k in NEW else ""
        o.write(f'<td{cls}><span class="r">{s["cagr"]:.1f}%</span>'
                f'<span class="d">{s["maxdd"]:.1f}% &middot; {s["calmar"]:.2f}</span>'
                f'<span class="d">{data["spans"][k][0][:7]}&ndash;{data["spans"][k][1][:7]}</span></td>')
    o.write('<td class="bo"></td><td class="bo"></td><td class="bo"></td></tr>')
    o.write("</tbody></table>")
    return o.getvalue()


order = [k for _, ks in GROUPS for k in ks if k in data["summary"]]
slim = dict(curves=data["curves"], dd=data["dd"])
html = (HTML.replace("__NOTE__", data["note"])
            .replace("__BOARD__", board_html())
            .replace("__TABLE__", table_html())
            .replace("__DATA__", json.dumps(slim))
            .replace("__COLORS__", json.dumps(COLORS))
            .replace("__ORDER__", json.dumps(order)))
(D / "roster.html").write_text(html, encoding="utf-8")
print("roster.html written", len(html), "bytes")
