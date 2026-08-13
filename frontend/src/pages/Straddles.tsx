import { useEffect, useMemo, useState } from 'react';

/* ---------- types ---------- */
interface V1 {
  version: string; trigger_pct: number; dte_max: number | null; lots: number;
  per_day: Record<string, { series: [string, number][]; final: number; stopped: boolean; dte: number; expiry: string }>;
  cum_curve: [string, number][];
}
interface V2Trade {
  entry_day: string; exit_day: string; strike: number; expiry: string;
  exit_reason: string; pnl: number; wing_pnl: number; series: [string, number][];
}
interface V2 { version: string; move_stop: number; pt: number; wings: number; lots: number; trades: V2Trade[]; book_curve: [string, number][]; }
interface DailyDay {
  series: [string, number][]; exit: { time: string; pnl: number } | null;
  final: number; stopped: boolean; low: number; high: number;
  dte: number; expiry: string; strike: number; credit: number;
}
interface V1Daily { version: string; trigger_pct: number; lots: number; lot: number; days: string[]; per_day: Record<string, DailyDay>; }
interface Leg { type: 'CE' | 'PE'; strike: number; qty: number; side: string; entry: number | null; ltp: number | null; pnl: number; entry_time?: string | null; exit_time?: string | null; max_ltp?: number | null; }

/* ---------- light theme tokens ---------- */
const C = { ink: '#1B1B1A', muted: '#888780', faint: '#B4B2A9', sec: '#5F5E5A', hair: 'rgba(0,0,0,0.10)',
  hairSoft: 'rgba(0,0,0,0.06)', pos: '#0F6E56', neg: '#A32D2D', navy: '#1E3A8A', navySoft: '#EFF3FA',
  amber: '#B45309', amberSoft: '#FEF3C7', surface: '#FFFFFF', canvas: '#FAFAF9' };

const inr = (n: number) => `${n >= 0 ? '+' : '−'}₹${Math.abs(Math.round(n)).toLocaleString('en-IN')}`;
const col = (n: number) => (n >= 0 ? C.pos : C.neg);

const fmtY = (v: number) => `${v >= 0 ? '+' : '−'}₹${Math.abs(Math.round(v)).toLocaleString('en-IN')}`;
function LineChart({ pts, h = 130, label, marker }: { pts: [string, number][]; h?: number; label?: string; marker?: { time: string; pnl: number; text?: string } | null }) {
  if (!pts || pts.length < 2) return <div style={{ color: C.faint, fontSize: 12, padding: 8 }}>—</div>;
  const W = 600, PAD_L = 56, PAD_R = 10, PAD_T = 8, PAD_B = 18;
  const ys = pts.map((p) => p[1]);
  const min = Math.min(0, ...ys), max = Math.max(0, ...ys), rng = max - min || 1;
  const X = (i: number) => PAD_L + (i / (pts.length - 1)) * (W - PAD_L - PAD_R);
  const Y = (v: number) => PAD_T + (1 - (v - min) / rng) * (h - PAD_T - PAD_B);
  const line = pts.map((p, i) => `${X(i)},${Y(p[1])}`).join(' ');
  const area = `${X(0)},${Y(0)} ${line} ${X(pts.length - 1)},${Y(0)}`;
  const last = ys[ys.length - 1];
  const yticks = [max, 0, min].filter((v, i, a) => a.indexOf(v) === i);
  const mIdx = marker ? pts.findIndex((p) => p[0] === marker.time) : -1;
  // auto x-axis ticks, but drop any that would collide with the exit-marker label
  const xi = [0, Math.floor((pts.length - 1) / 2), pts.length - 1]
    .filter((v, i, a) => a.indexOf(v) === i)
    .filter((i) => mIdx < 0 || i === 0 || i === pts.length - 1 ? true : Math.abs(i - mIdx) > 4);
  return (
    <div>
      {label && <div style={{ fontSize: 11, color: C.muted, marginBottom: 2 }}>{label}</div>}
      <svg viewBox={`0 0 ${W} ${h}`} width="100%" height={h} preserveAspectRatio="none">
        <defs>
          <linearGradient id={`sg${h}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={col(last)} stopOpacity="0.16" />
            <stop offset="100%" stopColor={col(last)} stopOpacity="0" />
          </linearGradient>
        </defs>
        {yticks.map((v) => (
          <g key={v}>
            <line x1={PAD_L} x2={W - PAD_R} y1={Y(v)} y2={Y(v)}
              stroke={v === 0 ? 'rgba(0,0,0,0.20)' : 'rgba(0,0,0,0.07)'} strokeWidth="1"
              strokeDasharray={v === 0 ? '0' : '3 3'} />
            <text x={PAD_L - 6} y={Y(v) + 3} textAnchor="end" fontSize="9.5" fill={C.muted}>{fmtY(v)}</text>
          </g>
        ))}
        <polygon points={area} fill={`url(#sg${h})`} />
        <polyline points={line} fill="none" stroke={col(last)} strokeWidth="2" />
        {xi.map((i, k) => (
          <text key={k} x={X(i)} y={h - 5} fontSize="9.5" fill={C.muted}
            textAnchor={k === 0 ? 'start' : k === xi.length - 1 ? 'end' : 'middle'}>{pts[i][0]}</text>
        ))}
        {mIdx >= 0 && marker && (
          <g>
            <line x1={X(mIdx)} x2={X(mIdx)} y1={PAD_T} y2={h - PAD_B}
              stroke={C.neg} strokeWidth="1" strokeDasharray="3 3" opacity="0.65" />
            <circle cx={X(mIdx)} cy={Y(marker.pnl)} r="3.6" fill={C.surface} stroke={C.neg} strokeWidth="2" />
            <text x={X(mIdx)} y={Y(marker.pnl) - 7} textAnchor="middle" fontSize="9.5" fontWeight="700" fill={C.neg}>
              {(marker.text ? marker.text + ' ' : '') + fmtY(marker.pnl)}
            </text>
            <text x={X(mIdx)} y={h - 5} textAnchor="middle" fontSize="9.5" fontWeight="700" fill={C.neg}>{marker.time}</text>
          </g>
        )}
      </svg>
    </div>
  );
}

/* ---------- payoff diagram: expiry (solid) + value-now T+0 (dashed), live spot ---------- */
function _ncdf(x: number) {            // standard normal CDF (Abramowitz–Stegun)
  const t = 1 / (1 + 0.2316419 * Math.abs(x)), d = 0.3989423 * Math.exp(-x * x / 2);
  const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  return x > 0 ? 1 - p : p;
}
function _bs(type: 'CE' | 'PE', S: number, K: number, T: number, sig: number) {
  if (T <= 0 || sig <= 0) return type === 'CE' ? Math.max(S - K, 0) : Math.max(K - S, 0);
  const v = sig * Math.sqrt(T), d1 = (Math.log(S / K) + 0.5 * sig * sig * T) / v, d2 = d1 - v;
  return type === 'CE' ? S * _ncdf(d1) - K * _ncdf(d2) : K * _ncdf(-d2) - S * _ncdf(-d1);
}
function PayoffChart({ legs, qty, spot, entrySpot, stopDn, stopUp, dte, iv, currentPnl, h = 200 }:
  { legs: any[]; qty: number; spot?: number | null; entrySpot: number; stopDn?: number; stopUp?: number;
    dte?: number; iv?: number; currentPnl?: number | null; h?: number }) {
  if (!legs || !legs.length || !entrySpot) return null;
  const val = (S: number, useBS: boolean) => {
    let pnl = 0;
    for (const l of legs) {
      const e = Number(l.entry), K = Number(l.strike);
      if (!isFinite(e) || !isFinite(K)) continue;
      const v = useBS ? _bs(l.instrument_type, S, K, dte || 0, iv || 0)
                      : (l.instrument_type === 'CE' ? Math.max(S - K, 0) : Math.max(K - S, 0));
      pnl += l.side === 'SELL' ? e - v : -e + v;
    }
    return pnl * qty;
  };
  const payoff = (S: number) => val(S, false);
  const showT0 = dte != null && dte > 0 && iv != null && iv > 0;
  const off = showT0 && spot && currentPnl != null ? currentPnl - val(spot, true) : 0;  // calibrate to live MTM
  const t0 = (S: number) => val(S, true) + off;
  const lo = entrySpot * 0.94, hi = entrySpot * 1.06, N = 140;
  const ptsE: [number, number][] = [], ptsT: [number, number][] = [];
  for (let i = 0; i <= N; i++) { const S = lo + (i / N) * (hi - lo); ptsE.push([S, payoff(S)]); if (showT0) ptsT.push([S, t0(S)]); }
  const eY = ptsE.map((p) => p[1]); const eMin = Math.min(...eY), eMax = Math.max(...eY);
  const allY = eY.concat(showT0 ? ptsT.map((p) => p[1]) : []);
  const ymin = Math.min(...allY), ymax = Math.max(...allY), yr = ymax - ymin || 1;
  const W = 600, PADL = 52, PADR = 10, PADT = 16, PADB = 22;
  const X = (S: number) => PADL + ((S - lo) / (hi - lo)) * (W - PADL - PADR);
  const Y = (v: number) => PADT + (1 - (v - ymin) / yr) * (h - PADT - PADB);
  const fL = (v: number) => `${v >= 0 ? '+' : '−'}${(Math.abs(v) / 1e5).toFixed(1)}L`;
  const bes: number[] = [];
  for (let i = 1; i < ptsE.length; i++) { const a = ptsE[i - 1], b = ptsE[i];
    if ((a[1] <= 0 && b[1] > 0) || (a[1] >= 0 && b[1] < 0)) bes.push(Math.round(a[0] + (a[1] / (a[1] - b[1])) * (b[0] - a[0]))); }
  const lineE = ptsE.map((p) => `${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(' ');
  const lineT = showT0 ? ptsT.map((p) => `${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(' ') : '';
  return (
    <div>
      <div style={{ fontSize: 11, color: C.muted, marginBottom: 2 }}>
        <b style={{ color: C.navy }}>— at expiry</b>{showT0 ? <b style={{ color: '#2563eb' }}> ··· value now (T+0)</b> : null}
        {' · '}max profit <b style={{ color: C.pos }}>{fL(eMax)}</b> · max loss <b style={{ color: C.neg }}>{fL(eMin)}</b> (capped by wings)
        {bes.length === 2 ? ` · BE ${bes[0].toLocaleString('en-IN')}/${bes[1].toLocaleString('en-IN')}` : ''}
      </div>
      <svg viewBox={`0 0 ${W} ${h}`} width="100%" height={h} preserveAspectRatio="none" style={{ display: 'block' }}>
        <line x1={PADL} x2={W - PADR} y1={Y(0)} y2={Y(0)} stroke="rgba(0,0,0,0.22)" strokeWidth="1" />
        <text x={PADL - 6} y={Y(0) + 3} textAnchor="end" fontSize="9.5" fill={C.muted}>0</text>
        <text x={PADL - 6} y={Y(eMax) + 9} textAnchor="end" fontSize="9.5" fill={C.pos}>{fL(eMax)}</text>
        <text x={PADL - 6} y={Y(eMin) - 2} textAnchor="end" fontSize="9.5" fill={C.neg}>{fL(eMin)}</text>
        {stopDn ? <line x1={X(stopDn)} x2={X(stopDn)} y1={PADT} y2={h - PADB} stroke={C.neg} strokeWidth="1" strokeDasharray="3 3" opacity="0.5" /> : null}
        {stopUp ? <line x1={X(stopUp)} x2={X(stopUp)} y1={PADT} y2={h - PADB} stroke={C.neg} strokeWidth="1" strokeDasharray="3 3" opacity="0.5" /> : null}
        {showT0 ? <polyline points={lineT} fill="none" stroke="#2563eb" strokeWidth="1.6" strokeDasharray="4 3" /> : null}
        <polyline points={lineE} fill="none" stroke={C.navy} strokeWidth="2" />
        {spot ? (
          <g>
            <line x1={X(spot)} x2={X(spot)} y1={PADT} y2={h - PADB} stroke={C.pos} strokeWidth="1.5" />
            <circle cx={X(spot)} cy={Y((showT0 ? t0 : payoff)(spot))} r="3.6" fill="#fff" stroke={C.pos} strokeWidth="2" />
            <text x={X(spot)} y={PADT - 4} textAnchor="middle" fontSize="9.5" fontWeight="700" fill={C.pos}>now {Math.round(spot).toLocaleString('en-IN')}</text>
          </g>
        ) : null}
        {[lo, entrySpot, hi].map((S, k) => (
          <text key={k} x={X(S)} y={h - 6} fontSize="9.5" fill={C.muted} textAnchor={k === 0 ? 'start' : k === 2 ? 'end' : 'middle'}>{Math.round(S).toLocaleString('en-IN')}</text>
        ))}
      </svg>
    </div>
  );
}

const card: React.CSSProperties = { border: `1px solid ${C.hair}`, background: C.surface, borderRadius: 10, padding: '16px 18px', marginBottom: 18, boxShadow: '0 1px 2px rgba(0,0,0,0.04)' };
const stat = (label: string, value: string, c?: string) => (
  <div><div style={{ fontSize: 11, color: C.muted }}>{label}</div><div style={{ fontSize: 19, fontWeight: 700, color: c || C.ink }}>{value}</div></div>
);
const chip = (bg: string, fg: string, t: string) => (
  <span style={{ background: bg, color: fg, fontSize: 11, fontWeight: 600, padding: '2px 8px', borderRadius: 6 }}>{t}</span>
);
const ecth: React.CSSProperties = { fontSize: 10, color: C.muted, fontWeight: 600, textAlign: 'right', padding: '2px 6px', borderBottom: `1px solid ${C.hairSoft}` };
const ectd: React.CSSProperties = { fontSize: 11, color: C.ink, textAlign: 'right', padding: '3px 6px', fontVariantNumeric: 'tabular-nums', borderTop: `1px solid ${C.hairSoft}` };

/* ---------- live positions table (trade-book style) ---------- */
function LegsTable({ legs, total }: { legs?: Leg[]; total: number }) {
  if (!legs || !legs.length) return null;
  const th: React.CSSProperties = { fontSize: 10, color: C.muted, fontWeight: 600, textAlign: 'right',
    padding: '3px 8px', textTransform: 'uppercase', letterSpacing: '0.04em', borderBottom: `1px solid ${C.hair}` };
  const td: React.CSSProperties = { fontSize: 12.5, color: C.ink, textAlign: 'right', padding: '5px 8px',
    fontVariantNumeric: 'tabular-nums', borderTop: `1px solid ${C.hairSoft}` };
  const px = (v: number | null) => (v == null ? '—' : v.toFixed(1));
  const tm = (v?: string | null) => (v ? v : '—');
  return (
    <table style={{ width: '100%', borderCollapse: 'collapse', margin: '2px 0 8px' }}>
      <thead><tr>
        <th style={{ ...th, textAlign: 'left' }}>Leg</th>
        <th style={th}>Strike</th><th style={th}>Qty</th>
        <th style={th}>In</th><th style={th}>Entry</th><th style={th}>LTP</th>
        <th style={{ ...th, color: C.faint }} title="Max premium the leg reached since entry (max adverse excursion)">Peak</th>
        <th style={th}>Out</th><th style={th} title="premium the leg was exited at">Out ₹</th><th style={th}>P&amp;L</th>
      </tr></thead>
      <tbody>
        {legs.map((l, i) => (
          <tr key={i}>
            <td style={{ ...td, textAlign: 'left' }}>
              {chip(l.type === 'CE' ? C.navySoft : C.amberSoft, l.type === 'CE' ? C.navy : C.amber, `SELL ${l.type}`)}
            </td>
            <td style={td}>{l.strike}</td>
            <td style={td}>{l.qty.toLocaleString('en-IN')}</td>
            <td style={{ ...td, color: C.sec }}>{tm(l.entry_time)}</td>
            <td style={td}>{px(l.entry)}</td>
            <td style={td}>{px(l.ltp)}</td>
            <td style={{ ...td, color: C.faint, fontSize: 11.5, fontWeight: 400 }}>{l.max_ltp == null ? '—' : l.max_ltp.toFixed(1)}</td>
            <td style={{ ...td, color: l.exit_time ? C.neg : C.faint }}>{tm(l.exit_time)}</td>
            <td style={{ ...td, color: l.exit_time ? C.ink : C.faint }}>{l.exit_time ? px(l.ltp) : '—'}</td>
            <td style={{ ...td, fontWeight: 700, color: col(l.pnl) }}>{inr(l.pnl)}</td>
          </tr>
        ))}
        <tr>
          <td style={{ ...td, textAlign: 'left', color: C.muted, borderTop: `1px solid ${C.hair}` }} colSpan={9}>
            Net · paper position (incl. costs)
          </td>
          <td style={{ ...td, fontWeight: 800, color: col(total), borderTop: `1px solid ${C.hair}` }}>{inr(total)}</td>
        </tr>
      </tbody>
    </table>
  );
}

/* ---------- collapsible system rules (both systems) ---------- */
function RulesBlock() {
  const head: React.CSSProperties = { fontWeight: 700, color: C.ink, fontSize: 13, margin: '0 0 4px' };
  const li: React.CSSProperties = { fontSize: 12, color: C.sec, lineHeight: 1.55, margin: '0 0 2px' };
  const k = (t: string) => <span style={{ color: C.ink, fontWeight: 600 }}>{t}</span>;
  return (
    <details style={{ marginTop: 12, borderTop: `1px solid ${C.hair}`, paddingTop: 10 }}>
      <summary style={{ cursor: 'pointer', fontSize: 12.5, fontWeight: 700, color: C.navy, listStyle: 'none', userSelect: 'none' }}>
        ▸ System rules — V1 &amp; V2 (click to expand)
      </summary>
      <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', marginTop: 10 }}>
        <div style={{ flex: 1, minWidth: 280 }}>
          <p style={head}>V1 · Intraday one-and-done</p>
          <ul style={{ margin: 0, paddingLeft: 16 }}>
            <li style={li}>{k('Instrument:')} short NIFTY weekly {k('ATM straddle')} (sell ATM CE + ATM PE), 10 lots · qty 650.</li>
            <li style={li}>{k('Entry:')} 09:20, only on {k('0-DTE or 1-DTE')} days (typically Mon/Tue).</li>
            <li style={li}>{k('Stop:')} underlying move {k('0.4% (0-DTE) / 0.5% (1-DTE)')} from entry strike → exit both legs flat. {k('One-and-done')} — no re-entry.</li>
            <li style={li}>{k('Else:')} hold to ~15:15 close.</li>
            <li style={li}>Edge concentrated on 0/1-DTE; P&amp;L net of brokerage + slippage.</li>
          </ul>
        </div>
        <div style={{ flex: 1, minWidth: 280 }}>
          <p style={head}>V2 · Positional bi-weekly (iron fly)</p>
          <ul style={{ margin: 0, paddingLeft: 16 }}>
            <li style={li}>{k('Instrument:')} sell 2nd-nearest weekly ATM straddle + buy {k('±500-pt wings')} (≈2.0% of ATM) = short {k('iron fly')}. Overnight carry.</li>
            <li style={li}>{k('Entry:')} 09:20, ~8 trading days to expiry; {k('roll')} 1 TD before expiry.</li>
            <li style={li}>{k('Exits:')} {k('2.0%')} underlying-move stop, or {k('+40%')} profit target, or roll at DTE≤1; {k('re-enter')} after exit.</li>
            <li style={li}>{k('Entry filter:')} India {k('VIX ≥ 13')} (backtested lock — lifts every full year positive).</li>
            <li style={li}>10 lots · qty 650. Net of taxes + ₹20/order + 0.25% slippage.</li>
          </ul>
        </div>
      </div>
      <div style={{ fontSize: 11, color: C.faint, marginTop: 8 }}>
        Live V2 card currently tracks the core short straddle; full wings / VIX / profit-target logic is the backtested spec being wired into the live engine.
      </div>
    </details>
  );
}

/* ---------- page ---------- */
export default function Straddles() {
  const [v1, setV1] = useState<V1 | null>(null);
  const [v2, setV2] = useState<V2 | null>(null);
  const [v2all, setV2all] = useState<{ [k: string]: V2 }>({});
  const [v2stop, setV2stop] = useState<'1.5' | '2.0'>('2.0');
  const [day1, setDay1] = useState<string | null>(null);
  const [tr2, setTr2] = useState<number | null>(null);
  const [live, setLive] = useState<any>(null);
  const [liveTs, setLiveTs] = useState<number | null>(null);
  const [daily, setDaily] = useState<V1Daily | null>(null);
  const [sl30, setSl30] = useState<any>(null);   // V1 + 30% premium SL backtest
  const [dayD, setDayD] = useState<string | null>(null);
  const [v2eng, setV2eng] = useState<any>(null);   // live paper executor state
  const [bo, setBo] = useState<any>(null);          // inside-week breakout sleeve
  const [v2engTs, setV2engTs] = useState<number | null>(null);  // last v2-engine live tick
  const [v2spot, setV2spot] = useState<number | null>(null);    // live NIFTY spot from the stream
  const [v2prev, setV2prev] = useState<any>(null);   // live deploy preview (legs + margin + gate)
  const [v2busy, setV2busy] = useState(false);       // an action is in flight
  const [condor, setCondor] = useState<any>(null);   // research/80 Wed->Fri iron-condor paper book
  const [v2msg, setV2msg] = useState<string | null>(null);  // last action result message
  const [ranks, setRanks] = useState<any>(null);     // weekly strategy leaderboard
  const [variants, setVariants] = useState<any>(null); // v2 stop x wings variant lab
  const [cslCfg, setCslCfg] = useState<any>(null);   // research/111 best-config lab (weekly regen)
  const [cslIdx, setCslIdx] = useState<'NIFTY' | 'SENSEX'>('NIFTY');
  const [cslDte, setCslDte] = useState<string>('all');
  const [csl2nd, setCsl2nd] = useState(false);       // stack the next-best non-overlapping slot
  const [cslPaper, setCslPaper] = useState<any>(null);     // live paper book state
  const [cslPaperCfg, setCslPaperCfg] = useState<any>(null); // frozen config
  const [cslPLive, setCslPLive] = useState<any>(null);     // intraday live series (today)
  const [cslPDay, setCslPDay] = useState<string>('');      // selected day for curves
  const [cslPBig, setCslPBig] = useState<string | null>(null); // expanded book curve

  useEffect(() => {
    fetch('/app/straddles/v1.json').then((r) => r.json()).then(setV1).catch(() => {});
    fetch('/app/straddles/v2_2.0.json').then((r) => r.json()).then((d) => setV2all((m) => ({ ...m, '2.0': d }))).catch(() => {});
    fetch('/app/straddles/v2_1.5.json').then((r) => r.json()).then((d) => setV2all((m) => ({ ...m, '1.5': d }))).catch(() => {});
    fetch('/app/straddles/v1_daily.json').then((r) => r.json()).then(setDaily).catch(() => {});
    fetch('/app/straddles/v1_sl30.json?t=' + Date.now()).then((r) => r.json()).then(setSl30).catch(() => {});
    fetch('/app/straddles/rankings.json?t=' + Date.now()).then((r) => r.json()).then(setRanks).catch(() => {});
    fetch('/app/straddles/variants.json?t=' + Date.now()).then((r) => r.json()).then(setVariants).catch(() => {});
    fetch('/app/straddles/csl_best_configs.json?t=' + Date.now()).then((r) => r.json()).then(setCslCfg).catch(() => {});
    fetch('/app/csl_paper.json?t=' + Date.now()).then((r) => r.json()).then(setCslPaper).catch(() => {});
    fetch('/app/csl_paper_config.json?t=' + Date.now()).then((r) => r.json()).then(setCslPaperCfg).catch(() => {});
    const loadPLive = () => fetch('/app/csl_paper_live.json?t=' + Date.now()).then((r) => r.json()).then(setCslPLive).catch(() => {});
    loadPLive(); const pl = setInterval(loadPLive, 60000);
    const loadLive = () => {
      fetch('/app/straddles_live.json?t=' + Date.now()).then((r) => r.json()).then(setLive).catch(() => {});
      fetch('/api/v2-ironfly/state?t=' + Date.now()).then((r) => r.json()).then(setV2eng).catch(() => {});
      fetch('/api/v2-breakout/state?t=' + Date.now()).then((r) => r.json()).then(setBo).catch(() => {});
      fetch('/app/condor_paper.json?t=' + Date.now()).then((r) => r.json()).then(setCondor).catch(() => {});
    };
    loadLive();
    const id = setInterval(loadLive, 30000);
    // Live-quote SSE overlay: ticks pnl_now + per-leg LTP/P&L every ~3s on top of the
    // cron base (which still supplies series, detail, entry/exit times). Mirrors the NAS stream.
    let es: EventSource | null = null;
    try {
      es = new EventSource('/api/straddles/stream');
      es.onmessage = (e) => {
        let m: any; try { m = JSON.parse(e.data); } catch { return; }
        if (m.type === 'tick') setLiveTs(m.ts);
        if (m.type !== 'tick' || !m.systems) return;
        setLive((prev: any) => {
          if (!prev) return prev;
          const next = { ...prev };
          (['v1', 'v2'] as const).forEach((kk) => {
            const s = m.systems[kk]; if (!s || !next[kk]) return;
            const d = { ...next[kk] };
            if (s.pnl_now != null) d.pnl_now = s.pnl_now;
            if (Array.isArray(d.legs)) d.legs = d.legs.map((l: Leg) => {
              const ltp = l.type === 'CE' ? s.ce_ltp : s.pe_ltp;
              const pnl = l.type === 'CE' ? s.ce_pnl : s.pe_pnl;
              return { ...l, ltp: ltp != null ? ltp : l.ltp, pnl: pnl != null ? pnl : l.pnl };
            });
            next[kk] = d;
          });
          return next;
        });
      };
    } catch { /* SSE unsupported — static poll still ticks every 30s */ }
    // V2 engine live-quote stream: ticks pnl_now + per-leg LTP/P&L every ~3s on the open fly.
    let es2: EventSource | null = null;
    try {
      es2 = new EventSource('/api/v2-ironfly/stream');
      es2.onmessage = (e) => {
        let m: any; try { m = JSON.parse(e.data); } catch { return; }
        if (m.type !== 'tick') return;
        setV2engTs(m.ts);
        if (m.spot != null) setV2spot(m.spot);
        setV2eng((prev: any) => {
          if (!prev || !prev.open) return prev;
          const open = { ...prev.open, pnl_now: m.pnl_now };
          if (Array.isArray(open.legs) && Array.isArray(m.legs)) {
            open.legs = open.legs.map((l: any) => {
              const t = m.legs.find((x: any) => x.strike === l.strike && x.instrument_type === l.instrument_type);
              return t ? { ...l, ltp: t.ltp, pnl: t.pnl } : l;
            });
          }
          return { ...prev, open };
        });
      };
    } catch { /* no SSE — 30s poll still refreshes */ }
    return () => { clearInterval(id); clearInterval(pl); if (es) es.close(); if (es2) es2.close(); };
  }, []);

  const v1stats = useMemo(() => {
    if (!v1) return null;
    const f = Object.values(v1.per_day).map((d) => d.final);
    const tot = f.reduce((a, b) => a + b, 0);
    return { n: f.length, tot, mean: tot / (f.length || 1), win: 100 * f.filter((x) => x > 0).length / (f.length || 1) };
  }, [v1]);
  useEffect(() => { if (v2all[v2stop]) setV2(v2all[v2stop]); }, [v2all, v2stop]);
  const v2stats = useMemo(() => {
    if (!v2) return null;
    const f = v2.trades.map((t) => t.pnl);
    const tot = f.reduce((a, b) => a + b, 0);
    return { n: f.length, tot, mean: tot / (f.length || 1), win: 100 * f.filter((x) => x > 0).length / (f.length || 1) };
  }, [v2]);
  const dailyStats = useMemo(() => {
    if (!daily) return null;
    const f = daily.days.map((k) => daily.per_day[k].final);
    const tot = f.reduce((a, b) => a + b, 0);
    const stops = daily.days.filter((k) => daily.per_day[k].stopped).length;
    return { n: f.length, tot, mean: tot / (f.length || 1), win: 100 * f.filter((x) => x > 0).length / (f.length || 1), stops };
  }, [daily]);
  useEffect(() => {
    if (daily && !dayD && daily.days.length) setDayD(daily.days[daily.days.length - 1]);
  }, [daily, dayD]);

  // ----- V2 live executor controls -----
  const refreshV2 = () => fetch('/api/v2-ironfly/state?t=' + Date.now()).then((r) => r.json()).then(setV2eng).catch(() => {});
  const v2post = async (path: string, body?: any) => {
    setV2busy(true); setV2msg(null);
    try {
      const r = await fetch('/api/v2-ironfly/' + path, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}),
      });
      const j = await r.json();
      if (j.error) setV2msg('⚠ ' + j.error);
      else if (path === 'deploy') setV2msg(j.entered ? '✓ Live trade deployed — automation armed.' : '⚠ Armed, but no entry today: ' + (j.reason || ''));
      else if (path === 'mode') setV2msg('Mode → ' + String(j.mode || '').toUpperCase() + ' (disarmed)');
      else if (path === 'kill-switch') setV2msg('Kill-switch: closed ' + (j.closed ?? 0) + ', automation disarmed.');
      await refreshV2();
      return j;
    } catch { setV2msg('⚠ request failed'); }
    finally { setV2busy(false); }
  };
  const v2SetMode = (mode: string) => {
    if (mode === 'live' && !window.confirm('Switch the V2 engine to LIVE (real-money) mode?\n\nThis only ENABLES live — no order is placed until you click Deploy.')) return;
    v2post('mode', { mode });
  };
  const v2Deploy = (force: boolean) => {
    const p = v2prev;
    const detail = p && p.ok
      ? `\n\nLegs: ${p.legs.map((l: any) => `${l.side} ${l.strike}${l.instrument_type}`).join(', ')}\nNet credit: ${p.net_credit}   Margin: ₹${(p.margin_need || 0).toLocaleString('en-IN')} (avail ₹${(p.margin_avail || 0).toLocaleString('en-IN')})\nFilter gate: ${p.gate}` : '';
    if (!window.confirm(`Deploy a REAL 10-lot NIFTY iron fly now?${detail}\n\n${force ? '⚠ FORCE = bypass the VIX / skip-filter gate.\n\n' : ''}After this, rolls & re-entries run automatically until you kill-switch.`)) return;
    v2post('deploy', { force });
  };
  const v2Kill = () => {
    if (!window.confirm('KILL-SWITCH — square off the open position now (real orders if live) and disarm automation?')) return;
    v2post('kill-switch');
  };
  useEffect(() => {
    if (v2eng?.deployable) fetch('/api/v2-ironfly/preview?t=' + Date.now()).then((r) => r.json()).then(setV2prev).catch(() => setV2prev(null));
    else setV2prev(null);
  }, [v2eng?.deployable, v2eng?.mode]);

  const days1 = v1 ? Object.keys(v1.per_day).sort() : [];
  const btn = (sel: boolean, c: string): React.CSSProperties => ({
    cursor: 'pointer', border: `1px solid ${sel ? C.navy : C.hair}`, background: sel ? C.navySoft : C.surface,
    color: c, borderRadius: 6, padding: '4px 8px', fontSize: 11, fontWeight: 600,
  });

  const cth2: React.CSSProperties = { fontSize: 9.5, color: C.muted, fontWeight: 600, textAlign: 'right', padding: '2px 8px', textTransform: 'uppercase', borderBottom: `1px solid ${C.hairSoft}` };
  const ctd2: React.CSSProperties = { fontSize: 11.5, color: C.ink, textAlign: 'right', padding: '4px 8px', borderTop: `1px solid ${C.hairSoft}`, fontVariantNumeric: 'tabular-nums' };
  const thL: React.CSSProperties = { fontSize: 9.5, color: C.muted, fontWeight: 600, textAlign: 'left', padding: '2px 8px', textTransform: 'uppercase', borderBottom: `1px solid ${C.hairSoft}` };
  const thR: React.CSSProperties = { ...thL, textAlign: 'right' };
  const tdL: React.CSSProperties = { fontSize: 11.5, color: C.sec, textAlign: 'left', padding: '5px 8px', borderTop: `1px solid ${C.hairSoft}`, fontVariantNumeric: 'tabular-nums' };
  const tdR: React.CSSProperties = { ...tdL, textAlign: 'right', color: C.ink };
  const gradeBadge = (g: string) => {
    const m: Record<string, [string, string]> = { A: [C.pos, '#E6F4EF'], B: [C.navy, C.navySoft], C: [C.amber, C.amberSoft], D: [C.neg, '#FBE9E9'], F: [C.neg, '#FBE9E9'] };
    const [fg, bg] = m[g] || [C.muted, C.hairSoft];
    return <span style={{ background: bg, color: fg, fontWeight: 800, fontSize: 11, padding: '1px 8px', borderRadius: 5 }}>{g}</span>;
  };
  return (
    <div style={{ maxWidth: 1000 }}>
      <style>{`@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}`}</style>
      <div className="page-title">Straddle Systems</div>
      <div className="page-subtitle">Two short-straddle systems on NIFTY · backtested on the recorded chain · paper-forward 10 lots</div>

      {/* ===== research/111 STUDY HUB ===== */}
      <section id="hub" style={{ ...card, marginTop: 14, borderColor: C.ink }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap', marginBottom: 6 }}>
          <span style={{ fontSize: 16, fontWeight: 800, color: C.ink }}>Short-Straddle Study Hub</span>
          {chip(C.navySoft, C.navy, 'research/111 · concluded 13-AUG-26')}
          {chip('#E7F2EE', C.pos, 'validation: paper-forward since 14-AUG')}
        </div>
        <div style={{ fontSize: 12.5, color: C.sec, lineHeight: 1.55, marginBottom: 8 }}>
          <b style={{ color: C.ink }}>Conclusion:</b> (1) <b>Time-boxed, DTE-scheduled straddles with a combined-premium SL (CSL)</b> are the best-constructed
          short-vol system on both indices — the <b>time window is the edge</b> (SL level barely binds inside it; 30% stays as disaster backstop; drawdowns cut 10–25×).
          (2) <b>Combined-SL ≫ per-leg SL</b>: on identical trades the per-leg mechanic lost while combined earned (+₹5.7L swing, 10 lots, 50d) — NAS's per-leg books survive
          only via their trail/adjust rescue layers. (3) <b>Schedule beats stop-tuning</b>: NIFTY-Thu &amp; SENSEX-expiry-Thu are the full-day holds; Wednesdays only in the
          10:30→12:00 window. (4) <b>Portfolio</b>: CSL-NIFTY 2u : CSL-SENSEX 1u alongside the live NAS sleeves (corr ≈ 0 — genuine diversification).
          <b style={{ color: C.amber }}> Grade: strong SIGNAL, in-sample</b> — the paper books below convert it to STRATEGY (or kill it) by ~mid-Sep.
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', fontSize: 12 }}>
          {[['/app/backtest/csl-best-config-straddles', '📄 Full study card (backtest page)'],
            ['#csl-paper', '📗 Paper Books (live validation)'], ['#csl-lab', '🔬 Best-Config Lab (weekly)'],
            ['#leaderboard', '🏆 Strategy Leaderboard'], ['#variant-lab', '🧪 V2 Variant Lab'],
            ['/app/nifty_csl_vs_nas.png', '📈 NIFTY: CSL vs NAS (chart)'], ['/app/sensex_csl_vs_nas.png', '📈 SENSEX: CSL vs NAS (chart)'],
            ['/app/perleg_vs_comb.png', '📉 Per-leg vs Combined SL (chart)'], ['/app/csl30_vs_nas916.png', '📊 CSL30 vs NAS-916 (chart)'],
            ['/app/options-study', '🕯 Opt-Study (decay/CPR/candles)']].map(([href, label]) => (
            <a key={href} href={href}
              onClick={(e) => { if (href.startsWith('#')) { e.preventDefault(); document.getElementById(href.slice(1))?.scrollIntoView({ behavior: 'smooth' }); } }}
              style={{ border: `1px solid ${C.hair}`, borderRadius: 8, padding: '6px 10px', textDecoration: 'none', color: C.navy, fontWeight: 600 }}>
              {label}</a>
          ))}
        </div>
        <div style={{ fontSize: 10.5, color: C.faint, marginTop: 8 }}>
          Full artifacts: research/111_sensex_manual_mgmt (STATUS doc, sweeps, comparisons, portfolio scan) · basis stated on every card (lots · n-days · date range) ·
          rules: 3-sec-first data, live-first backfill, dwell-mechanic fills, per-DTE per-index.
        </div>
      </section>

      {ranks && ranks.systems && (
        <section id="leaderboard" style={{ ...card, marginTop: 14, scrollMarginTop: 70 }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, flexWrap: 'wrap', marginBottom: 8 }}>
            <span style={{ fontSize: 16, fontWeight: 700, color: C.ink }}>Strategy Leaderboard</span>
            <span style={{ fontSize: 11, color: C.faint }}>rated by risk-adjusted return (Calmar) · updated {ranks.generated_at} · {ranks.cadence}</span>
          </div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead><tr>
                <th style={thL}>#</th><th style={thL}>Grade</th><th style={thL}>System</th>
                <th style={thR}>Net P&amp;L</th><th style={thR}>Calmar</th><th style={thR}>MaxDD</th>
                <th style={thR}>Win</th><th style={thR}>N</th><th style={thL}>Confidence</th>
              </tr></thead>
              <tbody>
                {ranks.systems.map((r: any) => (
                  <tr key={r.label}>
                    <td style={tdL}>{r.rank}</td>
                    <td style={tdL}>{gradeBadge(r.grade)}</td>
                    <td style={{ ...tdL, color: C.ink, fontWeight: 600 }}>
                      {r.anchor
                        ? <a href={'#' + r.anchor} onClick={(e) => { e.preventDefault(); document.getElementById(r.anchor)?.scrollIntoView({ behavior: 'smooth', block: 'start' }); }} style={{ color: C.navy, textDecoration: 'none', cursor: 'pointer' }}>{r.label} ↗</a>
                        : (r.page ? <a href={r.page} style={{ color: C.navy, textDecoration: 'none' }}>{r.label} ↗</a> : r.label)}
                      <div style={{ fontSize: 10, color: C.faint, fontWeight: 400 }}>{r.kind} · {r.note}
                        {r.report && <> · <a href={r.report} style={{ color: C.navy }}>📄 report</a></>}
                        {r.chart && <> · <a href={r.chart} style={{ color: C.navy }}>📈 tearsheet</a></>}
                      </div></td>
                    <td style={{ ...tdR, color: col(r.net), fontWeight: 700 }}>{inr(r.net)}</td>
                    <td style={tdR}>{r.calmar ?? '—'}</td>
                    <td style={{ ...tdR, color: C.neg }}>{inr(r.maxdd)}</td>
                    <td style={tdR}>{r.win}%</td>
                    <td style={tdR}>{r.n}</td>
                    <td style={{ ...tdL, color: r.confidence === 'medium' ? C.sec : C.amber }}>{r.confidence}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div style={{ fontSize: 11, color: C.amber, marginTop: 8 }}>⚠ {ranks.caveat}</div>
        </section>
      )}

      {variants && variants.variants && (
        <section id="variant-lab" style={{ ...card, marginTop: 14, scrollMarginTop: 70 }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, flexWrap: 'wrap', marginBottom: 8 }}>
            <span style={{ fontSize: 16, fontWeight: 700, color: C.ink }}>V2 Variant Lab</span>
            <span style={{ fontSize: 11, color: C.faint }}>same recorded chain · naked vs iron-fly × move-stop · {variants.note}</span>
          </div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead><tr>
                <th style={thL}>Variant</th><th style={thR}>Net</th><th style={thR}>Mean/tr</th>
                <th style={thR}>Win</th><th style={thR}>MaxDD</th><th style={thR}>Calmar</th><th style={thR}>N</th>
              </tr></thead>
              <tbody>
                {variants.variants.map((v: any) => (
                  <tr key={v.label}>
                    <td style={{ ...tdL, color: C.ink }}>{v.label}</td>
                    <td style={{ ...tdR, color: col(v.net), fontWeight: 700 }}>{inr(v.net)}</td>
                    <td style={{ ...tdR, color: col(v.mean) }}>{inr(v.mean)}</td>
                    <td style={tdR}>{v.win}%</td>
                    <td style={{ ...tdR, color: C.neg }}>{inr(v.maxdd)}</td>
                    <td style={tdR}>{v.calmar ?? '—'}</td>
                    <td style={tdR}>{v.trades}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div style={{ fontSize: 11, color: C.faint, marginTop: 8 }}>
            In this calm regime wide/no stop wins &amp; tight stops whipsaw — but naked's small drawdown is only because no crash hit the sample (unbounded tail). Signal, not a verdict.
          </div>
        </section>
      )}

      <section id="csl-paper" style={{ ...card, marginTop: 14, borderColor: C.pos, scrollMarginTop: 70 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap', marginBottom: 4 }}>
          <span style={{ fontSize: 16, fontWeight: 700, color: C.ink }}>CSL Paper Books</span>
          {chip('#E7F2EE', C.pos, 'PAPER · CSL-N 12L + CSL-S 6L + NAS-COMB20 3L')}
          {chip(C.navySoft, C.navy, 'frozen config' + (cslPaperCfg ? ' · ' + String(cslPaperCfg.frozen_at || '').slice(0, 10) : ''))}
          <span style={{ marginLeft: 'auto', fontSize: 12, fontWeight: 700 }}>
            {cslPaper && Object.keys(cslPaper.cum || {}).length
              ? Object.entries(cslPaper.cum).map(([b, v]: any, i: number) => <span key={b}>{i > 0 && ' · '}{b} <span style={{ color: col(v) }}>{inr(v)}</span></span>)
              : <span style={{ color: C.muted }}>no trades yet</span>}
          </span>
        </div>
        <div style={{ fontSize: 11, color: C.muted, marginBottom: 8 }}>
          Out-of-sample validation of the Lab's best configs (frozen 13-AUG-26; weekly Lab drift does NOT move this book) ·
          entries/exits per frozen schedule · combined-SL, 5s polling, 2-poll dwell, market exit next poll · 'none'-SL days carry a 50% disaster backstop · runs 09:12 via cron, first trading day 14-AUG-26.
        </div>
        {cslPaperCfg && (
          <div style={{ overflowX: 'auto', marginBottom: 8 }}>
            <table style={{ borderCollapse: 'collapse', fontSize: 11.5 }}>
              <thead><tr><th style={thL}>Book</th>{[0, 1, 2, 3, 4].map((k) => <th key={k} style={thL}>DTE{k}</th>)}</tr></thead>
              <tbody>{Object.keys(cslPaperCfg.books || {}).map((sym) => (
                <tr key={sym}><td style={{ ...tdL, fontWeight: 700 }}>{sym}</td>
                  {[0, 1, 2, 3, 4].map((k) => { const c2 = cslPaperCfg.books[sym][String(k)];
                    return <td key={k} style={tdL}>{c2 ? `${c2.entry}→${c2.exit} SL${c2.sl === 'none' ? '∅' : c2.sl}` : '—'}</td>; })}
                </tr>))}
              </tbody>
            </table>
          </div>
        )}
        {(() => {
          const recDays: string[] = Array.from(new Set(((cslPaper?.records) || []).filter((r: any) => r.series && r.series.length > 1).map((r: any) => r.day)));
          if (cslPLive?.day && Object.values(cslPLive.books || {}).some((b: any) => (b.series || []).length > 1) && !recDays.includes(cslPLive.day)) recDays.push(cslPLive.day);
          recDays.sort();
          const selDay = cslPDay && recDays.includes(cslPDay) ? cslPDay : recDays[recDays.length - 1];
          if (!selDay) return null;
          const curves: { bk: string; pts: [string, number][]; live: boolean }[] = [];
          (((cslPaper?.records) || []) as any[]).filter((r) => r.day === selDay && r.series && r.series.length > 1)
            .forEach((r) => curves.push({ bk: r.book || r.sym, pts: r.series, live: false }));
          if (cslPLive?.day === selDay) Object.entries(cslPLive.books || {}).forEach(([bk, b]: any) => {
            if ((b.series || []).length > 1 && !curves.some((c2) => c2.bk === bk)) curves.push({ bk, pts: b.series, live: b.state === 'OPEN' });
          });
          if (!curves.length) return null;
          const big = curves.find((c2) => c2.bk === cslPBig);
          return (
            <div style={{ margin: '10px 0' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', marginBottom: 6 }}>
                <span style={{ fontSize: 12, fontWeight: 700, color: C.ink }}>Day P&amp;L curves — all variants</span>
                <select value={selDay} onChange={(e) => setCslPDay(e.target.value)}
                  style={{ background: 'transparent', color: C.ink, border: `1px solid ${C.hair}`, borderRadius: 6, padding: '2px 8px', fontSize: 11.5 }}>
                  {recDays.map((d2) => <option key={d2} value={d2}>{d2}</option>)}
                </select>
                <span style={{ fontSize: 10.5, color: C.faint }}>~60s samples · click a curve to expand/collapse</span>
              </div>
              {big && <div style={{ marginBottom: 8 }}>
                <LineChart pts={big.pts} h={260} label={`${big.bk} · ${selDay}${big.live ? ' · LIVE (open)' : ''} — expanded (click tile to collapse)`} />
              </div>}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 10 }}>
                {curves.map((c2) => (
                  <div key={c2.bk} onClick={() => setCslPBig(cslPBig === c2.bk ? null : c2.bk)}
                    style={{ border: `1px solid ${cslPBig === c2.bk ? C.navy : C.hair}`, borderRadius: 8, padding: '6px 8px', cursor: 'pointer' }}>
                    <div style={{ fontSize: 11, fontWeight: 700, color: C.ink }}>
                      {c2.bk} {c2.live && <span style={{ color: C.pos }}>· LIVE</span>}
                      <span style={{ float: 'right', color: col(c2.pts[c2.pts.length - 1][1]) }}>{inr(c2.pts[c2.pts.length - 1][1])}</span>
                    </div>
                    <LineChart pts={c2.pts} h={90} />
                  </div>
                ))}
              </div>
            </div>
          );
        })()}
        {cslPaper && (cslPaper.records || []).length > 0 ? (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead><tr><th style={thL}>Day</th><th style={thL}>Book</th><th style={thL}>Cfg</th><th style={thR}>Strike</th><th style={thR}>Credit</th><th style={thR}>Exit</th><th style={thL}>Reason</th><th style={thR}>P&amp;L</th><th style={thR}>Lots</th></tr></thead>
              <tbody>{cslPaper.records.slice().reverse().map((r: any, i: number) => (
                <tr key={i}><td style={tdL}>{r.day}</td><td style={tdL}>{r.book || r.sym} D{r.dte}</td><td style={tdL}>{r.cfg}</td>
                  <td style={tdR}>{r.strike}</td><td style={tdR}>{r.credit}</td><td style={tdR}>{r.exit_comb} <span style={{ color: C.faint }}>@{String(r.exit_ts || '').slice(0, 5)}</span></td>
                  <td style={{ ...tdL, color: String(r.reason).startsWith('SL') ? C.neg : C.muted }}>{r.reason}</td>
                  <td style={{ ...tdR, fontWeight: 700, color: col(r.pnl) }}>{inr(r.pnl)}</td><td style={tdR}>{r.lots}</td></tr>))}
              </tbody>
            </table>
          </div>
        ) : (
          <div style={{ fontSize: 12, color: C.faint }}>No completed paper trades yet — first entries fire Friday 14-AUG (NIFTY 10:00→12:00 · SENSEX 10:30→12:00). Records appear here automatically.</div>
        )}
      </section>

      {!cslCfg && (
        <section id="csl-lab" style={{ ...card, marginTop: 14, borderColor: C.navy }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
            <span style={{ fontSize: 16, fontWeight: 700, color: C.ink }}>CSL Best-Config Lab</span>
            {chip(C.navySoft, C.navy, 'research/111')}
            {chip(C.amberSoft, C.amber, 'first data generation in progress…')}
          </div>
          <div style={{ fontSize: 12, color: C.muted, marginTop: 6 }}>
            The entry×exit×SL sweep (NIFTY + SENSEX, all recorded days, 3-sec dwell) is computing on the VPS.
            This card fills in automatically when it finishes, and refreshes every Friday 15:45 IST thereafter.
          </div>
        </section>
      )}

      {cslCfg && cslCfg.best && (() => {
        const DLBL: any = { NIFTY: { 0: 'Tue·exp', 1: 'Mon', 2: 'Fri', 3: 'Thu', 4: 'Wed' },
                            SENSEX: { 0: 'Thu·exp', 1: 'Wed', 2: 'Tue', 3: 'Mon', 4: 'Fri' } };
        const b = cslCfg.best[cslIdx] || {};
        const dtes = Object.keys(b).sort();
        const use = cslDte === 'all' ? dtes : dtes.filter((k) => k === cslDte);
        // next-best NON-OVERLAPPING second slot for the selected DTE (quality-gated)
        const second = (cslDte !== 'all' && b[cslDte]) ? ((cslCfg.cells || [])
          .filter((c: any) => c.sym === cslIdx && String(c.dte) === cslDte && c.n >= 8 && c.total > 0 &&
            (c.ratio ?? 0) >= 1.5 && c.series &&
            (c.entry >= b[cslDte].exit || c.exit <= b[cslDte].entry))
          .sort((a: any, z: any) => (z.ratio ?? -9) - (a.ratio ?? -9))[0] || null) : null;
        const daily: Record<string, number> = {};
        use.forEach((k) => (b[k]?.series || []).forEach(([d, v]: any) => { daily[d] = (daily[d] || 0) + v; }));
        if (csl2nd && second) second.series.forEach(([d, v]: any) => { daily[d] = (daily[d] || 0) + v; });
        let cum = 0, peak = 0, shownDD = 0;
        const pts: [string, number][] = [], ddPts: [string, number][] = [];
        Object.keys(daily).sort().forEach((d) => { cum += daily[d]; peak = Math.max(peak, cum);
          shownDD = Math.min(shownDD, cum - peak); pts.push([d, cum]); ddPts.push([d, Math.round(cum - peak)]); });
        const shownTotal = Math.round(cum);
        const mi = (cslCfg.meta || {})[cslIdx] || {};
        const alts = cslDte === 'all' ? [] : (cslCfg.cells || [])
          .filter((c: any) => c.sym === cslIdx && String(c.dte) === cslDte && c.n >= 8)
          .sort((a: any, z: any) => (z.ratio ?? -9) - (a.ratio ?? -9)).slice(0, 6);
        return (
          <section id="csl-lab" style={{ ...card, marginTop: 14, borderColor: C.navy, scrollMarginTop: 70 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap', marginBottom: 4 }}>
              <span style={{ fontSize: 16, fontWeight: 700, color: C.ink }}>CSL Best-Config Lab</span>
              {chip(C.navySoft, C.navy, 'research/111')}
              {chip(C.amberSoft, C.amber, 'weekly regen · Fri 15:45 IST')}
              <span style={{ marginLeft: 'auto', fontSize: 11, color: C.muted }}>updated {cslCfg.generated_at}</span>
            </div>
            <div style={{ fontSize: 11, color: C.muted, marginBottom: 8 }}>
              Basis — NIFTY: <b>{(cslCfg.meta?.NIFTY?.days) ?? '—'} days</b> ({cslCfg.meta?.NIFTY?.from} → {cslCfg.meta?.NIFTY?.to}) @ {cslCfg.meta?.NIFTY?.lots} lots (qty {cslCfg.meta?.NIFTY?.qty}) ·
              SENSEX: <b>{(cslCfg.meta?.SENSEX?.days) ?? '—'} days</b> ({cslCfg.meta?.SENSEX?.from} → {cslCfg.meta?.SENSEX?.to}) @ {cslCfg.meta?.SENSEX?.lots} lots (qty {cslCfg.meta?.SENSEX?.qty}) ·
              3-sec dwell mechanic · ATM at entry moment
            </div>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 8 }}>
              {(['NIFTY', 'SENSEX'] as const).map((s2) => (
                <button key={s2} onClick={() => setCslIdx(s2)} style={btn(cslIdx === s2, C.ink)}>{s2}</button>))}
              <span style={{ width: 14 }} />
              {['all', ...dtes].map((k) => (
                <button key={k} onClick={() => setCslDte(k)} style={btn(cslDte === k, C.sec)}>
                  {k === 'all' ? 'All DTEs (book)' : `DTE${k} · ${DLBL[cslIdx][k] || ''}`}</button>))}
            </div>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', marginBottom: 8 }}>
                <thead><tr><th style={thL}>DTE</th><th style={thL}>Window</th><th style={thL}>SL</th>
                  <th style={thR}>Total</th><th style={thR}>Mean/day</th><th style={thR}>Win</th>
                  <th style={thR}>MaxDD</th><th style={thR}>Ratio</th><th style={thR}>n</th></tr></thead>
                <tbody>{dtes.map((k) => { const r = b[k]; return (
                  <tr key={k} style={{ background: cslDte === k ? C.navySoft : undefined }}>
                    <td style={tdL}>DTE{k} <span style={{ color: C.faint }}>({DLBL[cslIdx][k]})</span></td>
                    <td style={{ ...tdL, fontWeight: 700, color: C.ink }}>{r.entry} → {r.exit}</td>
                    <td style={tdL}>{r.sl === 'none' ? 'none' : r.sl + '%'}</td>
                    <td style={{ ...tdR, color: col(r.total), fontWeight: 700 }}>{inr(r.total)}</td>
                    <td style={{ ...tdR, color: col(r.mean) }}>{inr(r.mean)}</td>
                    <td style={tdR}>{r.win}%</td>
                    <td style={{ ...tdR, color: C.neg }}>{inr(r.maxdd)}</td>
                    <td style={{ ...tdR, fontWeight: 700 }}>{r.ratio}</td>
                    <td style={tdR}>{r.n}</td></tr>); })}
                </tbody>
              </table>
            </div>
            {second && (
              <div style={{ margin: '4px 0 8px' }}>
                <button onClick={() => setCsl2nd(!csl2nd)} style={btn(csl2nd, C.pos)}>
                  {csl2nd ? '✓ ' : '+ '}2nd slot: {second.entry} → {second.exit} SL{second.sl === 'none' ? 'none' : second.sl + '%'}
                  {'  '}({inr(second.total)} · win {second.win}% · DD {inr(second.maxdd)} · r{second.ratio})
                </button>
              </div>
            )}
            {pts.length >= 2 && <LineChart pts={pts} h={150}
              label={`cumulative P&L — ${cslIdx} ${cslDte === 'all' ? 'best-config book (all DTEs)' : 'DTE' + cslDte + (csl2nd && second ? ' best + 2nd slot' : ' best config')} · ${(cslCfg.meta || {})[cslIdx]?.lots} lots · shown total ${inr(shownTotal)} · maxDD ${inr(shownDD)}`} />}
            {ddPts.length >= 2 && <LineChart pts={ddPts} h={90}
              label={`drawdown — same selection · trough ${inr(shownDD)}`} />}
            {alts.length > 0 && (
              <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: 8 }}>
                <thead><tr><th style={thL}>Alternative configs (DTE{cslDte})</th><th style={thL}>SL</th>
                  <th style={thR}>Total</th><th style={thR}>Win</th><th style={thR}>MaxDD</th><th style={thR}>Ratio</th><th style={thR}>n</th></tr></thead>
                <tbody>{alts.map((c: any, i: number) => (
                  <tr key={i}><td style={tdL}>{c.entry} → {c.exit}</td>
                    <td style={tdL}>{c.sl === 'none' ? 'none' : c.sl + '%'}</td>
                    <td style={{ ...tdR, color: col(c.total) }}>{inr(c.total)}</td>
                    <td style={tdR}>{c.win}%</td>
                    <td style={{ ...tdR, color: C.neg }}>{inr(c.maxdd)}</td>
                    <td style={tdR}>{c.ratio}</td><td style={tdR}>{c.n}</td></tr>))}
                </tbody>
              </table>
            )}
            <div style={{ fontSize: 11, color: C.amber, marginTop: 8 }}>
              ⚠ Grid maxima on ~15-day cells (multiple-testing risk) — SL-level invariance is the robustness signal; validate via paper before live sizing. Refreshed automatically every Friday 15:45 IST as recorded days accumulate.
            </div>
          </section>
        );
      })()}

      {condor && (
        <section id="condor" style={{ ...card, marginTop: 14, borderColor: C.amber, scrollMarginTop: 70 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6, flexWrap: 'wrap' }}>
            <span style={{ fontSize: 16, fontWeight: 700, color: C.ink }}>Wed&#8594;Fri Iron Condor</span>
            {chip(C.amberSoft, C.amber, 'PAPER · 2 lots')}
            <a href="/app/backtest/fardte-rescue" style={{ textDecoration: 'none' }}>
              {chip(C.navySoft, C.navy, 'research/80 ↗')}
            </a>
            {condor.closed_trades > 0 && (
              <span style={{ marginLeft: 'auto', fontSize: 12, color: C.muted }}>
                closed {condor.closed_trades} · <b style={{ color: col(condor.closed_total_pnl) }}>{inr(condor.closed_total_pnl)}</b>
                {condor.win_rate != null ? ` · ${condor.win_rate}% win` : ''}
              </span>
            )}
          </div>
          <div style={{ fontSize: 11.5, color: C.muted, marginBottom: 10 }}>{condor.subtitle}</div>

          {condor.open ? (
            <div>
              <div style={{ display: 'flex', gap: 22, flexWrap: 'wrap', alignItems: 'baseline', marginBottom: 10 }}>
                <div>
                  <div style={{ fontSize: 10.5, color: C.muted, textTransform: 'uppercase' }}>Day P&amp;L</div>
                  <div style={{ fontSize: 22, fontWeight: 800, color: col(condor.open.day_pnl || 0) }}>{inr(condor.open.day_pnl || 0)}</div>
                </div>
                <div style={{ fontSize: 12, color: C.sec }}>
                  entered <b>{condor.open.entry_day}</b> (DTE {condor.open.dte_at_entry}) · spot {condor.open.spot_at_entry} ·
                  credit <b>{condor.open.credit}</b> pts ({inr(condor.open.credit_inr)}) ·
                  max loss <b style={{ color: C.neg }}>{inr(-condor.open.max_loss_inr)}</b> · exit Fri close
                </div>
              </div>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead><tr>
                  <th style={{ ...cth2, textAlign: 'left' }}>LEG</th><th style={cth2}>STRIKE</th><th style={cth2}>QTY</th>
                  <th style={cth2}>ENTRY</th><th style={cth2}>LTP</th><th style={cth2}>P&amp;L</th>
                </tr></thead>
                <tbody>{condor.open.legs.map((l: any, i: number) => (
                  <tr key={i}>
                    <td style={{ ...ctd2, textAlign: 'left', fontWeight: 700, color: l.side === 'SELL' ? C.neg : C.pos }}>{l.side} {l.type}</td>
                    <td style={ctd2}>{l.strike}</td>
                    <td style={ctd2}>{l.qty}</td>
                    <td style={ctd2}>{l.entry}</td>
                    <td style={ctd2}>{l.ltp}</td>
                    <td style={{ ...ctd2, fontWeight: 700, color: col(l.pnl) }}>{inr(l.pnl)}</td>
                  </tr>))}</tbody>
              </table>
            </div>
          ) : (
            <div style={{ fontSize: 12, color: C.faint, padding: '4px 0' }}>
              Flat — no open position. Enters automatically on the next Wednesday close.
            </div>
          )}

          {condor.history && condor.history.length > 0 && (
            <details style={{ marginTop: 10 }}>
              <summary style={{ cursor: 'pointer', fontSize: 11.5, fontWeight: 600, color: C.navy, listStyle: 'none' }}>
                &#9656; Completed cycles ({condor.history.length})
              </summary>
              <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: 6 }}>
                <thead><tr>
                  <th style={{ ...cth2, textAlign: 'left' }}>ENTERED</th><th style={{ ...cth2, textAlign: 'left' }}>EXITED</th>
                  <th style={cth2}>CREDIT</th><th style={cth2}>EXIT VAL</th><th style={cth2}>P&amp;L</th>
                </tr></thead>
                <tbody>{condor.history.map((h: any, i: number) => (
                  <tr key={i}>
                    <td style={{ ...ctd2, textAlign: 'left' }}>{h.entry_day}</td>
                    <td style={{ ...ctd2, textAlign: 'left' }}>{h.exit_day}</td>
                    <td style={ctd2}>{h.credit}</td>
                    <td style={ctd2}>{h.exit_value}</td>
                    <td style={{ ...ctd2, fontWeight: 700, color: col(h.pnl) }}>{inr(h.pnl)}</td>
                  </tr>))}</tbody>
              </table>
            </details>
          )}
          <div style={{ fontSize: 10.5, color: C.faint, marginTop: 8 }}>
            SIGNAL, not yet a proven strategy — paper-forward only while the portfolio-correlation and real-chain
            checks are open. Uses the capital NAS-OPT leaves idle Wed&#8211;Fri; flat before Monday.
          </div>
        </section>
      )}


      {/* ===== TODAY · LIVE ===== */}
      {live && (
        <section id="live-box" style={{ ...card, marginTop: 14, borderColor: C.navy, scrollMarginTop: 70 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8, flexWrap: 'wrap' }}>
            <span style={{ fontSize: 16, fontWeight: 700, color: C.ink }}>Today · Live</span>
            {chip('#E7F2EE', C.pos, 'PAPER · 10 lots')}
            {liveTs && (Date.now() / 1000 - liveTs) < 12 && (
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5, fontSize: 11, fontWeight: 700, color: C.pos }}>
                <span style={{ width: 7, height: 7, borderRadius: '50%', background: C.pos, display: 'inline-block', animation: 'pulse 1.4s ease-in-out infinite' }} />
                LIVE
              </span>
            )}
            <span style={{ marginLeft: 'auto', fontSize: 11, color: C.muted }}>
              {liveTs ? `live-quote ${new Date(liveTs * 1000).toLocaleTimeString('en-IN', { hour12: false })}` : `updated ${String(live.updated_at || '').slice(11, 19)}`} · {live.day}
            </span>
          </div>
          <div style={{ display: 'flex', gap: 18, flexWrap: 'wrap' }}>
            {[['V1 · intraday one-and-done (naked straddle)', live.v1], ['V2 · positional bi-weekly (naked straddle · legacy)', live.v2]].map(([title, d]: any) => {
              const J = d.journey;
              const useJourney = Array.isArray(J) && J.length >= 2;
              const isV1 = String(title).startsWith('V1');
              // V1 is one-and-done: build its completed log from the daily history (strike/final/stop/exit-time per day).
              const v1Comp = isV1 && daily
                ? daily.days.slice().reverse().map((dd) => {
                    const x = daily.per_day[dd];
                    return { n: daily.days.indexOf(dd) + 1, entry_day: dd,
                      exit_day: x.stopped && x.exit ? x.exit.time : '15:15', strike: x.strike,
                      exit_pnl: x.final, reason: x.stopped ? 'stop hit' : 'held to close' };
                  })
                : null;
              const dowOf = (ds: string) => { const a = String(ds || '').split('-').map(Number); return a.length === 3 && !a.some(Number.isNaN) ? ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'][new Date(a[0], a[1] - 1, a[2]).getDay()] : ''; };
              const comp = isV1 ? v1Comp : (Array.isArray(d.completed) ? d.completed : null);
              const cth: React.CSSProperties = { fontSize: 9.5, color: C.muted, fontWeight: 600, textAlign: 'right', padding: '2px 6px', textTransform: 'uppercase', borderBottom: `1px solid ${C.hairSoft}` };
              const ctd: React.CSSProperties = { fontSize: 11, color: C.ink, textAlign: 'right', padding: '3px 6px', borderTop: `1px solid ${C.hairSoft}`, fontVariantNumeric: 'tabular-nums' };
              return (
              <div key={title} style={{ flex: 1, minWidth: 300, border: `1px solid ${C.hair}`, borderRadius: 8, padding: 12 }}>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
                  <span style={{ fontWeight: 700, color: C.ink }}>{title}</span>
                  <span style={{ fontSize: 22, fontWeight: 800, marginLeft: 'auto', color: d.status === 'idle' || d.status === 'flat' ? C.muted : col(d.pnl_now) }}>
                    {d.status === 'idle' || d.status === 'flat' ? d.status : inr(d.pnl_now)}
                  </span>
                </div>
                <div style={{ fontSize: 11, color: C.muted, margin: '2px 0 6px' }}>{d.detail}</div>
                {useJourney && d.entry_day && (
                  <div style={{ marginBottom: 6 }}>
                    {chip(C.navySoft, C.navy, `entered ${d.entry_day}`)}{' '}
                    <span style={{ fontSize: 11, color: C.muted }}>positional carry · {J.length} day{J.length === 1 ? '' : 's'} of marks</span>
                  </div>
                )}
                <LegsTable legs={d.legs} total={d.pnl_now} />
                {useJourney
                  ? <LineChart pts={J} h={140}
                      label={`trade journey · continuous (intraday across days) · entered ${d.entry_day} → now · low ${inr(d.low || 0)} · high ${inr(d.high || 0)}`} />
                  : (d.series && d.series.length >= 2
                      ? <LineChart pts={d.series} h={120}
                          marker={d.exit && d.exit.time ? { time: d.exit.time, pnl: d.exit.pnl, text: 'exit' } : null}
                          label={`intraday running P&L · low ${inr(d.low || 0)} · high ${inr(d.high || 0)}${d.exit && d.exit.time ? ` · stop-exit ${d.exit.time} @ ${inr(d.exit.pnl)}` : ''}`} />
                      : <div style={{ fontSize: 12, color: C.faint, padding: 8 }}>{d.status === 'idle' ? 'no trade today (not 0/1-DTE)' : '—'}</div>)}
                {comp && (
                  <details style={{ marginTop: 8 }}>
                    <summary style={{ cursor: 'pointer', fontSize: 11.5, fontWeight: 600, color: C.navy, listStyle: 'none' }}>
                      ▸ Completed trades ({comp.length}) · total{' '}
                      <span style={{ color: col(comp.reduce((a: number, t: any) => a + (t.exit_pnl || 0), 0)) }}>{inr(comp.reduce((a: number, t: any) => a + (t.exit_pnl || 0), 0))}</span>
                    </summary>
                    {comp.length === 0
                      ? <div style={{ fontSize: 11, color: C.faint, padding: '6px 0' }}>{isV1 ? 'No recorded days yet.' : `None yet — the current open position is the first (entered ${d.entry_day}).`}</div>
                      : <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: 6 }}>
                          <thead><tr>
                            <th style={{ ...cth, textAlign: 'left' }}>#</th><th style={{ ...cth, textAlign: 'left' }}>Entry</th>
                            <th style={cth}>Day</th>
                            <th style={{ ...cth, textAlign: 'left' }}>Exit</th><th style={cth}>Strike</th>
                            <th style={cth}>Exit P&amp;L</th><th style={{ ...cth, textAlign: 'left' }}>Reason</th>
                          </tr></thead>
                          <tbody>{comp.map((t: any, i: number) => (
                            <tr key={i}>
                              <td style={{ ...ctd, textAlign: 'left' }}>{t.n}</td>
                              <td style={{ ...ctd, textAlign: 'left' }}>{t.entry_day}</td>
                              <td style={{ ...ctd, color: C.muted }}>{dowOf(t.entry_day)}</td>
                              <td style={{ ...ctd, textAlign: 'left' }}>{t.exit_day}</td>
                              <td style={ctd}>{t.strike}</td>
                              <td style={{ ...ctd, fontWeight: 700, color: col(t.exit_pnl) }}>{inr(t.exit_pnl)}</td>
                              <td style={{ ...ctd, textAlign: 'left', color: C.muted }}>{t.reason}</td>
                            </tr>))}</tbody>
                        </table>}
                  </details>
                )}
              </div>
              );
            })}
          </div>

          {sl30 && sl30.stats && (() => {
            const s = sl30.stats;
            const today = sl30.days[sl30.days.length - 1];
            const tp = sl30.per_day[today];
            const comp = sl30.trades.slice().reverse();
            return (
              <div id="sl30-card" style={{ marginTop: 14, border: `1px solid ${C.hair}`, borderRadius: 8, padding: 12, scrollMarginTop: 70 }}>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, flexWrap: 'wrap' }}>
                  <span style={{ fontWeight: 700, color: C.ink }}>V1 + 30% combined-premium SL</span>
                  {chip(C.amberSoft, C.amber, 'BACKTEST · recorded chain')}
                  <a href="/app/options-study" style={{ textDecoration: 'none' }} title="Peak+% / decay analysis behind this stop">{chip(C.navySoft, C.navy, 'Opt-Study report ↗')}</a>
                  <span style={{ marginLeft: 'auto', fontSize: 22, fontWeight: 800, color: col(s.total) }}>{inr(s.total)}</span>
                </div>
                <div style={{ fontSize: 11, color: C.muted, margin: '2px 0 8px' }}>sell ATM ~09:20 · exit once if combined premium rises ≥30% (a 30%-of-credit loss), else hold to close · {s.n} recorded days · 10 lots</div>
                <div style={{ display: 'flex', gap: 18, flexWrap: 'wrap', marginBottom: 8 }}>
                  {[['Mean/day', inr(s.mean), col(s.mean)], ['Win', s.win + '%', C.ink], ['SL hit', s.sl_hit_pct + '%', C.ink], ['Max DD', inr(s.maxdd), C.neg], ['Worst day', inr(s.worst), C.neg]].map(([k, v, c]: any) => (
                    <div key={k}><div style={{ fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.04em', color: C.faint }}>{k}</div><div style={{ fontSize: 14, fontWeight: 800, color: c }}>{v}</div></div>
                  ))}
                </div>
                {tp && tp.series && tp.series.length >= 2 && (
                  <LineChart pts={tp.series} h={110}
                    marker={tp.exit && tp.exit.time ? { time: tp.exit.time, pnl: tp.exit.pnl, text: 'SL' } : null}
                    label={`today (${today}) intraday · ${tp.exit && tp.exit.time ? 'SL-exit ' + tp.exit.time : 'held to close'}`} />
                )}
                <LineChart pts={sl30.book_curve} h={140} label={`cumulative paper P&L · ${s.n} days · 10 lots (qty ${(sl30.lot || 65) * (sl30.lots || 10)}) · recorded chain`} />
                <div style={{ fontSize: 10.5, color: C.faint, textTransform: 'uppercase', letterSpacing: '0.04em', margin: '6px 0 2px' }}>By DTE · {sl30.lots || 10} lots (NIFTY · qty {(sl30.lot || 65) * (sl30.lots || 10)})</div>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead><tr><th style={{ ...ecth, textAlign: 'left' }}>DTE</th><th style={ecth}>Days</th><th style={ecth}>Total</th><th style={ecth}>Mean/day</th><th style={ecth}>Win</th><th style={ecth}>SL hit</th><th style={ecth}>Max DD</th></tr></thead>
                  <tbody>{Object.keys(s.by_dte).map((k) => { const r = s.by_dte[k]; return (
                    <tr key={k}><td style={{ ...ectd, textAlign: 'left' }}>DTE{k}</td><td style={ectd}>{r.n}</td>
                      <td style={{ ...ectd, color: col(r.total), fontWeight: 700 }}>{inr(r.total)}</td>
                      <td style={{ ...ectd, color: col(r.mean) }}>{inr(r.mean)}</td>
                      <td style={ectd}>{r.win}%</td><td style={ectd}>{r.sl_hit_pct}%</td>
                      <td style={{ ...ectd, color: C.neg }}>{inr(r.maxdd)}</td></tr>
                  ); })}</tbody>
                </table>
                <details style={{ marginTop: 8 }}>
                  <summary style={{ cursor: 'pointer', fontSize: 11.5, fontWeight: 600, color: C.navy, listStyle: 'none' }}>▸ Completed days ({comp.length})</summary>
                  <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: 6 }}>
                    <thead><tr><th style={{ ...ecth, textAlign: 'left' }}>Day</th><th style={ecth}>DTE</th><th style={ecth}>Peak+%</th><th style={{ ...ecth, textAlign: 'left' }}>Exit</th><th style={ecth}>P&amp;L</th></tr></thead>
                    <tbody>{comp.map((t: any, i: number) => (
                      <tr key={i}><td style={{ ...ectd, textAlign: 'left' }}>{t.day}</td><td style={ectd}>DTE{t.dte}</td>
                        <td style={{ ...ectd, color: t.peak_pct >= 30 ? C.neg : C.faint }}>+{t.peak_pct}%</td>
                        <td style={{ ...ectd, textAlign: 'left', color: t.stopped ? C.neg : C.muted }}>{t.stopped ? t.exit_time + ' · SL' : 'held to close'}</td>
                        <td style={{ ...ectd, fontWeight: 700, color: col(t.final) }}>{inr(t.final)}</td></tr>
                    ))}</tbody>
                  </table>
                </details>
              </div>
            );
          })()}

          <div style={{ fontSize: 11, color: C.muted, marginTop: 8 }}>Live paper · live-quote P&amp;L ticks ~every 3s during market (positions/chart refresh every minute) · recorded daily. Backtest history below.</div>
          <RulesBlock />
        </section>
      )}

      {/* ===== V2 ENGINE · live paper executor ===== */}
      {v2eng && (
        <section id="v2-engine" style={{ ...card, marginTop: 14, borderColor: C.pos, scrollMarginTop: 70 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8, flexWrap: 'wrap' }}>
            <span style={{ fontSize: 16, fontWeight: 700, color: C.ink }}>V2 Engine · iron-fly executor</span>
            {v2eng.mode === 'live'
              ? chip('#FBEEEE', C.neg, v2eng.armed ? 'LIVE · armed' : 'LIVE · disarmed')
              : chip('#E7F2EE', C.pos, 'PAPER · combo filter')}
            {v2engTs && (Date.now() / 1000 - v2engTs) < 12 && (
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5, fontSize: 11, fontWeight: 700, color: C.pos }}>
                <span style={{ width: 7, height: 7, borderRadius: '50%', background: C.pos, display: 'inline-block', animation: 'pulse 1.4s ease-in-out infinite' }} />
                LIVE
              </span>
            )}
            <span style={{ marginLeft: 'auto', fontSize: 11, color: C.muted }}>
              {v2engTs ? `live-quote ${new Date(v2engTs * 1000).toLocaleTimeString('en-IN', { hour12: false })} · ` : ''}closed {v2eng.closed_trades} · <b style={{ color: col(v2eng.closed_total_pnl) }}>{inr(v2eng.closed_total_pnl)}</b>
            </span>
          </div>

          {/* ---- PAPER / LIVE control bar ---- */}
          {v2eng.mode != null && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', marginBottom: 10,
                          padding: '8px 10px', borderRadius: 8,
                          background: v2eng.mode === 'live' ? '#FBEEEE' : '#F3F7F5',
                          border: `1px solid ${v2eng.mode === 'live' ? '#E3B7B7' : C.hairSoft}` }}>
              <span style={{ fontSize: 11, fontWeight: 700, color: C.muted, letterSpacing: 0.4 }}>MODE</span>
              <div style={{ display: 'inline-flex', border: `1px solid ${C.hair}`, borderRadius: 6, overflow: 'hidden' }}>
                {['paper', 'live'].map((mm) => {
                  const disabled = v2busy || !!v2eng.open || (mm === 'live' && v2eng.live_enabled === false);
                  return (
                    <button key={mm} disabled={disabled} onClick={() => v2SetMode(mm)} title={v2eng.open ? 'square off before switching mode' : ''}
                      style={{ cursor: disabled ? 'not-allowed' : 'pointer', border: 'none', padding: '5px 14px',
                               fontSize: 12, fontWeight: 700, opacity: disabled && v2eng.mode !== mm ? 0.5 : 1,
                               background: v2eng.mode === mm ? (mm === 'live' ? C.neg : C.pos) : C.surface,
                               color: v2eng.mode === mm ? '#fff' : C.muted }}>
                      {mm.toUpperCase()}
                    </button>
                  );
                })}
              </div>
              {v2eng.mode === 'live' && (
                <span style={{ fontSize: 11, fontWeight: 700, color: v2eng.armed ? C.neg : C.muted }}>
                  {v2eng.armed ? '● ARMED · rolls automated' : '○ disarmed'}
                </span>
              )}
              <div style={{ marginLeft: 'auto', display: 'flex', gap: 8, alignItems: 'center' }}>
                {v2eng.deployable && (
                  <button disabled={v2busy} onClick={() => v2Deploy(false)}
                    style={{ cursor: v2busy ? 'wait' : 'pointer', border: 'none', borderRadius: 6, padding: '6px 16px',
                             fontSize: 12.5, fontWeight: 800, background: C.neg, color: '#fff', boxShadow: '0 1px 3px rgba(0,0,0,.15)' }}>
                    ▶ Deploy live trade
                  </button>
                )}
                {v2eng.open && (
                  <button disabled={v2busy} onClick={v2Kill}
                    style={{ cursor: v2busy ? 'wait' : 'pointer', border: `1px solid ${C.neg}`, borderRadius: 6,
                             padding: '5px 12px', fontSize: 12, fontWeight: 700, background: '#fff', color: C.neg }}>
                    ✕ Kill-switch
                  </button>
                )}
              </div>
              {v2eng.deployable && v2prev?.ok && (
                <div style={{ flexBasis: '100%', fontSize: 11, color: C.sec, marginTop: 2 }}>
                  Will place: <b>{v2prev.legs.map((l: any) => `${l.side} ${l.strike}${l.instrument_type}`).join(' · ')}</b>
                  {' '}· net credit <b>{v2prev.net_credit}</b> · margin <b>₹{(v2prev.margin_need || 0).toLocaleString('en-IN')}</b>
                  {' '}(avail ₹{(v2prev.margin_avail || 0).toLocaleString('en-IN')}{v2prev.margin_ok ? '' : ' — INSUFFICIENT'})
                  {' '}· gate <b style={{ color: v2prev.gate === 'PASS' ? C.pos : C.neg }}>{v2prev.gate}</b>
                  {v2prev.gate !== 'PASS' && (
                    <button disabled={v2busy} onClick={() => v2Deploy(true)}
                      style={{ marginLeft: 8, cursor: 'pointer', border: `1px solid ${C.neg}`, borderRadius: 5,
                               padding: '2px 8px', fontSize: 10.5, fontWeight: 700, background: '#fff', color: C.neg }}>
                      force-deploy (ignore filter)
                    </button>
                  )}
                </div>
              )}
              {v2msg && <div style={{ flexBasis: '100%', fontSize: 11, fontWeight: 600, color: v2msg.startsWith('⚠') ? C.neg : C.pos }}>{v2msg}</div>}
              {v2eng.mode === 'live' && !v2eng.deployable && !v2eng.open && !v2eng.armed && (
                <div style={{ flexBasis: '100%', fontSize: 10.5, color: C.muted }}>Live, disarmed, flat — click Deploy to place the first trade and start automation.</div>
              )}
            </div>
          )}
          {v2eng.open ? (
            <div style={{ border: `1px solid ${C.hair}`, borderRadius: 8, padding: 12 }}>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
                <span style={{ fontWeight: 700, color: C.ink }}>Open fly · exp {v2eng.open.expiry} ({v2eng.open.dte_entry}d at entry)</span>
                <span style={{ fontSize: 22, fontWeight: 800, marginLeft: 'auto', color: col(v2eng.open.pnl_now || 0) }}>{inr(v2eng.open.pnl_now || 0)}</span>
              </div>
              <div style={{ fontSize: 11.5, color: C.sec, margin: '2px 0 6px' }}>
                <b style={{ color: C.ink }}>taken {v2eng.open.day} · {v2eng.open.entry_time}</b> · spot {Math.round(v2eng.open.entry_spot)} · VIX {v2eng.open.entry_vix} · net credit {Number(v2eng.open.net_entry).toFixed(1)} · exp {v2eng.open.expiry}
              </div>
              {v2eng.open.gap_day ? (
                <div style={{ fontSize: 11, color: C.sec, margin: '0 0 8px', padding: '5px 8px', background: '#FBEEEE', border: `1px solid #E3B7B7`, borderRadius: 6 }}>
                  <b style={{ color: C.neg }}>GAP DAY</b> — opened beyond ±2%, so the 2% stop is suspended. <b>No action first 5 min</b>; then exit on a 1-min close <b style={{ color: C.neg }}>&gt; {v2eng.open.or_high?.toLocaleString('en-IN')}</b> or <b style={{ color: C.neg }}>&lt; {v2eng.open.or_low?.toLocaleString('en-IN')}</b> (the 09:15–09:20 opening range).
                  {v2spot ? <> · spot now <b style={{ color: C.ink }}>{Math.round(v2spot).toLocaleString('en-IN')}</b></> : null}
                </div>
              ) : (
                <div style={{ fontSize: 11, color: C.sec, margin: '0 0 8px', padding: '5px 8px', background: C.amberSoft, border: `1px solid ${C.hairSoft}`, borderRadius: 6 }}>
                  <b>2% move-stop band:</b> exit if NIFTY ≤ <b style={{ color: C.neg }}>{v2eng.open.stop_dn?.toLocaleString('en-IN')}</b> or ≥ <b style={{ color: C.neg }}>{v2eng.open.stop_up?.toLocaleString('en-IN')}</b>
                  <span style={{ color: C.muted }}> (±2% from {Math.round(v2eng.open.entry_spot).toLocaleString('en-IN')})</span>
                  {v2spot ? <> · spot now <b style={{ color: C.ink }}>{Math.round(v2spot).toLocaleString('en-IN')}</b></> : null}
                  <span style={{ color: C.muted }}> · checked on 1-min candle close · on a &gt;2% gap-open this switches to the opening-range stop</span>
                </div>
              )}
              <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
                <thead><tr style={{ color: C.muted, textAlign: 'left' }}>
                  <th style={{ padding: '2px 0' }}>Leg</th><th>Strike</th><th>In</th><th>LTP</th><th style={{ textAlign: 'right' }}>P&amp;L</th></tr></thead>
                <tbody>
                  {v2eng.open.legs.map((l: any, i: number) => (
                    <tr key={i} style={{ borderTop: `1px solid ${C.hairSoft}` }}>
                      <td style={{ padding: '3px 0', color: l.side === 'SELL' ? C.neg : C.pos, fontWeight: 600 }}>{l.side} {l.instrument_type}</td>
                      <td>{l.strike}</td><td>{Number(l.entry).toFixed(1)}</td><td>{Number(l.ltp).toFixed(1)}</td>
                      <td style={{ textAlign: 'right', color: col(l.pnl || 0) }}>{l.pnl != null ? inr(l.pnl) : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {Array.isArray(v2eng.open.series) && v2eng.open.series.length >= 2 && (
                <div style={{ marginTop: 8 }}>
                  <LineChart pts={v2eng.open.series} h={120}
                    label={`intraday running P&L · low ${inr(v2eng.open.low || 0)} · high ${inr(v2eng.open.high || 0)}`} />
                </div>
              )}
              <div style={{ marginTop: 10 }}>
                <PayoffChart legs={v2eng.open.legs} qty={650} spot={v2spot || v2eng.open.entry_spot}
                  entrySpot={v2eng.open.entry_spot} stopDn={v2eng.open.stop_dn} stopUp={v2eng.open.stop_up}
                  dte={Math.max((new Date(v2eng.open.expiry + 'T15:30:00').getTime() - Date.now()) / (365 * 86400000), 0)}
                  iv={(v2eng.open.entry_vix || 13) / 100} currentPnl={v2eng.open.pnl_now} />
              </div>
            </div>
          ) : (
            <div style={{ fontSize: 13, color: C.muted, padding: '6px 2px' }}>Flat — waiting for a qualifying entry (VIX≥13 + filter not triggered).</div>
          )}
          <div style={{ fontSize: 11, color: C.muted, marginTop: 8 }}>
            Filter: {v2eng.config?.filter}. Shadow-skips logged: <b>{v2eng.shadow_skips?.length || 0}</b>
            {v2eng.shadow_skips?.length ? ` (latest ${v2eng.shadow_skips[0].day} — ${v2eng.shadow_skips[0].reasons})` : ''}. Margin {v2eng.config?.margin_est}.
          </div>
          {bo && (
            <div style={{ fontSize: 11, color: C.muted, marginTop: 4 }}>
              Inside-week breakout sleeve: <b>{bo.open ? `OPEN (${bo.open.legs?.length}-leg)` : 'flat'}</b> · closed {bo.closed_trades} ({inr(bo.closed_total_pnl)}) · UP→call debit · DOWN→broken-wing fly · paper-only.
            </div>
          )}
          {Array.isArray(v2eng.closed) && v2eng.closed.length > 0 && (
            <details style={{ marginTop: 10 }}>
              <summary style={{ cursor: 'pointer', fontSize: 12, fontWeight: 600, color: C.navy, listStyle: 'none' }}>▸ Completed trades ({v2eng.closed_trades})</summary>
              <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: 6 }}>
                <thead><tr>
                  <th style={ecth}>#</th><th style={{ ...ecth, textAlign: 'left' }}>Entry (date · time)</th>
                  <th style={ecth} title="Day of week of entry">DOW</th>
                  <th style={ecth} title="Calendar days to expiry at entry">DTE</th>
                  <th style={ecth} title="India VIX at entry (gate = 13)">VIX</th>
                  <th style={ecth} title="Weekly CPR width % (research/67) — narrow = trend-prone">wCPR</th>
                  <th style={ecth} title="Prior-day CPR width % (research/67) — narrow = calm next day">dCPR</th>
                  <th style={{ ...ecth, textAlign: 'left' }}>Exit (date · time)</th>
                  <th style={{ ...ecth, textAlign: 'left' }}>Reason</th><th style={ecth}>P&amp;L</th>
                </tr></thead>
                <tbody>{v2eng.closed.map((t: any, i: number) => (
                  <tr key={i}>
                    <td style={ectd}>{t.id}</td>
                    <td style={{ ...ectd, textAlign: 'left' }}>{t.day} · {t.entry_time}</td>
                    <td style={{ ...ectd, color: C.muted }}>{t.dow || '—'}</td>
                    <td style={ectd}>{t.dte_entry != null ? `${t.dte_entry}d` : '—'}</td>
                    <td style={{ ...ectd, color: t.entry_vix != null && t.entry_vix < 13 ? C.neg : C.sec }}>{t.entry_vix != null ? Number(t.entry_vix).toFixed(2) : '—'}</td>
                    <td style={{ ...ectd, color: C.sec }}>{t.cpr_w != null ? `${Number(t.cpr_w).toFixed(3)}%` : '—'}</td>
                    <td style={{ ...ectd, color: C.sec }}>{t.cpr_d != null ? `${Number(t.cpr_d).toFixed(3)}%` : '—'}</td>
                    <td style={{ ...ectd, textAlign: 'left' }}>{(t.exit_day || '—')} · {(t.exit_time || '—')}</td>
                    <td style={{ ...ectd, textAlign: 'left', color: C.muted }}>{t.exit_reason}</td>
                    <td style={{ ...ectd, fontWeight: 700, color: col(t.pnl) }}>{inr(t.pnl)}</td>
                  </tr>))}</tbody>
              </table>
              <div style={{ fontSize: 11.5, color: C.sec, marginTop: 8, lineHeight: 1.7 }}>
                <b>Overall (realized):</b>{' '}
                <b style={{ color: col(v2eng.closed_total_pnl) }}>{inr(v2eng.closed_total_pnl)}</b>
                {' '}over <b>{v2eng.closed_trades}</b> closed trades
                {v2eng.closed_trades > 0 && <> · mean/trade <b style={{ color: col(v2eng.closed_total_pnl / v2eng.closed_trades) }}>{inr(Math.round(v2eng.closed_total_pnl / v2eng.closed_trades))}</b></>}
                {Array.isArray(v2eng.closed) && v2eng.closed.length > 0 && <> · win rate <b>{Math.round(v2eng.closed.filter((x: any) => x.pnl > 0).length / v2eng.closed.length * 100)}%</b>{v2eng.closed_trades > v2eng.closed.length ? ` (last ${v2eng.closed.length})` : ''}</>}
                {v2eng.open && <> · <span style={{ color: C.muted }}>open position not included</span></>}
                <div style={{ color: C.faint, marginTop: 3 }}>
                  wCPR/dCPR = CPR width (research/67): weekly fixed Mon–Fri; daily re-draws. Blank where the trade predates the 160-day daily-bar window.
                </div>
              </div>
            </details>
          )}
          <details style={{ marginTop: 10 }}>
            <summary style={{ cursor: 'pointer', fontSize: 12, fontWeight: 600, color: C.navy }}>System rules — V2 engine (click to expand)</summary>
            <div style={{ fontSize: 11.5, color: C.sec, marginTop: 6, lineHeight: 1.7 }}>
              <div><b>Instrument:</b> short 2nd-nearest weekly NIFTY ATM straddle + <b>2%-of-ATM wings</b> (snapped to the 50-pt strike grid — ≈±450 at NIFTY 23.4k today; <b>not a fixed 500</b>) = short iron fly, overnight carry.</div>
              <div><b>Entry / when it starts:</b> 09:20 on any trading day the book is <b>flat</b> — sells the <b>2nd-nearest weekly</b> (must be <b>≥ 4 calendar days to expiry</b>), gated by <b>VIX ≥ 13</b> + the combo skip-filter. <b>No fixed weekday</b>: entries follow the roll cycle — it re-arms the morning after the prior fly rolls (DTE ≤ 1), so in practice it re-enters ~weekly. The scheduler checks at 09:20 Mon–Fri.</div>
              <div><b>Skip-filter (live):</b> skip entry when prior-day CPR width &lt; 0.10% OR last week was an inside week — every skip is shadow-logged for forward validation.</div>
              <div><b>Exits:</b> 2% underlying move-stop · +40% credit profit-target · roll at DTE ≤ 1, then re-enter.</div>
              <div><b>Gap-open handling:</b> if the session <b>opens outside the ±2% band</b>, the engine does <b>not</b> dump at the gap print. It takes no stop action for the first 5 min, then uses the <b>09:15–09:20 opening range</b> as the stop — exit on a 1-min close beyond its high or low (either side). Because that exit can only happen after 09:20 (while still holding), there's <b>no same-day re-entry</b>. Normal-day stop is unchanged. (Discretionary overlay — not in the AlgoTest backtest; paper-validating live.)</div>
              <div><b>Stop band:</b> fixed at entry = entry-spot ±2% (the exact NIFTY levels are shown on the open position). A move-stop, not a premium-stop — it triggers on the underlying, not on the option P&L.</div>
              <div><b>When it's checked (AlgoTest-synced):</b> evaluated on the <b>close of each 1-min NIFTY candle</b> — the same resolution the backtest used (all 110 backtested SL exits land on :00 minute boundaries; AlgoTest is a 1-min candle-close engine, not tick/intra-candle). On the first candle whose close is ≥2% from entry, it exits at that close. The on-screen P&L still ticks every ~3s for display, but the exit decision is candle-close each minute. +40% PT and the DTE≤1 roll run on the same cycle.</div>
              <div><b>Sizing:</b> 10 lots = qty 650 · blocked margin ≈ ₹7.0L (₹70k/lot, Kite SPAN — floats with VIX; hedged by the long wings, ~3× cheaper than the ₹21L naked straddle).</div>
              <div><b>Live control:</b> the engine runs PAPER by default. Toggle PAPER→LIVE (only when flat), then <b>Deploy</b> places the first real trade (wings bought first, then the straddle sold; margin pre-checked; auto-rollback if any leg fails) and arms automation. From then rolls/re-entries are automatic; <b>Kill-switch</b> squares off and disarms.</div>
              <div><b>Costs:</b> P&amp;L net of ₹20/order + 0.25% slippage — same basis as the backtest.</div>
              <div><b>Breakout sleeve (paper-only):</b> on inside-week-skip weeks, first daily close beyond the inside week's prior-week H/L → UP: call debit spread (runner edge) · DOWN: broken-wing fly skewed down.</div>
              <div style={{ color: C.muted, marginTop: 4 }}>Paper engine · live-quote P&amp;L ticks ~every 3s during market · 1-min position marks · backtest spec wired into the live engine.</div>
            </div>
          </details>
        </section>
      )}

      {/* ===== JADE ENGINE · directional shadow logger ===== */}
      {v2eng && v2eng.jade_today && (() => {
        const jt = v2eng.jade_today;
        const log = Array.isArray(v2eng.jade_log) ? v2eng.jade_log : [];
        const dirColor = jt.direction === 'BULL' ? C.pos : jt.direction === 'BEAR' ? C.neg : C.muted;
        const dirBg = jt.direction === 'BULL' ? '#E7F2EE' : jt.direction === 'BEAR' ? '#FBEEEE' : C.hairSoft;
        const dirLabel = jt.direction === 'BULL' ? 'BULL · jade lizard' : jt.direction === 'BEAR' ? 'BEAR · reverse-jade' : 'no signal today';
        const pct = (n: number) => `${n >= 0 ? '+' : '−'}${Math.abs(n)}%`;
        const jth: React.CSSProperties = { fontSize: 9.5, color: C.muted, fontWeight: 600, textAlign: 'right', padding: '2px 6px', textTransform: 'uppercase', borderBottom: `1px solid ${C.hairSoft}` };
        const jtd: React.CSSProperties = { fontSize: 11, color: C.ink, textAlign: 'right', padding: '3px 6px', borderTop: `1px solid ${C.hairSoft}`, fontVariantNumeric: 'tabular-nums' };
        return (
        <section style={{ ...card, marginTop: 14, borderColor: C.navy }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8, flexWrap: 'wrap' }}>
            <span style={{ fontSize: 16, fontWeight: 700, color: C.ink }}>Jade Engine · directional shadow logger</span>
            {chip(C.navySoft, C.navy, 'SHADOW · no orders')}
            <span style={{ marginLeft: 'auto', fontSize: 11, color: C.muted }}>{log.length} signal day{log.length === 1 ? '' : 's'} logged</span>
          </div>
          <div style={{ fontSize: 11.5, color: C.muted, marginBottom: 10, lineHeight: 1.6 }}>
            Day-1-confirmed directional book (research/64): the <b>prior daily close</b> sets the side — up &gt;+0.5% → <b style={{ color: C.pos }}>bull jade lizard</b>, down &lt;−0.5% → <b style={{ color: C.neg }}>bear reverse-jade</b>. Records the structure + <b>live entry premiums</b> every signal day (09:25 Mon–Fri) to accumulate <b>real forward ₹</b> — the missing input for the blended-book Calmar. Places <b>no orders</b>.
          </div>
          <div style={{ display: 'flex', gap: 22, flexWrap: 'wrap', alignItems: 'center', marginBottom: 10 }}>
            {chip(dirBg, dirColor, dirLabel)}
            {stat('Prior close', pct(jt.prev_ret), col(jt.prev_ret))}
            {stat('NIFTY', jt.spot != null ? jt.spot.toLocaleString('en-IN') : '—')}
            {stat('VIX', `${jt.vix ?? '—'} · ${jt.vix_regime}`)}
            {stat('Would enter', jt.would_enter ? 'YES' : 'no', jt.would_enter ? C.pos : C.muted)}
          </div>
          {Array.isArray(jt.legs) && jt.legs.length > 0 && (
            <table style={{ width: '100%', maxWidth: 460, borderCollapse: 'collapse', marginBottom: 8 }}>
              <thead><tr>
                <th style={{ ...jth, textAlign: 'left' }}>Leg</th><th style={jth}>Type</th>
                <th style={jth}>Strike</th><th style={jth}>Entry prem</th>
              </tr></thead>
              <tbody>{jt.legs.map((l: any, i: number) => (
                <tr key={i}>
                  <td style={{ ...jtd, textAlign: 'left', fontWeight: 700, color: l.side === 'SELL' ? C.neg : C.pos }}>{l.side}</td>
                  <td style={jtd}>{l.instrument_type}</td>
                  <td style={jtd}>{l.strike}</td>
                  <td style={jtd}>{l.entry != null ? l.entry : '—'}</td>
                </tr>))}</tbody>
            </table>
          )}
          <div style={{ fontSize: 11, color: C.muted, marginBottom: 4 }}>
            {jt.priced
              ? <>Net credit <b style={{ color: C.ink }}>{jt.net_credit} pts</b> (×650 = <b style={{ color: col(jt.net_credit ?? 0) }}>₹{Math.round((jt.net_credit ?? 0) * 650).toLocaleString('en-IN')}</b>) · long ±4% wing = defined-risk tail.</>
              : <>Premiums priced live at the 09:25 Mon–Fri logging tick (market closed now). Long ±4% wing = defined-risk tail.</>}
          </div>
          <details style={{ marginTop: 8 }}>
            <summary style={{ cursor: 'pointer', fontSize: 12, fontWeight: 600, color: C.navy, listStyle: 'none' }}>▸ Logged signal days ({log.length})</summary>
            {log.length === 0
              ? <div style={{ fontSize: 11, color: C.faint, padding: '6px 0' }}>None yet — the first fires at the next 09:25 Mon–Fri logging tick.</div>
              : <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: 6 }}>
                  <thead><tr>
                    <th style={{ ...jth, textAlign: 'left' }}>Day</th><th style={jth}>Prior%</th>
                    <th style={{ ...jth, textAlign: 'left' }}>Direction</th><th style={jth}>VIX</th>
                    <th style={jth}>NIFTY</th><th style={jth}>Net credit</th><th style={jth}>Enter?</th>
                  </tr></thead>
                  <tbody>{log.map((r: any, i: number) => (
                    <tr key={i}>
                      <td style={{ ...jtd, textAlign: 'left' }}>{r.day}</td>
                      <td style={{ ...jtd, color: col(r.prev_ret) }}>{pct(r.prev_ret)}</td>
                      <td style={{ ...jtd, textAlign: 'left', fontWeight: 700, color: r.direction === 'BULL' ? C.pos : r.direction === 'BEAR' ? C.neg : C.muted }}>{r.direction}</td>
                      <td style={jtd}>{r.vix ?? '—'}</td>
                      <td style={jtd}>{r.spot != null ? r.spot.toLocaleString('en-IN') : '—'}</td>
                      <td style={jtd}>{r.priced && r.net_credit != null ? `${r.net_credit}` : '—'}</td>
                      <td style={{ ...jtd, fontWeight: 700, color: r.would_enter ? C.pos : C.muted }}>{r.would_enter ? 'YES' : '—'}</td>
                    </tr>))}</tbody>
                </table>}
          </details>
        </section>
        );
      })()}

      <details style={{ marginTop: 14 }}>
        <summary style={{ cursor: 'pointer', fontSize: 15, fontWeight: 700, color: C.navy, padding: '10px 0' }}>📊 Backtests &amp; historical replays — click to expand (live books are above)</summary>
      {/* ===== ALL DAYS · DAILY JOURNEY ===== */}
      {daily && (
        <section style={{ ...card, marginTop: 14 }}>
          <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap', marginBottom: 4 }}>
            <span style={{ fontSize: 16, fontWeight: 700, color: C.ink }}>All recorded days · V1 intraday journey</span>
            {chip(C.navySoft, C.navy, `${daily.days.length} days · 0.4% one-and-done`)}
            {chip(C.amberSoft, C.amber, 'PAPER · replayed on recorded chain')}
          </div>
          <div style={{ fontSize: 11, color: C.muted, marginBottom: 8 }}>
            Every recorded day replayed (incl. non-0/1-DTE). The edge lives on 0/1-DTE — see the V1 backtest below for edge-only stats. Click any day for its intraday journey with the stop-exit marked (·h = held to close, no stop).
          </div>
          {dailyStats && (
            <div style={{ display: 'flex', gap: 26, flexWrap: 'wrap', margin: '4px 0 10px' }}>
              {stat('Total · all days', inr(dailyStats.tot), col(dailyStats.tot))}
              {stat('Mean/day', inr(dailyStats.mean), col(dailyStats.mean))}
              {stat('Days', String(dailyStats.n))}
              {stat('Win rate', `${Math.round(dailyStats.win)}%`)}
              {stat('Stopped', `${dailyStats.stops}/${dailyStats.n}`)}
            </div>
          )}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 6 }}>
            {daily.days.map((d) => {
              const x = daily.per_day[d];
              return (
                <button key={d} onClick={() => setDayD(d)} style={btn(d === dayD, col(x.final))}>
                  {d.slice(5)} {inr(x.final)}{x.stopped ? '' : ' ·h'}
                </button>
              );
            })}
          </div>
          {dayD && daily.per_day[dayD] && (() => {
            const x = daily.per_day[dayD];
            return (
              <div style={{ marginTop: 12, borderTop: `1px solid ${C.hair}`, paddingTop: 10 }}>
                <LineChart pts={x.series} h={130}
                  marker={x.exit ? { time: x.exit.time, pnl: x.exit.pnl, text: 'exit' } : null}
                  label={`${dayD} · ${x.strike} straddle (DTE ${x.dte}, credit ₹${x.credit}) · low ${inr(x.low)} · high ${inr(x.high)} · ${x.stopped ? `stop-exit ${x.exit!.time} @ ${inr(x.exit!.pnl)}` : `held to 15:15, final ${inr(x.final)}`}`} />
              </div>
            );
          })()}
        </section>
      )}

      {/* ===== V1 ===== */}
      <section style={{ ...card, marginTop: 14 }}>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap', marginBottom: 4 }}>
          <span style={{ fontSize: 16, fontWeight: 700, color: C.ink }}>V1 · Intraday one-and-done</span>
          {chip(C.navySoft, C.navy, '0.4% move-stop · 0/1-DTE · exit 14:45')}
          {chip(C.amberSoft, C.amber, 'BACKTEST')}
        </div>
        {v1stats && (
          <div style={{ display: 'flex', gap: 26, flexWrap: 'wrap', margin: '10px 0' }}>
            {stat('Total', inr(v1stats.tot), col(v1stats.tot))}
            {stat('Mean/day', inr(v1stats.mean), col(v1stats.mean))}
            {stat('Days', String(v1stats.n))}
            {stat('Win rate', `${Math.round(v1stats.win)}%`)}
          </div>
        )}
        {v1 && <LineChart pts={v1.cum_curve} h={100} label="Cumulative P&L across days (click a day below for its intraday curve)" />}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 10 }}>
          {days1.map((d) => {
            const f = v1!.per_day[d].final;
            return (
              <button key={d} onClick={() => setDay1(d === day1 ? null : d)} style={btn(d === day1, col(f))}>
                {d.slice(5)} {inr(f)}
              </button>
            );
          })}
        </div>
        {day1 && v1 && (
          <div style={{ marginTop: 12, borderTop: `1px solid ${C.hair}`, paddingTop: 10 }}>
            <LineChart pts={v1.per_day[day1].series} h={110}
              label={`${day1} · intraday P&L (entry → close) · ${v1.per_day[day1].stopped ? '0.4% STOP hit → flat' : 'held to 15:15'} · DTE ${v1.per_day[day1].dte} · final ${inr(v1.per_day[day1].final)}`} />
          </div>
        )}
      </section>

      {/* ===== V2 ===== */}
      <section style={card}>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap', marginBottom: 4 }}>
          <span style={{ fontSize: 16, fontWeight: 700, color: C.ink }}>V2 · Positional bi-weekly</span>
          <span style={{ display: 'inline-flex', border: `1px solid ${C.hair}`, borderRadius: 6, overflow: 'hidden' }}>
            {(['1.5', '2.0'] as const).map((sv) => (
              <button key={sv} onClick={() => setV2stop(sv)} style={{ border: 'none', cursor: 'pointer', fontSize: 11, fontWeight: 700, padding: '3px 10px', background: v2stop === sv ? C.navy : 'transparent', color: v2stop === sv ? '#fff' : C.muted }}>{sv}% stop</button>
            ))}
          </span>
          {chip(C.navySoft, C.navy, `${v2stop}% move-stop · PT-40% · ±500pt wings · re-enter · roll 1-DTE`)}
          {chip(C.amberSoft, C.amber, 'BACKTEST')}
        </div>
        {v2stats && (
          <div style={{ display: 'flex', gap: 26, flexWrap: 'wrap', margin: '10px 0' }}>
            {stat('Total', inr(v2stats.tot), col(v2stats.tot))}
            {stat('Mean/trade', inr(v2stats.mean), col(v2stats.mean))}
            {stat('Trades', String(v2stats.n))}
            {stat('Win rate', `${Math.round(v2stats.win)}%`)}
          </div>
        )}
        {v2 && <LineChart pts={v2.book_curve} h={100} label="Book cumulative P&L per trade (click a trade below for its day-by-day curve)" />}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 10 }}>
          {v2 && v2.trades.map((t, i) => (
            <button key={i} onClick={() => setTr2(i === tr2 ? null : i)} style={btn(i === tr2, col(t.pnl))}>
              {t.entry_day.slice(5)}→{t.exit_day.slice(5)} {inr(t.pnl)}
            </button>
          ))}
        </div>
        {tr2 != null && v2 && (
          <div style={{ marginTop: 12, borderTop: `1px solid ${C.hair}`, paddingTop: 10 }}>
            <LineChart pts={v2.trades[tr2].series} h={110}
              label={`${v2.trades[tr2].entry_day} → ${v2.trades[tr2].exit_day} · ${v2.trades[tr2].strike} straddle · exit: ${v2.trades[tr2].exit_reason} · wings ${inr(v2.trades[tr2].wing_pnl)} · final ${inr(v2.trades[tr2].pnl)}`} />
          </div>
        )}
        <div style={{ fontSize: 11, color: C.muted, marginTop: 10 }}>
          ⚠ Wings here are OVERNIGHT-ONLY (buy 15:20 / sell 09:20), net {v2 ? inr(v2.trades.reduce((a, t) => a + t.wing_pnl, 0)) : ''} — NOT the live engine's held-to-expiry wings. Held wings can't be backtested on this recorder: far-OTM wing quotes go stale (research/89) and produce impossible P&L. So read this as an ATM short-straddle behaviour probe, not a fly validation. Recorder replay Apr–Jul 2026 (~3 months, {v2stats ? v2stats.n : 13} trades, {v2stats ? Math.round(v2stats.win) : 0}% win, {v2stop}% stop) — SIGNAL only. The real held-wing fly is validated by the LIVE engine above + research/60 (AlgoTest).
        </div>
      </section>
      </details>
    </div>
  );
}
