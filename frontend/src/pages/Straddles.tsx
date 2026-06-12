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
        <th style={th}>Out</th><th style={th}>P&amp;L</th>
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
            <td style={{ ...td, fontWeight: 700, color: col(l.pnl) }}>{inr(l.pnl)}</td>
          </tr>
        ))}
        <tr>
          <td style={{ ...td, textAlign: 'left', color: C.muted, borderTop: `1px solid ${C.hair}` }} colSpan={8}>
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
            <li style={li}>{k('Exits:')} {k('1.5%')} underlying-move stop, or {k('+40%')} profit target, or roll at DTE≤1; {k('re-enter')} after exit.</li>
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
  const [day1, setDay1] = useState<string | null>(null);
  const [tr2, setTr2] = useState<number | null>(null);
  const [live, setLive] = useState<any>(null);
  const [liveTs, setLiveTs] = useState<number | null>(null);
  const [daily, setDaily] = useState<V1Daily | null>(null);
  const [dayD, setDayD] = useState<string | null>(null);
  const [v2eng, setV2eng] = useState<any>(null);   // live paper executor state
  const [bo, setBo] = useState<any>(null);          // inside-week breakout sleeve
  const [v2engTs, setV2engTs] = useState<number | null>(null);  // last v2-engine live tick
  const [v2spot, setV2spot] = useState<number | null>(null);    // live NIFTY spot from the stream

  useEffect(() => {
    fetch('/app/straddles/v1.json').then((r) => r.json()).then(setV1).catch(() => {});
    fetch('/app/straddles/v2.json').then((r) => r.json()).then(setV2).catch(() => {});
    fetch('/app/straddles/v1_daily.json').then((r) => r.json()).then(setDaily).catch(() => {});
    const loadLive = () => {
      fetch('/app/straddles_live.json?t=' + Date.now()).then((r) => r.json()).then(setLive).catch(() => {});
      fetch('/api/v2-ironfly/state?t=' + Date.now()).then((r) => r.json()).then(setV2eng).catch(() => {});
      fetch('/api/v2-breakout/state?t=' + Date.now()).then((r) => r.json()).then(setBo).catch(() => {});
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
    return () => { clearInterval(id); if (es) es.close(); if (es2) es2.close(); };
  }, []);

  const v1stats = useMemo(() => {
    if (!v1) return null;
    const f = Object.values(v1.per_day).map((d) => d.final);
    const tot = f.reduce((a, b) => a + b, 0);
    return { n: f.length, tot, mean: tot / (f.length || 1), win: 100 * f.filter((x) => x > 0).length / (f.length || 1) };
  }, [v1]);
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

  const days1 = v1 ? Object.keys(v1.per_day).sort() : [];
  const btn = (sel: boolean, c: string): React.CSSProperties => ({
    cursor: 'pointer', border: `1px solid ${sel ? C.navy : C.hair}`, background: sel ? C.navySoft : C.surface,
    color: c, borderRadius: 6, padding: '4px 8px', fontSize: 11, fontWeight: 600,
  });

  return (
    <div style={{ maxWidth: 1000 }}>
      <style>{`@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}`}</style>
      <div className="page-title">Straddle Systems</div>
      <div className="page-subtitle">Two short-straddle systems on NIFTY · backtested on the recorded chain · paper-forward 10 lots</div>

      {/* ===== TODAY · LIVE ===== */}
      {live && (
        <section style={{ ...card, marginTop: 14, borderColor: C.navy }}>
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
            {[['V1 · intraday one-and-done', live.v1], ['V2 · positional bi-weekly', live.v2]].map(([title, d]: any) => {
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
                      ▸ Completed trades ({comp.length})
                    </summary>
                    {comp.length === 0
                      ? <div style={{ fontSize: 11, color: C.faint, padding: '6px 0' }}>{isV1 ? 'No recorded days yet.' : `None yet — the current open position is the first (entered ${d.entry_day}).`}</div>
                      : <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: 6 }}>
                          <thead><tr>
                            <th style={{ ...cth, textAlign: 'left' }}>#</th><th style={{ ...cth, textAlign: 'left' }}>Entry</th>
                            <th style={{ ...cth, textAlign: 'left' }}>Exit</th><th style={cth}>Strike</th>
                            <th style={cth}>Exit P&amp;L</th><th style={{ ...cth, textAlign: 'left' }}>Reason</th>
                          </tr></thead>
                          <tbody>{comp.map((t: any, i: number) => (
                            <tr key={i}>
                              <td style={{ ...ctd, textAlign: 'left' }}>{t.n}</td>
                              <td style={{ ...ctd, textAlign: 'left' }}>{t.entry_day}</td>
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
          <div style={{ fontSize: 11, color: C.muted, marginTop: 8 }}>Live paper · live-quote P&amp;L ticks ~every 3s during market (positions/chart refresh every minute) · recorded daily. Backtest history below.</div>
          <RulesBlock />
        </section>
      )}

      {/* ===== V2 ENGINE · live paper executor ===== */}
      {v2eng && (
        <section style={{ ...card, marginTop: 14, borderColor: C.pos }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8, flexWrap: 'wrap' }}>
            <span style={{ fontSize: 16, fontWeight: 700, color: C.ink }}>V2 Engine · live paper executor</span>
            {chip('#E7F2EE', C.pos, 'PAPER · combo filter')}
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
          {v2eng.open ? (
            <div style={{ border: `1px solid ${C.hair}`, borderRadius: 8, padding: 12 }}>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
                <span style={{ fontWeight: 700, color: C.ink }}>Open fly · exp {v2eng.open.expiry} ({v2eng.open.dte_entry}d at entry)</span>
                <span style={{ fontSize: 22, fontWeight: 800, marginLeft: 'auto', color: col(v2eng.open.pnl_now || 0) }}>{inr(v2eng.open.pnl_now || 0)}</span>
              </div>
              <div style={{ fontSize: 11, color: C.muted, margin: '2px 0 6px' }}>
                entry {v2eng.open.entry_time} · spot {Math.round(v2eng.open.entry_spot)} · VIX {v2eng.open.entry_vix} · net credit {Number(v2eng.open.net_entry).toFixed(1)}
              </div>
              <div style={{ fontSize: 11, color: C.sec, margin: '0 0 8px', padding: '5px 8px', background: C.amberSoft, border: `1px solid ${C.hairSoft}`, borderRadius: 6 }}>
                <b>2% move-stop band:</b> exit if NIFTY ≤ <b style={{ color: C.neg }}>{v2eng.open.stop_dn?.toLocaleString('en-IN')}</b> or ≥ <b style={{ color: C.neg }}>{v2eng.open.stop_up?.toLocaleString('en-IN')}</b>
                <span style={{ color: C.muted }}> (±2% from {Math.round(v2eng.open.entry_spot).toLocaleString('en-IN')})</span>
                {v2spot ? <> · spot now <b style={{ color: C.ink }}>{Math.round(v2spot).toLocaleString('en-IN')}</b></> : null}
                <span style={{ color: C.muted }}> · checked on 1-min candle close</span>
              </div>
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
              <div style={{ fontSize: 11, color: C.muted, marginTop: 6 }}>
                Entered <b style={{ color: C.ink }}>{v2eng.open.day} · {v2eng.open.entry_time}</b> <span style={{ color: C.faint }}>(position-level — one entry/exit timestamp for all legs)</span>
              </div>
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
                  <th style={{ ...ecth, textAlign: 'left' }}>Exit (date · time)</th>
                  <th style={{ ...ecth, textAlign: 'left' }}>Reason</th><th style={ecth}>P&amp;L</th>
                </tr></thead>
                <tbody>{v2eng.closed.map((t: any, i: number) => (
                  <tr key={i}>
                    <td style={ectd}>{t.id}</td>
                    <td style={{ ...ectd, textAlign: 'left' }}>{t.day} · {t.entry_time}</td>
                    <td style={{ ...ectd, textAlign: 'left' }}>{(t.exit_day || '—')} · {(t.exit_time || '—')}</td>
                    <td style={{ ...ectd, textAlign: 'left', color: C.muted }}>{t.exit_reason}</td>
                    <td style={{ ...ectd, fontWeight: 700, color: col(t.pnl) }}>{inr(t.pnl)}</td>
                  </tr>))}</tbody>
              </table>
            </details>
          )}
          <details style={{ marginTop: 10 }}>
            <summary style={{ cursor: 'pointer', fontSize: 12, fontWeight: 600, color: C.navy }}>System rules — V2 engine (click to expand)</summary>
            <div style={{ fontSize: 11.5, color: C.sec, marginTop: 6, lineHeight: 1.7 }}>
              <div><b>Instrument:</b> short 2nd-nearest weekly NIFTY ATM straddle + <b>2%-of-ATM wings</b> (snapped to the 50-pt strike grid — ≈±450 at NIFTY 23.4k today; <b>not a fixed 500</b>) = short iron fly, overnight carry.</div>
              <div><b>Entry / when it starts:</b> 09:20 on any trading day the book is <b>flat</b> — sells the <b>2nd-nearest weekly</b> (must be <b>≥ 4 calendar days to expiry</b>), gated by <b>VIX ≥ 13</b> + the combo skip-filter. <b>No fixed weekday</b>: entries follow the roll cycle — it re-arms the morning after the prior fly rolls (DTE ≤ 1), so in practice it re-enters ~weekly. The scheduler checks at 09:20 Mon–Fri.</div>
              <div><b>Skip-filter (live):</b> skip entry when prior-day CPR width &lt; 0.10% OR last week was an inside week — every skip is shadow-logged for forward validation.</div>
              <div><b>Exits:</b> 2% underlying move-stop · +40% credit profit-target · roll at DTE ≤ 1, then re-enter.</div>
              <div><b>Stop band:</b> fixed at entry = entry-spot ±2% (the exact NIFTY levels are shown on the open position). A move-stop, not a premium-stop — it triggers on the underlying, not on the option P&L.</div>
              <div><b>When it's checked (AlgoTest-synced):</b> evaluated on the <b>close of each 1-min NIFTY candle</b> — the same resolution the backtest used (all 110 backtested SL exits land on :00 minute boundaries; AlgoTest is a 1-min candle-close engine, not tick/intra-candle). On the first candle whose close is ≥2% from entry, it exits at that close. The on-screen P&L still ticks every ~3s for display, but the exit decision is candle-close each minute. +40% PT and the DTE≤1 roll run on the same cycle.</div>
              <div><b>Sizing:</b> 10 lots = qty 650 · blocked margin ≈ ₹9.6L (₹95.8k/lot).</div>
              <div><b>Costs:</b> P&amp;L net of ₹20/order + 0.25% slippage — same basis as the backtest.</div>
              <div><b>Breakout sleeve (paper-only):</b> on inside-week-skip weeks, first daily close beyond the inside week's prior-week H/L → UP: call debit spread (runner edge) · DOWN: broken-wing fly skewed down.</div>
              <div style={{ color: C.muted, marginTop: 4 }}>Paper engine · live-quote P&amp;L ticks ~every 3s during market · 1-min position marks · backtest spec wired into the live engine.</div>
            </div>
          </details>
        </section>
      )}

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
          {chip(C.navySoft, C.navy, '1.5% stop · PT-40% · ±500pt wings · re-enter · roll 1-DTE')}
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
          Note: wings cost a net {v2 ? inr(v2.trades.reduce((a, t) => a + t.wing_pnl, 0)) : ''} over the book (overnight-gap protection you opted to keep). Single regime, ~6 weeks — SIGNAL, not yet validated.
        </div>
      </section>
    </div>
  );
}
