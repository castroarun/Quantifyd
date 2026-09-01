import React, { useEffect, useMemo, useState } from 'react';

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
const evTh: React.CSSProperties = { textAlign: 'left', padding: '5px 8px', fontSize: 10.5,
  fontWeight: 700, letterSpacing: 0.3, textTransform: 'uppercase', color: '#6B7280',
  borderBottom: '1px solid #E5E7EB' };
const evK: React.CSSProperties = { padding: '5px 8px', verticalAlign: 'top', whiteSpace: 'nowrap',
  color: '#6B7280', borderBottom: '1px solid #F3F4F6', width: 150 };
const evV: React.CSSProperties = { padding: '5px 8px', verticalAlign: 'top', color: '#374151',
  borderBottom: '1px solid #F3F4F6' };
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

/* ---------- CSL paper-books curve explorer (full-screen popup: day navigator + by-system wall) ---------- */
const CSL_BOOK_ORDER = ['CSL_TIMEB_NIFTY', 'CSL_TIMEB2_NIFTY', 'CSL_TIMEB_SENSEX', 'NAS_COMB20', 'NAS_C20_TRAIL', 'NAS_C20_SHIFT', 'CSL30F_NIFTY', 'CSL30F_SENSEX'];
const WD_DTE: Record<string, Record<number, number>> = {
  NIFTY: { 0: 1, 1: 0, 2: 4, 3: 3, 4: 2 }, SENSEX: { 0: 3, 1: 2, 2: 1, 3: 0, 4: 4 },
};
const dmon2 = (s: string) => s.slice(8, 10) + '-' + ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'][+s.slice(5, 7) - 1];
const srcChipStyle = (s: string): React.CSSProperties => ({
  fontSize: 9, fontWeight: 800, padding: '0 5px', borderRadius: 3,
  color: s === 'REAL' ? C.neg : s === 'BACKTEST' ? C.amber : C.pos,
  background: s === 'REAL' ? '#FBEEEE' : s === 'BACKTEST' ? C.amberSoft : '#E7F2EE',
});
const srcName = (s: string) => (s === 'REAL' ? 'LIVE REAL' : s === 'BACKTEST' ? 'BACKTEST' : 'LIVE PAPER');

function CslCurvesModal({ recs, mode0, day0, nasBase, onClose }: { recs: any[]; mode0: 'day' | 'sys'; day0: string; nasBase?: any; onClose: () => void }) {
  const [mode, setMode] = useState<'day' | 'sys'>(mode0);
  const [day, setDay] = useState<string>(day0);
  const [sys, setSys] = useState<string>('CSL_TIMEB_NIFTY');
  const [sysDay, setSysDay] = useState<string>('');
  const [venue, setVenue] = useState<'ALL' | 'NIFTY' | 'SENSEX'>('ALL');
  const [perLot, setPerLot] = useState(false);
  const inVenue = (name: string) => venue === 'ALL' || (venue === 'SENSEX' ? name.includes('SENSEX') : !name.includes('SENSEX'));
  const pv = (r: any, v: number) => (perLot ? Math.round(v / (r.lots || 1)) : v);
  const scaled = (r: any): [string, number][] => (perLot ? r.series.map((q: any) => [q[0], Math.round(q[1] / (r.lots || 1))]) : r.series);
  const days = useMemo(() => Array.from(new Set(recs.map((r) => r.day))).sort(), [recs]);
  const sysRecs = useMemo(() => recs.filter((r) => (r.book || r.sym) === sys).sort((a: any, b: any) => a.day.localeCompare(b.day)), [recs, sys]);
  const sysDays = sysRecs.map((r: any) => r.day);
  const move = (dir: number) => {
    if (mode === 'day') {
      const j = days.indexOf(day) + dir;
      if (j >= 0 && j < days.length) setDay(days[j]);
    } else {
      const cur = sysDay || sysDays[sysDays.length - 1];
      const j = sysDays.indexOf(cur) + dir;
      if (j >= 0 && j < sysDays.length) setSysDay(sysDays[j]);
    }
  };
  useEffect(() => {
    if (!inVenue(sys)) { const t = CSL_BOOK_ORDER.filter(inVenue)[0]; if (t) { setSys(t); setSysDay(''); } }
  }, [venue]);
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft') { e.preventDefault(); move(-1); }
      else if (e.key === 'ArrowRight') { e.preventDefault(); move(1); }
      else if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', h);
    return () => window.removeEventListener('keydown', h);
  });
  const navBtn: React.CSSProperties = { border: `1px solid ${C.hair}`, background: C.surface, color: C.navy, borderRadius: 6, padding: '3px 14px', fontSize: 15, fontWeight: 800, cursor: 'pointer' };
  const tabBtn = (on: boolean): React.CSSProperties => ({ border: `1px solid ${on ? C.navy : C.hair}`, background: on ? C.navySoft : C.surface, color: on ? C.navy : C.sec, borderRadius: 6, padding: '3px 12px', fontSize: 12, fontWeight: 700, cursor: 'pointer' });
  const dayRecs = recs.filter((r) => r.day === day);
  const selSysDay = sysDay || sysDays[sysDays.length - 1];
  const sysRec = sysRecs.find((r: any) => r.day === selSysDay);
  const symOf = sys.includes('SENSEX') ? 'SENSEX' : 'NIFTY';
  const wdName = ['MON', 'TUE', 'WED', 'THU', 'FRI'];
  const baseRow = (d2: string) => {
    const rows = ((nasBase?.days || {})[d2] || []).filter((b: any) => inVenue(b.book));
    if (!rows.length) return null;
    return (
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center', margin: '0 0 10px' }}>
        <span style={{ fontSize: 11, fontWeight: 800, color: C.sec, letterSpacing: 0.4 }}>NAS BASELINE (live books)</span>
        {rows.map((b: any) => (
          <span key={b.book} style={{ fontSize: 11.5, border: `1px solid ${C.hair}`, borderRadius: 6, padding: '2px 8px', color: C.sec, background: C.surface }}>
            {b.book} <span style={srcChipStyle(b.source === 'PAPER' ? 'PAPER' : 'REAL')}>{b.source === 'PAPER' ? 'PAPER' : 'REAL'}</span>{' '}
            <b style={{ color: col(b.pnl), fontVariantNumeric: 'tabular-nums' }}>{inr(perLot && b.lots ? Math.round(b.pnl / b.lots) : b.pnl)}</b>
            <span style={{ color: C.faint }}>{perLot && b.lots ? '/lot' : b.lots ? ` · ${b.lots}L` : ''}</span>
          </span>
        ))}
      </div>
    );
  };
  return (
    <div onClick={onClose} style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.55)', zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 12 }}>
      <div onClick={(e) => e.stopPropagation()} style={{ background: C.surface, border: `1px solid ${C.hair}`, borderRadius: 12, padding: '14px 18px', width: 'min(1550px, 97vw)', height: '94vh', overflowY: 'auto' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap', marginBottom: 10 }}>
          <span style={{ fontSize: 16.5, fontWeight: 800, color: C.ink, letterSpacing: 0.2 }}>Paper-book day curves</span>
          <button style={tabBtn(mode === 'day')} onClick={() => setMode('day')}>By day (all books)</button>
          <button style={tabBtn(mode === 'sys')} onClick={() => setMode('sys')}>By system (all days)</button>
          <span style={{ display: 'inline-flex', gap: 4, marginLeft: 6 }}>
            {(['ALL', 'NIFTY', 'SENSEX'] as const).map((v) => <button key={v} style={tabBtn(venue === v)} onClick={() => setVenue(v)}>{v === 'ALL' ? 'Both' : v}</button>)}
          </span>
          <button style={tabBtn(perLot)} onClick={() => setPerLot(!perLot)}>₹ / lot</button>
          <span style={{ fontSize: 11, color: C.faint }}>← → arrow keys or on-screen arrows to traverse · Esc to close</span>
          <button onClick={onClose} style={{ ...navBtn, marginLeft: 'auto', color: C.neg }}>✕ close</button>
        </div>
        {mode === 'day' ? (
          <>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10 }}>
              <button style={navBtn} onClick={() => move(-1)} disabled={days.indexOf(day) <= 0}>◀</button>
              <select value={day} onChange={(e) => setDay(e.target.value)}
                style={{ background: 'transparent', color: C.ink, border: `1px solid ${C.hair}`, borderRadius: 6, padding: '4px 10px', fontSize: 13, fontWeight: 700 }}>
                {days.map((d2) => <option key={d2} value={d2}>{dmon2(d2)} ({['SUN', 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT'][new Date(d2 + 'T00:00:00').getDay()]})</option>)}
              </select>
              <button style={navBtn} onClick={() => move(1)} disabled={days.indexOf(day) >= days.length - 1}>▶</button>
              <span style={{ fontSize: 12, color: C.muted }}>{dayRecs.length} book{dayRecs.length !== 1 ? 's' : ''} traded · day {days.indexOf(day) + 1}/{days.length}</span>
            </div>
            {baseRow(day)}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(460px, 1fr))', gap: 12 }}>
              {CSL_BOOK_ORDER.filter((bk) => inVenue(bk) && dayRecs.some((r) => (r.book || r.sym) === bk)).map((bk) => {
                const r = dayRecs.find((r2) => (r2.book || r2.sym) === bk);
                return (
                  <div key={bk} style={{ border: `1px solid ${C.hair}`, borderRadius: 8, padding: '8px 10px' }}>
                    <div style={{ fontSize: 12, fontWeight: 700, color: C.ink, marginBottom: 4 }}>
                      {bk} <span style={srcChipStyle(r.source || 'BACKTEST')}>{srcName(r.source || 'BACKTEST')}</span>
                      <span style={{ fontSize: 10.5, color: C.faint, fontWeight: 400 }}> · {r.cfg} · {r.lots} lots</span>
                      <span style={{ float: 'right', color: col(r.pnl ?? r.series[r.series.length - 1][1]), fontVariantNumeric: 'tabular-nums' }}>{inr(pv(r, r.pnl ?? r.series[r.series.length - 1][1]))}{perLot ? '/lot' : ''}</span>
                    </div>
                    <LineChart pts={scaled(r)} h={200} />
                  </div>
                );
              })}
            </div>
          </>
        ) : (
          <>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', marginBottom: 10 }}>
              {CSL_BOOK_ORDER.filter(inVenue).map((bk) => <button key={bk} style={tabBtn(sys === bk)} onClick={() => { setSys(bk); setSysDay(''); }}>{bk}</button>)}
            </div>
            {sysRec && (
              <div style={{ border: `1px solid ${C.navy}`, borderRadius: 8, padding: '8px 10px', marginBottom: 12 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
                  <button style={navBtn} onClick={() => move(-1)} disabled={sysDays.indexOf(selSysDay) <= 0}>◀</button>
                  <span style={{ fontSize: 13, fontWeight: 700, color: C.ink }}>
                    {sys} · {dmon2(selSysDay)} ({['SUN', 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT'][new Date(selSysDay + 'T00:00:00').getDay()]} · DTE{sysRec.dte})
                    {' '}<span style={srcChipStyle(sysRec.source || 'BACKTEST')}>{srcName(sysRec.source || 'BACKTEST')}</span>
                  </span>
                  <button style={navBtn} onClick={() => move(1)} disabled={sysDays.indexOf(selSysDay) >= sysDays.length - 1}>▶</button>
                  <span style={{ fontSize: 11, color: C.muted }}>{sysRec.cfg} · {sysRec.lots} lots · day {sysDays.indexOf(selSysDay) + 1}/{sysDays.length}</span>
                  <span style={{ marginLeft: 'auto', fontSize: 14, fontWeight: 700, color: col(sysRec.pnl), fontVariantNumeric: 'tabular-nums' }}>{inr(pv(sysRec, sysRec.pnl))}{perLot ? '/lot' : ''}</span>
                </div>
                <LineChart pts={scaled(sysRec)} h={280} />
              </div>
            )}
            {baseRow(selSysDay)}
            {(() => {
              if (sysRecs.length < 2) return null;
              let cum = 0;
              const cpts: [string, number][] = sysRecs.map((r: any) => { cum += pv(r, r.pnl || 0); return [dmon2(r.day), Math.round(cum)]; });
              return (
                <div style={{ border: `1px solid ${C.hair}`, borderRadius: 8, padding: '8px 10px', marginBottom: 12 }}>
                  <div style={{ fontSize: 12.5, fontWeight: 700, color: C.ink, marginBottom: 4 }}>
                    Cumulative P&L — {sys} · all {sysRecs.length} days{perLot ? ' · per lot' : ''}
                    <span style={{ float: 'right', color: col(cum), fontVariantNumeric: 'tabular-nums' }}>{inr(cum)}</span>
                  </div>
                  <LineChart pts={cpts} h={160} />
                </div>
              );
            })()}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 10 }}>
              {[0, 1, 2, 3, 4].map((w) => {
                const colRecs = sysRecs.filter((r: any) => new Date(r.day + 'T00:00:00').getDay() - 1 === w).slice().reverse();
                const net = colRecs.reduce((a: number, r: any) => a + pv(r, r.pnl || 0), 0);
                return (
                  <div key={w}>
                    <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', borderBottom: `2px solid ${C.navy}`, paddingBottom: 5, marginBottom: 8 }}>
                      <span style={{ fontSize: 12.5, fontWeight: 800, color: C.navy, letterSpacing: 0.4 }}>{wdName[w]} · DTE{WD_DTE[symOf][w]}</span>
                      <span style={{ fontSize: 12, fontWeight: 700, color: col(net), fontVariantNumeric: 'tabular-nums' }}>{colRecs.length ? inr(net) : '—'}</span>
                    </div>
                    <div style={{ fontSize: 9.5, color: C.faint, textAlign: 'right', margin: '-4px 0 6px' }}>{colRecs.length} day{colRecs.length !== 1 ? 's' : ''} · newest first</div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                      {colRecs.map((r: any) => (
                        <div key={r.day} onClick={() => setSysDay(r.day)}
                          style={{ border: `1px solid ${r.day === selSysDay ? C.navy : C.hair}`, borderRadius: 8, padding: '6px 8px 3px', cursor: 'pointer',
                            background: r.day === selSysDay ? C.navySoft : C.surface, boxShadow: r.day === selSysDay ? 'none' : '0 1px 2px rgba(0,0,0,0.04)' }}>
                          <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', marginBottom: 2 }}>
                            <span style={{ fontSize: 11.5, color: C.ink, fontWeight: 700, letterSpacing: 0.2 }}>
                              {dmon2(r.day)}{r.source && r.source !== 'BACKTEST' && <span style={{ ...srcChipStyle(r.source), marginLeft: 5 }}>{r.source}</span>}
                            </span>
                            <span style={{ fontSize: 12, fontWeight: 700, color: col(r.pnl), fontVariantNumeric: 'tabular-nums' }}>{inr(pv(r, r.pnl))}{perLot ? '/lot' : ''}</span>
                          </div>
                          <LineChart pts={scaled(r)} h={80} />
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

/* ---------- collapsed ops-info block: jobs + manual triggers per lab section ---------- */
function OpsInfo({ rows }: { rows: (string | undefined)[][] }) {
  return (
    <details style={{ marginTop: 10, borderTop: `1px solid ${C.hairSoft}`, paddingTop: 8 }}>
      <summary style={{ cursor: 'pointer', fontSize: 11.5, fontWeight: 700, color: C.navy, userSelect: 'none' }}>
        ⚙ Jobs &amp; manual triggers (click to expand)</summary>
      <div style={{ marginTop: 6 }}>
        {rows.map(([lbl, desc, cmd], i) => (
          <div key={i} style={{ fontSize: 11.5, color: C.sec, marginBottom: 5 }}>
            <b style={{ color: C.ink }}>{lbl}</b> — {desc}
            {cmd && <div><code style={{ display: 'inline-block', marginTop: 2, fontSize: 10.5, background: C.canvas,
              border: `1px solid ${C.hairSoft}`, borderRadius: 4, padding: '1px 6px', color: C.sec, overflowWrap: 'anywhere' }}>{cmd}</code></div>}
          </div>
        ))}
        <div style={{ fontSize: 10, color: C.faint }}>Commands run on the VPS from /home/arun/quantifyd · full reference: docs/LABS_AND_JOBS_REFERENCE.md</div>
      </div>
    </details>
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
  // Deep-link support: the TB-CSL alert opens /app/straddles#csl-paper, but the
  // sections mount after data fetches, so the browser's native anchor jump misses.
  // Retry until the element exists (~6s), then smooth-scroll to it.
  useEffect(() => {
    const h = window.location.hash.replace('#', '');
    if (!h) return;
    let tries = 0;
    const t = window.setInterval(() => {
      const el = document.getElementById(h);
      tries += 1;
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        window.clearInterval(t);
      } else if (tries > 20) {
        window.clearInterval(t);
      }
    }, 300);
    return () => window.clearInterval(t);
  }, []);
  const [v1, setV1] = useState<V1 | null>(null);
  const [v2, setV2] = useState<V2 | null>(null);
  const [v2all, setV2all] = useState<{ [k: string]: V2 }>({});
  const [v2stop, setV2stop] = useState<'1.5' | '2.0'>('2.0');
  const [day1, setDay1] = useState<string | null>(null);
  const [tr2, setTr2] = useState<number | null>(null);
  const [live, setLive] = useState<any>(null);
  const [liveTs, setLiveTs] = useState<number | null>(null);
  const [daily, setDaily] = useState<V1Daily | null>(null);
  const [stopMx, setStopMx] = useState<any>(null);  // what each stop level would
  // have done on the trades the book actually took, from the recorded chain
  const [sl30, setSl30] = useState<any>(null);   // HYPOTHETICAL V1 variant: a 30%
  // combined-premium stop replayed over the recorded chain. Not a running system —
  // live V1 stops on an underlying move of 0.4% (0-DTE) / 0.5% (1-DTE).
  const [sl30Modal, setSl30Modal] = useState(false);   // deep-dive popup
  const [sl30Day, setSl30Day] = useState<string>('');  // selected day in popup
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
  const dmon = (s: string) => s.slice(8, 10) + '-' + ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'][+s.slice(5, 7) - 1];
  const [variants, setVariants] = useState<any>(null); // v2 stop x wings variant lab
  const [cslCfg, setCslCfg] = useState<any>(null);   // research/111 best-config lab (weekly regen)
  const [expLab, setExpLab] = useState<any>(null);   // research/125 expiry-afternoon lab (Tue+Thu 16:05 regen)
  const [cslIdx, setCslIdx] = useState<'NIFTY' | 'SENSEX'>('NIFTY');
  const [cslDte, setCslDte] = useState<string>('all');
  const [csl2nd, setCsl2nd] = useState(false);       // stack the next-best non-overlapping slot
  const [cslPaper, setCslPaper] = useState<any>(null);     // live paper book state
  const [cslPaperCfg, setCslPaperCfg] = useState<any>(null); // frozen config
  const [cslPLive, setCslPLive] = useState<any>(null);     // intraday live series (today)
  const [cslPDay, setCslPDay] = useState<string>('');      // selected day for curves
  const [cslPBig, setCslPBig] = useState<string | null>(null); // expanded book curve
  const [cslPBF, setCslPBF] = useState<any>(null);   // backfilled BACKTEST day curves (recorded chain replay)
  const [cslPModal, setCslPModal] = useState<null | 'day' | 'sys'>(null); // full-screen curve explorer
  const [cslPVenue, setCslPVenue] = useState<'ALL' | 'NIFTY' | 'SENSEX'>('ALL'); // inline grid venue filter
  const [lbVenue, setLbVenue] = useState<'ALL' | 'NIFTY' | 'SENSEX'>('ALL');     // leaderboard venue filter
  const [nasBase, setNasBase] = useState<any>(null);  // NAS live-book day P&L baseline
  const [pfLab, setPfLab] = useState<any>(null);      // options portfolio lab (daily regen)
  const [pfSel, setPfSel] = useState<number | null>(null); // selected lab portfolio row (charts)
  const [pfDte, setPfDte] = useState<string>('ALL');       // lab charts DTE filter (NIFTY weekday map)
  const [pfRe, setPfRe] = useState<any>(null);             // weekly stack re-assessment (Fri 16:35)
  const [pfReH, setPfReH] = useState<any>(null);           // re-assessment run history (permanent trail)
  const [opsC, setOpsC] = useState<any>(null);             // ops & review center registry

  useEffect(() => {
    fetch('/app/straddles/v1.json').then((r) => r.json()).then(setV1).catch(() => {});
    fetch('/app/straddles/v2_2.0.json').then((r) => r.json()).then((d) => setV2all((m) => ({ ...m, '2.0': d }))).catch(() => {});
    fetch('/app/straddles/v2_1.5.json').then((r) => r.json()).then((d) => setV2all((m) => ({ ...m, '1.5': d }))).catch(() => {});
    fetch('/app/straddles/v1_daily.json').then((r) => r.json()).then(setDaily).catch(() => {});
    fetch('/app/v2_stop_matrix.json?t=' + Date.now()).then((r) => r.json())
      .then(setStopMx).catch(() => {});
    fetch('/app/straddles/v1_sl30.json?t=' + Date.now()).then((r) => r.json()).then(setSl30).catch(() => {});
    fetch('/app/straddles/rankings.json?t=' + Date.now()).then((r) => r.json()).then(setRanks).catch(() => {});
    fetch('/app/straddles/variants.json?t=' + Date.now()).then((r) => r.json()).then(setVariants).catch(() => {});
    fetch('/app/straddles/csl_best_configs.json?t=' + Date.now()).then((r) => r.json()).then(setCslCfg).catch(() => {});
    fetch('/app/csl_paper.json?t=' + Date.now()).then((r) => r.json()).then(setCslPaper).catch(() => {});
    fetch('/app/csl_paper_config.json?t=' + Date.now()).then((r) => r.json()).then(setCslPaperCfg).catch(() => {});
    fetch('/app/csl_paper_backfill.json?t=' + Date.now()).then((r) => r.json()).then(setCslPBF).catch(() => {});
    fetch('/app/nas_baseline.json?t=' + Date.now()).then((r) => r.json()).then(setNasBase).catch(() => {});
    fetch('/app/straddles/portfolio_lab.json?t=' + Date.now()).then((r) => r.json()).then(setPfLab).catch(() => {});
    fetch('/app/straddles/reassessment.json?t=' + Date.now()).then((r) => r.json()).then(setPfRe).catch(() => {});
    fetch('/app/straddles/reassessment_history.json?t=' + Date.now()).then((r) => r.json()).then(setPfReH).catch(() => {});
    fetch('/app/straddles/ops_center.json?t=' + Date.now()).then((r) => r.json()).then(setOpsC).catch(() => {});
    fetch('/app/straddles/expiry_lab.json?t=' + Date.now()).then((r) => r.json()).then(setExpLab).catch(() => {});
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
            ['#csl-paper', '📗 Paper Books (live validation)'], ['#csl-lab', '🔬 Best-Config Lab (weekly)'], ['#expiry-lab', '⏱ Expiry-Afternoon Lab'],
            ['#portfolio-lab', '🎯 Options Portfolio Lab'], ['#ops-center', '🛠 Ops & Reviews'], ['#leaderboard', '🏆 Strategy Leaderboard'], ['#variant-lab', '🧪 V2 Variant Lab'],
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

      {pfLab && pfLab.portfolios && (
        <section id="portfolio-lab" style={{ ...card, marginTop: 14, scrollMarginTop: 70, borderColor: C.pos }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
            <span style={{ fontSize: 16, fontWeight: 800, color: C.ink }}>Options Portfolio Lab</span>
            {chip('#E7F2EE', C.pos, 'NIFTY · the stack')}
            {chip(C.amberSoft, C.amber, 'refreshed daily 15:40')}
            <span style={{ fontSize: 11, color: C.faint }}>updated {pfLab.generated_at} · live-first merge (paper overrides backfill)</span>
          </div>
          <div style={{ fontSize: 12.5, color: C.sec, margin: '8px 0' }}>{pfLab.verdict}</div>
          <div style={{ overflowX: 'auto', marginBottom: 10 }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead><tr><th style={thL}>Portfolio</th><th style={thR}>Lots</th><th style={thL}>Scope</th>
                <th style={thR}>Total</th><th style={thR}>Mean/d</th><th style={thR}>MaxDD</th><th style={thR}>Ratio</th><th style={thR}>corr(parts)</th><th style={thR}>N</th></tr></thead>
              <tbody>
                {pfLab.portfolios.map((p2: any, i: number) => {
                  const selI = pfSel ?? pfLab.portfolios.findIndex((q2: any) => q2.label.startsWith('THE STACK') && q2.scope === 'all');
                  return (
                  <tr key={i} onClick={() => setPfSel(i)}
                    style={{ cursor: 'pointer', outline: i === selI ? `2px solid ${C.navy}` : undefined,
                      background: p2.label.startsWith('THE STACK') ? '#F0F7F4' : undefined }}>
                    <td style={{ ...tdL, fontWeight: p2.label.startsWith('THE STACK') ? 800 : 600, color: C.ink }}>{p2.label}</td>
                    <td style={tdR}>{p2.lots}L</td>
                    <td style={{ ...tdL, color: p2.scope === 'ex-Wed' ? C.navy : C.muted, fontWeight: p2.scope === 'ex-Wed' ? 700 : 400 }}>{p2.scope}</td>
                    <td style={{ ...tdR, fontWeight: 700, color: col(p2.total), fontVariantNumeric: 'tabular-nums' }}>{inr(p2.total)}</td>
                    <td style={{ ...tdR, fontVariantNumeric: 'tabular-nums' }}>{inr(p2.mean)}</td>
                    <td style={{ ...tdR, color: C.neg, fontVariantNumeric: 'tabular-nums' }}>{inr(p2.maxdd)}</td>
                    <td style={{ ...tdR, fontWeight: 800, color: p2.ratio >= 7 ? C.pos : C.ink }}>{p2.ratio}</td>
                    <td style={{ ...tdR, fontWeight: 700, color: p2.corr_parts == null ? C.faint : p2.corr_parts <= 0.4 ? C.pos : p2.corr_parts <= 0.7 ? C.amber : C.neg }}>
                      {p2.corr_parts == null ? '—' : p2.corr_parts.toFixed(2)}</td>
                    <td style={tdR}>{p2.n}</td>
                  </tr>);
                })}
              </tbody>
            </table>
          </div>
          {(() => {
            const selI = pfSel ?? pfLab.portfolios.findIndex((q2: any) => q2.label.startsWith('THE STACK') && q2.scope === 'all');
            const sel = pfLab.portfolios[selI >= 0 ? selI : 0];
            if (!sel || !sel.series || sel.series.length < 2) return null;
            const dteOf = (d2: string) => WD_DTE.NIFTY[new Date(d2 + 'T00:00:00').getDay() - 1];
            const fser: [string, number][] = pfDte === 'ALL' ? sel.series : sel.series.filter(([d2]: any) => dteOf(d2) === +pfDte);
            const dteLbl = ['TUE', 'MON', 'FRI', 'THU', 'WED'];
            const dteSel = (
              <select value={pfDte} onChange={(e) => setPfDte(e.target.value)}
                style={{ background: 'transparent', color: C.navy, border: `1px solid ${C.navy}`, borderRadius: 6, padding: '2px 8px', fontSize: 11.5, fontWeight: 700 }}>
                <option value="ALL">All DTEs</option>
                {[0, 1, 2, 3, 4].map((k) => <option key={k} value={String(k)}>DTE{k} · {dteLbl[k]}</option>)}
              </select>);
            if (fser.length < 2) {
              return (
                <div style={{ border: `1px solid ${C.navy}`, borderRadius: 8, padding: '10px 12px', marginBottom: 12 }}>
                  <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, flexWrap: 'wrap' }}>
                    <span style={{ fontSize: 13.5, fontWeight: 800, color: C.ink }}>{sel.label} · {sel.lots}L · {sel.scope}</span>
                    {dteSel}
                    <span style={{ fontSize: 12, color: C.faint }}>fewer than 2 days at this DTE in this scope (ex-Wed excludes DTE4)</span>
                  </div>
                </div>);
            }
            let cum = 0; let pk = 0; let ddm = 0; let wins = 0;
            const cpts: [string, number][] = []; const dpts: [string, number][] = [];
            fser.forEach(([d2, v2]: any) => {
              cum += v2; pk = Math.max(pk, cum); ddm = Math.min(ddm, cum - pk);
              if (v2 > 0) wins++;
              cpts.push([dmon(d2), Math.round(cum)]); dpts.push([dmon(d2), Math.round(cum - pk)]);
            });
            const fTotal = Math.round(cum); const fN = fser.length;
            const fMean = Math.round(fTotal / fN); const fDD = Math.round(ddm);
            const fRatio = fDD < 0 ? Math.round(fTotal / Math.abs(fDD) * 10) / 10 : 99;
            return (
              <div style={{ border: `1px solid ${C.navy}`, borderRadius: 8, padding: '10px 12px', marginBottom: 12 }}>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, flexWrap: 'wrap', marginBottom: 6 }}>
                  <span style={{ fontSize: 13.5, fontWeight: 800, color: C.ink }}>{sel.label} · {sel.lots}L · {sel.scope}{pfDte !== 'ALL' ? ` · DTE${pfDte}/${dteLbl[+pfDte]}` : ''}</span>
                  {dteSel}
                  <span style={{ fontSize: 12, color: C.sec }}>
                    net <b style={{ color: col(fTotal) }}>{inr(fTotal)}</b> · mean/day <b style={{ color: col(fMean) }}>{inr(fMean)}</b> ·
                    win <b>{Math.round(100 * wins / fN)}%</b> ·
                    maxDD <b style={{ color: C.neg }}>{inr(fDD)}</b> · ratio <b style={{ color: fRatio >= 7 ? C.pos : C.ink }}>{fRatio}</b>
                    {pfDte === 'ALL' && <> · corr <b>{sel.corr_parts ?? '—'}</b></>} · {fN} days
                  </span>
                  <span style={{ fontSize: 10.5, color: C.faint }}>click any row above to chart it</span>
                </div>
                <LineChart pts={cpts} h={200} label="Cumulative P&L" />
                <LineChart pts={dpts} h={90} label="Drawdown" />
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(230px, 1fr))', gap: 8, marginTop: 8 }}>
                  {Object.entries(pfLab.components).map(([n2, c2]: any) => {
                    const ser2 = pfDte === 'ALL' ? (c2.series || []) : (c2.series || []).filter(([d2]: any) => dteOf(d2) === +pfDte);
                    if (ser2.length < 2) return null;
                    let cc = 0;
                    const pts2: [string, number][] = ser2.map(([d2, v2]: any) => { cc += v2; return [dmon(d2), Math.round(cc)]; });
                    return (
                      <div key={n2} style={{ border: `1px solid ${C.hairSoft}`, borderRadius: 6, padding: '4px 8px' }}>
                        <div style={{ fontSize: 10.5, fontWeight: 700, color: C.sec }}>{n2.replace(/_/g, ' ')} (cum)
                          <span style={{ float: 'right', color: col(cc) }}>{inr(cc)}</span></div>
                        <LineChart pts={pts2} h={64} />
                      </div>);
                  })}
                </div>
              </div>
            );
          })()}
          <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap' }}>
            <div>
              <div style={{ fontSize: 12, fontWeight: 700, color: C.ink, marginBottom: 4 }}>Correlation matrix (daily P&amp;L)</div>
              <table style={{ borderCollapse: 'collapse', fontSize: 11 }}>
                <thead><tr><th style={thL}></th>{pfLab.names.map((n2: string) => <th key={n2} style={thR}>{n2.replace(/_/g, ' ').slice(0, 10)}</th>)}</tr></thead>
                <tbody>{pfLab.names.map((a: string, i: number) => (
                  <tr key={a}><td style={{ ...tdL, fontWeight: 700 }}>{a.replace(/_/g, ' ')}</td>
                    {pfLab.matrix[i].map((v2: number | null, j2: number) => (
                      <td key={j2} style={{ ...tdR, fontWeight: i === j2 ? 400 : 700,
                        color: v2 == null ? C.faint : i === j2 ? C.faint : v2 <= 0.4 ? C.pos : v2 <= 0.7 ? C.amber : C.neg }}>
                        {v2 == null ? '—' : v2.toFixed(2)}</td>))}
                  </tr>))}
                </tbody>
              </table>
            </div>
            <div style={{ flex: 1, minWidth: 260 }}>
              <div style={{ fontSize: 12, fontWeight: 700, color: C.ink, marginBottom: 4 }}>Components (n · range · source mix)</div>
              {Object.entries(pfLab.components).map(([n2, c2]: any) => (
                <div key={n2} style={{ fontSize: 11.5, color: C.sec, marginBottom: 3 }}>
                  <b style={{ color: C.ink }}>{n2.replace(/_/g, ' ')}</b> · {c2.n}d {c2.from?.slice(5)}→{c2.to?.slice(5)} ·{' '}
                  {Object.entries(c2.sources).map(([src, cnt]: any) => `${src} ${cnt}d`).join(' + ')} ·{' '}
                  <span style={{ color: col(c2.stats.total), fontWeight: 700 }}>{inr(c2.stats.total)}</span>
                </div>))}
              <div style={{ fontSize: 10.5, color: C.faint, marginTop: 6 }}>{pfLab.basis}</div>
              <OpsInfo rows={[
                ['Auto-refresh', 'daily 15:40 IST, last step of the regen chain (after backfill + NAS baseline)', 'venv/bin/python3 research/111_sensex_manual_mgmt/scripts/portfolio_lab.py'],
                ['Inputs (live-first)', 'csl_paper_state.json (PAPER/REAL records override) + csl_paper_backfill.json (BACKTEST) + nas_baseline.json (suite actuals)'],
                ['Sizing studies (sec 18/18b)', 're-run anytime; auto-include new live days', 'venv/bin/python3 research/111_sensex_manual_mgmt/scripts/comb_tb_overweight_grid.py'],
                ['Weekly RE-ASSESSMENT', 'Fridays 16:35 IST — corr-drift, per-DTE shifts, TB-window revalidation, sizing revalidation, live-vs-model tracking (panel below)', 'venv/bin/python3 research/111_sensex_manual_mgmt/scripts/stack_reassessment.py'],
              ]} />
              {pfRe && pfRe.checks && (
                <details style={{ marginTop: 8, border: `1px solid ${pfRe.overall === 'DRIFT' ? C.amber : C.hair}`, borderRadius: 8, padding: '6px 10px' }}>
                  <summary style={{ cursor: 'pointer', fontSize: 12, fontWeight: 700, userSelect: 'none',
                    color: pfRe.overall === 'DRIFT' ? C.amber : C.pos }}>
                    🔬 Weekly re-assessment — overall: {pfRe.overall} · {pfRe.generated_at} · {pfRe.window?.n}d window (recent {pfRe.window?.recent_n}d)
                  </summary>
                  <div style={{ marginTop: 6 }}>
                    {pfRe.checks.map((c2: any) => (
                      <div key={c2.name} style={{ fontSize: 11.5, marginBottom: 5 }}>
                        <span style={{ fontWeight: 800, padding: '0 6px', borderRadius: 4,
                          color: c2.verdict === 'DRIFT' ? C.amber : c2.verdict === 'INFO' ? C.navy : C.pos,
                          background: c2.verdict === 'DRIFT' ? C.amberSoft : c2.verdict === 'INFO' ? C.navySoft : '#E7F2EE' }}>
                          {c2.verdict}</span>{' '}
                        <b style={{ color: C.ink }}>{c2.name}</b>
                        <span style={{ color: C.sec }}> — {c2.detail}</span>
                      </div>))}
                    <div style={{ fontSize: 10, color: C.faint }}>DRIFT = behavior changed vs the frozen/sized basis — review, nothing auto-changes. Small recent windows (15d) are noisy; confirm before acting.</div>
                    {pfReH && pfReH.runs && pfReH.runs.length > 0 && (
                      <details style={{ marginTop: 8, borderTop: `1px solid ${C.hairSoft}`, paddingTop: 6 }}>
                        <summary style={{ cursor: 'pointer', fontSize: 11.5, fontWeight: 700, color: C.navy, userSelect: 'none' }}>
                          📜 Assessment run history ({pfReH.runs.length} logged)</summary>
                        <div style={{ marginTop: 6 }}>
                          {pfReH.runs.slice().reverse().map((run2: any, ri: number) => (
                            <details key={ri} style={{ marginBottom: 4 }}>
                              <summary style={{ cursor: 'pointer', fontSize: 11, userSelect: 'none' }}>
                                <span style={{ fontWeight: 800, color: run2.overall === 'DRIFT' ? C.amber : C.pos }}>{run2.overall}</span>
                                <b style={{ color: C.ink }}> {run2.generated_at}</b>
                                <span style={{ color: C.faint }}> · data {run2.window?.from?.slice(5)}→{run2.window?.to?.slice(5)} ({run2.window?.n}d, recent {run2.window?.recent_n}d) · </span>
                                {run2.checks?.map((c2: any) => (
                                  <span key={c2.name} title={c2.name} style={{ fontSize: 9, fontWeight: 800, padding: '0 4px', marginRight: 2, borderRadius: 3,
                                    color: c2.verdict === 'DRIFT' ? C.amber : c2.verdict === 'INFO' ? C.navy : C.pos,
                                    background: c2.verdict === 'DRIFT' ? C.amberSoft : c2.verdict === 'INFO' ? C.navySoft : '#E7F2EE' }}>
                                    {c2.name.split('-')[0]}</span>
                                ))}
                              </summary>
                              <div style={{ margin: '4px 0 6px 14px' }}>
                                {run2.checks?.map((c2: any) => (
                                  <div key={c2.name} style={{ fontSize: 10.5, color: C.sec, marginBottom: 3 }}>
                                    <b style={{ color: c2.verdict === 'DRIFT' ? C.amber : C.ink }}>{c2.verdict} · {c2.name}</b>
                                    {c2.objective && <span style={{ color: C.faint }}> — {c2.objective}</span>}
                                    <div style={{ marginLeft: 8 }}>{c2.detail}</div>
                                  </div>
                                ))}
                              </div>
                            </details>
                          ))}
                        </div>
                      </details>
                    )}
                  </div>
                </details>
              )}
            </div>
          </div>
        </section>
      )}

      {ranks && ranks.systems && (
        <section id="leaderboard" style={{ ...card, marginTop: 14, scrollMarginTop: 70 }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, flexWrap: 'wrap', marginBottom: 8 }}>
            <span style={{ fontSize: 16, fontWeight: 700, color: C.ink }}>Strategy Leaderboard</span>
            <span style={{ fontSize: 11, color: C.faint }}>rated by risk-adjusted return (Calmar) · updated {ranks.generated_at} · {ranks.cadence}</span>
            <span style={{ display: 'inline-flex', gap: 4 }}>
              {(['ALL', 'NIFTY', 'SENSEX'] as const).map((v) => (
                <button key={v} onClick={() => setLbVenue(v)}
                  style={{ border: `1px solid ${lbVenue === v ? C.navy : C.hair}`, background: lbVenue === v ? C.navySoft : 'transparent', color: lbVenue === v ? C.navy : C.sec, borderRadius: 6, padding: '1px 8px', fontSize: 10.5, fontWeight: 700, cursor: 'pointer' }}>{v === 'ALL' ? 'Both' : v}</button>
              ))}
            </span>
          </div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead><tr>
                <th style={thL}>#</th><th style={thL}>Grade</th><th style={thL}>System</th>
                <th style={thR}>Net P&amp;L</th><th style={thR}>Calmar</th><th style={thR}>MaxDD</th>
                <th style={thR}>Win</th><th style={thR}>N</th><th style={thL}>Period</th><th style={thR}>Corr·book</th><th style={thL}>Confidence</th>
              </tr></thead>
              <tbody>
                {ranks.systems.filter((r: any) => lbVenue === 'ALL' || (lbVenue === 'SENSEX' ? r.label.includes('SENSEX') : !r.label.includes('SENSEX'))).map((r: any) => (
                  <tr key={r.label}>
                    <td style={tdL}>{r.rank}</td>
                    <td style={tdL}>{gradeBadge(r.grade)}</td>
                    <td style={{ ...tdL, color: C.ink, fontWeight: 600 }}>
                      {r.anchor
                        ? <a href={'#' + r.anchor} onClick={(e) => { e.preventDefault(); document.getElementById(r.anchor)?.scrollIntoView({ behavior: 'smooth', block: 'start' }); }} style={{ color: C.navy, textDecoration: 'none', cursor: 'pointer' }}>{r.label} ↗</a>
                        : (r.page ? <a href={r.page} style={{ color: C.navy, textDecoration: 'none' }}>{r.label} ↗</a> : r.label)}
                      <div style={{ fontSize: 10, color: C.faint, fontWeight: 400 }}>
                        <span style={{ fontWeight: 800, padding: '0 6px', borderRadius: 4,
                          color: r.kind === 'live-real' ? C.neg : (String(r.kind).includes('paper') ? C.pos : C.amber),
                          background: r.kind === 'live-real' ? '#FBEEEE' : (String(r.kind).includes('paper') ? '#E7F2EE' : C.amberSoft) }}>
                          {r.kind === 'live-real' ? 'LIVE REAL' : (String(r.kind).includes('paper') ? 'LIVE PAPER' : 'BACKTEST')}</span>
                        {' '}· {r.note}
                        {r.report && <> · <a href={r.report} style={{ color: C.navy }}>📄 report</a></>}
                        {r.chart && <> · <a href={r.chart} style={{ color: C.navy }}>📈 tearsheet</a></>}
                      </div></td>
                    <td style={{ ...tdR, color: col(r.net), fontWeight: 700 }}>{inr(r.net)}</td>
                    <td style={tdR}>{r.calmar ?? '—'}</td>
                    <td style={{ ...tdR, color: C.neg }}>{inr(r.maxdd)}</td>
                    <td style={tdR}>{r.win}%</td>
                    <td style={tdR}>{r.n}</td>
                    <td style={{ ...tdL, whiteSpace: 'nowrap', fontSize: 11, color: C.muted }}>{r.from ? `${dmon(r.from)} → ${dmon(r.to)}` : '—'}</td>
                    <td style={{ ...tdR, fontWeight: 700, fontVariantNumeric: 'tabular-nums',
                      color: r.corr == null ? C.faint : r.corr <= 0.4 ? C.pos : r.corr <= 0.7 ? C.amber : C.neg }}>
                      {r.corr == null ? '—' : r.corr.toFixed(2)}</td>
                    <td style={{ ...tdL, color: r.confidence === 'medium' ? C.sec : C.amber }}>{r.confidence}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div style={{ fontSize: 11, color: C.amber, marginTop: 8 }}>⚠ {ranks.caveat}</div>
          <div style={{ fontSize: 10.5, color: C.faint, marginTop: 4 }}>Refresh jobs &amp; manual triggers: see <a href="#ops-center" style={{ color: C.navy }}>Ops &amp; Review Center</a>.</div>
        </section>
      )}

      {opsC && opsC.groups && (
        <section id="ops-center" style={{ ...card, marginTop: 14, scrollMarginTop: 70 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
            <span style={{ fontSize: 16, fontWeight: 800, color: C.ink }}>🛠 Operations &amp; Review Center</span>
            {chip(C.navySoft, C.navy, 'the standing registry')}
            <span style={{ fontSize: 11, color: C.faint }}>updated {opsC.generated_at} · every new lab job / periodic review is registered here (binding convention)</span>
          </div>

          <div style={{ fontSize: 12.5, fontWeight: 700, color: C.ink, margin: '10px 0 4px' }}>Periodic reviews &amp; re-assessments</div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead><tr><th style={thL}></th><th style={thL}>Review</th><th style={thL}>Due</th><th style={thL}>How / evidence</th></tr></thead>
              <tbody>
                {opsC.reviews.map((r: any, i: number) => (
                  <tr key={i} style={r.flag === 'OVERDUE' || r.flag === 'DUE SOON' ? { background: C.amberSoft } : undefined}>
                    <td style={tdL}><span style={{ fontSize: 10, fontWeight: 800, padding: '0 6px', borderRadius: 4,
                      color: r.flag === 'OVERDUE' ? C.neg : r.flag === 'DUE SOON' ? C.amber : r.status === 'PARKED' ? C.faint : C.navy,
                      background: r.flag === 'OVERDUE' ? '#FBEEEE' : r.flag === 'DUE SOON' ? '#FFF' : 'transparent' }}>{r.flag}</span></td>
                    <td style={{ ...tdL, fontWeight: 600, color: C.ink }}>{r.title}</td>
                    <td style={{ ...tdL, whiteSpace: 'nowrap' }}>{r.due ?? '—'}</td>
                    <td style={{ ...tdL, fontSize: 11, color: C.sec }}>{r.note}</td>
                  </tr>))}
              </tbody>
            </table>
          </div>

          <div style={{ fontSize: 12.5, fontWeight: 700, color: C.ink, margin: '12px 0 2px' }}>All jobs &amp; manual triggers</div>
          {opsC.groups.map((g: any) => (
            <details key={g.title} style={{ marginTop: 6, borderTop: `1px solid ${C.hairSoft}`, paddingTop: 6 }}>
              <summary style={{ cursor: 'pointer', fontSize: 12, fontWeight: 700, color: C.navy, userSelect: 'none' }}>▸ {g.title} ({g.jobs.length})</summary>
              <div style={{ marginTop: 6 }}>
                {g.jobs.map((j: any, i: number) => (
                  <div key={i} style={{ fontSize: 11.5, color: C.sec, marginBottom: 5 }}>
                    <b style={{ color: C.ink }}>{j.name}</b> <span style={{ color: C.muted }}>· {j.schedule}</span> — {j.what}
                    {j.cmd && <div><code style={{ display: 'inline-block', marginTop: 2, fontSize: 10.5, background: C.canvas,
                      border: `1px solid ${C.hairSoft}`, borderRadius: 4, padding: '1px 6px', color: C.sec, overflowWrap: 'anywhere' }}>{j.cmd}</code></div>}
                  </div>))}
              </div>
            </details>
          ))}
          <div style={{ fontSize: 10, color: C.faint, marginTop: 8 }}>Commands run on the VPS from /home/arun/quantifyd · full reference doc: docs/LABS_AND_JOBS_REFERENCE.md · registry source: research/111_sensex_manual_mgmt/scripts/ops_center.py</div>
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
        <details style={{ margin: '6px 0 10px' }}>
          <summary style={{ cursor: 'pointer', fontSize: 12, fontWeight: 600, color: C.navy }}>System rules — the 5 paper books (click to expand)</summary>
          <div style={{ fontSize: 11.5, color: C.sec, marginTop: 6, lineHeight: 1.7 }}>
            <div><b>Common mechanic (all 5):</b> sell 1× ATM straddle (strike from live spot at the entry moment, nearest weekly expiry) · combined premium polled ~every 5s · <b>combined-premium SL</b> = exit when CE+PE ≥ (1+SL%)×entry credit on <b>2 consecutive polls</b> (dwell), market exit at the next poll · otherwise hold to the config's exit time (15:26 hard force) · “SL ∅” books still carry a <b>50% disaster backstop</b> · entry skipped if the process starts &gt;15 min late · ₹160/day cost modeled.</div>
            <div style={{ marginTop: 6 }}><b>CSL_TIMEB_NIFTY (8 lots · qty 520 — sec-18b step-2):</b> the <b>optimized time-blocked book (TB-CSL)</b> — per-DTE entry→exit windows + SL from the Lab sweep, <b>frozen 13-AUG</b> (schedule above; weekly Lab regens do NOT move it). 2-unit weight per the portfolio scan.</div>
            <div><b>CSL_TIMEB2_NIFTY (2 lots · qty 130 · PAPER):</b> second-slots evidence book — the sweep's qualifying 2nd windows only: <b>DTE0 (Tue) 13:00→14:00 SL25 · DTE1 (Mon) 10:00→12:00 SL25</b>; no other days. If it tracks OOS, these merge into TB-CSL as a second slot per day (review ~05-SEP).</div>
            <div><b>CSL_TIMEB_SENSEX (8 lots · qty 160 — REAL, notional-parity with NIFTY TB@8L):</b> same TB-CSL frozen schedule on SENSEX; live from 19-Aug, suite Wednesday to paper the same night.</div>
            <div><b>NAS_COMB20 (3 lots · qty 195 · NIFTY):</b> the full CSL-replacement arm — same 09:16→15:20 full-day venue, but a <b>combined-premium SL, per-DTE frozen (DTE0→4: 25/30/30/20/30, sec-15 sweep)</b> replaces NAS's per-leg SLs + trail. Live test of “combined beats per-leg”.</div>
            <div><b>NAS_C20_TRAIL (3 lots · qty 195 · NIFTY):</b> NAS_COMB20 + management — on CSL hit, close only the LOSER leg and <b>trail the winner</b> (exit on ≥30% bounce off its post-trigger low). The ATM-style rescue.</div>
            <div><b>NAS_C20_SHIFT (3 lots · qty 195 · NIFTY):</b> NAS_COMB20 + management — on CSL hit, close both and <b>re-sell a fresh straddle at the new ATM</b> (own 20% CSL, max 3 shifts, none after 14:30). The ATM4-style recenter.</div>
            <div><b>CSL30F_NIFTY (3 lots · qty 195):</b> <b>F = FIXED</b> — the flat un-windowed rule: 09:16→15:20 every day, combined <b>30% SL</b>, no per-DTE windows. Control arm for “do the time-blocks add value over plain CSL30?”.</div>
            <div><b>CSL30F_SENSEX (3 lots · qty 60):</b> the same fixed CSL30 rule on SENSEX.</div>
            <div style={{ marginTop: 6, color: C.faint }}>The three 3-lot books are sized to match the live NAS books for like-for-like daily comparison. Day curves below: BACKTEST = recorded-chain replay of these exact rules · LIVE PAPER records start 14-AUG.</div>
          </div>
        </details>
        {(() => {
          // merge THREE sources, provenance-labeled: BACKTEST (chain replay) < PAPER (live paper) < REAL
          const srcColor = (s: string) => (s === 'REAL' ? C.neg : s === 'BACKTEST' ? C.amber : C.pos);
          const srcBg = (s: string) => (s === 'REAL' ? '#FBEEEE' : s === 'BACKTEST' ? C.amberSoft : '#E7F2EE');
          const srcLbl = (s: string) => (s === 'REAL' ? 'LIVE REAL' : s === 'BACKTEST' ? 'BACKTEST' : 'LIVE PAPER');
          const liveRecs = (((cslPaper?.records) || []) as any[]).filter((r) => r.series && r.series.length > 1)
            .map((r) => ({ ...r, source: r.source || 'PAPER' }));
          const bfRecs = (((cslPBF?.records) || []) as any[]).filter((r) => r.series && r.series.length > 1
            && !liveRecs.some((lr) => lr.day === r.day && (lr.book || lr.sym) === (r.book || r.sym)));
          const recDays: string[] = Array.from(new Set([...liveRecs, ...bfRecs].map((r: any) => r.day)));
          if (cslPLive?.day && Object.values(cslPLive.books || {}).some((b: any) => (b.series || []).length > 1) && !recDays.includes(cslPLive.day)) recDays.push(cslPLive.day);
          recDays.sort();
          const selDay = cslPDay && recDays.includes(cslPDay) ? cslPDay : recDays[recDays.length - 1];
          if (!selDay) return null;
          const curves: { bk: string; pts: [string, number][]; live: boolean; src: string }[] = [];
          liveRecs.filter((r) => r.day === selDay).forEach((r) => curves.push({ bk: r.book || r.sym, pts: r.series, live: false, src: r.source }));
          bfRecs.filter((r) => r.day === selDay).forEach((r) => { if (!curves.some((c2) => c2.bk === (r.book || r.sym))) curves.push({ bk: r.book || r.sym, pts: r.series, live: false, src: 'BACKTEST' }); });
          if (cslPLive?.day === selDay) Object.entries(cslPLive.books || {}).forEach(([bk, b]: any) => {
            if ((b.series || []).length > 1 && !curves.some((c2) => c2.bk === bk)) curves.push({ bk, pts: b.series, live: b.state === 'OPEN', src: 'PAPER' });
          });
          if (!curves.length) return null;
          curves.sort((a, b) => a.bk.localeCompare(b.bk));
          const shown = curves.filter((c2) => cslPVenue === 'ALL' || (cslPVenue === 'SENSEX' ? c2.bk.includes('SENSEX') : !c2.bk.includes('SENSEX')));
          const big = shown.find((c2) => c2.bk === cslPBig);
          const rng = (rs: any[]) => { const ds = Array.from(new Set(rs.map((r) => r.day))).sort(); return ds.length ? `${ds.length}d ${ds[0].slice(5)}→${ds[ds.length - 1].slice(5)}` : null; };
          const bfR = rng(bfRecs); const lpR = rng(liveRecs.filter((r) => r.source !== 'REAL')); const rlR = rng(liveRecs.filter((r) => r.source === 'REAL'));
          return (
            <div style={{ margin: '10px 0' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', marginBottom: 6 }}>
                <span style={{ fontSize: 12, fontWeight: 700, color: C.ink }}>Day P&amp;L curves — all variants</span>
                <button onClick={() => setCslPDay(recDays[Math.max(0, recDays.indexOf(selDay) - 1)])} disabled={recDays.indexOf(selDay) <= 0}
                  style={{ border: `1px solid ${C.hair}`, background: 'transparent', color: C.navy, borderRadius: 6, padding: '1px 9px', fontSize: 12, fontWeight: 800, cursor: 'pointer' }}>◀</button>
                <select value={selDay} onChange={(e) => setCslPDay(e.target.value)}
                  style={{ background: 'transparent', color: C.ink, border: `1px solid ${C.hair}`, borderRadius: 6, padding: '2px 8px', fontSize: 11.5 }}>
                  {recDays.map((d2) => <option key={d2} value={d2}>{d2}</option>)}
                </select>
                <button onClick={() => setCslPDay(recDays[Math.min(recDays.length - 1, recDays.indexOf(selDay) + 1)])} disabled={recDays.indexOf(selDay) >= recDays.length - 1}
                  style={{ border: `1px solid ${C.hair}`, background: 'transparent', color: C.navy, borderRadius: 6, padding: '1px 9px', fontSize: 12, fontWeight: 800, cursor: 'pointer' }}>▶</button>
                <button onClick={() => setCslPModal('day')}
                  style={{ border: `1px solid ${C.navy}`, background: C.navySoft, color: C.navy, borderRadius: 6, padding: '2px 10px', fontSize: 11.5, fontWeight: 700, cursor: 'pointer' }}>⤢ day navigator</button>
                <button onClick={() => setCslPModal('sys')}
                  style={{ border: `1px solid ${C.navy}`, background: C.navySoft, color: C.navy, borderRadius: 6, padding: '2px 10px', fontSize: 11.5, fontWeight: 700, cursor: 'pointer' }}>🗓 by system — all days</button>
                {(['ALL', 'NIFTY', 'SENSEX'] as const).map((v) => (
                  <button key={v} onClick={() => setCslPVenue(v)}
                    style={{ border: `1px solid ${cslPVenue === v ? C.navy : C.hair}`, background: cslPVenue === v ? C.navySoft : 'transparent', color: cslPVenue === v ? C.navy : C.sec, borderRadius: 6, padding: '1px 8px', fontSize: 10.5, fontWeight: 700, cursor: 'pointer' }}>{v === 'ALL' ? 'Both' : v}</button>
                ))}
                <span style={{ fontSize: 10.5, color: C.faint }}>~60s samples · click a curve to expand/collapse · ← → in popup</span>
              </div>
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 8, fontSize: 10 }}>
                {bfR && <span style={{ fontWeight: 800, padding: '1px 7px', borderRadius: 4, color: C.amber, background: C.amberSoft }}>BACKTEST · recorded-chain replay · {bfR}</span>}
                {lpR && <span style={{ fontWeight: 800, padding: '1px 7px', borderRadius: 4, color: C.pos, background: '#E7F2EE' }}>LIVE PAPER · {lpR}</span>}
                {rlR && <span style={{ fontWeight: 800, padding: '1px 7px', borderRadius: 4, color: C.neg, background: '#FBEEEE' }}>LIVE REAL · {rlR}</span>}
                {!lpR && <span style={{ color: C.faint }}>live paper recording starts 14-AUG (books deployed 13-AUG)</span>}
              </div>
              {big && <div style={{ marginBottom: 8 }}>
                <LineChart pts={big.pts} h={260} label={`${big.bk} · ${selDay} · ${big.live ? 'LIVE (open)' : srcLbl(big.src)} — expanded (click tile to collapse)`} />
              </div>}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: 10 }}>
                {shown.map((c2) => (
                  <div key={c2.bk} onClick={() => setCslPBig(cslPBig === c2.bk ? null : c2.bk)}
                    style={{ border: `1px solid ${cslPBig === c2.bk ? C.navy : C.hair}`, borderRadius: 8, padding: '6px 8px', cursor: 'pointer' }}>
                    <div style={{ fontSize: 11, fontWeight: 700, color: C.ink }}>
                      {c2.bk}{' '}
                      {c2.live ? <span style={{ color: C.pos }}>· LIVE</span> :
                        <span style={{ fontSize: 9, fontWeight: 800, padding: '0 5px', borderRadius: 3, color: srcColor(c2.src), background: srcBg(c2.src) }}>{srcLbl(c2.src)}</span>}
                      <span style={{ float: 'right', color: col(c2.pts[c2.pts.length - 1][1]) }}>{inr(c2.pts[c2.pts.length - 1][1])}</span>
                    </div>
                    <LineChart pts={c2.pts} h={90} />
                  </div>
                ))}
              </div>
              {cslPModal && <CslCurvesModal recs={[...liveRecs, ...bfRecs]} mode0={cslPModal} day0={selDay} nasBase={nasBase} onClose={() => setCslPModal(null)} />}
            </div>
          );
        })()}
        {cslPaper && (cslPaper.records || []).length > 0 ? (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead><tr><th style={thL}>Day</th><th style={thL}>Book</th><th style={thL}>Cfg</th><th style={thR}>Strike</th><th style={thR}>Credit</th><th style={thR}>Exit</th><th style={thL}>Reason</th><th style={thR}>P&amp;L</th><th style={thR}>Lots</th></tr></thead>
              <tbody>{cslPaper.records.slice().reverse().map((r: any, i: number) => (
                <tr key={i}><td style={tdL}>{r.day}</td><td style={tdL}>{r.book || r.sym} D{r.dte}{' '}
                  <span style={{ fontSize: 9, fontWeight: 800, padding: '0 4px', borderRadius: 3,
                    color: r.source === 'REAL' ? C.neg : C.pos, background: r.source === 'REAL' ? '#FBEEEE' : '#E7F2EE' }}>
                    {r.source === 'REAL' ? 'REAL' : 'PAPER'}</span></td><td style={tdL}>{r.cfg}</td>
                  <td style={tdR}>{r.strike}</td><td style={tdR}>{r.credit}</td><td style={tdR}>{r.exit_comb} <span style={{ color: C.faint }}>@{String(r.exit_ts || '').slice(0, 5)}</span></td>
                  <td style={{ ...tdL, color: String(r.reason).startsWith('SL') ? C.neg : C.muted }}>{r.reason}</td>
                  <td style={{ ...tdR, fontWeight: 700, color: col(r.pnl) }}>{inr(r.pnl)}</td><td style={tdR}>{r.lots}</td></tr>))}
              </tbody>
            </table>
          </div>
        ) : (
          <div style={{ fontSize: 12, color: C.faint }}>No completed paper trades yet — first entries fire Friday 14-AUG (NIFTY 10:00→12:00 · SENSEX 10:30→12:00). Records appear here automatically.</div>
        )}
        <OpsInfo rows={[
          ['Executor (the 7 books)', 'cron 09:12 Mon–Fri; NAS_COMB20 + CSL_TIMEB_NIFTY place REAL orders (marketable-LIMIT), rest paper; log /tmp/csl_paper.log', 'tail -f /tmp/csl_paper.log'],
          ['Safe dry-run (no orders)', 'gates + legs + margin check', 'venv/bin/python3 research/111_sensex_manual_mgmt/scripts/csl_paper_exec.py --probe'],
          ['History backfill', 'nightly 15:40 (recorded-chain replay, ~75 min; live records always win)', 'setsid nohup venv/bin/python3 research/111_sensex_manual_mgmt/scripts/csl_paper_backfill.py > /tmp/csl_backfill.log 2>&1 &'],
          ['Desktop popups', 'CSL feed (executor events) + NAS feed (1-min cron); laptop watcher scripts/csl_alert_watcher.pyw auto-starts at login'],
          ['Kill levers', 'nas_manual_freeze.flag (blocks ALL orders) · master mode paper (whole stack) · per-book mode flag · POST /api/nas/kill-switch (suite)'],
        ]} />
      </section>

      {expLab && (
        <section id="expiry-lab" style={{ ...card, marginTop: 14 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
            <span style={{ fontSize: 15, fontWeight: 800, color: C.navy }}>⏱ Expiry-Afternoon Lab</span>
            <span style={{ fontSize: 11, color: C.sec }}>research/125 · DTE0 afternoon straddle slots · re-assessed Tue+Thu 16:05 · regen {expLab.generated_at}</span>
          </div>
          <div style={{ fontSize: 11, color: C.sec, margin: '6px 0' }}>
            Spot-ATM straddle on expiry afternoons. The old 13:45→15:00 idea is real but sub-optimal: the money is
            13:15–13:30 → 14:15–15:00, before the last-hour gamma storm (variability triples after 14:30, drift turns positive — the late IV pop).
            All figures net (r/123 cost model), sized as shown. TimeB2 = the live Tuesday slot.
          </div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ borderCollapse: 'collapse', fontSize: 12, minWidth: 620 }}>
              <thead><tr style={{ color: C.sec, textAlign: 'left' }}>
                <th style={{ padding: '3px 10px' }}>Slot</th><th>SL</th><th>Lots</th><th>Mean/day</th><th>Win</th><th>MaxDD</th><th>Ratio</th><th>n</th></tr></thead>
              <tbody>
                {(expLab.winners || []).map((w: any, i: number) => (
                  <tr key={i} style={{ borderTop: `1px solid ${C.hair}`, fontWeight: i === 0 ? 700 : 400 }}>
                    <td style={{ padding: '3px 10px' }}>{w.sym} {w.entry}→{w.exit}{i === 0 ? ' · TimeB2' : ''}</td>
                    <td>{String(w.sl) === 'none' ? 'none+50%BS' : `CSL${w.sl}%`}</td><td>{w.lots}L</td>
                    <td style={{ color: w.mean >= 0 ? '#22a06b' : '#e5484d' }}>{w.mean >= 0 ? '+' : ''}{w.mean.toLocaleString('en-IN')}</td>
                    <td>{w.win}%</td><td>{w.maxdd.toLocaleString('en-IN')}</td><td>{w.ratio}</td><td>{w.n}</td>
                  </tr>))}
                {(expLab.algotest_ref || []).map((w: any, i: number) => (
                  <tr key={'r' + i} style={{ borderTop: `1px solid ${C.hair}`, color: C.sec }}>
                    <td style={{ padding: '3px 10px' }}>{w.sym} {w.entry}→{w.exit} (old AlgoTest slot)</td>
                    <td>best SL{w.sl}</td><td>—</td><td>{w.mean_lot >= 0 ? '+' : ''}{w.mean_lot}/lot</td>
                    <td>{w.win}%</td><td>—</td><td>{w.ratio}</td><td>—</td>
                  </tr>))}
              </tbody>
            </table>
          </div>
          {(expLab.live_vs_model || []).length > 0 && (
            <div style={{ marginTop: 8, fontSize: 12 }}>
              <span style={{ fontWeight: 700, color: C.navy }}>TimeB2 live days vs model:</span>{' '}
              {(expLab.live_vs_model || []).map((r: any) => `${r.day}: live ${r.live >= 0 ? '+' : ''}${r.live} vs model ${r.model ?? 'n/a'} (${r.reason})`).join(' · ')}
            </div>
          )}
          {(expLab.runs || []).length > 0 && (
            <div style={{ marginTop: 6, fontSize: 11, color: C.sec }}>
              Assessment history: {(expLab.runs || []).slice(-5).map((r: any) => `${r.at} — ${r.verdict}`).join(' · ')}
            </div>
          )}
          <OpsInfo rows={[
            ['Cadence', 'Tue+Thu 16:05 IST cron re-runs the sweep on all recorded expiry days, re-scores the frozen slots, flags DRIFT/WEAK, appends run history'],
            ['Manual run', 'venv/bin/python3 research/125_expiry_afternoon_straddle/scripts/expiry_lab_assessment.py'],
            ['Live slot', 'TimeB2: NIFTY Tue 13:15→14:30 CSL30 8L — one-shot runner timeb2_oneshot.py (results/timeb2_live_days.json)'],
            ['Study docs', 'research/125_expiry_afternoon_straddle/ — STATUS-MD + results/RESULTS.md · verdict: SIGNAL, paper/live-forward validating'],
          ]} />
        </section>
      )}

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
              {chip(C.amberSoft, C.amber, 'BACKTEST · replay')}
              {chip(C.amberSoft, C.amber, 'weekly regen · Fri 15:45 IST')}
              <span style={{ marginLeft: 'auto', fontSize: 11, color: C.muted }}>updated {cslCfg.generated_at}</span>
            </div>
            <div style={{ fontSize: 11, color: C.muted, marginBottom: 8 }}>
              Basis — NIFTY: <b>{(cslCfg.meta?.NIFTY?.days) ?? '—'} days</b> ({cslCfg.meta?.NIFTY?.from} → {cslCfg.meta?.NIFTY?.to}) @ {cslCfg.meta?.NIFTY?.lots} lots (qty {cslCfg.meta?.NIFTY?.qty}) ·
              SENSEX: <b>{(cslCfg.meta?.SENSEX?.days) ?? '—'} days</b> ({cslCfg.meta?.SENSEX?.from} → {cslCfg.meta?.SENSEX?.to}) @ {cslCfg.meta?.SENSEX?.lots} lots (qty {cslCfg.meta?.SENSEX?.qty}) ·
              3-sec dwell mechanic · ATM at entry moment
            </div>
            <div style={{ fontSize: 11, color: C.muted, marginBottom: 8 }}>
              Costs charged — <b>NET, and MEASURED</b> from 443 real live fills (Kite average_price vs
              chain LTP at the same minute). Entry sells land ~0.23pt <i>favourable</i> (booked at 0); a
              time-exit buy-back costs ~{cslCfg.cost_model?.slip_time ?? 0.18}pt per leg-side; an <b>SL-exit costs ~{cslCfg.cost_model?.slip_stop ?? 6.55}pt per leg-side</b> —
              slippage lives almost entirely in stop-outs, so a window that time-exits is nearly free and a
              stop-heavy book is not. Charged per outcome, plus the exact Zerodha rate card. A typical
              time-exit round trip ≈ <b>₹{cslCfg.meta?.NIFTY?.cost ?? '—'}</b> NIFTY · <b>₹{cslCfg.meta?.SENSEX?.cost ?? '—'}</b> SENSEX.
            </div>
            <OpsInfo rows={[
              ['Weekly sweep', 'Fridays 15:45 IST — entry×exit×SL grid per DTE per venue on all recorded days (informational: the live books’ FROZEN config does NOT move with it)', 'setsid nohup venv/bin/python3 -u research/111_sensex_manual_mgmt/scripts/entry_exit_sweep.py > /tmp/eesweep.log 2>&1 &'],
              ['Monitor a running sweep', 'progress + final tables', 'tail -f /tmp/eesweep.log'],
            ]} />
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
            {b && (
              <div style={{ fontSize: 11, color: C.muted, marginTop: 6 }}>
                Cost sensitivity — mean/day per DTE at 0.5x / <b>1x (measured, shown above)</b> / 2x slippage:{' '}
                {dtes.map((k) => {
                  const cs: any = b[k];
                  if (!cs?.sens) return null;
                  return (
                    <span key={k} style={{ marginRight: 10 }}>
                      DTE{k} <b>{inr(cs.sens['0.5x'])}</b>/<b>{inr(cs.sens['1.0x'])}</b>/<b>{inr(cs.sens['2.0x'])}</b>
                    </span>
                  );
                })}
                — a cell that only survives at 0.5x measured slippage is not tradable.
              </div>
            )}
          </section>
        );
      })()}

      {condor && (
        <section id="condor" style={{ ...card, marginTop: 14, borderColor: C.amber, scrollMarginTop: 70 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6, flexWrap: 'wrap' }}>
            <span style={{ fontSize: 16, fontWeight: 700, color: C.ink }}>Wed&#8594;Fri Iron Condor</span>
            {chip(C.amberSoft, C.amber, `PAPER · book ${condor.lots} lots · shown at 10`)}
            <a href="/app/backtest/fardte-rescue" style={{ textDecoration: 'none' }}>
              {chip(C.navySoft, C.navy, 'research/80 ↗')}
            </a>
            {condor.closed_trades > 0 && (
              <span style={{ marginLeft: 'auto', fontSize: 12, color: C.muted }}>
                closed {condor.closed_trades} · <b style={{ color: col(condor.closed_total_pnl * (650 / (condor.qty || 130))) }}>{inr(Math.round(condor.closed_total_pnl * (650 / (condor.qty || 130))))}</b>
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
                    <td style={ctd2}>{Math.round((l.qty || 0) * (650 / (condor.qty || 130)))}</td>
                    <td style={ctd2}>{l.entry}</td>
                    <td style={ctd2}>{l.ltp}</td>
                    <td style={{ ...ctd2, fontWeight: 700, color: col(l.pnl) }}>{inr(Math.round((l.pnl || 0) * (650 / (condor.qty || 130))))}</td>
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
                <tbody>{condor.history.map((h: any, i: number) => {
                  // premium points are per-unit; only rupee P&L scales with size
                  const k = 650 / (condor.qty || 130);
                  return (
                  <React.Fragment key={i}>
                    <tr>
                      <td style={{ ...ctd2, textAlign: 'left' }}>{h.entry_day}</td>
                      <td style={{ ...ctd2, textAlign: 'left' }}>{h.exit_day}</td>
                      <td style={ctd2}>{h.credit}</td>
                      <td style={ctd2}>{h.exit_value}</td>
                      <td style={{ ...ctd2, fontWeight: 700, color: col(h.pnl) }}>{inr(Math.round(h.pnl * k))}</td>
                    </tr>
                    {Array.isArray(h.legs) && h.legs.length > 0 && (
                      <tr>
                        <td colSpan={5} style={{ padding: '0 0 10px 14px' }}>
                          <details>
                            <summary style={{ cursor: 'pointer', fontSize: 10.5, color: C.muted, listStyle: 'none' }}>
                              &#9656; {h.legs.length} legs · spot {h.spot_at_entry} &rarr; {h.spot_at_exit} · expiry {h.expiry}
                            </summary>
                            <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: 4 }}>
                              <thead><tr>
                                <th style={{ ...cth2, textAlign: 'left' }}>LEG</th><th style={cth2}>STRIKE</th>
                                <th style={cth2}>QTY</th><th style={cth2}>ENTRY</th><th style={cth2}>EXIT</th>
                                <th style={cth2}>P&amp;L</th>
                              </tr></thead>
                              <tbody>{h.legs.map((l: any, j: number) => (
                                <tr key={j}>
                                  <td style={{ ...ctd2, textAlign: 'left', fontWeight: 700,
                                               color: l.side === 'SELL' ? C.neg : C.pos }}>{l.side} {l.type}</td>
                                  <td style={ctd2}>{l.strike}</td>
                                  <td style={ctd2}>{Math.round((l.qty || 0) * k)}</td>
                                  <td style={ctd2}>{l.entry}</td>
                                  <td style={ctd2}>{l.ltp}</td>
                                  <td style={{ ...ctd2, fontWeight: 700, color: col(l.pnl) }}>{inr(Math.round((l.pnl || 0) * k))}</td>
                                </tr>))}</tbody>
                            </table>
                          </details>
                        </td>
                      </tr>)}
                  </React.Fragment>);
                })}</tbody>
              </table>
            </details>
          )}
          <details style={{ marginTop: 10 }}>
            <summary style={{ cursor: 'pointer', fontSize: 12, fontWeight: 600, color: C.navy, listStyle: 'none' }}>
              &#9656; The ruleset — Wed&rarr;Fri iron condor
            </summary>
            <div style={{ fontSize: 11.5, color: C.sec, lineHeight: 1.85, marginTop: 6 }}>
              <div><b>Why it exists:</b> NAS-OPT trades the days near expiry and leaves capital
                idle from Wednesday to Friday. research/80 tested five ways to use those days;
                four died and this one worked — a condor, entered Wednesday, out Friday.</div>
              <div><b>Structure:</b> iron condor — sell a call and a put roughly <b>0.8% out of the
                money</b>, then buy each wing a further <b>1.0% beyond its own short</b> (not 1.0%
                from spot). That makes each vertical about <b>250 points</b> wide. Four legs,
                defined risk, no naked side. Example, 26-Aug cycle at spot 24,277: short CE 24450
                (+0.71%) / long CE 24700 (+1.74%), short PE 24100 (−0.73%) / long PE 23850 (−1.76%).</div>
              <div><b>Entry:</b> Wednesday close. <b>Exit:</b> Friday close. Flat over the weekend,
                and flat before Monday, so it never competes with NAS-OPT for margin.</div>
              <div><b>Size:</b> the book runs <b>{condor.lots} lots</b> (qty {condor.qty}); figures
                on this card are scaled to <b>10 lots</b> so it compares with the other books.
                Premium points are per-unit and unchanged — only rupee P&amp;L scales.</div>
              <div><b>The wings are the stop</b> — there is no separate stop-loss rule because
                there does not need to be one: maximum loss is the spread width less the credit,
                and it is <b>known at entry</b> rather than depending on a stop firing in time.
                On the credits seen so far (68–107 pts) that is <b>≈145–180 points</b>, i.e.
                <b>≈₹19,000–24,000 at the book's 2 lots</b> and ₹93,000–₹1.18L at 10 lots.
                Worth holding against the whole live book's max drawdown to date of ₹36,082 —
                the tail is bounded, but it is not small, and no cycle has yet come near it
                (worst so far −₹2,856, about 12% of the cap).</div>
              <div style={{ color: C.faint }}>Study: <a href="/app/backtest/fardte-rescue"
                style={{ color: C.navy }}>research/80 — rescuing the far-from-expiry days</a>{' '}
                (~11 years, Calmar 1.63, the best of the five ideas tested).</div>
            </div>
          </details>

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
                  <span style={{ fontWeight: 700, color: C.ink }}>V1 variant · 30% combined-premium SL</span>
                  <span style={{ marginLeft: 8, fontSize: 10.5, fontWeight: 700, letterSpacing: 0.04, textTransform: 'uppercase', color: C.amber, border: `1px solid ${C.amber}`, borderRadius: 3, padding: '1px 6px' }}>not deployed</span>
                  {chip(C.amberSoft, C.amber, 'BACKTEST · recorded chain')}
                  <button onClick={() => setSl30Modal(true)} style={{ cursor: 'pointer', border: `1px solid ${C.hair}`, background: C.surface, color: C.navy, borderRadius: 6, padding: '2px 10px', fontSize: 11.5, fontWeight: 700 }}>⤢ deep-dive</button>
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
                {sl30Modal && (() => {
                  const days: string[] = sl30.days || [];
                  const dSel = sl30Day && days.includes(sl30Day) ? sl30Day : days[days.length - 1];
                  const idx = days.indexOf(dSel);
                  const pd = (sl30.per_day || {})[dSel] || {};
                  let cum2 = 0, pk2 = 0;
                  const ddPts: [string, number][] = (sl30.book_curve || []).map(([d2, c2]: any) => { pk2 = Math.max(pk2, c2); return [d2, Math.round(c2 - pk2)]; });
                  return (
                    <div onClick={() => setSl30Modal(false)}
                      style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.5)', zIndex: 1000, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 20 }}>
                      <div onClick={(e) => e.stopPropagation()}
                        style={{ background: C.surface, border: `1px solid ${C.hair}`, borderRadius: 12, padding: '16px 20px', width: 'min(1250px, 96vw)', maxHeight: '92vh', overflowY: 'auto' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap', marginBottom: 6 }}>
                          <b style={{ fontSize: 15, color: C.ink }}>CSL30 deep-dive</b>
                          {chip(C.amberSoft, C.amber, `${s.n} days · 10 lots (qty 650) · refreshed daily 15:40 IST`)}
                          <span style={{ fontSize: 12 }}>total <b style={{ color: col(s.total) }}>{inr(s.total)}</b> · mean/day <b style={{ color: col(s.mean) }}>{inr(s.mean)}</b> · win {s.win}% · SL-hit {s.sl_hit_pct}% · maxDD <b style={{ color: C.neg }}>{inr(s.maxdd)}</b></span>
                          <button onClick={() => setSl30Modal(false)} style={{ marginLeft: 'auto', cursor: 'pointer', border: `1px solid ${C.hair}`, background: C.surface, borderRadius: 6, padding: '2px 10px' }}>✕</button>
                        </div>
                        <LineChart pts={sl30.book_curve} h={190} label={`cumulative P&L · ${s.n} days · recorded chain`} />
                        {ddPts.length > 1 && <LineChart pts={ddPts} h={90} label={`drawdown · trough ${inr(s.maxdd)}`} />}
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', margin: '12px 0 4px' }}>
                          <b style={{ fontSize: 12.5, color: C.ink }}>Day navigator</b>
                          <button disabled={idx <= 0} onClick={() => setSl30Day(days[idx - 1])} style={{ cursor: idx > 0 ? 'pointer' : 'not-allowed', border: `1px solid ${C.hair}`, background: C.surface, borderRadius: 6, padding: '2px 10px' }}>◀ prev</button>
                          <select value={dSel} onChange={(e) => setSl30Day(e.target.value)}
                            style={{ background: 'transparent', color: C.ink, border: `1px solid ${C.hair}`, borderRadius: 6, padding: '2px 8px', fontSize: 12 }}>
                            {days.map((d2) => <option key={d2} value={d2}>{d2}</option>)}
                          </select>
                          <button disabled={idx >= days.length - 1} onClick={() => setSl30Day(days[idx + 1])} style={{ cursor: idx < days.length - 1 ? 'pointer' : 'not-allowed', border: `1px solid ${C.hair}`, background: C.surface, borderRadius: 6, padding: '2px 10px' }}>next ▶</button>
                          <span style={{ fontSize: 12, color: C.muted }}>
                            POSITION: sold {pd.strike} straddle · credit ₹{pd.credit} · DTE{pd.dte} · peak +{pd.peak_pct}% ·
                            {pd.stopped && pd.exit ? ` SL exit ${pd.exit.time}` : ' held to close'} · P&L <b style={{ color: col(pd.final || 0) }}>{inr(pd.final || 0)}</b>
                          </span>
                        </div>
                        {pd.series && pd.series.length > 1 &&
                          <LineChart pts={pd.series} h={200}
                            marker={pd.stopped && pd.exit ? { time: pd.exit.time, pnl: pd.exit.pnl, text: 'SL' } : null}
                            label={`${dSel} intraday P&L (5-min marks, display only) · low ${inr(pd.low || 0)} · high ${inr(pd.high || 0)}`} />}
                        <div style={{ fontSize: 10.5, color: C.faint, marginTop: 6 }}>Click any row in the card's "Completed days" table to jump here directly · data regenerates every trading day post-close (15:40 IST cron).</div>
                      </div>
                    </div>
                  );
                })()}
                <details style={{ marginTop: 8 }}>
                  <summary style={{ cursor: 'pointer', fontSize: 11.5, fontWeight: 600, color: C.navy, listStyle: 'none' }}>▸ Completed days ({comp.length})</summary>
                  <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: 6 }}>
                    <thead><tr><th style={{ ...ecth, textAlign: 'left' }}>Day</th><th style={ecth}>DTE</th><th style={ecth}>Peak+%</th><th style={{ ...ecth, textAlign: 'left' }}>Exit</th><th style={ecth}>P&amp;L</th></tr></thead>
                    <tbody>{comp.map((t: any, i: number) => (
                      <tr key={i} onClick={() => { setSl30Day(t.day); setSl30Modal(true); }} style={{ cursor: 'pointer' }}
                        title="click: open this day's position + intraday curve in the deep-dive">
                        <td style={{ ...ectd, textAlign: 'left', color: C.navy }}>{t.day} ↗</td><td style={ectd}>DTE{t.dte}</td>
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
            <a href="/app/backtest/v2-nifty-ironfly-sl-vix" style={{ textDecoration: 'none' }}
               title="The stop-loss x VIX optimization these parameters come from — the AlgoTest runs, rebuilt and published">
              {chip(C.navySoft, C.navy, 'research/60 · study ↗')}
            </a>
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

          {/* ---- the evidence this book rests on: two engines, one table ---- */}
          <details style={{ marginBottom: 10 }}>
            <summary style={{ cursor: 'pointer', fontSize: 12, fontWeight: 700, color: C.navy, listStyle: 'none' }}>
              &#9656; The evidence &mdash; 7.3 years, two independent engines
            </summary>

            <div style={{ overflowX: 'auto', marginTop: 8 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11.5 }}>
                <thead>
                  <tr>
                    <th style={evTh}>&nbsp;</th>
                    <th style={evTh}>AlgoTest &mdash; the live spec</th>
                    <th style={evTh}>Our bhavcopy engine</th>
                  </tr>
                </thead>
                <tbody>
                  <tr><td style={evK}>what it models</td>
                      <td style={evV}>full rules, incl. the 2% move-stop + 40% profit target</td>
                      <td style={evV}>structure only &mdash; EOD closes, <b>no stop, no PT</b></td></tr>
                  <tr><td style={evK}>data path</td>
                      <td style={evV}>AlgoTest historical chain</td>
                      <td style={evV}>NSE bhavcopy (nse_options_bhav), look-ahead-free</td></tr>
                  <tr><td style={evK}>window</td>
                      <td style={evV}>2019-02 &rarr; 2026-05 (7.3y)</td>
                      <td style={evV}>7 years</td></tr>
                  <tr><td style={evK}>trades</td><td style={evV}>204 (VIX &ge; 13)</td><td style={evV}>&mdash;</td></tr>
                  <tr><td style={evK}>net P&amp;L</td>
                      <td style={{ ...evV, fontWeight: 700, color: C.pos }}>+&#8377;8,80,110</td>
                      <td style={{ ...evV, fontWeight: 700, color: C.pos }}>+&#8377;6,93,000</td></tr>
                  <tr><td style={evK}>max drawdown</td>
                      <td style={{ ...evV, fontWeight: 700 }}>&minus;&#8377;1,16,834</td>
                      <td style={{ ...evV, fontWeight: 700, color: C.neg }}>&minus;&#8377;4,87,000</td></tr>
                  <tr><td style={evK}>Calmar</td>
                      <td style={{ ...evV, fontWeight: 700 }}>1.03</td>
                      <td style={evV}>0.19</td></tr>
                  <tr><td style={evK}>green years</td><td style={evV}>7 / 8</td><td style={evV}>&mdash;</td></tr>
                </tbody>
              </table>
            </div>

            <div style={{ marginTop: 8, padding: '8px 10px', borderRadius: 8,
                          background: C.amberSoft, border: '1px solid #E7D8A8',
                          fontSize: 11.5, color: C.sec, lineHeight: 1.7 }}>
              <b style={{ color: C.amber }}>Read the disagreement, do not average it.</b> The two engines
              agree on the <b>edge</b> (+&#8377;8.80L vs +&#8377;6.93L) and disagree on <b>drawdown</b>
              (&minus;&#8377;1.17L vs &minus;&#8377;4.87L). That is expected: our engine prices daily closes,
              so it cannot see the intraday 2% move-stop or the 40% profit target and holds through
              falls the live rules would have cut. So the <b>edge is cross-validated on two data
              paths; the drawdown control is single-source</b>. Sizing on Calmar 1.03 trusts one engine.
            </div>

            <div style={{ overflowX: 'auto', marginTop: 10 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11.5 }}>
                <thead><tr><th style={evTh} colSpan={2}>What it takes, and what it pays &mdash; 10 lots (qty 650)</th></tr></thead>
                <tbody>
                  <tr><td style={evK}>capital required</td>
                      <td style={evV}><b>&#8377;8,24,580</b> &mdash; &#8377;82,458/lot, verified Zerodha
                          SPAN+exposure via the Kite margin API (2026-06-08). A naked straddle is
                          &#8377;2,10,088/lot, i.e. 2.5&times; more.</td></tr>
                  <tr><td style={evK}>CAGR</td>
                      <td style={evV}><b>~10.5%</b> on &#8377;8.25L compounding &middot; simple return on
                          margin <b>14.6%/yr</b> &middot; on 1.5&times; buffered working capital ~9.7%/yr</td></tr>
                  <tr><td style={evK}>mean P&amp;L / trade</td>
                      <td style={evV}><b>+&#8377;4,314</b> (&#8377;8,80,110 &divide; 204 trades)</td></tr>
                  <tr><td style={evK}>win rate</td>
                      <td style={evV}>56% on the raw 273-trade platform run, before the VIX floor</td></tr>
                  <tr><td style={evK}>avg risk / trade</td>
                      <td style={evV}>Loss is <b>bounded by construction</b> &mdash; the &plusmn;2%-of-ATM wings
                          cap it at (wing width &minus; credit), known at entry. <b>Worst observed trade
                          &minus;&#8377;71,000</b>; worst week &minus;&#8377;1.84L. The wings, not the stop,
                          are the real risk control.</td></tr>
                  <tr><td style={evK}>drawdown vs capital</td>
                      <td style={evV}>&minus;&#8377;1.17L on &#8377;8.25L margin = <b>14%</b> (AlgoTest).
                          On our unmanaged EOD reproduction it is &minus;&#8377;4.87L = 59%.</td></tr>
                </tbody>
              </table>
            </div>

            <div style={{ fontSize: 11, color: C.faint, marginTop: 8, lineHeight: 1.7 }}>
              The <b>CPR compression gate this engine runs</b> (skip when prior-day CPR &lt; 0.10%) is
              correct <i>for this structure</i>, and both engines agree: AlgoTest +&#8377;8.1L &rarr; +&#8377;11.0L
              (Calmar 0.95 &rarr; 1.59), ours +&#8377;6.93L &rarr; +&#8377;13.79L. The same filter <i>hurts</i> the
              separate DTE-3 fly in the look-ahead audit &mdash; it is structure-specific, not universal.
              {' '}<a href="/app/backtest/v2-nifty-ironfly-sl-vix" style={{ color: C.navy }}>V2 study &#8599;</a>
              {' · '}<a href="/app/backtest/nifty-straddle-lookahead-audit" style={{ color: C.navy }}>look-ahead audit &amp; independent reproduction &#8599;</a>
            </div>
          </details>

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
                  <th style={ecth} title="Furthest the underlying travelled from the entry spot while the position was open. A stop level can only fire if the move reached it.">Moved</th>
                  {(stopMx?.stops || []).map((s: number) => (
                    <th key={s} style={ecth}
                        title={`What a ${(s * 100).toFixed(1)}% move-stop would have realised on this same position, `
                               + `rebuilt from the recorded option chain. Live stop is 2.0%.`}>
                      {(s * 100).toFixed(1)}%
                    </th>))}
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
                    {(() => {
                      const row = (stopMx?.trades || []).find((x: any) => x.pos === t.id);
                      const mv = row?.max_move_held;
                      return (
                        <td style={{ ...ectd, color: C.sec }}
                            title={row?.max_move_after != null
                              ? `${mv?.toFixed(2)}% while held · ${row.max_move_after.toFixed(2)}% after the exit`
                              : undefined}>
                          {mv != null ? `${mv.toFixed(2)}%` : '—'}
                        </td>);
                    })()}
                    {(stopMx?.stops || []).map((s: number) => {
                      const row = (stopMx.trades || []).find((x: any) => x.pos === t.id);
                      const cell = row?.stops?.[String(s)];
                      if (!cell) return <td key={s} style={{ ...ectd, color: C.faint }}>—</td>;
                      if (cell.identical) return (
                        <td key={s} style={{ ...ectd, color: C.faint }}
                            title="This level never triggered — the move did not reach it — so the trade is unchanged.">
                          {inr(cell.pnl)}
                        </td>);
                      if (cell.no_data) return (
                        <td key={s} style={{ ...ectd, color: C.faint }} title={cell.note}>no data</td>);
                      // colour against the live outcome, not against zero — the question is
                      // whether this level would have been better, not whether it made money
                      return (
                        <td key={s} style={{ ...ectd, fontWeight: 600, color: col(cell.vs_actual) }}
                            title={`${cell.at} · spot ${cell.spot} · vs live ${inr(cell.vs_actual)}`}>
                          {inr(cell.pnl)}
                        </td>);
                    })}
                  </tr>))}</tbody>
                {stopMx && (
                  <tfoot><tr>
                    <td style={{ ...ectd, borderTop: `1px solid ${C.ink}` }} />
                    <td style={{ ...ectd, borderTop: `1px solid ${C.ink}`, textAlign: 'left',
                                 fontWeight: 700, color: C.ink }}>Book net</td>
                    <td style={{ ...ectd, borderTop: `1px solid ${C.ink}` }} colSpan={7} />
                    <td style={{ ...ectd, borderTop: `1px solid ${C.ink}`, fontWeight: 700,
                                 color: col(v2eng.closed_total_pnl) }}>
                      {inr(v2eng.closed_total_pnl)}
                    </td>
                    {(stopMx.stops || []).map((s: number) => {
                      // the variant's OWN net: its P&L where it would have changed the
                      // trade, the live P&L where it would not. Not a delta.
                      let net = 0;
                      (v2eng.closed || []).forEach((tr: any) => {
                        const cell = (stopMx.trades || [])
                          .find((x: any) => x.pos === tr.id)?.stops?.[String(s)];
                        net += (cell && cell.pnl != null) ? cell.pnl : (tr.pnl || 0);
                      });
                      const diff = net - (v2eng.closed_total_pnl || 0);
                      return (
                        <td key={s} style={{ ...ectd, borderTop: `1px solid ${C.ink}`,
                                             fontWeight: 700, color: col(net) }}>
                          {inr(Math.round(net))}
                          <div style={{ fontWeight: 400, fontSize: 10, color: col(diff) }}>
                            {inr(Math.round(diff))} vs live
                          </div>
                        </td>);
                    })}
                  </tr></tfoot>)}
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
                {stopMx && (
                  <div style={{ color: C.faint, marginTop: 6, lineHeight: 1.7 }}>
                    <b style={{ color: C.sec }}>Stop variants</b> — what each level would have realised on these same positions, rebuilt from the option chain recorded since 2026-04-20. A greyed figure equals the live result because that level never triggered — check it against the Moved column. “no data” = a gap exit stamped before the recorder’s first bar, so the decisive minutes are missing.
                    <div style={{ marginTop: 3 }}>
                      {(stopMx.stops || []).map((s: number) => {
                        const eff = (stopMx.trades || [])
                          .map((r: any) => r.stops?.[String(s)]?.vs_actual)
                          .filter((v: any) => v != null);
                        const tot = eff.reduce((a: number, b: number) => a + b, 0);
                        return (
                          <span key={s} style={{ marginRight: 16 }}>
                            {(s * 100).toFixed(1)}%: changed {eff.length} trade{eff.length === 1 ? '' : 's'},{' '}
                            <b style={{ color: col(tot) }}>{inr(tot)}</b> vs live
                          </span>);
                      })}
                      <span>— the live <b>2.0%</b> is the best of the four on this book. Six
                      trades though, of which only two or three are touched by any one level, so
                      this is a direction and not a verdict.</span>
                    </div>
                  </div>)}
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
