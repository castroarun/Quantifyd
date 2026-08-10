import { useEffect, useMemo, useState } from 'react';

/* Purpose-built charts for the momentum book. A generic price chart cannot show WHY a position is
   held or about to be exited — these draw the system's own levels: the Donchian trailing stop the
   book actually exits on, the entry price, and the live close. */

type Bar = { t: string; o: number; h: number; l: number; c: number; v: number; stop: number | null };
type Pos = {
  entry: number | null; entry_date: string | null; stop: number | null;
  qty: number | null; pnl_pct: number | null; days: number | null;
};
type Payload = { updated: string; donchian: number; positions: Record<string, Pos>; symbols: Record<string, Bar[]> };

const WINDOWS = [
  { k: '3M', d: 63 }, { k: '6M', d: 126 }, { k: '1Y', d: 252 },
];

function fmt(n: number) {
  return n >= 1000 ? n.toLocaleString('en-IN', { maximumFractionDigits: 0 }) : n.toFixed(1);
}

function Chart({ sym, bars, pos, days }: { sym: string; bars: Bar[]; pos?: Pos; days: number }) {
  const W = 520, H = 190, L = 46, R = 58, T = 12, B = 20;
  const win = bars.slice(-days);
  if (win.length < 3) return null;
  const stops = win.map((b) => b.stop).filter((x): x is number => x != null);
  const lo = Math.min(...win.map((b) => b.l), ...stops);
  const hi = Math.max(...win.map((b) => b.h));
  const pad = (hi - lo) * 0.06 || 1;
  const y = (v: number) => T + (hi + pad - v) / (hi - lo + 2 * pad) * (H - T - B);
  const x = (i: number) => L + (i / (win.length - 1)) * (W - L - R);
  const last = win[win.length - 1];
  const up = last.c >= win[0].c;
  const bw = Math.max(1.2, (W - L - R) / win.length * 0.62);

  // distance from the live close to the stop — the number that decides the next exit
  const stopNow = pos?.stop ?? last.stop ?? null;
  const dist = stopNow ? ((last.c / stopNow - 1) * 100) : null;
  const danger = dist != null && dist < 2;

  return (
    <div style={{ background: 'var(--panel,#161b22)', border: '1px solid var(--border,#293039)', borderRadius: 10, padding: '10px 12px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 4 }}>
        <b style={{ fontSize: 13.5 }}>{sym}</b>
        <span style={{ fontSize: 12, color: (pos?.pnl_pct ?? 0) >= 0 ? '#1f9d55' : '#c0392b', fontWeight: 700 }}>
          {(pos?.pnl_pct ?? 0) >= 0 ? '+' : ''}{(pos?.pnl_pct ?? 0).toFixed(1)}%
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
        {[0, 0.5, 1].map((f) => {
          const v = lo - pad + f * (hi - lo + 2 * pad);
          return (
            <g key={f}>
              <line x1={L} x2={W - R} y1={y(v)} y2={y(v)} stroke="var(--gl,#222a33)" strokeWidth="1" />
              <text x={L - 5} y={y(v) + 3} textAnchor="end" fontSize="9.5" fill="var(--ink-muted,#6b7581)">{fmt(v)}</text>
            </g>
          );
        })}
        {/* candles */}
        {win.map((b, i) => {
          const g = b.c >= b.o;
          const col = g ? '#2f9152' : '#e0564f';
          return (
            <g key={b.t}>
              <line x1={x(i)} x2={x(i)} y1={y(b.h)} y2={y(b.l)} stroke={col} strokeWidth="0.9" opacity="0.85" />
              <rect x={x(i) - bw / 2} y={y(Math.max(b.o, b.c))} width={bw}
                    height={Math.max(1, Math.abs(y(b.o) - y(b.c)))} fill={col} opacity="0.9" />
            </g>
          );
        })}
        {/* THE DONCHIAN TRAILING STOP — the level the book actually exits on */}
        <path d={win.map((b, i) => (b.stop == null ? '' : `${i === 0 || win[i - 1].stop == null ? 'M' : 'L'}${x(i)},${y(b.stop)}`)).join(' ')}
              fill="none" stroke="#e0564f" strokeWidth="1.6" strokeDasharray="5 3" opacity="0.95" />
        {/* entry price */}
        {pos?.entry != null && (
          <>
            <line x1={L} x2={W - R} y1={y(pos.entry)} y2={y(pos.entry)} stroke="#58a6ff" strokeWidth="1.2" strokeDasharray="2 3" />
            <text x={W - R + 4} y={y(pos.entry) + 3} fontSize="9.5" fill="#58a6ff">entry {fmt(pos.entry)}</text>
          </>
        )}
        {stopNow != null && (
          <text x={W - R + 4} y={y(stopNow) + 3} fontSize="9.5" fill="#e0564f">stop {fmt(stopNow)}</text>
        )}
        <circle cx={x(win.length - 1)} cy={y(last.c)} r="2.8" fill={up ? '#2f9152' : '#e0564f'} />
        <text x={W - R + 4} y={y(last.c) + 3} fontSize="10" fontWeight="700"
              fill={up ? '#2f9152' : '#e0564f'}>{fmt(last.c)}</text>
      </svg>
      <div style={{ display: 'flex', gap: 12, fontSize: 11.5, color: 'var(--ink-muted,#9aa4af)', marginTop: 2 }}>
        <span>held {pos?.days ?? 0}d</span>
        <span style={{ color: danger ? '#e0564f' : undefined, fontWeight: danger ? 700 : 400 }}>
          {dist == null ? '' : `${dist.toFixed(1)}% above stop${danger ? ' — at risk' : ''}`}
        </span>
      </div>
    </div>
  );
}

export default function MomentumCharts() {
  const [d, setD] = useState<Payload | null>(null);
  const [win, setWin] = useState(126);
  useEffect(() => {
    fetch(`/static/momentum_ohlc.json?t=${Math.floor(Date.now() / 300000)}`, { cache: 'no-store' })
      .then((r) => r.json()).then(setD).catch(() => setD(null));
  }, []);
  const held = useMemo(() => (d ? Object.keys(d.positions || {}) : []), [d]);
  if (!d) return <div style={{ fontSize: 13, color: 'var(--ink-muted,#888)' }}>Loading charts…</div>;
  const syms = held.length ? held : Object.keys(d.symbols);
  return (
    <>
      <div style={{ display: 'flex', gap: 14, alignItems: 'center', marginBottom: 10, flexWrap: 'wrap' }}>
        <span style={{ display: 'inline-flex', border: '1px solid var(--border,#293039)', borderRadius: 7, overflow: 'hidden' }}>
          {WINDOWS.map((w) => (
            <button key={w.k} onClick={() => setWin(w.d)}
              style={{
                background: win === w.d ? 'var(--panel2,#1c232c)' : 'transparent',
                color: 'inherit', border: 0, padding: '4px 11px', fontSize: 11.5, fontWeight: 600, cursor: 'pointer',
              }}>{w.k}</button>
          ))}
        </span>
        <span style={{ fontSize: 11.5, color: 'var(--ink-muted,#888)' }}>
          <span style={{ color: '#e0564f' }}>╌╌</span> Donchian-{d.donchian} trailing stop (the exit rule) &nbsp;
          <span style={{ color: '#58a6ff' }}>╌╌</span> entry &nbsp;· data {d.updated}
        </span>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(330px,1fr))', gap: 12 }}>
        {syms.map((s) => (
          <Chart key={s} sym={s} bars={d.symbols[s] || []} pos={d.positions?.[s]} days={win} />
        ))}
      </div>
    </>
  );
}
