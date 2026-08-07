import { useEffect, useMemo, useState } from 'react';

/**
 * Cumulative P&L curve across past sessions, split NIFTY / SENSEX / Combined.
 *
 * NIFTY  = sum of the three 9:16 systems' cumulative_pnl (/api/nas-916-atm|atm2|atm4/equity-curve).
 * SENSEX = running sum of /api/sensex/sessions day P&L.
 * Pure inline SVG (no chart dep), theme-aware via CSS vars. Frontend-only.
 */

type Pt = { day: string; cum: number };
type Venue = 'combined' | 'nifty' | 'sensex';

const NIFTY_EPS = [
  '/api/nas-916-atm/equity-curve',
  '/api/nas-916-atm2/equity-curve',
  '/api/nas-916-atm4/equity-curve',
];

function fmtInr(v: number): string {
  const s = v < 0 ? '−' : '+';
  return `${s}₹${Math.abs(Math.round(v)).toLocaleString('en-IN')}`;
}

async function jget(url: string): Promise<any> {
  const sep = url.includes('?') ? '&' : '?';
  const r = await fetch(`${url}${sep}t=${Date.now()}`, { cache: 'no-store' });
  return r.json();
}

// Sum of the three NIFTY systems' cumulative_pnl per date (carry-forward missing dates).
async function niftyCurve(): Promise<Pt[]> {
  const maps = await Promise.all(
    NIFTY_EPS.map(async (ep) => {
      const rows = await jget(ep);
      const m = new Map<string, number>();
      for (const r of rows || []) {
        if (r && r.trade_date) m.set(r.trade_date, Number(r.cumulative_pnl) || 0);
      }
      return m;
    }),
  );
  const dates = Array.from(new Set(maps.flatMap((m) => Array.from(m.keys())))).sort();
  const last = [0, 0, 0];
  return dates.map((d) => {
    let cum = 0;
    maps.forEach((m, i) => {
      const v = m.get(d);
      if (v !== undefined) last[i] = v;
      cum += last[i];
    });
    return { day: d, cum };
  });
}

async function sensexCurve(): Promise<Pt[]> {
  const d = await jget('/api/sensex/sessions');
  const days = ((d && d.sessions) || [])
    .map((s: any) => ({ day: s.day as string, pnl: Number(s.pnl) || 0 }))
    .sort((a: { day: string }, b: { day: string }) => a.day.localeCompare(b.day));
  let run = 0;
  return days.map((x: { day: string; pnl: number }) => {
    run += x.pnl;
    return { day: x.day, cum: run };
  });
}

function combine(a: Pt[], b: Pt[]): Pt[] {
  const dates = Array.from(new Set([...a.map((p) => p.day), ...b.map((p) => p.day)])).sort();
  const ma = new Map(a.map((p) => [p.day, p.cum] as const));
  const mb = new Map(b.map((p) => [p.day, p.cum] as const));
  let la = 0;
  let lb = 0;
  return dates.map((d) => {
    if (ma.has(d)) la = ma.get(d) as number;
    if (mb.has(d)) lb = mb.get(d) as number;
    return { day: d, cum: la + lb };
  });
}

// Drop leading flat-zero points so the curve starts where money first moved.
function trimLead(pts: Pt[]): Pt[] {
  let i = 0;
  while (i < pts.length - 1 && pts[i].cum === 0) i += 1;
  return pts.slice(i);
}

const GREEN = '#3fb950';
const RED = '#ef4444';

export default function CumulativePnL() {
  const [nifty, setNifty] = useState<Pt[]>([]);
  const [sensex, setSensex] = useState<Pt[]>([]);
  const [venue, setVenue] = useState<Venue>('combined');
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let dead = false;
    (async () => {
      try {
        const [n, s] = await Promise.all([niftyCurve(), sensexCurve()]);
        if (!dead) {
          setNifty(n);
          setSensex(s);
          setErr(null);
        }
      } catch (e: any) {
        if (!dead) setErr(String(e?.message ?? e));
      }
    })();
    return () => {
      dead = true;
    };
  }, []);

  const pts = useMemo(() => {
    const raw =
      venue === 'nifty' ? nifty : venue === 'sensex' ? sensex : combine(nifty, sensex);
    return trimLead(raw);
  }, [venue, nifty, sensex]);

  const stats = useMemo(() => {
    if (pts.length < 1) return null;
    const cums = pts.map((p) => p.cum);
    const last = cums[cums.length - 1];
    // per-day deltas
    let best = { day: '', v: -Infinity };
    let worst = { day: '', v: Infinity };
    let prev = 0;
    for (const p of pts) {
      const d = p.cum - prev;
      if (d > best.v) best = { day: p.day, v: d };
      if (d < worst.v) worst = { day: p.day, v: d };
      prev = p.cum;
    }
    return { last, peak: Math.max(...cums), trough: Math.min(...cums), best, worst, n: pts.length };
  }, [pts]);

  // SVG geometry
  const W = 920;
  const H = 240;
  const PL = 8;
  const PR = 8;
  const PT = 14;
  const PB = 22;
  const geom = useMemo(() => {
    if (pts.length < 2) return null;
    const ys = pts.map((p) => p.cum);
    const yMin = Math.min(0, ...ys);
    const yMax = Math.max(0, ...ys);
    const span = yMax - yMin || 1;
    const x = (i: number) => PL + (i / (pts.length - 1)) * (W - PL - PR);
    const y = (v: number) => PT + (1 - (v - yMin) / span) * (H - PT - PB);
    const line = pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(1)},${y(p.cum).toFixed(1)}`).join(' ');
    const area = `${line} L${x(pts.length - 1).toFixed(1)},${y(0).toFixed(1)} L${x(0).toFixed(1)},${y(0).toFixed(1)} Z`;
    const zeroY = y(0);
    // ~5 date ticks
    const ticks: Array<{ x: number; label: string }> = [];
    const step = Math.max(1, Math.floor(pts.length / 5));
    for (let i = 0; i < pts.length; i += step) ticks.push({ x: x(i), label: pts[i].day.slice(5) });
    return { line, area, zeroY, x, y, ticks };
  }, [pts]);

  const pos = (stats?.last ?? 0) >= 0;
  const stroke = pos ? GREEN : RED;

  const toggle = (v: Venue, label: string) => (
    <button
      type="button"
      onClick={() => setVenue(v)}
      style={{
        background: venue === v ? 'var(--line)' : 'transparent',
        border: '1px solid var(--line)',
        borderRadius: 6,
        padding: '2px 10px',
        fontSize: 12,
        cursor: 'pointer',
        color: venue === v ? 'var(--ink)' : 'var(--ink-muted)',
        fontWeight: venue === v ? 700 : 500,
      }}
    >
      {label}
    </button>
  );

  return (
    <section
      style={{
        border: '1px solid var(--line)',
        borderRadius: 12,
        padding: '12px 16px 8px',
        margin: '0 0 16px',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, flexWrap: 'wrap', marginBottom: 8 }}>
        <span style={{ fontSize: 14, fontWeight: 800 }}>Cumulative P&amp;L · past sessions</span>
        <span style={{ display: 'inline-flex', gap: 4 }}>
          {toggle('combined', 'Combined')}
          {toggle('nifty', 'NIFTY')}
          {toggle('sensex', 'SENSEX')}
        </span>
        {stats ? (
          <span style={{ marginLeft: 'auto', fontSize: 12, color: 'var(--ink-muted)' }}>
            <span style={{ color: pos ? GREEN : RED, fontWeight: 800, fontSize: 15 }}>{fmtInr(stats.last)}</span>
            {'  '}· {stats.n} sessions · peak {fmtInr(stats.peak)} · trough {fmtInr(stats.trough)}
          </span>
        ) : null}
      </div>

      {err ? (
        <div style={{ color: RED, fontSize: 12, padding: '20px 0' }}>Couldn&apos;t load P&amp;L: {err}</div>
      ) : geom ? (
        <>
          <svg viewBox={`0 0 ${W} ${H}`} width="100%" preserveAspectRatio="none" style={{ display: 'block' }}>
            <defs>
              <linearGradient id="cpnlFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={stroke} stopOpacity="0.20" />
                <stop offset="100%" stopColor={stroke} stopOpacity="0" />
              </linearGradient>
            </defs>
            {/* zero baseline */}
            <line x1={PL} y1={geom.zeroY} x2={W - PR} y2={geom.zeroY} stroke="var(--ink-muted)" strokeOpacity="0.35" strokeDasharray="3 3" />
            <path d={geom.area} fill="url(#cpnlFill)" />
            <path d={geom.line} fill="none" stroke={stroke} strokeWidth={2} strokeLinejoin="round" />
            {geom.ticks.map((t, i) => (
              <text key={i} x={t.x} y={H - 6} fontSize={10} fill="var(--ink-muted)" textAnchor="middle">
                {t.label}
              </text>
            ))}
          </svg>
          {stats ? (
            <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', fontSize: 11.5, color: 'var(--ink-muted)', marginTop: 4 }}>
              <span>best day <span style={{ color: GREEN, fontWeight: 700 }}>{fmtInr(stats.best.v)}</span> ({stats.best.day.slice(5)})</span>
              <span>worst day <span style={{ color: RED, fontWeight: 700 }}>{fmtInr(stats.worst.v)}</span> ({stats.worst.day.slice(5)})</span>
              <span style={{ marginLeft: 'auto', opacity: 0.7 }}>system P&amp;L (live + paper days), as traded</span>
            </div>
          ) : null}
        </>
      ) : (
        <div style={{ color: 'var(--ink-muted)', fontSize: 12, padding: '20px 0' }}>Loading…</div>
      )}
    </section>
  );
}
