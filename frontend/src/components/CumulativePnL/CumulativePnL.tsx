import { useEffect, useMemo, useState } from 'react';

/**
 * Cumulative P&L + drawdown across past sessions, split NIFTY / SENSEX / Combined.
 * Collapsed by default. Date-range presets. Pure inline SVG, theme-aware.
 *
 * NIFTY  = sum of the three 9:16 systems' cumulative_pnl (/api/nas-916-atm|atm2|atm4/equity-curve).
 * SENSEX = running sum of /api/sensex/sessions day P&L.
 *
 * BASIS: as-traded ₹, live + paper days combined. Lot size varied over the history
 * (1 / 2 / 10 lots early → 3 lots now); this curve is NOT lot-normalized.
 */

type Pt = { day: string; cum: number };
type Venue = 'combined' | 'nifty' | 'sensex';
type Range = '1M' | '3M' | '6M' | 'ALL';

const NIFTY_EPS = [
  '/api/nas-916-atm/equity-curve',
  '/api/nas-916-atm2/equity-curve',
  '/api/nas-916-atm4/equity-curve',
];
const RANGE_DAYS: Record<Range, number> = { '1M': 30, '3M': 90, '6M': 180, ALL: 1e9 };
const GREEN = '#3fb950';
const RED = '#ef4444';

function fmtInr(v: number): string {
  const s = v < 0 ? '−' : '+';
  return `${s}₹${Math.abs(Math.round(v)).toLocaleString('en-IN')}`;
}

async function jget(url: string): Promise<any> {
  const sep = url.includes('?') ? '&' : '?';
  const r = await fetch(`${url}${sep}t=${Date.now()}`, { cache: 'no-store' });
  return r.json();
}

async function niftyCurve(): Promise<Pt[]> {
  const maps = await Promise.all(
    NIFTY_EPS.map(async (ep) => {
      const rows = await jget(ep);
      const m = new Map<string, number>();
      for (const r of rows || []) if (r && r.trade_date) m.set(r.trade_date, Number(r.cumulative_pnl) || 0);
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

function trimLead(pts: Pt[]): Pt[] {
  let i = 0;
  while (i < pts.length - 1 && pts[i].cum === 0) i += 1;
  return pts.slice(i);
}

function tms(day: string): number {
  return new Date(`${day}T00:00:00`).getTime();
}

function Chart({ vals, dates, color, height }: { vals: number[]; dates: string[]; color: string; height: number }) {
  const W = 920;
  const PT = 12;
  const PB = 18;
  const PL = 6;
  const PR = 6;
  if (vals.length < 2) return null;
  const yMin = Math.min(0, ...vals);
  const yMax = Math.max(0, ...vals);
  const span = yMax - yMin || 1;
  const x = (i: number) => PL + (i / (vals.length - 1)) * (W - PL - PR);
  const y = (v: number) => PT + (1 - (v - yMin) / span) * (height - PT - PB);
  const line = vals.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
  const zeroY = y(0);
  const area = `${line} L${x(vals.length - 1).toFixed(1)},${zeroY.toFixed(1)} L${x(0).toFixed(1)},${zeroY.toFixed(1)} Z`;
  const step = Math.max(1, Math.floor(vals.length / 5));
  const ticks: Array<{ x: number; label: string }> = [];
  for (let i = 0; i < vals.length; i += step) ticks.push({ x: x(i), label: dates[i].slice(5) });
  const gid = `cpnl${color.replace('#', '')}${height}`;
  return (
    <svg viewBox={`0 0 ${W} ${height}`} width="100%" preserveAspectRatio="none" style={{ display: 'block' }}>
      <defs>
        <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.18" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      <line x1={PL} y1={zeroY} x2={W - PR} y2={zeroY} stroke="var(--ink-muted)" strokeOpacity="0.3" strokeDasharray="3 3" />
      <path d={area} fill={`url(#${gid})`} />
      <path d={line} fill="none" stroke={color} strokeWidth={2} strokeLinejoin="round" />
      {ticks.map((t, i) => (
        <text key={i} x={t.x} y={height - 4} fontSize={10} fill="var(--ink-muted)" textAnchor="middle">
          {t.label}
        </text>
      ))}
    </svg>
  );
}

export default function CumulativePnL() {
  const [nifty, setNifty] = useState<Pt[]>([]);
  const [sensex, setSensex] = useState<Pt[]>([]);
  const [open, setOpen] = useState(false);
  const [venue, setVenue] = useState<Venue>('combined');
  const [range, setRange] = useState<Range>('ALL');
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

  const full = useMemo(() => {
    const raw = venue === 'nifty' ? nifty : venue === 'sensex' ? sensex : combine(nifty, sensex);
    return trimLead(raw);
  }, [venue, nifty, sensex]);

  // headline total is always the all-time last (independent of the zoom range)
  const headline = full.length ? full[full.length - 1].cum : 0;

  // drawdown over the full series (all-time running peak), then window to the range
  const view = useMemo(() => {
    if (full.length === 0) return { pts: [] as Pt[], dd: [] as number[] };
    let peak = -Infinity;
    const ddFull = full.map((p) => {
      peak = Math.max(peak, p.cum);
      return p.cum - peak;
    });
    let lo = 0;
    if (range !== 'ALL' && full.length > 1) {
      const cut = tms(full[full.length - 1].day) - RANGE_DAYS[range] * 86400000;
      lo = full.findIndex((p) => tms(p.day) >= cut);
      if (lo < 0) lo = 0;
    }
    return { pts: full.slice(lo), dd: ddFull.slice(lo) };
  }, [full, range]);

  const stats = useMemo(() => {
    const pts = view.pts;
    if (pts.length < 1) return null;
    const first = pts[0].cum;
    const last = pts[pts.length - 1].cum;
    let best = { day: '', v: -Infinity };
    let worst = { day: '', v: Infinity };
    let prev = pts[0].cum - (pts.length ? 0 : 0);
    // per-day deltas within the window (delta of the first point uses its own step vs prior full point unknown -> skip)
    for (let i = 1; i < pts.length; i += 1) {
      const d = pts[i].cum - pts[i - 1].cum;
      if (d > best.v) best = { day: pts[i].day, v: d };
      if (d < worst.v) worst = { day: pts[i].day, v: d };
    }
    void prev;
    const maxDD = view.dd.length ? Math.min(...view.dd) : 0;
    return { periodPnl: last - first, maxDD, best, worst, n: pts.length };
  }, [view]);

  const btn = (active: boolean, onClick: () => void, label: string) => (
    <button
      type="button"
      onClick={onClick}
      style={{
        background: active ? 'var(--line)' : 'transparent',
        border: '1px solid var(--line)',
        borderRadius: 6,
        padding: '2px 10px',
        fontSize: 12,
        cursor: 'pointer',
        color: active ? 'var(--ink)' : 'var(--ink-muted)',
        fontWeight: active ? 700 : 500,
      }}
    >
      {label}
    </button>
  );

  const pos = headline >= 0;

  return (
    <section style={{ border: '1px solid var(--line)', borderRadius: 12, margin: '0 0 16px' }}>
      {/* header — always visible, click to expand */}
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        style={{
          width: '100%',
          display: 'flex',
          alignItems: 'center',
          gap: 12,
          padding: '12px 16px',
          background: 'transparent',
          border: 'none',
          cursor: 'pointer',
          textAlign: 'left',
        }}
      >
        <span style={{ fontSize: 14, fontWeight: 800, color: 'var(--ink)' }}>Cumulative P&amp;L · past sessions</span>
        <span style={{ color: pos ? GREEN : RED, fontWeight: 800, fontSize: 15 }}>{fmtInr(headline)}</span>
        <span style={{ fontSize: 11.5, color: 'var(--ink-muted)' }}>combined · all systems</span>
        <span style={{ marginLeft: 'auto', fontSize: 12, color: 'var(--ink-muted)' }}>{open ? 'Hide ▴' : 'Show ▾'}</span>
      </button>

      {open ? (
        <div style={{ padding: '0 16px 12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap', marginBottom: 8 }}>
            <span style={{ display: 'inline-flex', gap: 4 }}>
              {btn(venue === 'combined', () => setVenue('combined'), 'Combined')}
              {btn(venue === 'nifty', () => setVenue('nifty'), 'NIFTY')}
              {btn(venue === 'sensex', () => setVenue('sensex'), 'SENSEX')}
            </span>
            <span style={{ display: 'inline-flex', gap: 4, marginLeft: 8 }}>
              {(['1M', '3M', '6M', 'ALL'] as Range[]).map((r) => btn(range === r, () => setRange(r), r))}
            </span>
            {stats ? (
              <span style={{ marginLeft: 'auto', fontSize: 12, color: 'var(--ink-muted)' }}>
                period{' '}
                <span style={{ color: stats.periodPnl >= 0 ? GREEN : RED, fontWeight: 700 }}>{fmtInr(stats.periodPnl)}</span>
                {'  '}· maxDD <span style={{ color: RED, fontWeight: 700 }}>{fmtInr(stats.maxDD)}</span> · {stats.n} sessions
              </span>
            ) : null}
          </div>

          {err ? (
            <div style={{ color: RED, fontSize: 12, padding: '20px 0' }}>Couldn&apos;t load P&amp;L: {err}</div>
          ) : view.pts.length >= 2 ? (
            <>
              <div style={{ fontSize: 10.5, color: 'var(--ink-muted)', textTransform: 'uppercase', letterSpacing: 0.4, margin: '2px 0' }}>
                Equity (cumulative)
              </div>
              <Chart vals={view.pts.map((p) => p.cum)} dates={view.pts.map((p) => p.day)} color={headline >= 0 ? GREEN : RED} height={200} />
              <div style={{ fontSize: 10.5, color: 'var(--ink-muted)', textTransform: 'uppercase', letterSpacing: 0.4, margin: '8px 0 2px' }}>
                Drawdown (from peak)
              </div>
              <Chart vals={view.dd} dates={view.pts.map((p) => p.day)} color={RED} height={110} />
              <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', fontSize: 11.5, color: 'var(--ink-muted)', marginTop: 6 }}>
                {stats ? (
                  <>
                    <span>
                      best day <span style={{ color: GREEN, fontWeight: 700 }}>{fmtInr(stats.best.v)}</span> ({stats.best.day.slice(5)})
                    </span>
                    <span>
                      worst day <span style={{ color: RED, fontWeight: 700 }}>{fmtInr(stats.worst.v)}</span> ({stats.worst.day.slice(5)})
                    </span>
                  </>
                ) : null}
              </div>
              <div
                style={{
                  marginTop: 8,
                  padding: '6px 10px',
                  borderRadius: 6,
                  background: 'rgba(148,163,184,0.10)',
                  fontSize: 11.5,
                  color: 'var(--ink-muted)',
                  lineHeight: 1.5,
                }}
              >
                <strong style={{ color: 'var(--ink)' }}>Basis:</strong> live + paper days combined, <strong>as-traded ₹</strong> —
                lot size varied over the history (1 / 2 / 10 lots early → <strong>3 lots now</strong>), so this is{' '}
                <em>not</em> lot-normalized; older steps partly reflect size, not edge.
              </div>
            </>
          ) : (
            <div style={{ color: 'var(--ink-muted)', fontSize: 12, padding: '20px 0' }}>Loading…</div>
          )}
        </div>
      ) : null}
    </section>
  );
}
