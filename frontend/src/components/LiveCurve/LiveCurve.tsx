import { useEffect, useMemo, useState } from 'react';
import styles from './LiveCurve.module.css';

/* Live book vs market indices.
 *
 * Everything is plotted as PERCENT RETURN rebased to the visible window's first day — never raw
 * levels. Two reasons, both load-bearing:
 *   1. NAV is in rupees and an index is in points. On one chart in native units that needs two
 *      y-axes, and any crossover between them would be meaningless.
 *   2. The book takes deposits. Its NAV went Rs3L -> Rs7.69L in nine days, so a raw-NAV line would
 *      have shown +156% against Nifty 50's -2.15%. The `r` values come from book_curve(), which
 *      chains daily returns with that day's deposit backed out, so cash added moves the line by zero.
 */

type BookPt = { d: string; r: number; nav: number };
type Series = { label: string; points: { d: string; c: number }[] };
type Payload = { inception: string; book: BookPt[]; series: Record<string, Series> };
type Row = { d: string; nav: number; [k: string]: string | number | undefined };

/* Fixed identity order — validated for CVD separation (worst adjacent dE 15.0 deutan) on the light
 * surface. Colour belongs to the ENTITY: toggling Midcap off must never repaint Nifty 50. */
const SERIES: { key: string; label: string; color: string }[] = [
  { key: 'BOOK', label: 'Momentum-30', color: '#2563EB' },
  { key: 'NIFTY50', label: 'Nifty 50', color: '#B45309' },
  { key: 'NIFTY500', label: 'Nifty 500', color: '#7C3AED' },
  { key: 'NIFTYMIDCAP150', label: 'Midcap 150', color: '#0891B2' },
  { key: 'NIFTYSMLCAP250', label: 'Smallcap 250', color: '#9D174D' },
];

const RANGES = [
  { k: '1W', d: 5 }, { k: '1M', d: 22 }, { k: '3M', d: 66 },
  { k: '6M', d: 132 }, { k: '1Y', d: 252 }, { k: 'All', d: 0 },
];

const pctTxt = (v: number) => (v >= 0 ? '+' : '') + v.toFixed(2) + '%';

export default function LiveCurve({ compact = false }: { compact?: boolean }) {
  const [data, setData] = useState<Payload | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [on, setOn] = useState<Record<string, boolean>>({
    BOOK: true, NIFTY50: true, NIFTY500: false, NIFTYMIDCAP150: false, NIFTYSMLCAP250: false,
  });
  const [range, setRange] = useState<[number, number] | null>(null);
  const [preset, setPreset] = useState('All');
  const [hover, setHover] = useState<number | null>(null);
  const [drag, setDrag] = useState<{ a: number; b: number } | null>(null);
  const [big, setBig] = useState(false);

  useEffect(() => {
    fetch('/api/momentum-paper/benchmarks')
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error('HTTP ' + r.status))))
      .then(setData)
      .catch((e) => setErr(String(e && e.message ? e.message : e)));
  }, []);

  useEffect(() => {
    const esc = (e: KeyboardEvent) => { if (e.key === 'Escape') setBig(false); };
    window.addEventListener('keydown', esc);
    return () => window.removeEventListener('keydown', esc);
  }, []);

  /* The book's dates are the spine; an index is carried forward onto any date it lacks, so a
     holiday mismatch cannot punch a hole in a line. */
  const rows: Row[] = useMemo(() => {
    if (!data || !data.book || !data.book.length) return [];
    const idx: Record<string, Record<string, number>> = {};
    Object.entries(data.series || {}).forEach(([k, s]) => {
      idx[k] = {};
      s.points.forEach((p) => { idx[k][p.d] = p.c; });
    });
    const last: Record<string, number> = {};
    return data.book.map((bp) => {
      const o: Row = { d: bp.d, nav: bp.nav };
      o.BOOK_raw = 1 + bp.r / 100;
      SERIES.slice(1).forEach((s) => {
        const v = idx[s.key] ? idx[s.key][bp.d] : undefined;
        if (v != null) last[s.key] = v;
        o[s.key + '_raw'] = last[s.key];
      });
      return o;
    });
  }, [data]);

  const a = range ? range[0] : 0;
  const b = range ? range[1] : Math.max(0, rows.length - 1);
  const win = rows.slice(a, b + 1);

  /* Rebase every visible series to the FIRST DAY OF THE VISIBLE WINDOW, so zooming answers
     "who won over this period" rather than always replaying inception. */
  const lines = useMemo(() => {
    if (win.length < 2) return [];
    return SERIES.filter((s) => on[s.key]).map((s) => {
      const base = win[0][s.key + '_raw'] as number | undefined;
      const vals = win.map((r) => {
        const v = r[s.key + '_raw'] as number | undefined;
        return v == null || !base ? null : (v / base - 1) * 100;
      });
      return { key: s.key, label: s.label, color: s.color, vals, last: vals[vals.length - 1] };
    }).filter((l) => l.vals.some((v) => v != null));
  }, [win, on]);

  if (err) {
    return (
      <div className={styles.hint}>
        Curve data unavailable ({err}) — the benchmark endpoint ships with the next backend restart
        (after 15:40 IST). The book itself is unaffected.
      </div>
    );
  }
  if (!data) return <div className={styles.hint}>Loading curve…</div>;
  if (rows.length < 2) {
    return <div className={styles.hint}>Not enough history yet — the curve starts once the book has two daily marks.</div>;
  }

  const H = big ? 460 : compact ? 210 : 300;
  const W = 1000, L = 54, R = 66, T = 14, B = 26;
  const flat = lines.flatMap((l) => l.vals.filter((v): v is number => v != null));
  const lo = flat.length ? Math.min(0, ...flat) : -1;
  const hi = flat.length ? Math.max(0, ...flat) : 1;
  const pad = (hi - lo) * 0.12 || 1;
  const y = (v: number) => T + ((hi + pad - v) / (hi - lo + 2 * pad)) * (H - T - B);
  const x = (i: number) => L + (win.length < 2 ? 0 : (i / (win.length - 1)) * (W - L - R));

  const rawStep = (hi + pad - (lo - pad)) / 4;
  const mag = Math.pow(10, Math.floor(Math.log10(Math.max(rawStep, 1e-6))));
  const nn = rawStep / mag;
  const step = (nn < 1.5 ? 1 : nn < 3 ? 2 : nn < 7 ? 5 : 10) * mag;
  const ticks: number[] = [];
  for (let t = Math.ceil((lo - pad) / step) * step; t <= hi + pad; t += step) ticks.push(t);

  const idxFromEvent = (e: React.MouseEvent<SVGSVGElement>) => {
    const el = e.currentTarget.getBoundingClientRect();
    const rel = ((e.clientX - el.left) / el.width) * W;
    const i = Math.round(((rel - L) / (W - L - R)) * (win.length - 1));
    return Math.max(0, Math.min(win.length - 1, i));
  };

  const applyPreset = (k: string, d: number) => {
    setPreset(k); setHover(null);
    if (!d || d >= rows.length) { setRange(null); return; }
    setRange([Math.max(0, rows.length - 1 - d), rows.length - 1]);
  };

  const labelEvery = Math.max(1, Math.ceil(win.length / 7));

  const chart = (
    <div className={styles.plot}>
      <svg viewBox={'0 0 ' + W + ' ' + H} role="img"
        aria-label={'Percent return of the momentum book versus market indices, ' + win[0].d + ' to ' + win[win.length - 1].d}
        onMouseMove={(e) => { const i = idxFromEvent(e); setHover(i); if (drag) setDrag({ a: drag.a, b: i }); }}
        onMouseLeave={() => { setHover(null); setDrag(null); }}
        onMouseDown={(e) => { const i = idxFromEvent(e); setDrag({ a: i, b: i }); }}
        onMouseUp={() => {
          if (drag && Math.abs(drag.b - drag.a) >= 2) {
            setRange([a + Math.min(drag.a, drag.b), a + Math.max(drag.a, drag.b)]);
            setPreset('');
          }
          setDrag(null);
        }}
        onDoubleClick={() => { setRange(null); setPreset('All'); }}>
        {ticks.map((t) => (
          <g key={t}>
            <line x1={L} x2={W - R} y1={y(t)} y2={y(t)}
              stroke={Math.abs(t) < 1e-9 ? 'var(--hairline,rgba(0,0,0,0.18))' : 'var(--hairline-softest,rgba(0,0,0,0.05))'}
              strokeWidth={Math.abs(t) < 1e-9 ? 1.2 : 1} />
            <text x={L - 7} y={y(t) + 3.5} textAnchor="end" fontSize="10.5" fill="var(--ink-muted,#888780)">
              {t.toFixed(step < 1 ? 1 : 0)}%
            </text>
          </g>
        ))}
        {win.map((r, i) => ((i % labelEvery === 0 || i === win.length - 1) ? (
          <text key={r.d} x={x(i)} y={H - 8} textAnchor="middle" fontSize="10.5" fill="var(--ink-muted,#888780)">
            {r.d.slice(5)}
          </text>
        ) : null))}
        {drag && Math.abs(drag.b - drag.a) >= 2 && (
          <rect x={x(Math.min(drag.a, drag.b))} y={T} width={Math.abs(x(drag.b) - x(drag.a))}
            height={H - T - B} fill="#2563EB" opacity="0.08" />
        )}
        {lines.map((l) => (
          <path key={l.key} fill="none" stroke={l.color} strokeWidth={l.key === 'BOOK' ? 2.6 : 1.8}
            strokeLinejoin="round" strokeLinecap="round"
            d={l.vals.map((v, i) => (v == null ? '' : (i === 0 || l.vals[i - 1] == null ? 'M' : 'L') + x(i) + ',' + y(v))).join(' ')} />
        ))}
        {hover != null && win[hover] && (
          <line x1={x(hover)} x2={x(hover)} y1={T} y2={H - B}
            stroke="var(--ink-faint,#B4B2A9)" strokeWidth="1" strokeDasharray="3 3" />
        )}
        {lines.map((l) => {
          const i = hover != null ? hover : l.vals.length - 1;
          const v = l.vals[i];
          if (v == null) return null;
          return (
            <g key={l.key + '-end'}>
              <circle cx={x(i)} cy={y(v)} r="3.6" fill={l.color} stroke="var(--surface,#fff)" strokeWidth="2" />
              {hover == null && (
                <text x={W - R + 6} y={y(v) + 3.5} fontSize="11" fontWeight="700" fill={l.color}>{pctTxt(v)}</text>
              )}
            </g>
          );
        })}
      </svg>
      {hover != null && win[hover] && (
        <div className={styles.tip} style={{ left: 'min(calc(100% - 190px), ' + ((x(hover) / W) * 100) + '%)', top: 6 }}>
          <div className={styles.tipDate}>{win[hover].d}</div>
          {lines.map((l) => (
            <div key={l.key} className={styles.tipRow}>
              <span className={styles.dot} style={{ background: l.color }} />
              <span className={styles.tipName}>{l.label}</span>
              <span className={styles.tipVal}>{l.vals[hover] == null ? '—' : pctTxt(l.vals[hover] as number)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );

  const bookLine = lines.find((l) => l.key === 'BOOK');
  const n50 = lines.find((l) => l.key === 'NIFTY50');
  const alpha = bookLine && n50 && bookLine.last != null && n50.last != null
    ? bookLine.last - n50.last : null;

  const summary = (
    <div className={styles.summary}>
      {bookLine && bookLine.last != null && (
        <span><b style={{ color: bookLine.color }}>Momentum-30</b> {pctTxt(bookLine.last)}</span>
      )}
      {alpha != null && (
        <span>{alpha >= 0 ? 'ahead of' : 'behind'} Nifty 50 by <b>{Math.abs(alpha).toFixed(2)}pp</b></span>
      )}
      <span>{win.length} trading day{win.length === 1 ? '' : 's'} shown</span>
    </div>
  );

  const controls = (
    <>
      {summary}
      <div className={styles.bar}>
        {SERIES.map((s) => (
          <button key={s.key} className={styles.chip} data-on={on[s.key] ? '1' : '0'} aria-pressed={!!on[s.key]}
            onClick={() => setOn((o) => ({ ...o, [s.key]: !o[s.key] }))}>
            <span className={styles.dot} style={{ background: s.color }} />{s.label}
          </button>
        ))}
        <span className={styles.spacer} />
        <span className={styles.seg}>
          {RANGES.map((r) => (
            <button key={r.k} className={styles.segBtn} data-on={preset === r.k ? '1' : '0'}
              disabled={r.d > 0 && r.d >= rows.length} onClick={() => applyPreset(r.k, r.d)}>{r.k}</button>
          ))}
        </span>
        {range && <button className={styles.btn} onClick={() => { setRange(null); setPreset('All'); }}>Reset zoom</button>}
        <button className={styles.btn} onClick={() => setBig(!big)}>{big ? 'Close' : 'Expand'}</button>
      </div>
      <div className={styles.hint}>
        Percent return, rebased to {win[0].d} — the book excludes deposits (time-weighted), so adding
        cash does not move its line. Drag across the chart to zoom · double-click to reset.
      </div>
    </>
  );

  const table = (
    <details>
      <summary className={styles.hint} style={{ cursor: 'pointer' }}>Table view</summary>
      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr><th>Date</th>{lines.map((l) => <th key={l.key}>{l.label}</th>)}</tr>
          </thead>
          <tbody>
            {win.map((r, i) => (
              <tr key={r.d}>
                <td>{r.d}</td>
                {lines.map((l) => <td key={l.key}>{l.vals[i] == null ? '—' : pctTxt(l.vals[i] as number)}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </details>
  );

  if (big) {
    return (
      <div className={styles.overlay}>
        <div style={{ fontWeight: 700, fontSize: 15 }}>Momentum-30 vs market indices</div>
        {controls}{chart}{table}
      </div>
    );
  }
  return <div className={styles.wrap}>{controls}{chart}{table}</div>;
}
