/**
 * BookCurve — this book against the market, drawn in the panel it belongs to.
 *
 * Replaces the separate "Live P&L vs Nifty 50" card that used to sit further down the
 * page. It opens from the chevron on the summary panel, so a page that is not being
 * asked the question costs nothing to render.
 *
 * Everything is plotted as PERCENT RETURN rebased to the visible window's first day,
 * never raw levels — NAV is in rupees, an index is in points, and the book takes
 * deposits. The `r` values come from the book's own time-weighted curve with each day's
 * flow backed out, so adding cash moves the line by exactly zero.
 *
 * The drawing is laid out in the element's REAL PIXELS. A fixed viewBox stretched to
 * width scales x and y by different factors, which leaves the lines correct and every
 * label squashed — that bug shipped once and is the reason for the measurement below.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import styles from '../../pages/MomentumPaper.module.css';

type BookPt = { d: string; r: number; nav: number };
type Payload = {
  inception?: string;
  book: BookPt[];
  series: Record<string, { label: string; points: { d: string; c: number }[] }>;
};

/* Colour belongs to the ENTITY, not to its position in the list: switching Midcap off
   must never repaint Nifty 50. Order fixed, and validated for CVD separation. */
const INDICES: { key: string; label: string; color: string }[] = [
  { key: 'NIFTY50', label: 'Nifty 50', color: '#B45309' },
  { key: 'NIFTY500', label: 'Nifty 500', color: '#7C3AED' },
  { key: 'NIFTYMIDCAP150', label: 'Midcap 150', color: '#0891B2' },
  { key: 'NIFTYSMLCAP250', label: 'Smallcap 250', color: '#9D174D' },
];
const BOOK_COLOR = '#2563EB';

const RANGES = [
  { k: '1W', d: 5 },
  { k: '1M', d: 22 },
  { k: '3M', d: 66 },
  { k: '1Y', d: 252 },
  { k: 'All', d: 0 },
];

const MONS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
/** dd-Mon-yyyy everywhere; `short` drops the year where the axis is tight. */
function dmy(iso: string, short = false) {
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(iso);
  if (!m) return iso;
  return `${m[3]}-${MONS[parseInt(m[2], 10) - 1]}${short ? '' : '-' + m[1]}`;
}
const pctTxt = (v: number) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(2) + '%';

export default function BookCurve({ url, label }: { url: string; label: string }) {
  const [data, setData] = useState<Payload | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [preset, setPreset] = useState('All');
  const [on, setOn] = useState<Record<string, boolean>>({
    NIFTY50: true, NIFTY500: false, NIFTYMIDCAP150: false, NIFTYSMLCAP250: false,
  });
  const [w, setW] = useState(720);
  const box = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    let dead = false;
    fetch(url, { cache: 'no-store' })
      .then((r) => (r.ok ? r.json()
        : Promise.reject(new Error(r.status === 404 ? 'NOT_WIRED' : 'HTTP ' + r.status))))
      .then((d) => { if (!dead) setData(d); })
      .catch((e) => { if (!dead) setErr(String(e?.message || e)); });
    return () => { dead = true; };
  }, [url]);

  /* Measure the box, and re-measure on resize: the geometry is in pixels. */
  useEffect(() => {
    const measure = () => {
      const el = box.current;
      if (el) setW(Math.max(320, Math.round(el.getBoundingClientRect().width)));
    };
    measure();
    window.addEventListener('resize', measure);
    return () => window.removeEventListener('resize', measure);
  }, [data]);

  const days = RANGES.find((r) => r.k === preset)?.d ?? 0;

  /* Rebased percent series for the visible window. The book's dates are the spine; an
     index is carried forward onto any date it lacks, so a holiday mismatch cannot punch
     a hole in a line. */
  const view = useMemo(() => {
    const bk = data?.book ?? [];
    if (bk.length < 2) return null;
    const i0 = days ? Math.max(0, bk.length - days) : 0;
    const win = bk.slice(i0);
    const b0 = bk[i0].r;
    const out: Record<string, number[]> = {
      BOOK: win.map((p) => ((1 + p.r / 100) / (1 + b0 / 100) - 1) * 100),  // re-chain
    };
    const dates = win.map((p) => p.d);
    INDICES.forEach((s) => {
      const pts = data?.series?.[s.key]?.points;
      if (!pts?.length) return;
      const by: Record<string, number> = {};
      pts.forEach((p) => { by[p.d] = p.c; });
      let last: number | null = null;
      const vals: number[] = [];
      for (const d of dates) {
        if (by[d] != null) last = by[d];
        vals.push(last ?? NaN);
      }
      const base = vals.find((v) => !isNaN(v));
      if (base == null) return;
      out[s.key] = vals.map((v) => (isNaN(v) ? NaN : (v / base - 1) * 100));
    });
    return { dates, out };
  }, [data, days]);

  if (err === 'NOT_WIRED') {
    return (
      <p className={styles.note}>
        This book's curve is not being served yet — the route lands on the next backend
        deploy. Its record above is live.
      </p>
    );
  }
  if (err) {
    return (
      <p className={styles.note}>
        Curve unavailable — {err}. The book's own figures above are unaffected.
      </p>
    );
  }
  if (!data) return <p className={styles.note}>Loading the curve…</p>;
  if (!view) {
    const n = data.book?.length ?? 0;
    return (
      <p className={styles.note}>
        Not enough history to draw a curve yet — <b>{n}</b> session{n === 1 ? '' : 's'} since
        {' '}{data.inception ? dmy(data.inception) : 'inception'}. Two sessions make a line, not a
        record; this fills in on its own as the book runs.
      </p>
    );
  }

  const { dates, out } = view;
  const n = dates.length;
  const live = [{ key: 'BOOK', label, color: BOOK_COLOR }]
    .concat(INDICES.filter((s) => on[s.key] && out[s.key]));

  const h = 210, padL = 46, padR = 62, padT = 12, padB = 22;
  const W = w - padL - padR, H = h - padT - padB;
  let vals: number[] = [0];
  live.forEach((s) => { vals = vals.concat((out[s.key] || []).filter((v) => !isNaN(v))); });
  let lo = Math.min(...vals), hi = Math.max(...vals);
  const padv = Math.max(0.35, (hi - lo) * 0.14);
  lo -= padv; hi += padv;
  const X = (i: number) => padL + (n < 2 ? W / 2 : (i / (n - 1)) * W);
  const Y = (v: number) => padT + H - ((v - lo) / (hi - lo || 1)) * H;

  const grid = [0, 1, 2, 3].map((g) => lo + ((hi - lo) / 3) * g);
  const bookLast = out.BOOK[n - 1];
  const n50 = out.NIFTY50 ? out.NIFTY50[n - 1] : null;

  return (
    <div ref={box} className={styles.curveBox}>
      <div className={styles.curveHead}>
        <div className={styles.curveTitle}>
          {label} vs the market{' '}
          <span className={styles.muted} style={{ fontWeight: 400 }}>
            · time-weighted, deposits backed out
          </span>
        </div>
        <div className={styles.curveChips}>
          {RANGES.map((r) => (
            <button key={r.k} type="button"
                    className={`${styles.curveChip} ${preset === r.k ? styles.curveChipOn : ''}`}
                    onClick={() => setPreset(r.k)}>{r.k}</button>
          ))}
        </div>
      </div>

      <svg className={styles.curveSvg} viewBox={`0 0 ${w} ${h}`} width="100%" height={h}
           role="img" aria-label={`${label} return against the indices since ${dmy(dates[0])}`}>
        {grid.map((gv, i) => (
          <g key={i}>
            <line x1={padL} y1={Y(gv)} x2={padL + W} y2={Y(gv)}
                  stroke="var(--hairline,rgba(0,0,0,0.10))" strokeWidth={1} />
            <text x={padL - 8} y={Y(gv) + 3.5} textAnchor="end" fontSize={9.5}
                  fill="var(--ink-muted,#888780)">{pctTxt(gv)}</text>
          </g>
        ))}
        {lo < 0 && hi > 0 && (
          /* break-even has to read differently from a gridline */
          <line x1={padL} y1={Y(0)} x2={padL + W} y2={Y(0)}
                stroke="var(--ink-faint,#B4B2A9)" strokeWidth={1} strokeDasharray="3 3" />
        )}
        {live.map((s) => {
          const ser = out[s.key] || [];
          const pts = ser.map((v, i) => (isNaN(v) ? null : `${X(i).toFixed(1)},${Y(v).toFixed(1)}`))
            .filter(Boolean).join(' ');
          const last = ser[n - 1];
          return (
            <g key={s.key}>
              <polyline points={pts} fill="none" stroke={s.color}
                        strokeWidth={s.key === 'BOOK' ? 2.1 : 1.4}
                        strokeLinejoin="round" strokeLinecap="round" />
              {!isNaN(last) && <>
                <circle cx={X(n - 1)} cy={Y(last)} r={2.6} fill={s.color} />
                <text x={X(n - 1) + 6} y={Y(last) + 3.5} fontSize={10} fontWeight={600}
                      fill={s.color}>{pctTxt(last)}</text>
              </>}
            </g>
          );
        })}
        <text x={X(0)} y={h - 6} textAnchor="start" fontSize={9.5}
              fill="var(--ink-muted,#888780)">{dmy(dates[0], true)}</text>
        <text x={X(n - 1)} y={h - 6} textAnchor="end" fontSize={9.5}
              fill="var(--ink-muted,#888780)">{dmy(dates[n - 1], true)}</text>
      </svg>

      {/* the legend doubles as the toggles, each carrying its own current number */}
      <div className={styles.curveLegend}>
        <span className={styles.curveKey}>
          <i style={{ background: BOOK_COLOR }} />{label}
          <b>{pctTxt(bookLast)}</b>
        </span>
        {INDICES.map((s) => (
          <button key={s.key} type="button" disabled={!out[s.key]}
                  aria-pressed={!!on[s.key]}
                  className={`${styles.curveKey} ${on[s.key] ? styles.curveKeyOn : ''}`}
                  onClick={() => setOn((o) => ({ ...o, [s.key]: !o[s.key] }))}>
            <i style={{ background: s.color }} />{s.label}
            <b>{out[s.key] ? pctTxt(out[s.key][n - 1]) : '—'}</b>
          </button>
        ))}
      </div>

      <div className={styles.curveFoot}>
        <span>
          {n50 == null ? `${n} session${n === 1 ? '' : 's'} of live record.` : (
            <>
              <b className={bookLast - n50 >= 0 ? styles.pos : styles.neg}>
                {(bookLast - n50 >= 0 ? '+' : '−') + Math.abs(bookLast - n50).toFixed(2)}pp
              </b>{' '}
              {bookLast - n50 >= 0 ? 'ahead of' : 'behind'} Nifty 50 over this window
            </>
          )}
        </span>
        <span className={styles.muted}>
          {n} sessions · {dmy(dates[0])} to {dmy(dates[n - 1])}
        </span>
      </div>
    </div>
  );
}
