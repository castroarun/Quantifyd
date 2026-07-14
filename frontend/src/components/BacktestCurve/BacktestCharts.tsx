import { useEffect, useMemo, useState } from 'react';
import styles from './BacktestCurve.module.css';

type Pt = { d: string; p: number; n: number };
type Data = {
  meta: { cap: number; start: string; end: string; port_cagr: number; port_dd: number; nifty_dd: number };
  points: Pt[];
};

const W = 920, PAD_L = 64, PAD_R = 18, PAD_T = 16, PAD_B = 26;

function fmtInr(v: number): string {
  if (v >= 1e7) return '₹' + (v / 1e7).toFixed(2) + ' cr';
  if (v >= 1e5) return '₹' + (v / 1e5).toFixed(v >= 1e6 ? 0 : 1) + 'L';
  return '₹' + Math.round(v).toLocaleString('en-IN');
}

/** Adaptive Y ticks: log 1-2-5 for wide ranges, linear "nice" steps when zoomed narrow. */
function niceYTicks(lo: number, hi: number): number[] {
  const out: number[] = [];
  if (hi / lo > 6) {
    for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++)
      for (const m of [1, 2, 5]) { const t = m * 10 ** e; if (t >= lo * 0.9 && t <= hi * 1.1) out.push(t); }
    return out;
  }
  const span = hi - lo, raw = span / 5;
  const mag = 10 ** Math.floor(Math.log10(raw)), nn = raw / mag;
  const step = (nn < 1.5 ? 1 : nn < 3 ? 2 : nn < 7 ? 5 : 10) * mag;
  for (let t = Math.ceil((lo * 0.995) / step) * step; t <= hi * 1.005; t += step) out.push(t);
  return out;
}

export default function BacktestCharts() {
  const [data, setData] = useState<Data | null>(null);
  const [range, setRange] = useState<[number, number] | null>(null); // GLOBAL index range; null = full
  const [hi, setHi] = useState<number | null>(null);                 // hover LOCAL index
  const [drag, setDrag] = useState<{ a: number; b: number } | null>(null);

  useEffect(() => {
    fetch('/app/momentum30-backtest-curve.json').then((r) => r.json()).then(setData).catch(() => {});
  }, []);

  const years = useMemo(() => {
    if (!data) return [] as { y: string; a: number; b: number }[];
    const m: Record<string, { a: number; b: number }> = {};
    data.points.forEach((p, i) => {
      const y = p.d.slice(0, 4);
      if (!m[y]) m[y] = { a: i, b: i }; else m[y].b = i;
    });
    return Object.keys(m).sort().map((y) => ({ y, ...m[y] }));
  }, [data]);

  const view = useMemo(() => {
    if (!data) return null;
    const N = data.points.length;
    const [a, b] = range ?? [0, N - 1];
    const cap = data.meta.cap;
    const p0 = data.points[a].p, n0 = data.points[a].n;
    const pts = data.points.slice(a, b + 1).map((pt) => ({ d: pt.d, p: (pt.p / p0) * cap, n: (pt.n / n0) * cap }));
    // drawdowns (rebased window, from each series' own peak within the window)
    const dd = (k: 'p' | 'n') => { let pk = -Infinity; return pts.map((q) => { pk = Math.max(pk, q[k]); return (q[k] / pk - 1) * 100; }); };
    return { a, b, pts, ddP: dd('p'), ddN: dd('n'), n: pts.length };
  }, [data, range]);

  if (!data || !view) return <div className={styles.loading}>Loading backtest…</div>;
  const { a, pts, ddP, ddN, n } = view;
  const j = hi != null ? Math.max(0, Math.min(n - 1, hi)) : n - 1;
  const xf = (i: number) => PAD_L + (i / Math.max(n - 1, 1)) * (W - PAD_L - PAD_R);
  const idxAt = (clientX: number, rect: DOMRect) =>
    Math.max(0, Math.min(n - 1, Math.round((((clientX - rect.left) / rect.width) * W - PAD_L) / (W - PAD_L - PAD_R) * (n - 1))));

  const onDown = (e: React.MouseEvent<SVGSVGElement>) => {
    const i = idxAt(e.clientX, e.currentTarget.getBoundingClientRect());
    setDrag({ a: i, b: i });
  };
  const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const i = idxAt(e.clientX, e.currentTarget.getBoundingClientRect());
    setHi(i);
    if (drag) setDrag({ a: drag.a, b: i });
  };
  const onUp = () => {
    if (drag && Math.abs(drag.a - drag.b) >= 3) {
      const lo = Math.min(drag.a, drag.b), hiL = Math.max(drag.a, drag.b);
      setRange([a + lo, a + hiL]);
    }
    setDrag(null);
  };

  const last = pts[n - 1];
  const mult = (v: number) => (v / data.meta.cap).toFixed(1) + '×';

  // ---- equity (log) chart geometry ----
  const HE = 380;
  const vals = pts.flatMap((p) => [p.p, p.n]);
  const lo = Math.min(...vals), hiV = Math.max(...vals);
  const lLo = Math.log10(lo), lHi = Math.log10(hiV);
  const yE = (v: number) => PAD_T + (1 - (Math.log10(v) - lLo) / (lHi - lLo || 1)) * (HE - PAD_T - PAD_B);
  const pathE = (k: 'p' | 'n') => pts.map((p, i) => `${i ? 'L' : 'M'}${xf(i).toFixed(1)},${yE(p[k]).toFixed(1)}`).join(' ');
  const yTicks = niceYTicks(lo, hiV);

  // ---- drawdown chart geometry ----
  const HD = 220;
  const ddLo = Math.min(...ddP, ...ddN, -1);
  const yMinD = ddLo * 1.08;
  const yD = (v: number) => PAD_T + (1 - (v - yMinD) / (0 - yMinD)) * (HD - PAD_T - PAD_B);
  const lineD = (arr: number[]) => arr.map((v, i) => `${i ? 'L' : 'M'}${xf(i).toFixed(1)},${yD(v).toFixed(1)}`).join(' ');
  const areaD = (arr: number[]) => `M${xf(0).toFixed(1)},${yD(0).toFixed(1)} ` + arr.map((v, i) => `L${xf(i).toFixed(1)},${yD(v).toFixed(1)}`).join(' ') + ` L${xf(n - 1).toFixed(1)},${yD(0).toFixed(1)} Z`;
  const ddTicks = [0, -10, -20, -30, -40, -50].filter((t) => t >= yMinD);

  // x-axis labels (dates) for the current window
  const labels: { i: number; t: string }[] = [];
  const step = Math.max(1, Math.floor(n / 7));
  for (let i = 0; i < n; i += step) labels.push({ i, t: pts[i].d.slice(0, 7) });

  const activeYear = (yr: { a: number; b: number }) => range && range[0] === yr.a && range[1] === yr.b;

  return (
    <div className={styles.wrap}>
      {/* range controls */}
      <div className={styles.controls}>
        <button className={`${styles.chip} ${range == null ? styles.chipOn : ''}`} onClick={() => setRange(null)}>Full</button>
        {years.map((yr) => (
          <button key={yr.y} className={`${styles.chip} ${activeYear(yr) ? styles.chipOn : ''}`} onClick={() => setRange([yr.a, yr.b])}>{yr.y}</button>
        ))}
        {range != null && <button className={styles.chipReset} onClick={() => setRange(null)}>✕ reset</button>}
        <span className={styles.hint}>drag on the chart to zoom · both re-based to {fmtInr(data.meta.cap)} at {pts[0].d}</span>
      </div>

      <div className={styles.legend}>
        <span><i className={styles.dotP} /> Portfolio <b>{fmtInr(last.p)}</b> ({mult(last.p)})</span>
        <span><i className={styles.dotN} /> NIFTYBEES <b>{fmtInr(last.n)}</b> ({mult(last.n)})</span>
        <span className={styles.muted}>{pts[0].d} → {last.d}{range == null ? ` · ${data.meta.port_cagr}% CAGR` : ''} · log scale</span>
      </div>

      {/* equity chart */}
      <svg viewBox={`0 0 ${W} ${HE}`} className={styles.svg} onMouseDown={onDown} onMouseMove={onMove} onMouseUp={onUp} onMouseLeave={() => { setHi(null); setDrag(null); }}>
        {yTicks.map((t) => (<g key={t}><line x1={PAD_L} x2={W - PAD_R} y1={yE(t)} y2={yE(t)} className={styles.grid} /><text x={PAD_L - 7} y={yE(t) + 3} className={styles.ytick}>{fmtInr(t)}</text></g>))}
        {labels.map((l) => (<text key={l.i} x={xf(l.i)} y={HE - 8} className={styles.xtick}>{l.t}</text>))}
        {drag && <rect x={xf(Math.min(drag.a, drag.b))} y={PAD_T} width={Math.abs(xf(drag.b) - xf(drag.a))} height={HE - PAD_T - PAD_B} className={styles.selRect} />}
        <path d={pathE('n')} className={styles.lineN} />
        <path d={pathE('p')} className={styles.lineP} />
        {hi != null && !drag && (<g><line x1={xf(j)} x2={xf(j)} y1={PAD_T} y2={HE - PAD_B} className={styles.cross} /><circle cx={xf(j)} cy={yE(pts[j].n)} r={4} className={styles.markN} /><circle cx={xf(j)} cy={yE(pts[j].p)} r={4.5} className={styles.markP} /></g>)}
      </svg>

      {/* drawdown chart */}
      <div className={styles.ddTitle}>Drawdown (underwater) · worst in window: portfolio {Math.min(...ddP).toFixed(1)}% vs NIFTYBEES {Math.min(...ddN).toFixed(1)}%</div>
      <svg viewBox={`0 0 ${W} ${HD}`} className={styles.svg} onMouseMove={onMove} onMouseLeave={() => setHi(null)}>
        {ddTicks.map((t) => (<g key={t}><line x1={PAD_L} x2={W - PAD_R} y1={yD(t)} y2={yD(t)} className={styles.grid} /><text x={PAD_L - 7} y={yD(t) + 3} className={styles.ytick}>{t}%</text></g>))}
        <path d={areaD(ddN)} className={styles.fillN} />
        <path d={areaD(ddP)} className={styles.fillP} />
        <path d={lineD(ddN)} className={styles.lineN} />
        <path d={lineD(ddP)} className={styles.lineP} />
        {hi != null && (<g><line x1={xf(j)} x2={xf(j)} y1={PAD_T} y2={HD - PAD_B} className={styles.cross} /><circle cx={xf(j)} cy={yD(ddN[j])} r={4} className={styles.markN} /><circle cx={xf(j)} cy={yD(ddP[j])} r={4.5} className={styles.markP} /></g>)}
      </svg>

      {hi != null && (
        <div className={styles.tip} style={{ left: `${(xf(j) / W) * 100}%` }}>
          <div className={styles.tipDate}>{pts[j].d}</div>
          <div className={styles.tipRow}><i className={styles.dotP} /> {fmtInr(pts[j].p)} · dd {ddP[j].toFixed(1)}%</div>
          <div className={styles.tipRow}><i className={styles.dotN} /> {fmtInr(pts[j].n)} · dd {ddN[j].toFixed(1)}%</div>
          <div className={styles.tipAhead}>{(pts[j].p / pts[j].n).toFixed(2)}× vs index</div>
        </div>
      )}
    </div>
  );
}
