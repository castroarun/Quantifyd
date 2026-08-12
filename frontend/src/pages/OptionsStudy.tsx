import { useEffect, useMemo, useRef, useState } from 'react';
import uPlot from 'uplot';
import 'uplot/dist/uPlot.min.css';
import styles from './OptionsStudy.module.css';

type Day = {
  date: string; weekday: string; dte: number; atm: number;
  entry: number; close: number; high: number; low: number;
  decay_pct: number; rng: number; spot_open: number; spot_close: number; spot_move: number;
  series: [string, number, number, number, number][]; // hhmm, straddle, ce, pe, spot
  otm?: Record<string, [string, number][]>;            // offset -> [hhmm, strangle]
  ohlc?: [string, number, number, number, number][];   // hhmm, o, h, l, c (NIFTY 5-min candles)
  cpr?: { tc: number; pivot: number; bc: number; width_pct: number };  // prior-day CPR
};

const WDS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'];
const WD_COLOR: Record<string, string> = { Mon: '#79c0ff', Tue: '#3fb950', Wed: '#e3b341', Thu: '#ff7b72', Fri: '#d2a8ff' };
const DTES = [0, 1, 2, 3, 4];
const DTE_COLOR: Record<number, string> = { 0: '#f85149', 1: '#ff9e64', 2: '#e3b341', 3: '#3fb950', 4: '#79c0ff' };
const OTM_OFFS = ['100', '200', '300'];
const OTM_COLOR: Record<string, string> = { atm: '#3fb950', '100': '#e3b341', '200': '#ff9e64', '300': '#f85149' };

const toMin = (hhmm: string) => { const [h, m] = hhmm.split(':').map(Number); return h * 60 + m - (9 * 60 + 16); };
// snap a bar's minute-offset to a clean 5-min grid so stray marks (e.g. 13:56 vs 13:55)
// collapse into the same bucket — otherwise a slot only one odd day fills becomes a null gap
const snap5 = (m: number) => Math.round(m / 5) * 5;
const fmtT = (x: number) => {
  const v = 9 * 60 + 16 + x;
  return `${String(Math.floor(v / 60)).padStart(2, '0')}:${String(((v % 60) + 60) % 60).padStart(2, '0')}`;
};
const MON3 = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'];
const fmtDMY = (s: string) => { const [y, m, d] = s.split('-'); return `${d}-${MON3[+m - 1]}-${y.slice(2)}`; };
const XAXIS = { stroke: '#8b949e', grid: { stroke: 'rgba(139,148,158,0.10)' }, values: (_u: any, v: number[]) => v.map(fmtT) };
const pctAxis = { stroke: '#8b949e', grid: { stroke: 'rgba(139,148,158,0.10)' }, values: (_u: any, v: number[]) => v.map((x) => x + '%') };

function Chart({ opts, data, height }: { opts: any; data: any; height: number }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current; if (!el) return;
    const u = new (uPlot as any)({ ...opts, width: el.clientWidth, height }, data, el);
    const onR = () => el && u.setSize({ width: el.clientWidth, height });
    window.addEventListener('resize', onR);
    return () => { window.removeEventListener('resize', onR); u.destroy(); };
  }, [opts, data, height]);
  return <div ref={ref} className={styles.chart} />;
}
function Tile({ label, v, good }: { label: string; v: string; good?: boolean }) {
  return <div className={styles.tile}><div className={styles.tl}>{label}</div>
    <div className={styles.tv} style={{ color: good == null ? undefined : good ? '#3fb950' : '#f85149' }}>{v}</div></div>;
}
// tiny SVG sparkline of one day's straddle path (pts = [x,y]); green if it decayed, red if it rose
function Spark({ pts, up }: { pts: [number, number][]; up: boolean }) {
  const W = 150, H = 46, pad = 3;
  if (pts.length < 2) return <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} />;
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  const sx = (x: number) => pad + (x1 > x0 ? (x - x0) / (x1 - x0) : 0) * (W - 2 * pad);
  const sy = (y: number) => H - pad - (y1 > y0 ? (y - y0) / (y1 - y0) : 0.5) * (H - 2 * pad);
  const d = pts.map((p, i) => (i ? 'L' : 'M') + sx(p[0]).toFixed(1) + ' ' + sy(p[1]).toFixed(1)).join(' ');
  const c = up ? '#f85149' : '#3fb950';
  const yEntry = sy(pts[0][1]);
  const pk = ys.indexOf(y1);  // index of the intraday max premium (seller's peak pain)
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} preserveAspectRatio="none">
      <line x1={0} y1={yEntry} x2={W} y2={yEntry} stroke="rgba(139,148,158,0.35)" strokeWidth={0.6} strokeDasharray="3 3" />
      <path d={d} fill="none" stroke={c} strokeWidth={1.5} vectorEffect="non-scaling-stroke" />
      <circle cx={sx(pts[pk][0])} cy={sy(pts[pk][1])} r={2.2} fill="#d29922" />
    </svg>
  );
}

// NIFTY intraday candlesticks + prior-day CPR band (inline SVG; responsive)
function Candles({ ohlc, cpr, height = 240 }: { ohlc?: [string, number, number, number, number][]; cpr?: { tc: number; pivot: number; bc: number; width_pct: number }; height?: number }) {
  if (!ohlc || ohlc.length < 2) return <div style={{ color: '#8b949e', fontSize: 12, padding: 10 }}>no NIFTY OHLC for this day</div>;
  const n = ohlc.length, W = 960, padL = 46, padR = 8, padT = 8, padB = 18;
  let yMin = Math.min(...ohlc.map((b) => b[3])), yMax = Math.max(...ohlc.map((b) => b[2]));
  if (cpr) { yMin = Math.min(yMin, cpr.bc); yMax = Math.max(yMax, cpr.tc); }
  const pad = (yMax - yMin) * 0.05 || 1; yMin -= pad; yMax += pad;
  const cx = (i: number) => padL + ((i + 0.5) / n) * (W - padL - padR);
  const bw = Math.max(1.5, ((W - padL - padR) / n) * 0.62);
  const Y = (v: number) => padT + (1 - (v - yMin) / (yMax - yMin)) * (height - padT - padB);
  const up = '#3fb950', down = '#f85149';
  return (
    <svg viewBox={`0 0 ${W} ${height}`} width="100%" height={height} preserveAspectRatio="none" style={{ display: 'block' }}>
      {cpr && <g>
        <rect x={padL} y={Y(cpr.tc)} width={W - padL - padR} height={Math.max(1, Y(cpr.bc) - Y(cpr.tc))} fill="rgba(180,83,9,0.10)" />
        <line x1={padL} y1={Y(cpr.tc)} x2={W - padR} y2={Y(cpr.tc)} stroke="#b45309" strokeWidth={0.8} vectorEffect="non-scaling-stroke" />
        <line x1={padL} y1={Y(cpr.bc)} x2={W - padR} y2={Y(cpr.bc)} stroke="#b45309" strokeWidth={0.8} vectorEffect="non-scaling-stroke" />
        <line x1={padL} y1={Y(cpr.pivot)} x2={W - padR} y2={Y(cpr.pivot)} stroke="#b45309" strokeWidth={0.7} strokeDasharray="4 4" vectorEffect="non-scaling-stroke" />
      </g>}
      {ohlc.map((b, i) => { const o = b[1], h = b[2], l = b[3], c = b[4], x = cx(i), clr = c >= o ? up : down;
        return <g key={i}>
          <line x1={x} y1={Y(h)} x2={x} y2={Y(l)} stroke={clr} strokeWidth={0.8} vectorEffect="non-scaling-stroke" />
          <rect x={x - bw / 2} y={Y(Math.max(o, c))} width={bw} height={Math.max(1, Y(Math.min(o, c)) - Y(Math.max(o, c)))} fill={clr} />
        </g>; })}
      {[yMax, cpr?.pivot, yMin].filter((v): v is number => v != null).map((v, i) => (
        <text key={i} x={4} y={Y(v) + 3} fontSize={9} fill="#8b949e">{Math.round(v)}</text>))}
      {[0, Math.floor(n / 2), n - 1].map((i) => (
        <text key={'x' + i} x={cx(i)} y={height - 5} fontSize={9} fill="#8b949e" textAnchor="middle">{ohlc[i][0]}</text>))}
    </svg>
  );
}

export default function OptionsStudy() {
  const [d, setD] = useState<{ generated_at: string; n_days: number; days: Day[] } | null>(null);
  const [sel, setSel] = useState<string | null>(null);
  const [showLegs, setShowLegs] = useState(false);
  const [wd, setWd] = useState('All');
  const [startT, setStartT] = useState('09:16');
  const [endT, setEndT] = useState('15:30');
  const [grp, setGrp] = useState('none');
  const [expand, setExpand] = useState<string | null>(null);
  const [tableSort, setTableSort] = useState<'date' | 'decay'>('date');

  useEffect(() => {
    fetch(`/app/options_study.json?t=${Date.now()}`, { cache: 'no-store' })
      .then((r) => r.json()).then((j) => { setD(j); setSel(j.days?.[j.days.length - 1]?.date ?? null); }).catch(() => {});
  }, []);

  const allDays = d?.days ?? [];
  const times = useMemo(() => { const s = new Set<string>(); allDays.forEach((dy) => dy.series.forEach((b) => s.add(b[0]))); return [...s].sort(); }, [allDays]);
  const days = useMemo(() => (wd === 'All' ? allDays : allDays.filter((x) => x.weekday === wd)), [allDays, wd]);
  useEffect(() => { if (days.length && !days.some((x) => x.date === sel)) setSel(days[days.length - 1].date); }, [days, sel]);
  const day = useMemo(() => days.find((x) => x.date === sel) ?? days[days.length - 1], [days, sel]);

  // grid = the ACTUAL recorded bar times within the window (bars are at 09:16 then :20/:25/:30…,
  // i.e. minutes 0,4,9,14… — NOT multiples of 5 — so build the grid from the real slots, else the
  // median/overlay grid only ever matches the 09:16 point and everything else is null).
  const gridMins = useMemo(() => [...new Set(times.filter((t) => t >= startT && t <= endT).map((t) => snap5(toMin(t))))].sort((a, b) => a - b), [times, startT, endT]);

  // generic window helpers (a "getter" returns a day's full [hhmm,value] series)
  const getStrad = (dy: Day): [string, number][] => dy.series.map((b) => [b[0], b[1]]);
  const getOtm = (off: string) => (dy: Day) => dy.otm?.[off];
  const win = (full?: [string, number][]) => (full ?? []).filter((b) => b[0] >= startT && b[0] <= endT);
  const winDecay = (dy: Day, get: (d: Day) => [string, number][] | undefined) => {
    const w = win(get(dy)); if (w.length < 2 || !w[0][1]) return null;
    return Math.round((w[w.length - 1][1] / w[0][1] - 1) * 1000) / 10;
  };
  const medNorm = (dset: Day[], get: (d: Day) => [string, number][] | undefined) => {
    const norm = dset.map((dy) => {
      const w = win(get(dy)); const map = new Map<number, number>();
      if (w.length >= 2 && w[0][1]) { const e = w[0][1]; w.forEach((b) => map.set(snap5(toMin(b[0])), (b[1] / e) * 100)); }
      return gridMins.map((m) => (map.has(m) ? Math.round(map.get(m)! * 10) / 10 : null));
    });
    return gridMins.map((_, i) => {
      const vals = norm.map((n) => n[i]).filter((v): v is number => v != null).sort((a, b) => a - b);
      return vals.length ? vals[Math.floor(vals.length / 2)] : null;
    });
  };
  // median ABSOLUTE ₹ P&L of a SHORT position = entry − premium_t (profit as premium decays)
  const medPnl = (dset: Day[], get: (d: Day) => [string, number][] | undefined) => {
    const rows = dset.map((dy) => {
      const w = win(get(dy)); const map = new Map<number, number>();
      if (w.length >= 2 && w[0][1]) { const e = w[0][1]; w.forEach((b) => map.set(snap5(toMin(b[0])), Math.round((e - b[1]) * 10) / 10)); }
      return gridMins.map((m) => (map.has(m) ? map.get(m)! : null));
    });
    return gridMins.map((_, i) => {
      const vals = rows.map((n) => n[i]).filter((v): v is number => v != null).sort((a, b) => a - b);
      return vals.length ? vals[Math.floor(vals.length / 2)] : null;
    });
  };
  const statOf = (dy: Day) => {
    const w = win(getStrad(dy)); if (w.length < 2) return null;
    const strad = w.map((b) => b[1]);
    return { w, entry: w[0][1], close: w[w.length - 1][1], hi: Math.max(...strad), lo: Math.min(...strad),
             decay: w[0][1] ? Math.round((w[w.length - 1][1] / w[0][1] - 1) * 1000) / 10 : 0 };
  };

  // Chart 1 — intraday straddle (window) + spot dotted (right axis)
  const c1 = useMemo(() => {
    if (!day) return null; const st = statOf(day); if (!st) return null;
    const w = day.series.filter((b) => b[0] >= startT && b[0] <= endT); const xs = w.map((b) => toMin(b[0]));
    const data: any = showLegs ? [xs, w.map((b) => b[1]), w.map((b) => b[2]), w.map((b) => b[3]), w.map((b) => b[4])]
      : [xs, w.map((b) => b[1]), w.map((b) => b[4])];
    const series: any[] = [{}, { label: 'Straddle', stroke: '#3fb950', width: 2, points: { show: false } }];
    if (showLegs) { series.push({ label: 'CE', stroke: '#e3b341', width: 1, points: { show: false } }); series.push({ label: 'PE', stroke: '#79c0ff', width: 1, points: { show: false } }); }
    series.push({ label: 'NIFTY', scale: 'spot', stroke: '#8b949e', width: 1, dash: [4, 4], points: { show: false } });
    const opts: any = { series, scales: { spot: {} }, axes: [XAXIS,
      { stroke: '#8b949e', grid: { stroke: 'rgba(139,148,158,0.10)' }, values: (_u: any, v: number[]) => v.map((x) => '₹' + x) },
      { scale: 'spot', side: 1, stroke: '#8b949e', grid: { show: false }, values: (_u: any, v: number[]) => v.map(String) }],
      legend: { show: true }, cursor: { drag: { x: true, y: false } } };
    return { opts, data };
  }, [day, showLegs, startT, endT]);

  // Chart 2 — filtered days normalised to window-start=100 + median
  const c2 = useMemo(() => {
    if (!days.length) return null;
    const norm = days.map((dy) => { const w = win(getStrad(dy)); const map = new Map<number, number>();
      if (w.length >= 2 && w[0][1]) { const e = w[0][1]; w.forEach((b) => map.set(snap5(toMin(b[0])), Math.round((b[1] / e) * 1000) / 10)); }
      return gridMins.map((m) => (map.has(m) ? map.get(m)! : null)); });
    const data: any = [gridMins, ...norm, medNorm(days, getStrad)];
    const series: any[] = [{}]; days.forEach((dy) => series.push({ label: dy.date, stroke: dy.date === sel ? '#e3b341' : 'rgba(139,148,158,0.15)', width: dy.date === sel ? 2 : 1, points: { show: false } }));
    series.push({ label: 'median', stroke: '#3fb950', width: 2.5, points: { show: false } });
    return { opts: { series, axes: [XAXIS, pctAxis], legend: { show: false }, cursor: { drag: { x: false, y: false }, points: { show: false } } }, data };
  }, [days, sel, startT, endT]);

  // Chart — median by weekday
  const cG = useMemo(() => {
    if (!allDays.length) return null;
    const data: any = [gridMins, ...WDS.map((w) => medNorm(allDays.filter((x) => x.weekday === w), getStrad))];
    const series: any[] = [{}, ...WDS.map((w) => ({ label: w, stroke: WD_COLOR[w], width: 2, points: { show: false } }))];
    return { opts: { series, axes: [XAXIS, pctAxis], legend: { show: true }, cursor: { drag: { x: false, y: false } } }, data };
  }, [allDays, startT, endT]);

  // Chart — median by DTE
  const cDTE = useMemo(() => {
    if (!allDays.length) return null;
    const data: any = [gridMins, ...DTES.map((k) => medNorm(allDays.filter((x) => x.dte === k), getStrad))];
    const series: any[] = [{}, ...DTES.map((k) => ({ label: 'DTE' + k, stroke: DTE_COLOR[k], width: 2, points: { show: false } }))];
    return { opts: { series, axes: [XAXIS, pctAxis], legend: { show: true }, cursor: { drag: { x: false, y: false } } }, data };
  }, [allDays, startT, endT]);

  // Chart — ATM straddle vs OTM strangles (filtered days), median normalised
  const cOTM = useMemo(() => {
    if (!days.length) return null;
    const data: any = [gridMins, medNorm(days, getStrad), ...OTM_OFFS.map((o) => medNorm(days, getOtm(o)))];
    const series: any[] = [{}, { label: 'ATM straddle', stroke: OTM_COLOR.atm, width: 2.5, points: { show: false } },
      ...OTM_OFFS.map((o) => ({ label: '±' + o + 'pt', stroke: OTM_COLOR[o], width: 1.5, points: { show: false } }))];
    return { opts: { series, axes: [XAXIS, pctAxis], legend: { show: true }, cursor: { drag: { x: false, y: false } } }, data };
  }, [days, startT, endT]);

  // Chart — ABSOLUTE ₹ P&L of a short ATM straddle vs short OTM strangles (median, filtered days)
  const cAbs = useMemo(() => {
    if (!days.length) return null;
    const rupee = { stroke: '#8b949e', grid: { stroke: 'rgba(139,148,158,0.10)' }, values: (_u: any, v: number[]) => v.map((x) => '₹' + x) };
    const data: any = [gridMins, medPnl(days, getStrad), ...OTM_OFFS.map((o) => medPnl(days, getOtm(o)))];
    const series: any[] = [{}, { label: 'ATM straddle', stroke: OTM_COLOR.atm, width: 2.5, points: { show: false } },
      ...OTM_OFFS.map((o) => ({ label: '±' + o + 'pt', stroke: OTM_COLOR[o], width: 1.5, points: { show: false } }))];
    return { opts: { series, axes: [XAXIS, rupee], legend: { show: true }, cursor: { drag: { x: false, y: false } } }, data };
  }, [days, startT, endT]);

  // heatmap weekday x DTE (median straddle window-decay)
  const heat = useMemo(() => WDS.map((w) => DTES.map((k) => {
    const ds = allDays.filter((x) => x.weekday === w && x.dte === k).map((dy) => winDecay(dy, getStrad)).filter((v): v is number => v != null).sort((a, b) => a - b);
    return { n: ds.length, med: ds.length ? ds[Math.floor(ds.length / 2)] : null };
  })), [allDays, startT, endT]);

  // group the daily-decay bars by a chosen criterion (with per-group median)
  const groups = useMemo(() => {
    const keyOf = (x: Day) => grp === 'weekday' ? x.weekday : grp === 'dte' ? 'DTE' + x.dte : grp === 'month' ? x.date.slice(0, 7) : 'all';
    const m = new Map<string, Day[]>();
    days.forEach((x) => { const k = keyOf(x); if (!m.has(k)) m.set(k, []); m.get(k)!.push(x); });
    let keys = [...m.keys()];
    if (grp === 'weekday') keys = WDS.filter((k) => m.has(k));
    else if (grp === 'dte') keys = DTES.map((k) => 'DTE' + k).filter((k) => m.has(k));
    else keys.sort();
    return keys.map((k) => {
      const gd = m.get(k)!.slice().sort((a, b) => a.date.localeCompare(b.date));
      const dec = gd.map((x) => statOf(x)?.decay).filter((v): v is number => v != null).sort((a, b) => a - b);
      return { key: k, days: gd, med: dec.length ? dec[Math.floor(dec.length / 2)] : null };
    });
  }, [days, grp, startT, endT]);

  // per-day ATM decay table (one row per day in the current weekday filter)
  const dayRows = useMemo(() => {
    const rows = days.map((x) => { const s = statOf(x); if (!s) return null;
      const peakBar = s.w.find((b) => b[1] === s.hi);
      return { x, entry: s.entry, close: s.close, hi: s.hi, decay: s.decay,
        mae: s.entry ? Math.round((s.hi / s.entry - 1) * 1000) / 10 : 0,
        peakT: peakBar ? peakBar[0] : '',
        rng: s.entry ? Math.round(((s.hi - s.lo) / s.entry) * 1000) / 10 : 0 }; })
      .filter(Boolean) as { x: Day; entry: number; close: number; hi: number; decay: number; mae: number; peakT: string; rng: number }[];
    rows.sort((a, b) => tableSort === 'decay' ? a.decay - b.decay : b.x.date.localeCompare(a.x.date));
    return rows;
  }, [days, tableSort, startT, endT]);

  // weekday-aligned grid: columns = Mon..Fri, rows = weeks (newest first)
  const weekGrid = useMemo(() => {
    const mondayOf = (s: string) => { const [y, mo, da] = s.split('-').map(Number);
      const dt = new Date(Date.UTC(y, mo - 1, da)); const wd = (dt.getUTCDay() + 6) % 7;
      dt.setUTCDate(dt.getUTCDate() - wd); return dt.toISOString().slice(0, 10); };
    const m = new Map<string, (typeof dayRows[number] | null)[]>();
    dayRows.forEach((r) => { const wk = mondayOf(r.x.date);
      if (!m.has(wk)) m.set(wk, [null, null, null, null, null]);
      const ci = WDS.indexOf(r.x.weekday); if (ci >= 0) m.get(wk)![ci] = r; });
    return [...m.keys()].sort((a, b) => b.localeCompare(a)).map((wk) => ({ wk, cells: m.get(wk)! }));
  }, [dayRows]);

  // EOD decay captured by DTE (median decay a seller keeps by holding to window end)
  const eodByDte = useMemo(() => DTES.map((k) => {
    const ds = allDays.filter((x) => x.dte === k).map((x) => statOf(x)?.decay)
      .filter((v): v is number => v != null).sort((a, b) => a - b);
    if (!ds.length) return { dte: k, n: 0, med: null as number | null, win: 0 };
    return { dte: k, n: ds.length, med: ds[Math.floor(ds.length / 2)],
             win: Math.round(100 * ds.filter((x) => x < 0).length / ds.length) };
  }), [allDays, startT, endT]);

  if (!d) return <div className={styles.wrap}>Loading options study…</div>;
  const stats = days.map(statOf).filter(Boolean) as NonNullable<ReturnType<typeof statOf>>[];
  const decayed = stats.filter((s) => s.decay < 0).length;
  const medDecay = stats.length ? [...stats].map((s) => s.decay).sort((a, b) => a - b)[Math.floor(stats.length / 2)] : 0;
  const dStat = day ? statOf(day) : null;

  // registry of the line-charts so any of them can open fullscreen
  const chartsMap: Record<string, { title: string; opts: any; data: any }> = {};
  if (c1) chartsMap.c1 = { title: 'Intraday ATM straddle premium' + (day ? ` — ${fmtDMY(day.date)} · ${day.weekday} · DTE${day.dte}` : ''), opts: c1.opts, data: c1.data };
  if (cOTM) chartsMap.cOTM = { title: 'ATM straddle vs OTM strangles — % normalised', opts: cOTM.opts, data: cOTM.data };
  if (cAbs) chartsMap.cAbs = { title: 'Short-straddle P&L — absolute ₹ kept as premium decays', opts: cAbs.opts, data: cAbs.data };
  if (c2) chartsMap.c2 = { title: (wd === 'All' ? 'All days' : wd + ' days') + ' — normalised to window-start = 100', opts: c2.opts, data: c2.data };
  if (cG) chartsMap.cG = { title: 'Median decay by weekday', opts: cG.opts, data: cG.data };
  if (cDTE) chartsMap.cDTE = { title: 'Median decay by DTE', opts: cDTE.opts, data: cDTE.data };
  const ExpandBtn = ({ k }: { k: string }) => (
    <button className={styles.exp} title="Open fullscreen" onClick={() => setExpand(k)}>⤢</button>
  );

  return (
    <div className={styles.wrap}>
      <div className={styles.head}>
        <h1>Options Behaviour Study &middot; NIFTY</h1>
        <span className={styles.sub}>{d.n_days} days &middot; updated {d.generated_at}</span>
      </div>

      <div className={styles.controls}>
        <div className={styles.wdrow}>
          {['All', ...WDS].map((w) => (
            <button key={w} className={styles.wdbtn} onClick={() => setWd(w)}
              style={{ background: wd === w ? 'var(--line)' : 'transparent', fontWeight: wd === w ? 700 : 400, color: wd === w && w !== 'All' ? WD_COLOR[w] : undefined }}>
              {w} ({w === 'All' ? allDays.length : allDays.filter((x) => x.weekday === w).length})
            </button>
          ))}
        </div>
        <div className={styles.timerow}>
          <label>Window</label>
          <select value={startT} onChange={(e) => setStartT(e.target.value)}>{times.filter((t) => t < endT).map((t) => <option key={t} value={t}>{t}</option>)}</select>
          <span>→</span>
          <select value={endT} onChange={(e) => setEndT(e.target.value)}>{times.filter((t) => t > startT).map((t) => <option key={t} value={t}>{t}</option>)}</select>
        </div>
      </div>

      <div className={styles.tiles}>
        <Tile label={`Days (${wd})`} v={String(days.length)} />
        <Tile label={`Median decay ${startT}→${endT}`} v={medDecay + '%'} good={medDecay < 0} />
        <Tile label="Days straddle decayed (seller win)" v={stats.length ? Math.round((100 * decayed) / stats.length) + '%' : '—'} good />
        {day && dStat && <Tile label={`${fmtDMY(day.date)} · ${day.weekday} · DTE${day.dte}`} v={`₹${dStat.entry} → ₹${dStat.close} (${dStat.decay}%) · spot ${day.spot_move >= 0 ? '+' : ''}${day.spot_move}`} good={dStat.decay < 0} />}
      </div>

      <section className={styles.card}>
        <div className={styles.cardHead}><b>Intraday ATM straddle premium</b>
          <select className={styles.sel} value={sel ?? ''} onChange={(e) => setSel(e.target.value)}>{[...days].reverse().map((x) => <option key={x.date} value={x.date}>{fmtDMY(x.date)} &middot; {x.weekday} &middot; DTE{x.dte}</option>)}</select>
          <label className={styles.toggle}><input type="checkbox" checked={showLegs} onChange={(e) => setShowLegs(e.target.checked)} /> CE / PE split</label>
          <span className={styles.sub}>dotted grey = NIFTY spot (right axis)</span><ExpandBtn k="c1" /></div>
        {c1 && <Chart opts={c1.opts} data={c1.data} height={320} />}
      </section>

      <section className={styles.card}>
        <div className={styles.cardHead}><b>NIFTY intraday &mdash; candles + CPR</b>
          <span className={styles.sub}>{day ? fmtDMY(day.date) : ''} · {day?.cpr ? `CPR width ${day.cpr.width_pct}% (prior-day pivot range: ${Math.round(day.cpr.bc)}–${Math.round(day.cpr.tc)})` : 'no CPR'} · amber band = CPR · green up / red down candle</span></div>
        <Candles ohlc={day?.ohlc} cpr={day?.cpr} height={260} />
      </section>

      <section className={styles.card}>
        <div className={styles.cardHead}><b>ATM straddle vs OTM strangles</b><span className={styles.sub}>median normalised to 100 · {wd} days · OTM decays faster in % (cheaper, more extrinsic)</span><ExpandBtn k="cOTM" /></div>
        {cOTM && <Chart opts={cOTM.opts} data={cOTM.data} height={300} />}
      </section>

      <section className={styles.card}>
        <div className={styles.cardHead}><b>Short-straddle P&amp;L &mdash; absolute ₹</b><span className={styles.sub}>median ₹ kept as a seller (entry − premium) · {wd} days · ATM loses more % slowly but banks far more rupees</span><ExpandBtn k="cAbs" /></div>
        {cAbs && <Chart opts={cAbs.opts} data={cAbs.data} height={300} />}
      </section>

      <section className={styles.card}>
        <div className={styles.cardHead}><b>{wd === 'All' ? 'All days' : wd + ' days'}, normalised to window-start = 100</b><span className={styles.sub}>faint = each day · gold = selected · bold green = median</span><ExpandBtn k="c2" /></div>
        {c2 && <Chart opts={c2.opts} data={c2.data} height={320} />}
      </section>

      <div className={styles.two}>
        <section className={styles.card}>
          <div className={styles.cardHead}><b>Median decay by weekday</b><ExpandBtn k="cG" /></div>
          {cG && <Chart opts={cG.opts} data={cG.data} height={260} />}
        </section>
        <section className={styles.card}>
          <div className={styles.cardHead}><b>Median decay by DTE</b><span className={styles.sub}>the 0/1-DTE edge</span><ExpandBtn k="cDTE" /></div>
          {cDTE && <Chart opts={cDTE.opts} data={cDTE.data} height={260} />}
        </section>
      </div>

      <section className={styles.card}>
        <div className={styles.cardHead}><b>Median decay heatmap &mdash; weekday × DTE</b><span className={styles.sub}>green = decayed (seller profit) · red = expanded · {startT}→{endT}</span></div>
        <div className={styles.heat}>
          <div className={styles.hrow}><span className={styles.hcorner} />{DTES.map((k) => <span key={k} className={styles.hhead}>DTE{k}</span>)}</div>
          {heat.map((row, ri) => (
            <div key={ri} className={styles.hrow}>
              <span className={styles.hhead} style={{ color: WD_COLOR[WDS[ri]] }}>{WDS[ri]}</span>
              {row.map((cell, ci) => (
                <span key={ci} className={styles.hcell}
                  title={cell.med == null ? 'no data' : `${WDS[ri]} DTE${DTES[ci]}: median ${cell.med}% (n=${cell.n})`}
                  style={{ background: cell.med == null ? 'transparent' : cell.med < 0 ? `rgba(63,185,80,${Math.min(0.85, Math.abs(cell.med) / 40)})` : `rgba(248,81,73,${Math.min(0.85, Math.abs(cell.med) / 40)})`, color: cell.med == null ? 'var(--ink-faint, #6e7681)' : '#fff' }}>
                  {cell.med == null ? '·' : cell.med + '%'}
                </span>
              ))}
            </div>
          ))}
        </div>
      </section>

      <section className={styles.card}>
        <div className={styles.cardHead}><b>Decay per day &mdash; click a day</b>
          <span className={styles.sub}>green = cheaper (profit) · red = expanded · {startT}→{endT}</span>
          <select className={styles.sel} value={grp} onChange={(e) => setGrp(e.target.value)}>
            <option value="none">Chronological</option>
            <option value="weekday">Group by weekday</option>
            <option value="dte">Group by DTE</option>
            <option value="month">Group by month</option>
          </select>
        </div>
        <div className={styles.stripGroups}>
          {groups.map((g) => (
            <div key={g.key} className={styles.stripG}>
              {grp !== 'none' && <div className={styles.stripLabel}>{g.key} <span>({g.days.length}{g.med != null ? ` · med ${g.med}%` : ''})</span></div>}
              <div className={styles.strip}>
                {g.days.map((x) => { const s = statOf(x); const dc = s ? s.decay : 0;
                  return <button key={x.date} title={`${x.date} ${x.weekday} DTE${x.dte}\n${startT}→${endT}: ₹${s?.entry} → ₹${s?.close}\ndecay ${dc}%`} onClick={() => setSel(x.date)} className={`${styles.bar} ${x.date === sel ? styles.barSel : ''}`}>
                    <span style={{ height: Math.min(100, Math.abs(dc)) + '%', background: dc < 0 ? '#3fb950' : '#f85149' }} /></button>; })}
              </div>
            </div>
          ))}
        </div>
        <div className={styles.sub} style={{ marginTop: 8 }}>bar height = |decay %| (capped 100) · green = premium decayed (seller profit) · red = expanded (loss)</div>
      </section>

      <section className={styles.card}>
        <div className={styles.cardHead}><b>EOD decay captured by DTE</b>
          <span className={styles.sub}>median premium a seller keeps by holding to {endT} · % of days that decayed (seller win) · window {startT}→{endT}</span></div>
        <div className={styles.tableWrap} style={{ maxHeight: 'none' }}>
          <table className={styles.table}>
            <thead><tr><th>DTE</th><th className={styles.num}>Days</th><th className={styles.num}>Median EOD decay</th><th className={styles.num}>Seller win%</th><th className={styles.num}>Premium kept</th></tr></thead>
            <tbody>
              {eodByDte.filter((r) => r.n > 0).map((r) => (
                <tr key={r.dte}>
                  <td>DTE{r.dte}</td>
                  <td className={styles.num}>{r.n}</td>
                  <td className={styles.num} style={{ color: (r.med ?? 0) < 0 ? '#3fb950' : '#f85149', fontWeight: 700 }}>{r.med != null ? Math.round(r.med * 10) / 10 : '—'}%</td>
                  <td className={styles.num}>{r.win}%</td>
                  <td className={styles.num}>{r.med != null && r.med < 0 ? Math.round(-r.med) + '% of credit' : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className={styles.sub} style={{ marginTop: 8 }}>Decay accelerates into expiry (DTE0 biggest theta); DTE4 barely decays intraday (directional, not theta). DTE0 median ≫ mean = the gamma tail — a few expiry-day spikes drag the average.</div>
      </section>

      <section className={styles.card}>
        <div className={styles.cardHead}><b>Every day &mdash; ATM straddle curve ({wd})</b>
          <span className={styles.sub}>aligned by weekday · rows = weeks (newest top) · dashed = entry · gold dot = intraday peak · green decayed / red rose · <b>click a tile</b> to open it full-size with axes</span>
        </div>
        <div className={styles.spWeekGrid}>
          {WDS.map((w) => <div key={'h' + w} className={styles.spColHead} style={{ color: WD_COLOR[w] }}>{w}</div>)}
          {weekGrid.map(({ wk, cells }) => cells.map((r, i) => r ? (
            <button key={wk + i} className={`${styles.spCell} ${r.x.date === sel ? styles.spSel : ''}`}
              title={`${fmtDMY(r.x.date)} ${r.x.weekday} DTE${r.x.dte}\n₹${r.entry} → ₹${r.close} · peak ₹${r.hi}${r.peakT ? ' @ ' + r.peakT : ''} · decay ${r.decay}%`}
              onClick={() => { setSel(r.x.date); setExpand('c1'); }}>
              <div className={styles.spHead}>
                <span>{fmtDMY(r.x.date)} <b>D{r.x.dte}</b></span>
                <span style={{ color: r.decay < 0 ? '#3fb950' : '#f85149', fontWeight: 700 }}>{r.decay > 0 ? '+' : ''}{r.decay}%</span>
              </div>
              <Spark pts={win(getStrad(r.x)).map((b) => [toMin(b[0]), b[1]] as [number, number])} up={r.decay > 0} />
            </button>
          ) : <div key={wk + i} className={styles.spEmpty} title="no data — likely a market holiday">holiday</div>))}
        </div>
      </section>

      <section className={styles.card}>
        <div className={styles.cardHead}><b>Every day &mdash; ATM straddle decay ({wd})</b>
          <span className={styles.sub}>window {startT}→{endT} · click a row to load its intraday curve above</span>
          <select className={styles.sel} value={tableSort} onChange={(e) => setTableSort(e.target.value as 'date' | 'decay')}>
            <option value="date">Newest first</option>
            <option value="decay">Biggest decay first</option>
          </select>
        </div>
        <div className={styles.tableWrap}>
          <table className={styles.table}>
            <thead><tr><th>Date</th><th>WD</th><th>DTE</th><th className={styles.num}>Entry ₹</th><th className={styles.num} title="highest straddle premium reached intraday — a seller's peak pain">Max ₹</th><th className={styles.num}>Close ₹</th><th className={styles.num}>Decay</th><th className={styles.num} title="peak premium above entry = max adverse excursion for a seller">Peak+%</th><th className={styles.num}>Range</th><th className={styles.num}>Spot</th></tr></thead>
            <tbody>
              {dayRows.map((r) => (
                <tr key={r.x.date} onClick={() => setSel(r.x.date)} className={r.x.date === sel ? styles.trSel : ''}>
                  <td>{fmtDMY(r.x.date)}</td>
                  <td style={{ color: WD_COLOR[r.x.weekday] }}>{r.x.weekday}</td>
                  <td>DTE{r.x.dte}</td>
                  <td className={styles.num}>{r.entry}</td>
                  <td className={styles.num} style={{ color: '#d29922' }} title={r.peakT ? `peaked at ${r.peakT}` : ''}>{r.hi}</td>
                  <td className={styles.num}>{r.close}</td>
                  <td className={styles.num} style={{ color: r.decay < 0 ? '#3fb950' : '#f85149', fontWeight: 700 }}>{r.decay > 0 ? '+' : ''}{r.decay}%</td>
                  <td className={styles.num} style={{ color: r.mae > 0 ? '#f85149' : 'var(--ink-faint,#6e7681)' }}>+{r.mae}%</td>
                  <td className={styles.num}>{r.rng}%</td>
                  <td className={styles.num} style={{ color: r.x.spot_move < 0 ? '#f85149' : '#3fb950' }}>{r.x.spot_move >= 0 ? '+' : ''}{r.x.spot_move}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {expand && chartsMap[expand] && (
        <div className={styles.modal} onClick={() => setExpand(null)}>
          <div className={styles.modalInner} onClick={(e) => e.stopPropagation()}>
            <div className={styles.cardHead}><b>{chartsMap[expand].title}</b>
              <button className={styles.exp} style={{ marginLeft: 'auto' }} title="Close" onClick={() => setExpand(null)}>✕</button></div>
            <Chart opts={chartsMap[expand].opts} data={chartsMap[expand].data}
              height={Math.round((typeof window !== 'undefined' ? window.innerHeight : 720) * 0.74)} />
          </div>
        </div>
      )}
    </div>
  );
}
