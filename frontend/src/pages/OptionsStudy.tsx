import { useEffect, useMemo, useRef, useState } from 'react';
import uPlot from 'uplot';
import 'uplot/dist/uPlot.min.css';
import styles from './OptionsStudy.module.css';

type Day = {
  date: string; weekday: string; dte: number; atm: number;
  entry: number; close: number; high: number; low: number;
  decay_pct: number; rng: number; spot_open: number; spot_close: number; spot_move: number;
  series: [string, number, number, number, number][]; // hhmm, straddle, ce, pe, spot
};

const WDS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'];
const WD_COLOR: Record<string, string> = { Mon: '#79c0ff', Tue: '#3fb950', Wed: '#e3b341', Thu: '#ff7b72', Fri: '#d2a8ff' };

const toMin = (hhmm: string) => {
  const [h, m] = hhmm.split(':').map(Number);
  return h * 60 + m - (9 * 60 + 16);
};
const fmtT = (x: number) => {
  const v = 9 * 60 + 16 + x;
  return `${String(Math.floor(v / 60)).padStart(2, '0')}:${String(((v % 60) + 60) % 60).padStart(2, '0')}`;
};

function Chart({ opts, data, height }: { opts: any; data: any; height: number }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const u = new (uPlot as any)({ ...opts, width: el.clientWidth, height }, data, el);
    const onR = () => el && u.setSize({ width: el.clientWidth, height });
    window.addEventListener('resize', onR);
    return () => { window.removeEventListener('resize', onR); u.destroy(); };
  }, [opts, data, height]);
  return <div ref={ref} className={styles.chart} />;
}

function Tile({ label, v, good }: { label: string; v: string; good?: boolean }) {
  return (
    <div className={styles.tile}>
      <div className={styles.tl}>{label}</div>
      <div className={styles.tv} style={{ color: good == null ? undefined : good ? '#3fb950' : '#f85149' }}>{v}</div>
    </div>
  );
}

const XAXIS = { stroke: '#8b949e', grid: { stroke: 'rgba(139,148,158,0.10)' }, values: (_u: any, v: number[]) => v.map(fmtT) };

export default function OptionsStudy() {
  const [d, setD] = useState<{ generated_at: string; n_days: number; days: Day[] } | null>(null);
  const [sel, setSel] = useState<string | null>(null);
  const [showLegs, setShowLegs] = useState(false);
  const [wd, setWd] = useState('All');
  const [startT, setStartT] = useState('09:16');
  const [endT, setEndT] = useState('15:30');

  useEffect(() => {
    fetch(`/app/options_study.json?t=${Date.now()}`, { cache: 'no-store' })
      .then((r) => r.json())
      .then((j) => { setD(j); setSel(j.days?.[j.days.length - 1]?.date ?? null); })
      .catch(() => {});
  }, []);

  const allDays = d?.days ?? [];
  const times = useMemo(() => {
    const s = new Set<string>();
    allDays.forEach((dy) => dy.series.forEach((b) => s.add(b[0])));
    return [...s].sort();
  }, [allDays]);

  const days = useMemo(() => (wd === 'All' ? allDays : allDays.filter((x) => x.weekday === wd)), [allDays, wd]);
  useEffect(() => {
    if (days.length && !days.some((x) => x.date === sel)) setSel(days[days.length - 1].date);
  }, [days, sel]);
  const day = useMemo(() => days.find((x) => x.date === sel) ?? days[days.length - 1], [days, sel]);

  // window helpers — everything is computed on [startT, endT]
  const gridMins = useMemo(() => {
    const out: number[] = [];
    for (let m = toMin(startT); m <= toMin(endT); m += 5) out.push(m);
    return out;
  }, [startT, endT]);
  const winOf = (dy: Day) => dy.series.filter((b) => b[0] >= startT && b[0] <= endT);
  const statOf = (dy: Day) => {
    const w = winOf(dy);
    if (w.length < 2) return null;
    const entry = w[0][1], close = w[w.length - 1][1];
    const strad = w.map((b) => b[1]);
    return { w, entry, close, hi: Math.max(...strad), lo: Math.min(...strad),
             decay: entry ? Math.round((close / entry - 1) * 1000) / 10 : 0 };
  };
  const medPath = (dset: Day[]) => {
    const norm = dset.map((dy) => {
      const st = statOf(dy);
      const map = new Map<number, number>();
      if (st) st.w.forEach((b) => map.set(toMin(b[0]), (b[1] / st.entry) * 100));
      return gridMins.map((m) => (map.has(m) ? map.get(m)! : null));
    });
    return gridMins.map((_, i) => {
      const vals = norm.map((n) => n[i]).filter((v): v is number => v != null).sort((a, b) => a - b);
      return vals.length ? Math.round(vals[Math.floor(vals.length / 2)] * 10) / 10 : null;
    });
  };

  // Chart 1 — intraday straddle (window) + NIFTY spot dotted on right axis
  const c1 = useMemo(() => {
    if (!day) return null;
    const st = statOf(day);
    if (!st) return null;
    const w = st.w;
    const xs = w.map((b) => toMin(b[0]));
    const data: any = showLegs
      ? [xs, w.map((b) => b[1]), w.map((b) => b[2]), w.map((b) => b[3]), w.map((b) => b[4])]
      : [xs, w.map((b) => b[1]), w.map((b) => b[4])];
    const series: any[] = [{}, { label: 'Straddle', stroke: '#3fb950', width: 2, points: { show: false } }];
    if (showLegs) {
      series.push({ label: 'CE', stroke: '#e3b341', width: 1, points: { show: false } });
      series.push({ label: 'PE', stroke: '#79c0ff', width: 1, points: { show: false } });
    }
    series.push({ label: 'NIFTY', scale: 'spot', stroke: '#8b949e', width: 1, dash: [4, 4], points: { show: false } });
    const opts: any = {
      series, scales: { spot: {} },
      axes: [XAXIS,
        { stroke: '#8b949e', grid: { stroke: 'rgba(139,148,158,0.10)' }, values: (_u: any, v: number[]) => v.map((x) => '₹' + x) },
        { scale: 'spot', side: 1, stroke: '#8b949e', grid: { show: false }, values: (_u: any, v: number[]) => v.map(String) },
      ],
      legend: { show: true }, cursor: { drag: { x: true, y: false } },
    };
    return { opts, data };
  }, [day, showLegs, startT, endT]);

  // Chart 2 — all (filtered) days normalised to window-start = 100, + median
  const c2 = useMemo(() => {
    if (!days.length) return null;
    const norm = days.map((dy) => {
      const st = statOf(dy);
      const map = new Map<number, number>();
      if (st) st.w.forEach((b) => map.set(toMin(b[0]), Math.round((b[1] / st.entry) * 1000) / 10));
      return gridMins.map((m) => (map.has(m) ? map.get(m)! : null));
    });
    const data: any = [gridMins, ...norm, medPath(days)];
    const series: any[] = [{}];
    days.forEach((dy) => series.push({
      label: dy.date, stroke: dy.date === sel ? '#e3b341' : 'rgba(139,148,158,0.15)',
      width: dy.date === sel ? 2 : 1, points: { show: false },
    }));
    series.push({ label: 'median', stroke: '#3fb950', width: 2.5, points: { show: false } });
    const opts: any = {
      series, axes: [XAXIS, { stroke: '#8b949e', grid: { stroke: 'rgba(139,148,158,0.10)' }, values: (_u: any, v: number[]) => v.map((x) => x + '%') }],
      legend: { show: false }, cursor: { drag: { x: false, y: false }, points: { show: false } },
    };
    return { opts, data };
  }, [days, sel, startT, endT]);

  // Chart 3 — median decay path grouped by weekday (all days, aligned on the window)
  const cG = useMemo(() => {
    if (!allDays.length) return null;
    const data: any = [gridMins, ...WDS.map((w) => medPath(allDays.filter((x) => x.weekday === w)))];
    const series: any[] = [{}, ...WDS.map((w) => ({ label: w, stroke: WD_COLOR[w], width: 2, points: { show: false } }))];
    const opts: any = {
      series, axes: [XAXIS, { stroke: '#8b949e', grid: { stroke: 'rgba(139,148,158,0.10)' }, values: (_u: any, v: number[]) => v.map((x) => x + '%') }],
      legend: { show: true }, cursor: { drag: { x: false, y: false } },
    };
    return { opts, data };
  }, [allDays, startT, endT]);

  if (!d) return <div className={styles.wrap}>Loading options study…</div>;

  const stats = days.map(statOf).filter(Boolean) as NonNullable<ReturnType<typeof statOf>>[];
  const decayed = stats.filter((s) => s.decay < 0).length;
  const medDecay = stats.length ? [...stats].map((s) => s.decay).sort((a, b) => a - b)[Math.floor(stats.length / 2)] : 0;
  const dStat = day ? statOf(day) : null;

  return (
    <div className={styles.wrap}>
      <div className={styles.head}>
        <h1>Options Behaviour Study &middot; NIFTY ATM straddle</h1>
        <span className={styles.sub}>{d.n_days} days &middot; updated {d.generated_at}</span>
      </div>

      {/* controls: weekday filter + time window */}
      <div className={styles.controls}>
        <div className={styles.wdrow}>
          {['All', ...WDS].map((w) => (
            <button key={w} className={styles.wdbtn} onClick={() => setWd(w)}
              style={{ background: wd === w ? 'var(--line)' : 'transparent', fontWeight: wd === w ? 700 : 400,
                       color: wd === w && w !== 'All' ? WD_COLOR[w] : undefined }}>
              {w} ({w === 'All' ? allDays.length : allDays.filter((x) => x.weekday === w).length})
            </button>
          ))}
        </div>
        <div className={styles.timerow}>
          <label>Window</label>
          <select value={startT} onChange={(e) => setStartT(e.target.value)}>
            {times.filter((t) => t < endT).map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
          <span>→</span>
          <select value={endT} onChange={(e) => setEndT(e.target.value)}>
            {times.filter((t) => t > startT).map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
        </div>
      </div>

      <div className={styles.tiles}>
        <Tile label={`Days (${wd})`} v={String(days.length)} />
        <Tile label={`Median decay ${startT}→${endT}`} v={medDecay + '%'} good={medDecay < 0} />
        <Tile label="Days straddle decayed (seller win)" v={stats.length ? Math.round((100 * decayed) / stats.length) + '%' : '—'} good />
        {day && dStat && (
          <Tile label={`${day.date} · ${day.weekday} · DTE${day.dte}`}
            v={`₹${dStat.entry} → ₹${dStat.close} (${dStat.decay}%) · spot ${day.spot_move >= 0 ? '+' : ''}${day.spot_move}`}
            good={dStat.decay < 0} />
        )}
      </div>

      <section className={styles.card}>
        <div className={styles.cardHead}>
          <b>Intraday ATM straddle premium</b>
          <select className={styles.sel} value={sel ?? ''} onChange={(e) => setSel(e.target.value)}>
            {[...days].reverse().map((x) => <option key={x.date} value={x.date}>{x.date} &middot; {x.weekday} &middot; DTE{x.dte}</option>)}
          </select>
          <label className={styles.toggle}><input type="checkbox" checked={showLegs} onChange={(e) => setShowLegs(e.target.checked)} /> CE / PE split</label>
          <span className={styles.sub}>dotted grey = NIFTY spot (right axis)</span>
        </div>
        {c1 && <Chart opts={c1.opts} data={c1.data} height={260} />}
      </section>

      <section className={styles.card}>
        <div className={styles.cardHead}>
          <b>All {wd} days, normalised to window-start = 100</b>
          <span className={styles.sub}>faint = each day &middot; gold = selected &middot; bold green = median</span>
        </div>
        {c2 && <Chart opts={c2.opts} data={c2.data} height={260} />}
      </section>

      <section className={styles.card}>
        <div className={styles.cardHead}>
          <b>Median decay by weekday</b>
          <span className={styles.sub}>each line = that weekday's median straddle path over the window ({startT}→{endT})</span>
        </div>
        {cG && <Chart opts={cG.opts} data={cG.data} height={260} />}
      </section>

      <section className={styles.card}>
        <div className={styles.cardHead}>
          <b>Decay per day ({startT}→{endT}) &mdash; click a day</b>
          <span className={styles.sub}>green = straddle got cheaper (seller profit) &middot; red = expanded</span>
        </div>
        <div className={styles.strip}>
          {days.map((x) => {
            const s = statOf(x);
            const dc = s ? s.decay : 0;
            return (
              <button key={x.date} title={`${x.date} ${x.weekday} DTE${x.dte}\n${startT}→${endT}: ₹${s?.entry} → ₹${s?.close}\ndecay ${dc}%`}
                onClick={() => setSel(x.date)} className={styles.bar}
                style={{ outline: x.date === sel ? '1px solid var(--ink)' : 'none' }}>
                <span style={{ height: Math.min(100, Math.abs(dc)) + '%', background: dc < 0 ? '#3fb950' : '#f85149' }} />
              </button>
            );
          })}
        </div>
      </section>
    </div>
  );
}
