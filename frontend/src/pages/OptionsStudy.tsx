import { useEffect, useMemo, useRef, useState } from 'react';
import uPlot from 'uplot';
import 'uplot/dist/uPlot.min.css';
import styles from './OptionsStudy.module.css';

type Day = {
  date: string; weekday: string; dte: number; atm: number;
  entry: number; close: number; high: number; low: number;
  decay_pct: number; rng: number; spot_open: number; spot_close: number; spot_move: number;
  series: [string, number, number, number][];
};

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

export default function OptionsStudy() {
  const [d, setD] = useState<{ generated_at: string; n_days: number; days: Day[] } | null>(null);
  const [sel, setSel] = useState<string | null>(null);
  const [showLegs, setShowLegs] = useState(false);

  useEffect(() => {
    fetch(`/app/options_study.json?t=${Date.now()}`, { cache: 'no-store' })
      .then((r) => r.json())
      .then((j) => { setD(j); setSel(j.days?.[j.days.length - 1]?.date ?? null); })
      .catch(() => {});
  }, []);

  const days = d?.days ?? [];
  const day = useMemo(() => days.find((x) => x.date === sel) ?? days[days.length - 1], [days, sel]);

  // Chart 1 — intraday straddle for the selected day
  const c1 = useMemo(() => {
    if (!day) return null;
    const xs = day.series.map((s) => toMin(s[0]));
    const data: any = showLegs
      ? [xs, day.series.map((s) => s[1]), day.series.map((s) => s[2]), day.series.map((s) => s[3])]
      : [xs, day.series.map((s) => s[1])];
    const series: any[] = [{}, { label: 'Straddle', stroke: '#3fb950', width: 2, points: { show: false } }];
    if (showLegs) {
      series.push({ label: 'CE', stroke: '#e3b341', width: 1, points: { show: false } });
      series.push({ label: 'PE', stroke: '#79c0ff', width: 1, points: { show: false } });
    }
    const opts: any = {
      series,
      axes: [
        { stroke: '#8b949e', grid: { stroke: 'rgba(139,148,158,0.10)' }, values: (_u: any, v: number[]) => v.map(fmtT) },
        { stroke: '#8b949e', grid: { stroke: 'rgba(139,148,158,0.10)' }, values: (_u: any, v: number[]) => v.map((x) => '₹' + x) },
      ],
      legend: { show: true },
      cursor: { drag: { x: true, y: false } },
    };
    return { opts, data };
  }, [day, showLegs]);

  // Chart 2 — all days normalised to entry = 100, + median path
  const c2 = useMemo(() => {
    if (!days.length) return null;
    const grid: number[] = [];
    for (let m = 0; m <= 374; m += 5) grid.push(m);
    const norm = days.map((dy) => {
      const map = new Map<number, number>();
      dy.series.forEach((s) => map.set(toMin(s[0]), (s[1] / dy.entry) * 100));
      return grid.map((m) => (map.has(m) ? Math.round(map.get(m)! * 10) / 10 : null));
    });
    const med = grid.map((_, i) => {
      const vals = norm.map((n) => n[i]).filter((v): v is number => v != null).sort((a, b) => a - b);
      return vals.length ? vals[Math.floor(vals.length / 2)] : null;
    });
    const data: any = [grid, ...norm, med];
    const series: any[] = [{}];
    days.forEach((dy) => series.push({
      label: dy.date,
      stroke: dy.date === sel ? '#e3b341' : 'rgba(139,148,158,0.15)',
      width: dy.date === sel ? 2 : 1,
      points: { show: false },
    }));
    series.push({ label: 'median', stroke: '#3fb950', width: 2.5, points: { show: false } });
    const opts: any = {
      series,
      axes: [
        { stroke: '#8b949e', grid: { stroke: 'rgba(139,148,158,0.10)' }, values: (_u: any, v: number[]) => v.map(fmtT) },
        { stroke: '#8b949e', grid: { stroke: 'rgba(139,148,158,0.10)' }, values: (_u: any, v: number[]) => v.map((x) => x + '%') },
      ],
      legend: { show: false },
      cursor: { drag: { x: false, y: false }, points: { show: false } },
    };
    return { opts, data };
  }, [days, sel]);

  if (!d) return <div className={styles.wrap}>Loading options study…</div>;

  const decayed = days.filter((x) => x.decay_pct < 0).length;
  const sorted = [...days].map((x) => x.decay_pct).sort((a, b) => a - b);
  const medDecay = sorted[Math.floor(sorted.length / 2)];

  return (
    <div className={styles.wrap}>
      <div className={styles.head}>
        <h1>Options Behaviour Study &middot; NIFTY ATM straddle</h1>
        <span className={styles.sub}>{d.n_days} recorded days &middot; updated {d.generated_at}</span>
      </div>

      <div className={styles.tiles}>
        <Tile label="Recorded days" v={String(d.n_days)} />
        <Tile label="Median decay (entry→close)" v={medDecay + '%'} good={medDecay < 0} />
        <Tile label="Days straddle decayed (seller win)" v={Math.round((100 * decayed) / days.length) + '%'} good />
        {day && (
          <Tile
            label={`${day.date} · ${day.weekday} · DTE${day.dte} · ATM ${day.atm}`}
            v={`₹${day.entry} → ₹${day.close} (${day.decay_pct}%) · spot ${day.spot_move >= 0 ? '+' : ''}${day.spot_move}`}
            good={day.decay_pct < 0}
          />
        )}
      </div>

      <section className={styles.card}>
        <div className={styles.cardHead}>
          <b>Intraday ATM straddle premium</b>
          <select className={styles.sel} value={sel ?? ''} onChange={(e) => setSel(e.target.value)}>
            {[...days].reverse().map((x) => (
              <option key={x.date} value={x.date}>{x.date} &middot; {x.weekday} &middot; DTE{x.dte}</option>
            ))}
          </select>
          <label className={styles.toggle}>
            <input type="checkbox" checked={showLegs} onChange={(e) => setShowLegs(e.target.checked)} /> CE / PE split
          </label>
        </div>
        {c1 && <Chart opts={c1.opts} data={c1.data} height={260} />}
      </section>

      <section className={styles.card}>
        <div className={styles.cardHead}>
          <b>All days, normalised to entry = 100</b>
          <span className={styles.sub}>faint = each day &middot; gold = selected &middot; bold green = median decay path</span>
        </div>
        {c2 && <Chart opts={c2.opts} data={c2.data} height={260} />}
      </section>

      <section className={styles.card}>
        <div className={styles.cardHead}>
          <b>Daily decay (entry→close) &mdash; click a day</b>
          <span className={styles.sub}>green = straddle got cheaper (seller profit) &middot; red = expanded</span>
        </div>
        <div className={styles.strip}>
          {days.map((x) => (
            <button
              key={x.date}
              title={`${x.date} ${x.weekday} DTE${x.dte}\nentry ₹${x.entry} → close ₹${x.close}\ndecay ${x.decay_pct}% · range ₹${x.rng} · spot ${x.spot_move >= 0 ? '+' : ''}${x.spot_move}`}
              onClick={() => setSel(x.date)}
              className={styles.bar}
              style={{ outline: x.date === sel ? '1px solid var(--ink)' : 'none' }}
            >
              <span style={{ height: Math.min(100, Math.abs(x.decay_pct)) + '%', background: x.decay_pct < 0 ? '#3fb950' : '#f85149' }} />
            </button>
          ))}
        </div>
      </section>
    </div>
  );
}
