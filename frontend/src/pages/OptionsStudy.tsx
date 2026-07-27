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
};

const WDS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'];
const WD_COLOR: Record<string, string> = { Mon: '#79c0ff', Tue: '#3fb950', Wed: '#e3b341', Thu: '#ff7b72', Fri: '#d2a8ff' };
const DTES = [0, 1, 2, 3, 4];
const DTE_COLOR: Record<number, string> = { 0: '#f85149', 1: '#ff9e64', 2: '#e3b341', 3: '#3fb950', 4: '#79c0ff' };
const OTM_OFFS = ['100', '200', '300'];
const OTM_COLOR: Record<string, string> = { atm: '#3fb950', '100': '#e3b341', '200': '#ff9e64', '300': '#f85149' };

const toMin = (hhmm: string) => { const [h, m] = hhmm.split(':').map(Number); return h * 60 + m - (9 * 60 + 16); };
const fmtT = (x: number) => {
  const v = 9 * 60 + 16 + x;
  return `${String(Math.floor(v / 60)).padStart(2, '0')}:${String(((v % 60) + 60) % 60).padStart(2, '0')}`;
};
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

export default function OptionsStudy() {
  const [d, setD] = useState<{ generated_at: string; n_days: number; days: Day[] } | null>(null);
  const [sel, setSel] = useState<string | null>(null);
  const [showLegs, setShowLegs] = useState(false);
  const [wd, setWd] = useState('All');
  const [startT, setStartT] = useState('09:16');
  const [endT, setEndT] = useState('15:30');
  const [grp, setGrp] = useState('none');

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
  const gridMins = useMemo(() => times.filter((t) => t >= startT && t <= endT).map(toMin), [times, startT, endT]);

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
      if (w.length >= 2 && w[0][1]) { const e = w[0][1]; w.forEach((b) => map.set(toMin(b[0]), (b[1] / e) * 100)); }
      return gridMins.map((m) => (map.has(m) ? Math.round(map.get(m)! * 10) / 10 : null));
    });
    return gridMins.map((_, i) => {
      const vals = norm.map((n) => n[i]).filter((v): v is number => v != null).sort((a, b) => a - b);
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
      if (w.length >= 2 && w[0][1]) { const e = w[0][1]; w.forEach((b) => map.set(toMin(b[0]), Math.round((b[1] / e) * 1000) / 10)); }
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

  if (!d) return <div className={styles.wrap}>Loading options study…</div>;
  const stats = days.map(statOf).filter(Boolean) as NonNullable<ReturnType<typeof statOf>>[];
  const decayed = stats.filter((s) => s.decay < 0).length;
  const medDecay = stats.length ? [...stats].map((s) => s.decay).sort((a, b) => a - b)[Math.floor(stats.length / 2)] : 0;
  const dStat = day ? statOf(day) : null;

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
        {day && dStat && <Tile label={`${day.date} · ${day.weekday} · DTE${day.dte}`} v={`₹${dStat.entry} → ₹${dStat.close} (${dStat.decay}%) · spot ${day.spot_move >= 0 ? '+' : ''}${day.spot_move}`} good={dStat.decay < 0} />}
      </div>

      <section className={styles.card}>
        <div className={styles.cardHead}><b>Intraday ATM straddle premium</b>
          <select className={styles.sel} value={sel ?? ''} onChange={(e) => setSel(e.target.value)}>{[...days].reverse().map((x) => <option key={x.date} value={x.date}>{x.date} &middot; {x.weekday} &middot; DTE{x.dte}</option>)}</select>
          <label className={styles.toggle}><input type="checkbox" checked={showLegs} onChange={(e) => setShowLegs(e.target.checked)} /> CE / PE split</label>
          <span className={styles.sub}>dotted grey = NIFTY spot (right axis)</span></div>
        {c1 && <Chart opts={c1.opts} data={c1.data} height={260} />}
      </section>

      <section className={styles.card}>
        <div className={styles.cardHead}><b>ATM straddle vs OTM strangles</b><span className={styles.sub}>median normalised to 100 · {wd} days · how much slower OTM decays</span></div>
        {cOTM && <Chart opts={cOTM.opts} data={cOTM.data} height={240} />}
      </section>

      <section className={styles.card}>
        <div className={styles.cardHead}><b>{wd === 'All' ? 'All days' : wd + ' days'}, normalised to window-start = 100</b><span className={styles.sub}>faint = each day · gold = selected · bold green = median</span></div>
        {c2 && <Chart opts={c2.opts} data={c2.data} height={240} />}
      </section>

      <div className={styles.two}>
        <section className={styles.card}>
          <div className={styles.cardHead}><b>Median decay by weekday</b></div>
          {cG && <Chart opts={cG.opts} data={cG.data} height={220} />}
        </section>
        <section className={styles.card}>
          <div className={styles.cardHead}><b>Median decay by DTE</b><span className={styles.sub}>the 0/1-DTE edge</span></div>
          {cDTE && <Chart opts={cDTE.opts} data={cDTE.data} height={220} />}
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
                  return <button key={x.date} title={`${x.date} ${x.weekday} DTE${x.dte}\n${startT}→${endT}: ₹${s?.entry} → ₹${s?.close}\ndecay ${dc}%`} onClick={() => setSel(x.date)} className={styles.bar} style={{ outline: x.date === sel ? '1px solid var(--ink)' : 'none' }}>
                    <span style={{ height: Math.min(100, Math.abs(dc)) + '%', background: dc < 0 ? '#3fb950' : '#f85149' }} /></button>; })}
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
