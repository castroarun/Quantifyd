/**
 * Index Pulse — three layers:
 *   1. regime strip: every index as a clickable tile (day chg + sparkline)
 *   2. comparison chart: rebased growth-of-1, absolute or RS-vs-Nifty50
 *   3. drill-down for the focused index: top-10 by weight-proxy, leaders, laggards
 */

import { useEffect, useState } from 'react';
import { apiGet } from '../api/client';
import styles from './IndexPulse.module.css';

type Tile = { symbol: string; label: string; group: string; last: number;
              chg_pct: number; spark: number[]; asof: string };
type Compare = { window: string; mode: string; dates: string[];
                 series: Record<string, number[]>;
                 perf: { symbol: string; label: string; ret_pct: number }[] };
type DrillRow = { symbol: string; last: number; chg_1d_pct: number;
                  ret_window_pct: number; turnover_cr: number;
                  weight_proxy_pct: number };
type Drill = { error?: string; index: string; label: string; window: string;
               n_constituents: number; n_with_data: number; weight_note: string;
               top_by_weight: DrillRow[]; leaders: DrillRow[]; laggards: DrillRow[] };

const WINDOWS = ['1w', '1m', '3m', '6m', '1y', '3y'] as const;
const PALETTE = ['#4e8ef7', '#f2b632', '#43b97f', '#e05d5d', '#9b6ef3', '#31b0c6',
                 '#e78b3a', '#7a9a01', '#d05ca8', '#8a8f98', '#c9b458', '#5bc0de',
                 '#b0713f', '#66a5ad', '#c96567'];

function Spark({ v }: { v: number[] }) {
  if (v.length < 2) return null;
  const W = 64, H = 22, lo = Math.min(...v), hi = Math.max(...v);
  const x = (i: number) => (i / (v.length - 1)) * W;
  const y = (p: number) => H - ((p - lo) / (hi - lo || 1)) * (H - 3) - 1.5;
  const d = v.map((p, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(1)},${y(p).toFixed(1)}`).join(' ');
  const up = v[v.length - 1] >= v[0];
  return (
    <svg width={W} height={H} className={styles.spark}>
      <path d={d} fill="none" strokeWidth="1.5"
            stroke={up ? 'var(--accent-pos, #1f9d55)' : 'var(--accent-neg, #d23f3f)'} />
    </svg>
  );
}

function CompareChart({ c, focus, onFocus }: {
  c: Compare; focus: string | null; onFocus: (s: string) => void;
}) {
  const syms = Object.keys(c.series);
  if (!syms.length) return <div className={styles.empty}>No series.</div>;
  const W = 880, H = 340, P = { l: 8, r: 92, t: 10, b: 22 };
  const all = syms.flatMap((s) => c.series[s]);
  const lo = Math.min(...all), hi = Math.max(...all);
  const n = c.dates.length;
  const x = (i: number) => P.l + (i / Math.max(n - 1, 1)) * (W - P.l - P.r);
  const y = (v: number) => P.t + (1 - (v - lo) / (hi - lo || 1)) * (H - P.t - P.b);
  const color = (s: string) => PALETTE[syms.indexOf(s) % PALETTE.length];
  const months: { i: number; lbl: string }[] = [];
  let lastM = '';
  c.dates.forEach((d, i) => {
    const m = d.slice(0, 7);
    if (m !== lastM) { months.push({ i, lbl: d.slice(5, 7) + '/' + d.slice(2, 4) }); lastM = m; }
  });
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className={styles.chart} preserveAspectRatio="none">
      <line x1={P.l} x2={W - P.r} y1={y(1)} y2={y(1)}
            stroke="var(--ink-muted)" strokeDasharray="4 4" strokeWidth="0.7" opacity="0.5" />
      {months.filter((_, k) => k % Math.ceil(months.length / 8) === 0).map((m) => (
        <text key={m.i} x={x(m.i)} y={H - 6} fontSize="9" fill="var(--ink-muted)">{m.lbl}</text>
      ))}
      {syms.map((s) => {
        const v = c.series[s];
        const d = v.map((p, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(1)},${y(p).toFixed(1)}`).join(' ');
        const dim = focus && focus !== s;
        return (
          <g key={s} onClick={() => onFocus(s)} style={{ cursor: 'pointer' }}>
            <path d={d} fill="none" stroke={color(s)}
                  strokeWidth={focus === s ? 2.6 : 1.3} opacity={dim ? 0.18 : 1} />
            <text x={W - P.r + 4} y={y(v[v.length - 1]) + 3} fontSize="10"
                  fill={color(s)} opacity={dim ? 0.3 : 1}>
              {c.perf.find((p) => p.symbol === s)?.label ?? s}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

export default function IndexPulse() {
  const [tiles, setTiles] = useState<Tile[] | null>(null);
  const [cmp, setCmp] = useState<Compare | null>(null);
  const [drill, setDrill] = useState<Drill | null>(null);
  const [win, setWin] = useState<string>('3m');
  const [mode, setMode] = useState<'abs' | 'rs'>('abs');
  const [focus, setFocus] = useState<string>('NIFTY50');
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    apiGet<{ indices: Tile[] }>('/api/index-pulse/strip')
      .then((r) => setTiles(r.indices)).catch((e) => setErr(String(e)));
  }, []);
  useEffect(() => {
    apiGet<Compare>(`/api/index-pulse/compare?window=${win}&mode=${mode}`)
      .then(setCmp).catch((e) => setErr(String(e)));
  }, [win, mode]);
  useEffect(() => {
    if (!focus || focus === 'INDIAVIX') { setDrill(null); return; }
    setDrill(null);
    apiGet<Drill>(`/api/index-pulse/drilldown?index=${focus}&window=${win}`)
      .then(setDrill).catch((e) => setErr(String(e)));
  }, [focus, win]);

  const tile = (t: Tile) => (
    <div key={t.symbol}
         className={`${styles.tile} ${focus === t.symbol ? styles.tileActive : ''}`}
         onClick={() => setFocus(t.symbol)}>
      <div className={styles.tileLabel}>{t.label}</div>
      <div className={styles.tileVal}>{t.last.toLocaleString('en-IN')}</div>
      <div className={t.chg_pct >= 0 ? styles.pos : styles.neg}>
        {(t.chg_pct >= 0 ? '+' : '') + t.chg_pct.toFixed(2)}%
      </div>
      <Spark v={t.spark} />
    </div>
  );

  return (
    <div className={styles.root}>
      <div className={styles.headerRow}>
        <div>
          <h1 className={styles.title}>Index Pulse</h1>
          <p className={styles.sub}>
            Broad + sector indices, rebased comparison, and constituent drill-down ·
            click a tile or a chart line to focus{tiles?.length ? ` · data as of ${tiles[0].asof}` : ''}
          </p>
        </div>
      </div>

      {err && <div className={styles.error}>{err}</div>}

      {/* Layer 1: regime strip */}
      {tiles && (
        <>
          <div className={styles.stripGroupLabel}>Broad</div>
          <div className={styles.strip}>{tiles.filter((t) => t.group === 'broad').map(tile)}</div>
          <div className={styles.stripGroupLabel}>Sectors · Volatility</div>
          <div className={styles.strip}>
            {tiles.filter((t) => t.group !== 'broad').map(tile)}
          </div>
        </>
      )}

      {/* Layer 2: comparison */}
      <div className={styles.card}>
        <div className={styles.cardTitle}>
          Growth of ₹1 · {mode === 'rs' ? 'relative to Nifty 50 (RS)' : 'absolute'}
          <div className={styles.segRow}>
            {WINDOWS.map((w) => (
              <button key={w} className={`${styles.seg} ${win === w ? styles.segActive : ''}`}
                      onClick={() => setWin(w)}>{w.toUpperCase()}</button>
            ))}
            <span className={styles.segGap} />
            <button className={`${styles.seg} ${mode === 'abs' ? styles.segActive : ''}`}
                    onClick={() => setMode('abs')}>Absolute</button>
            <button className={`${styles.seg} ${mode === 'rs' ? styles.segActive : ''}`}
                    onClick={() => setMode('rs')}>RS vs Nifty50</button>
          </div>
        </div>
        {mode === 'rs' && (
          <p className={styles.note}>
            Each line is the index divided by Nifty 50, rebased to 1 — a rising line is
            outperforming the market, flat = matching it. This separates leadership from
            the market's own drift.
          </p>
        )}
        {cmp ? <CompareChart c={cmp} focus={focus} onFocus={setFocus} />
             : <div className={styles.empty}>Loading…</div>}
        {cmp && (
          <div className={styles.perfRow}>
            {cmp.perf.map((p) => (
              <button key={p.symbol}
                      className={`${styles.perfChip} ${focus === p.symbol ? styles.perfChipActive : ''}`}
                      onClick={() => setFocus(p.symbol)}>
                {p.label}
                <b className={p.ret_pct >= 0 ? styles.pos : styles.neg}>
                  {(p.ret_pct >= 0 ? '+' : '') + p.ret_pct.toFixed(1)}%
                </b>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Layer 3: drill-down */}
      {focus !== 'INDIAVIX' && (
        <div className={styles.card}>
          <div className={styles.cardTitle}>
            {drill ? `${drill.label} · inside the move (${win.toUpperCase()})` : 'Loading constituents…'}
            {drill && <span className={styles.count}>
              {drill.n_with_data} of {drill.n_constituents} constituents with data
            </span>}
          </div>
          {drill?.error && <div className={styles.empty}>{drill.error}</div>}
          {drill && !drill.error && (
            <div className={styles.drillGrid}>
              <div>
                <div className={styles.drillHead}>Top 10 by weight*</div>
                <table className={styles.table}>
                  <thead><tr><th>Stock</th><th>Wt*</th><th>1D</th><th>{win.toUpperCase()}</th></tr></thead>
                  <tbody>
                    {drill.top_by_weight.map((r) => (
                      <tr key={r.symbol}>
                        <td className={styles.sym}>{r.symbol}</td>
                        <td>{r.weight_proxy_pct.toFixed(1)}%</td>
                        <td className={r.chg_1d_pct >= 0 ? styles.pos : styles.neg}>
                          {(r.chg_1d_pct >= 0 ? '+' : '') + r.chg_1d_pct.toFixed(1)}%</td>
                        <td className={r.ret_window_pct >= 0 ? styles.pos : styles.neg}>
                          {(r.ret_window_pct >= 0 ? '+' : '') + r.ret_window_pct.toFixed(1)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div>
                <div className={styles.drillHead}>Leaders · {win.toUpperCase()}</div>
                <table className={styles.table}>
                  <thead><tr><th>Stock</th><th>{win.toUpperCase()}</th><th>1D</th><th>T/O ₹cr</th></tr></thead>
                  <tbody>
                    {drill.leaders.map((r) => (
                      <tr key={r.symbol}>
                        <td className={styles.sym}>{r.symbol}</td>
                        <td className={styles.pos}>+{r.ret_window_pct.toFixed(1)}%</td>
                        <td className={r.chg_1d_pct >= 0 ? styles.pos : styles.neg}>
                          {(r.chg_1d_pct >= 0 ? '+' : '') + r.chg_1d_pct.toFixed(1)}%</td>
                        <td className={styles.muted}>{r.turnover_cr.toFixed(0)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div>
                <div className={styles.drillHead}>Laggards · {win.toUpperCase()}</div>
                <table className={styles.table}>
                  <thead><tr><th>Stock</th><th>{win.toUpperCase()}</th><th>1D</th><th>T/O ₹cr</th></tr></thead>
                  <tbody>
                    {drill.laggards.map((r) => (
                      <tr key={r.symbol}>
                        <td className={styles.sym}>{r.symbol}</td>
                        <td className={r.ret_window_pct >= 0 ? styles.pos : styles.neg}>
                          {(r.ret_window_pct >= 0 ? '+' : '') + r.ret_window_pct.toFixed(1)}%</td>
                        <td className={r.chg_1d_pct >= 0 ? styles.pos : styles.neg}>
                          {(r.chg_1d_pct >= 0 ? '+' : '') + r.chg_1d_pct.toFixed(1)}%</td>
                        <td className={styles.muted}>{r.turnover_cr.toFixed(0)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
          {drill && !drill.error && <p className={styles.note}>* {drill.weight_note}</p>}
        </div>
      )}
    </div>
  );
}
