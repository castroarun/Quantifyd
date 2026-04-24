import { useEffect, useMemo, useState } from 'react';
import { apiGet } from '../../api/client';
import styles from './BookPnLChart.module.css';

/**
 * Live book-P&L chart for the ORB dashboard.
 *
 * Polls `/api/orb/book-pnl-series` every 10 seconds and draws the
 * running aggregate (realized + unrealized) P&L as a single SVG line
 * with horizontal reference bands at the soft (-Rs 7,500) and hard
 * (-Rs 15,000) drawdown-cut thresholds, plus a breakeven line.
 *
 * No external charting dependency — pure SVG so bundle stays small.
 */

type Sample = { ts: string; pnl_inr: number; realized?: number; unrealized?: number };
type Series = {
  series: Sample[];
  threshold_soft_inr: number;
  threshold_hard_inr: number;
  current: number | null;
};

const W = 900;           // viewBox width
const H = 220;           // viewBox height
const PAD_L = 60;
const PAD_R = 20;
const PAD_T = 20;
const PAD_B = 28;
const POLL_MS = 10_000;

function formatInr(v: number | null): string {
  if (v == null || !Number.isFinite(v)) return '—';
  const sign = v < 0 ? '−' : v > 0 ? '+' : '';
  return `${sign}Rs ${Math.abs(Math.round(v)).toLocaleString('en-IN')}`;
}

function hhmm(iso: string): string {
  const d = new Date(iso);
  return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
}

export default function BookPnLChart() {
  const [data, setData] = useState<Series | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const r = await apiGet<Series>('/api/orb/book-pnl-series');
        if (!cancelled) {
          setData(r);
          setErr(null);
        }
      } catch (e: any) {
        if (!cancelled) setErr(String(e?.message ?? e));
      }
    }
    load();
    const t = window.setInterval(load, POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(t);
    };
  }, []);

  const plot = useMemo(() => {
    if (!data || data.series.length === 0) return null;
    const soft = data.threshold_soft_inr;
    const hard = data.threshold_hard_inr;
    const values = data.series.map(s => s.pnl_inr);
    const rawMin = Math.min(...values, hard, -500);
    const rawMax = Math.max(...values, 500, Math.abs(soft));
    // Symmetric padding — always show breakeven + both thresholds
    const padY = Math.max(1500, (rawMax - rawMin) * 0.08);
    const yMin = rawMin - padY;
    const yMax = rawMax + padY;
    const yScale = (v: number) =>
      PAD_T + (1 - (v - yMin) / (yMax - yMin)) * (H - PAD_T - PAD_B);
    const xScale = (i: number) =>
      PAD_L + (i / Math.max(data.series.length - 1, 1)) * (W - PAD_L - PAD_R);

    const linePath = data.series.map((s, i) => {
      const x = xScale(i);
      const y = yScale(s.pnl_inr);
      return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(' ');

    // Area path under the line (gradient fill)
    const zeroY = yScale(0);
    const lastX = xScale(data.series.length - 1);
    const firstX = xScale(0);
    const areaPath = `${linePath} L${lastX.toFixed(1)},${zeroY.toFixed(1)} L${firstX.toFixed(1)},${zeroY.toFixed(1)} Z`;

    // Y-axis tick labels
    const ticks = [yMax, soft === 0 ? null : soft, 0, hard === 0 ? null : hard, yMin]
      .filter((v): v is number => v !== null);

    // X-axis time labels — first, middle, last
    const xTicks =
      data.series.length >= 2
        ? [0, Math.floor(data.series.length / 2), data.series.length - 1]
        : [0];

    return { linePath, areaPath, yScale, xScale, yMin, yMax, soft, hard, zeroY, ticks, xTicks };
  }, [data]);

  if (err) return <div className={styles.card}><div className={styles.errMsg}>Chart error: {err}</div></div>;
  if (!data) return <div className={styles.card}><div className={styles.loading}>Loading book P&amp;L chart…</div></div>;

  const current = data.current;
  const last = data.series[data.series.length - 1];
  const pnlClass =
    current == null ? '' : current >= 0 ? styles.pnlPos : styles.pnlNeg;

  return (
    <div className={styles.card}>
      <div className={styles.header}>
        <div className={styles.title}>Book P&amp;L · live</div>
        <div className={styles.currentRow}>
          <span className={`${styles.current} ${pnlClass}`}>{formatInr(current)}</span>
          {last ? <span className={styles.stamp}>@ {hhmm(last.ts)}</span> : null}
          <span className={styles.points}>{data.series.length} samples</span>
        </div>
      </div>

      {plot && data.series.length >= 2 ? (
        <svg viewBox={`0 0 ${W} ${H}`} className={styles.svg} preserveAspectRatio="none">
          <defs>
            <linearGradient id="pnlGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--pnl-grad-top, rgba(52,211,153,0.22))" />
              <stop offset="100%" stopColor="var(--pnl-grad-bot, rgba(52,211,153,0.02))" />
            </linearGradient>
          </defs>

          {/* threshold bands */}
          <rect
            x={PAD_L} y={plot.yScale(0)}
            width={W - PAD_L - PAD_R} height={Math.max(0, plot.yScale(plot.hard) - plot.yScale(0))}
            className={styles.bandLoss}
          />

          {/* Y grid + labels */}
          {plot.ticks.map(v => {
            const y = plot.yScale(v);
            const isHard = v === plot.hard;
            const isSoft = v === plot.soft;
            const isZero = v === 0;
            const cls = isHard
              ? styles.gridHard
              : isSoft
              ? styles.gridSoft
              : isZero
              ? styles.gridZero
              : styles.gridMinor;
            return (
              <g key={v}>
                <line x1={PAD_L} x2={W - PAD_R} y1={y} y2={y} className={cls} />
                <text x={PAD_L - 8} y={y + 3} textAnchor="end" className={styles.yLabel}>
                  {formatInr(v)}
                </text>
              </g>
            );
          })}

          {/* X time labels */}
          {plot.xTicks.map(i => {
            const x = plot.xScale(i);
            const s = data.series[i];
            if (!s) return null;
            return (
              <text key={i} x={x} y={H - 8} textAnchor="middle" className={styles.xLabel}>
                {hhmm(s.ts)}
              </text>
            );
          })}

          {/* area + line */}
          <path d={plot.areaPath} fill="url(#pnlGrad)" />
          <path d={plot.linePath} className={current != null && current >= 0 ? styles.lineGreen : styles.lineRed} />
        </svg>
      ) : (
        <div className={styles.waiting}>Waiting for first monitor tick…</div>
      )}
    </div>
  );
}
