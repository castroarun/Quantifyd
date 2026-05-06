/**
 * Journal Insights — equity curve, drawdown windows, per-strategy
 * attribution, win-rate by tag, R-distribution.
 *
 * Route: /journal/insights
 */

import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import styles from './Journal.module.css';
import { apiGet } from '../api/client';
import type { JournalInsightsResponse } from '../api/types';
import { formatPnl } from '../utils/format';

const RANGE_OPTIONS = [
  { key: '30', label: 'Last 30 days', days: 30 },
  { key: '90', label: 'Last 90 days', days: 90 },
  { key: '180', label: 'Last 180 days', days: 180 },
  { key: '365', label: 'Last 365 days', days: 365 },
];

function isoNDaysAgo(n: number): string {
  const d = new Date();
  d.setDate(d.getDate() - n);
  return d.toISOString().slice(0, 10);
}

export default function JournalInsights() {
  const navigate = useNavigate();
  const [rangeKey, setRangeKey] = useState('365');
  const [data, setData] = useState<JournalInsightsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    const days = RANGE_OPTIONS.find((r) => r.key === rangeKey)?.days || 180;
    const from = isoNDaysAgo(days);
    apiGet<JournalInsightsResponse>(`/api/journal/insights?from=${from}`)
      .then((d) => {
        if (cancelled) return;
        setData(d);
        setLoading(false);
      })
      .catch((e: Error) => {
        if (cancelled) return;
        setErr(e.message);
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [rangeKey]);

  if (loading || !data) {
    return (
      <div className={styles.root}>
        <div className={styles.loading}>Loading insights…</div>
      </div>
    );
  }

  return (
    <div className={styles.root}>
      <div className={styles.headerRow}>
        <div>
          <div className={styles.eyebrow}>Performance review</div>
          <h1 className={styles.title}>Insights</h1>
          <div className={styles.sub}>
            {data.from} to {data.to} · {data.metrics.trades} trades
          </div>
        </div>
        <div className={styles.actions}>
          {RANGE_OPTIONS.map((r) => (
            <span
              key={r.key}
              className={`${styles.chipFilter} ${rangeKey === r.key ? styles.active : ''}`}
              onClick={() => setRangeKey(r.key)}
            >
              {r.label}
            </span>
          ))}
          <button className={styles.btnSecondary} onClick={() => navigate('/journal')}>
            &#8592; Calendar
          </button>
        </div>
      </div>

      {err && <div className={styles.error}>{err}</div>}

      {/* Headline metrics */}
      <div className={styles.metrics}>
        <div className={styles.metric}>
          <div className={styles.metricLabel}>Net P&amp;L</div>
          <div className={`${styles.metricValue} ${data.metrics.pnl_net >= 0 ? styles.pos : styles.neg}`}>
            {formatPnl(data.metrics.pnl_net)}
          </div>
          <div className={styles.metricSub}>{data.metrics.trades} trades</div>
        </div>
        <div className={styles.metric}>
          <div className={styles.metricLabel}>Win rate</div>
          <div className={styles.metricValue}>{data.metrics.win_rate.toFixed(1)}%</div>
          <div className={styles.metricSub}>across all systems</div>
        </div>
        <div className={styles.metric}>
          <div className={styles.metricLabel}>Profit factor</div>
          <div className={styles.metricValue}>
            {data.metrics.profit_factor != null ? data.metrics.profit_factor.toFixed(2) : '—'}
          </div>
          <div className={styles.metricSub}>gross wins / gross losses</div>
        </div>
        <div className={styles.metric}>
          <div className={styles.metricLabel}>Expectancy R</div>
          <div className={styles.metricValue}>
            {data.metrics.expectancy_r != null ? data.metrics.expectancy_r.toFixed(2) + 'R' : '—'}
          </div>
          <div className={styles.metricSub}>avg R-multiple</div>
        </div>
        <div className={styles.metric}>
          <div className={styles.metricLabel}>Max drawdown</div>
          <div className={`${styles.metricValue} ${styles.neg}`}>
            {formatPnl(-Math.abs(data.metrics.max_drawdown))}
          </div>
          <div className={styles.metricSub}>peak-to-trough</div>
        </div>
      </div>

      {/* Equity curve */}
      <section className={styles.panel}>
        <div className={styles.panelHead}>
          <div className={styles.panelTitle}>
            Equity curve
            <span className={styles.panelTitleAccent}>· cumulative net P&amp;L</span>
          </div>
        </div>
        <div className={styles.sectionBody}>
          <EquityCurveSvg points={data.equity_curve} />
        </div>
      </section>

      <div className={styles.insightsGrid}>
        {/* Per-strategy attribution */}
        <section className={styles.panel}>
          <div className={styles.panelHead}>
            <div className={styles.panelTitle}>Per-strategy attribution</div>
          </div>
          <div>
            {data.per_strategy.length === 0 ? (
              <div className={styles.empty}>No strategies in range.</div>
            ) : (
              <table className={styles.attrTable}>
                <thead>
                  <tr>
                    <th>Strategy</th>
                    <th className="right">Trades</th>
                    <th className="right">WR%</th>
                    <th className="right">PF</th>
                    <th className="right">Net P&amp;L</th>
                  </tr>
                </thead>
                <tbody>
                  {data.per_strategy.map((s) => (
                    <tr key={s.strategy}>
                      <td>{s.strategy}</td>
                      <td className="right">{s.trades}</td>
                      <td className="right">{s.win_rate.toFixed(1)}</td>
                      <td className="right">{s.profit_factor != null ? s.profit_factor.toFixed(2) : '—'}</td>
                      <td className={`right ${s.pnl_net >= 0 ? styles.pos : styles.neg}`}>
                        {formatPnl(s.pnl_net)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </section>

        {/* Drawdown windows */}
        <section className={styles.panel}>
          <div className={styles.panelHead}>
            <div className={styles.panelTitle}>Drawdown windows</div>
          </div>
          <div>
            {data.drawdowns.length === 0 ? (
              <div className={styles.empty}>No drawdown windows.</div>
            ) : (
              <table className={styles.attrTable}>
                <thead>
                  <tr>
                    <th>Start</th>
                    <th>Trough</th>
                    <th>Recovery</th>
                    <th className="right">Depth</th>
                    <th className="right">Days</th>
                  </tr>
                </thead>
                <tbody>
                  {data.drawdowns.map((d, i) => (
                    <tr key={i}>
                      <td>{d.start_date}</td>
                      <td>{d.trough_date}</td>
                      <td>{d.recovery_date || '—'}</td>
                      <td className={`right ${styles.neg}`}>{formatPnl(d.depth)}</td>
                      <td className="right">{d.duration_days ?? '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </section>

        {/* Win-rate by tag */}
        <section className={styles.panel}>
          <div className={styles.panelHead}>
            <div className={styles.panelTitle}>Win-rate by tag</div>
          </div>
          <div>
            {data.win_rate_by_tag.length === 0 ? (
              <div className={styles.empty}>No tags applied yet. Tag trades on the trade detail page.</div>
            ) : (
              <table className={styles.attrTable}>
                <thead>
                  <tr>
                    <th>Tag</th>
                    <th>Category</th>
                    <th className="right">Trades</th>
                    <th className="right">WR%</th>
                    <th className="right">Net P&amp;L</th>
                  </tr>
                </thead>
                <tbody>
                  {data.win_rate_by_tag.map((t) => (
                    <tr key={t.name}>
                      <td>{t.name}</td>
                      <td>{t.category}</td>
                      <td className="right">{t.trades}</td>
                      <td className="right">{t.win_rate.toFixed(1)}</td>
                      <td className={`right ${t.pnl_net >= 0 ? styles.pos : styles.neg}`}>
                        {formatPnl(t.pnl_net)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </section>

        {/* R-distribution */}
        <section className={styles.panel}>
          <div className={styles.panelHead}>
            <div className={styles.panelTitle}>R-distribution</div>
          </div>
          <div className={styles.sectionBody}>
            <RDistributionChart points={data.r_distribution.map((x) => x.r)} />
          </div>
        </section>
      </div>
    </div>
  );
}

/* ---------- inline SVG charts ---------- */

function EquityCurveSvg({ points }: { points: { date: string; cum_net: number }[] }) {
  const W = 880;
  const H = 240;
  const PAD = 32;
  if (points.length === 0) {
    return <div className={styles.empty}>No equity data.</div>;
  }
  const ys = points.map((p) => p.cum_net);
  const minY = Math.min(0, ...ys);
  const maxY = Math.max(0, ...ys);
  const span = maxY - minY || 1;
  const stepX = (W - PAD * 2) / Math.max(1, points.length - 1);
  const scaleY = (v: number) => H - PAD - ((v - minY) / span) * (H - PAD * 2);

  const pathD = points
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${PAD + i * stepX} ${scaleY(p.cum_net)}`)
    .join(' ');

  const zeroY = scaleY(0);
  const lastPoint = points[points.length - 1];
  const lastClass = lastPoint.cum_net >= 0 ? styles.pos : styles.neg;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className={styles.equityChart} preserveAspectRatio="none">
      {/* Zero line */}
      <line x1={PAD} x2={W - PAD} y1={zeroY} y2={zeroY} stroke="rgba(0,0,0,0.08)" strokeDasharray="2 3" />
      {/* Curve */}
      <path
        d={pathD}
        fill="none"
        stroke={lastPoint.cum_net >= 0 ? 'var(--accent-pos)' : 'var(--accent-neg)'}
        strokeWidth={1.6}
      />
      {/* End marker */}
      <circle
        cx={PAD + (points.length - 1) * stepX}
        cy={scaleY(lastPoint.cum_net)}
        r={3.2}
        fill={lastPoint.cum_net >= 0 ? 'var(--accent-pos)' : 'var(--accent-neg)'}
      />
      {/* Y-axis labels */}
      <text x={2} y={PAD + 4} fontSize={10} fill="var(--ink-muted)">
        {Math.round(maxY)}
      </text>
      <text x={2} y={H - PAD + 4} fontSize={10} fill="var(--ink-muted)">
        {Math.round(minY)}
      </text>
      <text x={W - PAD - 50} y={PAD + 4} fontSize={11} fill="var(--ink-muted)" className={lastClass}>
        {points[points.length - 1].cum_net >= 0 ? '+' : ''}
        {Math.round(points[points.length - 1].cum_net)}
      </text>
    </svg>
  );
}

function RDistributionChart({ points }: { points: number[] }) {
  if (points.length === 0) {
    return (
      <div className={styles.empty}>
        No R-multiples recorded yet. R-multiples are populated for ORB trades
        with an SL price.
      </div>
    );
  }
  // Bin into [-3, -2, -1, 0, 1, 2, 3+] (7 bins)
  const bins = [0, 0, 0, 0, 0, 0, 0];
  const labels = ['-3R', '-2R', '-1R', '0', '+1R', '+2R', '+3R'];
  points.forEach((r) => {
    const idx = Math.max(0, Math.min(6, Math.round(r) + 3));
    bins[idx]++;
  });
  const max = Math.max(...bins, 1);
  const W = 600;
  const H = 220;
  const PAD = 28;
  const barW = (W - PAD * 2) / bins.length - 6;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 220 }}>
      {bins.map((count, i) => {
        const x = PAD + i * (barW + 6);
        const h = (count / max) * (H - PAD * 2);
        const y = H - PAD - h;
        const isNeg = i < 3;
        const isPos = i > 3;
        const fill = isNeg ? 'var(--accent-neg)' : isPos ? 'var(--accent-pos)' : 'var(--ink-muted)';
        return (
          <g key={i}>
            <rect x={x} y={y} width={barW} height={h} fill={fill} rx={2} />
            <text
              x={x + barW / 2}
              y={H - PAD + 14}
              fontSize={11}
              textAnchor="middle"
              fill="var(--ink-muted)"
            >
              {labels[i]}
            </text>
            <text
              x={x + barW / 2}
              y={y - 4}
              fontSize={11}
              textAnchor="middle"
              fill="var(--ink-secondary)"
            >
              {count > 0 ? count : ''}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/* eslint-disable-next-line @typescript-eslint/no-unused-vars */
const _useMemo = useMemo;
