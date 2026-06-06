import { useMemo, useState } from 'react';
import styles from './FdScenarios.module.css';

/**
 * FD Scenarios — "what happens to ₹1 Cr in a fixed deposit?"
 *
 * Compares three ways of handling the annual interest on a fixed deposit:
 *
 *   1. Spend all interest   — withdraw the full interest each year, corpus
 *                             stays flat, you live off a constant income.
 *   2. Reinvest half        — withdraw half the interest, let the rest
 *                             compound; corpus grows at (1 + r/2) per year.
 *   3. Leave untouched       — reinvest everything; full compounding P(1+r)^n.
 *
 * For an apples-to-apples comparison we track quantities per year:
 *   - FD corpus      = money still in the deposit
 *   - cash withdrawn = cumulative interest you took out and (presumably) spent
 *   - total wealth   = corpus + cash withdrawn  (net worth attributable to the FD)
 *   - return         = total wealth as a % gain over the original capital
 *
 * Pure client-side maths + hand-rolled SVG charts — no backend, no chart lib,
 * matching the BookPnLChart pattern so the bundle stays small.
 */

type MetricKey = 'total' | 'corpus' | 'cash' | 'ret' | 'effRate';
type ChartKind = 'bars' | 'lines';
type RetView = 'cumulative' | 'rate';
type ScenarioKey = 'spend' | 'half' | 'compound';

interface Scenario {
  key: ScenarioKey;
  name: string;
  short: string;
  color: string;
  desc: string;
}

const SCENARIOS: Scenario[] = [
  { key: 'spend', name: 'Spend all interest', short: 'Spend all', color: '#B45309',
    desc: 'Withdraw the full interest every year — corpus stays flat.' },
  { key: 'half', name: 'Reinvest half', short: 'Half & half', color: '#0F6E56',
    desc: 'Withdraw half the interest, let the other half compound.' },
  { key: 'compound', name: 'Leave untouched', short: 'Compound', color: '#1E3A8A',
    desc: 'Reinvest everything — full compounding.' },
];

const METRICS: { key: MetricKey; label: string }[] = [
  { key: 'total', label: 'Total wealth' },
  { key: 'corpus', label: 'FD corpus' },
  { key: 'cash', label: 'Cash withdrawn' },
];

const YEAR_OPTIONS = [5, 10, 15, 20, 25, 30];

interface YearRow {
  n: number;
  corpus: Record<ScenarioKey, number>;
  cash: Record<ScenarioKey, number>;
  total: Record<ScenarioKey, number>;
  /** cumulative gain as a % of the original capital (e.g. 65 = +65%) */
  ret: Record<ScenarioKey, number>;
  /** this year's gross interest as a % of the ORIGINAL capital (climbs as the
   *  corpus grows: 6.5% on a bigger base is worth >6.5% of what you put in) */
  effRate: Record<ScenarioKey, number>;
  /** interest pocketed during this specific year (0 for compound) */
  payout: Record<ScenarioKey, number>;
}

function buildSeries(P: number, ratePct: number, years: number): YearRow[] {
  const r = ratePct / 100;
  const rows: YearRow[] = [];
  for (let n = 0; n <= years; n++) {
    // Case 1 — spend all interest: flat corpus, linear cash.
    const corpusSpend = P;
    const cashSpend = n * P * r;
    const totalSpend = corpusSpend + cashSpend;

    // Case 2 — reinvest half: corpus grows at (1 + r/2), cash = growth so far.
    const corpusHalf = P * Math.pow(1 + r / 2, n);
    const cashHalf = corpusHalf - P;
    const totalHalf = corpusHalf + cashHalf;

    // Case 3 — leave untouched: full compounding, nothing withdrawn.
    const corpusComp = P * Math.pow(1 + r, n);
    const totalComp = corpusComp;

    const pct = (total: number) => (P > 0 ? ((total - P) / P) * 100 : 0);
    const prev = rows[n - 1];
    // This year's gross interest = rate x (corpus at start of year), expressed
    // as a % of the ORIGINAL capital. Year 0 anchors at the nominal rate.
    const baseRate = r * 100;
    const eff = (prevCorpus: number) =>
      n === 0 || P <= 0 ? baseRate : (r * prevCorpus / P) * 100;
    rows.push({
      n,
      corpus: { spend: corpusSpend, half: corpusHalf, compound: corpusComp },
      cash: { spend: cashSpend, half: cashHalf, compound: 0 },
      total: { spend: totalSpend, half: totalHalf, compound: totalComp },
      ret: { spend: pct(totalSpend), half: pct(totalHalf), compound: pct(totalComp) },
      effRate: {
        spend: baseRate,
        half: eff(prev ? prev.corpus.half : P),
        compound: eff(prev ? prev.corpus.compound : P),
      },
      payout: {
        spend: P * r,
        half: prev ? corpusHalf - prev.corpus.half : 0,
        compound: 0,
      },
    });
  }
  // Year 0 has no payout for any scenario.
  if (rows[0]) rows[0].payout = { spend: 0, half: 0, compound: 0 };
  return rows;
}

function fmtMoney(v: number): string {
  if (v >= 1e7) return `₹${(v / 1e7).toFixed(2)} Cr`;
  if (v >= 1e5) return `₹${(v / 1e5).toFixed(2)} L`;
  return `₹${Math.round(v).toLocaleString('en-IN')}`;
}

function fmtAxis(v: number): string {
  if (v >= 1e7) return `${(v / 1e7).toFixed(1)} Cr`;
  if (v >= 1e5) return `${(v / 1e5).toFixed(0)} L`;
  return `${Math.round(v / 1e3)}k`;
}

function fmtFull(v: number): string {
  return `₹${Math.round(v).toLocaleString('en-IN')}`;
}

function cagr(value: number, P: number, years: number): number {
  if (P <= 0 || years <= 0) return 0;
  return (Math.pow(value / P, 1 / years) - 1) * 100;
}

// ---- SVG chart geometry -------------------------------------------------
const W = 960;
const H = 360;
const PAD_L = 64;
const PAD_R = 16;
const PAD_T = 16;
const PAD_B = 36;
const PLOT_W = W - PAD_L - PAD_R;
const PLOT_H = H - PAD_T - PAD_B;

function Chart({
  rows,
  metric,
  kind,
  pct = false,
}: {
  rows: YearRow[];
  metric: MetricKey;
  kind: ChartKind;
  pct?: boolean;
}) {
  const years = rows.length - 1;
  const fmtAxisFn = pct ? (v: number) => `${Math.round(v)}%` : fmtAxis;
  const fmtValFn = pct ? (v: number) => `${v.toFixed(1)}%` : fmtMoney;

  const { yMax, ticks } = useMemo(() => {
    let max = 0;
    for (const row of rows) {
      for (const s of SCENARIOS) max = Math.max(max, row[metric][s.key]);
    }
    if (max <= 0) max = 1;
    const padded = max * 1.08;
    const t: number[] = [];
    for (let k = 0; k <= 4; k++) t.push((padded * k) / 4);
    return { yMax: padded, ticks: t };
  }, [rows, metric]);

  const yScale = (v: number) => PAD_T + PLOT_H - (v / yMax) * PLOT_H;

  // Show every other year-label when the axis gets crowded.
  const labelStep = rows.length > 16 ? 2 : 1;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className={styles.svg} preserveAspectRatio="xMidYMid meet">
      {/* Y grid + labels */}
      {ticks.map((v, i) => {
        const y = yScale(v);
        return (
          <g key={i}>
            <line x1={PAD_L} x2={W - PAD_R} y1={y} y2={y} className={styles.grid} />
            <text x={PAD_L - 10} y={y + 3} textAnchor="end" className={styles.yLabel}>
              {fmtAxisFn(v)}
            </text>
          </g>
        );
      })}

      {kind === 'bars' ? (
        // ---- Grouped bars: one group per year, 3 bars per group ----
        rows.map((row, i) => {
          const groupW = PLOT_W / rows.length;
          const barAreaW = groupW * 0.72;
          const barW = barAreaW / SCENARIOS.length;
          const groupStart = PAD_L + i * groupW + (groupW - barAreaW) / 2;
          return (
            <g key={i}>
              {SCENARIOS.map((s, j) => {
                const v = row[metric][s.key];
                const x = groupStart + j * barW;
                const y = yScale(v);
                const h = PAD_T + PLOT_H - y;
                return (
                  <rect
                    key={s.key}
                    x={x}
                    y={y}
                    width={Math.max(barW - 1, 1)}
                    height={Math.max(h, 0)}
                    fill={s.color}
                    rx={1.5}
                  >
                    <title>{`${s.short} · Year ${row.n}: ${fmtValFn(v)}`}</title>
                  </rect>
                );
              })}
            </g>
          );
        })
      ) : (
        // ---- Lines: one polyline per scenario over the years ----
        SCENARIOS.map((s) => {
          const xLine = (n: number) =>
            PAD_L + (years === 0 ? 0 : (n / years) * PLOT_W);
          const pts = rows.map((r) => `${xLine(r.n).toFixed(1)},${yScale(r[metric][s.key]).toFixed(1)}`);
          return (
            <g key={s.key}>
              <polyline points={pts.join(' ')} fill="none" stroke={s.color} strokeWidth={2.4}
                strokeLinejoin="round" strokeLinecap="round" />
              {rows.map((r) => (
                <circle key={r.n} cx={xLine(r.n)} cy={yScale(r[metric][s.key])} r={2.6} fill={s.color}>
                  <title>{`${s.short} · Year ${r.n}: ${fmtValFn(r[metric][s.key])}`}</title>
                </circle>
              ))}
            </g>
          );
        })
      )}

      {/* Baseline */}
      <line x1={PAD_L} x2={W - PAD_R} y1={yScale(0)} y2={yScale(0)} className={styles.axis} />

      {/* X labels (year numbers) */}
      {rows.map((row, i) => {
        if (i % labelStep !== 0 && i !== rows.length - 1) return null;
        const groupW = PLOT_W / rows.length;
        const x = kind === 'bars'
          ? PAD_L + i * groupW + groupW / 2
          : PAD_L + (years === 0 ? 0 : (row.n / years) * PLOT_W);
        return (
          <text key={i} x={x} y={H - 12} textAnchor="middle" className={styles.xLabel}>
            {row.n}
          </text>
        );
      })}
    </svg>
  );
}

export default function FdScenarios() {
  const [principal, setPrincipal] = useState(10_000_000);
  const [rate, setRate] = useState(6.5);
  const [years, setYears] = useState(10);
  const [metric, setMetric] = useState<MetricKey>('total');
  const [kind, setKind] = useState<ChartKind>('bars');
  const [retView, setRetView] = useState<RetView>('cumulative');

  const rows = useMemo(
    () => buildSeries(principal, rate, years),
    [principal, rate, years],
  );
  const last = rows[rows.length - 1];

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <div className={styles.title}>FD interest — three ways to handle it</div>
        <div className={styles.subtitle}>
          A fixed deposit of {fmtMoney(principal)} at {rate}% p.a. — compare spending the
          interest, reinvesting half, or leaving it to compound over {years} years.
        </div>
      </div>

      {/* ---- Controls ---- */}
      <div className={styles.controls}>
        <label className={styles.control}>
          <span className={styles.controlLabel}>Principal (₹)</span>
          <input
            type="number"
            className={styles.input}
            value={principal}
            min={0}
            step={100000}
            onChange={(e) => setPrincipal(Math.max(0, Number(e.target.value) || 0))}
          />
          <span className={styles.controlHint}>= {fmtMoney(principal)}</span>
        </label>

        <label className={styles.control}>
          <span className={styles.controlLabel}>Interest rate (% p.a.)</span>
          <input
            type="number"
            className={styles.input}
            value={rate}
            min={0}
            step={0.1}
            onChange={(e) => setRate(Math.max(0, Number(e.target.value) || 0))}
          />
          <span className={styles.controlHint}>annual, simple per year</span>
        </label>

        <div className={styles.control}>
          <span className={styles.controlLabel}>Horizon</span>
          <div className={styles.segmented}>
            {YEAR_OPTIONS.map((y) => (
              <button
                key={y}
                className={`${styles.seg} ${years === y ? styles.segActive : ''}`}
                onClick={() => setYears(y)}
              >
                {y}y
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* ---- Scenario summary cards ---- */}
      <div className={styles.cardsRow}>
        {SCENARIOS.map((s) => (
          <div key={s.key} className={styles.scenarioCard}>
            <div className={styles.scenarioTop}>
              <span className={styles.swatch} style={{ background: s.color }} />
              <span className={styles.scenarioName}>{s.name}</span>
            </div>
            <div className={styles.scenarioDesc}>{s.desc}</div>
            <div className={styles.scenarioBig} style={{ color: s.color }}>
              {fmtMoney(last.total[s.key])}
            </div>
            <div className={styles.scenarioBigLabel}>total wealth after {years}y</div>
            <div className={styles.scenarioBreak}>
              <div className={styles.breakRow}>
                <span>FD corpus</span>
                <span className={styles.breakVal}>{fmtMoney(last.corpus[s.key])}</span>
              </div>
              <div className={styles.breakRow}>
                <span>Cash taken out</span>
                <span className={styles.breakVal}>{fmtMoney(last.cash[s.key])}</span>
              </div>
              <div className={styles.breakRow}>
                <span>Return on capital</span>
                <span className={styles.breakVal} style={{ color: s.color }}>
                  +{last.ret[s.key].toFixed(1)}%
                </span>
              </div>
              <div className={styles.breakRow}>
                <span>CAGR (effective)</span>
                <span className={styles.breakVal}>
                  {cagr(last.total[s.key], principal, years).toFixed(2)}%
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* ---- Chart: rupee metrics ---- */}
      <div className={styles.chartCard}>
        <div className={styles.chartHead}>
          <div className={styles.segmented}>
            {METRICS.map((m) => (
              <button
                key={m.key}
                className={`${styles.seg} ${metric === m.key ? styles.segActive : ''}`}
                onClick={() => setMetric(m.key)}
              >
                {m.label}
              </button>
            ))}
          </div>
          <div className={styles.segmented}>
            <button
              className={`${styles.seg} ${kind === 'bars' ? styles.segActive : ''}`}
              onClick={() => setKind('bars')}
            >
              Bars
            </button>
            <button
              className={`${styles.seg} ${kind === 'lines' ? styles.segActive : ''}`}
              onClick={() => setKind('lines')}
            >
              Lines
            </button>
          </div>
        </div>

        <div className={styles.legend}>
          {SCENARIOS.map((s) => (
            <span key={s.key} className={styles.legendItem}>
              <span className={styles.swatch} style={{ background: s.color }} />
              {s.short}
            </span>
          ))}
        </div>

        <Chart rows={rows} metric={metric} kind={kind} />
        <div className={styles.chartFoot}>
          {METRICS.find((m) => m.key === metric)?.label} per year ·
          x-axis = year, y-axis = ₹
        </div>
      </div>

      {/* ---- Chart: return / effective-rate over original capital (%) ---- */}
      <div className={styles.chartCard}>
        <div className={styles.chartHead}>
          <div className={styles.chartTitle}>
            {retView === 'cumulative'
              ? `Interest growth alone — cumulative return on the original ${fmtMoney(principal)}`
              : `Effective interest rate each year — measured against the original ${fmtMoney(principal)}`}
          </div>
          <div className={styles.segmented}>
            <button
              className={`${styles.seg} ${retView === 'cumulative' ? styles.segActive : ''}`}
              onClick={() => setRetView('cumulative')}
            >
              Cumulative return
            </button>
            <button
              className={`${styles.seg} ${retView === 'rate' ? styles.segActive : ''}`}
              onClick={() => setRetView('rate')}
            >
              Annual rate
            </button>
          </div>
        </div>
        <div className={styles.legend}>
          {SCENARIOS.map((s) => (
            <span key={s.key} className={styles.legendItem}>
              <span className={styles.swatch} style={{ background: s.color }} />
              {s.short} ·{' '}
              {retView === 'cumulative'
                ? `+${last.ret[s.key].toFixed(1)}%`
                : `${last.effRate[s.key].toFixed(2)}%`}
            </span>
          ))}
        </div>
        <Chart rows={rows} metric={retView === 'cumulative' ? 'ret' : 'effRate'} kind="lines" pct />
        <div className={styles.chartFoot}>
          {retView === 'cumulative'
            ? 'Gain over the starting capital, ignoring the principal · x-axis = year, y-axis = % return'
            : 'Each year’s interest as a % of the original capital — 6.5% on a grown corpus is worth more than 6.5% of what you first put in · x-axis = year, y-axis = % rate'}
        </div>
      </div>

      {/* ---- Year-by-year table ---- */}
      <div className={styles.tableCard}>
        <table className={styles.table}>
          <thead>
            <tr className={styles.groupHead}>
              <th />
              <th colSpan={2} style={{ color: SCENARIOS[0].color }}>Spend all interest</th>
              <th colSpan={2} style={{ color: SCENARIOS[1].color }}>Reinvest half</th>
              <th colSpan={2} style={{ color: SCENARIOS[2].color }}>Compound</th>
            </tr>
            <tr className={styles.colHead}>
              <th>Year</th>
              <th>This yr income</th>
              <th>Total wealth</th>
              <th>This yr income</th>
              <th>Total wealth</th>
              <th>Total wealth</th>
              <th>Return %</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.n} className={row.n === years ? styles.finalRow : ''}>
                <td className={styles.yearCell}>{row.n}</td>
                <td>{fmtFull(row.payout.spend)}</td>
                <td className={styles.strong}>{fmtFull(row.total.spend)}</td>
                <td>{fmtFull(row.payout.half)}</td>
                <td className={styles.strong}>{fmtFull(row.total.half)}</td>
                <td className={styles.strong}>{fmtFull(row.total.compound)}</td>
                <td>+{row.ret.compound.toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
