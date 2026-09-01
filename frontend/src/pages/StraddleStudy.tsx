import { useCallback, useEffect, useMemo, useState } from 'react';
import type { CSSProperties } from 'react';
import styles from './StraddleStudy.module.css';
import { apiGet } from '../api/client';

/* ------------------------------------------------------------------ types */

interface RunRow {
  run_id: string;
  index_name: string;
  sl_pct: number;
  entry_time: string;
  exit_time: string;
  qty: number;
  lot_size: number;
  lots: number;
  n_trades: number;
  period_start: string;
  period_end: string;
}

interface RunsResp {
  ok: boolean;
  runs: RunRow[];
  indices: string[];
  years: number[];
  dtes: number[];
  defaults: { cost_rate: number; cost_flat: number; wr_min: number; streak_max: number };
}

interface MetricRow {
  label: string;
  n: number;
  net: number;
  mean: number;
  median: number;
  win_pct: number;
  avg_win: number;
  avg_loss: number;
  rr: number | null;
  pf: number | null;
  maxdd: number;
  net_dd: number | null;
  calmar: number | null;
  t: number;
  worst: number;
  best: number;
  lose_streak: number;
  win_streak: number;
  years_positive: number;
  years_total: number;
  per_year: Record<string, number>;
  per_month: Record<string, number>;
  verdict: string;
  gate_stats: boolean;
  gate_tradeable: boolean;
}

interface SeriesPoint {
  date: string;
  pnl: number;
  cum: number;
  dd: number;
}

interface SeriesResp {
  ok: boolean;
  label: string;
  n: number;
  points: SeriesPoint[];
}

interface QueryResp {
  ok: boolean;
  rows: MetricRow[];
  n_groups: number;
}

/* ------------------------------------------------------------ formatting */

const inr = (v: number) =>
  v.toLocaleString('en-IN', { maximumFractionDigits: 0 });

const SORTS: [string, string][] = [
  ['net', 'Net profit'],
  ['per_trade', 'Per-trade mean'],
  ['median', 'Per-trade median'],
  ['win', 'Win rate'],
  ['net_dd', 'Net / MaxDD'],
  ['calmar', 'Calmar (yearly / MaxDD)'],
  ['pf', 'Profit factor'],
  ['t', 't-stat'],
  ['worst', 'Worst trade (best first)'],
  ['maxdd', 'MaxDD (shallowest first)'],
  ['lose_streak', 'Losing streak (shortest first)'],
];

const GROUPS: [string, string][] = [
  ['run', 'Run (index + SL)'],
  ['run_dte', 'Run x DTE'],
  ['dte', 'DTE only'],
  ['year', 'Year'],
  ['weekday', 'Entry weekday'],
];

/* Column definitions for the results table; key=null means not sortable.
   asc=true marks columns where "smaller first" is the natural first click
   (losing streak). Everything else starts descending. */
interface Col {
  key: keyof MetricRow | null;
  label: string;
  left?: boolean;
  asc?: boolean;
}

const COLS: Col[] = [
  { key: null, label: '#' },
  { key: 'label', label: 'System', left: true, asc: true },
  { key: 'n', label: 'n' },
  { key: 'net', label: 'Net Rs.' },
  { key: 'mean', label: 'Mean' },
  { key: 'median', label: 'Median' },
  { key: 'win_pct', label: 'Win%' },
  { key: 'avg_win', label: 'Avg win' },
  { key: 'avg_loss', label: 'Avg loss' },
  { key: 'maxdd', label: 'MaxDD' },
  { key: 'worst', label: 'Worst' },
  { key: 'lose_streak', label: 'Streak', asc: true },
  { key: 'net_dd', label: 'Net/DD' },
  { key: 'calmar', label: 'Calmar' },
  { key: 'pf', label: 'PF' },
  { key: 't', label: 't' },
  { key: 'years_positive', label: 'Yrs+' },
  { key: 'verdict', label: 'Verdict', asc: true },
];

/* Human-readable explanation for the verdict chip tooltip. */
function verdictReasons(m: MetricRow): string {
  const bad: string[] = [];
  if (m.t < 2.0) bad.push(`t ${m.t.toFixed(2)} < 2.0`);
  if ((m.pf ?? 0) < 1.3) bad.push(`PF ${m.pf ?? '-'} < 1.3`);
  if (m.years_total > 0 && m.years_positive / m.years_total < 0.8)
    bad.push(`only ${m.years_positive}/${m.years_total} years positive (< 80%)`);
  if (m.win_pct < 45) bad.push(`WR ${m.win_pct.toFixed(1)}% < 45%`);
  if (m.lose_streak > 7) bad.push(`losing streak ${m.lose_streak} > 7`);
  return bad.length ? `Fails: ${bad.join('; ')}` : 'Passes both gates (stats + tradeability)';
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

/* Month x year P&L heatmap. */
function Heatmap({ perMonth, perYear }: {
  perMonth: Record<string, number>;
  perYear: Record<string, number>;
}) {
  const years = Array.from(new Set(Object.keys(perMonth).map((k) => k.slice(0, 4)))).sort();
  const vals = Object.values(perMonth);
  const maxAbs = Math.max(1, ...vals.map((v) => Math.abs(v)));
  const cellStyle = (v: number | undefined): CSSProperties => {
    if (v === undefined) return {};
    const a = 0.08 + 0.62 * (Math.abs(v) / maxAbs);
    return {
      background: v >= 0 ? `rgba(15, 110, 86, ${a})` : `rgba(163, 45, 45, ${a})`,
      color: Math.abs(v) / maxAbs > 0.55 ? '#fff' : undefined,
    };
  };
  const fmtK = (v: number) =>
    Math.abs(v) >= 100000 ? `${(v / 100000).toFixed(1)}L` : `${Math.round(v / 1000)}k`;
  return (
    <table className={styles.heat}>
      <thead>
        <tr>
          <th></th>
          {MONTHS.map((mo) => <th key={mo}>{mo}</th>)}
          <th>Year</th>
        </tr>
      </thead>
      <tbody>
        {years.map((y) => (
          <tr key={y}>
            <td className={styles.heatYear}>{y}</td>
            {MONTHS.map((_, i) => {
              const k = `${y}-${String(i + 1).padStart(2, '0')}`;
              const v = perMonth[k];
              return (
                <td key={k} style={cellStyle(v)} title={v !== undefined ? `${k}: ${inr(v)}` : ''}>
                  {v !== undefined ? fmtK(v) : ''}
                </td>
              );
            })}
            <td className={(perYear[y] ?? 0) >= 0 ? styles.pos : styles.neg}>
              <b>{perYear[y] !== undefined ? fmtK(perYear[y]) : ''}</b>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/* Cumulative P&L + drawdown, inline SVG (no chart lib). */
function EquityChart({ points }: { points: SeriesPoint[] }) {
  const W = 1000, EQ_TOP = 8, EQ_BOT = 185, DD_TOP = 200, DD_BOT = 288;
  const n = points.length;
  if (n < 2) return null;
  const cums = points.map((p) => p.cum);
  const cMin = Math.min(0, ...cums);
  const cMax = Math.max(0, ...cums);
  const dMin = Math.min(...points.map((p) => p.dd), -1);
  const x = (i: number) => 12 + (i / (n - 1)) * (W - 24);
  const yEq = (v: number) => EQ_BOT - ((v - cMin) / (cMax - cMin || 1)) * (EQ_BOT - EQ_TOP);
  const yDd = (v: number) => DD_TOP + (v / dMin) * (DD_BOT - DD_TOP);
  const eqPath = points.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${yEq(p.cum).toFixed(1)}`).join('');
  const ddArea =
    `M${x(0).toFixed(1)},${DD_TOP}` +
    points.map((p, i) => `L${x(i).toFixed(1)},${yDd(p.dd).toFixed(1)}`).join('') +
    `L${x(n - 1).toFixed(1)},${DD_TOP}Z`;
  // year boundaries
  const marks: { i: number; y: string }[] = [];
  for (let i = 1; i < n; i++) {
    if (points[i].date.slice(0, 4) !== points[i - 1].date.slice(0, 4))
      marks.push({ i, y: points[i].date.slice(0, 4) });
  }
  const last = points[n - 1];
  const worstDd = Math.min(...points.map((p) => p.dd));
  return (
    <svg viewBox={`0 0 ${W} 300`} className={styles.chart} preserveAspectRatio="none">
      {marks.map((mk) => (
        <g key={mk.y}>
          <line x1={x(mk.i)} x2={x(mk.i)} y1={EQ_TOP} y2={DD_BOT} className={styles.chartGrid} />
          <text x={x(mk.i) + 4} y={EQ_TOP + 12} className={styles.chartYear}>{mk.y}</text>
        </g>
      ))}
      {cMin < 0 && (
        <line x1={12} x2={W - 12} y1={yEq(0)} y2={yEq(0)} className={styles.chartZero} />
      )}
      <path d={eqPath} className={styles.chartEq} />
      <text x={W - 14} y={yEq(last.cum) - 6} className={styles.chartLblPos} textAnchor="end">
        {inr(last.cum)}
      </text>
      <line x1={12} x2={W - 12} y1={DD_TOP} y2={DD_TOP} className={styles.chartZero} />
      <path d={ddArea} className={styles.chartDd} />
      <text x={14} y={DD_BOT - 4} className={styles.chartLblNeg}>
        MaxDD {inr(worstDd)}
      </text>
    </svg>
  );
}

/* ------------------------------------------------------------------ page */

export default function StraddleStudy() {
  const [meta, setMeta] = useState<RunsResp | null>(null);
  const [rows, setRows] = useState<MetricRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState('');

  // filters
  const [indices, setIndices] = useState<string[]>(['NIFTY']);
  const [sls, setSls] = useState<number[]>([]);
  const [dtes, setDtes] = useState<number[]>([]);
  const [yearFrom, setYearFrom] = useState<number | ''>('');
  const [yearTo, setYearTo] = useState<number | ''>('');
  const [groupBy, setGroupBy] = useState('run');
  const [sortBy, setSortBy] = useState('net_dd');
  const [exclEvents, setExclEvents] = useState(true);
  const [costRate, setCostRate] = useState('0.59');
  const [lotsScale, setLotsScale] = useState('1');
  const [expanded, setExpanded] = useState<string | null>(null);
  // client-side column sort; null = keep the server "Rank by" order
  const [colSort, setColSort] = useState<{ key: keyof MetricRow; dir: 1 | -1 } | null>(null);

  useEffect(() => {
    apiGet<RunsResp>('/api/straddle-study/runs')
      .then(setMeta)
      .catch((e) => setErr(String(e)));
  }, []);

  // one place to build the filter querystring so /query and /series always agree
  const buildParams = useCallback(() => {
    const p = new URLSearchParams();
    if (indices.length) p.set('index', indices.join(','));
    if (sls.length) p.set('sl', sls.join(','));
    if (dtes.length) p.set('dte', dtes.join(','));
    if (yearFrom !== '') p.set('year_from', String(yearFrom));
    if (yearTo !== '') p.set('year_to', String(yearTo));
    p.set('group_by', groupBy);
    p.set('exclude_events', exclEvents ? '1' : '0');
    if (costRate) p.set('cost_rate', costRate);
    if (lotsScale && lotsScale !== '1') p.set('lots_scale', lotsScale);
    return p;
  }, [indices, sls, dtes, yearFrom, yearTo, groupBy, exclEvents, costRate, lotsScale]);

  const runQuery = useCallback(() => {
    setLoading(true);
    setErr('');
    const p = buildParams();
    p.set('sort', sortBy);
    apiGet<QueryResp>(`/api/straddle-study/query?${p.toString()}`)
      .then((r) => {
        setRows(r.rows);
        setColSort(null); // fresh data comes back in server rank order
        setExpanded(null);
        setSeries({});
      })
      .catch((e) => setErr(String(e)))
      .finally(() => setLoading(false));
  }, [buildParams, sortBy]);

  // equity/dd series per expanded bucket, fetched lazily
  const [series, setSeries] = useState<Record<string, SeriesPoint[] | 'loading'>>({});

  const expandRow = (label: string) => {
    const next = expanded === label ? null : label;
    setExpanded(next);
    if (next && !series[next]) {
      setSeries((s) => ({ ...s, [next]: 'loading' }));
      const p = buildParams();
      p.set('label', next);
      apiGet<SeriesResp>(`/api/straddle-study/series?${p.toString()}`)
        .then((r) => setSeries((s) => ({ ...s, [next]: r.points })))
        .catch(() => setSeries((s) => ({ ...s, [next]: [] })));
    }
  };

  useEffect(() => {
    if (meta) runQuery();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [meta, indices, sls, dtes, yearFrom, yearTo, groupBy, sortBy, exclEvents]);

  const toggle = <T,>(arr: T[], v: T, set: (x: T[]) => void) =>
    set(arr.includes(v) ? arr.filter((x) => x !== v) : [...arr, v]);

  const clickCol = (c: Col) => {
    if (!c.key) return;
    const first: 1 | -1 = c.asc ? 1 : -1;
    setColSort((cur) =>
      cur && cur.key === c.key
        ? { key: c.key as keyof MetricRow, dir: (cur.dir * -1) as 1 | -1 }
        : { key: c.key as keyof MetricRow, dir: first },
    );
  };

  const view = useMemo(() => {
    if (!colSort) return rows;
    const { key, dir } = colSort;
    return [...rows].sort((a, b) => {
      const av = a[key] as number | string | null;
      const bv = b[key] as number | string | null;
      if (av == null && bv == null) return 0;
      if (av == null) return 1; // nulls always last
      if (bv == null) return -1;
      if (typeof av === 'string' || typeof bv === 'string')
        return dir * String(av).localeCompare(String(bv));
      return dir * ((bv as number) - (av as number)) * -1;
    });
  }, [rows, colSort]);

  const slOptions = meta
    ? Array.from(
        new Set(
          meta.runs
            .filter((r) => !indices.length || indices.includes(r.index_name))
            .map((r) => r.sl_pct),
        ),
      ).sort((a, b) => a - b)
    : [];

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <h1>Straddle Intraday Study</h1>
        <p className={styles.sub}>
          AlgoTest archive - 09:16 ATM short straddle, exit 15:15, per-leg %SL, partial
          square-off, trail-to-BE. 10 lots (NIFTY qty 650 / SENSEX qty 200). All figures NET of
          0.59% turnover + Rs.80/trade unless you change the cost rate. Full doctrine and verdicts:
          research/136 STATUS doc.
        </p>
      </div>

      <div className={styles.finding}>
        <strong>Standing verdicts (2026-09-01):</strong> the stop level is NOT an optimisable
        parameter (14-pt NIFTY ladder flat, |t| &lt; 1.5 paired) - risk is monotonic in the stop,
        return is not. The book is <b>DTE-0 (expiry day)</b>: NIFTY t=4.28 &amp; OOS t=2.24 @60%,
        SENSEX confirms with t=3.61 @30%. DTE 1-6 rejected (IS-only edge or nothing). Tradeability
        gate binding: WR &ge; 45%, losing streak &le; 7 - median trade must be positive to hold.
        Budget + election days excluded by calendar rule.
      </div>

      {meta && (
        <div className={styles.controls}>
          <div className={styles.ctlGroup}>
            <span className={styles.ctlLabel}>Index</span>
            {meta.indices.map((ix) => (
              <button
                key={ix}
                className={`${styles.chip} ${
                  indices.length === 1 && indices[0] === ix ? styles.chipOn : ''
                }`}
                onClick={() => setIndices([ix])}
              >
                {ix}
              </button>
            ))}
            <button
              className={`${styles.chip} ${indices.length !== 1 ? styles.chipOn : ''}`}
              onClick={() => setIndices([])}
              title="Both indices - separate rows when grouping by run, POOLED onto one equity curve when grouping by DTE/year/weekday"
            >
              BOTH
            </button>
          </div>

          <div className={styles.ctlGroup}>
            <span className={styles.ctlLabel}>Stop %</span>
            {slOptions.map((sl) => (
              <button
                key={sl}
                className={`${styles.chip} ${sls.includes(sl) ? styles.chipOn : ''}`}
                onClick={() => toggle(sls, sl, setSls)}
              >
                {sl}
              </button>
            ))}
          </div>

          <div className={styles.ctlGroup}>
            <span className={styles.ctlLabel}>DTE</span>
            {meta.dtes.map((d) => (
              <button
                key={d}
                className={`${styles.chip} ${dtes.includes(d) ? styles.chipOn : ''}`}
                onClick={() => toggle(dtes, d, setDtes)}
              >
                {d}
              </button>
            ))}
          </div>

          <div className={styles.ctlGroup}>
            <span className={styles.ctlLabel}>Years</span>
            <select
              className={styles.select}
              value={yearFrom}
              onChange={(e) => setYearFrom(e.target.value ? Number(e.target.value) : '')}
            >
              <option value="">from</option>
              {meta.years.map((y) => (
                <option key={y} value={y}>{y}</option>
              ))}
            </select>
            <select
              className={styles.select}
              value={yearTo}
              onChange={(e) => setYearTo(e.target.value ? Number(e.target.value) : '')}
            >
              <option value="">to</option>
              {meta.years.map((y) => (
                <option key={y} value={y}>{y}</option>
              ))}
            </select>
          </div>

          <div className={styles.ctlGroup}>
            <span className={styles.ctlLabel}>Group</span>
            <select className={styles.select} value={groupBy} onChange={(e) => setGroupBy(e.target.value)}>
              {GROUPS.map(([v, l]) => (
                <option key={v} value={v}>{l}</option>
              ))}
            </select>
            <span className={styles.ctlLabel}>Rank by</span>
            <select className={styles.select} value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
              {SORTS.map(([v, l]) => (
                <option key={v} value={v}>{l}</option>
              ))}
            </select>
          </div>

          <div className={styles.ctlGroup}>
            <label className={styles.check}>
              <input
                type="checkbox"
                checked={exclEvents}
                onChange={(e) => setExclEvents(e.target.checked)}
              />
              exclude budget/election days
            </label>
            <span className={styles.ctlLabel}>Cost %</span>
            <input
              className={styles.numInput}
              value={costRate}
              onChange={(e) => setCostRate(e.target.value)}
              onBlur={runQuery}
            />
            <span className={styles.ctlLabel}>Lots x</span>
            <input
              className={styles.numInput}
              value={lotsScale}
              onChange={(e) => setLotsScale(e.target.value)}
              onBlur={runQuery}
            />
          </div>
        </div>
      )}

      {meta && (
        <div className={styles.slice}>
          Slice:{' '}
          {indices.length ? (
            indices.join(' + ')
          ) : (
            <b>ALL indices pooled ({meta.indices.join(' + ')} on one equity curve)</b>
          )}
          {' · SL '}
          {sls.length ? sls.map((s) => `${s}%`).join('/') : 'all'}
          {' · DTE '}
          {dtes.length ? dtes.join(',') : 'all'}
          {' · '}
          {yearFrom || 'start'}&ndash;{yearTo || 'end'}
          {exclEvents ? ' · ex budget/election days' : ' · events INCLUDED'}
          {lotsScale !== '1' && lotsScale !== '' ? ` · P&L scaled x${lotsScale}` : ''}
        </div>
      )}

      {err && <div className={styles.error}>{err}</div>}
      {loading && <div className={styles.loading}>computing...</div>}

      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr>
              {COLS.map((c) => (
                <th
                  key={c.label}
                  className={`${c.left ? styles.left : ''} ${
                    colSort && colSort.key === c.key ? styles.thActive : ''
                  }`}
                  onClick={() => clickCol(c)}
                  title={c.key ? 'click to sort' : undefined}
                >
                  {c.label}
                  {colSort && colSort.key === c.key && (
                    <span className={styles.sortInd}>{colSort.dir === 1 ? '▲' : '▼'}</span>
                  )}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {view.map((m, i) => (
              <>
                <tr
                  key={m.label}
                  className={styles.row}
                  onClick={() => expandRow(m.label)}
                >
                  <td className={styles.dim}>{i + 1}</td>
                  <td className={`${styles.left} ${styles.sys}`}>{m.label}</td>
                  <td>{m.n}</td>
                  <td className={m.net >= 0 ? styles.pos : styles.neg}>{inr(m.net)}</td>
                  <td className={m.mean >= 0 ? styles.pos : styles.neg}>{inr(m.mean)}</td>
                  <td className={m.median >= 0 ? styles.pos : styles.neg}>{inr(m.median)}</td>
                  <td>{m.win_pct.toFixed(1)}</td>
                  <td className={styles.dim}>{inr(m.avg_win)}</td>
                  <td className={styles.dim}>{inr(m.avg_loss)}</td>
                  <td className={styles.neg}>{inr(m.maxdd)}</td>
                  <td className={styles.neg}>{inr(m.worst)}</td>
                  <td className={m.lose_streak > 7 ? styles.neg : undefined}>{m.lose_streak}</td>
                  <td>{m.net_dd ?? '-'}</td>
                  <td>{m.calmar ?? '-'}</td>
                  <td>{m.pf ?? '-'}</td>
                  <td>{m.t.toFixed(2)}</td>
                  <td>{m.years_positive}/{m.years_total}</td>
                  <td>
                    <span
                      title={verdictReasons(m)}
                      className={
                        m.verdict === 'PASS'
                          ? styles.vPass
                          : m.verdict === 'rej: WR/streak'
                            ? styles.vWarn
                            : styles.vRej
                      }
                    >
                      {m.verdict}
                    </span>
                  </td>
                </tr>
                {expanded === m.label && (
                  <tr key={`${m.label}-x`} className={styles.expandRow}>
                    <td colSpan={18}>
                      <div className={styles.expandBody}>
                        <div className={styles.expandTitle}>
                          {m.label} - monthly P&amp;L heatmap
                          <span className={styles.expandExtras}>
                            best trade <b className={styles.pos}>{inr(m.best)}</b> · win streak{' '}
                            <b>{m.win_streak}</b> · R:R <b>{m.rr ?? '-'}</b>
                          </span>
                        </div>
                        <div className={styles.heatWrap}>
                          <Heatmap perMonth={m.per_month} perYear={m.per_year} />
                        </div>
                        <div className={styles.expandTitle}>
                          Cumulative P&amp;L and drawdown
                        </div>
                        {series[m.label] === 'loading' && (
                          <div className={styles.loading}>loading curve...</div>
                        )}
                        {Array.isArray(series[m.label]) && (
                          <EquityChart points={series[m.label] as SeriesPoint[]} />
                        )}
                      </div>
                    </td>
                  </tr>
                )}
              </>
            ))}
            {!loading && rows.length === 0 && (
              <tr>
                <td colSpan={18} className={styles.empty}>No trades match the current filters.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <p className={styles.footnote}>
        Gates: stats = t &ge; 2.0, PF &ge; 1.3, &ge; 80% of years positive - tradeability = WR &ge;
        45%, losing streak &le; 7. Click a row for per-year P&amp;L. Data:
        backtest_data/algotest_studies.db (gross + turnover stored, costs applied at query time).
      </p>
    </div>
  );
}
