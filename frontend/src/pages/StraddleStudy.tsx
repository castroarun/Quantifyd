import { useCallback, useEffect, useMemo, useState } from 'react';
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
  verdict: string;
  gate_stats: boolean;
  gate_tradeable: boolean;
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

/* ------------------------------------------------------------------ page */

export default function StraddleStudy() {
  const [meta, setMeta] = useState<RunsResp | null>(null);
  const [rows, setRows] = useState<MetricRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState('');

  // filters
  const [indices, setIndices] = useState<string[]>([]);
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

  const runQuery = useCallback(() => {
    setLoading(true);
    setErr('');
    const p = new URLSearchParams();
    if (indices.length) p.set('index', indices.join(','));
    if (sls.length) p.set('sl', sls.join(','));
    if (dtes.length) p.set('dte', dtes.join(','));
    if (yearFrom !== '') p.set('year_from', String(yearFrom));
    if (yearTo !== '') p.set('year_to', String(yearTo));
    p.set('group_by', groupBy);
    p.set('sort', sortBy);
    p.set('exclude_events', exclEvents ? '1' : '0');
    if (costRate) p.set('cost_rate', costRate);
    if (lotsScale && lotsScale !== '1') p.set('lots_scale', lotsScale);
    apiGet<QueryResp>(`/api/straddle-study/query?${p.toString()}`)
      .then((r) => {
        setRows(r.rows);
        setColSort(null); // fresh data comes back in server rank order
      })
      .catch((e) => setErr(String(e)))
      .finally(() => setLoading(false));
  }, [indices, sls, dtes, yearFrom, yearTo, groupBy, sortBy, exclEvents, costRate, lotsScale]);

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
                className={`${styles.chip} ${indices.includes(ix) ? styles.chipOn : ''}`}
                onClick={() => toggle(indices, ix, setIndices)}
              >
                {ix}
              </button>
            ))}
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
                  onClick={() => setExpanded(expanded === m.label ? null : m.label)}
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
                      <div className={styles.perYear}>
                        {Object.entries(m.per_year).map(([y, v]) => (
                          <span key={y} className={styles.yearCell}>
                            <span className={styles.yearLbl}>{y}</span>
                            <span className={v >= 0 ? styles.pos : styles.neg}>{inr(v)}</span>
                          </span>
                        ))}
                        <span className={styles.yearCell}>
                          <span className={styles.yearLbl}>best trade</span>
                          <span className={styles.pos}>{inr(m.best)}</span>
                        </span>
                        <span className={styles.yearCell}>
                          <span className={styles.yearLbl}>win streak</span>
                          <span>{m.win_streak}</span>
                        </span>
                        <span className={styles.yearCell}>
                          <span className={styles.yearLbl}>R:R</span>
                          <span>{m.rr ?? '-'}</span>
                        </span>
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
