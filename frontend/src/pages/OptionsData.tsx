import { useEffect, useMemo, useState } from 'react';
import styles from './OptionsData.module.css';
import { apiGet } from '../api/client';
import MetricCard from '../components/Cards/MetricCard';
import { formatInt } from '../utils/format';

interface SessionRow {
  date: string;
  rows: number;
  symbols: number;
  first_snapshot: string;
  last_snapshot: string;
  per_index: Record<string, number>;
  status: 'ok' | 'failed';
}

interface CoverageResp {
  cumulative: {
    rows: number;
    symbols: number;
    sessions: number;
    date_min: string | null;
    date_max: string | null;
    spot_rows: number;
    ohlc_rows: number;
    size_mb: number;
  };
  sessions: SessionRow[];
  granularity: string;
  capture_window: string;
}

function fmtDate(d: string): string {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(d);
  if (!m) return d;
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  return `${parseInt(m[3], 10)} ${months[parseInt(m[2], 10) - 1]} ${m[1]}`;
}

function weekday(d: string): string {
  const dt = new Date(d + 'T00:00:00Z');
  const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  return days[dt.getUTCDay()];
}

function fmtSize(mb: number): string {
  if (mb >= 1024) return `${(mb / 1024).toFixed(2)} GB`;
  return `${mb.toFixed(1)} MB`;
}

export default function OptionsData() {
  const [data, setData] = useState<CoverageResp | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    const load = () => {
      apiGet<CoverageResp>('/api/options/coverage')
        .then((r) => {
          if (cancelled) return;
          setData(r);
          setErr(null);
          setLoading(false);
        })
        .catch((e) => {
          if (cancelled) return;
          setErr(String(e));
          setLoading(false);
        });
    };
    load();
    const timer = setInterval(load, 60_000);  // refresh every 60s
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  const indices = useMemo(() => {
    const set = new Set<string>();
    (data?.sessions ?? []).forEach((s) => {
      Object.keys(s.per_index || {}).forEach((k) => set.add(k));
    });
    return Array.from(set).sort();
  }, [data]);

  if (loading && !data) {
    return <div className={styles.loading}>Loading options coverage…</div>;
  }
  if (err) {
    return <div className={styles.error}>Failed to load: {err}</div>;
  }
  if (!data) return null;

  const { cumulative, sessions, granularity, capture_window } = data;
  const failed = sessions.filter((s) => s.status === 'failed').length;
  const dateRange = cumulative.date_min && cumulative.date_max
    ? `${fmtDate(cumulative.date_min)} – ${fmtDate(cumulative.date_max)}`
    : '—';

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <div className={styles.title}>Options data capture</div>
        <div className={styles.subtitle}>
          1-min snapshots of NIFTY + BANKNIFTY + SENSEX option chains · {capture_window}
        </div>
      </div>

      <div className={styles.cards}>
        <MetricCard label="Sessions" value={formatInt(cumulative.sessions)} hint={dateRange} />
        <MetricCard label="Rows (option_chain)" value={formatInt(cumulative.rows)} hint={`${formatInt(cumulative.symbols)} distinct symbols`} />
        <MetricCard label="DB size" value={fmtSize(cumulative.size_mb)} hint={granularity} />
        <MetricCard
          label={failed ? 'Failed days' : 'All sessions OK'}
          value={failed ? String(failed) : 'all clean'}
          hint={failed ? 'days with zero captures' : 'no silent failures'}
          valueClassName={failed ? styles.valueBad : styles.valueGood}
        />
      </div>

      <div className={styles.sectionTitle}>Day-wise capture log</div>
      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>Date</th>
              <th>Day</th>
              <th>Status</th>
              <th style={{ textAlign: 'right' }}>Rows</th>
              <th style={{ textAlign: 'right' }}>Symbols</th>
              <th>Window</th>
              {indices.map((idx) => (
                <th key={idx} style={{ textAlign: 'right' }}>{idx}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sessions.map((s) => (
              <tr key={s.date} className={s.status === 'failed' ? styles.rowFail : styles.rowOk}>
                <td className={styles.dateCell}>{fmtDate(s.date)}</td>
                <td className={styles.muted}>{weekday(s.date)}</td>
                <td>
                  <span className={s.status === 'ok' ? styles.pillOk : styles.pillFail}>
                    {s.status === 'ok' ? '✓ captured' : '✕ empty'}
                  </span>
                </td>
                <td className={styles.num}>{formatInt(s.rows)}</td>
                <td className={styles.num}>{formatInt(s.symbols)}</td>
                <td className={styles.muted}>
                  {s.first_snapshot && s.last_snapshot
                    ? `${s.first_snapshot} → ${s.last_snapshot}`
                    : '—'}
                </td>
                {indices.map((idx) => (
                  <td key={idx} className={styles.num}>
                    {s.per_index[idx] ? formatInt(s.per_index[idx]) : '—'}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className={styles.foot}>
        underlying_spot rows: {formatInt(cumulative.spot_rows)}
        {cumulative.ohlc_rows > 0 ? ` · option_ohlc rows: ${formatInt(cumulative.ohlc_rows)}` : ''}
      </div>
    </div>
  );
}
