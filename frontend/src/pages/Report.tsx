import { useEffect, useMemo, useState } from 'react';
import styles from './Report.module.css';
import { apiGet } from '../api/client';
import type { NASReportData, NASReportSystem } from '../api/types';
import MetricCard from '../components/Cards/MetricCard';
import DataTable from '../components/DataTable/DataTable';
import type { Column } from '../components/DataTable/DataTable';
import Chip from '../components/Chip/Chip';
import { formatInt, formatNumber, formatPct, formatPnl, pnlClass } from '../utils/format';

const SYS_KEYS = [
  'OTM',
  'ATM',
  'ATM2',
  'ATM4',
  '916-OTM',
  '916-ATM',
  '916-ATM2',
  '916-ATM4',
];

const SYS_LABEL: Record<string, string> = {
  OTM: 'Squeeze · OTM',
  ATM: 'Squeeze · ATM',
  ATM2: 'Squeeze · ATM 2.0',
  ATM4: 'Squeeze · ATM V4',
  '916-OTM': '9:16 · OTM',
  '916-ATM': '9:16 · ATM',
  '916-ATM2': '9:16 · ATM 2.0',
  '916-ATM4': '9:16 · ATM V4',
};

const DOW_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'];

interface SystemRow {
  key: string;
  label: string;
  trades: number;
  win_rate: number;
  pf: number;
  total_pnl: number;
  avg_pnl: number;
  max_win: number;
  max_loss: number;
  error?: string;
}

interface DaySummary {
  date: string;
  dayName: string;
  totalPnl: number;
  perSystem: Array<{ key: string; label: string; pnl: number; trades: number }>;
}

export default function Report() {
  const [data, setData] = useState<NASReportData | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    apiGet<NASReportData>('/api/nas/report-data')
      .then((d) => {
        if (!cancelled) {
          setData(d);
          setErr(null);
        }
      })
      .catch((e) => {
        if (!cancelled) setErr(e instanceof Error ? e.message : 'Load failed');
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const rows: SystemRow[] = useMemo(() => {
    if (!data?.systems) return [];
    return SYS_KEYS.map((k) => {
      const s: NASReportSystem = data.systems[k] ?? {};
      return {
        key: k,
        label: SYS_LABEL[k] ?? k,
        trades: s.total_trades ?? 0,
        win_rate: s.win_rate ?? 0,
        pf: s.profit_factor ?? 0,
        total_pnl: s.total_pnl ?? 0,
        avg_pnl: s.avg_pnl ?? 0,
        max_win: s.max_win ?? 0,
        max_loss: s.max_loss ?? 0,
        error: s.error,
      };
    });
  }, [data]);

  const summary = useMemo(() => {
    const totalTrades = rows.reduce((a, r) => a + r.trades, 0);
    const totalPnl = rows.reduce((a, r) => a + r.total_pnl, 0);
    const totalWinners = rows.reduce((a, r) => {
      const sys: NASReportSystem = data?.systems?.[r.key] ?? {};
      return a + (sys.winners ?? 0);
    }, 0);
    const overallWinRate = totalTrades > 0 ? (totalWinners / totalTrades) * 100 : 0;
    // Aggregate PF: sum of (pf * trades) doesn't make sense; keep weighted by pf if all positive
    const pfs = rows.filter((r) => r.pf > 0).map((r) => r.pf);
    const avgPf = pfs.length ? pfs.reduce((a, b) => a + b, 0) / pfs.length : 0;
    return { totalTrades, totalPnl, overallWinRate, avgPf };
  }, [rows, data]);

  const daily: DaySummary[] = useMemo(() => {
    if (!data?.daily_snapshots) return [];
    const out: DaySummary[] = [];
    for (const [date, daySys] of Object.entries(data.daily_snapshots)) {
      const dt = new Date(date + 'T00:00:00');
      const dayName = dt.toLocaleDateString('en-IN', {
        weekday: 'short',
        day: '2-digit',
        month: 'short',
        year: 'numeric',
      });
      const perSystem = SYS_KEYS.map((k) => {
        const sd = daySys[k] ?? { day_pnl: 0, trade_count: 0, positions: [], trades: [] };
        return {
          key: k,
          label: SYS_LABEL[k] ?? k,
          pnl: sd.day_pnl ?? 0,
          trades: sd.trade_count ?? 0,
        };
      });
      const totalPnl = perSystem.reduce((a, p) => a + p.pnl, 0);
      out.push({ date, dayName, totalPnl, perSystem });
    }
    // newest first
    out.sort((a, b) => b.date.localeCompare(a.date));
    return out;
  }, [data]);

  const bestWorst = useMemo(() => {
    if (!daily.length) return { best: null, worst: null };
    let best = daily[0];
    let worst = daily[0];
    for (const d of daily) {
      if (d.totalPnl > best.totalPnl) best = d;
      if (d.totalPnl < worst.totalPnl) worst = d;
    }
    return { best, worst };
  }, [daily]);

  const dowAgg = useMemo(() => {
    const agg = DOW_NAMES.map((name) => ({
      name,
      total: 0,
      count: 0,
      avg: 0,
    }));
    for (const d of daily) {
      const dt = new Date(d.date + 'T00:00:00');
      const dow = dt.getDay(); // 0=Sun..6=Sat
      if (dow < 1 || dow > 5) continue;
      agg[dow - 1].total += d.totalPnl;
      agg[dow - 1].count += 1;
    }
    for (const a of agg) {
      a.avg = a.count > 0 ? a.total / a.count : 0;
    }
    return agg;
  }, [daily]);

  const maxAbsAvg = useMemo(() => {
    return Math.max(1, ...dowAgg.map((a) => Math.abs(a.avg)));
  }, [dowAgg]);

  const columns: Column<SystemRow>[] = [
    {
      key: 'label',
      header: 'System',
      width: '1.6fr',
      render: (r) => (
        <span className={styles.bold}>
          {r.label}
          {r.error ? <span className={styles.errTag}> · err</span> : null}
        </span>
      ),
    },
    {
      key: 'trades',
      header: 'Trades',
      width: '80px',
      align: 'right',
      render: (r) => formatInt(r.trades),
    },
    {
      key: 'wr',
      header: 'Win rate',
      width: '100px',
      align: 'right',
      render: (r) => formatPct(r.win_rate, 1),
    },
    {
      key: 'pf',
      header: 'PF',
      width: '70px',
      align: 'right',
      render: (r) => (r.pf ? formatNumber(r.pf, 2) : '—'),
    },
    {
      key: 'total',
      header: 'Total P&L',
      width: '1.1fr',
      align: 'right',
      render: (r) => (
        <span className={pnlClass(r.total_pnl)}>{formatPnl(r.total_pnl)}</span>
      ),
    },
    {
      key: 'avg',
      header: 'Avg P&L',
      width: '1fr',
      align: 'right',
      render: (r) => (
        <span className={pnlClass(r.avg_pnl)}>{formatPnl(r.avg_pnl)}</span>
      ),
    },
    {
      key: 'mw',
      header: 'Max win',
      width: '1fr',
      align: 'right',
      render: (r) => (
        <span className={pnlClass(r.max_win)}>{formatPnl(r.max_win)}</span>
      ),
    },
    {
      key: 'ml',
      header: 'Max loss',
      width: '1fr',
      align: 'right',
      render: (r) => (
        <span className={pnlClass(r.max_loss)}>{formatPnl(r.max_loss)}</span>
      ),
    },
  ];

  if (loading) {
    return <div className={styles.loading}>Loading report data…</div>;
  }

  return (
    <div className={styles.root}>
      <div className="page-title">NAS performance report</div>
      <div className="page-subtitle">
        All 4 squeeze + 4 x 9:16 systems, combined view
      </div>

      {err ? <div className={styles.error}>{err}</div> : null}

      {/* Summary cards */}
      <div className={styles.metrics}>
        <MetricCard
          label="Total trades"
          value={formatInt(summary.totalTrades)}
          hint="Across all 8 systems"
        />
        <MetricCard
          label="Overall win rate"
          value={formatPct(summary.overallWinRate, 1)}
          hint="Weighted by trades"
        />
        <MetricCard
          label="Avg profit factor"
          value={summary.avgPf ? formatNumber(summary.avgPf, 2) : '—'}
          hint="Mean across systems"
        />
        <MetricCard
          label="Total P&L"
          value={
            <span className={pnlClass(summary.totalPnl)}>
              {formatPnl(summary.totalPnl)}
            </span>
          }
          hint="Sum of 8 systems"
        />
        <MetricCard
          label="Best day"
          value={
            bestWorst.best ? (
              <span className={pnlClass(bestWorst.best.totalPnl)}>
                {formatPnl(bestWorst.best.totalPnl)}
              </span>
            ) : (
              '—'
            )
          }
          hint={bestWorst.best?.dayName ?? '—'}
        />
        <MetricCard
          label="Worst day"
          value={
            bestWorst.worst ? (
              <span className={pnlClass(bestWorst.worst.totalPnl)}>
                {formatPnl(bestWorst.worst.totalPnl)}
              </span>
            ) : (
              '—'
            )
          }
          hint={bestWorst.worst?.dayName ?? '—'}
        />
      </div>

      <div className={styles.secondaryMetrics}>
        <MetricCard
          label="Active trading days"
          value={formatInt(daily.length)}
          hint="Days with at least one trade"
        />
      </div>

      {/* Per-system summary */}
      <section className={styles.section}>
        <div className={styles.sectionHead}>
          <div className="section-title">Per-system summary</div>
          <Chip>{rows.length} systems</Chip>
        </div>
        <DataTable
          columns={columns}
          rows={rows}
          emptyText="No system data"
          rowKey={(r) => r.key}
        />
      </section>

      {/* Day of week analysis */}
      <section className={styles.section}>
        <div className={styles.sectionHead}>
          <div className="section-title">Day of week</div>
          <Chip>Avg P&L</Chip>
        </div>
        <div className={styles.dowCard}>
          {dowAgg.map((d) => {
            const pct = (Math.abs(d.avg) / maxAbsAvg) * 100;
            const positive = d.avg >= 0;
            return (
              <div key={d.name} className={styles.dowRow}>
                <div className={styles.dowName}>{d.name}</div>
                <div className={styles.dowTrack}>
                  <div
                    className={`${styles.dowFill} ${positive ? styles.dowFillPos : styles.dowFillNeg}`}
                    style={{ width: `${pct}%` }}
                  />
                </div>
                <div className={styles.dowValue}>
                  <span className={pnlClass(d.avg)}>{formatPnl(d.avg)}</span>
                </div>
                <div className={styles.dowCount}>
                  <span className={styles.mute}>
                    {d.count} {d.count === 1 ? 'day' : 'days'}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* Daily snapshots */}
      <section className={styles.section}>
        <div className={styles.sectionHead}>
          <div className="section-title">Daily snapshots</div>
          <Chip>{daily.length} days</Chip>
        </div>
        <div className={styles.daysList}>
          {daily.length === 0 ? (
            <div className={styles.empty}>No trading days recorded yet.</div>
          ) : (
            daily.map((d) => <DayBlock key={d.date} day={d} />)
          )}
        </div>
      </section>
    </div>
  );
}

function DayBlock({ day }: { day: DaySummary }) {
  return (
    <details className={styles.dayBlock}>
      <summary className={styles.daySummary}>
        <span className={styles.dayName}>{day.dayName}</span>
        <span className={styles.dayDate}>{day.date}</span>
        <span className={styles.daySpacer} />
        <span className={`${styles.dayPnl} ${pnlClass(day.totalPnl)}`}>
          {formatPnl(day.totalPnl)}
        </span>
      </summary>
      <div className={styles.daySystems}>
        {day.perSystem.map((ps) => (
          <div key={ps.key} className={styles.daySystemRow}>
            <span className={styles.daySystemName}>{ps.label}</span>
            <span className={styles.daySystemTrades}>
              {ps.trades} {ps.trades === 1 ? 'trade' : 'trades'}
            </span>
            <span className={`${styles.daySystemPnl} ${pnlClass(ps.pnl)}`}>
              {formatPnl(ps.pnl)}
            </span>
          </div>
        ))}
      </div>
    </details>
  );
}
