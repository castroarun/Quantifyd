import { useEffect, useMemo, useState } from 'react';
import styles from './Report.module.css';
import { apiGet } from '../api/client';
import type {
  NASReportData,
  NASReportSystem,
  ORBBacktestRun,
  ORBBacktestSignal,
  ORBLiveDailyResponse,
  ORBLiveDay,
  ORBLiveTrade,
} from '../api/types';
import MetricCard from '../components/Cards/MetricCard';
import DataTable from '../components/DataTable/DataTable';
import type { Column } from '../components/DataTable/DataTable';
import Chip from '../components/Chip/Chip';
import { formatInt, formatNumber, formatPct, formatPnl, formatPnlBare, pnlClass } from '../utils/format';

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

interface DayTrade {
  systemKey: string;
  systemLabel: string;
  leg: string;           // 'CE' | 'PE'
  tradingsymbol: string;
  strike: number | null;
  entryTime: string;     // "HH:MM"
  entryPrice: number;
  exitTime: string;      // "HH:MM" or ''
  exitPrice: number | null;
  exitReason: string | null;
  pnl: number;
  qty: number;
  status: string;
}

interface DaySummary {
  date: string;
  dayName: string;
  totalPnl: number;
  perSystem: Array<{ key: string; label: string; pnl: number; trades: number }>;
  trades: DayTrade[];
}

export default function Report() {
  const [data, setData] = useState<NASReportData | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [orb, setOrb] = useState<ORBBacktestRun | null>(null);
  const [orbHistory, setOrbHistory] = useState<ORBBacktestRun[]>([]);
  const [orbErr, setOrbErr] = useState<string | null>(null);
  const [orbLive, setOrbLive] = useState<ORBLiveDailyResponse | null>(null);
  const [orbLiveErr, setOrbLiveErr] = useState<string | null>(null);

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
    // Load ORB backtest — latest run + history
    apiGet<ORBBacktestRun>('/api/orb/backtest')
      .then((r) => {
        if (!cancelled) setOrb(r);
      })
      .catch((e) => {
        if (!cancelled) setOrbErr(e instanceof Error ? e.message : 'ORB load failed');
      });
    apiGet<ORBBacktestRun[]>('/api/orb/backtest?list=1')
      .then((list) => {
        if (!cancelled) setOrbHistory(list);
      })
      .catch(() => {
        /* no history available yet */
      });
    apiGet<ORBLiveDailyResponse>('/api/orb/live-daily')
      .then((r) => {
        if (!cancelled) setOrbLive(r);
      })
      .catch((e) => {
        if (!cancelled) setOrbLiveErr(e instanceof Error ? e.message : 'ORB live load failed');
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
      // Build flat trade log: one row per position (leg), sorted ASC by entry_time
      const trades: DayTrade[] = [];
      for (const k of SYS_KEYS) {
        const sd = daySys[k];
        if (!sd?.positions) continue;
        for (const p of sd.positions) {
          const entryPrice = (p.entry_price as number | undefined) ?? 0;
          const exitPrice = (p.exit_price as number | null | undefined) ?? null;
          const qty = (p.qty as number | undefined) ?? 0;
          const pnl =
            exitPrice != null && entryPrice != null && qty
              ? Math.round((entryPrice - exitPrice) * qty * 100) / 100
              : 0;
          const tsym = (p.tradingsymbol as string | undefined) ?? '';
          // Tradingsymbol format: UNDERLYING + YY + M + DD + STRIKE + CE/PE
          // e.g. NIFTY 26 4 21 24750 CE  ->  strike = 24750
          // Skip the 5-char expiry (YY/M/DD) that appears after the underlying.
          const m = tsym.match(/^[A-Z]+(\d{2}[A-Z0-9]\d{2})(\d+)(CE|PE)$/);
          const strike = m ? parseInt(m[2], 10) : null;
          const toHM = (s: string | undefined | null) =>
            s ? s.slice(11, 16) : '';
          trades.push({
            systemKey: k,
            systemLabel: SYS_LABEL[k] ?? k,
            leg: (p.leg as string | undefined) ?? '',
            tradingsymbol: tsym,
            strike,
            entryTime: toHM(p.entry_time),
            entryPrice,
            exitTime: toHM(p.exit_time),
            exitPrice,
            exitReason: (p.exit_reason as string | undefined) ?? null,
            pnl,
            qty,
            status: (p.status as string | undefined) ?? '',
          });
        }
      }
      trades.sort((a, b) => a.entryTime.localeCompare(b.entryTime));
      const totalPnl = perSystem.reduce((a, p) => a + p.pnl, 0);
      out.push({ date, dayName, totalPnl, perSystem, trades });
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
      header: 'Total P&L (Rs)',
      width: '1.1fr',
      align: 'right',
      render: (r) => (
        <span className={pnlClass(r.total_pnl)}>{formatPnlBare(r.total_pnl)}</span>
      ),
    },
    {
      key: 'avg',
      header: 'Avg P&L (Rs)',
      width: '1fr',
      align: 'right',
      render: (r) => (
        <span className={pnlClass(r.avg_pnl)}>{formatPnlBare(r.avg_pnl)}</span>
      ),
    },
    {
      key: 'mw',
      header: 'Max win (Rs)',
      width: '1fr',
      align: 'right',
      render: (r) => (
        <span className={pnlClass(r.max_win)}>{formatPnlBare(r.max_win)}</span>
      ),
    },
    {
      key: 'ml',
      header: 'Max loss (Rs)',
      width: '1fr',
      align: 'right',
      render: (r) => (
        <span className={pnlClass(r.max_loss)}>{formatPnlBare(r.max_loss)}</span>
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
        />
        <MetricCard
          label="Overall win rate"
          value={formatPct(summary.overallWinRate, 1)}
        />
        <MetricCard
          label="Avg profit factor"
          value={summary.avgPf ? formatNumber(summary.avgPf, 2) : '—'}
        />
        <MetricCard
          label="Total P&L (Rs)"
          value={
            <span className={pnlClass(summary.totalPnl)}>
              {formatPnlBare(summary.totalPnl)}
            </span>
          }
        />
        <MetricCard
          label="Best day (Rs)"
          labelRight={bestWorst.best?.dayName}
          value={
            bestWorst.best ? (
              <span className={pnlClass(bestWorst.best.totalPnl)}>
                {formatPnlBare(bestWorst.best.totalPnl)}
              </span>
            ) : (
              '—'
            )
          }
        />
        <MetricCard
          label="Worst day (Rs)"
          labelRight={bestWorst.worst?.dayName}
          value={
            bestWorst.worst ? (
              <span className={pnlClass(bestWorst.worst.totalPnl)}>
                {formatPnlBare(bestWorst.worst.totalPnl)}
              </span>
            ) : (
              '—'
            )
          }
        />
      </div>

      <div className={styles.secondaryMetrics}>
        <MetricCard
          label="Active trading days"
          value={formatInt(daily.length)}
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

      <OrbLiveSection data={orbLive} error={orbLiveErr} />
      <OrbBacktestSection latest={orb} history={orbHistory} error={orbErr} />
    </div>
  );
}

function OrbLiveSection({
  data,
  error,
}: {
  data: ORBLiveDailyResponse | null;
  error: string | null;
}) {
  if (error && !data) {
    return (
      <section className={styles.section}>
        <div className={styles.sectionHead}>
          <div className="section-title">ORB cash · live daily</div>
        </div>
        <div className={styles.error}>{error}</div>
      </section>
    );
  }
  if (!data) {
    return (
      <section className={styles.section}>
        <div className={styles.sectionHead}>
          <div className="section-title">ORB cash · live daily</div>
        </div>
        <div className={styles.empty}>Loading live trades…</div>
      </section>
    );
  }
  const s = data.summary;
  return (
    <section className={styles.section}>
      <div className={styles.sectionHead}>
        <div className="section-title">ORB cash · live daily</div>
        <Chip>{s.active_days} days</Chip>
      </div>
      <div className={styles.subtitle}>
        Actual fills from the live ORB cash system (paper or live mode). Click
        any day to see position-level detail.
      </div>

      <div className={styles.metrics}>
        <MetricCard label="Live trades" value={formatInt(s.total_trades)} />
        <MetricCard label="Win rate" value={formatPct(s.win_rate, 1)} />
        <MetricCard
          label="Total P&L (Rs)"
          value={
            <span className={pnlClass(s.total_pnl_inr)}>
              {formatPnlBare(s.total_pnl_inr)}
            </span>
          }
        />
        <MetricCard label="Active days" value={formatInt(s.active_days)} />
      </div>

      {data.days.length === 0 ? (
        <div className={styles.empty}>No live ORB trades recorded yet.</div>
      ) : (
        <div className={styles.daysList}>
          {data.days.map((d) => (
            <OrbLiveDayBlock key={d.trade_date} day={d} />
          ))}
        </div>
      )}
    </section>
  );
}

function OrbLiveDayBlock({ day }: { day: ORBLiveDay }) {
  const dt = new Date(day.trade_date + 'T00:00:00');
  const dayName = dt.toLocaleDateString('en-IN', {
    weekday: 'short',
    day: '2-digit',
    month: 'short',
    year: 'numeric',
  });
  return (
    <details className={styles.dayBlock}>
      <summary className={styles.daySummary}>
        <span className={styles.dayName}>{dayName}</span>
        <span className={styles.dayDate}>{day.trade_date}</span>
        <span className={styles.daySpacer} />
        <span className={styles.dayMeta}>
          {day.trades_count} {day.trades_count === 1 ? 'trade' : 'trades'} ·{' '}
          {day.winners}W / {day.losers}L
        </span>
        <span className={`${styles.dayPnl} ${pnlClass(day.daily_pnl_inr)}`}>
          {formatPnl(day.daily_pnl_inr)}
        </span>
      </summary>
      <div className={styles.tradeLog}>
        <div className={styles.orbTableHead}>
          <div>Stock</div>
          <div>Dir</div>
          <div>Entry</div>
          <div>Exit</div>
          <div>Reason</div>
          <div className={styles.orbRight}>P&amp;L</div>
        </div>
        {day.trades.map((t) => (
          <OrbLiveTradeRow key={t.id} t={t} />
        ))}
      </div>
    </details>
  );
}

function OrbLiveTradeRow({ t }: { t: ORBLiveTrade }) {
  const entryT = (t.entry_time ?? '').slice(11, 16);
  const exitT = (t.exit_time ?? '').slice(11, 16);
  return (
    <div className={styles.orbTableRow}>
      <div className={styles.orbStock}>{t.instrument}</div>
      <div className={styles.orbDir}>{t.direction}</div>
      <div className={styles.orbCell}>
        {entryT || '—'} @ {formatNumber(t.entry_price, 2)}
      </div>
      <div className={styles.orbCell}>
        {exitT || '—'} @{' '}
        {t.exit_price != null ? formatNumber(t.exit_price, 2) : '—'}
      </div>
      <div className={styles.orbReason}>{t.exit_reason ?? '—'}</div>
      <div className={`${styles.orbRight} ${pnlClass(t.pnl_inr ?? 0)}`}>
        {formatPnl(t.pnl_inr ?? 0)}
      </div>
    </div>
  );
}

function OrbBacktestSection({
  latest,
  history,
  error,
}: {
  latest: ORBBacktestRun | null;
  history: ORBBacktestRun[];
  error: string | null;
}) {
  if (error && !latest) {
    return (
      <section className={styles.section}>
        <div className={styles.sectionHead}>
          <div className="section-title">ORB daily backtest</div>
        </div>
        <div className={styles.error}>
          {error.includes('404') || error.includes('no backtest') ? (
            <>No ORB backtest runs stored yet. The scheduler runs daily at 15:45 IST.</>
          ) : (
            error
          )}
        </div>
      </section>
    );
  }
  if (!latest) {
    return null;
  }
  const signals = latest.signals ?? [];
  const taken = signals.filter((s) => s.signal_type === 'TAKEN');
  const blocked = signals.filter((s) => s.signal_type === 'BLOCKED');
  const noBreakout = signals.filter((s) => s.signal_type === 'NO_BREAKOUT');
  const wideCpr = signals.filter((s) => s.signal_type === 'SKIP_WIDE_CPR');

  return (
    <section className={styles.section}>
      <div className={styles.sectionHead}>
        <div className="section-title">ORB daily backtest</div>
        <Chip>{history.length || 1} runs</Chip>
      </div>
      <div className={styles.subtitle}>
        Legacy daily backtest — simpler rules than live (Rs 20k/trade · OR15 · 0.5% SL · 1R target).
        Retained for historical day-by-day reference. Latest: {latest.run_date}.
        See the V9t_lock50 250-day backtest above for the current live strategy.
      </div>

      <div className={styles.metrics}>
        <MetricCard
          label="Trades taken"
          value={formatInt(latest.trades_taken)}
          hint="Passed CPR + RSI + gap filters"
        />
        <MetricCard
          label="Net P&L"
          value={
            <span className={pnlClass(latest.net_pnl_inr)}>
              {formatPnl(latest.net_pnl_inr)}
            </span>
          }
          hint={`Across ${latest.trades_taken} trades`}
        />
        <MetricCard
          label="Blocked by filters"
          value={formatInt(latest.signals_blocked)}
          hint="Breakout but filter vetoed"
        />
        <MetricCard
          label="No breakout"
          value={formatInt(noBreakout.length)}
          hint={`${wideCpr.length} wide CPR skip`}
        />
      </div>

      <OrbSignalTable title="Trades taken" rows={taken} />
      {blocked.length > 0 && (
        <OrbSignalTable title="Blocked by filters" rows={blocked} blocked />
      )}

      {history.length > 1 && (
        <>
          <div className={styles.sectionHead} style={{ marginTop: 24 }}>
            <div className="section-title">Daily history</div>
            <Chip>{history.length} days</Chip>
          </div>
          <div className={styles.daysList}>
            {history.map((h) => (
              <OrbDayBlock key={h.run_date} summary={h} />
            ))}
          </div>
        </>
      )}
    </section>
  );
}

function OrbDayBlock({ summary }: { summary: ORBBacktestRun }) {
  const [details, setDetails] = useState<ORBBacktestRun | null>(
    summary.signals && summary.signals.length > 0 ? summary : null,
  );
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const dt = new Date(summary.run_date + 'T00:00:00');
  const dayName = dt.toLocaleDateString('en-IN', {
    weekday: 'short',
    day: '2-digit',
    month: 'short',
    year: 'numeric',
  });

  const onToggle = (ev: React.SyntheticEvent<HTMLDetailsElement>) => {
    if (!ev.currentTarget.open) return;
    if (details || loading) return;
    setLoading(true);
    apiGet<ORBBacktestRun>(`/api/orb/backtest?date=${summary.run_date}`)
      .then((r) => setDetails(r))
      .catch((e) => setErr(e instanceof Error ? e.message : 'load failed'))
      .finally(() => setLoading(false));
  };

  const signals = details?.signals ?? [];
  const taken = signals.filter((s) => s.signal_type === 'TAKEN');
  const blocked = signals.filter((s) => s.signal_type === 'BLOCKED');

  return (
    <details className={styles.dayBlock} onToggle={onToggle}>
      <summary className={styles.daySummary}>
        <span className={styles.dayName}>{dayName}</span>
        <span className={styles.dayDate}>{summary.run_date}</span>
        <span className={styles.daySpacer} />
        <span className={styles.dayMeta}>
          {summary.trades_taken} taken · {summary.signals_blocked} blocked
        </span>
        <span className={`${styles.dayPnl} ${pnlClass(summary.net_pnl_inr)}`}>
          {formatPnl(summary.net_pnl_inr)}
        </span>
      </summary>
      {loading ? (
        <div className={styles.orbEmpty}>Loading…</div>
      ) : err ? (
        <div className={styles.error}>{err}</div>
      ) : details ? (
        <div style={{ padding: '0 12px 12px' }}>
          {taken.length === 0 && blocked.length === 0 ? (
            <div className={styles.orbEmpty}>No signals on this day.</div>
          ) : (
            <>
              {taken.length > 0 && (
                <OrbSignalTable title="Trades taken" rows={taken} />
              )}
              {blocked.length > 0 && (
                <OrbSignalTable title="Blocked by filters" rows={blocked} blocked />
              )}
            </>
          )}
        </div>
      ) : null}
    </details>
  );
}

function OrbSignalTable({
  title,
  rows,
  blocked = false,
}: {
  title: string;
  rows: ORBBacktestSignal[];
  blocked?: boolean;
}) {
  if (rows.length === 0) {
    return (
      <div className={styles.orbEmpty}>No {title.toLowerCase()}.</div>
    );
  }
  return (
    <div className={styles.orbTable}>
      <div className={styles.orbHead}>{title} · {rows.length}</div>
      <div className={styles.orbTableHead}>
        <div>Stock</div>
        <div>Dir</div>
        <div>Entry</div>
        <div>Exit</div>
        <div>Reason</div>
        <div className={styles.orbRight}>P&amp;L</div>
      </div>
      {rows.map((r, i) => (
        <div key={i} className={styles.orbTableRow}>
          <div className={styles.orbStock}>{r.instrument}</div>
          <div className={styles.orbDir}>{r.direction ?? '—'}</div>
          <div className={styles.orbCell}>
            {r.entry_time ?? '—'} @ {r.entry_price?.toFixed(1) ?? '—'}
          </div>
          <div className={styles.orbCell}>
            {r.exit_time ?? '—'} @ {r.exit_price?.toFixed(1) ?? '—'}
          </div>
          <div className={styles.orbReason}>
            {blocked ? r.block_reason ?? '—' : r.exit_reason ?? '—'}
          </div>
          <div className={`${styles.orbRight} ${pnlClass(r.pnl_inr ?? 0)}`}>
            {formatPnl(r.pnl_inr ?? 0)}
          </div>
        </div>
      ))}
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
        <span className={styles.dayMeta}>{day.trades.length} legs</span>
        <span className={`${styles.dayPnl} ${pnlClass(day.totalPnl)}`}>
          {formatPnl(day.totalPnl)}
        </span>
      </summary>
      <div className={styles.daySystems}>
        {day.perSystem.map((ps) =>
          ps.trades === 0 && ps.pnl === 0 ? null : (
            <div key={ps.key} className={styles.daySystemRow}>
              <span className={styles.daySystemName}>{ps.label}</span>
              <span className={styles.daySystemTrades}>
                {ps.trades} {ps.trades === 1 ? 'trade' : 'trades'}
              </span>
              <span className={`${styles.daySystemPnl} ${pnlClass(ps.pnl)}`}>
                {formatPnl(ps.pnl)}
              </span>
            </div>
          ),
        )}
      </div>
      {day.trades.length > 0 ? (
        <div className={styles.tradeLog}>
          {groupTradesBySystem(day.trades).map((group) => (
            <div key={group.systemKey} className={styles.tradeGroup}>
              <div className={styles.tradeGroupHead}>
                <span className={styles.tradeGroupName}>{group.systemLabel}</span>
                <span className={styles.tradeGroupMeta}>
                  {group.trades.length} {group.trades.length === 1 ? 'leg' : 'legs'}
                </span>
                <span
                  className={`${styles.tradeGroupPnl} ${pnlClass(group.pnl)}`}
                >
                  {formatPnl(group.pnl)}
                </span>
              </div>
              <div className={styles.tradeLogHead}>
                <div>Time</div>
                <div>Side</div>
                <div>Leg</div>
                <div>Strike</div>
                <div>Entry</div>
                <div>Exit</div>
                <div>Reason</div>
                <div className={styles.tradeRight}>P&amp;L</div>
              </div>
              {group.trades.map((t, i) => (
                <div key={i} className={styles.tradeLogRow}>
                  <div className={styles.tradeTime}>
                    {t.entryTime}
                    {t.exitTime ? (
                      <span className={styles.tradeArrow}>→ {t.exitTime}</span>
                    ) : null}
                  </div>
                  <div>
                    <span className={styles.tradeSell}>S</span>
                    {t.exitPrice != null ? (
                      <span className={styles.tradeBuy}>B</span>
                    ) : null}
                  </div>
                  <div className={styles.tradeLeg}>{t.leg}</div>
                  <div className={styles.tradeStrike}>
                    {t.strike != null ? t.strike : '—'}
                  </div>
                  <div className={styles.tradeCell}>
                    {formatNumber(t.entryPrice, 2)}
                  </div>
                  <div className={styles.tradeCell}>
                    {t.exitPrice != null ? formatNumber(t.exitPrice, 2) : '—'}
                  </div>
                  <div className={styles.tradeReason}>
                    {t.exitReason ?? (t.status === 'ACTIVE' ? 'open' : '—')}
                  </div>
                  <div className={`${styles.tradeRight} ${pnlClass(t.pnl)}`}>
                    {formatPnl(t.pnl)}
                  </div>
                </div>
              ))}
            </div>
          ))}
        </div>
      ) : null}
    </details>
  );
}

function groupTradesBySystem(trades: DayTrade[]) {
  const bucket = new Map<string, { systemKey: string; systemLabel: string; trades: DayTrade[]; pnl: number }>();
  for (const t of trades) {
    const g = bucket.get(t.systemKey);
    if (g) {
      g.trades.push(t);
      g.pnl += t.pnl;
    } else {
      bucket.set(t.systemKey, {
        systemKey: t.systemKey,
        systemLabel: t.systemLabel,
        trades: [t],
        pnl: t.pnl,
      });
    }
  }
  // Order groups by the SYS_KEYS order so the view is consistent across days
  const order = new Map(SYS_KEYS.map((k, i) => [k, i] as const));
  const groups = Array.from(bucket.values()).sort(
    (a, b) => (order.get(a.systemKey) ?? 99) - (order.get(b.systemKey) ?? 99),
  );
  // Within each group, sort by entry time ASC
  for (const g of groups) {
    g.trades.sort((a, b) => a.entryTime.localeCompare(b.entryTime));
    g.pnl = Math.round(g.pnl * 100) / 100;
  }
  return groups;
}
