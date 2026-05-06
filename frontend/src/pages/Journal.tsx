/**
 * Journal — calendar P&L heatmap + recent trades stream.
 *
 * Mirrors frontend/mockups/journal/01_calendar_overview.html. Click a cell
 * to drill in to the day page. Click a recent-trade row to drill in to a
 * single trade.
 */

import { useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import styles from './Journal.module.css';
import { apiGet, apiPost } from '../api/client';
import type {
  JournalSummaryResponse,
  JournalTrade,
  JournalDailyRow,
} from '../api/types';
import { formatPnl, formatInt } from '../utils/format';

const STRATEGY_FILTERS = [
  { key: 'ALL', label: 'All strategies' },
  { key: 'ORB-CASH', label: 'ORB Cash' },
  { key: 'NAS', label: 'NAS' },
  { key: 'KC6', label: 'KC6' },
];

function monthBounds(year: number, monthIndex: number): { from: string; to: string } {
  const start = new Date(Date.UTC(year, monthIndex, 1));
  const end = new Date(Date.UTC(year, monthIndex + 1, 0));
  const fmt = (d: Date) => d.toISOString().slice(0, 10);
  return { from: fmt(start), to: fmt(end) };
}

function pnlBucketClass(pnl: number, peak: number): string {
  if (!pnl) return styles.pnlFlat;
  const pct = Math.min(1, Math.abs(pnl) / Math.max(1, peak));
  const step = Math.min(5, Math.max(1, Math.ceil(pct * 5)));
  const bucket = pnl > 0 ? `pnlPos${step}` : `pnlNeg${step}`;
  // CSS modules: dynamic class lookup
  return (styles as Record<string, string>)[bucket] || styles.pnlFlat;
}

function stratChipClass(strategy: string): string {
  const s = strategy.toUpperCase();
  if (s.startsWith('NAS')) return styles.nas;
  if (s.startsWith('KC6')) return styles.kc6;
  if (s.includes('DIAMOND')) return styles.diamond;
  if (s.includes('LMR') || s.includes('LONG-MR')) return styles.lmr;
  if (s.includes('LTC') || s.includes('LONG-TC')) return styles.ltc;
  return '';
}

function shortStratLabel(s: string): string {
  if (s.length <= 14) return s;
  return s.slice(0, 12) + '...';
}

export default function Journal() {
  const navigate = useNavigate();
  const [now] = useState(() => new Date());
  const [year, setYear] = useState(now.getFullYear());
  const [monthIdx, setMonthIdx] = useState(now.getMonth());
  const [strategy, setStrategy] = useState('ALL');
  const [summary, setSummary] = useState<JournalSummaryResponse | null>(null);
  const [recent, setRecent] = useState<JournalTrade[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  const { from, to } = useMemo(() => monthBounds(year, monthIdx), [year, monthIdx]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setErr(null);
    // Fire sync first (cheap), then load data — sync is idempotent.
    apiPost('/api/journal/sync').catch(() => undefined).finally(() => {
      Promise.all([
        apiGet<JournalSummaryResponse>(
          `/api/journal/summary?from=${from}&to=${to}` +
            (strategy !== 'ALL' ? `&strategy=${encodeURIComponent(strategy)}` : ''),
        ),
        apiGet<{ trades: JournalTrade[] }>(
          `/api/journal/trades?limit=20` +
            (strategy !== 'ALL' ? `&strategy=${encodeURIComponent(strategy)}` : ''),
        ),
      ])
        .then(([sum, recent]) => {
          if (cancelled) return;
          setSummary(sum);
          setRecent(recent.trades);
          setLoading(false);
        })
        .catch((e: Error) => {
          if (cancelled) return;
          setErr(e.message || 'Failed to load journal');
          setLoading(false);
        });
    });
    return () => {
      cancelled = true;
    };
  }, [from, to, strategy]);

  const monthName = new Date(year, monthIdx, 1).toLocaleString('en-US', {
    month: 'long',
    year: 'numeric',
  });

  // Build day map { 'YYYY-MM-DD': row }
  const dayMap = useMemo(() => {
    const m: Record<string, JournalDailyRow> = {};
    (summary?.days ?? []).forEach((d) => {
      m[d.trade_date] = d;
    });
    return m;
  }, [summary]);

  const peak = useMemo(() => {
    let p = 0;
    (summary?.days ?? []).forEach((d) => {
      const v = Math.abs(d.pnl_net || 0);
      if (v > p) p = v;
    });
    return p || 1;
  }, [summary]);

  // Build calendar grid (Mon-Sun)
  const cells = useMemo(() => {
    const first = new Date(year, monthIdx, 1);
    const lastDate = new Date(year, monthIdx + 1, 0).getDate();
    // dayOfWeek: 0=Sun..6=Sat; we want Mon=0
    const startDow = (first.getDay() + 6) % 7;
    const out: { day: number; inMonth: boolean; date?: string }[] = [];
    // Leading blanks
    for (let i = 0; i < startDow; i++) {
      const d = new Date(year, monthIdx, -startDow + i + 1);
      out.push({ day: d.getDate(), inMonth: false });
    }
    for (let d = 1; d <= lastDate; d++) {
      const dt = new Date(Date.UTC(year, monthIdx, d));
      const ds = dt.toISOString().slice(0, 10);
      out.push({ day: d, inMonth: true, date: ds });
    }
    // Trailing to fill grid (multiple of 7, then up to 6 weeks)
    while (out.length % 7 !== 0) {
      const idx = out.length;
      const fakeDay = idx - startDow - lastDate + 1;
      out.push({ day: fakeDay, inMonth: false });
    }
    while (out.length < 42) {
      const idx = out.length;
      const fakeDay = idx - startDow - lastDate + 1;
      out.push({ day: fakeDay, inMonth: false });
    }
    return out;
  }, [year, monthIdx]);

  const todayISO = new Date().toISOString().slice(0, 10);

  function handlePrevMonth() {
    if (monthIdx === 0) {
      setMonthIdx(11);
      setYear((y) => y - 1);
    } else {
      setMonthIdx((m) => m - 1);
    }
  }
  function handleNextMonth() {
    if (monthIdx === 11) {
      setMonthIdx(0);
      setYear((y) => y + 1);
    } else {
      setMonthIdx((m) => m + 1);
    }
  }

  if (loading && !summary) {
    return (
      <div className={styles.root}>
        <div className={styles.loading}>Loading journal…</div>
      </div>
    );
  }

  return (
    <div className={styles.root}>
      <div className={styles.headerRow}>
        <div>
          <div className={styles.eyebrow}>The trader's ledger</div>
          <h1 className={styles.title}>Journal</h1>
          <div className={styles.sub}>
            {monthName}
            {' · '}
            {summary?.metrics.trades ?? 0} trades
            {' · '}
            {(summary?.days ?? []).length} sessions traded
          </div>
        </div>
        <div className={styles.actions}>
          {STRATEGY_FILTERS.map((f) => (
            <span
              key={f.key}
              className={`${styles.chipFilter} ${strategy === f.key ? styles.active : ''}`}
              onClick={() => setStrategy(f.key)}
            >
              {f.label}
            </span>
          ))}
          <button
            className={styles.btnSecondary}
            onClick={() => apiPost('/api/journal/sync').then(() => window.location.reload())}
            title="Re-sync trades from strategy DBs"
          >
            Sync
          </button>
          <button
            className={styles.btnPrimary}
            onClick={() => alert('Manual trade entry — coming in Phase 2')}
          >
            + Log manual trade
          </button>
        </div>
      </div>

      {err && <div className={styles.error}>Failed to load: {err}</div>}

      {/* Metric strip */}
      <div className={styles.metrics}>
        <div className={styles.metric}>
          <div className={styles.metricLabel}>Month net P&amp;L</div>
          <div className={`${styles.metricValue} ${(summary?.metrics.pnl_net ?? 0) >= 0 ? styles.pos : styles.neg}`}>
            {formatPnl(summary?.metrics.pnl_net ?? 0)}
          </div>
          <div className={styles.metricSub}>
            Gross {formatPnl(summary?.metrics.pnl_gross ?? 0)}
          </div>
        </div>
        <div className={styles.metric}>
          <div className={styles.metricLabel}>Win rate</div>
          <div className={styles.metricValue}>{(summary?.metrics.win_rate ?? 0).toFixed(1)}%</div>
          <div className={styles.metricSub}>
            {summary?.metrics.wins ?? 0} wins · {summary?.metrics.losses ?? 0} losses
          </div>
        </div>
        <div className={styles.metric}>
          <div className={styles.metricLabel}>Profit factor</div>
          <div className={styles.metricValue}>
            {summary?.metrics.profit_factor != null ? summary.metrics.profit_factor.toFixed(2) : '—'}
          </div>
          <div className={styles.metricSub}>
            Avg win {formatPnl(summary?.metrics.avg_win ?? 0)} · loss {formatPnl(summary?.metrics.avg_loss ?? 0)}
          </div>
        </div>
        <div className={styles.metric}>
          <div className={styles.metricLabel}>Best day</div>
          <div className={`${styles.metricValue} ${styles.pos}`}>
            {summary?.metrics.best_day ? formatPnl(summary.metrics.best_day.pnl_net) : '—'}
          </div>
          <div className={styles.metricSub}>
            {summary?.metrics.best_day?.trade_date ?? ''}
            {summary?.metrics.best_day ? ` · ${summary.metrics.best_day.trades} trades` : ''}
          </div>
        </div>
        <div className={styles.metric}>
          <div className={styles.metricLabel}>Worst day</div>
          <div className={`${styles.metricValue} ${styles.neg}`}>
            {summary?.metrics.worst_day ? formatPnl(summary.metrics.worst_day.pnl_net) : '—'}
          </div>
          <div className={styles.metricSub}>
            {summary?.metrics.worst_day?.trade_date ?? ''}
            {summary?.metrics.worst_day ? ` · ${summary.metrics.worst_day.trades} trades` : ''}
          </div>
        </div>
      </div>

      {/* Body grid */}
      <div className={styles.bodyGrid}>
        {/* Calendar panel */}
        <section className={styles.panel}>
          <div className={styles.panelHead}>
            <div className={styles.panelTitle}>
              Calendar
              <span className={styles.panelTitleAccent}>· a month in P&amp;L</span>
            </div>
            <div className={styles.monthNav}>
              <button className={styles.iconBtn} onClick={handlePrevMonth} aria-label="Previous month">
                &#8249;
              </button>
              <span className={styles.calMonthName}>{monthName}</span>
              <button className={styles.iconBtn} onClick={handleNextMonth} aria-label="Next month">
                &#8250;
              </button>
            </div>
          </div>
          <div className={styles.cal}>
            <div className={styles.calWeekHeaders}>
              <div>Mon</div><div>Tue</div><div>Wed</div><div>Thu</div><div>Fri</div><div>Sat</div><div>Sun</div>
            </div>
            <div className={styles.calGrid}>
              {cells.map((cell, idx) => {
                if (!cell.inMonth) {
                  return (
                    <div key={idx} className={`${styles.calCell} ${styles.muted}`}>
                      <div className={styles.dayNum}>{cell.day}</div>
                    </div>
                  );
                }
                const data = cell.date ? dayMap[cell.date] : undefined;
                const pnl = data?.pnl_net ?? 0;
                const trades = data?.trades ?? 0;
                const tint = data ? pnlBucketClass(pnl, peak) : styles.pnlFlat;
                const isToday = cell.date === todayISO;
                return (
                  <div
                    key={idx}
                    className={`${styles.calCell} ${tint} ${isToday ? styles.today : ''}`}
                    onClick={() => cell.date && navigate(`/journal/day/${cell.date}`)}
                  >
                    <div className={styles.dayNum}>{cell.day}</div>
                    <div className={`${styles.dayPnl} ${pnl > 0 ? styles.pos : pnl < 0 ? styles.neg : ''}`}>
                      {data ? formatPnl(pnl) : '—'}
                    </div>
                    <div className={styles.dayTrades}>
                      {trades > 0 ? `${trades} trade${trades === 1 ? '' : 's'}${isToday ? ' · today' : ''}` : isToday ? 'today' : ''}
                    </div>
                  </div>
                );
              })}
            </div>

            <div className={styles.calLegend}>
              <span>Loss</span>
              <div className={styles.swatchRow}>
                <div className={`${styles.swatch} ${styles.pnlNeg5}`} />
                <div className={`${styles.swatch} ${styles.pnlNeg4}`} />
                <div className={`${styles.swatch} ${styles.pnlNeg3}`} />
                <div className={`${styles.swatch} ${styles.pnlNeg2}`} />
                <div className={`${styles.swatch} ${styles.pnlNeg1}`} />
              </div>
              <div className={`${styles.swatch} ${styles.pnlFlat}`} />
              <div className={styles.swatchRow}>
                <div className={`${styles.swatch} ${styles.pnlPos1}`} />
                <div className={`${styles.swatch} ${styles.pnlPos2}`} />
                <div className={`${styles.swatch} ${styles.pnlPos3}`} />
                <div className={`${styles.swatch} ${styles.pnlPos4}`} />
                <div className={`${styles.swatch} ${styles.pnlPos5}`} />
              </div>
              <span>Profit</span>
              <span style={{ marginLeft: 'auto' }}>click any day to drill in</span>
            </div>
          </div>
        </section>

        {/* Recent trades */}
        <section className={styles.panel}>
          <div className={styles.panelHead}>
            <div className={styles.panelTitle}>Recent trades</div>
            <a
              className={styles.chipFilter}
              onClick={() => navigate('/journal/insights')}
            >
              View insights
            </a>
          </div>
          <div className={styles.recentList}>
            {recent.length === 0 ? (
              <div className={styles.empty}>
                No trades yet. Run a sync, or place a live trade.
              </div>
            ) : (
              recent.map((t) => {
                const time = (t.entry_time || '').slice(11, 16) || (t.entry_time || '').slice(0, 10);
                const r = t.r_multiple;
                return (
                  <div
                    className={styles.recentRow}
                    key={t.id}
                    onClick={() => navigate(`/journal/trade/${t.id}`)}
                  >
                    <div className="time">{time}</div>
                    <div>
                      <div className={styles.sym}>
                        {t.instrument}
                        {t.mistake_flag ? <span className={styles.mistakeDot} title="Mistake-flagged" /> : null}
                      </div>
                      <div>
                        <span className={`${styles.stratChip} ${stratChipClass(t.strategy)}`}>
                          {shortStratLabel(t.strategy)}
                        </span>
                      </div>
                    </div>
                    <div>
                      <span className={`${styles.side} ${t.direction === 'LONG' ? styles.long : styles.short}`}>
                        {t.direction === 'LONG' ? 'Long' : 'Short'}
                      </span>
                    </div>
                    <div className={`${styles.pnlCell} ${(t.pnl_net ?? 0) >= 0 ? styles.pos : styles.neg}`}>
                      {formatPnl(t.pnl_net ?? 0)}
                    </div>
                    <div className={styles.rMult}>
                      {r != null ? `${r > 0 ? '+' : ''}${r.toFixed(2)}R` : `${formatInt(t.qty)}q`}
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </section>
      </div>

      <div className={styles.footnote}>
        <div className={styles.footnoteLabel}>From the journal</div>
        Auto-imported from <em>orb_trading.db</em>, <em>kc6_trading.db</em>, <em>nas_trading.db</em>
        and <em>strangle_trading.db</em>. Tag a trade, write a post-trade note, grade the process.
        The journal is a layer over execution data — it never re-records, it
        only enriches. (See <em>docs/Design/TRADING-JOURNAL-DESIGN.md</em>.)
      </div>
    </div>
  );
}
