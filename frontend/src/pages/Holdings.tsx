import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import styles from './Holdings.module.css';
import { apiGet } from '../api/client';
import type {
  HoldingsDigest,
  HoldingsEvent,
  HoldingsRecord,
} from '../api/types';
import Chip from '../components/Chip/Chip';
import { formatInt, formatNumber, formatPct, formatPnl, pnlClass } from '../utils/format';

const EXTREME_LABELS: Record<string, string> = {
  at_ath: 'at all-time high',
  at_52h: 'at 52wk high',
  near_52h: 'near 52wk high',
  at_52l: 'at 52wk low',
  near_52l: 'near 52wk low',
};

const EVENT_TAG_LABEL: Record<string, string> = {
  results: 'results',
  dividend: 'dividend',
  split: 'split',
  bonus: 'bonus',
  buyback: 'buyback',
  meeting: 'meeting',
};

function formatRs(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return '—';
  const abs = Math.abs(n);
  const sign = n < 0 ? '−' : '';
  if (abs >= 1e7) return `${sign}Rs ${(abs / 1e7).toFixed(2)}cr`;
  if (abs >= 1e5) return `${sign}Rs ${(abs / 1e5).toFixed(2)}L`;
  return `${sign}Rs ${formatInt(abs)}`;
}

function formatShortDate(iso: string): { day: string; month: string } {
  const dt = new Date(iso + 'T00:00:00');
  return {
    day: dt.getDate().toString().padStart(2, '0'),
    month: dt.toLocaleDateString('en-IN', { month: 'short' }),
  };
}

function daysUntil(iso: string): number {
  const dt = new Date(iso + 'T00:00:00');
  const now = new Date();
  now.setHours(0, 0, 0, 0);
  return Math.round((dt.getTime() - now.getTime()) / 86400000);
}

export default function Holdings() {
  const [data, setData] = useState<HoldingsDigest | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    const load = () => {
      apiGet<HoldingsDigest>('/api/holdings/digest')
        .then((d) => {
          if (cancelled) return;
          setData(d);
          setErr(null);
        })
        .catch((e) => {
          if (cancelled) return;
          setErr(e instanceof Error ? e.message : 'Failed to load');
        })
        .finally(() => {
          if (!cancelled) setLoading(false);
        });
    };
    load();
    const id = setInterval(load, 30_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  if (loading && !data) {
    return <div className={styles.root}>Loading holdings…</div>;
  }
  if (err && !data) {
    return <div className={styles.root}>Error: {err}</div>;
  }
  if (!data) return null;

  const { summary, movers_today, movers_weekly, extremes, events, next_event } = data;

  return (
    <div className={styles.root}>
      <div className="page-title">Holdings</div>
      <div className="page-subtitle">
        {summary.count} stocks · signals Kite doesn't flag — movers, extremes, events
      </div>

      {/* Hero (B) */}
      <HeroSummary summary={summary} next={next_event} />

      {/* 52-week extremes (A, with rail bars) */}
      {(extremes.high.length > 0 || extremes.low.length > 0) ? (
        <section className={styles.section}>
          <div className={styles.sectionHead}>
            <div className="section-title">52-week extremes</div>
            <Chip>{extremes.high.length + extremes.low.length} signals</Chip>
          </div>
          <div className={styles.extremesCard}>
            {extremes.high.length > 0 ? (
              <ExtremesGroup title="At / near 52wk high" rows={extremes.high} side="hi" />
            ) : null}
            {extremes.low.length > 0 ? (
              <ExtremesGroup title="At / near 52wk low" rows={extremes.low} side="lo" />
            ) : null}
          </div>
        </section>
      ) : null}

      {/* Today's movers (B) */}
      <section className={styles.section}>
        <div className={styles.sectionHead}>
          <div className="section-title">Today's movers</div>
        </div>
        <MoversGrid gainers={movers_today.gainers} losers={movers_today.losers} metricKey="day_pct" pnlKey="day_pnl_inr" />
      </section>

      {/* Weekly movers (B) */}
      {(movers_weekly.gainers.length > 0 || movers_weekly.losers.length > 0) ? (
        <section className={styles.section}>
          <div className={styles.sectionHead}>
            <div className="section-title">Weekly movers · last 5 trading days</div>
          </div>
          <MoversGrid gainers={movers_weekly.gainers} losers={movers_weekly.losers} metricKey="change_5d_pct" />
        </section>
      ) : null}

      {/* Upcoming events (B) */}
      <section className={styles.section}>
        <div className={styles.sectionHead}>
          <div className="section-title">Upcoming events · next 30 days</div>
          <Chip>{events.length} events</Chip>
        </div>
        {events.length === 0 ? (
          <div className={styles.emptyCard}>
            No upcoming corporate actions recorded for these symbols.
            NSE feed refreshes daily at 07:00 IST.
          </div>
        ) : (
          <div className={styles.eventsTimeline}>
            {events.map((ev, i) => (
              <EventRow key={i} ev={ev} />
            ))}
          </div>
        )}
      </section>

      {/* History footer */}
      <div className={styles.historyFooter}>
        <Link to="/holdings/history" className={styles.historyLink}>
          View holdings history →
        </Link>
      </div>
    </div>
  );
}

function HeroSummary({ summary, next }: { summary: HoldingsDigest['summary']; next: HoldingsEvent | null }) {
  return (
    <div className={styles.hero}>
      <div className={styles.heroBlock}>
        <div className={styles.heroLabel}>Current value</div>
        <div className={styles.heroCurrent}>{formatRs(summary.current)}</div>
        <div className={styles.heroSub}>
          Invested {formatRs(summary.invested)} · unrealized{' '}
          <span className={pnlClass(summary.total_pnl)}>
            {formatPnl(summary.total_pnl)} ({formatPct(summary.total_pct, 2)})
          </span>
        </div>
      </div>
      <div className={styles.heroBlock}>
        <div className={styles.heroLabel}>Day P&amp;L</div>
        <div className={`${styles.heroValue} ${pnlClass(summary.day_pnl)}`}>
          {formatPnl(summary.day_pnl)}
        </div>
        <div className={styles.heroSub}>{formatPct(summary.day_pct, 2)}</div>
      </div>
      <div className={styles.heroBlock}>
        <div className={styles.heroLabel}>Next event</div>
        {next ? (
          <>
            <div className={styles.heroValue}>
              {next.tradingsymbol} · {EVENT_TAG_LABEL[next.event_type] ?? next.event_type}
            </div>
            <div className={styles.heroSub}>
              in {daysUntil(next.event_date)} days · {next.event_date}
            </div>
          </>
        ) : (
          <>
            <div className={styles.heroValueMute}>none</div>
            <div className={styles.heroSub}>within next 30 days</div>
          </>
        )}
      </div>
    </div>
  );
}

function ExtremesGroup({ title, rows, side }: { title: string; rows: HoldingsRecord[]; side: 'hi' | 'lo' }) {
  return (
    <div className={styles.extremesGroup}>
      <div className={styles.extremesLabel}>
        <span className={`${styles.extremesBadge} ${side === 'hi' ? styles.badgeHi : styles.badgeLo}`}>
          {title}
        </span>
        <span className={styles.extremesCount}>{rows.length}</span>
      </div>
      <div className={styles.extremesRows}>
        {rows.map((r) => (
          <ExtremeRow key={r.tradingsymbol} r={r} />
        ))}
      </div>
    </div>
  );
}

function ExtremeRow({ r }: { r: HoldingsRecord }) {
  const low = r.week52_low ?? 0;
  const high = r.week52_high ?? 0;
  // Marker position 0..100
  const range = high > low ? high - low : 1;
  const pos = high > low ? Math.max(0, Math.min(100, ((r.ltp - low) / range) * 100)) : 50;

  const atAth = r.tag === 'at_ath';
  const atHi = r.tag === 'at_52h' || atAth;
  const atLo = r.tag === 'at_52l';
  const markerClass = atAth
    ? styles.markerAth
    : atHi
    ? styles.markerHi
    : atLo
    ? styles.markerLo
    : styles.markerMid;

  const distLabel = r.tag?.includes('_52l')
    ? `+${(r.pct_from_52l ?? 0).toFixed(1)}% above low`
    : atAth
    ? `at ATH`
    : `${(r.pct_from_52h ?? 0).toFixed(1)}% from 52wH`;

  return (
    <div className={styles.extremeRow}>
      <div className={styles.extremeSym}>{r.tradingsymbol}</div>
      <div className={styles.extremeTagCol}>
        <span className={`${styles.tag} ${atAth ? styles.tagAth : atHi ? styles.tagHi : styles.tagLo}`}>
          {EXTREME_LABELS[r.tag ?? ''] ?? r.tag ?? ''}
        </span>
      </div>
      <div className={styles.extremeRail}>
        <div className={styles.railBar} />
        <div className={`${styles.railMarker} ${markerClass}`} style={{ left: `${pos}%` }} />
        <div className={styles.railLo}>{formatNumber(low, 2)}</div>
        <div className={styles.railHi}>{formatNumber(high, 2)}</div>
      </div>
      <div className={styles.extremeLtp}>{formatNumber(r.ltp, 2)}</div>
      <div className={styles.extremeDist}>{distLabel}</div>
    </div>
  );
}

function MoversGrid({
  gainers,
  losers,
  metricKey,
  pnlKey,
}: {
  gainers: HoldingsRecord[];
  losers: HoldingsRecord[];
  metricKey: 'day_pct' | 'change_5d_pct';
  pnlKey?: 'day_pnl_inr';
}) {
  // Scale bars against the largest absolute move within the group so they
  // fill the card proportionally.
  const maxPos = Math.max(0.01, ...gainers.map((r) => Math.abs((r as any)[metricKey] ?? 0)));
  const maxNeg = Math.max(0.01, ...losers.map((r) => Math.abs((r as any)[metricKey] ?? 0)));

  return (
    <div className={styles.moversGrid}>
      <MoverCard title="Top gainers" rows={gainers} metricKey={metricKey} pnlKey={pnlKey} max={maxPos} kind="pos" />
      <MoverCard title="Top losers" rows={losers} metricKey={metricKey} pnlKey={pnlKey} max={maxNeg} kind="neg" />
    </div>
  );
}

function MoverCard({
  title,
  rows,
  metricKey,
  pnlKey,
  max,
  kind,
}: {
  title: string;
  rows: HoldingsRecord[];
  metricKey: 'day_pct' | 'change_5d_pct';
  pnlKey?: 'day_pnl_inr';
  max: number;
  kind: 'pos' | 'neg';
}) {
  return (
    <div className={styles.moverCard}>
      <div className={styles.moverCardTitle}>{title}</div>
      {rows.length === 0 ? (
        <div className={styles.moverEmpty}>None</div>
      ) : (
        rows.map((r) => {
          const pct = ((r as any)[metricKey] ?? 0) as number;
          const width = Math.min(100, (Math.abs(pct) / max) * 100);
          const pnl = pnlKey ? ((r as any)[pnlKey] as number | undefined) : undefined;
          return (
            <div key={r.tradingsymbol} className={styles.moverItem}>
              <div className={styles.moverSym}>{r.tradingsymbol}</div>
              <div className={styles.moverBarWrap}>
                <div
                  className={`${styles.moverBar} ${kind === 'pos' ? styles.moverBarPos : styles.moverBarNeg}`}
                  style={{ width: `${width}%` }}
                />
              </div>
              <div className={styles.moverRight}>
                <div className={`${styles.moverPct} ${kind === 'pos' ? styles.pnlPos : styles.pnlNeg}`}>
                  {pct > 0 ? '+' : ''}
                  {pct.toFixed(2)}%
                </div>
                {pnl !== undefined ? (
                  <div className={styles.moverInr}>{formatPnl(pnl)}</div>
                ) : null}
              </div>
            </div>
          );
        })
      )}
    </div>
  );
}

function EventRow({ ev }: { ev: HoldingsEvent }) {
  const d = formatShortDate(ev.event_date);
  const tag = ev.event_type ?? 'meeting';
  return (
    <div className={styles.timelineItem}>
      <div className={styles.timelineDate}>
        <div className={styles.timelineDay}>{d.day}</div>
        <div className={styles.timelineMonth}>{d.month}</div>
      </div>
      <div className={styles.timelineBody}>
        <div className={styles.timelineSym}>{ev.tradingsymbol}</div>
        <div className={styles.timelineKind}>
          <span className={`${styles.timelineTag} ${styles['tag_' + tag]}`}>
            {EVENT_TAG_LABEL[tag] ?? tag}
          </span>
          {ev.detail || ev.purpose}
        </div>
        {ev.record_date ? (
          <div className={styles.timelineDetail}>record {ev.record_date}</div>
        ) : null}
      </div>
    </div>
  );
}
