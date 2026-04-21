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
  near_ath: 'near all-time high',
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

  const { summary, movers_today, movers_weekly, extremes, events, next_event, holdings } = data;

  return (
    <div className={styles.root}>
      <div className="page-title">Holdings</div>
      <div className="page-subtitle">
        {summary.count} stocks · signals Kite doesn't flag — movers, extremes, events
      </div>

      {/* Hero (B) */}
      <HeroSummary summary={summary} next={next_event} />

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

      {/* Price extremes — stocks at/near ATH or 52-week high/low */}
      {(extremes.high.length > 0 || extremes.low.length > 0) ? (
        <section className={styles.section}>
          <div className={styles.sectionHead}>
            <div className="section-title">Price extremes</div>
            <Chip>{extremes.high.length + extremes.low.length} signals</Chip>
          </div>
          <div className={styles.extremesCard}>
            {extremes.high.length > 0 ? (
              <ExtremesGroup title="At / near highs" rows={extremes.high} side="hi" />
            ) : null}
            {extremes.low.length > 0 ? (
              <ExtremesGroup title="At / near lows" rows={extremes.low} side="lo" />
            ) : null}
          </div>
        </section>
      ) : null}

      {/* Return multiples: bucket all holdings by unrealized P&L % */}
      {holdings && holdings.length > 0 ? (
        <ReturnMultiplesSection holdings={holdings} />
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
  const tag = r.tag ?? '';
  const atAth = tag === 'at_ath';
  const nearAth = tag === 'near_ath';
  const athRow = atAth || nearAth;
  const atHi = tag === 'at_52h' || tag === 'near_52h';
  const atLo = tag === 'at_52l' || tag === 'near_52l';

  const tagClass = atAth
    ? styles.tagAth
    : nearAth
    ? styles.tagNearAth
    : tag === 'at_52h'
    ? styles.tagHi
    : tag === 'near_52h'
    ? styles.tagNearHi
    : tag === 'at_52l'
    ? styles.tagLo
    : tag === 'near_52l'
    ? styles.tagNearLo
    : styles.tagNearHi;

  const distLabel = (() => {
    if (atAth) return 'at ATH';
    if (nearAth) {
      const p = r.pct_from_ath ?? 0;
      return `${p >= 0 ? '+' : ''}${p.toFixed(1)}% from ATH`;
    }
    if (atLo) {
      const p = r.pct_from_52l ?? 0;
      return `+${p.toFixed(1)}% from 52wL`;
    }
    const p = r.pct_from_52h ?? 0;
    return `${p >= 0 ? '+' : ''}${p.toFixed(1)}% from 52wH`;
  })();

  const spark = r.sparkline ?? [];

  return (
    <div className={styles.extremeRow}>
      <div className={styles.extremeSym}>{r.tradingsymbol}</div>
      <div className={styles.extremeTagCol}>
        <span className={`${styles.tag} ${tagClass}`}>
          {EXTREME_LABELS[tag] ?? tag}
        </span>
      </div>
      <Sparkline
        values={spark}
        ltp={r.ltp}
        high={r.week52_high ?? null}
        low={r.week52_low ?? null}
        ath={r.all_time_high ?? null}
        highlight={athRow ? 'ath' : atHi ? 'hi' : atLo ? 'lo' : 'mid'}
      />
      <div className={styles.extremeLtp}>{formatNumber(r.ltp, 2)}</div>
      <div className={styles.extremeDist}>{distLabel}</div>
    </div>
  );
}

function Sparkline({
  values,
  ltp,
  high,
  low,
  ath,
  highlight,
}: {
  values: number[];
  ltp: number;
  high: number | null;
  low: number | null;
  ath: number | null;
  highlight: 'ath' | 'hi' | 'lo' | 'mid';
}) {
  // Guard: need at least 2 points to draw a line. Fall back to a flat rail.
  if (!values || values.length < 2) {
    return <div className={styles.sparkEmpty}>—</div>;
  }
  const W = 300;
  const H = 36;
  const pad = 3;
  // Clamp wide y-range so the line uses most of the vertical space.
  const dataMin = Math.min(...values, ltp);
  const dataMax = Math.max(...values, ltp);
  const range = dataMax > dataMin ? dataMax - dataMin : 1;
  const x = (i: number) => pad + (i / (values.length - 1)) * (W - 2 * pad);
  const y = (v: number) => H - pad - ((v - dataMin) / range) * (H - 2 * pad);

  const pathD =
    `M ${x(0)} ${y(values[0])} ` +
    values.slice(1).map((v, i) => `L ${x(i + 1)} ${y(v)}`).join(' ');
  const fillD = `${pathD} L ${x(values.length - 1)} ${H} L ${x(0)} ${H} Z`;

  // 20-day SMA overlay — adds depth without height
  const sma: Array<[number, number]> = [];
  const win = 20;
  for (let i = win - 1; i < values.length; i++) {
    let s = 0;
    for (let j = i - win + 1; j <= i; j++) s += values[j];
    sma.push([i, s / win]);
  }
  const smaD = sma.length > 1
    ? `M ${x(sma[0][0])} ${y(sma[0][1])} ` +
      sma.slice(1).map(([i, v]) => `L ${x(i)} ${y(v)}`).join(' ')
    : '';

  const isUp = values[values.length - 1] >= values[0];
  const strokeClass = isUp ? styles.sparkStrokeUp : styles.sparkStrokeDown;
  const fillId = `sf-${Math.abs(Math.round(values[0] * 1000))}-${Math.round(ltp * 100)}`;

  const maxI = values.indexOf(dataMax);
  const minI = values.indexOf(dataMin);

  const nowMarkerColor =
    highlight === 'ath'
      ? '#B45309'
      : highlight === 'hi'
      ? 'var(--accent-pos)'
      : highlight === 'lo'
      ? 'var(--accent-neg)'
      : 'var(--ink)';

  const athY = ath != null && ath >= dataMin && ath <= dataMax ? y(ath) : null;
  const hiY = high != null && high >= dataMin && high <= dataMax && high !== ath ? y(high) : null;
  const loY = low != null && low >= dataMin && low <= dataMax ? y(low) : null;

  return (
    <div className={styles.sparkWrap}>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} preserveAspectRatio="none">
        <defs>
          <linearGradient id={fillId} x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor={isUp ? 'var(--accent-pos)' : 'var(--accent-neg)'} stopOpacity="0.28" />
            <stop offset="100%" stopColor={isUp ? 'var(--accent-pos)' : 'var(--accent-neg)'} stopOpacity="0.02" />
          </linearGradient>
        </defs>
        {athY !== null ? (
          <line x1="0" x2={W} y1={athY} y2={athY} className={styles.sparkAthLine} strokeDasharray="3 3" />
        ) : null}
        {hiY !== null ? (
          <line x1="0" x2={W} y1={hiY} y2={hiY} className={styles.sparkHiLine} strokeDasharray="2 3" />
        ) : null}
        {loY !== null ? (
          <line x1="0" x2={W} y1={loY} y2={loY} className={styles.sparkLoLine} strokeDasharray="2 3" />
        ) : null}
        <path d={fillD} fill={`url(#${fillId})`} />
        {smaD ? <path d={smaD} className={styles.sparkSma} /> : null}
        <path d={pathD} className={`${styles.sparkPath} ${strokeClass}`} />
        <circle cx={x(maxI)} cy={y(values[maxI])} r="2" className={styles.sparkMarkerHi} />
        <circle cx={x(minI)} cy={y(values[minI])} r="2" className={styles.sparkMarkerLo} />
        <circle
          cx={x(values.length - 1)}
          cy={y(ltp)}
          r="3.5"
          fill={nowMarkerColor}
          stroke="var(--surface)"
          strokeWidth="1.5"
        />
      </svg>
    </div>
  );
}

// ========= Return multiples bucket ==========

type MultiplesBucket = { label: string; min: number; max: number | null; rows: HoldingsRecord[] };

function bucketizeByMultiple(holdings: HoldingsRecord[]): MultiplesBucket[] {
  const defs: Array<Omit<MultiplesBucket, 'rows'>> = [
    { label: '+400%', min: 400, max: null },
    { label: '+300%', min: 300, max: 400 },
    { label: '+200%', min: 200, max: 300 },
    { label: '+100%', min: 100, max: 200 },
    { label: '+50%', min: 50, max: 100 },
    { label: '+0%', min: 0, max: 50 },
    { label: 'underwater', min: -Infinity, max: 0 },
  ];
  const buckets: MultiplesBucket[] = defs.map((d) => ({ ...d, rows: [] }));
  for (const h of holdings) {
    const p = h.total_pnl_pct ?? 0;
    for (const b of buckets) {
      if (p >= b.min && (b.max === null || p < b.max)) {
        b.rows.push(h);
        break;
      }
    }
  }
  return buckets;
}

function ReturnMultiplesSection({ holdings }: { holdings: HoldingsRecord[] }) {
  const allBuckets = bucketizeByMultiple(holdings);
  // Drop empty buckets at the top — start from the first bucket that has stocks.
  const firstNonEmpty = allBuckets.findIndex((b) => b.rows.length > 0);
  const buckets = firstNonEmpty === -1 ? allBuckets : allBuckets.slice(firstNonEmpty);
  const maxCount = Math.max(1, ...buckets.map((b) => b.rows.length));
  const totalCurrent = holdings.reduce((s, h) => s + (h.current ?? 0), 0);
  return (
    <section className={styles.section}>
      <div className={styles.sectionHead}>
        <div className="section-title">Return multiples</div>
        <Chip>{holdings.length} positions</Chip>
      </div>
      <div className={styles.bucketsCard}>
        {buckets.map((b) => {
          const pctWidth = (b.rows.length / maxCount) * 100;
          const hot = b.min >= 200;
          const underwater = b.max === 0;
          const invested = b.rows.reduce((s, r) => s + (r.invested ?? 0), 0);
          const current = b.rows.reduce((s, r) => s + (r.current ?? 0), 0);
          const shareOfFund = totalCurrent > 0 ? (current / totalCurrent) * 100 : 0;
          const donutTone: 'hot' | 'mid' | 'cold' = hot ? 'hot' : underwater ? 'cold' : 'mid';
          return (
            <div key={b.label} className={styles.bucketRow}>
              <div className={styles.bucketLabel}>{b.label}</div>
              <div className={styles.bucketBar}>
                <div
                  className={`${styles.bucketBarFill} ${hot ? styles.bucketBarFillHot : underwater ? styles.bucketBarFillCold : styles.bucketBarFillMid}`}
                  style={{ width: `${pctWidth}%` }}
                />
                <div className={styles.bucketBarCount}>{b.rows.length > 0 ? b.rows.length : '—'}</div>
              </div>
              <div className={styles.bucketChips}>
                {b.rows
                  .slice()
                  .sort((a, c) => (c.total_pnl_pct ?? 0) - (a.total_pnl_pct ?? 0))
                  .slice(0, 6)
                  .map((r) => (
                    <span key={r.tradingsymbol} className={styles.bucketChip}>
                      {r.tradingsymbol}
                      <span className={styles.bucketChipPct}>{formatPct(r.total_pnl_pct, 0)}</span>
                    </span>
                  ))}
                {b.rows.length > 6 ? (
                  <span className={styles.bucketChipMore}>+{b.rows.length - 6}</span>
                ) : null}
              </div>
              <div className={styles.bucketFund}>
                <BucketDonut pct={shareOfFund} tone={donutTone} />
                <div className={styles.bucketFundText}>
                  <div className={styles.bucketFundNow}>{formatRs(current)}</div>
                  <div className={styles.bucketFundIn}>from {formatRs(invested)}</div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function BucketDonut({ pct, tone }: { pct: number; tone: 'hot' | 'mid' | 'cold' }) {
  const r = 16;
  const circ = 2 * Math.PI * r;
  const filled = Math.max(0, Math.min(100, pct));
  const dash = (filled / 100) * circ;
  const stroke =
    tone === 'hot' ? '#B45309' : tone === 'cold' ? 'var(--accent-neg)' : 'var(--accent-pos)';
  return (
    <svg className={styles.bucketDonut} viewBox="0 0 40 40" width="36" height="36">
      <circle cx="20" cy="20" r={r} className={styles.bucketDonutTrack} />
      <circle
        cx="20"
        cy="20"
        r={r}
        fill="none"
        stroke={stroke}
        strokeWidth="4"
        strokeLinecap="round"
        strokeDasharray={`${dash} ${circ}`}
        transform="rotate(-90 20 20)"
      />
      <text x="20" y="24" className={styles.bucketDonutLabel}>
        {filled.toFixed(0)}%
      </text>
    </svg>
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
