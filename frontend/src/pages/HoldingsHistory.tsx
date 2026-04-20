import { useEffect, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import styles from './HoldingsHistory.module.css';
import { apiGet } from '../api/client';
import type {
  HoldingsSnapshot,
  HoldingsSnapshotSummary,
  HoldingsRecord,
} from '../api/types';
import Chip from '../components/Chip/Chip';
import { formatNumber, formatPct, formatPnl, pnlClass } from '../utils/format';

function formatRs(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return '—';
  const abs = Math.abs(n);
  const sign = n < 0 ? '−' : '';
  if (abs >= 1e7) return `${sign}Rs ${(abs / 1e7).toFixed(2)}cr`;
  if (abs >= 1e5) return `${sign}Rs ${(abs / 1e5).toFixed(2)}L`;
  return `${sign}Rs ${abs.toLocaleString('en-IN')}`;
}

function formatDayName(iso: string): string {
  const dt = new Date(iso + 'T00:00:00');
  return dt.toLocaleDateString('en-IN', {
    weekday: 'short',
    day: '2-digit',
    month: 'short',
    year: 'numeric',
  });
}

export default function HoldingsHistory() {
  const [list, setList] = useState<HoldingsSnapshotSummary[]>([]);
  const [snaps, setSnaps] = useState<Record<string, HoldingsSnapshot>>({});
  const [idx, setIdx] = useState(0); // 0 = newest
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  // Load snapshot list on mount
  useEffect(() => {
    let cancelled = false;
    apiGet<HoldingsSnapshotSummary[]>('/api/holdings/snapshots?limit=120')
      .then((l) => {
        if (cancelled) return;
        setList(l);
        setErr(null);
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

  // Lazy-load full snapshot for currently visible index
  const currentDate = list[idx]?.snap_date;
  useEffect(() => {
    if (!currentDate || snaps[currentDate]) return;
    apiGet<HoldingsSnapshot>(`/api/holdings/snapshot?date=${currentDate}`)
      .then((s) => setSnaps((prev) => ({ ...prev, [currentDate]: s })))
      .catch(() => {
        /* ignore */
      });
  }, [currentDate, snaps]);

  const currentSnap = currentDate ? snaps[currentDate] : undefined;

  // Swipe handling on mobile
  const swipeRef = useRef<{ x0: number; t0: number } | null>(null);
  const onTouchStart = (e: React.TouchEvent) => {
    swipeRef.current = { x0: e.touches[0].clientX, t0: Date.now() };
  };
  const onTouchEnd = (e: React.TouchEvent) => {
    if (!swipeRef.current) return;
    const dx = e.changedTouches[0].clientX - swipeRef.current.x0;
    const dt = Date.now() - swipeRef.current.t0;
    swipeRef.current = null;
    if (Math.abs(dx) < 40 || dt > 700) return;
    if (dx < 0) {
      // swipe left → go older (idx +)
      setIdx((i) => Math.min(list.length - 1, i + 1));
    } else {
      setIdx((i) => Math.max(0, i - 1));
    }
  };

  if (loading) return <div className={styles.root}>Loading history…</div>;
  if (err) return <div className={styles.root}>Error: {err}</div>;
  if (list.length === 0) {
    return (
      <div className={styles.root}>
        <div className="page-title">Holdings history</div>
        <div className={styles.emptyCard}>
          No snapshots stored yet. The scheduler captures one daily at 16:00 IST.
        </div>
        <div className={styles.backLink}>
          <Link to="/holdings">← Holdings</Link>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.root}>
      <div className={styles.topRow}>
        <div>
          <div className="page-title">Holdings history</div>
          <div className="page-subtitle">
            {list.length} daily snapshots · swipe or use arrows to navigate
          </div>
        </div>
        <Link to="/holdings" className={styles.backLink}>
          ← Today
        </Link>
      </div>

      {/* Date picker strip */}
      <div className={styles.dateStrip}>
        <button
          className={styles.dateNavBtn}
          onClick={() => setIdx((i) => Math.min(list.length - 1, i + 1))}
          disabled={idx >= list.length - 1}
          aria-label="older"
        >
          ←
        </button>
        <div className={styles.dateDropdownWrap}>
          <select
            className={styles.dateSelect}
            value={idx}
            onChange={(e) => setIdx(Number(e.target.value))}
          >
            {list.map((s, i) => (
              <option key={s.snap_date} value={i}>
                {formatDayName(s.snap_date)}
              </option>
            ))}
          </select>
        </div>
        <button
          className={styles.dateNavBtn}
          onClick={() => setIdx((i) => Math.max(0, i - 1))}
          disabled={idx === 0}
          aria-label="newer"
        >
          →
        </button>
      </div>

      {/* Swipeable snapshot card */}
      <div
        className={styles.snapshotCard}
        onTouchStart={onTouchStart}
        onTouchEnd={onTouchEnd}
      >
        {!currentSnap ? (
          <div className={styles.emptyCard}>Loading snapshot…</div>
        ) : (
          <SnapshotView snap={currentSnap} />
        )}
      </div>

      {/* Mini strip: all snapshots at-a-glance */}
      <section className={styles.miniSection}>
        <div className="section-title">All snapshots</div>
        <div className={styles.miniStrip}>
          {list.map((s, i) => (
            <button
              key={s.snap_date}
              onClick={() => setIdx(i)}
              className={`${styles.miniItem} ${i === idx ? styles.miniActive : ''}`}
            >
              <div className={styles.miniDate}>
                {new Date(s.snap_date + 'T00:00:00').toLocaleDateString('en-IN', {
                  day: '2-digit',
                  month: 'short',
                })}
              </div>
              <div className={`${styles.miniPnl} ${pnlClass(s.day_pnl)}`}>
                {formatPnl(s.day_pnl)}
              </div>
            </button>
          ))}
        </div>
      </section>
    </div>
  );
}

function SnapshotView({ snap }: { snap: HoldingsSnapshot }) {
  const { summary, movers_today, extremes } = snap;
  return (
    <>
      <div className={styles.snapHead}>
        <div>
          <div className={styles.snapDay}>{formatDayName(snap.snap_date)}</div>
          <div className={styles.snapTime}>
            captured {new Date(snap.generated_at).toLocaleTimeString('en-IN')}
          </div>
        </div>
        <div className={styles.snapHeadMetrics}>
          <div className={styles.snapMetric}>
            <div className={styles.snapMetricLabel}>Current</div>
            <div className={styles.snapMetricVal}>{formatRs(summary.current)}</div>
          </div>
          <div className={styles.snapMetric}>
            <div className={styles.snapMetricLabel}>Day P&amp;L</div>
            <div className={`${styles.snapMetricVal} ${pnlClass(summary.day_pnl)}`}>
              {formatPnl(summary.day_pnl)}
            </div>
          </div>
          <div className={styles.snapMetric}>
            <div className={styles.snapMetricLabel}>Total P&amp;L</div>
            <div className={`${styles.snapMetricVal} ${pnlClass(summary.total_pnl)}`}>
              {formatPnl(summary.total_pnl)}
            </div>
          </div>
        </div>
      </div>

      <div className={styles.snapGrid}>
        <div>
          <div className={styles.subHead}>Movers</div>
          <MoverList title="Gainers" rows={movers_today.gainers} kind="pos" />
          <MoverList title="Losers" rows={movers_today.losers} kind="neg" />
        </div>
        <div>
          <div className={styles.subHead}>52-week extremes</div>
          <ExtremesMini rows={extremes.high} side="hi" />
          <ExtremesMini rows={extremes.low} side="lo" />
          {extremes.high.length === 0 && extremes.low.length === 0 ? (
            <div className={styles.noneText}>No extremes on this day</div>
          ) : null}
        </div>
      </div>
    </>
  );
}

function MoverList({ title, rows, kind }: { title: string; rows: HoldingsRecord[]; kind: 'pos' | 'neg' }) {
  if (rows.length === 0) return null;
  return (
    <div className={styles.moverBlock}>
      <div className={styles.moverBlockLabel}>{title}</div>
      {rows.map((r) => (
        <div key={r.tradingsymbol} className={styles.moverRow}>
          <span className={styles.moverRowSym}>{r.tradingsymbol}</span>
          <span className={`${styles.moverRowPct} ${kind === 'pos' ? styles.pnlPos : styles.pnlNeg}`}>
            {r.day_pct > 0 ? '+' : ''}
            {r.day_pct.toFixed(2)}%
          </span>
          <span className={`${styles.moverRowPnl} ${kind === 'pos' ? styles.pnlPos : styles.pnlNeg}`}>
            {formatPnl(r.day_pnl_inr)}
          </span>
        </div>
      ))}
    </div>
  );
}

function ExtremesMini({ rows, side }: { rows: HoldingsRecord[]; side: 'hi' | 'lo' }) {
  if (rows.length === 0) return null;
  return (
    <div className={styles.extremesBlock}>
      <div className={styles.moverBlockLabel}>
        {side === 'hi' ? 'At / near 52wk high' : 'At / near 52wk low'}
      </div>
      {rows.map((r) => (
        <div key={r.tradingsymbol} className={styles.extremeRow}>
          <span className={styles.moverRowSym}>{r.tradingsymbol}</span>
          <span className={styles.extremeTag}>{(r.tag ?? '').replace(/_/g, ' ')}</span>
          <span className={styles.extremeLtp}>{formatNumber(r.ltp, 2)}</span>
        </div>
      ))}
    </div>
  );
}
