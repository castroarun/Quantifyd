/**
 * BookActivity — one line telling you whether THIS book has actually traded.
 *
 * Every book page shows today and hides its own record, so a book that has not
 * traded in three months looks the same as one that simply had a quiet session.
 * This strip closes that gap: last trade, how long it has been quiet, trades in
 * the last 30 days, running total, win rate, and the shape of the curve.
 *
 * Reads /api/books/liveness (a read-only projection over each book's own trade
 * table). One shared fetch serves every mounted instance. If the endpoint is
 * missing the strip renders nothing rather than breaking the page.
 */

import { useEffect, useState } from 'react';
import styles from './BookActivity.module.css';
import { formatPnl, pnlClass } from '../../utils/format';

export interface BookLivenessRecord {
  trades: number;
  last_trade: string | null;
  days_idle: number | null;
  trades_30d: number;
  net_total: number | null;
  net_30d: number | null;
  win_rate: number | null;
  series: Array<{ d: string; c: number }>;
  /** Positions held right now — a book with no exits yet is still working. */
  open_positions?: number;
  last_entry?: string | null;
  /** Days since the last entry OR exit, whichever is later. */
  days_since_activity?: number | null;
}

type Books = Record<string, BookLivenessRecord>;

// Module-level cache: many pages can mount this without N requests.
let cache: { at: number; data: Books } | null = null;
let inflight: Promise<Books> | null = null;
const TTL_MS = 120_000;

function fetchBooks(): Promise<Books> {
  const now = Date.now();
  if (cache && now - cache.at < TTL_MS) return Promise.resolve(cache.data);
  if (inflight) return inflight;
  inflight = fetch('/api/books/liveness', { cache: 'no-store' })
    .then((r) => (r.ok ? r.json() : null))
    .then((d) => {
      const books: Books = d?.books ?? {};
      cache = { at: Date.now(), data: books };
      return books;
    })
    .catch(() => ({} as Books))
    .finally(() => {
      inflight = null;
    });
  return inflight;
}

function Spark({ series }: { series: Array<{ d: string; c: number }> }) {
  if (!series || series.length < 2) return null;
  const vals = series.map((p) => p.c);
  const min = Math.min(...vals, 0);
  const max = Math.max(...vals, 0);
  const span = max - min || 1;
  const w = 120;
  const h = 26;
  const pts = series.map((p, i) => {
    const x = (i / (series.length - 1)) * w;
    const y = h - ((p.c - min) / span) * h;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  const zeroY = h - ((0 - min) / span) * h;
  const last = vals[vals.length - 1];
  return (
    <svg className={styles.spark} width={w} height={h} viewBox={`0 0 ${w} ${h}`} aria-hidden="true">
      <line x1="0" y1={zeroY} x2={w} y2={zeroY} className={styles.zero} />
      <polyline points={pts.join(' ')} className={last >= 0 ? styles.pos : styles.neg} />
    </svg>
  );
}

export default function BookActivity({ bookId }: { bookId: string }) {
  const [rec, setRec] = useState<BookLivenessRecord | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchBooks().then((b) => {
      if (!cancelled) setRec(b[bookId] ?? null);
    });
    return () => {
      cancelled = true;
    };
  }, [bookId]);

  if (!rec) return null;

  if (!rec.trades) {
    return (
      <div className={styles.strip}>
        <span className={styles.label}>Book record</span>
        <span className={styles.quiet}>
          {(rec?.open_positions ?? 0) > 0
            ? `holding ${rec?.open_positions} — nothing closed yet`
            : 'no trades recorded yet'}
        </span>
      </div>
    );
  }

  const d = rec.days_since_activity ?? rec.days_idle;
  const idleLabel = d === 0 ? 'today' : d === 1 ? 'yesterday' : `${d}d ago`;
  const idleCls = d != null && d >= 30 ? styles.stale : d != null && d >= 7 ? styles.warn : '';

  return (
    <div className={styles.strip}>
      <span className={styles.label}>Book record</span>
      <span className={styles.item}>
        <b>{rec.trades}</b> trades
      </span>
      <span className={styles.item}>
        last <b className={idleCls}>{idleLabel}</b>
        {rec.last_trade ? <span className={styles.dim}> ({rec.last_trade})</span> : null}
      </span>
      <span className={styles.item}>
        <b>{rec.trades_30d}</b> in 30d
      </span>
      {(rec.open_positions ?? 0) > 0 && (
        <span className={styles.item}>
          holding <b>{rec.open_positions}</b>
        </span>
      )}
      {rec.net_total != null && (
        <span className={styles.item}>
          <span title="Realised P&L on closed trades. Paper books book fills at the "
                       + "signal price, so this is gross of brokerage, STT and slippage.">
            P&amp;L</span>{' '}
          <b className={pnlClass(rec.net_total)}>{formatPnl(rec.net_total)}</b>
        </span>
      )}
      {rec.win_rate != null && (
        <span className={styles.item}>
          <b>{rec.win_rate.toFixed(0)}%</b> wins
        </span>
      )}
      <Spark series={rec.series} />
    </div>
  );
}
