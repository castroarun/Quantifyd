/**
 * Day-by-day trade history for a book.
 *
 * The per-book pages showed only today, so a book with months of trades behind
 * it read as a book that had done nothing. This is the history: one row per
 * trading day with that day's realised P&L, the day's record, and a running
 * cumulative — click a day to see the trades that made it.
 *
 * Reads static/app/book_daily.json (built by scripts/build_book_daily.py), so
 * it needs no backend restart to appear.
 */
import { useEffect, useMemo, useState } from 'react';
import styles from './DailyPerformance.module.css';

export interface DayRow {
  date: string;
  trades: number;
  wins: number;
  losses: number;
  pnl: number;
  cum: number;
  mode: string;
  rows: Array<Record<string, unknown>>;
}

interface BookFeed {
  days: DayRow[];
  summary: {
    days: number; trades: number; wins: number; win_rate: number | null;
    net: number; best_day: number | null; worst_day: number | null;
    green_days: number; first: string | null; last: string | null;
  } | null;
  error?: string;
}

interface Feed {
  generated_at: string;
  books: Record<string, BookFeed>;
}

const inr = (v: number | null | undefined) => {
  if (v == null) return '—';
  const s = Math.abs(v) >= 1000
    ? Math.round(Math.abs(v)).toLocaleString('en-IN')
    : Math.abs(v).toFixed(0);
  return `${v < 0 ? '−' : '+'}₹${s}`;
};

const day = (d: string) => {
  const dt = new Date(d + 'T00:00:00');
  if (Number.isNaN(dt.getTime())) return d;
  return dt.toLocaleDateString('en-IN', { day: '2-digit', month: 'short' });
};

const weekday = (d: string) => {
  const dt = new Date(d + 'T00:00:00');
  return Number.isNaN(dt.getTime()) ? '' : dt.toLocaleDateString('en-IN', { weekday: 'short' });
};

const num = (v: unknown) =>
  typeof v === 'number' ? (Number.isInteger(v) ? String(v) : v.toFixed(2)) : String(v ?? '—');

const time = (v: unknown) => {
  const s = String(v ?? '');
  const m = s.match(/(\d{2}:\d{2})/);
  return m ? m[1] : '—';
};

/** Columns to show inside an expanded day, per book. */
const DETAIL: Record<string, Array<[string, string]>> = {
  orb: [['direction', 'Dir'], ['qty', 'Qty'], ['entry_price', 'Entry'], ['exit_price', 'Exit'],
        ['entry_time', 'In'], ['exit_time', 'Out'], ['exit_reason', 'Why'], ['conviction_grade', 'Grade']],
  n500m: [['direction', 'Dir'], ['signal_type', 'Signal'], ['timeframe', 'TF'], ['qty', 'Qty'],
          ['entry_price', 'Entry'], ['exit_price', 'Exit'], ['entry_time', 'In'], ['exit_time', 'Out'],
          ['exit_reason', 'Why']],
  mst: [['side', 'Side'], ['leg_role', 'Leg'], ['strike', 'Strike'], ['option_type', 'CE/PE'],
        ['qty', 'Qty'], ['entry_price', 'Entry'], ['exit_price', 'Exit'], ['exit_reason', 'Why']],
};

export default function DailyPerformance({ book, title = 'Daily performance' }:
    { book: string; title?: string }) {
  const [feed, setFeed] = useState<Feed | null>(null);
  const [failed, setFailed] = useState(false);
  const [open, setOpen] = useState<string | null>(null);
  const [limit, setLimit] = useState(20);

  useEffect(() => {
    let alive = true;
    fetch(`/app/book_daily.json?t=${Date.now()}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((d: Feed) => { if (alive) setFeed(d); })
      .catch(() => { if (alive) setFailed(true); });
    return () => { alive = false; };
  }, []);

  const b = feed?.books?.[book];
  const days = b?.days ?? [];
  const shown = useMemo(() => days.slice(0, limit), [days, limit]);

  // the cumulative in the feed runs oldest-first; the bar scale needs the extremes
  const peak = useMemo(
    () => Math.max(1, ...days.map((d) => Math.abs(d.pnl))),
    [days],
  );

  if (failed) return null;                    // the feed is optional furniture, never an error state
  if (!feed) return <div className={styles.loading}>Loading history…</div>;
  if (!days.length) {
    return (
      <section className={styles.wrap}>
        <div className={styles.head}><h3 className={styles.title}>{title}</h3></div>
        <div className={styles.empty}>No closed trades recorded yet.</div>
      </section>
    );
  }

  const s = b!.summary!;
  const cols = DETAIL[book] ?? DETAIL.orb;

  return (
    <section className={styles.wrap}>
      <div className={styles.head}>
        <h3 className={styles.title}>{title}</h3>
        <span className={styles.range}>
          {s.first === s.last ? day(s.first!) : `${day(s.first!)} → ${day(s.last!)}`}
          <span className={styles.dim}> · {s.days} trading days</span>
        </span>
      </div>

      <div className={styles.rail}>
        {[
          ['Net', inr(s.net), s.net >= 0 ? styles.pos : styles.neg],
          ['Trades', String(s.trades), ''],
          ['Win rate', s.win_rate == null ? '—' : `${s.win_rate}%`, ''],
          ['Green days', `${s.green_days}/${s.days}`, ''],
          ['Best day', inr(s.best_day), styles.pos],
          ['Worst day', inr(s.worst_day), styles.neg],
        ].map(([k, v, c]) => (
          <div key={k as string} className={styles.railItem}>
            <span className={styles.railK}>{k}</span>
            <span className={`${styles.railV} ${c as string}`}>{v}</span>
          </div>
        ))}
      </div>

      <table className={styles.tbl}>
        <thead>
          <tr>
            <th>Date</th>
            <th>Book</th>
            <th className={styles.numr}>Trades</th>
            <th className={styles.numr}>W / L</th>
            <th className={styles.numr}>Day P&amp;L</th>
            <th className={styles.bar}>&nbsp;</th>
            <th className={styles.numr}>Cumulative</th>
          </tr>
        </thead>
        <tbody>
          {shown.map((d) => {
            const isOpen = open === d.date;
            const w = Math.round((Math.abs(d.pnl) / peak) * 100);
            return [
              <tr key={d.date} className={styles.dayRow}
                  onClick={() => setOpen(isOpen ? null : d.date)}>
                <td>
                  <span className={styles.caret}>{isOpen ? '▾' : '▸'}</span>
                  {day(d.date)}
                  <span className={styles.dim}> {weekday(d.date)}</span>
                </td>
                <td>
                  <span className={`${styles.mode} ${d.mode === 'live' ? styles.modeLive : ''}`}>
                    {d.mode}
                  </span>
                </td>
                <td className={styles.numr}>{d.trades}</td>
                <td className={styles.numr}>
                  <span className={styles.pos}>{d.wins}</span>
                  <span className={styles.dim}> / </span>
                  <span className={styles.neg}>{d.losses}</span>
                </td>
                <td className={`${styles.numr} ${d.pnl >= 0 ? styles.pos : styles.neg}`}>
                  {inr(d.pnl)}
                </td>
                <td className={styles.bar}>
                  <span className={`${styles.barFill} ${d.pnl >= 0 ? styles.barPos : styles.barNeg}`}
                        style={{ width: `${w}%` }} />
                </td>
                <td className={`${styles.numr} ${styles.dim}`}>{inr(d.cum)}</td>
              </tr>,
              isOpen ? (
                <tr key={`${d.date}-x`} className={styles.detailRow}>
                  <td colSpan={7}>
                    <table className={styles.inner}>
                      <thead>
                        <tr>
                          <th>Symbol</th>
                          {cols.map(([, label]) => (
                            <th key={label} className={styles.numr}>{label}</th>
                          ))}
                          <th className={styles.numr}>P&amp;L</th>
                        </tr>
                      </thead>
                      <tbody>
                        {d.rows.map((r, i) => (
                          <tr key={i}>
                            <td>{String(r.symbol ?? '—')}</td>
                            {cols.map(([key, label]) => (
                              <td key={label} className={styles.numr}>
                                {key.endsWith('_time') ? time(r[key]) : num(r[key])}
                              </td>
                            ))}
                            <td className={`${styles.numr} ${(r.pnl as number) >= 0 ? styles.pos : styles.neg}`}>
                              {inr(r.pnl as number)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </td>
                </tr>
              ) : null,
            ];
          })}
        </tbody>
      </table>

      {days.length > shown.length && (
        <button type="button" className={styles.more} onClick={() => setLimit((n) => n + 30)}>
          Show {Math.min(30, days.length - shown.length)} more of {days.length} days
        </button>
      )}
      <div className={styles.foot}>
        Realised P&amp;L on closed trades, from the book's own record ·
        rebuilt {new Date(feed.generated_at).toLocaleString('en-IN',
          { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' })}
      </div>
    </section>
  );
}
