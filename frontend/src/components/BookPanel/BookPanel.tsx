/**
 * BookPanel — the summary block at the top of every book in the Momentum Portfolio.
 *
 * One component so True North, Open Alpha and IPO Base cannot drift apart again: they had
 * three near-copies of this markup, and the copies had already diverged on the date
 * format, on where the timestamp sat, and on whether the return appeared beside the value.
 *
 * Shape, decided with Arun 08-Sep-2026:
 *   line 1  what the book is worth, and what that cost — the return sits BESIDE the value
 *   line 2  what went in, and when it started (the two facts that do not move intraday)
 *   bar     where the money actually sits
 *   status  how the book is configured today, and whether every order filled
 *   right   where the return came from, footed by the total
 *   corner  freshness top-right; one chevron bottom-right, which is the only thing the
 *           closed panel spends on what is underneath it
 *
 * The chevron opens the book's trading record and its curve against the indices. Both used
 * to be separate sections further down the page; putting them behind the arrow is what
 * lets the page open on positions instead.
 */

import { useEffect, useState } from 'react';
import styles from '../../pages/MomentumPaper.module.css';
import LiveTick from '../LiveTick/LiveTick';
import BookActivity from '../BookActivity/BookActivity';
import BookCurve from './BookCurve';

export interface Seg { k: string; v: number; c: string }
export interface PnlRow { k: string; v: number; hint?: string }

const MONS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
/** dd-Mon-yyyy, the one date format on these pages (Arun, 08-Sep-2026). */
export function dmy(s?: string | null) {
  if (!s) return '—';
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(s);
  return m ? `${m[3]}-${MONS[parseInt(m[2], 10) - 1]}-${m[1]}` : s;
}
const inr = (n: number) => '₹' + Math.round(n).toLocaleString('en-IN');
const lakh = (n: number) => '₹' + (n / 100000).toFixed(2) + 'L';
const pct = (n: number | null | undefined) =>
  n == null ? '—' : (n >= 0 ? '+' : '−') + Math.abs(n).toFixed(1) + '%';
const tone = (v: number) =>
  v > 0 ? 'var(--accent-pos,#0F6E56)' : v < 0 ? 'var(--accent-neg,#A32D2D)' : 'var(--ink,#1B1B1A)';
const signed = (v: number) => (v >= 0 ? '+' : '−') + inr(Math.abs(v));

export interface BookPanelProps {
  /** 'Current value' unless the book has a reason to call it something else. */
  label?: string;
  /** The headline figure, already formatted — pages that tick wrap it in <Tick>. */
  hero: React.ReactNode;
  gain: number;
  returnPct: number | null;
  capital: number;
  /** 'invested' | 'of capital' | 'of notional capital' */
  capitalWord?: string;
  inception?: string | null;
  /** CAGR / worst drawdown and anything else that belongs on the second line. */
  extraSub?: React.ReactNode;
  /** Naive IST timestamp from the feed. */
  updated?: string | null;
  tickLabel?: string;
  segs: Seg[];
  /** The book's own configuration facts, as <span> children. */
  status: React.ReactNode;
  pnl: PnlRow[];
  /** Book id in /api/books/liveness. Omit and the record line is left out. */
  bookId?: string | null;
  /** Feed for the curve. Omit and only the record line appears. */
  curveUrl?: string | null;
  curveLabel?: string;
  /** Remembers open/closed for THIS book. */
  storageKey: string;
  /** Today's move, shown beside the total. Omit and the line is left out. */
  today?: number | null;
}


/** The cost rate every book is modelled at: 25 bps a side, the r/142 study assumption. */
export const COST_PCT = 0.0025;

/** A position priced well enough to know what it did today. */
export type DayRow = { qty?: number | null; ltp?: number | null;
                       prev_close?: number | null; day_move_pct?: number | null;
                       value?: number | null };

/**
 * Today's P&L: sum of qty x (last price - previous close).
 *
 * Falls back to day_move_pct against the position's value where a previous close is not
 * published, which is the same quantity by another route. Returns null when NOTHING could
 * be priced - a blank is honest, where a zero would read as "flat today".
 */
export function todayPnl(rows: DayRow[] | undefined | null): number | null {
  let tot = 0, n = 0;
  for (const r of rows ?? []) {
    if (r.qty != null && r.ltp != null && r.prev_close) {
      tot += r.qty * (r.ltp - r.prev_close);
      n++;
    } else if (r.day_move_pct != null && r.value) {
      tot += r.value - r.value / (1 + r.day_move_pct / 100);
      n++;
    }
  }
  return n ? tot : null;
}

/**
 * The P&L parts, with costs MEASURED rather than left over.
 *
 * Every book used to show `gain - (the parts)` as "Costs & fees", which is not a cost - it
 * is whatever the parts failed to explain, and it printed POSITIVE on Open Alpha. Costs are
 * now computed from what is actually known, and any genuine discrepancy gets its own line
 * where it can be seen instead of being dressed up as a fee.
 */
export function pnlBreakdown(o: {
  gain: number; unrealised: number; realised: number; invested: number;
  yieldRs?: number; yieldLabel?: string;
}): PnlRow[] {
  // Open positions only: a closed trade's costs are already inside its realised figure.
  const costs = -COST_PCT * (o.invested || 0);
  const rows: PnlRow[] = [
    { k: 'Unrealised', v: o.unrealised, hint: 'open positions at market, before entry costs' },
    { k: 'Realised (net)', v: o.realised, hint: 'closed trades, already net of their own costs' },
  ];
  if (o.yieldRs) {
    rows.push({ k: o.yieldLabel ?? 'Liquid fund yield', v: o.yieldRs,
                hint: 'gain on cash parked in the liquid ETF' });
  }
  rows.push({ k: 'Costs & fees', v: costs,
              hint: `modelled at ${(COST_PCT * 100).toFixed(2)}% a side on ${'\u20b9'}${Math.round(o.invested).toLocaleString('en-IN')} of open positions` });
  const gap = o.gain - (o.unrealised + o.realised + (o.yieldRs ?? 0) + costs);
  if (Math.abs(gap) >= 1) {
    rows.push({ k: 'Unreconciled', v: gap,
                hint: 'what the parts above do not explain. Mostly modelled charges the cash ledger never paid; it shrinks as trades reconcile through the corrected path. If it grows, something is wrong.' });
  }
  return rows;
}

export default function BookPanel(p: BookPanelProps) {
  const canOpen = !!(p.bookId || p.curveUrl);
  const [open, setOpen] = useState(false);

  /* The choice is per book, so True North can stay open while IPO Base stays shut.
     Storage can throw (private windows, blocked site data) — never let that break the page. */
  useEffect(() => {
    try {
      setOpen(localStorage.getItem('bookpanel:' + p.storageKey) === '1');
    } catch { /* no stored preference; closed is the right default */ }
  }, [p.storageKey]);

  const toggle = () => {
    const next = !open;
    setOpen(next);
    try { localStorage.setItem('bookpanel:' + p.storageKey, next ? '1' : '0'); } catch { /* ignore */ }
  };

  const total = p.segs.reduce((a, x) => a + x.v, 0) || 1;

  return (
    <div className={`${styles.sumWrap} ${open ? styles.sumWrapOpen : ''}`}>
      <div className={styles.bookSummary}>
        <div className={styles.sumMain}>
          <div className={styles.sumHead}>
            <span className={styles.sumLabel}>{p.label ?? 'Current value'}</span>
            <span className={styles.sumStamp}>
              <LiveTick updated={p.updated} label={p.tickLabel ?? 'marks'} />
              {p.updated && <span>updated {dmy(p.updated)} {p.updated.slice(11, 16)} IST</span>}
            </span>
          </div>

          <div className={styles.sumHeroLine}>
            <span className={styles.sumHero}>{p.hero}</span>
            <span className={styles.sumDelta} style={{ color: tone(p.gain) }}>
              {signed(p.gain)} · {pct(p.returnPct)}
            </span>
          </div>

          <div className={styles.sumSub}>
            on <b>{inr(p.capital)}</b> {p.capitalWord ?? 'invested'}
            {p.inception ? <> · since {dmy(p.inception)}</> : null}
            {p.extraSub}
          </div>

          <div className={styles.barWrap} role="img"
               aria-label={p.segs.map((x) => `${x.k} ${Math.round((x.v / total) * 100)}%`).join(', ')}>
            {p.segs.map((x) => (
              <div key={x.k} className={styles.barSeg}
                   style={{ width: `${(x.v / total) * 100}%`, background: x.c }} />
            ))}
          </div>
          <div className={styles.legend}>
            {p.segs.map((x) => (
              <span key={x.k} className={styles.legendItem}>
                <i className={styles.swatch} style={{ background: x.c }} />
                {x.k} <b>{lakh(x.v)}</b>
                <span className={styles.legendPct}>{((x.v / total) * 100).toFixed(0)}%</span>
              </span>
            ))}
          </div>

          <div className={styles.sumStatus}>
            {p.status}
            {canOpen && (
              <button type="button" className={styles.sumTog} onClick={toggle}
                      aria-expanded={open} aria-controls={'reveal-' + p.storageKey}
                      title={open ? 'Hide the record and the curve'
                                  : 'Trading record, and this book against the indices'}>
                {'▾'}
              </button>
            )}
          </div>
        </div>

        <div className={styles.sumPnl}>
          <div className={styles.sumLabel}>Profit &amp; loss</div>
          {p.pnl.map((r) => (
            <div key={r.k} className={styles.pnlRow} title={r.hint}>
              <span>{r.k}</span>
              <b style={{ color: tone(r.v) }}>{signed(r.v)}</b>
            </div>
          ))}
          <div className={`${styles.pnlRow} ${styles.pnlTotal}`}>
            <span>
              Total return
              {p.today != null && (
                /* on the row, not in the list above: a day is a slice of TIME, while the
                   rows are a decomposition by KIND. Listing it would imply it adds up
                   with them. */
                <span className={styles.pnlWas}>
                  today <b style={{ color: tone(p.today), fontWeight: 600 }}>
                    {signed(p.today)}</b>
                </span>
              )}
            </span>
            <b style={{ color: tone(p.gain) }}>{signed(p.gain)} · {pct(p.returnPct)}</b>
          </div>
        </div>
      </div>

      {canOpen && open && (
        <div className={styles.sumReveal} id={'reveal-' + p.storageKey}>
          {p.bookId && <BookActivity bookId={p.bookId} inline />}
          {p.curveUrl && <BookCurve url={p.curveUrl} label={p.curveLabel ?? 'This book'} />}
        </div>
      )}
    </div>
  );
}
