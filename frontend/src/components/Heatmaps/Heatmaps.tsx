/**
 * Heatmaps — books × days and stocks × days.
 *
 * Two matrices over the same idea: one cell per row per session, coloured by
 * what happened. The books grid answers "which systems actually earned this
 * quarter, and on which days"; the stocks grid answers "what did the market do
 * underneath them", grouped by sector so leadership is visible as a block.
 *
 * Canvas rather than DOM: the stock grid is ~12,000 cells and 12,000 divs is a
 * scroll-jank machine. Reads one static feed (overview_heatmaps.json) that a
 * daily job writes; renders nothing if it is absent.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import styles from './Heatmaps.module.css';

interface BookRow { label: string; v: number[]; total: number; days: number }
interface StockRow { s: string; sec: string; v: Array<number | null>; sum: number }
interface Feed {
  generated_at: string;
  books: { dates: string[]; rows: BookRow[] };
  stocks: { dates: string[]; rows: StockRow[] };
}

const POS = [15, 110, 86];
const NEG = [163, 45, 45];
const MID = [237, 234, 228];

function mix(a: number[], b: number[], t: number) {
  return `rgb(${a.map((x, i) => Math.round(x + (b[i] - x) * t)).join(',')})`;
}
function colour(v: number | null, clamp: number) {
  if (v === null || v === undefined) return '#F4F2ED';
  const t = Math.max(-1, Math.min(1, v / clamp));
  return t < 0 ? mix(MID, NEG, -t) : mix(MID, POS, t);
}
const inr = (v: number) =>
  `${v < 0 ? '−' : ''}₹${Math.abs(Math.round(v)).toLocaleString('en-IN')}`;

export default function Heatmaps() {
  const [feed, setFeed] = useState<Feed | null>(null);
  const [tab, setTab] = useState<'books' | 'stocks'>('books');
  const [tip, setTip] = useState<{ x: number; y: number; html: string } | null>(null);
  const cv = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    fetch(`/app/overview_heatmaps.json?t=${Date.now()}`, { cache: 'no-store' })
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => setFeed(d))
      .catch(() => undefined);
  }, []);

  // p90 of |value| keeps one outlier from washing the whole grid out
  const clamp = useMemo(() => {
    if (!feed) return 1;
    const vals =
      tab === 'books'
        ? feed.books.rows.flatMap((r) => r.v).filter((v) => v)
        : feed.stocks.rows.flatMap((r) => r.v).filter((v): v is number => v != null && v !== 0);
    if (!vals.length) return 1;
    const s = vals.map(Math.abs).sort((a, b) => a - b);
    return s[Math.floor(s.length * 0.9)] || 1;
  }, [feed, tab]);

  const geom = useMemo(() => {
    if (!feed) return null;
    const rows = tab === 'books' ? feed.books.rows.length : feed.stocks.rows.length;
    const cols = tab === 'books' ? feed.books.dates.length : feed.stocks.dates.length;
    const rh = tab === 'books' ? 15 : 3.2;
    const padL = tab === 'books' ? 108 : 78;
    return { rows, cols, rh, padL, h: Math.ceil(rows * rh) + 20 };
  }, [feed, tab]);

  useEffect(() => {
    if (!feed || !geom || !cv.current) return;
    const c = cv.current;
    const dpr = window.devicePixelRatio || 1;
    const w = c.getBoundingClientRect().width;
    c.width = w * dpr;
    c.height = geom.h * dpr;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, geom.h);
    const cw = (w - geom.padL - 8) / geom.cols;
    const font = getComputedStyle(document.body).fontFamily;

    if (tab === 'books') {
      feed.books.rows.forEach((r, i) => {
        const y = 16 + i * geom.rh;
        ctx.fillStyle = '#1B1B1A';
        ctx.font = `500 10px ${font}`;
        ctx.fillText(r.label.slice(0, 16), 2, y + geom.rh - 4);
        r.v.forEach((v, j) => {
          ctx.fillStyle = colour(v, clamp);
          ctx.fillRect(geom.padL + j * cw, y, Math.max(1, cw - 0.6), geom.rh - 2);
        });
      });
    } else {
      let last = '';
      feed.stocks.rows.forEach((r, i) => {
        const y = 16 + i * geom.rh;
        if (r.sec !== last) {
          last = r.sec;
          ctx.fillStyle = '#888780';
          ctx.font = `9px ${font}`;
          ctx.fillText(r.sec.slice(0, 13), 2, y + 4);
          ctx.strokeStyle = 'rgba(0,0,0,.10)';
          ctx.beginPath();
          ctx.moveTo(geom.padL, y - 1);
          ctx.lineTo(w - 8, y - 1);
          ctx.stroke();
        }
        r.v.forEach((v, j) => {
          ctx.fillStyle = colour(v, clamp);
          ctx.fillRect(geom.padL + j * cw, y, Math.max(1, cw - 0.4), Math.max(1.6, geom.rh - 0.8));
        });
      });
    }

    // date ticks
    const dates = tab === 'books' ? feed.books.dates : feed.stocks.dates;
    ctx.fillStyle = '#B4B2A9';
    ctx.font = `9px ${font}`;
    const step = Math.max(1, Math.floor(dates.length / 8));
    dates.forEach((d, j) => {
      if (j % step === 0) ctx.fillText(d.slice(5), geom.padL + j * cw, 10);
    });
  }, [feed, geom, tab, clamp]);

  if (!feed) return null;

  const onMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!geom) return;
    const b = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - b.left;
    const y = e.clientY - b.top - 16;
    const cw = (b.width - geom.padL - 8) / geom.cols;
    const i = Math.floor(y / geom.rh);
    const j = Math.floor((x - geom.padL) / cw);
    if (i < 0 || j < 0 || j >= geom.cols) return setTip(null);
    if (tab === 'books') {
      const r = feed.books.rows[i];
      if (!r) return setTip(null);
      const v = r.v[j];
      setTip({ x: e.clientX, y: e.clientY,
        html: `<b>${r.label}</b> · ${feed.books.dates[j]}<br>${v ? inr(v) : 'no trade'}` });
    } else {
      const r = feed.stocks.rows[i];
      if (!r) return setTip(null);
      const v = r.v[j];
      setTip({ x: e.clientX, y: e.clientY,
        html: `<b>${r.s}</b> <span class="g">${r.sec}</span> · ${feed.stocks.dates[j]}<br>` +
              (v == null ? 'no data' : `${v > 0 ? '+' : ''}${v.toFixed(2)}% · ${r.sum > 0 ? '+' : ''}${r.sum.toFixed(1)}% over the window`) });
    }
  };

  const rows = tab === 'books' ? feed.books.rows.length : feed.stocks.rows.length;
  const cols = tab === 'books' ? feed.books.dates.length : feed.stocks.dates.length;

  return (
    <section className={styles.panel}>
      <div className={styles.head}>
        <h2 className={styles.title}>Heatmap</h2>
        <div className={styles.tabs}>
          <button type="button" className={`${styles.tab} ${tab === 'books' ? styles.on : ''}`}
                  onClick={() => setTab('books')}>Books × days</button>
          <button type="button" className={`${styles.tab} ${tab === 'stocks' ? styles.on : ''}`}
                  onClick={() => setTab('stocks')}>Stocks × days</button>
        </div>
        <span className={styles.sub}>
          {rows} rows × {cols} sessions · {tab === 'books'
            ? 'green = the book made money that day'
            : 'grouped by sector, sorted by window return'}
        </span>
      </div>
      <div className={tab === 'stocks' ? styles.scroll : undefined}>
        <canvas
          ref={cv}
          style={{ height: geom ? geom.h : 200 }}
          className={styles.canvas}
          onMouseMove={onMove}
          onMouseLeave={() => setTip(null)}
        />
      </div>
      {tip && (
        <div className={styles.tip}
             style={{ left: Math.min(tip.x + 14, window.innerWidth - 220), top: tip.y + 14 }}
             dangerouslySetInnerHTML={{ __html: tip.html }} />
      )}
    </section>
  );
}
