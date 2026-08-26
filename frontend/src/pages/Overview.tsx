/**
 * Overview — the desk view. One screen, top-down by what you check first.
 *
 *   Tape      — the numbers you glance at before anything else.
 *   Equity    — cumulative net P&L with the underwater plot beneath it.
 *               The drawdown is not optional: an equity curve without one
 *               is a sales chart.
 *   Systems   — dense table, one row per book, sorted by 30-day impact.
 *   Journal   — the month, day by day, as a heatmap.
 *
 * Everything is derived from two read-only feeds — /api/books/liveness and
 * /api/journal/summary. This page holds no state and places no orders.
 *
 * Metric honesty: "Sharpe" here is computed on daily *rupee* P&L
 * (mean/σ × √252), not on returns over a capital base — it is a
 * consistency ratio, and is labelled as such.
 */

import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import styles from './Overview.module.css';
import { apiGet } from '../api/client';

// ---------------------------------------------------------------- types

interface BookLiveness {
  trades: number;
  last_trade: string | null;
  days_idle: number | null;
  trades_30d: number;
  net_total: number | null;
  net_30d: number | null;
  win_rate: number | null;
  series: Array<{ d: string; c: number }>;
}

interface JournalDay {
  trade_date: string;
  pnl_net: number;
  pnl_gross: number;
  trades: number;
  wins: number;
  losses: number;
  best: number;
  worst: number;
}

const BOOK_LABELS: Record<string, string> = {
  'nas-nifty': 'NAS · NIFTY',
  'nas-sensex': 'NAS · SENSEX',
  'orb-cash': 'ORB Cash',
  'orb-index': 'ORB Index',
  'orb-paper': 'ORB Paper',
  kc6: 'KC6 Mean Reversion',
  n500m: 'Nifty 500 Momentum',
  nwv: 'NWV',
  mst: 'MST',
  i75wr: 'Intraday 75WR',
  pairs: 'Pair Trading',
  'ha-paper': 'Heikin Ashi',
  'ohol-paper': 'OHOL',
  'momentum-3l': 'Momentum 3L',
  'fnoms-paper': 'FNO Multi-Signal',
  'breakout-paper': 'Breakout',
};

const REAL_MONEY = new Set(['nas-nifty', 'nas-sensex', 'orb-cash', 'orb-index', 'kc6', 'n500m']);

const RANGES = [
  { key: '1M', days: 30 },
  { key: '3M', days: 90 },
  { key: '6M', days: 182 },
  { key: 'ALL', days: 100000 },
] as const;

// ------------------------------------------------------------ formatting

function inr(v: number | null | undefined, sign = true) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  const abs = Math.abs(v);
  let body: string;
  if (abs >= 100000) body = `${(abs / 100000).toFixed(2)}L`;
  else if (abs >= 1000) body = `${(abs / 1000).toFixed(1)}k`;
  else body = abs.toFixed(0);
  const pre = sign ? (v > 0 ? '+' : v < 0 ? '−' : '') : v < 0 ? '−' : '';
  return `${pre}₹${body}`;
}

const pct = (v: number | null | undefined, d = 0) =>
  v === null || v === undefined || Number.isNaN(v) ? '—' : `${v.toFixed(d)}%`;

const tone = (v: number | null | undefined) =>
  v === null || v === undefined || v === 0 ? styles.flat : v > 0 ? styles.pos : styles.neg;

const shortDate = (iso: string) =>
  new Date(iso + 'T00:00:00').toLocaleDateString('en-GB', { day: '2-digit', month: 'short' });

// --------------------------------------------------------- equity chart

function EquityChart({ days }: { days: JournalDay[] }) {
  const W = 1000;
  const H = 190;
  const D = 62; // drawdown panel height
  const PAD_L = 4;

  const pts = useMemo(() => {
    let c = 0;
    let peak = 0;
    return days.map((d) => {
      c += d.pnl_net || 0;
      peak = Math.max(peak, c);
      return { date: d.trade_date, cum: c, dd: c - peak, pnl: d.pnl_net || 0 };
    });
  }, [days]);

  if (pts.length < 2) return <div className={styles.chartEmpty}>Not enough history to plot.</div>;

  const cums = pts.map((p) => p.cum);
  const lo = Math.min(...cums, 0);
  const hi = Math.max(...cums, 0);
  const span = hi - lo || 1;
  const x = (i: number) => PAD_L + (i / (pts.length - 1)) * (W - PAD_L * 2);
  const y = (v: number) => H - ((v - lo) / span) * H;

  const line = pts.map((p, i) => `${x(i).toFixed(1)},${y(p.cum).toFixed(1)}`).join(' ');
  const zeroY = y(0);
  const area = `${PAD_L},${zeroY} ${line} ${x(pts.length - 1).toFixed(1)},${zeroY}`;

  const ddMin = Math.min(...pts.map((p) => p.dd), -1);
  const dy = (v: number) => (v / ddMin) * D;
  const ddArea =
    `${PAD_L},0 ` +
    pts.map((p, i) => `${x(i).toFixed(1)},${dy(p.dd).toFixed(1)}`).join(' ') +
    ` ${x(pts.length - 1).toFixed(1)},0`;

  const last = pts[pts.length - 1];
  const up = last.cum >= 0;

  // gridlines at 0, mid, max
  const grid = [hi, (hi + lo) / 2, lo].filter((v, i, a) => a.indexOf(v) === i);

  return (
    <div className={styles.chart}>
      <svg viewBox={`0 0 ${W} ${H}`} className={styles.chartSvg} preserveAspectRatio="none">
        {grid.map((g, i) => (
          <line key={i} x1="0" x2={W} y1={y(g)} y2={y(g)} className={styles.gridLine} />
        ))}
        <line x1="0" x2={W} y1={zeroY} y2={zeroY} className={styles.zeroLine} />
        <polygon points={area} className={up ? styles.areaPos : styles.areaNeg} />
        <polyline points={line} className={up ? styles.linePos : styles.lineNeg} />
        <circle cx={x(pts.length - 1)} cy={y(last.cum)} r="3.5" className={styles.tip} />
      </svg>

      <div className={styles.chartAxis}>
        <span>{inr(lo)}</span>
        <span className={styles.axisMid}>cumulative net P&amp;L</span>
        <span>{inr(hi)}</span>
      </div>

      <div className={styles.ddLabel}>
        Drawdown <span className={styles.ddMax}>max {inr(ddMin)}</span>
      </div>
      <svg viewBox={`0 0 ${W} ${D}`} className={styles.ddSvg} preserveAspectRatio="none">
        <polygon points={ddArea} className={styles.ddArea} />
        <polyline
          points={pts.map((p, i) => `${x(i).toFixed(1)},${dy(p.dd).toFixed(1)}`).join(' ')}
          className={styles.ddLine}
        />
      </svg>

      <div className={styles.chartDates}>
        <span>{shortDate(pts[0].date)}</span>
        <span>{shortDate(pts[Math.floor(pts.length / 2)].date)}</span>
        <span>{shortDate(last.date)}</span>
      </div>
    </div>
  );
}

// ------------------------------------------------------------- sparkline

function Spark({ series }: { series: Array<{ d: string; c: number }> }) {
  if (!series || series.length < 2) return <span className={styles.sparkNil}>—</span>;
  const vals = series.map((p) => p.c);
  const min = Math.min(...vals, 0);
  const max = Math.max(...vals, 0);
  const span = max - min || 1;
  const w = 68;
  const h = 20;
  const pts = series
    .map((p, i) => `${((i / (series.length - 1)) * w).toFixed(1)},${(h - ((p.c - min) / span) * h).toFixed(1)}`)
    .join(' ');
  const up = vals[vals.length - 1] >= 0;
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} className={styles.spark} aria-hidden="true">
      <polyline points={pts} className={up ? styles.linePos : styles.lineNeg} />
    </svg>
  );
}

// -------------------------------------------------------------- calendar

function Calendar({ year, monthIdx, days }: { year: number; monthIdx: number; days: JournalDay[] }) {
  const map = useMemo(() => {
    const m: Record<string, JournalDay> = {};
    days.forEach((d) => (m[d.trade_date] = d));
    return m;
  }, [days]);

  const peak = useMemo(
    () => days.reduce((p, d) => Math.max(p, Math.abs(d.pnl_net || 0)), 0) || 1,
    [days],
  );

  const cells = useMemo(() => {
    const first = new Date(year, monthIdx, 1);
    const lastDate = new Date(year, monthIdx + 1, 0).getDate();
    const startDow = (first.getDay() + 6) % 7;
    const out: Array<{ day: number; date?: string }> = [];
    for (let i = 0; i < startDow; i++) out.push({ day: 0 });
    const p = (n: number) => String(n).padStart(2, '0');
    for (let d = 1; d <= lastDate; d++)
      out.push({ day: d, date: `${year}-${p(monthIdx + 1)}-${p(d)}` });
    return out;
  }, [year, monthIdx]);

  return (
    <div className={styles.cal}>
      <div className={styles.calGrid}>
        {['M', 'T', 'W', 'T', 'F', 'S', 'S'].map((d, i) => (
          <div key={i} className={styles.calDow}>
            {d}
          </div>
        ))}
        {cells.map((c, i) => {
          if (!c.date) return <div key={i} />;
          const row = map[c.date];
          const v = row?.pnl_net ?? null;
          const k = v === null ? 0 : Math.min(1, Math.abs(v) / peak);
          const bg =
            v === null || v === 0
              ? undefined
              : v > 0
                ? `rgba(15,110,86,${(0.1 + k * 0.5).toFixed(3)})`
                : `rgba(163,45,45,${(0.1 + k * 0.5).toFixed(3)})`;
          return (
            <Link
              key={i}
              to={`/journal/day/${c.date}`}
              className={`${styles.calCell} ${row ? styles.calOn : ''} ${k > 0.6 ? styles.calStrong : ''}`}
              style={bg ? { background: bg } : undefined}
              title={
                row
                  ? `${c.date} · ${inr(row.pnl_net)} · ${row.trades} trades · ${row.wins}W/${row.losses}L`
                  : `${c.date} · flat`
              }
            >
              <span className={styles.calNum}>{c.day}</span>
              {row && <span className={styles.calVal}>{inr(row.pnl_net)}</span>}
            </Link>
          );
        })}
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ page

export default function Overview() {
  const [books, setBooks] = useState<Record<string, BookLiveness>>({});
  const [allDays, setAllDays] = useState<JournalDay[]>([]);
  const [loading, setLoading] = useState(true);
  const [range, setRange] = useState<string>('ALL');
  const [now] = useState(() => new Date());

  useEffect(() => {
    let dead = false;
    const to = now.toISOString().slice(0, 10);
    Promise.all([
      apiGet<{ books: Record<string, BookLiveness> }>(`/api/books/liveness?t=${Date.now()}`).catch(
        () => ({ books: {} }),
      ),
      apiGet<{ days: JournalDay[] }>(`/api/journal/summary?from=2020-01-01&to=${to}`).catch(() => ({
        days: [],
      })),
    ]).then(([b, j]) => {
      if (dead) return;
      setBooks(b?.books ?? {});
      setAllDays((j?.days ?? []).slice().sort((a, b2) => a.trade_date.localeCompare(b2.trade_date)));
      setLoading(false);
    });
    return () => {
      dead = true;
    };
  }, [now]);

  const days = useMemo(() => {
    const n = RANGES.find((r) => r.key === range)?.days ?? 100000;
    return allDays.slice(-n);
  }, [allDays, range]);

  /** Desk metrics, all from daily net P&L. */
  const m = useMemo(() => {
    if (!days.length)
      return { net: 0, mdd: 0, sharpe: null as number | null, pf: null as number | null, winDays: null as number | null, trades: 0, best: 0, worst: 0, streak: 0 };
    let c = 0;
    let peak = 0;
    let mdd = 0;
    let gw = 0;
    let gl = 0;
    let wins = 0;
    let trades = 0;
    days.forEach((d) => {
      const v = d.pnl_net || 0;
      c += v;
      peak = Math.max(peak, c);
      mdd = Math.min(mdd, c - peak);
      if (v > 0) {
        gw += v;
        wins += 1;
      } else gl += -v;
      trades += d.trades || 0;
    });
    const vals = days.map((d) => d.pnl_net || 0);
    const mean = vals.reduce((s, v) => s + v, 0) / vals.length;
    const sd = Math.sqrt(vals.reduce((s, v) => s + (v - mean) ** 2, 0) / Math.max(1, vals.length - 1));
    // trailing streak of same-sign days
    let streak = 0;
    const sgn = Math.sign(vals[vals.length - 1] || 0);
    for (let i = vals.length - 1; i >= 0 && Math.sign(vals[i]) === sgn && sgn !== 0; i--) streak++;
    return {
      net: c,
      mdd,
      sharpe: sd > 0 ? (mean / sd) * Math.sqrt(252) : null,
      pf: gl > 0 ? gw / gl : null,
      winDays: (wins / days.length) * 100,
      trades,
      best: Math.max(...vals),
      worst: Math.min(...vals),
      streak: streak * (sgn || 1),
    };
  }, [days]);

  const rows = useMemo(
    () =>
      Object.entries(books)
        .map(([key, b]) => ({ key, ...b }))
        .filter((r) => r.trades > 0)
        .sort((a, b) => (b.net_30d ?? 0) - (a.net_30d ?? 0)),
    [books],
  );

  const liveCount = rows.filter((r) => r.days_idle !== null && r.days_idle <= 1).length;
  const monthDays = useMemo(
    () =>
      allDays.filter((d) => {
        const dt = new Date(d.trade_date + 'T00:00:00');
        return dt.getFullYear() === now.getFullYear() && dt.getMonth() === now.getMonth();
      }),
    [allDays, now],
  );
  const monthNet = monthDays.reduce((s, d) => s + (d.pnl_net || 0), 0);

  return (
    <div className={styles.page}>
      {/* ------------------------------------------------------ header */}
      <header className={styles.head}>
        <div className={styles.headL}>
          <h1 className={styles.title}>Overview</h1>
          <span className={styles.kicker}>
            Systematic derivatives &amp; equities · net of costs
          </span>
        </div>
        <div className={styles.headR}>
          <span className={styles.live}>
            <span className={`${styles.dot} ${liveCount ? styles.dotOn : ''}`} />
            {liveCount} live
          </span>
          <span className={styles.sep} />
          <span className={styles.stamp}>
            {now.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' })}
          </span>
        </div>
      </header>

      {/* -------------------------------------------------------- tape */}
      <section className={styles.tape}>
        {[
          { k: 'Net P&L', v: inr(m.net), c: tone(m.net), s: `${days.length} sessions` },
          { k: 'Max DD', v: inr(m.mdd, false), c: styles.neg, s: 'peak to trough' },
          { k: 'Sharpe', v: m.sharpe === null ? '—' : m.sharpe.toFixed(2), c: tone(m.sharpe), s: 'daily P&L · ann.' },
          { k: 'Profit factor', v: m.pf === null ? '—' : m.pf.toFixed(2), c: tone(m.pf ? m.pf - 1 : 0), s: 'gross win / loss' },
          { k: 'Win days', v: pct(m.winDays), c: styles.flat, s: `of ${days.length}` },
          { k: 'Trades', v: m.trades.toLocaleString('en-IN'), c: styles.flat, s: 'closed' },
          { k: 'Best day', v: inr(m.best), c: styles.pos, s: 'single session' },
          { k: 'Worst day', v: inr(m.worst), c: styles.neg, s: 'single session' },
        ].map((t) => (
          <div key={t.k} className={styles.tapeCell}>
            <div className={styles.tapeK}>{t.k}</div>
            <div className={`${styles.tapeV} ${t.c}`}>{loading ? '·' : t.v}</div>
            <div className={styles.tapeS}>{t.s}</div>
          </div>
        ))}
      </section>

      {/* ------------------------------------------------------ equity */}
      <section className={styles.panel}>
        <div className={styles.panelHead}>
          <h2 className={styles.panelTitle}>Equity curve</h2>
          <div className={styles.ranges}>
            {RANGES.map((r) => (
              <button
                key={r.key}
                className={`${styles.rangeBtn} ${range === r.key ? styles.rangeOn : ''}`}
                onClick={() => setRange(r.key)}
              >
                {r.key}
              </button>
            ))}
          </div>
        </div>
        {loading ? <div className={styles.chartEmpty}>Loading…</div> : <EquityChart days={days} />}
      </section>

      {/* --------------------------------------------- systems + journal */}
      <div className={styles.split}>
        <section className={styles.panel}>
          <div className={styles.panelHead}>
            <h2 className={styles.panelTitle}>Systems</h2>
            <Link to="/strategies" className={styles.panelLink}>
              Register →
            </Link>
          </div>
          <div className={styles.tableWrap}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th className={styles.thL}>Book</th>
                  <th>30D</th>
                  <th className={styles.thSpark}>Curve</th>
                  <th>All-time</th>
                  <th>Win</th>
                  <th>Trades</th>
                  <th className={styles.thR}>Last</th>
                </tr>
              </thead>
              <tbody>
                {loading && (
                  <tr>
                    <td colSpan={7} className={styles.tdEmpty}>
                      Loading books…
                    </td>
                  </tr>
                )}
                {rows.map((r) => {
                  const on = r.days_idle !== null && r.days_idle <= 1;
                  return (
                    <tr key={r.key}>
                      <td className={styles.tdName}>
                        <span className={`${styles.dot} ${on ? styles.dotOn : styles.dotOff}`} />
                        <span className={styles.bookName}>{BOOK_LABELS[r.key] ?? r.key}</span>
                        <span className={REAL_MONEY.has(r.key) ? styles.tagLive : styles.tagPaper}>
                          {REAL_MONEY.has(r.key) ? 'LIVE' : 'PAPER'}
                        </span>
                      </td>
                      <td className={`${styles.num} ${tone(r.net_30d)}`}>{inr(r.net_30d)}</td>
                      <td className={styles.tdSpark}>
                        <Spark series={r.series} />
                      </td>
                      <td className={`${styles.num} ${tone(r.net_total)}`}>{inr(r.net_total)}</td>
                      <td className={styles.num}>{pct(r.win_rate)}</td>
                      <td className={styles.num}>{r.trades.toLocaleString('en-IN')}</td>
                      <td className={`${styles.num} ${styles.tdMuted}`}>
                        {r.last_trade ? shortDate(r.last_trade) : '—'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </section>

        <section className={styles.panel}>
          <div className={styles.panelHead}>
            <h2 className={styles.panelTitle}>
              {now.toLocaleString('en-US', { month: 'long' })}
              <span className={`${styles.monthNet} ${tone(monthNet)}`}>{inr(monthNet)}</span>
            </h2>
            <Link to="/journal" className={styles.panelLink}>
              Journal →
            </Link>
          </div>
          <Calendar year={now.getFullYear()} monthIdx={now.getMonth()} days={monthDays} />
          <div className={styles.calFoot}>
            <span className={styles.legLoss} />
            loss
            <span className={styles.legWin} />
            win
            <span className={styles.calFootR}>
              {monthDays.filter((d) => d.pnl_net > 0).length}W ·{' '}
              {monthDays.filter((d) => d.pnl_net < 0).length}L
            </span>
          </div>
        </section>
      </div>
    </div>
  );
}
