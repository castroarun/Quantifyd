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
import Heatmaps from '../components/Heatmaps/Heatmaps';

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

interface IndexTick {
  symbol: string;
  label: string;
  last: number;
  chg_pct: number;
  spark: number[];
  group: string;
}

interface DeskFeed {
  exposure: {
    option_legs: number; cash_legs: number; short_option_legs: number;
    naked: string[]; naked_count: number; open_pnl: number;
    age_mins: number | null;
    legs: Array<{ symbol: string; qty: number; avg: number; ltp: number; pnl: number }>;
  };
  recon: { alerts: number; warns: number; clean: boolean; age_mins: number | null;
           items: Array<{ level: string; kind: string; symbol: string; detail: string }> };
  gates: { nas_master_mode: string | null; freeze_flag: boolean;
           nas_matrix: { live: string[]; paper: string[] } };
  ops: { overdue: Array<{ title: string; due: string; in_days: number }>;
         due_soon: Array<{ title: string; due: string; in_days: number }>; tracked: number };
  health: { summary: { ok: number; warn: number; fail: number } | null;
            token_file_age_mins: number | null;
            token_chain: Record<string, { last: string | null; today: boolean }> };
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

// ------------------------------------------------------------- ticker

/** Market ticker — indices with last, change and a micro sparkline. */
function Ticker() {

  const [ticks, setTicks] = useState<IndexTick[]>([]);

  useEffect(() => {
    let dead = false;
    const pull = () =>
      apiGet<{ indices: IndexTick[] }>(`/api/index-pulse/strip?t=${Date.now()}`)
        .then((d) => {
          if (!dead) setTicks((d?.indices ?? []).filter((i) => i.group === 'broad').slice(0, 6));
        })
        .catch(() => undefined);
    pull();
    const id = setInterval(pull, 60000);
    return () => {
      dead = true;
      clearInterval(id);
    };
  }, []);

  if (!ticks.length) return null;

  return (
    <div className={styles.ticker}>
      <span className={styles.tickerTag}>
        <span className={styles.blip} />
        NSE
      </span>
      <div className={styles.tickerRail}>
        {ticks.map((t) => {
          const up = t.chg_pct >= 0;
          const vals = t.spark || [];
          const lo = Math.min(...vals);
          const hi = Math.max(...vals);
          const sp = hi - lo || 1;
          const pts = vals
            .map((v, i) => `${((i / Math.max(1, vals.length - 1)) * 40).toFixed(1)},${(14 - ((v - lo) / sp) * 14).toFixed(1)}`)
            .join(' ');
          return (
            <div key={t.symbol} className={styles.tick}>
              <span className={styles.tickName}>{t.label}</span>
              <span className={styles.tickLast}>
                {t.last.toLocaleString('en-IN', { maximumFractionDigits: 2 })}
              </span>
              {vals.length > 1 && (
                <svg width="40" height="14" viewBox="0 0 40 14" className={styles.tickSpark}>
                  <polyline points={pts} className={up ? styles.linePos : styles.lineNeg} />
                </svg>
              )}
              <span className={`${styles.tickChg} ${up ? styles.pos : styles.neg}`}>
                {up ? '\u25B2' : '\u25BC'} {Math.abs(t.chg_pct).toFixed(2)}%
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// -------------------------------------------------------------- clock

/** IST clock + NSE session state. Ticks every second. */
function Clock() {
  const [t, setT] = useState(() => new Date());
  useEffect(() => {
    const id = setInterval(() => setT(new Date()), 1000);
    return () => clearInterval(id);
  }, []);

  const ist = new Date(t.getTime() + (t.getTimezoneOffset() + 330) * 60000);
  const hh = ist.getHours();
  const mm = ist.getMinutes();
  const mins = hh * 60 + mm;
  const weekday = ist.getDay() >= 1 && ist.getDay() <= 5;
  const open = weekday && mins >= 555 && mins <= 930; // 09:15 - 15:30
  const pre = weekday && mins >= 540 && mins < 555;

  const state = open ? 'OPEN' : pre ? 'PRE-OPEN' : 'CLOSED';

  return (
    <span className={styles.clock}>
      <span className={`${styles.mktDot} ${open ? styles.mktOpen : pre ? styles.mktPre : ''}`} />
      <span className={styles.mktState}>{state}</span>
      <span className={styles.clockTime}>
        {String(hh).padStart(2, '0')}:{String(mm).padStart(2, '0')}
        <span className={styles.clockSec}>:{String(ist.getSeconds()).padStart(2, '0')}</span>
      </span>
      <span className={styles.clockTz}>IST</span>
    </span>
  );
}

// ----------------------------------------------------------- heartbeat

/** Session heartbeat — one bar per trading day, height and colour by net P&L. */
function Heartbeat({ days }: { days: JournalDay[] }) {
  const recent = days.slice(-70);
  if (!recent.length) return null;
  const peak = recent.reduce((p, d) => Math.max(p, Math.abs(d.pnl_net || 0)), 0) || 1;
  return (
    <div className={styles.hbWrap}>
      <div className={styles.hbHead}>
        <span className={styles.hbLabel}>Session heartbeat</span>
        <span className={styles.hbMeta}>last {recent.length} sessions</span>
      </div>
      <div className={styles.hb}>
        {recent.map((d) => {
          const v = d.pnl_net || 0;
          const h = Math.max(3, (Math.abs(v) / peak) * 26);
          return (
            <Link
              key={d.trade_date}
              to={`/journal/day/${d.trade_date}`}
              className={styles.hbCol}
              title={`${d.trade_date} · ${inr(v)} · ${d.trades} trades`}
            >
              <span className={styles.hbUp}>
                {v > 0 && <span className={styles.hbBarPos} style={{ height: `${h}px` }} />}
              </span>
              <span className={styles.hbMid} />
              <span className={styles.hbDown}>
                {v < 0 && <span className={styles.hbBarNeg} style={{ height: `${h}px` }} />}
              </span>
            </Link>
          );
        })}
      </div>
    </div>
  );
}

// --------------------------------------------------------- distribution

/** Daily P&L distribution — the shape of the edge, not just its size. */
function Distribution({ days }: { days: JournalDay[] }) {
  const bins = useMemo(() => {
    const vals = days.map((d) => d.pnl_net || 0);
    if (vals.length < 3) return null;
    const lo = Math.min(...vals);
    const hi = Math.max(...vals);
    const N = 17;
    const w = (hi - lo) / N || 1;
    const out = Array.from({ length: N }, (_, i) => ({ lo: lo + i * w, hi: lo + (i + 1) * w, n: 0 }));
    vals.forEach((v) => {
      const i = Math.min(N - 1, Math.max(0, Math.floor((v - lo) / w)));
      out[i].n += 1;
    });
    return { out, max: Math.max(...out.map((b) => b.n)) || 1, lo, hi };
  }, [days]);

  if (!bins) return null;

  return (
    <div className={styles.dist}>
      <div className={styles.distBars}>
        {bins.out.map((b, i) => {
          const mid = (b.lo + b.hi) / 2;
          return (
            <span
              key={i}
              className={`${styles.distBar} ${mid >= 0 ? styles.distPos : styles.distNeg}`}
              style={{ height: `${Math.max(2, (b.n / bins.max) * 100)}%` }}
              title={`${inr(b.lo)} to ${inr(b.hi)} — ${b.n} session${b.n === 1 ? '' : 's'}`}
            />
          );
        })}
      </div>
      <div className={styles.distAxis}>
        <span>{inr(bins.lo)}</span>
        <span className={styles.distZero}>0</span>
        <span>{inr(bins.hi)}</span>
      </div>
    </div>
  );
}

// --------------------------------------------------------- equity chart


const OVERLAY_COLOURS: Record<string, string> = {
  NIFTY50: '#1E3A8A',
  NIFTYMIDCAP150: '#B45309',
  NIFTYSMLCAP250: '#0F6E56',
  NIFTY500: '#8A6BBF',
};
const OVERLAY_LABELS: Record<string, string> = {
  NIFTY50: 'Nifty 50',
  NIFTYMIDCAP150: 'Midcap 150',
  NIFTYSMLCAP250: 'Smallcap 250',
  NIFTY500: 'Nifty 500',
};

interface Candle { d: string; end: string; o: number; h: number; l: number; c: number; n: number }

/** ISO week key, so the buckets line up with the index series. */
function isoWeek(iso: string): string {
  const dt = new Date(iso + 'T00:00:00Z');
  const day = (dt.getUTCDay() + 6) % 7;                 // Mon = 0
  dt.setUTCDate(dt.getUTCDate() - day + 3);             // nearest Thursday
  const first = new Date(Date.UTC(dt.getUTCFullYear(), 0, 4));
  const week = 1 + Math.round(((dt.getTime() - first.getTime()) / 86400000
    - 3 + ((first.getUTCDay() + 6) % 7)) / 7);
  return `${dt.getUTCFullYear()}-${String(week).padStart(2, '0')}`;
}

/** Weekly candles on the book's cumulative P&L, indices overlaid in %. */
function EquityCandles({ days }: { days: JournalDay[] }) {
  const [on, setOn] = useState<Record<string, boolean>>({
    NIFTY50: true, NIFTYMIDCAP150: true, NIFTYSMLCAP250: true, NIFTY500: false,
  });
  const [idx, setIdx] = useState<{ dates: string[]; series: Record<string, number[]> } | null>(null);

  useEffect(() => {
    apiGet<{ dates: string[]; series: Record<string, number[]> }>(
      `/api/index-pulse/compare?window=1y&mode=rel&symbols=${Object.keys(OVERLAY_COLOURS).join(',')}`,
    ).then(setIdx).catch(() => undefined);
  }, []);

  const candles = useMemo<Candle[]>(() => {
    const sorted = [...days].sort((a, b) => a.trade_date.localeCompare(b.trade_date));
    let cum = 0;
    const byWeek = new Map<string, { d: string; end: string; vals: number[] }>();
    for (const d of sorted) {
      cum += d.pnl_net || 0;
      const k = isoWeek(d.trade_date);
      const b = byWeek.get(k);
      if (b) { b.vals.push(cum); b.end = d.trade_date; }
      else byWeek.set(k, { d: d.trade_date, end: d.trade_date, vals: [cum] });
    }
    const out: Candle[] = [];
    let prev: number | null = null;
    for (const [, b] of byWeek) {
      const o = prev ?? b.vals[0];
      out.push({
        d: b.d, end: b.end, o,
        h: Math.max(o, ...b.vals), l: Math.min(o, ...b.vals),
        c: b.vals[b.vals.length - 1], n: b.vals.length,
      });
      prev = b.vals[b.vals.length - 1];
    }
    return out;
  }, [days]);

  const overlays = useMemo(() => {
    if (!idx || !candles.length) return {} as Record<string, Array<{ d: string; p: number }>>;
    const from = candles[0].d;
    const i0 = idx.dates.findIndex((d) => d >= from);
    if (i0 < 0) return {};
    const out: Record<string, Array<{ d: string; p: number }>> = {};
    for (const [sym, vals] of Object.entries(idx.series)) {
      const base = vals[i0];
      if (!base) continue;
      const seen = new Set<string>();
      const pts: Array<{ d: string; p: number }> = [];
      for (let i = i0; i < idx.dates.length; i++) {
        const k = isoWeek(idx.dates[i]);
        if (seen.has(k)) { pts[pts.length - 1] = { d: idx.dates[i], p: (vals[i] / base - 1) * 100 }; }
        else { seen.add(k); pts.push({ d: idx.dates[i], p: (vals[i] / base - 1) * 100 }); }
      }
      out[sym] = pts;
    }
    return out;
  }, [idx, candles]);

  if (candles.length < 2) return <div className={styles.chartEmpty}>Not enough history to plot.</div>;

  const W = 1000, H = 250, PAD_L = 62, PAD_R = 52, PAD_B = 18;
  const lo = Math.min(...candles.map((c) => c.l), 0);
  const hi = Math.max(...candles.map((c) => c.h), 0);
  const span = hi - lo || 1;
  const plotW = W - PAD_L - PAD_R;
  const bw = plotW / candles.length;
  const x = (i: number) => PAD_L + i * bw + bw / 2;
  const y = (v: number) => H - PAD_B - ((v - lo) / span) * (H - PAD_B - 8);

  const shown = Object.keys(overlays).filter((k) => on[k]);
  const allPct = shown.flatMap((k) => overlays[k].map((q) => q.p));
  const pMax = Math.max(2, ...allPct.map(Math.abs));
  const py = (v: number) => H - PAD_B - ((v + pMax) / (2 * pMax)) * (H - PAD_B - 8);
  const inr0 = (v: number) =>
    Math.abs(v) >= 1e5 ? `${(v / 1e5).toFixed(1)}L` : `${(v / 1e3).toFixed(0)}k`;

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className={styles.chartSvg} preserveAspectRatio="none">
        {[0, 0.25, 0.5, 0.75, 1].map((f) => {
          const v = lo + span * f;
          return (
            <g key={f}>
              <line x1={PAD_L} y1={y(v)} x2={W - PAD_R} y2={y(v)} className={styles.grid} />
              <text x={PAD_L - 6} y={y(v) + 3} className={styles.axisL}>{inr0(v)}</text>
            </g>
          );
        })}
        <line x1={PAD_L} y1={y(0)} x2={W - PAD_R} y2={y(0)} className={styles.zero} />

        {shown.map((k) => {
          const pts = overlays[k];
          const step = plotW / Math.max(1, pts.length - 1);
          const d = pts.map((q, i) => `${(PAD_L + i * step).toFixed(1)},${py(q.p).toFixed(1)}`).join(' ');
          return <polyline key={k} points={d} fill="none" stroke={OVERLAY_COLOURS[k]}
                           strokeWidth={1.4} strokeOpacity={0.8} />;
        })}

        {candles.map((c, i) => {
          const up = c.c >= c.o;
          const top = y(Math.max(c.o, c.c)), bot = y(Math.min(c.o, c.c));
          return (
            <g key={c.d} className={up ? styles.candleUp : styles.candleDn}>
              <title>{`${c.d} → ${c.end} · ${c.n} session${c.n === 1 ? '' : 's'}\nopen ₹${Math.round(c.o).toLocaleString('en-IN')}   close ₹${Math.round(c.c).toLocaleString('en-IN')}\nhigh ₹${Math.round(c.h).toLocaleString('en-IN')}   low ₹${Math.round(c.l).toLocaleString('en-IN')}\nweek ${c.c - c.o >= 0 ? '+' : '−'}₹${Math.abs(Math.round(c.c - c.o)).toLocaleString('en-IN')}`}</title>
              <line x1={x(i)} y1={y(c.h)} x2={x(i)} y2={y(c.l)} strokeWidth={1} />
              <rect x={x(i) - bw * 0.3} y={top} width={Math.max(1.2, bw * 0.6)} height={Math.max(1, bot - top)} />
            </g>
          );
        })}

        {shown.length > 0 && [pMax, 0, -pMax].map((v) => (
          <text key={v} x={W - PAD_R + 6} y={py(v) + 3} className={styles.axisR}>
            {v > 0 ? '+' : ''}{v.toFixed(0)}%
          </text>
        ))}
      </svg>

      <div className={styles.legend}>
        <span className={styles.legendNote}>
          candles = this book, cumulative ₹ (weekly) · lines = index % since {candles[0].d}
        </span>
        {Object.keys(OVERLAY_COLOURS).map((k) => {
          const last = overlays[k]?.slice(-1)[0]?.p;
          return (
            <button key={k} type="button"
              className={`${styles.legendBtn} ${on[k] ? styles.legendOn : ''}`}
              onClick={() => setOn((s) => ({ ...s, [k]: !s[k] }))}>
              <span className={styles.legendSwatch} style={{ background: OVERLAY_COLOURS[k] }} />
              {OVERLAY_LABELS[k]}
              {last != null && (
                <span className={last >= 0 ? styles.pos : styles.neg}>
                  {last >= 0 ? '+' : ''}{last.toFixed(1)}%
                </span>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}

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
  const [desk, setDesk] = useState<DeskFeed | null>(null);
  useEffect(() => {
    let off = false;
    const load = () =>
      apiGet<DeskFeed>(`/api/overview/desk?t=${Date.now()}`)
        .then((d) => { if (!off) setDesk(d); })
        .catch(() => undefined);          // absent until the next restart
    load();
    const id = setInterval(load, 60_000);
    return () => { off = true; clearInterval(id); };
  }, []);
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
          <h1 className={styles.title}>The Desk</h1>
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
          <Clock />
          <span className={styles.sep} />
          <span className={styles.stamp}>
            {now.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' })}
          </span>
        </div>
      </header>

      <Ticker />

      {/* -------------------------------------------------- desk band */}
      {desk && (
        <section className={styles.desk}>
          <div className={`${styles.deskCard} ${desk.exposure.naked_count ? styles.deskWarn : ''}`}>
            <div className={styles.deskK}>Exposed now</div>
            <div className={styles.deskV}>
              {desk.exposure.short_option_legs}
              <span className={styles.deskUnit}> short legs</span>
            </div>
            <div className={styles.deskS}>
              {desk.exposure.naked_count > 0
                ? `${desk.exposure.naked_count} with NO exchange stop`
                : 'all protected at the exchange'}
              {' · open '}
              <span className={desk.exposure.open_pnl >= 0 ? styles.pos : styles.neg}>
                {inr(desk.exposure.open_pnl)}
              </span>
            </div>
          </div>

          <div className={`${styles.deskCard} ${desk.recon.alerts ? styles.deskBad : ''}`}>
            <div className={styles.deskK}>Broker vs app</div>
            <div className={styles.deskV}>
              {desk.recon.clean ? 'match' : `${desk.recon.alerts} off`}
            </div>
            <div className={styles.deskS}>
              {desk.recon.warns} warning{desk.recon.warns === 1 ? '' : 's'}
              {desk.recon.age_mins != null ? ` · checked ${desk.recon.age_mins}m ago` : ''}
            </div>
          </div>

          <div className={styles.deskCard}>
            <div className={styles.deskK}>Trading today</div>
            <div className={styles.deskV}>
              {desk.gates.nas_matrix.live.length}
              <span className={styles.deskUnit}> live</span>
              <span className={styles.deskFaint}> / {desk.gates.nas_matrix.paper.length} paper</span>
            </div>
            <div className={styles.deskS}>
              master {desk.gates.nas_master_mode ?? '—'}
              {desk.gates.freeze_flag ? ' · FREEZE ON' : ''}
            </div>
          </div>

          <div className={`${styles.deskCard} ${desk.ops.overdue.length ? styles.deskWarn : ''}`}>
            <div className={styles.deskK}>Review queue</div>
            <div className={styles.deskV}>
              {desk.ops.overdue.length}
              <span className={styles.deskUnit}> overdue</span>
            </div>
            <div className={styles.deskS}>
              {desk.ops.overdue[0]?.title.slice(0, 44)
                ?? `${desk.ops.due_soon.length} due this week · ${desk.ops.tracked} tracked`}
            </div>
          </div>

          <div className={`${styles.deskCard} ${desk.health.summary?.fail ? styles.deskBad : ''}`}>
            <div className={styles.deskK}>Plumbing</div>
            <div className={styles.deskV}>
              {desk.health.summary
                ? `${desk.health.summary.ok}/${desk.health.summary.ok + desk.health.summary.warn + desk.health.summary.fail}`
                : '—'}
              <span className={styles.deskUnit}> ok</span>
            </div>
            <div className={styles.deskS}>
              token {desk.health.token_file_age_mins != null ? `${desk.health.token_file_age_mins}m old` : '—'}
              {' · '}
              {Object.values(desk.health.token_chain).every((c) => c.today)
                ? 'morning chain ran'
                : 'CHAIN MISSED'}
            </div>
          </div>
        </section>
      )}

      {desk && desk.exposure.naked_count > 0 && (
        <div className={styles.nakedStrip}>
          <b>{desk.exposure.naked_count} short option{desk.exposure.naked_count === 1 ? '' : 's'} with no stop resting at the exchange</b>
          {' — '}
          {desk.exposure.naked.slice(0, 6).join(', ')}
          {desk.exposure.naked.length > 6 ? ` +${desk.exposure.naked.length - 6}` : ''}
          {'. Software-side stops only: if the process stops, nothing protects these.'}
        </div>
      )}


      {/* -------------------------------------------------------- tape */}
      <section className={styles.metrics}>
        {(() => {
          const sessions = days.length || 1;
          // Calmar on rupee P&L: annualised net over the worst peak-to-trough.
          // Dimensionless, and labelled as such — there is no capital base here.
          const annual = (m.net / sessions) * 252;
          const calmar = m.mdd ? annual / Math.abs(m.mdd) : null;
          const cards = [
            { k: 'Net P&L', v: inr(m.net), c: tone(m.net),
              s: `${days.length} sessions · ${m.trades.toLocaleString('en-IN')} trades`, hero: true },
            { k: 'Max drawdown', v: inr(m.mdd, false), c: styles.neg, s: 'worst peak to trough' },
            { k: 'Calmar', v: calmar === null ? '—' : calmar.toFixed(2), c: tone(calmar ? calmar - 1 : 0),
              s: 'annualised ÷ max DD' },
            { k: 'Sharpe', v: m.sharpe === null ? '—' : m.sharpe.toFixed(2), c: tone(m.sharpe),
              s: 'daily ₹ P&L, annualised' },
            { k: 'Profit factor', v: m.pf === null ? '—' : m.pf.toFixed(2), c: tone(m.pf ? m.pf - 1 : 0),
              s: 'gross win ÷ gross loss' },
            { k: 'Winning days', v: pct(m.winDays), c: styles.flat, s: `of ${days.length} sessions` },
            { k: 'Best day', v: inr(m.best), c: styles.pos, s: 'single session' },
            { k: 'Worst day', v: inr(m.worst), c: styles.neg, s: 'single session' },
          ];
          return cards.map((card) => (
            <div key={card.k} className={`${styles.mCard} ${card.hero ? styles.mHero : ''}`}>
              <div className={styles.mK}>{card.k}</div>
              <div className={`${styles.mV} ${card.c}`}>{loading ? '·' : card.v}</div>
              <div className={styles.mS}>{card.s}</div>
            </div>
          ));
        })()}
      </section>

      {/* equity at half width, the books grid beside it — half the page
          height, and the two questions sit side by side */}
      <div className={styles.duo}>
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
        {loading ? <div className={styles.chartEmpty}>Loading…</div> : <EquityCandles days={days} />}
        {!loading && <Heartbeat days={days} />}
      </section>
        <Heatmaps />
      </div>

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
          <div className={styles.distHead}>
            <span className={styles.distTitle}>Daily P&amp;L distribution</span>
            <span className={styles.distMeta}>{days.length} sessions</span>
          </div>
          <Distribution days={days} />

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
