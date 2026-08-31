import { useEffect, useMemo, useRef, useState } from 'react';
import type { HoldingsRecord } from '../../api/types';
import styles from './HoldingsCharts.module.css';
import MomentumScanner from '../MomentumScanner/MomentumScanner';

// Phase 1 — renders every holding's 1-year daily chart on one page, off the
// `sparkline` (252 daily closes) already in /api/holdings/digest. No backend
// changes: the endpoint dot reflects the live LTP, which the parent Holdings
// page re-polls every 30s. Real OHLC candlesticks arrive in Phase 2 via a new
// /api/holdings/candles route (needs a backend restart → after market close).

function readVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || '#888780';
}

function sma(a: number[], w: number): (number | null)[] {
  const o: (number | null)[] = new Array(a.length).fill(null);
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    s += a[i];
    if (i >= w) s -= a[i - w];
    if (i >= w - 1) o[i] = s / w;
  }
  return o;
}

interface Series {
  sym: string;
  closes: number[];
  d50: (number | null)[];
  d200: (number | null)[];
  hi52: number;
  lo52: number;
  ath: number;
  ltp: number;
  day: number;
  d5: number;
  fromAth: number;
  ret: number;
  invested: number;
  current: number;
  pnl: number;
  qty: number;
  avg: number;
  prevClose: number;
}

function toSeries(r: HoldingsRecord, bars?: Bar[]): Series | null {
  // Prefer the digest sparkline; fall back to the OHLC file's closes when the
  // sparkline is empty (Yahoo occasionally returns nothing for a symbol at
  // digest-build time). This keeps every holding on the wall, not just the
  // ones Yahoo happened to serve.
  let closes: number[] | null = null;
  if (r.sparkline && r.sparkline.length >= 2) closes = r.sparkline.slice();
  else if (bars && bars.length >= 2) closes = bars.map((b) => b.c);
  if (!closes) return null;
  // Live endpoint: overlay the current LTP on the most recent point.
  if (Number.isFinite(r.ltp) && r.ltp > 0) closes[closes.length - 1] = r.ltp;
  const yr = closes.slice(-252);
  // ATH can never be below the current price — a positive "distance from ATH"
  // just means the stored all_time_high is stale (doesn't include the live
  // price). Take the max of recorded ATH, live LTP and the series, so distance
  // is always <= 0 and a fresh high reads as "at ATH".
  const ath = Math.max(r.all_time_high ?? 0, r.ltp, Math.max(...closes));
  return {
    sym: r.tradingsymbol,
    closes,
    d50: sma(closes, 50),
    d200: sma(closes, 200),
    hi52: r.week52_high ?? Math.max(...yr),
    lo52: r.week52_low ?? Math.min(...yr),
    ath,
    ltp: r.ltp,
    day: r.day_pct ?? 0,
    d5: r.change_5d_pct ?? 0,
    fromAth: ath > 0 ? Math.min(0, (r.ltp / ath - 1) * 100) : 0,
    ret: r.total_pnl_pct ?? 0,
    invested: r.invested ?? 0,
    current: r.current ?? 0,
    pnl: r.total_pnl_inr ?? 0,
    qty: r.qty ?? 0,
    avg: r.avg_price ?? 0,
    prevClose: r.prev_close ?? 0,
  };
}

interface Bar { t: string; o: number; h: number; l: number; c: number; v: number; stop?: number | null; }
type OhlcMap = Record<string, Bar[]>;

function ymd(d: Date): string {
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
}

// The OHLC file is regenerated once a day (after close), so intraday its last
// bar is yesterday's. Append a live "today" forming candle from the digest LTP
// (open ≈ prev close, high/low bracket the move) so the chart reflects today
// while the market is open. Skipped on weekends and once Yahoo already has today.
function augmentToday(bars: Bar[] | undefined, s: Series): Bar[] | undefined {
  if (!bars || bars.length < 2) return bars;
  if (!(Number.isFinite(s.ltp) && s.ltp > 0)) return bars;
  const today = ymd(new Date());
  const last = bars[bars.length - 1];
  if (last.t >= today) return bars;                 // already has today (or newer)
  const dow = new Date().getDay();
  if (dow === 0 || dow === 6) return bars;          // weekend — no synthetic bar
  // Guard: if the series is stale (its last close is wildly off from the live LTP),
  // splicing a live bar onto it draws a fake cliff. Skip rather than fabricate a spike.
  if (last.c > 0 && Math.abs(s.ltp / last.c - 1) > 0.20) return bars;
  const o = s.prevClose > 0 && Math.abs(s.ltp / s.prevClose - 1) <= 0.20 ? s.prevClose : last.c;
  return [...bars, { t: today, o, h: Math.max(o, s.ltp), l: Math.min(o, s.ltp), c: s.ltp, v: 0 }];
}

function roundTick(p: number): number { return Math.round(p / 0.05) * 0.05; }
interface ExecStep { t: string; price: number; action: string; }
interface ExecState {
  status: string; message: string; steps: ExecStep[];
  exec_id?: string; qty?: number; ref_ltp?: number; cap?: number; used?: number;
  paper?: boolean; filled_qty?: number; avg_price?: number | null;
}
const EXEC_LABEL: Record<string, string> = {
  working: 'Working…', filled: 'Filled ✓', partial: 'Partially filled',
  cancelled: 'Not filled', error: 'Rejected',
};
// Buy + exit are live once the account-aware order/exit/funds routes are deployed.
const BUY_ENABLED = true;

function fmtRs(n: number): string {
  const a = Math.abs(n);
  const sign = n < 0 ? '−' : '';
  if (a >= 1e7) return `${sign}₹${(a / 1e7).toFixed(2)}cr`;
  if (a >= 1e5) return `${sign}₹${(a / 1e5).toFixed(2)}L`;
  return `${sign}₹${Math.round(a).toLocaleString('en-IN')}`;
}
function pct(n: number, dp = 2): string {
  return `${n >= 0 ? '+' : ''}${n.toFixed(dp)}%`;
}
function upDn(n: number): string {
  return n >= 0 ? styles.up : styles.dn;
}

function setupCanvas(cv: HTMLCanvasElement): CanvasRenderingContext2D | null {
  const W = cv.clientWidth || 300;
  const H = cv.clientHeight || 96;
  if (W < 2) return null;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  cv.width = W * dpr;
  cv.height = H * dpr;
  const ctx = cv.getContext('2d');
  if (!ctx) return null;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

// ---------- mini area chart (wall tile) ----------
function drawMini(cv: HTMLCanvasElement, s: Series, win: number) {
  const x = setupCanvas(cv);
  if (!x) return;
  const W = cv.clientWidth, H = cv.clientHeight;
  const cl = s.closes.slice(-win);
  const d50 = s.d50.slice(-win), d200 = s.d200.slice(-win);
  const pad = 4, top = pad, bot = H - pad;
  let mn = Math.min(...cl), mx = Math.max(...cl);
  [s.hi52, s.lo52].forEach((v) => { if (v < mn) mn = v; if (v > mx) mx = v; });
  const gap = (mx - mn) * 0.08 || 1; mn -= gap; mx += gap;
  const px = (i: number) => pad + (i / (cl.length - 1)) * (W - 2 * pad);
  const py = (v: number) => bot - ((v - mn) / (mx - mn)) * (bot - top);
  const up = cl[cl.length - 1] >= cl[0];
  const pos = readVar('--accent-pos'), neg = readVar('--accent-neg');
  const line = up ? pos : neg;

  x.clearRect(0, 0, W, H);
  // fill
  x.beginPath();
  x.moveTo(px(0), py(cl[0]));
  cl.forEach((v, i) => x.lineTo(px(i), py(v)));
  x.lineTo(px(cl.length - 1), bot);
  x.lineTo(px(0), bot);
  x.closePath();
  const g = x.createLinearGradient(0, top, 0, bot);
  g.addColorStop(0, up ? 'rgba(15,110,86,0.16)' : 'rgba(163,45,45,0.14)');
  g.addColorStop(1, 'rgba(0,0,0,0)');
  x.fillStyle = g;
  x.fill();
  // 52w hi/lo rails
  x.setLineDash([2, 3]); x.lineWidth = 1; x.strokeStyle = readVar('--hairline');
  [s.hi52, s.lo52].forEach((v) => {
    if (v >= mn && v <= mx) { x.beginPath(); x.moveTo(pad, py(v)); x.lineTo(W - pad, py(v)); x.stroke(); }
  });
  x.setLineDash([]);
  // DMAs
  const dma = (arr: (number | null)[], col: string) => {
    x.strokeStyle = col; x.lineWidth = 1; x.globalAlpha = 0.85; x.beginPath();
    let started = false;
    arr.forEach((v, i) => {
      if (v == null) return;
      if (!started) { x.moveTo(px(i), py(v)); started = true; } else x.lineTo(px(i), py(v));
    });
    x.stroke(); x.globalAlpha = 1;
  };
  dma(d200, readVar('--brand-navy'));
  dma(d50, readVar('--brand-amber'));
  // price line
  x.strokeStyle = line; x.lineWidth = 1.6; x.lineJoin = 'round'; x.beginPath();
  x.moveTo(px(0), py(cl[0]));
  cl.forEach((v, i) => x.lineTo(px(i), py(v)));
  x.stroke();
  // endpoint
  x.fillStyle = line; x.beginPath(); x.arc(px(cl.length - 1), py(cl[cl.length - 1]), 3, 0, 7); x.fill();
  x.strokeStyle = readVar('--surface'); x.lineWidth = 1.5; x.stroke();
}

// ---------- big area chart (focus) ----------
function drawFocus(cv: HTMLCanvasElement, s: Series, win: number, hover: number | null) {
  const x = setupCanvas(cv);
  if (!x) return;
  const W = cv.clientWidth, H = cv.clientHeight;
  const cl = s.closes.slice(-win);
  const d50 = s.d50.slice(-win), d200 = s.d200.slice(-win);
  const padL = 6, padR = 54, padT = 10, padB = 22;
  const top = padT, bot = H - padB;
  let mn = Math.min(...cl), mx = Math.max(...cl);
  d50.concat(d200).forEach((v) => { if (v != null) { if (v < mn) mn = v; if (v > mx) mx = v; } });
  const showHi = s.hi52 <= mx * 1.02 && s.hi52 >= mn * 0.98;
  const showLo = s.lo52 <= mx * 1.02 && s.lo52 >= mn * 0.98;
  if (showHi) mx = Math.max(mx, s.hi52);
  if (showLo) mn = Math.min(mn, s.lo52);
  const gap = (mx - mn) * 0.05 || 1; mn -= gap; mx += gap;
  const px = (i: number) => padL + (i / (cl.length - 1)) * (W - padL - padR);
  const py = (v: number) => top + ((mx - v) / (mx - mn)) * (bot - top);
  const up = cl[cl.length - 1] >= cl[0];
  const pos = readVar('--accent-pos'), neg = readVar('--accent-neg');
  const line = up ? pos : neg;

  x.clearRect(0, 0, W, H);
  // ticker watermark (left)
  x.save();
  x.font = `700 22px ${readVar('--font-family')}`; x.textAlign = 'left';
  x.fillStyle = readVar('--ink'); x.globalAlpha = 0.08;
  x.fillText(s.sym, padL + 10, top + 26);
  x.restore();
  // grid + price axis
  x.strokeStyle = readVar('--hairline-soft'); x.fillStyle = readVar('--ink-faint');
  x.lineWidth = 1; x.font = `10px ${readVar('--font-family')}`; x.textAlign = 'left';
  for (let i = 0; i <= 4; i++) {
    const v = mn + (mx - mn) * i / 4; const yy = py(v);
    x.beginPath(); x.moveTo(padL, yy); x.lineTo(W - padR, yy); x.stroke();
    x.fillText(v.toFixed(v > 1000 ? 0 : 1), W - padR + 5, yy + 3);
  }
  // 52w hi/lo
  x.setLineDash([4, 3]); x.strokeStyle = readVar('--ink-faint');
  if (showHi) { x.beginPath(); x.moveTo(padL, py(s.hi52)); x.lineTo(W - padR, py(s.hi52)); x.stroke();
    x.fillStyle = readVar('--ink-muted'); x.fillText(`52wH ${s.hi52.toFixed(0)}`, padL + 3, py(s.hi52) - 3); }
  if (showLo) { x.beginPath(); x.moveTo(padL, py(s.lo52)); x.lineTo(W - padR, py(s.lo52)); x.stroke();
    x.fillStyle = readVar('--ink-muted'); x.fillText(`52wL ${s.lo52.toFixed(0)}`, padL + 3, py(s.lo52) + 11); }
  x.setLineDash([]);
  // fill
  x.beginPath(); x.moveTo(px(0), py(cl[0]));
  cl.forEach((v, i) => x.lineTo(px(i), py(v)));
  x.lineTo(px(cl.length - 1), bot); x.lineTo(px(0), bot); x.closePath();
  const g = x.createLinearGradient(0, top, 0, bot);
  g.addColorStop(0, up ? 'rgba(15,110,86,0.16)' : 'rgba(163,45,45,0.14)');
  g.addColorStop(1, 'rgba(0,0,0,0)');
  x.fillStyle = g; x.fill();
  // DMAs
  const dma = (arr: (number | null)[], col: string) => {
    x.strokeStyle = col; x.lineWidth = 1.4; x.globalAlpha = 0.9; x.beginPath();
    let started = false;
    arr.forEach((v, i) => {
      if (v == null) return;
      if (!started) { x.moveTo(px(i), py(v)); started = true; } else x.lineTo(px(i), py(v));
    });
    x.stroke(); x.globalAlpha = 1;
  };
  dma(d200, readVar('--brand-navy'));
  dma(d50, readVar('--brand-amber'));
  // price line
  x.strokeStyle = line; x.lineWidth = 1.8; x.lineJoin = 'round'; x.beginPath();
  x.moveTo(px(0), py(cl[0]));
  cl.forEach((v, i) => x.lineTo(px(i), py(v)));
  x.stroke();
  // last price tag
  const last = cl[cl.length - 1]; const yl = py(last);
  x.fillStyle = line; x.fillRect(W - padR, yl - 8, padR - 2, 16);
  x.fillStyle = readVar('--surface'); x.font = `700 10px ${readVar('--font-family')}`;
  x.fillText(last.toFixed(1), W - padR + 4, yl + 3);
  // crosshair
  if (hover != null && hover >= 0 && hover < cl.length) {
    const hx = px(hover), hy = py(cl[hover]);
    x.strokeStyle = readVar('--ink-faint'); x.setLineDash([3, 3]); x.lineWidth = 1;
    x.beginPath(); x.moveTo(hx, top); x.lineTo(hx, bot); x.stroke(); x.setLineDash([]);
    x.fillStyle = readVar('--ink'); x.beginPath(); x.arc(hx, hy, 3.5, 0, 7); x.fill();
    x.strokeStyle = readVar('--surface'); x.lineWidth = 1.5; x.stroke();
    const label = `${cl[hover].toFixed(1)}  ·  t-${cl.length - 1 - hover}d`;
    x.font = `600 11px ${readVar('--font-family')}`;
    const tw = x.measureText(label).width + 12;
    const bxx = Math.min(Math.max(hx - tw / 2, padL), W - padR - tw);
    x.fillStyle = readVar('--ink'); x.fillRect(bxx, top + 2, tw, 18);
    x.fillStyle = readVar('--surface'); x.textAlign = 'left'; x.fillText(label, bxx + 6, top + 14);
  }
}

// Viewport for the candle chart: `span` bars visible, `end` = float index at
// the right edge (may exceed n-1 to leave blank space so the latest candle can
// sit mid-chart). Clamped so you can't scroll the data fully off-screen.
interface View { span: number; end: number; }
function clampView(span: number, end: number, n: number): View {
  const sp = Math.max(8, Math.min(Math.round(span), n));
  const rightPad = sp * 0.6;               // how far past the last bar you can scroll
  const maxEnd = (n - 1) + rightPad;
  const minEnd = sp - 1;                    // keeps left edge >= 0
  return { span: sp, end: Math.max(minEnd, Math.min(end, maxEnd)) };
}

// ---------- big candlestick chart (focus, when OHLC available) ----------
interface Overlay { avg?: number; band?: [number, number]; sellZone?: [number, number]; }
function drawFocusCandles(cv: HTMLCanvasElement, bars: Bar[], s: Series, view: View, hover: number | null, overlay?: Overlay) {
  const x = setupCanvas(cv);
  if (!x) return;
  const W = cv.clientWidth, H = cv.clientHeight;
  const all = bars.map((b) => ({ ...b }));
  const n = all.length;
  if (n < 2) return;
  // live: overlay current LTP onto the most recent bar
  if (Number.isFinite(s.ltp) && s.ltp > 0) {
    const last = all[n - 1];
    last.c = s.ltp; last.h = Math.max(last.h, s.ltp); last.l = Math.min(last.l, s.ltp);
  }
  const closesAll = all.map((b) => b.c);
  const d50All = sma(closesAll, 50), d200All = sma(closesAll, 200);

  const span = Math.max(8, Math.min(view.span, n));
  const end = view.end;
  const left = end - (span - 1);
  const padL = 6, padR = 56, padT = 10, padB = 20;
  const volH = H * 0.15, top = padT, bot = H - padB - volH - 6;
  const plotW = W - padL - padR;
  const slot = plotW / span;
  const bw = Math.max(1, Math.min(slot * 0.68, 14));
  const px = (i: number) => padL + (i - left) * slot + slot / 2;

  const vs = Math.max(0, Math.floor(left));         // first visible real bar
  const ve = Math.min(n - 1, Math.ceil(end));       // last visible real bar
  if (ve < vs) return;
  let mn = Infinity, mx = -Infinity;
  for (let i = vs; i <= ve; i++) {
    if (all[i].l < mn) mn = all[i].l;
    if (all[i].h > mx) mx = all[i].h;
    const a = d50All[i], b = d200All[i];
    if (a != null) { if (a < mn) mn = a; if (a > mx) mx = a; }
    if (b != null) { if (b < mn) mn = b; if (b > mx) mx = b; }
  }
  const showHi = s.hi52 <= mx * 1.02 && s.hi52 >= mn * 0.98;
  const showLo = s.lo52 <= mx * 1.02 && s.lo52 >= mn * 0.98;
  if (showHi) mx = Math.max(mx, s.hi52);
  if (showLo) mn = Math.min(mn, s.lo52);
  const gap = (mx - mn) * 0.05 || 1; mn -= gap; mx += gap;
  const py = (v: number) => top + ((mx - v) / (mx - mn)) * (bot - top);
  let vmax = 1;
  for (let i = vs; i <= ve; i++) if (all[i].v > vmax) vmax = all[i].v;
  const vy = (v: number) => H - padB - (v / vmax) * volH;
  const pos = readVar('--accent-pos'), neg = readVar('--accent-neg');
  const font = readVar('--font-family');

  x.clearRect(0, 0, W, H);
  // ticker watermark (left)
  x.save();
  x.font = `700 22px ${font}`; x.textAlign = 'left';
  x.fillStyle = readVar('--ink'); x.globalAlpha = 0.08;
  x.fillText(s.sym, padL + 10, top + 26);
  x.restore();
  // grid + price axis
  x.strokeStyle = readVar('--hairline-soft'); x.fillStyle = readVar('--ink-faint');
  x.lineWidth = 1; x.font = `10px ${font}`; x.textAlign = 'left';
  for (let i = 0; i <= 4; i++) {
    const v = mn + (mx - mn) * i / 4, yy = py(v);
    x.beginPath(); x.moveTo(padL, yy); x.lineTo(W - padR, yy); x.stroke();
    x.fillText(v.toFixed(v > 1000 ? 0 : 1), W - padR + 5, yy + 3);
  }
  // 52w hi/lo
  x.setLineDash([4, 3]); x.strokeStyle = readVar('--ink-faint');
  if (showHi) { x.beginPath(); x.moveTo(padL, py(s.hi52)); x.lineTo(W - padR, py(s.hi52)); x.stroke();
    x.fillStyle = readVar('--ink-muted'); x.fillText(`52wH ${s.hi52.toFixed(0)}`, padL + 3, py(s.hi52) - 3); }
  if (showLo) { x.beginPath(); x.moveTo(padL, py(s.lo52)); x.lineTo(W - padR, py(s.lo52)); x.stroke();
    x.fillStyle = readVar('--ink-muted'); x.fillText(`52wL ${s.lo52.toFixed(0)}`, padL + 3, py(s.lo52) + 11); }
  x.setLineDash([]);
  // volume
  x.globalAlpha = 0.26;
  for (let i = vs; i <= ve; i++) {
    const b = all[i]; x.fillStyle = b.c >= b.o ? pos : neg;
    x.fillRect(px(i) - bw / 2, vy(b.v), bw, Math.max(1, H - padB - vy(b.v)));
  }
  x.globalAlpha = 1;
  // DMAs
  const dma = (arr: (number | null)[], col: string) => {
    x.strokeStyle = col; x.lineWidth = 1.4; x.globalAlpha = 0.9; x.beginPath();
    let started = false;
    for (let i = vs; i <= ve; i++) {
      const v = arr[i]; if (v == null) continue;
      if (!started) { x.moveTo(px(i), py(v)); started = true; } else x.lineTo(px(i), py(v));
    }
    x.stroke(); x.globalAlpha = 1;
  };
  // Donchian trailing stop — the momentum book's actual exit level (only present on that feed)
  if (all.some((b) => b.stop != null)) {
    x.strokeStyle = readVar('--accent-neg'); x.lineWidth = 1.5;
    x.setLineDash([5, 3]); x.beginPath();
    let st = false;
    for (let i = vs; i <= ve; i++) {
      const v = all[i].stop; if (v == null) continue;
      if (!st) { x.moveTo(px(i), py(v)); st = true; } else x.lineTo(px(i), py(v));
    }
    x.stroke(); x.setLineDash([]);
    const lastStop = all[ve]?.stop;
    if (lastStop != null) {
      x.fillStyle = readVar('--accent-neg'); x.font = `700 10px ${font}`; x.textAlign = 'left';
      x.fillText(`stop ${lastStop.toFixed(lastStop > 1000 ? 0 : 1)}`, padL + 3, py(lastStop) - 4);
    }
  }
  dma(d200All, readVar('--brand-navy'));
  dma(d50All, readVar('--brand-amber'));
  // candles
  for (let i = vs; i <= ve; i++) {
    const b = all[i]; const up = b.c >= b.o, col = up ? pos : neg;
    x.strokeStyle = col; x.fillStyle = col; x.lineWidth = 1;
    const cx = px(i);
    x.beginPath(); x.moveTo(cx, py(b.h)); x.lineTo(cx, py(b.l)); x.stroke();
    const yo = py(b.o), yc = py(b.c);
    x.fillRect(cx - bw / 2, Math.min(yo, yc), bw, Math.max(1, Math.abs(yc - yo)));
  }
  // ---- trade overlay: buy fill-band, sell realized-zone, avg-cost line ----
  if (overlay) {
    const clip = (v: number) => Math.max(top, Math.min(bot, py(v)));
    // buy smart-limit fill band (start..cap) — where a buy will land
    if (overlay.band) {
      const [b0, b1] = overlay.band;
      const y0 = clip(Math.min(b0, b1)), y1 = clip(Math.max(b0, b1));
      x.fillStyle = pos; x.globalAlpha = 0.12; x.fillRect(padL, y0, plotW, Math.max(2, y1 - y0)); x.globalAlpha = 1;
      x.strokeStyle = pos; x.globalAlpha = 0.5; x.setLineDash([4, 3]); x.lineWidth = 1;
      x.beginPath(); x.moveTo(padL, y0); x.lineTo(W - padR, y0); x.moveTo(padL, y1); x.lineTo(W - padR, y1); x.stroke();
      x.setLineDash([]); x.globalAlpha = 1;
      x.fillStyle = pos; x.font = `700 9px ${font}`; x.textAlign = 'left';
      x.fillText('buy fills here', padL + 4, y0 - 3);
    }
    // sell realized-P&L zone (avg..ltp)
    if (overlay.sellZone) {
      const [z0, z1] = overlay.sellZone;
      const profit = z1 >= z0;   // ltp >= avg
      const y0 = clip(Math.min(z0, z1)), y1 = clip(Math.max(z0, z1));
      x.fillStyle = profit ? pos : neg; x.globalAlpha = 0.1; x.fillRect(padL, y0, plotW, Math.max(2, y1 - y0)); x.globalAlpha = 1;
    }
    // average-cost line
    if (overlay.avg && overlay.avg > 0) {
      const ya = clip(overlay.avg);
      x.strokeStyle = readVar('--ink-muted'); x.setLineDash([2, 3]); x.lineWidth = 1;
      x.beginPath(); x.moveTo(padL, ya); x.lineTo(W - padR, ya); x.stroke(); x.setLineDash([]);
      x.fillStyle = readVar('--ink-muted'); x.font = `700 9px ${font}`; x.textAlign = 'left';
      x.fillText(`avg ${overlay.avg.toFixed(overlay.avg > 1000 ? 0 : 1)}`, padL + 4, ya + 10);
    }
  }
  // last price tag (real last bar, if in view vertically)
  const lb = all[n - 1], yl = py(lb.c);
  if (yl >= top && yl <= bot) {
    x.fillStyle = lb.c >= lb.o ? pos : neg; x.fillRect(W - padR, yl - 8, padR - 2, 16);
    x.fillStyle = readVar('--surface'); x.font = `700 10px ${font}`; x.textAlign = 'left';
    x.fillText(lb.c.toFixed(1), W - padR + 4, yl + 3);
  }
  // crosshair + OHLC label
  if (hover != null && hover >= vs && hover <= ve) {
    const b = all[hover], hx = px(hover);
    x.strokeStyle = readVar('--ink-faint'); x.setLineDash([3, 3]); x.lineWidth = 1;
    x.beginPath(); x.moveTo(hx, top); x.lineTo(hx, bot); x.stroke(); x.setLineDash([]);
    const up = b.c >= b.o;
    const label = `${b.t}   O ${b.o.toFixed(1)}  H ${b.h.toFixed(1)}  L ${b.l.toFixed(1)}  C ${b.c.toFixed(1)}`;
    x.font = `600 11px ${font}`;
    const tw = x.measureText(label).width + 14;
    const bxx = Math.min(Math.max(hx - tw / 2, padL), W - padR - tw);
    x.fillStyle = readVar('--ink'); x.fillRect(bxx, top + 2, tw, 18);
    x.fillStyle = up ? '#8ff0d8' : '#ffb3b1'; x.textAlign = 'left'; x.fillText(label, bxx + 7, top + 14.5);
  }
}

// ---------- reactive chart wrappers ----------
function MiniChart({ s, win }: { s: Series; win: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const draw = () => drawMini(cv, s, win);
    const raf = requestAnimationFrame(draw);
    const ro = new ResizeObserver(draw);
    ro.observe(cv);
    return () => { cancelAnimationFrame(raf); ro.disconnect(); };
  }, [s, win]);
  return <canvas ref={ref} className={styles.tileCanvas} />;
}

function FocusChart({ s, win, bars, overlay }: { s: Series; win: number; bars?: Bar[]; overlay?: Overlay }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [hover, setHover] = useState<number | null>(null);
  const [view, setView] = useState<View | null>(null);
  const dragRef = useRef<{ x: number; end: number; span: number } | null>(null);
  const hasCandles = !!bars && bars.length >= 2;
  const n = hasCandles ? (bars as Bar[]).length : 0;

  // reset viewport when the symbol or timeframe changes. Leave a small right
  // margin by default (~10% of the window) so the latest candle + price tag
  // aren't jammed against the axis.
  useEffect(() => {
    if (hasCandles) {
      const span = Math.min(win, n);
      setView(clampView(span, (n - 1) + span * 0.1, n));
    } else setView(null);
  }, [win, n, s.sym, hasCandles]);

  // draw
  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const draw = () => (hasCandles && view)
      ? drawFocusCandles(cv, bars as Bar[], s, view, hover, overlay)
      : drawFocus(cv, s, win, hover);
    const raf = requestAnimationFrame(draw);
    const ro = new ResizeObserver(draw);
    ro.observe(cv);
    return () => { cancelAnimationFrame(raf); ro.disconnect(); };
  }, [s, win, hover, bars, view, hasCandles, overlay]);

  // native, non-passive wheel zoom (anchored under the cursor)
  useEffect(() => {
    const cv = ref.current;
    if (!cv || !hasCandles || !view) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const r = cv.getBoundingClientRect();
      const W = cv.clientWidth, padL = 6, padR = 56;
      const slot = (W - padL - padR) / view.span;
      const left = view.end - (view.span - 1);
      const anchor = left + (e.clientX - r.left - padL) / slot;
      const factor = e.deltaY > 0 ? 1.15 : 1 / 1.15;
      const span = Math.max(8, Math.min(Math.round(view.span * factor), n));
      const end = anchor + (view.end - anchor) * (span / view.span);
      setView(clampView(span, end, n));
    };
    cv.addEventListener('wheel', onWheel, { passive: false });
    return () => cv.removeEventListener('wheel', onWheel);
  }, [view, n, hasCandles]);

  const onDown = (e: React.MouseEvent) => {
    if (hasCandles && view) dragRef.current = { x: e.clientX, end: view.end, span: view.span };
  };
  const endDrag = () => { dragRef.current = null; };
  const onMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const cv = ref.current; if (!cv) return;
    const r = cv.getBoundingClientRect();
    const W = cv.clientWidth, padL = 6;
    const padR = hasCandles ? 56 : 54;
    if (hasCandles && view) {
      if (dragRef.current) {
        const slot = (W - padL - padR) / dragRef.current.span;
        const dx = e.clientX - dragRef.current.x;
        setView(clampView(view.span, dragRef.current.end - dx / slot, n));
        return;
      }
      const slot = (W - padL - padR) / view.span;
      const left = view.end - (view.span - 1);
      const i = Math.round(left + (e.clientX - r.left - padL) / slot - 0.5);
      setHover(Math.max(0, Math.min(n - 1, i)));
    } else {
      const nn = Math.min(win, s.closes.length);
      const frac = (e.clientX - r.left - padL) / (W - padL - padR);
      setHover(Math.max(0, Math.min(nn - 1, Math.round(frac * (nn - 1)))));
    }
  };

  const zoomBy = (f: number) => {
    if (!view) return;
    const center = view.end - (view.span - 1) / 2;
    const span = Math.max(8, Math.min(Math.round(view.span * f), n));
    setView(clampView(span, center + (span - 1) / 2, n));
  };
  const centerLatest = () => { if (view) setView(clampView(view.span, (n - 1) + view.span / 2, n)); };

  return (
    <div className={styles.focusWrap}>
      {hasCandles && (
        <div className={styles.chartTools}>
          <button onClick={() => zoomBy(1.3)} title="Zoom out (show more)">−</button>
          <button onClick={centerLatest} title="Center the latest candle">⊙ latest</button>
          <button onClick={() => zoomBy(1 / 1.3)} title="Zoom in (show fewer)">+</button>
          <span className={styles.toolsHint}>scroll = zoom · drag = pan · dbl-click = center latest</span>
        </div>
      )}
      <canvas
        ref={ref}
        className={styles.focusCanvas}
        onMouseDown={onDown}
        onMouseUp={endDrag}
        onMouseMove={onMove}
        onMouseLeave={() => { setHover(null); endDrag(); }}
        onDoubleClick={centerLatest}
        style={{ cursor: hasCandles ? 'grab' : 'default' }}
      />
    </div>
  );
}

// ---------- main ----------
const FILTERS: [string, string][] = [
  ['all', 'All'], ['win', 'Winners'], ['lose', 'Losers'],
  ['ath', 'Near ATH'], ['low', 'At 52w-low'], ['neg', 'Underwater'],
];
const TFS: [string, number][] = [['3M', 63], ['6M', 126], ['1Y', 252]];

// Watchlist flags — persisted in localStorage (per-browser; no backend).
const FLAGS = ['green', 'yellow', 'red'] as const;
type FlagColor = typeof FLAGS[number];
const FLAG_HEX: Record<FlagColor, string> = { green: '#16a34a', yellow: '#ca8a04', red: '#dc2626' };
const FLAG_RANK: Record<FlagColor, number> = { green: 0, yellow: 1, red: 2 };
function loadFlags(): Record<string, FlagColor> {
  try { return JSON.parse(localStorage.getItem('holdingsFlags') || '{}'); } catch { return {}; }
}
function saveFlags(f: Record<string, FlagColor>) {
  try { localStorage.setItem('holdingsFlags', JSON.stringify(f)); } catch { /* ignore */ }
}

export default function HoldingsCharts({ holdings, ohlcUrl, ohlcUrls, account = 'me' }: { holdings: HoldingsRecord[]; ohlcUrl?: string; ohlcUrls?: string[]; account?: 'me' | 'dad' | 'both' }) {
  const [view, setView] = useState<'wall' | 'focus'>('wall');
  const [sort, setSort] = useState('day');
  const [filter, setFilter] = useState('all');
  const [tf, setTf] = useState(252);
  const [density, setDensity] = useState<'comfy' | 'compact'>('comfy');
  const [currentSym, setCurrentSym] = useState<string | null>(null);
  const [ohlc, setOhlc] = useState<OhlcMap>({});
  const [flags, setFlags] = useState<Record<string, FlagColor>>(loadFlags);
  const setFlag = (sym: string, color: FlagColor) => setFlags((prev) => {
    const next = { ...prev };
    if (next[sym] === color) delete next[sym]; else next[sym] = color; // click active = unflag
    saveFlags(next);
    return next;
  });
  const clearFlag = (sym: string) => setFlags((prev) => {
    const next = { ...prev }; delete next[sym]; saveFlags(next); return next;
  });

  // Daily OHLC for candlesticks — a static file (static/holdings_ohlc.json)
  // regenerated out-of-band, same pattern as nifty_5m.json. No backend route.
  useEffect(() => {
    let on = true;
    setOhlc({});
    const urls = ohlcUrls && ohlcUrls.length ? ohlcUrls : [ohlcUrl || "/static/holdings_ohlc.json"];
    const bust = Math.floor(Date.now() / 300000);
    Promise.all(urls.map((u) => fetch(`${u}?t=${bust}`, { cache: 'no-store' })
      .then((r) => (r.ok ? r.json() : null)).catch(() => null)))
      .then((list) => {
        if (!on) return;
        const merged: OhlcMap = {};
        list.forEach((d) => { if (d && d.symbols) Object.assign(merged, d.symbols as OhlcMap); });
        setOhlc(merged);
      });
    return () => { on = false; };
  }, [ohlcUrl, (ohlcUrls || []).join(',')]);

  const allSeries = useMemo(
    () => holdings.map((r) => toSeries(r, ohlc[r.tradingsymbol])).filter((s): s is Series => s !== null),
    [holdings, ohlc],
  );

  const list = useMemo(() => {
    const f = allSeries.filter((d) => {
      if (filter === 'win') return d.ret > 0;
      if (filter === 'lose') return d.day < 0;
      if (filter === 'ath') return d.fromAth > -4;
      if (filter === 'low') return (d.ltp / d.lo52 - 1) * 100 < 5;
      if (filter === 'neg') return d.ret < 0;
      return true;
    });
    f.sort((p, q) => {
      if (sort === 'flag') {
        const pr = flags[p.sym] != null ? FLAG_RANK[flags[p.sym]] : 3;
        const qr = flags[q.sym] != null ? FLAG_RANK[flags[q.sym]] : 3;
        return pr !== qr ? pr - qr : q.day - p.day;
      }
      return sort === 'az' ? p.sym.localeCompare(q.sym)
        : sort === 'pnl' ? q.ret - p.ret
        : sort === 'ath' ? q.fromAth - p.fromAth
        : sort === 'val' ? q.current - p.current
        : q.day - p.day;
    });
    return f;
  }, [allSeries, filter, sort, flags]);

  const current = useMemo(() => {
    if (!list.length) return null;
    return list.find((d) => d.sym === currentSym) ?? list[0];
  }, [list, currentSym]);

  const step = (dir: number) => {
    if (!list.length || !current) return;
    const i = list.findIndex((d) => d.sym === current.sym);
    const next = list[(i + dir + list.length) % list.length];
    setCurrentSym(next.sym);
  };

  useEffect(() => {
    if (view !== 'focus') return;
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
      const kk = e.key.toLowerCase();
      if (kk === 't' || (e.altKey && kk === 'b')) { e.preventDefault(); setDockOpen((o) => !o); return; }  // toggle trade dock
      if (e.key === 'ArrowRight') { e.preventDefault(); step(1); }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); step(-1); }
      else if (e.key === 'Escape') setView('wall');
      else if (current) {
        const k = e.key.toLowerCase();
        if (k === 'g') setFlag(current.sym, 'green');
        else if (k === 'y') setFlag(current.sym, 'yellow');
        else if (k === 'r') setFlag(current.sym, 'red');
        else if (k === 'u') clearFlag(current.sym);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [view, list, current]);

  const openFocus = (sym: string) => { setCurrentSym(sym); setView('focus'); };

  // ---- buy / sell state ----
  const [amt, setAmt] = useState<number | null>(null);
  const [buyQtyOverride, setBuyQtyOverride] = useState<number | null>(null);
  const [customAmt, setCustomAmt] = useState('');
  const [sellCustomQty, setSellCustomQty] = useState('');
  const [sellPct, setSellPct] = useState<number | null>(null);
  const [orderSide, setOrderSide] = useState<'buy' | 'sell'>('buy');
  const [dockOpen, setDockOpen] = useState(false);
  const [paper, setPaper] = useState(false);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [exec, setExec] = useState<ExecState | null>(null);
  const [placing, setPlacing] = useState(false);
  const pollRef = useRef<number | undefined>(undefined);

  // ---- funds (per account) ----
  const [funds, setFunds] = useState<{ available: number; cash: number; live_balance: number; error?: string } | null>(null);
  const loadFunds = () => {
    if (account === 'both') return;  // trading is single-account; no funds panel here
    fetch(`/api/holdings/funds?account=${account}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => { if (d) setFunds(d); })
      .catch(() => {});
  };
  useEffect(() => { setFunds(null); loadFunds(); /* eslint-disable-next-line */ }, [account]);

  const curSym = current?.sym;
  useEffect(() => {
    // reset both order forms whenever the focused symbol (or account) changes
    setAmt(null); setCustomAmt(''); setBuyQtyOverride(null); setSellCustomQty(''); setSellPct(null);
    setConfirmOpen(false); setExec(null); setPlacing(false);
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = undefined; }
  }, [curSym, account]);
  useEffect(() => () => { if (pollRef.current) clearInterval(pollRef.current); }, []);

  const buyAmount = customAmt ? Number(customAmt) : (amt ?? 0);
  // qty from the chosen amount, unless the user hand-scaled it (stepper override)
  const buyQtyAuto = current && current.ltp > 0 && buyAmount > 0 ? Math.floor(buyAmount / current.ltp) : 0;
  const buyQty = buyQtyOverride != null ? buyQtyOverride : buyQtyAuto;
  const buyUsed = current ? buyQty * current.ltp : 0;
  const buyLeft = buyAmount - buyUsed;
  const setBuyAmt = (a: number) => { setAmt(a); setCustomAmt(''); setBuyQtyOverride(null); };
  const bumpQty = (d: number) => setBuyQtyOverride(Math.max(1, (buyQty || 0) + d));
  const startP = current ? roundTick(current.ltp * (1 - 0.002)) : 0;
  const capP = current ? roundTick(current.ltp * (1 + 0.003)) : 0;
  const inr0 = (n: number) => Math.round(n).toLocaleString('en-IN');

  // sell qty: a % chip of the holding, or a custom share count, clamped to holding
  const sellQty = (() => {
    if (!current) return 0;
    const raw = sellCustomQty ? Number(sellCustomQty) : (sellPct != null ? Math.floor(current.qty * sellPct / 100) : 0);
    return Math.max(0, Math.min(current.qty, Math.floor(raw || 0)));
  })();
  const sellValue = current ? sellQty * current.ltp : 0;
  const sellRealized = current ? sellQty * (current.ltp - current.avg) : 0;

  // what the chart draws for the order in progress: avg line always, buy fill-band
  // when an amount is picked, realized-P&L zone when a sell qty is picked
  const overlay: Overlay | undefined = current ? {
    avg: current.avg,
    band: buyQty >= 1 ? [startP, capP] : undefined,
    sellZone: sellQty >= 1 ? [current.avg, current.ltp] : undefined,
  } : undefined;

  const openConfirm = (side: 'buy' | 'sell') => { setOrderSide(side); setExec(null); setConfirmOpen(true); };

  const submitOrder = async () => {
    if (!current) return;
    const isSell = orderSide === 'sell';
    if (isSell ? sellQty < 1 : buyQty < 1) return;
    setPlacing(true);
    try {
      const url = isSell ? '/api/holdings/exit' : '/api/holdings/order';
      const body = isSell
        ? { symbol: current.sym, qty: sellQty, paper, account }
        : { symbol: current.sym, amount: buyAmount, qty: buyQty, paper, account };
      const r = await fetch(url, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const j = await r.json();
      if (j.error) { setExec({ status: 'error', message: j.error, steps: [] }); return; }
      setExec({ status: 'working', message: 'placing…', steps: [], ...j });
      if (pollRef.current) clearInterval(pollRef.current);
      pollRef.current = window.setInterval(async () => {
        try {
          const sr = await fetch(`/api/holdings/order/${j.exec_id}`);
          const s: ExecState = await sr.json();
          setExec(s);
          if (['filled', 'partial', 'cancelled', 'error'].includes(s.status)) {
            if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = undefined; }
            loadFunds();  // refresh available funds after a fill
          }
        } catch { /* keep polling */ }
      }, 1500);
    } catch {
      setExec({ status: 'error', message: 'network error', steps: [] });
    } finally {
      setPlacing(false);
    }
  };

  return (
    <div className={styles.root}>
      <div className={styles.controls}>
        <div className={styles.sortSel}>
          Sort
          <select value={sort} onChange={(e) => setSort(e.target.value)}>
            <option value="day">Day %</option>
            <option value="pnl">Total P&amp;L %</option>
            <option value="ath">Distance from ATH</option>
            <option value="val">Current value</option>
            <option value="az">A–Z</option>
            <option value="flag">Flag (🟢🟡🔴)</option>
          </select>
        </div>
        <div className={styles.chips}>
          {FILTERS.map(([k, label]) => (
            <button
              key={k}
              className={`${styles.chip} ${filter === k ? styles.chipOn : ''}`}
              onClick={() => setFilter(k)}
            >{label}</button>
          ))}
        </div>
        <div className={styles.spring} />
        <div className={styles.seg}>
          {TFS.map(([label, v]) => (
            <button key={v} className={`${styles.segBtn} ${tf === v ? styles.segOn : ''}`} onClick={() => setTf(v)}>{label}</button>
          ))}
        </div>
        {view === 'wall' && (
          <div className={styles.seg}>
            <button className={`${styles.segBtn} ${density === 'comfy' ? styles.segOn : ''}`} onClick={() => setDensity('comfy')}>Comfy</button>
            <button className={`${styles.segBtn} ${density === 'compact' ? styles.segOn : ''}`} onClick={() => setDensity('compact')}>Compact</button>
          </div>
        )}
        <div className={styles.seg}>
          <button className={`${styles.segBtn} ${view === 'wall' ? styles.segOn : ''}`} onClick={() => setView('wall')}>Wall</button>
          <button className={`${styles.segBtn} ${view === 'focus' ? styles.segOn : ''}`} onClick={() => setView('focus')}>Focus &amp; browse</button>
        </div>
        <span className={styles.cnt}>{list.length} shown</span>
      </div>

      {list.length === 0 ? (
        <div className={styles.empty}>No holdings match this filter.</div>
      ) : view === 'wall' ? (
        <div className={`${styles.wall} ${density === 'compact' ? styles.wallCompact : ''}`}>
          {list.map((d) => (
            <div
              key={d.sym}
              className={styles.tile}
              tabIndex={0}
              role="button"
              onClick={() => openFocus(d.sym)}
              onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openFocus(d.sym); } }}
            >
              <div className={styles.tTop}>
                <div className={styles.tSym}>
                  {flags[d.sym] ? <span className={styles.flagDot} style={{ background: FLAG_HEX[flags[d.sym]] }} /> : null}
                  {d.sym}
                </div>
                <div>
                  <div className={styles.tLtp}>{d.ltp.toFixed(1)}</div>
                  <div className={`${styles.tDay} ${upDn(d.day)}`}>{pct(d.day)}</div>
                </div>
              </div>
              <MiniChart s={d} win={tf} />
              <div className={styles.tFoot}>
                <span>P&amp;L <span className={`${styles.pnlPill} ${d.ret >= 0 ? styles.upBg : styles.dnBg}`}>{pct(d.ret, 0)}</span></span>
                <span>ATH <b className={upDn(d.fromAth)}>{pct(d.fromAth, 1)}</b></span>
                <span>5d <b className={upDn(d.d5)}>{pct(d.d5, 1)}</b></span>
              </div>
            </div>
          ))}
        </div>
      ) : current ? (
        <div className={styles.focus}>
          <div className={`${styles.panel} ${styles.rail}`}>
            <div className={styles.railH}>{list.length} holdings</div>
            {list.map((h) => {
              const rv = sort === 'pnl' ? pct(h.ret, 0) : sort === 'ath' ? pct(h.fromAth, 0) : sort === 'val' ? fmtRs(h.current) : pct(h.day);
              const rvn = sort === 'pnl' ? h.ret : sort === 'ath' ? h.fromAth : h.day;
              return (
                <div
                  key={h.sym}
                  className={`${styles.rRow} ${h.sym === current.sym ? styles.rRowOn : ''}`}
                  onClick={() => setCurrentSym(h.sym)}
                >
                  <span className={styles.rs}>
                    {flags[h.sym] ? <span className={styles.flagDot} style={{ background: FLAG_HEX[flags[h.sym]] }} /> : null}
                    {h.sym}
                  </span>
                  <span className={`${styles.rv} ${upDn(rvn)}`}>{rv}</span>
                </div>
              );
            })}
          </div>

          <div className={`${styles.panel} ${styles.stage}`}>
            <div className={styles.stHead}>
              <div style={{ order: 2, textAlign: 'right' }}>
                <div className={styles.stTitle}>{current.sym}</div>
                <div className={styles.stSub}>{current.qty} sh @ ₹{current.avg.toFixed(0)} · daily {(ohlc[current.sym]?.length ?? 0) >= 2 ? 'candles' : 'close'}</div>
                <div className={styles.flagRow} style={{ justifyContent: 'flex-end' }}>
                  <span className={styles.flagLbl}>Flag</span>
                  {FLAGS.map((c) => (
                    <button
                      key={c}
                      className={`${styles.flagBtn} ${flags[current.sym] === c ? styles.flagBtnOn : ''}`}
                      onClick={() => setFlag(current.sym, c)}
                      title={`${c} flag (press ${c[0].toUpperCase()})`}
                    >
                      <span className={styles.flagIco} style={{ background: FLAG_HEX[c] }} />
                    </button>
                  ))}
                  {flags[current.sym] ? (
                    <button className={styles.flagClearBtn} onClick={() => clearFlag(current.sym)} title="Unflag (press U)">clear</button>
                  ) : null}
                </div>
              </div>
              <div style={{ textAlign: 'left', order: 1 }}>
                <div><span className={styles.stPrice}>{current.ltp.toFixed(1)}</span><span className={`${styles.stDay} ${upDn(current.day)}`}>{pct(current.day)}</span></div>
                <div className={styles.navBtns} style={{ justifyContent: 'flex-start' }}>
                  <button onClick={() => step(-1)} title="Previous (←)">‹</button>
                  <button onClick={() => step(1)} title="Next (→)">›</button>
                </div>
              </div>
            </div>
            <div className={styles.metricStrip} style={{ justifyContent: 'flex-end' }}>
              <div className={styles.metric}><span className={styles.mLabel}>Invested</span><span className={styles.mVal}>{fmtRs(current.invested)}</span></div>
              <div className={styles.metric}><span className={styles.mLabel}>Current</span><span className={styles.mVal}>{fmtRs(current.current)}</span></div>
              <div className={styles.metric}><span className={styles.mLabel}>Unrealized P&amp;L</span><span className={`${styles.mVal} ${upDn(current.pnl)}`}>{fmtRs(current.pnl)} · {pct(current.ret, 1)}</span></div>
              <div className={styles.metric}><span className={styles.mLabel}>Qty · Avg</span><span className={styles.mVal}>{current.qty} · ₹{current.avg.toFixed(0)}</span></div>
            </div>
            <div className={styles.chartArea}>
              <FocusChart s={current} win={tf} bars={augmentToday(ohlc[current.sym], current)} overlay={overlay} />

              {BUY_ENABLED && account !== 'both' && (
              <div className={`${styles.tradeDock} ${dockOpen ? '' : styles.tradeDockMin}`}>
                <div className={styles.dockHead} onClick={() => setDockOpen((o) => !o)}>
                  <span className={styles.dockChevron}>{dockOpen ? '▾' : '▸'}</span>
                  <span className={styles.dockTitle}>Trade {current.sym}</span>
                  <span className={styles.dockAcct}>{account === 'dad' ? 'Stanly' : 'Me'}</span>
                  <span className={styles.dockFunds} title="Cash available to deploy">{funds ? (funds.error ? '—' : fmtRs(funds.live_balance)) : '…'}</span>
                  {!dockOpen && <span className={styles.dockKbd} title="Shortcut: T (or Alt+B)">T</span>}
                </div>
                {dockOpen && (
                <div className={styles.dockBody}>
                  <label className={styles.paperTog}>
                    <input type="checkbox" checked={paper} onChange={(e) => setPaper(e.target.checked)} />
                    Paper mode
                  </label>

                  <div className={styles.tradeSub}>Buy · smart limit</div>
                  <div className={styles.amtRow}>
                    {[5000, 10000, 15000].map((a) => (
                      <button
                        key={a}
                        className={`${styles.amtChip} ${!customAmt && amt === a ? styles.amtChipOn : ''}`}
                        onClick={() => setBuyAmt(a)}
                      >₹{(a / 1000)}k</button>
                    ))}
                    <input
                      className={styles.amtInput}
                      type="number"
                      min={0}
                      placeholder="Custom ₹"
                      value={customAmt}
                      onChange={(e) => { setCustomAmt(e.target.value); setAmt(null); setBuyQtyOverride(null); }}
                    />
                  </div>
                  {/* manual qty scaler — nudge share count up/down past the amount chip */}
                  <div className={styles.qtyStep}>
                    <button className={styles.stepBtn} onClick={() => bumpQty(-1)} disabled={buyQty < 2}>−</button>
                    <input
                      className={styles.qtyNum}
                      type="number" min={1}
                      value={buyQty || ''}
                      placeholder="0"
                      onChange={(e) => setBuyQtyOverride(Math.max(1, Math.floor(Number(e.target.value) || 0)))}
                    />
                    <span className={styles.qtyUnit}>sh</span>
                    <button className={styles.stepBtn} onClick={() => bumpQty(1)} disabled={buyQty < 1}>+</button>
                    <span className={styles.qtyCost}>
                      {buyQty >= 1
                        ? <>₹{inr0(buyUsed)}{buyQtyOverride != null ? <span className={styles.buyLeft}> · scaled</span> : null}</>
                        : <span className={styles.buyLeft}>{buyAmount > 0 ? `below 1 sh (₹${current.ltp.toFixed(1)})` : 'pick an amount →'}</span>}
                    </span>
                  </div>
                  <button className={styles.reviewBtn} disabled={buyQty < 1} onClick={() => openConfirm('buy')}>
                    Review buy →
                  </button>

                  <div className={styles.tradeSub} style={{ marginTop: 12 }}>Sell · market · hold {current.qty}</div>
                  <div className={styles.amtRow}>
                    {[25, 50, 100].map((p) => (
                      <button
                        key={p}
                        className={`${styles.amtChip} ${!sellCustomQty && sellPct === p ? styles.amtChipOn : ''}`}
                        onClick={() => { setSellPct(p); setSellCustomQty(''); }}
                      >{p === 100 ? 'All' : `${p}%`}</button>
                    ))}
                    <input
                      className={styles.amtInput}
                      type="number"
                      min={0}
                      max={current.qty}
                      placeholder="Qty"
                      value={sellCustomQty}
                      onChange={(e) => { setSellCustomQty(e.target.value); setSellPct(null); }}
                    />
                  </div>
                  <div className={styles.buyCalc}>
                    {sellQty >= 1 ? (
                      <>sell <b>{sellQty} sh</b> ≈ ₹{inr0(sellValue)} <span className={sellRealized >= 0 ? styles.up : styles.dn}>({sellRealized >= 0 ? '+' : '−'}₹{inr0(Math.abs(sellRealized))})</span></>
                    ) : (
                      <span className={styles.buyLeft}>pick a quantity →</span>
                    )}
                  </div>
                  <button className={`${styles.reviewBtn} ${styles.sellBtn}`} disabled={sellQty < 1} onClick={() => openConfirm('sell')}>
                    Review sell →
                  </button>
                </div>
                )}
              </div>
              )}
            </div>
            <div className={styles.leg}>
              <span><i style={{ borderColor: 'var(--brand-amber)' }} />50-DMA</span>
              <span><i style={{ borderColor: 'var(--brand-navy)' }} />200-DMA</span>
              <span><i style={{ borderColor: 'var(--ink-faint)', borderTopStyle: 'dashed' }} />52-wk hi / lo</span>
              <span><i style={{ borderColor: 'var(--ink-muted)', borderTopStyle: 'dotted' }} />avg cost</span>
            </div>
            <div className={styles.kHint}>
              <span className={styles.kbd}><b>←</b><b>→</b></span> browse holdings
              <span className={styles.kbd}><b>T</b></span> trade
              <span className={styles.kbd}><b>Esc</b></span> back to wall
              <span style={{ color: 'var(--ink-faint)' }}>· the dock trades the on-screen stock; the green band shows where a buy fills</span>
            </div>
          </div>

          <div className={`${styles.panel} ${styles.stats}`}>
            <h4>Position</h4>
            <div className={styles.sRow}><span className={styles.k}>Last price</span><span className={styles.v}>{current.ltp.toFixed(2)}</span></div>
            <div className={styles.sRow}><span className={styles.k}>Day</span><span className={`${styles.v} ${upDn(current.day)}`}>{pct(current.day)}</span></div>
            <div className={styles.sRow}><span className={styles.k}>Total P&amp;L</span><span className={`${styles.v} ${upDn(current.pnl)}`}>{fmtRs(current.pnl)} · {pct(current.ret, 1)}</span></div>
            <div className={styles.sRow}><span className={styles.k}>Invested</span><span className={styles.v}>{fmtRs(current.invested)}</span></div>
            <div className={styles.sRow}><span className={styles.k}>Current</span><span className={styles.v}>{fmtRs(current.current)}</span></div>
            <div className={styles.sRow}><span className={styles.k}>From ATH</span><span className={`${styles.v} ${upDn(current.fromAth)}`}>{pct(current.fromAth, 1)}</span></div>
            <h4 style={{ marginTop: 16 }}>52-week range</h4>
            <div className={styles.rangeBar}>
              <div className={styles.rangeMk} style={{ left: `${Math.max(0, Math.min(100, (current.ltp - current.lo52) / (current.hi52 - current.lo52) * 100))}%` }} />
            </div>
            <div className={styles.rangeLbl}><span>{current.lo52.toFixed(0)}</span><span>now {current.ltp.toFixed(0)}</span><span>{current.hi52.toFixed(0)}</span></div>
            <h4 style={{ marginTop: 16 }}>Trend</h4>
            {(() => {
              const l50 = current.d50[current.d50.length - 1];
              const l200 = current.d200[current.d200.length - 1];
              return (
                <>
                  <div className={styles.sRow}><span className={styles.k}>50-DMA</span><span className={styles.v}>{l50 != null ? l50.toFixed(0) : '—'} <span className={current.ltp >= (l50 ?? current.ltp) ? styles.up : styles.dn}>{l50 != null ? (current.ltp >= l50 ? 'above' : 'below') : ''}</span></span></div>
                  <div className={styles.sRow}><span className={styles.k}>200-DMA</span><span className={styles.v}>{l200 != null ? l200.toFixed(0) : '—'} <span className={current.ltp >= (l200 ?? current.ltp) ? styles.up : styles.dn}>{l200 != null ? (current.ltp >= l200 ? 'above' : 'below') : ''}</span></span></div>
                  <div className={styles.sRow}><span className={styles.k}>5-day</span><span className={`${styles.v} ${upDn(current.d5)}`}>{pct(current.d5, 1)}</span></div>
                </>
              );
            })()}
          </div>
        </div>
      ) : null}

      {confirmOpen && current && (
        <div className={styles.modalWrap} onClick={() => { if (!exec) setConfirmOpen(false); }}>
          <div className={styles.modal} onClick={(e) => e.stopPropagation()}>
            <div className={styles.modalHead}>
              <span>Confirm {orderSide === 'sell' ? 'sell' : 'buy'} · {account === 'dad' ? "Stanly's account" : 'My account'}</span>
              <span className={paper ? styles.paperBadge : styles.liveBadge}>{paper ? 'PAPER' : 'LIVE'}</span>
            </div>
            {!exec ? (
              <>
                {orderSide === 'sell' ? (
                  <>
                    <div className={styles.modalBig} style={{ color: 'var(--accent-neg)' }}>SELL {sellQty} × {current.sym}</div>
                    <div className={styles.modalRows}>
                      <div><span>Product</span><b>CNC · delivery</b></div>
                      <div><span>Order</span><b>MARKET — fills immediately</b></div>
                      <div><span>Est. proceeds</span><b>≈ ₹{inr0(sellValue)} @ ₹{current.ltp.toFixed(1)}</b></div>
                      <div><span>Realized P&amp;L</span><b className={sellRealized >= 0 ? styles.up : styles.dn}>{sellRealized >= 0 ? '+' : '−'}₹{inr0(Math.abs(sellRealized))}</b></div>
                      <div><span>Holding after</span><b>{current.qty - sellQty} sh{current.qty - sellQty === 0 ? ' · fully exited' : ''}</b></div>
                    </div>
                  </>
                ) : (
                  <>
                    <div className={styles.modalBig}>BUY {buyQty} × {current.sym}</div>
                    <div className={styles.modalRows}>
                      <div><span>Product</span><b>CNC · delivery</b></div>
                      <div><span>Smart limit</span><b>start ₹{startP.toFixed(2)} → cap ₹{capP.toFixed(2)}</b></div>
                      <div><span>Execution</span><b>shave, then chase to fill (~20s)</b></div>
                      <div><span>Est. cost</span><b>₹{inr0(buyUsed)}</b></div>
                    </div>
                  </>
                )}
                <div className={styles.modalBtns}>
                  <button className={styles.btnGhost} onClick={() => setConfirmOpen(false)}>Cancel</button>
                  <button
                    className={`${styles.btnGo} ${paper ? '' : (orderSide === 'sell' ? styles.btnGoSell : styles.btnGoLive)}`}
                    disabled={placing}
                    onClick={submitOrder}
                  >{placing ? 'Placing…' : paper ? 'Place paper order' : orderSide === 'sell' ? 'Sell at market' : 'Place order'}</button>
                </div>
              </>
            ) : (
              <div className={styles.execBox}>
                <div className={`${styles.execStatus} ${styles['ex_' + exec.status] ?? ''}`}>
                  {EXEC_LABEL[exec.status] ?? exec.status}
                </div>
                <div className={styles.execMsg}>{exec.message}</div>
                {exec.avg_price ? (
                  <div className={styles.execFill}>{exec.filled_qty} sh @ ₹{exec.avg_price}</div>
                ) : null}
                {exec.steps && exec.steps.length > 0 ? (
                  <div className={styles.execSteps}>
                    {exec.steps.map((st, i) => (
                      <div key={i}><span>{st.t}</span> {st.action} @ ₹{st.price}</div>
                    ))}
                  </div>
                ) : null}
                <div className={styles.modalBtns}>
                  <button className={styles.btnGhost} onClick={() => { setConfirmOpen(false); setExec(null); }}>
                    Close
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      <MomentumScanner heldSyms={new Set(holdings.map((h) => h.tradingsymbol))} />
    </div>
  );
}
