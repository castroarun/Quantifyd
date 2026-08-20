import React, { createContext, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import styles from './Nas.module.css';
import { apiGet } from '../api/client';
import type { NASState, NASPosition } from '../api/types';
import StatusDot from '../components/StatusDot/StatusDot';
import Chip from '../components/Chip/Chip';
import MetricCard from '../components/Cards/MetricCard';
import NiftyChart from '../components/NiftyChart/NiftyChart';
import CumulativePnL from '../components/CumulativePnL/CumulativePnL';
import {
  formatInt,
  formatNumber,
  formatPnl,
  formatPct,
  formatRs,
  pnlClass,
} from '../utils/format';

/* ---------- live ticks context ---------- */

interface LiveTicks {
  spot: number | null;
  legs: Record<string, number>; // tradingsymbol → ltp
  highs: Record<string, number>; // tradingsymbol → day-high (max traded premium)
  connected: boolean;
}
const LiveTicksContext = createContext<LiveTicks>({ spot: null, legs: {}, connected: false, highs: {} });
const useLiveTicks = () => useContext(LiveTicksContext);

/* ---------- system definitions ---------- */

interface SystemDef {
  id: string;
  key: string; // used for API paths /api/{key}/...
  label: string;
  subtitle: string;
  rules: string;
  configNote: string;
  group: 'squeeze' | '916';
}

const SQUEEZE_SYSTEMS: SystemDef[] = [
  {
    id: 'nas',
    key: 'nas',
    label: 'Squeeze · OTM',
    subtitle: 'Original 1.5 ATR OTM strangle',
    rules:
      'Entry: ATR(14) < SMA(ATR,50) on 5-min → SELL OTM CE+PE at approx Rs 20. Adj: Cross-leg imbalance >= 2x → ROLL_OUT or ROLL_IN alternating. Exit: Time 14:45, EOD 15:15.',
    configNote: 'OTM: 10L | Premium Rs 20-24',
    group: 'squeeze',
  },
  {
    id: 'nas-atm',
    key: 'nas-atm',
    label: 'Squeeze · ATM',
    subtitle: 'ATM strangle with alternating adjustment',
    rules:
      'Entry: ATR squeeze → SELL ATM CE+PE, SL = entry x 1.3 (30%). 1st SL: Close stopped leg. Naked leg: ST(7,3) exit. EOD 15:15.',
    configNote: 'ATM: 5L | 30% SL',
    group: 'squeeze',
  },
  {
    id: 'nas-atm2',
    key: 'nas-atm2',
    label: 'Squeeze · ATM 2.0',
    subtitle: 'Cascading ATM — ±0.4% move-stop, re-center to new CMP',
    rules:
      'Entry: ATR squeeze → SELL ATM CE+PE. Exit: ±0.4% underlying move closes BOTH legs AND re-enters (cascades) at the new ATM with the same ±0.4% stop. Move-stop is the sole trigger (no per-leg SL). Max 5 re-centers/day. EOD 15:15.',
    configNote: 'ATM 2.0: 5L | ±0.4% move-stop + re-center | max 5/day',
    group: 'squeeze',
  },
  {
    id: 'nas-atm4',
    key: 'nas-atm4',
    label: 'Squeeze · ATM V4',
    subtitle: 'ATM V4 with cross-leg topup',
    rules:
      'Entry: ATR squeeze → SELL ATM, SL = 1.3x. 1st SL: Roll stopped leg to match surviving leg CMP (both re-get 30% SLs). 2nd SL: Close stopped leg, naked surviving leg uses ST(7,3) exit. EOD 15:15.',
    configNote: 'ATM V4: 5L | 1.3x SL | Roll-to-match',
    group: 'squeeze',
  },
];

const ENTRY_916_SYSTEMS: SystemDef[] = [
  {
    id: 'nas-916-otm',
    key: 'nas-916-otm',
    label: 'NIFTY OTM',
    subtitle: 'Time-based 9:16 entry, OTM legs',
    rules:
      'Entry: Auto-enter at 9:16 AM. SELL OTM CE+PE at approx Rs 20. Adj: Cross-leg imbalance >= 2x → ROLL_OUT or ROLL_IN alternating. Exit: Time 14:45, EOD 15:15.',
    configNote: '916 OTM: 10L | Premium Rs 20-24',
    group: '916',
  },
  {
    id: 'nas-916-atm',
    key: 'nas-916-atm',
    label: 'NIFTY ATM',
    subtitle: 'Time-based 9:16 entry, ATM legs',
    rules:
      'Entry: Auto-enter at 9:16 AM. SELL ATM CE+PE, SL = entry x 1.3 (30%). 1st SL: Close stopped leg. Naked leg: ST(7,3) exit. EOD 15:15.',
    configNote: '916 ATM: 5L | 30% SL',
    group: '916',
  },
  {
    id: 'nas-916-atm2',
    key: 'nas-916-atm2',
    label: 'NIFTY ATM2',
    subtitle: '9:16 entry, ±0.4% move-stop → re-center (cascade)',
    rules:
      'Entry: Auto-enter at 9:16 AM. SELL ATM CE+PE. Exit: ±0.4% underlying move closes BOTH legs AND re-enters (cascades) at the new ATM with the same ±0.4% stop. Move-stop is the sole trigger (no per-leg SL). Max 5 re-centers/day. EOD 15:15.',
    configNote: '916 ATM 2.0: 5L | ±0.4% move-stop + re-center | max 5/day',
    group: '916',
  },
  {
    id: 'nas-916-atm4',
    key: 'nas-916-atm4',
    label: 'NIFTY ATM4',
    subtitle: '9:16 entry, ATM V4 cross-leg',
    rules:
      'Entry: Auto-enter at 9:16 AM. SELL ATM, SL = 1.3x. 1st SL: Roll stopped leg to match surviving leg CMP. 2nd SL: Close stopped leg, naked surviving leg uses ST(7,3) exit. EOD 15:15.',
    configNote: '916 ATM V4: 5L | 1.3x SL | Roll-to-match',
    group: '916',
  },
];

const ALL_SYSTEMS: SystemDef[] = [...SQUEEZE_SYSTEMS, ...ENTRY_916_SYSTEMS];

// UI ONLY (user 2026-07-13): hide the two OTM variants' CARDS from the dashboard.
// Nothing else changes -- both systems keep running, keep trading, and still appear in the
// Trade Book, the combined P&L and the day matrix. This hides a card, not a strategy.
const HIDDEN_CARDS = new Set(['nas', 'nas-916-otm']);

// The uniform size everything is restated to (2 lots x 65).
const TARGET_QTY = 130;

// NAS-OPT (research/54 paper system) — shown in the Trade Book alongside the 8 variants.
const NAS_OPT_DEF: SystemDef = {
  id: 'nas-opt',
  key: 'nas-opt',
  label: 'NAS-OPT',
  subtitle: '0/1-DTE ~100pt-OTM strangle + move-stop (paper)',
  rules:
    'Entry 09:20 on 0/1-DTE only (Mon & Tue). SELL ~100pt OTM CE+PE. Exit: ±0.4% underlying move-stop or 14:45.',
  configNote: 'paper · research/54',
  group: '916',
};

// COMB + TimeB sleeves in the Trade Book (after the 9:16 systems, before NAS-OPT).
// SENSEX 9:16 suite in the Trade Book (legs from static/app/sensex_live.json).
const SENSEX_TB_DEFS: SystemDef[] = [
  { id: 'sx-atm', key: 'sx-atm', label: 'SENSEX ATM', subtitle: '9:16 straddle', rules: '', configNote: 'Wed/Thu', group: '916' },
  { id: 'sx-atm2', key: 'sx-atm2', label: 'SENSEX ATM2', subtitle: '9:16 straddle · move-stop', rules: '', configNote: 'Wed/Thu', group: '916' },
  { id: 'sx-atm4', key: 'sx-atm4', label: 'SENSEX ATM4', subtitle: '9:16 straddle · roll', rules: '', configNote: 'Wed/Thu', group: '916' },
];

const SLEEVE_TB_DEFS: SystemDef[] = [
  { id: 'csl-comb', key: 'csl-comb', label: 'NIFTY COMB', subtitle: 'full-day combined-SL', rules: '', configNote: 'live · 2L (Thu 5L) · ex-Wed', group: '916' },
  { id: 'csl-timeb', key: 'csl-timeb', label: 'NIFTY TimeB', subtitle: 'windowed combined-SL', rules: '', configNote: 'live · Mon/Tue/Fri windows', group: '916' },
  { id: 'csl-comb-sx', key: 'csl-comb-sx', label: 'SENSEX COMB · all-week', subtitle: 'study per-DTE stops · all 5 days (paper)', rules: '', configNote: 'paper A/B', group: '916' },
  { id: 'csl-timeb-sx', key: 'csl-timeb-sx', label: 'COMB SENSEX', subtitle: 'Wed window + Thu full-day', rules: '', configNote: 'live · Wed 8L window / Thu 5L full-day', group: '916' },
];

// Map NAS-OPT's today-position + closed trades into the Trade Book's NASState leg shape.
function nasOptTradeBookState(today: any, trades: any[]): NASState {
  const QTY = 130; // lots_per_leg 2 × 65
  const ce: NASPosition[] = [];
  const pe: NASPosition[] = [];
  const closed_today: NASPosition[] = [];
  if (today && today.status && today.status !== 'CLOSED') {
    // entry_spot drives the +/-0.4% move-stop band in the ARM column -- NAS-OPT's ONLY exit
    // trigger besides the 14:45 time exit. Without it the arm rendered as '--', which read as
    // 'no stop' when in fact the band is checked every minute (services/nas_opt.py MOVE_PCT).
    ce.push({ leg: 'CE', tradingsymbol: today.ce_sym, strike: today.ce_strike, entry_price: today.ce_entry, qty: QTY, entry_time: today.entry_time, status: 'ACTIVE', entry_spot: today.entry_spot, mode: 'paper' });
    pe.push({ leg: 'PE', tradingsymbol: today.pe_sym, strike: today.pe_strike, entry_price: today.pe_entry, qty: QTY, entry_time: today.entry_time, status: 'ACTIVE', entry_spot: today.entry_spot, mode: 'paper' });
  }
  const dayStr = new Date().toISOString().slice(0, 10);
  (trades || [])
    .filter((x) => x.status === 'CLOSED' && x.mode === 'paper' && x.day === dayStr)
    .forEach((x) => {
      closed_today.push({ leg: 'CE', strike: x.ce_strike, entry_price: x.ce_entry, exit_price: x.ce_exit, qty: QTY, pnl_inr: Math.round(((x.ce_entry ?? 0) - (x.ce_exit ?? 0)) * QTY), entry_time: x.entry_time, exit_time: x.exit_time, exit_reason: x.exit_reason, entry_spot: x.entry_spot, mode: 'paper' });
      closed_today.push({ leg: 'PE', strike: x.pe_strike, entry_price: x.pe_entry, exit_price: x.pe_exit, qty: QTY, pnl_inr: Math.round(((x.pe_entry ?? 0) - (x.pe_exit ?? 0)) * QTY), entry_time: x.entry_time, exit_time: x.exit_time, exit_reason: x.exit_reason, entry_spot: x.entry_spot, mode: 'paper' });
    });
  return { stats: {}, positions: { ce, pe, total_active: ce.length + pe.length, closed_today } };
}

/* ---------- page ---------- */

interface SystemStateRecord {
  state: NASState | null;
  err: string | null;
}

type MtmPoint = [string, number];

interface MtmEvent {
  ts: string;
  type: 'entry' | 'adjust' | 'sl_hit' | 'exit';
  label: string;
  sig?: string | null;
  sym?: string | null;
  tx?: string | null;
  price?: number | null;
}

interface MtmSystem { points: MtmPoint[]; events: MtmEvent[]; }

/* ---------- Integrity Watchdog (independent 5-min poller, read-only) ---------- */
function WatchdogSection() {
  const [wd, setWd] = useState<any>(null);
  const [open, setOpen] = useState(true);
  useEffect(() => {
    const load = () => fetch('/app/watchdog.json?t=' + Date.now()).then((r) => r.json()).then(setWd).catch(() => {});
    load();
    const id = setInterval(load, 60000);
    return () => clearInterval(id);
  }, []);
  if (!wd) return null;
  const s = wd.summary || { ok: 0, warn: 0, fail: 0 };
  const attn = (s.warn || 0) + (s.fail || 0);
  const icon = (st: string) => {
    const m: Record<string, [string, string, string]> = {
      ok: ['#E7F2EE', '#0F6E56', '✓'], warn: ['#FEF3C7', '#B45309', '!'], fail: ['#FBEAEA', '#A32D2D', '✕'],
    };
    const [bg, c, g] = m[st] || m.ok;
    return <span style={{ display: 'inline-flex', alignItems: 'center', justifyContent: 'center', width: 22, height: 22, borderRadius: '50%', background: bg, color: c, fontSize: 13, fontWeight: 800 }}>{g}</span>;
  };
  const th: React.CSSProperties = { textAlign: 'left', fontSize: 11, textTransform: 'uppercase', letterSpacing: '.04em', color: '#888780', fontWeight: 600, padding: '8px 16px' };
  const td: React.CSSProperties = { padding: '9px 16px', borderTop: '1px solid rgba(0,0,0,0.06)', fontSize: 13 };
  const rows: any[] = [];
  (wd.groups || []).forEach((g: any) => { rows.push({ grp: g.name }); (g.checks || []).forEach((c: any) => rows.push(c)); });
  return (
    <section style={{ border: '1px solid rgba(0,0,0,0.10)', background: '#fff', borderRadius: 10, marginTop: 18, overflow: 'hidden', boxShadow: '0 1px 2px rgba(0,0,0,0.04)' }}>
      <div onClick={() => setOpen(!open)} style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '12px 16px', cursor: 'pointer' }}>
        <span style={{ color: '#888780', fontSize: 12, transform: open ? 'rotate(90deg)' : 'none', transition: 'transform .15s' }}>▶</span>
        <span style={{ color: '#1E3A8A' }}>🛡</span>
        <span style={{ fontWeight: 700, fontSize: 15, color: '#1B1B1A' }}>Integrity Watchdog</span>
        <span style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 12, fontSize: 12, color: '#888780', flexWrap: 'wrap' }}>
          <span><span style={{ color: '#0F6E56', fontWeight: 600 }}>{s.ok} OK</span> · <span style={{ color: '#B45309', fontWeight: 600 }}>{s.warn} warn</span> · <span style={{ color: '#A32D2D', fontWeight: 600 }}>{s.fail} fail</span></span>
          {attn > 0
            ? <span style={{ background: '#FBEAEA', color: '#A32D2D', fontWeight: 700, fontSize: 12, padding: '3px 10px', borderRadius: 20 }}>{attn} NEED ATTENTION</span>
            : <span style={{ background: '#E7F2EE', color: '#0F6E56', fontWeight: 700, fontSize: 12, padding: '3px 10px', borderRadius: 20 }}>ALL CLEAR</span>}
          <span>polled {String(wd.polled_at || '').slice(11, 19)}</span>
        </span>
      </div>
      {open && (
        <div style={{ borderTop: '1px solid rgba(0,0,0,0.10)' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead><tr>
              <th style={{ ...th, width: '24%' }}>Check</th><th style={{ ...th, width: '16%' }}>Scope</th>
              <th style={{ ...th, textAlign: 'center', width: '8%' }}>Status</th><th style={th}>Detail</th>
            </tr></thead>
            <tbody>
              {rows.map((r, i) => r.grp ? (
                <tr key={i}><td colSpan={4} style={{ background: '#FAFAF9', color: '#888780', fontSize: 11, textTransform: 'uppercase', letterSpacing: '.04em', fontWeight: 600, padding: '5px 16px' }}>{r.grp}</td></tr>
              ) : (
                <tr key={i}>
                  <td style={{ ...td, color: '#1B1B1A' }}>{r.check}</td>
                  <td style={{ ...td, color: '#5F5E5A', fontWeight: 600 }}>{r.scope}</td>
                  <td style={{ ...td, textAlign: 'center' }}>{icon(r.status)}</td>
                  <td style={{ ...td, color: '#5F5E5A' }}>{r.detail}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div style={{ padding: '10px 16px', fontSize: 11, color: '#B4B2A9', borderTop: '1px solid rgba(0,0,0,0.06)' }}>
            Independent 5-min poller · read-only · places no orders
          </div>
        </div>
      )}
    </section>
  );
}

function NasOptCard() {
  const [state, setState] = useState<any>(null);
  const [showLog, setShowLog] = useState(false);
  const [equity, setEquity] = useState<Array<{ day: string; pnl: number; cum: number }>>([]);
  const [trades, setTrades] = useState<any[]>([]);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const [s, e, t] = await Promise.all([
          fetch('/api/nas-opt/state').then((r) => r.json()),
          fetch('/api/nas-opt/equity').then((r) => r.json()),
          fetch('/api/nas-opt/trades').then((r) => r.json()),
        ]);
        if (!cancelled) {
          setState(s);
          setEquity(Array.isArray(e) ? e : []);
          setTrades(Array.isArray(t) ? t : []);
        }
      } catch {
        /* endpoint may be momentarily unavailable */
      }
    };
    load();
    const id = setInterval(load, 30000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  if (!state) return null;
  void equity;

  const today = state.today;
  const closedAll = trades.filter((t) => t.status === 'CLOSED');
  const sumPnl = (arr: any[]) => arr.reduce((a, t) => a + Number(t.pnl || 0), 0);
  const paper = closedAll.filter((t) => t.mode === 'paper');
  const bt = closedAll.filter((t) => t.mode === 'backtest');
  const paperPnl = sumPnl(paper);
  const paperN = paper.length;
  const paperWin = paperN ? Math.round((paper.filter((t) => Number(t.pnl) > 0).length / paperN) * 100) : 0;
  const btPnl = sumPnl(bt);
  const btN = bt.length;

  // Daily record (newest first) + the split that actually explains the P&L:
  // days held to the 14:45 time-exit vs days the +/-0.4% move-stop fired.
  const paperDesc = [...paper].sort((a, b) => (String(a.day) < String(b.day) ? 1 : -1));
  const held = paper.filter((t) => t.exit_reason === 'time1445');
  const stopped = paper.filter((t) => t.exit_reason !== 'time1445');
  const heldPnl = sumPnl(held);
  const stopPnl = sumPnl(stopped);
  const rs = (n: number) => `${n >= 0 ? '+' : '-'}\u20B9${Math.abs(Math.round(n)).toLocaleString('en-IN')}`;
  const LOGCOLS = '104px 34px 108px 56px 92px 74px';
  // Position size -- mirrors services/nas_opt.py (LOT 65, lots_per_leg 2 => 130 qty/leg).
  // Every rupee figure on this card is on that size; per-lot lets it be compared against
  // the live books, which run 2 lots live / 10 lots paper.
  const NASOPT_LOTS = 2;
  const NASOPT_QTY = 130;
  const perLot = (n: number) => rs(n / NASOPT_LOTS);

  // live-paper equity curve (built from paper trades only)
  const paperSorted = [...paper].sort((a, b) => (String(a.day) < String(b.day) ? -1 : 1));
  let cum = 0;
  const ys = paperSorted.map((t) => (cum += Number(t.pnl || 0)));
  let spark: JSX.Element | null = null;
  if (ys.length >= 2) {
    const W = 320;
    const H = 60;
    const min = Math.min(0, ...ys);
    const max = Math.max(0, ...ys);
    const rng = max - min || 1;
    const pts = ys
      .map((y, i) => `${(i / (ys.length - 1)) * W},${H - ((y - min) / rng) * H}`)
      .join(' ');
    const zeroY = H - ((0 - min) / rng) * H;
    spark = (
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} preserveAspectRatio="none">
        <line x1="0" y1={zeroY} x2={W} y2={zeroY} stroke="rgba(0,0,0,0.15)" strokeWidth="1" strokeDasharray="3 3" />
        <polyline points={pts} fill="none" stroke="#0F6E56" strokeWidth="2" />
      </svg>
    );
  }

  const chip = (bg: string, fg: string, text: string) => (
    <span style={{ background: bg, color: fg, fontSize: 11, fontWeight: 600, padding: '2px 8px', borderRadius: 6 }}>
      {text}
    </span>
  );

  return (
    <section
      style={{
        border: '1px solid rgba(0,0,0,0.10)',
        background: '#FFFFFF',
        boxShadow: '0 1px 2px rgba(0,0,0,0.04)',
        borderRadius: 10,
        padding: '16px 18px',
        marginBottom: 18,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap', marginBottom: 4 }}>
        <span style={{ fontSize: 16, fontWeight: 700, color: '#1B1B1A' }}>NAS-OPT</span>
        {chip('#FEF3C7', '#B45309', 'PAPER')}
        <Link
          to="/app/backtest/nasopt-full-replay"
          style={{ textDecoration: 'none' }}
          title="Full study: NAS-OPT replayed on all 58 recorded chain days — is the 0/1-DTE gate the edge?"
        >
          {chip('#EFF3FA', '#1E3A8A', 'research/54 · full study \u2197')}
        </Link>
        <Link
          to="/app/backtest/fardte-rescue"
          style={{ textDecoration: 'none' }}
          title="Why NAS-OPT only trades 0/1-DTE, and what the other days are worth (research/80)"
        >
          {chip('#FEF3C7', '#B45309', 'the other days ↗')}
        </Link>
        {chip('#F1F0EC', '#5A5852', `${NASOPT_LOTS} lots/leg · ${NASOPT_QTY} qty`)}
        <span style={{ color: '#888780', fontSize: 12 }}>{state.system}</span>
      </div>
      <div style={{ display: 'flex', gap: 26, flexWrap: 'wrap', margin: '10px 0' }}>
        <div>
          <div style={{ fontSize: 11, color: '#888780' }}>Live paper P&amp;L</div>
          <div style={{ fontSize: 20, fontWeight: 700, color: paperN === 0 ? '#888780' : paperPnl >= 0 ? '#0F6E56' : '#A32D2D' }}>
            {paperN === 0 ? '₹0' : `${paperPnl >= 0 ? '+' : ''}₹${paperPnl.toLocaleString('en-IN')}`}
          </div>
          {paperN > 0 && (
            <div style={{ fontSize: 11, color: '#888780' }}>{perLot(paperPnl)}/lot</div>
          )}
        </div>
        <div>
          <div style={{ fontSize: 11, color: '#888780' }}>Paper trades</div>
          <div style={{ fontSize: 20, fontWeight: 700, color: '#1B1B1A' }}>{paperN}</div>
        </div>
        <div>
          <div style={{ fontSize: 11, color: '#888780' }}>Win rate</div>
          <div style={{ fontSize: 20, fontWeight: 700, color: '#1B1B1A' }}>{paperN ? `${paperWin}%` : '—'}</div>
        </div>
        <div style={{ flex: 1, minWidth: 220 }}>
          <div style={{ fontSize: 11, color: '#888780', marginBottom: 2 }}>Live paper equity</div>
          {spark ?? (
            <div style={{ color: '#B4B2A9', fontSize: 12 }}>No paper trades yet · first entry Monday</div>
          )}
        </div>
      </div>
      <div style={{ fontSize: 11, color: '#888780', marginBottom: 6 }}>
        Backtest baseline (research/54, not live): {btPnl >= 0 ? '+' : ''}₹{btPnl.toLocaleString('en-IN')} · {btN} trades
      </div>
      {paperN > 0 && (
        <div style={{ marginBottom: 8 }}>
          <button
            onClick={() => setShowLog((v) => !v)}
            style={{
              background: 'transparent', border: '1px solid rgba(0,0,0,0.12)', borderRadius: 6,
              padding: '3px 9px', fontSize: 11, fontWeight: 600, color: '#1B1B1A', cursor: 'pointer',
            }}
          >
            {showLog ? 'Hide daily record \u25B4' : `Daily record \u2014 ${paperN} paper days \u25BE`}
          </button>
          {showLog && (
            <div style={{ marginTop: 8, overflowX: 'auto' }}>
              <div style={{ minWidth: 'max-content', fontSize: 11, fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace' }}>
                <div style={{
                  display: 'grid', gridTemplateColumns: LOGCOLS, gap: 8, padding: '3px 0',
                  color: '#888780', fontSize: 10, letterSpacing: '0.04em',
                  borderBottom: '1px solid rgba(0,0,0,0.10)',
                }}>
                  <span>DAY</span><span>DTE</span><span>STRIKES</span><span>CREDIT</span>
                  <span>EXIT</span><span style={{ textAlign: 'right' }}>P&amp;L</span>
                </div>
                {paperDesc.map((t) => {
                  const timeExit = t.exit_reason === 'time1445';
                  const p = Number(t.pnl || 0);
                  return (
                    <div
                      key={t.id}
                      style={{
                        display: 'grid', gridTemplateColumns: LOGCOLS, gap: 8, padding: '4px 0',
                        borderBottom: '1px solid rgba(0,0,0,0.05)',
                      }}
                    >
                      <span style={{ color: '#1B1B1A' }}>{t.day} {t.weekday}</span>
                      <span style={{ color: '#888780' }}>{t.dte}</span>
                      <span style={{ color: '#888780' }}>{t.pe_strike}P/{t.ce_strike}C</span>
                      <span style={{ color: '#888780' }}>{Number(t.credit || 0).toFixed(1)}</span>
                      <span style={{ color: timeExit ? '#0F6E56' : '#A32D2D' }}>
                        {timeExit ? 'held 14:45' : 'move-stop'}
                      </span>
                      <span style={{ textAlign: 'right', fontWeight: 600, color: p >= 0 ? '#0F6E56' : '#A32D2D' }}>
                        {rs(p)}
                      </span>
                    </div>
                  );
                })}
                <div style={{ marginTop: 8, fontSize: 11, fontFamily: 'inherit', color: '#5A5852', lineHeight: 1.55 }}>
                  <div>
                    Held to 14:45 &middot; <strong>{held.length}</strong> days &middot;{' '}
                    <span style={{ color: heldPnl >= 0 ? '#0F6E56' : '#A32D2D', fontWeight: 600 }}>{rs(heldPnl)}</span>
                    <span style={{ color: '#888780' }}> ({perLot(heldPnl)}/lot)</span>
                  </div>
                  <div>
                    Move-stop fired &middot; <strong>{stopped.length}</strong> days &middot;{' '}
                    <span style={{ color: stopPnl >= 0 ? '#0F6E56' : '#A32D2D', fontWeight: 600 }}>{rs(stopPnl)}</span>
                    <span style={{ color: '#888780' }}> ({perLot(stopPnl)}/lot)</span>
                  </div>
                  <div style={{ color: '#888780', marginTop: 4 }}>
                    All figures on {NASOPT_LOTS} lots/leg ({NASOPT_QTY} qty × CE + PE), net of ₹80/leg
                    round-trip brokerage. 1 lot = 65.
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
      <div style={{ fontSize: 12, color: today ? '#0F6E56' : '#888780' }}>
        {today
          ? today.status === 'CLOSED'
            ? `Today: ${today.pe_strike}PE + ${today.ce_strike}CE from spot ${Math.round(
                today.entry_spot,
              )} — closed ${
                today.exit_reason === 'time1445' ? 'at the 14:45 time-exit' : 'on the ±0.4% move-stop'
              } · ${rs(Number(today.pnl || 0))}`
            : `Open today: ${today.pe_strike}PE + ${today.ce_strike}CE · entry spot ${Math.round(
                today.entry_spot,
              )} · running`
          : 'Idle today — NAS-OPT trades 0/1-DTE only (Mon & Tue); enters 09:20, ±0.4% move-stop, 14:45 exit.'}
      </div>
    </section>
  );
}


/* ---------- SENSEX LIVE book (real-money 9:16 executors) ----------
   The 3 SENSEX 9:16 systems (ATM / ATM2 / ATM4) trade REAL MONEY on Wed/Thu — SENSEX's 0/1-DTE,
   the days NIFTY has nothing to harvest and NIFTY real money is switched off. This renders their
   live legs at the TOP of the page with a LIVE marking in the MODE column. Data comes from a
   standalone writer (scripts/sensex_live_writer.py -> /app/sensex_live.json) so it needs no backend
   restart while positions are open. Refreshes every 8s. */
function SensexLiveCard() {
  const [d, setD] = useState<any>(null);
  const [hist, setHist] = useState<any>(null);
  const [mtm, setMtm] = useState<any>(null);
  const [openHist, setOpenHist] = useState(false);
  const [openDay, setOpenDay] = useState<string | null>(null);
  useEffect(() => {
    const load = () => {
      fetch(`/app/sensex_live.json?t=${Date.now()}`, { cache: 'no-store' })
        .then((r) => r.json()).then(setD).catch(() => {});
      fetch(`/api/sensex/sessions`, { cache: 'no-store' })
        .then((r) => r.json()).then(setHist).catch(() => {});
      fetch(`/app/sensex_mtm.json?t=${Date.now()}`, { cache: 'no-store' })
        .then((r) => r.json()).then(setMtm).catch(() => {});
    };
    load();
    const id = setInterval(load, 8000);
    return () => clearInterval(id);
  }, []);
  if (!d && !hist) return null;
  const rs = (n: number) => `${(n || 0) >= 0 ? '+' : '−'}₹${Math.abs(Math.round(n || 0)).toLocaleString('en-IN')}`;
  const cc = (n: number) => ((n || 0) >= 0 ? '#3fb950' : '#f85149');
  const withLegs = ((d && d.systems) || []).filter((x: any) => (x.legs || []).length);
  const cell: React.CSSProperties = { fontSize: 11, padding: '3px 8px', textAlign: 'right',
    borderTop: '1px solid var(--line)', fontVariantNumeric: 'tabular-nums' };
  const head: React.CSSProperties = { fontSize: 9.5, color: 'var(--ink-faint, #6e7681)', padding: '2px 8px',
    textAlign: 'right', textTransform: 'uppercase', letterSpacing: '0.04em' };
  const modePill = (m: string) => (
    <span style={{ display: 'inline-block', fontSize: 9.5, fontWeight: 800, letterSpacing: '0.05em',
      padding: '1px 7px', borderRadius: 4,
      background: m === 'live' ? 'rgba(63,185,80,0.16)' : 'rgba(139,148,158,0.14)',
      color: m === 'live' ? '#3fb950' : 'var(--ink-muted)',
      border: `1px solid ${m === 'live' ? 'rgba(63,185,80,0.4)' : 'var(--line)'}` }}>
      {m === 'live' ? 'LIVE' : (m || '').toUpperCase()}
    </span>
  );
  const GRID = '54px 68px 46px 60px 60px 60px 60px 1fr 78px 82px';
  const sessions = (hist && hist.sessions) || [];
  const anyLive = withLegs.some((x: any) => x.mode === 'live');

  return (
    <section className={styles.sectionBlock} style={{ marginTop: 14,
      border: '1px solid var(--line)', borderRadius: 10, padding: '12px 14px',
      background: 'transparent' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4, flexWrap: 'wrap' }}>
        <span className="section-title">SENSEX</span>
        {modePill(anyLive ? 'live' : 'paper')}
        <span style={{ fontSize: 11, color: 'var(--ink-muted)' }}>{d ? d.expiry : 'Wed/Thu live · else paper'}</span>
        {d && d.spot != null && (
          <span style={{ fontSize: 11, color: 'var(--ink-muted)' }}>spot {d.spot.toLocaleString('en-IN')}</span>
        )}
        <span style={{ marginLeft: 'auto', fontSize: 12, color: 'var(--ink-muted)' }}>
          today <b style={{ color: cc(d ? d.day_pnl : 0) }}>{rs(d ? d.day_pnl : 0)}</b>
        </span>
      </div>
      <div style={{ fontSize: 10, color: 'var(--ink-faint, #6e7681)', marginBottom: 8 }}>
        9:16 systems on SENSEX (Wed/Thu 0/1-DTE) · {d ? `updated ${d.generated_at}` : 'after-hours — showing recorded sessions'}
      </div>

      {mtm && mtm.points && mtm.points.length >= 2 ? (() => {
        const pts = mtm.points as [string, number][];
        const W = 900, H = 120, PL = 6, PR = 6, PT = 8, PB = 16;
        const ys = pts.map((p) => p[1]);
        const yMin = Math.min(0, ...ys), yMax = Math.max(0, ...ys), span = (yMax - yMin) || 1;
        const xf = (i: number) => PL + (i / (pts.length - 1)) * (W - PL - PR);
        const yf = (v: number) => PT + (1 - (v - yMin) / span) * (H - PT - PB);
        const line = pts.map((p, i) => `${i ? 'L' : 'M'}${xf(i).toFixed(1)},${yf(p[1]).toFixed(1)}`).join(' ');
        const zeroY = yf(0);
        const area = `${line} L${xf(pts.length - 1).toFixed(1)},${zeroY.toFixed(1)} L${xf(0).toFixed(1)},${zeroY.toFixed(1)} Z`;
        const last = ys[ys.length - 1];
        const col = last >= 0 ? '#3fb950' : '#f85149';
        const ticks: Array<[number, string]> = [];
        const step = Math.max(1, Math.floor(pts.length / 6));
        for (let i = 0; i < pts.length; i += step) ticks.push([xf(i), pts[i][0]]);
        return (
          <div style={{ marginBottom: 10, border: '1px solid var(--line)', borderRadius: 8, padding: '6px 8px 2px' }}>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 2 }}>
              <span style={{ fontSize: 10.5, color: 'var(--ink-muted)', textTransform: 'uppercase', letterSpacing: '0.4px' }}>
                SENSEX intraday P&amp;L · today
              </span>
              <span style={{ fontSize: 12, fontWeight: 800, color: col }}>{rs(last)}</span>
              <span style={{ marginLeft: 'auto', fontSize: 10, color: 'var(--ink-faint, #6e7681)' }}>
                lo {rs(mtm.lo)} · hi {rs(mtm.hi)}
              </span>
            </div>
            <svg viewBox={`0 0 ${W} ${H}`} width="100%" preserveAspectRatio="none" style={{ display: 'block' }}>
              <defs>
                <linearGradient id="sxmtmfill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={col} stopOpacity="0.18" />
                  <stop offset="100%" stopColor={col} stopOpacity="0" />
                </linearGradient>
              </defs>
              <line x1={PL} y1={zeroY} x2={W - PR} y2={zeroY} stroke="var(--ink-muted)" strokeOpacity="0.3" strokeDasharray="3 3" />
              <path d={area} fill="url(#sxmtmfill)" />
              <path d={line} fill="none" stroke={col} strokeWidth={2} strokeLinejoin="round" />
              {ticks.map(([tx, tl], i) => (
                <text key={i} x={tx} y={H - 4} fontSize={9} fill="var(--ink-muted)" textAnchor="middle">{tl}</text>
              ))}
            </svg>
          </div>
        );
      })() : null}

      {!withLegs.length && (
        <div style={{ fontSize: 11.5, color: 'var(--ink-faint, #6e7681)',
          border: '1px solid var(--line)', borderRadius: 8, padding: '10px 12px' }}>
          No open SENSEX positions right now — see recorded sessions below.
        </div>
      )}

      {withLegs.map((sy: any, si: number) => (
        <div key={sy.key} style={{ marginTop: si ? 12 : 2 }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, marginBottom: 2 }}>
            <span style={{ fontSize: 12, fontWeight: 700 }}>{sy.label}</span>
            {modePill(sy.mode)}
            <span style={{ marginLeft: 'auto', fontSize: 12, fontWeight: 800, color: cc(sy.pnl) }}>
              {rs(sy.pnl)}
            </span>
          </div>
          <div style={{ overflowX: 'auto' }}>
            <div style={{ display: 'grid', gridTemplateColumns: GRID, gap: 8, minWidth: 700,
              fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace' }}>
              <span style={{ ...head, textAlign: 'left' }}>C/P</span><span style={head}>STRIKE</span>
              <span style={head}>QTY</span><span style={{ ...head, textAlign: 'left' }}>MODE</span>
              <span style={head}>ENTRY</span><span style={head}>LTP</span><span style={head}>MAX</span>
              <span style={{ ...head, textAlign: 'left' }}>ARM</span><span style={head}>STATUS</span>
              <span style={head}>P&amp;L</span>
              {(sy.legs || []).map((l: any, li: number) => (
                <React.Fragment key={li}>
                  <span style={{ ...cell, textAlign: 'left', fontWeight: 700, color: '#f85149' }}>SELL {l.cp}</span>
                  <span style={cell}>{l.strike}</span>
                  <span style={cell}>{l.qty}</span>
                  <span style={{ ...cell, textAlign: 'left' }}>{modePill(l.mode)}</span>
                  <span style={cell}>{l.entry}</span>
                  <span style={cell}>{l.ltp ?? '—'}</span>
                  <span style={{ ...cell, color: 'var(--ink-muted)' }}>{l.max ?? '—'}</span>
                  <span style={{ ...cell, textAlign: 'left',
                    color: (l.arm || '').includes('UNARMED') ? '#f85149' : 'var(--ink-muted)' }}>{l.arm}</span>
                  <span style={{ ...cell, color: l.status === 'ACTIVE' ? '#3fb950' : 'var(--ink-muted)' }}>{l.status}</span>
                  <span style={{ ...cell, fontWeight: 700, color: cc(l.pnl) }}>{rs(l.pnl)}</span>
                </React.Fragment>
              ))}
            </div>
          </div>
        </div>
      ))}

      {sessions.length > 0 && (
        <div style={{ marginTop: 12, borderTop: '1px solid var(--line)', paddingTop: 8 }}>
          <button type="button" onClick={() => setOpenHist((v) => !v)}
            style={{ display: 'flex', alignItems: 'center', gap: 8, width: '100%', background: 'transparent',
              border: 0, padding: '2px 0', cursor: 'pointer', textAlign: 'left' }}>
            <span style={{ fontSize: 12, color: 'var(--ink-muted)' }}>{openHist ? '▾' : '▸'}</span>
            <span style={{ fontSize: 11.5, fontWeight: 700 }}>Past sessions — {sessions.length} days</span>
            <span style={{ marginLeft: 'auto', fontSize: 12, fontWeight: 800, color: cc(hist.total) }}>
              {rs(hist.total)}
            </span>
          </button>
          {openHist && (
            <div style={{ marginTop: 8 }}>
              {(() => {
                const chron = [...sessions].reverse();
                let c = 0; const pts = chron.map((se: any) => (c += se.pnl));
                if (pts.length < 2) return null;
                const W = 560, H = 60;
                const mn = Math.min(0, ...pts), mx = Math.max(0, ...pts), rg = (mx - mn) || 1;
                const poly = pts.map((y: number, i: number) =>
                  `${(i / (pts.length - 1)) * W},${H - ((y - mn) / rg) * H}`).join(' ');
                const zy = H - ((0 - mn) / rg) * H;
                return (
                  <div style={{ marginBottom: 10 }}>
                    <div style={{ fontSize: 10, color: 'var(--ink-faint, #6e7681)', marginBottom: 2 }}>
                      Cumulative P&amp;L (SENSEX paper + live, all days)
                    </div>
                    <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} preserveAspectRatio="none">
                      <line x1="0" y1={zy} x2={W} y2={zy} stroke="var(--line)" strokeWidth="1" strokeDasharray="3 3" />
                      <polyline points={poly} fill="none"
                        stroke={pts[pts.length - 1] >= 0 ? '#3fb950' : '#f85149'} strokeWidth="2" />
                    </svg>
                  </div>
                );
              })()}
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 8 }}>
                {sessions.map((se: any) => (
                  <button key={se.day} type="button" onClick={() => setOpenDay(openDay === se.day ? null : se.day)}
                    style={{ border: '1px solid var(--line)', borderRadius: 6, padding: '4px 9px',
                      background: openDay === se.day ? 'var(--line)' : 'transparent', cursor: 'pointer' }}>
                    <div style={{ fontSize: 10, color: 'var(--ink-muted)' }}>{se.day.slice(5)}</div>
                    <div style={{ fontSize: 12, fontWeight: 800, color: cc(se.pnl) }}>{rs(se.pnl)}</div>
                  </button>
                ))}
              </div>
              {openDay && (() => {
                const se = sessions.find((x: any) => x.day === openDay);
                if (!se) return null;
                return (
                  <div style={{ border: '1px solid var(--line)', borderRadius: 8, padding: '8px 10px' }}>
                    <div style={{ fontSize: 11, fontWeight: 700, marginBottom: 4 }}>Positions · {se.day}</div>
                    {se.systems.map((sy: any, i: number) => (
                      <div key={i} style={{ marginTop: i ? 8 : 0 }}>
                        <div style={{ display: 'flex', gap: 8, marginBottom: 2 }}>
                          <span style={{ fontSize: 11, fontWeight: 700 }}>{sy.label}</span>
                          {modePill((sy.legs[0] && sy.legs[0].mode) || 'paper')}
                          <span style={{ marginLeft: 'auto', fontSize: 11, fontWeight: 700, color: cc(sy.pnl) }}>{rs(sy.pnl)}</span>
                        </div>
                        <div style={{ overflowX: 'auto' }}>
                          <div style={{ display: 'grid', gridTemplateColumns: '54px 68px 46px 62px 62px 1fr 80px',
                            gap: 8, minWidth: 520, fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace' }}>
                            <span style={{ ...head, textAlign: 'left' }}>C/P</span><span style={head}>STRIKE</span>
                            <span style={head}>QTY</span><span style={head}>ENTRY</span><span style={head}>EXIT</span>
                            <span style={{ ...head, textAlign: 'left' }}>EXIT REASON</span><span style={head}>P&amp;L</span>
                            {sy.legs.map((l: any, li: number) => (
                              <React.Fragment key={li}>
                                <span style={{ ...cell, textAlign: 'left', fontWeight: 700, color: '#f85149' }}>SELL {l.leg}</span>
                                <span style={cell}>{l.strike}</span>
                                <span style={cell}>{l.qty}</span>
                                <span style={cell}>{l.entry}</span>
                                <span style={cell}>{l.exit ?? '—'}</span>
                                <span style={{ ...cell, textAlign: 'left', color: 'var(--ink-muted)' }}>{l.reason}</span>
                                <span style={{ ...cell, fontWeight: 700, color: cc(l.pnl || 0) }}>{l.pnl != null ? rs(l.pnl) : '—'}</span>
                              </React.Fragment>
                            ))}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                );
              })()}
            </div>
          )}
        </div>
      )}
    </section>
  );
}

/* ---------- SENSEX expiry-day paper book (research/82) ----------
   SENSEX expires THURSDAY, NIFTY Tuesday. NAS-OPT harvests NIFTY's 0/1-DTE edge on Mon/Tue;
   this covers Wed(DTE1)/Thu(DTE0) — the days that capital would otherwise sit idle.
   Runs every weekday, but DTE0/1 is the SYSTEM and other DTEs are recorded as OBSERVATION
   only — the same split research/79 forced on NAS-OPT after far-DTE proved EV-negative. */
function SensexPaperCard() {
  const [d, setD] = useState<any>(null);
  const [open, setOpen] = useState(false);
  const [showLog, setShowLog] = useState(false);
  const [variant, setVariant] = useState<'straddle' | 'strangle'>('strangle');
  useEffect(() => {
    const load = () => fetch(`/app/sensex_paper.json?t=${Date.now()}`, { cache: 'no-store' })
      .then((r) => r.json()).then(setD).catch(() => {});
    load();
    const id = setInterval(load, 60000);
    return () => clearInterval(id);
  }, []);
  if (!d) return null;
  const V = d.variants?.[variant];
  if (!V) return null;
  const rs = (n: number) => `${n >= 0 ? '+' : '−'}₹${Math.abs(Math.round(n)).toLocaleString('en-IN')}`;
  const cc = (n: number) => (n >= 0 ? '#3fb950' : '#f85149');
  const rows = (d.history || []).filter((h: any) => h.variant === variant);

  // cumulative P&L curve — SYSTEM days only (the honest number)
  const curve: [string, number][] = V.curve || [];
  let spark: JSX.Element | null = null;
  if (curve.length >= 2) {
    const W = 560, H = 74;
    const ys = curve.map((p) => p[1]);
    const mn = Math.min(0, ...ys), mx = Math.max(0, ...ys), rg = mx - mn || 1;
    const pts = ys.map((y, i) => `${(i / (ys.length - 1)) * W},${H - ((y - mn) / rg) * H}`).join(' ');
    const zy = H - ((0 - mn) / rg) * H;
    spark = (
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} preserveAspectRatio="none">
        <line x1="0" y1={zy} x2={W} y2={zy} stroke="var(--line)" strokeWidth="1" strokeDasharray="3 3" />
        <polyline points={pts} fill="none" stroke={ys[ys.length - 1] >= 0 ? '#3fb950' : '#f85149'} strokeWidth="2" />
      </svg>
    );
  }
  const cell: React.CSSProperties = { fontSize: 11, padding: '3px 8px', textAlign: 'right',
    borderTop: '1px solid var(--line)', fontVariantNumeric: 'tabular-nums' };
  const head: React.CSSProperties = { fontSize: 9.5, color: 'var(--ink-faint, #6e7681)', padding: '2px 8px',
    textAlign: 'right', textTransform: 'uppercase', letterSpacing: '0.04em' };

  return (
    <section className={styles.sectionBlock} style={{ marginTop: 14 }}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        style={{ display: 'flex', alignItems: 'center', gap: 10, width: '100%', background: 'transparent',
          border: 0, padding: '4px 0', cursor: 'pointer', textAlign: 'left' }}
      >
        <span style={{ fontSize: 13, color: 'var(--ink-muted)' }}>{open ? '▾' : '▸'}</span>
        <span className="section-title">SENSEX expiry-day · paper</span>
        <Chip>{d.lots} lots · {d.qty} qty</Chip>
        <span style={{ fontSize: 11, color: 'var(--ink-muted)' }}>Thu expiry → fills Wed/Thu</span>
        <span style={{ marginLeft: 'auto', fontSize: 12, color: 'var(--ink-muted)' }}>
          system {V.system.n}d{' '}
          <b style={{ color: cc(V.system.pnl) }}>{rs(V.system.pnl)}</b>
          {V.system.win != null ? ` · ${V.system.win}% win` : ''}
        </span>
      </button>

      {open && (
        <div style={{ marginTop: 10 }}>
          <div style={{ fontSize: 11.5, color: 'var(--ink-muted)', marginBottom: 10 }}>{d.subtitle}</div>

          <div style={{ display: 'flex', gap: 6, marginBottom: 10 }}>
            {(['strangle', 'straddle'] as const).map((v) => (
              <button key={v} type="button" onClick={() => setVariant(v)}
                style={{ background: variant === v ? 'var(--line)' : 'transparent',
                  border: '1px solid var(--line)', borderRadius: 5, padding: '2px 9px', fontSize: 11,
                  cursor: 'pointer', color: variant === v ? 'var(--ink)' : 'var(--ink-muted)',
                  fontWeight: variant === v ? 700 : 400 }}>
                {v === 'strangle' ? 'OTM strangle (NAS-OPT shape)' : 'ATM straddle (NAS-916 shape)'}
              </button>
            ))}
          </div>

          <div style={{ display: 'flex', gap: 26, flexWrap: 'wrap', marginBottom: 10 }}>
            <div>
              <div style={{ fontSize: 10.5, color: 'var(--ink-faint, #6e7681)', textTransform: 'uppercase' }}>
                System (DTE 0/1 — Wed/Thu)
              </div>
              <div style={{ fontSize: 20, fontWeight: 800, color: cc(V.system.pnl) }}>{rs(V.system.pnl)}</div>
              <div style={{ fontSize: 11, color: 'var(--ink-muted)' }}>
                {V.system.n} days{V.system.win != null ? ` · ${V.system.win}% win` : ''}
              </div>
            </div>
            <div>
              <div style={{ fontSize: 10.5, color: 'var(--ink-faint, #6e7681)', textTransform: 'uppercase' }}>
                Observational (DTE ≥ 2)
              </div>
              <div style={{ fontSize: 20, fontWeight: 800, color: cc(V.observational.pnl) }}>{rs(V.observational.pnl)}</div>
              <div style={{ fontSize: 11, color: 'var(--ink-muted)' }}>
                {V.observational.n} days · not the system
              </div>
            </div>
            <div style={{ flex: 1, minWidth: 240 }}>
              <div style={{ fontSize: 10.5, color: 'var(--ink-faint, #6e7681)', marginBottom: 2 }}>
                Cumulative P&amp;L — system days only
              </div>
              {spark ?? <div style={{ fontSize: 11, color: 'var(--ink-faint, #6e7681)' }}>building…</div>}
            </div>
          </div>

          {(() => {
            // Today's actual legs, both variants — the live book. `legs` is emitted by
            // sensex_paper.py; during market hours `ltp` is the live mark off the chain.
            const today = (d.history || []).filter((h: any) => h.day === d.today_day && h.legs);
            if (!today.length) {
              return (
                <div style={{ fontSize: 11.5, color: 'var(--ink-faint, #6e7681)',
                  border: '1px solid var(--line)', borderRadius: 8, padding: '10px 12px', marginBottom: 10 }}>
                  No position today yet — enters 09:20 each weekday.
                </div>
              );
            }
            return (
              <div style={{ border: '1px solid var(--line)', borderRadius: 8, padding: '10px 12px',
                marginBottom: 10 }}>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, marginBottom: 6 }}>
                  <span style={{ fontSize: 12, fontWeight: 700 }}>Positions · {d.today_day}</span>
                  <span style={{ fontSize: 10.5, color: 'var(--ink-muted)' }}>
                    {today[0].weekday} · DTE {today[0].dte} · {today[0].signal_class}
                  </span>
                  <span style={{ marginLeft: 'auto', fontSize: 13, fontWeight: 800,
                    color: cc(today.reduce((a: number, t: any) => a + t.pnl, 0)) }}>
                    {rs(today.reduce((a: number, t: any) => a + t.pnl, 0))}
                  </span>
                </div>
                {today.map((t: any, ti: number) => (
                  <div key={ti} style={{ marginTop: ti ? 8 : 0 }}>
                    <div style={{ fontSize: 10.5, color: 'var(--ink-muted)', marginBottom: 2 }}>
                      {t.variant === 'strangle' ? 'OTM strangle' : 'ATM straddle'} · credit {t.credit} ·{' '}
                      <span style={{ color: t.exit_reason?.startsWith('time') ? '#3fb950' : '#f85149' }}>
                        {t.exit_reason}
                      </span>
                      <span style={{ marginLeft: 8, fontWeight: 700, color: cc(t.pnl) }}>{rs(t.pnl)}</span>
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: '68px 76px 46px 62px 62px 78px',
                      gap: 8, fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace' }}>
                      <span style={head}>LEG</span><span style={head}>STRIKE</span><span style={head}>QTY</span>
                      <span style={head}>ENTRY</span><span style={head}>LTP</span><span style={head}>P&amp;L</span>
                      {(t.legs || []).map((l: any, li: number) => (
                        <React.Fragment key={li}>
                          <span style={{ ...cell, textAlign: 'left', fontWeight: 700, color: '#f85149' }}>
                            SELL {l.type}
                          </span>
                          <span style={cell}>{l.strike}</span>
                          <span style={cell}>{l.qty}</span>
                          <span style={cell}>{l.entry}</span>
                          <span style={cell}>{l.ltp}</span>
                          <span style={{ ...cell, fontWeight: 700, color: cc(l.pnl) }}>{rs(l.pnl)}</span>
                        </React.Fragment>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            );
          })()}

          <button type="button" onClick={() => setShowLog((v) => !v)}
            style={{ background: 'transparent', border: '1px solid var(--line)', borderRadius: 6,
              padding: '3px 9px', fontSize: 11, fontWeight: 600, color: 'var(--ink)', cursor: 'pointer' }}>
            {showLog ? 'Hide daily record ▴' : `Daily record — ${rows.length} days ▾`}
          </button>

          {showLog && (
            <div style={{ marginTop: 8, overflowX: 'auto' }}>
              <div style={{ minWidth: 'max-content', fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace' }}>
                <div style={{ display: 'grid', gridTemplateColumns: '104px 34px 34px 132px 62px 82px 78px',
                  gap: 8, borderBottom: '1px solid var(--line)' }}>
                  <span style={head}>DAY</span><span style={head}>WD</span><span style={head}>DTE</span>
                  <span style={head}>STRIKES</span><span style={head}>CREDIT</span><span style={head}>EXIT</span>
                  <span style={head}>P&amp;L</span>
                </div>
                {rows.map((h: any, i: number) => (
                  <div key={i} style={{ display: 'grid',
                    gridTemplateColumns: '104px 34px 34px 132px 62px 82px 78px', gap: 8,
                    opacity: h.signal_class === 'system' ? 1 : 0.5 }}>
                    <span style={{ ...cell, textAlign: 'left' }}>{h.day}</span>
                    <span style={cell}>{h.weekday}</span>
                    <span style={cell}>{h.dte}</span>
                    <span style={cell}>{h.pe_strike}P/{h.ce_strike}C</span>
                    <span style={cell}>{h.credit}</span>
                    <span style={{ ...cell, color: h.exit_reason?.startsWith('time') ? '#3fb950' : '#f85149' }}>
                      {h.exit_reason}
                    </span>
                    <span style={{ ...cell, fontWeight: 700, color: cc(h.pnl) }}>{rs(h.pnl)}</span>
                  </div>
                ))}
                <div style={{ fontSize: 10.5, color: 'var(--ink-faint, #6e7681)', marginTop: 8 }}>
                  Faded rows are DTE≥2 — observation only, never counted in the system number.
                  All figures on {d.lots} lots ({d.qty} qty), SENSEX lot {d.lot_size}, net of ₹20/leg.
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  );
}

/** NSR-W v1.5 weekly-strangle PAPER books (research/90 G5) — live cards + positions.
 *  Backend: services/nsrw_paper.py -> /api/nsrw/state (books t30/t20, 1-min m2m series).
 *  Study: /app/backtest/nifty-strangle-rules-research90 · report: /app/nsrw-travel-research90.html */
function NsrwSpark({ series }: { series: [string, number][] }) {
  if (!series || series.length < 2) return null;
  const W = 560, H = 92, P = 6;
  const vals = series.map((p) => p[1]);
  const lo = Math.min(...vals, 0), hi = Math.max(...vals, 0);
  const x = (i: number) => P + (i / (series.length - 1)) * (W - 2 * P);
  const y = (v: number) => P + ((hi - v) / (hi - lo || 1)) * (H - 2 * P);
  const now = vals[vals.length - 1];
  const col = now >= 0 ? '#0ca30c' : '#e66767';
  const pts = series.map((p, i) => `${x(i).toFixed(1)},${y(p[1]).toFixed(1)}`).join(' ');
  const area = `${x(0).toFixed(1)},${y(0).toFixed(1)} ${pts} ${x(series.length - 1).toFixed(1)},${y(0).toFixed(1)}`;
  return (
    <div style={{ margin: '6px 0 2px' }}>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
        <polygon points={area} fill={col} opacity={0.12} />
        <line x1={P} x2={W - P} y1={y(0)} y2={y(0)} stroke="#8b8fa3" strokeWidth={1} strokeDasharray="3 4" opacity={0.5} />
        <polyline fill="none" stroke={col} strokeWidth={2} points={pts} />
        <circle cx={x(series.length - 1)} cy={y(now)} r={3.5} fill={col} />
      </svg>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11.5 }}>
        <span style={{ color: col, fontWeight: 800 }}>
          now {now < 0 ? '−₹' : '+₹'}{Math.abs(Math.round(now)).toLocaleString('en-IN')}
        </span>
        <span style={{ opacity: 0.6 }}>
          lo −₹{Math.abs(Math.round(Math.min(...vals, 0))).toLocaleString('en-IN')} · hi
          +₹{Math.abs(Math.round(Math.max(...vals, 0))).toLocaleString('en-IN')} · {series[0][0]}→{series[series.length - 1][0]}
        </span>
      </div>
    </div>
  );
}

const NSRW_M = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const nsrwExp = (e?: string) => {
  if (!e) return '';
  const d = new Date(e);
  return `${d.getDate()} ${NSRW_M[d.getMonth()]}`;
};
const nsrwLeg = (l: any, expiry?: string) =>
  `NIFTY ${l.strike} ${String(l.tsym || '').endsWith('PE') ? 'PE' : 'CE'}` +
  (expiry ? ` · ${nsrwExp(expiry)}` : '');
const nsrwDT = (s?: string) => {
  if (!s) return '—';
  const d = new Date(String(s).replace(' ', 'T'));
  return isNaN(d.getTime()) ? String(s)
    : `${d.getDate()} ${NSRW_M[d.getMonth()]} ${String(s).slice(11, 16)}`;
};

function NsrwBook({ id, b }: { id: string; b: any }) {
  const c: any = b.cycle;
  const legs: any[] = c?.legs ?? [];
  const rs = (v: number) => (v < 0 ? '−₹' : '+₹') + Math.abs(Math.round(v)).toLocaleString('en-IN');
  const legPnl = (l: any) => (l.entry - (l.status === 'live' ? (l.ltp ?? l.entry) : (l.exit ?? l.entry))) * Math.abs(l.qty);
  const chip = (txt: string, col: string) => (
    <span style={{ fontSize: 10.5, fontWeight: 700, letterSpacing: 0.4, border: `1px solid ${col}`,
      color: col, borderRadius: 99, padding: '1px 8px' }}>{txt}</span>
  );
  return (
    <div style={{ flex: '1 1 420px', minWidth: 340, border: '1px solid rgba(148,163,184,0.15)', borderRadius: 10, padding: '10px 14px' }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, flexWrap: 'wrap' }}>
        <span style={{ fontSize: 13, fontWeight: 800 }}>{b.label}</span>
        {b.killed ? chip('KILLED', '#e66767') : c ? chip(`OPEN · exp ${nsrwExp(c.expiry)}`, '#3987e5') : chip('FLAT', '#8b8fa3')}
        <span style={{ marginLeft: 'auto', fontSize: 11.5, opacity: 0.75 }}>
          book {b.totals.weeks}w · {b.totals.wins}W · <b>{rs(b.totals.net_rs)}</b>
        </span>
      </div>
      {c ? (
        <>
          <div style={{ display: 'flex', gap: 22, alignItems: 'baseline', margin: '8px 0 0', flexWrap: 'wrap' }}>
            <div>
              <div style={{ fontSize: 11, opacity: 0.6 }}>Day P&amp;L</div>
              <div style={{ fontSize: 20, fontWeight: 800,
                color: ((c.m2m_rs ?? 0) - ((c.mtm_series ?? [])[0]?.[1] ?? 0)) >= 0 ? '#0ca30c' : '#e66767' }}>
                {rs((c.m2m_rs ?? 0) - ((c.mtm_series ?? [])[0]?.[1] ?? 0))}
              </div>
            </div>
            <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', fontSize: 12, opacity: 0.9 }}>
              <span>credit <b>{c.credit0}</b></span>
              <span>week m2m <b style={{ color: (c.m2m_rs ?? 0) >= 0 ? '#0ca30c' : '#e66767' }}>{rs(c.m2m_rs ?? 0)}</b></span>
              <span>restrangled <b>{c.rolled ? 'yes' : 'no'}</b></span>
              <span>recentered <b>{c.recentered ? 'yes' : 'no'}</b></span>
            </div>
          </div>
          <NsrwSpark series={c.mtm_series ?? []} />
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', fontSize: 12, borderCollapse: 'collapse' }}>
              <thead><tr style={{ opacity: 0.6, textAlign: 'left' }}>
                <th style={{ padding: '3px 6px' }}>Leg</th><th>Sold @</th><th>Now / Exit</th>
                <th>Min</th><th>Max</th><th>Stop</th><th>P&amp;L</th><th>Status</th>
              </tr></thead>
              <tbody>
                {legs.map((l, i) => {
                  const p = legPnl(l);
                  return (
                    <tr key={i} style={{ borderTop: '1px solid rgba(148,163,184,0.1)',
                      opacity: l.status === 'live' ? 1 : 0.75 }}>
                      <td style={{ padding: '3px 6px', fontWeight: 600 }}>
                        {nsrwLeg(l, c.expiry)} ×{Math.abs(l.qty)}</td>
                      <td>{l.entry}
                        <div style={{ fontSize: 10, opacity: 0.55 }}>{nsrwDT(l.opened)}</div></td>
                      <td>{l.status === 'live' ? (l.ltp ?? '—') : l.exit}
                        <div style={{ fontSize: 10, opacity: 0.55 }}>
                          {l.status === 'live' ? 'live' : nsrwDT(l.closed)}</div></td>
                      <td>{l.px_min ?? '—'}</td>
                      <td>{l.px_max ?? '—'}</td>
                      <td>{l.stop}</td>
                      <td style={{ color: p >= 0 ? '#0ca30c' : '#e66767', fontWeight: 700 }}>{rs(p)}</td>
                      <td style={{ color: l.status === 'live' ? '#3987e5' : l.status === 'STOP' ? '#e66767' : '#8b8fa3', fontWeight: 700 }}>
                        {l.status}
                        {l.reason_detail && (
                          <div style={{ fontSize: 10, fontWeight: 400, opacity: 0.6,
                            maxWidth: 240, whiteSpace: 'normal' }}>{l.reason_detail}</div>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', textAlign: 'center', minHeight: 150, gap: 6, opacity: 0.72 }}>
          <div style={{ fontSize: 13, fontWeight: 700 }}>Flat — no open cycle</div>
          <div style={{ fontSize: 12, opacity: 0.85 }}>Enters Monday 15:14 · ₹{b.target}/leg · adjust at ₹{b.adjust}</div>
        </div>
      )}
      {(b.history ?? []).length > 0 && (
        <div style={{ marginTop: 10 }}>
          <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: 0.3, opacity: 0.6,
            marginBottom: 3 }}>COMPLETED WEEKS</div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', fontSize: 11.5, borderCollapse: 'collapse' }}>
              <tbody>
                {b.history.slice(-6).reverse().flatMap((h: any, i: number) => [
                  <tr key={`h${i}`} style={{ borderTop: '1px solid rgba(148,163,184,0.18)' }}>
                    <td style={{ padding: '4px 6px', fontWeight: 700 }}>wk {h.week}</td>
                    <td>credit {h.credit0}</td>
                    <td>{h.reason}</td>
                    <td style={{ fontWeight: 700,
                      color: h.net_rs >= 0 ? '#0ca30c' : '#e66767' }}>
                      {(h.net_rs < 0 ? '−₹' : '+₹') + Math.abs(Math.round(h.net_rs)).toLocaleString('en-IN')}
                    </td>
                    <td style={{ opacity: 0.6 }}>closed {nsrwDT(h.closed)}</td>
                  </tr>,
                  ...(h.legs ?? []).map((l: any, j: number) => (
                    <tr key={`h${i}l${j}`} style={{ opacity: 0.62 }}>
                      <td style={{ padding: '2px 6px 2px 14px' }}>{nsrwLeg(l, h.expiry)}</td>
                      <td>in {l.entry} <span style={{ opacity: 0.6 }}>{nsrwDT(l.opened)}</span></td>
                      <td>out {l.exit ?? '—'} <span style={{ opacity: 0.6 }}>{nsrwDT(l.closed)}</span></td>
                      <td>min {l.px_min ?? '—'} · max {l.px_max ?? '—'}</td>
                      <td title={l.reason_detail || ''}>{l.status}
                        {l.reason_detail && (
                          <span style={{ fontSize: 10, opacity: 0.7 }}> — {l.reason_detail}</span>
                        )}
                      </td>
                    </tr>
                  )),
                ])}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

function NsrwPaperCard() {
  const [s, setS] = useState<any>(null);
  useEffect(() => {
    const load = () =>
      fetch('/api/nsrw/state', { cache: 'no-store' })
        .then((r) => r.json())
        .then(setS)
        .catch(() => {});
    load();
    const t = setInterval(load, 10000);
    return () => clearInterval(t);
  }, []);
  if (!s || !s.books) return null;
  return (
    <section className={styles.sectionBlock} style={{ marginTop: 14 }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, flexWrap: 'wrap', marginBottom: 8 }}>
        <span style={{ fontSize: 14, fontWeight: 800 }}>NSR-W — Weekly Strangle Paper Books</span>
        <span style={{ fontSize: 11, fontWeight: 700, letterSpacing: 0.4, border: '1px solid #8b8fa3',
          color: '#8b8fa3', borderRadius: 99, padding: '1px 9px' }}>PAPER · 10 LOTS EACH</span>
        <span style={{ marginLeft: 'auto', fontSize: 12, opacity: 0.75 }}>
          <a href="/app/backtest/nifty-strangle-rules-research90" style={{ color: '#3987e5' }}>study</a>
          {' · '}<a href="/app/nsrw-travel-research90.html" style={{ color: '#3987e5' }}>travel report</a>
        </span>
      </div>
      <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap' }}>
        <NsrwBook id="t30" b={s.books.t30} />
        <NsrwBook id="t20" b={s.books.t20} />
      </div>
      <div style={{ fontSize: 11.5, opacity: 0.6, marginTop: 6 }}>{s.spec}</div>
    </section>
  );
}

function SleeveCard({ label, sub, rules, info }: { label: string; sub: string; rules: string; info: any }) {
  const lv = info.live;
  const status: 'live' | 'paper' | 'closed' | 'off' =
    info.state === 'CLOSED' ? 'closed' : info.src === 'live' ? 'live' : info.src === 'paper' ? 'paper' : 'off';
  const stTitle = info.state === 'CLOSED' ? 'Closed for the day' : info.src === 'live' ? 'Live - real money'
    : info.src === 'paper' ? 'Paper - simulated' : 'Not trading today';
  const qty = lv?.qty ?? info.qty ?? 130;
  const legs = lv?.ce_sym ? [
    { side: 'CE', color: '#d29922', e: lv.ce0, l: lv.ce_last },
    { side: 'PE', color: '#a371f7', e: lv.pe0, l: lv.pe_last },
  ] : [];
  const legPnl = (e: number, l: number) => (Number(e) - Number(l)) * qty;
  return (
    <div className={styles.panel}>
      <div className={styles.panelHead}>
        <div className={styles.panelHeadLeft}>
          <div className={styles.panelTitle}>{label}</div>
          <div className={styles.panelSub}>{sub}</div>
        </div>
        <div className={styles.panelStatus}>
          <StatusDot kind={status} title={stTitle} />
          <div className={styles.panelStatusMeta}>{info.lots} lots · {info.state}</div>
        </div>
      </div>
      <div className={styles.metricsRow}>
        <MiniMetric label="Day P&L" value={<span className={pnlClass(info.pnl)}>{formatPnl(info.pnl)}</span>} />
        <MiniMetric label="Credit" value={lv?.credit ?? info.rec?.credit ?? '—'} />
      </div>
      {info.series.length >= 2 ? <NsrwSpark series={info.series} /> : null}
      <div className={styles.legs}>
        {legs.length === 0 ? (
          <div className={styles.noLegs}>{info.state === 'CLOSED' ? 'Closed today' : 'Legs appear from Monday'}</div>
        ) : legs.map((lg) => (
          <div key={lg.side} style={{ display: 'flex', alignItems: 'center', gap: 10, fontSize: 13, padding: '3px 7px' }}>
            <span style={{ fontWeight: 700, color: lg.color }}>{lg.side}</span>
            <span>{lv.K}</span>
            <span style={{ color: 'var(--ink-muted)' }}>x{qty}</span>
            <span style={{ color: 'var(--ink-muted)' }}>{Number(lg.e).toFixed(2)} @{lv.entry_ts} → {Number(lg.l).toFixed(2)}</span>
            <span className={pnlClass(legPnl(lg.e, lg.l))} style={{ marginLeft: 'auto', fontWeight: 600 }}>{formatPnl(legPnl(lg.e, lg.l))}</span>
          </div>
        ))}
      </div>
      <details className={styles.rules}>
        <summary className={styles.rulesSummary}>Rules &amp; snapshot</summary>
        <div className={styles.rulesBody}><div className={styles.rulesText}>{rules}</div></div>
      </details>
    </div>
  );
}

function Collapsible({ title, meta, defaultOpen = true, children }: { title: string; meta?: any; defaultOpen?: boolean; children: any }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section style={{ marginTop: 24, border: '1px solid var(--line)', borderRadius: 12, overflow: 'hidden', background: 'var(--card-bg, transparent)' }}>
      <button
        onClick={() => setOpen((o) => !o)}
        style={{
          width: '100%', display: 'flex', alignItems: 'center', gap: 10, padding: '13px 16px',
          background: 'rgba(0,0,0,0.025)', border: 'none', borderBottom: open ? '1px solid var(--line)' : 'none',
          cursor: 'pointer', textAlign: 'left',
        }}
      >
        <span style={{ display: 'inline-block', transition: 'transform 0.15s ease', transform: open ? 'rotate(90deg)' : 'none', color: 'var(--ink-muted)', fontSize: 11 }}>▶</span>
        <span className="section-title" style={{ margin: 0 }}>{title}</span>
        {meta ? <span style={{ fontSize: 12, color: 'var(--ink-muted)', marginLeft: 4 }}>{meta}</span> : null}
        <span style={{ marginLeft: 'auto', fontSize: 11, color: 'var(--ink-faint, #9aa0a6)' }}>{open ? 'collapse' : 'expand'}</span>
      </button>
      {open ? <div style={{ padding: 16 }}>{children}</div> : null}
    </section>
  );
}

export default function Nas() {
  const [states, setStates] = useState<Record<string, SystemStateRecord>>({});
  const [toast, setToast] = useState<string | null>(null);
  const [mtmData, setMtmData] = useState<Record<string, MtmSystem>>({});
  const [mtmCombined, setMtmCombined] = useState<MtmSystem | null>(null);
  const [expandedKey, setExpandedKey] = useState<string | null>(null);
  const [historyDays, setHistoryDays] = useState<Array<{
    date: string; combined_last: number; n_fired: number; n_systems: number;
    combined_last_2lot?: number;
  }>>([]);
  // Past sessions were traded at 1-10 lots across two NIFTY lot sizes, so the as-traded
  // series is not comparable day to day. Default to the 2-lot restatement; 'as traded' is
  // what actually hit the account and stays one click away.
  const [histBasis, setHistBasis] = useState<'per2' | 'raw'>(
    () => (localStorage.getItem('nasHistBasis') === 'raw' ? 'raw' : 'per2'),
  );
  const setBasis = (b: 'per2' | 'raw') => {
    setHistBasis(b);
    localStorage.setItem('nasHistBasis', b);
  };
  const [historyModal, setHistoryModal] = useState<{
    title: string; points: MtmPoint[]; events: MtmEvent[];
    points2lot?: MtmPoint[];
  } | null>(null);

  // COMB + TimeB sleeves (research/111 paper/live books) — static json, no backend.
  const [cslLive, setCslLive] = useState<any>(null);
  const [cslDay, setCslDay] = useState<any>(null);
  useEffect(() => {
    const load = () => {
      fetch(`/app/csl_paper_live.json?t=${Date.now()}`, { cache: 'no-store' })
        .then((r) => r.json()).then(setCslLive).catch(() => {});
      fetch(`/app/csl_paper.json?t=${Date.now()}`, { cache: 'no-store' })
        .then((r) => r.json()).then(setCslDay).catch(() => {});
    };
    load();
    const id = setInterval(load, 30000);
    return () => clearInterval(id);
  }, []);

  // NAS-OPT (paper) — fetched here so it can appear in the Trade Book like the 8 variants.
  const [nasOptTb, setNasOptTb] = useState<SystemStateRecord>({ state: null, err: null });
  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const [s, t] = await Promise.all([
          fetch('/api/nas-opt/state').then((r) => r.json()),
          fetch('/api/nas-opt/trades').then((r) => r.json()),
        ]);
        if (!cancelled) {
          setNasOptTb({ state: nasOptTradeBookState(s?.today, Array.isArray(t) ? t : []), err: null });
        }
      } catch {
        /* endpoint may be momentarily unavailable */
      }
    };
    load();
    const id = setInterval(load, 30000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  // Load saved daily snapshots (written by scripts/snapshot_nas_eod.py at 15:32).
  useEffect(() => {
    fetch(`/static/snapshots/index.json?t=${Date.now()}`, { cache: 'no-store' })
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => { if (d?.days) setHistoryDays(d.days); })
      .catch(() => { /* no snapshots yet */ });
  }, []);

  async function openHistory(d: string) {
    try {
      const r = await fetch(`/static/snapshots/nas_mtm_${d}.json?t=${Date.now()}`,
        { cache: 'no-store' });
      if (!r.ok) return;
      const data = await r.json();
      const c = data.combined ?? {};
      // combined_2lot: the same whole-book curve with every system restated at 2 lots
      // (scripts/normalize_snapshots.py). Absent on days written before the restatement.
      const c2 = data.combined_2lot ?? {};
      setHistoryModal({
        title: `NAS overall — ${d}`,
        points: c.points ?? [],
        events: c.events ?? [],
        points2lot: c2.points ?? undefined,
      });
    } catch { /* swallow */ }
  }
  const [liveTicks, setLiveTicks] = useState<LiveTicks>({
    spot: null,
    legs: {},
    connected: false,
    highs: {},
  });
  const evtRef = useRef<EventSource | null>(null);

  // Poll per-leg DAY HIGH (max traded premium) written by scripts/leg_day_highs.py (cron, 1/min).
  useEffect(() => {
    let dead = false;
    const load = () =>
      fetch(`/app/leg_highs.json?t=${Date.now()}`, { cache: 'no-store' })
        .then((r) => r.json())
        .then((d) => {
          if (!dead && d && d.highs) setLiveTicks((prev) => ({ ...prev, highs: d.highs }));
        })
        .catch(() => {});
    load();
    const id = setInterval(load, 30000);
    return () => {
      dead = true;
      clearInterval(id);
    };
  }, []);

  // Poll the per-system intraday MTM curves every 30s. Cheap (one row per
  // system per 3 min), reuses the same snapshots the EOD report renders.
  useEffect(() => {
    let cancelled = false;
    async function pull() {
      // Read the cron-written static dump (cache-busted). Doesn't depend on
      // any backend route registration; works even if gunicorn is running
      // older code than this bundle.
      try {
        let data: {
          systems: Record<string, MtmSystem>;
          combined?: MtmSystem;
        } | null = null;
        try {
          const resp = await fetch(`/static/nas_mtm.json?t=${Date.now()}`,
            { cache: 'no-store' });
          if (resp.ok) data = await resp.json();
        } catch { /* try API fallback */ }
        if (!data) {
          data = await apiGet<{
            systems: Record<string, MtmSystem>;
            combined?: MtmSystem;
          }>('/api/nas/mtm');
        }
        if (cancelled || !data?.systems) return;
        const next: Record<string, MtmSystem> = {};
        for (const k of Object.keys(data.systems)) {
          next[k] = {
            points: data.systems[k]?.points ?? [],
            events: data.systems[k]?.events ?? [],
          };
        }
        setMtmData(next);
        setMtmCombined(data.combined
          ? { points: data.combined.points ?? [],
              events: data.combined.events ?? [] }
          : null);
      } catch { /* sparkline stays in empty state */ }
    }
    pull();
    const t = setInterval(pull, 30000);
    return () => { cancelled = true; clearInterval(t); };
  }, []);

  function updateState(id: string, rec: SystemStateRecord) {
    setStates((prev) => ({ ...prev, [id]: rec }));
  }

  // One SSE connection for the entire dashboard — pushes spot + all option
  // leg LTPs across 8 systems in a single stream. Reconnects on error.
  useEffect(() => {
    let cancelled = false;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    const open = () => {
      if (cancelled) return;
      const es = new EventSource('/api/nas/stream');
      evtRef.current = es;
      es.onopen = () => {
        if (!cancelled) setLiveTicks((prev) => ({ ...prev, connected: true }));
      };
      es.onmessage = (ev) => {
        if (cancelled) return;
        try {
          const d = JSON.parse(ev.data);
          if (d.type === 'tick') {
            const legs: Record<string, number> = {};
            for (const [tsym, info] of Object.entries(d.legs || {})) {
              const ltp = (info as { ltp?: number }).ltp;
              if (typeof ltp === 'number') legs[tsym] = ltp;
            }
            setLiveTicks((prev) => ({
              ...prev,
              spot: typeof d.spot === 'number' && d.spot > 0 ? d.spot : null,
              legs,
              connected: true,
            }));
          } else if (d.type === 'offline') {
            setLiveTicks((prev) => ({ ...prev, connected: false }));
          }
        } catch {
          /* ignore malformed payload */
        }
      };
      es.onerror = () => {
        if (cancelled) return;
        setLiveTicks((prev) => ({ ...prev, connected: false }));
        es.close();
        evtRef.current = null;
        reconnectTimer = setTimeout(open, 3000);
      };
    };

    open();
    return () => {
      cancelled = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (evtRef.current) {
        evtRef.current.close();
        evtRef.current = null;
      }
    };
  }, []);

  const squeezeSystems = SQUEEZE_SYSTEMS.map((s) => states[s.id]?.state).filter(
    Boolean,
  ) as NASState[];
  const nineSixteenSystems = ENTRY_916_SYSTEMS.map((s) => states[s.id]?.state).filter(
    Boolean,
  ) as NASState[];

  // Pick the first squeeze state that has ATR/squeeze data for the shared header.
  const headerState: NASState | null = useMemo(() => {
    for (const s of SQUEEZE_SYSTEMS) {
      const rec = states[s.id]?.state;
      if (rec?.state && typeof rec.state.atr_value === 'number') return rec;
    }
    return states[SQUEEZE_SYSTEMS[0].id]?.state ?? null;
  }, [states]);

  // Per-system day P&L = DB-persisted today_pnl (closed trades) + live open-leg P&L.
  // Open-leg P&L = sum of (entry_price - ltp) * qty across CE + PE legs (we short).
  // Prefer live SSE tick LTP over polled state LTP when available.
  const liveSystemPnl = (s: NASState | undefined | null): number => {
    if (!s) return 0;
    const persisted = (s.stats?.today_pnl as number | undefined) ?? 0;
    const legs = [...(s.positions?.ce ?? []), ...(s.positions?.pe ?? [])];
    const open = legs.reduce((acc, p) => {
      const entry = p.entry_price ?? p.entry_premium;
      const liveLtp = p.tradingsymbol ? liveTicks.legs[p.tradingsymbol] : undefined;
      const ltp = liveLtp ?? p.ltp;
      const qty = p.qty ?? 0;
      if (entry == null || ltp == null || !qty) return acc;
      return acc + (entry - ltp) * qty;
    }, 0);
    return persisted + open;
  };
  const squeezeDayPnl = squeezeSystems.reduce((acc, s) => acc + liveSystemPnl(s), 0);
  const nineSixteenDayPnl = nineSixteenSystems.reduce((acc, s) => acc + liveSystemPnl(s), 0);
  const nineSixteenActive = nineSixteenSystems.filter(
    (s) => (((s?.positions?.ce?.length ?? 0) + (s?.positions?.pe?.length ?? 0)) > 0),
  ).length;

  const core = headerState?.state ?? {};
  const isSqueezing = !!core.is_squeezing;
  // Prefer live SSE spot over polled state spot when the stream is up.
  const pollSpot = core.spot_price as number | undefined;
  const spot = liveTicks.spot ?? pollSpot;
  const atr = core.atr_value as number | undefined;
  const atrMa = core.atr_ma as number | undefined;

  // Market hours check (IST 09:15-15:30)
  const nowIst = new Date();
  const mins = nowIst.getHours() * 60 + nowIst.getMinutes();
  const marketOpen = mins >= 9 * 60 + 15 && mins <= 15 * 60 + 30;
  const hasData = atr !== undefined && atrMa !== undefined;

  // Squeeze dot kind: green if active + has data, red if no squeeze + has data, grey if no data/market closed
  let squeezeDotKind: 'connected' | 'disconnected' | 'warning' = 'warning';
  if (!hasData || !marketOpen) {
    squeezeDotKind = 'warning'; // grey
  } else if (isSqueezing) {
    squeezeDotKind = 'connected'; // green
  } else {
    squeezeDotKind = 'disconnected'; // red
  }

  // Margin shape (served by backend's _orb_get_margin):
  //   available  = eq.net (Kite UI 'Available margin', total pool)
  //   cash_cap   = min(2 × live_balance, net) — actual cap on new F&O margin
  //                under SEBI's 50:50 cash:collateral rule. This is what
  //                the trader can ACTUALLY size against today.
  //   live_balance = real free cash right now
  const margin = headerState?.margin as
    | { available?: number; cash_cap?: number; live_balance?: number }
    | undefined;
  // Display the cash-constrained cap as the primary number — matches the
  // trader's sizing intuition. Show total pool in the hint for context.
  const cashCap = margin?.cash_cap;
  const totalPool = margin?.available;

  function showToast(msg: string) {
    setToast(msg);
    setTimeout(() => setToast(null), 2500);
  }

  /* ---------- whats next schedule ---------- */

  const nextEvents = useMemo(() => buildNextEvents(states), [states]);

  // 9:16 and Squeeze sub-curves for the combined "Overall NAS" modal — sum each
  // family's per-system P&L series (forward-filled across timestamps).
  const squeezeMtmPts = useMemo(
    () => sumSeries(SQUEEZE_SYSTEMS.map((s) => mtmData[s.key]?.points ?? [])), [mtmData]);
  const nineMtmPts = useMemo(
    () => sumSeries(ENTRY_916_SYSTEMS.map((s) => mtmData[s.key]?.points ?? [])), [mtmData]);

  // COMB + TimeB sleeves: today's series/P&L from the live json, cfg/source from the day record.
  const SLEEVES: Array<{ key: string; label: string; hint: string }> = [
    { key: 'NAS_COMB20', label: 'COMB sleeve', hint: 'combined-SL · ex-Wed' },
    { key: 'CSL_TIMEB_NIFTY', label: 'TimeB', hint: 'time-windows · ex-Wed' },
  ];
  const sleeveInfo = (bk: string) => {
    const live = cslLive?.books?.[bk];
    const rec = (cslDay?.records ?? []).find((r: any) => r.day === cslLive?.day && r.book === bk);
    const series: [string, number][] = (live?.series?.length ? live.series : (rec?.series ?? [])) as [string, number][];
    const pnl = rec ? Number(rec.pnl || 0) : (series.length ? Number(series[series.length - 1][1]) : 0);
    const state = rec ? 'CLOSED' : (live?.state ?? '—');
    const src = rec ? (rec.source === 'REAL' ? 'live' : 'paper') : null;
    const lots = rec ? Number(rec.lots || 2) : 2;
    const qty = rec ? Number(rec.qty || 130) : 130;
    return { live, series, pnl, state, src, rec, lots, qty };
  };
  // Uniform card status: green live / blue paper / grey closed / faint off.
  const famStatus = (defs: SystemDef[]): 'live' | 'paper' | 'closed' | 'off' => {
    let anyLive = false, anyActive = false;
    for (const sd of defs) {
      const pos = states[sd.id]?.state?.positions as any;
      const legs = [...((pos?.ce) ?? []), ...((pos?.pe) ?? [])];
      if (legs.length) anyActive = true;
      if (legs.some((l: any) => (l.mode || '').toLowerCase() === 'live')) anyLive = true;
    }
    if (anyLive) return 'live';
    if (anyActive) return 'paper';
    return marketOpen ? 'closed' : 'off';
  };
  const sleeveStatus = (info: any): 'live' | 'paper' | 'closed' | 'off' =>
    info.state === 'CLOSED' ? 'closed' : info.src === 'live' ? 'live'
      : info.src === 'paper' ? 'paper' : (marketOpen ? 'paper' : 'off');
  // Trade-Book state for a sleeve: OPEN -> CE/PE legs; WAIT_ENTRY -> planned window (by today's DTE).
  const TB_WIN: Record<string, Record<number, [string, string, number | string]>> = {
    NAS_COMB20: { 0: ['09:16', '15:20', 25], 1: ['09:16', '15:20', 30], 2: ['09:16', '15:20', 30], 3: ['09:16', '15:20', 20] },
    CSL_TIMEB_NIFTY: { 0: ['09:30', '11:00', 25], 1: ['13:00', '14:00', 20], 2: ['10:00', '12:00', 20] },
    CSL_TIMEB_SENSEX: { 0: ['13:00', '15:20', 'none'], 1: ['10:30', '12:00', 20] },
    CSL30F_SENSEX: { 0: ['09:16', '15:20', 30], 1: ['09:16', '15:20', 30], 2: ['09:16', '15:20', 30], 3: ['09:16', '15:20', 30], 4: ['09:16', '15:20', 30] },
  };
  const [sxLive, setSxLive] = useState<any>(null);
  useEffect(() => {
    const load = () =>
      fetch(`/app/sensex_live.json?t=${Date.now()}`, { cache: 'no-store' })
        .then((r) => r.json()).then(setSxLive).catch(() => {});
    load();
    const id = setInterval(load, 8000);
    return () => clearInterval(id);
  }, []);
  const sxTbState = (label: string): SystemStateRecord => {
    const sys = (sxLive?.systems || []).find((x: any) => x.label === label);
    if (!sys || !(sys.legs || []).length) return { state: null, err: null };
    const mk = (l: any): any => ({
      leg: l.cp, strike: l.strike, qty: l.qty, entry_price: l.entry,
      ltp: l.ltp, exit_price: l.status === 'ACTIVE' ? null : l.ltp,
      pnl_inr: l.pnl, mode: l.mode, status: l.status === 'ACTIVE' ? 'ACTIVE' : 'CLOSED',
      exit_reason: l.status === 'ACTIVE' ? undefined : l.status,
      sl_price: (typeof l.arm === 'string' && l.arm.startsWith('SL ')) ? parseFloat(l.arm.slice(3)) : undefined,
      entry_time: '09:16',
    });
    const act = (sys.legs || []).filter((l: any) => l.status === 'ACTIVE');
    const done = (sys.legs || []).filter((l: any) => l.status !== 'ACTIVE');
    return { state: { positions: {
      ce: act.filter((l: any) => l.cp === 'CE').map(mk),
      pe: act.filter((l: any) => l.cp === 'PE').map(mk),
      closed_today: done.map(mk) } } as any, err: null };
  };
  const sleeveTbState = (bk: string, qty: number): SystemStateRecord => {
    const b: any = cslLive?.books?.[bk];
    if (!b) return { state: null, err: null };
    // venue-aware trading-DTE: executor-published if present, else weekday table
    // (NIFTY Tue-expiry: Mon..Fri -> 1,0,4,3,2 · SENSEX Thu-expiry: 3,2,1,0,4).
    // Fixes the Wed bug where COMB (ex-Wed) carried no dte and armed rows vanished.
    const wd = new Date().getDay() - 1;
    const dteTbl = bk.includes('SENSEX') ? [3, 2, 1, 0, 4] : [1, 0, 4, 3, 2];
    const dte = b.dte ?? (bk.includes('SENSEX') ? null : cslLive?.books?.NAS_COMB20?.dte) ?? (wd >= 0 && wd <= 4 ? dteTbl[wd] : null);
    const w = (dte != null && TB_WIN[bk]) ? TB_WIN[bk][dte] : null;
    if (b.state === 'OPEN' && b.ce_sym) {
      const slNone = b.sl === 'none' || b.sl == null;
      const thr = b.credit
        ? Math.round((slNone ? 1.5 : 1 + b.sl / 100) * b.credit)
        : null;
      const curComb = (Number(b.ce_last) || 0) + (Number(b.pe_last) || 0);
      const entryComb = Number(b.credit) || ((Number(b.ce0) || 0) + (Number(b.pe0) || 0));
      const armTxt = thr != null
        ? `${entryComb.toFixed(1)} · ${thr}${slNone ? ' bkstp' : ''} (${curComb.toFixed(1)})`
        : undefined;
      const mk = (leg: string, sym: string, e: number, l: number): any => ({
        leg, tradingsymbol: sym, strike: b.K, qty, entry_price: e, ltp: l,
        mode: b.live ? 'live' : 'paper', entry_time: b.entry_ts, status: 'ACTIVE', sl_price: thr, arm_text: armTxt,
        exit_planned: w ? w[1] : undefined,
      });
      return { state: { positions: { ce: [mk('CE', b.ce_sym, b.ce0, b.ce_last)], pe: [mk('PE', b.pe_sym, b.pe0, b.pe_last)], closed_today: [] } } as any, err: null };
    }
    if (b.state === 'WAIT_ENTRY' && w) {
      return { state: { planned: { entry: w[0], exit: w[1], sl: w[2], qty, mode: b.live === false ? 'paper' : 'live' } } as any, err: null };
    }
    // CLOSED: sleeve traded and exited today -> keep it in the book as one done combined
    // straddle line (the live book is nulled on exit; the record holds combined credit/exit).
    const rec: any = (cslDay?.records ?? []).find((r: any) => r.day === cslLive?.day && r.book === bk);
    if (rec) {
      const closed: any = {
        leg: 'C+P', tradingsymbol: undefined, strike: rec.strike, qty: rec.qty ?? qty,
        entry_price: rec.credit, exit_price: rec.exit_comb, pnl_inr: rec.pnl,
        entry_time: rec.entry_ts, exit_time: rec.exit_ts, exit_reason: rec.reason,
        status: 'CLOSED', mode: rec.source === 'REAL' ? 'live' : 'paper',
      };
      return { state: { positions: { ce: [], pe: [], closed_today: [closed] } } as any, err: null };
    }
    return { state: null, err: null };
  };
  // Merge the sleeve intraday series into the Overall curve (convert HH:MM -> ISO to match mtm).
  const sleevePts: MtmPoint[] = useMemo(() => {
    const day = cslLive?.day;
    if (!day) return [];
    const iso = (hm: string) => `${day}T${hm}:00`;
    const lists = SLEEVES
      .map((sv) => (cslLive?.books?.[sv.key]?.series ?? []) as [string, number][])
      .filter((x) => x.length)
      .map((x) => x.map(([hm, v]) => [iso(hm), v] as MtmPoint));
    return lists.length ? sumSeries(lists) : [];
  }, [cslLive]);
  const [sxMtm, setSxMtm] = useState<any>(null);
  useEffect(() => {
    const load = () =>
      fetch(`/app/sensex_mtm.json?t=${Date.now()}`, { cache: 'no-store' })
        .then((r) => r.json()).then(setSxMtm).catch(() => {});
    load();
    const id = setInterval(load, 15000);
    return () => clearInterval(id);
  }, []);
  const sxPts: MtmPoint[] = useMemo(() => (sxMtm?.points ?? []) as MtmPoint[], [sxMtm]);
  const overallPts: MtmPoint[] = useMemo(() => {
    const lists = [mtmCombined?.points ?? [], sleevePts, sxPts].filter((x) => x.length);
    if (!lists.length) return [];
    return lists.length === 1 ? lists[0] : sumSeries(lists);
  }, [mtmCombined, sleevePts, sxPts]);

  return (
    <LiveTicksContext.Provider value={liveTicks}>
    <div className={styles.root}>
      {/* Tier 1 (exchange-side SL-M) not yet built — remove this block when it ships. */}
      <div className={styles.slmWarning} role="alert">
        <span className={styles.slmWarningIcon} aria-hidden="true">⚠</span>
        <div className={styles.slmWarningText}>
          <strong>NAS LIVE — exchange-side SL-M not yet implemented.</strong>
          <span className={styles.slmWarningDetail}>
            {' '}If Flask or ticker dies during an open position, the short is
            unprotected until the process recovers. Tier 1 build pending.
          </span>
        </div>
      </div>

      <div className={styles.titleRow}>
        <div>
          <div className="page-title">NAS options</div>
          <div className="page-subtitle">
            Eight Nifty options systems running in parallel. ATR squeeze entries on the
            left, time-based 9:16 entries on the right.
          </div>
        </div>
        <div className={styles.titleRowActions}>
          <MasterModeToggle onToast={setToast} />
          <Link
            to="/nas-panic"
            className={styles.panicLink}
            title="Closes all positions and disables all 8 NAS variants. Survives Flask/VPS restart."
          >
            ⚠ Emergency stop
          </Link>
        </div>
      </div>

      {toast ? <div className={styles.toast}>{toast}</div> : null}

      <ChartModal
        open={!!expandedKey}
        title={
          expandedKey === '_combined'
            ? 'Overall NAS — Intraday P&L'
            : ((ALL_SYSTEMS.find((s) => s.key === expandedKey)?.label ?? '') +
               ' — Intraday P&L')
        }
        points={
          expandedKey === '_combined'
            ? overallPts
            : expandedKey
            ? (mtmData[expandedKey]?.points || [])
            : []
        }
        events={
          expandedKey === '_combined'
            ? (mtmCombined?.events || [])
            : expandedKey
            ? (mtmData[expandedKey]?.events || [])
            : []
        }
        extraSeries={expandedKey === '_combined' ? [
          { label: 'NIFTY 9:16', color: '#3b82f6', points: nineMtmPts },
          { label: 'Squeeze', color: '#f59e0b', points: squeezeMtmPts },
          { label: 'Sleeves', color: '#a371f7', points: sleevePts },
          { label: 'SENSEX', color: '#0F6E56', points: sxPts },
        ].filter((x) => x.points.length) : undefined}
        onClose={() => setExpandedKey(null)}
      />

      <ChartModal
        open={!!historyModal}
        title={
          (historyModal?.title ?? '') +
          (histBasis === 'per2' && historyModal?.points2lot ? '  · per 2 lots' : '  · as traded')
        }
        points={
          histBasis === 'per2'
            ? (historyModal?.points2lot ?? historyModal?.points ?? [])
            : (historyModal?.points ?? [])
        }
        events={historyModal?.events ?? []}
        onClose={() => setHistoryModal(null)}
      />

      {/* Shared ATR squeeze header */}
      <div className={styles.headerMetrics}>
        <MetricCard
          label="ATR squeeze"
          value={
            <span className={styles.squeezeValue}>
              <StatusDot kind={squeezeDotKind} className={styles.squeezeDot} />
              <span>
                {!hasData || !marketOpen
                  ? '—'
                  : isSqueezing
                  ? 'Squeeze'
                  : 'Normal'}
              </span>
            </span>
          }
          hint={
            hasData
              ? `ATR ${formatNumber(atr)} / MA ${formatNumber(atrMa)}`
              : 'ATR(14) vs SMA(ATR,50)'
          }
        />
        <MetricCard
          label="Nifty spot"
          value={spot !== undefined ? formatNumber(spot) : '—'}
          hint="Live index price"
        />
        <MetricCard
          label="Available margin"
          // Cash-constrained F&O cap (SEBI 50:50 rule), NOT Kite UI's
          // 'Available margin' (which is the larger total pool).
          value={cashCap !== undefined ? formatRs(cashCap) : '—'}
          hint={
            totalPool !== undefined
              ? `Max new F&O margin · 50:50 cash rule (Total pool ${formatRs(totalPool)})`
              : 'Max new F&O margin (50:50 cash rule)'
          }
        />
        <MetricCard
          label="Squeeze day P&L"
          value={
            <span className={pnlClass(squeezeDayPnl)}>
              {formatPnl(squeezeDayPnl)}
            </span>
          }
          hint="OTM + ATM + ATM 2.0 + ATM V4"
          status={famStatus(SQUEEZE_SYSTEMS)}
        />
        <MetricCard
          label="9:16 day P&L"
          value={
            <span className={pnlClass(nineSixteenDayPnl)}>
              {formatPnl(nineSixteenDayPnl)}
            </span>
          }
          hint={`${nineSixteenActive} of ${ENTRY_916_SYSTEMS.length} systems traded today`}
          status={famStatus(ENTRY_916_SYSTEMS)}
        />
        {SLEEVES.map((sv) => {
          const info = sleeveInfo(sv.key);
          return (
            <MetricCard
              key={sv.key}
              label={sv.label}
              value={<span className={pnlClass(info.pnl)}>{formatPnl(info.pnl)}</span>}
              hint={`${info.lots} lots (${info.qty} qty) · ${sv.hint}`}
              status={sleeveStatus(info)}
            />
          );
        })}
      </div>

      {mtmCombined && mtmCombined.points.length >= 2 ? (
        <section className={styles.combinedHero}>
          <div className={styles.combinedHead}>
            <div className="section-title">Overall NAS · intraday P&amp;L</div>
            <div className={styles.combinedMeta}>
              {mtmCombined.points.length} pts · click to expand
            </div>
          </div>
          <button
            type="button"
            className={styles.sparkButton}
            onClick={() => setExpandedKey('_combined')}
            title="Expand combined chart"
          >
            <div className={styles.combinedBox}>
              <PnlChart
                points={overallPts.length ? overallPts : mtmCombined.points}
                events={mtmCombined.events}
              />
              <div className={styles.sparkMeta}>
                {(() => {
                  const ys = (overallPts.length ? overallPts : mtmCombined.points).map((p) => p[1]);
                  const last = ys[ys.length - 1];
                  const yMin = Math.min(0, ...ys);
                  const yMax = Math.max(0, ...ys);
                  return (
                    <>
                      <span className={last >= 0 ? styles.sparkPos : styles.sparkNeg}>
                        now {fmtPnl(last)}
                      </span>
                      <span className={styles.sparkRange}>
                        lo {fmtPnl(yMin)} · hi {fmtPnl(yMax)} · 8 systems + sleeves ⤢
                      </span>
                    </>
                  );
                })()}
              </div>
            </div>
          </button>
        </section>
      ) : null}

      {historyDays.length > 0 ? (
        <section className={styles.historyStrip}>
          <div className={styles.historyHead} style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
            {(() => {
              const tot = historyDays.reduce(
                (a, d) => a + (histBasis === 'per2' ? (d.combined_last_2lot ?? d.combined_last) : d.combined_last),
                0,
              );
              return (
                <span>
                  Past sessions · {historyDays.length} days ·{' '}
                  <span style={{ color: tot >= 0 ? '#3fb950' : '#ef4444', fontWeight: 700 }}>
                    {fmtPnl(tot)}
                  </span>{' '}
                  total · click to view
                </span>
              );
            })()}
            <span style={{ display: 'inline-flex', gap: 4 }}>
              {(['per2', 'raw'] as const).map((b) => (
                <button
                  key={b}
                  type="button"
                  onClick={() => setBasis(b)}
                  title={
                    b === 'per2'
                      ? 'Every day restated at a uniform 2 lots, so sessions are comparable. The book actually traded 1-10 lots at different times, which makes the raw series jump ~10x on 25 Jun for no reason other than size.'
                      : 'Exactly what hit the account that day, at whatever size was traded (1-10 lots).'
                  }
                  style={{
                    background: histBasis === b ? 'var(--line)' : 'transparent',
                    border: '1px solid var(--line)', borderRadius: 5,
                    padding: '1px 7px', fontSize: 10, cursor: 'pointer',
                    color: histBasis === b ? 'var(--ink)' : 'var(--ink-muted)',
                    fontWeight: histBasis === b ? 700 : 400,
                  }}
                >
                  {b === 'per2' ? 'per 2 lots' : 'as traded'}
                </button>
              ))}
            </span>
          </div>
          <div className={styles.historyList}>
            {historyDays.map((d) => (
              <button
                key={d.date}
                type="button"
                className={styles.historyChip}
                onClick={() => openHistory(d.date)}
              >
                <div className={styles.historyDate}>{d.date}</div>
                {(() => {
                  const per2 = d.combined_last_2lot ?? d.combined_last;
                  const v = histBasis === 'per2' ? per2 : d.combined_last;
                  const other = histBasis === 'per2' ? d.combined_last : per2;
                  return (
                    <div
                      className={v >= 0 ? styles.sparkPos : styles.sparkNeg}
                      title={`${histBasis === 'per2' ? 'Restated at 2 lots' : 'As traded'}: ${fmtPnl(v)}  ·  ${histBasis === 'per2' ? 'as traded' : 'at 2 lots'}: ${fmtPnl(other)}`}
                    >
                      {fmtPnl(v)}
                    </div>
                  );
                })()}
                <div className={styles.historyMeta}>
                  {d.n_fired}/{d.n_systems} fired
                </div>
              </button>
            ))}
          </div>
        </section>
      ) : null}

      {/* Trade Book — grouped active+closed trades with group P&L (EOD report). Paper legs are
          restated at 2 lots on the 'per 2 lots' basis; live legs always show as traded. */}
      <Collapsible title="Trade Book" meta="NAS positions - live + closed today" defaultOpen>
        <TradeBook
          systems={[...ENTRY_916_SYSTEMS, ...SENSEX_TB_DEFS, ...SLEEVE_TB_DEFS, NAS_OPT_DEF, ...SQUEEZE_SYSTEMS]}
          states={{ ...states, 'nas-opt': nasOptTb, 'csl-comb': sleeveTbState('NAS_COMB20', 130), 'csl-timeb': sleeveTbState('CSL_TIMEB_NIFTY', 520), 'csl-comb-sx': sleeveTbState('CSL30F_SENSEX', 60), 'csl-timeb-sx': sleeveTbState('CSL_TIMEB_SENSEX', 160), 'sx-atm': sxTbState('SENSEX ATM'), 'sx-atm2': sxTbState('SENSEX ATM2'), 'sx-atm4': sxTbState('SENSEX ATM4') }}
          liveLegs={liveTicks.legs}
          basis={histBasis}
        />
      </Collapsible>

      {/* Live NIFTY index + SENSEX positions — moved below the NAS trade book so NAS
          positions sit with the NAS cards/curve (user 2026-08-14). */}
      <NiftyChart />

      <Collapsible title="NIFTY · systems" meta="9:16 + Squeeze + COMB/TimeB sleeves" defaultOpen>
      {/* Section 1 — 9:16 ATMs */}
      <section className={styles.sectionBlock} style={{ marginTop: 8 }}>
        <div className={styles.colHead}>
          <div className="section-title">NIFTY ATMs</div>
          <Chip>{ENTRY_916_SYSTEMS.filter((x) => !HIDDEN_CARDS.has(x.id)).length} systems</Chip>
        </div>
        <div className={styles.grid3}>
          {ENTRY_916_SYSTEMS.filter((x) => !HIDDEN_CARDS.has(x.id)).map((s) => (
            <SystemPanel key={s.id} def={s} onStateChange={(rec) => updateState(s.id, rec)} onToast={showToast} series={mtmData[s.key]?.points || []} events={mtmData[s.key]?.events || []} onExpand={() => setExpandedKey(s.key)} />
          ))}
        </div>
      </section>

      {/* Section 2 — COMB + TimeB sleeves (3rd slot reserved) */}
      <section className={styles.sectionBlock} style={{ marginTop: 22 }}>
        <div className={styles.colHead}>
          <div className="section-title">COMB + TimeB sleeves</div>
          <Chip>NIFTY 2-lot · ex-Wed · 2 books</Chip>
        </div>
        <div className={styles.grid3}>
          <SleeveCard label="NIFTY COMB" sub="Full-day combined-SL · ex-Wed" rules="09:16 to 15:20. Combined-premium SL per DTE (DTE0 25 / DTE1 30 / DTE2 30 / DTE3 20). Replaces per-leg SLs + trail. 2 lots — except Thursday, which trades 5 lots (the two former Thursday books merged into this single trade, 19-Aug). Wednesday off." info={sleeveInfo('NAS_COMB20')} />
          <SleeveCard label="NIFTY TimeB" sub="Windowed combined-SL · Mon/Tue/Fri" rules="Per-DTE entry-to-exit windows + SL: DTE0 09:30-11:00 SL25 / DTE1 13:00-14:00 SL20 / DTE2 10:00-12:00 SL20 / DTE3 full-day SL20. 2 lots, Wednesday off. Frozen 13-Aug." info={sleeveInfo('CSL_TIMEB_NIFTY')} />
        </div>
      </section>

      {/* Section 3 — Squeeze ATMs */}
      <section className={styles.sectionBlock} style={{ marginTop: 22 }}>
        <div className={styles.colHead}>
          <div className="section-title">Squeeze · ATMs</div>
          <Chip>{SQUEEZE_SYSTEMS.filter((x) => !HIDDEN_CARDS.has(x.id)).length} systems</Chip>
        </div>
        <div className={styles.grid3}>
          {SQUEEZE_SYSTEMS.filter((x) => !HIDDEN_CARDS.has(x.id)).map((s) => (
            <SystemPanel key={s.id} def={s} onStateChange={(rec) => updateState(s.id, rec)} onToast={showToast} series={mtmData[s.key]?.points || []} events={mtmData[s.key]?.events || []} onExpand={() => setExpandedKey(s.key)} />
          ))}
        </div>
      </section>

      {/* Section 4 — NAS-OPT (lone) */}
      <section className={styles.sectionBlock} style={{ marginTop: 22 }}>
        <div className={styles.colHead}>
          <div className="section-title">NAS-OPT</div>
          <Chip>research/54 · paper</Chip>
        </div>
        <div className={styles.grid3}>
          <NasOptCard />
          <div />
          <div />
        </div>
      </section>

      <CumulativePnL />
      </Collapsible>

      <Collapsible title="SENSEX" meta="Wed/Thu live · else paper" defaultOpen>
        <SensexLiveCard />
        <SensexPaperCard />
        <section className={styles.sectionBlock} style={{ marginTop: 22 }}>
          <div className={styles.colHead}>
            <div className="section-title">SENSEX sleeves</div>
            <Chip>paper · CSL A/B books</Chip>
          </div>
          <div className={styles.grid3}>
            <SleeveCard label="SENSEX COMB · all-week" sub="Full-day · study per-DTE stops · paper" rules="09:16 to 15:20, combined-premium SL 30%. Same construction and stops as the live COMB, but it trades ALL five days — so the ex-Wed rule is measured on real days rather than assumed. 3 lots, paper." info={sleeveInfo('CSL30F_SENSEX')} />
            <SleeveCard label="COMB (SENSEX)" sub="Time-blocked windows + SL · LIVE" rules="Per-DTE from the lab config: Wed 10:30-12:00 SL20 at 8 lots; Thu 13:00-15:20 at 8 lots, no %-SL (50% disaster backstop) - the afternoon decay window, chosen 20-Aug over the full-day for 62% less time-in-market. REAL from 19-Aug." info={sleeveInfo('CSL_TIMEB_SENSEX')} />
            <div />
          </div>
        </section>
      </Collapsible>

      {/* What's next - retained but hidden per user 2026-08-14; set to true to restore */}
      {false && (
      <section className={styles.sectionBlock}>
        <div className={styles.sectionHead}>
          <div className="section-title">What's next</div>
          <Chip>{nextEvents.length} events</Chip>
        </div>
        <div className={styles.eventsTable}>
          <div className={styles.eventsHead}>
            <div>System</div>
            <div>Event</div>
            <div>Scheduled</div>
            <div>Status</div>
            <div className={styles.eventsHeadRight}>In</div>
          </div>
          {nextEvents.map((ev, i) => (
            <div key={`${ev.system}-${ev.event}-${i}`} className={styles.eventsRow}>
              <div className={styles.eventsSystem}>{ev.system}</div>
              <div>{ev.event}</div>
              <div className={styles.eventsTime}>{ev.scheduled}</div>
              <div>
                <span className={`${styles.status} ${styles[`status_${ev.tone}`]}`}>
                  {ev.status}
                </span>
              </div>
              <div className={styles.eventsHeadRight}>
                <span className={styles.mute}>{ev.relative}</span>
              </div>
            </div>
          ))}
        </div>
      </section>
      )}

      <Collapsible title="Integrity watchdog" meta="pipeline / candle-freeze health monitor" defaultOpen={false}>
        <WatchdogSection />
      </Collapsible>

      {/* Cross-book comparison: NAS-916 x3 vs V1+30% combined-premium SL (2026-08-12, research/111) */}
      <section className={styles.sectionBlock}>
        <div className={styles.sectionHead}>
          <div className="section-title">vs V1 + 30% combined-premium SL (NIFTY straddle)</div>
          <Chip>corr 0.04 - independent</Chip>
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ borderCollapse: 'collapse', fontSize: 12.5, minWidth: 520 }}>
            <thead><tr>{['', 'NAS-916 x3', 'V1 + 30% SL', 'Both stacked'].map((h) => (
              <th key={h} style={{ textAlign: 'right', padding: '4px 12px', opacity: 0.6, fontWeight: 600 }}>{h}</th>))}
            </tr></thead>
            <tbody>
              {[['Days', '67', '79', '66 common'],
                ['Total', '+₹4,53,075', '+₹7,56,050', '+₹10,26,039'],
                ['Mean/day', '+₹6,762', '+₹9,570', '+₹15,546'],
                ['MaxDD', '−₹60,669', '−₹1,21,930', '−₹1,11,545'],
                ['Return/DD', '7.5', '6.2', '9.2']].map((r) => (
                <tr key={r[0]}>{r.map((cVal, i) => (
                  <td key={i} style={{ textAlign: i ? 'right' : 'left', padding: '4px 12px',
                    fontWeight: r[0] === 'Return/DD' || i === 3 ? 700 : 400,
                    borderTop: '1px solid rgba(128,128,128,0.15)', fontVariantNumeric: 'tabular-nums' }}>{cVal}</td>))}
                </tr>))}
            </tbody>
          </table>
        </div>
        <div style={{ fontSize: 11, opacity: 0.6, marginTop: 6 }}>
          Daily-P&L correlation 0.04 over 66 common days - independent streams; stacking beats either alone.
          SL30 = recorded-chain backtest @10 lots; NAS = live paper @1-3 lots; single regime Apr-Aug 2026 (as of 12-AUG).
          Rules, visual comparisons &amp; per-system charts: <a href="/app/straddles" style={{ color: 'inherit' }}>Strategy Leaderboard</a> -
          <a href="/app/straddles#sl30-card" style={{ color: 'inherit' }}> SL30 card</a> - <a href="/app/backtest/csl-best-config-straddles" style={{ color: 'inherit' }}>full study card</a>.
        </div>
      </section>

      {/* Config footer */}
      <section className={styles.sectionBlock}>
        <div className={styles.configFooter}>
          <div className={styles.configTitle}>Config overview</div>
          <div className={styles.configList}>
            {ALL_SYSTEMS.map((s) => (
              <div key={s.id} className={styles.configItem}>
                <span className={styles.configSysLabel}>{s.label}</span>
                <span className={styles.configSysNote}>{s.configNote}</span>
              </div>
            ))}
          </div>
        </div>
      </section>
      <Collapsible title="Paper books — NSR-W weekly strangle" meta="paper · 10 lots each · research only" defaultOpen={false}>
        <NsrwPaperCard />
      </Collapsible>

    </div>
    </LiveTicksContext.Provider>
  );
}

/* ---------- Trade Book (grouped active + closed trades, EOD report) ---------- */

type TBGroupMode = 'system' | 'family' | 'all';

interface TBRow {
  sysId: string;
  sysLabel: string;
  family: string;
  side: string;          // 'CE' | 'PE'
  strike: number | null;
  qty: number;
  entry: number | null;
  exit: number | null;   // exit price (closed) or live ltp (open)
  pnl: number;
  open: boolean;
  reason?: string;
  inTime: string;        // entry time HH:MM
  outTime: string;       // exit time HH:MM ('' while open)
  arm: number | null;    // exit/SL monitoring trigger value (premium) while open
  arm_text?: string;     // custom ARM display (sleeve combined-SL: current premium · exit level)
  armLive?: boolean;     // ticker actually subscribed/watching this leg right now
  armLo?: number;        // move-stop systems: NIFTY level below which BOTH legs close + re-center
  armHi?: number;        // move-stop systems: NIFTY level above which BOTH legs close + re-center
  armSpot?: number;      // entry spot the band is anchored to
  mode?: string;         // 'live' | 'paper' -- from 2026-07-14 both trade 130 qty, so size
                         // no longer distinguishes them; this is the only reliable tell
  restated?: boolean;    // paper leg shown at the uniform 130 qty rather than as traded
  tradingsymbol?: string;
}

// Systems whose ONLY exit trigger is the underlying move-stop, not the per-leg premium SL.
// On these the stored sl_price is never evaluated -- nas_atm2_executor.py guards the SL check
// with `if (not move_stop_pct) and live_prem >= sl_price`, always false when a move-stop is set.
// So the arm we display must be the NIFTY band that actually fires.
const MOVE_STOP_PCT: Record<string, number> = {
  'nas-atm2': 0.004,
  'nas-916-atm2': 0.004,
  'nas-opt': 0.004,
};

// Compact status tags so the column stays narrow and the P&L stays next to it.
const REASON_SHORT: Record<string, string> = {
  adj_boundary_exit_no_strike: 'BOUNDARY',
  PHANTOM_PAPER_NO_BROKER: 'PHANTOM',
  MANUAL_USER_ROLL: 'MANUAL',
  time_exit: 'TIME',
  eod_squareoff: 'EOD',
  SL_EXIT_BOTH: 'SL-BOTH',
  SL_HIT: 'SL',
};
function shortReason(r?: string): string {
  if (!r) return 'CLOSED';
  return REASON_SHORT[r] ?? r.toUpperCase();
}

function buildTradeBook(
  systems: SystemDef[],
  states: Record<string, SystemStateRecord>,
  liveLegs: Record<string, number>,
  basis: 'per2' | 'raw' = 'per2',
): TBRow[] {
  const rows: TBRow[] = [];
  for (const sys of systems) {
    const pos = states[sys.id]?.state?.positions;
    if (!pos) {
      const pl = (states[sys.id]?.state as any)?.planned;
      if (pl) {
        rows.push({ sysId: sys.id, sysLabel: sys.label, family: sys.group, side: '—', strike: null,
          qty: pl.qty ?? 0, entry: null, exit: null, pnl: 0, open: false,
          reason: (pl.sl === 'none' || pl.sl == null) ? 'PLANNED · SL none · 50% bkstp' : `PLANNED · SL${pl.sl}`, inTime: pl.entry ? pl.entry + '*' : '', outTime: pl.exit ? pl.exit + '*' : '',
          arm: null, mode: pl.mode ?? 'live' });
      }
      continue;
    }
    const push = (p: NASPosition, open: boolean) => {
      const entry = (p.entry_price ?? p.entry_premium) ?? null;
      const ltp = open
        ? ((p.tradingsymbol ? liveLegs[p.tradingsymbol] : undefined) ?? p.ltp ?? null)
        : (p.exit_price ?? null);
      const rawQty = p.qty ?? 0;
      const isLive = (p.mode || '').toLowerCase() === 'live';
      // Restate PAPER legs at the uniform 2-lot size PER VENUE: NIFTY 2x65=130,
      // SENSEX 2x20=40 (restating SENSEX to 130 inflated its rows 2.17x - user
      // catch 20-Aug). LIVE legs are real money and are NEVER restated.
      const isSensex = (p.tradingsymbol || '').startsWith('SENSEX') || (p.strike ?? 0) >= 40000;
      const targetQty = isSensex ? 40 : TARGET_QTY;
      const sc = basis === 'per2' && !isLive && rawQty > 0 ? targetQty / rawQty : 1;
      const qty = Math.round(rawQty * sc);
      const rawComputed = entry != null && ltp != null && rawQty ? (entry - ltp) * rawQty : 0;
      const pnl = (open ? rawComputed : (p.pnl_inr ?? rawComputed)) * sc;
      const msPct = MOVE_STOP_PCT[sys.id];
      const eSpot = p.entry_spot ?? null;
      const band = open && msPct != null && eSpot != null && eSpot > 0;
      rows.push({
        sysId: sys.id, sysLabel: sys.label, family: sys.group,
        side: (p.leg ?? '').toUpperCase(), strike: p.strike ?? null, qty,
        entry, exit: ltp, pnl, open, reason: p.exit_reason ?? undefined,
        inTime: formatLegTime(p.entry_time) ?? '',
        outTime: open ? ((p as any).exit_planned ? (p as any).exit_planned + '*' : '') : (formatLegTime(p.exit_time) ?? ''),
        arm: open ? (p.sl_price ?? null) : null,
        arm_text: (p as any).arm_text,
        armLive: open ? p.arm_live : undefined,
        armLo: band ? (eSpot as number) * (1 - (msPct as number)) : undefined,
        armHi: band ? (eSpot as number) * (1 + (msPct as number)) : undefined,
        armSpot: band ? (eSpot as number) : undefined,
        mode: (p.mode || '').toLowerCase(),
        restated: sc !== 1,
        tradingsymbol: p.tradingsymbol,
      });
    };
    [...(pos.ce ?? []), ...(pos.pe ?? [])].forEach((p) => push(p, true));
    (pos.closed_today ?? []).forEach((p) => push(p, false));
  }
  return rows;
}

// CE before PE, then open before closed, then strike ascending.
function tbSort(a: TBRow, b: TBRow): number {
  if (a.side !== b.side) return a.side === 'CE' ? -1 : 1;
  if (a.open !== b.open) return a.open ? -1 : 1;
  return (a.strike ?? 0) - (b.strike ?? 0);
}

function TradeBook({ systems, states, liveLegs, basis }: {
  systems: SystemDef[];
  states: Record<string, SystemStateRecord>;
  liveLegs: Record<string, number>;
  basis: 'per2' | 'raw';
}) {
  const [mode, setMode] = useState<TBGroupMode>('system');
  const [liveOnly, setLiveOnly] = useState(false);
  const [venue, setVenue] = useState<'all' | 'nifty' | 'sensex'>('all');
  // ARMED FOR TODAY -- the day's plan straight from the rules matrix, visible
  // before anything enters (executors arm at 09:00/09:12; this needs neither).
  const [rulesRm, setRulesRm] = useState<any | null>(null);
  useEffect(() => {
    fetch('/api/nas/rules-matrix').then(r => r.json()).then(setRulesRm).catch(() => {});
  }, []);
  const armedToday = useMemo(() => {
    if (!rulesRm) return [] as { name: string; venue: string; win: string; stop: string; lots: number }[];
    const wd = new Date().getDay();
    const DTE: Record<string, number> = { NIFTY: [-1, 1, 0, 4, 3, 2, -1][wd], SENSEX: [-1, 3, 2, 1, 0, 4, -1][wd] };
    const out: { name: string; venue: string; win: string; stop: string; lots: number }[] = [];
    for (const e of (rulesRm.entry916 || [])) {
      if (e.live && (e.live_dtes || []).includes(DTE[e.venue]))
        out.push({ name: e.label, venue: e.venue, win: `${e.entry}*→${e.exit}*`, stop: e.stop, lots: e.lots });
    }
    for (const s of (rulesRm.sleeves || [])) {
      if (s.mode !== 'live') continue;
      const c = (s.perdte || {})[String(DTE[s.venue])];
      if (c) out.push({ name: s.label, venue: s.venue, win: `${String(c.win).replace('-', '*→')}*`,
                        stop: c.sl === 'none' ? 'no %-SL · 50% backstop' : `combined-SL ${c.sl}%`, lots: c.lots });
    }
    return out;
  }, [rulesRm]);
  const { spot } = useLiveTicks();   // live NIFTY -- distance to the move-stop band
  // ST-trail value for naked-survivor legs (sl_price sentinel 999999) — from the ticker.
  const [stTrail, setStTrail] = useState<Record<string, number>>({});
  useEffect(() => {
    let on = true;
    const load = () =>
      fetch(`/api/nas/ticker/status`, { cache: 'no-store' })
        .then((r) => r.json())
        .then((d) => {
          if (!on) return;
          const m: Record<string, number> = {};
          for (const k of ['atm_naked_st', 'atm4_naked_st']) {
            const nst = d?.[k];
            if (nst && nst.active && nst.tradingsymbol && typeof nst.st_value === 'number') m[nst.tradingsymbol] = nst.st_value;
          }
          setStTrail(m);
        })
        .catch(() => {});
    load();
    const id = setInterval(load, 5000);
    return () => { on = false; clearInterval(id); };
  }, []);
  const rowVenue = (r: TBRow): 'nifty' | 'sensex' =>
    (r.sysId.startsWith('sx-') || /SENSEX/i.test(r.sysLabel) ||
     (r.tradingsymbol || '').startsWith('SENSEX') || (r.strike ?? 0) >= 40000) ? 'sensex' : 'nifty';
  const rows = buildTradeBook(systems, states, liveLegs, basis)
    .filter((r) => !liveOnly || r.mode === 'live')
    .filter((r) => venue === 'all' || rowVenue(r) === venue);

  const dayPnl = rows.reduce((a, r) => a + r.pnl, 0);
  const realized = rows.filter((r) => !r.open).reduce((a, r) => a + r.pnl, 0);
  const openPnl = rows.filter((r) => r.open).reduce((a, r) => a + r.pnl, 0);

  const byLabel = (a: TBRow, b: TBRow) =>
    a.sysLabel === b.sysLabel ? tbSort(a, b) : a.sysLabel.localeCompare(b.sysLabel);

  const groups: { key: string; label: string; rows: TBRow[] }[] = [];
  if (mode === 'all') {
    groups.push({ key: 'all', label: 'All systems', rows: [...rows].sort(byLabel) });
  } else if (mode === 'family') {
    for (const fam of ['916', 'squeeze']) {
      const fr = rows.filter((r) => r.family === fam).sort(byLabel);
      if (fr.length) groups.push({ key: fam, label: fam === 'squeeze' ? 'Squeeze (ATR)' : '9:16 (timed)', rows: fr });
    }
  } else {
    for (const sys of systems) {
      const sr = rows.filter((r) => r.sysId === sys.id).sort(tbSort);
      if (sr.length) groups.push({ key: sys.id, label: sys.label, rows: sr });
    }
  }

  const stamp = new Date().toLocaleString('en-IN', {
    timeZone: 'Asia/Kolkata', day: '2-digit', month: 'short', year: 'numeric',
    hour: '2-digit', minute: '2-digit', hour12: false,
  });
  const col = (v: number) => (v > 0 ? '#3fb950' : v < 0 ? '#f85149' : 'var(--ink-muted)');
  const inr = (v: number) => (v >= 0 ? '+₹' : '−₹') + Math.abs(Math.round(v)).toLocaleString('en-IN');
  const gridCols = mode === 'system'
    ? '34px 58px 46px 50px 104px 124px 116px 48px 48px 86px'
    : '120px 34px 58px 46px 50px 104px 124px 116px 48px 48px 86px';

  return (
    <section className={styles.sectionBlock}>
      <div className={styles.sectionHead}>
        <div className="section-title">Trade Book · {stamp} IST</div>
        <div style={{ display: 'flex', gap: 6 }}>
          {(['system', 'family', 'all'] as TBGroupMode[]).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setMode(m)}
              style={{
                fontSize: 'var(--text-xs)', padding: '3px 10px', borderRadius: 6,
                border: '1px solid var(--line)', cursor: 'pointer',
                background: mode === m ? '#2f81f7' : 'transparent',
                color: mode === m ? '#fff' : 'var(--ink-muted)',
              }}
            >
              {m === 'system' ? 'By system' : m === 'family' ? 'By family' : 'All together'}
            </button>
          ))}
          <label
            style={{ display: 'inline-flex', alignItems: 'center', gap: 5, fontSize: 'var(--text-xs)', color: liveOnly ? '#ef4444' : 'var(--ink-muted)', cursor: 'pointer', marginLeft: 8, fontWeight: liveOnly ? 700 : 400 }}
            title="Show only real-money (LIVE) legs; hide the paper systems"
          >
            <input type="checkbox" checked={liveOnly} onChange={(e) => setLiveOnly(e.target.checked)} style={{ cursor: 'pointer', accentColor: '#ef4444' }} />
            LIVE only
          </label>
          <div style={{ display: 'inline-flex', gap: 2, marginLeft: 8 }}>
            {(['all', 'nifty', 'sensex'] as const).map((v) => (
              <button
                key={v}
                type="button"
                onClick={() => setVenue(v)}
                title={v === 'all' ? 'Both venues' : v === 'nifty' ? 'NIFTY legs only' : 'SENSEX legs only'}
                style={{
                  fontSize: 'var(--text-xs)', padding: '3px 9px', borderRadius: 6,
                  border: '1px solid var(--line)', cursor: 'pointer',
                  background: venue === v ? (v === 'sensex' ? '#0F6E56' : v === 'nifty' ? '#2f81f7' : 'var(--ink-muted)') : 'transparent',
                  color: venue === v ? '#fff' : 'var(--ink-muted)',
                  fontWeight: venue === v ? 700 : 400,
                }}
              >
                {v === 'all' ? 'Both' : v.toUpperCase()}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div style={{ display: 'flex', gap: 18, alignItems: 'baseline', margin: '4px 0 12px', flexWrap: 'wrap' }}>
        <span style={{ fontSize: 18, fontWeight: 700, color: col(dayPnl) }}>Day P&amp;L {inr(dayPnl)}</span>
        <span style={{ fontSize: 'var(--text-xs)', color: 'var(--ink-muted)' }}>
          realized {inr(realized)} · open {inr(openPnl)} · {rows.length} legs
        </span>
      </div>

      <div style={{ fontSize: 'var(--text-xs)', fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace', overflowX: 'auto', WebkitOverflowScrolling: 'touch' }}>
        <div style={{ minWidth: 'max-content' }}>
        {/* column header */}
        <div style={{
          display: 'grid', gridTemplateColumns: gridCols, gap: 8, padding: '2px 0',
          color: 'var(--ink-faint, #6e7681)', fontSize: 10, letterSpacing: '0.04em',
          borderBottom: '1px solid var(--line)',
        }}>
          {mode !== 'system' && <span>SYSTEM</span>}
          <span>C/P</span><span>STRIKE</span><span>QTY</span><span>MODE</span><span>ENTRY→EXIT</span>
          <span>ARM</span><span>STATUS</span><span>IN</span><span>OUT</span>
          <span style={{ textAlign: 'right' }}>P&amp;L</span>
        </div>
        {groups.map((g) => {
          const gp = g.rows.reduce((a, r) => a + r.pnl, 0);
          return (
            <div key={g.key} style={{ marginBottom: 10 }}>
              <div style={{
                display: 'flex', justifyContent: 'space-between',
                borderBottom: '1px solid var(--line)', padding: '5px 0', fontWeight: 600, marginTop: 6,
              }}>
                <span>{g.label}</span>
                <span style={{ color: col(gp) }}>group {inr(gp)}</span>
              </div>
              {g.rows.map((r, i) => (
                <div
                  key={`${r.sysId}-${r.side}-${r.strike}-${r.open ? 'o' : 'c'}-${i}`}
                  title={r.mode === 'live'
                    ? 'LIVE — real money. This leg went to the exchange.'
                    : 'Paper — simulated. No order went to the exchange.'}
                  style={{
                    display: 'grid', gridTemplateColumns: gridCols, gap: 8,
                    padding: '3px 0 3px 7px', alignItems: 'center', opacity: r.open ? 1 : 0.62,
                    // Whole-row tell for real money. inset shadow, not a border, so it can
                    // never shift the grid columns.
                    boxShadow: r.mode === 'live'
                      ? 'inset 2px 0 0 #ef4444'
                      : 'inset 2px 0 0 rgba(110,118,129,0.40)',
                    background: r.mode === 'live' ? 'rgba(239,68,68,0.045)' : 'transparent',
                  }}
                >
                  {mode !== 'system' && (
                    <span style={{ color: 'var(--ink-muted)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{r.sysLabel}</span>
                  )}
                  <span style={{ fontWeight: 700, color: r.side === 'CE' ? '#d29922' : r.side === 'PE' ? '#a371f7' : 'var(--ink-muted)' }}>{r.side}</span>
                  <span>{r.strike ?? '—'}</span>
                  <span
                    style={{ color: r.restated ? '#58a6ff' : 'var(--ink-muted)' }}
                    title={r.restated
                      ? 'Paper leg restated to the uniform 2 lots (130 qty). Switch the history basis to "as traded" to see the size it actually simulated.'
                      : undefined}
                  >×{r.qty}</span>
                  {(() => {
                    const live = r.mode === 'live';
                    return (
                      <span
                        style={{
                          color: live ? '#ef4444' : 'var(--ink-faint, #6e7681)',
                          fontWeight: live ? 700 : 400,
                          whiteSpace: 'nowrap',
                        }}
                        title={
                          live
                            ? 'LIVE — real money. This leg was sent to the exchange.'
                            : 'Paper — simulated. No order was sent to the exchange.'
                        }
                      >
                        {live ? 'LIVE' : 'paper'}
                      </span>
                    );
                  })()}
                  <span style={{ color: 'var(--ink-muted)' }}>
                    {r.entry != null ? r.entry.toFixed(1) : '—'} → {r.exit != null ? r.exit.toFixed(1) : '—'}
                  </span>
                  {(() => {
                    if (r.open && r.arm_text) {
                      return (
                        <span style={{ color: '#d29922', whiteSpace: 'nowrap' }}
                          title="Combined premium now · combined-SL exit level — the whole straddle exits when CE+PE reaches it">
                          {r.arm_text}
                        </span>
                      );
                    }
                    // Move-stop systems (ATM2 / OTM): the enforced arm is a NIFTY level, not a
                    // premium. Show the band that actually closes BOTH legs and re-centers.
                    if (r.open && r.armLo != null && r.armHi != null) {
                      const lo = Math.round(r.armLo);
                      const hi = Math.round(r.armHi);
                      const near = spot != null
                        ? Math.round(Math.min(Math.abs(hi - spot), Math.abs(spot - lo)))
                        : null;
                      const color = near == null ? '#d29922'
                        : near <= 15 ? '#ef4444'
                        : near <= 40 ? '#d29922'
                        : '#3fb950';
                      const title =
                        `MOVE-STOP (the only exit trigger on this system): closes BOTH legs and ` +
                        `re-centers when NIFTY leaves ${lo}-${hi} (+/-0.40% from entry spot ` +
                        `${Math.round(r.armSpot ?? 0)}).` +
                        (spot != null ? ` NIFTY ${Math.round(spot)} - ${near} pts from firing.` : '') +
                        ` The per-leg premium SL (${r.arm != null ? r.arm.toFixed(1) : 'n/a'}) is ` +
                        `deliberately disabled on this system and will NEVER fire.`;
                      return (
                        <span
                          style={{ color, whiteSpace: 'nowrap', fontWeight: near != null && near <= 15 ? 700 : 400 }}
                          title={title}
                        >
                          {lo}-{hi}{near != null ? ` (${near})` : ''}
                        </span>
                      );
                    }
                    const hasArm = r.open && r.arm != null && r.arm > 0;
                    if (!hasArm) {
                      return <span style={{ color: 'var(--ink-faint, #6e7681)', whiteSpace: 'nowrap' }} title="No active exit arm">—</span>;
                    }
                    if ((r.arm as number) >= 900000) {
                      const stv = r.tradingsymbol ? stTrail[r.tradingsymbol] : undefined;
                      return (
                        <span style={{ color: '#58a6ff', whiteSpace: 'nowrap' }} title="SuperTrend(7,3) trailing exit on the naked survivor leg (no fixed-price stop)">
                          ST {stv != null ? stv.toFixed(1) : '\u2026'}
                        </span>
                      );
                    }
                    const v = (r.arm as number).toFixed(1);
                    let color = '#d29922', icon = '●';
                    let title = `Armed at premium ${v} (live monitoring status unknown)`;
                    if (r.armLive === true) {
                      color = '#3fb950'; icon = '●';
                      title = `Live-armed — ticker is actively watching this leg; triggers at premium ${v}`;
                    } else if (r.armLive === false) {
                      color = '#ef4444'; icon = '⚠';
                      title = `WARNING: SL ${v} is set but the ticker is NOT subscribed to this leg — monitoring gap`;
                    }
                    return (
                      <span style={{ color, whiteSpace: 'nowrap', fontWeight: r.armLive === false ? 700 : 400 }} title={title}>
                        {icon} {v}
                      </span>
                    );
                  })()}
                  {(() => {
                    if (r.open) {
                      return (
                        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 3, color: '#3fb950', fontWeight: 700, fontSize: 10, background: 'rgba(63,185,80,0.15)', padding: '1px 7px', borderRadius: 9, whiteSpace: 'nowrap' }}>
                          &#9679; OPEN
                        </span>
                      );
                    }
                    const up = (r.reason || '').toUpperCase();
                    const c = up.includes('SL') ? '#f85149'
                      : up.includes('ST') ? '#58a6ff'
                      : (up.includes('ROLL') || up.includes('ADJ') || up.includes('SHIFT')) ? '#d29922'
                      : '#8b949e';
                    return (
                      <span style={{ display: 'inline-flex', alignItems: 'center', gap: 3, color: c, fontWeight: 600, fontSize: 10, background: `${c}22`, padding: '1px 7px', borderRadius: 9, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                        {shortReason(r.reason)}
                      </span>
                    );
                  })()}
                  <span style={{ color: 'var(--ink-muted)' }}>{r.inTime || '—'}</span>
                  <span style={{ color: 'var(--ink-muted)' }}>{r.outTime || '—'}</span>
                  <span style={{ textAlign: 'right', color: col(r.pnl), fontWeight: 600 }}>{inr(r.pnl)}</span>
                </div>
              ))}
            </div>
          );
        })}
        {armedToday.length > 0 && (
          <div style={{ margin: '6px 0 4px', border: '1px dashed #30363d', borderRadius: 8, padding: '6px 10px' }}>
            <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: 0.8, color: 'var(--ink-muted)', marginBottom: 4 }}>
              ARMED FOR TODAY · plan from the rules matrix · * = planned time
            </div>
            {armedToday.map((a, i) => (
              <div key={i} style={{ display: 'flex', gap: 12, fontSize: 11, padding: '2px 0', alignItems: 'center', flexWrap: 'wrap' }}>
                <span style={{ fontWeight: 600, minWidth: 120 }}>{a.name}</span>
                <span style={{ color: a.venue === 'SENSEX' ? '#d29922' : '#58a6ff', fontSize: 10, fontWeight: 700, minWidth: 48 }}>{a.venue}</span>
                <span style={{ fontFamily: 'monospace' }}>{a.win}</span>
                <span style={{ color: 'var(--ink-muted)', minWidth: 40 }}>{a.lots}L</span>
                <span style={{ color: 'var(--ink-muted)' }}>{a.stop}</span>
              </div>
            ))}
          </div>
        )}
        {rows.length === 0 && <div style={{ color: 'var(--ink-muted)', padding: 8 }}>No trades today yet.</div>}
        </div>
      </div>
    </section>
  );
}

/* ---------- panel per system ---------- */

interface PanelProps {
  def: SystemDef;
  onStateChange: (rec: SystemStateRecord) => void;
  onToast: (msg: string) => void;
  series: MtmPoint[];
  events: MtmEvent[];
  onExpand: () => void;
}

const EVENT_COLOR: Record<string, string> = {
  entry:  '#22c55e',  // green — open
  adjust: '#f59e0b',  // amber — roll / adj
  sl_hit: '#ef4444',  // red   — SL hit
  exit:   '#94a3b8',  // grey  — time / EOD exit
};

function fmtPnl(v: number): string {
  return (v >= 0 ? '+₹' : '-₹') + Math.abs(Math.round(v)).toLocaleString('en-IN');
}

function shortSym(s: string | undefined | null): string {
  // "NIFTY26MAY23400CE+NIFTY26MAY23400PE" -> "23400CE+23400PE"
  // Single symbol gets its trailing strike+leg fragment.
  if (!s) return '';
  return s.split('+')
          .filter(Boolean)
          .map((p) => p.slice(-7))
          .join('+');
}

interface ChartSeries { label: string; color: string; points: MtmPoint[]; }

// Sum several per-system P&L series into one, forward-filling each system's
// last value across the union of timestamps (systems tick at slightly
// different times). Used to build the 9:16 and Squeeze sub-curves.
function sumSeries(seriesList: MtmPoint[][]): MtmPoint[] {
  const lists = seriesList.filter((s) => s && s.length);
  if (!lists.length) return [];
  const allTs = Array.from(new Set(lists.flatMap((s) => s.map((p) => p[0])))).sort();
  const idx = lists.map(() => 0);
  const lastVal = lists.map(() => 0);
  const out: MtmPoint[] = [];
  for (const ts of allTs) {
    const tm = new Date(ts).getTime();
    let sum = 0;
    lists.forEach((s, si) => {
      while (idx[si] < s.length && new Date(s[idx[si]][0]).getTime() <= tm) {
        lastVal[si] = s[idx[si]][1];
        idx[si] += 1;
      }
      sum += lastVal[si];
    });
    out.push([ts, sum]);
  }
  return out;
}

interface PnlChartProps {
  points: MtmPoint[];
  events: MtmEvent[];
  expanded?: boolean;
  extraSeries?: ChartSeries[];
  hiddenSeries?: Set<string>;   // labels to hide ('Overall' | extra labels)
}

function PnlChart({ points, events, expanded = false, extraSeries, hiddenSeries }: PnlChartProps) {
  // SVG day-P&L curve. Color graded by intensity:
  //   above 0 → green (deeper as profit grows)
  //   below 0 → red   (deeper as loss grows)
  // Event markers (entry / adjust / sl_hit / exit) drawn as dotted verticals.
  const W = expanded ? 920 : 320;
  const H = expanded ? 340 : 56;
  const PAD_X = expanded ? 56 : 4;
  const PAD_Y = expanded ? 28 : 4;
  const ys = points.map((p) => p[1]);
  const showOverall = !hiddenSeries?.has('Overall');
  const visibleExtra = (extraSeries ?? []).filter((s) => !hiddenSeries?.has(s.label));
  // Scale to whatever's visible so toggling a curve off rescales the y-axis.
  const scaleYs = [
    ...(showOverall ? ys : []),
    ...visibleExtra.flatMap((s) => s.points.map((p) => p[1])),
  ];
  const baseYs = scaleYs.length ? scaleYs : ys;
  const yMinRaw = Math.min(0, ...baseYs);
  const yMaxRaw = Math.max(0, ...baseYs);
  // pad y a bit in expanded so events at edges don't get clipped
  const yPad = expanded ? Math.max(50, (yMaxRaw - yMinRaw) * 0.08) : 0;
  const yMin = yMinRaw - yPad;
  const yMax = yMaxRaw + yPad;
  const ySpan = yMax - yMin || 1;
  const tMin = new Date(points[0][0]).getTime();
  const tMax = new Date(points[points.length - 1][0]).getTime();
  const tSpan = tMax - tMin || 1;
  const xOf = (ts: string) => {
    const m = new Date(ts).getTime();
    const x = PAD_X + ((m - tMin) / tSpan) * (W - 2 * PAD_X);
    return Math.max(PAD_X, Math.min(W - PAD_X, x));
  };
  const yOf = (v: number) =>
    H - PAD_Y - ((v - yMin) / ySpan) * (H - 2 * PAD_Y);
  const zeroY = yOf(0);
  const d = points
    .map((p, i) => `${i ? 'L' : 'M'} ${xOf(p[0])} ${yOf(p[1])}`)
    .join(' ');
  const firstX = xOf(points[0][0]);
  const lastX = xOf(points[points.length - 1][0]);
  const area = `${d} L ${lastX} ${zeroY} L ${firstX} ${zeroY} Z`;
  const last = ys[ys.length - 1];
  // intensity (0.25..1) scales gradient stop opacity by how deep min/max reach
  const denom = Math.max(1, Math.abs(yMaxRaw) + Math.abs(yMinRaw));
  const gIntensity = Math.min(1, Math.max(0.25, Math.abs(yMaxRaw) / denom + 0.25));
  const rIntensity = Math.min(1, Math.max(0.25, Math.abs(yMinRaw) / denom + 0.25));
  const zeroFrac = ((zeroY - PAD_Y) / (H - 2 * PAD_Y)) * 100;
  const gid = `pnlg${Math.random().toString(36).slice(2, 8)}`;

  // Interpolate event y onto the curve once, for the overlay markers.
  function interpY(ts: string): number {
    const t = new Date(ts).getTime();
    if (t <= tMin) return ys[0];
    if (t >= tMax) return ys[ys.length - 1];
    for (let j = 0; j < points.length - 1; j++) {
      const t0 = new Date(points[j][0]).getTime();
      const t1 = new Date(points[j + 1][0]).getTime();
      if (t >= t0 && t <= t1) {
        const f = (t - t0) / Math.max(1, t1 - t0);
        return ys[j] + f * (ys[j + 1] - ys[j]);
      }
    }
    return ys[ys.length - 1];
  }

  return (
    <div className={styles.chartWrap}>
    <svg
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="none"
      className={expanded ? styles.chartSvg : styles.sparkSvg}
    >
      <defs>
        <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#22c55e"
                stopOpacity={(0.55 * gIntensity).toFixed(3)} />
          <stop offset={`${Math.max(0, Math.min(100, zeroFrac)).toFixed(2)}%`}
                stopColor="#a3a3a3" stopOpacity="0.04" />
          <stop offset="100%" stopColor="#ef4444"
                stopOpacity={(0.55 * rIntensity).toFixed(3)} />
        </linearGradient>
      </defs>
      {expanded ? (() => {
        // Y-axis "nice" ticks: pick a step from [1,2,2.5,5]×10^k that lands
        // ~6 ticks across the range. Draw a dashed gridline + ₹-labelled
        // tick at each, skipping the zero line (rendered separately below).
        const range = yMax - yMin;
        if (range <= 0) return null;
        const targetTicks = 6;
        const rawStep = range / targetTicks;
        const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
        const norm = rawStep / mag;
        const step = norm <= 1.5 ? mag
                   : norm <= 3   ? 2 * mag
                   : norm <= 4   ? 2.5 * mag
                   : norm <= 7   ? 5 * mag
                   : 10 * mag;
        const ticks: number[] = [];
        const first = Math.ceil(yMin / step) * step;
        for (let v = first; v <= yMax + step * 0.001; v += step) {
          if (Math.abs(v) < step * 0.01) continue;  // skip near-zero
          ticks.push(v);
        }
        return (
          <g>
            {ticks.map((v) => {
              const yy = yOf(v);
              return (
                <g key={`yt-${v}`}>
                  <line x1={PAD_X} x2={W - PAD_X}
                        y1={yy} y2={yy}
                        stroke="#2a2a2a" strokeWidth="0.5"
                        strokeDasharray="2 4" />
                  <text x={PAD_X - 6} y={yy + 3.5}
                        fontSize="9.5" fill="#94a3b8"
                        textAnchor="end">
                    {fmtPnl(v)}
                  </text>
                </g>
              );
            })}
          </g>
        );
      })() : null}
      <line x1={PAD_X} x2={W - PAD_X} y1={zeroY} y2={zeroY}
            stroke="#3a3a3a" strokeDasharray="2 3"
            strokeWidth={expanded ? 1 : 0.7} />
      {expanded ? (
        <text x={PAD_X - 6} y={zeroY + 3.5}
              fontSize="9.5" fill="#94a3b8" textAnchor="end">
          ₹0
        </text>
      ) : null}
      {showOverall ? (
        <>
          <path d={area} fill={`url(#${gid})`} stroke="none" />
          <path d={d} fill="none"
                stroke={last >= 0 ? '#16a34a' : '#dc2626'}
                strokeWidth={expanded ? 2 : 1.4}
                strokeLinejoin="round" strokeLinecap="round" />
        </>
      ) : null}
      {/* 9:16 / Squeeze sub-curves (only the visible ones) */}
      {visibleExtra.map((s) => {
        if (s.points.length < 2) return null;
        const sd = s.points
          .map((p, i) => `${i ? 'L' : 'M'} ${xOf(p[0])} ${yOf(p[1])}`)
          .join(' ');
        return (
          <path key={s.label} d={sd} fill="none" stroke={s.color}
                strokeWidth={expanded ? 1.5 : 1} strokeOpacity={0.95}
                strokeLinejoin="round" strokeLinecap="round" />
        );
      })}
      {expanded ? (
        <>
          <text x={W - PAD_X} y={PAD_Y - 9} fontSize="11"
                fill={last >= 0 ? '#22c55e' : '#ef4444'}
                textAnchor="end" fontWeight="700">
            now {fmtPnl(last)}
          </text>
          {(() => {
            // 5-min time ticks along the x-axis (expanded mode only).
            const ticks: Array<{ x: number; lab: string }> = [];
            const start = new Date(tMin);
            start.setSeconds(0, 0);
            // round up to next 15-min boundary (15-min marks, 30-min labels)
            const m = start.getMinutes();
            start.setMinutes(m + ((15 - (m % 15)) % 15));
            for (let t = start.getTime(); t <= tMax + 1; t += 15 * 60 * 1000) {
              const dt = new Date(t);
              const x =
                PAD_X + ((t - tMin) / tSpan) * (W - 2 * PAD_X);
              if (x < PAD_X - 2 || x > W - PAD_X + 2) continue;
              const lab = `${String(dt.getHours()).padStart(2, '0')}:${String(
                dt.getMinutes()
              ).padStart(2, '0')}`;
              ticks.push({ x, lab });
            }
            // label every other 15-min tick (= every 30 min) when dense
            const step = ticks.length > 14 ? 2 : 1;
            return (
              <g>
                {ticks.map((t, i) => (
                  <g key={t.x}>
                    <line
                      x1={t.x} x2={t.x}
                      y1={H - PAD_Y} y2={H - PAD_Y + 4}
                      stroke="#525252" strokeWidth="0.7" />
                    {i % step === 0 ? (
                      <text x={t.x} y={H - PAD_Y + 16}
                            fontSize="9.5" fill="#94a3b8"
                            textAnchor="middle">
                        {t.lab}
                      </text>
                    ) : null}
                  </g>
                ))}
              </g>
            );
          })()}
        </>
      ) : null}
    </svg>
    {(() => {
      // Cluster events into small time-buckets so overlapping markers collapse
      // into ONE dot; hovering lists every event in that bucket. Dots are event
      // markers riding the Overall curve (not points of the 9:16/Squeeze lines).
      const BUCKET = 1.6; // ~5-6 min at full-day width
      const buckets = new Map<number, { xPct: number; yPct: number; evs: MtmEvent[] }>();
      events.forEach((e) => {
        const tMs = new Date(e.ts).getTime();
        if (tMs < tMin - 1 || tMs > tMax + 1) return;
        const xPct = (xOf(e.ts) / W) * 100;
        const key = Math.round(xPct / BUCKET);
        const ex = buckets.get(key);
        if (ex) ex.evs.push(e);
        else buckets.set(key, { xPct, yPct: (yOf(interpY(e.ts)) / H) * 100, evs: [e] });
      });
      const size = expanded ? 8 : 6;
      return Array.from(buckets.values()).map((b, i) => {
        const types = new Set(b.evs.map((e) => e.type));
        const color = types.size === 1 ? (EVENT_COLOR[b.evs[0].type] || '#888') : '#9ca3af';
        const multi = b.evs.length > 1;
        const tip = b.evs
          .map((e) => `${e.ts.slice(11, 16)}  ${e.label}${e.sym ? ' ' + e.sym : ''}`)
          .join('\n');
        return (
          <span
            key={`mk-${i}`}
            className={styles.markerDot}
            style={{
              left: `${b.xPct}%`, top: `${b.yPct}%`,
              width: size, height: size, background: color,
              boxShadow: multi ? '0 0 0 1.5px rgba(255,255,255,0.6)' : undefined,
            }}
            title={multi ? `${b.evs.length} events\n${tip}` : tip}
          />
        );
      });
    })()}
    </div>
  );
}

interface SparkProps {
  points: MtmPoint[];
  events: MtmEvent[];
  onExpand: () => void;
}

function Sparkline({ points, events, onExpand }: SparkProps) {
  if (!points || points.length < 2) {
    return (
      <div className={styles.sparkEmpty}>
        Live P&amp;L curve appears once snapshots flow (09:15+).
      </div>
    );
  }
  const last = points[points.length - 1][1];
  const ys = points.map((p) => p[1]);
  const yMin = Math.min(0, ...ys);
  const yMax = Math.max(0, ...ys);
  return (
    <button type="button" onClick={onExpand} className={styles.sparkButton}
            title="Expand chart">
      <div className={styles.sparkBox}>
        <PnlChart points={points} events={events} />
        <div className={styles.sparkMeta}>
          <span className={last >= 0 ? styles.sparkPos : styles.sparkNeg}>
            now {fmtPnl(last)}
          </span>
          <span className={styles.sparkRange}>
            lo {fmtPnl(yMin)} · hi {fmtPnl(yMax)} · {points.length} pts
            {events.length ? ` · ${events.length} ev` : ''} ⤢
          </span>
        </div>
      </div>
    </button>
  );
}

interface ChartModalProps {
  open: boolean;
  title: string;
  points: MtmPoint[];
  events: MtmEvent[];
  extraSeries?: ChartSeries[];
  onClose: () => void;
}

function ChartModal({ open, title, points, events, extraSeries, onClose }: ChartModalProps) {
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const toggleSeries = (label: string) =>
    setHidden((prev) => {
      const next = new Set(prev);
      if (next.has(label)) next.delete(label); else next.add(label);
      return next;
    });
  useEffect(() => {
    if (!open) return;
    const k = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', k);
    return () => window.removeEventListener('keydown', k);
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div className={styles.modalBackdrop} onClick={onClose}>
      <div className={styles.modalCard} onClick={(e) => e.stopPropagation()}>
        <div className={styles.modalHead}>
          <div className={styles.modalTitle}>{title}</div>
          <button type="button" onClick={onClose}
                  className={styles.modalClose} aria-label="Close">×</button>
        </div>
        {points.length >= 2 ? (
          <PnlChart points={points} events={events} expanded extraSeries={extraSeries} hiddenSeries={hidden} />
        ) : (
          <div className={styles.sparkEmpty} style={{ margin: 24 }}>
            No snapshots yet — comes alive after 09:15.
          </div>
        )}
        {extraSeries && extraSeries.length ? (
          <div className={styles.eventLegend}>
            <span style={{ color: 'var(--ink-faint, #6e7681)', fontSize: 11, marginRight: 2 }}>show:</span>
            {[{ label: 'Overall', color: '#16a34a' }, ...extraSeries].map((s) => {
              const off = hidden.has(s.label);
              return (
                <button
                  key={s.label}
                  type="button"
                  onClick={() => toggleSeries(s.label)}
                  className={styles.legendItem}
                  title={off ? `Show ${s.label}` : `Hide ${s.label}`}
                  style={{
                    cursor: 'pointer', background: 'none', border: 'none',
                    font: 'inherit', color: 'inherit', padding: 0,
                    opacity: off ? 0.4 : 1,
                    textDecoration: off ? 'line-through' : 'none',
                  }}
                >
                  <span className={styles.legendDot} style={{ background: s.color }} />{s.label}
                </button>
              );
            })}
          </div>
        ) : null}
        {events.length ? (
          <div className={styles.eventLegend}>
            {Object.entries({
              entry: 'Entry', adjust: 'Adjust', sl_hit: 'SL hit', exit: 'Exit',
            }).map(([k, lab]) => (
              <span key={k} className={styles.legendItem}>
                <span className={styles.legendDot}
                      style={{ background: EVENT_COLOR[k] }} />
                {lab}
              </span>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  );
}

function SystemPanel({ def, onStateChange, onToast, series, events, onExpand }: PanelProps) {
  const [state, setState] = useState<NASState | null>(null);
  const [err, setErr] = useState<string | null>(null);
  // Live tick prices from the parent SSE stream — keyed by tradingsymbol.
  const liveTicks = useLiveTicks();
  const ticks = liveTicks.legs;
  const streamAlive = liveTicks.connected;

  const stateUrl = `/api/${def.key}/state`;

  // Poll state every 5s as a safety net — positions, config, stats. Live
  // leg LTPs come from the shared SSE stream in the parent component.
  useEffect(() => {
    let cancelled = false;
    const load = () => {
      apiGet<NASState>(stateUrl)
        .then((s) => {
          if (cancelled) return;
          setState(s);
          setErr(null);
          onStateChange({ state: s, err: null });
        })
        .catch((e) => {
          if (cancelled) return;
          const msg = e instanceof Error ? e.message : 'Load failed';
          setErr(msg);
          onStateChange({ state: null, err: msg });
        });
    };
    load();
    const id = setInterval(load, 5_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stateUrl]);

  const positions: NASPosition[] = [
    ...(state?.positions?.ce ?? []),
    ...(state?.positions?.pe ?? []),
  ];

  const enriched = positions.map((p) => {
    const live = p.tradingsymbol ? ticks[p.tradingsymbol] : undefined;
    const ltp = live ?? p.ltp;
    const entry = p.entry_price ?? p.entry_premium;
    let pnl = p.pnl_inr;
    if (live !== undefined && entry !== undefined && p.qty) {
      // NAS systems short options → profit when LTP drops below entry
      pnl = Math.round((entry - live) * p.qty * 100) / 100;
    }
    return { ...p, ltp, pnl_inr: pnl };
  });

  const totalPnl = enriched.reduce((acc, p) => acc + (p.pnl_inr ?? 0), 0);
  // Day P&L = closed trades today (persisted) + live open-leg P&L.
  // Old code used `??` which preferred 0 (no closed trades) over the open-leg
  // MTM, so system-level Day P&L stuck at Rs 0 even when legs were profitable.
  const persistedTodayPnl = (state?.stats?.today_pnl as number | undefined) ?? 0;
  const dayPnl = persistedTodayPnl + totalPnl;

  // Closed-today legs (sorted newest first), with pnl computed from entry/exit.
  const closedLegs = (state?.positions?.closed_today ?? [])
    .map((p) => {
      const entry = p.entry_price ?? p.entry_premium;
      const exit = p.exit_price;
      const qty = p.qty ?? 0;
      const pnl =
        entry != null && exit != null && qty
          ? Math.round((entry - exit) * qty * 100) / 100
          : undefined;
      return { ...p, pnl_inr: pnl };
    })
    .sort((a, b) => (b.exit_time ?? '').localeCompare(a.exit_time ?? ''));
  const reentries = (state?.stats?.total_reentries as number | undefined) ?? 0;
  const winRate = state?.stats?.win_rate as number | undefined;
  const pf = state?.stats?.profit_factor as number | undefined;
  const slHits = state?.stats?.sl_hits_today as number | undefined;

  const enabled = !!state?.config?.enabled;
  const paper = !!state?.config?.paper_trading_mode;

  // action buttons removed — header is title/subtitle only, consistent across pages

  return (
    <div className={styles.panel}>
      <div className={styles.panelHead}>
        <div className={styles.panelHeadLeft}>
          <div className={styles.panelTitle}>{def.label}</div>
          <div className={styles.panelSub}>{def.subtitle}</div>
        </div>
        <div className={styles.panelStatus}>
          <StatusDot
            kind={!enabled ? 'off' : !streamAlive ? 'warn' : paper ? 'paper' : 'live'}
            title={!enabled ? 'Disabled' : !streamAlive ? 'Stream down - check the system'
              : paper ? 'Paper - simulated' : 'Live - real money'}
          />
          <div className={styles.panelStatusMeta}>
            {formatInt(state?.positions?.total_active ?? 0)} active · {formatInt(reentries)} re-entry
          </div>
        </div>
      </div>

      <div className={styles.metricsRow}>
        <MiniMetric
          label="Day P&L"
          value={
            <span className={pnlClass(dayPnl)}>{formatPnl(dayPnl)}</span>
          }
        />
        <MiniMetric label="SL hits today" value={formatInt(slHits ?? 0)} />
      </div>

      <Sparkline points={series} events={events} onExpand={onExpand} />

      <div className={styles.legs}>
        {enriched.length === 0 ? (
          <div className={styles.noLegs}>No open legs</div>
        ) : (
          enriched.map((p, i) => <LegRow key={(p.tradingsymbol ?? '') + i} leg={p} high={liveTicks.highs?.[p.tradingsymbol ?? '']} />)
        )}
      </div>

      {closedLegs.length > 0 ? (
        <div className={styles.closedLegs}>
          <div className={styles.closedHead}>
            Closed today · {closedLegs.length}
          </div>
          {closedLegs.map((p, i) => (
            <LegRow
              key={'c' + (p.tradingsymbol ?? '') + i}
              leg={p}
              closed
              reason={p.exit_reason}
            />
          ))}
        </div>
      ) : null}

      <details className={styles.rules}>
        <summary className={styles.rulesSummary}>Rules &amp; snapshot</summary>
        <div className={styles.rulesBody}>
          <div className={styles.snapshotRow}>
            <div className={styles.snapshotItem}>
              <span className={styles.snapshotLabel}>Win rate (all-time)</span>
              <span className={styles.snapshotValue}>
                {winRate !== undefined ? formatPct(winRate, 1) : '—'}
              </span>
            </div>
            <div className={styles.snapshotItem}>
              <span className={styles.snapshotLabel}>Profit factor</span>
              <span className={styles.snapshotValue}>
                {pf !== undefined ? formatNumber(pf, 2) : '—'}
              </span>
            </div>
          </div>
          <div className={styles.rulesText}>{def.rules}</div>
        </div>
      </details>

      {err ? <div className={styles.errRow}>{err}</div> : null}
    </div>
  );
}

/* ---------- Master mode toggle (OFF / PAPER / LIVE for all 8 systems) ---------- */

interface MasterModeState {
  mode: 'off' | 'paper' | 'live' | 'mixed' | null;
  busy: boolean;
}

function MasterModeToggle({ onToast }: { onToast: (msg: string) => void }) {
  const [state, setState] = useState<MasterModeState>({ mode: null, busy: false });

  const refresh = async () => {
    try {
      const r = await fetch('/api/nas/master-mode');
      const d = await r.json();
      setState((p) => ({ ...p, mode: d.mode ?? null }));
    } catch {
      /* ignore */
    }
  };

  useEffect(() => {
    void refresh();
    const id = setInterval(refresh, 10_000);
    return () => clearInterval(id);
  }, []);

  const setMode = async (target: 'off' | 'paper' | 'live') => {
    if (state.busy || state.mode === target) return;
    if (target === 'live') {
      const ok = window.confirm(
        'Switch ALL 8 NAS systems to LIVE trading? Real money will be at risk on the next entry signal.',
      );
      if (!ok) return;
    }
    setState((p) => ({ ...p, busy: true }));
    try {
      const r = await fetch('/api/nas/master-mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: target }),
      });
      const d = await r.json();
      if (!r.ok) {
        onToast(`Master toggle failed: ${d.error || r.statusText}`);
      } else {
        setState({ mode: d.mode ?? target, busy: false });
        onToast(`All NAS systems → ${target.toUpperCase()}`);
        return;
      }
    } catch (e) {
      onToast(`Master toggle error: ${e instanceof Error ? e.message : 'unknown'}`);
    }
    setState((p) => ({ ...p, busy: false }));
  };

  const buttons: { id: 'off' | 'paper' | 'live'; label: string; cls: string }[] = [
    { id: 'off',   label: 'OFF',   cls: styles.masterBtnOff },
    { id: 'paper', label: 'PAPER', cls: styles.masterBtnPaper },
    { id: 'live',  label: 'LIVE',  cls: styles.masterBtnLive },
  ];

  return (
    <div className={styles.masterToggle}>
      <div className={styles.masterToggleLabel}>All NAS systems</div>
      <div className={styles.masterToggleGroup}>
        {buttons.map((b) => {
          const active = state.mode === b.id;
          return (
            <button
              key={b.id}
              type="button"
              className={`${styles.masterBtn} ${b.cls} ${active ? styles.masterBtnActive : ''}`}
              disabled={state.busy}
              onClick={() => void setMode(b.id)}
            >
              {b.label}
            </button>
          );
        })}
      </div>
      {state.mode === 'mixed' ? (
        <div className={styles.masterMixed}>mixed — variants in different states</div>
      ) : null}
    </div>
  );
}

function MiniMetric({
  label,
  value,
}: {
  label: string;
  value: React.ReactNode;
}) {
  return (
    <div className={styles.mini}>
      <div className={styles.miniLabel}>{label}</div>
      <div className={styles.miniValue}>{value}</div>
    </div>
  );
}

function LegRow({
  leg,
  closed = false,
  reason,
  high,
}: {
  leg: NASPosition;
  closed?: boolean;
  reason?: string;
  high?: number;
}) {
  // Show just the strike (e.g. "24300") — the CE/PE side is already shown as
  // the badge to the left, so the full contract code is noise. Fall back to
  // the shortened symbol only if the numeric strike is unavailable.
  const tsym = leg.strike != null
    ? String(Math.round(leg.strike))
    : shortOptionSymbol(leg.tradingsymbol);
  const entry = leg.entry_price ?? leg.entry_premium;
  const ltp = leg.ltp ?? leg.exit_price;
  const pnl = leg.pnl_inr;
  const qty = leg.qty;
  const entryTime = formatLegTime(leg.entry_time);
  const exitTime = closed ? formatLegTime(leg.exit_time) : undefined;

  return (
    <div className={`${styles.leg} ${closed ? styles.legClosed : ''}`}>
      <div className={styles.legMain}>
        <span className={styles.legSide}>{leg.leg}</span>
        <span className={styles.legSym}>{tsym}</span>
        {qty ? <span className={styles.legQty}>×{qty}</span> : null}
        {closed && reason ? (
          <span className={styles.legReason}>{reason}</span>
        ) : null}
      </div>
      <div className={styles.legNums}>
        <span className={styles.legSmall}>
          {entry !== undefined ? formatNumber(entry) : '—'}
          {entryTime ? <span className={styles.legTime}> @{entryTime}</span> : null}
        </span>
        <span className={styles.legArrow}>→</span>
        <span className={styles.legSmall}>
          {ltp !== undefined ? formatNumber(ltp) : '—'}
          {exitTime ? <span className={styles.legTime}> @{exitTime}</span> : null}
        </span>
        {!closed && high !== undefined ? (
          <span className={styles.legSmall} style={{ opacity: 0.65 }}>max {formatNumber(high)}</span>
        ) : null}
        <span className={pnlClass(pnl)} style={{ fontSize: 'var(--text-xs)' }}>
          {formatPnl(pnl)}
        </span>
      </div>
    </div>
  );
}

/** Shorten an option tradingsymbol like 'NIFTY2650523800PE' to '23800PE'.
 *  Drops the underlying + expiry prefix so the strike + CE/PE suffix is
 *  always visible even on narrow leg rows that would otherwise truncate. */
function shortOptionSymbol(tsym?: string | null): string {
  if (!tsym) return '—';
  const m = /(\d+)(CE|PE)$/.exec(tsym);
  return m ? `${m[1]}${m[2]}` : tsym;
}

function formatLegTime(iso?: string | null): string | null {
  if (!iso) return null;
  // Extract HH:MM from either ISO ("2026-04-22T11:51:42") or "11:51" fallback.
  const m = /T(\d{2}:\d{2})/.exec(iso) || /^(\d{2}:\d{2})/.exec(iso);
  return m ? m[1] : null;
}

/* ---------- next events helper ---------- */

interface NextEvent {
  system: string;
  event: string;
  scheduled: string;
  status: string;
  tone: 'pos' | 'neg' | 'neutral';
  relative: string;
  sortKey: number;
}

function minutesFromNowIST(hhmm: string): number {
  // IST current time
  const now = new Date();
  const parts = new Intl.DateTimeFormat('en-IN', {
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
    timeZone: 'Asia/Kolkata',
  })
    .format(now)
    .split(':');
  const nowMin = parseInt(parts[0], 10) * 60 + parseInt(parts[1], 10);
  const [h, m] = hhmm.split(':').map((x) => parseInt(x, 10));
  const targetMin = h * 60 + m;
  return targetMin - nowMin;
}

function relativeLabel(diffMin: number): string {
  if (diffMin < 0) return 'passed';
  if (diffMin === 0) return 'now';
  const h = Math.floor(diffMin / 60);
  const m = diffMin % 60;
  if (h === 0) return `in ${m}m`;
  return `in ${h}h ${m}m`;
}

function buildNextEvents(
  states: Record<string, SystemStateRecord>,
): NextEvent[] {
  const events: NextEvent[] = [];

  for (const def of ALL_SYSTEMS) {
    const rec = states[def.id];
    const enabled = !!rec?.state?.config?.enabled;
    const cfg = rec?.state?.config ?? {};

    // Entry event — for 9:16 systems only
    if (def.group === '916') {
      const diff = minutesFromNowIST('09:16');
      events.push({
        system: def.label,
        event: 'Auto-entry at 9:16',
        scheduled: '09:16',
        status: diff < 0 ? 'Done' : enabled ? 'Pending' : 'Disabled',
        tone: diff < 0 ? 'neutral' : enabled ? 'pos' : 'neutral',
        relative: relativeLabel(diff),
        sortKey: diff < 0 ? 9999 : diff,
      });
    } else {
      // Squeeze entry — continuous during entry window
      const startHHMM =
        (cfg.entry_start_time as string | undefined) ?? '09:30';
      const endHHMM =
        (cfg.entry_end_time as string | undefined) ?? '14:30';
      const startDiff = minutesFromNowIST(startHHMM);
      const endDiff = minutesFromNowIST(endHHMM);
      const active = startDiff <= 0 && endDiff > 0;
      events.push({
        system: def.label,
        event: 'Re-enter on squeeze',
        scheduled: `${startHHMM}-${endHHMM}`,
        status: !enabled ? 'Disabled' : active ? 'Active' : endDiff <= 0 ? 'Done' : 'Pending',
        tone: !enabled ? 'neutral' : active ? 'pos' : 'neutral',
        relative: active ? 'active' : startDiff > 0 ? relativeLabel(startDiff) : 'passed',
        sortKey: active ? -1 : startDiff > 0 ? startDiff : 9999,
      });
    }

    // Time exit (14:45) for OTM-flavoured systems
    if (def.id === 'nas' || def.id === 'nas-916-otm') {
      const t = (cfg.time_exit as string | undefined) ?? '14:45';
      const diff = minutesFromNowIST(t);
      events.push({
        system: def.label,
        event: 'Time exit',
        scheduled: t,
        status: diff < 0 ? 'Done' : 'Pending',
        tone: diff < 0 ? 'neutral' : 'pos',
        relative: relativeLabel(diff),
        sortKey: diff < 0 ? 9999 : diff,
      });
    }

    // EOD squareoff
    const eod = (cfg.eod_squareoff_time as string | undefined) ?? '15:15';
    const diff = minutesFromNowIST(eod);
    events.push({
      system: def.label,
      event: 'EOD squareoff',
      scheduled: eod,
      status: diff < 0 ? 'Done' : 'Pending',
      tone: diff < 0 ? 'neutral' : 'pos',
      relative: relativeLabel(diff),
      sortKey: diff < 0 ? 9999 : diff,
    });
  }

  // Daily summary at 15:20
  const diff = minutesFromNowIST('15:20');
  events.push({
    system: 'All systems',
    event: 'Daily summary',
    scheduled: '15:20',
    status: diff < 0 ? 'Done' : 'Pending',
    tone: 'neutral',
    relative: relativeLabel(diff),
    sortKey: diff < 0 ? 9999 : diff,
  });

  // Sort by nearest event first, past events at end
  events.sort((a, b) => a.sortKey - b.sortKey);
  return events;
}
