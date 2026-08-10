import { useEffect, useState } from 'react';
import { apiGet } from '../api/client';
import styles from './OholPaper.module.css';

type Position = { symbol: string; side: string; lot_size: number; entry: number; since: string };
type Setup = { symbol: string; side: string; trigger: number; status: string };
type NavPt = { d: string; nav: number; n_pos: number };
type Fill = { ts: string; symbol: string; side: string; price: number; lot_size: number; reason: string; pnl: number | null };
type State = {
  mode: string; system: string; capital: number; inception: string | null;
  killed: boolean; last_scan: string | null; last_cycle: string | null;
  realized: number; n_positions: number; positions: Position[];
  today_setups: Setup[]; navcurve: NavPt[]; recent_fills: Fill[];
};

const inr = (n: number) => '₹' + Math.round(n).toLocaleString('en-IN');
const lakh = (n: number) => '₹' + (n / 100000).toFixed(2) + 'L';

const RULES: [string, string][] = [
  ['Scan (09:21)', 'Liquid F&O list (77 names): first 5-min candle with OPEN = LOW → long candidate; OPEN = HIGH → short candidate. Day gap must be within ±0.5%; first candle range ≤ 0.5% (no long candles).'],
  ['Entry', 'Break of the first candle\'s high (long) / low (short), checked every 5 minutes → 1 LOT futures at market (paper fill on stock price × lot size).'],
  ['Exit', 'SuperTrend(7,2) on 5-min as trailing stop — exit when a COMPLETED 5-min candle closes through it. No target, no time stop; positions may carry overnight.'],
  ['Limits', 'Max 10 concurrent positions. 3bp/side paper costs.'],
  ['Status', 'Deployed per user spec on 2026-08-09 WITHOUT a prior backtest — this paper record is the system\'s first and only test. Judge it on the NAV curve below.'],
];

function Curve({ data }: { data: NavPt[] }) {
  if (data.length < 2) return <div className={styles.chartEmpty}>NAV curve builds daily ({data.length} point{data.length === 1 ? '' : 's'} so far).</div>;
  const W = 760, H = 220, P = 8;
  const navs = data.map((d) => d.nav);
  const lo = Math.min(...navs), hi = Math.max(...navs);
  const x = (i: number) => P + (i / (data.length - 1)) * (W - 2 * P);
  const y = (v: number) => P + (1 - (v - lo) / (hi - lo || 1)) * (H - 2 * P);
  const path = navs.map((v, i) => `${i === 0 ? 'M' : 'L'}${x(i)},${y(v)}`).join(' ');
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className={styles.chart} preserveAspectRatio="none">
      <path d={path} fill="none" stroke="var(--accent-pos, #1f9d55)" strokeWidth="2" />
    </svg>
  );
}

export default function OholPaper() {
  const [s, setS] = useState<State | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const load = () => apiGet<State>('/api/ohol-paper/state').then(setS).catch((e) => setErr(String(e)));
  useEffect(() => { load(); const t = setInterval(load, 20000); return () => clearInterval(t); }, []);

  if (err) return <div className={styles.root}><div className={styles.loading}>Error: {err}</div></div>;
  if (!s) return <div className={styles.root}><div className={styles.loading}>Loading paper book…</div></div>;

  const nav = s.navcurve.length ? s.navcurve[s.navcurve.length - 1].nav : s.capital;
  const totalRet = ((nav - s.capital) / s.capital) * 100;
  const running = !s.killed;

  // pair raw fills into round-trip trades (fills arrive newest-first)
  type Trade = { symbol: string; dir: string; lot: number; entryTs: string; entry: number;
                 exitTs: string | null; exit: number | null; exitVia: string | null; pnl: number | null };
  const trades: Trade[] = [];
  [...s.recent_fills].reverse().forEach((f) => {
    if (f.reason === 'OHOL') {
      const dir = f.side === 'SELL' ? 'SHORT' : 'LONG';
      trades.push({ symbol: f.symbol, dir, lot: f.lot_size, entryTs: f.ts, entry: f.price,
                    exitTs: null, exit: null, exitVia: null, pnl: null });
    } else {
      const t = trades.filter((x) => x.symbol === f.symbol && x.exit == null).pop();
      if (t) { t.exitTs = f.ts; t.exit = f.price; t.exitVia = f.reason; t.pnl = f.pnl; }
    }
  });
  const tkey = (ts: string) => ts.replace(' ', 'T');
  trades.sort((a, b) => tkey(b.entryTs).localeCompare(tkey(a.entryTs)));
  const fmtTs = (ts: string | null) => {
    if (!ts) return '—';
    const d = new Date(tkey(ts));
    return isNaN(d.getTime()) ? ts
      : d.toLocaleDateString('en-IN', { day: '2-digit', month: 'short' }) + ' ' +
        d.toTimeString().slice(0, 5);
  };

  return (
    <div className={styles.root}>
      <div className={styles.headerRow}>
        <div>
          <h1 className={styles.title}>O=H / O=L First-Candle — Live Paper Futures Book</h1>
          <p className={styles.sub}>
            1-lot futures per signal · open=low breaks long, open=high breaks short · SuperTrend(7,2) 5-min trail
            {s.inception ? ` · since ${s.inception}` : ''} · scan {s.last_scan || '—'} · cycle {s.last_cycle || '—'}
          </p>
        </div>
        <div className={`${styles.gateBadge} ${running ? styles.on : styles.off}`}>
          <span className={styles.dot} />
          {running ? 'PAPER · running' : 'KILLED'}
        </div>
      </div>

      <div className={styles.kpis}>
        <Kpi label="NAV" value={lakh(nav)} tone="" />
        <Kpi label="Total return" value={(totalRet >= 0 ? '+' : '') + totalRet.toFixed(2) + '%'} tone={totalRet >= 0 ? 'pos' : 'neg'} />
        <Kpi label="Realized P&L" value={inr(s.realized)} tone={s.realized >= 0 ? 'pos' : 'neg'} />
        <Kpi label="Open positions" value={`${s.n_positions} / 10`} tone="" />
        <Kpi label="Today's setups" value={String(s.today_setups.length)} tone="" />
        <Kpi label="Margin pool" value={lakh(s.capital)} tone="" />
      </div>

      <div className={styles.card}>
        <div className={styles.cardTitle}>Paper NAV (₹10L margin pool + P&L)</div>
        <Curve data={s.navcurve} />
      </div>

      <div className={styles.card}>
        <div className={styles.cardTitle}>Today's setups ({s.today_setups.length})</div>
        {s.today_setups.length === 0 ? (
          <div className={styles.chartEmpty}>The 09:21 scan lists O=L / O=H candidates here each morning.</div>
        ) : (
          <table className={styles.table}>
            <thead><tr><th>Stock</th><th>Side</th><th>Trigger</th><th>Status</th></tr></thead>
            <tbody>
              {s.today_setups.map((u) => (
                <tr key={u.symbol}>
                  <td className={styles.sym}>{u.symbol}</td>
                  <td className={u.side === 'LONG' ? styles.pos : styles.neg}>{u.side}</td>
                  <td>{u.trigger.toFixed(2)}</td>
                  <td><span className={styles.reason}>{u.status}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className={styles.card}>
        <div className={styles.cardTitle}>Open positions ({s.n_positions})</div>
        {s.positions.length === 0 ? (
          <div className={styles.chartEmpty}>No open positions.</div>
        ) : (
          <table className={styles.table}>
            <thead><tr><th>Stock</th><th>Side</th><th>Lot size</th><th>Entry ₹</th><th>Since</th></tr></thead>
            <tbody>
              {s.positions.map((p) => (
                <tr key={p.symbol}>
                  <td className={styles.sym}>{p.symbol}</td>
                  <td className={p.side === 'LONG' ? styles.pos : styles.neg}>{p.side}</td>
                  <td>{p.lot_size}</td>
                  <td>{p.entry.toFixed(2)}</td>
                  <td className={styles.muted}>{p.since}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className={styles.card}>
        <div className={styles.cardTitle}>Trades (entry ⇄ exit)</div>
        {trades.length === 0 ? (
          <div className={styles.chartEmpty}>No trades yet.</div>
        ) : (
          <table className={styles.table}>
            <thead><tr>
              <th>Stock</th><th>Side</th><th>Lot</th><th>Entry time</th><th>Entry ₹</th>
              <th>Exit time</th><th>Exit ₹</th><th>Exit via</th><th>P&L</th>
            </tr></thead>
            <tbody>
              {trades.map((t, i) => (
                <tr key={i}>
                  <td className={styles.sym}>{t.symbol}</td>
                  <td className={t.dir === 'LONG' ? styles.pos : styles.neg}>{t.dir}</td>
                  <td>{t.lot}</td>
                  <td className={styles.muted}>{fmtTs(t.entryTs)}</td>
                  <td>{t.entry.toFixed(2)}</td>
                  <td className={styles.muted}>{fmtTs(t.exitTs)}</td>
                  <td>{t.exit == null ? '—' : t.exit.toFixed(2)}</td>
                  <td><span className={styles.reason}>{t.exitVia ?? 'OPEN'}</span></td>
                  <td className={t.pnl == null ? styles.muted : t.pnl >= 0 ? styles.pos : styles.neg}>
                    {t.pnl == null ? 'open' : inr(t.pnl)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <p className={styles.note}>
          A scanned setup becomes a trade only when its trigger level actually breaks — ARMED setups
          that never break (e.g. a long whose first-candle high is never crossed) expire untraded at the close.
          Shorts enter with a SELL fill and exit with a BUY; the P&L belongs to the round-trip, shown here as one row.
        </p>
      </div>

      <div className={styles.card}>
        <div className={styles.cardTitle}>System rules (your spec, verbatim mechanics)</div>
        <div className={styles.rules}>
          {RULES.map(([k, v]) => (
            <div key={k} className={styles.ruleRow}><div className={styles.ruleK}>{k}</div><div className={styles.ruleV}>{v}</div></div>
          ))}
        </div>
        <p className={styles.note}>Paper only — cannot place a real order. Kill switch: POST /api/ohol-paper/kill-switch.</p>
      </div>
    </div>
  );
}

function Kpi({ label, value, tone }: { label: string; value: string; tone: string }) {
  return (
    <div className={styles.kpi}>
      <div className={`${styles.kpiVal} ${tone === 'pos' ? styles.pos : tone === 'neg' ? styles.neg : ''}`}>{value}</div>
      <div className={styles.kpiLabel}>{label}</div>
    </div>
  );
}
