import { useEffect, useState } from 'react';
import { apiGet } from '../api/client';
import styles from './OrbPaper.module.css';

type Position = { symbol: string; qty: number; entry: number; since: string; stop: number; sessions: number };
type NavPt = { d: string; nav: number; gate: number; bench: number | null };
type Fill = { ts: string; symbol: string; side: string; price: number; reason: string; pnl: number | null };
type State = {
  mode: string; system: string; capital: number; inception: string | null;
  killed: boolean; last_cycle: string | null; cash: number; n_positions: number;
  gate_on: boolean; positions: Position[]; navcurve: NavPt[]; recent_fills: Fill[];
};

const inr = (n: number) => '₹' + Math.round(n).toLocaleString('en-IN');
const lakh = (n: number) => '₹' + (n / 100000).toFixed(2) + 'L';
const pct = (n: number | null) => (n == null ? '—' : (n >= 0 ? '+' : '') + n.toFixed(2) + '%');

const RULES: [string, string][] = [
  ['Entry', 'Day gaps up ≥0.4% AND price closes above the 90-minute opening range (09:15–10:45) → buy at market. Long only, checked every 30 min.'],
  ['Stop', 'Opening-range low — checked every cycle, exit at market on breach.'],
  ['Time exit', 'Close of the 4th session (entry day = session 1). The edge is multi-day continuation — research/89 proved the intraday version never worked.'],
  ['Gate', 'NIFTY 50 above its 50-day average, else no new entries (sidesteps the down-tape quarters like 2026Q1 that caused the historical drawdowns).'],
  ['Sizing', '₹10L / 20 equal slots of ₹50k; idle cash earns 5.5%; 5bp/side costs.'],
  ['Provenance', 'research/89: never-died grid passer (+27/+21/+22bps per trade across 2015-21 / 2022-23 / 2024-26, t≈6.7/3.8/4.2). OOS was consumed, so THIS 90-day soak is the validation — target +15-20bps/trade, verdict ~Nov 2026.'],
];

function Curve({ data }: { data: NavPt[] }) {
  if (data.length < 2) return <div className={styles.chartEmpty}>Equity curve builds daily from tomorrow's close ({data.length} point{data.length === 1 ? '' : 's'} so far).</div>;
  const W = 760, H = 220, P = 8;
  const navs = data.map((d) => d.nav);
  const benchRaw = data.map((d) => d.bench);
  const n0 = navs[0];
  const b0 = benchRaw.find((x) => x != null) || 1;
  const navN = navs.map((v) => v / n0);
  const benN = benchRaw.map((v) => (v == null ? null : v / (b0 as number)));
  const all = [...navN, ...(benN.filter((x) => x != null) as number[])];
  const lo = Math.min(...all), hi = Math.max(...all);
  const x = (i: number) => P + (i / (data.length - 1)) * (W - 2 * P);
  const y = (v: number) => P + (1 - (v - lo) / (hi - lo || 1)) * (H - 2 * P);
  const path = (arr: (number | null)[]) =>
    arr.map((v, i) => (v == null ? '' : `${i === 0 || arr[i - 1] == null ? 'M' : 'L'}${x(i)},${y(v)}`)).join(' ');
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className={styles.chart} preserveAspectRatio="none">
      <path d={path(benN)} fill="none" stroke="var(--ink-muted)" strokeWidth="1.2" strokeDasharray="4 3" />
      <path d={path(navN)} fill="none" stroke="var(--accent-pos, #1f9d55)" strokeWidth="2" />
    </svg>
  );
}

export default function OrbPaper() {
  const [s, setS] = useState<State | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const load = () => apiGet<State>('/api/orb-paper/state').then(setS).catch((e) => setErr(String(e)));
  useEffect(() => { load(); const t = setInterval(load, 30000); return () => clearInterval(t); }, []);

  if (err) return <div className={styles.root}><div className={styles.loading}>Error: {err}</div></div>;
  if (!s) return <div className={styles.root}><div className={styles.loading}>Loading paper book…</div></div>;

  const nav = s.navcurve.length ? s.navcurve[s.navcurve.length - 1].nav : s.capital;
  const totalRet = ((nav - s.capital) / s.capital) * 100;
  const sells = s.recent_fills.filter((f) => f.side === 'SELL' && f.pnl != null);
  const realized = sells.reduce((a, f) => a + (f.pnl as number), 0);
  const running = !s.killed;

  type Trade = { symbol: string; entryTs: string; entry: number; exitTs: string | null;
                 exit: number | null; exitVia: string | null; pnl: number | null };
  const trades: Trade[] = [];
  [...s.recent_fills].reverse().forEach((f) => {
    if (f.side === 'BUY') {
      trades.push({ symbol: f.symbol, entryTs: f.ts, entry: f.price,
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
          <h1 className={styles.title}>Gap-ORB 4-Day Revival — Live Paper Book</h1>
          <p className={styles.sub}>
            ₹10L paper deployment of the research/89 revival candidate · 90-min OR breakout on gap-up days,
            long only, ≤4-session hold, NIFTY&gt;50DMA gate
            {s.inception ? ` · since ${s.inception}` : ''} · last cycle {s.last_cycle || '—'}
          </p>
        </div>
        <div className={`${styles.gateBadge} ${s.gate_on ? styles.on : styles.off}`}>
          <span className={styles.dot} />
          {!running ? 'KILLED' : s.gate_on ? 'GATE ON · taking entries' : 'GATE OFF · no new entries'}
        </div>
      </div>

      <div className={styles.kpis}>
        <Kpi label="NAV (EOD mark)" value={lakh(nav)} tone="" />
        <Kpi label="Total return" value={pct(totalRet)} tone={totalRet >= 0 ? 'pos' : 'neg'} />
        <Kpi label="Cash" value={lakh(s.cash)} tone="" />
        <Kpi label="Open positions" value={`${s.n_positions} / 20`} tone="" />
        <Kpi label="Realized (recent)" value={inr(realized)} tone={realized >= 0 ? 'pos' : 'neg'} />
        <Kpi label="Capital" value={lakh(s.capital)} tone="" />
      </div>

      <div className={styles.card}>
        <div className={styles.cardTitle}>Paper NAV vs NIFTYBEES (growth of ₹1, EOD marks)</div>
        <Curve data={s.navcurve} />
      </div>

      <div className={styles.card}>
        <div className={styles.cardTitle}>Open positions ({s.n_positions})</div>
        {s.positions.length === 0 ? (
          <div className={styles.chartEmpty}>Flat — entries arrive on gap-up days that break the 90-min range (gate permitting).</div>
        ) : (
          <table className={styles.table}>
            <thead><tr><th>Stock</th><th>Qty</th><th>Entry ₹</th><th>Stop (OR-low)</th><th>Session</th><th>Since</th></tr></thead>
            <tbody>
              {s.positions.map((p) => (
                <tr key={p.symbol}>
                  <td className={styles.sym}>{p.symbol}</td>
                  <td>{p.qty}</td>
                  <td>{p.entry.toFixed(2)}</td>
                  <td className={styles.muted}>{p.stop.toFixed(2)}</td>
                  <td>{p.sessions} / 4</td>
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
          <div className={styles.chartEmpty}>No trades yet — entries need a ≥0.4% gap-up day whose 90-min range breaks (gate permitting).</div>
        ) : (
          <table className={styles.table}>
            <thead><tr>
              <th>Stock</th><th>Entry time</th><th>Entry ₹</th><th>Exit time</th>
              <th>Exit ₹</th><th>Exit via</th><th>P&L</th>
            </tr></thead>
            <tbody>
              {trades.map((t, i) => (
                <tr key={i}>
                  <td className={styles.sym}>{t.symbol}</td>
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
      </div>

      <div className={styles.card}>
        <div className={styles.cardTitle}>System rules (locked — this soak IS the validation)</div>
        <div className={styles.rules}>
          {RULES.map(([k, v]) => (
            <div key={k} className={styles.ruleRow}><div className={styles.ruleK}>{k}</div><div className={styles.ruleV}>{v}</div></div>
          ))}
        </div>
        <p className={styles.note}>Paper only — cannot place a real order. Kill switch: POST /api/orb-paper/kill-switch.</p>
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
