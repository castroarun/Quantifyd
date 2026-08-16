import { useEffect, useState } from 'react';
import { apiGet } from '../api/client';
import styles from './FnomsPaper.module.css';

type Position = { symbol: string; system: string; qty: number; entry: number; since: string;
                  stop: number; target: number; last: number | null;
                  mtm: number | null; mtm_pct: number | null };
type NavPt = { d: string; nav: number; bench: number | null };
type Fill = { ts: string; symbol: string; system: string; side: string; price: number;
              qty: number; reason: string; pnl: number | null };
type State = {
  mode: string; system: string; capital: number; inception: string | null;
  killed: boolean; last_run: string | null; cash: number; n_positions: number;
  positions: Position[]; navcurve: NavPt[]; recent_fills: Fill[];
};

const inr = (n: number) => '₹' + Math.round(n).toLocaleString('en-IN');
const lakh = (n: number) => '₹' + (n / 100000).toFixed(2) + 'L';

const SYSTEMS: [string, string, string][] = [
  ['DON55', 'Donchian-55 + volume 3× + ATR floor', 'OOS PF 1.97 · Sharpe 1.25 · MaxDD 7.7% · CAGR 13.5% (175 trades)'],
  ['ADX30', 'Donchian-55 + volume 3× + ADX(14) > 30', 'OOS PF 1.77 · Sharpe 1.30 · MaxDD 16.5% · CAGR 15.5% (225 trades)'],
  ['PBRSI', 'Uptrend pullback: close > 200-SMA, RSI < 40, then prior-high break', 'OOS PF 1.55 · Sharpe 1.01 · MaxDD 18.2% · CAGR 13.0% (233 trades)'],
];

const RULES: [string, string][] = [
  ['Universe', 'F&O stocks, daily bars. All three systems are LONG only — the 24 SHORT variants tested alongside them were dead on arrival.'],
  ['Entry', 'Evaluated at 15:40 on completed bars; fills at the signal price + 10bp. A symbol already held by any system is skipped, so the book never doubles up on one name.'],
  ['Stop', 'max(entry − 2×ATR14, entry − 8%) — the wider of the volatility stop and the hard floor.'],
  ['Target / time', '+25% target; 60-session max hold.'],
  ['Sizing', '1% of equity risked per trade, capped at equity/30 notional; max 10 concurrent per system (30 book-wide). Idle cash earns 5.5%.'],
  ['Validation', 'All three PASSED walk-forward on OOS 2023-01-01→2025-12-31 against pre-set gates (PF ≥ 1.20, Sharpe ≥ 0.8, MaxDD ≤ 30%). This paper book is the forward test of that validation.'],
];

function Curve({ data }: { data: NavPt[] }) {
  if (data.length < 2) return <div className={styles.chartEmpty}>NAV curve builds daily ({data.length} point{data.length === 1 ? '' : 's'} so far).</div>;
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

export default function FnomsPaper() {
  const [s, setS] = useState<State | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const load = () => apiGet<State>('/api/fnoms-paper/state').then(setS).catch((e) => setErr(String(e)));
  useEffect(() => { load(); const t = setInterval(load, 30000); return () => clearInterval(t); }, []);

  if (err) return <div className={styles.root}><div className={styles.loading}>Error: {err}</div></div>;
  if (!s) return <div className={styles.root}><div className={styles.loading}>Loading paper book…</div></div>;

  const nav = s.navcurve.length ? s.navcurve[s.navcurve.length - 1].nav : s.capital;
  const totalRet = ((nav - s.capital) / s.capital) * 100;
  const unreal = s.positions.reduce((a, p) => a + (p.mtm ?? 0), 0);
  const sells = s.recent_fills.filter((f) => f.side === 'SELL' && f.pnl != null);
  const realized = sells.reduce((a, f) => a + (f.pnl as number), 0);
  const running = !s.killed;

  type Trade = { symbol: string; system: string; qty: number; entryTs: string; entry: number;
                 exitTs: string | null; exit: number | null; exitVia: string | null; pnl: number | null };
  const trades: Trade[] = [];
  [...s.recent_fills].reverse().forEach((f) => {
    if (f.side === 'BUY') {
      trades.push({ symbol: f.symbol, system: f.system, qty: f.qty, entryTs: f.ts, entry: f.price,
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
      : d.toLocaleDateString('en-IN', { day: '2-digit', month: 'short' }) + ' ' + d.toTimeString().slice(0, 5);
  };
  const bySystem = SYSTEMS.map(([k]) => ({
    k, n: s.positions.filter((p) => p.system === k).length,
    mtm: s.positions.filter((p) => p.system === k).reduce((a, p) => a + (p.mtm ?? 0), 0),
  }));

  return (
    <div className={styles.root}>
      <div className={styles.headerRow}>
        <div>
          <h1 className={styles.title}>F&amp;O Multi-Signal — Live Paper Book</h1>
          <p className={styles.sub}>
            ₹20L across 3 walk-forward-validated LONG systems · Donchian-55 breakout, ADX-30 trend, RSI-40 pullback
            {s.inception ? ` · since ${s.inception}` : ''} · last run {s.last_run || '—'}
          </p>
        </div>
        <div className={`${styles.gateBadge} ${running ? styles.on : styles.off}`}>
          <span className={styles.dot} />{running ? 'PAPER · running' : 'KILLED'}
        </div>
      </div>

      <div className={styles.kpis}>
        <Kpi label="NAV" value={lakh(nav)} tone="" />
        <Kpi label="Total return" value={(totalRet >= 0 ? '+' : '') + totalRet.toFixed(2) + '%'} tone={totalRet >= 0 ? 'pos' : 'neg'} />
        <Kpi label="Cash" value={lakh(s.cash)} tone="" />
        <Kpi label="Open positions" value={`${s.n_positions} / 30`} tone="" />
        <Kpi label="Unrealized" value={inr(unreal)} tone={unreal >= 0 ? 'pos' : 'neg'} />
        <Kpi label="Realized (recent)" value={inr(realized)} tone={realized >= 0 ? 'pos' : 'neg'} />
        <Kpi label="Capital" value={lakh(s.capital)} tone="" />
      </div>

      <div className={styles.card}>
        <div className={styles.cardTitle}>The three systems (each independently walk-forward validated)</div>
        <table className={styles.table}>
          <thead><tr><th>System</th><th>Rule</th><th>Out-of-sample record</th><th>Open now</th><th>Unrealized</th></tr></thead>
          <tbody>
            {SYSTEMS.map(([k, rule, oos]) => {
              const b = bySystem.find((x) => x.k === k)!;
              return (
                <tr key={k}>
                  <td className={styles.sym}>{k}</td>
                  <td>{rule}</td>
                  <td className={styles.muted}>{oos}</td>
                  <td>{b.n} / 10</td>
                  <td className={b.mtm >= 0 ? styles.pos : styles.neg}>{inr(b.mtm)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className={styles.card}>
        <div className={styles.cardTitle}>Paper NAV vs NIFTYBEES (growth of ₹1)</div>
        <Curve data={s.navcurve} />
      </div>

      <div className={styles.card}>
        <div className={styles.cardTitle}>Open positions ({s.n_positions})</div>
        {s.positions.length === 0 ? (
          <div className={styles.chartEmpty}>Flat — the book enters at 15:40 when a system fires.</div>
        ) : (
          <table className={styles.table}>
            <thead><tr>
              <th>Stock</th><th>System</th><th>Qty</th><th>Entry ₹</th><th>Now ₹</th>
              <th>P&L ₹</th><th>P&L %</th><th>Stop</th><th>Target</th><th>Since</th>
            </tr></thead>
            <tbody>
              {s.positions.map((p) => (
                <tr key={p.symbol}>
                  <td className={styles.sym}>{p.symbol}</td>
                  <td><span className={styles.reason}>{p.system}</span></td>
                  <td>{p.qty}</td>
                  <td>{p.entry.toFixed(2)}</td>
                  <td>{p.last == null ? '—' : p.last.toFixed(2)}</td>
                  <td className={p.mtm == null ? styles.muted : p.mtm >= 0 ? styles.pos : styles.neg}>
                    {p.mtm == null ? '—' : inr(p.mtm)}</td>
                  <td className={p.mtm_pct == null ? styles.muted : p.mtm_pct >= 0 ? styles.pos : styles.neg}>
                    {p.mtm_pct == null ? '—' : (p.mtm_pct >= 0 ? '+' : '') + p.mtm_pct.toFixed(2) + '%'}</td>
                  <td className={styles.muted}>{p.stop.toFixed(2)}</td>
                  <td className={styles.muted}>{p.target.toFixed(2)}</td>
                  <td className={styles.muted}>{fmtTs(p.since)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className={styles.card}>
        <div className={styles.cardTitle}>Trades (entry ⇄ exit — long only)</div>
        {trades.length === 0 ? (
          <div className={styles.chartEmpty}>No trades yet.</div>
        ) : (
          <table className={styles.table}>
            <thead><tr>
              <th>Stock</th><th>System</th><th>Qty</th><th>Entry time</th><th>Entry ₹</th>
              <th>Exit time</th><th>Exit ₹</th><th>Exit via</th><th>P&L</th>
            </tr></thead>
            <tbody>
              {trades.map((t, i) => (
                <tr key={i}>
                  <td className={styles.sym}>{t.symbol}</td>
                  <td><span className={styles.reason}>{t.system}</span></td>
                  <td>{t.qty}</td>
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
        <div className={styles.cardTitle}>System rules</div>
        <div className={styles.rules}>
          {RULES.map(([k, v]) => (
            <div key={k} className={styles.ruleRow}><div className={styles.ruleK}>{k}</div><div className={styles.ruleV}>{v}</div></div>
          ))}
        </div>
        <p className={styles.note}>Paper only — cannot place a real order. Kill switch: POST /api/fnoms-paper/kill-switch.</p>
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
