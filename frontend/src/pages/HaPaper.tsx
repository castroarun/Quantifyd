import { useEffect, useState } from 'react';
import { apiGet } from '../api/client';
import styles from './HaPaper.module.css';
import BookActivity from '../components/BookActivity/BookActivity';

type Position = { symbol: string; qty: number; entry: number; since: string;
                  last: number | null; mtm: number | null; mtm_pct: number | null };
type NavPt = { d: string; nav: number; bench: number | null };
type Fill = {
  ts: string; symbol: string; side: string; price: number;
  reason: string; pnl: number | null;
};
type State = {
  mode: string; system: string; capital: number; inception: string | null;
  killed: boolean; last_cycle: string | null; cash: number; n_positions: number;
  positions: Position[]; navcurve: NavPt[]; recent_fills: Fill[];
};

const inr = (n: number) => '₹' + Math.round(n).toLocaleString('en-IN');
const lakh = (n: number) => '₹' + (n / 100000).toFixed(2) + 'L';
const pct = (n: number | null) => (n == null ? '—' : (n >= 0 ? '+' : '') + n.toFixed(2) + '%');

const RULES: [string, string][] = [
  ['Entry', '2 consecutive GREEN 30-min Heikin-Ashi candles, both with no lower wick, then a bar CLOSES above their real high → buy at next bar open. Long only.'],
  ['Exit', '2 consecutive RED HA candles, then a bar closes below their real low → sell at next bar open. No stops, no targets — the mirror pattern is the exit.'],
  ['Sizing', '81 equal sleeves of ₹24.7k (₹20L total), one per F&O name. A sleeve is either in its trade or earning 5.5% liquid yield. No selection, no ranking.'],
  ['Cadence', 'Recomputed 1 min after every 30-min bar close (09:46 … 15:16, 15:31). Stateless replay of 10 days of bars each cycle — restart-proof.'],
  ['Costs', '3bp per side (slippage + charges) deducted on every paper fill.'],
  ['Provenance', 'research/86 — the only system of the 6-study edge-discovery program (r/81-86) to pass IS → Validation → out-of-sample. OOS: +25bps/trade (t=3.7), book 11.6% CAGR vs 5.6% NIFTYBEES, max DD −11%, beat the index in all 3 OOS years.'],
  ['Soak question', 'Per-trade edge faded 47 → 36 → 25bps across IS/Val/OOS. This book answers: is that decay or drought? Review ~Oct 2026 vs the +25bps OOS benchmark.'],
];

function EquityCurve({ data }: { data: NavPt[] }) {
  if (data.length < 2) return <div className={styles.chartEmpty}>Equity curve builds as the book runs ({data.length} point{data.length === 1 ? '' : 's'} so far).</div>;
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

export default function HaPaper() {
  const [s, setS] = useState<State | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const load = () => apiGet<State>('/api/ha-paper/state').then(setS).catch((e) => setErr(String(e)));
  useEffect(() => {
    load();
    const t = setInterval(load, 30000);
    return () => clearInterval(t);
  }, []);

  if (err) return <div className={styles.root}><div className={styles.loading}>Error: {err}</div></div>;
  if (!s) return <div className={styles.root}><div className={styles.loading}>Loading paper book…</div></div>;

  const nav = s.navcurve.length ? s.navcurve[s.navcurve.length - 1].nav : s.capital;
  const totalRet = ((nav - s.capital) / s.capital) * 100;
  const investedPct = nav > 0 ? ((nav - s.cash) / nav) * 100 : 0;
  const sells = s.recent_fills.filter((f) => f.side === 'SELL' && f.pnl != null);
  const realized = sells.reduce((a, f) => a + (f.pnl as number), 0);
  const wins = sells.filter((f) => (f.pnl as number) > 0).length;
  const unreal = s.positions.reduce((a, p) => a + (p.mtm ?? 0), 0);
  const running = !s.killed;

  // pair fills into round-trip trades: long-only, so BUY = entry, SELL = exit
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
          <h1 className={styles.title}>HA 2-Green Break — Live Paper Book</h1>
          <p className={styles.sub}>
            ₹20L paper deployment of the research/86 survivor · 30-min Heikin-Ashi 2-green no-wick breakout,
            long only, 81 F&O sleeves, idle cash in liquid
            {s.inception ? ` · since ${s.inception}` : ''} · last cycle {s.last_cycle || '—'}
          </p>
        </div>
        <div className={`${styles.gateBadge} ${running ? styles.on : styles.off}`}>
          <span className={styles.dot} />
          {running ? 'PAPER · running' : 'KILLED · cycles paused'}
        </div>
      </div>
      <BookActivity bookId="ha-paper" />

      {/* KPI strip */}
      <div className={styles.kpis}>
        <Kpi label="NAV (EOD mark)" value={lakh(nav)} tone="" />
        <Kpi label="Total return" value={pct(totalRet)} tone={totalRet >= 0 ? 'pos' : 'neg'} />
        <Kpi label="Invested" value={investedPct.toFixed(0) + '%'} tone="" />
        <Kpi label="Sleeve cash" value={lakh(s.cash)} tone="" />
        <Kpi label="Open positions" value={`${s.n_positions} / 81`} tone="" />
        <Kpi label="Realized (recent)" value={inr(realized)} tone={realized >= 0 ? 'pos' : 'neg'} />
        <Kpi label="Unrealized (live)" value={inr(unreal)} tone={unreal >= 0 ? 'pos' : 'neg'} />
        <Kpi label="Recent win rate" value={sells.length ? `${wins}/${sells.length}` : '—'} tone="" />
        <Kpi label="Capital" value={lakh(s.capital)} tone="" />
      </div>

      {/* Equity curve */}
      <div className={styles.card}>
        <div className={styles.cardTitle}>Paper NAV vs NIFTYBEES (growth of ₹1, EOD marks)</div>
        <EquityCurve data={s.navcurve} />
      </div>

      {/* Open positions */}
      <div className={styles.card}>
        <div className={styles.cardTitle}>Open positions ({s.n_positions})</div>
        {s.positions.length === 0 ? (
          <div className={styles.chartEmpty}>
            Flat — all 81 sleeves in liquid. The book only takes fresh pattern fires (it was seeded flat, mid-pattern names excluded).
          </div>
        ) : (
          <table className={styles.table}>
            <thead><tr>
              <th>Stock</th><th>Qty</th><th>Entry ₹</th><th>Now ₹</th><th>P&L ₹</th><th>P&L %</th><th>Since</th>
            </tr></thead>
            <tbody>
              {s.positions.map((p) => (
                <tr key={p.symbol}>
                  <td className={styles.sym}>{p.symbol}</td>
                  <td>{p.qty}</td>
                  <td>{p.entry.toFixed(2)}</td>
                  <td>{p.last == null ? '—' : p.last.toFixed(2)}</td>
                  <td className={p.mtm == null ? styles.muted : p.mtm >= 0 ? styles.pos : styles.neg}>
                    {p.mtm == null ? '—' : inr(p.mtm)}</td>
                  <td className={p.mtm_pct == null ? styles.muted : p.mtm_pct >= 0 ? styles.pos : styles.neg}>
                    {p.mtm_pct == null ? '—' : (p.mtm_pct >= 0 ? '+' : '') + p.mtm_pct.toFixed(2) + '%'}</td>
                  <td className={styles.muted}>{fmtTs(p.since)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Trades */}
      <div className={styles.card}>
        <div className={styles.cardTitle}>Trades (entry ⇄ exit — long only; a SELL is always the exit of a long)</div>
        {trades.length === 0 ? (
          <div className={styles.chartEmpty}>No trades yet — entries arrive as fresh 2-green patterns fire and break.</div>
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
        <p className={styles.note}>
          The book is LONG ONLY. Entry = a bar closes above the 2-green pattern high (BUY).
          Exit = a bar closes below the 2-red pattern low (SELL) — the exit of the long, not a short.
          Trades still open show live P&L in the Open positions table above.
        </p>
      </div>

      {/* Rules */}
      <div className={styles.card}>
        <div className={styles.cardTitle}>System rules (locked — OOS is consumed, no further tuning)</div>
        <div className={styles.rules}>
          {RULES.map(([k, v]) => (
            <div key={k} className={styles.ruleRow}><div className={styles.ruleK}>{k}</div><div className={styles.ruleV}>{v}</div></div>
          ))}
        </div>
        <p className={styles.note}>
          Paper only — this module cannot place a real order. Kill switch: POST /api/ha-paper/kill-switch.
        </p>
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
