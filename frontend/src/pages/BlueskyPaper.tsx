import { useEffect, useState } from 'react';
import styles from './BlueskyPaper.module.css';

type Pos = {
  symbol: string; qty: number; buy: number; entry_date: string; pivot: number | null;
  src?: string; ltp: number | null; value: number; pnl: number; pnl_pct: number | null;
  days: number; stop: number; to_stop_pct: number | null; trail: number | null;
  to_trail_pct: number | null; weight: number | null;
};
type Pending = { symbol: string; pivot: number; rs: number; signal_date: string };
type Trade = {
  symbol: string; entry_date: string; exit_date: string; buy: number; sell: number;
  qty: number | null; net_pnl?: number | null; ret_pct: number; reason: string; src?: string;
};
type NavPt = { date: string; nav: number; bench: number | null };
type Feed = {
  updated: string; nav: number; capital: number; cash: number; invested_pct: number;
  unrealized: number; ret_pct: number; max_dd_pct: number;
  gate_weak: boolean; gate_nb: number | null; gate_sma: number | null; gate_gap_pct: number | null;
  positions: Pos[]; pending: Pending[]; trades: Trade[]; n_trades: number;
  n_live_trades: number; win_pct: number | null; nav_curve: NavPt[]; spec: string;
  provenance: string | null; study: string; log: string[];
};

const inr = (n: number) => '₹' + Math.round(n).toLocaleString('en-IN');
const lakh = (n: number) => '₹' + (n / 100000).toFixed(2) + 'L';
const pct = (n: number | null | undefined) =>
  n == null ? '—' : (n >= 0 ? '+' : '') + n.toFixed(1) + '%';
const MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const fmtD = (s: string | null | undefined) => {
  if (!s) return '—';
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(s);
  return m ? `${m[3]}-${MON[parseInt(m[2], 10) - 1]}-${m[1]}` : s;
};
const reasonLabel: Record<string, string> = {
  stop_8pct: '−8% stop', trail_sma20: '20-SMA trail', trail_50d: '50-SMA trail',
  open_marked: 'still open',
};
/* Heat tints — same visual language as the Momentum page. */
const pnlTint = (p: number | null | undefined): React.CSSProperties => {
  if (p == null || !isFinite(p)) return {};
  const t = Math.min(1, Math.abs(p) / 10);
  const a = 0.10 + 0.34 * t;
  return { background: p >= 0 ? `rgba(47,145,82,${a})` : `rgba(224,86,79,${a})`,
           fontWeight: Math.abs(p) >= 5 ? 700 : 600 };
};
const exitTint = (d: number | null | undefined): React.CSSProperties => {
  if (d == null || !isFinite(d)) return {};
  if (d < 2) return { background: 'rgba(224,86,79,0.44)', fontWeight: 700 };
  if (d < 5) return { background: 'rgba(217,119,6,0.30)', fontWeight: 650 };
  if (d < 10) return { background: 'rgba(217,119,6,0.13)' };
  return { background: 'rgba(47,145,82,0.15)' };
};

function Curve({ data }: { data: NavPt[] }) {
  if (data.length < 2)
    return <div className={styles.empty}>Equity curve builds as the soak runs.</div>;
  const W = 760, H = 220, P = 8;
  const n0 = data[0].nav;
  const b0 = data.find((d) => d.bench != null)?.bench ?? 1;
  const navN = data.map((d) => d.nav / n0);
  const benN = data.map((d) => (d.bench == null ? null : d.bench / b0));
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

export default function BlueskyPaper() {
  const [f, setF] = useState<Feed | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    fetch('/app/bluesky_paper.json')
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then(setF)
      .catch((e) => setErr(String(e)));
  }, []);
  if (err)
    return <div className={styles.page}><div className={styles.empty}>
      No feed yet ({err}) — first snapshot lands after the 18:40 IST run.</div></div>;
  if (!f) return <div className={styles.page}><div className={styles.empty}>Loading…</div></div>;
  const totPnl = f.positions.reduce((a, p) => a + (p.pnl || 0), 0);
  const totVal = f.positions.reduce((a, p) => a + (p.value || 0), 0);
  return (
    <div className={styles.page}>
      <div className={styles.studyBar}>
        <span className={styles.studyBarLabel}>Evidence</span>
        <a className={styles.studyLink} href={f.study}>The study this book runs</a>
        <a className={styles.studyLink} href="/app/strategies#bluesky-paper">Register entry</a>
      </div>
      <div className={styles.head}>
        <div>
          <h1>BlueSky ATH Breakout — Paper Book</h1>
          <div className={styles.sub}>
            {f.spec} · updated {fmtD(f.updated)} · live trades so far: {f.n_live_trades}
          </div>
        </div>
        <span className={f.gate_weak ? styles.chipWeak : styles.chipOk}>
          {f.gate_weak ? 'GATE WEAK — no new entries' : 'RISK-ON — gate open'}
        </span>
      </div>
      <div className={styles.tiles}>
        <div className={styles.tile}><div>NAV</div><b>{inr(f.nav)}</b></div>
        <div className={styles.tile}><div>Return</div>
          <b className={f.ret_pct >= 0 ? styles.pos : styles.neg}>{pct(f.ret_pct)}</b></div>
        <div className={styles.tile}><div>Unrealised</div>
          <b className={f.unrealized >= 0 ? styles.pos : styles.neg}>{inr(f.unrealized)}</b></div>
        <div className={styles.tile}><div>Cash / Invested</div>
          <b>{lakh(f.cash)} / {f.invested_pct}%</b></div>
        <div className={styles.tile}><div>Max DD</div><b className={styles.neg}>{pct(f.max_dd_pct)}</b></div>
        <div className={styles.tile}><div>Trades · Win%</div>
          <b>{f.n_trades} · {f.win_pct == null ? '—' : f.win_pct + '%'}</b></div>
      </div>

      <div className={styles.card}>
        <div className={styles.cardTitle}>Equity vs NIFTYBEES (both rebased at book start; dashed = index)</div>
        <Curve data={f.nav_curve} />
      </div>

      <div className={styles.card}>
        <div className={styles.cardTitle}>Holdings</div>
        {f.positions.length === 0 ? <div className={styles.empty}>none — in cash</div> : (
        <table className={styles.tbl}><thead><tr>
          <th>Holding</th><th>Wt</th><th>Entry</th><th>Entry ₹</th><th>Now ₹</th><th>Value</th>
          <th>P&L ₹</th><th>P&L %</th><th>Days</th><th>Stop −8%</th><th>To stop</th>
          <th>20-SMA trail</th><th>To trail</th>
        </tr></thead><tbody>
          {f.positions.map((p) => (
            <tr key={p.symbol + p.entry_date}>
              <td className={styles.sym}>{p.symbol}</td>
              <td>{p.weight == null ? '—' : p.weight + '%'}</td>
              <td className={styles.muted}>{fmtD(p.entry_date)}</td>
              <td>{p.buy}</td><td>{p.ltp ?? '—'}</td><td>{lakh(p.value)}</td>
              <td className={(p.pnl ?? 0) >= 0 ? styles.pos : styles.neg} style={pnlTint(p.pnl_pct)}>
                {(p.pnl ?? 0) >= 0 ? '+' : ''}{inr(p.pnl ?? 0)}</td>
              <td className={(p.pnl_pct ?? 0) >= 0 ? styles.pos : styles.neg} style={pnlTint(p.pnl_pct)}>
                {pct(p.pnl_pct)}</td>
              <td>{p.days}</td>
              <td className={styles.muted}>{p.stop}</td>
              <td style={exitTint(p.to_stop_pct)} title="distance above the −8% hard stop">
                {p.to_stop_pct == null ? '—' : '+' + p.to_stop_pct + '%'}</td>
              <td className={styles.muted}>{p.trail ?? '—'}</td>
              <td style={exitTint(p.to_trail_pct)}
                  title="distance above the 20-SMA trail — the usual exit; red means an exit is imminent">
                {p.to_trail_pct == null ? '—' : (p.to_trail_pct >= 0 ? '+' : '') + p.to_trail_pct + '%'}</td>
            </tr>
          ))}
        </tbody>
        <tfoot><tr className={styles.totals}>
          <td>TOTAL ({f.positions.length} stocks)</td><td /><td /><td /><td />
          <td>{lakh(totVal)}</td>
          <td className={totPnl >= 0 ? styles.pos : styles.neg}>{totPnl >= 0 ? '+' : ''}{inr(totPnl)}</td>
          <td colSpan={6} />
        </tr></tfoot>
        </table>)}
      </div>

      <div className={styles.grid2}>
        <div className={styles.card}>
          <div className={styles.cardTitle}>Macro gate — NIFTYBEES vs 200-DMA</div>
          <div className={styles.gateRow}>
            <div><span className={styles.muted}>NIFTYBEES</span><b>{f.gate_nb ?? '—'}</b></div>
            <div><span className={styles.muted}>200-DMA</span><b>{f.gate_sma ?? '—'}</b></div>
            <div><span className={styles.muted}>Gap</span>
              <b className={(f.gate_gap_pct ?? 0) >= 0 ? styles.pos : styles.neg}>{pct(f.gate_gap_pct)}</b></div>
            <div><span className={styles.muted}>State</span>
              <b className={f.gate_weak ? styles.neg : styles.pos}>{f.gate_weak ? 'WEAK' : 'RISK-ON'}</b></div>
          </div>
          <p className={styles.note}>Below the 200-DMA no NEW breakouts are taken; open positions still run their stop and trail.</p>
        </div>
        <div className={styles.card}>
          <div className={styles.cardTitle}>Pending buy-stops (tomorrow)</div>
          {f.pending.length === 0 ? <div className={styles.empty}>none</div> : (
          <table className={styles.tbl}><thead><tr>
            <th>Symbol</th><th>Pivot (buy-stop)</th><th>RS</th><th>Signalled</th>
          </tr></thead><tbody>
            {f.pending.map((p) => (
              <tr key={p.symbol}><td className={styles.sym}>{p.symbol}</td>
                <td>{p.pivot}</td><td>{p.rs}</td><td className={styles.muted}>{fmtD(p.signal_date)}</td></tr>
            ))}
          </tbody></table>)}
        </div>
      </div>

      <div className={styles.card}>
        <div className={styles.cardTitle}>Closed trades (latest first)</div>
        {f.trades.length === 0 ? <div className={styles.empty}>none yet</div> : (
        <table className={styles.tbl}><thead><tr>
          <th>Symbol</th><th>Entry</th><th>Exit</th><th>Buy ₹</th><th>Sell ₹</th>
          <th>Return</th><th>Why exited</th><th>Source</th>
        </tr></thead><tbody>
          {[...f.trades].reverse().map((t, i) => (
            <tr key={i}>
              <td className={styles.sym}>{t.symbol}</td>
              <td className={styles.muted}>{fmtD(t.entry_date)}</td>
              <td className={styles.muted}>{fmtD(t.exit_date)}</td>
              <td>{t.buy}</td><td>{t.sell}</td>
              <td className={t.ret_pct >= 0 ? styles.pos : styles.neg} style={pnlTint(t.ret_pct)}>
                {pct(t.ret_pct)}</td>
              <td className={styles.muted}>{reasonLabel[t.reason] ?? t.reason}</td>
              <td className={styles.muted}>{t.src === 'live' ? 'LIVE' : 'backfill'}</td>
            </tr>
          ))}
        </tbody></table>)}
      </div>

      {f.provenance && <p className={styles.note}>{f.provenance}</p>}
      <div className={styles.foot}>Last run: {f.log.join(' · ')}</div>
    </div>
  );
}
