import { useEffect, useState } from 'react';
import styles from './BlueskyPaper.module.css';

type Pos = { symbol: string; qty: number; buy: number; entry_date: string; pivot: number };
type Pending = { symbol: string; pivot: number; rs: number; signal_date: string };
type Trade = {
  symbol: string; entry_date: string; exit_date: string; buy: number; sell: number;
  qty: number; ret_pct: number; reason: string;
};
type NavPt = { date: string; nav: number };
type Feed = {
  updated: string; nav: number; capital: number; ret_pct: number; max_dd_pct: number;
  gate_weak: boolean; positions: Pos[]; pending: Pending[]; trades: Trade[];
  n_trades: number; win_pct: number | null; nav_curve: NavPt[]; spec: string;
  study: string; log: string[];
};

const inr = (n: number) => '₹' + Math.round(n).toLocaleString('en-IN');
const pct = (n: number | null | undefined) =>
  n == null ? '—' : (n >= 0 ? '+' : '') + n.toFixed(2) + '%';

function Curve({ data }: { data: NavPt[] }) {
  if (data.length < 2)
    return <div className={styles.empty}>Equity curve builds as the soak runs ({data.length} point so far).</div>;
  const W = 760, H = 200, P = 8;
  const v = data.map((d) => d.nav);
  const lo = Math.min(...v), hi = Math.max(...v);
  const x = (i: number) => P + (i / (data.length - 1)) * (W - 2 * P);
  const y = (n: number) => H - P - ((n - lo) / Math.max(1, hi - lo)) * (H - 2 * P);
  const path = v.map((n, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(n).toFixed(1)}`).join(' ');
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className={styles.chart} preserveAspectRatio="none">
      <path d={path} fill="none" stroke="var(--accent-pos, #2f9e6e)" strokeWidth="1.8" />
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
      No feed yet ({err}) — the book publishes its first snapshot after its first 18:40 IST run.
    </div></div>;
  if (!f) return <div className={styles.page}><div className={styles.empty}>Loading…</div></div>;
  return (
    <div className={styles.page}>
      <div className={styles.head}>
        <h1>BlueSky Breakout — ₹10L Paper</h1>
        <span className={f.gate_weak ? styles.chipWeak : styles.chipOk}>
          {f.gate_weak ? 'GATE WEAK — no new entries' : 'gate OK'}
        </span>
      </div>
      <div className={styles.sub}>
        {f.spec} · G5 soak of <a href={f.study}>the research/142 study</a> · updated {f.updated}
      </div>
      <div className={styles.tiles}>
        <div className={styles.tile}><div>NAV</div><b>{inr(f.nav)}</b></div>
        <div className={styles.tile}><div>Return</div><b>{pct(f.ret_pct)}</b></div>
        <div className={styles.tile}><div>Max DD</div><b>{pct(f.max_dd_pct)}</b></div>
        <div className={styles.tile}><div>Positions</div><b>{f.positions.length}/8</b></div>
        <div className={styles.tile}><div>Closed trades</div><b>{f.n_trades}</b></div>
        <div className={styles.tile}><div>Win%</div><b>{f.win_pct == null ? '—' : f.win_pct + '%'}</b></div>
      </div>
      <Curve data={f.nav_curve} />

      <h2>Open positions</h2>
      {f.positions.length === 0 ? <div className={styles.empty}>none</div> : (
        <table className={styles.tbl}><thead><tr>
          <th>Symbol</th><th>Qty</th><th>Buy</th><th>Pivot</th><th>Entered</th>
        </tr></thead><tbody>
          {f.positions.map((p) => (
            <tr key={p.symbol + p.entry_date}>
              <td>{p.symbol}</td><td>{p.qty}</td><td>{p.buy}</td><td>{p.pivot}</td><td>{p.entry_date}</td>
            </tr>
          ))}
        </tbody></table>
      )}

      <h2>Pending buy-stops (tomorrow)</h2>
      {f.pending.length === 0 ? <div className={styles.empty}>none</div> : (
        <table className={styles.tbl}><thead><tr>
          <th>Symbol</th><th>Pivot</th><th>RS</th><th>Signal date</th>
        </tr></thead><tbody>
          {f.pending.map((p) => (
            <tr key={p.symbol}><td>{p.symbol}</td><td>{p.pivot}</td><td>{p.rs}</td><td>{p.signal_date}</td></tr>
          ))}
        </tbody></table>
      )}

      <h2>Recent closed trades</h2>
      {f.trades.length === 0 ? <div className={styles.empty}>none yet</div> : (
        <table className={styles.tbl}><thead><tr>
          <th>Symbol</th><th>Entry</th><th>Exit</th><th>Buy</th><th>Sell</th><th>Return</th><th>Why</th>
        </tr></thead><tbody>
          {[...f.trades].reverse().map((t, i) => (
            <tr key={i}>
              <td>{t.symbol}</td><td>{t.entry_date}</td><td>{t.exit_date}</td>
              <td>{t.buy}</td><td>{t.sell}</td>
              <td className={t.ret_pct >= 0 ? styles.pos : styles.neg}>{pct(t.ret_pct)}</td>
              <td>{t.reason}</td>
            </tr>
          ))}
        </tbody></table>
      )}
      <div className={styles.foot}>Last run log: {f.log.join(' · ')}</div>
    </div>
  );
}
