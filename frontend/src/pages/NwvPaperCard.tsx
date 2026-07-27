import { useEffect, useState } from 'react';
import { apiGet, apiPost } from '../api/client';
import styles from './Nwv.module.css';

type Leg = {
  side: string; tsym: string; strike: number; qty: number; entry: number;
  status: string; exit?: number; closed?: string;
};
type Watch = {
  s1: number; r2: number; spot?: number | null;
  dist_s1_pts?: number | null; dist_r2_pts?: number | null;
  put_side_live: boolean; call_side_live: boolean;
  pt_rs: number; stop_rs: number; last_pivot_check?: string | null; rule: string;
};
type Cycle = {
  entry_date: string; expiry: string; view: string; spot0: number;
  credit0: number; m2m_rs?: number; last_check?: string; status: string;
  strikes: Record<string, number>; legs: Leg[]; events: string[];
};
type Hist = {
  week: string; view?: string; expiry?: string; credit0?: number;
  reason: string; net_rs: number; net_pts?: number;
};
type PaperState = {
  spec: string; killed: boolean; watch: Watch | null; cycle: Cycle | null;
  history: Hist[]; totals: { weeks: number; net_rs: number; wins: number };
};

function rs(v?: number | null): string {
  if (v == null) return '—';
  const sign = v < 0 ? '−' : '';
  return `${sign}₹${Math.abs(Math.round(v)).toLocaleString('en-IN')}`;
}

export default function NwvPaperCard() {
  const [st, setSt] = useState<PaperState | null>(null);
  const [offline, setOffline] = useState(false);

  const load = async () => {
    try {
      setSt(await apiGet<PaperState>('/api/nwv-trade/state'));
      setOffline(false);
    } catch {
      setOffline(true);
    }
  };
  useEffect(() => {
    void load();
    const t = setInterval(() => void load(), 30000);
    return () => clearInterval(t);
  }, []);

  const kill = async () => {
    try { await apiPost('/api/nwv-trade/kill-switch', {}); void load(); } catch { /* noop */ }
  };

  const c = st?.cycle;
  const w = st?.watch;
  const liveLegs = c?.legs?.filter(l => l.status === 'live') ?? [];

  return (
    <div className={styles.card} style={{ marginBottom: 18 }}>
      <div className={styles.pbHead}>
        <div className={styles.cardTitle}>
          Phase 1 — Jade-Lizard Paper Book <span className={styles.muted}>· research/94</span>
        </div>
        <div className={styles.pbChips}>
          <span className={`${styles.pbChip} ${styles.pbChipPaper}`}>PAPER · 10 lots</span>
          {st?.killed
            ? <span className={`${styles.pbChip} ${styles.pbChipKilled}`}>KILLED</span>
            : c
              ? <span className={`${styles.pbChip} ${styles.pbChipOpen}`}>{c.status}</span>
              : <span className={styles.pbChip}>FLAT</span>}
          <button className={styles.pbKill} onClick={kill}>
            {st?.killed ? 'un-kill' : 'kill'}
          </button>
        </div>
      </div>

      {offline && (
        <div className={styles.mute}>
          Book registered — API activates at the next backend restart (pre-open 09:00).
          Today it runs via the standalone runner; state refreshes once the route is live.
        </div>
      )}

      {!offline && !c && (
        <div className={styles.mute}>
          No open cycle. Next entry: Monday 09:50 IST on any non-ignore view
          (short PE @S1 +550 wing / short CE @R2 +200 wing, next-week expiry).
        </div>
      )}

      {c && (
        <>
          <div className={styles.pbStats}>
            <div><span>Week / view</span><b>{c.entry_date} · {c.view}</b></div>
            <div><span>Expiry</span><b>{c.expiry}</b></div>
            <div><span>Net credit</span><b>{c.credit0} pts ({rs(c.credit0 * 650)})</b></div>
            <div><span>MTM now</span>
              <b className={(c.m2m_rs ?? 0) >= 0 ? styles.pbPos : styles.pbNeg}>
                {rs(c.m2m_rs)}
              </b>
            </div>
            {w && <div><span>Take-profit / stop</span><b>{rs(w.pt_rs)} / {rs(w.stop_rs)}</b></div>}
          </div>

          {w && (
            <>
              <div className={styles.pbWatch}>
                <div className={`${styles.pbLevel} ${!w.put_side_live ? styles.pbLevelDone : ''}`}>
                  <span>EXIT PUTS BELOW</span>
                  <b>S1 {w.s1.toFixed(2)}</b>
                  <em>{w.put_side_live
                    ? (w.dist_s1_pts != null ? `${w.dist_s1_pts > 0 ? '+' : ''}${w.dist_s1_pts} pts away` : '—')
                    : 'put side exited'}</em>
                </div>
                <div className={styles.pbLevel}>
                  <span>SPOT (last)</span>
                  <b>{w.spot != null ? w.spot.toFixed(2) : '—'}</b>
                  <em>{w.last_pivot_check ? `pivot check ${w.last_pivot_check.slice(11)}` : 'first check 09:45'}</em>
                </div>
                <div className={`${styles.pbLevel} ${!w.call_side_live ? styles.pbLevelDone : ''}`}>
                  <span>EXIT CALLS ABOVE</span>
                  <b>R2 {w.r2.toFixed(2)}</b>
                  <em>{w.call_side_live
                    ? (w.dist_r2_pts != null ? `${w.dist_r2_pts > 0 ? '+' : ''}${w.dist_r2_pts} pts away` : '—')
                    : 'call side exited'}</em>
                </div>
              </div>
              <div className={styles.pbRule}>{w.rule}</div>
            </>
          )}

          <table className={styles.pbTable}>
            <thead>
              <tr><th>Leg</th><th>Qty</th><th>Entry</th><th>Status</th><th>Exit</th></tr>
            </thead>
            <tbody>
              {c.legs.map((l, i) => (
                <tr key={i} className={l.status !== 'live' ? styles.pbLegClosed : ''}>
                  <td>{l.tsym}</td>
                  <td>{l.qty > 0 ? `+${l.qty}` : l.qty}</td>
                  <td>{l.entry.toFixed(2)}</td>
                  <td>{l.status}</td>
                  <td>{l.exit != null ? l.exit.toFixed(2) : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {liveLegs.length === 0 && <div className={styles.mute}>All legs closed.</div>}
        </>
      )}

      {!offline && st && (
        <div className={styles.pbFoot}>
          <span>
            Book: {st.totals.weeks} wk · {st.totals.wins} wins · net {rs(st.totals.net_rs)}
          </span>
          <span className={styles.muted}>{st.spec}</span>
        </div>
      )}

      {!offline && (st?.history?.length ?? 0) > 0 && (
        <table className={styles.pbTable}>
          <thead>
            <tr><th>Week</th><th>View</th><th>Credit</th><th>Exit</th><th>Net</th></tr>
          </thead>
          <tbody>
            {st!.history.slice(-8).reverse().map((h, i) => (
              <tr key={i}>
                <td>{h.week}</td>
                <td>{h.view ?? '—'}</td>
                <td>{h.credit0 != null ? `${h.credit0} pts` : '—'}</td>
                <td>{h.reason}</td>
                <td className={h.net_rs >= 0 ? styles.pbPos : styles.pbNeg}>{rs(h.net_rs)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
