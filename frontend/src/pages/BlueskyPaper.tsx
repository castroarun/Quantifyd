import { useEffect, useState } from 'react';
import { getStudy } from '../data/backtests';
import styles from './MomentumPaper.module.css';

/* BlueSky paper book — deliberately shares the Momentum page's stylesheet and layout
   (headline book-summary, KPI language, cards, tables) so the two sleeves read as one
   family. Structure mirrors MomentumPaper: summary → holdings → gate/pending → curve →
   closed trades. */

type Pos = {
  symbol: string; is_cash?: boolean; qty: number; buy: number | null; entry_date: string | null; pivot: number | null;
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
  unrealized: number; ret_pct: number; cagr_pct: number | null; max_dd_pct: number;
  gate_weak: boolean; gate_nb: number | null; gate_sma: number | null; gate_gap_pct: number | null;
  positions: Pos[]; pending: Pending[]; trades: Trade[]; n_trades: number;
  n_live_trades: number; interest_earned?: number; cash_yield_pct?: number;
  swept_value?: number; sweep_units?: number;
  win_pct: number | null; nav_curve: NavPt[]; spec: string;
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
};
const SEG_COLORS = ['#1f9d55', '#2f7fd1', '#d98a00', '#9057c9', '#c94f7c', '#3fa9a5', '#8a8f3d', '#b3593a'];
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

function BacktestEvidence() {
  const study = getStudy('bluesky-ath-breakout-research142');
  const [open, setOpen] = useState(false);
  const [expanded, setExpanded] = useState(false);
  if (!study || !study.results || !study.results.metrics) return null;
  const m = study.results.metrics;
  return (
    <div className={styles.evidence}>
      <div className={styles.evidenceHead}>
        <span className={styles.evidenceTag}>Backtest evidence</span>
        <span className={styles.evidenceSub}>
          {study.title} · {study.status}
          {study.date ? ` · ${study.date}` : ''} · this is the study the paper book implements, not
          live performance
        </span>
        <a className={styles.studyLink} href="/app/backtest/bluesky-ath-breakout-research142">Study</a>
        <a className={styles.studyLink} href="/app/sleeves">Sleeves 50-50</a>
        <a className={styles.studyLink} href="/app/strategies#bluesky-paper">Register</a>
        <button className={styles.evidenceBtn} onClick={() => setExpanded(!expanded)}>
          {expanded ? 'Hide' : 'Show numbers'}
        </button>
        {expanded && (
        <button className={styles.evidenceBtn} onClick={() => setOpen(!open)}>
          {open ? 'Hide caveats' : 'Caveats'}
        </button>
        )}
      </div>
      {expanded && (<>
      <div className={styles.evidenceGrid}>
        {m.map((y) => (
          <div key={y.label} className={styles.evidenceCell} title={y.hint || ''}>
            <div className={styles.evidenceVal}
                 style={{ color: y.tone === 'pos' ? 'var(--accent-pos,#0F6E56)'
                                : y.tone === 'neg' ? 'var(--accent-neg,#A32D2D)'
                                : 'var(--ink,#1B1B1A)' }}>{y.value}</div>
            <div className={styles.evidenceLab}>{y.label}</div>
            {y.hint && <div className={styles.evidenceHint}>{y.hint}</div>}
          </div>
        ))}
      </div>
      {open && (
        <div className={styles.evidenceCaveat}>
          <b>What this number is not.</b>
          <ul>
            {(study.caveats || []).map((c, i) => <li key={i}>{c}</li>)}
          </ul>
        </div>
      )}
      </>)}
    </div>
  );
}

function EquityCurve({ data }: { data: NavPt[] }) {
  if (data.length < 2)
    return <div className={styles.chartEmpty}>Equity curve builds as the soak runs.</div>;
  const W = 760, H = 220, P = 8;
  const n0 = data[0].nav;
  const b0 = data.find((d) => d.bench != null)?.bench ?? 1;
  const navN = data.map((d) => d.nav / n0);
  const benN = data.map((d) => (d.bench == null ? null : d.bench / (b0 as number)));
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

function BookSummary({ f }: { f: Feed }) {
  const realized = f.trades.reduce((a, t) => a + (t.net_pnl ?? 0), 0);
  const stocks = f.positions.filter((p) => !p.is_cash);
  const cashW = Math.max(0, 100 - f.invested_pct);
  return (
    <div className={styles.bookSummary}>
      <div className={styles.sumMain}>
        <div className={styles.sumLabel}>Book value</div>
        <div className={styles.sumHero}>{inr(f.nav)}</div>
        <div className={styles.sumSub}>
          CAGR {pct(f.cagr_pct)} (incl. backfill) · max drawdown {pct(f.max_dd_pct)} ·
          updated {fmtD(f.updated)}
        </div>
        <div className={styles.barWrap}>
          <div className={styles.barSeg}
               style={{ width: `${f.invested_pct}%`, background: 'var(--accent-pos, #1f9d55)' }}
               title={`stocks ${f.invested_pct}%`} />
          <div className={styles.barSeg}
               style={{ width: `${cashW}%`, background: 'var(--ink-faint, #b7b7b0)' }}
               title={`cash ${cashW.toFixed(0)}%`} />
        </div>
        <div className={styles.legend}>
          <span className={styles.legendItem}>
            <span className={styles.swatch} style={{ background: 'var(--accent-pos, #1f9d55)' }} />
            stocks ({stocks.length}) <span className={styles.legendPct}>{f.invested_pct}%</span>
          </span>
          <span className={styles.legendItem}>
            <span className={styles.swatch} style={{ background: 'var(--ink-faint, #b7b7b0)' }} />
            CASHIETF sweep + cash <span className={styles.legendPct}>{cashW.toFixed(0)}%</span>
          </span>
        </div>
        <div className={styles.sumStatus}>
          <span>{stocks.length}/8 slots held</span>
          <span>{f.pending.length} pending buy-stops</span>
          <span>{f.n_trades} closed · {f.win_pct == null ? '—' : f.win_pct + '% win'}</span>
          <span>live trades: {f.n_live_trades}</span>
        </div>
      </div>
      <div className={styles.sumPnl}>
        <div className={styles.sumLabel}>P&L</div>
        <div className={styles.pnlRow}><span>Unrealised (open)</span>
          <b className={f.unrealized >= 0 ? styles.pos : styles.neg}>
            {f.unrealized >= 0 ? '+' : ''}{inr(f.unrealized)}</b></div>
        <div className={styles.pnlRow}><span>Realised (closed, net)</span>
          <b className={realized >= 0 ? styles.pos : styles.neg}>
            {realized >= 0 ? '+' : ''}{inr(realized)}</b></div>
        <div className={styles.pnlRow}><span>Sweep interest earned</span>
          <b className={styles.pos}>+{inr(f.interest_earned ?? 0)}</b></div>
        <div className={styles.pnlRow}><span>Cash (in liquid sweep)</span><b>{lakh(f.cash)}</b></div>
        <div className={`${styles.pnlRow} ${styles.pnlTotal}`}><span>NAV</span><b>{inr(f.nav)}</b></div>
      </div>
    </div>
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
    return <div className={styles.root}><div className={styles.loading}>
      No feed yet ({err}) — first snapshot lands after the 18:40 IST run.</div></div>;
  if (!f) return <div className={styles.root}><div className={styles.loading}>Loading book…</div></div>;
  const totPnl = f.positions.reduce((a, p) => a + (p.pnl || 0), 0);
  const totVal = f.positions.reduce((a, p) => a + (p.value || 0), 0);
  const gateOn = !f.gate_weak;
  return (
    <div className={styles.root}>
      <BacktestEvidence />
      <div className={styles.headerRow}>
        <div>
          <h1 className={styles.title}>Open Alpha — Paper Book</h1>
          <p className={styles.sub}>
            <b>Formerly BlueSky</b> · all-time-high close breakouts in RS≥70, ₹5cr/day-liquid NSE names ·
            buy-stop at the pivot next day · −8% stop · 20-SMA trail ·
            sized level with the Momentum sleeve · live trades so far: {f.n_live_trades}
          </p>
        </div>
        <div className={`${styles.gateBadge} ${gateOn ? styles.on : styles.off}`}>
          <span className={styles.dot} />
          {gateOn ? 'RISK-ON — gate open' : 'GATE WEAK — no new entries'}
        </div>
      </div>

      <BookSummary f={f} />

      <div className={styles.card}>
        <div className={styles.cardTitle}>Holdings</div>
        {f.positions.length === 0 ? <div className={styles.loading}>none — in cash</div> : (
        <table className={styles.table}>
          <thead><tr>
            <th>Holding</th><th>Wt</th><th>Entry</th><th>Entry ₹</th><th>Now ₹</th><th>Value</th>
            <th>P&L ₹</th><th>P&L %</th><th>Days</th><th>Stop −8%</th><th>To stop</th>
            <th>20-SMA trail</th><th>To trail</th>
          </tr></thead>
          <tbody>
            {f.positions.map((p) => p.is_cash ? (
              <tr key="cashietf" className={styles.cashRow}>
                <td className={styles.sym}>{p.symbol}<span className={styles.cashTag}>liquid fund</span></td>
                <td>{p.weight == null ? '—' : p.weight + '%'}</td>
                <td className={styles.muted}>—</td><td>—</td><td>{p.ltp ?? '—'}</td><td>{lakh(p.value)}</td>
                <td className={styles.pos} title="sweep interest earned to date">+{inr(p.pnl ?? 0)}</td>
                <td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td>
              </tr>
            ) : (
              <tr key={p.symbol + p.entry_date}>
                <td className={styles.sym}>{p.symbol}</td>
                <td>{p.weight == null ? '—' : p.weight + '%'}</td>
                <td className={styles.muted}>{fmtD(p.entry_date)}</td>
                <td>{p.buy ?? '—'}</td><td>{p.ltp ?? '—'}</td><td>{lakh(p.value)}</td>
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
          <tfoot>
            <tr style={{ borderTop: '2px solid var(--hairline,rgba(0,0,0,0.14))', fontWeight: 700 }}>
              <td>TOTAL ({f.positions.filter((p) => !p.is_cash).length} stocks + sweep)</td>
              <td>{f.invested_pct}%</td><td /><td /><td />
              <td>{lakh(totVal)}</td>
              <td className={totPnl >= 0 ? styles.pos : styles.neg}>{totPnl >= 0 ? '+' : ''}{inr(totPnl)}</td>
              <td colSpan={6} />
            </tr>
          </tfoot>
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
              <b className={gateOn ? styles.pos : styles.neg}>{gateOn ? 'RISK-ON' : 'WEAK'}</b></div>
          </div>
          <p className={styles.note}>Below the 200-DMA no NEW breakouts are taken; open positions keep their −8% stop and 20-SMA trail.</p>
        </div>
        <div className={styles.card}>
          <div className={styles.cardTitle}>Pending buy-stops (tomorrow)</div>
          {f.pending.length === 0 ? <div className={styles.loading}>none</div> : (
          <table className={styles.table}>
            <thead><tr><th>Symbol</th><th>Pivot (buy-stop)</th><th>RS</th><th>Signalled</th></tr></thead>
            <tbody>
              {f.pending.map((p) => (
                <tr key={p.symbol}><td className={styles.sym}>{p.symbol}</td>
                  <td>{p.pivot}</td><td>{p.rs}</td><td className={styles.muted}>{fmtD(p.signal_date)}</td></tr>
              ))}
            </tbody>
          </table>)}
        </div>
      </div>

      <div className={styles.card}>
        <div className={styles.cardTitle}>Equity vs NIFTYBEES (both rebased at book start · dashed = index)</div>
        <EquityCurve data={f.nav_curve} />
      </div>

      <div className={styles.card}>
        <div className={styles.cardTitle}>Closed trades (latest first)</div>
        {f.trades.length === 0 ? <div className={styles.loading}>none yet</div> : (
        <table className={styles.table}>
          <thead><tr>
            <th>Symbol</th><th>Entry</th><th>Exit</th><th>Buy ₹</th><th>Sell ₹</th><th>Qty</th>
            <th>Net ₹</th><th>Return</th><th>Why exited</th><th>Source</th>
          </tr></thead>
          <tbody>
            {[...f.trades].reverse().map((t, i) => (
              <tr key={i}>
                <td className={styles.sym}>{t.symbol}</td>
                <td className={styles.muted}>{fmtD(t.entry_date)}</td>
                <td className={styles.muted}>{fmtD(t.exit_date)}</td>
                <td>{t.buy}</td><td>{t.sell}</td><td>{t.qty ?? '—'}</td>
                <td className={(t.net_pnl ?? 0) >= 0 ? styles.pos : styles.neg}>
                  {t.net_pnl == null ? '—' : (t.net_pnl >= 0 ? '+' : '') + inr(t.net_pnl)}</td>
                <td className={t.ret_pct >= 0 ? styles.pos : styles.neg} style={pnlTint(t.ret_pct)}>
                  {pct(t.ret_pct)}</td>
                <td><span className={styles.reason}>{reasonLabel[t.reason] ?? t.reason}</span></td>
                <td className={styles.muted}>{t.src === 'live' ? 'LIVE' : 'backfill'}</td>
              </tr>
            ))}
          </tbody>
        </table>)}
      </div>

      {f.provenance && <p className={styles.note}>{f.provenance}</p>}
      <p className={styles.note}>Last run: {f.log.join(' · ')}</p>
    </div>
  );
}
