import { useEffect, useState } from 'react';
import { getStudy } from '../data/backtests';
import styles from './MomentumPaper.module.css';
import HoldingsCharts from '../components/HoldingsCharts/HoldingsCharts';
import type { HoldingsRecord } from '../api/types';

/* BlueSky paper book — deliberately shares the Momentum page's stylesheet and layout
   (headline book-summary, KPI language, cards, tables) so the two sleeves read as one
   family. Structure mirrors MomentumPaper: summary → holdings → gate/pending → curve →
   closed trades. */

type Pos = {
  symbol: string; is_cash?: boolean; qty: number; buy: number | null; entry_date: string | null; pivot: number | null;
  day_move_pct?: number | null;
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
          updated {fmtD(f.updated)}{' '}
          <button style={{ marginLeft: 10, padding: '3px 10px', borderRadius: 6, fontSize: 12,
                           border: '1px solid var(--hairline, #888)', cursor: 'pointer',
                           background: 'var(--surface, transparent)', color: 'inherit' }}
            onClick={() => fetch('/api/sleeves/openalpha/run', { method: 'POST', credentials: 'include' })
              .then((r) => r.json()).then((d) => alert(d.mode || 'started'))
              .catch(() => alert('run API not loaded yet (arrives with the next service reload)'))}
            title="Initiate the engine: full nightly cycle after 17:50 IST, display refresh during market hours">
            Run cycle now</button>
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

type RealPos = {
  symbol: string; qty: number; buy: number; entry_date: string; stop: number; src: string;
  ltp: number | null; days: number; weight: number; day_move_pct: number | null;
  value: number; pnl: number; pnl_pct: number | null; trail: number | null;
  to_stop_pct: number | null; to_trail_pct: number | null;
};
type RealFeed = { updated: string; positions: RealPos[]; invested: number; value: number;
  cash: number; nav: number; pnl: number; realized: number; pnl_pct: number;
  inception: string; navcurve: { d: string; nav: number }[]; note: string; trades: any[] };

function CurveCard({ nc }: { nc: { d: string; nav: number }[] }) {
  if (!nc || nc.length < 2)
    return (
      <div className={styles.card}>
        <div className={styles.cardTitle}>Equity curve</div>
        <p className={styles.note}>The curve begins at tomorrow&apos;s close — the book is one day old.
        Each post-close mark appends a point.</p>
      </div>
    );
  const vals = nc.map((x) => x.nav);
  const min = Math.min(...vals), max = Math.max(...vals), span = max - min || 1;
  const W = 720, H = 160;
  const pts = nc.map((x, k) => `${(k / (nc.length - 1)) * W},${H - 14 - ((x.nav - min) / span) * (H - 28)}`).join(' ');
  return (
    <div className={styles.card}>
      <div className={styles.cardTitle}>Equity curve — since {fmtD(nc[0].d)}</div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }} role="img"
           aria-label="real book equity curve">
        <polyline points={pts} fill="none" stroke="var(--accent-pos,#0F6E56)" strokeWidth="2" />
      </svg>
    </div>
  );
}

export default function BlueskyPaper() {
  const [r, setR] = useState<RealFeed | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    const load = () =>
      fetch('/app/oa_real.json?t=' + Date.now())
        .then((x) => (x.ok ? x.json() : Promise.reject(new Error(String(x.status)))))
        .then(setR)
        .catch((e) => setErr(String(e)));
    load();
    const id = setInterval(load, 30000);   // marks refresh every minute in market hours
    return () => clearInterval(id);
  }, []);
  if (err)
    return <div className={styles.root}><div className={styles.loading}>Real-book feed unavailable ({err}).</div></div>;
  if (!r) return <div className={styles.root}><div className={styles.loading}>Loading book…</div></div>;

  const gain = r.pnl + r.realized;
  const tone = (v: number) => (v > 0 ? 'var(--accent-pos,#0F6E56)' : v < 0 ? 'var(--accent-neg,#A32D2D)' : 'var(--ink,#1B1B1A)');
  const segs = [
    { k: 'Stocks', v: r.value, c: '#2563EB' },
    { k: 'Cash', v: r.cash, c: 'var(--ink-faint,#B4B2A9)' },
  ].filter((x) => x.v > 0);
  const total = segs.reduce((a, x) => a + x.v, 0) || 1;
  const pnlRows = [
    { k: 'Unrealised', v: r.pnl, hint: 'open positions vs actual fills' },
    { k: 'Realised (net)', v: r.realized, hint: 'closed trades, after costs' },
  ];

  return (
    <div className={styles.root}>
      <BacktestEvidence />
      <div className={styles.headerRow}>
        <div>
          <h1 className={styles.title}>Open Alpha — REAL Book</h1>
          <p className={styles.sub}>
            <b>LIVE MONEY (RA6610)</b> · ATH-close breakouts, top-16 by RS of 04-Sep&apos;s 21 candidates ·
            −8% close stop · 15-SMA close trail (entry-day exempt) · exits manual-assisted:
            the 15:18 IST checker alerts the exact sell order — no automated selling yet.
          </p>
        </div>
      </div>

      <div className={styles.bookSummary}>
        <div className={styles.sumMain}>
          <div className={styles.sumLabel}>Current value</div>
          <div className={styles.sumHero}>{inr(r.nav)}</div>
          <div className={styles.sumSub}>
            on <b>{inr(r.invested)}</b> invested{' '}
            <span style={{ color: tone(gain), fontWeight: 700 }}>
              {gain >= 0 ? '+' : '−'}{inr(Math.abs(gain))} · {pct(r.pnl_pct)}
            </span>
            {' '}· since {r.inception}
          </div>
          <div className={styles.sumSub}>updated {fmtD(r.updated)} {r.updated?.slice(11, 16)} IST
            · marks every 10 min market hours</div>
          <div className={styles.barWrap} role="img"
               aria-label={segs.map((x) => `${x.k} ${Math.round((x.v / total) * 100)}%`).join(', ')}>
            {segs.map((x) => (
              <div key={x.k} className={styles.barSeg}
                   style={{ width: `${(x.v / total) * 100}%`, background: x.c }} />
            ))}
          </div>
          <div className={styles.legend}>
            {segs.map((x) => (
              <span key={x.k} className={styles.legendItem}>
                <i className={styles.swatch} style={{ background: x.c }} />
                {x.k} <b>{lakh(x.v)}</b>
                <span className={styles.legendPct}>{((x.v / total) * 100).toFixed(0)}%</span>
              </span>
            ))}
          </div>
          <div className={styles.sumStatus}>
            <span><b>{r.positions.length}</b> holdings</span>
            <span><b>{((r.value / (r.nav || 1)) * 100).toFixed(0)}%</b> deployed</span>
            <span>no market gate</span>
            <span>exit check <b>15:18</b> IST daily</span>
          </div>
        </div>
        <div className={styles.sumPnl}>
          <div className={styles.sumLabel}>Profit &amp; loss</div>
          {pnlRows.map((x) => (
            <div key={x.k} className={styles.pnlRow} title={x.hint}>
              <span>{x.k}</span>
              <b style={{ color: tone(x.v) }}>{x.v >= 0 ? '+' : '−'}{inr(Math.abs(x.v))}</b>
            </div>
          ))}
          <div className={`${styles.pnlRow} ${styles.pnlTotal}`}>
            <span>Total return</span>
            <b style={{ color: tone(gain) }}>{gain >= 0 ? '+' : '−'}{inr(Math.abs(gain))} · {pct(r.pnl_pct)}</b>
          </div>
        </div>
      </div>

      <div className={styles.card}>
        <div className={styles.cardTitle}>Holdings — real positions</div>
        <div style={{ overflowX: 'auto' }}>
        <table className={styles.table}>
          <thead><tr>
            <th>Holding</th><th>Entry</th><th>Buy ₹</th><th>Now ₹</th><th>Value</th><th>Today</th>
            <th>P&L ₹</th><th>P&L %</th><th>Days</th>
            <th>Stop −8%</th><th>To stop</th><th>15-SMA trail</th><th>To trail</th>
          </tr></thead>
          <tbody>
            {r.positions.map((p) => (
              <tr key={p.symbol}>
                <td className={styles.sym}>{p.symbol}
                  <span className={styles.muted} style={{ fontSize: 11, marginLeft: 6 }}>{p.weight}%</span></td>
                <td className={styles.muted}>{fmtD(p.entry_date)}</td>
                <td>{p.buy}</td><td>{p.ltp ?? '—'}</td>
                <td>{lakh(p.value)}</td>
                <td className={(p.day_move_pct ?? 0) >= 0 ? styles.pos : styles.neg}
                    style={pnlTint(p.day_move_pct == null ? null : p.day_move_pct * 3)}>
                  {p.day_move_pct == null ? '—' : (p.day_move_pct >= 0 ? '+' : '') + p.day_move_pct + '%'}</td>
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
                    title="distance above the 15-SMA trail — the usual exit">
                  {p.to_trail_pct == null ? '—' : (p.to_trail_pct >= 0 ? '+' : '') + p.to_trail_pct + '%'}</td>
              </tr>
            ))}
          </tbody>
          <tfoot>
            <tr style={{ borderTop: '2px solid var(--hairline,rgba(0,0,0,0.14))', fontWeight: 700 }}>
              <td>TOTAL ({r.positions.length} stocks · {((r.value / (r.nav || 1)) * 100).toFixed(0)}% deployed)</td>
              <td /><td /><td />
              <td>{lakh(r.value)}</td>
              <td />
              <td className={r.pnl >= 0 ? styles.pos : styles.neg}>
                {r.pnl >= 0 ? '+' : ''}{inr(r.pnl)}</td>
              <td className={r.pnl >= 0 ? styles.pos : styles.neg}>{pct(r.pnl_pct)}</td>
              <td colSpan={5} />
            </tr>
          </tfoot>
        </table>
        </div>
      </div>

      <CurveCard nc={r.navcurve} />

      {r.positions.length > 0 && (
        <div className={styles.chartsSection}>
          <div className={styles.cardTitle}>
            Charts — live positions
            <span style={{ fontSize: 11.5, fontWeight: 400, color: 'var(--ink-muted,#888)', marginLeft: 8 }}>
              scroll to zoom · drag to pan · red dashed line = 15-SMA trail floored at the −8% stop (the exit rule)
            </span>
          </div>
          <HoldingsCharts
            ohlcUrl="/static/oa_real_ohlc.json"
            stopLabel="15-SMA trail · floored at the −8% stop (the exit rule)"
            holdings={r.positions.map((p) => ({
              tradingsymbol: p.symbol,
              qty: p.qty,
              avg_price: p.buy,
              ltp: p.ltp ?? 0,
              prev_close: p.buy,
              day_pct: p.day_move_pct ?? 0,
              day_pnl_inr: 0,
              invested: (p.value ?? 0) - (p.pnl ?? 0),
              current: p.value ?? 0,
              total_pnl_inr: p.pnl ?? 0,
              total_pnl_pct: p.pnl_pct ?? 0,
            })) as HoldingsRecord[]}
          />
        </div>
      )}

      <div className={styles.card}>
        <div className={styles.cardTitle}>Closed trades</div>
        {(!r.trades || r.trades.length === 0)
          ? <p className={styles.note}>None yet — exits land here when the 15:18 checker fires and a sell executes.</p>
          : <p className={styles.note}>{r.trades.length} closed.</p>}
        <p className={styles.note}>{r.note}</p>
        <p className={styles.note}>
          Paper model retired from this page (Arun, 04-Sep-2026); its engine runs headless as the
          reference model until Sleeves/dividends are rewired to this book.
          Study: <a href="/app/backtest/bluesky-ath-breakout-research142">bluesky-ath-breakout-research142</a>.
        </p>
      </div>
    </div>
  );
}
