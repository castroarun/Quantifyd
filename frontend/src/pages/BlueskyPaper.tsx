import { useEffect, useState } from 'react';
import { getStudy } from '../data/backtests';
import styles from './MomentumPaper.module.css';
import HoldingsCharts from '../components/HoldingsCharts/HoldingsCharts';
import BookPanel, { pnlBreakdown, todayPnl } from '../components/BookPanel/BookPanel';
import LiveTick, { Tick } from '../components/LiveTick/LiveTick';
import type { HoldingsRecord } from '../api/types';

/* OPEN ALPHA — the real-money book (/app/bluesky-paper).

   Shares MomentumPaper's stylesheet and section order so the three books read as one
   family, and keeps its own loader: a raw fetch of the cron-baked static feed, which is
   why this page renders in ~2ms while True North's API route costs ~0.7s warm (it does
   a live Kite call plus a large pandas pivot on the request path).

   Money controls are deliberately NOT here. Every deposit and withdrawal lives on the
   Capital Desk so there is one place to look and one path to audit; this page links to
   it and shows the resulting balances.

   The retired paper model's types, EquityCurve and BookSummary were deleted on
   06-Sep-2026 — none had been rendered since the page was rebuilt around the real book. */

type LedgerTrade = {
  symbol: string; entry_date: string; exit_date: string; pivot: number | null;
  entry: number; exit: number; qty: number; day_open: number; day_high: number;
  day_close: number; held_days: number; reason: string; ret_pct: number;
  pnl: number; cum_pnl: number; dd_pct: number | null;
};
type LedgerArm = {
  key: string; label: string; trail: number; entry_time: string; note: string;
  cagr: number; dd: number; n: number; win: number; trades: LedgerTrade[];
};
type Ledger = { generated: string; window: string[]; seed: number; arms: LedgerArm[] };

/* The trade ledger, collapsed by default and fetched only when opened: 305 KB has no
   business loading on first paint. Two arms from the same signal and the same exits,
   differing only in the entry, so the pivot / price-paid / day-close columns can be read
   across and compared without any commentary from me. */
function LedgerCard() {
  const [open, setOpen] = useState(false);
  const [data, setData] = useState<Ledger | null>(null);
  const [arm, setArm] = useState(0);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!open || data) return;
    fetch('/app/oa_ledger.json?t=' + Date.now())
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error('HTTP ' + r.status))))
      .then(setData)
      .catch((e) => setErr(String(e)));
  }, [open, data]);

  const a = data?.arms[arm];
  return (
    <div className={styles.card}>
      <div className={styles.cardTitleTog} onClick={() => setOpen(!open)}
           role="button" tabIndex={0}
           onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setOpen(!open); }}>
        Trade ledger — last two years, for manual verification
        <span className={styles.sumTog}>{open ? '▾' : '▸'}</span>
      </div>
      {!open && (
        <p className={styles.note}>
          Every trade both entry mechanics would have taken between Sep-2024 and Sep-2026,
          with the pivot, the price paid and that day’s open, high and close side by side.
          Expand to load.
        </p>
      )}
      {open && err && <p className={styles.note}>Could not load the ledger: {err}</p>}
      {open && !data && !err && <p className={styles.note}>Loading…</p>}
      {open && a && (
        <>
          <div className={styles.chips}>
            {data!.arms.map((x, i) => (
              <span key={x.key} className={i === arm ? styles.curveChipOn : styles.curveChip}
                    onClick={() => setArm(i)} role="button" tabIndex={0}
                    onKeyDown={(e) => { if (e.key === 'Enter') setArm(i); }}>
                {x.label}
              </span>
            ))}
          </div>
          <p className={styles.note}>
            {a.note} <b>{a.n}</b> trades, CAGR <b>{a.cagr.toFixed(1)}%</b>, max drawdown{' '}
            <b>{a.dd.toFixed(1)}%</b>, win rate <b>{a.win.toFixed(0)}%</b>. Entry timing:{' '}
            {a.entry_time}. The study runs on daily bars, so there is no intraday clock in
            it — the time shown is the one this mechanic implies, not a recorded
            timestamp. One deterministic path (seed {data!.seed}), ranked by relative
            strength. Generated {data!.generated}.
          </p>
          <div style={{ maxHeight: 460, overflow: 'auto' }}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>Symbol</th><th>Entry</th><th>Pivot</th><th>Paid</th>
                  <th>Day open</th><th>Day high</th><th>Day close</th>
                  <th>Exit</th><th>Reason</th><th>Days</th><th>Qty</th>
                  <th>Return</th><th>P&amp;L</th><th>Cumulative</th><th>Book DD</th>
                </tr>
              </thead>
              <tbody>
                {a.trades.map((tr, i) => (
                  <tr key={i}>
                    <td className={styles.sym}>{tr.symbol}</td>
                    <td>{fmtD(tr.entry_date)}</td>
                    <td>{tr.pivot == null ? '—' : tr.pivot.toFixed(2)}</td>
                    <td>{tr.entry.toFixed(2)}</td>
                    <td className={styles.muted}>{tr.day_open.toFixed(2)}</td>
                    <td className={styles.muted}>{tr.day_high.toFixed(2)}</td>
                    <td style={tr.pivot != null && tr.day_close > tr.pivot
                               ? { color: 'var(--pos, #2f9152)' } : {}}>
                      {tr.day_close.toFixed(2)}
                    </td>
                    <td>{fmtD(tr.exit_date)}</td>
                    <td className={styles.muted}>{reasonLabel[tr.reason] ?? tr.reason}</td>
                    <td>{tr.held_days}</td>
                    <td>{tr.qty}</td>
                    <td className={tr.ret_pct >= 0 ? styles.pos : styles.neg}>
                      {pct(tr.ret_pct)}
                    </td>
                    <td className={tr.pnl >= 0 ? styles.pos : styles.neg}>{inr(tr.pnl)}</td>
                    <td>{inr(tr.cum_pnl)}</td>
                    <td className={styles.muted}>
                      {tr.dd_pct == null ? '—' : tr.dd_pct.toFixed(1) + '%'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className={styles.note}>
            Read the <b>Pivot</b>, <b>Paid</b> and <b>Day close</b> columns across. On the
            published mechanic the close finishes above the pivot on every single trade,
            because that mechanic only counts the trades where it did — a median 1.29%
            above the price paid, on all 550. A real order gets the same kind of entry and
            is right about half the time: 231 of 465, median 0.31% below. That difference,
            not the price, is what the published figure rests on.
          </p>
        </>
      )}
    </div>
  );
}

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

type RealPos = {
  symbol: string; qty: number; buy: number; entry_date: string; stop: number; src: string;
  ltp: number | null; days: number; weight: number; day_move_pct: number | null;
  value: number; pnl: number; pnl_pct: number | null; trail: number | null;
  to_stop_pct: number | null; to_trail_pct: number | null;
};
type RealTrade = { symbol?: string; qty?: number; buy?: number; sell?: number;
  entry_date?: string; exit_date?: string; net_pnl?: number; pnl_pct?: number; reason?: string };
type FailedOrder = { ts: string; book: string; symbol: string; qty: number;
  reason: string; detail?: string };
type RealFeed = { updated: string; positions: RealPos[]; invested: number; value: number;
  cash: number; nav: number; pnl: number; realized: number; pnl_pct: number;
  inception: string; navcurve: { d: string; nav: number }[]; note: string;
  trades: RealTrade[];
  /* Added with the capital ledger (05-Sep-2026). Optional because a feed baked by the
     previous build is still valid until the next cron mark — the page must not blank
     out in between, so every read falls back. */
  capital?: number; gain?: number; return_pct?: number; stale?: boolean;
  flows?: { ts: string; kind: string; amount: number; via?: string }[];
  failed_orders?: FailedOrder[];
};

/* CAGR and the worst drawdown, from the book's own nav curve. Returns nulls rather than
   zeros while the curve is too short to mean anything — the book is days old, and a
   confident "0.0%" would read as a measurement rather than an absence. */
function curveStats(nc: { d: string; nav: number }[]) {
  if (!nc || nc.length < 20) return { cagr: null as number | null, dd: null as number | null };
  const yrs = (Date.parse(nc[nc.length - 1].d) - Date.parse(nc[0].d)) / 3.15576e10;
  const cagr = yrs > 0.08 ? (Math.pow(nc[nc.length - 1].nav / nc[0].nav, 1 / yrs) - 1) * 100 : null;
  let peak = nc[0].nav, dd = 0;
  for (const x of nc) { peak = Math.max(peak, x.nav); dd = Math.min(dd, x.nav / peak - 1); }
  return { cagr, dd: dd * 100 };
}


export default function BlueskyPaper() {
  const [showHoldings, setShowHoldings] = useState(true);
  const [r, setR] = useState<RealFeed | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    const load = () =>
      fetch('/app/oa_real.json?t=' + Date.now())
        .then((x) => (x.ok ? x.json() : Promise.reject(new Error(String(x.status)))))
        .then(setR)
        .catch((e) => setErr(String(e)));
    load();
    const id = setInterval(load, 10000);   // marks are baked every minute   // marks refresh every minute in market hours
    return () => clearInterval(id);
  }, []);
  if (err)
    return <div className={styles.root}><div className={styles.loading}>Real-book feed unavailable ({err}).</div></div>;
  if (!r) return <div className={styles.root}><div className={styles.loading}>Loading book…</div></div>;

  /* Gain is measured against CONTRIBUTED CAPITAL, not against the cost of today's
     positions. Before the capital ledger the page divided by `invested`, which is only
     right while capital never moves — and it is about to start moving. */
  const capital = r.capital ?? r.invested;
  const gain = r.gain ?? (r.pnl + r.realized);
  const retPct = r.return_pct ?? (capital ? (gain / capital) * 100 : 0);
  const cs = curveStats(r.navcurve);
  const deployedPct = (r.value / (r.nav || 1)) * 100;
  const tone = (v: number) => (v > 0 ? 'var(--accent-pos,#0F6E56)' : v < 0 ? 'var(--accent-neg,#A32D2D)' : 'var(--ink,#1B1B1A)');
  const segs = [
    { k: 'Stocks', v: r.value, c: '#2563EB' },
    { k: 'Cash', v: r.cash, c: 'var(--ink-faint,#B4B2A9)' },
  ].filter((x) => x.v > 0);
  const total = segs.reduce((a, x) => a + x.v, 0) || 1;
  const pnlRows = pnlBreakdown({
    gain, unrealised: r.pnl, realised: r.realized, invested: r.invested ?? 0,
  });

  return (
    <div className={styles.root}>
      <BacktestEvidence />
      <div className={styles.headerRow}>
        <div>
          <h1 className={styles.title}>
            Open Alpha — REAL Book
            <span className={`${styles.gateBadge} ${styles.on}`} style={{ marginLeft: 10 }}>
              <i className={styles.dot} />LIVE · real money
            </span>
            {r.stale && (
              <span className={styles.gateBadge} style={{ marginLeft: 8 }}
                    title="the feed was rebuilt from state without live quotes">
                prices as of last mark
              </span>
            )}
          </h1>
          <p className={styles.sub}>
            <b>LIVE MONEY (RA6610)</b> · ATH-close breakouts, top-16 by RS of 04-Sep&apos;s 21 candidates ·
            −8% close stop · 15-SMA close trail (entry-day exempt) · exits manual-assisted:
            the 15:18 IST checker alerts the exact sell order — no automated selling yet.
          </p>
        </div>
      </div>

      {(r.failed_orders ?? []).length > 0 && (
        <div style={{
          border: '1px solid var(--accent-neg,#A32D2D)', borderLeftWidth: 4,
          borderRadius: 7, padding: '11px 14px', marginBottom: 14,
          background: 'var(--surface,#fff)',
        }}>
          <b style={{ color: 'var(--accent-neg,#A32D2D)' }}>
            {(r.failed_orders ?? []).length} order
            {(r.failed_orders ?? []).length === 1 ? '' : 's'} did NOT go through
          </b>
          <table className={styles.table} style={{ marginTop: 6 }}>
            <tbody>
              {(r.failed_orders ?? []).map((f, i) => (
                <tr key={i}>
                  <td className={styles.sym}>{f.symbol}</td>
                  <td>x{f.qty}</td>
                  <td className={styles.neg}>{f.reason}</td>
                  <td className={styles.muted}>{f.detail}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className={styles.note}>
            These trades are not in the account, and the book does not hold them. Email and
            WhatsApp were sent when this happened; the banner clears on the next clean run.
          </p>
        </div>
      )}
      <div className={styles.studyBar}>
        <span className={styles.studyBarLabel}>Money in and out</span>
        <a className={styles.studyLink} href="/app/capital">Capital Desk →</a>
        <a className={styles.studyLink} href="/app/strategies#bluesky-paper">Register</a>
      </div>

      <BookPanel
        hero={<Tick v={r.nav} render={(n) => inr(n ?? 0)} />}
        gain={gain}
        returnPct={retPct}
        capital={capital}
        capitalWord="of capital"
        inception={r.inception}
        extraSub={<>
          {cs.cagr != null && <> · CAGR <b>{cs.cagr.toFixed(1)}%</b></>}
          {cs.dd != null && <> · worst drawdown <b>{cs.dd.toFixed(1)}%</b></>}
        </>}
        updated={r.updated}
        today={todayPnl(r.positions as never)}
        tickLabel="marks"
        segs={segs}
        pnl={pnlRows}
        bookId="oa-real"
        curveUrl="/api/books/oa-real/benchmarks"
        curveLabel="Open Alpha"
        storageKey="oa-real"
        status={<>
          <span><b>{r.positions.length}</b> / 16 slots</span>
          <span><b>{((r.value / (r.nav || 1)) * 100).toFixed(0)}%</b> deployed</span>
          <span>no market gate</span>
          <span>exit check <b>15:18</b> IST daily</span>
          {(r.failed_orders ?? []).length > 0
            ? <span className={styles.neg}>
                <b>{(r.failed_orders ?? []).length}</b> order{(r.failed_orders ?? []).length === 1 ? '' : 's'} unfilled
              </span>
            : <span><b>all</b> orders filled</span>}
        </>}
      />

      <div className={styles.card}>
        <button type="button" className={`${styles.cardTitle} ${styles.cardTitleTog}`}
                aria-expanded={showHoldings} onClick={() => setShowHoldings((v) => !v)}>
          Holdings — real positions
          <span className={styles.cardCount}>
            {r.positions.length} position{r.positions.length === 1 ? '' : 's'}
          </span>
          <span className={styles.caret} aria-hidden="true">{'\u25be'}</span>
        </button>
        {showHoldings && (
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
                <td>{p.buy}</td>
                <td><Tick v={p.ltp} render={(n) => (n == null ? '—' : String(n))} /></td>
                <td><Tick v={p.value} render={(n) => lakh(n ?? 0)} /></td>
                <td className={(p.day_move_pct ?? 0) >= 0 ? styles.pos : styles.neg}
                    style={pnlTint(p.day_move_pct == null ? null : p.day_move_pct * 3)}>
                  {p.day_move_pct == null ? '—' : (p.day_move_pct >= 0 ? '+' : '') + p.day_move_pct + '%'}</td>
                <td className={(p.pnl ?? 0) >= 0 ? styles.pos : styles.neg} style={pnlTint(p.pnl_pct)}>
                  <Tick v={p.pnl} render={(n) => ((n ?? 0) >= 0 ? '+' : '') + inr(n ?? 0)} /></td>
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
        )}
      </div>

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
        <div className={styles.cardTitle}>Cash &amp; capital</div>
        <div style={{ display: 'grid', gap: 14, gridTemplateColumns: 'repeat(auto-fit,minmax(150px,1fr))' }}>
          {[
            { k: 'Contributed capital', v: inr(capital), hint: 'every rupee paid in, less anything taken out' },
            { k: 'Idle cash', v: inr(r.cash), hint: 'credited but not yet in a position',
              warn: r.cash > capital * 0.1 },
            { k: 'Deployed', v: deployedPct.toFixed(0) + '%', hint: 'positions as a share of book value' },
            { k: 'Slots used', v: `${r.positions.length} / 16`, hint: '16 slots at 6.25% of NAV each' },
          ].map((x) => (
            <div key={x.k} title={x.hint}>
              <div className={styles.muted} style={{ fontSize: 11.5 }}>{x.k}</div>
              <div style={{ fontSize: 19, fontWeight: 700,
                            color: x.warn ? 'var(--accent-warn,#B45309)' : undefined }}>{x.v}</div>
            </div>
          ))}
        </div>
        {r.cash > 1000 && (
          <p className={styles.note}>
            Idle cash sits undeployed until you buy — this book has no automated executor, so a
            deposit is recorded and alerted, never placed. Deploy it from the{' '}
            <a href="/app/capital">Capital Desk</a>.
          </p>
        )}
        {(r.flows ?? []).length > 0 && (
          <p className={styles.note}>
            Flows: {(r.flows ?? []).slice(-5).map((f) =>
              `${f.kind} ${inr(f.amount)} (${String(f.ts).slice(0, 10)})`).join(' · ')}
          </p>
        )}
      </div>

      <div className={styles.card}>
        <div className={styles.cardTitle}>Closed trades</div>
        {(!r.trades || r.trades.length === 0)
          ? <p className={styles.note}>None yet — exits land here when the 15:18 checker fires and a sell executes.</p>
          : (
            <table className={styles.table}>
              <thead><tr>
                <th>Stock</th><th>Entry</th><th>Exit</th><th>Qty</th>
                <th>Buy ₹</th><th>Sell ₹</th><th>P&amp;L ₹</th><th>P&amp;L %</th><th>Why</th>
              </tr></thead>
              <tbody>
                {r.trades.map((tr, i) => (
                  <tr key={i}>
                    <td className={styles.sym}>{tr.symbol ?? '—'}</td>
                    <td className={styles.muted}>{tr.entry_date ? fmtD(tr.entry_date) : '—'}</td>
                    <td className={styles.muted}>{tr.exit_date ? fmtD(tr.exit_date) : '—'}</td>
                    <td>{tr.qty ?? '—'}</td>
                    <td>{tr.buy ?? '—'}</td><td>{tr.sell ?? '—'}</td>
                    <td className={(tr.net_pnl ?? 0) >= 0 ? styles.pos : styles.neg}>
                      {(tr.net_pnl ?? 0) >= 0 ? '+' : ''}{inr(tr.net_pnl ?? 0)}</td>
                    <td className={(tr.pnl_pct ?? 0) >= 0 ? styles.pos : styles.neg}>{pct(tr.pnl_pct ?? null)}</td>
                    <td className={styles.reason}>{tr.reason ?? '—'}</td>
                  </tr>
                ))}
              </tbody>
              <tfoot>
                <tr style={{ borderTop: '2px solid var(--hairline,rgba(0,0,0,0.14))', fontWeight: 700 }}>
                  <td>TOTAL ({r.trades.length})</td><td colSpan={5} />
                  <td className={r.realized >= 0 ? styles.pos : styles.neg}>
                    {r.realized >= 0 ? '+' : ''}{inr(r.realized)}</td>
                  <td colSpan={2} />
                </tr>
              </tfoot>
            </table>
          )}
        <p className={styles.note}>{r.note}</p>
        <p className={styles.note}>
          Paper model retired from this page (Arun, 04-Sep-2026); its engine still runs headless as
          the reference model. The Capital Desk and the dividend engine were rewired onto THIS book
          on 05-Sep-2026 — before that they were both operating on the paper book's balances.
          Study: <a href="/app/backtest/bluesky-ath-breakout-research142">bluesky-ath-breakout-research142</a>.
        </p>
      </div>

      <div className={styles.card}>
        <div className={styles.cardTitle}>How it works</div>
        <table className={`${styles.table} ${styles.rulesTable}`}>
          <tbody>
            {[
              ['Universe', 'NSE equities clearing a Rs 5 cr 20-day median traded value floor; ETFs excluded'],
              ['Entry (as designed)', 'close at an all-time high, relative strength >= 70, bought the next day'],
              ['Entry (PAUSED 11-Sep-2026)', 'buying is switched off. The scanner arms the opposite condition to the one above \u2014 it picks names still BELOW their high \u2014 and separately, the published entry cannot be placed by any real order. See the ledger below. Selling, stops and the trail keep running.'],
              ['Sizing', '16 slots at 6.25% of NAV each — equal weight, no pyramiding'],
              ['Hard stop', 'close 8% below the fill'],
              ['Trail', 'close below the 15-day SMA (the entry day is exempt)'],
              ['Market gate', 'none — the gate was retired in r/142; per-stock stops carry the risk'],
              ['Exit check', '15:18 IST daily on a close proxy, and it PLACES the sell orders itself (automated since 08-Sep-2026)'],
              ['Costs', '25 bps per side, and 20% STCG on gains held under a year'],
            ].map(([k, v]) => (
              <tr key={k}><td className={styles.sym}>{k}</td><td className={styles.muted}>{v}</td></tr>
            ))}
          </tbody>
        </table>
        <p className={styles.note}>
          Liveness and day-by-day history are not shown here yet: both components read
          registries keyed to SQLite books, and this one keeps JSON state. With no closed
          trades so far they would render empty, so the plumbing waits until there is
          something to put in them.
        </p>
      </div>

      <LedgerCard />
    </div>
  );
}
