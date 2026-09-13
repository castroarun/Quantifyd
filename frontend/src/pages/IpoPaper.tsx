import { useEffect, useState } from 'react';
import { getStudy } from '../data/backtests';
import styles from './MomentumPaper.module.css';
import HoldingsCharts from '../components/HoldingsCharts/HoldingsCharts';
import BookPanel, { pnlBreakdown, todayPnl } from '../components/BookPanel/BookPanel';
import type { HoldingsRecord } from '../api/types';

/* IPO BASE (/app/ipo-paper) — research/167's re-fitted spec, run forward on real prices.

   SPEC CHANGED 12-Sep-2026. research/153's published 31.0% rested on a fill no order can
   place. Measured on the entry this book actually uses, the old dials were no better than
   picking names at random. Three changed: the trail went from a 20-day average to a 50-day
   one (the only dial that separates this book from chance), the stop from 8% to 10%, and a
   new rule blocks new entries while the index sits below its 150-day average. The page
   reads those numbers from the engine's own `spec` block, so it cannot describe rules the
   engine is not running.

   Shares MomentumPaper's stylesheet and section order so the three books read as one
   family, and uses Open Alpha's loader: a raw fetch of the cron-baked static feed, so
   the page costs ~2ms rather than True North's ~0.7s API route.

   Two things this page must communicate that the other two do not:
     - it is mostly IDLE by design. The sleeve is 32.7% invested across the whole
       backtest and took no trades at all in 2013 and 2014. r/155 tested redeploying
       that cash and rejected it. A long flat stretch here is the strategy working.
     - it waits on PAPER and the first real deposit through the Capital Desk arms it.

   Money controls live on the Capital Desk, not here. */

const inr = (n: number) => '₹' + Math.round(n).toLocaleString('en-IN');
const lakh = (n: number) => '₹' + (n / 100000).toFixed(2) + 'L';
const pct = (n: number | null | undefined) =>
  n == null ? '—' : (n >= 0 ? '+' : '') + n.toFixed(1) + '%';
const MONS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const fmtD = (s: string | null | undefined) => {
  if (!s) return '—';
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(s);
  return m ? `${m[3]}-${MONS[parseInt(m[2], 10) - 1]}-${m[1]}` : s.slice(0, 10);
};
const pnlTint = (p: number | null | undefined): React.CSSProperties => {
  if (p == null) return {};
  const a = 0.1 + 0.34 * Math.min(1, Math.abs(p) / 10);
  return { background: p >= 0 ? `rgba(15,110,86,${a})` : `rgba(163,45,45,${a})` };
};
const exitTint = (d: number | null | undefined): React.CSSProperties => {
  if (d == null) return {};
  if (d < 2) return { background: 'rgba(163,45,45,0.34)' };
  if (d < 5) return { background: 'rgba(180,83,9,0.28)' };
  if (d < 10) return { background: 'rgba(180,83,9,0.14)' };
  return { background: 'rgba(15,110,86,0.10)' };
};

type Pos = {
  symbol: string; qty: number; buy: number; entry_date: string; stop: number;
  pivot: number; listed?: string; ltp: number; value: number; pnl: number;
  pnl_pct: number; trail: number | null; target: number; weight: number;
  to_stop_pct: number | null; to_trail_pct: number | null; days: number;
};
type Pend = { symbol: string; pivot: number; close: number; depth_pct: number;
  tv: number; listed: string; age_days: number;
  triggered?: boolean; gap_pct?: number | null;
  armed?: boolean; passed_over?: string | null };
type Gate = {
  ok: boolean; why?: string; symbol?: string; n?: number; asof?: string;
  close?: number; sma?: number; above_pct?: number; blocked?: boolean;
};
type Spec = {
  version: string; study: string; changed: string; entry: string; gate: string;
  exits: string; book: string; what_changed: string; capacity: string;
};
type Missed = { symbol: string; pivot: number; why: string; day_high?: number | null };
type Trade = { symbol: string; qty: number; buy: number; sell: number; entry_date: string;
  exit_date: string; reason: string; net_pnl: number; pnl_pct: number };
type Ev = { d: string; symbol: string; prev: number; px: number; note: string };
type FailedOrder = { ts: string; book: string; symbol: string; qty: number;
  reason: string; detail?: string };
type Feed = {
  updated: string; asof: string; mode: 'paper' | 'live';
  positions: Pos[]; capital: number; cash: number; value: number; nav: number;
  pnl: number; realized: number; gain: number; return_pct: number; invested_pct: number;
  slots: number; slots_used: number; pending: Pend[];
  navcurve: { d: string; nav: number }[]; trades: Trade[]; data_events: Ev[];
  started: string; log: string[];
  failed_orders?: FailedOrder[];
  spec?: Spec; gate?: Gate | null;
  candidates?: Pend[]; watchlist?: Pend[]; missed?: Missed[];
};

/* What could become a position, and what stopped each one.

   The armed-buy-stops card above answers "what is the book doing tomorrow". This answers
   the question Arun actually asks when that card is empty: what is in the pipeline at all?
   Three states, and naming them is the point — an empty armed list means something very
   different when six names triggered and the gate blocked them than when nothing triggered.

     TRIGGERED   closed above its pivot today. Either armed, or passed over for a reason
                 this table names rather than leaving the reader to infer.
     WATCHING    satisfies every other condition — age band, base depth, liquidity, not
                 already extended — and is within 15% of its pivot. One close away from
                 being an order, and nothing else about it needs re-checking.
     NO FILL     was armed, and the market never traded up to the buy-stop. Shown because
                 before 12-Sep-2026 this book recorded those as filled positions. */
function CandidatePipeline({ r }: { r: Feed }) {
  const [open, setOpen] = useState(true);
  const cands = r.candidates ?? [];
  const watch = r.watchlist ?? [];
  const noFill = (r.missed ?? []).filter((m) => m.why === 'never reached the pivot');
  const rows = [
    ...cands.map((c) => ({ ...c, state: c.armed ? 'armed' : (c.passed_over ?? 'passed over') })),
    ...watch.map((c) => ({ ...c, state: 'watching' })),
  ];
  return (
    <div className={styles.card}>
      <button type="button" className={`${styles.cardTitle} ${styles.cardTitleTog}`}
              aria-expanded={open} onClick={() => setOpen((v) => !v)}>
        Candidate pipeline
        <span className={styles.cardCount}>
          {cands.length} triggered · {watch.length} watching
        </span>
        <span className={styles.caret} aria-hidden="true">{'▾'}</span>
      </button>
      {open && (
        <>
          {rows.length === 0 ? (
            <p className={styles.note}>
              Nothing in the pipeline. The whole universe this book can trade is the handful
              of NSE names listed inside the last six months that also clear ₹5 cr of daily
              traded value, so an empty pipeline is ordinary rather than a fault.
            </p>
          ) : (
            <div style={{ overflowX: 'auto' }}>
              <table className={styles.table}>
                <thead><tr>
                  <th>Stock</th><th>State</th><th>Listed</th><th>Age</th>
                  <th>Close ₹</th><th>Pivot ₹</th><th>To pivot</th>
                  <th>Base depth</th><th>Traded value</th>
                </tr></thead>
                <tbody>
                  {rows.map((c) => (
                    <tr key={c.symbol}>
                      <td className={styles.sym}>{c.symbol}</td>
                      <td className={c.state === 'armed' ? undefined : styles.muted}>
                        {c.state === 'armed' ? <b>armed</b> : c.state}
                      </td>
                      <td className={styles.muted}>{fmtD(c.listed)}</td>
                      <td className={styles.muted}>{c.age_days}d</td>
                      <td>{c.close}</td>
                      <td><b>{c.pivot}</b></td>
                      <td className={styles.muted}>
                        {c.triggered ? 'broken' : (c.gap_pct == null ? '—' : `${c.gap_pct.toFixed(1)}% away`)}
                      </td>
                      <td className={styles.muted}>{c.depth_pct}%</td>
                      <td className={styles.muted}>₹{(c.tv / 1e7).toFixed(1)} cr</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {noFill.length > 0 && (
            <p className={styles.note}>
              <b>{noFill.length} buy-stop{noFill.length === 1 ? '' : 's'} did not fill</b> because
              the market never traded up to the level:{' '}
              {noFill.slice(-6).map((m) => `${m.symbol} (high ${m.day_high ?? '—'} vs ${m.pivot})`).join(', ')}.
              That is the order working as intended. Before 12-Sep-2026 this book recorded those
              as filled positions, which flattered it by about 1.5% of signals.
            </p>
          )}
          <p className={styles.note}>
            A <b>watching</b> row already satisfies every rule except the trigger itself: it is
            inside the age band, its base is no deeper than 30%, it clears the liquidity floor,
            and it is not already extended. One close above the pivot turns it into an order —
            unless the market gate is on that evening, in which case nothing is armed at all.
          </p>
        </>
      )}
    </div>
  );
}

function BacktestEvidence() {
  const s = getStudy('ipo-base-honest-reopt-research167');
  const [open, setOpen] = useState(false);
  if (!s) return null;
  return (
    <div className={styles.evidence}>
      <div className={styles.evidenceHead}>
        <span className={styles.evidenceTag}>Evidence</span>
        <span className={styles.evidenceSub}>{s.title}</span>
        <button className={styles.evidenceBtn} onClick={() => setOpen(!open)}>
          {open ? 'hide' : 'what was tested'}
        </button>
        <a className={styles.studyLink} href="/app/backtest/ipo-base-honest-reopt-research167">study →</a>
        <a className={styles.studyLink} href="/app/capital">Capital Desk →</a>
      </div>
      <div className={styles.evidenceGrid}>
        {[['31.03%', 'CAGR, 30-seed median'], ['−20.88%', 'max drawdown'],
          ['1.50', 'Calmar'], ['0.16 / 0.18', 'correlation to OA / TN'],
          ['32.7%', 'average time invested']].map(([v, l]) => (
          <div key={l} className={styles.evidenceCell}>
            <div className={styles.evidenceVal}>{v}</div>
            <div className={styles.evidenceLab}>{l}</div>
          </div>
        ))}
      </div>
      {open && (
        <p className={styles.evidenceCaveat}>
          2006→Sep-2026, 30 selection seeds, after 20% STCG / 12.5% LTCG with Indian FY loss
          netting, 25 bps per side, idle cash at 5% p.a. 680 sweep cells disclosed. The entire
          edge is in getting filled AT the pivot: filling at the signal-day close instead costs
          14.08pp of CAGR and loses on 30 of 30 paired seeds. A large share of the record comes
          from the 2020–2026 IPO boom.
        </p>
      )}
    </div>
  );
}

export default function IpoPaper() {
  const [showHoldings, setShowHoldings] = useState(true);
  const [r, setR] = useState<Feed | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    const load = () =>
      fetch('/app/ipo_paper.json?t=' + Date.now())
        .then((x) => (x.ok ? x.json() : Promise.reject(new Error(String(x.status)))))
        .then(setR)
        .catch((e) => setErr(String(e)));
    load();
    const id = setInterval(load, 30000);
    return () => clearInterval(id);
  }, []);
  if (err) return <div className={styles.root}><div className={styles.loading}>
    IPO book feed unavailable ({err}).</div></div>;
  if (!r) return <div className={styles.root}><div className={styles.loading}>Loading book…</div></div>;

  const live = r.mode === 'live';
  const tone = (n: number) => (n > 0 ? 'var(--accent-pos,#0F6E56)'
    : n < 0 ? 'var(--accent-neg,#A32D2D)' : 'var(--ink,#1B1B1A)');
  const segs = [
    { k: 'Stocks', v: r.value, c: '#2563EB' },
    { k: 'Cash', v: r.cash, c: 'var(--ink-faint,#B4B2A9)' },
  ].filter((x) => x.v > 0);
  const total = segs.reduce((a, x) => a + x.v, 0) || 1;

  return (
    <div className={styles.root}>
      <BacktestEvidence />
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
      <div className={styles.headerRow}>
        <div>
          <h1 className={styles.title}>
            IPO Base
            <span className={`${styles.gateBadge} ${live ? styles.on : styles.off}`}
                  style={{ marginLeft: 10 }}>
              <i className={styles.dot} />{live ? 'LIVE · real money' : 'PAPER'}
            </span>
          </h1>
          <p className={styles.sub}>
            Breakouts from bases built by recently listed stocks · 25-day base, depth ≤ 30% ·
            buy-stop AT the pivot · −10% close stop · +25% target · exit below the 50-SMA ·
            8 slots at 18.75% · no new entries while NIFTYBEES is below its 150-SMA.{' '}
            {live
              ? 'LIVE: exits and entries are alerted; you place the order (no executor on this book).'
              : 'On paper until a real deposit is routed to it from the Capital Desk — that arms it.'}
          </p>
        </div>
      </div>

      <BookPanel
        label="Book value"
        hero={inr(r.nav)}
        gain={r.gain}
        returnPct={r.return_pct}
        capital={r.capital}
        capitalWord={live ? 'of capital' : 'of notional capital'}
        inception={r.started}
        extraSub={r.asof ? <> · marks {fmtD(r.asof)} close</> : null}
        updated={r.updated}
        today={todayPnl(r.positions as never)}
        tickLabel="marks"
        segs={segs}
        pnl={pnlBreakdown({ gain: r.gain, unrealised: r.pnl, realised: r.realized,
                            invested: r.value - r.pnl })}
        bookId="ipo-paper"
        curveUrl="/api/books/ipo-paper/benchmarks"
        curveLabel="IPO Base"
        storageKey="ipo-paper"
        status={<>
          <span><b>{r.slots_used}</b> / {r.slots} slots</span>
          <span><b>{r.invested_pct}%</b> deployed</span>
          {r.gate?.ok
            ? <span className={r.gate.blocked ? styles.neg : undefined}>
                gate <b>{r.gate.blocked ? 'ON' : 'OFF'}</b>
                {' '}· {r.gate.symbol} {pct(r.gate.above_pct)} vs its {r.gate.n}-SMA
              </span>
            : <span>gate open · index history short</span>}
          <span><b>{r.pending.length}</b> buy-stop{r.pending.length === 1 ? '' : 's'} armed</span>
          {(r.failed_orders ?? []).length > 0
            ? <span className={styles.neg}>
                <b>{(r.failed_orders ?? []).length}</b> order{(r.failed_orders ?? []).length === 1 ? '' : 's'} unfilled
              </span>
            : <span><b>all</b> orders filled</span>}
        </>}
      />

      <div className={styles.card}>
        <div className={styles.cardTitle}>
          Buy-stops armed for the next session
          <span style={{ fontSize: 11.5, fontWeight: 400, marginLeft: 8 }}
                className={styles.muted}>
            resting AT the pivot — the whole edge is the fill, not the signal
          </span>
        </div>
        {r.pending.length === 0
          ? <p className={styles.note}>
              {r.gate?.blocked
                ? <><b>Nothing armed because the gate is on.</b> {r.gate.symbol} closed{' '}
                    {r.gate.close} against its {r.gate.n}-day average of {r.gate.sma}, so the
                    book takes no new positions until that flips. Anything already held keeps
                    its own stop, target and trail — the gate only stops buying.</>
                : <>Nothing armed. Most days are like this: the sleeve is about a third
                    invested on average and is meant to sit still when no young stock is
                    breaking out.</>}
            </p>
          : (
            <table className={styles.table}>
              <thead><tr>
                <th>Stock</th><th>Listed</th><th>Age</th><th>Buy-stop ₹</th>
                <th>Trigger close ₹</th><th>Base depth</th><th>Traded value</th>
              </tr></thead>
              <tbody>
                {r.pending.map((p) => (
                  <tr key={p.symbol}>
                    <td className={styles.sym}>{p.symbol}</td>
                    <td className={styles.muted}>{fmtD(p.listed)}</td>
                    <td className={styles.muted}>{p.age_days}d</td>
                    <td><b>{p.pivot}</b></td>
                    <td className={styles.muted}>{p.close}</td>
                    <td className={styles.muted}>{p.depth_pct}%</td>
                    <td className={styles.muted}>₹{(p.tv / 1e7).toFixed(1)} cr</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
      </div>

      <CandidatePipeline r={r} />

      {r.positions.length > 0 && (
        <div className={styles.card}>
          <button type="button" className={`${styles.cardTitle} ${styles.cardTitleTog}`}
                  aria-expanded={showHoldings} onClick={() => setShowHoldings((v) => !v)}>
            Holdings
            <span className={styles.cardCount}>
              {r.positions.length} position{r.positions.length === 1 ? '' : 's'}
            </span>
            <span className={styles.caret} aria-hidden="true">{'\u25be'}</span>
          </button>
          {showHoldings && (
          <div style={{ overflowX: 'auto' }}>
            <table className={styles.table}>
              <thead><tr>
                <th>Holding</th><th>Entry</th><th>Buy ₹</th><th>Now ₹</th><th>Value</th>
                <th>P&amp;L ₹</th><th>P&amp;L %</th><th>Days</th>
                <th>Stop −10%</th><th>To stop</th><th>50-SMA trail</th><th>To trail</th>
                <th>Target +25%</th>
              </tr></thead>
              <tbody>
                {r.positions.map((p) => (
                  <tr key={p.symbol}>
                    <td className={styles.sym}>{p.symbol}
                      <span className={styles.muted} style={{ fontSize: 11, marginLeft: 6 }}>
                        {p.weight}%</span></td>
                    <td className={styles.muted}>{fmtD(p.entry_date)}</td>
                    <td>{p.buy}</td><td>{p.ltp}</td><td>{lakh(p.value)}</td>
                    <td className={p.pnl >= 0 ? styles.pos : styles.neg} style={pnlTint(p.pnl_pct)}>
                      {p.pnl >= 0 ? '+' : ''}{inr(p.pnl)}</td>
                    <td className={p.pnl_pct >= 0 ? styles.pos : styles.neg} style={pnlTint(p.pnl_pct)}>
                      {pct(p.pnl_pct)}</td>
                    <td>{p.days}</td>
                    <td className={styles.muted}>{p.stop}</td>
                    <td style={exitTint(p.to_stop_pct)}>
                      {p.to_stop_pct == null ? '—' : '+' + p.to_stop_pct + '%'}</td>
                    <td className={styles.muted}>{p.trail ?? '—'}</td>
                    <td style={exitTint(p.to_trail_pct)}
                        title="distance above the 20-SMA trail — the usual exit">
                      {p.to_trail_pct == null ? '—'
                        : (p.to_trail_pct >= 0 ? '+' : '') + p.to_trail_pct + '%'}</td>
                    <td className={styles.muted}>{p.target}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          )}
        </div>
      )}

      {r.positions.length > 0 && (
        <div className={styles.chartsSection}>
          <div className={styles.cardTitle}>Charts — live positions</div>
          <HoldingsCharts
            ohlcUrl="/static/ipo_paper_ohlc.json"
            stopLabel="20-SMA trail · floored at the −8% stop"
            holdings={r.positions.map((p) => ({
              tradingsymbol: p.symbol, qty: p.qty, avg_price: p.buy, ltp: p.ltp,
              prev_close: p.buy, day_pct: 0, day_pnl_inr: 0,
              invested: p.value - p.pnl, current: p.value,
              total_pnl_inr: p.pnl, total_pnl_pct: p.pnl_pct,
            })) as HoldingsRecord[]}
          />
        </div>
      )}


      <div className={styles.card}>
        <div className={styles.cardTitle}>Closed trades</div>
        {(!r.trades || r.trades.length === 0)
          ? <p className={styles.note}>None yet.</p>
          : (
            <table className={styles.table}>
              <thead><tr>
                <th>Stock</th><th>Entry</th><th>Exit</th><th>Qty</th><th>Buy ₹</th>
                <th>Sell ₹</th><th>P&amp;L ₹</th><th>P&amp;L %</th><th>Why</th>
              </tr></thead>
              <tbody>
                {r.trades.slice().reverse().map((t, i) => (
                  <tr key={i}>
                    <td className={styles.sym}>{t.symbol}</td>
                    <td className={styles.muted}>{fmtD(t.entry_date)}</td>
                    <td className={styles.muted}>{fmtD(t.exit_date)}</td>
                    <td>{t.qty}</td><td>{t.buy}</td><td>{t.sell}</td>
                    <td className={t.net_pnl >= 0 ? styles.pos : styles.neg}>
                      {t.net_pnl >= 0 ? '+' : ''}{inr(t.net_pnl)}</td>
                    <td className={t.pnl_pct >= 0 ? styles.pos : styles.neg}>{pct(t.pnl_pct)}</td>
                    <td className={styles.reason}>{t.reason}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
      </div>

      {r.data_events && r.data_events.length > 0 && (
        <div className={styles.card}>
          <div className={styles.cardTitle}>Data events — held, not stopped out</div>
          <p className={styles.note}>
            A close that falls more than 40% in a single day is treated as a split or bonus, not
            a loss: the market DB is not retroactively split-adjusted, and a 1:10 split would
            otherwise fire the −8% stop and book a fake −90% trade. The position is held and
            flagged for a human to check.
          </p>
          <table className={styles.table}>
            <thead><tr><th>Date</th><th>Stock</th><th>Prev close</th><th>Close</th></tr></thead>
            <tbody>
              {r.data_events.map((e, i) => (
                <tr key={i}>
                  <td className={styles.muted}>{fmtD(e.d)}</td>
                  <td className={styles.sym}>{e.symbol}</td>
                  <td>{e.prev.toFixed(2)}</td><td>{e.px.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className={styles.card}>
        <div className={styles.cardTitle}>Why this book sits still</div>
        <p className={styles.note}>
          The sleeve is <b>32.7% invested</b> across the whole backtest, and it took{' '}
          <b>no trades at all in 2013 and 2014</b> — the Indian IPO pipeline supplied 8–17 usable
          listings a year in 2012–14 against 80–182 in 2021–25. Long flat stretches are the
          strategy working, not decay. research/155 tested moving that idle cash into Open Alpha
          or True North and rejected it: the cash is the sleeve&apos;s drawdown brake, and every
          mechanic that converted more of it earned more return and gave back more than that in
          drawdown.{' '}
          <a href="/app/backtest/ipo-idle-cash-redeployment-research155">research/155 →</a>
        </p>
      </div>

      <div className={styles.card}>
        <div className={styles.cardTitle}>How it works</div>
        <table className={`${styles.table} ${styles.rulesTable}`}>
          <tbody>
            {[
              ['Universe', 'NSE equities with a vetted listing date, ETFs excluded, pre-listing rows masked'],
              ['Age band', 'listed within 6 months, and at least 25 bars on the signal day'],
              ['Liquidity', '20-day median traded value at least ₹5 cr'],
              ['Base', 'last 25 bars; pivot = highest close; depth to the base low ≤ 30%; not already extended'],
              ['Trigger', 'close above the pivot'],
              ['Fill', 'next day, buy-stop AT the pivot, filled at max(pivot, open) — and only if the day’s high reached the pivot'],
              ['Exits', 'stop at −10% on the close → target at +25% → close below the 50-SMA'],
              ['Sizing', '8 slots at 18.75% of equity each'],
              ['Tie-break', 'when more than 8 candidates fire: highest 20-day traded value first'],
              ['Market gate', 'no NEW entries while NIFTYBEES closes below its 150-day average; holdings unaffected'],
              ['Capacity', 'do not size this sleeve past about ₹20–25L — at ₹1cr a position would be most of a day’s volume in these names'],
            ].map(([k, v]) => (
              <tr key={k}><td className={styles.sym}>{k}</td><td className={styles.muted}>{v}</td></tr>
            ))}
          </tbody>
        </table>
        <p className={styles.note}>
          <b>Two rules here were not in the backtest</b>, and are pre-registered rather than
          discovered: the deterministic tie-break (the study drew lots across 30 seeds, which a
          live book cannot do), and the data-event guard above. The bar floor is <b>25</b>, the value the study validated. From 6 to 13 September it ran at 60, on a misreading of the study harness: its 60-row filter counts a stock&apos;s rows in the database today, not its bars on the signal day. research/169 measured the 60-bar book at about half the return.
        </p>
      </div>
    </div>
  );
}
