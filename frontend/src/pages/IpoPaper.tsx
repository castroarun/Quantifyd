import { useEffect, useState } from 'react';
import { getStudy } from '../data/backtests';
import styles from './MomentumPaper.module.css';
import HoldingsCharts from '../components/HoldingsCharts/HoldingsCharts';
import type { HoldingsRecord } from '../api/types';

/* IPO BASE (/app/ipo-paper) — research/153's adopted spec, run forward on real prices.

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
  const d = new Date(s.slice(0, 10));
  return isNaN(+d) ? s.slice(0, 10) : `${d.getDate()} ${MONS[d.getMonth()]}`;
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
  tv: number; listed: string; age_days: number };
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
};

function BacktestEvidence() {
  const s = getStudy('ipo-base-breakout-research153');
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
        <a className={styles.studyLink} href="/app/backtest/ipo-base-breakout-research153">study →</a>
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

function CurveCard({ nc }: { nc: { d: string; nav: number }[] }) {
  if (!nc || nc.length < 2)
    return (
      <div className={styles.card}>
        <div className={styles.cardTitle}>Equity curve</div>
        <p className={styles.note}>One point so far — the curve builds a point per close.</p>
      </div>
    );
  const v = nc.map((x) => x.nav);
  const min = Math.min(...v), max = Math.max(...v), span = max - min || 1;
  const W = 720, H = 160;
  const pts = nc.map((x, k) =>
    `${(k / (nc.length - 1)) * W},${H - 14 - ((x.nav - min) / span) * (H - 28)}`).join(' ');
  return (
    <div className={styles.card}>
      <div className={styles.cardTitle}>Equity curve — since {fmtD(nc[0].d)}</div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto' }} role="img"
           aria-label="IPO base equity curve">
        <polyline points={pts} fill="none" stroke="var(--accent-pos,#0F6E56)" strokeWidth="2" />
      </svg>
    </div>
  );
}

export default function IpoPaper() {
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
            buy-stop AT the pivot · −8% close stop · +25% target · exit below the 20-SMA ·
            8 slots at 18.75% · no market gate.{' '}
            {live
              ? 'LIVE: exits and entries are alerted; you place the order (no executor on this book).'
              : 'On paper until a real deposit is routed to it from the Capital Desk — that arms it.'}
          </p>
        </div>
      </div>

      <div className={styles.bookSummary}>
        <div className={styles.sumMain}>
          <div className={styles.sumLabel}>Book value</div>
          <div className={styles.sumHero}>{inr(r.nav)}</div>
          <div className={styles.sumSub}>
            on <b>{inr(r.capital)}</b> of {live ? 'capital' : 'notional capital'}{' '}
            <span style={{ color: tone(r.gain), fontWeight: 700 }}>
              {r.gain >= 0 ? '+' : '−'}{inr(Math.abs(r.gain))} · {pct(r.return_pct)}
            </span>{' '}· since {fmtD(r.started)}
          </div>
          <div className={styles.sumSub}>
            marks {fmtD(r.asof)} close · updated {fmtD(r.updated)} {r.updated?.slice(11, 16)} IST
          </div>
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
            <span><b>{r.slots_used}</b> / {r.slots} slots</span>
            <span><b>{r.invested_pct}%</b> deployed</span>
            <span>no market gate</span>
            <span>{r.pending.length} buy-stop{r.pending.length === 1 ? '' : 's'} armed</span>
          </div>
        </div>
        <div className={styles.sumPnl}>
          <div className={styles.sumLabel}>Profit &amp; loss</div>
          {[{ k: 'Unrealised', v: r.pnl }, { k: 'Realised (net)', v: r.realized },
            { k: 'Costs & fees', v: r.gain - (r.pnl + r.realized) }].map((x) => (
            <div key={x.k} className={styles.pnlRow}>
              <span>{x.k}</span>
              <b style={{ color: tone(x.v) }}>{x.v >= 0 ? '+' : '−'}{inr(Math.abs(x.v))}</b>
            </div>
          ))}
          <div className={`${styles.pnlRow} ${styles.pnlTotal}`}>
            <span>Total return</span>
            <b style={{ color: tone(r.gain) }}>
              {r.gain >= 0 ? '+' : '−'}{inr(Math.abs(r.gain))} · {pct(r.return_pct)}</b>
          </div>
        </div>
      </div>

      <div className={styles.card}>
        <div className={styles.cardTitle}>
          Buy-stops armed for the next session
          <span style={{ fontSize: 11.5, fontWeight: 400, marginLeft: 8 }}
                className={styles.muted}>
            resting AT the pivot — the whole edge is the fill, not the signal
          </span>
        </div>
        {r.pending.length === 0
          ? <p className={styles.note}>Nothing armed. Most days are like this: the sleeve is
              32.7% invested on average and is meant to sit still when no young stock is
              breaking out.</p>
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

      {r.positions.length > 0 && (
        <div className={styles.card}>
          <div className={styles.cardTitle}>Holdings</div>
          <div style={{ overflowX: 'auto' }}>
            <table className={styles.table}>
              <thead><tr>
                <th>Holding</th><th>Entry</th><th>Buy ₹</th><th>Now ₹</th><th>Value</th>
                <th>P&amp;L ₹</th><th>P&amp;L %</th><th>Days</th>
                <th>Stop −8%</th><th>To stop</th><th>20-SMA trail</th><th>To trail</th>
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

      <CurveCard nc={r.navcurve} />

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
              ['Age band', 'listed within 6 months, and at least 60 bars of history'],
              ['Liquidity', '20-day median traded value at least ₹5 cr'],
              ['Base', 'last 25 bars; pivot = highest close; depth to the base low ≤ 30%; not already extended'],
              ['Trigger', 'close above the pivot'],
              ['Fill', 'next day, buy-stop AT the pivot, filled at max(pivot, open)'],
              ['Exits', 'stop at −8% on the close → target at +25% → close below the 20-SMA'],
              ['Sizing', '8 slots at 18.75% of equity each'],
              ['Tie-break', 'when more than 8 candidates fire: highest 20-day traded value first'],
              ['Market gate', 'none — it lost on 30 of 30 seeds'],
            ].map(([k, v]) => (
              <tr key={k}><td className={styles.sym}>{k}</td><td className={styles.muted}>{v}</td></tr>
            ))}
          </tbody>
        </table>
        <p className={styles.note}>
          <b>Two rules here were not in the backtest</b>, and are pre-registered rather than
          discovered: the deterministic tie-break (the study drew lots across 30 seeds, which a
          live book cannot do), and the data-event guard above. The bar floor is <b>60</b>, not
          the 25 the written spec says — the study&apos;s own harness only ever admitted stocks
          with 60+ bars, so 60 is what was actually validated.
        </p>
      </div>
    </div>
  );
}
