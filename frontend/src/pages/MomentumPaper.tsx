import { useEffect, useState, type CSSProperties } from 'react';
import { apiGet, apiPost } from '../api/client';
import BacktestCharts from '../components/BacktestCurve/BacktestCharts';
import HoldingsCharts from '../components/HoldingsCharts/HoldingsCharts';
import LiveCurve from '../components/LiveCurve/LiveCurve';
import { getStudy } from '../data/backtests';
import type { HoldingsRecord } from '../api/types';
import styles from './MomentumPaper.module.css';
import BookActivity from '../components/BookActivity/BookActivity';
import DailyPerformance from '../components/DailyPerformance/DailyPerformance';

type Holding = {
  symbol: string; qty: number | null; entry_date: string; entry_price: number | null;
  price: number | null; value: number; weight: number; pnl: number; pnl_pct: number | null;
  days: number; stop: number | null; stop_dist_pct: number | null; is_cash?: boolean;
};
type Target = { symbol: string; rank: number; tier: string };
type Closed = {
  symbol: string; entry_date: string; entry_price: number; exit_date: string; exit_price: number;
  qty: number; gross_pnl: number; gross_pct: number; net_pnl: number; reason: string;
  holding_days: number; stcg_tax: number;
};
type NavPt = { d: string; nav: number; bench: number | null; gate: string };
type Hedge = {
  tsym: string; strike: number; expiry: string; qty: number; lots: number | null;
  entry_date: string; entry_price: number; price: number | null; cost: number;
  value: number; pnl: number; pnl_pct: number; dte: number;
} | null;
type HedgeClosed = {
  id: number; tsym: string; strike: number; expiry: string; qty: number; entry_date: string;
  entry_price: number; exit_date: string; exit_price: number; cost: number; proceeds: number;
  pnl: number; reason: string;
};
type State = {
  seeded: boolean; gate: string; inception: string | null; capital: number; nav: number;
  cash: number; equity: number; invested_pct: number; total_return_pct: number;
  unrealized: number; realized_net: number; n_holdings: number; interest_earned: number;
  cash_yield_pct: number; stcg_unbooked: number; stcg_booked: number;
  last_daily: string | null; last_weekly: string | null; last_monthly: string | null;
  mode: string; live_mode: boolean;
  data_asof: string | null; target_basket: Target[]; gate_last: number | null;
  gate_sma: number | null; gate_gap_pct: number | null;
  holdings: Holding[]; navcurve: NavPt[]; closed: Closed[]; rules: [string, string, string][];
  hedge: Hedge; hedge_closed: HedgeClosed[];
  idle_cash: number; idle_pct: number; days_to_rebalance: number; cash_reserve: number;
  ledger_cash?: number; swept_value?: number; sweep_gain?: number;
  hedge_viability?: {
    enabled: boolean; ratio: number; lot: number; spot: number; lot_notional: number;
    equity: number; target_notional: number; lots_needed: number; viable: boolean;
    over_hedge_x: number | null; equity_for_one_lot: number; capital_needed: number | null;
    capital_now: number; shortfall: number | null; dte_target: number;
  } | null;
  sweep: { enabled: boolean; symbol: string; units: number; value: number };
};

/* Heat tints — intensity scales with the number, so the eye lands on what matters.
   P&L: green/red, full strength at +/-10%. To-stop: red when a stop is imminent. */
const pnlTint = (pctv: number | null | undefined): React.CSSProperties => {
  if (pctv == null || !isFinite(pctv)) return {};
  const t = Math.min(1, Math.abs(pctv) / 10);
  const a = 0.10 + 0.34 * t;
  return {
    background: pctv >= 0 ? `rgba(47,145,82,${a})` : `rgba(224,86,79,${a})`,
    fontWeight: Math.abs(pctv) >= 5 ? 700 : 600,
  };
};
const stopTint = (dist: number | null | undefined): React.CSSProperties => {
  if (dist == null || !isFinite(dist)) return {};
  if (dist < 2) return { background: 'rgba(224,86,79,0.44)', fontWeight: 700 };   // about to stop
  if (dist < 5) return { background: 'rgba(217,119,6,0.30)', fontWeight: 650 };   // close
  if (dist < 10) return { background: 'rgba(217,119,6,0.13)' };                    // watch
  return { background: 'rgba(47,145,82,0.15)' };                                   // safe
};

const inr = (n: number) => '₹' + Math.round(n).toLocaleString('en-IN');
const lakh = (n: number) => '₹' + (n / 100000).toFixed(2) + 'L';
const pct = (n: number | null) => (n == null ? '—' : (n >= 0 ? '+' : '') + n.toFixed(1) + '%');
const reasonLabel: Record<string, string> = {
  DONCHIAN: 'Donchian stop', GATE_RISK_OFF: 'Macro gate (risk-off)',
  BUFFER_ROTATE: 'Buffer rotation', REBALANCE: 'Monthly rebalance', SEED: 'Seed',
};

function EquityCurve({ data }: { data: NavPt[] }) {
  if (data.length < 2) return <div className={styles.chartEmpty}>Equity curve builds as the book runs (1 point so far).</div>;
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

export default function MomentumPaper() {
  const [s, setS] = useState<State | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const load = () => apiGet<State>('/api/momentum-paper/state').then(setS).catch((e) => setErr(String(e)));
  useEffect(() => {
    load();
    const t = setInterval(load, 30000);
    return () => clearInterval(t);
  }, []);

  if (err) return <div className={styles.root}><div className={styles.loading}>Error: {err}</div></div>;
  if (!s) return <div className={styles.root}><div className={styles.loading}>Loading book…</div></div>;

  const riskOn = s.gate === 'ON';
  const retPos = s.total_return_pct >= 0;

  return (
    <div className={styles.root}>
      
      <BacktestEvidence />
      <div className={styles.headerRow}>
        <div>
          <h1 className={styles.title}>
            True North (Momentum-30) — {s.live_mode ? 'Live Book' : 'Paper Book'}
          </h1>
          <p className={styles.sub}>
            <b>Universe = the Nifty 200</b> (200 largest NSE stocks by market cap) → <b>ranked by momentum</b> → <b>hold the top 8</b>.
            {lakh(s.capital)} {s.live_mode ? 'LIVE (real money, shared account)' : 'paper'} (research/62 winner)
            {s.inception ? ` · since ${s.inception}` : ''} · data as-of {s.data_asof || '—'}
          </p>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <span style={{ padding: '5px 11px', borderRadius: 6, fontSize: 12, fontWeight: 700, color: '#fff', background: s.live_mode ? '#c0392b' : '#39424e' }}>
            {s.mode}
          </span>
          <div className={`${styles.gateBadge} ${riskOn ? styles.on : styles.off}`}>
            <span className={styles.dot} />
            {riskOn ? 'RISK-ON · invested' : 'RISK-OFF · in cash'}
          </div>
        </div>
      </div>
      <BookActivity bookId="momentum-3l" />

      <BookSummary s={s} />

      {s.holdings.length > 0 && (
        <div className={styles.card}>
          <div className={styles.cardTitle}>Holdings</div>
          <table className={styles.table}>
            <thead><tr>
              <th>Holding</th><th>Wt</th><th>Entry</th><th>Entry ₹</th><th>Now ₹</th>
              <th>Value</th><th>P&L ₹</th><th>P&L %</th><th>Days</th><th>Donchian stop</th><th>To stop</th>
            </tr></thead>
            <tbody>
              {s.holdings.map((h) => (
                <tr key={h.symbol} className={h.is_cash ? styles.cashRow : ''}>
                  <td className={styles.sym}>
                    {h.symbol}
                    {h.is_cash && <span className={styles.cashTag}>liquid fund @{h.pnl_pct}%</span>}
                  </td>
                  <td>{h.weight}%</td>
                  <td className={styles.muted}>{h.entry_date}</td>
                  <td>{h.entry_price ?? '—'}</td>
                  <td>{h.is_cash ? lakh(h.value) : h.price}</td>
                  <td>{lakh(h.value)}</td>
                  <td className={(h.pnl ?? 0) >= 0 ? styles.pos : styles.neg}
                      style={h.is_cash ? {} : pnlTint(h.pnl_pct)}>
                    {(h.pnl ?? 0) >= 0 ? '+' : ''}{inr(h.pnl ?? 0)}</td>
                  <td className={(h.pnl_pct ?? 0) >= 0 ? styles.pos : styles.neg}
                      style={h.is_cash ? {} : pnlTint(h.pnl_pct)}>
                    {h.is_cash ? '—' : pct(h.pnl_pct)}</td>
                  <td>{h.days}</td>
                  <td className={styles.muted}>{h.is_cash ? '—' : (h.stop ?? '—')}</td>
                  <td style={h.is_cash ? {} : stopTint(h.stop_dist_pct)}
                      title={h.is_cash ? '' : 'distance to the Donchian stop — red means a stop is imminent'}>
                    {h.stop_dist_pct == null ? '—' : '+' + h.stop_dist_pct + '%'}</td>
                </tr>
              ))}
            </tbody>
            <tfoot>
              {(() => {
                // VALUE totals every row — that IS the book's worth. P&L totals only the STOCK
                // rows: the cash row's "pnl" is the liquid ETF's accrued interest, not an
                // unrealised gain, and the header reports it on its own line. Including it here
                // made this table read Rs182 above the header's Unrealised for the same book.
                const stockRows = s.holdings.filter((h) => !h.is_cash);
                const tv = s.holdings.reduce((a, h) => a + (h.value || 0), 0);
                const tp = stockRows.reduce((a, h) => a + (h.pnl || 0), 0);
                const cost = stockRows.reduce((a, h) => a + ((h.value || 0) - (h.pnl || 0)), 0);
                const tpc = cost > 0 ? (tp / cost) * 100 : 0;
                return (
                  <tr style={{ borderTop: '2px solid var(--hairline,rgba(0,0,0,0.14))', fontWeight: 700 }}>
                    <td>TOTAL ({stockRows.length} stocks + cash)</td>
                    <td>100%</td><td /><td /><td />
                    <td>{lakh(tv)}</td>
                    <td className={tp >= 0 ? styles.pos : styles.neg}
                        title="Unrealised P&L on the stocks only — the liquid ETF's interest is reported separately in the header">
                      {tp >= 0 ? '+' : ''}{inr(tp)}</td>
                    <td className={tpc >= 0 ? styles.pos : styles.neg}>{pct(tpc)}</td>
                    <td /><td /><td />
                  </tr>
                );
              })()}
            </tfoot>
          </table>
        </div>
      )}

      {/* Charts — rendered exactly like /app/holdings (no card wrapper) so it gets full width */}
      {s.holdings.filter((h) => !h.is_cash).length > 0 && (
        <div className={styles.chartsSection}>
          <div className={styles.cardTitle}>
            Charts — live positions
            <span style={{ fontSize: 11.5, fontWeight: 400, color: 'var(--ink-muted,#888)', marginLeft: 8 }}>
              scroll to zoom · drag to pan · red dashed line = Donchian trailing stop (the exit rule)
            </span>
          </div>
          <HoldingsCharts
            ohlcUrl="/static/momentum_ohlc.json"
            holdings={s.holdings.filter((h) => !h.is_cash).map((h) => ({
              tradingsymbol: h.symbol,
              qty: h.qty ?? 0,
              avg_price: h.entry_price ?? 0,
              ltp: h.price ?? 0,
              prev_close: h.entry_price ?? null,
              day_pct: 0,
              day_pnl_inr: 0,
              invested: (h.value ?? 0) - (h.pnl ?? 0),
              current: h.value ?? 0,
              total_pnl_inr: h.pnl ?? 0,
              total_pnl_pct: h.pnl_pct ?? 0,
            })) as HoldingsRecord[]}
          />
        </div>
      )}

      <CashStatus s={s} />

      <HedgePanel s={s} />

      <CashPanel s={s} reload={load} />

      <div className={styles.grid2}>
        {/* Gate panel */}
        <div className={styles.card}>
          <div className={styles.cardTitle}>Macro gate · NIFTYBEES vs 100-day SMA (weekly)</div>
          <div className={styles.gateRow}>
            <div><span className={styles.muted}>NIFTYBEES</span><b>{s.gate_last ?? '—'}</b></div>
            <div><span className={styles.muted}>100-DMA</span><b>{s.gate_sma ?? '—'}</b></div>
            <div><span className={styles.muted}>Gap</span>
              <b className={(s.gate_gap_pct ?? 0) >= 0 ? styles.pos : styles.neg}>{pct(s.gate_gap_pct)}</b></div>
            <div><span className={styles.muted}>State</span>
              <b className={riskOn ? styles.pos : styles.neg}>{riskOn ? 'RISK-ON' : 'RISK-OFF'}</b></div>
          </div>
          <p className={styles.note}>
            {riskOn
              ? 'Above the 100-DMA → the book is deployed in the 8 momentum names below.'
              : 'Below the 100-DMA → the book is in cash (earning the liquid yield) and will redeploy into the target basket at the next month-end once NIFTYBEES reclaims its 100-DMA.'}
          </p>
        </div>

        {/* Tax panel */}
        <div className={styles.card}>
          <div className={styles.cardTitle}>Tax (STCG, shown separately — not in NAV)</div>
          <div className={styles.gateRow}>
            <div><span className={styles.muted}>Booked STCG</span><b>{inr(s.stcg_booked)}</b></div>
            <div><span className={styles.muted}>If booked now</span><b>{inr(s.stcg_unbooked)}</b></div>
            <div><span className={styles.muted}>Capital</span><b>{lakh(s.capital)}</b></div>
          </div>
          <p className={styles.note}>20% short-term capital-gains on gains held &lt; 1 year. Tracked separately so the NAV reflects pre-tax market performance net of ~0.3% trading cost.</p>
        </div>
      </div>

      {/* Backtest equity + drawdown — interactive, range-selectable */}
      <div className={styles.card}>
        <div className={styles.cardTitle}>Backtest — Portfolio vs NIFTYBEES (choose any period; both re-based to ₹20L at the start)</div>
        <BacktestCharts />
        <p className={styles.note}>Pick a year (or drag on the chart) to zoom — both lines restart at ₹20L on the selected date, so you can read the *relative* race over any window (e.g. select <b>2018</b> to see the gate sit in cash while the index runs). Equity on top (log scale), drawdown below. This is the validated history; the live book below tracks it forward.</p>
      </div>

      {/* Live P&L vs Nifty 50 */}
      <LivePnL s={s} />


      {/* Target basket — what it will buy at the next risk-on month-end */}
      {!riskOn && (
        <div className={styles.card}>
          <div className={styles.cardTitle}>Target basket — buys at next month-end once risk-on</div>
          <table className={styles.table}>
            <thead><tr><th>Momentum #</th><th>Stock</th><th>Size (by market cap)</th></tr></thead>
            <tbody>
              {s.target_basket.map((t) => (
                <tr key={t.symbol}>
                  <td className={styles.muted}>#{t.rank}</td>
                  <td className={styles.sym}>{t.symbol}</td>
                  <td><span className={`${styles.tier} ${t.tier === 'Nifty 50' ? styles.tierLg : t.tier === 'Next 50' ? styles.tierMd : styles.tierSm}`}>{t.tier}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className={styles.note}>
            Ranked #1–8 by momentum. Size tier is from official index membership (free-float market-cap defined):
            Nifty 50 ≈ mcap rank 1–50, Next 50 ≈ 51–100, Midcap ≈ 101–200. The book buys these equal-weight
            (~₹2.5L each) at the next month-end once NIFTYBEES reclaims its 100-DMA.
          </p>
        </div>
      )}

      {/* Closed trades */}
      <div className={styles.card}>
        <div className={styles.cardTitle}>Closed trades ({s.closed.length})</div>
        {s.closed.length === 0 ? (
          <div className={styles.chartEmpty}>No closed trades yet.</div>
        ) : (
          <table className={styles.table}>
            <thead><tr>
              <th>Stock</th><th>Entry</th><th>Exit</th><th>Entry ₹</th><th>Exit ₹</th>
              <th>Net P&L</th><th>%</th><th>Days</th><th>Why exited</th>
            </tr></thead>
            <tbody>
              {s.closed.map((c, i) => (
                <tr key={i}>
                  <td className={styles.sym}>{c.symbol}</td>
                  <td className={styles.muted}>{c.entry_date}</td>
                  <td className={styles.muted}>{c.exit_date}</td>
                  <td>{c.entry_price}</td>
                  <td>{c.exit_price}</td>
                  <td className={c.net_pnl >= 0 ? styles.pos : styles.neg} style={pnlTint(c.gross_pct)}>
                    {inr(c.net_pnl)}</td>
                  <td className={c.gross_pct >= 0 ? styles.pos : styles.neg} style={pnlTint(c.gross_pct)}>
                    {pct(c.gross_pct)}</td>
                  <td>{c.holding_days}</td>
                  <td><span className={styles.reason}>{reasonLabel[c.reason] || c.reason}</span></td>
                </tr>
              ))}
            </tbody>
            <tfoot>
              {(() => {
                const tp = s.closed.reduce((a, c) => a + (c.net_pnl || 0), 0);
                const tax = s.closed.reduce((a, c) => a + (c.stcg_tax || 0), 0);
                const wins = s.closed.filter((c) => c.net_pnl > 0).length;
                return (
                  <tr style={{ borderTop: '2px solid var(--hairline,rgba(0,0,0,0.14))', fontWeight: 700 }}>
                    <td>TOTAL ({s.closed.length} closed · {wins} winners)</td>
                    <td /><td /><td /><td />
                    <td className={tp >= 0 ? styles.pos : styles.neg}>{tp >= 0 ? '+' : ''}{inr(tp)}</td>
                    <td className={styles.muted} colSpan={3}>
                      after {inr(tax)} STCG · net {inr(tp - tax)}
                    </td>
                  </tr>
                );
              })()}
            </tfoot>
          </table>
        )}
      </div>

      {/* Rules + frequency */}
      <div className={styles.card}>
        <div className={styles.cardTitle}>How it works — rules, frequency &amp; action (all automated)</div>
        <p className={styles.summary}>
          <b>Universe = the Nifty 200</b> (the 200 largest NSE stocks by market cap) → <b>rank all 200 by momentum</b>
          {' '}(6-month + 12-month relative strength vs NIFTYBEES) → <b>hold the top 8</b>, equal-weight. A top-22
          retention buffer, a per-stock Donchian stop, and a market-regime cash gate then manage the book.
        </p>
        <table className={`${styles.table} ${styles.rulesTable}`}>
          <thead><tr><th>Rule</th><th>Frequency — check &amp; action</th><th>What happens</th></tr></thead>
          <tbody>
            {s.rules.map(([k, freq, v]) => (
              <tr key={k}>
                <td className={styles.sym}>{k}</td>
                <td className={styles.freq}>{freq}</td>
                <td className={styles.muted}>{v}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className={styles.note}>
          {s.live_mode
            ? 'LIVE — these jobs place real orders in the shared Zerodha account.'
            : 'Paper only — never places a real order.'}{' '}Last automated runs: daily {s.last_daily || '—'} ·
          weekly {s.last_weekly || '—'} · monthly {s.last_monthly || '—'}.
        </p>
      </div>
    </div>
  );
}


/** The backtest this book implements, shown ON the book so live results have something to be read
 *  against. Numbers are pulled from the study record itself rather than retyped here — a hardcoded
 *  copy would drift the moment the study is revised, and a stale backtest figure next to live P&L
 *  is worse than none. */
/** Headline block. Hierarchy, not a row of equal tiles: what the book is worth, where that value
 *  sits, then the P&L parts before the total they add up to. */
function BookSummary({ s }: { s: State }) {
  const gain = s.nav - s.capital;
  // CAGR / max-drawdown subline (display-only) from the book's own nav curve —
  // same headline language as the Open Alpha page (Arun, 02-Sep-2026).
  const nc = s.navcurve || [];
  let cagr: number | null = null, mdd: number | null = null, yrsSpan = 0;
  if (nc.length > 5) {
    yrsSpan = (Date.parse(nc[nc.length - 1].d) - Date.parse(nc[0].d)) / 3.15576e10;
    if (yrsSpan > 0.02) cagr = (Math.pow(nc[nc.length - 1].nav / nc[0].nav, 1 / yrsSpan) - 1) * 100;
    let peak = nc[0].nav, ddv = 0;
    for (const q of nc) { peak = Math.max(peak, q.nav); ddv = Math.min(ddv, q.nav / peak - 1); }
    mdd = ddv * 100;
  }
  const stocks = s.equity || 0;
  const etf = s.swept_value || 0;
  const idle = s.ledger_cash != null ? s.ledger_cash : Math.max(0, s.nav - stocks - etf);
  const total = stocks + etf + idle || 1;
  const segs = [
    { k: 'Stocks', v: stocks, c: '#2563EB' },
    { k: s.sweep?.symbol || 'Liquid ETF', v: etf, c: '#0891B2' },
    { k: 'Un-swept cash', v: idle, c: 'var(--ink-faint,#B4B2A9)' },
  ].filter((x) => x.v > 0);
  const yieldRs = s.interest_earned || 0;
  // Shown as a residual on purpose. Unrealised is measured before buy-side costs while realised is
  // already net of its sell-side ones, so the three parts land ~Rs900 above the real gain. Printing
  // a total under a divider that does not equal the rows above it is how a P&L block loses trust,
  // so the gap is named and shown rather than hidden.
  const costs = gain - (s.unrealized + s.realized_net + yieldRs);
  const rows = [
    { k: 'Unrealised', v: s.unrealized, hint: 'open positions, before entry costs' },
    { k: 'Realised (net)', v: s.realized_net, hint: 'closed trades, after costs' },
    { k: `${s.sweep?.symbol || 'Liquid'} yield`, v: yieldRs, hint: 'gain on parked cash' },
    { k: 'Costs & fees', v: costs, hint: 'brokerage, STT and stamp duty not already netted above' },
  ];
  const tone = (v: number) => (v > 0 ? 'var(--accent-pos,#0F6E56)' : v < 0 ? 'var(--accent-neg,#A32D2D)' : 'var(--ink,#1B1B1A)');
  const gapTone = s.gate_gap_pct == null ? 'var(--ink,#1B1B1A)'
    : s.gate_gap_pct < 0 ? 'var(--accent-neg,#A32D2D)'
    : s.gate_gap_pct < 2 ? 'var(--status-warning,#C97B20)' : 'var(--accent-pos,#0F6E56)';

  return (
    <div className={styles.bookSummary}>
      <div className={styles.sumMain}>
        <div className={styles.sumLabel}>Current value</div>
        <div className={styles.sumHero}>{inr(s.nav)}</div>
        <div className={styles.sumSub}>
          on <b>{inr(s.capital)}</b> invested{' '}
          <span style={{ color: tone(gain), fontWeight: 700 }}>
            {gain >= 0 ? '+' : '−'}{inr(Math.abs(gain))} · {pct(s.total_return_pct)}
          </span>
          {s.inception ? ` · since ${s.inception}` : ''}
        </div>
        {cagr != null && (
          <div className={styles.sumSub}>
            CAGR {pct(cagr)}{yrsSpan < 1 ? ' (annualized — early days)' : ''} · max drawdown {pct(mdd)} · updated {s.data_asof || '—'}
          </div>
        )}

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
          <span><b>{s.n_holdings}</b> holdings</span>
          <span><b>{s.invested_pct.toFixed(0)}%</b> deployed</span>
          <span>gate{' '}
            <b style={{ color: gapTone }}>
              {s.gate_gap_pct == null ? '—' : (s.gate_gap_pct >= 0 ? '+' : '') + s.gate_gap_pct.toFixed(2) + '%'}
            </b>{' '}vs 100-DMA
          </span>
          <span><b>{s.days_to_rebalance}d</b> to rebalance</span>
        </div>
      </div>

      <div className={styles.sumPnl}>
        <div className={styles.sumLabel}>Profit &amp; loss</div>
        {rows.map((r) => (
          <div key={r.k} className={styles.pnlRow} title={r.hint}>
            <span>{r.k}</span>
            <b style={{ color: tone(r.v) }}>{r.v >= 0 ? '+' : '−'}{inr(Math.abs(r.v))}</b>
          </div>
        ))}
        <div className={`${styles.pnlRow} ${styles.pnlTotal}`}>
          <span>Total return</span>
          <b style={{ color: tone(gain) }}>
            {gain >= 0 ? '+' : '−'}{inr(Math.abs(gain))} · {pct(s.total_return_pct)}
          </b>
        </div>
      </div>
    </div>
  );
}

function BacktestEvidence() {
  const study = getStudy('momentum30-subselect');
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
          {study.date ? ` · ${study.date}` : ''} — this is the study the live book implements, not
          live performance
        </span>
        <a className={styles.studyLink} href="/app/backtest/momentum30-subselect">
          The backtest this book runs
        </a>
        <a className={styles.studyLink} href="/app/backtest/momentum-universe-bakeoff">Universe bake-off</a>
        <a className={styles.studyLink} href="/app/backtest/momentum-put-hedge-overlay">Put-hedge overlay</a>
        <a className={styles.studyLink} href="/app/backtest/momentum-250-leverage-frontier">Leverage frontier</a>
        <a className={styles.studyLink} href="/app/strategies#momentum-3l">Register entry</a>
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
        {m.map((x) => (
          <div key={x.label} className={styles.evidenceCell} title={x.hint || ''}>
            <div className={styles.evidenceVal}
                 style={{ color: x.tone === 'pos' ? 'var(--accent-pos,#0F6E56)'
                                : x.tone === 'neg' ? 'var(--accent-neg,#A32D2D)'
                                : 'var(--ink,#1B1B1A)' }}>{x.value}</div>
            <div className={styles.evidenceLab}>{x.label}</div>
            {x.hint && <div className={styles.evidenceHint}>{x.hint}</div>}
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

function LivePnL({ s }: { s: State }) {
  return (
    <div className={styles.card}>
      <div className={styles.cardTitle}>
        P&amp;L since go-live — book vs market indices
        <span style={{ fontSize: 11.5, fontWeight: 400, color: 'var(--ink-muted,#888)', marginLeft: 8 }}>
          percent return, deposits excluded{s.inception ? ` · live since ${s.inception}` : ''}
        </span>
      </div>
      <LiveCurve />
    </div>
  );
}

function CashStatus({ s }: { s: State }) {
  // idle_cash now INCLUDES the swept CASHIETF value (it is cash-equivalent, redeemable same-day),
  // so subtracting sweep.value again would understate what the book can actually deploy.
  const deployable = Math.max(0, s.idle_cash - s.cash_reserve);
  const heavy = s.idle_pct >= 25;
  const cell: React.CSSProperties = {
    background: 'var(--canvas,#FAFAF9)', border: '1px solid var(--hairline,rgba(0,0,0,0.10))',
    borderRadius: 8, padding: '10px 13px',
  };
  const lab: React.CSSProperties = { fontSize: 11.5, color: 'var(--ink-muted,#888)' };
  return (
    <div className={styles.card}>
      <div className={styles.cardTitle}>Cash &amp; redeployment</div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(160px,1fr))', gap: 10 }}>
        <div style={cell}><div style={lab}>Idle cash</div>
          <div style={{ fontSize: 17, fontWeight: 700, color: heavy ? '#d97706' : 'inherit' }}>
            {inr(s.idle_cash)}</div>
          <div style={lab}>{s.idle_pct}% of book</div></div>
        <div style={cell}><div style={lab}>Awaiting redeployment</div>
          <div style={{ fontSize: 17, fontWeight: 700 }}>{inr(deployable)}</div>
          <div style={lab}>deployed at the month-end rebalance</div></div>
        <div style={cell}><div style={lab}>Days to rebalance</div>
          <div style={{ fontSize: 17, fontWeight: 700 }}>{s.days_to_rebalance}</div>
          <div style={lab}>then empty slots are refilled to 8</div></div>
        <div style={cell}><div style={lab}>Hedge reserve (3%)</div>
          <div style={{ fontSize: 17, fontWeight: 700, color: '#1f9d55' }}>{inr(s.cash_reserve)}</div>
          <div style={lab}>never deployed — keeps the put fundable</div></div>
        {s.sweep?.enabled && (
          <div style={cell}><div style={lab}>Parked in {s.sweep.symbol}</div>
            <div style={{ fontSize: 17, fontWeight: 700 }}>{inr(s.sweep.value)}</div>
            <div style={lab}>{s.sweep.units} units · sold before the rebalance</div></div>
        )}
      </div>
      <div style={{ fontSize: 12.5, color: 'var(--ink-muted,#888)', marginTop: 10 }}>
        {heavy
          ? `A large idle balance is normal after stop-outs — research/108 showed redeploying sooner than month-end HALVES the Calmar (0.91→0.45), because empty slots exist precisely when the market is falling. The cash waits ${s.days_to_rebalance} days by design.`
          : 'Stop-out proceeds wait for the month-end rebalance rather than being redeployed immediately — refilling faster was tested and performs much worse.'}
        {!s.sweep?.enabled && ' Liquid-fund sweep is OFF, so in LIVE this cash would earn nothing.'}
      </div>
    </div>
  );
}

function HedgePanel({ s }: { s: State }) {
  const h = s.hedge;
  const hist = s.hedge_closed || [];
  const riskOn = s.gate !== 'OFF';
  const totPnl = hist.reduce((a, x) => a + x.pnl, 0);
  const wins = hist.filter((x) => x.pnl > 0).length;
  const box: React.CSSProperties = {
    background: 'var(--canvas,#FAFAF9)', border: '1px solid var(--hairline,rgba(0,0,0,0.10))',
    borderRadius: 8, padding: '10px 13px', fontSize: 13,
  };
  return (
    <div className={styles.card}>
      <div className={styles.cardTitle}>
        Downside hedge — bi-weekly NIFTY puts @ 2× equity
        <span style={{
          marginLeft: 10, padding: '2px 9px', borderRadius: 5, fontSize: 11, fontWeight: 700,
          color: '#fff', background: h ? '#c0392b' : '#39424e',
        }}>{h ? 'HEDGE ACTIVE' : (s.hedge_viability && !s.hedge_viability.viable ? 'NOT VIABLE YET' : 'NOT HEDGED')}</span>
      </div>
      {h ? (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(150px,1fr))', gap: 10, marginBottom: 10 }}>
            <div style={box}><b>{h.tsym}</b><div style={{ color: 'var(--ink-muted,#888)', fontSize: 12 }}>
              {h.lots ?? '—'} lot{(h.lots ?? 0) === 1 ? '' : 's'} · {h.qty} qty · {h.dte}d to expiry</div></div>
            <div style={box}>Premium paid<div><b>{inr(h.cost)}</b> @ {h.entry_price}</div></div>
            <div style={box}>Now worth<div><b>{inr(h.value)}</b>{h.price ? ` @ ${h.price}` : ''}</div></div>
            <div style={box}>Hedge P&amp;L<div style={{ color: h.pnl >= 0 ? '#1f9d55' : '#c0392b' }}>
              <b>{h.pnl >= 0 ? '+' : ''}{inr(h.pnl)}</b> ({pct(h.pnl_pct)})</div></div>
          </div>
          <div style={{ fontSize: 12.5, color: 'var(--ink-muted,#888)' }}>
            Bought {h.entry_date} on the risk-off gate. Strike {h.strike}, expiry {h.expiry}. It re-sizes if
            holdings stop out, rolls at expiry while the gate stays risk-off, and is sold when the gate turns risk-on.
          </div>
        </>
      ) : (
        <div style={{ fontSize: 12.5, color: 'var(--ink-muted,#888)' }}>
          {riskOn
            ? 'Gate is risk-ON, so the book is fully invested and unhedged — as intended. A put is bought automatically at the next risk-off gate.'
            : 'Gate is risk-OFF but no hedge is open — check the alert email (premium may have exceeded the cash or the 6%-of-NAV cap).'}
        </div>
      )}
      {!h && s.hedge_viability && !s.hedge_viability.viable && (() => {
        const v = s.hedge_viability!;
        const progress = v.capital_needed ? Math.min(100, (v.capital_now / v.capital_needed) * 100) : 0;
        return (
          <div style={{ marginTop: 12, ...box }}>
            <div style={{ fontWeight: 650, marginBottom: 6 }}>
              Why it is not running yet — one NIFTY lot is bigger than this book
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(160px,1fr))', gap: 8, marginBottom: 10 }}>
              <div>1 lot ({v.lot} × {v.spot.toLocaleString('en-IN')})<div><b>{lakh(v.lot_notional)}</b> of notional</div></div>
              <div>Book equity<div><b>{lakh(v.equity)}</b></div></div>
              <div>{v.ratio}× target<div><b>{lakh(v.target_notional)}</b></div></div>
              <div>Lots that buys<div><b style={{ color: '#c0392b' }}>{v.lots_needed}</b> — must be a whole 1</div></div>
            </div>
            <div style={{ fontSize: 12.5, color: 'var(--ink-muted,#888)', marginBottom: 10, lineHeight: 1.55 }}>
              Buying one lot anyway would hedge <b>{v.over_hedge_x}× equity</b>, not {v.ratio}×. That is the
              research/105 failure: as holdings stop out the fixed put position stops being a hedge and becomes
              a net-short directional bet. So the book stays unhedged until a whole lot fits.
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12.5, marginBottom: 4 }}>
              <span>Capital now <b>{lakh(v.capital_now)}</b></span>
              <span>Hedge unlocks at <b>{v.capital_needed ? lakh(v.capital_needed) : '—'}</b>
                {v.shortfall ? <span style={{ color: '#c0392b' }}> · {lakh(v.shortfall)} to go</span> : null}</span>
            </div>
            <div style={{ height: 7, borderRadius: 4, background: 'var(--hairline,rgba(0,0,0,0.10))', overflow: 'hidden' }}>
              <div style={{ width: `${progress}%`, height: '100%', background: '#1f9d55' }} />
            </div>
            <div style={{ fontSize: 11.5, color: 'var(--ink-muted,#888)', marginTop: 6 }}>
              Needs equity of {lakh(v.equity_for_one_lot)} (half a lot's notional at {v.ratio}×). Target moves
              with NIFTY, so it is recomputed live rather than fixed. Tenor when it starts: {v.dte_target}-day
              (bi-weekly) puts, per research/105.
            </div>
          </div>
        );
      })()}
      {hist.length > 0 && (
        <details style={{ marginTop: 12 }}>
          <summary style={{ cursor: 'pointer', fontSize: 13 }}>
            Past hedges ({hist.length}) — {wins} profitable, net{' '}
            <b style={{ color: totPnl >= 0 ? '#1f9d55' : '#c0392b' }}>{totPnl >= 0 ? '+' : ''}{inr(totPnl)}</b>
          </summary>
          <div style={{ maxHeight: 260, overflow: 'auto', marginTop: 8 }}>
            <table className={styles.table}>
              <thead><tr><th>Contract</th><th>Held</th><th>Premium</th><th>Exit</th><th>P&amp;L</th><th>Why closed</th></tr></thead>
              <tbody>
                {hist.map((x) => (
                  <tr key={x.id}>
                    <td>{x.tsym}</td>
                    <td>{x.entry_date} → {x.exit_date}</td>
                    <td>{inr(x.cost)}</td>
                    <td>{inr(x.proceeds)}</td>
                    <td style={{ color: x.pnl >= 0 ? '#1f9d55' : '#c0392b' }}>{x.pnl >= 0 ? '+' : ''}{inr(x.pnl)}</td>
                    <td style={{ color: 'var(--ink-muted,#888)' }}>{x.reason}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
      )}
    </div>
  );
}

/* eslint-disable @typescript-eslint/no-explicit-any */
function CashPanel({ s, reload }: { s: State; reload: () => void }) {
  const [rec, setRec] = useState<any>(null);
  const [dep, setDep] = useState<{ amount: string; mode: string; plan: any; busy: boolean }>({ amount: '', mode: 'immediate', plan: null, busy: false });
  const [wd, setWd] = useState<{ amount: string; plan: any; busy: boolean }>({ amount: '', plan: null, busy: false });

  // An even top-up splits the deposit equally, but CNC buys WHOLE shares — so every holding needs at
  // least one share of the DEAREST name or its whole slice falls back to cash (2026-08-14: POWERINDIA
  // at Rs35,760 swallowed Rs20,000 of a Rs1L deposit without a word).
  // fall back to entry_price: the live quote is null whenever the Kite token is stale (weekends,
  // before the 08:5x auto-login) and the hint must not silently vanish then.
  const shPx = (h: Holding) => h.price ?? h.entry_price ?? 0;
  const stocks = s.holdings.filter((h) => !h.is_cash && shPx(h) > 0);
  const dearest = stocks.reduce<Holding | null>((a, h) => (shPx(h) > (a ? shPx(a) : 0) ? h : a), null);
  const minFull = dearest ? Math.ceil(shPx(dearest) * stocks.length) : 0;
  const depAmt = parseFloat(dep.amount);
  const perSlot = depAmt > 0 && stocks.length ? depAmt / stocks.length : 0;
  const skipped = perSlot > 0 ? stocks.filter((h) => perSlot < shPx(h)) : [];
  const willDeploy = perSlot > 0
    ? stocks.reduce((a, h) => a + Math.floor(perSlot / shPx(h)) * shPx(h), 0)
    : 0;

  const previewDep = async () => {
    const amt = parseFloat(dep.amount); if (!(amt > 0)) return;
    setDep((d) => ({ ...d, busy: true }));
    const r: any = await apiPost('/api/momentum-paper/deposit', { amount: amt, mode: dep.mode, dry_run: true }).catch(() => null);
    setDep((d) => ({ ...d, plan: r, busy: false }));
  };
  const execDep = async () => {
    const amt = parseFloat(dep.amount);
    if (!window.confirm(`Deposit ₹${amt.toLocaleString('en-IN')} (${dep.mode})? This ${s.live_mode ? 'PLACES REAL ORDERS' : 'updates the paper book'}.`)) return;
    setDep((d) => ({ ...d, busy: true }));
    await apiPost('/api/momentum-paper/deposit', { amount: amt, mode: dep.mode, dry_run: false }).catch(() => null);
    setDep({ amount: '', mode: 'immediate', plan: null, busy: false }); reload();
  };
  const previewWd = async () => {
    const amt = parseFloat(wd.amount); if (!(amt > 0)) return;
    setWd((w) => ({ ...w, busy: true }));
    const r: any = await apiPost('/api/momentum-paper/withdraw', { amount: amt, dry_run: true }).catch(() => null);
    setWd((w) => ({ ...w, plan: r, busy: false }));
  };
  const execWd = async () => {
    const amt = parseFloat(wd.amount);
    if (!window.confirm(`Withdraw ₹${amt.toLocaleString('en-IN')}? This ${s.live_mode ? 'PLACES REAL SELL ORDERS' : 'updates the paper book'}.`)) return;
    setWd((w) => ({ ...w, busy: true }));
    await apiPost('/api/momentum-paper/withdraw', { amount: amt, dry_run: false }).catch(() => null);
    setWd({ amount: '', plan: null, busy: false }); reload();
  };
  const reconcile = async () => setRec(await apiGet('/api/momentum-paper/reconcile').catch((e) => ({ error: String(e) })));

  const btn: CSSProperties = { padding: '7px 12px', borderRadius: 6, border: '1px solid var(--hairline,rgba(0,0,0,0.10))', cursor: 'pointer', fontSize: 13, background: 'var(--canvas,#FAFAF9)', color: 'var(--ink,#1B1B1A)', fontWeight: 600 };
  const inp: CSSProperties = { padding: '7px 10px', borderRadius: 6, border: '1px solid var(--hairline,rgba(0,0,0,0.10))', background: 'var(--surface,#FFFFFF)', color: 'var(--ink,#1B1B1A)', width: 140, fontSize: 13 };

  return (
    <div className={styles.card}>
      <div className={styles.cardTitle}>Live controls &amp; cash management</div>
      <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap', marginBottom: 10 }}>
        <span>Mode: <b style={{ color: s.live_mode ? '#c0392b' : 'inherit' }}>{s.mode}</b></span>
        <button style={{ ...btn, opacity: 0.5, cursor: 'not-allowed' }} disabled title="Arming is gated until the Kite account + two-key safety (Phase 0)">
          Go LIVE — 🔒 gated (Phase 0)
        </button>
        <button style={btn} onClick={reconcile}>Reconcile vs broker</button>
        {rec && (
          <span style={{ fontSize: 12, color: rec.match ? '#1f9d55' : rec.error ? '#d97706' : '#c0392b' }}>
            {rec.live === false ? 'paper mode — reconcile is live-only' : rec.error ? `error: ${rec.error}` : rec.match ? '✓ book matches broker' : `⚠ mismatch: ${JSON.stringify(rec.diffs)}`}
          </span>
        )}
      </div>
      <div style={{ fontSize: 12, color: 'var(--ink-muted,#888)', marginBottom: 14 }}>
        {s.live_mode
          ? 'ARMED AND LIVE — deposits, withdrawals and every scheduled job place real orders in the shared Zerodha account. The sell guard never sells more than the quantity this book itself recorded, so personal holdings in the same account cannot be touched.'
          : 'Live-arming is disabled: the cash tools below run in paper and carry straight to live once armed.'}
      </div>
      <div className={styles.grid2}>
        <div>
          <div style={{ fontWeight: 600, marginBottom: 6 }}>Add funds (deposit)</div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
            <input style={inp} placeholder="₹ amount" value={dep.amount} onChange={(e) => setDep((d) => ({ ...d, amount: e.target.value, plan: null }))} />
            <select style={{ ...inp, width: 'auto' }} value={dep.mode} onChange={(e) => setDep((d) => ({ ...d, mode: e.target.value, plan: null }))}>
              <option value="immediate">Immediate equal top-up (default)</option>
              <option value="park">Park → deploy at next rebalance</option>
            </select>
            <button style={btn} onClick={previewDep} disabled={dep.busy}>Preview</button>
          </div>
          {stocks.length > 0 && dearest && (
            <div style={{ marginTop: 6, fontSize: 12, color: 'var(--ink-muted,#888)', lineHeight: 1.55 }}>
              An even top-up needs <b>{inr(minFull)}</b> to reach all {stocks.length} holdings — whole
              {' '}shares only, so every slot needs one share of the dearest name
              {' '}({dearest.symbol} at {inr(shPx(dearest))}). Below that, the priciest names are
              {' '}skipped and their slice stays in cash.
              {perSlot > 0 && (
                <div style={{ marginTop: 3, color: skipped.length ? '#c0392b' : '#1f9d55' }}>
                  {dep.mode === 'immediate' ? '' : 'Deployed immediately, '}
                  {inr(depAmt)} splits to {inr(perSlot)}/name → ~<b>{inr(willDeploy)}</b> buys stock,
                  {' '}~<b>{inr(Math.max(0, depAmt - willDeploy))}</b> stays in cash
                  {skipped.length > 0 && (
                    <> · skipped: <b>{skipped.map((h) => h.symbol).join(', ')}</b></>
                  )}
                </div>
              )}
            </div>
          )}
          {dep.plan && (
            <div style={{ marginTop: 8, fontSize: 12.5 }}>
              <div style={{ color: 'var(--ink-muted,#888)' }}>{dep.plan.note}</div>
              {dep.plan.plan?.length > 0 && (
                <ul style={{ margin: '4px 0', paddingLeft: 18 }}>{dep.plan.plan.map((p: any, i: number) => <li key={i}>BUY {p.qty} {p.symbol} (~{inr(p.value)})</li>)}</ul>
              )}
              <button style={{ ...btn, background: '#1f9d55', color: '#fff', marginTop: 4 }} onClick={execDep} disabled={dep.busy}>Confirm &amp; deploy</button>
            </div>
          )}
        </div>
        <div>
          <div style={{ fontWeight: 600, marginBottom: 6 }}>Withdraw funds</div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
            <input style={inp} placeholder="₹ amount" value={wd.amount} onChange={(e) => setWd((w) => ({ ...w, amount: e.target.value, plan: null }))} />
            <button style={btn} onClick={previewWd} disabled={wd.busy}>Preview plan</button>
          </div>
          {wd.plan && (
            <div style={{ marginTop: 8, fontSize: 12.5 }}>
              <div style={{ color: 'var(--ink-muted,#888)' }}>Raises {inr(wd.plan.raised)}{wd.plan.shortfall > 0 ? ` (short ${inr(wd.plan.shortfall)})` : ''} — sells the weakest names first, keeps the winners.</div>
              <ul style={{ margin: '4px 0', paddingLeft: 18 }}>{wd.plan.plan?.map((p: any, i: number) => <li key={i}>{p.source === 'idle cash' ? `Idle cash ${inr(p.value)}` : `SELL ${p.qty} ${p.source} (~${inr(p.value)}, score ${p.score})`}</li>)}</ul>
              <button style={{ ...btn, background: '#c0392b', color: '#fff', marginTop: 4 }} onClick={execWd} disabled={wd.busy}>Confirm &amp; withdraw</button>
            </div>
          )}
        </div>
      </div>
      <DailyPerformance book="momentum-3l" title="Daily performance · trade history" />

    </div>
  );
}

// Fold-marker-done
