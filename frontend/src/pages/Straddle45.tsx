/* 45-DTE NIFTY short straddle — research/119.
 *
 * NOT ARMED. There is no executor and no book behind this page; it is the
 * control room for a strategy that has passed G3 but still owes a stress-margin
 * test before real money. The positions table is wired to the same shape the
 * live books use so it fills in the day it is armed, and reads empty until then.
 *
 * Everything numeric here scales off the lot selector. The study is stored in
 * POINTS (lot-size agnostic); rupees are derived as points x 65 x lots, so
 * changing lots re-prices the whole page — returns, drawdown, margin and payoff.
 */
import { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import styles from './Straddle45.module.css';

const LOT = 65;                     // NIFTY contract size (Kite instrument master, 2026-08-24)
const STUDY = '/app/backtest/nifty-45dte-short-straddle';

/* ---- the study, in points. Real NSE bhavcopy prices, daily-close monitoring,
   Jan-2019 -> Jun-2026, 89 non-overlapping monthly campaigns. ---------------- */
const YEARS = [
  { y: '2019', n: 12, pts: 15.2, dd: -337.0, win: 66.7 },
  { y: '2020', n: 12, pts: 860.7, dd: -466.3, win: 58.3 },
  { y: '2021', n: 12, pts: 928.3, dd: -489.8, win: 66.7 },
  { y: '2022', n: 12, pts: 1039.4, dd: -452.3, win: 75.0 },
  { y: '2023', n: 12, pts: -240.1, dd: -998.4, win: 66.7 },
  { y: '2024', n: 12, pts: 971.2, dd: -212.2, win: 66.7 },
  { y: '2025', n: 12, pts: 1915.5, dd: -564.1, win: 83.3 },
  { y: '2026 H1', n: 5, pts: 1462.2, dd: 0.0, win: 100.0 },
];
const TOTAL_PTS = 6952.4;
const MAXDD_PTS = -998.4;
const WORST_TRADE_PTS = -811.8;
const AVG_CREDIT = 786.3;
const AVG_DAYS = 24.2;
const WIN_RATE = 70.8;
const TSTAT = 3.03;
const N_TRADES = 89;
const YEARS_SPAN = 7.48;

/* ---- margin, measured live from Kite basket_order_margins on 2026-08-24,
   NIFTY ATM straddle, consider_positions=false. NRML == MIS to the rupee. ---- */
const MARGIN_ASOF = '2026-08-24';
const MARGIN_BY_DTE = [
  { dte: 1, perLot: 215348 },
  { dte: 8, perLot: 212973 },
  { dte: 15, perLot: 212583 },
  { dte: 22, perLot: 213432 },
  { dte: 29, perLot: 228843 },
  { dte: 36, perLot: 216043 },
  { dte: 45, perLot: 224328, interp: true },
  { dte: 64, perLot: 241789 },
];
/* Margin inflates when the position moves against you — measured at DTE 22 by
   pricing the same structure at strikes away from spot. This is the number that
   actually sizes the book, because it bites in the same event as the drawdown. */
const MARGIN_STRESS = [
  { label: 'At entry (ATM, ~45 DTE)', perLot: 224328, freq: null },
  { label: 'Peak tenor in the 45→21 hold', perLot: 228843, freq: null },
  { label: 'After a 2% adverse move', perLot: 252228, freq: 87.6 },
  { label: 'After a 3% adverse move', perLot: 268974, freq: 66.3 },
  { label: 'After a 5% adverse move', perLot: 300135, freq: 29.2 },
  { label: 'After a 7% adverse move', perLot: 329045, freq: null },
];
const SIZING_MARGIN = 268974;      // survive a 3% adverse move — the base case

const inr = (v: number) => (v < 0 ? '-' : '') + '₹' + Math.abs(Math.round(v)).toLocaleString('en-IN');
const lakh = (v: number) => (v < 0 ? '-' : '') + '₹' + (Math.abs(v) / 1e5).toFixed(2) + 'L';
const pct = (v: number) => (v >= 0 ? '+' : '') + v.toFixed(1) + '%';

type Position = {
  leg: string; strike: number; expiry: string; dte: number; qty: number;
  entry_date: string; entry_price: number; price: number | null; pnl: number;
};

export default function Straddle45() {
  const [lots, setLots] = useState<number>(() => {
    try {
      const v = localStorage.getItem('straddle45.lots');
      const n = v ? parseInt(v, 10) : NaN;
      return Number.isFinite(n) && n >= 1 && n <= 20 ? n : 3;
    } catch { return 3; }
  });
  const setLotsPersist = (n: number) => {
    setLots(n);
    try { localStorage.setItem('straddle45.lots', String(n)); } catch { /* private mode */ }
  };

  const positions: Position[] = [];      // no executor yet — see the banner

  const m = useMemo(() => {
    const qty = LOT * lots;
    const rupee = (p: number) => p * qty;
    const marginAtEntry = MARGIN_BY_DTE.find((x) => x.dte === 45)!.perLot * lots;
    const buffer = 2 * Math.abs(MAXDD_PTS) * LOT * lots;
    const capital = SIZING_MARGIN * lots + buffer;
    const total = rupee(TOTAL_PTS);
    const maxdd = rupee(Math.abs(MAXDD_PTS));
    const cagr = (Math.pow((capital + total) / capital, 1 / YEARS_SPAN) - 1) * 100;
    const ddPct = (maxdd / capital) * 100;
    return {
      qty, rupee, marginAtEntry, buffer, capital, total, maxdd,
      cagr, ddPct, calmar: ddPct ? cagr / ddPct : 0,
      worst: rupee(Math.abs(WORST_TRADE_PTS)),
      credit: rupee(AVG_CREDIT),
      headroom: capital / marginAtEntry,
      maxLots: Math.floor(capital / (SIZING_MARGIN + 2 * Math.abs(MAXDD_PTS) * LOT)),
    };
  }, [lots]);

  return (
    <div className={styles.root}>
      <div className={styles.headerRow}>
        <div>
          <h1 className={styles.title}>45-DTE NIFTY Short Straddle</h1>
          <p className={styles.sub}>
            Sell the <b>ATM straddle 45 calendar days</b> before the NIFTY monthly expiry,
            close it at <b>21 DTE</b> — collecting the fat part of the decay and leaving before
            gamma turns. Replication of “The Long &amp; The Short Ep. 48” on real NSE bhavcopy
            prices: <b>+78.1 pts/trade, t = {TSTAT}</b> across {N_TRADES} non-overlapping campaigns.{' '}
            <Link className={styles.link} to="/backtest/nifty-45dte-short-straddle">Full study →</Link>
          </p>
        </div>
        <div className={`${styles.gateBadge} ${styles.off}`}>
          <span className={styles.dot} />
          NOT ARMED — study only
        </div>
      </div>

      <div className={styles.warn}>
        <b>No executor is running for this system.</b> It has passed robustness (G3) but still owes
        a <b>stress-margin test</b> — SPAN inflates in the same event that drives the drawdown, and
        that has not been reconstructed across 2019–26 yet. This page is the control room and the
        sizing calculator; it does not place, hold or monitor anything. The positions table below
        will populate the day it is armed.
      </div>

      {/* ── lot configuration — everything on this page scales from here ────── */}
      <div className={styles.card}>
        <div className={styles.cardTitle}>Position size</div>
        <div className={styles.lotRow}>
          <div className={styles.lotPick}>
            {[1, 2, 3, 4, 5, 6].map((n) => (
              <button
                key={n}
                className={`${styles.lotBtn} ${n === lots ? styles.lotOn : ''}`}
                onClick={() => setLotsPersist(n)}
                aria-pressed={n === lots}
              >
                {n}
              </button>
            ))}
            <span className={styles.lotUnit}>lots · {m.qty} qty · 1 pt = {inr(LOT * lots)}</span>
          </div>
          {lots > m.maxLots && (
            <span className={styles.overSize}>
              over the {m.maxLots}-lot ceiling for this capital
            </span>
          )}
        </div>
        <div className={styles.gateRow}>
          <div><b>{lakh(m.marginAtEntry)}</b><span className={styles.muted}>margin at entry (45 DTE)</span></div>
          <div><b>{lakh(m.buffer)}</b><span className={styles.muted}>drawdown buffer (2× MaxDD)</span></div>
          <div><b>{lakh(m.capital)}</b><span className={styles.muted}>capital to block</span></div>
          <div><b>{m.headroom.toFixed(2)}×</b><span className={styles.muted}>margin headroom</span></div>
          <div><b>{m.maxLots}</b><span className={styles.muted}>max lots at this capital</span></div>
        </div>
        <p className={styles.note}>
          Capital is sized on the <b>3%-adverse-move margin</b> ({inr(SIZING_MARGIN)}/lot), not the
          entry margin — the underlying moves ≥3% from the entry anchor in <b>66% of campaigns</b>,
          so it is the base case, not a tail. Buffer is 2× the historical max drawdown.
        </p>
      </div>

      {/* ── KPI strip, scaled to the configured lots ────────────────────────── */}
      <div className={styles.kpis}>
        <Kpi label="CAGR" value={m.cagr.toFixed(2) + '%'} tone="pos" />
        <Kpi label="Max drawdown" value={'-' + m.ddPct.toFixed(1) + '%'} tone="neg" />
        <Kpi label="Calmar" value={m.calmar.toFixed(2)} tone="" />
        <Kpi label="Total net" value={lakh(m.total)} tone="pos" />
        <Kpi label="Max DD ₹" value={lakh(m.maxdd)} tone="neg" />
        <Kpi label="Worst trade" value={lakh(m.worst)} tone="neg" />
        <Kpi label="Win rate" value={WIN_RATE + '%'} tone="" />
        <Kpi label="Avg credit" value={lakh(m.credit)} tone="" />
        <Kpi label="Avg hold" value={AVG_DAYS + ' d'} tone="" />
      </div>

      {/* ── positions ───────────────────────────────────────────────────────── */}
      <div className={styles.card}>
        <div className={styles.cardTitle}>Open position</div>
        <table className={styles.table}>
          <thead><tr>
            <th>Leg</th><th>Strike</th><th>Expiry</th><th>DTE</th><th>Qty</th>
            <th>Entry</th><th>Entry ₹</th><th>Now ₹</th><th>P&amp;L ₹</th>
          </tr></thead>
          <tbody>
            {positions.length === 0 ? (
              <tr><td colSpan={9} className={styles.emptyCell}>
                Nothing open — the system is not armed. When it is, both legs appear here with
                live marks, exactly as the Momentum and NAS books show theirs.
              </td></tr>
            ) : positions.map((p) => (
              <tr key={p.leg}>
                <td className={styles.sym}>{p.leg}</td>
                <td>{p.strike}</td>
                <td className={styles.muted}>{p.expiry}</td>
                <td>{p.dte}</td>
                <td>{p.qty}</td>
                <td className={styles.muted}>{p.entry_date}</td>
                <td>{p.entry_price}</td>
                <td>{p.price ?? '—'}</td>
                <td className={p.pnl >= 0 ? styles.pos : styles.neg}>{inr(p.pnl)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* ── payoff ──────────────────────────────────────────────────────────── */}
      <div className={styles.card}>
        <div className={styles.cardTitle}>
          Payoff at expiry — {lots} lot{lots > 1 ? 's' : ''}, average credit {AVG_CREDIT} pts
        </div>
        <Payoff credit={AVG_CREDIT} qty={m.qty} />
        <p className={styles.note}>
          Drawn at expiry for shape. <b>The rule never gets there</b> — it closes at 21 DTE, which is
          why the realised loss distribution is far tamer than this diagram suggests. Breakevens sit
          at ±{AVG_CREDIT} points ({(AVG_CREDIT / 24170 * 100).toFixed(1)}% of spot); the 200% stop
          fires long before either.
        </p>
      </div>

      {/* ── rules ───────────────────────────────────────────────────────────── */}
      <div className={styles.grid2}>
        <div className={styles.card}>
          <div className={styles.cardTitle}>Rules</div>
          <div className={styles.rules}>
            <Rule k="Instrument" v="NIFTY monthly expiry — the last expiry of the month already listed 45 days out (from 2025 a weekly can expire AFTER the monthly; the listing test separates them)." />
            <Rule k="Entry" v="Expiry − 45 calendar days, at the close. Sell 1× ATM CE + 1× ATM PE. ATM = strike nearest spot. Both legs must have actually traded that day." />
            <Rule k="Size" v={`${lots} lot${lots > 1 ? 's' : ''} = ${m.qty} qty. Fixed — the rule does not compound.`} />
            <Rule k="Monitoring" v="Hourly candle closes. Nothing below 60-min changes a single trade — confirmed on 28.3M real 1-minute quotes." />
            <Rule k="Optional filter" v="India VIX rank > 25 vs the previous 252 sessions. Fewer trades, no losing year, better Calmar. Off by default." />
          </div>
        </div>
        <div className={styles.card}>
          <div className={styles.cardTitle}>Exits — first to trigger</div>
          <div className={styles.rules}>
            <Rule k="Target — 50%" v={`Combined premium ≤ 50% of entry credit. Fires once in ${N_TRADES} trades.`} />
            <Rule k="Stop — 200%" v={`Combined premium ≥ 200% of entry credit. Fires 2–3 times in ${N_TRADES} trades.`} />
            <Rule k="Time — 21 DTE" v="Expiry − 21 calendar days. This is how 85 of 89 trades end, and it is the whole design." />
            <Rule k="Do NOT delta-manage" v="Every move-threshold exit and re-centring scheme tested is worse than holding. Cycles cut on a move realise −28.6 pts; cycles left alone earn +83.0. To cut risk, cut lots." />
          </div>
        </div>
      </div>

      {/* ── management ──────────────────────────────────────────────────────── */}
      <div className={styles.card}>
        <div className={styles.cardTitle}>Management</div>
        <table className={styles.table}>
          <thead><tr><th>When</th><th>Check</th><th>Action</th></tr></thead>
          <tbody>
            <tr><td className={styles.freq}>Entry day, 15:15</td><td>Contract listed 45 days out; both ATM legs traded today</td><td>Sell the ATM straddle, {lots} lots. Record the credit — every exit is a ratio to it.</td></tr>
            <tr><td className={styles.freq}>Hourly, each session</td><td>Combined premium vs 50% / 200% of credit</td><td>Close both legs together if either level is breached.</td></tr>
            <tr><td className={styles.freq}>On an adverse move</td><td>Margin inflates {'>'}25% once spot is 3% away</td><td><b>Do nothing to the position.</b> Ensure the buffer is funded — this is what it is for.</td></tr>
            <tr><td className={styles.freq}>Expiry − 21 days</td><td>Position still open</td><td>Close at the day&apos;s close regardless of P&amp;L. No extensions — holding to expiry triples the drawdown.</td></tr>
            <tr><td className={styles.freq}>Never</td><td>Rolling, re-centring, one-leg adjustment</td><td>Not part of this system. Tested and refuted.</td></tr>
          </tbody>
        </table>
      </div>

      {/* ── yearly, scaled ──────────────────────────────────────────────────── */}
      <div className={styles.card}>
        <div className={styles.cardTitle}>
          Year by year at {lots} lot{lots > 1 ? 's' : ''} — capital {lakh(m.capital)}
        </div>
        <table className={styles.table}>
          <thead><tr>
            <th>Year</th><th>Trades</th><th>Net pts</th><th>Net ₹</th><th>Return</th>
            <th>Intra-year DD</th><th>Win%</th><th>Equity end</th>
          </tr></thead>
          <tbody>
            {(() => {
              let eq = m.capital;
              return YEARS.map((r) => {
                const rs = m.rupee(r.pts);
                eq += rs;
                const ret = (rs / m.capital) * 100;
                const ddp = (m.rupee(r.dd) / m.capital) * 100;
                return (
                  <tr key={r.y}>
                    <td className={styles.sym}>{r.y}</td>
                    <td>{r.n}</td>
                    <td className={r.pts >= 0 ? styles.pos : styles.neg}>{r.pts.toFixed(1)}</td>
                    <td className={rs >= 0 ? styles.pos : styles.neg}>{inr(rs)}</td>
                    <td className={ret >= 0 ? styles.pos : styles.neg}>{pct(ret)}</td>
                    <td className={styles.neg}>{ddp === 0 ? '—' : ddp.toFixed(1) + '%'}</td>
                    <td>{r.win}%</td>
                    <td>{lakh(eq)}</td>
                  </tr>
                );
              });
            })()}
          </tbody>
          <tfoot>
            <tr className={styles.totalRow}>
              <td>TOTAL</td><td>{N_TRADES}</td>
              <td className={styles.pos}>{TOTAL_PTS.toFixed(1)}</td>
              <td className={styles.pos}>{inr(m.total)}</td>
              <td className={styles.pos}>CAGR {m.cagr.toFixed(2)}%</td>
              <td className={styles.neg}>{m.ddPct.toFixed(1)}%</td>
              <td>{WIN_RATE}%</td>
              <td>{lakh(m.capital + m.total)}</td>
            </tr>
          </tfoot>
        </table>
        <p className={styles.note}>
          Real NSE bhavcopy prices, daily-close monitoring, Jan-2019 → Jun-2026.{' '}
          <Link className={styles.link} to="/backtest/nifty-45dte-short-straddle">
            Study — hourly-monitoring variant, VIX filter, delta-management tests →
          </Link>
        </p>
      </div>

      {/* ── margin ──────────────────────────────────────────────────────────── */}
      <div className={styles.grid2}>
        <div className={styles.card}>
          <div className={styles.cardTitle}>Margin by DTE — {lots} lot{lots > 1 ? 's' : ''}</div>
          <table className={styles.table}>
            <thead><tr><th>DTE</th><th>Per lot</th><th>At {lots} lots</th></tr></thead>
            <tbody>
              {MARGIN_BY_DTE.map((r) => (
                <tr key={r.dte} className={r.dte === 45 ? styles.hiRow : ''}>
                  <td className={styles.sym}>{r.dte}{r.interp ? ' *' : ''}</td>
                  <td>{inr(r.perLot)}</td>
                  <td>{lakh(r.perLot * lots)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className={styles.note}>
            Live from Kite <code>basket_order_margins</code> on {MARGIN_ASOF}, ATM straddle,
            standalone. <b>NRML = MIS to the rupee</b> — no intraday benefit on short index options.
            * interpolated. Margin is flat 1–22 DTE and rises with tenor, not into expiry.
          </p>
        </div>
        <div className={styles.card}>
          <div className={styles.cardTitle}>Margin under an adverse move — {lots} lot{lots > 1 ? 's' : ''}</div>
          <table className={styles.table}>
            <thead><tr><th>Scenario</th><th>Per lot</th><th>At {lots} lots</th><th>How often</th></tr></thead>
            <tbody>
              {MARGIN_STRESS.map((r) => (
                <tr key={r.label} className={r.perLot === SIZING_MARGIN ? styles.hiRow : ''}>
                  <td>{r.label}</td>
                  <td>{inr(r.perLot)}</td>
                  <td>{lakh(r.perLot * lots)}</td>
                  <td className={styles.muted}>{r.freq == null ? '—' : r.freq + '% of campaigns'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className={styles.note}>
            Margin rises with <b>moneyness, not tenor</b> — a straddle 3% away from spot costs 26%
            more, 5% away costs 41% more, and <b>down-moves cost more than up-moves</b>. It inflates
            in the same event that drains P&amp;L, which is why the book is sized on the stressed
            number.
          </p>
        </div>
      </div>
    </div>
  );
}

function Kpi({ label, value, tone }: { label: string; value: string; tone: string }) {
  return (
    <div className={styles.kpi}>
      <div className={`${styles.kpiVal} ${tone === 'pos' ? styles.pos : tone === 'neg' ? styles.neg : ''}`}>
        {value}
      </div>
      <div className={styles.kpiLabel}>{label}</div>
    </div>
  );
}

function Rule({ k, v }: { k: string; v: string }) {
  return (
    <div className={styles.ruleRow}>
      <div className={styles.ruleK}>{k}</div>
      <div className={styles.ruleV}>{v}</div>
    </div>
  );
}

/* Short-straddle payoff at expiry, in rupees for the configured size. */
function Payoff({ credit, qty }: { credit: number; qty: number }) {
  const W = 900, H = 240, padL = 62, padR = 16, padT = 14, padB = 26;
  const span = credit * 2.2;                       // x-axis: strike ± 2.2 x credit
  const maxProfit = credit * qty;
  const maxLoss = -(span - credit) * qty;
  const X = (d: number) => padL + ((d + span) / (2 * span)) * (W - padL - padR);
  const Y = (v: number) => padT + (1 - (v - maxLoss) / (maxProfit - maxLoss)) * (H - padT - padB);
  const pl = (d: number) => (credit - Math.abs(d)) * qty;
  const pts = [-span, -credit, 0, credit, span];
  const line = pts.map((d, i) => `${i ? 'L' : 'M'} ${X(d).toFixed(1)} ${Y(pl(d)).toFixed(1)}`).join(' ');
  const area = `${line} L ${X(span).toFixed(1)} ${Y(0).toFixed(1)} L ${X(-span).toFixed(1)} ${Y(0).toFixed(1)} Z`;

  return (
    <svg className={styles.chart} viewBox={`0 0 ${W} ${H}`} role="img"
         aria-label="Short straddle payoff at expiry: peak profit at the strike, losing beyond the breakevens.">
      <line x1={padL} y1={Y(0)} x2={W - padR} y2={Y(0)} className={styles.axis} />
      <path d={area} className={styles.payoffFill} />
      <path d={line} className={styles.payoffLine} />
      {[-credit, credit].map((d) => (
        <g key={d}>
          <line x1={X(d)} y1={padT} x2={X(d)} y2={H - padB} className={styles.beDash} />
          <text x={X(d)} y={H - 8} className={styles.axLab} textAnchor="middle">
            breakeven {d > 0 ? '+' : '−'}{credit.toFixed(0)}
          </text>
        </g>
      ))}
      <line x1={X(0)} y1={padT} x2={X(0)} y2={H - padB} className={styles.strikeDash} />
      <text x={X(0)} y={H - 8} className={styles.axLab} textAnchor="middle">ATM strike</text>
      <circle cx={X(0)} cy={Y(maxProfit)} r={3.5} className={styles.peak} />
      <text x={X(0) + 8} y={Y(maxProfit) + 4} className={styles.peakLab}>
        max {inr(maxProfit)} if it pins
      </text>
      <text x={6} y={Y(maxProfit) + 4} className={styles.axLab}>{inr(maxProfit)}</text>
      <text x={6} y={Y(0) + 4} className={styles.axLab}>0</text>
      <text x={6} y={H - padB - 2} className={styles.axLab}>{inr(maxLoss)}</text>
    </svg>
  );
}
