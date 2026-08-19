import { useEffect, useState } from 'react';
import { apiGet } from '../api/client';
import styles from './BreakoutPaper.module.css';
import BookActivity from '../components/BookActivity/BookActivity';

type Holding = {
  symbol: string; qty: number | null; entry_date: string | null; entry_price: number | null;
  price: number | null; value: number; weight: number; pnl: number; pnl_pct: number | null;
  days: number | null; stop: number | null; stop_dist_pct: number | null; is_cash?: boolean;
  tag?: string;
};
type Candidate = { symbol: string; run_pct: number; turn_cr: number };
type Closed = {
  symbol: string; entry_date: string; entry_price: number; exit_date: string; exit_price: number;
  qty: number; gross_pnl: number; gross_pct: number; net_pnl: number; reason: string;
  holding_days: number; stcg_tax: number;
};
type NavPt = { d: string; nav: number; bench: number | null; gate: string };
type State = {
  seeded: boolean; gate: string; gate_on: boolean; inception: string | null; capital: number;
  nav: number; cash: number; equity: number; invested_pct: number; total_return_pct: number;
  unrealized: number; realized_net: number; n_holdings: number; interest_earned: number;
  cash_yield_pct: number; stcg_unbooked: number; stcg_booked: number;
  last_daily: string | null; data_asof: string | null; today_candidates: Candidate[];
  gate_last: number | null; gate_sma: number | null; gate_gap_pct: number | null;
  holdings: Holding[]; navcurve: NavPt[]; closed: Closed[]; rules: [string, string][];
};

const inr = (n: number) => '₹' + Math.round(n).toLocaleString('en-IN');
const lakh = (n: number) => '₹' + (n / 100000).toFixed(2) + 'L';
const pct = (n: number | null) => (n == null ? '—' : (n >= 0 ? '+' : '') + n.toFixed(1) + '%');
const reasonLabel: Record<string, string> = {
  DONCHIAN: 'Donchian-20 trail', CATASTROPHE: '20% catastrophe', BREAKOUT: 'Breakout entry',
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

export default function BreakoutPaper() {
  const [s, setS] = useState<State | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const load = () => apiGet<State>('/api/breakout-paper/state').then(setS).catch((e) => setErr(String(e)));
  useEffect(() => {
    load();
    const t = setInterval(load, 30000);
    return () => clearInterval(t);
  }, []);

  if (err) return <div className={styles.root}><div className={styles.loading}>Error: {err}</div></div>;
  if (!s) return <div className={styles.root}><div className={styles.loading}>Loading paper book…</div></div>;

  const riskOn = s.gate_on;
  const retPos = s.total_return_pct >= 0;

  return (
    <div className={styles.root}>
      <div className={styles.headerRow}>
        <div>
          <h1 className={styles.title}>Breakout Swing — Live Paper Book</h1>
          <p className={styles.sub}>
            ₹10L paper deployment of the research/71 winner · MTF-bullish volume breakouts, Donchian-20 trail + NIFTY&gt;200DMA gate, 1 entry/day ·
            {s.inception ? ` since ${s.inception}` : ''} · data as-of {s.data_asof || '—'}
          </p>
        </div>
        <div className={`${styles.gateBadge} ${riskOn ? styles.on : styles.off}`}>
          <span className={styles.dot} />
          {riskOn ? 'RISK-ON · taking entries' : 'RISK-OFF · no new entries'}
        </div>
      </div>
      <BookActivity bookId="breakout-paper" />

      {/* KPI strip */}
      <div className={styles.kpis}>
        <Kpi label="NAV" value={lakh(s.nav)} tone="" />
        <Kpi label="Total return" value={pct(s.total_return_pct)} tone={retPos ? 'pos' : 'neg'} />
        <Kpi label="Invested" value={s.invested_pct.toFixed(0) + '%'} tone="" />
        <Kpi label="Cash" value={lakh(s.cash)} tone="" />
        <Kpi label="Open positions" value={`${s.n_holdings} / 8`} tone="" />
        <Kpi label="Unrealized" value={inr(s.unrealized)} tone={s.unrealized >= 0 ? 'pos' : 'neg'} />
        <Kpi label="Realized (net)" value={inr(s.realized_net)} tone={s.realized_net >= 0 ? 'pos' : 'neg'} />
        <Kpi label={`Cash yield @${s.cash_yield_pct}%`} value={inr(s.interest_earned)} tone="pos" />
      </div>

      <div className={styles.grid2}>
        {/* Gate panel */}
        <div className={styles.card}>
          <div className={styles.cardTitle}>Regime gate · NIFTYBEES vs 200-day SMA</div>
          <div className={styles.gateRow}>
            <div><span className={styles.muted}>NIFTYBEES</span><b>{s.gate_last ?? '—'}</b></div>
            <div><span className={styles.muted}>200-DMA</span><b>{s.gate_sma ?? '—'}</b></div>
            <div><span className={styles.muted}>Gap</span>
              <b className={(s.gate_gap_pct ?? 0) >= 0 ? styles.pos : styles.neg}>{pct(s.gate_gap_pct)}</b></div>
            <div><span className={styles.muted}>State</span>
              <b className={riskOn ? styles.pos : styles.neg}>{riskOn ? 'RISK-ON' : 'RISK-OFF'}</b></div>
          </div>
          <p className={styles.note}>
            {riskOn
              ? 'Above the 200-DMA → the book takes up to 1 new breakout per day (top of the candidates below) until 8 are held.'
              : 'Below the 200-DMA → NO new entries (the single biggest drawdown reducer). Existing positions keep their trailing stops; freed cash earns the liquid yield.'}
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
          <p className={styles.note}>20% short-term capital-gains on gains held &lt; 1 year (these ~7-week holds are short-term). Tracked separately so the NAV reflects pre-tax performance net of 0.20% trading cost.</p>
        </div>
      </div>

      {/* Equity curve */}
      <div className={styles.card}>
        <div className={styles.cardTitle}>Paper NAV vs NIFTYBEES (growth of ₹1)</div>
        <EquityCurve data={s.navcurve} />
      </div>

      {/* Holdings */}
      {s.holdings.length > 0 && (
        <div className={styles.card}>
          <div className={styles.cardTitle}>Open positions</div>
          <table className={styles.table}>
            <thead><tr>
              <th>Holding</th><th>Wt</th><th>Entry</th><th>Entry ₹</th><th>Now ₹</th>
              <th>P&L</th><th>Days</th><th>Trailing stop (prior 20-day low; hard floor −20% from entry)</th><th>To stop</th>
            </tr></thead>
            <tbody>
              {s.holdings.map((h) => (
                <tr key={h.symbol} className={h.is_cash ? styles.cashRow : ''}>
                  <td className={styles.sym}>
                    {h.symbol}
                    {h.is_cash && <span className={styles.cashTag}>{h.tag || 'liquid fund @6.5%'}</span>}
                  </td>
                  <td>{h.weight}%</td>
                  <td className={styles.muted}>{h.entry_date}</td>
                  <td>{h.entry_price ?? '—'}</td>
                  <td>{h.is_cash ? lakh(h.value) : h.price}</td>
                  <td className={(h.pnl_pct ?? 0) >= 0 ? styles.pos : styles.neg}>
                    {h.is_cash ? '+' + inr(h.pnl) : pct(h.pnl_pct)}</td>
                  <td>{h.days}</td>
                  <td className={styles.muted}>{h.is_cash ? '—' : (h.stop ?? '—')}</td>
                  <td className={!h.is_cash && (h.stop_dist_pct ?? 9) < 3 ? styles.warn : ''}>
                    {h.stop_dist_pct == null ? '—' : '+' + h.stop_dist_pct + '%'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Today's breakout candidates */}
      <div className={styles.card}>
        <div className={styles.cardTitle}>
          Today's qualifying breakouts {riskOn ? '— the book takes the top 1' : '— (gate risk-off, none taken)'}
        </div>
        {s.today_candidates.length === 0 ? (
          <div className={styles.chartEmpty}>No names pass the full MTF-bullish volume-breakout filter today.</div>
        ) : (
          <table className={styles.table}>
            <thead><tr><th>Rank</th><th>Stock</th><th>Today's run</th><th>20d median turnover</th></tr></thead>
            <tbody>
              {s.today_candidates.map((c, i) => (
                <tr key={c.symbol} className={i === 0 && riskOn ? styles.pickRow : ''}>
                  <td className={styles.muted}>#{i + 1}{i === 0 && riskOn ? ' ← pick' : ''}</td>
                  <td className={styles.sym}>{c.symbol}</td>
                  <td className={styles.pos}>{pct(c.run_pct)}</td>
                  <td>₹{c.turn_cr}cr</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <p className={styles.note}>
          Qualifiers: daily+weekly+monthly MACD&gt;0, close ≥98% of the 252-day high, volume ≥2× its 20-day average,
          20-day median turnover ≥₹5cr, price ≥₹20, not up &gt;15% today. Ranked by today's %-run; the book takes the top one (when risk-on and a slot is free).
        </p>
      </div>

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
                  <td className={c.net_pnl >= 0 ? styles.pos : styles.neg}>{inr(c.net_pnl)}</td>
                  <td className={c.gross_pct >= 0 ? styles.pos : styles.neg}>{pct(c.gross_pct)}</td>
                  <td>{c.holding_days}</td>
                  <td><span className={styles.reason}>{reasonLabel[c.reason] || c.reason}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Rules */}
      <div className={styles.card}>
        <div className={styles.cardTitle}>System rules (run automatically)</div>
        <div className={styles.rules}>
          {s.rules.map(([k, v]) => (
            <div key={k} className={styles.ruleRow}><div className={styles.ruleK}>{k}</div><div className={styles.ruleV}>{v}</div></div>
          ))}
        </div>
        <p className={styles.note}>
          Automation: daily 15:45 IST — settle T+1 money + accrue fund interest, check exits (Donchian-20 / 20% catastrophe),
          take up to 1 new breakout from the settled buffer if risk-on, then refill/sweep the buffer (a buy triggers a same-day
          fund redemption so tomorrow's slot is ready). Paper only — never places a real order. Last daily run: {s.last_daily || '—'}.
        </p>
      </div>
    </div>
  );
}

function Kpi({ label, value, tone }: { label: string; value: string; tone: string }) {
  return (
    <div className={styles.kpi}>
      <div className={`${styles.kpiVal} ${tone === 'pos' ? styles.pos : tone === 'neg' ? styles.neg : ''}`}>{value}</div>
      <div className={styles.kpiLabel}>{label}</div>
    </div>
  );
}
