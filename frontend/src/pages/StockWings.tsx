/* Stock 45->21 DTE winged short strangle — research/127. PAPER.
 *
 * Backed by services/stock_wings_paper.py (cron: 16:50 + 20:30 IST), which
 * seeds entry cycles, marks/exits from real NSE bhavcopy closes and publishes
 * a static JSON — no API route, no backend restart. One universal ruleset for
 * every F&O stock; the liquidity gate IS the stock filter.
 * Study: /app/backtest/stock-45dte-neutral-wings  (STRATEGY-candidate; the
 * real-margin check is still owed before this could ever be real money.)
 */
import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import s from './StockWings.module.css';

const STUDY = '/app/backtest/stock-45dte-neutral-wings';
const TEARSHEET = '/app/stock45_wings_tearsheet.png';
const GH = 'https://github.com/castroarun/Quantifyd/tree/main/research/127_stock_neutral_wings';

const inr = (v: number | null | undefined) =>
  v == null ? '—' : (v < 0 ? '-' : '') + '₹' + Math.abs(Math.round(v)).toLocaleString('en-IN');
const lakh = (v: number) => (v < 0 ? '-' : '') + '₹' + (Math.abs(v) / 1e5).toFixed(2) + 'L';
const MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const dmy = (v: string | null | undefined) => {
  if (!v) return '—';
  const [y, m, d] = v.slice(0, 10).split('-');
  return `${d}-${MON[parseInt(m, 10) - 1]}-${y.slice(2)}`;
};

type Pos = {
  id: number; symbol: string; expiry: string; entry_date: string; entry_spot: number;
  kce: number; kpe: number; wce: number; wpe: number;
  credit: number; lots: number; lot: number; qty: number;
  atm_vol: number; wing_vol_min: number; src: 'SEED' | 'LIVE';
  exit_date: string | null; exit_val: number | null; exit_reason: string | null;
  gross_rs: number | null; cost_rs: number | null; net_rs: number | null;
  status: 'OPEN' | 'CLOSED';
  mark_val: number | null; mark_date: string | null; mtm_rs: number | null; mark_spot: number | null;
  exit_due: string; dte: number;
  curve?: Array<{ d: string; val: number; mtm: number }>;
};
type Paper = {
  asof: string; mode: string; capital: number; max_slots: number;
  slot_margin: number; margin_pct_est: number;
  rules: { dte_in: number; dte_out: number; k_off: number; wing_pct: number; tp: number;
    stop: null; atm_vol_min: number; wing_vol_min: number; slip: number };
  links: { study: string; tearsheet: string; github: string };
  bhav_through: string; realised: number; unrealised: number; nav: number;
  n_open: number; n_closed: number; win_rate: number | null;
  upcoming?: Array<{ expiry: string; entry_date: string; entry_weekday: string;
    exit_due: string; days_away: number }>;
  open_positions: Pos[]; closed_trades: Pos[]; note?: string;
};

export default function StockWings() {
  const [p, setP] = useState<Paper | null>(null);
  useEffect(() => {
    const load = () =>
      fetch('/app/stock_wings_paper.json?t=' + Date.now())
        .then((r) => (r.ok ? r.json() : null))
        .then(setP)
        .catch(() => setP(null));
    load();
    const t = setInterval(load, 60000);
    return () => clearInterval(t);
  }, []);

  const next = p?.upcoming?.[0] ?? null;
  const totalRet = p ? ((p.nav - p.capital) / p.capital) * 100 : 0;

  return (
    <div className={s.root}>
      <div className={s.headerRow}>
        <div>
          <h1 className={s.title}>Stock Winged Strangle — 45→21 DTE</h1>
          <p className={s.sub}>
            One universal ruleset across the F&amp;O stock universe: at 45 DTE sell the ±2.5%
            monthly strangle, buy wings ~7% away (crash cap), no stop, 50% target, exit at
            21 DTE. Liquidity is the only stock filter — all four legs must have traded
            (shorts ≥100 contracts, wings ≥10). Up to 10 slots ranked by option volume.
            {' '}<Link className={s.link} to={STUDY}>Backtest study</Link>
            {' · '}<a className={s.link} href={TEARSHEET} target="_blank" rel="noreferrer">Tearsheet</a>
            {' · '}<a className={s.link} href={GH} target="_blank" rel="noreferrer">research/127</a>
          </p>
        </div>
        <span className={`${s.gateBadge} ${s.paperOn}`}>
          <span className="dot" /> PAPER · ₹20L
          {p && <span className={s.tm}>as of {p.asof.slice(0, 16)}</span>}
        </span>
      </div>

      <div className={s.warn}>
        Backtest verdict: <b>STRATEGY-candidate</b> — net +0.264% of spot/trade (t 5.06,
        628 liquid trades 2016–26); portfolio 20–26% CAGR at <b>stressed</b> margin, corr
        to NIFTY −0.09. Sizing here uses a 10%-of-notional margin <b>estimate</b>; the
        real Kite basket-margin check is still owed, and marks are EOD bhavcopy (stock
        options have no intraday recorder).
      </div>

      <div className={s.kpis}>
        <div className={s.kpi}><div className={`${s.kpiVal} ${p && p.nav >= (p?.capital ?? 0) ? s.pos : s.neg}`}>{p ? lakh(p.nav) : '—'}</div><div className={s.kpiLabel}>NAV</div></div>
        <div className={s.kpi}><div className={`${s.kpiVal} ${totalRet >= 0 ? s.pos : s.neg}`}>{p ? (totalRet >= 0 ? '+' : '') + totalRet.toFixed(2) + '%' : '—'}</div><div className={s.kpiLabel}>Return on ₹20L</div></div>
        <div className={s.kpi}><div className={`${s.kpiVal} ${p && p.realised >= 0 ? s.pos : s.neg}`}>{p ? inr(p.realised) : '—'}</div><div className={s.kpiLabel}>Realised</div></div>
        <div className={s.kpi}><div className={`${s.kpiVal} ${p && p.unrealised >= 0 ? s.pos : s.neg}`}>{p ? inr(p.unrealised) : '—'}</div><div className={s.kpiLabel}>Open MTM</div></div>
        <div className={s.kpi}><div className={s.kpiVal}>{p ? `${p.n_open}/${p.max_slots}` : '—'}</div><div className={s.kpiLabel}>Slots in use</div></div>
        <div className={s.kpi}><div className={s.kpiVal}>{p ? p.n_closed : '—'}</div><div className={s.kpiLabel}>Closed trades</div></div>
        <div className={s.kpi}><div className={s.kpiVal}>{p?.win_rate != null ? p.win_rate.toFixed(0) + '%' : '—'}</div><div className={s.kpiLabel}>Win rate</div></div>
        <div className={s.kpi}><div className={s.kpiVal}>{p ? dmy(p.bhav_through) : '—'}</div><div className={s.kpiLabel}>Bhav through</div></div>
      </div>

      <div className={s.card}>
        <div className={s.cardTitle}>
          Open positions {p ? `(${p.n_open})` : ''}
          {next && (
            <span className={s.tm}>
              &nbsp;· next entry cycle {dmy(next.entry_date)} ({next.entry_weekday}), expiry {dmy(next.expiry)}
            </span>
          )}
        </div>
        <table className={s.table}>
          <thead>
            <tr>
              <th>Symbol</th><th>Entry</th><th>Expiry (DTE)</th><th>Shorts PE/CE</th>
              <th>Wings PE/CE</th><th>Qty (lots)</th><th>Credit</th><th>Mark</th>
              <th>MTM</th><th>Exit due</th><th>Src</th>
            </tr>
          </thead>
          <tbody>
            {(p?.open_positions ?? []).map((r) => (
              <tr key={r.id}>
                <td className={s.sym}>{r.symbol}</td>
                <td>{dmy(r.entry_date)}</td>
                <td>{dmy(r.expiry)} ({r.dte})</td>
                <td>{r.kpe.toLocaleString('en-IN')} / {r.kce.toLocaleString('en-IN')}</td>
                <td>{r.wpe.toLocaleString('en-IN')} / {r.wce.toLocaleString('en-IN')}</td>
                <td>{r.qty.toLocaleString('en-IN')} ({r.lots}×{r.lot})</td>
                <td>{inr(r.credit * r.qty)}</td>
                <td>{r.mark_val != null ? r.mark_val.toFixed(2) + ' · ' + dmy(r.mark_date) : '—'}</td>
                <td className={(r.mtm_rs ?? 0) >= 0 ? s.pos : s.neg}>{inr(r.mtm_rs)}</td>
                <td>{dmy(r.exit_due)}</td>
                <td><span className={s.srcTag}>{r.src}</span></td>
              </tr>
            ))}
            {p && p.open_positions.length === 0 && (
              <tr><td colSpan={11} className={s.emptyCell}>Book is flat — next cycle {next ? dmy(next.entry_date) : '—'}.</td></tr>
            )}
          </tbody>
        </table>
      </div>

      <div className={s.card}>
        <div className={s.cardTitle}>Closed trades {p ? `(${p.n_closed})` : ''}</div>
        <table className={s.table}>
          <thead>
            <tr>
              <th>Symbol</th><th>Entry → Exit</th><th>Expiry</th><th>Shorts PE/CE</th>
              <th>Qty</th><th>Credit</th><th>Exit @</th><th>Reason</th><th>Net</th><th>Src</th>
            </tr>
          </thead>
          <tbody>
            {(p?.closed_trades ?? []).slice().reverse().map((r) => (
              <tr key={r.id}>
                <td className={s.sym}>{r.symbol}</td>
                <td>{dmy(r.entry_date)} → {dmy(r.exit_date)}</td>
                <td>{dmy(r.expiry)}</td>
                <td>{r.kpe.toLocaleString('en-IN')} / {r.kce.toLocaleString('en-IN')}</td>
                <td>{r.qty.toLocaleString('en-IN')}</td>
                <td>{inr(r.credit * r.qty)}</td>
                <td>{r.exit_val != null ? r.exit_val.toFixed(2) : '—'}</td>
                <td className={s.freq}>{r.exit_reason}</td>
                <td className={(r.net_rs ?? 0) >= 0 ? s.pos : s.neg}>{inr(r.net_rs)}</td>
                <td><span className={s.srcTag}>{r.src}</span></td>
              </tr>
            ))}
            {p && p.closed_trades.length === 0 && (
              <tr><td colSpan={10} className={s.emptyCell}>No closed trades yet.</td></tr>
            )}
          </tbody>
        </table>
      </div>

      <div className={s.grid2}>
        <div className={s.card}>
          <div className={s.cardTitle}>The ruleset (research/127 C1 — identical for every stock)</div>
          <div className={s.rules}>
            <div className={s.ruleRow}><div className={s.ruleK}>Entry</div><div className={s.ruleV}>Monthly stock expiry − 45 calendar days, at EOD close (rolled back to a session)</div></div>
            <div className={s.ruleRow}><div className={s.ruleK}>Structure</div><div className={s.ruleV}>Sell CE @ spot+2.5% and PE @ spot−2.5% (nearest traded strikes); buy wing CE/PE ~7% of spot beyond the shorts</div></div>
            <div className={s.ruleRow}><div className={s.ruleK}>Liquidity gate</div><div className={s.ruleV}>All 4 legs traded that day; short legs ≥100 contracts combined; each wing ≥10. This is the ONLY stock filter</div></div>
            <div className={s.ruleRow}><div className={s.ruleK}>Exits</div><div className={s.ruleV}>First of: 50% of net credit captured · time exit at 21 DTE. No premium stop — the wings are the risk cap</div></div>
            <div className={s.ruleRow}><div className={s.ruleK}>Slots &amp; sizing</div><div className={s.ruleV}>10 slots × ₹2L margin; candidates ranked by option volume; lots sized to ~₹20L notional/slot (10%-of-notional margin estimate)</div></div>
            <div className={s.ruleRow}><div className={s.ruleK}>Costs</div><div className={s.ruleV}>0.5% of premium per side slippage + STT + txn + ₹20×8 brokerage, applied at close</div></div>
          </div>
        </div>
        <div className={s.card}>
          <div className={s.cardTitle}>Why this exists (the study, in one card)</div>
          <div className={s.gateRow}>
            <div><b>+0.264% S0</b><span className={s.muted}>net/trade · t 5.06 · n 628</span></div>
            <div><b>20–26%</b><span className={s.muted}>CAGR at stressed margin</span></div>
            <div><b>−0.09</b><span className={s.muted}>corr vs NIFTY (monthly)</span></div>
            <div><b>+1.65%/mo</b><span className={s.muted}>in NIFTY down&gt;3% months</span></div>
          </div>
          <div className={s.assump}>
            <b>Evidence highlights:</b>
            <ul>
              <li>DTE placebo: the same structure at 35/55 DTE earns ~zero — the 45→21 theta window IS the edge.</li>
              <li>Edge rises monotonically with liquidity (the anti-phantom-fill test).</li>
              <li>Survives dropping the top-5 contributing names (t 3.49) and every era split.</li>
              <li>No premium stop beats every stop tested; wider wings beat tighter ones.</li>
            </ul>
          </div>
          <p className={s.note}>
            Full report: <Link className={s.link} to={STUDY}>/app/backtest/stock-45dte-neutral-wings</Link> —
            rules, robustness gauntlet, margin stress, year-wise tables and the complete trade log.
            Owed before real money: real basket-margin check (recorder pattern from research/119).
          </p>
        </div>
      </div>

      {p?.note && <div className={s.note}>{p.note}</div>}
    </div>
  );
}
