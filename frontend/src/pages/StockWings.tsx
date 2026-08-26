/* Stock 45->21 DTE winged short strangle — research/127. PAPER.
 *
 * Backed by services/stock_wings_paper.py (cron: 16:50 + 20:30 IST), which
 * seeds entry cycles, marks/exits from real NSE bhavcopy closes and publishes
 * a static JSON — no API route, no backend restart. One universal ruleset for
 * every F&O stock; the liquidity gate IS the stock filter.
 * Study: /app/backtest/stock-45dte-neutral-wings  (STRATEGY-candidate; the
 * real-margin check is still owed before this could ever be real money.)
 */
import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import s from './StockWings.module.css';

const STUDY = '/app/backtest/stock-45dte-neutral-wings';
const TEARSHEET = '/app/stock45_wings_tearsheet.png';
const GH = 'https://github.com/castroarun/Quantifyd/tree/main/research/127_stock_neutral_wings';

const inr = (v: number | null | undefined) =>
  v == null ? '—' : (v < 0 ? '-' : '') + '₹' + Math.abs(Math.round(v)).toLocaleString('en-IN');
const lakh = (v: number) => (v < 0 ? '-' : '') + '₹' + (Math.abs(v) / 1e5).toFixed(2) + 'L';
const MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
type Leg = {
  side: 'SHORT' | 'LONG'; opt: 'CE' | 'PE'; strike: number;
  price: number | null; volume: number;
};

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
  /* read-only projections added for the expandable row */
  legs_entry?: Leg[] | null; legs_now?: Leg[] | null; legs_asof?: string | null;
  margin_real?: number | null; margin_est?: number | null;
  margin_peak?: number | null; mtm_pct?: number | null;
  live?: boolean; stale_legs?: number;
};

type BtTrade = {
  symbol: string; expiry: string; entry: string; exit: string; reason: string;
  year: number; spot: number; kce: number; kpe: number; wce: number; wpe: number;
  hold_days: number; credit_pct: number; gross_pct: number; net_pct: number;
};
type BtAgg = {
  n: number; win: number; avg: number; med: number; best: number; worst: number;
  avg_credit: number; avg_hold: number;
};
type Bt = {
  config: string; source: string; overall: BtAgg;
  by_symbol: (BtAgg & { symbol: string })[];
  by_year: (BtAgg & { year: number })[];
  by_reason: (BtAgg & { reason: string })[];
  trades: BtTrade[];
};
type Paper = {
  asof: string; mode: string; capital: number; max_slots: number;
  slot_margin: number; margin_pct_est: number;
  rules: { dte_in: number; dte_out: number; k_off: number; wing_pct: number; tp: number;
    stop: null; atm_vol_min: number; wing_vol_min: number; slip: number };
  links: { study: string; tearsheet: string; github: string };
  bhav_through: string; realised: number; unrealised: number; nav: number;
  n_open: number; n_closed: number; win_rate: number | null;
  capital_deployed?: number | null; capital_deployed_est?: number | null;
  margin_asof?: string; running_pnl?: number;
  capital_deployed_peak?: number | null; running_pnl_pct?: number | null;
  live_ts?: string | null; live_n?: number;
  upcoming?: Array<{ expiry: string; entry_date: string; entry_weekday: string;
    exit_due: string; days_away: number }>;
  open_positions: Pos[]; closed_trades: Pos[]; note?: string;
};

export default function StockWings() {
  /* Row expansion — per row, plus a one-click expand/collapse all. */
  const [bt, setBt] = useState<Bt | null>(null);
  const [btAll, setBtAll] = useState(false);
  const [btSym, setBtSym] = useState<string | null>(null);
  useEffect(() => {
    fetch('/app/stock_wings_backtest.json')
      .then((r) => (r.ok ? r.json() : null))
      .then(setBt)
      .catch(() => setBt(null));
  }, []);
  const [openRows, setOpenRows] = useState<Set<number>>(new Set());
  const toggleRow = (id: number) =>
    setOpenRows((prev) => {
      const n = new Set(prev);
      if (n.has(id)) n.delete(id); else n.add(id);
      return n;
    });
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
  const held = new Set((p?.open_positions ?? []).map((r) => r.symbol));
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
          {p && (
            <span className={s.tm}>
              {p.live_ts
                ? <><span className={s.liveDot} /> live {p.live_ts}</>
                : <>as of {p.asof.slice(0, 16)}</>}
            </span>
          )}
        </span>
      </div>

      <div className={s.warn}>
        Backtest verdict: <b>STRATEGY-candidate</b> — net +0.264% of spot/trade (t 5.06,
        628 liquid trades 2016–26); portfolio 20–26% CAGR at <b>stressed</b> margin, corr
        to NIFTY −0.09. Margin is now the <b>real Kite basket requirement</b>, not the old
        10%-of-notional estimate — the check the study left owed. Entries, the target and
        the stop still resolve on the EOD bhavcopy close, exactly as backtested; the live
        quotes below only keep the marks moving during the session.
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
          <span className={s.headRight}>
            <button
              className={s.expandBtn}
              onClick={() => {
                const all = p?.open_positions ?? [];
                setOpenRows(openRows.size === all.length && all.length > 0
                  ? new Set()
                  : new Set(all.map((r) => r.id)));
              }}
            >
              {p && openRows.size === p.open_positions.length && p.n_open > 0
                ? '▾ Collapse all' : '▸ Expand all'}
            </button>
            {p && (
              <>
                <span className={s.headStat}
                  title={p.capital_deployed_peak != null
                    ? `peak requirement at entry ${inr(p.capital_deployed_peak)}` : ''}>
                  <b>{p.capital_deployed != null ? inr(p.capital_deployed) : '—'}</b>
                  <i>capital deployed{p.capital_deployed != null
                    ? ` · ${(100 * p.capital_deployed / p.capital).toFixed(0)}% of ₹20L` : ''}</i>
                </span>
                <span className={s.headStat}>
                  <b className={(p.running_pnl ?? 0) >= 0 ? s.pos : s.neg}>
                    {inr(p.running_pnl ?? 0)}
                    {p.running_pnl_pct != null && (
                      <span className={s.pctPar}>
                        ({p.running_pnl_pct >= 0 ? '+' : ''}{p.running_pnl_pct.toFixed(2)}%)
                      </span>
                    )}
                  </b>
                  <i>running P&amp;L{p.running_pnl_pct != null ? ' · on margin' : ''}</i>
                </span>
              </>
            )}
          </span>
        </div>
        <table className={s.table}>
          <thead>
            <tr>
              <th style={{ width: 22 }} />
              <th>Symbol</th><th>Entry</th><th>Expiry (DTE)</th><th>Shorts PE/CE</th>
              <th>Wings PE/CE</th><th>Qty (lots)</th><th>Credit</th><th>Mark</th>
              <th>MTM</th><th>Margin</th><th>Exit due</th><th>Src</th>
            </tr>
          </thead>
          <tbody>
            {(p?.open_positions ?? []).flatMap((r) => [
              <tr key={r.id} className={s.clickRow} onClick={() => toggleRow(r.id)}>
                <td className={s.caret}>{openRows.has(r.id) ? '▾' : '▸'}</td>
                <td className={s.sym}>{r.symbol}</td>
                <td>{dmy(r.entry_date)}</td>
                <td>{dmy(r.expiry)} ({r.dte})</td>
                <td>{r.kpe.toLocaleString('en-IN')} / {r.kce.toLocaleString('en-IN')}</td>
                <td>{r.wpe.toLocaleString('en-IN')} / {r.wce.toLocaleString('en-IN')}</td>
                <td>{r.qty.toLocaleString('en-IN')} ({r.lots}×{r.lot})</td>
                <td>{inr(r.credit * r.qty)}</td>
                <td>{r.mark_val != null ? r.mark_val.toFixed(2) + ' · ' + dmy(r.mark_date) : '—'}</td>
                <td className={(r.mtm_rs ?? 0) >= 0 ? s.pos : s.neg}>
                  {inr(r.mtm_rs)}
                  {r.mtm_pct != null && (
                    <span className={s.pctPar}>
                      ({r.mtm_pct >= 0 ? '+' : ''}{r.mtm_pct.toFixed(1)}%)
                    </span>
                  )}
                </td>
                <td title={[
                  r.margin_peak != null ? `needs ${inr(r.margin_peak)} free at entry` : '',
                  r.margin_est != null ? `old 10%-of-notional estimate ${inr(r.margin_est)}` : '',
                ].filter(Boolean).join(' · ')}>
                  {r.margin_real != null ? inr(r.margin_real) : '—'}
                </td>
                <td>{dmy(r.exit_due)}</td>
                <td><span className={s.srcTag}>{r.src}</span></td>
              </tr>,
              openRows.has(r.id) ? (
                <tr key={r.id + '-legs'} className={s.legRow}>
                  <td colSpan={13}>
                    <div className={s.legWrap}>
                      <div className={s.legTitle}>
                        {r.symbol} — the four legs actually held
                        {r.legs_asof ? <span className={s.tm}>marked {dmy(r.legs_asof)}</span> : null}
                      </div>
                      <table className={s.legTable}>
                        <thead><tr>
                          <th>Leg</th><th>Strike</th><th>Entry</th><th>Now</th>
                          <th>Move</th><th>Qty</th><th>Value now</th><th>Entry vol</th>
                        </tr></thead>
                        <tbody>
                          {(r.legs_entry ?? []).map((le: Leg, i: number) => {
                            const nowLeg = (r.legs_now ?? []).find(
                              (x: Leg) => x.side === le.side && x.opt === le.opt);
                            const now = nowLeg?.price ?? null;
                            const sign = le.side === 'SHORT' ? -1 : 1;
                            const move = le.price && now != null ? (now / le.price - 1) * 100 : null;
                            return (
                              <tr key={i}>
                                <td>
                                  <span className={le.side === 'SHORT' ? s.legShort : s.legLong}>
                                    {le.side === 'SHORT' ? 'SHORT' : 'LONG'}
                                  </span>{' '}{le.opt}
                                </td>
                                <td>{le.strike.toLocaleString('en-IN')}</td>
                                <td>{le.price != null ? le.price.toFixed(2) : '—'}</td>
                                <td>{now != null ? now.toFixed(2) : '—'}</td>
                                <td className={move == null ? '' :
                                  (move * sign) >= 0 ? s.pos : s.neg}>
                                  {move == null ? '—' : (move >= 0 ? '+' : '') + move.toFixed(1) + '%'}
                                </td>
                                <td>{r.qty.toLocaleString('en-IN')}</td>
                                <td>{now != null ? inr(now * r.qty * (le.side === 'SHORT' ? -1 : 1)) : '—'}</td>
                                <td className={s.muted}>{le.volume ? le.volume.toLocaleString('en-IN') : '—'}</td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                      <div className={s.legNote}>
                        Net of the four legs = <b>{r.credit.toFixed(2)}</b> credit per share at entry
                        ({inr(r.credit * r.qty)} on {r.qty.toLocaleString('en-IN')} qty), marked at{' '}
                        <b>{r.mark_val != null ? r.mark_val.toFixed(2) : '—'}</b> to close.
                        Shorts pay you; the two long wings cap the tail and are what pull the real
                        margin below a naked strangle&apos;s.
                        {r.margin_real != null && (
                          <> Kite blocks <b>{inr(r.margin_real)}</b> for this position
                          ({inr(r.margin_real / r.lots)}/lot on {r.lots} lots)
                          {r.margin_peak != null && (
                            <>, and wants {inr(r.margin_peak)} free at the moment of entry
                            — the gap is the hedge benefit, which only lands once all four
                            legs are on</>
                          )}.</>
                        )}
                      </div>
                    </div>
                  </td>
                </tr>
              ) : null,
            ])}
            {p && p.open_positions.length === 0 && (
              <tr><td colSpan={13} className={s.emptyCell}>Book is flat — next cycle {next ? dmy(next.entry_date) : '—'}.</td></tr>
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

      {/* ---- the same structure, historically ------------------------------ */}
      <div className={s.card}>
        <div className={s.cardTitle}>
          The same structure, backtested
          {bt && <span className={s.tm}>{bt.overall.n} trades &middot; {bt.by_symbol.length} stocks &middot; 2016&ndash;26</span>}
          <span className={s.headRight}>
            <button className={s.expandBtn} onClick={() => { setBtAll(!btAll); setBtSym(null); }}>
              {btAll ? 'Held symbols only' : 'All symbols'}
            </button>
          </span>
        </div>
        {!bt ? (
          <p className={s.note}>Loading the study&apos;s trade log&hellip;</p>
        ) : (
          <>
            <p className={s.note}>
              Every prior instance of the construction the book is running right now &mdash; same
              45&rarr;21 DTE window, same &plusmn;2.5% shorts, same 7% wings, no stop, TP at half the
              credit, and the same liquidity gate. Returns are <b>net</b> (gross less 0.5% of
              premium turnover) and quoted as a <b>% of spot</b>, which is how the study compares
              risk across stocks of very different prices.
            </p>
            <div className={s.gateRow} style={{ marginTop: 14 }}>
              <div><b className={s.pos}>+{bt.overall.avg.toFixed(3)}%</b>
                <span className={s.muted}>net per trade (of spot)</span></div>
              <div><b>{bt.overall.win.toFixed(1)}%</b><span className={s.muted}>win rate</span></div>
              <div><b>{bt.overall.avg_credit.toFixed(2)}%</b><span className={s.muted}>avg credit taken</span></div>
              <div><b>{bt.overall.avg_hold.toFixed(0)} d</b><span className={s.muted}>avg hold</span></div>
              <div><b className={s.neg}>{bt.overall.worst.toFixed(1)}%</b><span className={s.muted}>worst single trade</span></div>
            </div>

            <table className={s.table} style={{ marginTop: 16 }}>
              <thead>
                <tr>
                  <th style={{ width: 22 }} />
                  <th>Symbol</th><th>Trades</th><th>Win %</th><th>Avg net</th>
                  <th>Median</th><th>Best</th><th>Worst</th><th>Avg credit</th>
                  <th>Avg hold</th><th>Held now</th>
                </tr>
              </thead>
              <tbody>
                {bt.by_symbol
                  .filter((b) => btAll || held.has(b.symbol))
                  .map((b) => [
                    <tr key={b.symbol} className={s.clickRow}
                      onClick={() => setBtSym(btSym === b.symbol ? null : b.symbol)}>
                      <td className={s.caret}>{btSym === b.symbol ? '\u25be' : '\u25b8'}</td>
                      <td className={s.sym}>{b.symbol}</td>
                      <td>{b.n}</td>
                      <td className={b.win >= 50 ? s.pos : s.neg}>{b.win.toFixed(0)}%</td>
                      <td className={b.avg >= 0 ? s.pos : s.neg}>
                        {(b.avg >= 0 ? '+' : '') + b.avg.toFixed(3)}%
                      </td>
                      <td className={b.med >= 0 ? s.pos : s.neg}>
                        {(b.med >= 0 ? '+' : '') + b.med.toFixed(3)}%
                      </td>
                      <td className={s.pos}>+{b.best.toFixed(2)}%</td>
                      <td className={s.neg}>{b.worst.toFixed(2)}%</td>
                      <td>{b.avg_credit.toFixed(2)}%</td>
                      <td>{b.avg_hold.toFixed(0)} d</td>
                      <td>{held.has(b.symbol)
                        ? <span className={s.heldTag}>open</span>
                        : <span className={s.muted}>&mdash;</span>}</td>
                    </tr>,
                    btSym === b.symbol ? (
                      <tr key={b.symbol + '-t'} className={s.legRow}>
                        <td colSpan={11}>
                          <div className={s.legWrap}>
                            <div className={s.legTitle}>
                              {b.symbol} &mdash; every backtested instance
                            </div>
                            <table className={s.legTable}>
                              <thead>
                                <tr>
                                  <th>Entry</th><th>Exit</th><th>Held</th><th>Spot</th>
                                  <th>Shorts PE/CE</th><th>Wings PE/CE</th>
                                  <th>Credit</th><th>Net</th><th>Exit reason</th>
                                </tr>
                              </thead>
                              <tbody>
                                {bt.trades.filter((t) => t.symbol === b.symbol).map((t, i) => (
                                  <tr key={i}>
                                    <td>{dmy(t.entry)}</td>
                                    <td>{dmy(t.exit)}</td>
                                    <td>{t.hold_days} d</td>
                                    <td>{t.spot.toLocaleString('en-IN')}</td>
                                    <td>{t.kpe} / {t.kce}</td>
                                    <td className={s.muted}>{t.wpe} / {t.wce}</td>
                                    <td>{t.credit_pct.toFixed(2)}%</td>
                                    <td className={t.net_pct >= 0 ? s.pos : s.neg}>
                                      {(t.net_pct >= 0 ? '+' : '') + t.net_pct.toFixed(3)}%
                                    </td>
                                    <td><span className={s.srcTag}>{t.reason}</span></td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </td>
                      </tr>
                    ) : null,
                  ])}
              </tbody>
            </table>

            <div className={s.legTitle} style={{ marginTop: 18 }}>By year &mdash; the same trades, grouped</div>
            <table className={s.table}>
              <thead>
                <tr><th>Year</th><th>Trades</th><th>Win %</th><th>Avg net</th>
                  <th>Best</th><th>Worst</th><th>Avg hold</th></tr>
              </thead>
              <tbody>
                {bt.by_year.map((y) => (
                  <tr key={y.year}>
                    <td className={s.sym}>{y.year}</td>
                    <td>{y.n}</td>
                    <td className={y.win >= 50 ? s.pos : s.neg}>{y.win.toFixed(0)}%</td>
                    <td className={y.avg >= 0 ? s.pos : s.neg}>
                      {(y.avg >= 0 ? '+' : '') + y.avg.toFixed(3)}%
                    </td>
                    <td className={s.pos}>+{y.best.toFixed(2)}%</td>
                    <td className={s.neg}>{y.worst.toFixed(2)}%</td>
                    <td>{y.avg_hold.toFixed(0)} d</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className={s.note}>
              The thin early years are the honest part: before 2021 stock options rarely cleared
              the liquidity gate, so few trades qualified. The edge being traded lives in the
              dense era, and the study says so rather than averaging it away. How trades ended:{' '}
              {bt.by_reason.map((x, i) => (
                <span key={x.reason}>
                  {i > 0 ? ' \u00b7 ' : ''}<b>{x.reason}</b> {x.n} ({x.win.toFixed(0)}% win)
                </span>
              ))}.
            </p>
          </>
        )}
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
