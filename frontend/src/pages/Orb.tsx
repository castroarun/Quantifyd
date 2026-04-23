import { useEffect, useMemo, useState } from 'react';
import styles from './Orb.module.css';
import { apiGet } from '../api/client';
import type {
  ORBState,
  ORBStockSummary,
  ORBPosition,
  ORBClosedTrade,
  ORBSignal,
  ORBCandidates,
} from '../api/types';
import MetricCard from '../components/Cards/MetricCard';
import DataTable from '../components/DataTable/DataTable';
import type { Column } from '../components/DataTable/DataTable';
import Chip from '../components/Chip/Chip';
import {
  formatInt,
  formatNumber,
  formatPnl,
  formatRs,
  pnlClass,
} from '../utils/format';
import { formatTime } from '../utils/time';
// action buttons removed — header is title/subtitle only, consistent across pages

/* ---------- helpers ---------- */

/** Micro gauge bar: red end = original SL (OR edge), green end = Target.
 * Entry tick stays at its natural ~40% position even after the 14:30 trail.
 * Current SL (after trail) renders as a secondary marker that moves toward
 * target as profit is locked. Dot = live LTP. */
function PositionGauge({ p }: { p: ORBPosition }) {
  const tgt = p.target_price ?? p.target;
  const entry = p.entry_price;
  const ltp = p.ltp;
  // Anchor to the ORIGINAL SL (the OR edge). For LONG this is or_low,
  // for SHORT it's or_high. This never moves, so the gauge scale stays
  // fixed for the life of the trade. Falls back to live sl_price if the
  // OR edges aren't available (should be rare).
  const origSl =
    p.direction === 'LONG'
      ? (p.or_low ?? p.sl_price ?? p.stop_loss)
      : (p.or_high ?? p.sl_price ?? p.stop_loss);
  const liveSl = p.sl_price ?? p.stop_loss;

  if (origSl == null || tgt == null || entry == null) {
    return <span className={styles.mute}>—</span>;
  }

  // Normalize a price to 0..100 along origSL → Target.
  const toPct = (price: number) => {
    const span = tgt - origSl;
    if (span === 0) return 50;
    return Math.max(0, Math.min(100, ((price - origSl) / span) * 100));
  };

  const entryPct = toPct(entry);
  const ltpPct = ltp != null ? toPct(ltp) : null;
  // Show trailed-SL marker only if the live SL has moved away from
  // the original SL (profit locked). Tolerance = 0.05.
  const trailed = liveSl != null && Math.abs(liveSl - origSl) > 0.05;
  const trailedSlPct = trailed && liveSl != null ? toPct(liveSl) : null;

  return (
    <div className={styles.gaugeCell}>
      <div className={styles.gaugeTrack}>
        <div className={styles.gaugeMark} style={{ left: `${entryPct}%` }} />
        {trailedSlPct != null && (
          <div
            className={styles.gaugeTrailedSl}
            style={{ left: `${trailedSlPct}%` }}
            title={`Trailed SL: ${liveSl?.toFixed(2)}`}
          />
        )}
        {ltpPct != null && (
          <div className={styles.gaugeLtp} style={{ left: `${ltpPct}%` }} />
        )}
      </div>
      <div className={styles.gaugeCap}>
        <span>SL</span>
        <span>Entry</span>
        <span>TGT</span>
      </div>
    </div>
  );
}

function stockStatus(s: ORBStockSummary): string {
  if (s.position) return 'In position';
  if (s.today_result) return 'Traded';
  const ds = s.daily_state;
  if (ds?.is_wide_cpr_day) return 'Wide CPR';
  if (ds?.or_finalized) return 'Scanning';
  if (ds?.today_open) return 'Waiting for OR';
  return 'Idle';
}

function statusTone(label: string): 'neutral' | 'pos' | 'neg' {
  if (label === 'In position' || label === 'Scanning') return 'pos';
  if (label === 'Wide CPR') return 'neg';
  return 'neutral';
}

interface FilterDot {
  label: string;
  tone: 'pos' | 'neg' | 'neutral';
  title: string;
}

function filterDots(s: ORBStockSummary): FilterDot[] {
  const ds = s.daily_state || ({} as ORBStockSummary['daily_state']);
  const dots: FilterDot[] = [];

  // VWAP
  if (ds?.vwap && s.price) {
    const above = s.price > ds.vwap;
    dots.push({
      label: 'VWAP',
      tone: above ? 'pos' : 'neg',
      title: `VWAP ${formatNumber(ds.vwap)} — price ${above ? 'above' : 'below'}`,
    });
  } else {
    dots.push({ label: 'VWAP', tone: 'neutral', title: 'VWAP pending' });
  }

  // RSI
  if (typeof ds?.rsi === 'number') {
    const rsi = ds.rsi;
    const tone = rsi > 55 ? 'pos' : rsi < 45 ? 'neg' : 'neutral';
    dots.push({ label: 'RSI', tone, title: `RSI ${formatNumber(rsi)}` });
  } else {
    dots.push({ label: 'RSI', tone: 'neutral', title: 'RSI pending' });
  }

  // CPR direction
  if (ds?.cpr_dir) {
    const tone = ds.cpr_dir === 'BULL' ? 'pos' : ds.cpr_dir === 'BEAR' ? 'neg' : 'neutral';
    dots.push({ label: 'CPR', tone, title: `CPR ${ds.cpr_dir}` });
  } else {
    dots.push({ label: 'CPR', tone: 'neutral', title: 'CPR pending' });
  }

  return dots;
}

/* ---------- page ---------- */

export default function Orb() {
  const [state, setState] = useState<ORBState | null>(null);
  const [signals, setSignals] = useState<ORBSignal[]>([]);
  const [candidates, setCandidates] = useState<ORBCandidates | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    // Heavy load (state + signals) — refreshed slowly (10s).
    const loadHeavy = async () => {
      try {
        const [s, sig] = await Promise.all([
          apiGet<ORBState>('/api/orb/state'),
          apiGet<ORBSignal[]>('/api/orb/signals').catch(() => [] as ORBSignal[]),
        ]);
        if (cancelled) return;
        setState(s);
        setSignals(Array.isArray(sig) ? sig : []);
        setErr(null);
      } catch (e) {
        if (!cancelled) setErr(e instanceof Error ? e.message : 'Failed to load ORB state');
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    // Light load (candidates) — tick every 3s so LTP / %s feel live.
    const loadCandidates = async () => {
      try {
        const cand = await apiGet<ORBCandidates>('/api/orb/candidates');
        if (!cancelled) setCandidates(cand);
      } catch {
        /* silent — keep last candidates snapshot */
      }
    };
    loadHeavy();
    loadCandidates();
    const idHeavy = setInterval(loadHeavy, 10_000);
    const idLight = setInterval(loadCandidates, 3_000);
    return () => {
      cancelled = true;
      clearInterval(idHeavy);
      clearInterval(idLight);
    };
  }, []);

  const stocksArr = useMemo(() => {
    if (!state?.stocks) return [];
    return Object.values(state.stocks).sort((a, b) =>
      a.daily_state.instrument.localeCompare(b.daily_state.instrument),
    );
  }, [state]);

  const marginAvail = (state?.margin?.available as number | undefined) ?? 0;

  /* ---------- table columns ---------- */

  const positionCols: Column<ORBPosition>[] = [
    {
      key: 'instrument',
      header: 'Stock',
      width: '1.2fr',
      render: (p) => <span className={styles.bold}>{p.instrument}</span>,
    },
    {
      key: 'direction',
      header: 'Direction',
      width: '90px',
      render: (p) => (
        <span className={p.direction === 'LONG' ? styles.dirLong : styles.dirShort}>
          {p.direction === 'LONG' ? 'Long' : 'Short'}
        </span>
      ),
    },
    { key: 'qty', header: 'Qty', width: '70px', align: 'right', render: (p) => formatInt(p.qty) },
    {
      key: 'entry',
      header: 'Entry',
      width: '1fr',
      align: 'right',
      render: (p) => formatNumber(p.entry_price),
    },
    {
      key: 'sl',
      header: 'SL',
      width: '1fr',
      align: 'right',
      render: (p) => formatNumber(p.sl_price ?? p.stop_loss),
    },
    {
      key: 'ltp',
      header: 'LTP',
      width: '1fr',
      align: 'right',
      render: (p) => <span className={styles.ltpCell}>{formatNumber(p.ltp)}</span>,
    },
    {
      key: 'target',
      header: 'Target',
      width: '1fr',
      align: 'right',
      render: (p) => formatNumber(p.target_price ?? p.target),
    },
    {
      key: 'gauge',
      header: 'SL ── TGT',
      width: '160px',
      render: (p) => <PositionGauge p={p} />,
    },
    {
      key: 'pnl',
      header: 'P&L',
      width: '1.1fr',
      align: 'right',
      render: (p) => (
        <span className={pnlClass(p.pnl_inr)}>{formatPnl(p.pnl_inr)}</span>
      ),
    },
    {
      key: 'time',
      header: 'Entry time',
      width: '90px',
      align: 'right',
      render: (p) => (
        <span className={styles.mute}>{formatTime(p.entry_time)}</span>
      ),
    },
    {
      key: 'grade',
      header: 'Grade',
      width: '60px',
      align: 'right',
      render: (p) => p.conviction_grade ? (
        <span
          className={
            p.conviction_grade === 'A+' ? styles.convApp
            : p.conviction_grade === 'A'  ? styles.convA
            : p.conviction_grade === 'B'  ? styles.convB
            : styles.convC
          }
          title={
            (p.conviction_stars ?? [])
              .map((s) => `${s.hit ? '✓' : '·'} ${s.desc}`)
              .join('\n')
          }
        >
          {p.conviction_grade}
        </span>
      ) : <span className={styles.mute}>—</span>,
    },
  ];

  const tradeCols: Column<ORBClosedTrade>[] = [
    {
      key: 'instrument',
      header: 'Stock',
      width: '1.2fr',
      render: (t) => <span className={styles.bold}>{t.instrument}</span>,
    },
    {
      key: 'direction',
      header: 'Direction',
      width: '90px',
      render: (t) => (t.direction === 'LONG' ? 'Long' : 'Short'),
    },
    { key: 'qty', header: 'Qty', width: '70px', align: 'right', render: (t) => formatInt(t.qty) },
    {
      key: 'entry',
      header: 'Entry',
      width: '1fr',
      align: 'right',
      render: (t) => formatNumber(t.entry_price),
    },
    {
      key: 'exit',
      header: 'Exit',
      width: '1fr',
      align: 'right',
      render: (t) => formatNumber(t.exit_price),
    },
    {
      key: 'reason',
      header: 'Exit reason',
      width: '1.4fr',
      render: (t) => <span className={styles.mute}>{t.exit_reason || '—'}</span>,
    },
    {
      key: 'pnl',
      header: 'P&L',
      width: '1.1fr',
      align: 'right',
      render: (t) => (
        <span className={pnlClass(t.pnl_inr)}>{formatPnl(t.pnl_inr)}</span>
      ),
    },
    {
      key: 'time',
      header: 'Exit time',
      width: '90px',
      align: 'right',
      render: (t) => <span className={styles.mute}>{formatTime(t.exit_time)}</span>,
    },
  ];

  const signalCols: Column<ORBSignal>[] = [
    {
      key: 'time',
      header: 'Time',
      width: '80px',
      render: (s) => <span className={styles.mute}>{formatTime(s.signal_time)}</span>,
    },
    {
      key: 'instrument',
      header: 'Stock',
      width: '1.2fr',
      render: (s) => <span className={styles.bold}>{s.instrument}</span>,
    },
    {
      key: 'direction',
      header: 'Direction',
      width: '90px',
      render: (s) => s.direction ?? '—',
    },
    {
      key: 'price',
      header: 'Price',
      width: '1fr',
      align: 'right',
      render: (s) => formatNumber(s.price),
    },
    {
      key: 'reason',
      header: 'Reason',
      width: '1.8fr',
      render: (s) => <span className={styles.mute}>{s.reason || '—'}</span>,
    },
    {
      key: 'action',
      header: 'Action',
      width: '1.2fr',
      render: (s) => <span className={styles.mute}>{s.action_taken || s.status || '—'}</span>,
    },
  ];

  /* ---------- render ---------- */

  if (loading) {
    return (
      <div className={styles.loading}>Loading ORB state…</div>
    );
  }

  return (
    <div className={styles.root}>
      <div className={styles.headerRow}>
        <div>
          <div className="page-title">Opening range breakout</div>
          <div className="page-subtitle">
            Cash intraday · {state?.universe?.length ?? 0} stocks ·{' '}
            {state?.live_trading ? 'Live trading' : 'Paper trading'}
          </div>
        </div>
      </div>

      {err ? <div className={styles.error}>{err}</div> : null}

      {/* summary metrics */}
      <div className={styles.metrics}>
        <MetricCard
          label="Day P&L"
          value={
            <span className={pnlClass(state?.today_pnl)}>
              {formatPnl(state?.today_pnl)}
            </span>
          }
          hint={`Daily limit ${formatRs(state?.daily_loss_limit)}`}
        />
        <MetricCard
          label="Open positions"
          value={formatInt(state?.open_positions?.length ?? 0)}
          hint={`Max ${state?.config?.max_concurrent_trades ?? '—'} concurrent`}
        />
        <MetricCard
          label="Available margin"
          value={formatRs(marginAvail)}
          hint={`Per trade ${formatRs(state?.config?.allocation_per_trade)}`}
        />
        <MetricCard
          label="Trades today"
          value={formatInt(
            (state?.today_closed?.length ?? 0) + (state?.open_positions?.length ?? 0)
          )}
          hint={
            state?.open_positions?.length
              ? `${state.open_positions.length} open · ${state?.today_closed?.length ?? 0} closed`
              : `${state?.universe?.length ?? 0} stocks scanning`
          }
        />
      </div>

      {/* open positions */}
      <section className={styles.section}>
        <div className={styles.sectionHead}>
          <div className="section-title">Open positions</div>
          <Chip>
            {state?.open_positions?.length ?? 0} active
          </Chip>
        </div>
        <DataTable
          columns={positionCols}
          rows={state?.open_positions ?? []}
          emptyText="No open positions"
          rowKey={(p) => p.instrument + (p.entry_time ?? '')}
        />
      </section>

      {/* live candidates */}
      {candidates ? <CandidatesSection candidates={candidates} /> : null}

      {/* current indicators */}
      <section className={styles.section}>
        <div className={styles.sectionHead}>
          <div className="section-title">Current indicators</div>
          <Chip>Config</Chip>
        </div>
        <CurrentIndicators state={state} />
      </section>

      {/* stock scan grid */}
      <section className={styles.section}>
        <div className={styles.sectionHead}>
          <div className="section-title">Stock scan</div>
          <Chip>{stocksArr.length} stocks</Chip>
        </div>
        <div className={styles.stockGrid}>
          {stocksArr.map((s) => (
            <StockCard key={s.daily_state.instrument} stock={s} />
          ))}
          {stocksArr.length === 0 ? (
            <div className={styles.empty}>
              No stocks loaded yet. Click "Initialize day" to compute OR levels.
            </div>
          ) : null}
        </div>
      </section>

      {/* whats next */}
      <section className={styles.section}>
        <div className={styles.sectionHead}>
          <div className="section-title">What's next</div>
          <Chip>Today's schedule</Chip>
        </div>
        <WhatsNext state={state} />
      </section>

      {/* signals */}
      <section className={styles.section}>
        <div className={styles.sectionHead}>
          <div className="section-title">Today's signals</div>
          <Chip>{signals.length} events</Chip>
        </div>
        <DataTable
          columns={signalCols}
          rows={signals.slice(0, 30)}
          emptyText="No signals yet today"
          rowKey={(s, i) => s.id ?? `${s.instrument}-${i}`}
        />
      </section>

      {/* recent trades */}
      <section className={styles.section}>
        <div className={styles.sectionHead}>
          <div className="section-title">Today's trades</div>
          <Chip>{state?.today_closed?.length ?? 0} closed</Chip>
        </div>
        <DataTable
          columns={tradeCols}
          rows={state?.today_closed ?? []}
          emptyText="No trades closed today"
          rowKey={(t, i) => `${t.instrument}-${t.exit_time ?? i}`}
        />
      </section>

      {/* system rules */}
      <section className={styles.section}>
        <details className={styles.rulesBlock}>
          <summary className={styles.rulesSummary}>Strategy rules · V9t_lock50</summary>
          <div className={styles.rulesBody}>
            <div className={styles.ruleItem}>
              <span className={styles.ruleLabel}>Setup</span>
              <span>
                15 Nifty 500 stocks · MIS cash · OR = 09:15–09:30 (15 min) · max 5 concurrent · 1.2× margin buffer.
              </span>
            </div>
            <div className={styles.ruleItem}>
              <span className={styles.ruleLabel}>Allocation &amp; risk (live)</span>
              <span>
                <table className={styles.allocTable}>
                  <tbody>
                    <tr>
                      <td>Capital (deposit)</td>
                      <td>{formatRs(state?.capital)}</td>
                    </tr>
                    <tr>
                      <td>MIS leverage</td>
                      <td>{state?.mis_leverage ?? 1}× · buying power {formatRs((state?.capital ?? 0) * (state?.mis_leverage ?? 1))}</td>
                    </tr>
                    <tr>
                      <td>Per-trade notional cap</td>
                      <td>{formatRs(state?.max_notional_per_trade)} (= capital × leverage ÷ max_concurrent)</td>
                    </tr>
                    <tr>
                      <td>Risk per trade</td>
                      <td>{((state?.risk_per_trade_pct ?? 0) * 100).toFixed(1)}% of capital = {formatRs((state?.capital ?? 0) * (state?.risk_per_trade_pct ?? 0))}</td>
                    </tr>
                    <tr>
                      <td>Max DD if 5 SLs hit</td>
                      <td>
                        {formatRs((state?.capital ?? 0) * (state?.risk_per_trade_pct ?? 0) * 5)}
                        {' '}({((state?.risk_per_trade_pct ?? 0) * 5 * 100).toFixed(1)}% of capital)
                      </td>
                    </tr>
                    <tr>
                      <td>Daily loss cap (display)</td>
                      <td>
                        {formatRs(state?.daily_loss_limit)} ({((state?.daily_loss_limit_pct ?? 0) * 100).toFixed(1)}%)
                        {state?.enforce_daily_loss_cap === false
                          ? <span className={styles.mute}> · <b style={{ color: 'var(--accent-neg)' }}>enforcement OFF</b></span>
                          : null}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </span>
            </div>
            <div className={styles.ruleItem}>
              <span className={styles.ruleLabel}>Entry trigger</span>
              <span>
                5-min candle closes beyond OR. <b>LONG</b>: close &gt; OR_high AND prev_close ≤ OR_high ·
                <b> SHORT</b>: close &lt; OR_low AND prev_close ≥ OR_low. LIMIT order with 0.2% buffer.
              </span>
            </div>
            <div className={styles.ruleItem}>
              <span className={styles.ruleLabel}>Filters (all must pass)</span>
              <span>
                CPR width &lt; 0.65% · CPR direction (LONG if gap opens ≥ TC, SHORT if ≤ BC) ·
                Gap ≤ +0.3% for LONG · VWAP direction (close above for LONG / below for SHORT) ·
                RSI(15m) ≥ 50 for LONG, ≤ 50 for SHORT · Wide-CPR days skipped entirely ·
                Max 1 trade/stock/day · Last entry 14:00.
              </span>
            </div>
            <div className={styles.ruleItem}>
              <span className={styles.ruleLabel}>Stop loss</span>
              <span>
                OR opposite edge (LONG = OR_low, SHORT = OR_high). Risk R = |entry − SL|.
                Placed as <b>exchange SL order</b> (trigger=SL, limit=SL±0.5%) immediately after
                entry fill — survives restarts and crashes.
              </span>
            </div>
            <div className={styles.ruleItem}>
              <span className={styles.ruleLabel}>Target</span>
              <span>1.5 × R. LONG = entry + 1.5R · SHORT = entry − 1.5R. Monitored by 30s LTP poll; on hit, cancels exchange SL then market exits.</span>
            </div>
            <div className={styles.ruleItem}>
              <span className={styles.ruleLabel}>14:30 · V9t_lock50 trail (strict)</span>
              <span>
                <b>Profitable</b> → SL = <code>entry ± 0.5 × gain</code> (locks half the unrealized P&amp;L).
                <b> Losing</b> → SL = <code>entry</code> (breakeven — cuts afternoon drawdowns).
                Only tightens, never loosens. Modifies the exchange SL trigger to match. Frozen thereafter until SL hits or 15:18 EOD.
                <br /><span className={styles.mute}>Backtest 60-day: strict Calmar 3,152 vs lenient (profitable-only) 466 · strict beats lenient by Rs 6.2K/day.</span>
              </span>
            </div>
            <div className={styles.ruleItem}>
              <span className={styles.ruleLabel}>EOD squareoff</span>
              <span>15:18 sharp · hard close all remaining positions at market (2 min before Zerodha's 15:20 MIS auto-squareoff).</span>
            </div>
            <div className={styles.ruleItem}>
              <span className={styles.ruleLabel}>Recovery (catchup)</span>
              <span>
                If OR-finalize fails or service restarts after 09:30, <code>/api/orb/catchup</code> walks post-OR candles,
                picks the <b>latest</b> transition still consistent with LTP, applies all filters, enters only if slippage ≤ 0.5R.
              </span>
            </div>
            <div className={styles.ruleItem}>
              <span className={styles.ruleLabel}>Position sizing</span>
              <span>
                {state?.use_risk_based_sizing ? (
                  <>
                    <b>Risk-based</b>:
                    {' '}<code>qty = floor((capital × {(state.risk_per_trade_pct ?? 0) * 100}%) / |entry − SL|)</code>,
                    {' '}capped at notional {formatRs(state.max_notional_per_trade)}.
                    {' '}Target risk ≈ {formatRs((state.capital ?? 0) * (state.risk_per_trade_pct ?? 0))} per trade regardless
                    of which stock fires. Same SL rule (OR-opposite) — tight OR → bigger qty, wide OR → smaller qty.
                  </>
                ) : (
                  <code>qty = floor({formatRs(state?.config?.allocation_per_trade)} / entry_price)</code>
                )}
              </span>
            </div>
            <div className={styles.ruleItem}>
              <span className={styles.ruleLabel}>Conviction grade</span>
              <span>
                4-star rubric at entry (persisted on position): ⭐ CPR &lt; 0.3% · ⭐ OR width &lt; 0.8% ·
                ⭐ RSI ≥ 65 (LONG) / ≤ 35 (SHORT) · ⭐ Past% in 0.2–0.7.
                A+ = 4/4, A = 3/4, B = 2/4, C = ≤1/4.
              </span>
            </div>
            <div className={styles.ruleItem}>
              <span className={styles.ruleLabel}>Daily loss limit</span>
              <span>
                {state?.daily_loss_limit_pct != null
                  ? `${(state.daily_loss_limit_pct * 100).toFixed(1)}% of capital = ${formatRs(state.daily_loss_limit)}`
                  : `${formatRs(state?.daily_loss_limit ?? 3000)}`}
                {state?.enforce_daily_loss_cap === false ? (
                  <>
                    {' '}· <b style={{ color: 'var(--accent-neg)' }}>Enforcement OFF</b>{' '}
                    — cap is displayed but does not block entries or force-close.
                  </>
                ) : (
                  <>
                    {' '}— realized loss at cap <b>blocks new entries</b>; MTM loss at
                    1.5× cap triggers <b>panic force-close of all open positions</b>.
                  </>
                )}
              </span>
            </div>
          </div>
        </details>
      </section>
    </div>
  );
}

/* ---------- Current indicators ---------- */

function CurrentIndicators({ state }: { state: ORBState | null }) {
  const cfg = state?.config;
  const items: { label: string; value: React.ReactNode }[] = [
    {
      label: 'Allocation per trade',
      value: cfg ? formatRs(cfg.allocation_per_trade) : '—',
    },
    {
      label: 'Min margin',
      value: cfg ? formatRs(cfg.min_margin_for_trade) : '—',
    },
    {
      label: 'Max concurrent',
      value: cfg ? formatInt(cfg.max_concurrent_trades) : '—',
    },
    {
      label: 'R-multiple',
      value: cfg ? `${cfg.r_multiple}x` : '—',
    },
    {
      label: 'SL type',
      value: cfg?.sl_type ?? '—',
    },
    {
      label: 'Target type',
      value: cfg ? `${cfg.r_multiple}x risk` : '—',
    },
    {
      label: 'OR minutes',
      value: cfg ? `${cfg.or_minutes}m` : '—',
    },
    {
      label: 'Last entry',
      value: cfg?.last_entry_time ?? '—',
    },
    {
      label: 'EOD exit',
      value: cfg?.eod_exit_time ?? '—',
    },
  ];
  return (
    <div className={styles.indicatorsGrid}>
      {items.map((it) => (
        <div key={it.label} className={styles.indicatorItem}>
          <div className={styles.indicatorLabel}>{it.label}</div>
          <div className={styles.indicatorValue}>{it.value}</div>
        </div>
      ))}
    </div>
  );
}

/* ---------- Whats next ---------- */

interface NextEvent {
  event: string;
  scheduled: string;
  status: string;
  tone: 'pos' | 'neg' | 'neutral';
  relative: string;
  sortKey: number;
}

function minutesFromNowIST(hhmm: string): number {
  const now = new Date();
  const parts = new Intl.DateTimeFormat('en-IN', {
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
    timeZone: 'Asia/Kolkata',
  })
    .format(now)
    .split(':');
  const nowMin = parseInt(parts[0], 10) * 60 + parseInt(parts[1], 10);
  const [h, m] = hhmm.split(':').map((x) => parseInt(x, 10));
  return h * 60 + m - nowMin;
}

function relativeLabel(diffMin: number): string {
  if (diffMin < 0) return 'passed';
  if (diffMin === 0) return 'now';
  const h = Math.floor(diffMin / 60);
  const m = diffMin % 60;
  if (h === 0) return `in ${m}m`;
  return `in ${h}h ${m}m`;
}

function WhatsNext({ state }: { state: ORBState | null }) {
  // Determine current statuses based on state
  const stocksArr = Object.values(state?.stocks ?? {});
  const anyOrFinalized = stocksArr.some((s) => s.daily_state?.or_finalized);
  const anyTodayOpen = stocksArr.some((s) => s.daily_state?.today_open);
  const openPositions = state?.open_positions?.length ?? 0;

  // ORB init 9:14
  const initDiff = minutesFromNowIST('09:14');
  // OR tracking 9:15-9:30
  const orStartDiff = minutesFromNowIST('09:15');
  const orEndDiff = minutesFromNowIST('09:30');
  const orInProgress = orStartDiff <= 0 && orEndDiff > 0;
  // Signal eval 9:30 to 14:00
  const sigStartDiff = minutesFromNowIST('09:30');
  const sigEndDiff = minutesFromNowIST('14:00');
  const sigActive = sigStartDiff <= 0 && sigEndDiff > 0;
  // EOD 15:20
  const eodDiff = minutesFromNowIST('15:20');

  const events: NextEvent[] = [
    {
      event: 'ORB initialize',
      scheduled: '09:14',
      status: anyTodayOpen ? 'Done' : initDiff < 0 ? 'Missed' : 'Pending',
      tone: anyTodayOpen ? 'neutral' : initDiff < 0 ? 'neg' : 'pos',
      relative: initDiff < 0 ? 'passed' : relativeLabel(initDiff),
      sortKey: initDiff < 0 ? 9999 : initDiff,
    },
    {
      event: 'OR tracking (9:15-9:30)',
      scheduled: '09:15-09:30',
      status: anyOrFinalized ? 'Done' : orInProgress ? 'In progress' : orEndDiff < 0 ? 'Done' : 'Pending',
      tone: anyOrFinalized || orEndDiff < 0 ? 'neutral' : orInProgress ? 'pos' : 'pos',
      relative: orInProgress ? 'active' : orStartDiff > 0 ? relativeLabel(orStartDiff) : 'passed',
      sortKey: orInProgress ? -1 : orStartDiff > 0 ? orStartDiff : 9999,
    },
    {
      event: 'Signal eval (every 5 min)',
      scheduled: '09:30-14:00',
      status: sigActive ? 'Active' : sigEndDiff < 0 ? 'Done' : 'Pending',
      tone: sigActive ? 'pos' : 'neutral',
      relative: sigActive ? 'active' : sigStartDiff > 0 ? relativeLabel(sigStartDiff) : 'passed',
      sortKey: sigActive ? -1 : sigStartDiff > 0 ? sigStartDiff : 9999,
    },
    {
      event: 'Position monitor (every 30s)',
      scheduled: 'Continuous',
      status: openPositions > 0 ? 'Active' : 'Idle',
      tone: openPositions > 0 ? 'pos' : 'neutral',
      relative: openPositions > 0 ? `${openPositions} open` : 'no positions',
      sortKey: openPositions > 0 ? -1 : 0,
    },
    {
      event: 'EOD squareoff',
      scheduled: '15:20',
      status: eodDiff < 0 ? 'Done' : 'Pending',
      tone: eodDiff < 0 ? 'neutral' : 'pos',
      relative: relativeLabel(eodDiff),
      sortKey: eodDiff < 0 ? 9999 : eodDiff,
    },
  ];

  events.sort((a, b) => a.sortKey - b.sortKey);

  return (
    <div className={styles.eventsTable}>
      <div className={styles.eventsHead}>
        <div>Event</div>
        <div>Scheduled</div>
        <div>Status</div>
        <div className={styles.eventsHeadRight}>In</div>
      </div>
      {events.map((ev, i) => (
        <div key={i} className={styles.eventsRow}>
          <div className={styles.eventsEvent}>{ev.event}</div>
          <div className={styles.eventsTime}>{ev.scheduled}</div>
          <div>
            <span className={`${styles.eventStatus} ${styles[`eventStatus_${ev.tone}`]}`}>
              {ev.status}
            </span>
          </div>
          <div className={styles.eventsHeadRight}>
            <span className={styles.mute}>{ev.relative}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

/* ---------- Stock card ---------- */

function StockCard({ stock }: { stock: ORBStockSummary }) {
  const ds = stock.daily_state;
  const label = stockStatus(stock);
  const tone = statusTone(label);
  const dots = filterDots(stock);

  const pnl = stock.position?.pnl_inr ?? stock.today_result?.pnl_inr;

  return (
    <div className={styles.stockCard}>
      <div className={styles.stockHead}>
        <div className={styles.stockSym}>{ds.instrument}</div>
        <div className={styles.stockLtp}>
          {stock.position?.ltp
            ? formatNumber(stock.position.ltp)
            : stock.price
            ? formatNumber(stock.price)
            : '—'}
        </div>
      </div>

      <div className={styles.stockRanges}>
        <div className={styles.range}>
          <div className={styles.rangeLabel}>OR high</div>
          <div className={styles.rangeValue}>
            {ds.or_high ? formatNumber(ds.or_high) : '—'}
          </div>
        </div>
        <div className={styles.range}>
          <div className={styles.rangeLabel}>OR low</div>
          <div className={styles.rangeValue}>
            {ds.or_low ? formatNumber(ds.or_low) : '—'}
          </div>
        </div>
      </div>

      <div className={styles.stockDots}>
        {dots.map((d) => (
          <div key={d.label} className={styles.dotItem} title={d.title}>
            <span className={`${styles.fdot} ${styles[`dot_${d.tone}`]}`} />
            <span>{d.label}</span>
          </div>
        ))}
      </div>

      <div className={styles.stockFoot}>
        <span className={`${styles.status} ${styles[`status_${tone}`]}`}>{label}</span>
        {pnl !== undefined ? (
          <span className={pnlClass(pnl)} style={{ fontSize: 'var(--text-xxs)' }}>
            {formatPnl(pnl)}
          </span>
        ) : null}
      </div>
    </div>
  );
}

/* ---------- Candidates section ---------- */

function riskTone(pct: number | undefined): string {
  // Risk (OR width as % of price): <0.5% tight (green), 0.5–1% ok, >1% wide (red)
  const p = pct ?? 0;
  if (p < 0.5) return styles.riskTight;
  if (p < 1.0) return styles.riskOk;
  return styles.riskWide;
}

function cprTone(pct: number | undefined, wide: boolean | undefined): string {
  if (wide) return styles.cprWide;
  const p = pct ?? 0;
  if (p < 0.3) return styles.cprNarrow;
  return styles.cprOk;
}

function rsiTone(rsi: number | null | undefined): string {
  if (rsi == null) return styles.rsiMute;
  if (rsi >= 60) return styles.rsiLong;
  if (rsi <= 40) return styles.rsiShort;
  return styles.rsiDead;
}

function CandidatesSection({ candidates }: { candidates: ORBCandidates }) {
  const { broken_out, watching, excluded, as_of } = candidates;
  const wideCpr = excluded.filter((e) => e.reason.startsWith('wide_cpr'));
  const otherExcl = excluded.filter((e) => !e.reason.startsWith('wide_cpr'));
  const fmtTime = (iso: string) => {
    try {
      const d = new Date(iso);
      return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}:${String(d.getSeconds()).padStart(2, '0')}`;
    } catch {
      return iso.slice(11, 19);
    }
  };

  return (
    <section className={styles.section}>
      <div className={styles.sectionHead}>
        <div className="section-title">Live candidates</div>
        <Chip>as of {fmtTime(as_of)}</Chip>
      </div>
      <div className={styles.candidatesCard}>
        {/* Header (shared across groups) */}
        <div className={styles.candHeader}>
          <span>Symbol</span>
          <span>Status</span>
          <span>LTP</span>
          <span>OR range</span>
          <span title="OR width as % of LTP — ≈ SL risk if the trade fires">Risk %</span>
          <span title="CPR width % — narrow = clean setup, wide = skip">CPR %</span>
          <span title="15-min RSI — ≥60 allows LONG, ≤40 allows SHORT">RSI</span>
          <span>Signal</span>
        </div>

        {/* Broken-out */}
        <div className={styles.candGroup}>
          <div className={`${styles.candGroupHead} ${styles.broken}`}>
            <span className={`${styles.candGroupDot} ${styles.candGroupDotBroken}`} />
            <span className={styles.candGroupTitle}>Broken out · awaiting signal eval</span>
            <span className={styles.candCount}>{broken_out.length}</span>
          </div>
          {broken_out.length === 0 ? (
            <div className={styles.candEmpty}>No breakouts past OR levels right now.</div>
          ) : (
            <div className={styles.candRows}>
              {broken_out.map((r) => (
                <div key={r.sym} className={styles.candRow}>
                  <span className={styles.candSym}>{r.sym}</span>
                  <span className={`${styles.statusPill} ${r.side === 'LONG' ? styles.statusLong : styles.statusShort}`}>
                    {r.side === 'LONG' ? '▲' : '▼'} {r.side}
                  </span>
                  <span className={styles.candLtp}>{formatNumber(r.ltp)}</span>
                  <span className={styles.candOr}>
                    {formatNumber(r.or_low)}–{formatNumber(r.or_high)}
                  </span>
                  <span className={riskTone(r.or_width_pct)}>{(r.or_width_pct ?? 0).toFixed(2)}%</span>
                  <span className={cprTone(r.cpr_width_pct, r.cpr_is_wide)}>{r.cpr_width_pct.toFixed(2)}%</span>
                  <span className={rsiTone(r.rsi_15m)}>{r.rsi_15m != null ? r.rsi_15m.toFixed(0) : '—'}</span>
                  <span className={styles.candSignal}>
                    <span className={r.side === 'LONG' ? styles.pastPos : styles.pastNeg}>
                      {r.past_pct >= 0 ? '+' : ''}{r.past_pct.toFixed(2)}% past
                    </span>
                    {r.conviction_grade ? (
                      <span
                        className={`${styles.convBadge} ${
                          r.conviction_grade === 'A+' ? styles.convApp
                          : r.conviction_grade === 'A'  ? styles.convA
                          : r.conviction_grade === 'B'  ? styles.convB
                          : styles.convC
                        }`}
                        title={
                          (r.conviction_stars ?? [])
                            .map((s) => `${s.hit ? '✓' : '·'}  ${s.desc}`)
                            .join('\n')
                        }
                      >
                        {r.conviction_grade}
                      </span>
                    ) : null}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Watching */}
        <div className={styles.candGroup}>
          <div className={`${styles.candGroupHead} ${styles.watching}`}>
            <span className={`${styles.candGroupDot} ${styles.candGroupDotWatching}`} />
            <span className={styles.candGroupTitle}>Watching · inside OR</span>
            <span className={styles.candCount}>{watching.length}</span>
          </div>
          {watching.length === 0 ? (
            <div className={styles.candEmpty}>No stocks inside OR right now.</div>
          ) : (
            <div className={styles.candRows}>
              {watching.map((r) => {
                const nearest = Math.min(
                  Math.abs(r.dist_up_pct ?? 99),
                  Math.abs(r.dist_dn_pct ?? 99),
                );
                const statusClass =
                  r.side_hint === 'both'
                    ? styles.statusBoth
                    : r.side_hint === 'long'
                    ? styles.statusLong
                    : r.side_hint === 'short'
                    ? styles.statusShort
                    : styles.statusBlocked;
                return (
                  <div key={r.sym} className={styles.candRow}>
                    <span className={styles.candSym}>{r.sym}</span>
                    <span className={`${styles.statusPill} ${statusClass}`}>
                      {r.side_hint.toUpperCase()}
                    </span>
                    <span className={styles.candLtp}>{formatNumber(r.ltp)}</span>
                    <span className={styles.candOr}>
                      {formatNumber(r.or_low)}–{formatNumber(r.or_high)}
                    </span>
                    <span className={riskTone(r.or_width_pct)}>{(r.or_width_pct ?? 0).toFixed(2)}%</span>
                    <span className={cprTone(r.cpr_width_pct, r.cpr_is_wide)}>{r.cpr_width_pct.toFixed(2)}%</span>
                    <span className={rsiTone(r.rsi_15m)}>{r.rsi_15m != null ? r.rsi_15m.toFixed(0) : '—'}</span>
                    <span className={styles.candDist}>
                      ↑{(r.dist_up_pct ?? 0).toFixed(2)}% · ↓{(r.dist_dn_pct ?? 0).toFixed(2)}% · nearest{' '}
                      <b>{nearest.toFixed(2)}%</b>
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Excluded */}
        {(wideCpr.length > 0 || otherExcl.length > 0) ? (
          <div className={styles.candGroup}>
            <div className={`${styles.candGroupHead} ${styles.excluded}`}>
              <span className={`${styles.candGroupDot} ${styles.candGroupDotExcluded}`} />
              <span className={styles.candGroupTitle}>Excluded</span>
              <span className={styles.candCount}>{excluded.length}</span>
            </div>
            {wideCpr.length > 0 ? (
              <div className={styles.candExclRow}>
                <span className={styles.candExclLabel}>Wide CPR</span>
                <span className={styles.candExclList}>
                  {wideCpr.map((e) => (
                    <span key={e.sym} className={styles.exclChip}>
                      {e.sym}
                      <span className={styles.exclChipPct}>{(e.cpr ?? 0).toFixed(2)}%</span>
                    </span>
                  ))}
                </span>
              </div>
            ) : null}
            {otherExcl.length > 0 ? (
              <div className={styles.candExclRow}>
                <span className={styles.candExclLabel}>Other</span>
                <span className={styles.candExclList}>
                  {otherExcl.map((e) => (
                    <span key={e.sym} className={styles.exclChip}>
                      {e.sym}
                      <span className={styles.exclChipPct}>{e.reason}</span>
                    </span>
                  ))}
                </span>
              </div>
            ) : null}
          </div>
        ) : null}

        {/* Rubric explainer (collapsed, footer of Live candidates) */}
        <details className={styles.candRubric}>
          <summary className={styles.candRubricSummary}>
            How conviction grade is calculated (A+ / A / B / C)
          </summary>
          <div className={styles.candRubricBody}>
            <div className={styles.candRubricIntro}>
              Broken-out candidates earn 1 star for each quality criterion hit. Total stars → grade:{' '}
              <b>4★ A+</b> · <b>3★ A</b> · <b>2★ B</b> · <b>1★ C</b> · 0★ hidden.
            </div>
            <div className={styles.candRubricTable}>
              <div className={styles.candRubricRow}><span>🎯 CPR narrow</span><span>CPR width &lt; 0.3%</span><span>Clean prior-day compression → stronger setup</span></div>
              <div className={styles.candRubricRow}><span>📏 Tight risk</span><span>Risk % &lt; 0.8</span><span>Small SL → better R/R on the 1.5R target</span></div>
              <div className={styles.candRubricRow}><span>💪 RSI conviction</span><span>LONG RSI ≥ 65 · SHORT ≤ 35</span><span>Momentum clearly confirms, not borderline</span></div>
              <div className={styles.candRubricRow}><span>✅ Clean past %</span><span>0.2% ≤ past % ≤ 0.7%</span><span>Past enough to confirm, not so far you're chasing</span></div>
            </div>
          </div>
        </details>
      </div>
    </section>
  );
}
