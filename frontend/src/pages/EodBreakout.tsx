import { useEffect, useMemo, useState } from 'react';
import styles from './EodBreakout.module.css';
import { apiGet } from '../api/client';
import DataTable from '../components/DataTable/DataTable';
import type { Column } from '../components/DataTable/DataTable';
import Chip from '../components/Chip/Chip';
import { formatInt, formatPnl, formatRs, pnlClass } from '../utils/format';

type SystemId = 'nifty500' | 'smallcap' | 'fno';

interface SystemSummary {
  system_id: SystemId;
  description: string;
  capital: number;
  risk_per_trade_pct: number;
  max_concurrent: number;
  cost_pct: number;
  vol_threshold_mult: number;
  target_pct: number;
  initial_hard_stop_pct: number;
  universe_size: number;
  open_positions: number;
  total_trades: number;
  win_rate: number;
  profit_factor: number;
  total_pnl: number;
  avg_days_held: number;
  today_signals: number;
  today_fills: number;
  today_exits: number;
}

interface Position {
  id: number;
  symbol: string;
  entry_date: string;
  entry_price: number;
  qty: number;
  initial_stop: number;
  target: number;
  notional_inr: number;
}

interface Signal {
  id: number;
  signal_date: string;
  symbol: string;
  signal_close: number;
  breakout_high: number | null;
  vol_ratio: number | null;
  status: string;
  notes: string | null;
}

interface Trade {
  id: number;
  symbol: string;
  entry_date: string;
  exit_date: string;
  entry_price: number;
  exit_price: number;
  qty: number;
  days_held: number;
  exit_reason: string;
  net_pnl: number;
}

// Backtest summary metrics — sourced from research/17, 19, 21 FINDINGS.md
const BACKTEST_RESULTS: Record<SystemId, {
  variant: string;
  is_pf: number; is_sharpe: number; is_maxdd: number; is_cagr: number;
  oos_pf: number; oos_sharpe: number; oos_maxdd: number; oos_cagr: number;
  oos_trades: number;
  pass: boolean;
}> = {
  nifty500: {
    variant: 'D1_fixed_25_8 (research/17)',
    is_pf: 1.51, is_sharpe: 0.82, is_maxdd: 35.94, is_cagr: 13.36,
    oos_pf: 1.44, oos_sharpe: 0.95, oos_maxdd: 23.98, oos_cagr: 14.54,
    oos_trades: 208, pass: true,
  },
  smallcap: {
    variant: 'vol_3x (research/19)',
    is_pf: 1.93, is_sharpe: 0.79, is_maxdd: 17.46, is_cagr: 10.72,
    oos_pf: 1.46, oos_sharpe: 1.43, oos_maxdd: 12.85, oos_cagr: 26.79,
    oos_trades: 305, pass: true,
  },
  fno: {
    variant: 'vol_3x (research/21)',
    is_pf: 2.06, is_sharpe: 1.18, is_maxdd: 20.03, is_cagr: 11.22,
    oos_pf: 1.91, oos_sharpe: 1.04, oos_maxdd: 9.11, oos_cagr: 10.25,
    oos_trades: 134, pass: true,
  },
};

const SYSTEM_LABELS: Record<SystemId, string> = {
  nifty500: 'Nifty 500',
  smallcap: 'Small/Mid Cap',
  fno: 'F&O Universe',
};

export default function EodBreakout() {
  const [systems, setSystems] = useState<SystemSummary[]>([]);
  const [active, setActive] = useState<SystemId>('nifty500');
  const [positions, setPositions] = useState<Position[]>([]);
  const [signals, setSignals] = useState<Signal[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [err, setErr] = useState<string | null>(null);

  // Top-level state poll
  useEffect(() => {
    let cancelled = false;
    const load = () =>
      apiGet<{ systems: SystemSummary[] }>('/api/eod/state')
        .then((r) => {
          if (!cancelled) {
            setSystems(r.systems);
            setErr(null);
          }
        })
        .catch((e) => {
          if (!cancelled) setErr(e instanceof Error ? e.message : 'Load failed');
        });
    load();
    const id = setInterval(load, 30_000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  // Per-system poll (positions / signals / trades)
  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const [p, s, t] = await Promise.all([
          apiGet<{ positions: Position[] }>(`/api/eod/${active}/positions`).catch(() => ({ positions: [] })),
          apiGet<{ signals: Signal[] }>(`/api/eod/${active}/signals`).catch(() => ({ signals: [] })),
          apiGet<{ trades: Trade[] }>(`/api/eod/${active}/trades`).catch(() => ({ trades: [] })),
        ]);
        if (cancelled) return;
        setPositions(p.positions || []);
        setSignals(s.signals || []);
        setTrades(t.trades || []);
      } catch {
        /* keep last */
      }
    };
    load();
    const id = setInterval(load, 30_000);
    return () => { cancelled = true; clearInterval(id); };
  }, [active]);

  const activeSys = useMemo(
    () => systems.find((s) => s.system_id === active),
    [systems, active]
  );

  const positionCols: Column<Position>[] = [
    { key: 'symbol', header: 'Symbol', width: '1.2fr', render: (p) => p.symbol },
    { key: 'entry_date', header: 'Entry', width: '1fr', render: (p) => p.entry_date },
    { key: 'entry_price', header: 'Entry Px', width: '1fr', align: 'right', render: (p) => `Rs ${p.entry_price.toFixed(2)}` },
    { key: 'qty', header: 'Qty', width: '0.6fr', align: 'right', render: (p) => formatInt(p.qty) },
    { key: 'stop', header: 'Stop', width: '1fr', align: 'right', render: (p) => `Rs ${p.initial_stop.toFixed(2)}` },
    { key: 'target', header: 'Target', width: '1fr', align: 'right', render: (p) => `Rs ${p.target.toFixed(2)}` },
    { key: 'notional', header: 'Notional', width: '1fr', align: 'right', render: (p) => formatRs(p.notional_inr) },
  ];

  const signalCols: Column<Signal>[] = [
    { key: 'date', header: 'Date', width: '1fr', render: (s) => s.signal_date },
    { key: 'symbol', header: 'Symbol', width: '1.2fr', render: (s) => s.symbol },
    { key: 'close', header: 'Close', width: '1fr', align: 'right', render: (s) => `Rs ${s.signal_close.toFixed(2)}` },
    { key: 'vol', header: 'Vol×', width: '0.8fr', align: 'right', render: (s) => s.vol_ratio ? `${s.vol_ratio.toFixed(1)}×` : '—' },
    { key: 'status', header: 'Status', width: '1fr', render: (s) => <Chip>{s.status}</Chip> },
  ];

  const tradeCols: Column<Trade>[] = [
    { key: 'symbol', header: 'Symbol', width: '1.2fr', render: (t) => t.symbol },
    { key: 'entry', header: 'Entry', width: '1fr', render: (t) => t.entry_date },
    { key: 'exit', header: 'Exit', width: '1fr', render: (t) => t.exit_date },
    { key: 'days', header: 'Days', width: '0.6fr', align: 'right', render: (t) => formatInt(t.days_held) },
    { key: 'reason', header: 'Reason', width: '0.9fr', render: (t) => <Chip>{t.exit_reason}</Chip> },
    { key: 'pnl', header: 'Net P&L', width: '1.2fr', align: 'right',
      render: (t) => <span className={pnlClass(t.net_pnl)}>{formatPnl(t.net_pnl)}</span> },
  ];

  const bt = BACKTEST_RESULTS[active];

  return (
    <div className={styles.root}>
      <div className="page-title">EOD Breakout</div>
      <div className="page-subtitle">
        Daily-bar 252-day breakout + volume + 200-SMA regime filter. Three sub-systems
        on different universes — same rules. Paper-trading, runs daily 16:00 IST.
      </div>

      {err ? <div style={{ color: 'var(--accent-neg)' }}>API error: {err}</div> : null}

      <div className={styles.tabBar}>
        {(['nifty500', 'smallcap', 'fno'] as SystemId[]).map((sid) => {
          const sys = systems.find((s) => s.system_id === sid);
          return (
            <button
              key={sid}
              className={`${styles.tab} ${active === sid ? styles.tabActive : ''}`}
              onClick={() => setActive(sid)}
            >
              {SYSTEM_LABELS[sid]}
              {sys ? ` (${sys.universe_size})` : ''}
            </button>
          );
        })}
      </div>

      {activeSys ? (
        <>
          <div className={styles.systemHeader}>
            <div className={styles.systemTitle}>{activeSys.description}</div>
            <div className={styles.systemSub}>
              Universe: {activeSys.universe_size} stocks · Capital: {formatRs(activeSys.capital)}
              · Risk/trade: {(activeSys.risk_per_trade_pct * 100).toFixed(1)}%
              · Max concurrent: {activeSys.max_concurrent}
              · Cost: {(activeSys.cost_pct * 100).toFixed(2)}% round-trip
            </div>
            <div className={styles.metricsRow}>
              <div className={styles.metric}>
                <div className={styles.metricLabel}>Open positions</div>
                <div className={styles.metricValue}>{activeSys.open_positions}</div>
              </div>
              <div className={styles.metric}>
                <div className={styles.metricLabel}>Total trades</div>
                <div className={styles.metricValue}>{activeSys.total_trades}</div>
              </div>
              <div className={styles.metric}>
                <div className={styles.metricLabel}>Win rate</div>
                <div className={styles.metricValue}>{activeSys.win_rate.toFixed(1)}%</div>
              </div>
              <div className={styles.metric}>
                <div className={styles.metricLabel}>PF</div>
                <div className={styles.metricValue}>{activeSys.profit_factor.toFixed(2)}</div>
              </div>
              <div className={styles.metric}>
                <div className={styles.metricLabel}>Net P&L</div>
                <div className={styles.metricValue}>
                  <span className={pnlClass(activeSys.total_pnl)}>{formatPnl(activeSys.total_pnl)}</span>
                </div>
              </div>
              <div className={styles.metric}>
                <div className={styles.metricLabel}>Today</div>
                <div className={styles.metricValue}>
                  {activeSys.today_signals}sig · {activeSys.today_fills}fill · {activeSys.today_exits}exit
                </div>
              </div>
            </div>
          </div>

          {/* Rules */}
          <section className={styles.section}>
            <div className={styles.sectionHead}>
              <div className="section-title">Rules</div>
              <Chip>Locked from research/17, 19, 21</Chip>
            </div>
            <div className={styles.rulesBlock}>
              <div className={styles.rulesGrid}>
                <span className={styles.ruleLabel}>Entry</span>
                <span className={styles.ruleVal}>
                  Today's close &gt; 252-day high (excl today)
                  AND volume &ge; {activeSys.vol_threshold_mult}× 50-day avg
                  AND close &gt; 200-day SMA
                </span>
                <span className={styles.ruleLabel}>Execution</span>
                <span className={styles.ruleVal}>EOD signal → fill at next day's open price (paper)</span>
                <span className={styles.ruleLabel}>Initial stop</span>
                <span className={styles.ruleVal}>
                  max(entry &minus; 2×ATR(14), entry × {(1 - activeSys.initial_hard_stop_pct).toFixed(2)}) &mdash;
                  closer to entry of the two. Caps day-1 risk at {(activeSys.initial_hard_stop_pct * 100).toFixed(0)}%.
                </span>
                <span className={styles.ruleLabel}>Target</span>
                <span className={styles.ruleVal}>
                  entry × {(1 + activeSys.target_pct).toFixed(2)}
                  &mdash; fixed +{(activeSys.target_pct * 100).toFixed(0)}%, no trailing.
                </span>
                <span className={styles.ruleLabel}>Time stop</span>
                <span className={styles.ruleVal}>60-day max hold safety. Exit at close on day 60.</span>
                <span className={styles.ruleLabel}>Sizing</span>
                <span className={styles.ruleVal}>
                  Risk {(activeSys.risk_per_trade_pct * 100).toFixed(1)}% per trade.
                  Qty = floor((equity × risk_pct) / (entry &minus; stop)).
                  Capped at notional {formatRs(activeSys.capital / activeSys.max_concurrent)} per position.
                </span>
                <span className={styles.ruleLabel}>Slot allocation</span>
                <span className={styles.ruleVal}>
                  Max {activeSys.max_concurrent} concurrent positions.
                  Tie-break by volume rank (highest vol/avg ratio fills first).
                </span>
                <span className={styles.ruleLabel}>Daily lifecycle</span>
                <span className={styles.ruleVal}>
                  09:20 IST: fill PENDING signals at today's open ·
                  16:00 IST: scan today's bars for new breakouts ·
                  16:05 IST: check open positions for target/stop hit
                </span>
              </div>
            </div>
          </section>

          {/* Backtest summary */}
          <section className={styles.section}>
            <div className={styles.sectionHead}>
              <div className="section-title">Recorded backtest — walk-forward</div>
              <Chip>{bt.variant}</Chip>
            </div>
            <div className={styles.backtestTable}>
              <div className={`${styles.backtestRow} ${styles.backtestHead}`}>
                <span>Phase</span>
                <span style={{ textAlign: 'right' }}>Trades</span>
                <span style={{ textAlign: 'right' }}>PF</span>
                <span style={{ textAlign: 'right' }}>Sharpe</span>
                <span style={{ textAlign: 'right' }}>MaxDD</span>
                <span style={{ textAlign: 'right' }}>CAGR</span>
                <span style={{ textAlign: 'right' }}>Verdict</span>
                <span></span>
              </div>
              {bt.is_pf > 0 ? (
                <>
                  <div className={styles.backtestRow}>
                    <span>In-sample (2018–2022)</span>
                    <span style={{ textAlign: 'right' }} className={styles.muted}>—</span>
                    <span style={{ textAlign: 'right' }}>{bt.is_pf.toFixed(2)}</span>
                    <span style={{ textAlign: 'right' }}>{bt.is_sharpe.toFixed(2)}</span>
                    <span style={{ textAlign: 'right' }}>{bt.is_maxdd.toFixed(1)}%</span>
                    <span style={{ textAlign: 'right' }}>+{bt.is_cagr.toFixed(1)}%</span>
                    <span></span>
                    <span></span>
                  </div>
                  <div className={styles.backtestRow}>
                    <span>OOS (2023–2025)</span>
                    <span style={{ textAlign: 'right' }}>{bt.oos_trades}</span>
                    <span style={{ textAlign: 'right' }}>{bt.oos_pf.toFixed(2)}</span>
                    <span style={{ textAlign: 'right' }}>{bt.oos_sharpe.toFixed(2)}</span>
                    <span style={{ textAlign: 'right' }}>{bt.oos_maxdd.toFixed(1)}%</span>
                    <span style={{ textAlign: 'right' }}>+{bt.oos_cagr.toFixed(1)}%</span>
                    <span style={{ textAlign: 'right' }} className={bt.pass ? styles.backtestPass : styles.muted}>
                      {bt.pass ? 'PASS' : 'pending'}
                    </span>
                    <span></span>
                  </div>
                </>
              ) : (
                <div className={styles.empty}>Backtest pending — agent generating F&O sub-system results.</div>
              )}
            </div>
          </section>

          {/* Open positions */}
          <section className={styles.section}>
            <div className={styles.sectionHead}>
              <div className="section-title">Open positions</div>
              <Chip>{positions.length} open</Chip>
            </div>
            <DataTable
              columns={positionCols}
              rows={positions}
              emptyText="No open positions yet. First fills land tomorrow morning at 09:20 IST."
              rowKey={(p) => p.id}
            />
          </section>

          {/* Recent signals */}
          <section className={styles.section}>
            <div className={styles.sectionHead}>
              <div className="section-title">Recent signals</div>
              <Chip>last 30</Chip>
            </div>
            <DataTable
              columns={signalCols}
              rows={signals.slice(0, 30)}
              emptyText="No signals yet. Daily scan runs at 16:00 IST."
              rowKey={(s) => s.id}
            />
          </section>

          {/* Trade history */}
          <section className={styles.section}>
            <div className={styles.sectionHead}>
              <div className="section-title">Trade history</div>
              <Chip>{trades.length} trades</Chip>
            </div>
            <DataTable
              columns={tradeCols}
              rows={trades.slice(0, 50)}
              emptyText="No closed trades yet."
              rowKey={(t) => t.id}
            />
          </section>
        </>
      ) : (
        <div className={styles.empty}>Loading state…</div>
      )}
    </div>
  );
}
