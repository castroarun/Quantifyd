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

// Backtest summary metrics — sourced from walk-forward CSVs of research/17, 19, 21
type Phase = {
  trades: number; wins: number; losses: number;
  win_rate: number; avg_win: number; avg_loss: number;
  avg_days_held: number; profit_factor: number;
  net_pnl: number; cagr: number; sharpe: number;
  max_dd: number; calmar: number; final_equity: number;
};
type BacktestMetrics = {
  variant: string;
  research: string;
  period_is: string; period_oos: string;
  is_years: number; oos_years: number;
  is: Phase; oos: Phase;
  pass: boolean;
};

const BACKTEST_RESULTS: Record<SystemId, BacktestMetrics> = {
  nifty500: {
    variant: 'D1_fixed_25_8',
    research: 'research/17',
    period_is: '2018-01 to 2022-12', period_oos: '2023-01 to 2025-12',
    is_years: 5, oos_years: 3,
    is: { trades: 388, wins: 119, losses: 269, win_rate: 30.67, avg_win: 42864, avg_loss: -12521,
          avg_days_held: 54.7, profit_factor: 1.51, net_pnl: 1732705, cagr: 13.36, sharpe: 0.82,
          max_dd: 35.94, calmar: 0.37, final_equity: 1870784 },
    oos: { trades: 208, wins: 67, losses: 141, win_rate: 32.21, avg_win: 27790, avg_loss: -9150,
          avg_days_held: 49.3, profit_factor: 1.44, net_pnl: 571802, cagr: 14.54, sharpe: 0.95,
          max_dd: 23.98, calmar: 0.61, final_equity: 1501770 },
    pass: true,
  },
  smallcap: {
    variant: 'vol_3x',
    research: 'research/19',
    period_is: '2018-01 to 2022-12', period_oos: '2023-01 to 2025-12',
    is_years: 5, oos_years: 3,
    is: { trades: 375, wins: 125, losses: 250, win_rate: 33.33, avg_win: 37809, avg_loss: -9791,
          avg_days_held: 29.8, profit_factor: 1.93, net_pnl: 2278318, cagr: 10.72, sharpe: 0.79,
          max_dd: 17.46, calmar: 0.61, final_equity: 1662527 },
    oos: { trades: 305, wins: 116, losses: 189, win_rate: 38.03, avg_win: 28487, avg_loss: -12010,
          avg_days_held: 23.6, profit_factor: 1.46, net_pnl: 1034707, cagr: 26.79, sharpe: 1.43,
          max_dd: 12.85, calmar: 2.08, final_equity: 2036000 },
    pass: true,
  },
  fno: {
    variant: 'vol_3x',
    research: 'research/21',
    period_is: '2018-01 to 2022-12', period_oos: '2023-01 to 2025-12',
    is_years: 5, oos_years: 3,
    is: { trades: 187, wins: 81, losses: 106, win_rate: 43.32, avg_win: 21654, avg_loss: -8032,
          avg_days_held: 43.9, profit_factor: 2.06, net_pnl: 902558, cagr: 11.22, sharpe: 1.18,
          max_dd: 20.03, calmar: 0.56, final_equity: 1701002 },
    oos: { trades: 134, wins: 59, losses: 75, win_rate: 44.03, avg_win: 12841, avg_loss: -5285,
          avg_days_held: 34.5, profit_factor: 1.91, net_pnl: 361280, cagr: 10.25, sharpe: 1.04,
          max_dd: 9.11, calmar: 1.13, final_equity: 1339413 },
    pass: true,
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

      {/* Cross-system summary — always visible regardless of active tab */}
      <section className={styles.section} style={{ marginTop: 20 }}>
        <div className={styles.sectionHead}>
          <div className="section-title">Cross-system OOS comparison</div>
          <Chip>walk-forward 2023-2025 (3 yr OOS each)</Chip>
        </div>
        <div className={styles.backtestTable}>
          <div className={`${styles.backtestRow} ${styles.backtestHead}`}
               style={{ gridTemplateColumns: '1.6fr repeat(9, 1fr)' }}>
            <span>System</span>
            <span style={{ textAlign: 'right' }}>Universe</span>
            <span style={{ textAlign: 'right' }}>Trades</span>
            <span style={{ textAlign: 'right' }}>Tr/yr</span>
            <span style={{ textAlign: 'right' }}>WR%</span>
            <span style={{ textAlign: 'right' }}>PF</span>
            <span style={{ textAlign: 'right' }}>Sharpe</span>
            <span style={{ textAlign: 'right' }}>MaxDD</span>
            <span style={{ textAlign: 'right' }}>Calmar</span>
            <span style={{ textAlign: 'right' }}>CAGR</span>
          </div>
          {(['nifty500', 'smallcap', 'fno'] as SystemId[]).map((sid) => {
            const bt = BACKTEST_RESULTS[sid];
            const sys = systems.find((s) => s.system_id === sid);
            const trPerYr = bt.oos.trades / bt.oos_years;
            return (
              <div key={sid} className={styles.backtestRow}
                   style={{ gridTemplateColumns: '1.6fr repeat(9, 1fr)' }}>
                <span><strong>{SYSTEM_LABELS[sid]}</strong></span>
                <span style={{ textAlign: 'right' }} className={styles.muted}>
                  {sys?.universe_size ?? '—'} stocks
                </span>
                <span style={{ textAlign: 'right' }}>{bt.oos.trades}</span>
                <span style={{ textAlign: 'right' }}>{trPerYr.toFixed(0)}</span>
                <span style={{ textAlign: 'right' }}>{bt.oos.win_rate.toFixed(1)}</span>
                <span style={{ textAlign: 'right' }}>{bt.oos.profit_factor.toFixed(2)}</span>
                <span style={{ textAlign: 'right' }}>{bt.oos.sharpe.toFixed(2)}</span>
                <span style={{ textAlign: 'right' }}>{bt.oos.max_dd.toFixed(1)}%</span>
                <span style={{ textAlign: 'right' }}>{bt.oos.calmar.toFixed(2)}</span>
                <span style={{ textAlign: 'right' }} className={styles.pos}>
                  +{bt.oos.cagr.toFixed(1)}%
                </span>
              </div>
            );
          })}
        </div>
        <div style={{ fontSize: 'var(--text-xs)', color: 'var(--ink-muted)', marginTop: 8 }}>
          All three systems use IDENTICAL rules — only universe + cost assumption differ.
          Signal logic is shared, so do NOT stack them as parallel strategies. F&O sub-system
          is the options-overlay candidate (covered calls 25% OTM matches the +25% target).
        </div>
      </section>

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

          {/* Backtest summary — full walk-forward metrics */}
          <section className={styles.section}>
            <div className={styles.sectionHead}>
              <div className="section-title">Recorded backtest — walk-forward</div>
              <Chip>variant: {bt.variant} · {bt.research}</Chip>
            </div>

            {/* Headline metrics row */}
            <div className={styles.metricsRow}>
              <div className={styles.metric}>
                <div className={styles.metricLabel}>OOS verdict</div>
                <div className={`${styles.metricValue} ${bt.pass ? styles.pos : styles.muted}`}>
                  {bt.pass ? 'PASS (3 of 3 gates)' : 'pending'}
                </div>
              </div>
              <div className={styles.metric}>
                <div className={styles.metricLabel}>OOS PF</div>
                <div className={styles.metricValue}>{bt.oos.profit_factor.toFixed(2)}</div>
              </div>
              <div className={styles.metric}>
                <div className={styles.metricLabel}>OOS Sharpe</div>
                <div className={styles.metricValue}>{bt.oos.sharpe.toFixed(2)}</div>
              </div>
              <div className={styles.metric}>
                <div className={styles.metricLabel}>OOS CAGR</div>
                <div className={`${styles.metricValue} ${styles.pos}`}>+{bt.oos.cagr.toFixed(2)}%</div>
              </div>
              <div className={styles.metric}>
                <div className={styles.metricLabel}>OOS MaxDD</div>
                <div className={styles.metricValue}>{bt.oos.max_dd.toFixed(2)}%</div>
              </div>
              <div className={styles.metric}>
                <div className={styles.metricLabel}>OOS Calmar</div>
                <div className={styles.metricValue}>{bt.oos.calmar.toFixed(2)}</div>
              </div>
            </div>

            {/* Detailed phase comparison */}
            <div className={styles.backtestTable} style={{ marginTop: 12 }}>
              <div className={`${styles.backtestRow} ${styles.backtestHead}`} style={{ gridTemplateColumns: '1.4fr repeat(13, 1fr)' }}>
                <span>Phase</span>
                <span style={{ textAlign: 'right' }}>Trades</span>
                <span style={{ textAlign: 'right' }}>Tr/yr</span>
                <span style={{ textAlign: 'right' }}>WR%</span>
                <span style={{ textAlign: 'right' }}>W/L</span>
                <span style={{ textAlign: 'right' }}>Avg Win</span>
                <span style={{ textAlign: 'right' }}>Avg Loss</span>
                <span style={{ textAlign: 'right' }}>Days held</span>
                <span style={{ textAlign: 'right' }}>PF</span>
                <span style={{ textAlign: 'right' }}>Sharpe</span>
                <span style={{ textAlign: 'right' }}>MaxDD</span>
                <span style={{ textAlign: 'right' }}>Calmar</span>
                <span style={{ textAlign: 'right' }}>CAGR</span>
                <span style={{ textAlign: 'right' }}>Net P&L</span>
              </div>
              {(['is', 'oos'] as const).map((phase) => {
                const m = bt[phase];
                const period = phase === 'is' ? bt.period_is : bt.period_oos;
                const years = phase === 'is' ? bt.is_years : bt.oos_years;
                const trPerYr = m.trades / years;
                const label = phase === 'is' ? `In-sample (${period})` : `OOS (${period})`;
                return (
                  <div key={phase} className={styles.backtestRow}
                    style={{ gridTemplateColumns: '1.4fr repeat(13, 1fr)' }}>
                    <span>{label}</span>
                    <span style={{ textAlign: 'right' }}>{m.trades}</span>
                    <span style={{ textAlign: 'right' }}>{trPerYr.toFixed(0)}</span>
                    <span style={{ textAlign: 'right' }}>{m.win_rate.toFixed(1)}</span>
                    <span style={{ textAlign: 'right' }} className={styles.muted}>{m.wins}/{m.losses}</span>
                    <span style={{ textAlign: 'right' }} className={styles.pos}>+{(m.avg_win/1000).toFixed(1)}K</span>
                    <span style={{ textAlign: 'right' }} className={styles.neg}>{(m.avg_loss/1000).toFixed(1)}K</span>
                    <span style={{ textAlign: 'right' }}>{m.avg_days_held.toFixed(1)}d</span>
                    <span style={{ textAlign: 'right' }}>{m.profit_factor.toFixed(2)}</span>
                    <span style={{ textAlign: 'right' }}>{m.sharpe.toFixed(2)}</span>
                    <span style={{ textAlign: 'right' }}>{m.max_dd.toFixed(1)}%</span>
                    <span style={{ textAlign: 'right' }}>{m.calmar.toFixed(2)}</span>
                    <span style={{ textAlign: 'right' }} className={styles.pos}>+{m.cagr.toFixed(1)}%</span>
                    <span style={{ textAlign: 'right' }} className={pnlClass(m.net_pnl)}>
                      {formatRs(m.net_pnl)}
                    </span>
                  </div>
                );
              })}
            </div>
            <div style={{ fontSize: 'var(--text-xs)', color: 'var(--ink-muted)', marginTop: 8 }}>
              Capital base: Rs {(activeSys.capital / 100000).toFixed(0)}L. End-of-OOS equity:
              {' '}<span className={styles.pos}>{formatRs(bt.oos.final_equity)}</span>
              {' '}({((bt.oos.final_equity / activeSys.capital - 1) * 100).toFixed(1)}% return).
              Win rate {bt.oos.win_rate.toFixed(1)}% with avg-win/avg-loss ratio of
              {' '}<strong>{(Math.abs(bt.oos.avg_win / bt.oos.avg_loss)).toFixed(2)}×</strong>
              {' '}— losers are cut small, winners ride the {(activeSys.target_pct * 100).toFixed(0)}% target.
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
