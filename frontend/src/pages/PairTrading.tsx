import { useEffect, useState, useCallback } from 'react';
import styles from './PairTrading.module.css';
import { apiGet, apiPost } from '../api/client';
import { formatPnl, formatRs, pnlClass } from '../utils/format';

type Mode = 'off' | 'paper' | 'live';

interface PairConfig {
  name: string;
  symA: string;
  symB: string;
  entry_z: number;
  stop_z: number;
  hold_days: number;
  lookback: number;
  alpha?: number;
  beta?: number;
}

interface PairLive {
  name: string;
  symA: string;
  symB: string;
  current_z?: number | null;
  is_open?: boolean;
  direction?: 1 | -1 | null;
  entry_z?: number;
  stop_z?: number;
  hold_days?: number;
  pnl_inr?: number;
  days_in_trade?: number;
  last_scan_action?: string;
}

interface PairState {
  enabled: boolean;
  paper_trading_mode: boolean;
  live_trading_enabled: boolean;
  capital?: number;
  today_pnl?: number;
  realized_pnl_today?: number;
  unrealized_pnl?: number;
  open_pairs?: number;
  max_concurrent?: number;
  last_scan_ts?: string;
  pairs?: PairLive[];
}

interface Position {
  pair_name: string;
  direction: 1 | -1;
  symA: string;
  symB: string;
  qtyA: number;
  qtyB: number;
  entry_priceA: number;
  entry_priceB: number;
  entry_date?: string;
  entry_z?: number;
  current_z?: number;
  pnl_inr?: number;
  status?: string;
  paper_mode?: boolean;
  days_in_trade?: number;
}

interface Signal {
  ts: string;
  pair_name: string;
  z?: number;
  action: string;
  direction?: 1 | -1;
  block_reason?: string | null;
  paper_mode?: boolean;
}

function modeOf(s: PairState | null): Mode {
  if (!s || !s.enabled) return 'off';
  if (s.paper_trading_mode) return 'paper';
  if (s.live_trading_enabled) return 'live';
  return 'paper';
}

/** Map z in [-5, 5] to 0..100% along the gauge, clamped. */
function zPct(z: number | null | undefined): number | null {
  if (z == null || !Number.isFinite(z)) return null;
  const clamped = Math.max(-5, Math.min(5, z));
  return ((clamped + 5) / 10) * 100;
}

function ZGauge({ z, isOpen, entryZ }: { z: number | null | undefined; isOpen: boolean; entryZ: number }) {
  const pct = zPct(z);
  const entryLeft = zPct(-entryZ);
  const entryRight = zPct(+entryZ);
  return (
    <div>
      <div className={styles.zGauge}>
        <div className={styles.zGaugeMid} />
        {entryLeft != null && (
          <div className={styles.zGaugeEntry} style={{ left: `${entryLeft}%` }} title={`Entry threshold -${entryZ}`} />
        )}
        {entryRight != null && (
          <div className={styles.zGaugeEntry} style={{ left: `${entryRight}%` }} title={`Entry threshold +${entryZ}`} />
        )}
        {pct != null && (
          <div
            className={`${styles.zGaugeMarker} ${isOpen ? styles.zGaugeMarkerOpen : ''}`}
            style={{ left: `${pct}%` }}
            title={`z = ${z?.toFixed(2)}`}
          />
        )}
      </div>
      <div className={styles.zGaugeLabels}>
        <span>-5</span>
        <span>-2</span>
        <span>0</span>
        <span>+2</span>
        <span>+5</span>
      </div>
    </div>
  );
}

function PairCard({ p }: { p: PairLive }) {
  const isOpen = !!p.is_open;
  const z = p.current_z;
  const entryZ = p.entry_z ?? 2.0;
  return (
    <div className={`${styles.pairCard} ${isOpen ? styles.pairCardOpen : ''}`}>
      <div className={styles.pairHead}>
        <div>
          <div className={styles.pairName}>{p.name}</div>
          <div className={styles.pairSub}>
            {p.symA} / {p.symB}
          </div>
        </div>
        <span className={`${styles.statusChip} ${isOpen ? styles.statusChipOpen : styles.statusChipFlat}`}>
          {isOpen ? `OPEN ${p.direction === 1 ? 'LONG' : 'SHORT'}` : 'flat'}
        </span>
      </div>

      <ZGauge z={z} isOpen={isOpen} entryZ={entryZ} />

      <div className={styles.pairMetrics}>
        <div>
          <div className={styles.pairMetricLabel}>z-score</div>
          <div className={styles.pairMetricValue}>{z != null ? z.toFixed(2) : '—'}</div>
        </div>
        <div>
          <div className={styles.pairMetricLabel}>Entry / Stop / Hold</div>
          <div className={styles.pairMetricValue}>
            {entryZ.toFixed(1)} / {(p.stop_z ?? 0).toFixed(1)} / {p.hold_days ?? '—'}d
          </div>
        </div>
        <div>
          <div className={styles.pairMetricLabel}>{isOpen ? 'P&L · day' : 'Last action'}</div>
          <div className={`${styles.pairMetricValue} ${pnlClass(p.pnl_inr)}`}>
            {isOpen
              ? p.pnl_inr != null
                ? formatPnl(p.pnl_inr)
                : '—'
              : p.last_scan_action || '—'}
          </div>
        </div>
      </div>
    </div>
  );
}

export default function PairTrading() {
  const [state, setState] = useState<PairState | null>(null);
  const [positions, setPositions] = useState<Position[]>([]);
  const [signals, setSignals] = useState<Signal[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const [s, pos, sig] = await Promise.all([
        apiGet<PairState>('/api/pair_trading/state').catch(() => null),
        apiGet<{ positions: Position[] }>('/api/pair_trading/positions').catch(() => ({ positions: [] })),
        apiGet<{ signals: Signal[] }>('/api/pair_trading/signals').catch(() => ({ signals: [] })),
      ]);
      if (s) setState(s);
      setPositions(pos.positions || []);
      setSignals(sig.signals || []);
      setErr(null);
    } catch (e) {
      setErr(String(e));
    }
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 12000);
    return () => clearInterval(t);
  }, [refresh]);

  const onSetMode = useCallback(
    async (mode: Mode) => {
      if (
        mode === 'live' &&
        !window.confirm(
          'Flip Pair Trading to LIVE? Real F&O futures orders will be placed on the next signal. Are you sure?',
        )
      ) {
        return;
      }
      setBusy(true);
      try {
        await apiPost('/api/pair_trading/toggle-mode', { mode });
        await refresh();
      } catch (e) {
        setErr(String(e));
      } finally {
        setBusy(false);
      }
    },
    [refresh],
  );

  const onKillSwitch = useCallback(async () => {
    if (!window.confirm('Square off ALL open pairs and halt for the day?')) return;
    setBusy(true);
    try {
      await apiPost('/api/pair_trading/kill-switch', {});
      await refresh();
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }, [refresh]);

  const onManualScan = useCallback(async () => {
    setBusy(true);
    try {
      await apiPost('/api/pair_trading/scan', {});
      await refresh();
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }, [refresh]);

  const mode = modeOf(state);

  // Default 6-pair cohort if backend hasn't responded yet
  const fallbackPairs: PairLive[] = [
    { name: 'HAVELLS-MARICO', symA: 'HAVELLS', symB: 'MARICO', entry_z: 2.0, stop_z: 4.0, hold_days: 20, is_open: false },
    { name: 'BAJFINANCE-KOTAKBANK', symA: 'BAJFINANCE', symB: 'KOTAKBANK', entry_z: 2.0, stop_z: 4.0, hold_days: 20, is_open: false },
    { name: 'DABUR-HINDUNILVR', symA: 'DABUR', symB: 'HINDUNILVR', entry_z: 2.0, stop_z: 5.0, hold_days: 20, is_open: false },
    { name: 'COFORGE-HCLTECH', symA: 'COFORGE', symB: 'HCLTECH', entry_z: 2.0, stop_z: 4.0, hold_days: 15, is_open: false },
    { name: 'DABUR-TCS', symA: 'DABUR', symB: 'TCS', entry_z: 2.0, stop_z: 99.0, hold_days: 10, is_open: false },
    { name: 'APOLLOHOSP-COFORGE', symA: 'APOLLOHOSP', symB: 'COFORGE', entry_z: 2.0, stop_z: 5.0, hold_days: 10, is_open: false },
  ];
  const pairs: PairLive[] = state?.pairs && state.pairs.length > 0 ? state.pairs : fallbackPairs;

  return (
    <div className={styles.root}>
      <div className={styles.headerRow}>
        <div>
          <div className="page-title">Pair Trading</div>
          <div className="page-subtitle">
            Carry-forward F&amp;O · 6 cointegrated pairs · daily EOD scan @ 16:00 IST ·{' '}
            {!state?.enabled
              ? 'Disabled'
              : state?.live_trading_enabled
              ? 'Live trading'
              : 'Paper trading'}
          </div>
        </div>
        <div className={styles.actions}>
          <button
            className={`${styles.btn} ${mode === 'off' ? styles.btnActive : ''}`}
            disabled={busy}
            onClick={() => onSetMode('off')}
          >
            Off
          </button>
          <button
            className={`${styles.btn} ${mode === 'paper' ? styles.btnActive : ''}`}
            disabled={busy}
            onClick={() => onSetMode('paper')}
          >
            Paper
          </button>
          <button
            className={`${styles.btn} ${mode === 'live' ? styles.btnActiveLive : ''}`}
            disabled={busy}
            onClick={() => onSetMode('live')}
          >
            Live
          </button>
          <button className={styles.scanBtn} disabled={busy} onClick={onManualScan} title="Trigger scan now">
            Scan now
          </button>
          <button className={styles.killBtn} disabled={busy || !state?.enabled} onClick={onKillSwitch}>
            Kill
          </button>
        </div>
      </div>

      <div className={styles.bannerInfo}>
        <div className={styles.eyebrow}>Default state</div>
        <strong>Pair Trading defaults to PAPER MODE.</strong> Daily scan at 16:00 IST
        evaluates each of the 6 pairs and logs signals + synthetic fills only. No real
        F&amp;O futures orders until you flip to Live.{' '}
        <strong>Walk-forward stats: WR 78.7% · PF 3.57 · MaxDD 0.06% · n=108 trades</strong>{' '}
        on 22 months of held-out data.
      </div>

      {err ? <div className={styles.error}>{err}</div> : null}

      <div className={styles.metricsRow}>
        <div className={styles.metricCard}>
          <span className={styles.metricLabel}>Day P&amp;L</span>
          <span className={`${styles.metricValue} ${pnlClass(state?.today_pnl)}`}>
            {state?.today_pnl != null ? formatPnl(state.today_pnl) : '—'}
          </span>
          <span className={styles.metricHint}>
            Realized {state?.realized_pnl_today != null ? formatPnl(state.realized_pnl_today) : '—'} ·
            Unrealized {state?.unrealized_pnl != null ? formatPnl(state.unrealized_pnl) : '—'}
          </span>
        </div>
        <div className={styles.metricCard}>
          <span className={styles.metricLabel}>Open pairs</span>
          <span className={styles.metricValue}>
            {state?.open_pairs ?? 0} / {state?.max_concurrent ?? 5}
          </span>
          <span className={styles.metricHint}>Concurrency cap</span>
        </div>
        <div className={styles.metricCard}>
          <span className={styles.metricLabel}>Capital</span>
          <span className={styles.metricValue}>{formatRs(state?.capital ?? 1_000_000)}</span>
          <span className={styles.metricHint}>Rs.6,000 risk per pair-trade</span>
        </div>
        <div className={styles.metricCard}>
          <span className={styles.metricLabel}>Last scan</span>
          <span className={styles.metricValue}>{state?.last_scan_ts ?? '—'}</span>
          <span className={styles.metricHint}>Daily 16:00 IST · F&amp;O EOD</span>
        </div>
      </div>

      <h2 className={styles.sectionTitle}>The 6 Cointegrated Pairs</h2>
      <div className={styles.pairsGrid}>
        {pairs.map((p) => (
          <PairCard key={p.name} p={p} />
        ))}
      </div>

      <div className={styles.tableSection}>
        <h2 className={styles.sectionTitle}>Open Positions</h2>
        {positions.filter((p) => p.status !== 'CLOSED').length === 0 ? (
          <div className={styles.empty}>No open pair positions.</div>
        ) : (
          <table className={styles.tinyTable}>
            <thead>
              <tr>
                <th>Pair</th>
                <th>Direction</th>
                <th>Entry z</th>
                <th>Current z</th>
                <th>Days</th>
                <th>P&amp;L</th>
                <th>Mode</th>
              </tr>
            </thead>
            <tbody>
              {positions
                .filter((p) => p.status !== 'CLOSED')
                .map((p, i) => (
                  <tr key={`${p.pair_name}-${i}`}>
                    <td>{p.pair_name}</td>
                    <td>{p.direction === 1 ? 'LONG spread' : 'SHORT spread'}</td>
                    <td>{p.entry_z?.toFixed(2) ?? '—'}</td>
                    <td>{p.current_z?.toFixed(2) ?? '—'}</td>
                    <td>{p.days_in_trade ?? '—'}</td>
                    <td className={pnlClass(p.pnl_inr)}>
                      {p.pnl_inr != null ? formatPnl(p.pnl_inr) : '—'}
                    </td>
                    <td>{p.paper_mode ? 'paper' : 'live'}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        )}
      </div>

      <div className={styles.tableSection}>
        <h2 className={styles.sectionTitle}>Recent Signals</h2>
        {signals.length === 0 ? (
          <div className={styles.empty}>No signals logged yet.</div>
        ) : (
          <table className={styles.tinyTable}>
            <thead>
              <tr>
                <th>Time</th>
                <th>Pair</th>
                <th>z</th>
                <th>Action</th>
                <th>Reason</th>
              </tr>
            </thead>
            <tbody>
              {signals.slice(0, 12).map((s, i) => (
                <tr key={i}>
                  <td>{s.ts}</td>
                  <td>{s.pair_name}</td>
                  <td>{s.z?.toFixed(2) ?? '—'}</td>
                  <td>{s.action}</td>
                  <td className={styles.mute}>{s.block_reason ?? ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
