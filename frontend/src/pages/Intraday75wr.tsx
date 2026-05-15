import { useEffect, useState, useCallback } from 'react';
import styles from './Intraday75wr.module.css';
import { apiGet, apiPost } from '../api/client';
import { formatPnl, formatRs, pnlClass } from '../utils/format';

type Mode = 'off' | 'paper' | 'live';
type ConfigId = 'A' | 'B' | 'C';

interface ConfigState {
  config_id: ConfigId;
  name: string;
  enabled: boolean;
  paper_trading_mode: boolean;
  live_trading_enabled: boolean;
  capital?: number;
  today_pnl?: number;
  open_positions?: number;
  signals_today?: number;
  daily_loss_limit_rs?: number;
  exits_today?: number;
}

interface I75State {
  configs: ConfigState[];
  combined_open_positions?: number;
  max_concurrent?: number;
}

interface Position {
  config_id: ConfigId;
  sub_signal?: string;
  symbol: string;
  direction: 'LONG' | 'SHORT';
  qty: number;
  entry_price: number;
  ltp?: number;
  sl_price?: number;
  target_price?: number;
  pnl_pts?: number;
  pnl_inr?: number;
  paper_mode: boolean;
  entry_time?: string;
}

interface Signal {
  config_id: ConfigId;
  sub_signal?: string;
  symbol?: string;
  ts: string;
  action: string;
  direction?: 'LONG' | 'SHORT';
  paper_mode?: boolean;
  block_reason?: string | null;
}

const CONFIG_DESCRIPTIONS: Record<ConfigId, { title: string; sub: string }> = {
  A: {
    title: 'Config A — Original',
    sub: 'TP 0.5% / SL 1.5% · 78% WR · cost-fragile · ~895 trades/yr',
  },
  B: {
    title: 'Config B — Cost-Resilient',
    sub: 'TP 2.0% / SL 1.5% · 53% WR / RR 1.33 · ~830 trades/yr',
  },
  C: {
    title: 'Config C — Multi-Bar Bounce',
    sub: 'TP 1.5% / SL 1.0% · 60% WR / RR 1.5 · ~120 trades/yr',
  },
};

function modeOf(c: ConfigState): Mode {
  if (!c.enabled) return 'off';
  if (c.paper_trading_mode) return 'paper';
  if (c.live_trading_enabled) return 'live';
  return 'paper';
}

function ModeBadge({ mode }: { mode: Mode }) {
  const cls =
    mode === 'live'
      ? `${styles.modeBadge} ${styles.modeBadgeLive}`
      : mode === 'paper'
      ? `${styles.modeBadge} ${styles.modeBadgePaper}`
      : `${styles.modeBadge} ${styles.modeBadgeOff}`;
  return <span className={cls}>{mode}</span>;
}

function ConfigCard({
  cfg,
  positions,
  signals,
  busy,
  onSetMode,
  onKillSwitch,
}: {
  cfg: ConfigState;
  positions: Position[];
  signals: Signal[];
  busy: boolean;
  onSetMode: (id: ConfigId, mode: Mode) => void;
  onKillSwitch: (id: ConfigId) => void;
}) {
  const mode = modeOf(cfg);
  const meta = CONFIG_DESCRIPTIONS[cfg.config_id];
  const myPositions = positions.filter((p) => p.config_id === cfg.config_id);
  const mySignals = signals
    .filter((s) => s.config_id === cfg.config_id)
    .slice(0, 5);

  return (
    <div className={styles.configCard}>
      <div className={styles.configHead}>
        <div>
          <h3 className={styles.configTitle}>{meta.title}</h3>
          <div className={styles.configSub}>{meta.sub}</div>
        </div>
        <ModeBadge mode={mode} />
      </div>

      <div className={styles.toggleRow}>
        <button
          className={`${styles.btn} ${mode === 'off' ? styles.btnActive : ''}`}
          disabled={busy}
          onClick={() => onSetMode(cfg.config_id, 'off')}
        >
          Off
        </button>
        <button
          className={`${styles.btn} ${mode === 'paper' ? styles.btnActive : ''}`}
          disabled={busy}
          onClick={() => onSetMode(cfg.config_id, 'paper')}
        >
          Paper
        </button>
        <button
          className={`${styles.btn} ${mode === 'live' ? styles.btnActiveLive : ''}`}
          disabled={busy}
          onClick={() => onSetMode(cfg.config_id, 'live')}
        >
          Live
        </button>
      </div>

      <div className={styles.metricRow}>
        <div className={styles.metricCell}>
          <span className={styles.metricLabel}>Day P&amp;L</span>
          <span className={`${styles.metricValue} ${pnlClass(cfg.today_pnl)}`}>
            {cfg.today_pnl != null ? formatPnl(cfg.today_pnl) : '—'}
          </span>
        </div>
        <div className={styles.metricCell}>
          <span className={styles.metricLabel}>Open</span>
          <span className={styles.metricValue}>{cfg.open_positions ?? 0}</span>
        </div>
        <div className={styles.metricCell}>
          <span className={styles.metricLabel}>Signals</span>
          <span className={styles.metricValue}>{cfg.signals_today ?? 0}</span>
        </div>
      </div>

      <div>
        <div className={styles.metricLabel} style={{ marginBottom: 6 }}>
          Open positions
        </div>
        {myPositions.length === 0 ? (
          <div className={styles.empty}>No open positions.</div>
        ) : (
          <table className={styles.tinyTable}>
            <thead>
              <tr>
                <th>Sym</th>
                <th>Side</th>
                <th>Qty</th>
                <th>Entry</th>
                <th>LTP</th>
                <th>P&amp;L</th>
              </tr>
            </thead>
            <tbody>
              {myPositions.slice(0, 6).map((p, i) => (
                <tr key={`${p.symbol}-${i}`}>
                  <td>{p.symbol}</td>
                  <td>{p.direction}</td>
                  <td>{p.qty}</td>
                  <td>{p.entry_price?.toFixed(2)}</td>
                  <td>{p.ltp?.toFixed(2) ?? '—'}</td>
                  <td className={pnlClass(p.pnl_inr)}>
                    {p.pnl_inr != null ? formatPnl(p.pnl_inr) : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div>
        <div className={styles.metricLabel} style={{ marginBottom: 6 }}>
          Recent signals
        </div>
        {mySignals.length === 0 ? (
          <div className={styles.empty}>No signals today.</div>
        ) : (
          <table className={styles.tinyTable}>
            <thead>
              <tr>
                <th>Time</th>
                <th>Sym</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {mySignals.map((s, i) => (
                <tr key={i}>
                  <td>{s.ts?.split('T')[1]?.slice(0, 5) ?? s.ts}</td>
                  <td>{s.symbol ?? '—'}</td>
                  <td>
                    {s.action}
                    {s.block_reason ? ` (${s.block_reason})` : ''}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className={styles.killRow}>
        <span style={{ flex: 1 }}>
          Cap Rs.{(cfg.capital ?? 0).toLocaleString('en-IN')} · loss limit Rs.{(cfg.daily_loss_limit_rs ?? 9000).toLocaleString('en-IN')}
        </span>
        <button
          className={styles.killBtn}
          disabled={busy || !cfg.enabled}
          onClick={() => onKillSwitch(cfg.config_id)}
          title="Square off + halt for the day"
        >
          Kill
        </button>
      </div>
    </div>
  );
}

export default function Intraday75wr() {
  const [state, setState] = useState<I75State | null>(null);
  const [positions, setPositions] = useState<Position[]>([]);
  const [signals, setSignals] = useState<Signal[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const [s, posA, posB, posC, sigA, sigB, sigC] = await Promise.all([
        apiGet<I75State>('/api/intraday75wr/state').catch(() => null),
        apiGet<{ positions: Position[] }>('/api/intraday75wr/positions?config=A').catch(() => ({ positions: [] })),
        apiGet<{ positions: Position[] }>('/api/intraday75wr/positions?config=B').catch(() => ({ positions: [] })),
        apiGet<{ positions: Position[] }>('/api/intraday75wr/positions?config=C').catch(() => ({ positions: [] })),
        apiGet<{ signals: Signal[] }>('/api/intraday75wr/signals?config=A').catch(() => ({ signals: [] })),
        apiGet<{ signals: Signal[] }>('/api/intraday75wr/signals?config=B').catch(() => ({ signals: [] })),
        apiGet<{ signals: Signal[] }>('/api/intraday75wr/signals?config=C').catch(() => ({ signals: [] })),
      ]);
      if (s) setState(s);
      setPositions([
        ...(posA.positions || []),
        ...(posB.positions || []),
        ...(posC.positions || []),
      ]);
      setSignals([
        ...(sigA.signals || []),
        ...(sigB.signals || []),
        ...(sigC.signals || []),
      ]);
      setErr(null);
    } catch (e) {
      setErr(String(e));
    }
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 8000);
    return () => clearInterval(t);
  }, [refresh]);

  const onSetMode = useCallback(
    async (id: ConfigId, mode: Mode) => {
      if (
        mode === 'live' &&
        !window.confirm(
          `Flip Config ${id} to LIVE? Real Kite orders will be placed on the next signal. Are you sure?`,
        )
      ) {
        return;
      }
      setBusy(true);
      try {
        await apiPost('/api/intraday75wr/toggle-mode', { config: id, mode });
        await refresh();
      } catch (e) {
        setErr(String(e));
      } finally {
        setBusy(false);
      }
    },
    [refresh],
  );

  const onKillSwitch = useCallback(
    async (id: ConfigId) => {
      if (!window.confirm(`Square off Config ${id} and halt for the day?`)) return;
      setBusy(true);
      try {
        await apiPost('/api/intraday75wr/kill-switch', { config: id });
        await refresh();
      } catch (e) {
        setErr(String(e));
      } finally {
        setBusy(false);
      }
    },
    [refresh],
  );

  const configs: ConfigState[] =
    state?.configs && state.configs.length > 0
      ? state.configs
      : (['A', 'B', 'C'] as ConfigId[]).map((id) => ({
          config_id: id,
          name: CONFIG_DESCRIPTIONS[id].title,
          enabled: true,
          paper_trading_mode: true,
          live_trading_enabled: false,
          today_pnl: 0,
          open_positions: 0,
          signals_today: 0,
        }));

  return (
    <div className={styles.root}>
      <div className={styles.headerRow}>
        <div>
          <div className="page-title">Intraday 75WR</div>
          <div className="page-subtitle">
            3-config intraday system — Diamond Short / Long-MR / Long-TC / Multi-Bar Bounce ·{' '}
            {state
              ? `${state.combined_open_positions ?? 0}/${state.max_concurrent ?? 5} open`
              : 'loading…'}
          </div>
        </div>
      </div>

      <div className={styles.bannerInfo}>
        <div className={styles.eyebrow}>Default state</div>
        <strong>All 3 configs default to PAPER MODE.</strong> Signals fire and synthetic
        positions are recorded; no real Kite orders are placed. Flip any config to
        Live only after paper-trade validation. Belt-and-suspenders: live orders
        require both <code>paper_trading_mode=False</code> and{' '}
        <code>live_trading_enabled=True</code>.
      </div>

      {err ? <div className={styles.error}>{err}</div> : null}

      <div className={styles.configGrid}>
        {configs.map((cfg) => (
          <ConfigCard
            key={cfg.config_id}
            cfg={cfg}
            positions={positions}
            signals={signals}
            busy={busy}
            onSetMode={onSetMode}
            onKillSwitch={onKillSwitch}
          />
        ))}
      </div>
    </div>
  );
}
