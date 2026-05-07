import { useEffect, useState } from 'react';
import { apiGet, apiPost } from '../api/client';
import styles from './N500m.module.css';

type Mode = 'OFF' | 'PAPER' | 'LIVE';

type StockConfig = {
  symbol: string;
  signal: 'ccrb' | 'volbo';
  timeframe: string;
  direction: 'long' | 'short';
  exit_policy: string;
  expected_sharpe: number;
  expected_wr: number;
  expected_n: number;
  promote: boolean;
};

type Position = {
  id: number;
  symbol: string;
  signal_type: string;
  trade_date: string;
  timeframe: string;
  direction: string;
  qty: number;
  entry_price: number;
  entry_time: string;
  sl_price?: number | null;
  target_price?: number | null;
  exit_price?: number | null;
  exit_time?: string | null;
  exit_reason?: string | null;
  pnl_pts?: number | null;
  pnl_inr?: number | null;
  status: string;
  mode: string;
  exit_policy: string;
  expected_sharpe?: number | null;
};

type Signal = {
  id: number;
  symbol: string;
  signal_type: string;
  trade_date: string;
  signal_time: string;
  timeframe: string;
  direction: string;
  entry_price: number;
  sl_price?: number | null;
  target_price?: number | null;
  vm_ratio?: number | null;
  action_taken: string;
};

type EquityRow = {
  trade_date: string;
  ending_nav: number;
  realized_pnl: number;
  unrealized_pnl: number;
  n_trades: number;
  n_wins: number;
};

type State = {
  mode: Mode;
  kill_switch: boolean;
  config_count: number;
  unique_symbols: string[];
  open_positions: Position[];
  today_positions: Position[];
  today_signals: Signal[];
  equity_curve: EquityRow[];
};

function fmtRs(v?: number | null, dp = 2): string {
  if (v == null || !Number.isFinite(v)) return '—';
  return v.toLocaleString('en-IN', { maximumFractionDigits: dp, minimumFractionDigits: dp });
}

function fmtPnl(v?: number | null): string {
  if (v == null || !Number.isFinite(v)) return '—';
  const s = v >= 0 ? '+' : '';
  return `${s}${Math.round(v).toLocaleString('en-IN')}`;
}

function fmtTime(ts?: string | null): string {
  if (!ts) return '—';
  return ts.length >= 19 ? ts.slice(11, 19) : ts;
}

export default function N500m() {
  const [state, setState] = useState<State | null>(null);
  const [configs, setConfigs] = useState<StockConfig[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [toast, setToast] = useState<string | null>(null);

  const refetch = async () => {
    try {
      const [s, c] = await Promise.all([
        apiGet<State>('/api/n500m/state'),
        apiGet<{ configs: StockConfig[] }>('/api/n500m/configs'),
      ]);
      setState(s);
      setConfigs(c.configs);
      setErr(null);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  };

  useEffect(() => {
    refetch();
    const t = window.setInterval(refetch, 15_000);
    return () => window.clearInterval(t);
  }, []);

  const onSetMode = async (mode: Mode) => {
    if (busy) return;
    if (mode === 'LIVE' && !window.confirm(
      'Switch N500 Momentum to LIVE — real cash MIS orders will fire on entries. Continue?')) return;
    setBusy(true);
    try {
      await apiPost('/api/n500m/toggle-mode', { mode });
      setToast(`Mode → ${mode}`);
      window.setTimeout(() => setToast(null), 2500);
      await refetch();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const onScanNow = async () => {
    if (busy) return;
    setBusy(true);
    try {
      const r = await apiPost<{ scanned: number; submitted: { symbol: string }[] }>(
        '/api/n500m/scan', {});
      setToast(`Scan: ${r.scanned} signal(s), ${r.submitted.length} submitted`);
      window.setTimeout(() => setToast(null), 3500);
      await refetch();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const onKillSwitch = async () => {
    if (busy) return;
    if (!window.confirm('Engage KILL SWITCH — flatten all open positions and lock mode OFF. Continue?')) return;
    setBusy(true);
    try {
      const r = await apiPost<{ flattened: number }>('/api/n500m/kill-switch', {});
      setToast(`Killed — ${r.flattened} flattened`);
      window.setTimeout(() => setToast(null), 3500);
      await refetch();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  if (!state) {
    return <div className={styles.loading}>Loading N500 Momentum...</div>;
  }

  const open = state.open_positions;
  const closed = state.today_positions.filter((p) => p.status === 'CLOSED');
  const realizedToday = closed.reduce((s, p) => s + (p.pnl_inr ?? 0), 0);
  const winsToday = closed.filter((p) => (p.pnl_inr ?? 0) > 0).length;

  return (
    <div className={styles.root}>
      <div className={styles.headerRow}>
        <div>
          <div className="page-title">Nifty 500 Intraday Momentum</div>
          <div className="page-subtitle">
            Per-stock CCRB + vol-BO portfolio · {state.unique_symbols.length} stocks ·{' '}
            {state.config_count} rules ·{' '}
            <span className={state.mode === 'LIVE' ? styles.tagLive : state.mode === 'PAPER' ? styles.tagPaper : styles.tagOff}>
              {state.mode}
            </span>
            {state.kill_switch && <span className={styles.kill}> · KILL SWITCH</span>}
          </div>
        </div>
        <div className={styles.actions}>
          <button
            className={`${styles.btn} ${state.mode === 'OFF' ? styles.btnActive : ''}`}
            disabled={busy}
            onClick={() => onSetMode('OFF')}
          >
            Off
          </button>
          <button
            className={`${styles.btn} ${state.mode === 'PAPER' ? styles.btnActive : ''}`}
            disabled={busy}
            onClick={() => onSetMode('PAPER')}
          >
            Paper
          </button>
          <button
            className={`${styles.btn} ${state.mode === 'LIVE' ? styles.btnActiveLive : ''}`}
            disabled={busy}
            onClick={() => onSetMode('LIVE')}
          >
            Live
          </button>
          <button className={styles.btn} disabled={busy} onClick={onScanNow}>
            Scan now
          </button>
          <button className={`${styles.btn} ${styles.btnDanger}`} disabled={busy} onClick={onKillSwitch}>
            Kill
          </button>
        </div>
      </div>

      {err && <div className={styles.error}>{err}</div>}
      {toast && <div className={styles.toast}>{toast}</div>}

      <div className={styles.metrics}>
        <div className={styles.metric}>
          <div className={styles.metricLabel}>Open positions</div>
          <div className={styles.metricValue}>{open.length}</div>
        </div>
        <div className={styles.metric}>
          <div className={styles.metricLabel}>Trades today</div>
          <div className={styles.metricValue}>{closed.length}</div>
        </div>
        <div className={styles.metric}>
          <div className={styles.metricLabel}>Wins today</div>
          <div className={styles.metricValue}>
            {winsToday}/{closed.length}
          </div>
        </div>
        <div className={styles.metric}>
          <div className={styles.metricLabel}>Realised P&amp;L (today)</div>
          <div className={`${styles.metricValue} ${realizedToday >= 0 ? styles.pos : styles.neg}`}>
            {fmtPnl(realizedToday)}
          </div>
        </div>
      </div>

      <section className={styles.section}>
        <div className={styles.sectionHead}>
          <div className={styles.sectionTitle}>Open positions ({open.length})</div>
        </div>
        {open.length === 0 ? (
          <div className={styles.empty}>No open positions.</div>
        ) : (
          <table className={styles.tbl}>
            <thead>
              <tr>
                <th>Symbol</th>
                <th>Signal</th>
                <th>TF</th>
                <th>Dir</th>
                <th className={styles.numr}>Qty</th>
                <th className={styles.numr}>Entry</th>
                <th className={styles.numr}>SL</th>
                <th className={styles.numr}>TGT</th>
                <th>Exit</th>
                <th>Mode</th>
              </tr>
            </thead>
            <tbody>
              {open.map((p) => (
                <tr key={p.id}>
                  <td>{p.symbol}</td>
                  <td>{p.signal_type}</td>
                  <td>{p.timeframe}</td>
                  <td>{p.direction}</td>
                  <td className={styles.numr}>{p.qty}</td>
                  <td className={styles.numr}>{fmtRs(p.entry_price)}</td>
                  <td className={styles.numr}>{p.sl_price ? fmtRs(p.sl_price) : '—'}</td>
                  <td className={styles.numr}>{p.target_price ? fmtRs(p.target_price) : '—'}</td>
                  <td>{p.exit_policy}</td>
                  <td>{p.mode}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <section className={styles.section}>
        <div className={styles.sectionHead}>
          <div className={styles.sectionTitle}>Today’s closed trades ({closed.length})</div>
        </div>
        {closed.length === 0 ? (
          <div className={styles.empty}>No trades closed today.</div>
        ) : (
          <table className={styles.tbl}>
            <thead>
              <tr>
                <th>Symbol</th>
                <th>Signal</th>
                <th>Dir</th>
                <th className={styles.numr}>Qty</th>
                <th className={styles.numr}>Entry</th>
                <th className={styles.numr}>Exit</th>
                <th>Reason</th>
                <th className={styles.numr}>P&amp;L</th>
                <th>Mode</th>
              </tr>
            </thead>
            <tbody>
              {closed.map((p) => (
                <tr key={p.id}>
                  <td>{p.symbol}</td>
                  <td>{p.signal_type}</td>
                  <td>{p.direction}</td>
                  <td className={styles.numr}>{p.qty}</td>
                  <td className={styles.numr}>{fmtRs(p.entry_price)}</td>
                  <td className={styles.numr}>{p.exit_price ? fmtRs(p.exit_price) : '—'}</td>
                  <td>{p.exit_reason}</td>
                  <td className={`${styles.numr} ${(p.pnl_inr ?? 0) >= 0 ? styles.pos : styles.neg}`}>
                    {fmtPnl(p.pnl_inr)}
                  </td>
                  <td>{p.mode}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <section className={styles.section}>
        <div className={styles.sectionHead}>
          <div className={styles.sectionTitle}>
            Today’s signals ({state.today_signals.length})
          </div>
        </div>
        {state.today_signals.length === 0 ? (
          <div className={styles.empty}>No signals fired today.</div>
        ) : (
          <table className={styles.tbl}>
            <thead>
              <tr>
                <th>Time</th>
                <th>Symbol</th>
                <th>Signal</th>
                <th>TF</th>
                <th>Dir</th>
                <th className={styles.numr}>Entry</th>
                <th className={styles.numr}>VM</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {state.today_signals.map((s) => (
                <tr key={s.id}>
                  <td>{fmtTime(s.signal_time)}</td>
                  <td>{s.symbol}</td>
                  <td>{s.signal_type}</td>
                  <td>{s.timeframe}</td>
                  <td>{s.direction}</td>
                  <td className={styles.numr}>{fmtRs(s.entry_price)}</td>
                  <td className={styles.numr}>{s.vm_ratio ? s.vm_ratio.toFixed(2) : '—'}</td>
                  <td>{s.action_taken}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </section>

      <section className={styles.section}>
        <div className={styles.sectionHead}>
          <div className={styles.sectionTitle}>STOCK_CONFIGS — top {configs?.length ?? 0} per-stock rules</div>
        </div>
        {configs && configs.length > 0 ? (
          <table className={styles.tbl}>
            <thead>
              <tr>
                <th>Symbol</th>
                <th>Signal</th>
                <th>TF</th>
                <th>Dir</th>
                <th>Exit</th>
                <th className={styles.numr}>Sharpe</th>
                <th className={styles.numr}>WR%</th>
                <th className={styles.numr}>n</th>
                <th>Promote</th>
              </tr>
            </thead>
            <tbody>
              {configs.map((c, i) => (
                <tr key={`${c.symbol}-${c.signal}-${i}`}>
                  <td>{c.symbol}</td>
                  <td>{c.signal}</td>
                  <td>{c.timeframe}</td>
                  <td>{c.direction}</td>
                  <td>{c.exit_policy}</td>
                  <td className={styles.numr}>{c.expected_sharpe.toFixed(3)}</td>
                  <td className={styles.numr}>{c.expected_wr.toFixed(1)}</td>
                  <td className={styles.numr}>{c.expected_n}</td>
                  <td>{c.promote ? <span className={styles.tagPromote}>YES</span> : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div className={styles.empty}>Loading configs…</div>
        )}
      </section>
    </div>
  );
}
