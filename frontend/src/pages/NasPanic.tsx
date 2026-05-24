import { useCallback, useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import styles from './NasPanic.module.css';
import { apiGet, apiPost } from '../api/client';

interface VariantState {
  name: string;
  enabled: boolean;
  paper: boolean;
}

interface KillInfo {
  killed_at: string;
  reason: string;
  source: string;
}

interface PanicStatus {
  killed: boolean;
  kill_info: KillInfo | null;
  open_positions_total: number;
  variants: VariantState[];
}

interface PanicResult {
  status: string;
  kill_flag?: KillInfo;
  disabled?: string[];
  closed_positions_total?: number;
  closed_per_variant?: Record<string, number>;
  errors_per_variant?: Record<string, string>;
}

interface ResumeResult {
  status: string;
  had_kill_flag: boolean;
  still_killed: boolean;
  enabled: string[];
}

type ToastKind = 'ok' | 'err' | 'info';
interface Toast { kind: ToastKind; text: string; }

export default function NasPanic() {
  const [status, setStatus] = useState<PanicStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<'kill' | 'resume' | null>(null);
  const [toast, setToast] = useState<Toast | null>(null);

  const refresh = useCallback(async () => {
    try {
      const d = await apiGet<PanicStatus>('/api/nas/panic-status');
      setStatus(d);
      setLoading(false);
    } catch (e) {
      setToast({ kind: 'err', text: `Status fetch failed: ${(e as Error).message}` });
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 8000);
    return () => clearInterval(t);
  }, [refresh]);

  const doKill = async () => {
    if (!window.confirm(
      'PANIC: stop all NAS trading now?\n\n'
      + 'This closes any open positions on Kite and disables all 8 variants. '
      + 'The kill survives a Flask/VPS restart.'
    )) return;
    setBusy('kill');
    setToast({ kind: 'info', text: 'Sending kill…' });
    try {
      const r = await apiPost<PanicResult>('/api/nas/panic', {
        confirm: 'YES',
        source: 'nas-panic-react',
        reason: 'manual panic from /app/nas-panic',
      });
      const closed = r.closed_positions_total ?? 0;
      const errCount = Object.keys(r.errors_per_variant || {}).length;
      let msg = `KILLED. Closed ${closed} position${closed === 1 ? '' : 's'}.`;
      if (errCount > 0) {
        msg += ` ${errCount} variant${errCount === 1 ? '' : 's'} reported errors — check journal.`;
      }
      setToast({ kind: 'ok', text: msg });
    } catch (e) {
      setToast({ kind: 'err', text: `Kill failed: ${(e as Error).message}` });
    }
    setBusy(null);
    refresh();
  };

  const doResume = async () => {
    if (!window.confirm(
      'Resume NAS trading?\n\n'
      + 'Clears the kill flag and re-enables all 8 variants. '
      + 'Mode (paper vs live) is unchanged.'
    )) return;
    setBusy('resume');
    setToast({ kind: 'info', text: 'Resuming…' });
    try {
      const r = await apiPost<ResumeResult>('/api/nas/resume', { confirm: 'RESUME' });
      setToast({ kind: 'ok', text: `Resumed. ${r.enabled.length} variants enabled.` });
    } catch (e) {
      setToast({ kind: 'err', text: `Resume failed: ${(e as Error).message}` });
    }
    setBusy(null);
    refresh();
  };

  const killed = !!status?.killed;
  const openN = status?.open_positions_total || 0;

  return (
    <div className={styles.root}>
      <div className={styles.titleRow}>
        <div>
          <div className="page-title">NAS panic</div>
          <div className="page-subtitle">
            One-tap shutdown for all 8 NAS systems. Closes any open positions on Kite and persists the kill through Flask/VPS restart.
          </div>
        </div>
        <Link to="/nas" className={styles.backLink}>← Back to NAS</Link>
      </div>

      {loading ? (
        <div className={styles.loading}>Loading…</div>
      ) : (
        <>
          <div className={styles.stateCard} data-state={killed ? 'killed' : 'running'}>
            <div className={styles.stateLabel}>Status</div>
            <div className={styles.stateVal}>
              {killed ? (
                <>
                  <span className={styles.stateKilled}>KILLED — no new entries</span>
                  {openN > 0 ? (
                    <span className={styles.stateOpen}> &nbsp;+ {openN} open</span>
                  ) : null}
                </>
              ) : (
                <>
                  <span className={styles.stateRunning}>RUNNING</span>
                  {openN > 0 ? (
                    <span className={styles.stateOpenMuted}>
                      {' '}· {openN} open position{openN === 1 ? '' : 's'}
                    </span>
                  ) : null}
                </>
              )}
            </div>
            {killed && status?.kill_info ? (
              <div className={styles.killInfo}>
                killed at {status.kill_info.killed_at} — {status.kill_info.reason}
              </div>
            ) : null}
          </div>

          <div className={styles.variantsCard}>
            <div className={styles.cardLabel}>Variants</div>
            <table className={styles.variantsTable}>
              <tbody>
                {status?.variants.map((v) => (
                  <tr key={v.name}>
                    <td>{v.name}</td>
                    <td className={styles.variantPillCell}>
                      {!v.enabled ? (
                        <span className={`${styles.pill} ${styles.pillOff}`}>OFF</span>
                      ) : v.paper ? (
                        <span className={`${styles.pill} ${styles.pillPaper}`}>PAPER</span>
                      ) : (
                        <span className={`${styles.pill} ${styles.pillLive}`}>LIVE</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <button
            type="button"
            className={styles.killBtn}
            onClick={doKill}
            disabled={busy !== null || killed}
          >
            {killed ? 'ALREADY KILLED' : 'PANIC — STOP ALL NAS NOW'}
          </button>

          {killed ? (
            <button
              type="button"
              className={styles.resumeBtn}
              onClick={doResume}
              disabled={busy !== null}
            >
              Resume NAS trading
            </button>
          ) : null}

          {toast ? (
            <div className={`${styles.toast} ${styles[`toast_${toast.kind}`]}`}>
              {toast.text}
            </div>
          ) : null}

          <details className={styles.howCard}>
            <summary>How this works</summary>
            <ul>
              <li>Panic button: writes a kill flag, disables all 8 variants in-process, closes open Kite positions.</li>
              <li>Kill flag survives a Flask/VPS restart — even a reboot will not silently resume trading.</li>
              <li>Resume requires the second button: clears the flag and sets enabled=True on all 8.</li>
              <li>Mode (paper vs live) is not changed — only on/off and open positions.</li>
              <li>Bookmark this page on your phone home screen for one-tap access.</li>
            </ul>
          </details>
        </>
      )}
    </div>
  );
}
