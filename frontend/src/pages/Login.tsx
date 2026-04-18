import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { apiGet, apiPost } from '../api/client';
import type { AuthStatus } from '../api/types';
import styles from './Login.module.css';
import StatusDot from '../components/StatusDot/StatusDot';

interface LoginResponse {
  status: 'success' | 'error';
  message: string;
}

export default function Login() {
  const navigate = useNavigate();
  const [status, setStatus] = useState<AuthStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const r = await apiGet<AuthStatus>('/api/auth/status');
        if (cancelled) return;
        setStatus(r);
        if (r.authenticated) {
          navigate('/strategies', { replace: true });
          return;
        }
        // Not authenticated — auto-trigger TOTP login
        await handleLogin();
      } catch {
        if (!cancelled) setStatus({ authenticated: false });
      }
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function handleLogin() {
    setBusy(true);
    setError(null);
    try {
      const res = await apiPost<LoginResponse>('/api/auth/auto-login');
      if (res.status === 'success') {
        navigate('/strategies', { replace: true });
      } else {
        setError(res.message || 'Login failed');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed');
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className={styles.root}>
      <div className={styles.card}>
        <div className={styles.brand}>
          <div className={styles.brandIcon}>Q</div>
          <div className={styles.brandName}>Quantifyd</div>
        </div>

        <div className={styles.title}>Sign in</div>
        <div className={styles.sub}>
          Authenticate with Zerodha to access live strategies. Login uses TOTP — no
          browser redirect needed.
        </div>

        <div className={styles.statusRow}>
          <StatusDot
            kind={status?.authenticated ? 'connected' : 'disconnected'}
            label={status?.authenticated ? `Connected as ${status.user_name || 'user'}` : 'Not connected'}
          />
        </div>

        <button className={styles.btn} onClick={handleLogin} disabled={busy}>
          {busy ? 'Signing in…' : 'Retry login'}
        </button>

        {error ? <div className={styles.error}>{error}</div> : null}

        <div className={styles.foot}>
          Uses Kite credentials from the server environment. No password is stored in
          the browser.
        </div>
      </div>
    </div>
  );
}
