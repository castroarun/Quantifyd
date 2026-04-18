import { Routes, Route, Navigate, useNavigate } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { apiGet } from './api/client';
import type { AuthStatus } from './api/types';
import AppLayout from './components/Layout/AppLayout';
import Login from './pages/Login';
import Strategies from './pages/Strategies';
import Orb from './pages/Orb';
import Nas from './pages/Nas';
import NotFound from './pages/NotFound';

type AuthState = 'unknown' | 'auth' | 'noauth';

function useAuthGate(): AuthState {
  const [state, setState] = useState<AuthState>('unknown');
  useEffect(() => {
    let cancelled = false;
    apiGet<AuthStatus>('/api/auth/status')
      .then((r) => {
        if (cancelled) return;
        setState(r.authenticated ? 'auth' : 'noauth');
      })
      .catch(() => {
        if (!cancelled) setState('noauth');
      });
    return () => {
      cancelled = true;
    };
  }, []);
  return state;
}

function Protected({ children }: { children: React.ReactNode }) {
  const auth = useAuthGate();
  const navigate = useNavigate();
  useEffect(() => {
    if (auth === 'noauth') navigate('/login', { replace: true });
  }, [auth, navigate]);
  if (auth !== 'auth') {
    return (
      <div style={{ padding: '48px', color: 'var(--ink-muted)', fontSize: 'var(--text-sm)' }}>
        Loading…
      </div>
    );
  }
  return <>{children}</>;
}

function HomeRedirect() {
  const auth = useAuthGate();
  if (auth === 'unknown') {
    return (
      <div style={{ padding: '48px', color: 'var(--ink-muted)', fontSize: 'var(--text-sm)' }}>
        Loading…
      </div>
    );
  }
  return <Navigate to={auth === 'auth' ? '/strategies' : '/login'} replace />;
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<HomeRedirect />} />
      <Route path="/login" element={<Login />} />
      <Route
        path="/strategies"
        element={
          <Protected>
            <AppLayout active="strategies">
              <Strategies />
            </AppLayout>
          </Protected>
        }
      />
      <Route
        path="/orb"
        element={
          <Protected>
            <AppLayout active="orb">
              <Orb />
            </AppLayout>
          </Protected>
        }
      />
      <Route
        path="/nas"
        element={
          <Protected>
            <AppLayout active="nas">
              <Nas />
            </AppLayout>
          </Protected>
        }
      />
      <Route path="*" element={<NotFound />} />
    </Routes>
  );
}
