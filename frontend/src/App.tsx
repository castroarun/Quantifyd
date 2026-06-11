import { Routes, Route, Navigate, useNavigate } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { apiGet } from './api/client';
import type { AuthStatus } from './api/types';
import AppLayout from './components/Layout/AppLayout';
import Login from './pages/Login';
import Strategies from './pages/Strategies';
import Orb from './pages/Orb';
import Nas from './pages/Nas';
import NasConfig from './pages/NasConfig';
import Straddles from './pages/Straddles';
import NasPanic from './pages/NasPanic';
import Nwv from './pages/Nwv';
import N500m from './pages/N500m';
import Strangle from './pages/Strangle';
import Report from './pages/Report';
import Holdings from './pages/Holdings';
import HoldingsHistory from './pages/HoldingsHistory';
import OptionsData from './pages/OptionsData';
import FuturePlans from './pages/FuturePlans';
import EodBreakout from './pages/EodBreakout';
import Mst from './pages/Mst';
import Intraday75wr from './pages/Intraday75wr';
import PairTrading from './pages/PairTrading';
import Scanner from './pages/Scanner';
import BreakoutScanner from './pages/BreakoutScanner';
import Backtest from './pages/Backtest';
import BacktestStudy from './pages/BacktestStudy';
import Journal from './pages/Journal';
import JournalDay from './pages/JournalDay';
import JournalTrade from './pages/JournalTrade';
import JournalInsights from './pages/JournalInsights';
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
      <Route
        path="/nas-config"
        element={
          <Protected>
            <AppLayout active="nas-config">
              <NasConfig />
            </AppLayout>
          </Protected>
        }
      />
      <Route
        path="/straddles"
        element={
          <Protected>
            <AppLayout active="straddles">
              <Straddles />
            </AppLayout>
          </Protected>
        }
      />
      <Route
        path="/nas-panic"
        element={
          <Protected>
            <AppLayout active="nas">
              <NasPanic />
            </AppLayout>
          </Protected>
        }
      />
      <Route
        path="/nwv"
        element={
          <Protected>
            <AppLayout active="nwv">
              <Nwv />
            </AppLayout>
          </Protected>
        }
      />
      <Route
        path="/n500m"
        element={
          <Protected>
            <AppLayout active="n500m">
              <N500m />
            </AppLayout>
          </Protected>
        }
      />
      <Route
        path="/strangle"
        element={
          <Protected>
            <AppLayout active="strangle">
              <Strangle />
            </AppLayout>
          </Protected>
        }
      />
      <Route
        path="/mst"
        element={
          <Protected>
            <AppLayout active="mst">
              <Mst />
            </AppLayout>
          </Protected>
        }
      />
      <Route
        path="/intraday75wr"
        element={
          <Protected>
            <AppLayout active="intraday75wr">
              <Intraday75wr />
            </AppLayout>
          </Protected>
        }
      />
      <Route
        path="/pair-trading"
        element={
          <Protected>
            <AppLayout active="pair-trading">
              <PairTrading />
            </AppLayout>
          </Protected>
        }
      />
      <Route
        path="/scanner"
        element={
          <Protected>
            <AppLayout active="scanner">
              <Scanner />
            </AppLayout>
          </Protected>
        }
      />
      <Route
        path="/breakout-scanner"
        element={
          <Protected>
            <AppLayout active="breakout-scanner">
              <BreakoutScanner />
            </AppLayout>
          </Protected>
        }
      />
      <Route
        path="/backtest"
        element={
          <Protected>
            <AppLayout active="backtest">
              <Backtest />
            </AppLayout>
          </Protected>
        }
      />
      <Route
        path="/backtest/:slug"
        element={
          <Protected>
            <AppLayout active="backtest">
              <BacktestStudy />
            </AppLayout>
          </Protected>
        }
      />
      <Route
        path="/report"
        element={
          <Protected>
            <AppLayout active="reports">
              <Report />
            </AppLayout>
          </Protected>
        }
      />
      <Route
        path="/holdings"
        element={
          <Protected>
            <AppLayout active="holdings">
              <Holdings />
            </AppLayout>
          </Protected>
        }
      />
      <Route
        path="/holdings/history"
        element={
          <Protected>
            <AppLayout active="holdings">
              <HoldingsHistory />
            </AppLayout>
          </Protected>
        }
      />
      <Route
        path="/options-data"
        element={
          <Protected>
            <AppLayout active="options-data">
              <OptionsData />
            </AppLayout>
          </Protected>
        }
      />
      <Route
        path="/future-plans"
        element={
          <Protected>
            <AppLayout active="future-plans">
              <FuturePlans />
            </AppLayout>
          </Protected>
        }
      />
      <Route
        path="/eod-breakout"
        element={
          <Protected>
            <AppLayout active="eod-breakout">
              <EodBreakout />
            </AppLayout>
          </Protected>
        }
      />
      <Route
        path="/journal"
        element={
          <Protected>
            <AppLayout active="journal">
              <Journal />
            </AppLayout>
          </Protected>
        }
      />
      <Route
        path="/journal/insights"
        element={
          <Protected>
            <AppLayout active="journal">
              <JournalInsights />
            </AppLayout>
          </Protected>
        }
      />
      <Route
        path="/journal/day/:date"
        element={
          <Protected>
            <AppLayout active="journal">
              <JournalDay />
            </AppLayout>
          </Protected>
        }
      />
      <Route
        path="/journal/trade/:id"
        element={
          <Protected>
            <AppLayout active="journal">
              <JournalTrade />
            </AppLayout>
          </Protected>
        }
      />
      <Route path="*" element={<NotFound />} />
    </Routes>
  );
}
