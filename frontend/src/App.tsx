import { Routes, Route, Navigate, useNavigate } from 'react-router-dom';
import { useEffect, useState, lazy, Suspense } from 'react';
import { apiGet } from './api/client';
import type { AuthStatus } from './api/types';
import AppLayout from './components/Layout/AppLayout';
import Login from './pages/Login';

/* ── route-level code splitting (2026-09-07) ──
   One bundle meant every page carried all 45 of them: 1.9 MB of JavaScript before
   a single holding could be drawn. Each page is now its own chunk, fetched when its
   route is first opened, so the initial download is the shell plus the one page you
   asked for. Login stays eager — it is the entry point and must paint at once. */
const Overview = lazy(() => import('./pages/Overview'));
const Strategies = lazy(() => import('./pages/Strategies'));
const Orb = lazy(() => import('./pages/Orb'));
const Nas = lazy(() => import('./pages/Nas'));
const NasConfig = lazy(() => import('./pages/NasConfig'));
const ScaleUp = lazy(() => import('./pages/ScaleUp'));
const Straddles = lazy(() => import('./pages/Straddles'));
const NasPanic = lazy(() => import('./pages/NasPanic'));
const Nwv = lazy(() => import('./pages/Nwv'));
const OptionsStudy = lazy(() => import('./pages/OptionsStudy'));
const StraddleStudy = lazy(() => import('./pages/StraddleStudy'));
const N500m = lazy(() => import('./pages/N500m'));
const Strangle = lazy(() => import('./pages/Strangle'));
const Report = lazy(() => import('./pages/Report'));
const Holdings = lazy(() => import('./pages/Holdings'));
const HoldingsHistory = lazy(() => import('./pages/HoldingsHistory'));
const OptionsData = lazy(() => import('./pages/OptionsData'));
const FuturePlans = lazy(() => import('./pages/FuturePlans'));
const EodBreakout = lazy(() => import('./pages/EodBreakout'));
const Mst = lazy(() => import('./pages/Mst'));
const Intraday75wr = lazy(() => import('./pages/Intraday75wr'));
const PairTrading = lazy(() => import('./pages/PairTrading'));
const Scanner = lazy(() => import('./pages/Scanner'));
const BreakoutScanner = lazy(() => import('./pages/BreakoutScanner'));
const AthScanner = lazy(() => import('./pages/AthScanner'));
const IndexPulse = lazy(() => import('./pages/IndexPulse'));
const Backtest = lazy(() => import('./pages/Backtest'));
const BacktestStudy = lazy(() => import('./pages/BacktestStudy'));
const MomentumPaper = lazy(() => import('./pages/MomentumPaper'));
const Straddle45 = lazy(() => import('./pages/Straddle45'));
const StockWings = lazy(() => import('./pages/StockWings'));
const BreakoutPaper = lazy(() => import('./pages/BreakoutPaper'));
const BlueskyPaper = lazy(() => import('./pages/BlueskyPaper'));
const CapitalDesk = lazy(() => import('./pages/CapitalDesk'));
const IpoPaper = lazy(() => import('./pages/IpoPaper'));
const HaPaper = lazy(() => import('./pages/HaPaper'));
const OrbPaper = lazy(() => import('./pages/OrbPaper'));
const OholPaper = lazy(() => import('./pages/OholPaper'));
const FnomsPaper = lazy(() => import('./pages/FnomsPaper'));
const Journal = lazy(() => import('./pages/Journal'));
const JournalDay = lazy(() => import('./pages/JournalDay'));
const JournalTrade = lazy(() => import('./pages/JournalTrade'));
const JournalInsights = lazy(() => import('./pages/JournalInsights'));
const NotFound = lazy(() => import('./pages/NotFound'));

/* Shown while a page's chunk is in flight. Deliberately plain: a spinner that
   appears for 80ms reads as jank, so this is a quiet line that only registers if
   the network is genuinely slow. */
function ChunkFallback() {
  return (
    <div style={{ padding: '28px 22px', color: 'var(--ink-muted,#8a8a85)', fontSize: 13 }}>
      Loading…
    </div>
  );
}


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
  return <Navigate to={auth === 'auth' ? '/overview' : '/login'} replace />;
}

export default function App() {
  return (
    <Suspense fallback={<ChunkFallback />}>
    <Routes>
      <Route path="/" element={<HomeRedirect />} />
      <Route path="/login" element={<Login />} />
      <Route
        path="/overview"
        element={
          <Protected>
            <AppLayout active="overview">
              <Overview />
            </AppLayout>
          </Protected>
        }
      />
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
        path="/scaleup"
        element={
          <Protected>
            <AppLayout active="scaleup">
              <ScaleUp />
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
        path="/indices"
        element={
          <Protected>
            <AppLayout active="indices">
              <IndexPulse />
            </AppLayout>
          </Protected>
        }
      />
      <Route
        path="/ath-scanner"
        element={
          <Protected>
            <AppLayout active="ath-scanner">
              <AthScanner />
            </AppLayout>
          </Protected>
        }
      />
      {/* ORB-index strangle retired 2026-08-17 — redirect old links */}
      <Route path="/strangle" element={<Navigate to="/strategies" replace />} />
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
        path="/straddle45"
        element={
          <Protected>
            <AppLayout active="straddle45">
              <Straddle45 />
            </AppLayout>
          </Protected>
        }
      />
      <Route
        path="/stock-wings"
        element={
          <Protected>
            <AppLayout active="stock-wings">
              <StockWings />
            </AppLayout>
          </Protected>
        }
      />
      <Route
        path="/momentum-paper"
        element={
          <Protected>
            <AppLayout active="momentum-paper">
              <MomentumPaper />
            </AppLayout>
          </Protected>
        }
      />
      <Route
        path="/breakout-paper"
        element={
          <Protected>
            <AppLayout active="breakout-paper">
              <BreakoutPaper />
            </AppLayout>
          </Protected>
        }
      />
      <Route
        path="/bluesky-paper"
        element={
          <Protected>
            <AppLayout active="bluesky-paper">
              <BlueskyPaper />
            </AppLayout>
          </Protected>
        }
      />
      <Route
        path="/ipo-paper"
        element={
          <Protected>
            <AppLayout active="ipo-paper">
              <IpoPaper />
            </AppLayout>
          </Protected>
        }
      />
      <Route
        path="/capital"
        element={
          <Protected>
            <AppLayout active="capital">
              <CapitalDesk />
            </AppLayout>
          </Protected>
        }
      />
      {/* The page was called "Sleeves 50-50" until 05-Sep-2026. Any bookmark or link
          that still says /sleeves lands on the Capital Desk rather than a 404. */}
      <Route path="/sleeves" element={<Navigate to="/capital" replace />} />
      <Route
        path="/ha-paper"
        element={
          <Protected>
            <AppLayout active="ha-paper">
              <HaPaper />
            </AppLayout>
          </Protected>
        }
      />
      <Route
        path="/orb-paper"
        element={
          <Protected>
            <AppLayout active="orb-paper">
              <OrbPaper />
            </AppLayout>
          </Protected>
        }
      />
      <Route
        path="/ohol-paper"
        element={
          <Protected>
            <AppLayout active="ohol-paper">
              <OholPaper />
            </AppLayout>
          </Protected>
        }
      />
      <Route
        path="/fnoms-paper"
        element={
          <Protected>
            <AppLayout active="fnoms-paper">
              <FnomsPaper />
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
      <Route
        path="/straddle-study"
        element={
          <Protected>
            <AppLayout active="straddle-study">
              <StraddleStudy />
            </AppLayout>
          </Protected>
        }
      />
      <Route
        path="/options-study"
        element={
          <Protected>
            <AppLayout active="options-study">
              <OptionsStudy />
            </AppLayout>
          </Protected>
        }
      />
      <Route path="*" element={<NotFound />} />
    </Routes>
    </Suspense>
  );
}
