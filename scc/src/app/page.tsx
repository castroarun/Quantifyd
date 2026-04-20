'use client';

import { useStore } from '@/store';
import { TABS } from '@/lib/data';
import { useSCCDashboard } from '@/hooks/useSCC';
import { Header } from '@/components/Header';
import { AlertBanner } from '@/components/AlertBanner';
import { KillAllModal } from '@/components/KillAllModal';
import { Dashboard } from '@/components/Dashboard';
import { PositionsAndTrades } from '@/components/PositionsAndTrades';
import { Blueprints } from '@/components/Blueprints';
import { DayDeepDive } from '@/components/DayDeepDive';

export default function Home() {
  const { tab, setTab, connected } = useStore();
  useSCCDashboard(); // Poll Flask every 5s

  return (
    <div className="min-h-screen" style={{ background: 'linear-gradient(160deg,#f8fafc 0%,#eef2f7 50%,#e4eaf1 100%)' }}>
      <Header />
      <KillAllModal />
      <AlertBanner />

      {/* Tab navigation */}
      <nav className="px-4 pt-3 pb-1 overflow-x-auto" style={{ scrollbarWidth: 'none' }}>
        <div className="inline-flex bg-slate-100 rounded-xl p-1 gap-0.5">
          {TABS.map((t) => (
            <button key={t.id} onClick={() => setTab(t.id)} className={`px-3.5 py-1.5 rounded-lg text-sm font-semibold transition-all whitespace-nowrap ${tab === t.id ? 'tab-on text-slate-900' : 'text-slate-500 hover:text-slate-700'}`}>
              {t.icon} {t.label}
            </button>
          ))}
        </div>
        {!connected && (
          <span className="ml-3 text-xs text-amber-600 font-medium">⏳ Connecting to Flask...</span>
        )}
      </nav>

      <main className="px-4 pb-8 fu" key={tab}>
        {tab === 'dash' && <Dashboard />}
        {tab === 'positions' && <PositionsAndTrades />}
        {tab === 'blueprints' && <Blueprints />}
        {tab === 'deepdive' && <DayDeepDive />}
      </main>
    </div>
  );
}
