'use client';

import { useEffect, useState } from 'react';
import { useStore } from '@/store';

export function Header() {
  const { privacy, togglePrivacy, setShowKillModal, strategies, positions, connected } = useStore();
  const [time, setTime] = useState(new Date());

  useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  const dangerFlags = positions.filter((p) => p.flags.length > 0);
  const stratErrors = strategies.filter((s) => s.status === 'error');
  const hasAlerts = dangerFlags.length > 0 || stratErrors.length > 0;

  return (
    <header className="bg-white border-b border-slate-200 px-4 py-2.5 flex items-center justify-between sticky top-0 z-50" style={{ boxShadow: '0 1px 3px rgba(0,0,0,.03)' }}>
      <div className="flex items-center gap-2.5">
        <div className="w-8 h-8 rounded-lg flex items-center justify-center text-white font-bold text-xs" style={{ background: 'linear-gradient(135deg,#0f172a,#334155)' }}>SC</div>
        <h1 className="text-base font-bold text-slate-900 tracking-tight">Strategy Command Center</h1>
      </div>
      <div className="flex items-center gap-2.5">
        <span className="flex items-center gap-1.5 text-xs text-slate-500">
          <span className={`w-1.5 h-1.5 rounded-full ${connected ? 'bg-emerald-500 animate-pulse' : 'bg-amber-400'}`} />
          <span className="font-mono">{time.toLocaleTimeString('en-IN', { hour12: false })}</span>
        </span>
        <button onClick={togglePrivacy} className={`px-2.5 py-1 rounded-lg text-xs font-semibold border transition-all ${privacy ? 'bg-slate-900 text-white border-slate-900' : 'bg-white text-slate-500 border-slate-200'}`}>
          {privacy ? '🔒' : '🔓'}
        </button>
        {hasAlerts && <span className="w-2 h-2 rounded-full bg-red-500 flag-pulse" />}
        <button onClick={() => setShowKillModal(true)} className="kill-btn text-white px-3 py-1 rounded-lg text-xs font-bold hover:scale-105 transition-transform">
          ✕ KILL ALL
        </button>
      </div>
    </header>
  );
}
