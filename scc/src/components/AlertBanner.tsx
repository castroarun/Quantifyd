'use client';

import { useStore } from '@/store';

export function AlertBanner() {
  const { strategies, positions, setTab } = useStore();
  const tc = strategies.reduce((a, s) => a + s.capital, 0);
  const td = strategies.reduce((a, s) => a + s.deployed, 0);
  const capUtil = tc > 0 ? (td / tc) * 100 : 0;

  const dangerFlags = positions.filter((p) => p.flags.length > 0);
  const stratErrors = strategies.filter((s) => s.status === 'error');
  const hasAlerts = dangerFlags.length > 0 || stratErrors.length > 0 || capUtil > 75;

  if (!hasAlerts) return null;

  return (
    <div className="mx-4 mt-3 bg-red-50 border border-red-200 rounded-xl px-4 py-2.5 flex items-start gap-3">
      <span className="text-base mt-0.5">🚨</span>
      <div className="flex-1 space-y-1">
        {stratErrors.map((s) => (
          <p key={s.id} className="text-xs text-red-700 font-medium">⚠️ <strong>{s.name}</strong>: Strategy in error state — Max DD breached ({s.maxDD}%)</p>
        ))}
        {dangerFlags.map((p) => p.flags.map((f, i) => (
          <p key={p.id + i} className="text-xs text-red-700 font-medium">🔴 <strong>{p.symbol}</strong>: {f}</p>
        )))}
        {capUtil > 75 && <p className="text-xs text-amber-700 font-medium">⚡ Capital utilization at <strong>{capUtil.toFixed(0)}%</strong> — consider reducing exposure</p>}
      </div>
      <button onClick={() => setTab('positions')} className="text-xs text-red-600 font-bold hover:underline whitespace-nowrap">View →</button>
    </div>
  );
}
