'use client';

import { useState, useEffect } from 'react';
import { useStore } from '@/store';
import { fmt, pc, cl } from '@/lib/format';
import { DAY_SNAPSHOTS } from '@/lib/data';
import { MetricCard } from './ui/MetricCard';

export function DayDeepDive() {
  const { privacy } = useStore();
  const [idx, setIdx] = useState(0);
  const day = DAY_SNAPSHOTS[idx];
  const canPrev = idx < DAY_SNAPSHOTS.length - 1;
  const canNext = idx > 0;

  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft' && canPrev) setIdx((i) => i + 1);
      if (e.key === 'ArrowRight' && canNext) setIdx((i) => i - 1);
    };
    window.addEventListener('keydown', h);
    return () => window.removeEventListener('keydown', h);
  }, [canPrev, canNext]);

  const wr = day.trades > 0 ? (day.wins / day.trades * 100).toFixed(0) : '0';
  const stBreak = [...new Set(day.items.map((t) => t.strategy))].map((s) => ({
    name: s,
    pnl: day.items.filter((t) => t.strategy === s).reduce((a, t) => a + t.pnl, 0),
    cnt: day.items.filter((t) => t.strategy === s).length,
  }));

  const b = privacy ? 'blur-it' : '';

  return (
    <div className="pt-2 space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-base font-bold text-slate-900">Day Deep Dive</h2>
        <div className="flex items-center gap-2.5">
          <button className="arrow-btn" disabled={!canPrev} onClick={() => setIdx((i) => i + 1)}>←</button>
          <div className="text-center min-w-[120px]">
            <div className="font-bold text-slate-900 text-sm">{day.date}</div>
            <div className="text-xs text-slate-400">{day.dow} · {idx + 1}/{DAY_SNAPSHOTS.length}</div>
          </div>
          <button className="arrow-btn" disabled={!canNext} onClick={() => setIdx((i) => i - 1)}>→</button>
        </div>
      </div>

      <div className={`fu ${b}`} key={idx}>
        <div className="grid grid-cols-3 sm:grid-cols-6 gap-2.5 mb-4">
          <MetricCard label="Day P&L" value={fmt(day.pnl)} accent={cl(day.pnl)} />
          <MetricCard label="Trades" value={day.trades} sub={`${day.wins}W/${day.losses}L`} />
          <MetricCard label="Win Rate" value={wr + '%'} accent={Number(wr) >= 50 ? 'text-emerald-700' : 'text-red-600'} />
          <MetricCard label="Best" value={fmt(day.bestTrade)} accent="text-emerald-700" />
          <MetricCard label="Worst" value={fmt(day.worstTrade)} accent="text-red-600" />
          <MetricCard label="Capital" value={fmt(day.capitalUsed)} />
        </div>

        {/* Strategy P&L breakdown */}
        <div className="flex flex-wrap gap-2 mb-4">
          {stBreak.map((s) => (
            <div key={s.name} className="bg-white rounded-lg border border-slate-200 px-3 py-2 flex items-center gap-2">
              <span className="text-xs text-slate-600 font-medium">{s.name}</span>
              <span className={`font-mono text-xs font-bold ${cl(s.pnl)}`}>{fmt(s.pnl)}</span>
              <span className="text-xs text-slate-400">{s.cnt}t</span>
            </div>
          ))}
        </div>

        {/* Trades */}
        <div className="bg-white rounded-xl border border-slate-200 overflow-hidden shadow-sm">
          <div className="px-4 py-2.5 border-b border-slate-100 flex justify-between">
            <span className="text-sm font-semibold text-slate-700">All Trades</span>
            <span className={`font-mono text-sm font-bold ${cl(day.pnl)}`}>Net: {fmt(day.pnl)}</span>
          </div>
          <div className="divide-y divide-slate-100">
            {day.items.map((t, i) => (
              <div key={i} className="px-4 py-2.5 hovr">
                <div className="flex items-center justify-between mb-0.5">
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-slate-900 font-mono text-xs">{t.symbol}</span>
                    <span className={`text-xs font-bold ${t.side === 'LONG' ? 'text-emerald-600' : 'text-red-600'}`}>{t.side}</span>
                    <span className="text-xs bg-slate-100 text-slate-500 px-1.5 py-0.5 rounded">{t.strategy}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={`font-mono text-xs font-bold ${cl(t.pnl)}`}>{fmt(t.pnl)}</span>
                    <span className={`font-mono text-xs ${cl(t.pnlPct)}`}>{pc(t.pnlPct)}</span>
                  </div>
                </div>
                <div className="flex gap-3 text-xs text-slate-400">
                  <span>En: <span className="font-mono text-slate-600">{t.entry}</span></span>
                  <span>Ex: <span className="font-mono text-slate-600">{t.exit}</span></span>
                  <span>⏱ {t.timeWindow}</span>
                </div>
                {t.notes && <p className="text-xs text-slate-500 mt-1">📝 {t.notes}</p>}
              </div>
            ))}
          </div>
        </div>
      </div>

      <p className="text-xs text-slate-400 text-center">← → keys or buttons to navigate days</p>
    </div>
  );
}
