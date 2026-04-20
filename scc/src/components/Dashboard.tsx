'use client';

import { useStore } from '@/store';
import { fmt, pc, cl } from '@/lib/format';
import { PNL_WEEKLY, PNL_MONTHLY } from '@/lib/data';
import { MetricCard } from './ui/MetricCard';
import { MiniBar } from './ui/MiniBar';
import { StatusBadge, ModeBadge } from './ui/StatusBadge';
import { toggleStrategyAPI } from '@/hooks/useSCC';

export function Dashboard() {
  const { strategies, positions, trades, privacy, toggleStrategy, setTab } = useStore();
  const tc = strategies.reduce((a, s) => a + s.capital, 0);
  const td = strategies.reduce((a, s) => a + s.deployed, 0);
  const tp = strategies.reduce((a, s) => a + s.pnlToday, 0);
  const ta = strategies.reduce((a, s) => a + s.pnlTotal, 0);
  const run = strategies.filter((s) => s.status === 'running').length;
  const capUtil = tc > 0 ? (td / tc) * 100 : 0;
  const b = privacy ? 'blur-it' : '';

  const handleToggle = async (id: string) => {
    toggleStrategy(id); // Optimistic UI
    await toggleStrategyAPI(id); // Sync to Flask
  };

  return (
    <div className="space-y-4 pt-2">
      {/* Metrics row */}
      <div className={`grid grid-cols-3 lg:grid-cols-6 gap-2.5 ${b}`}>
        <MetricCard label="Today P&L" value={fmt(tp)} accent={cl(tp)} sub={pc(tp / tc * 100)} />
        <MetricCard label="Total P&L" value={fmt(ta)} accent={cl(ta)} />
        <MetricCard label="Deployed" value={fmt(td)} accent="text-sky-700" sub={`${capUtil.toFixed(0)}% of ${fmt(tc)}`} />
        <MetricCard label="Free Cash" value={fmt(tc - td)} accent="text-emerald-700" />
        <MetricCard label="Strategies" value={`${run}/${strategies.length}`} sub={`${run} live`} />
        <MetricCard label="Positions" value={positions.length} sub="open" />
      </div>

      {/* Capital utilization bar */}
      <div className={`bg-white rounded-xl border border-slate-200 p-4 shadow-sm ${b}`}>
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider">Capital Deployment</h3>
          <span className={`text-xs font-bold font-mono ${capUtil > 75 ? 'text-red-600' : capUtil > 60 ? 'text-amber-600' : 'text-emerald-600'}`}>{capUtil.toFixed(0)}% used</span>
        </div>
        <div className="h-2.5 bg-slate-100 rounded-full overflow-hidden mb-3">
          <div className="h-full rounded-full transition-all" style={{ width: `${capUtil}%`, background: capUtil > 80 ? '#dc2626' : capUtil > 60 ? '#f59e0b' : '#059669' }} />
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2">
          {strategies.map((st) => {
            const u = st.capital > 0 ? (st.deployed / st.capital * 100) : 0;
            return (
              <div key={st.id} className="text-xs">
                <div className="flex justify-between mb-0.5">
                  <span className="text-slate-600 truncate font-medium" style={{ maxWidth: 110 }}>{st.name}</span>
                  <span className="font-mono text-slate-400">{u.toFixed(0)}%</span>
                </div>
                <div className="h-1 bg-slate-100 rounded-full">
                  <div className="h-full rounded-full" style={{ width: `${u}%`, background: u > 85 ? '#dc2626' : u > 60 ? '#f59e0b' : '#059669' }} />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="grid lg:grid-cols-3 gap-4">
        {/* Weekly P&L */}
        <div className={`bg-white rounded-xl border border-slate-200 p-4 shadow-sm ${b}`}>
          <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">Weekly P&L</h3>
          <MiniBar data={PNL_WEEKLY} height={64} />
          <div className="mt-1.5 text-right">
            <span className={`font-mono text-xs font-bold ${cl(PNL_WEEKLY.reduce((a, d) => a + d.pnl, 0))}`}>{fmt(PNL_WEEKLY.reduce((a, d) => a + d.pnl, 0))}</span>
          </div>
        </div>

        {/* Monthly P&L */}
        <div className={`bg-white rounded-xl border border-slate-200 p-4 shadow-sm ${b}`}>
          <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">Monthly (6M)</h3>
          <MiniBar data={PNL_MONTHLY} height={64} />
          <div className="mt-1.5 text-right">
            <span className={`font-mono text-xs font-bold ${cl(PNL_MONTHLY.reduce((a, d) => a + d.pnl, 0))}`}>{fmt(PNL_MONTHLY.reduce((a, d) => a + d.pnl, 0))}</span>
          </div>
        </div>

        {/* Strategy quick controls */}
        <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
          <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">Strategies</h3>
          <div className="space-y-1.5">
            {strategies.map((st) => (
              <div key={st.id} className={`flex items-center justify-between py-1 px-2 rounded-lg ${st.status === 'error' ? 'bg-red-50' : ''}`}>
                <div className="flex items-center gap-2">
                  <StatusBadge status={st.status} />
                  <span className="text-xs font-medium text-slate-700 truncate" style={{ maxWidth: 100 }}>{st.name}</span>
                  <ModeBadge mode={st.mode} />
                </div>
                <div className="flex items-center gap-1.5">
                  <span className={`font-mono text-xs font-bold ${b} ${cl(st.pnlToday)}`}>{fmt(st.pnlToday)}</span>
                  <button onClick={() => handleToggle(st.id)} className={`w-6 h-6 rounded-md flex items-center justify-center text-xs ${st.status === 'running' ? 'bg-emerald-50 text-emerald-600' : 'bg-slate-100 text-slate-400'}`}>
                    {st.status === 'running' ? '⏸' : '▶'}
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Recent journal entries */}
      <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
        <div className="flex items-center justify-between mb-2">
          <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider">Recent Journal Notes</h3>
          <button onClick={() => setTab('positions')} className="text-xs text-sky-600 font-semibold hover:underline">All trades →</button>
        </div>
        <div className={`space-y-2 ${b}`}>
          {trades.filter((t) => t.journal).slice(0, 3).map((t) => (
            <div key={t.id} className="flex items-start gap-2 text-xs">
              <span className={`font-mono font-bold ${cl(t.pnl)} flex-shrink-0 w-16`}>{fmt(t.pnl)}</span>
              <span className="text-slate-600 flex-1">{t.journal}</span>
              <span className="text-slate-400 flex-shrink-0">{t.date}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
