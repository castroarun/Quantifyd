'use client';

import { useState } from 'react';
import { useStore } from '@/store';
import { BLUEPRINTS, INIT_STRATEGIES } from '@/lib/data';
import { StatusBadge, SystemTypeBadge } from './ui/StatusBadge';

function RulesList({ rules, color }: { rules: string[]; color: string }) {
  return (
    <div className="space-y-2">
      {rules.map((r, i) => (
        <div key={i} className="flex items-start gap-2">
          <span className={`min-w-[20px] h-5 rounded-[5px] flex items-center justify-center text-[10px] font-bold flex-shrink-0 ${color}`}>{i + 1}</span>
          <span className="text-sm text-slate-700 leading-relaxed">{r}</span>
        </div>
      ))}
    </div>
  );
}

function BMetric({ label, value, warn }: { label: string; value: string | number; warn?: boolean }) {
  return (
    <div>
      <span className="text-xs text-slate-400">{label}</span>
      <span className={`text-sm font-bold font-mono block ${warn ? 'text-red-600' : 'text-slate-800'}`}>{value}</span>
    </div>
  );
}

export function Blueprints() {
  const { privacy } = useStore();
  const [sel, setSel] = useState(BLUEPRINTS[0].id);
  const [view, setView] = useState<'rules' | 'backtest'>('rules');
  const bp = BLUEPRINTS.find((b) => b.id === sel);
  const bt = bp?.backtest;

  if (!bp || !bt) return null;

  return (
    <div className="pt-2 space-y-4">
      {/* Strategy selector pills */}
      <div className="flex flex-wrap gap-2 overflow-x-auto pb-1" style={{ scrollbarWidth: 'none' }}>
        {BLUEPRINTS.map((b) => {
          const st = INIT_STRATEGIES.find((s) => s.id === b.id);
          return (
            <button key={b.id} onClick={() => setSel(b.id)} className={`flex items-center gap-2 px-3 py-2 rounded-xl text-sm font-semibold whitespace-nowrap border transition-all ${sel === b.id ? 'bg-white border-slate-300 text-slate-900 shadow-sm' : 'border-transparent text-slate-500 hover:bg-white hover:border-slate-200'}`}>
              {st && <StatusBadge status={st.status} />}
              {b.name}
            </button>
          );
        })}
      </div>

      <div className="fu" key={bp.id + view}>
        {/* Header */}
        <div className="bg-white rounded-t-2xl border border-slate-200 border-b-0 p-5">
          <div className="flex flex-wrap items-start justify-between gap-3 mb-2">
            <div>
              <div className="flex flex-wrap items-center gap-2 mb-1">
                <h3 className="text-lg font-bold text-slate-900">{bp.name}</h3>
                <span className="font-mono text-xs bg-slate-100 text-slate-500 px-2 py-0.5 rounded-lg font-semibold">{bp.version}</span>
                <SystemTypeBadge type={bp.systemType} />
              </div>
              <p className="text-sm text-slate-600 max-w-xl">{bp.description}</p>
            </div>
            <a href={bp.github} target="_blank" rel="noopener noreferrer" className="flex items-center gap-2 bg-slate-900 text-white px-3 py-2 rounded-xl text-xs font-semibold hover:bg-slate-800 flex-shrink-0">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="white"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
              GitHub
            </a>
          </div>
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-400 mb-3">
            <span>⏱ <span className="text-slate-700 font-semibold">{bp.holdingPeriod}</span></span>
            <span>📊 <span className="text-slate-700 font-semibold">{bp.timeframe}</span></span>
            <span>🎯 <span className="text-slate-700 font-semibold">{bp.instruments}</span></span>
            <span>💰 <span className="text-slate-700 font-semibold">{bp.capital}</span></span>
          </div>
          <div className="text-xs text-slate-400 mb-3">Commit: <span className="font-mono text-slate-500">{bp.lastCommit}</span> — <span className="italic">{bp.commitMsg}</span></div>
          <div className="flex gap-1">
            <button onClick={() => setView('rules')} className={`px-3 py-1.5 rounded-lg text-xs font-semibold ${view === 'rules' ? 'bg-slate-900 text-white' : 'bg-slate-100 text-slate-500'}`}>📋 Rules</button>
            <button onClick={() => setView('backtest')} className={`px-3 py-1.5 rounded-lg text-xs font-semibold ${view === 'backtest' ? 'bg-slate-900 text-white' : 'bg-slate-100 text-slate-500'}`}>📊 Backtest</button>
          </div>
        </div>

        {view === 'rules' ? (
          <div className={privacy ? 'blur-it' : ''}>
            <div className="grid lg:grid-cols-3 border border-slate-200 border-t-0 bg-white">
              <div className="p-5 lg:border-r border-slate-100">
                <div className="flex items-center gap-2 mb-3">
                  <span className="w-7 h-7 rounded-lg bg-emerald-100 flex items-center justify-center text-sm">📥</span>
                  <div><h4 className="text-sm font-bold text-slate-900">Entry</h4><span className="text-xs text-slate-400">ALL must be true</span></div>
                </div>
                <RulesList rules={bp.entry} color="bg-emerald-50 text-emerald-700" />
              </div>
              <div className="p-5 lg:border-r border-slate-100 border-t lg:border-t-0">
                <div className="flex items-center gap-2 mb-3">
                  <span className="w-7 h-7 rounded-lg bg-sky-100 flex items-center justify-center text-sm">📤</span>
                  <div><h4 className="text-sm font-bold text-slate-900">Exit</h4><span className="text-xs text-slate-400">ANY triggers</span></div>
                </div>
                <RulesList rules={bp.exit} color="bg-sky-50 text-sky-700" />
              </div>
              <div className="p-5 border-t lg:border-t-0">
                <div className="flex items-center gap-2 mb-3">
                  <span className="w-7 h-7 rounded-lg bg-red-100 flex items-center justify-center text-sm">🛑</span>
                  <div><h4 className="text-sm font-bold text-slate-900">Stop Loss</h4><span className="text-xs text-slate-400">{bp.stopLoss.length} layers</span></div>
                </div>
                <RulesList rules={bp.stopLoss} color="bg-red-50 text-red-700" />
              </div>
            </div>
            <div className="bg-white border border-slate-200 border-t-0 px-5 py-3">
              <p className="text-sm text-slate-700 bg-amber-50 rounded-lg px-3 py-2 font-medium">⚖️ {bp.positionSizing}</p>
            </div>
            <div className="grid sm:grid-cols-2 bg-white border border-slate-200 border-t-0">
              <div className="px-5 py-3 sm:border-r border-slate-100">
                <span className="text-xs font-bold text-slate-400 uppercase">Indicators</span>
                <div className="flex flex-wrap gap-1 mt-1">{bp.indicators.map((x, i) => <span key={i} className="font-mono text-xs bg-slate-100 text-slate-700 px-2 py-0.5 rounded">{x}</span>)}</div>
              </div>
              <div className="px-5 py-3 border-t sm:border-t-0">
                <span className="text-xs font-bold text-slate-400 uppercase">Filters</span>
                <div className="flex flex-wrap gap-1 mt-1">{bp.filters.map((x, i) => <span key={i} className="text-xs bg-slate-50 text-slate-600 px-2 py-0.5 rounded border border-slate-200">{x}</span>)}</div>
              </div>
            </div>
            <div className="bg-white rounded-b-2xl border border-slate-200 border-t-0 px-5 py-2.5 flex items-center gap-2">
              <span className="text-xs text-slate-400 font-semibold">Tags:</span>
              {bp.tags.map((t, i) => <span key={i} className="text-xs bg-violet-50 text-violet-600 px-2 py-0.5 rounded-full font-semibold">#{t}</span>)}
            </div>
          </div>
        ) : (
          <div className="bg-white rounded-b-2xl border border-slate-200 border-t-0 p-5">
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-sm font-bold text-slate-900">Backtest Results</h4>
              <span className="text-xs bg-slate-100 text-slate-500 px-2 py-0.5 rounded-lg font-semibold font-mono">{bt.period}</span>
            </div>
            <div className="grid grid-cols-4 lg:grid-cols-8 gap-3 mb-4 pb-4 border-b border-slate-100">
              <BMetric label="Trades" value={bt.totalTrades} />
              <BMetric label="Win%" value={bt.winRate + '%'} />
              <BMetric label="PF" value={bt.profitFactor} />
              <BMetric label="Expectancy" value={(bt.expectancy > 0 ? '+' : '') + bt.expectancy + 'R'} warn={bt.expectancy < 0} />
              <BMetric label="Sharpe" value={bt.sharpe} />
              <BMetric label="Sortino" value={bt.sortino} />
              <BMetric label="Calmar" value={bt.calmar} />
              <BMetric label="Avg Hold" value={bt.avgHolding} />
            </div>
            <div className="bg-red-50 rounded-xl p-4 mb-4">
              <h5 className="text-xs font-bold text-red-800 uppercase tracking-wider mb-2">📉 Drawdown — What to Expect</h5>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div><span className="text-xs text-red-500">Max Strategy DD</span><span className="text-lg font-bold font-mono text-red-700 block">{bt.maxDD}%</span></div>
                <div><span className="text-xs text-red-500">Avg Strategy DD</span><span className="text-lg font-bold font-mono text-red-700 block">{bt.avgDD}%</span></div>
                <div><span className="text-xs text-red-500">Max DD/Trade</span><span className="text-lg font-bold font-mono text-red-700 block">{bt.maxDDTrade}%</span></div>
                <div><span className="text-xs text-red-500">Avg DD/Trade</span><span className="text-lg font-bold font-mono text-red-700 block">{bt.avgDDTrade}%</span></div>
              </div>
              <div className="mt-2 pt-2 border-t border-red-200 text-xs text-red-600">Max consecutive losses: <strong className="font-mono">{bt.maxConsecLoss}</strong> · Avg loss: <strong className="font-mono">{bt.avgLoss}%</strong></div>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-3">
              <BMetric label="Avg Win" value={'+' + bt.avgWin + '%'} />
              <BMetric label="Avg Loss" value={bt.avgLoss + '%'} warn />
              <BMetric label="Best Month" value={'+' + bt.bestMonth + '%'} />
              <BMetric label="Worst Month" value={bt.worstMonth + '%'} warn />
            </div>
            <div className="bg-emerald-50 rounded-lg px-3 py-2 text-sm">
              <span className="font-semibold text-emerald-800">Monthly:</span>
              <span className="font-mono font-bold text-emerald-700 ml-1">{bt.avgMonthly > 0 ? '+' : ''}{bt.avgMonthly}%</span>
              <span className="text-emerald-600 ml-1 text-xs">avg over {bt.period}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
