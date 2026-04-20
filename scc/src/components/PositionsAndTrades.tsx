'use client';

import { useState } from 'react';
import { useStore } from '@/store';
import { fmt, pc, cl } from '@/lib/format';
import { RiskBadge } from './ui/StatusBadge';

export function PositionsAndTrades() {
  const { positions, strategies, trades, closePosition, privacy } = useStore();
  const [confirmId, setConfirmId] = useState<string | null>(null);
  const [view, setView] = useState<'live' | 'history'>('live');
  const [histFilter, setHistFilter] = useState<'all' | 'wins' | 'losses'>('all');
  const [editJournal, setEditJournal] = useState<string | null>(null);
  const [journalText, setJournalText] = useState('');
  const [journals, setJournals] = useState<Record<string, string>>(() => {
    const m: Record<string, string> = {};
    trades.forEach((t) => { if (t.journal) m[t.id] = t.journal; });
    return m;
  });

  const filtHist = histFilter === 'all' ? trades
    : trades.filter((t) => t.pnl > 0 ? histFilter === 'wins' : histFilter === 'losses');

  const saveJournal = (id: string) => { setJournals({ ...journals, [id]: journalText }); setEditJournal(null); };
  const b = privacy ? 'blur-it' : '';

  return (
    <div className="pt-2 space-y-4">
      {/* Toggle live/history */}
      <div className="flex items-center justify-between">
        <div className="inline-flex bg-slate-100 rounded-lg p-0.5">
          <button onClick={() => setView('live')} className={`px-3.5 py-1.5 rounded-md text-sm font-semibold ${view === 'live' ? 'tab-on text-slate-900' : 'text-slate-500'}`}>📊 Live ({positions.length})</button>
          <button onClick={() => setView('history')} className={`px-3.5 py-1.5 rounded-md text-sm font-semibold ${view === 'history' ? 'tab-on text-slate-900' : 'text-slate-500'}`}>📜 History</button>
        </div>
        {view === 'history' && (
          <div className="inline-flex bg-slate-100 rounded-lg p-0.5 text-xs font-semibold">
            {(['all', 'wins', 'losses'] as const).map((f) => (
              <button key={f} onClick={() => setHistFilter(f)} className={`px-2.5 py-1 rounded-md capitalize ${histFilter === f ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-500'}`}>{f}</button>
            ))}
          </div>
        )}
      </div>

      {view === 'live' ? (
        <>
          {positions.length === 0 ? (
            <div className="bg-white rounded-xl border p-10 text-center text-slate-400">No open positions</div>
          ) : (
            <div className={`bg-white rounded-xl border border-slate-200 overflow-hidden shadow-sm ${b}`}>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="bg-slate-50 border-b border-slate-200">
                      {['', 'Strategy', 'Symbol', 'Side', 'Qty', 'Entry', 'Current', 'P&L', 'P&L%', 'SL', 'TP', ''].map((h) => (
                        <th key={h} className="px-2.5 py-2.5 text-left font-semibold text-slate-500 uppercase tracking-wider whitespace-nowrap">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {positions.map((x) => (
                      <tr key={x.id} className={`border-b border-slate-100 hovr ${x.flags.length > 0 ? 'bg-red-50 bg-opacity-50' : ''}`}>
                        <td className="pl-2.5 py-2">{x.flags.length > 0 && <span className="text-red-500 flag-pulse text-sm">⚠️</span>}</td>
                        <td className="px-2.5 py-2 text-slate-500 whitespace-nowrap">{x.strategy}</td>
                        <td className="px-2.5 py-2 font-bold text-slate-900 font-mono whitespace-nowrap">
                          {x.symbol}
                          {x.flags.length > 0 && (
                            <div className="flex flex-wrap gap-1 mt-0.5">
                              {x.flags.map((f, i) => <span key={i} className="text-xs bg-red-100 text-red-700 px-1.5 py-0.5 rounded font-semibold">{f}</span>)}
                            </div>
                          )}
                        </td>
                        <td className={`px-2.5 py-2 font-bold ${x.side === 'LONG' ? 'text-emerald-600' : 'text-red-600'}`}>{x.side}</td>
                        <td className="px-2.5 py-2 font-mono">{x.qty}</td>
                        <td className="px-2.5 py-2 font-mono">{x.entry.toFixed(2)}</td>
                        <td className="px-2.5 py-2 font-mono font-semibold">{x.current.toFixed(2)}</td>
                        <td className={`px-2.5 py-2 font-mono font-bold ${cl(x.pnl)}`}>{fmt(x.pnl)}</td>
                        <td className={`px-2.5 py-2 font-mono font-bold ${cl(x.pnlPct)}`}>{pc(x.pnlPct)}</td>
                        <td className="px-2.5 py-2 font-mono text-red-400">{x.sl}</td>
                        <td className="px-2.5 py-2 font-mono text-emerald-500">{x.tp || '—'}</td>
                        <td className="px-2.5 py-2">
                          {confirmId === x.id ? (
                            <div className="flex gap-1">
                              <button onClick={() => { closePosition(x.id); setConfirmId(null); }} className="kill-btn text-white px-2 py-0.5 rounded font-bold">Exit</button>
                              <button onClick={() => setConfirmId(null)} className="bg-slate-100 text-slate-500 px-2 py-0.5 rounded">No</button>
                            </div>
                          ) : (
                            <button onClick={() => setConfirmId(x.id)} className="bg-red-50 text-red-600 px-2 py-0.5 rounded font-semibold hover:bg-red-100 whitespace-nowrap">Close</button>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Risk summary strip */}
          <div className={`grid grid-cols-3 sm:grid-cols-5 gap-2 ${b}`}>
            {strategies.map((st) => (
              <div key={st.id} className={`bg-white rounded-lg border p-2.5 text-center ${st.status === 'error' ? 'border-red-200 bg-red-50' : 'border-slate-200'}`}>
                <div className="text-xs text-slate-500 truncate">{st.name}</div>
                <div className={`font-mono text-sm font-bold ${cl(st.pnlToday)}`}>{fmt(st.pnlToday)}</div>
                <div className="flex justify-center gap-1 mt-1">
                  <RiskBadge risk={st.risk} />
                  <span className="font-mono text-xs text-slate-400">{st.sharpe}S</span>
                </div>
              </div>
            ))}
          </div>
        </>
      ) : (
        /* TRADE HISTORY */
        <div className="space-y-2.5">
          {filtHist.map((t) => (
            <div key={t.id} className={`bg-white rounded-xl border shadow-sm ${t.pnl > 0 ? 'border-emerald-100' : 'border-red-100'}`}>
              <div className={`p-4 ${b}`}>
                <div className="flex flex-wrap items-start justify-between gap-2 mb-1.5">
                  <div>
                    <div className="flex items-center gap-2 mb-0.5">
                      <span className="font-bold text-slate-900 font-mono text-sm">{t.symbol}</span>
                      <span className={`text-xs font-bold ${t.side === 'LONG' ? 'text-emerald-600' : 'text-red-600'}`}>{t.side}</span>
                      <span className="text-xs bg-slate-100 text-slate-500 px-1.5 py-0.5 rounded">{t.strategy}</span>
                    </div>
                    <span className="text-xs text-slate-400">{t.date} · {t.holding}</span>
                  </div>
                  <div className="text-right">
                    <div className={`text-lg font-bold font-mono ${cl(t.pnl)}`}>{fmt(t.pnl)}</div>
                    <div className={`text-xs font-mono font-semibold ${cl(t.pnlPct)}`}>{pc(t.pnlPct)}</div>
                  </div>
                </div>
                {t.notes && <p className="text-xs text-slate-500 mb-2">📋 {t.notes}</p>}

                {/* Inline journal */}
                {editJournal === t.id ? (
                  <div className="bg-amber-50 rounded-lg p-3 space-y-2">
                    <textarea rows={2} value={journalText} onChange={(e) => setJournalText(e.target.value)} placeholder="Your thoughts, lessons, what you'd do differently..." className="w-full px-3 py-2 rounded-lg border border-amber-200 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-amber-300 bg-white" />
                    <div className="flex gap-2">
                      <button onClick={() => saveJournal(t.id)} className="bg-slate-900 text-white px-3 py-1 rounded-lg text-xs font-semibold">Save</button>
                      <button onClick={() => setEditJournal(null)} className="text-slate-500 text-xs font-semibold">Cancel</button>
                    </div>
                  </div>
                ) : journals[t.id] ? (
                  <div className="bg-amber-50 rounded-lg px-3 py-2 flex items-start gap-2 cursor-pointer hover:bg-amber-100 transition-colors" onClick={() => { setEditJournal(t.id); setJournalText(journals[t.id]); }}>
                    <span className="text-xs">📝</span>
                    <span className="text-xs text-amber-800 flex-1">{journals[t.id]}</span>
                    <span className="text-xs text-amber-500">edit</span>
                  </div>
                ) : (
                  <button onClick={() => { setEditJournal(t.id); setJournalText(''); }} className="text-xs text-slate-400 hover:text-slate-600 font-medium">+ Add journal note</button>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
