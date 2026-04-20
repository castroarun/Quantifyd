'use client';

import { useStore } from '@/store';
import { killAllAPI } from '@/hooks/useSCC';

export function KillAllModal() {
  const { showKillModal, setShowKillModal, killAll, positions, strategies } = useStore();
  if (!showKillModal) return null;

  const running = strategies.filter((s) => s.status === 'running').length;

  const handleKill = async () => {
    killAll(); // Optimistic UI
    await killAllAPI(); // Sync to Flask
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-40 flex items-center justify-center z-50" onClick={() => setShowKillModal(false)}>
      <div className="bg-white rounded-2xl p-7 max-w-sm w-full shadow-2xl fu" onClick={(e) => e.stopPropagation()}>
        <h3 className="text-lg font-bold text-slate-900 mb-2">⚠️ Emergency Kill All</h3>
        <div className="bg-red-50 rounded-xl p-3 mb-5 text-sm text-red-700">
          Close <strong>{positions.length}</strong> positions, stop <strong>{running}</strong> strategies. Cannot undo.
        </div>
        <div className="flex gap-3">
          <button onClick={() => setShowKillModal(false)} className="flex-1 px-4 py-2 rounded-xl border border-slate-200 text-slate-600 font-semibold text-sm">Cancel</button>
          <button onClick={handleKill} className="flex-1 kill-btn text-white px-4 py-2 rounded-xl font-bold text-sm">Confirm</button>
        </div>
      </div>
    </div>
  );
}
