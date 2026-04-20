import { create } from 'zustand';
import { Strategy, Position, Trade, TabId } from '@/lib/types';
import { INIT_STRATEGIES, INIT_POSITIONS, INIT_TRADES } from '@/lib/data';

interface AppState {
  tab: TabId;
  setTab: (tab: TabId) => void;

  strategies: Strategy[];
  setStrategies: (s: Strategy[]) => void;
  toggleStrategy: (id: string) => void;

  positions: Position[];
  setPositions: (p: Position[]) => void;
  closePosition: (id: string) => void;
  killAll: () => void;

  trades: Trade[];
  setTrades: (t: Trade[]) => void;

  // Connection status
  connected: boolean;
  setConnected: (v: boolean) => void;

  privacy: boolean;
  togglePrivacy: () => void;

  showKillModal: boolean;
  setShowKillModal: (v: boolean) => void;
}

export const useStore = create<AppState>((set) => ({
  tab: 'dash',
  setTab: (tab) => set({ tab }),

  // Start with mock data — replaced by API fetch on mount
  strategies: INIT_STRATEGIES,
  setStrategies: (strategies) => set({ strategies, connected: true }),
  toggleStrategy: (id) =>
    set((s) => ({
      strategies: s.strategies.map((st) =>
        st.id === id
          ? { ...st, status: st.status === 'running' ? 'paused' : 'running' }
          : st
      ),
    })),

  positions: INIT_POSITIONS,
  setPositions: (positions) => set({ positions }),
  closePosition: (id) =>
    set((s) => ({ positions: s.positions.filter((p) => p.id !== id) })),
  killAll: () =>
    set((s) => ({
      positions: [],
      strategies: s.strategies.map((st) => ({
        ...st,
        status: 'stopped' as const,
        deployed: 0,
      })),
      showKillModal: false,
    })),

  trades: INIT_TRADES,
  setTrades: (trades) => set({ trades }),

  connected: false,
  setConnected: (connected) => set({ connected }),

  privacy: false,
  togglePrivacy: () => set((s) => ({ privacy: !s.privacy })),

  showKillModal: false,
  setShowKillModal: (showKillModal) => set({ showKillModal }),
}));
