'use client';

import { useEffect, useRef, useCallback } from 'react';
import { useStore } from '@/store';
import { sccApi } from '@/lib/api';
import type { Strategy, Position, Trade } from '@/lib/types';

const POLL_INTERVAL = 5000; // 5 seconds

export function useSCCDashboard() {
  const { setStrategies, setPositions } = useStore();
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchDashboard = useCallback(async () => {
    try {
      const data = await sccApi.dashboard() as {
        strategies: Strategy[];
        positions: Position[];
        summary: Record<string, number>;
      };
      if (data.strategies) setStrategies(data.strategies);
      if (data.positions) setPositions(data.positions);
    } catch {
      // Silently fail on network errors — keep showing last known state
    }
  }, [setStrategies, setPositions]);

  useEffect(() => {
    fetchDashboard();
    intervalRef.current = setInterval(fetchDashboard, POLL_INTERVAL);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [fetchDashboard]);

  return { refresh: fetchDashboard };
}

export function useSCCTrades(filter?: string, strategy?: string) {
  const tradesRef = useRef<Trade[]>([]);
  const setRef = useRef<((t: Trade[]) => void) | null>(null);

  const fetchTrades = useCallback(async () => {
    try {
      const data = await sccApi.trades(filter, strategy) as Trade[];
      tradesRef.current = data;
      if (setRef.current) setRef.current(data);
    } catch {
      // keep last known
    }
  }, [filter, strategy]);

  return { fetchTrades, tradesRef };
}

export async function toggleStrategyAPI(id: string): Promise<{ status: string; enabled: boolean } | null> {
  try {
    const res = await fetch('/api/scc/toggle-strategy', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id }),
    });
    return res.json();
  } catch {
    return null;
  }
}

export async function killAllAPI(): Promise<boolean> {
  try {
    await sccApi.killAll();
    return true;
  } catch {
    return false;
  }
}
