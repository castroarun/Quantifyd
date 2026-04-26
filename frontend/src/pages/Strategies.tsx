import { useEffect, useRef, useState } from 'react';
import styles from './Strategies.module.css';
import StrategyCard from '../components/Cards/StrategyCard';
import { IconBarChart, IconLayers } from '../components/Icons';
import { apiGet } from '../api/client';
import type { ORBState, NASState, StrangleState } from '../api/types';
import { formatInt, formatPnl, pnlClass } from '../utils/format';

// All 8 NAS sub-system endpoints so the Strategies card reflects the whole
// NAS footprint, not just OTM.
const NAS_SYSTEMS = [
  'nas',
  'nas-atm',
  'nas-atm2',
  'nas-atm4',
  'nas-916-otm',
  'nas-916-atm',
  'nas-916-atm2',
  'nas-916-atm4',
] as const;

type NasAgg = {
  activeLegs: number;
  dayPnl: number;       // realized + unrealized across all 8 systems
  totalPnl: number;     // all-time
  anyEnabled: boolean;
  anyLive: boolean;
};

// Polled /api/{key}/state returns p.ltp=None for some open legs (the 916
// executors write to separate ticker caches that _enrich_nas_positions_with_ltp
// doesn't see). The NAS page masks this by reading LTPs from /api/nas/stream
// SSE, so its per-card + top-line totals reconcile. This aggregator must do
// the same or the Strategies card under-reports unrealized on those legs,
// leaving only the realized loss visible (e.g. −Rs 66K after morning SLs).
function aggregateNas(
  states: (NASState | null)[],
  liveLegLtps: Record<string, number>,
): NasAgg {
  let activeLegs = 0;
  let dayPnl = 0;
  let totalPnl = 0;
  let anyEnabled = false;
  let anyLive = false;
  for (const s of states) {
    if (!s) continue;
    activeLegs += s.positions?.total_active ?? 0;
    // Realized (stats.today_pnl computed server-side from closed_today)
    dayPnl += (s.stats?.today_pnl as number | undefined) ?? 0;
    // Unrealized — prefer live SSE tick, fall back to polled p.ltp
    const legs = [...(s.positions?.ce ?? []), ...(s.positions?.pe ?? [])];
    for (const p of legs) {
      const entry = p.entry_price ?? p.entry_premium;
      const liveLtp = p.tradingsymbol ? liveLegLtps[p.tradingsymbol] : undefined;
      const ltp = liveLtp ?? p.ltp;
      const qty = p.qty ?? 0;
      if (entry != null && ltp != null && qty) {
        dayPnl += (entry - ltp) * qty;
      }
    }
    totalPnl += (s.stats?.total_pnl as number | undefined) ?? 0;
    if (s.config?.enabled) anyEnabled = true;
    if (s.config?.enabled && !s.config?.paper_trading_mode) anyLive = true;
  }
  return { activeLegs, dayPnl: Math.round(dayPnl * 100) / 100, totalPnl, anyEnabled, anyLive };
}

export default function Strategies() {
  const [orb, setOrb] = useState<ORBState | null>(null);
  const [nasStates, setNasStates] = useState<(NASState | null)[]>([]);
  const [strangle, setStrangle] = useState<StrangleState | null>(null);
  const [liveLegLtps, setLiveLegLtps] = useState<Record<string, number>>({});
  const [err, setErr] = useState<string | null>(null);
  const evtRef = useRef<EventSource | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const [o, st, ...ns] = await Promise.all([
          apiGet<ORBState>('/api/orb/state').catch(() => null),
          apiGet<StrangleState>('/api/strangle/state').catch(() => null),
          ...NAS_SYSTEMS.map((s) =>
            apiGet<NASState>(`/api/${s}/state`).catch(() => null),
          ),
        ]);
        if (cancelled) return;
        setOrb(o);
        setStrangle(st);
        setNasStates(ns);
      } catch (e) {
        if (!cancelled) setErr(e instanceof Error ? e.message : 'Failed to load');
      }
    };
    load();
    const id = setInterval(load, 5_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  // Subscribe to the NAS SSE tick stream so the aggregator has live LTPs for
  // every open leg — matches what the NAS dashboard already consumes.
  useEffect(() => {
    let cancelled = false;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    const open = () => {
      if (cancelled) return;
      const es = new EventSource('/api/nas/stream');
      evtRef.current = es;
      es.onmessage = (ev) => {
        if (cancelled) return;
        try {
          const d = JSON.parse(ev.data);
          if (d.type === 'tick' && d.legs) {
            const next: Record<string, number> = {};
            for (const [tsym, info] of Object.entries(d.legs)) {
              const ltp = (info as { ltp?: number }).ltp;
              if (typeof ltp === 'number') next[tsym] = ltp;
            }
            setLiveLegLtps(next);
          }
        } catch {
          /* ignore malformed payload */
        }
      };
      es.onerror = () => {
        if (cancelled) return;
        es.close();
        evtRef.current = null;
        reconnectTimer = setTimeout(open, 3000);
      };
    };
    open();
    return () => {
      cancelled = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (evtRef.current) {
        evtRef.current.close();
        evtRef.current = null;
      }
    };
  }, []);

  const orbOpen = orb?.open_positions?.length ?? 0;
  const orbClosed = orb?.today_closed?.length ?? 0;
  const orbPnl = orb?.today_pnl ?? 0;
  const orbEnabled = !!orb?.enabled;
  const orbLive = !!orb?.live_trading;

  const nasAgg = aggregateNas(nasStates, liveLegLtps);
  const nasOpen = nasAgg.activeLegs;
  const nasPnl = nasAgg.dayPnl;
  const nasEnabled = nasAgg.anyEnabled;
  const nasPaper = nasAgg.anyEnabled && !nasAgg.anyLive;

  // Strangle: aggregate across all 10 variants from a single payload.
  const strangleVariants = strangle?.variants ?? [];
  const strangleEnabled = strangleVariants.some((v) => v.enabled);
  const strangleOpen = strangleVariants.reduce(
    (s, v) => s + (v.open_positions ?? 0),
    0,
  );
  const strangleTodayPnl = strangleVariants.reduce(
    (s, v) => s + (v.today_pnl ?? 0),
    0,
  );
  const strangleVariantsCount = strangleVariants.length;

  return (
    <div className={styles.root}>
      <div className="page-title">Strategies</div>
      <div className="page-subtitle">
        Live automated strategies. Click a card to open its dashboard.
      </div>

      {err ? <div className={styles.error}>Failed to load: {err}</div> : null}

      <div className={styles.grid}>
        <StrategyCard
          to="/orb"
          icon={<IconBarChart size={15} />}
          title="ORB Cash"
          description="Cash intraday on 15 stocks. OR15 breakout with VWAP, RSI, CPR filters. Runs 9:14 AM to 3:20 PM."
          status={orbEnabled ? 'connected' : 'disconnected'}
          statusLabel={
            orbEnabled ? (orbLive ? 'Live trading' : 'Paper trading') : 'Disabled'
          }
          dayPnl={orbPnl}
          stats={[
            { label: 'Open', value: formatInt(orbOpen) },
            { label: 'Closed today', value: formatInt(orbClosed) },
            { label: 'Universe', value: formatInt(orb?.universe?.length) },
          ]}
        />
        <StrategyCard
          to="/strangle"
          icon={<IconLayers size={15} />}
          title="ORB Index"
          description="ORB break on Nifty index → delta-skewed short strangle (PE -0.22, CE +0.10). 10 variants across 5/15/30/45/60-min OR windows + RSI/calm/CPR-against filters."
          status={strangleEnabled ? 'connected' : 'disconnected'}
          statusLabel={strangleEnabled ? 'Paper trading' : 'Disabled'}
          dayPnl={strangleTodayPnl}
          stats={[
            { label: 'Open positions', value: formatInt(strangleOpen) },
            { label: 'Variants', value: formatInt(strangleVariantsCount) },
            { label: 'Spot', value: formatInt(strangle?.spot_ltp ?? null) },
          ]}
        />
        <StrategyCard
          to="/nas"
          icon={<IconLayers size={15} />}
          title="NAS options"
          description="Nifty ATR squeeze + 9:16 entry. Eight variants running in parallel across OTM, ATM, ATM 2.0 and ATM V4."
          status={nasEnabled ? 'connected' : 'disconnected'}
          statusLabel={nasEnabled ? (nasPaper ? 'Paper trading' : 'Live trading') : 'Disabled'}
          dayPnl={nasPnl}
          stats={[
            { label: 'Active legs', value: formatInt(nasOpen) },
            { label: 'Systems', value: '8' },
            {
              label: 'Total P&L',
              value: (
                <span className={pnlClass(nasAgg.totalPnl)}>
                  {formatPnl(nasAgg.totalPnl)}
                </span>
              ),
            },
          ]}
        />
      </div>

      <div className={styles.section}>
        <div className="section-title">Today at a glance</div>
        <div className={styles.miniGrid}>
          <MiniStat
            label="ORB Cash day P&L"
            value={formatPnl(orbPnl)}
            cls={pnlClass(orbPnl)}
          />
          <MiniStat
            label="ORB Index day P&L"
            value={formatPnl(strangleTodayPnl)}
            cls={pnlClass(strangleTodayPnl)}
          />
          <MiniStat label="NAS day P&L" value={formatPnl(nasPnl)} cls={pnlClass(nasPnl)} />
          <MiniStat label="ORB Cash open" value={formatInt(orbOpen)} />
          <MiniStat label="ORB Index open" value={formatInt(strangleOpen)} />
          <MiniStat label="NAS open legs" value={formatInt(nasOpen)} />
        </div>
      </div>
    </div>
  );
}

function MiniStat({
  label,
  value,
  cls,
}: {
  label: string;
  value: string;
  cls?: string;
}) {
  return (
    <div className={styles.mini}>
      <div className={styles.miniLabel}>{label}</div>
      <div className={`${styles.miniValue} ${cls ?? ''}`}>{value}</div>
    </div>
  );
}
