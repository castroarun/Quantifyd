import { useEffect, useState } from 'react';
import styles from './Strategies.module.css';
import StrategyCard from '../components/Cards/StrategyCard';
import { IconBarChart, IconLayers } from '../components/Icons';
import { apiGet } from '../api/client';
import type { ORBState, NASState } from '../api/types';
import { formatInt, formatPnl, pnlClass } from '../utils/format';

export default function Strategies() {
  const [orb, setOrb] = useState<ORBState | null>(null);
  const [nas, setNas] = useState<NASState | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const [o, n] = await Promise.all([
          apiGet<ORBState>('/api/orb/state').catch(() => null),
          apiGet<NASState>('/api/nas/state').catch(() => null),
        ]);
        if (cancelled) return;
        setOrb(o);
        setNas(n);
      } catch (e) {
        if (!cancelled) setErr(e instanceof Error ? e.message : 'Failed to load');
      }
    };
    load();
    const id = setInterval(load, 15_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  const orbOpen = orb?.open_positions?.length ?? 0;
  const orbClosed = orb?.today_closed?.length ?? 0;
  const orbPnl = orb?.today_pnl ?? 0;
  const orbEnabled = !!orb?.enabled;
  const orbLive = !!orb?.live_trading;

  const nasOpen = nas?.positions?.total_active ?? 0;
  const nasPnl = (nas?.stats?.today_pnl as number | undefined) ?? 0;
  const nasEnabled = !!nas?.config?.enabled;
  const nasPaper = !!nas?.config?.paper_trading_mode;

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
          title="Opening range breakout"
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
                <span className={pnlClass(nas?.stats?.total_pnl as number | undefined)}>
                  {formatPnl(nas?.stats?.total_pnl as number | undefined)}
                </span>
              ),
            },
          ]}
        />
      </div>

      <div className={styles.section}>
        <div className="section-title">Today at a glance</div>
        <div className={styles.miniGrid}>
          <MiniStat label="ORB day P&L" value={formatPnl(orbPnl)} cls={pnlClass(orbPnl)} />
          <MiniStat label="NAS day P&L" value={formatPnl(nasPnl)} cls={pnlClass(nasPnl)} />
          <MiniStat label="ORB open" value={formatInt(orbOpen)} />
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
