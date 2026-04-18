import { useEffect, useState } from 'react';
import styles from './Nas.module.css';
import { apiGet } from '../api/client';
import { useSSE } from '../api/sse';
import type { NASState, NASPosition, NASTickPayload } from '../api/types';
import StatusDot from '../components/StatusDot/StatusDot';
import Chip from '../components/Chip/Chip';
import { formatInt, formatNumber, formatPnl, pnlClass } from '../utils/format';

/* ---------- system definitions ---------- */

interface SystemDef {
  id: string;
  label: string;
  subtitle: string;
  stateUrl: string;
  streamUrl?: string; // not all 916 variants have streams, but we'll still try
}

const SQUEEZE_SYSTEMS: SystemDef[] = [
  {
    id: 'nas',
    label: 'Squeeze · OTM',
    subtitle: 'Original 1.5 ATR OTM strangle',
    stateUrl: '/api/nas/state',
    streamUrl: '/api/nas/ticker/stream',
  },
  {
    id: 'nas-atm',
    label: 'Squeeze · ATM',
    subtitle: 'ATM strangle with alternating adjustment',
    stateUrl: '/api/nas-atm/state',
    streamUrl: '/api/nas-atm/ticker/stream',
  },
  {
    id: 'nas-atm2',
    label: 'Squeeze · ATM 2.0',
    subtitle: 'Cascading ATM, 4-24 premium bounds',
    stateUrl: '/api/nas-atm2/state',
    streamUrl: '/api/nas-atm2/ticker/stream',
  },
  {
    id: 'nas-atm4',
    label: 'Squeeze · ATM V4',
    subtitle: 'ATM V4 with cross-leg topup',
    stateUrl: '/api/nas-atm4/state',
    streamUrl: '/api/nas-atm4/ticker/stream',
  },
];

const ENTRY_916_SYSTEMS: SystemDef[] = [
  {
    id: 'nas-916-otm',
    label: '9:16 · OTM',
    subtitle: 'Time-based 9:16 entry, OTM legs',
    stateUrl: '/api/nas-916-otm/state',
    streamUrl: '/api/nas-916-otm/ticker/stream',
  },
  {
    id: 'nas-916-atm',
    label: '9:16 · ATM',
    subtitle: 'Time-based 9:16 entry, ATM legs',
    stateUrl: '/api/nas-916-atm/state',
    streamUrl: '/api/nas-916-atm/ticker/stream',
  },
  {
    id: 'nas-916-atm2',
    label: '9:16 · ATM 2.0',
    subtitle: '9:16 entry, cascading ATM 2.0',
    stateUrl: '/api/nas-916-atm2/state',
    streamUrl: '/api/nas-916-atm2/ticker/stream',
  },
  {
    id: 'nas-916-atm4',
    label: '9:16 · ATM V4',
    subtitle: '9:16 entry, ATM V4 cross-leg',
    stateUrl: '/api/nas-916-atm4/state',
    streamUrl: '/api/nas-916-atm4/ticker/stream',
  },
];

/* ---------- page ---------- */

export default function Nas() {
  return (
    <div className={styles.root}>
      <div className="page-title">NAS options</div>
      <div className="page-subtitle">
        Eight Nifty options systems running in parallel. ATR squeeze entries on the
        left, time-based 9:16 entries on the right.
      </div>

      <div className={styles.columns}>
        <div className={styles.col}>
          <div className={styles.colHead}>
            <div className="section-title">ATR squeeze</div>
            <Chip>4 systems</Chip>
          </div>
          <div className={styles.panelList}>
            {SQUEEZE_SYSTEMS.map((s) => (
              <SystemPanel key={s.id} def={s} />
            ))}
          </div>
        </div>

        <div className={styles.divider} aria-hidden />

        <div className={styles.col}>
          <div className={styles.colHead}>
            <div className="section-title">9:16 entry</div>
            <Chip>4 systems</Chip>
          </div>
          <div className={styles.panelList}>
            {ENTRY_916_SYSTEMS.map((s) => (
              <SystemPanel key={s.id} def={s} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

/* ---------- panel per system ---------- */

function SystemPanel({ def }: { def: SystemDef }) {
  const [state, setState] = useState<NASState | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [ticks, setTicks] = useState<Record<string, number>>({}); // tradingsymbol → ltp
  const [streamAlive, setStreamAlive] = useState(false);
  const [spot, setSpot] = useState<number | null>(null);

  // Poll state every 10s
  useEffect(() => {
    let cancelled = false;
    const load = () => {
      apiGet<NASState>(def.stateUrl)
        .then((s) => {
          if (!cancelled) {
            setState(s);
            setErr(null);
          }
        })
        .catch((e) => {
          if (!cancelled) setErr(e instanceof Error ? e.message : 'Load failed');
        });
    };
    load();
    const id = setInterval(load, 10_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [def.stateUrl]);

  // SSE for live tick updates
  useSSE<NASTickPayload>(def.streamUrl ?? null, (payload) => {
    if (payload.type === 'offline') {
      setStreamAlive(false);
      return;
    }
    setStreamAlive(true);
    if (typeof payload.spot === 'number') setSpot(payload.spot);
    if (payload.legs) {
      const next: Record<string, number> = {};
      for (const [tsym, info] of Object.entries(payload.legs)) {
        next[tsym] = info.ltp;
      }
      setTicks((prev) => ({ ...prev, ...next }));
    }
  });

  const positions: NASPosition[] = [
    ...(state?.positions?.ce ?? []),
    ...(state?.positions?.pe ?? []),
  ];

  // Merge live ticks into open positions for display P&L
  const enriched = positions.map((p) => {
    const live = p.tradingsymbol ? ticks[p.tradingsymbol] : undefined;
    const ltp = live ?? p.ltp;
    let pnl = p.pnl_inr;
    if (live !== undefined && p.entry_premium !== undefined && p.qty) {
      // Short premium: profit = (entry − ltp) × qty
      pnl = Math.round((p.entry_premium - live) * p.qty * 100) / 100;
    }
    return { ...p, ltp, pnl_inr: pnl };
  });

  const totalPnl = enriched.reduce((acc, p) => acc + (p.pnl_inr ?? 0), 0);
  const dayPnl = (state?.stats?.today_pnl as number | undefined) ?? totalPnl;
  const reentries = (state?.stats?.total_reentries as number | undefined) ?? 0;

  const enabled = !!state?.config?.enabled;
  const paper = !!state?.config?.paper_trading_mode;

  return (
    <div className={styles.panel}>
      <div className={styles.panelHead}>
        <div>
          <div className={styles.panelTitle}>{def.label}</div>
          <div className={styles.panelSub}>{def.subtitle}</div>
        </div>
        <div className={styles.panelStatus}>
          <StatusDot
            kind={enabled ? (streamAlive ? 'connected' : 'warning') : 'disconnected'}
            label={
              !enabled
                ? 'Disabled'
                : paper
                ? streamAlive
                  ? 'Paper · live'
                  : 'Paper'
                : streamAlive
                ? 'Live'
                : 'Standby'
            }
          />
        </div>
      </div>

      <div className={styles.metricsRow}>
        <MiniMetric
          label="Active"
          value={formatInt(state?.positions?.total_active ?? 0)}
        />
        <MiniMetric
          label="Re-entries"
          value={formatInt(reentries)}
        />
        <MiniMetric
          label="Day P&L"
          value={
            <span className={pnlClass(dayPnl)}>{formatPnl(dayPnl)}</span>
          }
        />
      </div>

      {spot !== null ? (
        <div className={styles.spotRow}>
          <span className={styles.spotLabel}>Nifty spot</span>
          <span className={styles.spotValue}>{formatNumber(spot)}</span>
        </div>
      ) : null}

      <div className={styles.legs}>
        {enriched.length === 0 ? (
          <div className={styles.noLegs}>No open legs</div>
        ) : (
          enriched.map((p, i) => <LegRow key={(p.tradingsymbol ?? '') + i} leg={p} />)
        )}
      </div>

      {err ? <div className={styles.errRow}>{err}</div> : null}
    </div>
  );
}

function MiniMetric({
  label,
  value,
}: {
  label: string;
  value: React.ReactNode;
}) {
  return (
    <div className={styles.mini}>
      <div className={styles.miniLabel}>{label}</div>
      <div className={styles.miniValue}>{value}</div>
    </div>
  );
}

function LegRow({ leg }: { leg: NASPosition }) {
  const tsym = leg.tradingsymbol ?? '—';
  const entry = leg.entry_premium;
  const ltp = leg.ltp;
  const pnl = leg.pnl_inr;

  return (
    <div className={styles.leg}>
      <div className={styles.legMain}>
        <span className={styles.legSide}>{leg.leg}</span>
        <span className={styles.legSym}>{tsym}</span>
      </div>
      <div className={styles.legNums}>
        <span className={styles.legSmall}>
          {entry !== undefined ? formatNumber(entry) : '—'}
        </span>
        <span className={styles.legArrow}>→</span>
        <span className={styles.legSmall}>
          {ltp !== undefined ? formatNumber(ltp) : '—'}
        </span>
        <span className={pnlClass(pnl)} style={{ fontSize: 'var(--text-xs)' }}>
          {formatPnl(pnl)}
        </span>
      </div>
    </div>
  );
}

