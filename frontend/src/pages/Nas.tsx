import { createContext, useContext, useEffect, useMemo, useRef, useState } from 'react';
import styles from './Nas.module.css';
import { apiGet } from '../api/client';
import type { NASState, NASPosition } from '../api/types';
import StatusDot from '../components/StatusDot/StatusDot';
import Chip from '../components/Chip/Chip';
import MetricCard from '../components/Cards/MetricCard';
import {
  formatInt,
  formatNumber,
  formatPnl,
  formatPct,
  formatRs,
  pnlClass,
} from '../utils/format';

/* ---------- live ticks context ---------- */

interface LiveTicks {
  spot: number | null;
  legs: Record<string, number>; // tradingsymbol → ltp
  connected: boolean;
}
const LiveTicksContext = createContext<LiveTicks>({ spot: null, legs: {}, connected: false });
const useLiveTicks = () => useContext(LiveTicksContext);

/* ---------- system definitions ---------- */

interface SystemDef {
  id: string;
  key: string; // used for API paths /api/{key}/...
  label: string;
  subtitle: string;
  rules: string;
  configNote: string;
  group: 'squeeze' | '916';
}

const SQUEEZE_SYSTEMS: SystemDef[] = [
  {
    id: 'nas',
    key: 'nas',
    label: 'Squeeze · OTM',
    subtitle: 'Original 1.5 ATR OTM strangle',
    rules:
      'Entry: ATR(14) < SMA(ATR,50) on 5-min → SELL OTM CE+PE at approx Rs 20. Adj: Cross-leg imbalance >= 2x → ROLL_OUT or ROLL_IN alternating. Exit: Time 14:45, EOD 15:15.',
    configNote: 'OTM: 10L | Premium Rs 20-24',
    group: 'squeeze',
  },
  {
    id: 'nas-atm',
    key: 'nas-atm',
    label: 'Squeeze · ATM',
    subtitle: 'ATM strangle with alternating adjustment',
    rules:
      'Entry: ATR squeeze → SELL ATM CE+PE, SL = entry x 1.3 (30%). 1st SL: Close stopped leg. Naked leg: ST(7,2) exit. EOD 15:15.',
    configNote: 'ATM: 5L | 30% SL',
    group: 'squeeze',
  },
  {
    id: 'nas-atm2',
    key: 'nas-atm2',
    label: 'Squeeze · ATM 2.0',
    subtitle: 'Cascading ATM, 4-24 premium bounds',
    rules:
      'Entry: ATR squeeze → SELL ATM CE+PE, SL = 1.3x. Any SL closes BOTH legs and re-enters a new ATM strangle. Max 5 re-entries. EOD 15:15.',
    configNote: 'ATM 2.0: 5L | 1.3x SL | 5 re-entries',
    group: 'squeeze',
  },
  {
    id: 'nas-atm4',
    key: 'nas-atm4',
    label: 'Squeeze · ATM V4',
    subtitle: 'ATM V4 with cross-leg topup',
    rules:
      'Entry: ATR squeeze → SELL ATM, SL = 1.3x. 1st SL: Roll stopped leg to match surviving leg CMP (both re-get 30% SLs). 2nd SL: Close stopped leg, naked surviving leg uses ST(7,2) exit. EOD 15:15.',
    configNote: 'ATM V4: 5L | 1.3x SL | Roll-to-match',
    group: 'squeeze',
  },
];

const ENTRY_916_SYSTEMS: SystemDef[] = [
  {
    id: 'nas-916-otm',
    key: 'nas-916-otm',
    label: '9:16 · OTM',
    subtitle: 'Time-based 9:16 entry, OTM legs',
    rules:
      'Entry: Auto-enter at 9:16 AM. SELL OTM CE+PE at approx Rs 20. Adj: Cross-leg imbalance >= 2x → ROLL_OUT or ROLL_IN alternating. Exit: Time 14:45, EOD 15:15.',
    configNote: '916 OTM: 10L | Premium Rs 20-24',
    group: '916',
  },
  {
    id: 'nas-916-atm',
    key: 'nas-916-atm',
    label: '9:16 · ATM',
    subtitle: 'Time-based 9:16 entry, ATM legs',
    rules:
      'Entry: Auto-enter at 9:16 AM. SELL ATM CE+PE, SL = entry x 1.3 (30%). 1st SL: Close stopped leg. Naked leg: ST(7,2) exit. EOD 15:15.',
    configNote: '916 ATM: 5L | 30% SL',
    group: '916',
  },
  {
    id: 'nas-916-atm2',
    key: 'nas-916-atm2',
    label: '9:16 · ATM 2.0',
    subtitle: '9:16 entry, cascading ATM 2.0',
    rules:
      'Entry: Auto-enter at 9:16 AM. SELL ATM CE+PE, SL = 1.3x. Any SL closes BOTH legs and re-enters a new ATM strangle. Max 5 re-entries. EOD 15:15.',
    configNote: '916 ATM 2.0: 5L | 1.3x SL | 5 re-entries',
    group: '916',
  },
  {
    id: 'nas-916-atm4',
    key: 'nas-916-atm4',
    label: '9:16 · ATM V4',
    subtitle: '9:16 entry, ATM V4 cross-leg',
    rules:
      'Entry: Auto-enter at 9:16 AM. SELL ATM, SL = 1.3x. 1st SL: Roll stopped leg to match surviving leg CMP. 2nd SL: Close stopped leg, naked surviving leg uses ST(7,2) exit. EOD 15:15.',
    configNote: '916 ATM V4: 5L | 1.3x SL | Roll-to-match',
    group: '916',
  },
];

const ALL_SYSTEMS: SystemDef[] = [...SQUEEZE_SYSTEMS, ...ENTRY_916_SYSTEMS];

/* ---------- page ---------- */

interface SystemStateRecord {
  state: NASState | null;
  err: string | null;
}

export default function Nas() {
  const [states, setStates] = useState<Record<string, SystemStateRecord>>({});
  const [toast, setToast] = useState<string | null>(null);
  const [liveTicks, setLiveTicks] = useState<LiveTicks>({
    spot: null,
    legs: {},
    connected: false,
  });
  const evtRef = useRef<EventSource | null>(null);

  function updateState(id: string, rec: SystemStateRecord) {
    setStates((prev) => ({ ...prev, [id]: rec }));
  }

  // One SSE connection for the entire dashboard — pushes spot + all option
  // leg LTPs across 8 systems in a single stream. Reconnects on error.
  useEffect(() => {
    let cancelled = false;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    const open = () => {
      if (cancelled) return;
      const es = new EventSource('/api/nas/stream');
      evtRef.current = es;
      es.onopen = () => {
        if (!cancelled) setLiveTicks((prev) => ({ ...prev, connected: true }));
      };
      es.onmessage = (ev) => {
        if (cancelled) return;
        try {
          const d = JSON.parse(ev.data);
          if (d.type === 'tick') {
            const legs: Record<string, number> = {};
            for (const [tsym, info] of Object.entries(d.legs || {})) {
              const ltp = (info as { ltp?: number }).ltp;
              if (typeof ltp === 'number') legs[tsym] = ltp;
            }
            setLiveTicks({
              spot: typeof d.spot === 'number' && d.spot > 0 ? d.spot : null,
              legs,
              connected: true,
            });
          } else if (d.type === 'offline') {
            setLiveTicks((prev) => ({ ...prev, connected: false }));
          }
        } catch {
          /* ignore malformed payload */
        }
      };
      es.onerror = () => {
        if (cancelled) return;
        setLiveTicks((prev) => ({ ...prev, connected: false }));
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

  const squeezeSystems = SQUEEZE_SYSTEMS.map((s) => states[s.id]?.state).filter(
    Boolean,
  ) as NASState[];
  const nineSixteenSystems = ENTRY_916_SYSTEMS.map((s) => states[s.id]?.state).filter(
    Boolean,
  ) as NASState[];

  // Pick the first squeeze state that has ATR/squeeze data for the shared header.
  const headerState: NASState | null = useMemo(() => {
    for (const s of SQUEEZE_SYSTEMS) {
      const rec = states[s.id]?.state;
      if (rec?.state && typeof rec.state.atr_value === 'number') return rec;
    }
    return states[SQUEEZE_SYSTEMS[0].id]?.state ?? null;
  }, [states]);

  // Per-system day P&L = DB-persisted today_pnl (closed trades) + live open-leg P&L.
  // Open-leg P&L = sum of (entry_price - ltp) * qty across CE + PE legs (we short).
  // Prefer live SSE tick LTP over polled state LTP when available.
  const liveSystemPnl = (s: NASState | undefined | null): number => {
    if (!s) return 0;
    const persisted = (s.stats?.today_pnl as number | undefined) ?? 0;
    const legs = [...(s.positions?.ce ?? []), ...(s.positions?.pe ?? [])];
    const open = legs.reduce((acc, p) => {
      const entry = p.entry_price ?? p.entry_premium;
      const liveLtp = p.tradingsymbol ? liveTicks.legs[p.tradingsymbol] : undefined;
      const ltp = liveLtp ?? p.ltp;
      const qty = p.qty ?? 0;
      if (entry == null || ltp == null || !qty) return acc;
      return acc + (entry - ltp) * qty;
    }, 0);
    return persisted + open;
  };
  const squeezeDayPnl = squeezeSystems.reduce((acc, s) => acc + liveSystemPnl(s), 0);
  const nineSixteenDayPnl = nineSixteenSystems.reduce((acc, s) => acc + liveSystemPnl(s), 0);

  const core = headerState?.state ?? {};
  const isSqueezing = !!core.is_squeezing;
  // Prefer live SSE spot over polled state spot when the stream is up.
  const pollSpot = core.spot_price as number | undefined;
  const spot = liveTicks.spot ?? pollSpot;
  const atr = core.atr_value as number | undefined;
  const atrMa = core.atr_ma as number | undefined;

  // Market hours check (IST 09:15-15:30)
  const nowIst = new Date();
  const mins = nowIst.getHours() * 60 + nowIst.getMinutes();
  const marketOpen = mins >= 9 * 60 + 15 && mins <= 15 * 60 + 30;
  const hasData = atr !== undefined && atrMa !== undefined;

  // Squeeze dot kind: green if active + has data, red if no squeeze + has data, grey if no data/market closed
  let squeezeDotKind: 'connected' | 'disconnected' | 'warning' = 'warning';
  if (!hasData || !marketOpen) {
    squeezeDotKind = 'warning'; // grey
  } else if (isSqueezing) {
    squeezeDotKind = 'connected'; // green
  } else {
    squeezeDotKind = 'disconnected'; // red
  }

  // Available margin (from any state's margin field, served by backend)
  const margin = headerState?.margin as { available?: number } | undefined;
  const availableMargin = margin?.available;

  function showToast(msg: string) {
    setToast(msg);
    setTimeout(() => setToast(null), 2500);
  }

  /* ---------- whats next schedule ---------- */

  const nextEvents = useMemo(() => buildNextEvents(states), [states]);

  return (
    <LiveTicksContext.Provider value={liveTicks}>
    <div className={styles.root}>
      <div className="page-title">NAS options</div>
      <div className="page-subtitle">
        Eight Nifty options systems running in parallel. ATR squeeze entries on the
        left, time-based 9:16 entries on the right.
      </div>

      {toast ? <div className={styles.toast}>{toast}</div> : null}

      {/* Shared ATR squeeze header */}
      <div className={styles.headerMetrics}>
        <MetricCard
          label="ATR squeeze"
          value={
            <span className={styles.squeezeValue}>
              <StatusDot kind={squeezeDotKind} className={styles.squeezeDot} />
              <span>
                {!hasData || !marketOpen
                  ? '—'
                  : isSqueezing
                  ? 'Squeeze'
                  : 'Normal'}
              </span>
            </span>
          }
          hint={
            hasData
              ? `ATR ${formatNumber(atr)} / MA ${formatNumber(atrMa)}`
              : 'ATR(14) vs SMA(ATR,50)'
          }
        />
        <MetricCard
          label="Nifty spot"
          value={spot !== undefined ? formatNumber(spot) : '—'}
          hint="Live index price"
        />
        <MetricCard
          label="Available margin"
          value={availableMargin !== undefined ? formatRs(availableMargin) : '—'}
          hint="Cash for MIS orders"
        />
        <MetricCard
          label="Squeeze day P&L"
          value={
            <span className={pnlClass(squeezeDayPnl)}>
              {formatPnl(squeezeDayPnl)}
            </span>
          }
          hint="OTM + ATM + ATM 2.0 + ATM V4"
        />
        <MetricCard
          label="9:16 day P&L"
          value={
            <span className={pnlClass(nineSixteenDayPnl)}>
              {formatPnl(nineSixteenDayPnl)}
            </span>
          }
          hint="All four 9:16 systems"
        />
      </div>

      <div className={styles.columns}>
        <div className={styles.col}>
          <div className={styles.colHead}>
            <div className="section-title">ATR squeeze</div>
            <Chip>4 systems</Chip>
          </div>
          <div className={styles.panelList}>
            {SQUEEZE_SYSTEMS.map((s) => (
              <SystemPanel
                key={s.id}
                def={s}
                onStateChange={(rec) => updateState(s.id, rec)}
                onToast={showToast}
              />
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
              <SystemPanel
                key={s.id}
                def={s}
                onStateChange={(rec) => updateState(s.id, rec)}
                onToast={showToast}
              />
            ))}
          </div>
        </div>
      </div>

      {/* What's next */}
      <section className={styles.sectionBlock}>
        <div className={styles.sectionHead}>
          <div className="section-title">What's next</div>
          <Chip>{nextEvents.length} events</Chip>
        </div>
        <div className={styles.eventsTable}>
          <div className={styles.eventsHead}>
            <div>System</div>
            <div>Event</div>
            <div>Scheduled</div>
            <div>Status</div>
            <div className={styles.eventsHeadRight}>In</div>
          </div>
          {nextEvents.map((ev, i) => (
            <div key={`${ev.system}-${ev.event}-${i}`} className={styles.eventsRow}>
              <div className={styles.eventsSystem}>{ev.system}</div>
              <div>{ev.event}</div>
              <div className={styles.eventsTime}>{ev.scheduled}</div>
              <div>
                <span className={`${styles.status} ${styles[`status_${ev.tone}`]}`}>
                  {ev.status}
                </span>
              </div>
              <div className={styles.eventsHeadRight}>
                <span className={styles.mute}>{ev.relative}</span>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Config footer */}
      <section className={styles.sectionBlock}>
        <div className={styles.configFooter}>
          <div className={styles.configTitle}>Config overview</div>
          <div className={styles.configList}>
            {ALL_SYSTEMS.map((s) => (
              <div key={s.id} className={styles.configItem}>
                <span className={styles.configSysLabel}>{s.label}</span>
                <span className={styles.configSysNote}>{s.configNote}</span>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
    </LiveTicksContext.Provider>
  );
}

/* ---------- panel per system ---------- */

interface PanelProps {
  def: SystemDef;
  onStateChange: (rec: SystemStateRecord) => void;
  onToast: (msg: string) => void;
}

function SystemPanel({ def, onStateChange, onToast }: PanelProps) {
  const [state, setState] = useState<NASState | null>(null);
  const [err, setErr] = useState<string | null>(null);
  // Live tick prices from the parent SSE stream — keyed by tradingsymbol.
  const liveTicks = useLiveTicks();
  const ticks = liveTicks.legs;
  const streamAlive = liveTicks.connected;

  const stateUrl = `/api/${def.key}/state`;

  // Poll state every 5s as a safety net — positions, config, stats. Live
  // leg LTPs come from the shared SSE stream in the parent component.
  useEffect(() => {
    let cancelled = false;
    const load = () => {
      apiGet<NASState>(stateUrl)
        .then((s) => {
          if (cancelled) return;
          setState(s);
          setErr(null);
          onStateChange({ state: s, err: null });
        })
        .catch((e) => {
          if (cancelled) return;
          const msg = e instanceof Error ? e.message : 'Load failed';
          setErr(msg);
          onStateChange({ state: null, err: msg });
        });
    };
    load();
    const id = setInterval(load, 5_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stateUrl]);

  const positions: NASPosition[] = [
    ...(state?.positions?.ce ?? []),
    ...(state?.positions?.pe ?? []),
  ];

  const enriched = positions.map((p) => {
    const live = p.tradingsymbol ? ticks[p.tradingsymbol] : undefined;
    const ltp = live ?? p.ltp;
    const entry = p.entry_price ?? p.entry_premium;
    let pnl = p.pnl_inr;
    if (live !== undefined && entry !== undefined && p.qty) {
      // NAS systems short options → profit when LTP drops below entry
      pnl = Math.round((entry - live) * p.qty * 100) / 100;
    }
    return { ...p, ltp, pnl_inr: pnl };
  });

  const totalPnl = enriched.reduce((acc, p) => acc + (p.pnl_inr ?? 0), 0);
  const dayPnl = (state?.stats?.today_pnl as number | undefined) ?? totalPnl;
  const reentries = (state?.stats?.total_reentries as number | undefined) ?? 0;
  const winRate = state?.stats?.win_rate as number | undefined;
  const pf = state?.stats?.profit_factor as number | undefined;
  const slHits = state?.stats?.sl_hits_today as number | undefined;

  const enabled = !!state?.config?.enabled;
  const paper = !!state?.config?.paper_trading_mode;

  // action buttons removed — header is title/subtitle only, consistent across pages

  return (
    <div className={styles.panel}>
      <div className={styles.panelHead}>
        <div className={styles.panelHeadLeft}>
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
        <MiniMetric label="Re-entries" value={formatInt(reentries)} />
        <MiniMetric
          label="Day P&L"
          value={
            <span className={pnlClass(dayPnl)}>{formatPnl(dayPnl)}</span>
          }
        />
      </div>

      <div className={styles.legs}>
        {enriched.length === 0 ? (
          <div className={styles.noLegs}>No open legs</div>
        ) : (
          enriched.map((p, i) => <LegRow key={(p.tradingsymbol ?? '') + i} leg={p} />)
        )}
      </div>

      <div className={styles.statsMiniRow}>
        <MiniMetric
          label="Win rate (all-time)"
          value={winRate !== undefined ? formatPct(winRate, 1) : '—'}
        />
        <MiniMetric
          label="Profit factor"
          value={pf !== undefined ? formatNumber(pf, 2) : '—'}
        />
        <MiniMetric label="SL hits today" value={formatInt(slHits ?? 0)} />
      </div>

      <details className={styles.rules}>
        <summary className={styles.rulesSummary}>Rules</summary>
        <div className={styles.rulesBody}>{def.rules}</div>
      </details>

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
  const entry = leg.entry_price ?? leg.entry_premium;
  const ltp = leg.ltp ?? leg.exit_price;
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

/* ---------- next events helper ---------- */

interface NextEvent {
  system: string;
  event: string;
  scheduled: string;
  status: string;
  tone: 'pos' | 'neg' | 'neutral';
  relative: string;
  sortKey: number;
}

function minutesFromNowIST(hhmm: string): number {
  // IST current time
  const now = new Date();
  const parts = new Intl.DateTimeFormat('en-IN', {
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
    timeZone: 'Asia/Kolkata',
  })
    .format(now)
    .split(':');
  const nowMin = parseInt(parts[0], 10) * 60 + parseInt(parts[1], 10);
  const [h, m] = hhmm.split(':').map((x) => parseInt(x, 10));
  const targetMin = h * 60 + m;
  return targetMin - nowMin;
}

function relativeLabel(diffMin: number): string {
  if (diffMin < 0) return 'passed';
  if (diffMin === 0) return 'now';
  const h = Math.floor(diffMin / 60);
  const m = diffMin % 60;
  if (h === 0) return `in ${m}m`;
  return `in ${h}h ${m}m`;
}

function buildNextEvents(
  states: Record<string, SystemStateRecord>,
): NextEvent[] {
  const events: NextEvent[] = [];

  for (const def of ALL_SYSTEMS) {
    const rec = states[def.id];
    const enabled = !!rec?.state?.config?.enabled;
    const cfg = rec?.state?.config ?? {};

    // Entry event — for 9:16 systems only
    if (def.group === '916') {
      const diff = minutesFromNowIST('09:16');
      events.push({
        system: def.label,
        event: 'Auto-entry at 9:16',
        scheduled: '09:16',
        status: diff < 0 ? 'Done' : enabled ? 'Pending' : 'Disabled',
        tone: diff < 0 ? 'neutral' : enabled ? 'pos' : 'neutral',
        relative: relativeLabel(diff),
        sortKey: diff < 0 ? 9999 : diff,
      });
    } else {
      // Squeeze entry — continuous during entry window
      const startHHMM =
        (cfg.entry_start_time as string | undefined) ?? '09:30';
      const endHHMM =
        (cfg.entry_end_time as string | undefined) ?? '14:30';
      const startDiff = minutesFromNowIST(startHHMM);
      const endDiff = minutesFromNowIST(endHHMM);
      const active = startDiff <= 0 && endDiff > 0;
      events.push({
        system: def.label,
        event: 'Re-enter on squeeze',
        scheduled: `${startHHMM}-${endHHMM}`,
        status: !enabled ? 'Disabled' : active ? 'Active' : endDiff <= 0 ? 'Done' : 'Pending',
        tone: !enabled ? 'neutral' : active ? 'pos' : 'neutral',
        relative: active ? 'active' : startDiff > 0 ? relativeLabel(startDiff) : 'passed',
        sortKey: active ? -1 : startDiff > 0 ? startDiff : 9999,
      });
    }

    // Time exit (14:45) for OTM-flavoured systems
    if (def.id === 'nas' || def.id === 'nas-916-otm') {
      const t = (cfg.time_exit as string | undefined) ?? '14:45';
      const diff = minutesFromNowIST(t);
      events.push({
        system: def.label,
        event: 'Time exit',
        scheduled: t,
        status: diff < 0 ? 'Done' : 'Pending',
        tone: diff < 0 ? 'neutral' : 'pos',
        relative: relativeLabel(diff),
        sortKey: diff < 0 ? 9999 : diff,
      });
    }

    // EOD squareoff
    const eod = (cfg.eod_squareoff_time as string | undefined) ?? '15:15';
    const diff = minutesFromNowIST(eod);
    events.push({
      system: def.label,
      event: 'EOD squareoff',
      scheduled: eod,
      status: diff < 0 ? 'Done' : 'Pending',
      tone: diff < 0 ? 'neutral' : 'pos',
      relative: relativeLabel(diff),
      sortKey: diff < 0 ? 9999 : diff,
    });
  }

  // Daily summary at 15:20
  const diff = minutesFromNowIST('15:20');
  events.push({
    system: 'All systems',
    event: 'Daily summary',
    scheduled: '15:20',
    status: diff < 0 ? 'Done' : 'Pending',
    tone: 'neutral',
    relative: relativeLabel(diff),
    sortKey: diff < 0 ? 9999 : diff,
  });

  // Sort by nearest event first, past events at end
  events.sort((a, b) => a.sortKey - b.sortKey);
  return events;
}
