import { createContext, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
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

type MtmPoint = [string, number];

interface MtmEvent {
  ts: string;
  type: 'entry' | 'adjust' | 'sl_hit' | 'exit';
  label: string;
  sig?: string | null;
  sym?: string | null;
  tx?: string | null;
  price?: number | null;
}

interface MtmSystem { points: MtmPoint[]; events: MtmEvent[]; }

export default function Nas() {
  const [states, setStates] = useState<Record<string, SystemStateRecord>>({});
  const [toast, setToast] = useState<string | null>(null);
  const [mtmData, setMtmData] = useState<Record<string, MtmSystem>>({});
  const [mtmCombined, setMtmCombined] = useState<MtmSystem | null>(null);
  const [expandedKey, setExpandedKey] = useState<string | null>(null);
  const [historyDays, setHistoryDays] = useState<Array<{
    date: string; combined_last: number; n_fired: number; n_systems: number;
  }>>([]);
  const [historyModal, setHistoryModal] = useState<{
    title: string; points: MtmPoint[]; events: MtmEvent[];
  } | null>(null);

  // Load saved daily snapshots (written by scripts/snapshot_nas_eod.py at 15:32).
  useEffect(() => {
    fetch(`/static/snapshots/index.json?t=${Date.now()}`, { cache: 'no-store' })
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => { if (d?.days) setHistoryDays(d.days); })
      .catch(() => { /* no snapshots yet */ });
  }, []);

  async function openHistory(d: string) {
    try {
      const r = await fetch(`/static/snapshots/nas_mtm_${d}.json?t=${Date.now()}`,
        { cache: 'no-store' });
      if (!r.ok) return;
      const data = await r.json();
      const c = data.combined ?? {};
      setHistoryModal({
        title: `NAS overall — ${d}`,
        points: c.points ?? [],
        events: c.events ?? [],
      });
    } catch { /* swallow */ }
  }
  const [liveTicks, setLiveTicks] = useState<LiveTicks>({
    spot: null,
    legs: {},
    connected: false,
  });
  const evtRef = useRef<EventSource | null>(null);

  // Poll the per-system intraday MTM curves every 30s. Cheap (one row per
  // system per 3 min), reuses the same snapshots the EOD report renders.
  useEffect(() => {
    let cancelled = false;
    async function pull() {
      // Read the cron-written static dump (cache-busted). Doesn't depend on
      // any backend route registration; works even if gunicorn is running
      // older code than this bundle.
      try {
        let data: {
          systems: Record<string, MtmSystem>;
          combined?: MtmSystem;
        } | null = null;
        try {
          const resp = await fetch(`/static/nas_mtm.json?t=${Date.now()}`,
            { cache: 'no-store' });
          if (resp.ok) data = await resp.json();
        } catch { /* try API fallback */ }
        if (!data) {
          data = await apiGet<{
            systems: Record<string, MtmSystem>;
            combined?: MtmSystem;
          }>('/api/nas/mtm');
        }
        if (cancelled || !data?.systems) return;
        const next: Record<string, MtmSystem> = {};
        for (const k of Object.keys(data.systems)) {
          next[k] = {
            points: data.systems[k]?.points ?? [],
            events: data.systems[k]?.events ?? [],
          };
        }
        setMtmData(next);
        setMtmCombined(data.combined
          ? { points: data.combined.points ?? [],
              events: data.combined.events ?? [] }
          : null);
      } catch { /* sparkline stays in empty state */ }
    }
    pull();
    const t = setInterval(pull, 30000);
    return () => { cancelled = true; clearInterval(t); };
  }, []);

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

  // Margin shape (served by backend's _orb_get_margin):
  //   available  = eq.net (Kite UI 'Available margin', total pool)
  //   cash_cap   = min(2 × live_balance, net) — actual cap on new F&O margin
  //                under SEBI's 50:50 cash:collateral rule. This is what
  //                the trader can ACTUALLY size against today.
  //   live_balance = real free cash right now
  const margin = headerState?.margin as
    | { available?: number; cash_cap?: number; live_balance?: number }
    | undefined;
  // Display the cash-constrained cap as the primary number — matches the
  // trader's sizing intuition. Show total pool in the hint for context.
  const cashCap = margin?.cash_cap;
  const totalPool = margin?.available;

  function showToast(msg: string) {
    setToast(msg);
    setTimeout(() => setToast(null), 2500);
  }

  /* ---------- whats next schedule ---------- */

  const nextEvents = useMemo(() => buildNextEvents(states), [states]);

  return (
    <LiveTicksContext.Provider value={liveTicks}>
    <div className={styles.root}>
      {/* Tier 1 (exchange-side SL-M) not yet built — remove this block when it ships. */}
      <div className={styles.slmWarning} role="alert">
        <span className={styles.slmWarningIcon} aria-hidden="true">⚠</span>
        <div className={styles.slmWarningText}>
          <strong>NAS LIVE — exchange-side SL-M not yet implemented.</strong>
          <span className={styles.slmWarningDetail}>
            {' '}If Flask or ticker dies during an open position, the short is
            unprotected until the process recovers. Tier 1 build pending.
          </span>
        </div>
      </div>

      <div className={styles.titleRow}>
        <div>
          <div className="page-title">NAS options</div>
          <div className="page-subtitle">
            Eight Nifty options systems running in parallel. ATR squeeze entries on the
            left, time-based 9:16 entries on the right.
          </div>
        </div>
        <div className={styles.titleRowActions}>
          <MasterModeToggle onToast={setToast} />
          <Link
            to="/nas-panic"
            className={styles.panicLink}
            title="Closes all positions and disables all 8 NAS variants. Survives Flask/VPS restart."
          >
            ⚠ Emergency stop
          </Link>
        </div>
      </div>

      {toast ? <div className={styles.toast}>{toast}</div> : null}

      <ChartModal
        open={!!expandedKey}
        title={
          expandedKey === '_combined'
            ? 'Overall NAS — Intraday P&L'
            : ((ALL_SYSTEMS.find((s) => s.key === expandedKey)?.label ?? '') +
               ' — Intraday P&L')
        }
        points={
          expandedKey === '_combined'
            ? (mtmCombined?.points || [])
            : expandedKey
            ? (mtmData[expandedKey]?.points || [])
            : []
        }
        events={
          expandedKey === '_combined'
            ? (mtmCombined?.events || [])
            : expandedKey
            ? (mtmData[expandedKey]?.events || [])
            : []
        }
        onClose={() => setExpandedKey(null)}
      />

      <ChartModal
        open={!!historyModal}
        title={historyModal?.title ?? ''}
        points={historyModal?.points ?? []}
        events={historyModal?.events ?? []}
        onClose={() => setHistoryModal(null)}
      />

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
          // Cash-constrained F&O cap (SEBI 50:50 rule), NOT Kite UI's
          // 'Available margin' (which is the larger total pool).
          value={cashCap !== undefined ? formatRs(cashCap) : '—'}
          hint={
            totalPool !== undefined
              ? `Max new F&O margin · 50:50 cash rule (Total pool ${formatRs(totalPool)})`
              : 'Max new F&O margin (50:50 cash rule)'
          }
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

      {mtmCombined && mtmCombined.points.length >= 2 ? (
        <section className={styles.combinedHero}>
          <div className={styles.combinedHead}>
            <div className="section-title">Overall NAS · intraday P&amp;L</div>
            <div className={styles.combinedMeta}>
              {mtmCombined.points.length} pts · click to expand
            </div>
          </div>
          <button
            type="button"
            className={styles.sparkButton}
            onClick={() => setExpandedKey('_combined')}
            title="Expand combined chart"
          >
            <div className={styles.combinedBox}>
              <PnlChart
                points={mtmCombined.points}
                events={mtmCombined.events}
              />
              <div className={styles.sparkMeta}>
                {(() => {
                  const ys = mtmCombined.points.map((p) => p[1]);
                  const last = ys[ys.length - 1];
                  const yMin = Math.min(0, ...ys);
                  const yMax = Math.max(0, ...ys);
                  return (
                    <>
                      <span className={last >= 0 ? styles.sparkPos : styles.sparkNeg}>
                        now {fmtPnl(last)}
                      </span>
                      <span className={styles.sparkRange}>
                        lo {fmtPnl(yMin)} · hi {fmtPnl(yMax)} · all 8 systems ⤢
                      </span>
                    </>
                  );
                })()}
              </div>
            </div>
          </button>
        </section>
      ) : null}

      {historyDays.length > 0 ? (
        <section className={styles.historyStrip}>
          <div className={styles.historyHead}>Past sessions · click to view</div>
          <div className={styles.historyList}>
            {historyDays.slice(0, 12).map((d) => (
              <button
                key={d.date}
                type="button"
                className={styles.historyChip}
                onClick={() => openHistory(d.date)}
              >
                <div className={styles.historyDate}>{d.date}</div>
                <div className={d.combined_last >= 0 ? styles.sparkPos : styles.sparkNeg}>
                  {fmtPnl(d.combined_last)}
                </div>
                <div className={styles.historyMeta}>
                  {d.n_fired}/{d.n_systems} fired
                </div>
              </button>
            ))}
          </div>
        </section>
      ) : null}

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
                series={mtmData[s.key]?.points || []}
                events={mtmData[s.key]?.events || []}
                onExpand={() => setExpandedKey(s.key)}
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
                series={mtmData[s.key]?.points || []}
                events={mtmData[s.key]?.events || []}
                onExpand={() => setExpandedKey(s.key)}
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
  series: MtmPoint[];
  events: MtmEvent[];
  onExpand: () => void;
}

const EVENT_COLOR: Record<string, string> = {
  entry:  '#22c55e',  // green — open
  adjust: '#f59e0b',  // amber — roll / adj
  sl_hit: '#ef4444',  // red   — SL hit
  exit:   '#94a3b8',  // grey  — time / EOD exit
};

function fmtPnl(v: number): string {
  return (v >= 0 ? '+₹' : '-₹') + Math.abs(Math.round(v)).toLocaleString('en-IN');
}

function shortSym(s: string | undefined | null): string {
  // "NIFTY26MAY23400CE+NIFTY26MAY23400PE" -> "23400CE+23400PE"
  // Single symbol gets its trailing strike+leg fragment.
  if (!s) return '';
  return s.split('+')
          .filter(Boolean)
          .map((p) => p.slice(-7))
          .join('+');
}

interface PnlChartProps {
  points: MtmPoint[];
  events: MtmEvent[];
  expanded?: boolean;
}

function PnlChart({ points, events, expanded = false }: PnlChartProps) {
  // SVG day-P&L curve. Color graded by intensity:
  //   above 0 → green (deeper as profit grows)
  //   below 0 → red   (deeper as loss grows)
  // Event markers (entry / adjust / sl_hit / exit) drawn as dotted verticals.
  const W = expanded ? 920 : 320;
  const H = expanded ? 340 : 56;
  const PAD_X = expanded ? 56 : 4;
  const PAD_Y = expanded ? 28 : 4;
  const ys = points.map((p) => p[1]);
  const yMinRaw = Math.min(0, ...ys);
  const yMaxRaw = Math.max(0, ...ys);
  // pad y a bit in expanded so events at edges don't get clipped
  const yPad = expanded ? Math.max(50, (yMaxRaw - yMinRaw) * 0.08) : 0;
  const yMin = yMinRaw - yPad;
  const yMax = yMaxRaw + yPad;
  const ySpan = yMax - yMin || 1;
  const tMin = new Date(points[0][0]).getTime();
  const tMax = new Date(points[points.length - 1][0]).getTime();
  const tSpan = tMax - tMin || 1;
  const xOf = (ts: string) => {
    const m = new Date(ts).getTime();
    const x = PAD_X + ((m - tMin) / tSpan) * (W - 2 * PAD_X);
    return Math.max(PAD_X, Math.min(W - PAD_X, x));
  };
  const yOf = (v: number) =>
    H - PAD_Y - ((v - yMin) / ySpan) * (H - 2 * PAD_Y);
  const zeroY = yOf(0);
  const d = points
    .map((p, i) => `${i ? 'L' : 'M'} ${xOf(p[0])} ${yOf(p[1])}`)
    .join(' ');
  const firstX = xOf(points[0][0]);
  const lastX = xOf(points[points.length - 1][0]);
  const area = `${d} L ${lastX} ${zeroY} L ${firstX} ${zeroY} Z`;
  const last = ys[ys.length - 1];
  // intensity (0.25..1) scales gradient stop opacity by how deep min/max reach
  const denom = Math.max(1, Math.abs(yMaxRaw) + Math.abs(yMinRaw));
  const gIntensity = Math.min(1, Math.max(0.25, Math.abs(yMaxRaw) / denom + 0.25));
  const rIntensity = Math.min(1, Math.max(0.25, Math.abs(yMinRaw) / denom + 0.25));
  const zeroFrac = ((zeroY - PAD_Y) / (H - 2 * PAD_Y)) * 100;
  const gid = `pnlg${Math.random().toString(36).slice(2, 8)}`;

  // Interpolate event y onto the curve once, for the overlay markers.
  function interpY(ts: string): number {
    const t = new Date(ts).getTime();
    if (t <= tMin) return ys[0];
    if (t >= tMax) return ys[ys.length - 1];
    for (let j = 0; j < points.length - 1; j++) {
      const t0 = new Date(points[j][0]).getTime();
      const t1 = new Date(points[j + 1][0]).getTime();
      if (t >= t0 && t <= t1) {
        const f = (t - t0) / Math.max(1, t1 - t0);
        return ys[j] + f * (ys[j + 1] - ys[j]);
      }
    }
    return ys[ys.length - 1];
  }

  return (
    <div className={styles.chartWrap}>
    <svg
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="none"
      className={expanded ? styles.chartSvg : styles.sparkSvg}
    >
      <defs>
        <linearGradient id={gid} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#22c55e"
                stopOpacity={(0.55 * gIntensity).toFixed(3)} />
          <stop offset={`${Math.max(0, Math.min(100, zeroFrac)).toFixed(2)}%`}
                stopColor="#a3a3a3" stopOpacity="0.04" />
          <stop offset="100%" stopColor="#ef4444"
                stopOpacity={(0.55 * rIntensity).toFixed(3)} />
        </linearGradient>
      </defs>
      {expanded ? (() => {
        // Y-axis "nice" ticks: pick a step from [1,2,2.5,5]×10^k that lands
        // ~6 ticks across the range. Draw a dashed gridline + ₹-labelled
        // tick at each, skipping the zero line (rendered separately below).
        const range = yMax - yMin;
        if (range <= 0) return null;
        const targetTicks = 6;
        const rawStep = range / targetTicks;
        const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
        const norm = rawStep / mag;
        const step = norm <= 1.5 ? mag
                   : norm <= 3   ? 2 * mag
                   : norm <= 4   ? 2.5 * mag
                   : norm <= 7   ? 5 * mag
                   : 10 * mag;
        const ticks: number[] = [];
        const first = Math.ceil(yMin / step) * step;
        for (let v = first; v <= yMax + step * 0.001; v += step) {
          if (Math.abs(v) < step * 0.01) continue;  // skip near-zero
          ticks.push(v);
        }
        return (
          <g>
            {ticks.map((v) => {
              const yy = yOf(v);
              return (
                <g key={`yt-${v}`}>
                  <line x1={PAD_X} x2={W - PAD_X}
                        y1={yy} y2={yy}
                        stroke="#2a2a2a" strokeWidth="0.5"
                        strokeDasharray="2 4" />
                  <text x={PAD_X - 6} y={yy + 3.5}
                        fontSize="9.5" fill="#94a3b8"
                        textAnchor="end">
                    {fmtPnl(v)}
                  </text>
                </g>
              );
            })}
          </g>
        );
      })() : null}
      <line x1={PAD_X} x2={W - PAD_X} y1={zeroY} y2={zeroY}
            stroke="#3a3a3a" strokeDasharray="2 3"
            strokeWidth={expanded ? 1 : 0.7} />
      {expanded ? (
        <text x={PAD_X - 6} y={zeroY + 3.5}
              fontSize="9.5" fill="#94a3b8" textAnchor="end">
          ₹0
        </text>
      ) : null}
      <path d={area} fill={`url(#${gid})`} stroke="none" />
      <path d={d} fill="none"
            stroke={last >= 0 ? '#16a34a' : '#dc2626'}
            strokeWidth={expanded ? 2 : 1.4}
            strokeLinejoin="round" strokeLinecap="round" />
      {expanded ? events.map((e, i) => {
        // Only the vertical dotted line stays in SVG (lines stretch fine).
        const ex = xOf(e.ts);
        if (ex < PAD_X - 2 || ex > W - PAD_X + 2) return null;
        const color = EVENT_COLOR[e.type] || '#888';
        return (
          <line key={`vl-${e.ts}-${i}`}
                x1={ex} x2={ex} y1={PAD_Y} y2={H - PAD_Y}
                stroke={color} strokeOpacity={0.30}
                strokeWidth={1} strokeDasharray="2 3" />
        );
      }) : null}
      {expanded ? (
        <>
          <text x={W - PAD_X} y={PAD_Y - 9} fontSize="11"
                fill={last >= 0 ? '#22c55e' : '#ef4444'}
                textAnchor="end" fontWeight="700">
            now {fmtPnl(last)}
          </text>
          {(() => {
            // 5-min time ticks along the x-axis (expanded mode only).
            const ticks: Array<{ x: number; lab: string }> = [];
            const start = new Date(tMin);
            start.setSeconds(0, 0);
            // round up to next 5-min boundary
            const m = start.getMinutes();
            start.setMinutes(m + ((5 - (m % 5)) % 5));
            for (let t = start.getTime(); t <= tMax + 1; t += 5 * 60 * 1000) {
              const dt = new Date(t);
              const x =
                PAD_X + ((t - tMin) / tSpan) * (W - 2 * PAD_X);
              if (x < PAD_X - 2 || x > W - PAD_X + 2) continue;
              const lab = `${String(dt.getHours()).padStart(2, '0')}:${String(
                dt.getMinutes()
              ).padStart(2, '0')}`;
              ticks.push({ x, lab });
            }
            // thin out labels if too many (>16 ticks → label every other)
            const step = ticks.length > 16 ? 2 : 1;
            return (
              <g>
                {ticks.map((t, i) => (
                  <g key={t.x}>
                    <line
                      x1={t.x} x2={t.x}
                      y1={H - PAD_Y} y2={H - PAD_Y + 4}
                      stroke="#525252" strokeWidth="0.7" />
                    {i % step === 0 ? (
                      <text x={t.x} y={H - PAD_Y + 16}
                            fontSize="9.5" fill="#94a3b8"
                            textAnchor="middle">
                        {t.lab}
                      </text>
                    ) : null}
                  </g>
                ))}
              </g>
            );
          })()}
        </>
      ) : null}
    </svg>
    {(() => {
      // HTML overlay markers: percent-positioned over the SVG so they're
      // immune to preserveAspectRatio="none" stretching. The SVG stays
      // responsive; the dots stay round.
      //
      // Deconflict labels: when multiple events fall in a narrow x-window
      // (8 NAS systems all entering near 9:16, all exiting near 14:45/15:15),
      // stagger labels vertically so they don't stack on top of each other.
      // Also flip the label to the left of the dot near the right edge to
      // avoid clipping at the chart border.
      const placed = events
        .map((e, i) => {
          const tMs = new Date(e.ts).getTime();
          if (tMs < tMin - 1 || tMs > tMax + 1) return null;
          const xPct = (xOf(e.ts) / W) * 100;
          const yPct = (yOf(interpY(e.ts)) / H) * 100;
          return { e, i, tMs, xPct, yPct };
        })
        .filter((p): p is NonNullable<typeof p> => p !== null)
        .sort((a, b) => a.xPct - b.xPct || a.tMs - b.tMs);

      const BUCKET_PCT = 3.5;       // ~32px @ W=920
      const LINE_HEIGHT_PX = 13;
      let bucketStart = -Infinity;
      let bucketIdx = 0;
      const decorated = placed.map((p) => {
        if (p.xPct - bucketStart > BUCKET_PCT) {
          bucketStart = p.xPct;
          bucketIdx = 0;
        } else {
          bucketIdx += 1;
        }
        return { ...p, stagger: bucketIdx };
      });

      return decorated.map(({ e, i, xPct, yPct, stagger }) => {
        const color = EVENT_COLOR[e.type] || '#888';
        const size = expanded ? 9 : 7;
        const flipLeft = expanded && xPct > 62;
        const stackPx = expanded ? stagger * LINE_HEIGHT_PX : 0;
        return (
          <span
            key={`m-${e.ts}-${i}`}
            className={styles.markerDot}
            style={{
              left: `${xPct}%`,
              top: `${yPct}%`,
              width: size,
              height: size,
              background: color,
            }}
            title={`${e.label}${e.sym ? ' · ' + e.sym : ''} @ ${e.ts.slice(11, 16)}`}
          >
            {expanded ? (
              <span
                className={flipLeft ? styles.markerLabelRight : styles.markerLabel}
                style={{ color, transform: `translateY(${stackPx}px)` }}
              >
                {e.label}{e.sym ? ` ${shortSym(e.sym)}` : ''}
              </span>
            ) : null}
          </span>
        );
      });
    })()}
    </div>
  );
}

interface SparkProps {
  points: MtmPoint[];
  events: MtmEvent[];
  onExpand: () => void;
}

function Sparkline({ points, events, onExpand }: SparkProps) {
  if (!points || points.length < 2) {
    return (
      <div className={styles.sparkEmpty}>
        Live P&amp;L curve appears once snapshots flow (09:15+).
      </div>
    );
  }
  const last = points[points.length - 1][1];
  const ys = points.map((p) => p[1]);
  const yMin = Math.min(0, ...ys);
  const yMax = Math.max(0, ...ys);
  return (
    <button type="button" onClick={onExpand} className={styles.sparkButton}
            title="Expand chart">
      <div className={styles.sparkBox}>
        <PnlChart points={points} events={events} />
        <div className={styles.sparkMeta}>
          <span className={last >= 0 ? styles.sparkPos : styles.sparkNeg}>
            now {fmtPnl(last)}
          </span>
          <span className={styles.sparkRange}>
            lo {fmtPnl(yMin)} · hi {fmtPnl(yMax)} · {points.length} pts
            {events.length ? ` · ${events.length} ev` : ''} ⤢
          </span>
        </div>
      </div>
    </button>
  );
}

interface ChartModalProps {
  open: boolean;
  title: string;
  points: MtmPoint[];
  events: MtmEvent[];
  onClose: () => void;
}

function ChartModal({ open, title, points, events, onClose }: ChartModalProps) {
  useEffect(() => {
    if (!open) return;
    const k = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', k);
    return () => window.removeEventListener('keydown', k);
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div className={styles.modalBackdrop} onClick={onClose}>
      <div className={styles.modalCard} onClick={(e) => e.stopPropagation()}>
        <div className={styles.modalHead}>
          <div className={styles.modalTitle}>{title}</div>
          <button type="button" onClick={onClose}
                  className={styles.modalClose} aria-label="Close">×</button>
        </div>
        {points.length >= 2 ? (
          <PnlChart points={points} events={events} expanded />
        ) : (
          <div className={styles.sparkEmpty} style={{ margin: 24 }}>
            No snapshots yet — comes alive after 09:15.
          </div>
        )}
        {events.length ? (
          <div className={styles.eventLegend}>
            {Object.entries({
              entry: 'Entry', adjust: 'Adjust', sl_hit: 'SL hit', exit: 'Exit',
            }).map(([k, lab]) => (
              <span key={k} className={styles.legendItem}>
                <span className={styles.legendDot}
                      style={{ background: EVENT_COLOR[k] }} />
                {lab}
              </span>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  );
}

function SystemPanel({ def, onStateChange, onToast, series, events, onExpand }: PanelProps) {
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
  // Day P&L = closed trades today (persisted) + live open-leg P&L.
  // Old code used `??` which preferred 0 (no closed trades) over the open-leg
  // MTM, so system-level Day P&L stuck at Rs 0 even when legs were profitable.
  const persistedTodayPnl = (state?.stats?.today_pnl as number | undefined) ?? 0;
  const dayPnl = persistedTodayPnl + totalPnl;

  // Closed-today legs (sorted newest first), with pnl computed from entry/exit.
  const closedLegs = (state?.positions?.closed_today ?? [])
    .map((p) => {
      const entry = p.entry_price ?? p.entry_premium;
      const exit = p.exit_price;
      const qty = p.qty ?? 0;
      const pnl =
        entry != null && exit != null && qty
          ? Math.round((entry - exit) * qty * 100) / 100
          : undefined;
      return { ...p, pnl_inr: pnl };
    })
    .sort((a, b) => (b.exit_time ?? '').localeCompare(a.exit_time ?? ''));
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
          <div className={styles.panelStatusMeta}>
            {formatInt(state?.positions?.total_active ?? 0)} active · {formatInt(reentries)} re-entry
          </div>
        </div>
      </div>

      <div className={styles.metricsRow}>
        <MiniMetric
          label="Day P&L"
          value={
            <span className={pnlClass(dayPnl)}>{formatPnl(dayPnl)}</span>
          }
        />
        <MiniMetric label="SL hits today" value={formatInt(slHits ?? 0)} />
      </div>

      <Sparkline points={series} events={events} onExpand={onExpand} />

      <div className={styles.legs}>
        {enriched.length === 0 ? (
          <div className={styles.noLegs}>No open legs</div>
        ) : (
          enriched.map((p, i) => <LegRow key={(p.tradingsymbol ?? '') + i} leg={p} />)
        )}
      </div>

      {closedLegs.length > 0 ? (
        <div className={styles.closedLegs}>
          <div className={styles.closedHead}>
            Closed today · {closedLegs.length}
          </div>
          {closedLegs.map((p, i) => (
            <LegRow
              key={'c' + (p.tradingsymbol ?? '') + i}
              leg={p}
              closed
              reason={p.exit_reason}
            />
          ))}
        </div>
      ) : null}

      <details className={styles.rules}>
        <summary className={styles.rulesSummary}>Rules &amp; snapshot</summary>
        <div className={styles.rulesBody}>
          <div className={styles.snapshotRow}>
            <div className={styles.snapshotItem}>
              <span className={styles.snapshotLabel}>Win rate (all-time)</span>
              <span className={styles.snapshotValue}>
                {winRate !== undefined ? formatPct(winRate, 1) : '—'}
              </span>
            </div>
            <div className={styles.snapshotItem}>
              <span className={styles.snapshotLabel}>Profit factor</span>
              <span className={styles.snapshotValue}>
                {pf !== undefined ? formatNumber(pf, 2) : '—'}
              </span>
            </div>
          </div>
          <div className={styles.rulesText}>{def.rules}</div>
        </div>
      </details>

      {err ? <div className={styles.errRow}>{err}</div> : null}
    </div>
  );
}

/* ---------- Master mode toggle (OFF / PAPER / LIVE for all 8 systems) ---------- */

interface MasterModeState {
  mode: 'off' | 'paper' | 'live' | 'mixed' | null;
  busy: boolean;
}

function MasterModeToggle({ onToast }: { onToast: (msg: string) => void }) {
  const [state, setState] = useState<MasterModeState>({ mode: null, busy: false });

  const refresh = async () => {
    try {
      const r = await fetch('/api/nas/master-mode');
      const d = await r.json();
      setState((p) => ({ ...p, mode: d.mode ?? null }));
    } catch {
      /* ignore */
    }
  };

  useEffect(() => {
    void refresh();
    const id = setInterval(refresh, 10_000);
    return () => clearInterval(id);
  }, []);

  const setMode = async (target: 'off' | 'paper' | 'live') => {
    if (state.busy || state.mode === target) return;
    if (target === 'live') {
      const ok = window.confirm(
        'Switch ALL 8 NAS systems to LIVE trading? Real money will be at risk on the next entry signal.',
      );
      if (!ok) return;
    }
    setState((p) => ({ ...p, busy: true }));
    try {
      const r = await fetch('/api/nas/master-mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: target }),
      });
      const d = await r.json();
      if (!r.ok) {
        onToast(`Master toggle failed: ${d.error || r.statusText}`);
      } else {
        setState({ mode: d.mode ?? target, busy: false });
        onToast(`All NAS systems → ${target.toUpperCase()}`);
        return;
      }
    } catch (e) {
      onToast(`Master toggle error: ${e instanceof Error ? e.message : 'unknown'}`);
    }
    setState((p) => ({ ...p, busy: false }));
  };

  const buttons: { id: 'off' | 'paper' | 'live'; label: string; cls: string }[] = [
    { id: 'off',   label: 'OFF',   cls: styles.masterBtnOff },
    { id: 'paper', label: 'PAPER', cls: styles.masterBtnPaper },
    { id: 'live',  label: 'LIVE',  cls: styles.masterBtnLive },
  ];

  return (
    <div className={styles.masterToggle}>
      <div className={styles.masterToggleLabel}>All NAS systems</div>
      <div className={styles.masterToggleGroup}>
        {buttons.map((b) => {
          const active = state.mode === b.id;
          return (
            <button
              key={b.id}
              type="button"
              className={`${styles.masterBtn} ${b.cls} ${active ? styles.masterBtnActive : ''}`}
              disabled={state.busy}
              onClick={() => void setMode(b.id)}
            >
              {b.label}
            </button>
          );
        })}
      </div>
      {state.mode === 'mixed' ? (
        <div className={styles.masterMixed}>mixed — variants in different states</div>
      ) : null}
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

function LegRow({
  leg,
  closed = false,
  reason,
}: {
  leg: NASPosition;
  closed?: boolean;
  reason?: string;
}) {
  const tsym = shortOptionSymbol(leg.tradingsymbol);
  const entry = leg.entry_price ?? leg.entry_premium;
  const ltp = leg.ltp ?? leg.exit_price;
  const pnl = leg.pnl_inr;
  const qty = leg.qty;
  const entryTime = formatLegTime(leg.entry_time);
  const exitTime = closed ? formatLegTime(leg.exit_time) : undefined;

  return (
    <div className={`${styles.leg} ${closed ? styles.legClosed : ''}`}>
      <div className={styles.legMain}>
        <span className={styles.legSide}>{leg.leg}</span>
        <span className={styles.legSym}>{tsym}</span>
        {qty ? <span className={styles.legQty}>×{qty}</span> : null}
        {closed && reason ? (
          <span className={styles.legReason}>{reason}</span>
        ) : null}
      </div>
      <div className={styles.legNums}>
        <span className={styles.legSmall}>
          {entry !== undefined ? formatNumber(entry) : '—'}
          {entryTime ? <span className={styles.legTime}> @{entryTime}</span> : null}
        </span>
        <span className={styles.legArrow}>→</span>
        <span className={styles.legSmall}>
          {ltp !== undefined ? formatNumber(ltp) : '—'}
          {exitTime ? <span className={styles.legTime}> @{exitTime}</span> : null}
        </span>
        <span className={pnlClass(pnl)} style={{ fontSize: 'var(--text-xs)' }}>
          {formatPnl(pnl)}
        </span>
      </div>
    </div>
  );
}

/** Shorten an option tradingsymbol like 'NIFTY2650523800PE' to '23800PE'.
 *  Drops the underlying + expiry prefix so the strike + CE/PE suffix is
 *  always visible even on narrow leg rows that would otherwise truncate. */
function shortOptionSymbol(tsym?: string | null): string {
  if (!tsym) return '—';
  const m = /(\d+)(CE|PE)$/.exec(tsym);
  return m ? `${m[1]}${m[2]}` : tsym;
}

function formatLegTime(iso?: string | null): string | null {
  if (!iso) return null;
  // Extract HH:MM from either ISO ("2026-04-22T11:51:42") or "11:51" fallback.
  const m = /T(\d{2}:\d{2})/.exec(iso) || /^(\d{2}:\d{2})/.exec(iso);
  return m ? m[1] : null;
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
