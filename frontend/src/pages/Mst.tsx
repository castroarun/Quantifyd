import { useEffect, useState } from 'react';
import styles from './Mst.module.css';
import { apiGet, apiPost } from '../api/client';
import type {
  MSTState, MSTEvent, MSTBar, MSTMode, MSTPosition,
} from '../api/mst-types';
import MetricCard from '../components/Cards/MetricCard';
import DataTable from '../components/DataTable/DataTable';
import type { Column } from '../components/DataTable/DataTable';
import Chip from '../components/Chip/Chip';
import StatusDot from '../components/StatusDot/StatusDot';
import { formatNumber, formatPnl, formatRs, pnlClass } from '../utils/format';
import { formatTime } from '../utils/time';

function modeLabel(state: MSTState | null): { label: string; tone: 'pos' | 'neg' | 'neutral' } {
  if (!state || !state.enabled) return { label: 'Disabled', tone: 'neutral' };
  if (state.live_trading) return { label: 'Live', tone: 'pos' };
  return { label: 'Paper', tone: 'neutral' };
}

function directionLabel(d: number): string {
  if (d === 1) return 'LONG';
  if (d === -1) return 'SHORT';
  return '—';
}

function seedOnlyChip(isSeed: boolean): string {
  return isSeed ? 'Seed warmup · awaiting first live 30m bar' : 'Live';
}

export default function Mst() {
  const [state, setState] = useState<MSTState | null>(null);
  const [events, setEvents] = useState<MSTEvent[]>([]);
  const [bars, setBars] = useState<MSTBar[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);

  const refresh = async () => {
    try {
      const [s, e, b] = await Promise.all([
        apiGet<MSTState>('/api/mst/state'),
        apiGet<{ events: MSTEvent[] }>('/api/mst/events?limit=50'),
        apiGet<{ bars: MSTBar[] }>('/api/mst/bars?limit=80'),
      ]);
      setState(s);
      setEvents(e.events || []);
      setBars(b.bars || []);
      setErr(null);
    } catch (ex: unknown) {
      const msg = ex instanceof Error ? ex.message : String(ex);
      setErr(msg);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 30_000);
    return () => clearInterval(id);
  }, []);

  const onSetMode = async (mode: MSTMode) => {
    if (busy) return;
    if (mode === 'live' && !window.confirm('Switch MST to LIVE — real orders will be placed. Continue?')) return;
    setBusy(true);
    try {
      await apiPost('/api/mst/toggle-mode', { mode });
      await refresh();
    } catch (ex: unknown) {
      const msg = ex instanceof Error ? ex.message : String(ex);
      setErr(msg);
    } finally {
      setBusy(false);
    }
  };

  const onKillSwitch = async () => {
    if (busy) return;
    if (!window.confirm('KILL SWITCH — close all open legs at market and halt entries. Continue?')) return;
    setBusy(true);
    try {
      await apiPost('/api/mst/kill-switch', {});
      await refresh();
    } catch (ex: unknown) {
      const msg = ex instanceof Error ? ex.message : String(ex);
      setErr(msg);
    } finally {
      setBusy(false);
    }
  };

  const positionCols: Column<MSTPosition>[] = [
    { key: 'leg_role', header: 'Role', width: 'minmax(120px, 1.4fr)',
      render: (p) => <span className={styles.bold}>{p.leg_role}</span> },
    { key: 'side', header: 'Side', width: '70px',
      render: (p) => <span className={p.side === 'BUY' ? styles.dirLong : styles.dirShort}>{p.side}</span> },
    { key: 'tradingsymbol', header: 'Symbol', width: 'minmax(140px, 1.6fr)',
      render: (p) => <span className={styles.mono}>{p.tradingsymbol}</span> },
    { key: 'strike', header: 'Strike', width: '80px', align: 'right',
      render: (p) => `${p.strike} ${p.option_type}` },
    { key: 'qty', header: 'Qty', width: '60px', align: 'right',
      render: (p) => p.qty },
    { key: 'entry_price', header: 'Entry', width: '80px', align: 'right',
      render: (p) => formatNumber(p.entry_price) },
    { key: 'pnl_inr', header: 'P&L', width: '90px', align: 'right',
      render: (p) => <span className={pnlClass(p.pnl_inr)}>{formatPnl(p.pnl_inr)}</span> },
    { key: 'status', header: 'Status', width: '90px',
      render: (p) => <Chip>{p.status}</Chip> },
    { key: 'expiry_date', header: 'Expiry', width: '100px',
      render: (p) => <span className={styles.mute}>{p.expiry_date}</span> },
  ];

  const eventCols: Column<MSTEvent>[] = [
    { key: 'bar_dt', header: 'Time', width: '140px',
      render: (e) => <span className={styles.mute}>{e.bar_dt.replace('T', ' ').slice(0, 16)}</span> },
    { key: 'event_type', header: 'Event', width: 'minmax(150px, 1.6fr)',
      render: (e) => <span className={styles.bold}>{e.event_type}</span> },
    { key: 'direction', header: 'Dir', width: '60px',
      render: (e) => directionLabel(e.direction ?? 0) },
    { key: 'pyramid_level', header: 'L', width: '40px', align: 'right',
      render: (e) => e.pyramid_level ?? '—' },
    { key: 'price', header: 'Price', width: '80px', align: 'right',
      render: (e) => formatNumber(e.price) },
    { key: 'notes', header: 'Notes', width: 'minmax(200px, 3fr)',
      render: (e) => <span className={styles.mute}>{e.notes ?? '—'}</span> },
  ];

  if (loading) return <div className={styles.loading}>Loading MST state…</div>;

  const ml = modeLabel(state);
  const cfg = state?.config;
  const dir = state?.mst_direction ?? 0;

  return (
    <div className={styles.root}>
      <div className={styles.headerRow}>
        <div>
          <div className="page-title">MST · NIFTY 30-min</div>
          <div className="page-subtitle">
            SuperTrend({cfg?.atr_period ?? 21}, {cfg?.multiplier ?? 5.0}) +
            {' '}Stoch({cfg?.stoch_k ?? 14},{cfg?.stoch_d ?? 3},{cfg?.stoch_smooth ?? 3})
            {' · '}{state?.live_trading ? 'Live trading' : 'Paper trading'}
            {state?.current_expiry_dt && ` · expiry ${state.current_expiry_dt}`}
          </div>
        </div>
        <div className={styles.headerActions}>
          <StatusDot
            kind={state?.enabled ? (state.live_trading ? 'connected' : 'warning') : 'disconnected'}
            label={ml.label}
          />
          <div className={styles.modeSwitch}>
            <button
              className={`${styles.modeBtn} ${!state?.enabled ? styles.modeActive : ''}`}
              disabled={busy}
              onClick={() => onSetMode('off')}
            >
              Off
            </button>
            <button
              className={`${styles.modeBtn} ${state?.enabled && !state.live_trading ? styles.modeActive : ''}`}
              disabled={busy}
              onClick={() => onSetMode('paper')}
            >
              Paper
            </button>
            <button
              className={`${styles.modeBtn} ${state?.enabled && state.live_trading ? styles.modeActiveLive : ''}`}
              disabled={busy}
              onClick={() => onSetMode('live')}
            >
              Live
            </button>
            <button
              className={styles.killBtn}
              disabled={busy}
              onClick={onKillSwitch}
              title="Close all legs at market + halt entries"
            >
              Kill switch
            </button>
          </div>
        </div>
      </div>

      {err ? <div className={styles.error}>{err}</div> : null}

      {(() => {
        const li = state?.live_indicators;
        const liveSpot = state?.live_spot;
        const seedOnly = !!li?.is_seed_only;
        const k = li?.stoch_k;
        const d = li?.stoch_d;
        const stoch = (k != null && d != null) ? `${k.toFixed(1)} / ${d.toFixed(1)}` : '—';
        const stochZone = state?.proximity?.stoch_zone;
        const spotShown = liveSpot ?? li?.close ?? null;
        const stocSubtitle = stochZone
          ? (stochZone === 'overbought' ? `Overbought · K is ${(k ?? 0) >= (state?.config?.stoch_ob ?? 80) ? 'past' : 'near'} OB` :
             stochZone === 'oversold'   ? `Oversold · K is ${(k ?? 0) <= (state?.config?.stoch_os ?? 20) ? 'past' : 'near'} OS` :
             `Neutral · OB ${(state?.config?.stoch_ob ?? 80)} / OS ${(state?.config?.stoch_os ?? 20)}`)
          : '—';
        return (
          <div className={styles.metrics}>
            <MetricCard
              label="Day P&L"
              value={<span className={pnlClass(state?.today_pnl)}>{formatPnl(state?.today_pnl)}</span>}
              hint={`${state?.open_legs?.length ?? 0} open · ${state?.closed_today?.length ?? 0} closed today`}
            />
            <MetricCard
              label="State"
              value={state?.state_machine ?? '—'}
              hint={`Direction ${directionLabel(dir)} · L${state?.pyramid_level ?? 1}`}
            />
            <MetricCard
              label="Stoch K / D"
              value={stoch}
              hint={stocSubtitle}
            />
            <MetricCard
              label="NIFTY Spot"
              value={spotShown != null ? formatNumber(spotShown) : '—'}
              hint={
                liveSpot
                  ? (li?.atr21 ? `Live tick · ATR21 ${li.atr21.toFixed(1)}` : 'Live tick')
                  : (li?.atr21 ? `${seedOnly ? 'Seed bar' : 'Last 30m close'} · ATR21 ${li.atr21.toFixed(1)}` : '—')
              }
            />
          </div>
        );
      })()}

      {/* Proximity to triggers — visual gauges. Always visible; values fill in
          live as bars stream from NasTicker. */}
      {(() => {
        const li = state?.live_indicators;
        const px = state?.proximity ?? {};
        const cfg = state?.config;
        const spot = state?.live_spot ?? li?.close ?? null;
        if (!li || spot == null) return null;
        const k = li.stoch_k;
        const dVal = li.stoch_d;
        const ob = cfg?.stoch_ob ?? 80;
        const os = cfg?.stoch_os ?? 20;
        const kPct = k != null ? Math.max(0, Math.min(100, k)) : null;
        const stPos = li.st_value != null && li.atr21 ? (spot - li.st_value) / Math.max(li.atr21, 1) : null;  // distance in ATRs
        return (
          <section className={styles.section}>
            <div className={styles.sectionHead}>
              <div className="section-title">Proximity to triggers</div>
              <Chip>{seedOnlyChip(li.is_seed_only)}</Chip>
            </div>
            <div className={styles.proxGrid}>
              {/* Spot vs SuperTrend line */}
              <div className={styles.proxItem}>
                <div className={styles.proxLabel}>
                  Spot vs SuperTrend
                  <span className={styles.proxValue}>
                    {px.spot_to_st_pts != null
                      ? `${px.spot_to_st_pts > 0 ? '+' : ''}${px.spot_to_st_pts.toFixed(1)} pts (${px.spot_to_st_pct?.toFixed(2)}%)`
                      : '—'}
                  </span>
                </div>
                <div className={styles.proxBar}>
                  <div
                    className={styles.proxFill}
                    style={{
                      left: '50%',
                      width: stPos != null ? `${Math.min(48, Math.abs(stPos) * 8)}%` : '0%',
                      transform: stPos != null && stPos < 0 ? 'translateX(-100%)' : undefined,
                      background: li.direction === 1 ? 'var(--accent-pos)' : 'var(--accent-neg)',
                    }}
                  />
                  <div className={styles.proxMidLine} />
                </div>
                <div className={styles.proxScale}>
                  <span>flip {li.direction === 1 ? 'down' : 'up'}</span>
                  <span>0</span>
                  <span>{li.direction === 1 ? '↑ trending up' : '↓ trending down'}</span>
                </div>
              </div>

              {/* Stoch K position */}
              <div className={styles.proxItem}>
                <div className={styles.proxLabel}>
                  Stoch %K position
                  <span className={styles.proxValue}>
                    {k != null ? `K=${k.toFixed(1)} D=${(dVal != null ? dVal.toFixed(1) : '—')}` : '—'}
                  </span>
                </div>
                <div className={styles.proxStochBar}>
                  <div className={styles.proxStochOs} style={{ width: `${os}%` }} />
                  <div className={styles.proxStochOb} style={{ left: `${ob}%`, width: `${100 - ob}%` }} />
                  {kPct != null && (
                    <div className={styles.proxStochMarker} style={{ left: `${kPct}%` }} title={`K=${k?.toFixed(1)}`}>K</div>
                  )}
                </div>
                <div className={styles.proxScale}>
                  <span>0 (OS)</span>
                  <span>{os}</span>
                  <span>50</span>
                  <span>{ob}</span>
                  <span>100 (OB)</span>
                </div>
              </div>

              {/* ARMED state — break-of-extreme distance */}
              {state?.state_machine === 'ARMED' && px.armed_break_distance_pts != null ? (
                <div className={styles.proxItem}>
                  <div className={styles.proxLabel}>
                    Break of extreme
                    <span className={styles.proxValue}>
                      Need {px.armed_break_distance_pts > 0
                        ? `${px.armed_break_distance_pts.toFixed(1)} more pts`
                        : 'BREAK CONFIRMED'}
                    </span>
                  </div>
                  <div className={styles.proxBar}>
                    <div className={styles.proxFill} style={{
                      left: 0,
                      width: px.armed_break_distance_pts <= 0 ? '100%' : `${Math.max(0, 100 - Math.min(100, Math.abs(px.armed_break_distance_pts) * 2))}%`,
                      background: px.armed_break_distance_pts <= 0 ? 'var(--accent-pos)' : 'var(--ink-muted)',
                    }} />
                  </div>
                  <div className={styles.proxScale}>
                    <span>now {formatNumber(spot)}</span>
                    <span>target {formatNumber(px.armed_break_target ?? null)}</span>
                  </div>
                </div>
              ) : null}

              {/* CONDOR_OPEN_L1 — distance to safety wing breach */}
              {state?.state_machine === 'CONDOR_OPEN_L1' && px.safety_distance_pts != null ? (
                <div className={styles.proxItem}>
                  <div className={styles.proxLabel}>
                    Safety wing breach (L2 pyramid trigger)
                    <span className={styles.proxValue}>
                      {px.safety_distance_pts > 0
                        ? `${px.safety_distance_pts.toFixed(1)} pts away`
                        : 'BREACHED — pyramid imminent'}
                    </span>
                  </div>
                  <div className={styles.proxBar}>
                    <div className={styles.proxFill} style={{
                      left: 0,
                      width: px.safety_distance_pts <= 0 ? '100%' : `${Math.max(0, 100 - Math.min(100, px.safety_distance_pts / 5))}%`,
                      background: px.safety_distance_pts <= 0 ? 'var(--accent-neg)' : 'var(--ink-muted)',
                    }} />
                  </div>
                  <div className={styles.proxScale}>
                    <span>spot {formatNumber(spot)}</span>
                    <span>safety {formatNumber(px.safety_breach_level ?? null)}</span>
                  </div>
                </div>
              ) : null}

              {/* CONDOR_OPEN_L1 — last CST level reference (for D_cumulative) */}
              {state?.state_machine === 'CONDOR_OPEN_L1' && px.last_cst_level != null ? (
                <div className={styles.proxItem}>
                  <div className={styles.proxLabel}>
                    Last CST bar level (D_cumulative reference)
                    <span className={styles.proxValue}>
                      {(px.spot_vs_cst_pts ?? 0) > 0
                        ? `+${px.spot_vs_cst_pts?.toFixed(1)} pts past CST level`
                        : `${px.spot_vs_cst_pts?.toFixed(1)} pts (still inside)`}
                    </span>
                  </div>
                  <div className={styles.proxScale}>
                    <span>CST {formatNumber(px.last_cst_level)}</span>
                    <span>spot {formatNumber(spot)}</span>
                  </div>
                </div>
              ) : null}
            </div>
          </section>
        );
      })()}

      {/* current state details */}
      {state?.state_machine === 'ARMED' ? (
        <section className={styles.section}>
          <div className={styles.sectionHead}>
            <div className="section-title">Armed — waiting for break of extreme</div>
            <Chip>{directionLabel(dir)}</Chip>
          </div>
          <div className={styles.armedDetails}>
            {dir === 1 ? (
              <span>Break above <b>{formatNumber(state.armed_high)}</b> to activate</span>
            ) : (
              <span>Break below <b>{formatNumber(state.armed_low)}</b> to activate</span>
            )}
          </div>
        </section>
      ) : null}

      {/* positions */}
      <section className={styles.section}>
        <div className={styles.sectionHead}>
          <div className="section-title">Positions</div>
          <Chip>
            {state?.open_legs?.length ?? 0} open · {state?.closed_today?.length ?? 0} closed today
          </Chip>
        </div>
        <DataTable
          columns={positionCols}
          rows={[
            ...(state?.open_legs ?? []),
            ...(state?.closed_today ?? []),
          ]}
          emptyText={state?.state_machine === 'NO_POSITION' || state?.state_machine === 'ARMED'
            ? 'No open positions'
            : 'Loading positions…'}
          rowKey={(p) => `${p.id}`}
        />
      </section>

      {/* events */}
      <section className={styles.section}>
        <div className={styles.sectionHead}>
          <div className="section-title">Recent events</div>
          <Chip>{events.length} events</Chip>
        </div>
        <DataTable
          columns={eventCols}
          rows={events}
          emptyText="No events yet"
          rowKey={(e) => `${e.id}`}
        />
      </section>

      {/* strategy rules */}
      <section className={styles.section}>
        <details className={styles.rulesBlock}>
          <summary className={styles.rulesSummary}>Strategy rules · MST + CST + Pyramid</summary>
          <div className={styles.rulesBody}>
            <div className={styles.ruleItem}>
              <span className={styles.ruleLabel}>Setup</span>
              <span>
                NIFTY 30-min · MST = SuperTrend({cfg?.atr_period},{cfg?.multiplier}) ·
                {' '}CST = Stoch({cfg?.stoch_k},{cfg?.stoch_d},{cfg?.stoch_smooth}) OB={cfg?.stoch_ob}/OS={cfg?.stoch_os} ·
                {' '}{cfg?.lots_per_leg} lot ({cfg?.lot_size} contracts) per leg per pyramid level.
              </span>
            </div>
            <div className={styles.ruleItem}>
              <span className={styles.ruleLabel}>MST entry</span>
              <span>
                On ST flip, wait for next bar to break flip-bar's high (long) or low (short).
                Then enter weekly bull-call (long) or bear-put (short) debit spread, anchored at <b>1st OTM</b>
                {' '}(ATM ± {cfg?.debit_otm_offset ?? 50} pts), {cfg?.spread_width} wide.
                Next NIFTY weekly Tuesday expiry with ≥{cfg?.min_dte_at_entry} DTE.
              </span>
            </div>
            <div className={styles.ruleItem}>
              <span className={styles.ruleLabel}>CST trigger (hedge)</span>
              <span>
                LONG bias: %K crosses below %D from K_prev≥{cfg?.stoch_ob}. Add bear-call spread at ATM+{cfg ? 2*cfg.spread_width : 400}/{cfg ? 3*cfg.spread_width : 600} (entry-anchored). Mirror for short.
                Subsequent CSTs in same week → log only.
                If credit &lt; ₹{cfg?.min_credit_per_lot}/lot, roll-and-reset to next week (Reading D, {cfg?.reset_width}-wide spot-centered).
              </span>
            </div>
            <div className={styles.ruleItem}>
              <span className={styles.ruleLabel}>Pyramid trigger (OR of two paths)</span>
              <span>
                In CONDOR_OPEN_L1, pyramid fires on whichever path triggers first:
                <br />
                <b>Path 1 — D_cumulative AND B (momentum):</b> within last {cfg?.pyramid_d_lookback ?? 6} bars after CST,
                {' '}<code>(closes_beyond − closes_against) ≥ {cfg?.pyramid_d_threshold ?? 3}</code> AND current bar closes beyond,
                {' '}AND %K has left the OB/OS zone and returned to ≥{cfg?.stoch_ob ?? 80}/≤{cfg?.stoch_os ?? 20}.
                <br />
                <b>Path 2 — Safety wing-breach (price-based):</b> spot has breached half-way into the wing
                past the credit-spread short strike — i.e. <code>spot ≥ entry_atm + {((2 + (cfg?.pyramid_safety_wing_pct ?? 0.5)) * (cfg?.spread_width ?? 200)).toFixed(0)}</code> for long
                {' '}(or <code>≤ entry_atm − {((2 + (cfg?.pyramid_safety_wing_pct ?? 0.5)) * (cfg?.spread_width ?? 200)).toFixed(0)}</code> for short).
                Backup for the case where directional move continues without classic momentum signal — prevents the condor bleeding out.
                <br />
                On either trigger: add a NEW debit spread at current spot's ATM (level 2). Next CST after pyramid adds level-2 hedge.
                Capped at level {cfg?.pyramid_max_level} — further triggers logged only. Each pyramid event records
                <code>trigger_kind</code> = "d_cumulative_and_b" / "safety_wing_breach" / "both".
              </span>
            </div>
            <div className={styles.ruleItem}>
              <span className={styles.ruleLabel}>Exits</span>
              <span>
                T-1 close (typically Mon {cfg?.t_minus_1_close_hour}:{String(cfg?.t_minus_1_close_minute).padStart(2,'0')} IST)
                — close all legs, immediately rollover at current ATM if MST still active (level resets to L1).
                MST flip → close + re-arm. Kill switch → close + halt.
                T-1 day is computed from NSE trading calendar (handles holidays).
              </span>
            </div>
            <div className={styles.ruleItem}>
              <span className={styles.ruleLabel}>Mode switch</span>
              <span>
                <b>Off</b> = no signal evaluation, no entries.
                <b> Paper</b> = full signals + alerts + DB logging, simulated orders only.
                <b> Live</b> = real orders via Kite NRML.
                Use <b>Kill switch</b> to immediately close all legs at market + halt.
              </span>
            </div>
            <div className={styles.ruleItem}>
              <span className={styles.ruleLabel}>Backtest backing</span>
              <span>
                research/35 (signal selection on 2-yr, 252 ST cells); research/36 (CST continuation + pyramid trigger
                validated on 6.3-yr extended period: 302 trends, 1495 CSTs).
                <br />
                Pyramid: D_cumulative+B FP=13.2% (vs strict's 18.7%) with same 79% coverage of trend continuations.
                Safety wing-breach validated at <b>0% false-positive rate</b>, +2.1% incremental coverage,
                fires earlier than D_cumulative+B in 8.6% of cases.
              </span>
            </div>
          </div>
        </details>
      </section>
    </div>
  );
}
