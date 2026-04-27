import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import styles from './Strangle.module.css';
import { apiGet, apiPost } from '../api/client';
import type {
  StrangleState,
  StrangleVariantSummary,
  StrangleVariantDetail,
  StrangleTrade,
} from '../api/types';
import MetricCard from '../components/Cards/MetricCard';
import Chip from '../components/Chip/Chip';
import DataTable, { type Column } from '../components/DataTable/DataTable';
import {
  formatInt,
  formatNumber,
  formatPnl,
  formatPct,
  pnlClass,
} from '../utils/format';

/* ---------- live ticks (SSE) ---------- */

interface LiveVariantTick {
  pe_now: number | null;
  ce_now: number | null;
  pe_mtm: number | null;
  ce_mtm: number | null;
  net_mtm: number | null;
}

interface LiveTicks {
  spot: number | null;
  variants: Record<string, LiveVariantTick>;
  connected: boolean;
}

/* ---------- variant definitions (UI-side metadata, mirrors config.py) ---------- */

interface VariantDef {
  id: string;
  subtitle: string;
  rules: string;
}

const GROUP_TF: VariantDef[] = [
  {
    id: 'or60-std',
    subtitle: 'OR60 Standard. Phase 1e champion baseline.',
    rules:
      'Entry: 60-min OR break (close > OR_high or < OR_low) after 10:15 IST + 5-min RSI > 60 (long) / < 40 (short). Sell PE @ -0.22δ + CE @ +0.10δ on nearest weekly Tuesday expiry (incl DTE=0). SL: opposite OR boundary (wick-based, OR-width × 1.0). Skip Q4 days (OR width > 0.67% of spot). EOD square-off 15:25.',
  },
  {
    id: 'or45-std',
    subtitle: 'OR45 Standard. Earlier entry, ~same WR.',
    rules:
      'Entry: 45-min OR break after 10:00 + 5-min RSI 60/40. Same delta-skewed strangle (PE -0.22δ + CE +0.10δ) on nearest weekly Tuesday. SL = opposite OR (wick). Skip Q4. EOD 15:25.',
  },
  {
    id: 'or30-std',
    subtitle: 'OR30 Standard. More theta time, lower WR.',
    rules:
      'Entry: 30-min OR break after 9:45 + 5-min RSI 60/40. Same strikes (PE -0.22δ + CE +0.10δ) / SL / exit / Q4 filter as OR60-Std.',
  },
  {
    id: 'or15-std',
    subtitle: 'OR15 Standard. High frequency, tighter SL.',
    rules:
      'Entry: 15-min OR break after 9:30 + 5-min RSI 60/40. Same strikes / SL / exit / Q4 filter as OR60-Std. SL distance scales with the day\'s OR15 width.',
  },
  {
    id: 'or5-std',
    subtitle: 'OR5 Standard. Max theta runway, signal noise.',
    rules:
      'Entry: 5-min OR break after 9:20 + 5-min RSI 60/40. Same strikes / SL / exit / Q4 filter. Earliest signal — captures most intraday theta but on a noisy 1-bar OR.',
  },
];

const GROUP_OR60_ALT: VariantDef[] = [
  {
    id: 'or60-norsi',
    subtitle: 'OR60 No-RSI. Tests if RSI confirmation is necessary.',
    rules:
      'Entry: 60-min OR break after 10:15 — NO RSI filter, take any first break in either direction. Same strikes (PE -0.22δ + CE +0.10δ) / SL / exit / Q4 filter as OR60-Std.',
  },
  {
    id: 'or60-tight',
    subtitle: 'OR60 Tight RSI. RSI 65/35 — fewer better signals.',
    rules:
      'Entry: 60-min OR break + 5-min RSI > 65 (long) / < 35 (short) — tighter than std 60/40. Same strikes / SL / exit / Q4 filter. Trades fewer signals at slightly higher WR.',
  },
  {
    id: 'or60-calm',
    subtitle: 'OR60 Calm-Only. Skip days where OR width >= 0.40% of spot.',
    rules:
      'Entry: 60-min OR break + 5-min RSI 60/40, ONLY on calm days where OR60 width < 0.40% of spot (additional filter on top of universal Q4). Same strikes / SL / exit. Targets the high-WR Q1 quartile from the backtest — fewer trades, much higher WR.',
  },
];

const GROUP_CPR: VariantDef[] = [
  {
    id: 'or60-cpr-against',
    subtitle:
      'OR60 CPR-Against. Take only signals that have NOT cleared CPR (tame breakouts → friendlier theta capture).',
    rules:
      'Entry: 60-min OR break + 5-min RSI 60/40 + entry close has NOT cleared the CPR zone in its direction. For LONG: close ≤ CPR_high. For SHORT: close ≥ CPR_low. CPR computed from prior trading day\'s daily HLC (P=(H+L+C)/3, BC=(H+L)/2, TC=2P-BC). Hypothesis: tame breakouts that haven\'t escaped CPR are friendlier to short-strangle theta than fully extended trend breaks. Same strikes / SL / exit / Q4 filter.',
  },
  {
    id: 'or30-cpr-against',
    subtitle: 'OR30 CPR-Against. Earlier entry version.',
    rules:
      'Entry: 30-min OR break + 5-min RSI 60/40 + entry close has NOT cleared CPR zone (against-CPR filter, same logic as V9 but on the 30-min OR). Same strikes / SL / exit / Q4 filter.',
  },
];

const ALL_VARIANTS = [...GROUP_TF, ...GROUP_OR60_ALT, ...GROUP_CPR];

const SUBTITLE_BY_ID: Record<string, string> = Object.fromEntries(
  ALL_VARIANTS.map((v) => [v.id, v.subtitle]),
);

const RULES_BY_ID: Record<string, string> = Object.fromEntries(
  ALL_VARIANTS.map((v) => [v.id, v.rules]),
);

/* ---------- page ---------- */

export default function Strangle() {
  const [state, setState] = useState<StrangleState | null>(null);
  const [details, setDetails] = useState<Record<string, StrangleVariantDetail | null>>({});
  const [trades, setTrades] = useState<StrangleTrade[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [toast, setToast] = useState<string | null>(null);
  const [liveTicks, setLiveTicks] = useState<LiveTicks>({
    spot: null,
    variants: {},
    connected: false,
  });
  const evtRef = useRef<EventSource | null>(null);

  const showToast = useCallback((msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 2500);
  }, []);

  const loadAll = useCallback(async () => {
    setRefreshing(true);
    try {
      const top = await apiGet<StrangleState>('/api/strangle/state');
      setState(top);
      setErr(null);

      // Fetch detail for each variant (parallel).
      const detailEntries = await Promise.all(
        top.variants.map(async (v) => {
          try {
            const d = await apiGet<StrangleVariantDetail>(
              `/api/strangle/variant/${v.id}`,
            );
            return [v.id, d] as const;
          } catch {
            return [v.id, null] as const;
          }
        }),
      );
      const map: Record<string, StrangleVariantDetail | null> = {};
      for (const [id, d] of detailEntries) map[id] = d;
      setDetails(map);

      // Build merged recent-trades list (newest first, capped at 30).
      const merged: StrangleTrade[] = [];
      for (const [, d] of detailEntries) {
        if (d?.recent_trades) merged.push(...d.recent_trades);
      }
      merged.sort((a, b) => (b.exit_ts ?? '').localeCompare(a.exit_ts ?? ''));
      setTrades(merged.slice(0, 30));

      setLastUpdated(new Date());
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Load failed';
      setErr(msg);
    } finally {
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    void loadAll();
    const id = setInterval(() => void loadAll(), 30_000);
    return () => clearInterval(id);
  }, [loadAll]);

  // SSE: live tick stream of spot + per-variant leg LTPs/MTM. Mirrors NAS pattern.
  useEffect(() => {
    let cancelled = false;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    const open = () => {
      if (cancelled) return;
      const es = new EventSource('/api/strangle/stream');
      evtRef.current = es;
      es.onopen = () => {
        if (!cancelled) setLiveTicks((prev) => ({ ...prev, connected: true }));
      };
      es.onmessage = (ev) => {
        if (cancelled) return;
        try {
          const d = JSON.parse(ev.data);
          if (d.type === 'tick') {
            setLiveTicks({
              spot: typeof d.spot === 'number' ? d.spot : null,
              variants: (d.variants ?? {}) as Record<string, LiveVariantTick>,
              connected: true,
            });
          } else if (d.type === 'offline') {
            setLiveTicks((p) => ({ ...p, connected: false }));
          }
        } catch {
          /* ignore malformed payload */
        }
      };
      es.onerror = () => {
        if (cancelled) return;
        es.close();
        evtRef.current = null;
        setLiveTicks((p) => ({ ...p, connected: false }));
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

  const summaryById: Record<string, StrangleVariantSummary> = useMemo(() => {
    const m: Record<string, StrangleVariantSummary> = {};
    for (const v of state?.variants ?? []) m[v.id] = v;
    return m;
  }, [state]);

  // Aggregate today's P&L across all variants for the header tile.
  // For variants with an open position we prefer the live SSE net_mtm
  // (≤2s stale) over the polled today_pnl (≤60s stale, only updates when
  // master_tick refreshes leg MTM in DB).
  const totalDayPnl = useMemo(() => {
    return (state?.variants ?? []).reduce((acc, v) => {
      if (v.open_positions && liveTicks.variants[v.id]?.net_mtm != null) {
        return acc + (liveTicks.variants[v.id].net_mtm ?? 0);
      }
      return acc + (v.today_pnl ?? 0);
    }, 0);
  }, [state, liveTicks]);

  // Total open positions across all variants.
  const totalOpen = useMemo(() => {
    return (state?.variants ?? []).reduce((acc, v) => acc + (v.open_positions ?? 0), 0);
  }, [state]);

  const today = state?.today ?? '';
  const spot = state?.spot_ltp ?? null;

  // Derive current ORB widths per OR window from the daily_state of the
  // representative variants (OR60-std, OR45-std, OR30-std, OR15-std, OR5-std).
  const orWidths = useMemo(() => {
    const out: { window: number; width_pct: number | null }[] = [];
    for (const id of ['or60-std', 'or45-std', 'or30-std', 'or15-std', 'or5-std']) {
      const d = details[id];
      const ds = d?.daily_state as { or_width_pct?: number | null } | undefined;
      const om = d?.variant?.or_min;
      if (om != null) {
        const w = ds && typeof ds.or_width_pct === 'number' ? ds.or_width_pct : null;
        out.push({ window: om, width_pct: w });
      }
    }
    return out;
  }, [details]);

  return (
    <div className={styles.root}>
      <div className="page-title">ORB Index</div>
      <div className="page-subtitle">
        Ten paper-trading variants of the Nifty ORB short-strangle. One sells PE+CE
        deltas after the OR breakout; exits on spot SL, target, or 15:25 EOD.
      </div>

      {toast ? <div className={styles.toast}>{toast}</div> : null}

      {/* Top metrics */}
      <div className={styles.headerMetrics}>
        <MetricCard
          label="Nifty spot"
          value={
            liveTicks.spot != null
              ? formatNumber(liveTicks.spot)
              : spot != null
                ? formatNumber(spot)
                : '—'
          }
          hint={
            liveTicks.connected
              ? `Live · ${today}`
              : today
                ? `Today ${today}`
                : 'Live index price'
          }
        />
        <MetricCard
          label="Open positions"
          value={formatInt(totalOpen)}
          hint={`Across ${state?.variants?.length ?? 10} variants`}
        />
        <MetricCard
          label="Day P&L (all variants)"
          value={
            <span className={pnlClass(totalDayPnl)}>{formatPnl(totalDayPnl)}</span>
          }
          hint="Sum of today's net P&L across variants"
        />
        <MetricCard
          label="Last refresh"
          value={
            <span className={styles.headerRight}>
              <button
                className={styles.refreshBtn}
                onClick={() => void loadAll()}
                disabled={refreshing}
              >
                {refreshing ? 'Refreshing…' : 'Refresh'}
              </button>
            </span>
          }
          hint={
            <span className={styles.lastUpdated}>
              {lastUpdated ? lastUpdated.toLocaleTimeString() : 'Loading…'}
            </span>
          }
        />
      </div>

      {/* Day-state summary */}
      <div className={styles.dayBanner}>
        {orWidths.map((o) => (
          <div key={o.window}>
            <span className={styles.dayBannerLabel}>OR{o.window} width:</span>
            <span className={styles.dayBannerValue}>
              {o.width_pct != null ? `${o.width_pct.toFixed(2)}%` : '—'}
            </span>
          </div>
        ))}
      </div>

      {err ? <div className={styles.errRow}>{err}</div> : null}

      {/* Group 1: OR-window timeframe sweep */}
      <section className={styles.group}>
        <div className={styles.groupHead}>
          <div>
            <div className={styles.groupTitle}>OR-window timeframe sweep</div>
            <div className={styles.groupSub}>
              Same rules, varying OR length. RSI 60/40 confirmation, Q4 daily filter.
            </div>
          </div>
          <Chip>{GROUP_TF.length} variants</Chip>
        </div>
        <div className={styles.cardGrid}>
          {GROUP_TF.map((v) => (
            <VariantCard
              key={v.id}
              defId={v.id}
              subtitle={SUBTITLE_BY_ID[v.id]}
              rules={RULES_BY_ID[v.id]}
              summary={summaryById[v.id]}
              detail={details[v.id]}
              liveTick={liveTicks.variants[v.id]}
              onAction={() => void loadAll()}
              onToast={showToast}
            />
          ))}
        </div>
      </section>

      {/* Group 2: OR60 alternates */}
      <section className={styles.group}>
        <div className={styles.groupHead}>
          <div>
            <div className={styles.groupTitle}>OR60 alternates</div>
            <div className={styles.groupSub}>
              Variations on OR60: drop RSI, tighten RSI, or only take calm-OR days.
            </div>
          </div>
          <Chip>{GROUP_OR60_ALT.length} variants</Chip>
        </div>
        <div className={styles.cardGrid}>
          {GROUP_OR60_ALT.map((v) => (
            <VariantCard
              key={v.id}
              defId={v.id}
              subtitle={SUBTITLE_BY_ID[v.id]}
              rules={RULES_BY_ID[v.id]}
              summary={summaryById[v.id]}
              detail={details[v.id]}
              liveTick={liveTicks.variants[v.id]}
              onAction={() => void loadAll()}
              onToast={showToast}
            />
          ))}
        </div>
      </section>

      {/* Group 3: CPR-against */}
      <section className={styles.group}>
        <div className={styles.groupHead}>
          <div>
            <div className={styles.groupTitle}>CPR-against (high-WR low-volume)</div>
            <div className={styles.groupSub}>
              Take only breakouts that have NOT cleared CPR — tame moves friendlier
              to short-strangle theta capture.
            </div>
          </div>
          <Chip>{GROUP_CPR.length} variants</Chip>
        </div>
        <div className={styles.cardGrid}>
          {GROUP_CPR.map((v) => (
            <VariantCard
              key={v.id}
              defId={v.id}
              subtitle={SUBTITLE_BY_ID[v.id]}
              rules={RULES_BY_ID[v.id]}
              summary={summaryById[v.id]}
              detail={details[v.id]}
              liveTick={liveTicks.variants[v.id]}
              onAction={() => void loadAll()}
              onToast={showToast}
            />
          ))}
        </div>
      </section>

      {/* Recent trades across all variants */}
      <section className={styles.tradesSection}>
        <details>
          <summary className={styles.tradesHead}>
            <div className={styles.groupTitle}>Recent trades · all variants</div>
            <Chip>{trades.length} rows</Chip>
          </summary>
          <RecentTradesTable trades={trades} />
        </details>
      </section>
    </div>
  );
}

/* ---------- variant card ---------- */

interface VariantCardProps {
  defId: string;
  subtitle: string;
  rules: string;
  summary: StrangleVariantSummary | undefined;
  detail: StrangleVariantDetail | null | undefined;
  liveTick: LiveVariantTick | undefined;
  onAction: () => void;
  onToast: (msg: string) => void;
}

function VariantCard({
  defId,
  subtitle,
  rules,
  summary,
  detail,
  liveTick,
  onAction,
  onToast,
}: VariantCardProps) {
  const [busy, setBusy] = useState<'scan' | 'close' | null>(null);

  const cfg = detail?.variant;
  const status = summary?.today_status ?? 'Idle';
  const allTimePnl = detail?.stats?.total_pnl ?? 0;
  const tradeCount = detail?.stats?.total_trades ?? 0;
  const enabled = summary?.enabled ?? true;
  const open = detail?.open_position;
  // Prefer live SSE tick (≤2s stale); fall back to polled DB MTM (≤60s stale).
  const mtm = liveTick ?? detail?.mtm;
  // Today's P&L: if a live tick is present and a position is open, prefer
  // live net_mtm so the card updates intra-tick. Otherwise use the polled
  // summary value (which only refreshes when master_tick or close-position runs).
  const todayPnl =
    open && liveTick?.net_mtm != null ? liveTick.net_mtm : summary?.today_pnl ?? 0;
  const todayTradeCount = (detail?.recent_trades ?? []).filter(
    (t) => t.exit_date === detail?.daily_state?.trade_date,
  ).length;

  const configNote = cfg
    ? buildConfigNote(cfg)
    : `OR${summary?.or_min ?? '?'}m`;

  async function runScan() {
    setBusy('scan');
    try {
      await apiPost(`/api/strangle/scan/${defId}`);
      onToast(`${cfg?.name ?? defId}: scan triggered`);
      onAction();
    } catch (e) {
      onToast(`Scan failed: ${e instanceof Error ? e.message : 'unknown'}`);
    } finally {
      setBusy(null);
    }
  }

  async function runClose() {
    if (!open) return;
    if (!confirm(`Force-close open position for ${cfg?.name ?? defId}?`)) return;
    setBusy('close');
    try {
      await apiPost(`/api/strangle/close/${defId}`);
      onToast(`${cfg?.name ?? defId}: close issued`);
      onAction();
    } catch (e) {
      onToast(`Close failed: ${e instanceof Error ? e.message : 'unknown'}`);
    } finally {
      setBusy(null);
    }
  }

  return (
    <div className={styles.card}>
      <div className={styles.cardHead}>
        <div className={styles.cardHeadLeft}>
          <div className={styles.cardTitle}>{cfg?.name ?? summary?.name ?? defId}</div>
          <div className={styles.cardSub}>{subtitle}</div>
          <div className={styles.cardConfig}>{configNote}</div>
        </div>
        <div className={styles.cardStatus}>
          <StatusChip status={status} />
          {!enabled ? (
            <span className={styles.disabledLabel}>disabled</span>
          ) : null}
        </div>
      </div>

      <div className={styles.metricsRow}>
        <MiniMetric label="Status" value={status} />
        <MiniMetric
          label="Today's P&L"
          value={
            <span className={pnlClass(todayPnl)}>{formatPnl(todayPnl)}</span>
          }
        />
        <MiniMetric label="Trades today" value={formatInt(todayTradeCount)} />
        <MiniMetric
          label="All-time P&L"
          value={
            <span className={pnlClass(allTimePnl)}>{formatPnl(allTimePnl)}</span>
          }
        />
      </div>

      {open ? (
        <div className={styles.openPanel}>
          <div className={styles.openHead}>
            <span>
              Open · {open.direction} · entered {formatLegTime(open.entry_ts)}
            </span>
            <span className={styles.openMeta}>
              {tradeCount} all-time trades
              {detail?.stats?.win_rate != null
                ? ` · WR ${formatPct(detail.stats.win_rate, 1)}`
                : ''}
            </span>
          </div>
          <LegLine
            type="PE"
            strike={open.pe_strike}
            qty={open.qty}
            entry={open.pe_entry_price}
            now={mtm?.pe_now ?? null}
            pnl={mtm?.pe_mtm ?? null}
          />
          <LegLine
            type="CE"
            strike={open.ce_strike}
            qty={open.qty}
            entry={open.ce_entry_price}
            now={mtm?.ce_now ?? null}
            pnl={mtm?.ce_mtm ?? null}
          />
        </div>
      ) : null}

      <div className={styles.actions}>
        <button
          className={styles.actionBtn}
          onClick={() => void runScan()}
          disabled={busy !== null || !enabled}
        >
          {busy === 'scan' ? 'Scanning…' : 'Manual Scan'}
        </button>
        <button
          className={`${styles.actionBtn} ${styles.actionBtnDanger}`}
          onClick={() => void runClose()}
          disabled={busy !== null || !open}
        >
          {busy === 'close' ? 'Closing…' : 'Force Close'}
        </button>
      </div>

      <details className={styles.rules}>
        <summary className={styles.rulesSummary}>Rules &amp; backtest</summary>
        <div className={styles.rulesBody}>
          <div className={styles.snapshotRow}>
            <div className={styles.snapshotItem}>
              <span className={styles.snapshotLabel}>Backtest WR</span>
              <span className={styles.snapshotValue}>
                {cfg?.backtest_wr_pct != null ? `${cfg.backtest_wr_pct}%` : '—'}
              </span>
            </div>
            <div className={styles.snapshotItem}>
              <span className={styles.snapshotLabel}>Wins / year</span>
              <span className={styles.snapshotValue}>
                {cfg?.backtest_wins_per_year != null
                  ? formatInt(cfg.backtest_wins_per_year)
                  : '—'}
              </span>
            </div>
            <div className={styles.snapshotItem}>
              <span className={styles.snapshotLabel}>Trades / year</span>
              <span className={styles.snapshotValue}>
                {cfg?.backtest_trades_per_year != null
                  ? formatInt(cfg.backtest_trades_per_year)
                  : '—'}
              </span>
            </div>
            <div className={styles.snapshotItem}>
              <span className={styles.snapshotLabel}>Live WR</span>
              <span className={styles.snapshotValue}>
                {detail?.stats?.win_rate != null
                  ? formatPct(detail.stats.win_rate, 1)
                  : '—'}
              </span>
            </div>
          </div>
          <div className={styles.rulesText}>{rules}</div>
        </div>
      </details>
    </div>
  );
}

function buildConfigNote(cfg: NonNullable<StrangleVariantDetail['variant']>): string {
  const parts: string[] = [`OR${cfg.or_min}m`];

  if (cfg.rsi_lo_long != null && cfg.rsi_hi_short != null) {
    parts.push(`RSI ${cfg.rsi_lo_long}/${cfg.rsi_hi_short}`);
  } else {
    parts.push('No RSI');
  }

  const filters: string[] = [];
  if (cfg.apply_q4_filter) filters.push('Q4');
  if (cfg.apply_calm_filter) filters.push('Calm');
  if (cfg.apply_cpr_against_filter) filters.push('CPR-against');
  if (filters.length) parts.push(filters.join('+'));

  parts.push(`Lot ${cfg.lot_size}`);
  return parts.join(' | ');
}

function StatusChip({ status }: { status: string }) {
  const cls =
    status === 'Open'
      ? styles.chipOpen
      : status === 'Watching'
      ? styles.chipWatching
      : status === 'Closed'
      ? styles.chipClosed
      : status === 'Skip'
      ? styles.chipSkip
      : styles.chipIdle;
  return <span className={cls}>{status}</span>;
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

function LegLine({
  type,
  strike,
  qty,
  entry,
  now,
  pnl,
}: {
  type: 'PE' | 'CE';
  strike: number;
  qty: number;
  entry: number;
  now: number | null;
  pnl: number | null;
}) {
  return (
    <div className={styles.legRow}>
      <span className={styles.legType}>{type}</span>
      <span className={styles.legStrike}>
        {strike} ×{qty}
      </span>
      <span className={styles.legNum}>{formatNumber(entry)}</span>
      <span className={styles.legArrow}>→</span>
      <span className={styles.legNum}>{now != null ? formatNumber(now) : '—'}</span>
      <span className={`${styles.legPnl} ${pnlClass(pnl)}`}>
        {formatPnl(pnl)}
      </span>
    </div>
  );
}

function formatLegTime(iso?: string | null): string {
  if (!iso) return '—';
  const m = /T(\d{2}:\d{2})/.exec(iso) || /\s(\d{2}:\d{2})/.exec(iso);
  return m ? m[1] : iso.slice(0, 16);
}

/* ---------- recent trades table ---------- */

function RecentTradesTable({ trades }: { trades: StrangleTrade[] }) {
  const columns: Column<StrangleTrade>[] = [
    {
      key: 'variant',
      header: 'Variant',
      width: '1.4fr',
      render: (r) => <span style={{ color: 'var(--ink)' }}>{r.variant_id}</span>,
    },
    {
      key: 'entry_date',
      header: 'Entry',
      width: '110px',
      render: (r) => r.entry_date,
    },
    {
      key: 'direction',
      header: 'Dir',
      width: '60px',
      render: (r) => r.direction ?? '—',
    },
    {
      key: 'pe',
      header: 'PE strike',
      width: '90px',
      align: 'right',
      render: (r) => (r.pe_strike != null ? formatInt(r.pe_strike) : '—'),
    },
    {
      key: 'ce',
      header: 'CE strike',
      width: '90px',
      align: 'right',
      render: (r) => (r.ce_strike != null ? formatInt(r.ce_strike) : '—'),
    },
    {
      key: 'reason',
      header: 'Exit',
      width: '110px',
      render: (r) =>
        r.exit_reason ? (
          <span className={styles.exitReason}>{r.exit_reason}</span>
        ) : (
          '—'
        ),
    },
    {
      key: 'pnl',
      header: 'Net P&L',
      width: '110px',
      align: 'right',
      render: (r) => (
        <span className={pnlClass(r.net_pnl)}>{formatPnl(r.net_pnl)}</span>
      ),
    },
  ];

  return (
    <DataTable<StrangleTrade>
      columns={columns}
      rows={trades}
      emptyText="No closed trades yet."
      rowKey={(r, i) => r.id ?? i}
      rowClassName={() => styles.tradeRowClosed}
    />
  );
}
