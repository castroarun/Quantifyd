import { useCallback, useEffect, useMemo, useState } from 'react';
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

/* ---------- variant definitions (UI-side metadata, mirrors config.py) ---------- */

interface VariantDef {
  id: string;
  subtitle: string;
}

const GROUP_TF: VariantDef[] = [
  { id: 'or60-std', subtitle: 'OR60 Standard. Phase 1e champion baseline.' },
  { id: 'or45-std', subtitle: 'OR45 Standard. Earlier entry, ~same WR.' },
  { id: 'or30-std', subtitle: 'OR30 Standard. More theta time, lower WR.' },
  { id: 'or15-std', subtitle: 'OR15 Standard. High frequency, tighter SL.' },
  { id: 'or5-std', subtitle: 'OR5 Standard. Max theta runway, signal noise.' },
];

const GROUP_OR60_ALT: VariantDef[] = [
  { id: 'or60-norsi', subtitle: 'OR60 No-RSI. Tests if RSI confirmation is necessary.' },
  { id: 'or60-tight', subtitle: 'OR60 Tight RSI. RSI 65/35 — fewer better signals.' },
  { id: 'or60-calm',  subtitle: 'OR60 Calm-Only. Skip days where OR width >= 0.40% of spot.' },
];

const GROUP_CPR: VariantDef[] = [
  {
    id: 'or60-cpr-against',
    subtitle:
      'OR60 CPR-Against. Take only signals that have NOT cleared CPR (tame breakouts → friendlier theta capture). Backtest 95.5% WR.',
  },
  {
    id: 'or30-cpr-against',
    subtitle: 'OR30 CPR-Against. Earlier entry version. Backtest 93.1% WR.',
  },
];

const SUBTITLE_BY_ID: Record<string, string> = Object.fromEntries(
  [...GROUP_TF, ...GROUP_OR60_ALT, ...GROUP_CPR].map((v) => [v.id, v.subtitle]),
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

  const summaryById: Record<string, StrangleVariantSummary> = useMemo(() => {
    const m: Record<string, StrangleVariantSummary> = {};
    for (const v of state?.variants ?? []) m[v.id] = v;
    return m;
  }, [state]);

  // Aggregate today's P&L across all variants for the header tile.
  const totalDayPnl = useMemo(() => {
    return (state?.variants ?? []).reduce((acc, v) => acc + (v.today_pnl ?? 0), 0);
  }, [state]);

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
      <div className="page-title">Nifty ORB Strangle</div>
      <div className="page-subtitle">
        Ten paper-trading variants of the Nifty ORB short-strangle. One sells PE+CE
        deltas after the OR breakout; exits on spot SL, target, or 15:25 EOD.
      </div>

      {toast ? <div className={styles.toast}>{toast}</div> : null}

      {/* Top metrics */}
      <div className={styles.headerMetrics}>
        <MetricCard
          label="Nifty spot"
          value={spot != null ? formatNumber(spot) : '—'}
          hint={today ? `Today ${today}` : 'Live index price'}
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
              summary={summaryById[v.id]}
              detail={details[v.id]}
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
              summary={summaryById[v.id]}
              detail={details[v.id]}
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
              summary={summaryById[v.id]}
              detail={details[v.id]}
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
  summary: StrangleVariantSummary | undefined;
  detail: StrangleVariantDetail | null | undefined;
  onAction: () => void;
  onToast: (msg: string) => void;
}

function VariantCard({
  defId,
  subtitle,
  summary,
  detail,
  onAction,
  onToast,
}: VariantCardProps) {
  const [busy, setBusy] = useState<'scan' | 'close' | null>(null);

  const cfg = detail?.variant;
  const status = summary?.today_status ?? 'Idle';
  const todayPnl = summary?.today_pnl ?? 0;
  const allTimePnl = detail?.stats?.total_pnl ?? 0;
  const tradeCount = detail?.stats?.total_trades ?? 0;
  const enabled = summary?.enabled ?? true;
  const open = detail?.open_position;
  const mtm = detail?.mtm;
  const todayTradeCount = (detail?.recent_trades ?? []).filter(
    (t) => t.exit_date === detail?.daily_state?.trade_date,
  ).length;

  const configNote = cfg
    ? buildConfigNote(cfg)
    : `OR${summary?.or_min ?? '?'}m`;

  const backtestNote = cfg
    ? `Backtest: WR ${cfg.backtest_wr_pct}% · ${cfg.backtest_wins_per_year} wins/yr · ${cfg.backtest_trades_per_year} trades/yr`
    : '';

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
          {backtestNote ? (
            <div className={styles.cardBacktest}>{backtestNote}</div>
          ) : null}
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
