import { useEffect, useMemo, useState } from 'react';
import styles from './Orb.module.css';
import { apiGet, apiPost } from '../api/client';
import type {
  ORBState,
  ORBStockSummary,
  ORBPosition,
  ORBClosedTrade,
  ORBSignal,
} from '../api/types';
import MetricCard from '../components/Cards/MetricCard';
import DataTable from '../components/DataTable/DataTable';
import type { Column } from '../components/DataTable/DataTable';
import Chip from '../components/Chip/Chip';
import {
  formatInt,
  formatNumber,
  formatPnl,
  formatRs,
  pnlClass,
} from '../utils/format';
import { formatTime } from '../utils/time';
import { IconPlay, IconRefresh, IconPower } from '../components/Icons';

/* ---------- helpers ---------- */

function stockStatus(s: ORBStockSummary): string {
  if (s.position) return 'In position';
  if (s.today_result) return 'Traded';
  const ds = s.daily_state;
  if (ds?.is_wide_cpr_day) return 'Wide CPR';
  if (ds?.or_finalized) return 'Scanning';
  if (ds?.today_open) return 'Waiting for OR';
  return 'Idle';
}

function statusTone(label: string): 'neutral' | 'pos' | 'neg' {
  if (label === 'In position' || label === 'Scanning') return 'pos';
  if (label === 'Wide CPR') return 'neg';
  return 'neutral';
}

interface FilterDot {
  label: string;
  tone: 'pos' | 'neg' | 'neutral';
  title: string;
}

function filterDots(s: ORBStockSummary): FilterDot[] {
  const ds = s.daily_state || ({} as ORBStockSummary['daily_state']);
  const dots: FilterDot[] = [];

  // VWAP
  if (ds?.vwap && s.price) {
    const above = s.price > ds.vwap;
    dots.push({
      label: 'VWAP',
      tone: above ? 'pos' : 'neg',
      title: `VWAP ${formatNumber(ds.vwap)} — price ${above ? 'above' : 'below'}`,
    });
  } else {
    dots.push({ label: 'VWAP', tone: 'neutral', title: 'VWAP pending' });
  }

  // RSI
  if (typeof ds?.rsi === 'number') {
    const rsi = ds.rsi;
    const tone = rsi > 55 ? 'pos' : rsi < 45 ? 'neg' : 'neutral';
    dots.push({ label: 'RSI', tone, title: `RSI ${formatNumber(rsi)}` });
  } else {
    dots.push({ label: 'RSI', tone: 'neutral', title: 'RSI pending' });
  }

  // CPR direction
  if (ds?.cpr_dir) {
    const tone = ds.cpr_dir === 'BULL' ? 'pos' : ds.cpr_dir === 'BEAR' ? 'neg' : 'neutral';
    dots.push({ label: 'CPR', tone, title: `CPR ${ds.cpr_dir}` });
  } else {
    dots.push({ label: 'CPR', tone: 'neutral', title: 'CPR pending' });
  }

  return dots;
}

/* ---------- page ---------- */

export default function Orb() {
  const [state, setState] = useState<ORBState | null>(null);
  const [signals, setSignals] = useState<ORBSignal[]>([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const [s, sig] = await Promise.all([
          apiGet<ORBState>('/api/orb/state'),
          apiGet<ORBSignal[]>('/api/orb/signals').catch(() => [] as ORBSignal[]),
        ]);
        if (cancelled) return;
        setState(s);
        setSignals(Array.isArray(sig) ? sig : []);
        setErr(null);
      } catch (e) {
        if (!cancelled) setErr(e instanceof Error ? e.message : 'Failed to load ORB state');
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    load();
    const id = setInterval(load, 10_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  async function onInitialize() {
    setBusy('init');
    setErr(null);
    try {
      await apiPost('/api/orb/initialize');
      setToast('Day initialized');
    } catch (e) {
      setErr(e instanceof Error ? e.message : 'Initialize failed');
    } finally {
      setBusy(null);
      setTimeout(() => setToast(null), 2500);
    }
  }

  async function onScan() {
    setBusy('scan');
    setErr(null);
    try {
      await apiPost('/api/orb/scan');
      setToast('Scan queued');
    } catch (e) {
      setErr(e instanceof Error ? e.message : 'Scan failed');
    } finally {
      setBusy(null);
      setTimeout(() => setToast(null), 2500);
    }
  }

  async function onKill() {
    if (!confirm('Close all open ORB positions now?')) return;
    setBusy('kill');
    setErr(null);
    try {
      await apiPost('/api/orb/kill-switch');
      setToast('Kill switch triggered');
    } catch (e) {
      setErr(e instanceof Error ? e.message : 'Kill switch failed');
    } finally {
      setBusy(null);
      setTimeout(() => setToast(null), 2500);
    }
  }

  const stocksArr = useMemo(() => {
    if (!state?.stocks) return [];
    return Object.values(state.stocks).sort((a, b) =>
      a.daily_state.instrument.localeCompare(b.daily_state.instrument),
    );
  }, [state]);

  const marginAvail = (state?.margin?.available as number | undefined) ?? 0;

  /* ---------- table columns ---------- */

  const positionCols: Column<ORBPosition>[] = [
    {
      key: 'instrument',
      header: 'Stock',
      width: '1.2fr',
      render: (p) => <span className={styles.bold}>{p.instrument}</span>,
    },
    {
      key: 'direction',
      header: 'Direction',
      width: '90px',
      render: (p) => (
        <span className={p.direction === 'LONG' ? styles.dirLong : styles.dirShort}>
          {p.direction === 'LONG' ? 'Long' : 'Short'}
        </span>
      ),
    },
    { key: 'qty', header: 'Qty', width: '70px', align: 'right', render: (p) => formatInt(p.qty) },
    {
      key: 'entry',
      header: 'Entry',
      width: '1fr',
      align: 'right',
      render: (p) => formatNumber(p.entry_price),
    },
    {
      key: 'sl',
      header: 'Stop',
      width: '1fr',
      align: 'right',
      render: (p) => formatNumber(p.stop_loss),
    },
    {
      key: 'target',
      header: 'Target',
      width: '1fr',
      align: 'right',
      render: (p) => formatNumber(p.target),
    },
    {
      key: 'ltp',
      header: 'LTP',
      width: '1fr',
      align: 'right',
      render: (p) => formatNumber(p.ltp),
    },
    {
      key: 'pnl',
      header: 'P&L',
      width: '1.1fr',
      align: 'right',
      render: (p) => (
        <span className={pnlClass(p.pnl_inr)}>{formatPnl(p.pnl_inr)}</span>
      ),
    },
    {
      key: 'time',
      header: 'Entry time',
      width: '90px',
      align: 'right',
      render: (p) => (
        <span className={styles.mute}>{formatTime(p.entry_time)}</span>
      ),
    },
  ];

  const tradeCols: Column<ORBClosedTrade>[] = [
    {
      key: 'instrument',
      header: 'Stock',
      width: '1.2fr',
      render: (t) => <span className={styles.bold}>{t.instrument}</span>,
    },
    {
      key: 'direction',
      header: 'Direction',
      width: '90px',
      render: (t) => (t.direction === 'LONG' ? 'Long' : 'Short'),
    },
    { key: 'qty', header: 'Qty', width: '70px', align: 'right', render: (t) => formatInt(t.qty) },
    {
      key: 'entry',
      header: 'Entry',
      width: '1fr',
      align: 'right',
      render: (t) => formatNumber(t.entry_price),
    },
    {
      key: 'exit',
      header: 'Exit',
      width: '1fr',
      align: 'right',
      render: (t) => formatNumber(t.exit_price),
    },
    {
      key: 'reason',
      header: 'Exit reason',
      width: '1.4fr',
      render: (t) => <span className={styles.mute}>{t.exit_reason || '—'}</span>,
    },
    {
      key: 'pnl',
      header: 'P&L',
      width: '1.1fr',
      align: 'right',
      render: (t) => (
        <span className={pnlClass(t.pnl_inr)}>{formatPnl(t.pnl_inr)}</span>
      ),
    },
    {
      key: 'time',
      header: 'Exit time',
      width: '90px',
      align: 'right',
      render: (t) => <span className={styles.mute}>{formatTime(t.exit_time)}</span>,
    },
  ];

  const signalCols: Column<ORBSignal>[] = [
    {
      key: 'time',
      header: 'Time',
      width: '80px',
      render: (s) => <span className={styles.mute}>{formatTime(s.signal_time)}</span>,
    },
    {
      key: 'instrument',
      header: 'Stock',
      width: '1.2fr',
      render: (s) => <span className={styles.bold}>{s.instrument}</span>,
    },
    {
      key: 'direction',
      header: 'Direction',
      width: '90px',
      render: (s) => s.direction ?? '—',
    },
    {
      key: 'price',
      header: 'Price',
      width: '1fr',
      align: 'right',
      render: (s) => formatNumber(s.price),
    },
    {
      key: 'reason',
      header: 'Reason',
      width: '1.8fr',
      render: (s) => <span className={styles.mute}>{s.reason || '—'}</span>,
    },
    {
      key: 'action',
      header: 'Action',
      width: '1.2fr',
      render: (s) => <span className={styles.mute}>{s.action_taken || s.status || '—'}</span>,
    },
  ];

  /* ---------- render ---------- */

  if (loading) {
    return (
      <div className={styles.loading}>Loading ORB state…</div>
    );
  }

  return (
    <div className={styles.root}>
      <div className={styles.headerRow}>
        <div>
          <div className="page-title">Opening range breakout</div>
          <div className="page-subtitle">
            Cash intraday · {state?.universe?.length ?? 0} stocks ·{' '}
            {state?.live_trading ? 'Live trading' : 'Paper trading'}
          </div>
        </div>
        <div className={styles.actions}>
          <button
            className={styles.btn}
            onClick={onInitialize}
            disabled={busy !== null}
          >
            <IconRefresh size={14} />
            <span>Initialize day</span>
          </button>
          <button className={styles.btn} onClick={onScan} disabled={busy !== null}>
            <IconPlay size={14} />
            <span>Scan now</span>
          </button>
          <button
            className={`${styles.btn} ${styles.btnDanger}`}
            onClick={onKill}
            disabled={busy !== null}
          >
            <IconPower size={14} />
            <span>Kill switch</span>
          </button>
        </div>
      </div>

      {err ? <div className={styles.error}>{err}</div> : null}
      {toast ? <div className={styles.toast}>{toast}</div> : null}

      {/* summary metrics */}
      <div className={styles.metrics}>
        <MetricCard
          label="Day P&L"
          value={
            <span className={pnlClass(state?.today_pnl)}>
              {formatPnl(state?.today_pnl)}
            </span>
          }
          hint={`Daily limit ${formatRs(state?.daily_loss_limit)}`}
        />
        <MetricCard
          label="Open positions"
          value={formatInt(state?.open_positions?.length ?? 0)}
          hint={`Max ${state?.config?.max_concurrent_trades ?? '—'} concurrent`}
        />
        <MetricCard
          label="Available margin"
          value={formatRs(marginAvail)}
          hint={`Per trade ${formatRs(state?.config?.allocation_per_trade)}`}
        />
        <MetricCard
          label="Trades today"
          value={formatInt(state?.today_closed?.length ?? 0)}
          hint={`${state?.universe?.length ?? 0} stocks scanning`}
        />
      </div>

      {/* open positions */}
      <section className={styles.section}>
        <div className={styles.sectionHead}>
          <div className="section-title">Open positions</div>
          <Chip>
            {state?.open_positions?.length ?? 0} active
          </Chip>
        </div>
        <DataTable
          columns={positionCols}
          rows={state?.open_positions ?? []}
          emptyText="No open positions"
          rowKey={(p) => p.instrument + (p.entry_time ?? '')}
        />
      </section>

      {/* stock scan grid */}
      <section className={styles.section}>
        <div className={styles.sectionHead}>
          <div className="section-title">Stock scan</div>
          <Chip>{stocksArr.length} stocks</Chip>
        </div>
        <div className={styles.stockGrid}>
          {stocksArr.map((s) => (
            <StockCard key={s.daily_state.instrument} stock={s} />
          ))}
          {stocksArr.length === 0 ? (
            <div className={styles.empty}>
              No stocks loaded yet. Click "Initialize day" to compute OR levels.
            </div>
          ) : null}
        </div>
      </section>

      {/* signals */}
      <section className={styles.section}>
        <div className={styles.sectionHead}>
          <div className="section-title">Today's signals</div>
          <Chip>{signals.length} events</Chip>
        </div>
        <DataTable
          columns={signalCols}
          rows={signals.slice(0, 30)}
          emptyText="No signals yet today"
          rowKey={(s, i) => s.id ?? `${s.instrument}-${i}`}
        />
      </section>

      {/* recent trades */}
      <section className={styles.section}>
        <div className={styles.sectionHead}>
          <div className="section-title">Today's trades</div>
          <Chip>{state?.today_closed?.length ?? 0} closed</Chip>
        </div>
        <DataTable
          columns={tradeCols}
          rows={state?.today_closed ?? []}
          emptyText="No trades closed today"
          rowKey={(t, i) => `${t.instrument}-${t.exit_time ?? i}`}
        />
      </section>
    </div>
  );
}

/* ---------- Stock card ---------- */

function StockCard({ stock }: { stock: ORBStockSummary }) {
  const ds = stock.daily_state;
  const label = stockStatus(stock);
  const tone = statusTone(label);
  const dots = filterDots(stock);

  const pnl = stock.position?.pnl_inr ?? stock.today_result?.pnl_inr;

  return (
    <div className={styles.stockCard}>
      <div className={styles.stockHead}>
        <div className={styles.stockSym}>{ds.instrument}</div>
        <div className={styles.stockLtp}>
          {stock.position?.ltp
            ? formatNumber(stock.position.ltp)
            : stock.price
            ? formatNumber(stock.price)
            : '—'}
        </div>
      </div>

      <div className={styles.stockRanges}>
        <div className={styles.range}>
          <div className={styles.rangeLabel}>OR high</div>
          <div className={styles.rangeValue}>
            {ds.or_high ? formatNumber(ds.or_high) : '—'}
          </div>
        </div>
        <div className={styles.range}>
          <div className={styles.rangeLabel}>OR low</div>
          <div className={styles.rangeValue}>
            {ds.or_low ? formatNumber(ds.or_low) : '—'}
          </div>
        </div>
      </div>

      <div className={styles.stockDots}>
        {dots.map((d) => (
          <div key={d.label} className={styles.dotItem} title={d.title}>
            <span className={`${styles.fdot} ${styles[`dot_${d.tone}`]}`} />
            <span>{d.label}</span>
          </div>
        ))}
      </div>

      <div className={styles.stockFoot}>
        <span className={`${styles.status} ${styles[`status_${tone}`]}`}>{label}</span>
        {pnl !== undefined ? (
          <span className={pnlClass(pnl)} style={{ fontSize: 'var(--text-xxs)' }}>
            {formatPnl(pnl)}
          </span>
        ) : null}
      </div>
    </div>
  );
}
