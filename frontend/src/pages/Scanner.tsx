import { useEffect, useMemo, useState } from 'react';
import { apiGet } from '../api/client';
import StatusDot from '../components/StatusDot/StatusDot';
import Chip from '../components/Chip/Chip';
import DataTable from '../components/DataTable/DataTable';
import type { Column } from '../components/DataTable/DataTable';
import { IconLayers } from '../components/Icons';
import styles from './Scanner.module.css';

type ScanRow = {
  symbol: string;
  ltp: number | null;
  day_change_pct: number | null;
  volume_surge: number | null;
  cpr_width_pct: number | null;
  cpr_narrow: boolean;
  trend: 'up' | 'down' | 'flat';
  range_state: string;
  range_escaped: boolean;
  clean_candle: 'long' | 'short' | 'none' | null;
  direction: 'long' | 'short' | null;
  score: number;
};

type ScanState = {
  generated_at: string;
  market_open: boolean;
  cpr_threshold: number;
  count: number;
  rows: ScanRow[];
};

type DirFilter = 'both' | 'long' | 'short';
type ViewMode = 'cards' | 'table';

function scoreClass(s: number): string {
  if (s >= 70) return styles.scoreHigh;
  if (s >= 45) return styles.scoreMid;
  return styles.scoreLow;
}

function trendChipClass(t: string): string {
  if (t === 'up') return styles.chipUp;
  if (t === 'down') return styles.chipDown;
  return styles.chipFlat;
}

function rangeLabel(r: ScanRow): string {
  switch (r.range_state) {
    case 'week_high': return 'Above PW high';
    case 'week_low':  return 'Below PW low';
    case 'day_high':  return 'Above PD high';
    case 'day_low':   return 'Below PD low';
    default:          return 'Inside range';
  }
}

function fmtTime(iso: string): string {
  try {
    return new Date(iso).toLocaleTimeString('en-IN', {
      hour: '2-digit', minute: '2-digit', second: '2-digit',
    });
  } catch {
    return iso;
  }
}

export default function Scanner() {
  const [data, setData] = useState<ScanState | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  // filters
  const [cpr, setCpr] = useState(0.5);
  const [dir, setDir] = useState<DirFilter>('both');
  const [escapedOnly, setEscapedOnly] = useState(false);
  const [minVol, setMinVol] = useState(0);
  const [minScore, setMinScore] = useState(0);
  const [view, setView] = useState<ViewMode>('cards');

  useEffect(() => {
    let cancelled = false;
    const load = () => {
      apiGet<ScanState>(`/api/scanner/state?cpr=${cpr}`)
        .then((s) => {
          if (cancelled) return;
          setData(s);
          setErr(null);
          setLoading(false);
        })
        .catch((e) => {
          if (cancelled) return;
          setErr(e instanceof Error ? e.message : 'Load failed');
          setLoading(false);
        });
    };
    load();
    const id = setInterval(load, 30_000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [cpr]);

  const filtered = useMemo(() => {
    const rows = data?.rows ?? [];
    return rows.filter((r) => {
      if (dir !== 'both' && r.direction !== dir) return false;
      if (escapedOnly && !r.range_escaped) return false;
      if (minVol > 0 && (r.volume_surge ?? 0) < minVol) return false;
      if (minScore > 0 && r.score < minScore) return false;
      return true;
    });
  }, [data, dir, escapedOnly, minVol, minScore]);

  const columns: Column<ScanRow>[] = [
    { key: 'symbol', header: 'Symbol', width: '1.1fr', render: (r) => <span className={styles.sym}>{r.symbol}</span> },
    {
      key: 'ltp', header: 'LTP', width: '0.9fr', align: 'right',
      render: (r) => (r.ltp != null ? r.ltp.toLocaleString('en-IN') : '—'),
    },
    {
      key: 'chg', header: 'Day %', width: '0.8fr', align: 'right',
      render: (r) => {
        const v = r.day_change_pct;
        if (v == null) return '—';
        return <span className={v >= 0 ? styles.pos : styles.neg}>{v >= 0 ? '+' : ''}{v}%</span>;
      },
    },
    {
      key: 'vol', header: 'Vol×', width: '0.7fr', align: 'right',
      render: (r) => (r.volume_surge != null ? `${r.volume_surge}×` : '—'),
    },
    {
      key: 'cpr', header: 'CPR w%', width: '0.8fr', align: 'right',
      render: (r) => (r.cpr_width_pct != null ? `${r.cpr_width_pct}%` : '—'),
    },
    {
      key: 'trend', header: 'Trend', width: '0.8fr',
      render: (r) => <Chip className={trendChipClass(r.trend)}>{r.trend}</Chip>,
    },
    {
      key: 'range', header: 'Range', width: '1.1fr',
      render: (r) => <Chip className={r.range_escaped ? styles.chipEscaped : ''}>{rangeLabel(r)}</Chip>,
    },
    {
      key: 'score', header: 'Score', width: '0.7fr', align: 'right',
      render: (r) => <span className={`${styles.scorePill} ${scoreClass(r.score)}`}>{r.score}</span>,
    },
  ];

  return (
    <div className={styles.root}>
      <div className={styles.headerRow}>
        <div>
          <div className="page-title">
            <span className={styles.titleIcon}><IconLayers /></span> F&amp;O Scanner
          </div>
          <div className="page-subtitle">
            Live setup ranking across {data?.count ?? 81} F&amp;O stocks · volume surge ·
            weekly CPR · trend &amp; range break · composite 0–100 score
          </div>
        </div>
        <div className={styles.headerRight}>
          <StatusDot
            kind={data?.market_open ? 'connected' : 'disconnected'}
            label={data?.market_open ? 'Market open' : 'Market closed'}
          />
          {data?.generated_at && (
            <span className={styles.genAt}>Updated {fmtTime(data.generated_at)}</span>
          )}
        </div>
      </div>

      {err ? <div className={styles.error}>{err}</div> : null}

      <div className={styles.body}>
        <aside className={styles.filters}>
          <div className={styles.filterGroup}>
            <label className={styles.filterLabel}>
              CPR-width threshold
              <span className={styles.filterVal}>{cpr.toFixed(2)}%</span>
            </label>
            <input
              type="range" min={0.1} max={2.0} step={0.05} value={cpr}
              onChange={(e) => setCpr(parseFloat(e.target.value))}
              className={styles.slider}
            />
            <div className={styles.sliderHint}>
              Stocks at/below this weekly-CPR width score the narrowness bonus.
            </div>
          </div>

          <div className={styles.filterGroup}>
            <label className={styles.filterLabel}>Direction</label>
            <div className={styles.segmented}>
              {(['both', 'long', 'short'] as DirFilter[]).map((d) => (
                <button
                  key={d}
                  className={`${styles.segBtn} ${dir === d ? styles.segActive : ''}`}
                  onClick={() => setDir(d)}
                >
                  {d}
                </button>
              ))}
            </div>
          </div>

          <div className={styles.filterGroup}>
            <label className={styles.toggleRow}>
              <input
                type="checkbox" checked={escapedOnly}
                onChange={(e) => setEscapedOnly(e.target.checked)}
              />
              Range escaped only (beyond prior week)
            </label>
          </div>

          <div className={styles.filterGroup}>
            <label className={styles.filterLabel}>
              Min volume surge
              <span className={styles.filterVal}>{minVol.toFixed(1)}×</span>
            </label>
            <input
              type="range" min={0} max={5} step={0.1} value={minVol}
              onChange={(e) => setMinVol(parseFloat(e.target.value))}
              className={styles.slider}
            />
          </div>

          <div className={styles.filterGroup}>
            <label className={styles.filterLabel}>
              Min score
              <span className={styles.filterVal}>{minScore}</span>
            </label>
            <input
              type="range" min={0} max={100} step={5} value={minScore}
              onChange={(e) => setMinScore(parseInt(e.target.value, 10))}
              className={styles.slider}
            />
          </div>

          <div className={styles.filterGroup}>
            <label className={styles.filterLabel}>View</label>
            <div className={styles.segmented}>
              {(['cards', 'table'] as ViewMode[]).map((v) => (
                <button
                  key={v}
                  className={`${styles.segBtn} ${view === v ? styles.segActive : ''}`}
                  onClick={() => setView(v)}
                >
                  {v}
                </button>
              ))}
            </div>
          </div>

          <div className={styles.resultCount}>
            {filtered.length} / {data?.count ?? 0} stocks
          </div>
        </aside>

        <main className={styles.main}>
          {loading && !data ? (
            <div className={styles.waiting}>Loading scanner…</div>
          ) : filtered.length === 0 ? (
            <div className={styles.waiting}>
              No stocks match the current filters.
            </div>
          ) : view === 'table' ? (
            <DataTable
              columns={columns}
              rows={filtered}
              rowKey={(r) => r.symbol}
              emptyText="No stocks match the current filters."
            />
          ) : (
            <div className={styles.grid}>
              {filtered.map((r) => (
                <div key={r.symbol} className={`${styles.card} ${scoreClass(r.score)}`}>
                  <div className={styles.cardTop}>
                    <span className={styles.cardSym}>{r.symbol}</span>
                    <span className={`${styles.scorePill} ${scoreClass(r.score)}`}>{r.score}</span>
                  </div>

                  <div className={styles.cardLtp}>
                    <span className={styles.ltpVal}>
                      {r.ltp != null ? r.ltp.toLocaleString('en-IN') : '—'}
                    </span>
                    {r.day_change_pct != null && (
                      <span className={r.day_change_pct >= 0 ? styles.pos : styles.neg}>
                        {r.day_change_pct >= 0 ? '+' : ''}{r.day_change_pct}%
                      </span>
                    )}
                  </div>

                  <div className={styles.cardRow}>
                    <span className={styles.cardKey}>Vol surge</span>
                    <span className={styles.cardNum}>
                      {r.volume_surge != null ? `${r.volume_surge}×` : '—'}
                    </span>
                  </div>

                  <div className={styles.cardRow}>
                    <span className={styles.cardKey}>Weekly CPR</span>
                    <span className={styles.cardNum}>
                      {r.cpr_width_pct != null ? `${r.cpr_width_pct}%` : '—'}
                    </span>
                  </div>
                  <div className={styles.cprBarTrack}>
                    <div
                      className={styles.cprBarFill}
                      style={{
                        width: `${Math.min(
                          100,
                          ((r.cpr_width_pct ?? 0) / (cpr * 2 || 1)) * 100,
                        )}%`,
                      }}
                    />
                    <div
                      className={styles.cprThreshold}
                      style={{ left: '50%' }}
                      title={`Threshold ${cpr.toFixed(2)}%`}
                    />
                  </div>

                  <div className={styles.cardChips}>
                    <Chip className={trendChipClass(r.trend)}>{r.trend}</Chip>
                    <Chip className={r.range_escaped ? styles.chipEscaped : ''}>
                      {rangeLabel(r)}
                    </Chip>
                    {r.clean_candle && r.clean_candle !== 'none' && (
                      <Chip className={styles.chipClean}>clean {r.clean_candle}</Chip>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
