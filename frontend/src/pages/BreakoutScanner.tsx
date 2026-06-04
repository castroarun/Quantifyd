import { useEffect, useMemo, useState } from 'react';
import { apiGet, apiPost } from '../api/client';
import StatusDot from '../components/StatusDot/StatusDot';
import Chip from '../components/Chip/Chip';
import DataTable from '../components/DataTable/DataTable';
import type { Column } from '../components/DataTable/DataTable';
import { IconBarChart } from '../components/Icons';
import styles from './BreakoutScanner.module.css';

type Match = {
  symbol: string;
  ltp: number;
  day_change_pct: number | null;
  rule: 'compression' | 'donchian';
  breakout_level: number | null;
  trend_sma50: number | null;
  cpr_width_pct: number | null;
  range10_pct: number | null;
  nr7: boolean;
  vol_surge: number | null;
  rsi: number | null;
  turn_cr: number | null;
};

type ScanState = {
  generated_at: string;
  market_open: boolean;
  universes: string[];
  rules: string[];
  scanned: number;
  count: number;
  matches: Match[];
  note?: string;
};

type Settings = {
  universes: string[];
  rules: string[];
  trend: string;
  cpr_narrow_pct: number;
  contract_pct: number;
  donchian_lb: number;
  vol_mult: number;
  rsi_min: number;
  min_price: number;
  min_turn_cr: number;
  email_enabled: boolean;
  email_to: string;
};

const UNIVERSES: { key: string; label: string; hint: string }[] = [
  { key: 'fno', label: 'F&O', hint: '~80 liquid F&O names' },
  { key: 'nifty500', label: 'Nifty 500', hint: 'incl. midcaps' },
  { key: 'smallcap', label: 'Smallcap', hint: 'turnover < 50cr' },
  { key: 'all_liquid', label: 'All liquid', hint: 'whole universe, price/turnover gated' },
];

const RULES: { key: string; label: string; hint: string }[] = [
  { key: 'compression', label: 'Compression breakout', hint: 'uptrend + coil + narrow CPR + prev-day-high break + volume' },
  { key: 'donchian', label: 'Donchian-20 + volume', hint: 'uptrend + 20-day-high break + volume surge' },
];

const DEFAULTS: Settings = {
  universes: ['fno'], rules: ['compression', 'donchian'], trend: 'sma50',
  cpr_narrow_pct: 0.6, contract_pct: 0.12, donchian_lb: 20, vol_mult: 1.5,
  rsi_min: 0, min_price: 30, min_turn_cr: 3, email_enabled: false,
  email_to: 'arun.castromin@gmail.com',
};

function fmtTime(iso: string): string {
  try {
    return new Date(iso).toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch { return iso; }
}

export default function BreakoutScanner() {
  const [data, setData] = useState<ScanState | null>(null);
  const [s, setS] = useState<Settings>(DEFAULTS);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [toast, setToast] = useState<string | null>(null);

  const setField = <K extends keyof Settings>(k: K, v: Settings[K]) => setS((p) => ({ ...p, [k]: v }));
  const toggleIn = (k: 'universes' | 'rules', val: string) =>
    setS((p) => {
      const arr = p[k].includes(val) ? p[k].filter((x) => x !== val) : [...p[k], val];
      return { ...p, [k]: arr.length ? arr : p[k] };
    });

  // load saved settings once
  useEffect(() => {
    apiGet<{ settings: Settings }>('/api/breakout-scanner/settings')
      .then((r) => setS({ ...DEFAULTS, ...r.settings })).catch(() => {});
  }, []);

  // poll current scan state
  useEffect(() => {
    let cancelled = false;
    const load = () => {
      apiGet<ScanState>('/api/breakout-scanner/state')
        .then((d) => { if (!cancelled) { setData(d); setErr(null); } })
        .catch((e) => { if (!cancelled) setErr(e instanceof Error ? e.message : 'Load failed'); });
    };
    load();
    const id = setInterval(load, 30_000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  const flash = (m: string) => { setToast(m); setTimeout(() => setToast(null), 2500); };

  const runScan = async () => {
    setBusy(true); setErr(null);
    try { setData(await apiPost<ScanState>('/api/breakout-scanner/scan', s)); }
    catch (e) { setErr(e instanceof Error ? e.message : 'Scan failed'); }
    finally { setBusy(false); }
  };
  const saveSettings = async () => {
    setBusy(true);
    try { await apiPost('/api/breakout-scanner/settings', s); flash('Settings saved'); }
    catch (e) { setErr(e instanceof Error ? e.message : 'Save failed'); }
    finally { setBusy(false); }
  };
  const testEmail = async () => {
    setBusy(true);
    try { await apiPost('/api/breakout-scanner/test-email'); flash('Test email sent'); }
    catch (e) { setErr(e instanceof Error ? e.message : 'Email failed'); }
    finally { setBusy(false); }
  };

  const matches = data?.matches ?? [];

  const columns: Column<Match>[] = useMemo(() => [
    { key: 'symbol', header: 'Symbol', width: '1.1fr', render: (r) => <span className={styles.sym}>{r.symbol}</span> },
    { key: 'ltp', header: 'LTP', width: '0.9fr', align: 'right', render: (r) => r.ltp?.toLocaleString('en-IN') },
    {
      key: 'chg', header: 'Day %', width: '0.7fr', align: 'right',
      render: (r) => r.day_change_pct == null ? '—'
        : <span className={r.day_change_pct >= 0 ? styles.pos : styles.neg}>{r.day_change_pct >= 0 ? '+' : ''}{r.day_change_pct}%</span>,
    },
    {
      key: 'rule', header: 'Rule', width: '1.2fr',
      render: (r) => <Chip className={r.rule === 'compression' ? styles.chipComp : styles.chipDon}>{r.rule === 'compression' ? 'Compression' : 'Donchian-20'}</Chip>,
    },
    { key: 'brk', header: 'Broke >', width: '0.9fr', align: 'right', render: (r) => r.breakout_level?.toLocaleString('en-IN') ?? '—' },
    { key: 'vol', header: 'Vol×', width: '0.6fr', align: 'right', render: (r) => r.vol_surge != null ? <span className={styles.volx}>{r.vol_surge}×</span> : '—' },
    { key: 'rsi', header: 'RSI', width: '0.6fr', align: 'right', render: (r) => r.rsi ?? '—' },
    { key: 'cpr', header: 'CPR w%', width: '0.7fr', align: 'right', render: (r) => r.cpr_width_pct != null ? `${r.cpr_width_pct}%` : '—' },
    { key: 'coil', header: 'Coil%', width: '0.6fr', align: 'right', render: (r) => r.range10_pct != null ? `${r.range10_pct}%` : '—' },
  ], []);

  return (
    <div className={styles.root}>
      <div className={styles.headerRow}>
        <div>
          <div className="page-title">
            <span className={styles.titleIcon}><IconBarChart /></span> Breakout Scanner
          </div>
          <div className="page-subtitle">
            Live intraday breakouts · compression &amp; Donchian-20 · multi-universe · email alerts every 15 min
          </div>
        </div>
        <div className={styles.headerRight}>
          <StatusDot kind={data?.market_open ? 'connected' : 'disconnected'} label={data?.market_open ? 'Market open' : 'Market closed'} />
          {data?.generated_at && <span className={styles.genAt}>Updated {fmtTime(data.generated_at)}</span>}
        </div>
      </div>

      {err ? <div className={styles.error}>{err}</div> : null}
      {toast ? <div className={styles.toast}>{toast}</div> : null}

      <div className={styles.body}>
        <aside className={styles.filters}>
          <div className={styles.filterGroup}>
            <label className={styles.filterLabel}>Universe</label>
            {UNIVERSES.map((u) => (
              <label key={u.key} className={styles.checkRow}>
                <input type="checkbox" checked={s.universes.includes(u.key)} onChange={() => toggleIn('universes', u.key)} />
                <span><b>{u.label}</b><span className={styles.checkHint}> — {u.hint}</span></span>
              </label>
            ))}
          </div>

          <div className={styles.filterGroup}>
            <label className={styles.filterLabel}>Breakout rule</label>
            {RULES.map((r) => (
              <label key={r.key} className={styles.checkRow}>
                <input type="checkbox" checked={s.rules.includes(r.key)} onChange={() => toggleIn('rules', r.key)} />
                <span><b>{r.label}</b><span className={styles.checkHint}> — {r.hint}</span></span>
              </label>
            ))}
          </div>

          <div className={styles.filterGroup}>
            <label className={styles.filterLabel}>Trend filter</label>
            <div className={styles.segmented}>
              {['sma50', 'sma200', 'both', 'off'].map((t) => (
                <button key={t} className={`${styles.segBtn} ${s.trend === t ? styles.segActive : ''}`} onClick={() => setField('trend', t)}>
                  {t === 'off' ? 'Off' : t.toUpperCase()}
                </button>
              ))}
            </div>
          </div>

          <div className={styles.filterGroup}>
            <label className={styles.filterLabel}>Volume surge ≥<span className={styles.filterVal}>{s.vol_mult.toFixed(1)}×</span></label>
            <input type="range" min={1} max={4} step={0.1} value={s.vol_mult} onChange={(e) => setField('vol_mult', parseFloat(e.target.value))} className={styles.slider} />
          </div>

          <div className={styles.filterGroup}>
            <label className={styles.filterLabel}>Narrow CPR ≤<span className={styles.filterVal}>{s.cpr_narrow_pct.toFixed(2)}%</span></label>
            <input type="range" min={0.2} max={1.5} step={0.05} value={s.cpr_narrow_pct} onChange={(e) => setField('cpr_narrow_pct', parseFloat(e.target.value))} className={styles.slider} />
            <div className={styles.sliderHint}>Compression rule only.</div>
          </div>

          <div className={styles.filterGroup}>
            <label className={styles.filterLabel}>Coil range ≤<span className={styles.filterVal}>{(s.contract_pct * 100).toFixed(0)}%</span></label>
            <input type="range" min={0.05} max={0.3} step={0.01} value={s.contract_pct} onChange={(e) => setField('contract_pct', parseFloat(e.target.value))} className={styles.slider} />
            <div className={styles.sliderHint}>10-day range as % of price.</div>
          </div>

          <div className={styles.filterGroup}>
            <label className={styles.filterLabel}>RSI ≥<span className={styles.filterVal}>{s.rsi_min === 0 ? 'off' : s.rsi_min}</span></label>
            <input type="range" min={0} max={80} step={5} value={s.rsi_min} onChange={(e) => setField('rsi_min', parseInt(e.target.value))} className={styles.slider} />
          </div>

          <div className={styles.filterGroup}>
            <label className={styles.filterLabel}>Liquidity floor</label>
            <div className={styles.inlineInputs}>
              <label className={styles.miniInput}>Min ₹<input type="number" value={s.min_price} onChange={(e) => setField('min_price', parseFloat(e.target.value) || 0)} /></label>
              <label className={styles.miniInput}>Turn ≥ cr<input type="number" value={s.min_turn_cr} onChange={(e) => setField('min_turn_cr', parseFloat(e.target.value) || 0)} /></label>
            </div>
          </div>

          <div className={styles.emailBox}>
            <label className={styles.toggleRow}>
              <input type="checkbox" checked={s.email_enabled} onChange={(e) => setField('email_enabled', e.target.checked)} />
              <span><b>Email me on breakouts</b><span className={styles.checkHint}> — every 15 min during market, new hits only</span></span>
            </label>
            <input className={styles.emailInput} type="email" value={s.email_to} onChange={(e) => setField('email_to', e.target.value)} placeholder="you@email.com" />
            <button className={styles.btnGhost} disabled={busy} onClick={testEmail}>Send test email</button>
          </div>

          <div className={styles.actions}>
            <button className={styles.btnPrimary} disabled={busy} onClick={runScan}>{busy ? 'Scanning…' : 'Run scan now'}</button>
            <button className={styles.btnSecondary} disabled={busy} onClick={saveSettings}>Save settings</button>
          </div>

          {data && (
            <div className={styles.resultCount}>
              {data.count} breakout{data.count === 1 ? '' : 's'} · {data.scanned} scanned
            </div>
          )}
        </aside>

        <div className={styles.main}>
          {matches.length === 0 ? (
            <div className={styles.waiting}>
              No breakouts match the current filters{data && !data.market_open ? ' (market closed — off-hours uses last close, volume is stale)' : ''}.
              Loosen volume×, widen universe, or wait for the next 15-min scan.
            </div>
          ) : (
            <DataTable<Match> columns={columns} rows={matches} rowKey={(r) => `${r.symbol}:${r.rule}`} />
          )}
          {data?.note && <div className={styles.note}>{data.note}</div>}
        </div>
      </div>
    </div>
  );
}
