import { useEffect, useState } from 'react';
import { apiGet } from '../api/client';
import styles from './AthScanner.module.css';

type Row = {
  symbol: string; price: number; day_high?: number; prior_ath?: number;
  pct_above_prior_ath?: number; box_high?: number; box_low?: number;
  box_days?: number; range_pct?: number; pct_above_box?: number;
  chg_pct: number; vol_x_10d: number | null; vol_x_20d: number | null;
  vol_spike: boolean; turnover_cr: number; asof: string; when?: string;
};
type Res = {
  error?: string; asof: string; asof_date: string; session_fraction: number;
  universe: string; universe_size: number; scanned: number; history_since: string;
  params: { vol_mult: number; vol_window: number; consol_days: number;
            consol_max_range_pct: number; min_turnover_cr: number };
  ath: Row[]; consolidation_breakout: Row[]; cached?: boolean;
  runs?: { asof: string; label?: string; date?: string; ath?: number; consol?: number; error?: string }[];
};

const ASOF = [
  ['live', 'Live (now)'], ['eod', 'EOD (last close)'], ['t1', 'T-1'],
  ['t2', 'T-2'], ['t3', 'T-3'],
] as const;
const UNIVERSES = [
  ['nifty500', 'Nifty 500'], ['fno', 'F&O'], ['smallcap', 'Smallcap'],
  ['all', 'All stored (~1600)'],
] as const;

export default function AthScanner() {
  const [asofs, setAsofs] = useState<string[]>(['eod']);
  const [universe, setUniverse] = useState('nifty500');
  const [volWindow, setVolWindow] = useState(20);
  const [volMult, setVolMult] = useState('1.5');
  const [consolDays, setConsolDays] = useState('12');
  const [minTurn, setMinTurn] = useState('1');
  const [onlyVol, setOnlyVol] = useState(false);
  const [hideCont, setHideCont] = useState(true);
  const [res, setRes] = useState<Res | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const run = () => {
    setLoading(true);
    setErr(null);
    const qs = new URLSearchParams({
      asof: asofs.join(','), universe, vol_window: String(volWindow), vol_mult: volMult,
      consol_days: consolDays, min_turnover_cr: minTurn,
      fresh_min_gap: hideCont ? '20' : '0',
    });
    apiGet<Res>(`/api/breakout-scan?${qs}`)
      .then((r) => { r.error ? setErr(r.error) : setRes(r); setLoading(false); })
      .catch((e) => { setErr(String(e)); setLoading(false); });
  };
  useEffect(run, []);  // first load with defaults

  const volKey = volWindow === 20 ? 'vol_x_20d' : 'vol_x_10d';
  const filt = (rows: Row[]) => (onlyVol ? rows.filter((r) => r.vol_spike) : rows);

  const volCell = (r: Row) => {
    const v = r[volKey as 'vol_x_20d'];
    return (
      <td className={r.vol_spike ? styles.spike : ''}>
        {v == null ? '—' : v.toFixed(2) + '×'}
        {r.vol_spike && <span className={styles.spikeTag}>VOL</span>}
      </td>
    );
  };

  return (
    <div className={styles.root}>
      <div className={styles.headerRow}>
        <div>
          <h1 className={styles.title}>ATH &amp; Breakout Scanner</h1>
          <p className={styles.sub}>
            New all-time highs and consolidation-box breakouts, with volume confirmation ·
            {res ? ` ${res.scanned}/${res.universe_size} scanned · ATH = highest close since ${res.history_since}` : ' loading…'}
            {res?.runs && ' · ' + res.runs.map((r) => r.error
              ? `${r.asof}: ${r.error}`
              : `${r.label} (${r.date}): ${r.ath} ATH / ${r.consol} box`).join(' · ')}
            {res?.asof === 'live' && res.session_fraction < 1 &&
              ` · live volume pro-rated to ${(res.session_fraction * 100).toFixed(0)}% of session`}
          </p>
        </div>
        <button className={styles.btnPrimary} onClick={run} disabled={loading}>
          {loading ? 'Scanning…' : 'Scan'}
        </button>
      </div>

      {/* Controls */}
      <div className={styles.controls}>
        <div className={styles.ctrl}>
          <label>As of</label>
          <div className={styles.segRow}>
            {ASOF.map(([k, lbl]) => (
              <button key={k} className={`${styles.seg} ${asofs.includes(k) ? styles.segActive : ''}`}
                      onClick={() => setAsofs((cur) =>
                        cur.includes(k) ? (cur.length > 1 ? cur.filter((x) => x !== k) : cur)
                                        : [...cur, k])}>{lbl}</button>
            ))}
            <span className={styles.hint}>multi-select — rows are tagged by day</span>
          </div>
        </div>
        <div className={styles.ctrl}>
          <label>Universe</label>
          <div className={styles.segRow}>
            {UNIVERSES.map(([k, lbl]) => (
              <button key={k} className={`${styles.seg} ${universe === k ? styles.segActive : ''}`}
                      onClick={() => setUniverse(k)}>{lbl}</button>
            ))}
          </div>
        </div>
        <div className={styles.ctrl}>
          <label>Volume window</label>
          <div className={styles.segRow}>
            <button className={`${styles.seg} ${volWindow === 20 ? styles.segActive : ''}`}
                    onClick={() => setVolWindow(20)}>20-day (recommended)</button>
            <button className={`${styles.seg} ${volWindow === 10 ? styles.segActive : ''}`}
                    onClick={() => setVolWindow(10)}>10-day</button>
          </div>
        </div>
        <div className={styles.ctrlInline}>
          <label>Vol ≥ <input className={styles.num} value={volMult}
                 onChange={(e) => setVolMult(e.target.value)} />×</label>
          <label>Box days <input className={styles.num} value={consolDays}
                 onChange={(e) => setConsolDays(e.target.value)} /></label>
          <label>Min turnover ₹<input className={styles.num} value={minTurn}
                 onChange={(e) => setMinTurn(e.target.value)} />cr</label>
          <label className={styles.chk}>
            <input type="checkbox" checked={onlyVol}
                   onChange={(e) => setOnlyVol(e.target.checked)} />
            volume-confirmed only
          </label>
          <label className={styles.chk}>
            <input type="checkbox" checked={hideCont}
                   onChange={(e) => setHideCont(e.target.checked)} />
            hide ATH continuations (prior high touched &lt;20 sessions ago)
          </label>
        </div>
      </div>

      {err && <div className={styles.error}>{err}</div>}

      {/* Section 1: New ATH */}
      <div className={styles.card}>
        <div className={styles.cardTitle}>
          New all-time highs
          <span className={styles.count}>
            {res ? `${filt(res.ath).length} of ${res.ath.length}` : ''}
          </span>
        </div>
        {!res || filt(res.ath).length === 0 ? (
          <div className={styles.empty}>No new ATHs {onlyVol ? 'with volume confirmation ' : ''}for this selection.</div>
        ) : (
          <table className={styles.table}>
            <thead><tr>
              <th>Day</th><th>Stock</th><th>Price</th><th>Prior ATH</th><th>Above ATH</th>
              <th>Day chg</th><th>Vol vs {volWindow}d</th><th>Vol vs {volWindow === 20 ? 10 : 20}d</th>
              <th>Turnover</th>
            </tr></thead>
            <tbody>
              {filt(res.ath).map((r, i) => (
                <tr key={r.symbol + (r.when ?? '') + i}>
                  <td><span className={styles.dayTag}>{r.when ?? 'EOD'}</span></td>
                  <td className={styles.sym}>{r.symbol}</td>
                  <td>{r.price.toFixed(2)}</td>
                  <td className={styles.muted}>{r.prior_ath?.toFixed(2)}</td>
                  <td className={styles.pos}>+{r.pct_above_prior_ath?.toFixed(2)}%</td>
                  <td className={r.chg_pct >= 0 ? styles.pos : styles.neg}>
                    {(r.chg_pct >= 0 ? '+' : '') + r.chg_pct.toFixed(2)}%</td>
                  {volCell(r)}
                  <td className={styles.muted}>
                    {(volWindow === 20 ? r.vol_x_10d : r.vol_x_20d)?.toFixed(2) ?? '—'}×</td>
                  <td className={styles.muted}>₹{r.turnover_cr.toFixed(1)}cr</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Section 2: Consolidation breakouts */}
      <div className={styles.card}>
        <div className={styles.cardTitle}>
          Consolidation-box breakouts
          <span className={styles.count}>
            {res ? `${filt(res.consolidation_breakout).length} of ${res.consolidation_breakout.length}` : ''}
          </span>
        </div>
        <p className={styles.note}>
          Price traded inside a FLAT box (range ≤ {res?.params.consol_max_range_pct ?? 8}%, net drift ≤ half the range,
          box high ≥3 sessions old) for the last {res?.params.consol_days ?? 12} sessions, and is now above the box high —
          rising channels that merely stay inside a narrow band no longer qualify.
        </p>
        {!res || filt(res.consolidation_breakout).length === 0 ? (
          <div className={styles.empty}>No box breakouts {onlyVol ? 'with volume confirmation ' : ''}for this selection.</div>
        ) : (
          <table className={styles.table}>
            <thead><tr>
              <th>Day</th><th>Stock</th><th>Price</th><th>Box</th><th>Box width</th>
              <th>Above box</th><th>Day chg</th><th>Vol vs {volWindow}d</th><th>Turnover</th>
            </tr></thead>
            <tbody>
              {filt(res.consolidation_breakout).map((r, i) => (
                <tr key={r.symbol + (r.when ?? '') + i}>
                  <td><span className={styles.dayTag}>{r.when ?? 'EOD'}</span></td>
                  <td className={styles.sym}>{r.symbol}</td>
                  <td>{r.price.toFixed(2)}</td>
                  <td className={styles.muted}>{r.box_low?.toFixed(1)} – {r.box_high?.toFixed(1)}</td>
                  <td>{r.range_pct?.toFixed(1)}%</td>
                  <td className={styles.pos}>+{r.pct_above_box?.toFixed(2)}%</td>
                  <td className={r.chg_pct >= 0 ? styles.pos : styles.neg}>
                    {(r.chg_pct >= 0 ? '+' : '') + r.chg_pct.toFixed(2)}%</td>
                  {volCell(r)}
                  <td className={styles.muted}>₹{r.turnover_cr.toFixed(1)}cr</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className={styles.card}>
        <div className={styles.cardTitle}>Why 20-day volume is the default</div>
        <p className={styles.note}>
          The 10-day average is contaminated by the quiet stretch that usually precedes a breakout —
          a 1.5× spike over 10 days is much easier to clear than it looks. The 20-day average is
          steadier and harder to fool; both ratios are shown so you can compare. Live scans pro-rate
          today's volume by the fraction of the session elapsed, so a morning spike is judged fairly.
        </p>
      </div>
    </div>
  );
}
