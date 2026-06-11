import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import styles from './NasConfig.module.css';
import { apiGet, apiPost } from '../api/client';

/* ---------- types ---------- */

interface SysRow {
  dte: Record<string, boolean>;
  gap_up: boolean;
  gap_down: boolean;
  live: boolean;
}
interface Matrix {
  gap_up_pct: number;
  gap_down_pct: number;
  systems: Record<string, SysRow>;
}
interface SystemMeta {
  key: string;
  label: string;
}
interface TodayInfo {
  dte: number | null;
  gap_pct: number | null;
  is_gap_up: boolean;
  is_gap_down: boolean;
}
interface MatrixResponse {
  matrix: Matrix;
  systems: SystemMeta[];
  today: TodayInfo;
}

const DTE_COLS: { d: number; day: string }[] = [
  { d: 4, day: 'Wed' },
  { d: 3, day: 'Thu' },
  { d: 2, day: 'Fri' },
  { d: 1, day: 'Mon' },
  { d: 0, day: 'Tue' },
];

/* ---------- component ---------- */

export default function NasConfig() {
  const [matrix, setMatrix] = useState<Matrix | null>(null);
  const [systems, setSystems] = useState<SystemMeta[]>([]);
  const [today, setToday] = useState<TodayInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [savedAt, setSavedAt] = useState<string | null>(null);

  const fetchMatrix = () => {
    setLoading(true);
    apiGet<MatrixResponse>('/api/nas/day-matrix')
      .then((r) => {
        setMatrix(r.matrix);
        setSystems(r.systems);
        setToday(r.today);
        setErr(null);
      })
      .catch((e) => setErr(String(e?.message || e)))
      .finally(() => setLoading(false));
  };

  useEffect(fetchMatrix, []);

  // immutable updates
  const editRow = (key: string, patch: Partial<SysRow>) => {
    setMatrix((m) => {
      if (!m) return m;
      const row = m.systems[key];
      return { ...m, systems: { ...m.systems, [key]: { ...row, ...patch } } };
    });
    setSavedAt(null);
  };
  const editDte = (key: string, d: number, val: boolean) => {
    setMatrix((m) => {
      if (!m) return m;
      const row = m.systems[key];
      return {
        ...m,
        systems: { ...m.systems, [key]: { ...row, dte: { ...row.dte, [String(d)]: val } } },
      };
    });
    setSavedAt(null);
  };
  const editGap = (which: 'gap_up_pct' | 'gap_down_pct', val: number) => {
    setMatrix((m) => (m ? { ...m, [which]: val } : m));
    setSavedAt(null);
  };

  const save = () => {
    if (!matrix) return;
    setSaving(true);
    apiPost<{ status: string; matrix: Matrix }>('/api/nas/day-matrix', matrix)
      .then((r) => {
        setMatrix(r.matrix);
        setSavedAt(new Date().toLocaleTimeString());
        setErr(null);
        fetchMatrix(); // refresh "today" with new thresholds
      })
      .catch((e) => setErr(String(e?.message || e)))
      .finally(() => setSaving(false));
  };

  // would this system enter today, per the (saved) matrix + today's DTE/gap?
  const firesToday = useMemo(() => {
    const out: Record<string, boolean> = {};
    if (!matrix || !today) return out;
    for (const { key } of systems) {
      const row = matrix.systems[key];
      if (!row) continue;
      const byDte = today.dte != null && !!row.dte[String(today.dte)];
      const byUp = today.is_gap_up && row.gap_up;
      const byDown = today.is_gap_down && row.gap_down;
      out[key] = byDte || byUp || byDown;
    }
    return out;
  }, [matrix, today, systems]);

  if (loading && !matrix) {
    return <div className={styles.loading}>Loading day-matrix…</div>;
  }
  if (err && !matrix) {
    return <div className={styles.error}>Failed to load: {err}</div>;
  }
  if (!matrix) return null;

  const gapLabel =
    today?.gap_pct == null
      ? 'gap n/a'
      : `${today.gap_pct >= 0 ? '+' : ''}${today.gap_pct.toFixed(2)}%`;
  const gapKind = today?.is_gap_up ? 'gap-UP' : today?.is_gap_down ? 'gap-DOWN' : 'no gap';

  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <div>
          <h1 className={styles.title}>NAS · Day &amp; Gap Matrix</h1>
          <p className={styles.sub}>
            Which system enters on which day (trading-DTE to Tue expiry) or gap, and whether it
            trades real money. Saved settings drive the 09:16 &amp; squeeze entries.{' '}
            <Link to="/nas" className={styles.link}>
              ← back to NAS
            </Link>
          </p>
        </div>
        <div className={styles.actions}>
          {savedAt && <span className={styles.saved}>saved {savedAt}</span>}
          {err && <span className={styles.errInline}>{err}</span>}
          <button className={styles.saveBtn} onClick={save} disabled={saving}>
            {saving ? 'Saving…' : 'Save'}
          </button>
        </div>
      </div>

      {today && (
        <div className={styles.todayBar}>
          <span className={styles.todayChip}>
            Today: <strong>DTE {today.dte ?? '–'}</strong>
          </span>
          <span
            className={`${styles.todayChip} ${
              today.is_gap_up ? styles.up : today.is_gap_down ? styles.down : ''
            }`}
          >
            {gapKind} <strong>{gapLabel}</strong>
          </span>
          <span className={styles.todayNote}>
            Gap thresholds: up ≥ {matrix.gap_up_pct}% · down ≤ -{matrix.gap_down_pct}%. Rows
            highlighted below would <strong>fire today</strong>.
          </span>
        </div>
      )}

      <div className={styles.tableWrap}>
        <table className={styles.matrix}>
          <thead>
            <tr>
              <th className={styles.sysCol}>System</th>
              {DTE_COLS.map((c) => (
                <th key={c.d} className={styles.dteCol}>
                  <div className={styles.dteHead}>{c.d}DTE</div>
                  <div className={styles.dteDay}>{c.day}</div>
                </th>
              ))}
              <th className={styles.gapCol}>Gap&nbsp;Up</th>
              <th className={styles.gapCol}>Gap&nbsp;Dn</th>
              <th className={styles.liveCol}>LIVE&nbsp;₹</th>
            </tr>
          </thead>
          <tbody>
            {systems.map(({ key, label }) => {
              const row = matrix.systems[key];
              if (!row) return null;
              const fires = firesToday[key];
              return (
                <tr key={key} className={fires ? styles.firesRow : ''}>
                  <td className={styles.sysCol}>
                    <span className={styles.sysLabel}>{label}</span>
                    {fires && <span className={styles.firesTag}>fires today</span>}
                  </td>
                  {DTE_COLS.map((c) => (
                    <td key={c.d} className={styles.cell}>
                      <input
                        type="checkbox"
                        checked={!!row.dte[String(c.d)]}
                        onChange={(e) => editDte(key, c.d, e.target.checked)}
                      />
                    </td>
                  ))}
                  <td className={styles.cell}>
                    <input
                      type="checkbox"
                      checked={row.gap_up}
                      onChange={(e) => editRow(key, { gap_up: e.target.checked })}
                    />
                  </td>
                  <td className={styles.cell}>
                    <input
                      type="checkbox"
                      checked={row.gap_down}
                      onChange={(e) => editRow(key, { gap_down: e.target.checked })}
                    />
                  </td>
                  <td className={`${styles.cell} ${styles.liveCell}`}>
                    <input
                      type="checkbox"
                      checked={row.live}
                      onChange={(e) => editRow(key, { live: e.target.checked })}
                    />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className={styles.gapConfig}>
        <div className={styles.gapField}>
          <label>Gap-UP threshold ≥</label>
          <div className={styles.numWrap}>
            <input
              type="number"
              step="0.05"
              min="0"
              value={matrix.gap_up_pct}
              onChange={(e) => editGap('gap_up_pct', parseFloat(e.target.value) || 0)}
            />
            <span>%</span>
          </div>
        </div>
        <div className={styles.gapField}>
          <label>Gap-DOWN threshold ≤ -</label>
          <div className={styles.numWrap}>
            <input
              type="number"
              step="0.05"
              min="0"
              value={matrix.gap_down_pct}
              onChange={(e) => editGap('gap_down_pct', parseFloat(e.target.value) || 0)}
            />
            <span>%</span>
          </div>
        </div>
        <p className={styles.gapHelp}>
          Gap = (today&apos;s official open − previous close) / previous close, from Kite. A system
          enters if its today-DTE is checked, <em>or</em> a matching gap box is checked on a gap
          day. <strong>LIVE ₹</strong> = real money (else paper). Master mode <em>paper/off</em>
          still forces paper for safety.
        </p>
      </div>
    </div>
  );
}
