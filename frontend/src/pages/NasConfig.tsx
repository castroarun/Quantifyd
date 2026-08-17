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
interface Entry916Row {
  key: string; label: string; venue: string; entry: string; exit: string;
  stop: string; mgmt: string; lots: number | null; live: boolean; live_dtes: number[];
}
interface SleeveRow {
  book: string; label: string; venue: string; mode: string;
  lots: number; qty: number; perdte: Record<string, string>;
}
interface PStopRow {
  venue: string; stop_per_lot: number; tp_per_lot: number | null;
  trail_arm: number | null; trail_give: number | null;
}
interface RulesMatrix {
  entry916: Entry916Row[]; sleeves: SleeveRow[]; portfolioStop: PStopRow[];
}

const DTE_COLS: { d: number; day: string }[] = [
  { d: 1, day: 'Mon' },
  { d: 0, day: 'Tue' },
  { d: 4, day: 'Wed' },
  { d: 3, day: 'Thu' },
  { d: 2, day: 'Fri' },
];

/* ---------- rules-matrix helpers ---------- */

const DTE_SEL: { d: number; day: string }[] = [
  { d: 1, day: 'Mon' }, { d: 0, day: 'Tue' }, { d: 4, day: 'Wed' },
  { d: 3, day: 'Thu' }, { d: 2, day: 'Fri' },
];
const DAY_START = 9 * 60 + 15;
const DAY_SPAN = 15 * 60 + 30 - DAY_START; // 09:15 -> 15:30
const HOURS = [10, 11, 12, 13, 14, 15];
const toMin = (t: string) => {
  const [h, m] = t.split(':').map(Number);
  return h * 60 + m;
};
const pctOf = (t: string) => Math.max(0, Math.min(100, ((toMin(t) - DAY_START) / DAY_SPAN) * 100));
const parseWin = (s?: string) => {
  const m = s ? s.match(/(\d{2}:\d{2})-(\d{2}:\d{2})\s*SL(\d+)/) : null;
  return m ? { entry: m[1], exit: m[2], sl: m[3] } : null;
};

/* ---------- component ---------- */

export default function NasConfig() {
  const [matrix, setMatrix] = useState<Matrix | null>(null);
  const [systems, setSystems] = useState<SystemMeta[]>([]);
  const [today, setToday] = useState<TodayInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [savedAt, setSavedAt] = useState<string | null>(null);
  const [rules, setRules] = useState<RulesMatrix | null>(null);
  const [dteSel, setDteSel] = useState<number | null>(null);

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
  useEffect(() => {
    apiGet<RulesMatrix>('/api/nas/rules-matrix').then(setRules).catch(() => undefined);
  }, []);

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

  const dte = dteSel ?? today?.dte ?? 0;

  // timeline rows for the selected DTE: 916 (constant window, live/paper by DTE) + sleeves (per-DTE window)
  const timelineRows = useMemo(() => {
    if (!rules) return [] as { label: string; venue: string; mode: string; entry: string; exit: string }[];
    const rows: { label: string; venue: string; mode: string; entry: string; exit: string }[] = [];
    for (const r of rules.entry916) {
      const isLive = r.live && r.live_dtes.includes(dte);
      rows.push({ label: r.label, venue: r.venue, mode: isLive ? 'live' : 'paper', entry: r.entry, exit: r.exit });
    }
    for (const s of rules.sleeves) {
      const w = parseWin(s.perdte[String(dte)]);
      if (w) rows.push({ label: s.label, venue: s.venue, mode: s.mode, entry: w.entry, exit: w.exit });
    }
    return rows.sort((a, b) =>
      a.mode === b.mode ? toMin(a.entry) - toMin(b.entry) : a.mode === 'live' ? -1 : 1
    );
  }, [rules, dte]);

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
  const dteDay = DTE_SEL.find((x) => x.d === dte)?.day ?? '';

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

      {rules && (
        <div className={styles.rulesSection}>
          <div className={styles.rulesHead}>
            <div>
              <h2 className={styles.rulesTitle}>Rules Matrix</h2>
              <span className={styles.rulesNote}>reads live config — any rule change reflects here</span>
            </div>
            <div className={styles.dteSeg} role="tablist" aria-label="Expiry day">
              {DTE_SEL.map(({ d, day }) => (
                <button
                  key={d}
                  className={`${styles.dteSegBtn} ${d === dte ? styles.dteSegOn : ''}`}
                  onClick={() => setDteSel(d)}
                >
                  <span className={styles.dteSegD}>DTE{d}</span>
                  <span className={styles.dteSegDay}>{day}</span>
                </button>
              ))}
            </div>
          </div>

          {/* ---- Trading-day timeline ---- */}
          <div className={styles.tlCaption}>
            Trading day · <strong>DTE{dte}</strong> ({dteDay})
            <span className={styles.tlLegend}>
              <i className={styles.legLive} /> live&nbsp;₹ <i className={styles.legPaper} /> paper
            </span>
          </div>
          <div className={styles.timeline}>
            <div className={styles.tlAxis}>
              {HOURS.map((h) => (
                <span key={h} className={styles.tlTick} style={{ left: `${pctOf(`${h}:00`)}%` }}>
                  {h}:00
                </span>
              ))}
            </div>
            <div className={styles.tlRows}>
              {timelineRows.map((r, i) => {
                const left = pctOf(r.entry);
                const width = Math.max(1.2, pctOf(r.exit) - left);
                return (
                  <div key={`${r.label}-${i}`} className={styles.tlRow}>
                    <div className={styles.tlName}>{r.label}</div>
                    <div className={styles.tlTrack}>
                      {HOURS.map((h) => (
                        <span key={h} className={styles.tlGrid} style={{ left: `${pctOf(`${h}:00`)}%` }} />
                      ))}
                      <div
                        className={`${styles.tlBar} ${r.mode === 'live' ? styles.tlBarLive : styles.tlBarPaper}`}
                        style={{ left: `${left}%`, width: `${width}%` }}
                        title={`${r.label}: ${r.entry}–${r.exit} (${r.mode})`}
                      >
                        <span className={styles.tlBarTime}>{r.entry}–{r.exit}</span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* ---- Entry-system cards ---- */}
          <div className={styles.rulesGroupLabel}>Entry systems</div>
          <div className={styles.cardGrid}>
            {rules.entry916.map((r) => {
              const isLive = r.live && r.live_dtes.includes(dte);
              return (
                <div key={r.key} className={`${styles.ruleCard} ${isLive ? styles.cardLive : ''}`}>
                  <div className={styles.cardTop}>
                    <span className={styles.cardName}>{r.label}</span>
                    <span className={styles.venueChip}>{r.venue}</span>
                    <span className={`${styles.modeBadge} ${isLive ? styles.badgeLive : styles.badgePaper}`}>
                      {isLive ? 'LIVE' : 'paper'}
                    </span>
                  </div>
                  <div className={styles.cardWindow}>
                    {r.entry} <span className={styles.arrow}>→</span> {r.exit}
                  </div>
                  <div className={styles.chipRow}>
                    <span className={styles.stopChip} title="Stop">⊘ {r.stop}</span>
                  </div>
                  <div className={styles.chipRow}>
                    <span className={styles.mgmtChip} title="Management">↻ {r.mgmt}</span>
                  </div>
                  <div className={styles.cardFoot}>
                    <span className={styles.lotTag}>×{r.lots} lots</span>
                    <span className={styles.dtePills} title="Live on these DTEs">
                      {[0, 1, 2, 3, 4].map((d) => (
                        <span
                          key={d}
                          className={`${styles.dtePill} ${r.live && r.live_dtes.includes(d) ? styles.pillLive : ''} ${d === dte ? styles.pillSel : ''}`}
                        >
                          {d}
                        </span>
                      ))}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>

          {/* ---- CSL sleeve cards ---- */}
          <div className={styles.rulesGroupLabel}>CSL sleeves</div>
          <div className={styles.cardGrid}>
            {rules.sleeves.map((s) => {
              const w = parseWin(s.perdte[String(dte)]);
              const isLive = s.mode === 'live';
              return (
                <div key={s.book} className={`${styles.ruleCard} ${isLive ? styles.cardLive : ''}`}>
                  <div className={styles.cardTop}>
                    <span className={styles.cardName}>{s.label}</span>
                    <span className={styles.venueChip}>{s.venue}</span>
                    <span className={`${styles.modeBadge} ${isLive ? styles.badgeLive : styles.badgePaper}`}>
                      {isLive ? 'LIVE' : 'paper'}
                    </span>
                  </div>
                  <div className={styles.cardWindow}>
                    {w ? (
                      <>
                        {w.entry} <span className={styles.arrow}>→</span> {w.exit}
                      </>
                    ) : (
                      <span className={styles.noWin}>no window · DTE{dte}</span>
                    )}
                  </div>
                  <div className={styles.chipRow}>
                    <span className={styles.stopChip}>⊘ combined-SL {w ? `${w.sl}%` : '—'}</span>
                    <span className={styles.sizeChip}>{s.lots}L ×{s.qty}</span>
                  </div>
                  <div className={styles.miniDte}>
                    {[0, 1, 2, 3, 4].map((d) => {
                      const ww = parseWin(s.perdte[String(d)]);
                      return (
                        <div key={d} className={`${styles.miniCell} ${d === dte ? styles.miniOn : ''}`}>
                          <span className={styles.miniLbl}>D{d}</span>
                          <span className={styles.miniVal}>{ww ? ww.entry : '—'}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>

          {/* ---- Portfolio stop ---- */}
          <div className={styles.rulesGroupLabel}>9:16 book stop · ATM + ATM2 + ATM4 only (not COMB/TimeB)</div>
          <div className={styles.pstopGrid}>
            {rules.portfolioStop.map((p) => (
              <div key={p.venue} className={styles.pstopCard}>
                <div className={styles.pstopVenue}>{p.venue}</div>
                <div className={styles.pstopTiles}>
                  <div className={`${styles.pstopTile} ${styles.tileStop}`}>
                    <span className={styles.pstopVal}>−₹{Math.abs(p.stop_per_lot)}</span>
                    <span className={styles.pstopLbl}>hard stop / lot</span>
                  </div>
                  {p.tp_per_lot != null && (
                    <div className={`${styles.pstopTile} ${styles.tileTp}`}>
                      <span className={styles.pstopVal}>₹{p.tp_per_lot}</span>
                      <span className={styles.pstopLbl}>take-profit / lot</span>
                    </div>
                  )}
                  {p.trail_arm != null && (
                    <div className={styles.pstopTile}>
                      <span className={styles.pstopVal}>+₹{p.trail_arm}</span>
                      <span className={styles.pstopLbl}>trail arm / lot</span>
                    </div>
                  )}
                  {p.trail_give != null && (
                    <div className={styles.pstopTile}>
                      <span className={styles.pstopVal}>₹{p.trail_give}</span>
                      <span className={styles.pstopLbl}>giveback / lot</span>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
