import { useEffect, useMemo, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import styles from './BacktestStudy.module.css';
import MetricCard from '../components/Cards/MetricCard';
import DataTable, { type Column } from '../components/DataTable/DataTable';
import { getStudy, type StudyTable, type KV } from '../data/backtests';

const STATUS_CLASS: Record<string, string> = {
  COMPLETE: styles.stComplete,
  RUNNING: styles.stRunning,
  STUCK: styles.stStuck,
  FAILED: styles.stFailed,
  PARKED: styles.stParked,
};

function fmtDate(d: string): string {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(d);
  if (!m) return d;
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  return `${parseInt(m[3], 10)} ${months[parseInt(m[2], 10) - 1]} ${m[1]}`;
}

type TableRow = { cells: string[]; _hl: boolean };

/** Parse a numeric cell like "+101.5", "−10.3", "35.3%", "—".
 *  Handles the unicode minus (−, U+2212) used in the data. */
function parseNum(s: string): number | null {
  if (s == null) return null;
  const cleaned = s
    .replace(/−/g, '-')   // unicode minus → ascii
    .replace(/[%+,]/g, '')
    .trim();
  if (cleaned === '' || cleaned === '-' || /^[—–-]$/.test(s.trim())) return null;
  const n = Number(cleaned);
  return Number.isFinite(n) ? n : null;
}

/** Diverging red → neutral → green background for a value within a
 *  symmetric range. Returns a translucent rgba so the dark theme shows
 *  through, plus a flag for whether the cell is "strong" (needs lighter
 *  text for contrast). */
function heatStyle(value: number, maxAbs: number): { background: string; strong: boolean } {
  if (maxAbs <= 0) return { background: 'transparent', strong: false };
  // normalized intensity 0..1, clamped
  const t = Math.min(1, Math.abs(value) / maxAbs);
  // ease so near-zero stays neutral, extremes pop
  const a = Math.round(0.65 * Math.pow(t, 0.85) * 100) / 100;
  const strong = t > 0.55;
  if (value >= 0) {
    // green (matches --accent-pos family)
    return { background: `rgba(31, 168, 122, ${a})`, strong };
  }
  // red
  return { background: `rgba(214, 69, 69, ${a})`, strong };
}

function StudyDataTable({ t }: { t: StudyTable }) {
  const hl = new Set(t.highlightRows ?? []);
  const rows: TableRow[] = t.rows.map((cells, i) => ({ cells, _hl: hl.has(i) }));

  // Which columns are numeric (heatmap-eligible): every column that has
  // at least one parseable number across the body, excluding column 0.
  const numericCols = new Set<number>();
  const colMaxAbs: Record<number, number> = {};
  if (t.heatmap) {
    t.columns.forEach((_h, ci) => {
      if (ci === 0) return;
      let any = false;
      let maxAbs = 0;
      for (const r of t.rows) {
        const n = parseNum(r[ci]);
        if (n != null) {
          any = true;
          maxAbs = Math.max(maxAbs, Math.abs(n));
        }
      }
      if (any) {
        numericCols.add(ci);
        colMaxAbs[ci] = maxAbs;
      }
    });
  }

  const columns: Column<TableRow>[] = t.columns.map((header, ci) => ({
    key: String(ci),
    header,
    align: ci === 0 ? 'left' : 'right',
    render: (row: TableRow) => {
      const raw = row.cells[ci];
      if (t.heatmap && numericCols.has(ci)) {
        const n = parseNum(raw);
        if (n != null) {
          const { background, strong } = heatStyle(n, colMaxAbs[ci]);
          return (
            <span
              className={`${styles.heatCell} ${strong ? styles.heatStrong : ''}`}
              style={{ background }}
            >
              {raw}
            </span>
          );
        }
      }
      return raw;
    },
  }));

  return (
    <div className={styles.tableBlock}>
      <div className={styles.tableTitle}>{t.title}</div>
      {t.caption ? <div className={styles.tableCaption}>{t.caption}</div> : null}
      <DataTable<TableRow>
        columns={columns}
        rows={rows}
        rowKey={(_r, i) => i}
        rowClassName={(r) => (r._hl ? styles.hlRow : undefined)}
      />
      {t.heatmap ? (
        <div className={styles.heatLegend}>
          <span className={styles.heatLegendLabel}>Under-perform</span>
          <span className={styles.heatLegendBar} aria-hidden="true" />
          <span className={styles.heatLegendLabel}>Out-perform</span>
        </div>
      ) : null}
    </div>
  );
}

function StudyFigure({ src, caption }: { src: string; caption: string }) {
  return (
    <div className={styles.figureBlock}>
      <img className={styles.figureImg} src={src} alt={caption} loading="lazy" />
      <div className={styles.tableCaption}>{caption}</div>
    </div>
  );
}

function KVList({ rows }: { rows: KV[] }) {
  return (
    <div className={styles.kvList}>
      {rows.map((r, i) => (
        <div className={styles.kvRow} key={i}>
          <div className={styles.kvK}>{r.k}</div>
          <div className={styles.kvV}>{r.v}</div>
        </div>
      ))}
    </div>
  );
}

/** Split one CSV line, honouring double-quoted fields that contain commas
 *  (the exit_reason column is "SuperTrend(14,4) trail"). */
function splitCsvLine(line: string): string[] {
  const out: string[] = [];
  let cur = '';
  let inQ = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (inQ) {
      if (ch === '"') {
        if (line[i + 1] === '"') { cur += '"'; i += 1; } else { inQ = false; }
      } else cur += ch;
    } else if (ch === '"') inQ = true;
    else if (ch === ',') { out.push(cur); cur = ''; }
    else cur += ch;
  }
  out.push(cur);
  return out;
}

/** Backtested trade list, fetched from a CSV served under /app/.
 *  Sortable by any column, scrollable, with a download link. */
function TradeTable({
  src, caption, maxRows = 400, note,
}: { src: string; caption?: string; maxRows?: number; note?: string }) {
  const [rows, setRows] = useState<string[][]>([]);
  const [head, setHead] = useState<string[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [sortCol, setSortCol] = useState<number | null>(null);
  const [desc, setDesc] = useState(true);

  useEffect(() => {
    let alive = true;
    fetch(src)
      .then((r) => (r.ok ? r.text() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((txt) => {
        if (!alive) return;
        const lines = txt.trim().split(/\r?\n/);
        setHead(splitCsvLine(lines[0]));
        setRows(lines.slice(1).map(splitCsvLine));
      })
      .catch((e) => alive && setErr(String(e)));
    return () => { alive = false; };
  }, [src]);

  const sorted = useMemo(() => {
    if (sortCol == null) return rows;
    const copy = [...rows];
    copy.sort((a, b) => {
      const x = a[sortCol] ?? '';
      const y = b[sortCol] ?? '';
      const nx = Number(x);
      const ny = Number(y);
      const both = !Number.isNaN(nx) && !Number.isNaN(ny) && x !== '' && y !== '';
      const c = both ? nx - ny : x.localeCompare(y);
      return desc ? -c : c;
    });
    return copy;
  }, [rows, sortCol, desc]);

  if (err) {
    return <div style={{ fontSize: 12, opacity: 0.7, margin: '14px 0' }}>
      Trade list unavailable ({err}) — <a href={src}>download CSV</a>
    </div>;
  }

  const columns: Column<string[]>[] = head.map((h, ci) => ({
    key: String(ci),
    header: h.replace(/_/g, ' '),
    align: ci === 0 ? 'left' : 'right',
    render: (row: string[]) => <span>{row[ci]}</span>,
  }));

  return (
    <div style={{ margin: '18px 0' }}>
      {caption && <div style={{ fontSize: 13, marginBottom: 8 }}>{caption}</div>}
      <div style={{ fontSize: 12, opacity: 0.7, marginBottom: 8 }}>
        {rows.length.toLocaleString()} trades · click a column header to sort ·{' '}
        <a href={src} download>download the full CSV</a>
        {note ? ` · ${note}` : ''}
      </div>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 8 }}>
        {head.map((h, ci) => (
          <button
            key={ci}
            onClick={() => { if (sortCol === ci) setDesc(!desc); else { setSortCol(ci); setDesc(true); } }}
            style={{
              fontSize: 11, padding: '3px 8px', borderRadius: 6, cursor: 'pointer',
              border: '1px solid rgba(148,163,184,0.25)',
              background: sortCol === ci ? 'rgba(148,163,184,0.18)' : 'transparent',
              color: 'inherit',
            }}
          >
            {h.replace(/_/g, ' ')}{sortCol === ci ? (desc ? ' \u2193' : ' \u2191') : ''}
          </button>
        ))}
      </div>
      <div style={{ maxHeight: 560, overflow: 'auto', border: '1px solid rgba(148,163,184,0.18)', borderRadius: 10 }}>
        <DataTable<string[]>
          columns={columns}
          rows={sorted.slice(0, maxRows)}
          rowKey={(_r, i) => i}
          emptyText="Loading trades…"
        />
      </div>
      {sorted.length > maxRows && (
        <div style={{ fontSize: 12, opacity: 0.65, marginTop: 6 }}>
          Showing the first {maxRows.toLocaleString()} of {sorted.length.toLocaleString()} rows in the current sort —
          the CSV has them all.
        </div>
      )}
    </div>
  );
}

function SectionHead({ n, label }: { n: number; label: string }) {
  return (
    <div className={styles.sectionHead}>
      <span className={styles.sectionNum}>{String(n).padStart(2, '0')}</span>
      <span className={styles.sectionLabel}>{label}</span>
    </div>
  );
}

export default function BacktestStudy() {
  const { slug } = useParams<{ slug: string }>();
  const study = slug ? getStudy(slug) : undefined;

  if (!study) {
    return (
      <div className={styles.page}>
        <Link to="/backtest" className={styles.back}>‹ Back to studies</Link>
        <div className={styles.notFound}>
          No backtest study found for “{slug}”.
        </div>
      </div>
    );
  }

  return (
    <div className={styles.page}>
      <Link to="/backtest" className={styles.back}>‹ Back to studies</Link>

      {/* 1. Header */}
      <div className={styles.headerCard}>
        <div className={styles.headerTop}>
          <div className={styles.studyTitle}>{study.title}</div>
          <span className={`${styles.statusChip} ${STATUS_CLASS[study.status] ?? ''}`}>
            {study.status}
          </span>
        </div>
        <div className={styles.verdict}>{study.verdict}</div>
        {study.slug === 'momentum30-subselect' && (
          <Link
            to="/momentum-paper"
            style={{
              display: 'inline-flex', alignItems: 'center', gap: 6, marginTop: 14,
              padding: '9px 16px', borderRadius: 8, background: 'var(--accent-pos, #1f9d55)',
              color: '#fff', fontSize: 14, fontWeight: 600, textDecoration: 'none',
            }}
          >
            ▶ View the live ₹20L paper book →
          </Link>
        )}
        <div className={styles.headerMeta}>
          <span>Study completed {fmtDate(study.date)}</span>
          <span className={styles.dot}>·</span>
          <span className={styles.slugMono}>{study.slug}</span>
        </div>
      </div>

      {/* 2. System Rules (optional — the actual traded rules, stated
          before the System/Conditions narrative and the evidence
          tables so the rules precede the results). */}
      {study.systemRules ? (
        <section className={styles.section}>
          <SectionHead n={2} label="System Rules" />
          {study.systemRules.intro ? (
            <div className={styles.sectionIntro}>{study.systemRules.intro}</div>
          ) : null}
          <div className={styles.subHead}>{study.systemRules.sharedCoreTitle}</div>
          <KVList rows={study.systemRules.sharedCore} />
          <div className={styles.systemRulesTable}>
            <StudyDataTable t={study.systemRules.riskLayer} />
          </div>
        </section>
      ) : null}

      {/* 3. System */}
      <section className={styles.section}>
        <SectionHead n={study.systemRules ? 3 : 2} label="System" />
        <div className={styles.sectionIntro}>{study.system.intro}</div>
        <KVList rows={study.system.rows} />
      </section>

      {/* 4. Conditions */}
      <section className={styles.section}>
        <SectionHead n={study.systemRules ? 4 : 3} label="Conditions" />
        {study.conditions.intro ? (
          <div className={styles.sectionIntro}>{study.conditions.intro}</div>
        ) : null}
        <KVList rows={study.conditions.rows} />
      </section>

      {/* 5. Comparisons */}
      <section className={styles.section}>
        <SectionHead n={study.systemRules ? 5 : 4} label="Comparisons" />
        {study.comparisons.map((t, i) => (
          <StudyDataTable key={i} t={t} />
        ))}
      </section>

      {/* 6. Results */}
      <section className={styles.section}>
        <SectionHead n={study.systemRules ? 6 : 5} label="Results" />
        <div className={styles.metricGrid}>
          {study.results.metrics.map((m, i) => (
            <MetricCard
              key={i}
              label={m.label}
              value={m.value}
              hint={m.hint}
              valueClassName={
                m.tone === 'pos' ? styles.tonePos : m.tone === 'neg' ? styles.toneNeg : undefined
              }
            />
          ))}
        </div>
        {study.results.tables.map((t, i) => (
          <StudyDataTable key={i} t={t} />
        ))}
        {study.results.charts?.map((c, i) => (
          <StudyFigure key={i} src={c.src} caption={c.caption} />
        ))}
        {study.tradeTable && (
          <TradeTable
            src={study.tradeTable.src}
            caption={study.tradeTable.caption}
            maxRows={study.tradeTable.maxRows}
            note={study.tradeTable.note}
          />
        )}
        {study.results.embeds?.map((e, i) => (
          <div key={`e${i}`} style={{ margin: '18px 0' }}>
            <iframe
              src={e.src}
              title={e.caption || 'embedded report'}
              style={{ width: '100%', height: e.height ?? 1200, border: '1px solid rgba(148,163,184,0.18)', borderRadius: 10, background: 'transparent' }}
            />
            {e.caption && <div style={{ fontSize: 12, opacity: 0.65, marginTop: 6 }}>{e.caption}</div>}
          </div>
        ))}
      </section>

      {/* 7. Winners */}
      <section className={styles.section}>
        <SectionHead n={study.systemRules ? 7 : 6} label="Winners" />
        {study.winners.map((w, i) => (
          <div key={i} className={styles.winnerCallout}>
            <div className={styles.winnerBadge}>WINNER</div>
            <div className={styles.winnerConfig}>{w.config}</div>
            <div className={styles.winnerSummary}>{w.summary}</div>
            <div className={styles.winnerMetrics}>
              {w.metrics.map((m, j) => (
                <div className={styles.winnerMetric} key={j}>
                  <div className={styles.wmK}>{m.k}</div>
                  <div className={styles.wmV}>{m.v}</div>
                </div>
              ))}
            </div>
            {w.rejected && w.rejected.length > 0 ? (
              <div className={styles.rejectedWrap}>
                <div className={styles.rejectedHead}>Rejected / void variants</div>
                <ul className={styles.rejectedList}>
                  {w.rejected.map((r, j) => (
                    <li key={j}>{r}</li>
                  ))}
                </ul>
              </div>
            ) : null}
          </div>
        ))}
      </section>

      {/* 8. Caveats */}
      <section className={styles.section}>
        <SectionHead n={study.systemRules ? 8 : 7} label="Caveats" />
        <ol className={styles.caveatList}>
          {study.caveats.map((c, i) => (
            <li key={i}>{c}</li>
          ))}
        </ol>
      </section>

      {/* 9. Links */}
      <section className={styles.section}>
        <SectionHead n={study.systemRules ? 9 : 8} label="Links" />
        <div className={styles.linkGrid}>
          <div className={styles.linkCol}>
            <div className={styles.linkColHead}>GitHub</div>
            <ul className={styles.linkList}>
              {study.githubLinks.map((l, i) => (
                <li key={i}>
                  <a href={l.href} target="_blank" rel="noreferrer">
                    {l.label}
                  </a>
                </li>
              ))}
            </ul>
          </div>
          {study.reports && study.reports.length > 0 ? (
            <div className={styles.linkCol}>
              <div className={styles.linkColHead}>Reports &amp; data</div>
              <ul className={styles.linkList}>
                {study.reports.map((l, i) => (
                  <li key={i}>
                    <a href={l.href} target="_blank" rel="noreferrer">{l.label}</a>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
          <div className={styles.linkCol}>
            <div className={styles.linkColHead}>Project paths (local)</div>
            <ul className={styles.pathList}>
              {study.projectPaths.map((p, i) => (
                <li key={i}><code>{p}</code></li>
              ))}
            </ul>
          </div>
        </div>
      </section>
    </div>
  );
}
