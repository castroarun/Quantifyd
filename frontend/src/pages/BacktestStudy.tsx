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
