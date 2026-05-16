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

function StudyDataTable({ t }: { t: StudyTable }) {
  const columns: Column<TableRow>[] = t.columns.map((header, ci) => ({
    key: String(ci),
    header,
    align: ci === 0 ? 'left' : 'right',
    render: (row: TableRow) => row.cells[ci],
  }));
  const hl = new Set(t.highlightRows ?? []);
  const rows: TableRow[] = t.rows.map((cells, i) => ({ cells, _hl: hl.has(i) }));

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

      {/* 2. System */}
      <section className={styles.section}>
        <SectionHead n={2} label="System" />
        <div className={styles.sectionIntro}>{study.system.intro}</div>
        <KVList rows={study.system.rows} />
      </section>

      {/* 3. Conditions */}
      <section className={styles.section}>
        <SectionHead n={3} label="Conditions" />
        {study.conditions.intro ? (
          <div className={styles.sectionIntro}>{study.conditions.intro}</div>
        ) : null}
        <KVList rows={study.conditions.rows} />
      </section>

      {/* 4. Comparisons */}
      <section className={styles.section}>
        <SectionHead n={4} label="Comparisons" />
        {study.comparisons.map((t, i) => (
          <StudyDataTable key={i} t={t} />
        ))}
      </section>

      {/* 5. Results */}
      <section className={styles.section}>
        <SectionHead n={5} label="Results" />
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
      </section>

      {/* 6. Winners */}
      <section className={styles.section}>
        <SectionHead n={6} label="Winners" />
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

      {/* 7. Caveats */}
      <section className={styles.section}>
        <SectionHead n={7} label="Caveats" />
        <ol className={styles.caveatList}>
          {study.caveats.map((c, i) => (
            <li key={i}>{c}</li>
          ))}
        </ol>
      </section>

      {/* 8. Links */}
      <section className={styles.section}>
        <SectionHead n={8} label="Links" />
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
