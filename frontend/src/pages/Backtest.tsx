import { Link } from 'react-router-dom';
import styles from './Backtest.module.css';
import { BACKTEST_STUDIES } from '../data/backtests';

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

export default function Backtest() {
  return (
    <div className={styles.page}>
      <div className={styles.header}>
        <div className={styles.title}>Backtest research</div>
        <div className={styles.subtitle}>
          Completed strategy studies — system, conditions, comparisons, results,
          winners and honest caveats. Click a study to open its full report.
        </div>
      </div>

      <div className={styles.grid}>
        {BACKTEST_STUDIES.map((s) => (
          <Link key={s.slug} to={`/backtest/${s.slug}`} className={styles.card}>
            <div className={styles.cardHead}>
              <div className={styles.cardTitle}>{s.title}</div>
              <span className={`${styles.statusChip} ${STATUS_CLASS[s.status] ?? ''}`}>
                {s.status}
              </span>
            </div>
            <div className={styles.cardVerdict}>{s.cardBlurb}</div>
            <div className={styles.cardStats}>
              {s.cardStats.map((st, i) => (
                <div className={styles.stat} key={i}>
                  <div className={styles.statLabel}>{st.label}</div>
                  <div className={styles.statValue}>{st.value}</div>
                </div>
              ))}
            </div>
            <div className={styles.cardFoot}>
              <span className={styles.metaDate}>{fmtDate(s.date)}</span>
              <span className={styles.openLink}>Open report ›</span>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}
