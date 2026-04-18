import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import styles from './StrategyCard.module.css';
import StatusDot from '../StatusDot/StatusDot';
import { formatPnl, pnlClass } from '../../utils/format';

interface Stat {
  label: string;
  value: ReactNode;
}

interface Props {
  to: string;
  title: string;
  description: string;
  icon?: ReactNode;
  status?: 'connected' | 'warning' | 'disconnected';
  statusLabel?: string;
  stats?: Stat[];
  dayPnl?: number;
}

export default function StrategyCard({
  to,
  title,
  description,
  icon,
  status = 'connected',
  statusLabel,
  stats,
  dayPnl,
}: Props) {
  return (
    <Link to={to} className={styles.card}>
      <div className={styles.head}>
        {icon ? <div className={styles.icon}>{icon}</div> : null}
        <div className={styles.titles}>
          <div className={styles.title}>{title}</div>
          <div className={styles.desc}>{description}</div>
        </div>
        <StatusDot kind={status} label={statusLabel} />
      </div>

      {stats && stats.length > 0 ? (
        <div className={styles.stats}>
          {stats.map((s, i) => (
            <div className={styles.stat} key={i}>
              <div className={styles.statLabel}>{s.label}</div>
              <div className={styles.statValue}>{s.value}</div>
            </div>
          ))}
        </div>
      ) : null}

      {dayPnl !== undefined ? (
        <div className={styles.foot}>
          <span className={styles.footLabel}>Day P&amp;L</span>
          <span className={pnlClass(dayPnl)}>{formatPnl(dayPnl)}</span>
        </div>
      ) : null}
    </Link>
  );
}
