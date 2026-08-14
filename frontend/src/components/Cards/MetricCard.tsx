import type { ReactNode } from 'react';
import styles from './MetricCard.module.css';
import StatusDot from '../StatusDot/StatusDot';

type CardStatus = 'live' | 'paper' | 'closed' | 'warn' | 'off';
const STATUS_TITLE: Record<CardStatus, string> = {
  live: 'Live - real money, active',
  paper: 'Paper - simulated, active',
  closed: 'Closed - squared off for the day',
  warn: 'Attention - check this system',
  off: 'Not trading today',
};

interface Props {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  /** Optional right-aligned text next to the label (e.g. context date) */
  labelRight?: ReactNode;
  valueClassName?: string;
  /** Uniform status dot (top-right). */
  status?: CardStatus;
  statusTitle?: string;
}

export default function MetricCard({ label, value, hint, labelRight, valueClassName, status, statusTitle }: Props) {
  return (
    <div className={styles.card}>
      <div className={styles.labelRow}>
        <div className={styles.label}>{label}</div>
        {(labelRight || status) ? (
          <div className={styles.labelRight}>
            {labelRight}
            {status ? <StatusDot kind={status} title={statusTitle ?? STATUS_TITLE[status]} /> : null}
          </div>
        ) : null}
      </div>
      <div className={`${styles.value} ${valueClassName ?? ''}`}>{value}</div>
      {hint ? <div className={styles.hint}>{hint}</div> : null}
    </div>
  );
}
