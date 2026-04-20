import type { ReactNode } from 'react';
import styles from './MetricCard.module.css';

interface Props {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  /** Optional right-aligned text next to the label (e.g. context date) */
  labelRight?: ReactNode;
  valueClassName?: string;
}

export default function MetricCard({ label, value, hint, labelRight, valueClassName }: Props) {
  return (
    <div className={styles.card}>
      <div className={styles.labelRow}>
        <div className={styles.label}>{label}</div>
        {labelRight ? <div className={styles.labelRight}>{labelRight}</div> : null}
      </div>
      <div className={`${styles.value} ${valueClassName ?? ''}`}>{value}</div>
      {hint ? <div className={styles.hint}>{hint}</div> : null}
    </div>
  );
}
