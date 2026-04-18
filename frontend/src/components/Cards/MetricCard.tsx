import type { ReactNode } from 'react';
import styles from './MetricCard.module.css';

interface Props {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  valueClassName?: string;
}

export default function MetricCard({ label, value, hint, valueClassName }: Props) {
  return (
    <div className={styles.card}>
      <div className={styles.label}>{label}</div>
      <div className={`${styles.value} ${valueClassName ?? ''}`}>{value}</div>
      {hint ? <div className={styles.hint}>{hint}</div> : null}
    </div>
  );
}
