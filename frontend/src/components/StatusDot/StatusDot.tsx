import styles from './StatusDot.module.css';

type Kind = 'connected' | 'warning' | 'disconnected';

interface Props {
  kind?: Kind;
  label?: string;
  className?: string;
}

export default function StatusDot({ kind = 'connected', label, className }: Props) {
  return (
    <span className={`${styles.wrap} ${className ?? ''}`}>
      <span className={`${styles.dot} ${styles[kind]}`} aria-hidden />
      {label ? <span className={styles.label}>{label}</span> : null}
    </span>
  );
}
