import styles from './StatusDot.module.css';

// Uniform status language across the app (2026-08-14): live/paper/closed/warn/off,
// plus the legacy connection kinds. Colour carries the meaning; `title` is the tooltip.
type Kind = 'connected' | 'warning' | 'disconnected' | 'live' | 'paper' | 'closed' | 'warn' | 'off';

interface Props {
  kind?: Kind;
  label?: string;
  className?: string;
  title?: string;
}

export default function StatusDot({ kind = 'connected', label, className, title }: Props) {
  return (
    <span className={`${styles.wrap} ${className ?? ''}`} title={title}>
      <span className={`${styles.dot} ${styles[kind]}`} aria-hidden aria-label={title} />
      {label ? <span className={styles.label}>{label}</span> : null}
    </span>
  );
}
