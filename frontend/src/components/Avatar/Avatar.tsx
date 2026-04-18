import styles from './Avatar.module.css';

interface Props {
  name: string;
  subtitle?: string;
  size?: number;
}

function initial(name: string): string {
  return (name?.trim()?.[0] || '?').toUpperCase();
}

export default function Avatar({ name, subtitle, size = 24 }: Props) {
  return (
    <div className={styles.wrap}>
      <div
        className={styles.circle}
        style={{ width: size, height: size, fontSize: Math.round(size * 0.46) }}
      >
        {initial(name)}
      </div>
      <div className={styles.meta}>
        <div className={styles.name}>{name}</div>
        {subtitle ? <div className={styles.sub}>{subtitle}</div> : null}
      </div>
    </div>
  );
}
