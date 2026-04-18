import type { ReactNode } from 'react';
import styles from './Chip.module.css';

interface Props {
  children: ReactNode;
  className?: string;
}

export default function Chip({ children, className }: Props) {
  return <span className={`${styles.chip} ${className ?? ''}`}>{children}</span>;
}
