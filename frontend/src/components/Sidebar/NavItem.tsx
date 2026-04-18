import { NavLink } from 'react-router-dom';
import type { ReactNode } from 'react';
import styles from './NavItem.module.css';

interface Props {
  to: string;
  icon: ReactNode;
  label: string;
  active?: boolean;
}

export default function NavItem({ to, icon, label, active }: Props) {
  // We accept an explicit `active` prop to stay in sync with page-level state,
  // but fall back to NavLink's own matcher when not provided.
  if (active !== undefined) {
    const cls = `${styles.item} ${active ? styles.active : ''}`;
    return (
      <NavLink to={to} className={cls}>
        <span className={styles.icon}>{icon}</span>
        <span className={styles.label}>{label}</span>
      </NavLink>
    );
  }
  return (
    <NavLink
      to={to}
      className={({ isActive }) => `${styles.item} ${isActive ? styles.active : ''}`}
    >
      <span className={styles.icon}>{icon}</span>
      <span className={styles.label}>{label}</span>
    </NavLink>
  );
}
