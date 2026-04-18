import { NavLink } from 'react-router-dom';
import type { ReactNode } from 'react';
import styles from './NavItem.module.css';

interface Props {
  to: string;
  icon: ReactNode;
  label: string;
  active?: boolean;
  collapsed?: boolean;
}

export default function NavItem({ to, icon, label, active, collapsed }: Props) {
  const titleAttr = collapsed ? label : undefined;

  if (active !== undefined) {
    const cls = `${styles.item} ${active ? styles.active : ''} ${collapsed ? styles.collapsed : ''}`;
    return (
      <NavLink to={to} className={cls} title={titleAttr}>
        <span className={styles.icon}>{icon}</span>
        {!collapsed && <span className={styles.label}>{label}</span>}
      </NavLink>
    );
  }
  return (
    <NavLink
      to={to}
      className={({ isActive }) => `${styles.item} ${isActive ? styles.active : ''} ${collapsed ? styles.collapsed : ''}`}
      title={titleAttr}
    >
      <span className={styles.icon}>{icon}</span>
      {!collapsed && <span className={styles.label}>{label}</span>}
    </NavLink>
  );
}
