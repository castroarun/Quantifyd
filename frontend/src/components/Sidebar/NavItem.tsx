import { NavLink } from 'react-router-dom';
import type { ReactNode } from 'react';
import styles from './NavItem.module.css';
import { PAGE_HOTKEYS } from './hotkeys';

interface Props {
  to: string;
  icon: ReactNode;
  label: string;
  active?: boolean;
  collapsed?: boolean;
}

export default function NavItem({ to, icon, label, active, collapsed }: Props) {
  const hotkey = PAGE_HOTKEYS[to];
  const titleAttr = collapsed
    ? hotkey
      ? `${label} (${hotkey})`
      : label
    : hotkey
      ? `Press ${hotkey}`
      : undefined;

  const body = (
    <>
      <span className={styles.icon}>{icon}</span>
      {!collapsed && <span className={styles.label}>{label}</span>}
      {!collapsed && hotkey && <span className={styles.hotkey}>{hotkey}</span>}
    </>
  );

  if (active !== undefined) {
    const cls = `${styles.item} ${active ? styles.active : ''} ${collapsed ? styles.collapsed : ''}`;
    return (
      <NavLink to={to} className={cls} title={titleAttr}>
        {body}
      </NavLink>
    );
  }
  return (
    <NavLink
      to={to}
      className={({ isActive }) => `${styles.item} ${isActive ? styles.active : ''} ${collapsed ? styles.collapsed : ''}`}
      title={titleAttr}
    >
      {body}
    </NavLink>
  );
}
