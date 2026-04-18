import { useEffect, useState } from 'react';
import NavItem from './NavItem';
import styles from './Sidebar.module.css';
import {
  IconGrid,
  IconBarChart,
  IconLayers,
  IconReport,
  IconSettings,
} from '../Icons';
import Avatar from '../Avatar/Avatar';

interface Props {
  active?: 'strategies' | 'orb' | 'nas' | 'reports' | 'settings';
  userName?: string;
}

const COLLAPSE_KEY = 'qf.sidebar.collapsed';

export default function Sidebar({ active, userName = 'Trader' }: Props) {
  const [collapsed, setCollapsed] = useState<boolean>(() => {
    try {
      return localStorage.getItem(COLLAPSE_KEY) === '1';
    } catch {
      return false;
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(COLLAPSE_KEY, collapsed ? '1' : '0');
      document.documentElement.dataset.sidebar = collapsed ? 'collapsed' : 'expanded';
    } catch {
      /* ignore */
    }
  }, [collapsed]);

  const toggle = () => setCollapsed((c) => !c);

  return (
    <aside className={`${styles.sidebar} ${collapsed ? styles.collapsed : ''}`}>
      <div className={styles.logo}>
        <div className={styles.logoIcon}>Q</div>
        {!collapsed && <span className={styles.logoText}>Quantifyd</span>}
        <button
          className={styles.collapseBtn}
          onClick={toggle}
          title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
            {collapsed ? (
              <polyline points="9 18 15 12 9 6" />
            ) : (
              <polyline points="15 18 9 12 15 6" />
            )}
          </svg>
        </button>
      </div>

      <div className={styles.section}>
        {!collapsed && <div className={styles.sectionLabel}>Workspace</div>}
        <nav className={styles.nav}>
          <NavItem
            to="/strategies"
            icon={<IconGrid />}
            label="Strategies"
            active={active === 'strategies'}
            collapsed={collapsed}
          />
          <NavItem
            to="/orb"
            icon={<IconBarChart />}
            label="ORB cash"
            active={active === 'orb'}
            collapsed={collapsed}
          />
          <NavItem
            to="/nas"
            icon={<IconLayers />}
            label="NAS options"
            active={active === 'nas'}
            collapsed={collapsed}
          />
          <NavItem
            to="/report"
            icon={<IconReport />}
            label="Performance"
            active={active === 'reports'}
            collapsed={collapsed}
          />
        </nav>
      </div>

      <div className={styles.section}>
        {!collapsed && <div className={styles.sectionLabel}>General</div>}
        <nav className={styles.nav}>
          <NavItem
            to="/settings"
            icon={<IconSettings />}
            label="Settings"
            active={active === 'settings'}
            collapsed={collapsed}
          />
        </nav>
      </div>

      {!collapsed && (
        <div className={styles.foot}>
          <Avatar name={userName} subtitle="Zerodha account" />
        </div>
      )}
    </aside>
  );
}
