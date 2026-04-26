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
  active?: 'strategies' | 'orb' | 'nas' | 'nwv' | 'strangle' | 'reports' | 'holdings' | 'options-data' | 'future-plans' | 'settings';
  userName?: string;
}

// Briefcase icon for holdings, inlined to avoid growing Icons export surface
function IconBriefcase() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="7" width="18" height="13" rx="2" />
      <path d="M8 7V5a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
      <path d="M3 13h18" />
    </svg>
  );
}

// Database/cylinder icon for options-data — represents captured market data
function IconDatabase() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      <ellipse cx="12" cy="5" rx="9" ry="3" />
      <path d="M3 5v14c0 1.66 4.03 3 9 3s9-1.34 9-3V5" />
      <path d="M3 12c0 1.66 4.03 3 9 3s9-1.34 9-3" />
    </svg>
  );
}

// Lightbulb icon for future-plans — idea / design sketches
function IconLightbulb() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9 18h6" />
      <path d="M10 22h4" />
      <path d="M12 2a7 7 0 0 1 7 7c0 3-1.5 4.5-3 6a3 3 0 0 0-1 2v1H9v-1a3 3 0 0 0-1-2C6.5 13.5 5 12 5 9a7 7 0 0 1 7-7z" />
    </svg>
  );
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
            to="/app/strangle"
            icon={<IconLayers />}
            label="ORB index"
            active={active === 'strangle'}
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
            to="/nwv"
            icon={<IconLayers />}
            label="NWV weekly"
            active={active === 'nwv'}
            collapsed={collapsed}
          />
          <NavItem
            to="/report"
            icon={<IconReport />}
            label="Performance"
            active={active === 'reports'}
            collapsed={collapsed}
          />
          <NavItem
            to="/holdings"
            icon={<IconBriefcase />}
            label="Holdings"
            active={active === 'holdings'}
            collapsed={collapsed}
          />
          <NavItem
            to="/options-data"
            icon={<IconDatabase />}
            label="Options data"
            active={active === 'options-data'}
            collapsed={collapsed}
          />
          <NavItem
            to="/future-plans"
            icon={<IconLightbulb />}
            label="Future plans"
            active={active === 'future-plans'}
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
