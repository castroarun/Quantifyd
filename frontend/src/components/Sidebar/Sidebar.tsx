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

export default function Sidebar({ active, userName = 'Trader' }: Props) {
  return (
    <aside className={styles.sidebar}>
      <div className={styles.logo}>
        <div className={styles.logoIcon}>Q</div>
        <span className={styles.logoText}>Quantifyd</span>
      </div>

      <div className={styles.section}>
        <div className={styles.sectionLabel}>Workspace</div>
        <nav className={styles.nav}>
          <NavItem
            to="/strategies"
            icon={<IconGrid />}
            label="Strategies"
            active={active === 'strategies'}
          />
          <NavItem
            to="/orb"
            icon={<IconBarChart />}
            label="ORB cash"
            active={active === 'orb'}
          />
          <NavItem
            to="/nas"
            icon={<IconLayers />}
            label="NAS options"
            active={active === 'nas'}
          />
        </nav>
      </div>

      <div className={styles.section}>
        <div className={styles.sectionLabel}>General</div>
        <nav className={styles.nav}>
          <NavItem
            to="/reports"
            icon={<IconReport />}
            label="Reports"
            active={active === 'reports'}
          />
          <NavItem
            to="/settings"
            icon={<IconSettings />}
            label="Settings"
            active={active === 'settings'}
          />
        </nav>
      </div>

      <div className={styles.foot}>
        <Avatar name={userName} subtitle="Zerodha account" />
      </div>
    </aside>
  );
}
