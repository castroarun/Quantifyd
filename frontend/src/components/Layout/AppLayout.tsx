import type { ReactNode } from 'react';
import { useEffect, useState } from 'react';
import { useLocation } from 'react-router-dom';
import Sidebar from '../Sidebar/Sidebar';
import TopBar from '../TopBar/TopBar';
import styles from './AppLayout.module.css';
import { apiGet } from '../../api/client';
import type { AuthStatus } from '../../api/types';

interface Props {
  active?: 'overview' | 'strategies' | 'orb' | 'nas' | 'nas-config' | 'scaleup' | 'straddles' | 'straddle45' | 'stock-wings' | 'nwv' | 'options-study' | 'straddle-study' | 'n500m' | 'strangle' | 'mst' | 'intraday75wr' | 'pair-trading' | 'scanner' | 'breakout-scanner' | 'ath-scanner' | 'indices' | 'backtest' | 'momentum-paper' | 'breakout-paper' | 'bluesky-paper' | 'ha-paper' | 'orb-paper' | 'ohol-paper' | 'fnoms-paper' | 'eod-breakout' | 'reports' | 'holdings' | 'options-data' | 'future-plans' | 'journal' | 'settings';
  children: ReactNode;
  topBarRight?: ReactNode;
}

export default function AppLayout({ active, children, topBarRight }: Props) {
  const [userName, setUserName] = useState('Trader');
  // Phone-only navigation drawer. Above 768px the sidebar is always in flow
  // and this flag does nothing.
  const [navOpen, setNavOpen] = useState(false);
  const location = useLocation();

  useEffect(() => {
    apiGet<AuthStatus>('/api/auth/status')
      .then((r) => {
        if (r.user_name) setUserName(r.user_name);
      })
      .catch(() => {
        /* swallow — topbar still renders */
      });
  }, []);

  // Navigating closes the drawer, so a tap on a page never leaves it covering
  useEffect(() => {
    setNavOpen(false);
  }, [location.pathname]);

  // Don't let the page behind the drawer scroll under a thumb
  useEffect(() => {
    if (!navOpen) return undefined;
    const prev = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      document.body.style.overflow = prev;
    };
  }, [navOpen]);

  return (
    <div className={styles.root}>
      <Sidebar
        active={active}
        userName={userName}
        mobileOpen={navOpen}
        onNavigate={() => setNavOpen(false)}
      />
      {navOpen && (
        <div
          className={styles.backdrop}
          onClick={() => setNavOpen(false)}
          role="presentation"
        />
      )}
      <div className={styles.main}>
        <TopBar userName={userName} right={topBarRight} onMenu={() => setNavOpen(true)} />
        <div className={styles.content} data-search-root>{children}</div>
      </div>
    </div>
  );
}
