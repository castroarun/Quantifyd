import type { ReactNode } from 'react';
import { useEffect, useState } from 'react';
import Sidebar from '../Sidebar/Sidebar';
import TopBar from '../TopBar/TopBar';
import styles from './AppLayout.module.css';
import { apiGet } from '../../api/client';
import type { AuthStatus } from '../../api/types';

interface Props {
  active?: 'strategies' | 'orb' | 'nas' | 'reports' | 'holdings' | 'options-data' | 'future-plans' | 'settings';
  children: ReactNode;
  topBarRight?: ReactNode;
}

export default function AppLayout({ active, children, topBarRight }: Props) {
  const [userName, setUserName] = useState('Trader');

  useEffect(() => {
    apiGet<AuthStatus>('/api/auth/status')
      .then((r) => {
        if (r.user_name) setUserName(r.user_name);
      })
      .catch(() => {
        /* swallow — topbar still renders */
      });
  }, []);

  return (
    <div className={styles.root}>
      <Sidebar active={active} userName={userName} />
      <div className={styles.main}>
        <TopBar userName={userName} right={topBarRight} />
        <div className={styles.content}>{children}</div>
      </div>
    </div>
  );
}
