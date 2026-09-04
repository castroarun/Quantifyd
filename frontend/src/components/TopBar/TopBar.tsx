import { useEffect, useMemo, useState } from 'react';
import { useLocation } from 'react-router-dom';
import styles from './TopBar.module.css';
import StatusDot from '../StatusDot/StatusDot';
import GlobalSearch from '../GlobalSearch/GlobalSearch';
import StarButton from '../GlobalSearch/StarButton';
import { resolvePage } from '../GlobalSearch/favourites';
import { nowStamp } from '../../utils/time';

interface Props {
  connected?: boolean;
  connectedLabel?: string;
  userName?: string;
  right?: React.ReactNode;
  /** Opens the navigation drawer — rendered only on phone widths. */
  onMenu?: () => void;
}

export default function TopBar({ connected = true, connectedLabel, userName, right, onMenu }: Props) {
  const [stamp, setStamp] = useState(nowStamp());
  const location = useLocation();

  useEffect(() => {
    const id = setInterval(() => setStamp(nowStamp()), 1000);
    return () => clearInterval(id);
  }, []);

  // What the star favourites: this route, by the name the app knows it by
  const page = useMemo(() => resolvePage(location.pathname), [location.pathname]);

  return (
    <div className={styles.bar}>
      <div className={styles.left}>
        <button type="button" className={styles.menu} onClick={onMenu} aria-label="Open navigation menu">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
            <path d="M4 7h16M4 12h16M4 17h16" />
          </svg>
        </button>
        <span className={styles.stamp}>{stamp}</span>
      </div>
      <div className={styles.right}>
        <GlobalSearch />
        <StarButton target={page} />
        {userName ? <span className={styles.user}>{userName}</span> : null}
        <StatusDot
          className={styles.status}
          kind={connected ? 'connected' : 'disconnected'}
          label={connectedLabel ?? (connected ? 'Connected to Kite' : 'Disconnected')}
          title={connectedLabel ?? (connected ? 'Connected to Kite' : 'Disconnected')}
        />
        {right}
      </div>
    </div>
  );
}
