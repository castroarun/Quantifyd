import { useEffect, useState } from 'react';
import styles from './TopBar.module.css';
import StatusDot from '../StatusDot/StatusDot';
import { nowStamp } from '../../utils/time';

interface Props {
  connected?: boolean;
  connectedLabel?: string;
  right?: React.ReactNode;
}

export default function TopBar({ connected = true, connectedLabel, right }: Props) {
  const [stamp, setStamp] = useState(nowStamp());
  useEffect(() => {
    const id = setInterval(() => setStamp(nowStamp()), 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <div className={styles.bar}>
      <div className={styles.left}>
        <span className={styles.stamp}>{stamp}</span>
      </div>
      <div className={styles.right}>
        <StatusDot
          kind={connected ? 'connected' : 'disconnected'}
          label={connectedLabel ?? (connected ? 'Connected to Kite' : 'Disconnected')}
        />
        {right}
      </div>
    </div>
  );
}
