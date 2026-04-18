import { Link } from 'react-router-dom';

export default function NotFound() {
  return (
    <div
      style={{
        padding: '96px 48px',
        textAlign: 'center',
        color: 'var(--ink-muted)',
      }}
    >
      <div
        style={{
          fontSize: 'var(--text-xl)',
          color: 'var(--ink)',
          fontWeight: 500,
          marginBottom: 8,
          letterSpacing: 'var(--tracking-tight)',
        }}
      >
        Page not found
      </div>
      <div style={{ fontSize: 'var(--text-sm)', marginBottom: 20 }}>
        The page you requested doesn't exist.
      </div>
      <Link
        to="/strategies"
        style={{
          fontSize: 'var(--text-sm)',
          color: 'var(--ink)',
          borderBottom: '0.5px solid var(--hairline)',
          paddingBottom: 2,
        }}
      >
        Back to strategies
      </Link>
    </div>
  );
}
