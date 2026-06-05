import { useEffect, useMemo, useState } from 'react';

/* ---------- types ---------- */
interface V1 {
  version: string; trigger_pct: number; dte_max: number | null; lots: number;
  per_day: Record<string, { series: [string, number][]; final: number; stopped: boolean; dte: number; expiry: string }>;
  cum_curve: [string, number][];
}
interface V2Trade {
  entry_day: string; exit_day: string; strike: number; expiry: string;
  exit_reason: string; pnl: number; wing_pnl: number; series: [string, number][];
}
interface V2 { version: string; move_stop: number; pt: number; wings: number; lots: number; trades: V2Trade[]; book_curve: [string, number][]; }

/* ---------- helpers ---------- */
const inr = (n: number) => `${n >= 0 ? '+' : '−'}₹${Math.abs(Math.round(n)).toLocaleString('en-IN')}`;
const col = (n: number) => (n >= 0 ? '#2dd4a7' : '#ff6b6b');

function LineChart({ pts, h = 90, label }: { pts: [string, number][]; h?: number; label?: string }) {
  if (!pts || pts.length < 2) return <div style={{ color: '#5a6072', fontSize: 12, padding: 8 }}>—</div>;
  const ys = pts.map((p) => p[1]);
  const W = 600;
  const min = Math.min(0, ...ys), max = Math.max(0, ...ys), rng = max - min || 1;
  const X = (i: number) => (i / (pts.length - 1)) * W;
  const Y = (y: number) => h - ((y - min) / rng) * h;
  const path = pts.map((p, i) => `${X(i)},${Y(p[1])}`).join(' ');
  const zeroY = Y(0);
  const last = ys[ys.length - 1];
  return (
    <div>
      {label && <div style={{ fontSize: 11, color: '#8b93a3', marginBottom: 2 }}>{label}</div>}
      <svg viewBox={`0 0 ${W} ${h}`} width="100%" height={h} preserveAspectRatio="none">
        <line x1="0" y1={zeroY} x2={W} y2={zeroY} stroke="#3a3f4b" strokeWidth="1" strokeDasharray="4 4" />
        <polyline points={path} fill="none" stroke={col(last)} strokeWidth="2" />
      </svg>
    </div>
  );
}

const card: React.CSSProperties = { border: '1px solid #2a2f3a', background: 'linear-gradient(180deg,#161a22,#12151c)', borderRadius: 12, padding: '16px 18px', marginBottom: 18 };
const stat = (label: string, value: string, c?: string) => (
  <div><div style={{ fontSize: 11, color: '#8b93a3' }}>{label}</div><div style={{ fontSize: 19, fontWeight: 700, color: c || '#e6e9ef' }}>{value}</div></div>
);

/* ---------- page ---------- */
export default function Straddles() {
  const [v1, setV1] = useState<V1 | null>(null);
  const [v2, setV2] = useState<V2 | null>(null);
  const [day1, setDay1] = useState<string | null>(null);
  const [tr2, setTr2] = useState<number | null>(null);

  useEffect(() => {
    fetch('/app/straddles/v1.json').then((r) => r.json()).then(setV1).catch(() => {});
    fetch('/app/straddles/v2.json').then((r) => r.json()).then(setV2).catch(() => {});
  }, []);

  const v1stats = useMemo(() => {
    if (!v1) return null;
    const f = Object.values(v1.per_day).map((d) => d.final);
    const tot = f.reduce((a, b) => a + b, 0);
    return { n: f.length, tot, mean: tot / (f.length || 1), win: 100 * f.filter((x) => x > 0).length / (f.length || 1) };
  }, [v1]);
  const v2stats = useMemo(() => {
    if (!v2) return null;
    const f = v2.trades.map((t) => t.pnl);
    const tot = f.reduce((a, b) => a + b, 0);
    return { n: f.length, tot, mean: tot / (f.length || 1), win: 100 * f.filter((x) => x > 0).length / (f.length || 1) };
  }, [v2]);

  const days1 = v1 ? Object.keys(v1.per_day).sort() : [];

  return (
    <div style={{ maxWidth: 1000 }}>
      <div className="page-title">Straddle Systems</div>
      <div className="page-subtitle">Two short-straddle systems on NIFTY · backtested on the recorded chain · paper-forward 10 lots</div>

      {/* ===== V1 ===== */}
      <section style={{ ...card, marginTop: 14 }}>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap', marginBottom: 4 }}>
          <span style={{ fontSize: 16, fontWeight: 700, color: '#e6e9ef' }}>V1 · Intraday one-and-done</span>
          <span style={{ background: '#1e2a3a', color: '#6db3f2', fontSize: 11, fontWeight: 600, padding: '2px 8px', borderRadius: 6 }}>0.4% move-stop · 0/1-DTE · exit 14:45</span>
          <span style={{ background: '#3a2a00', color: '#f5b301', fontSize: 11, fontWeight: 600, padding: '2px 8px', borderRadius: 6 }}>BACKTEST</span>
        </div>
        {v1stats && (
          <div style={{ display: 'flex', gap: 26, flexWrap: 'wrap', margin: '10px 0' }}>
            {stat('Total', inr(v1stats.tot), col(v1stats.tot))}
            {stat('Mean/day', inr(v1stats.mean), col(v1stats.mean))}
            {stat('Days', String(v1stats.n))}
            {stat('Win rate', `${Math.round(v1stats.win)}%`)}
          </div>
        )}
        {v1 && <LineChart pts={v1.cum_curve} h={100} label="Cumulative P&L across days (click a day below for its intraday curve)" />}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 10 }}>
          {days1.map((d) => {
            const f = v1!.per_day[d].final;
            return (
              <button key={d} onClick={() => setDay1(d === day1 ? null : d)}
                style={{ cursor: 'pointer', border: `1px solid ${d === day1 ? '#6db3f2' : '#2a2f3a'}`, background: d === day1 ? '#1e2a3a' : '#12151c', color: col(f), borderRadius: 6, padding: '4px 8px', fontSize: 11, fontWeight: 600 }}>
                {d.slice(5)} {inr(f)}
              </button>
            );
          })}
        </div>
        {day1 && v1 && (
          <div style={{ marginTop: 12, borderTop: '1px solid #2a2f3a', paddingTop: 10 }}>
            <LineChart pts={v1.per_day[day1].series} h={110}
              label={`${day1} · intraday P&L (entry → close) · ${v1.per_day[day1].stopped ? '0.4% STOP hit → flat' : 'held to 15:15'} · DTE ${v1.per_day[day1].dte} · final ${inr(v1.per_day[day1].final)}`} />
          </div>
        )}
      </section>

      {/* ===== V2 ===== */}
      <section style={card}>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap', marginBottom: 4 }}>
          <span style={{ fontSize: 16, fontWeight: 700, color: '#e6e9ef' }}>V2 · Positional bi-weekly</span>
          <span style={{ background: '#1e2a3a', color: '#6db3f2', fontSize: 11, fontWeight: 600, padding: '2px 8px', borderRadius: 6 }}>1.5% stop · PT-40% · ±500pt wings · re-enter · roll 1-DTE</span>
          <span style={{ background: '#3a2a00', color: '#f5b301', fontSize: 11, fontWeight: 600, padding: '2px 8px', borderRadius: 6 }}>BACKTEST</span>
        </div>
        {v2stats && (
          <div style={{ display: 'flex', gap: 26, flexWrap: 'wrap', margin: '10px 0' }}>
            {stat('Total', inr(v2stats.tot), col(v2stats.tot))}
            {stat('Mean/trade', inr(v2stats.mean), col(v2stats.mean))}
            {stat('Trades', String(v2stats.n))}
            {stat('Win rate', `${Math.round(v2stats.win)}%`)}
          </div>
        )}
        {v2 && <LineChart pts={v2.book_curve} h={100} label="Book cumulative P&L per trade (click a trade below for its day-by-day curve)" />}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 10 }}>
          {v2 && v2.trades.map((t, i) => (
            <button key={i} onClick={() => setTr2(i === tr2 ? null : i)}
              style={{ cursor: 'pointer', border: `1px solid ${i === tr2 ? '#6db3f2' : '#2a2f3a'}`, background: i === tr2 ? '#1e2a3a' : '#12151c', color: col(t.pnl), borderRadius: 6, padding: '4px 8px', fontSize: 11, fontWeight: 600 }}>
              {t.entry_day.slice(5)}→{t.exit_day.slice(5)} {inr(t.pnl)}
            </button>
          ))}
        </div>
        {tr2 != null && v2 && (
          <div style={{ marginTop: 12, borderTop: '1px solid #2a2f3a', paddingTop: 10 }}>
            <LineChart pts={v2.trades[tr2].series} h={110}
              label={`${v2.trades[tr2].entry_day} → ${v2.trades[tr2].exit_day} · ${v2.trades[tr2].strike} straddle · exit: ${v2.trades[tr2].exit_reason} · wings ${inr(v2.trades[tr2].wing_pnl)} · final ${inr(v2.trades[tr2].pnl)}`} />
          </div>
        )}
        <div style={{ fontSize: 11, color: '#6b7280', marginTop: 10 }}>
          Note: wings cost a net {v2 ? inr(v2.trades.reduce((a, t) => a + t.wing_pnl, 0)) : ''} over the book (overnight-gap protection you opted to keep). Single regime, ~6 weeks — SIGNAL, not yet validated.
        </div>
      </section>
    </div>
  );
}
