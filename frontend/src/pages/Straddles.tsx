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

/* ---------- light theme tokens ---------- */
const C = { ink: '#1B1B1A', muted: '#888780', faint: '#B4B2A9', sec: '#5F5E5A', hair: 'rgba(0,0,0,0.10)',
  hairSoft: 'rgba(0,0,0,0.06)', pos: '#0F6E56', neg: '#A32D2D', navy: '#1E3A8A', navySoft: '#EFF3FA',
  amber: '#B45309', amberSoft: '#FEF3C7', surface: '#FFFFFF', canvas: '#FAFAF9' };

const inr = (n: number) => `${n >= 0 ? '+' : '−'}₹${Math.abs(Math.round(n)).toLocaleString('en-IN')}`;
const col = (n: number) => (n >= 0 ? C.pos : C.neg);

const fmtY = (v: number) => `${v >= 0 ? '+' : '−'}₹${Math.abs(Math.round(v)).toLocaleString('en-IN')}`;
function LineChart({ pts, h = 130, label }: { pts: [string, number][]; h?: number; label?: string }) {
  if (!pts || pts.length < 2) return <div style={{ color: C.faint, fontSize: 12, padding: 8 }}>—</div>;
  const W = 600, PAD_L = 56, PAD_R = 10, PAD_T = 8, PAD_B = 18;
  const ys = pts.map((p) => p[1]);
  const min = Math.min(0, ...ys), max = Math.max(0, ...ys), rng = max - min || 1;
  const X = (i: number) => PAD_L + (i / (pts.length - 1)) * (W - PAD_L - PAD_R);
  const Y = (v: number) => PAD_T + (1 - (v - min) / rng) * (h - PAD_T - PAD_B);
  const line = pts.map((p, i) => `${X(i)},${Y(p[1])}`).join(' ');
  const area = `${X(0)},${Y(0)} ${line} ${X(pts.length - 1)},${Y(0)}`;
  const last = ys[ys.length - 1];
  const yticks = [max, 0, min].filter((v, i, a) => a.indexOf(v) === i);
  const xi = [0, Math.floor((pts.length - 1) / 2), pts.length - 1].filter((v, i, a) => a.indexOf(v) === i);
  return (
    <div>
      {label && <div style={{ fontSize: 11, color: C.muted, marginBottom: 2 }}>{label}</div>}
      <svg viewBox={`0 0 ${W} ${h}`} width="100%" height={h} preserveAspectRatio="none">
        <defs>
          <linearGradient id={`sg${h}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={col(last)} stopOpacity="0.16" />
            <stop offset="100%" stopColor={col(last)} stopOpacity="0" />
          </linearGradient>
        </defs>
        {yticks.map((v) => (
          <g key={v}>
            <line x1={PAD_L} x2={W - PAD_R} y1={Y(v)} y2={Y(v)}
              stroke={v === 0 ? 'rgba(0,0,0,0.20)' : 'rgba(0,0,0,0.07)'} strokeWidth="1"
              strokeDasharray={v === 0 ? '0' : '3 3'} />
            <text x={PAD_L - 6} y={Y(v) + 3} textAnchor="end" fontSize="9.5" fill={C.muted}>{fmtY(v)}</text>
          </g>
        ))}
        <polygon points={area} fill={`url(#sg${h})`} />
        <polyline points={line} fill="none" stroke={col(last)} strokeWidth="2" />
        {xi.map((i, k) => (
          <text key={k} x={X(i)} y={h - 5} fontSize="9.5" fill={C.muted}
            textAnchor={k === 0 ? 'start' : k === xi.length - 1 ? 'end' : 'middle'}>{pts[i][0]}</text>
        ))}
      </svg>
    </div>
  );
}

const card: React.CSSProperties = { border: `1px solid ${C.hair}`, background: C.surface, borderRadius: 10, padding: '16px 18px', marginBottom: 18, boxShadow: '0 1px 2px rgba(0,0,0,0.04)' };
const stat = (label: string, value: string, c?: string) => (
  <div><div style={{ fontSize: 11, color: C.muted }}>{label}</div><div style={{ fontSize: 19, fontWeight: 700, color: c || C.ink }}>{value}</div></div>
);
const chip = (bg: string, fg: string, t: string) => (
  <span style={{ background: bg, color: fg, fontSize: 11, fontWeight: 600, padding: '2px 8px', borderRadius: 6 }}>{t}</span>
);

/* ---------- page ---------- */
export default function Straddles() {
  const [v1, setV1] = useState<V1 | null>(null);
  const [v2, setV2] = useState<V2 | null>(null);
  const [day1, setDay1] = useState<string | null>(null);
  const [tr2, setTr2] = useState<number | null>(null);
  const [live, setLive] = useState<any>(null);

  useEffect(() => {
    fetch('/app/straddles/v1.json').then((r) => r.json()).then(setV1).catch(() => {});
    fetch('/app/straddles/v2.json').then((r) => r.json()).then(setV2).catch(() => {});
    const loadLive = () => fetch('/app/straddles_live.json?t=' + Date.now()).then((r) => r.json()).then(setLive).catch(() => {});
    loadLive();
    const id = setInterval(loadLive, 60000);
    return () => clearInterval(id);
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
  const btn = (sel: boolean, c: string): React.CSSProperties => ({
    cursor: 'pointer', border: `1px solid ${sel ? C.navy : C.hair}`, background: sel ? C.navySoft : C.surface,
    color: c, borderRadius: 6, padding: '4px 8px', fontSize: 11, fontWeight: 600,
  });

  return (
    <div style={{ maxWidth: 1000 }}>
      <div className="page-title">Straddle Systems</div>
      <div className="page-subtitle">Two short-straddle systems on NIFTY · backtested on the recorded chain · paper-forward 10 lots</div>

      {/* ===== TODAY · LIVE ===== */}
      {live && (
        <section style={{ ...card, marginTop: 14, borderColor: C.navy }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8, flexWrap: 'wrap' }}>
            <span style={{ fontSize: 16, fontWeight: 700, color: C.ink }}>Today · Live</span>
            {chip('#E7F2EE', C.pos, 'PAPER · 10 lots')}
            <span style={{ marginLeft: 'auto', fontSize: 11, color: C.muted }}>updated {String(live.updated_at || '').slice(11, 19)} · {live.day}</span>
          </div>
          <div style={{ display: 'flex', gap: 18, flexWrap: 'wrap' }}>
            {[['V1 · intraday one-and-done', live.v1], ['V2 · positional bi-weekly', live.v2]].map(([title, d]: any) => (
              <div key={title} style={{ flex: 1, minWidth: 300, border: `1px solid ${C.hair}`, borderRadius: 8, padding: 12 }}>
                <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
                  <span style={{ fontWeight: 700, color: C.ink }}>{title}</span>
                  <span style={{ fontSize: 22, fontWeight: 800, marginLeft: 'auto', color: d.status === 'idle' || d.status === 'flat' ? C.muted : col(d.pnl_now) }}>
                    {d.status === 'idle' || d.status === 'flat' ? d.status : inr(d.pnl_now)}
                  </span>
                </div>
                <div style={{ fontSize: 11, color: C.muted, margin: '2px 0 6px' }}>{d.detail}</div>
                {d.series && d.series.length >= 2
                  ? <LineChart pts={d.series} h={120} label={`intraday running P&L · low ${inr(d.low || 0)} · high ${inr(d.high || 0)}`} />
                  : <div style={{ fontSize: 12, color: C.faint, padding: 8 }}>{d.status === 'idle' ? 'no trade today (not 0/1-DTE)' : '—'}</div>}
              </div>
            ))}
          </div>
          <div style={{ fontSize: 11, color: C.muted, marginTop: 8 }}>Live paper · ticks every 5 min during market · recorded daily. Backtest history below.</div>
        </section>
      )}

      {/* ===== V1 ===== */}
      <section style={{ ...card, marginTop: 14 }}>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap', marginBottom: 4 }}>
          <span style={{ fontSize: 16, fontWeight: 700, color: C.ink }}>V1 · Intraday one-and-done</span>
          {chip(C.navySoft, C.navy, '0.4% move-stop · 0/1-DTE · exit 14:45')}
          {chip(C.amberSoft, C.amber, 'BACKTEST')}
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
              <button key={d} onClick={() => setDay1(d === day1 ? null : d)} style={btn(d === day1, col(f))}>
                {d.slice(5)} {inr(f)}
              </button>
            );
          })}
        </div>
        {day1 && v1 && (
          <div style={{ marginTop: 12, borderTop: `1px solid ${C.hair}`, paddingTop: 10 }}>
            <LineChart pts={v1.per_day[day1].series} h={110}
              label={`${day1} · intraday P&L (entry → close) · ${v1.per_day[day1].stopped ? '0.4% STOP hit → flat' : 'held to 15:15'} · DTE ${v1.per_day[day1].dte} · final ${inr(v1.per_day[day1].final)}`} />
          </div>
        )}
      </section>

      {/* ===== V2 ===== */}
      <section style={card}>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap', marginBottom: 4 }}>
          <span style={{ fontSize: 16, fontWeight: 700, color: C.ink }}>V2 · Positional bi-weekly</span>
          {chip(C.navySoft, C.navy, '1.5% stop · PT-40% · ±500pt wings · re-enter · roll 1-DTE')}
          {chip(C.amberSoft, C.amber, 'BACKTEST')}
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
            <button key={i} onClick={() => setTr2(i === tr2 ? null : i)} style={btn(i === tr2, col(t.pnl))}>
              {t.entry_day.slice(5)}→{t.exit_day.slice(5)} {inr(t.pnl)}
            </button>
          ))}
        </div>
        {tr2 != null && v2 && (
          <div style={{ marginTop: 12, borderTop: `1px solid ${C.hair}`, paddingTop: 10 }}>
            <LineChart pts={v2.trades[tr2].series} h={110}
              label={`${v2.trades[tr2].entry_day} → ${v2.trades[tr2].exit_day} · ${v2.trades[tr2].strike} straddle · exit: ${v2.trades[tr2].exit_reason} · wings ${inr(v2.trades[tr2].wing_pnl)} · final ${inr(v2.trades[tr2].pnl)}`} />
          </div>
        )}
        <div style={{ fontSize: 11, color: C.muted, marginTop: 10 }}>
          Note: wings cost a net {v2 ? inr(v2.trades.reduce((a, t) => a + t.wing_pnl, 0)) : ''} over the book (overnight-gap protection you opted to keep). Single regime, ~6 weeks — SIGNAL, not yet validated.
        </div>
      </section>
    </div>
  );
}
