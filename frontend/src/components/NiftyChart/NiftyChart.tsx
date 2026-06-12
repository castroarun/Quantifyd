import { useEffect, useRef, useState } from 'react';

// LIVE NIFTY 50 5-min candlestick, self-hosted from your own Kite feed.
// Candles: /static/nifty_5m.json (1-min cron from the options recorder).
// Live tick: /api/nas/ticker/status last_ltp overlays the forming candle (~5s).
// Daily CPR (cyan, solid) + Weekly CPR (amber, dashed) — subtle, clipped to view.

interface Candle { t: string; o: number; h: number; l: number; c: number; }
interface Cpr { P: number; TC: number; BC: number; R1: number; S1: number; R2: number; S2: number; }
interface Data { updated: string; day: string; last: number | null; candles: Candle[]; dailyCpr?: Cpr | null; weeklyCpr?: Cpr | null; }

const H = 440;
const DAILY = '#22d3ee';   // bright cyan — Daily CPR (solid)
const WEEKLY = '#f59e0b';  // bright amber — Weekly CPR (dashed)

export default function NiftyChart() {
  const [data, setData] = useState<Data | null>(null);
  const [ltp, setLtp] = useState<number | null>(null);
  const [w, setW] = useState(900);
  const wrapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setW(el.clientWidth || 900));
    ro.observe(el);
    setW(el.clientWidth || 900);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    let on = true;
    const load = () =>
      fetch(`/static/nifty_5m.json?t=${Date.now()}`, { cache: 'no-store' })
        .then((r) => r.json()).then((d: Data) => { if (on) setData(d); }).catch(() => {});
    load();
    const id = setInterval(load, 20000);
    return () => { on = false; clearInterval(id); };
  }, []);

  useEffect(() => {
    let on = true;
    const load = () =>
      fetch(`/api/nas/ticker/status`, { cache: 'no-store' })
        .then((r) => r.json()).then((d) => { if (on && d.last_ltp) setLtp(d.last_ltp); }).catch(() => {});
    load();
    const id = setInterval(load, 5000);
    return () => { on = false; clearInterval(id); };
  }, []);

  const padT = 14, padB = 24, padR = 58, padL = 8;
  const candles: Candle[] = data?.candles ? data.candles.map((c) => ({ ...c })) : [];
  if (ltp && candles.length) {
    const last = candles[candles.length - 1];
    last.c = ltp; last.h = Math.max(last.h, ltp); last.l = Math.min(last.l, ltp);
  }
  const lo = candles.length ? Math.min(...candles.map((c) => c.l)) : 0;
  const hi = candles.length ? Math.max(...candles.map((c) => c.h)) : 1;
  const pad = (hi - lo) * 0.06 || 1;
  const yMin = lo - pad, yMax = hi + pad;
  const plotW = Math.max(w - padL - padR, 100);
  const plotH = H - padT - padB;
  const n = candles.length || 1;
  const slot = plotW / n;
  const bodyW = Math.max(2.5, Math.min(slot * 0.72, 22)); // wider bodies → smaller gaps
  const y = (p: number) => padT + ((yMax - p) / (yMax - yMin)) * plotH;
  const x = (i: number) => padL + slot * i + slot / 2;
  const cur = ltp ?? data?.last ?? null;
  const ticks = 4;
  const gridVals = Array.from({ length: ticks + 1 }, (_, i) => yMin + ((yMax - yMin) * i) / ticks);

  const renderCpr = (cpr: Cpr | null | undefined, color: string, dash: string, tag: string) => {
    if (!cpr) return null;
    const inR = (v: number) => v >= yMin && v <= yMax;
    const pill = (yy: number, label: string, small?: boolean) => {
      const wd = label.length * (small ? 5.0 : 5.6) + 8;
      return (
        <>
          <rect x={padL} y={yy - (small ? 6 : 7)} width={wd} height={small ? 12 : 14} rx={3} fill={color} />
          <text x={padL + 4} y={yy + (small ? 2.5 : 3)} fontSize={small ? 8 : 9} fontWeight={700} fill="#06121a">{label}</text>
        </>
      );
    };
    return (
      <g>
        {cpr.BC <= yMax && cpr.TC >= yMin && (
          <g>
            <rect x={padL} y={y(cpr.TC)} width={w - padR - padL} height={Math.max(2, y(cpr.BC) - y(cpr.TC))} fill={color} opacity={0.16} />
            <line x1={padL} x2={w - padR} y1={y(cpr.P)} y2={y(cpr.P)} stroke={color} strokeWidth={1.4} strokeDasharray={dash} opacity={0.95} />
            {inR(cpr.P) && pill(y(cpr.P), `${tag} CPR`)}
          </g>
        )}
        {([['R1', cpr.R1], ['R2', cpr.R2], ['S1', cpr.S1], ['S2', cpr.S2]] as [string, number][]).map(([k, v]) => (
          inR(v) ? (
            <g key={tag + k}>
              <line x1={padL} x2={w - padR} y1={y(v)} y2={y(v)} stroke={color} strokeWidth={0.9} strokeDasharray={dash || '1 4'} opacity={0.7} />
              {pill(y(v), `${tag}·${k}`, true)}
            </g>
          ) : null
        ))}
        {cpr.P < yMin && <g>{pill(H - padB - 3, `${tag} CPR ↓ ${cpr.P.toFixed(0)}`, true)}</g>}
        {cpr.P > yMax && <g>{pill(padT + 8, `${tag} CPR ↑ ${cpr.P.toFixed(0)}`, true)}</g>}
      </g>
    );
  };

  return (
    <section style={{ margin: '0 0 18px' }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, margin: '0 0 8px', flexWrap: 'wrap' }}>
        <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--ink,#e6e8ec)' }}>NIFTY 50 · 5-min</span>
        {cur != null && <span style={{ fontSize: 16, fontWeight: 700, color: 'var(--accent,#f5b301)' }}>{cur.toFixed(2)}</span>}
        <span style={{ fontSize: 11, color: 'var(--ink-muted,#8b93a1)' }}>live {ltp ? '●' : '○'} · {data?.updated ?? '–'}</span>
        <span style={{ fontSize: 10.5, color: DAILY, marginLeft: 'auto' }}>— Daily CPR</span>
        <span style={{ fontSize: 10.5, color: WEEKLY }}>– – Weekly CPR</span>
      </div>
      <div ref={wrapRef} style={{ height: H, width: '100%', background: 'var(--surface,#0f131a)', border: '1px solid var(--border,#232936)', borderRadius: 12, overflow: 'hidden' }}>
        {candles.length === 0 ? (
          <div style={{ padding: 24, color: 'var(--ink-muted,#8b93a1)', fontSize: 13 }}>Loading NIFTY candles…</div>
        ) : (
          <svg width={w} height={H} style={{ display: 'block' }}>
            {gridVals.map((v, i) => (
              <g key={'g' + i}>
                <line x1={padL} x2={w - padR} y1={y(v)} y2={y(v)} stroke="var(--border,#232936)" strokeWidth={0.5} />
                <text x={w - padR + 5} y={y(v) + 3} fontSize={10} fill="var(--ink-muted,#8b93a1)">{v.toFixed(0)}</text>
              </g>
            ))}
            {renderCpr(data?.weeklyCpr, WEEKLY, '6 4', 'W')}
            {renderCpr(data?.dailyCpr, DAILY, '', 'D')}
            {candles.map((c, i) => {
              const up = c.c >= c.o;
              const col = up ? '#26a69a' : '#ef5350';
              const yo = y(c.o), yc = y(c.c), bx = x(i);
              return (
                <g key={'c' + i}>
                  <line x1={bx} x2={bx} y1={y(c.h)} y2={y(c.l)} stroke={col} strokeWidth={1} />
                  <rect x={bx - bodyW / 2} y={Math.min(yo, yc)} width={bodyW} height={Math.max(1, Math.abs(yc - yo))} fill={col} />
                </g>
              );
            })}
            {cur != null && (
              <g>
                <line x1={padL} x2={w - padR} y1={y(cur)} y2={y(cur)} stroke="var(--accent,#f5b301)" strokeWidth={0.8} strokeDasharray="4 3" />
                <rect x={w - padR} y={y(cur) - 8} width={padR - 2} height={16} fill="var(--accent,#f5b301)" />
                <text x={w - padR + 5} y={y(cur) + 3} fontSize={10} fontWeight={700} fill="#1a1300">{cur.toFixed(1)}</text>
              </g>
            )}
            {candles.map((c, i) => (i % 6 === 0 ? (
              <text key={'t' + i} x={x(i)} y={H - 7} fontSize={9} fill="var(--ink-muted,#8b93a1)" textAnchor="middle">{c.t}</text>
            ) : null))}
          </svg>
        )}
      </div>
    </section>
  );
}
