import { useEffect, useState } from 'react';

/* Is this page actually live, and when did it last move?

   Both book pages repainted every 30s with no visual cue, so a page whose numbers had
   just changed looked identical to one whose feed had died on Friday. This says which,
   in the only terms that matter: how long ago the marks were taken.

   Green and pulsing while the feed is fresh; amber once it is older than the bake
   interval by a clear margin; grey outside market hours, when "not ticking" is correct
   rather than broken. */

const IST_OFFSET_MIN = 330;

function istNow() {
  const d = new Date();
  return new Date(d.getTime() + (d.getTimezoneOffset() + IST_OFFSET_MIN) * 60000);
}

function marketOpen() {
  const t = istNow();
  const day = t.getDay();
  if (day === 0 || day === 6) return false;
  const mins = t.getHours() * 60 + t.getMinutes();
  return mins >= 9 * 60 + 15 && mins <= 15 * 60 + 30;
}

export default function LiveTick({ updated, staleAfterSec = 150, label = 'marks' }:
  { updated?: string | null; staleAfterSec?: number; label?: string }) {
  const [, force] = useState(0);
  useEffect(() => {
    const id = setInterval(() => force((n) => n + 1), 1000);   // re-render the age
    return () => clearInterval(id);
  }, []);
  if (!updated) return null;

  /* The feed writes naive IST timestamps ("2026-09-07 11:32:01.9"). Parsing that in the
     browser's own zone would be wrong by the offset, so compare it against IST-now. */
  const t = Date.parse(updated.replace(' ', 'T'));
  if (isNaN(t)) return null;
  const ageSec = Math.max(0, Math.round((istNow().getTime() - t) / 1000));

  const open = marketOpen();
  const fresh = ageSec <= staleAfterSec;
  const color = !open ? 'var(--ink-muted,#8a8a85)'
    : fresh ? 'var(--accent-pos,#0F6E56)' : 'var(--accent-warn,#B45309)';
  const age = ageSec < 60 ? `${ageSec}s ago`
    : ageSec < 3600 ? `${Math.floor(ageSec / 60)}m ago`
    : `${Math.floor(ageSec / 3600)}h ago`;

  return (
    <span title={`${label} taken ${updated.slice(0, 19)} IST`}
          style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 11.5,
                   color, fontWeight: 600, whiteSpace: 'nowrap' }}>
      <span style={{ width: 7, height: 7, borderRadius: '50%', background: color,
                     animation: open && fresh ? 'liveTickPulse 1.6s ease-in-out infinite' : 'none' }} />
      {open ? (fresh ? `live · ${label} ${age}` : `${label} ${age} — feed may be stuck`)
            : `market closed · ${label} ${age}`}
      <style>{`@keyframes liveTickPulse{0%,100%{opacity:1}50%{opacity:.25}}
        @media (prefers-reduced-motion:reduce){*{animation:none!important}}`}</style>
    </span>
  );
}

/* A number that briefly highlights when it changes, so a moving price is visible
   without watching the clock. Falls back to plain text if the value is not a number. */
export function Tick({ v, render, className, style }:
  { v: number | null | undefined; render: (n: number | null | undefined) => string;
    className?: string; style?: React.CSSProperties }) {
  const [prev, setPrev] = useState<number | null | undefined>(v);
  const [dir, setDir] = useState<0 | 1 | -1>(0);
  useEffect(() => {
    if (v == null || prev == null || v === prev) { setPrev(v); return; }
    setDir(v > prev ? 1 : -1);
    setPrev(v);
    const id = setTimeout(() => setDir(0), 900);
    return () => clearTimeout(id);
  }, [v]);                                    // eslint-disable-line react-hooks/exhaustive-deps
  const flash = dir === 0 ? undefined
    : { boxShadow: `inset 0 0 0 99px ${dir > 0 ? 'rgba(15,110,86,0.22)' : 'rgba(163,45,45,0.22)'}`,
        transition: 'box-shadow .9s ease-out' };
  return <span className={className} style={{ ...style, ...flash }}>{render(v)}</span>;
}
