'use client';

const STATUS_MAP: Record<string, [string, string, string, string]> = {
  running: ['bg-emerald-50', 'text-emerald-700', 'bg-emerald-500', 'Live'],
  paused: ['bg-amber-50', 'text-amber-700', 'bg-amber-400', 'Paused'],
  error: ['bg-red-50', 'text-red-700', 'bg-red-500', 'Error'],
  stopped: ['bg-slate-100', 'text-slate-500', 'bg-slate-400', 'Off'],
};

export function StatusBadge({ status }: { status: string }) {
  const [bg, tx, dt, lb] = STATUS_MAP[status] || STATUS_MAP.stopped;
  return (
    <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-semibold ${bg} ${tx}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${dt} ${status === 'running' ? 'animate-pulse' : ''}`} />
      {lb}
    </span>
  );
}

export function RiskBadge({ risk }: { risk: string }) {
  const cls = risk === 'low' ? 'bg-emerald-100 text-emerald-800'
    : risk === 'medium' ? 'bg-amber-100 text-amber-800'
    : 'bg-red-100 text-red-800';
  return <span className={`px-1.5 py-0.5 rounded text-xs font-bold uppercase ${cls}`}>{risk}</span>;
}

export function SystemTypeBadge({ type }: { type: string }) {
  const cls = type === 'Intraday' ? 'bg-violet-100 text-violet-800'
    : type === 'Swing' ? 'bg-sky-100 text-sky-800'
    : type === 'Positional' ? 'bg-teal-100 text-teal-800'
    : 'bg-rose-100 text-rose-800';
  return <span className={`px-2 py-0.5 rounded-full text-xs font-bold ${cls}`}>{type}</span>;
}

export function ModeBadge({ mode }: { mode: 'paper' | 'live' }) {
  return mode === 'live'
    ? <span className="px-1.5 py-0.5 rounded text-xs font-bold bg-emerald-100 text-emerald-800">LIVE</span>
    : <span className="px-1.5 py-0.5 rounded text-xs font-bold bg-sky-100 text-sky-800">PAPER</span>;
}
