'use client';

interface MetricCardProps {
  label: string;
  value: string | number;
  sub?: string;
  accent?: string;
}

export function MetricCard({ label, value, sub, accent }: MetricCardProps) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-3.5 flex flex-col gap-0.5 shadow-sm">
      <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">{label}</span>
      <span className={`text-xl font-bold tracking-tight font-mono ${accent || 'text-slate-900'}`}>{value}</span>
      {sub && <span className="text-xs text-slate-500">{sub}</span>}
    </div>
  );
}
