'use client';

interface BarData {
  label: string;
  pnl: number;
}

export function MiniBar({ data, height = 56 }: { data: BarData[]; height?: number }) {
  const max = Math.max(...data.map((d) => Math.abs(d.pnl) || 1));
  const w = 100 / data.length;

  return (
    <svg viewBox={`0 0 100 ${height}`} className="w-full" style={{ height }}>
      {data.map((d, i) => {
        const h2 = (Math.abs(d.pnl) / max) * (height / 2 - 4);
        const y = d.pnl >= 0 ? height / 2 - h2 : height / 2;
        return (
          <g key={i}>
            <rect x={i * w + w * 0.15} y={y} width={w * 0.7} height={Math.max(h2, 1)} rx="2" fill={d.pnl >= 0 ? '#059669' : '#dc2626'} opacity=".85" />
            <text x={i * w + w / 2} y={height - 1} textAnchor="middle" fontSize="5" fill="#64748b" fontFamily="monospace">{d.label}</text>
          </g>
        );
      })}
      <line x1="0" y1={height / 2} x2="100" y2={height / 2} stroke="#cbd5e1" strokeWidth=".4" strokeDasharray="2,2" />
    </svg>
  );
}
