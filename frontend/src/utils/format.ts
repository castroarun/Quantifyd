// Indian-locale number formatters. Keep use consistent across the app.

const inrFmt = new Intl.NumberFormat('en-IN', { maximumFractionDigits: 0 });
const inrFmt2 = new Intl.NumberFormat('en-IN', {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});
const numFmt = new Intl.NumberFormat('en-IN', { maximumFractionDigits: 2 });

export function formatRs(value: number | null | undefined, decimals = 0): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  const abs = Math.abs(value);
  const formatted = decimals === 2 ? inrFmt2.format(abs) : inrFmt.format(abs);
  return `Rs${formatted}`;
}

/** P&L formatting: "+Rs18,420" or "−Rs2,062" (U+2212 minus sign). */
export function formatPnl(value: number | null | undefined, decimals = 0): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  const abs = Math.abs(value);
  const formatted = decimals === 2 ? inrFmt2.format(abs) : inrFmt.format(abs);
  if (value > 0) return `+Rs${formatted}`;
  if (value < 0) return `\u2212Rs${formatted}`;
  return `Rs${formatted}`;
}

export function pnlClass(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return 'pnl-zero';
  if (value > 0) return 'pnl-pos';
  if (value < 0) return 'pnl-neg';
  return 'pnl-zero';
}

export function formatNumber(value: number | null | undefined, decimals = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return new Intl.NumberFormat('en-IN', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value);
}

export function formatInt(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return numFmt.format(value);
}

export function formatPct(value: number | null | undefined, decimals = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return `${value.toFixed(decimals)}%`;
}
