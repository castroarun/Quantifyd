// Time helpers. All times are IST.

const timeFmt = new Intl.DateTimeFormat('en-IN', {
  hour: '2-digit',
  minute: '2-digit',
  hour12: false,
  timeZone: 'Asia/Kolkata',
});

const dateFmt = new Intl.DateTimeFormat('en-IN', {
  day: '2-digit',
  month: 'short',
  year: 'numeric',
  timeZone: 'Asia/Kolkata',
});

const dateTimeFmt = new Intl.DateTimeFormat('en-IN', {
  day: '2-digit',
  month: 'short',
  hour: '2-digit',
  minute: '2-digit',
  hour12: false,
  timeZone: 'Asia/Kolkata',
});

function parse(value: string | Date | number | null | undefined): Date | null {
  if (!value && value !== 0) return null;
  const d = value instanceof Date ? value : new Date(value);
  return Number.isNaN(d.getTime()) ? null : d;
}

export function formatTime(value: string | Date | number | null | undefined): string {
  const d = parse(value);
  return d ? timeFmt.format(d) : '—';
}

export function formatDate(value: string | Date | number | null | undefined): string {
  const d = parse(value);
  return d ? dateFmt.format(d) : '—';
}

export function formatDateTime(value: string | Date | number | null | undefined): string {
  const d = parse(value);
  return d ? dateTimeFmt.format(d) : '—';
}

/** "14:32:07 · Fri 18 Apr" style timestamp for the top bar. */
export function nowStamp(): string {
  const d = new Date();
  const t = new Intl.DateTimeFormat('en-IN', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
    timeZone: 'Asia/Kolkata',
  }).format(d);
  const date = new Intl.DateTimeFormat('en-IN', {
    weekday: 'short',
    day: '2-digit',
    month: 'short',
    timeZone: 'Asia/Kolkata',
  }).format(d);
  return `${t} · ${date}`;
}
