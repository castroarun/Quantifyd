// Scale-Up plan: weekly targets & progress (NIFTY + SENSEX combined).
// Locked 19-Aug-2026 (Option B): venue-split with a reduced NIFTY Thursday book.
// Append a WEEKS row every Friday evening (registered in the Ops Center); targets
// change ONLY via the weekly stack reassessment — never silently.

export interface DayPlan {
  day: string;
  books: string;
  lots: string;
  margin: number; // Rs lakh, concurrent
}

// The locked 1.0x configuration (steady state from Mon 24-Aug; TB books 8L until then).
export const CONFIG_PLAN: DayPlan[] = [
  { day: 'Mon', books: 'NIFTY suite + COMB (TimeB dropped 23-Aug — paper study only)', lots: '6 + 2', margin: 13.2 },
  { day: 'Tue', books: 'NIFTY suite + COMB + TB-N (09:30–11:00) — NIFTY expiry', lots: '6 + 2 + 10', margin: 29.7 },
  { day: 'Wed', books: 'TB-SENSEX window (10:30–12:00) only', lots: '10', margin: 20.4 },
  { day: 'Thu', books: 'SENSEX suite + TB-SX full-day + NIFTY COMB + TB-N-Thu (09:25) — SENSEX expiry', lots: '6 + 10 + 2 + 3', margin: 40.9 },
  { day: 'Fri', books: 'NIFTY suite + COMB + TB-N (10:00–12:00)', lots: '6 + 2 + 10', margin: 29.7 },
];

export const TARGET = {
  label: '1.0× · locked 19-Aug-2026',
  pnlWeek: 44000,        // Rs — 50% haircut of the ~Rs87k lab basis after Monday TimeB was dropped (23-Aug; Fri kept)
  ddFloor: -30000,       // Rs, worst-week guide (joint study maxDDs)
  peakMarginL: 40.9,     // Rs lakh, Thursday
  capitalL: 44.7,        // Rs lakh, current
  cashEquivL: 20.8,      // cash + liquid-fund collateral (50% rule basis)
  basis:
    'The bar is ₹45k/wk = a 50% haircut of the ₹89k lab in-sample sum (research/111 verdict, 19-Aug: ' +
    'walk-forward retained ~half the in-sample edge OOS, and early live-vs-lab gaps point the same way — ' +
    'e.g. TB-N Tue live +276/lot vs lab 1,326/lot). Lab decomposition: NIFTY suite 2.8k/d ×3d + COMB 1.3k/d ×4d + ' +
    'TB-N (Mon 3.8k · Tue 13.3k · Fri 5.5k) + COMB-Thu 3L 5.1k + TB-SX (Wed 3.2k · Thu 28.6k) + SENSEX suite Thu 16.5k. ' +
    'Restore toward the lab basis only after 4 consecutive live weeks ≥ the ₹45k bar.',
};

export type WeekStatus = 'transition' | 'progress' | 'ontrack' | 'miss';

export interface WeekRow {
  week: string;          // e.g. 'W/C 24 Aug 2026'
  pnl: number | null;    // Rs, NIFTY + SENSEX combined, live books only
  dd: number | null;     // Rs, worst intra-week drawdown (combined)
  status: WeekStatus;
  note: string;
}

// Append one row per week (Friday evening ritual). NIFTY + SENSEX combined, live books.
export const WEEKS: WeekRow[] = [
  {
    week: 'W/C 18 Aug 2026',
    pnl: null,
    dd: null,
    status: 'transition',
    note: 'Transition week — not scored. Stepped through: SENSEX suite → 2 lots (18th), TB-SENSEX live 8L (19th), Thursday restructure + TB-SX Wed/Thu-only (19th eve). MAXV roll-stop + broker-truth guards landed.',
  },
  {
    week: 'W/C 24 Aug 2026',
    pnl: null,
    dd: null,
    status: 'progress',
    note: 'First full week at 1.0× — TB-N and TB-SX step 8 → 10 lots Monday (registered in Ops).',
  },
];

export interface LadderStep {
  step: string;
  nifty: string;
  sensex: string;
  peakL: number;       // Thursday concurrent margin, Rs lakh
  capitalL: number;    // required capital (peak ×1.15 + DD cushion)
  addL: number;        // vs current 44.7L
  liquidAddL: number;  // extra cash-equivalent needed for the 50% rule
  pnlWeek: number;     // Rs, scaled lab basis
  dd: number;          // Rs, scaled worst-week guide
}

// Future ladder — the whole table is SUBJECT TO the weekly reassessment and adjustments.
export const LADDER: LadderStep[] = [
  { step: '1.0× (now)', nifty: '6-2-10 (+3 Thu)', sensex: '6 / 10 · Wed+Thu', peakL: 40.9, capitalL: 44.7, addL: 0, liquidAddL: 0, pnlWeek: 89000, dd: -30000 },
  { step: '1.25×', nifty: '8-2-12 (+4 Thu)', sensex: '8 / 12 · Wed+Thu', peakL: 51.1, capitalL: 59.0, addL: 14.3, liquidAddL: 4.8, pnlWeek: 112000, dd: -38000 },
  { step: '1.5×', nifty: '9-3-15 (+5 Thu)', sensex: '9 / 15 · Wed+Thu', peakL: 61.4, capitalL: 71.0, addL: 26.3, liquidAddL: 9.9, pnlWeek: 134000, dd: -45000 },
  { step: '2.0×', nifty: '12-4-20 (+6 Thu)', sensex: '12 / 20 · Wed+Thu', peakL: 81.8, capitalL: 95.0, addL: 50.3, liquidAddL: 20.1, pnlWeek: 179000, dd: -60000 },
];

export const PAGE_UPDATED = '19 Aug 2026';
