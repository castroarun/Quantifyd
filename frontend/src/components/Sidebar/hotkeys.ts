/**
 * Single-letter page shortcuts.
 *
 * One map is the source of truth for both the badge rendered on each sidebar
 * row and the global key listener in Sidebar, so a badge can never advertise
 * a key that does not actually navigate.
 *
 * Mnemonics where the label allows (J journal, N nas, H holdings, M momentum,
 * O opt study); digits for the numeric books (5 = N500M, 7 = I75WR, 1 = OHOL
 * 1-Lot, 2 = the second ₹20L paper book).
 */
export const PAGE_HOTKEYS: Record<string, string> = {
  // Workspace
  '/strategies': 'S',
  '/orb': 'R',
  '/nas': 'N',
  '/nas-config': 'G',
  '/straddles': 'T',
  '/nwv': 'W',
  '/n500m': '5',
  '/mst': '3',
  '/intraday75wr': '7',
  '/pair-trading': 'P',
  '/backtest': 'B',
  '/eod-breakout': 'E',
  '/report': 'F',
  '/journal': 'J',
  '/future-plans': 'U',
  // Paper books
  '/ha-paper': 'A',
  '/fnoms-paper': '2',
  '/momentum-paper': 'M',
  '/breakout-paper': 'K',
  '/orb-paper': 'V',
  '/ohol-paper': '1',
  // Holdings
  '/holdings': 'H',
  // Options
  '/options-study': 'O',
  '/options-data': 'D',
  // Scanner
  '/scanner': 'C',
  '/breakout-scanner': 'Y',
  // General
  '/settings': 'Z',
};

/** Reverse map: lowercased key -> route, for the keydown handler. */
export const HOTKEY_ROUTES: Record<string, string> = Object.fromEntries(
  Object.entries(PAGE_HOTKEYS).map(([route, key]) => [key.toLowerCase(), route]),
);
