/**
 * Global-search destination index.
 *
 * Two kinds of result exist in the app search: matches ON THE CURRENT PAGE
 * (scanned live from the DOM by GlobalSearch) and DESTINATIONS — the pages,
 * systems and studies you can jump to. This file is the destination half.
 *
 * The page list mirrors the Sidebar so the two never disagree about what the
 * app contains; systems and studies are derived from their existing registries
 * (data/strategies.ts, data/backtests.ts) so a new system or a newly published
 * study becomes searchable with no edit here.
 */
import { SYSTEMS, STATUS_LABEL, LAB_PAGES } from '../../data/strategies';
import { BACKTEST_STUDIES } from '../../data/backtests';
import { PAGE_HOTKEYS } from '../Sidebar/hotkeys';

export type DestGroup = 'Pages' | 'Systems' | 'Studies';

export interface Destination {
  id: string;
  label: string;
  group: DestGroup;
  /** Router path (basename /app is applied by the router). */
  to: string;
  /** Secondary line under the label. */
  hint?: string;
  /** Extra text folded into matching but not displayed. */
  keywords?: string;
  /** Single-letter sidebar shortcut, shown as a badge where one exists. */
  hotkey?: string;
}

interface PageDef {
  to: string;
  label: string;
  section: string;
  keywords?: string;
}

/** Mirrors the Sidebar, plus routes that exist without a nav row. */
const PAGES: PageDef[] = [
  // Workspace
  { to: '/overview', label: 'Desk', section: 'Workspace', keywords: 'home dashboard summary today' },
  { to: '/indices', label: 'Index Pulse', section: 'Workspace', keywords: 'nifty banknifty sensex vix levels' },
  { to: '/strategies', label: 'Strategies', section: 'Workspace', keywords: 'register systems live paper parked rules index' },
  { to: '/nas-config', label: 'NAS Config', section: 'Workspace', keywords: 'day gap matrix dte live paper toggle' },
  { to: '/scaleup', label: 'Scale-Up', section: 'Workspace', keywords: 'lots sizing ramp capital' },
  { to: '/backtest', label: 'Backtest', section: 'Workspace', keywords: 'studies research results verdict' },
  { to: '/eod-breakout', label: 'EOD', section: 'Workspace', keywords: 'end of day breakout scan' },
  { to: '/report', label: 'Performance', section: 'Workspace', keywords: 'reports pnl equity drawdown' },
  { to: '/journal', label: 'Journal', section: 'Workspace', keywords: 'trades notes review diary' },
  { to: '/journal/insights', label: 'Journal insights', section: 'Workspace', keywords: 'patterns stats review' },
  { to: '/future-plans', label: 'Future plans', section: 'Workspace', keywords: 'ideas roadmap sketches' },
  // Live
  { to: '/nas', label: 'NAS', section: 'Live', keywords: 'straddle atm atm2 atm4 9:16 squeeze nifty sensex positions' },
  { to: '/nas-panic', label: 'NAS Panic', section: 'Live', keywords: 'gauge spike volatility panic' },
  { to: '/straddles', label: 'Straddles', section: 'Live', keywords: 'v1 v2 short straddle ops center wings' },
  { to: '/straddle45', label: '45-DTE Straddle', section: 'Live', keywords: 'positional monthly straddle' },
  { to: '/stock-wings', label: 'Stock Wings', section: 'Live', keywords: 'stock strangle wings 45 dte c1' },
  { to: '/strangle', label: 'Strangle', section: 'Live', keywords: 'otm strangle' },
  // Paper books
  { to: '/ha-paper', label: 'HA 2-Green ₹20L', section: 'Paper books', keywords: 'heikin ashi paper book sleeves' },
  { to: '/fnoms-paper', label: 'F&O Multi-Signal ₹20L', section: 'Paper books', keywords: 'fnoms multi signal paper' },
  { to: '/breakout-paper', label: 'Breakout ₹10L', section: 'Paper books', keywords: 'breakout paper book' },
  { to: '/orb-paper', label: 'ORB Revival ₹10L', section: 'Paper books', keywords: 'opening range breakout paper' },
  { to: '/ohol-paper', label: 'OHOL 1-Lot', section: 'Paper books', keywords: 'open high open low' },
  { to: '/orb', label: 'ORB Cash', section: 'Paper books', keywords: 'opening range breakout cash intraday' },
  { to: '/nwv', label: 'NWV', section: 'Paper books', keywords: 'weekly view jade lizard iron condor' },
  { to: '/n500m', label: 'N500M', section: 'Paper books', keywords: 'nifty 500 momentum' },
  { to: '/mst', label: 'MST', section: 'Paper books', keywords: 'multi supertrend' },
  { to: '/intraday75wr', label: 'I75WR', section: 'Paper books', keywords: 'intraday 75 win rate' },
  { to: '/pair-trading', label: 'Pairs', section: 'Paper books', keywords: 'pair trading spread' },
  // Holdings
  { to: '/holdings', label: 'Holdings', section: 'Holdings', keywords: 'portfolio stocks positions zerodha' },
  { to: '/holdings/history', label: 'Holdings history', section: 'Holdings', keywords: 'past holdings snapshots' },
  { to: '/momentum-paper', label: 'True North LIVE', section: 'Holdings', keywords: 'momentum book true north' },
  { to: '/bluesky-paper', label: 'Open Alpha', section: 'Holdings', keywords: 'bluesky ath breakout open alpha' },
  { to: '/sleeves', label: 'Sleeves 50-50', section: 'Holdings', keywords: 'blend allocation sleeves' },
  // Options
  { to: '/straddle-study', label: 'Straddle Study', section: 'Options', keywords: 'straddle decay study' },
  { to: '/options-study', label: 'Opt Study', section: 'Options', keywords: 'options behaviour decay cpr candles' },
  { to: '/options-data', label: 'Options data', section: 'Options', keywords: 'chain recorder capture database' },
  // Scanner
  { to: '/scanner', label: 'F&O Scanner', section: 'Scanner', keywords: 'fno scan signals' },
  { to: '/breakout-scanner', label: 'Breakout Scanner', section: 'Scanner', keywords: 'breakout scan' },
  { to: '/ath-scanner', label: 'ATH & Breakouts', section: 'Scanner', keywords: 'all time high 52 week' },
  // General
  { to: '/settings', label: 'Settings', section: 'General', keywords: 'preferences config account' },
];

function clip(s: string | undefined, n: number): string {
  if (!s) return '';
  const t = s.replace(/\s+/g, ' ').trim();
  return t.length > n ? t.slice(0, n - 1) + '…' : t;
}

/** Built once per session — all three registries are already in the bundle. */
let CACHE: Destination[] | null = null;

export function destinations(): Destination[] {
  if (CACHE) return CACHE;

  const pages: Destination[] = PAGES.map((p) => ({
    id: `page:${p.to}`,
    label: p.label,
    group: 'Pages' as const,
    to: p.to,
    hint: p.section,
    keywords: `${p.keywords ?? ''} ${p.to}`,
    hotkey: PAGE_HOTKEYS[p.to],
  }));

  const labs: Destination[] = LAB_PAGES
    .filter((l) => !PAGES.some((p) => p.to === l.to))
    .map((l) => ({
      id: `lab:${l.to}`,
      label: l.name,
      group: 'Pages' as const,
      to: l.to,
      hint: l.what,
    }));

  const systems: Destination[] = SYSTEMS.map((s) => ({
    id: `sys:${s.id}`,
    label: s.name,
    group: 'Systems' as const,
    to: s.dashboard ?? '/strategies',
    hint: `${STATUS_LABEL[s.status]} · ${s.size}`,
    keywords: `${s.subtitle} ${s.rule} ${s.status} ${s.id}`,
  }));

  const studies: Destination[] = BACKTEST_STUDIES.map((b) => ({
    id: `study:${b.slug}`,
    label: b.title,
    group: 'Studies' as const,
    to: `/backtest/${b.slug}`,
    hint: clip(b.verdict, 90),
    keywords: `${b.slug} ${clip(b.cardBlurb, 240)}`,
  }));

  CACHE = [...pages, ...labs, ...systems, ...studies];
  return CACHE;
}
