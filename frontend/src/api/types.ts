// TypeScript shapes for the Flask API responses we consume.
// These are intentionally permissive — backend returns a lot of extra fields
// we ignore.

export interface AuthStatus {
  authenticated: boolean;
  user_name?: string;
  user_id?: string;
}

/* ---------- ORB ---------- */

export interface ORBDailyState {
  instrument: string;
  today_open?: number | null;
  or_high?: number | null;
  or_low?: number | null;
  or_finalized?: boolean;
  cpr_pivot?: number | null;
  cpr_tc?: number | null;
  cpr_bc?: number | null;
  cpr_width_pct?: number | null;
  is_wide_cpr_day?: boolean;
  prev_day_date?: string | null;
  gap_pct?: number | null;
  trades_taken?: number;
  vwap?: number | null;
  rsi?: number | null;
  cpr_dir?: 'BULL' | 'BEAR' | 'NEUTRAL' | null;
  status?: string;
}

export interface ORBPosition {
  instrument: string;
  direction: 'LONG' | 'SHORT';
  qty: number;
  entry_price: number;
  sl_price?: number;
  target_price?: number;
  /** @deprecated use sl_price (kept for any legacy consumer) */
  stop_loss?: number;
  /** @deprecated use target_price */
  target?: number;
  entry_time?: string;
  ltp?: number;
  pnl_inr?: number;
  pnl_pts?: number;
  entry_reason?: string;
  kite_sl_order_id?: string;
  or_high?: number;
  or_low?: number;
  conviction_grade?: 'A+' | 'A' | 'B' | 'C';
  conviction_score?: number;
  conviction_stars?: Array<{ key: string; hit: boolean; desc: string }>;
  // Closed-row fields (present when the Positions table is showing history)
  status?: 'OPEN' | 'CLOSED';
  exit_price?: number;
  exit_time?: string;
  exit_reason?: string;
}

export interface ORBClosedTrade {
  instrument: string;
  direction: 'LONG' | 'SHORT';
  qty: number;
  entry_price: number;
  exit_price?: number;
  pnl_inr?: number;
  pnl_pts?: number;
  entry_time?: string;
  exit_time?: string;
  exit_reason?: string;
  trade_date?: string;
}

export interface ORBStockSummary {
  daily_state: ORBDailyState;
  position: ORBPosition | null;
  today_result: ORBClosedTrade | null;
  qty: number;
  capital_per_trade: number;
  sl_risk_inr: number;
  price: number;
}

export interface ORBConfig {
  capital: number;
  max_concurrent_trades: number;
  allocation_per_trade: number;
  min_margin_for_trade: number;
  margin_buffer: number;
  or_minutes: number;
  r_multiple: number;
  sl_type: string;
  last_entry_time: string;
  eod_exit_time: string;
  use_vwap_filter: boolean;
  use_rsi_filter: boolean;
  use_cpr_dir_filter: boolean;
  use_cpr_width_filter: boolean;
  cpr_width_threshold_pct: number;
  use_gap_filter: boolean;
  gap_threshold_pct: number;
}

export interface ORBStats {
  total_trades?: number;
  wins?: number;
  losses?: number;
  total_pnl?: number;
  win_rate?: number;
  [k: string]: unknown;
}

export interface ORBState {
  enabled: boolean;
  live_trading: boolean;
  universe: string[];
  capital: number;
  daily_loss_limit: number;
  daily_loss_limit_pct?: number;
  enforce_daily_loss_cap?: boolean;
  mis_leverage?: number;
  use_risk_based_sizing?: boolean;
  risk_per_trade_pct?: number;
  max_notional_per_trade?: number;
  stocks: Record<string, ORBStockSummary>;
  open_positions: ORBPosition[];
  today_closed: ORBClosedTrade[];
  today_pnl: number;
  stats: ORBStats;
  margin: { available?: number; used?: number; [k: string]: unknown };
  fund_alert: unknown;
  unread_notifications: number;
  config: ORBConfig;
}

export interface ORBSignal {
  id?: number;
  instrument: string;
  direction?: string;
  signal_time?: string;
  price?: number;
  reason?: string;
  action_taken?: string;
  status?: string;
}

export interface ORBCandidateBrokenOut {
  sym: string;
  ltp: number;
  or_high: number;
  or_low: number;
  or_width_pct?: number;
  cpr_width_pct: number;
  cpr_is_wide?: boolean;
  gap_pct: number;
  rsi_15m?: number | null;
  dist_up_pct?: number | null;
  dist_dn_pct?: number | null;
  side: 'LONG' | 'SHORT';
  past_pct: number;
  conviction_score?: number;
  conviction_grade?: 'A+' | 'A' | 'B' | 'C';
  conviction_stars?: Array<{ key: string; hit: boolean; desc: string }>;
}

export interface ORBCandidateWatching {
  sym: string;
  ltp: number;
  or_high: number;
  or_low: number;
  or_width_pct?: number;
  cpr_width_pct: number;
  cpr_is_wide?: boolean;
  gap_pct: number;
  rsi_15m?: number | null;
  dist_up_pct?: number | null;
  dist_dn_pct?: number | null;
  side_hint: 'both' | 'long' | 'short' | 'blocked';
  long_gap_ok?: boolean;
  long_rsi_ok?: boolean;
  short_rsi_ok?: boolean;
}

export interface ORBCandidateExcluded {
  sym: string;
  ltp: number;
  cpr?: number;
  reason: string;
}

export interface ORBCandidates {
  broken_out: ORBCandidateBrokenOut[];
  watching: ORBCandidateWatching[];
  excluded: ORBCandidateExcluded[];
  in_position: string[];
  traded_today: string[];
  as_of: string;
}

/* ---------- NAS ---------- */

export interface NASPosition {
  leg: 'CE' | 'PE';
  tradingsymbol?: string;
  strike?: number;
  entry_price?: number;
  entry_premium?: number; // legacy alias
  exit_price?: number;
  sl_price?: number;
  ltp?: number;
  qty?: number;
  lots?: number;
  pnl_inr?: number;
  entry_time?: string;
  exit_time?: string;
  exit_reason?: string;
  system?: string;
  status?: string;
}

export interface NASTrade {
  tradingsymbol?: string;
  leg?: string;
  pnl_inr?: number;
  entry_time?: string;
  exit_time?: string;
  exit_reason?: string;
}

export interface NASStats {
  total_trades?: number;
  wins?: number;
  losses?: number;
  total_pnl?: number;
  today_pnl?: number;
  total_reentries?: number;
  win_rate?: number;
  profit_factor?: number;
  sl_hits_today?: number;
  [k: string]: unknown;
}

export interface NASCoreState {
  atr_value?: number | null;
  atr_ma?: number | null;
  is_squeezing?: number | boolean | null;
  squeeze_count?: number | null;
  spot_price?: number | null;
  day_open?: number | null;
  daily_atr?: number | null;
  daily_pnl?: number | null;
  [k: string]: unknown;
}

export interface NASConfigShape {
  paper_trading_mode?: boolean;
  enabled?: boolean;
  lots_per_leg?: number;
  min_squeeze_bars?: number;
  entry_start_time?: string;
  entry_end_time?: string;
  time_exit?: string;
  eod_squareoff_time?: string;
  target_entry_premium?: number;
  max_daily_loss?: number;
  [k: string]: unknown;
}

export interface NASState {
  state?: NASCoreState;
  stats: NASStats;
  config?: NASConfigShape;
  positions: {
    ce: NASPosition[];
    pe: NASPosition[];
    total_active: number;
    closed_today: NASPosition[];
  };
  recent_signals?: unknown[];
  recent_trades?: NASTrade[];
  margin?: { available?: number; used?: number; [k: string]: unknown };
}

export interface NASTickPayload {
  type: 'tick' | 'offline';
  spot?: number;
  legs?: Record<string, { ltp: number; entry: number; leg: 'CE' | 'PE' }>;
  ts?: number;
}

/* ---------- NAS Report ---------- */

export interface NASReportSystem {
  total_trades?: number;
  total_pnl?: number;
  avg_pnl?: number;
  win_rate?: number;
  winners?: number;
  losers?: number;
  max_win?: number;
  max_loss?: number;
  profit_factor?: number;
  error?: string;
}

export interface NASReportDayPosition {
  id?: number;
  strangle_id?: number;
  leg?: string;
  tradingsymbol?: string;
  qty?: number;
  entry_price?: number;
  exit_price?: number | null;
  entry_time?: string;
  exit_time?: string | null;
  exit_reason?: string | null;
  status?: string;
}

export interface NASReportDaySystem {
  positions: NASReportDayPosition[];
  trades: Record<string, unknown>[];
  day_pnl: number;
  trade_count: number;
}

export interface NASReportData {
  systems: Record<string, NASReportSystem>;
  daily_snapshots: Record<string, Record<string, NASReportDaySystem>>;
}

/* ---------- ORB Backtest ---------- */

export interface ORBBacktestSignal {
  instrument: string;
  direction?: 'LONG' | 'SHORT' | null;
  signal_type: 'TAKEN' | 'BLOCKED' | 'NO_BREAKOUT' | 'SKIP_WIDE_CPR' | 'NO_DATA' | 'ERROR';
  block_reason?: string | null;
  entry_time?: string | null;
  entry_price?: number | null;
  exit_time?: string | null;
  exit_price?: number | null;
  exit_reason?: string | null;
  pnl_pct?: number | null;
  pnl_inr?: number | null;
  or_high?: number | null;
  or_low?: number | null;
  gap_pct?: number | null;
  cpr_width_pct?: number | null;
  rsi_15m?: number | null;
}

export interface ORBBacktestRun {
  run_date: string;
  generated_at?: string;
  universe_size?: number;
  trades_taken: number;
  signals_blocked: number;
  net_pnl_inr: number;
  signals?: ORBBacktestSignal[];
}

/* ---------- Holdings ---------- */

export interface HoldingsRecord {
  tradingsymbol: string;
  qty: number;
  avg_price: number;
  ltp: number;
  prev_close?: number | null;
  day_pct: number;
  day_pnl_inr: number;
  invested: number;
  current: number;
  total_pnl_inr: number;
  total_pnl_pct: number;
  week52_high?: number | null;
  week52_high_date?: string | null;
  week52_low?: number | null;
  week52_low_date?: string | null;
  all_time_high?: number | null;
  pct_from_52h?: number | null;
  pct_from_52l?: number | null;
  pct_from_ath?: number | null;
  change_5d_pct?: number | null;
  change_20d_pct?: number | null;
  sparkline?: number[] | null; // last ~252 daily closes
  tag?: string; // in extremes payloads only
}

export interface HoldingsSummary {
  count: number;
  invested: number;
  current: number;
  day_pnl: number;
  day_pct: number;
  total_pnl: number;
  total_pct: number;
}

export interface HoldingsEvent {
  id?: number;
  tradingsymbol: string;
  event_date: string;
  event_type: 'results' | 'dividend' | 'split' | 'bonus' | 'buyback' | 'meeting';
  purpose?: string | null;
  detail?: string | null;
  record_date?: string | null;
}

export interface HoldingsDigest {
  summary: HoldingsSummary;
  holdings: HoldingsRecord[];
  movers_today: { gainers: HoldingsRecord[]; losers: HoldingsRecord[] };
  movers_weekly: { gainers: HoldingsRecord[]; losers: HoldingsRecord[] };
  extremes: { high: HoldingsRecord[]; low: HoldingsRecord[] };
  events: HoldingsEvent[];
  next_event: HoldingsEvent | null;
}

export interface HoldingsSnapshot {
  snap_date: string;
  generated_at: string;
  summary: HoldingsSummary;
  movers_today: { gainers: HoldingsRecord[]; losers: HoldingsRecord[] };
  extremes: { high: HoldingsRecord[]; low: HoldingsRecord[] };
  holdings: HoldingsRecord[];
}

export interface HoldingsSnapshotSummary {
  snap_date: string;
  generated_at: string;
  day_pnl: number;
  day_pct: number;
  total_pnl: number;
  current: number;
  count: number;
}

/* ---------- Nifty ORB Strangle ---------- */

// Compact per-variant view returned by GET /api/strangle/state.
export interface StrangleVariantSummary {
  id: string;
  name: string;
  or_min: number;
  enabled: boolean;
  today_status: 'Idle' | 'Watching' | 'Open' | 'Closed' | 'Skip' | string;
  today_pnl: number;
  open_positions: number;
}

export interface StrangleState {
  today: string;
  spot_ltp: number | null;
  variants: StrangleVariantSummary[];
}

// Variant config row from STRANGLE_VARIANTS (echoed back by GET /api/strangle/variant/<id>).
export interface StrangleVariantConfig {
  id: string;
  name: string;
  or_min: number;
  rsi_lo_long: number | null;
  rsi_hi_short: number | null;
  apply_q4_filter: boolean;
  q4_threshold_pct?: number;
  apply_calm_filter: boolean;
  calm_threshold_pct?: number;
  apply_cpr_against_filter: boolean;
  pe_delta_target_long?: number;
  ce_delta_target_long?: number;
  pe_delta_target_short?: number;
  ce_delta_target_short?: number;
  lot_size: number;
  enabled: boolean;
  backtest_wr_pct: number;
  backtest_wins_per_year: number;
  backtest_trades_per_year: number;
}

export interface StrangleLeg {
  id?: number;
  position_id?: number;
  variant_id?: string;
  leg_type: 'PE' | 'CE';
  side?: string;            // 'SELL'
  tradingsymbol?: string | null;
  strike: number;
  expiry_date?: string;
  qty: number;
  entry_price: number;
  delta_at_entry?: number | null;
  iv_at_entry?: number | null;
  exit_price?: number | null;
  leg_pnl?: number | null;
  mtm_now?: number | null;
}

export interface StranglePosition {
  id: number;
  variant_id: string;
  symbol?: string;
  entry_date: string;
  entry_ts: string;
  direction: 'LONG' | 'SHORT';
  spot_at_entry: number;
  or_high: number;
  or_low: number;
  or_width_pct: number;
  sl_price: number;
  expiry_date: string;
  pe_strike: number;
  ce_strike: number;
  pe_entry_price: number;
  ce_entry_price: number;
  pe_delta_at_entry?: number | null;
  ce_delta_at_entry?: number | null;
  lot_size: number;
  qty: number;
  status: 'OPEN' | 'CLOSED';
  exit_date?: string | null;
  exit_ts?: string | null;
  pe_exit_price?: number | null;
  ce_exit_price?: number | null;
  spot_at_exit?: number | null;
  gross_pnl?: number | null;
  net_pnl?: number | null;
  exit_reason?: string | null;
  legs?: StrangleLeg[];
}

export interface StrangleMtm {
  pe_now: number | null;
  ce_now: number | null;
  pe_mtm: number | null;
  ce_mtm: number | null;
  net_mtm: number | null;
}

export interface StrangleDailyState {
  id?: number;
  variant_id: string;
  trade_date: string;
  or_high?: number | null;
  or_low?: number | null;
  or_width_pct?: number | null;
  day_filter_passed?: number;
  signal_seen?: number;
  rsi_confirmed?: number;
  entry_taken?: number;
  exit_taken?: number;
  exit_reason?: string | null;
  daily_pnl?: number;
  notes?: string | null;
  last_event_ts?: string | null;
}

export interface StrangleStats {
  total_trades: number;
  wins: number;
  losses: number;
  win_rate: number;
  total_pnl: number;
  avg_pnl: number;
  profit_factor: number | string;  // 'inf' when no losses
}

export interface StrangleTrade {
  id?: number;
  position_id?: number;
  variant_id: string;
  entry_date: string;
  exit_date: string;
  entry_ts?: string;
  exit_ts?: string;
  direction?: 'LONG' | 'SHORT';
  spot_at_entry?: number;
  spot_at_exit?: number;
  pe_strike?: number;
  ce_strike?: number;
  pe_entry?: number;
  ce_entry?: number;
  pe_exit?: number;
  ce_exit?: number;
  gross_pnl?: number;
  net_pnl: number;
  costs?: number;
  exit_reason?: string;
  hold_minutes?: number | null;
}

export interface StrangleVariantDetail {
  variant: StrangleVariantConfig;
  today_status: string;
  today_pnl: number;
  stats: StrangleStats;
  recent_trades: StrangleTrade[];
  open_position: StranglePosition | null;
  mtm: StrangleMtm | null;
  daily_state: StrangleDailyState | Record<string, never>;
}
