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
  stop_loss?: number;
  target?: number;
  entry_time?: string;
  ltp?: number;
  pnl_inr?: number;
  pnl_pts?: number;
  entry_reason?: string;
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
