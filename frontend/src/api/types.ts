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
  entry_premium?: number;
  ltp?: number;
  qty?: number;
  lots?: number;
  pnl_inr?: number;
  entry_time?: string;
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
  [k: string]: unknown;
}

export interface NASState {
  state?: Record<string, unknown>;
  stats: NASStats;
  config?: {
    paper_trading_mode?: boolean;
    enabled?: boolean;
    lots_per_leg?: number;
    [k: string]: unknown;
  };
  positions: {
    ce: NASPosition[];
    pe: NASPosition[];
    total_active: number;
    closed_today: NASPosition[];
  };
  recent_signals?: unknown[];
  recent_trades?: NASTrade[];
}

export interface NASTickPayload {
  type: 'tick' | 'offline';
  spot?: number;
  legs?: Record<string, { ltp: number; entry: number; leg: 'CE' | 'PE' }>;
  ts?: number;
}
