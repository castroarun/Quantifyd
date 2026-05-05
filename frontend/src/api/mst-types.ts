/* MST Index Strategy types — see docs/Design/MST-INDEX-STRATEGY-DESIGN.md */

export type MSTStateMachine =
  | 'NO_POSITION'
  | 'ARMED'
  | 'DEBIT_OPEN_L1'
  | 'CONDOR_OPEN_L1'
  | 'DEBIT_OPEN_L2'
  | 'CONDOR_OPEN_L2';

export type MSTMode = 'off' | 'paper' | 'live';

export interface MSTBar {
  bar_dt: string;
  open: number;
  high: number;
  low: number;
  close: number;
  atr21: number | null;
  st_upper: number | null;
  st_lower: number | null;
  st_value: number | null;
  direction: number | null;
  stoch_k: number | null;
  stoch_d: number | null;
  created_at?: string;
}

export interface MSTPosition {
  id: number;
  week_label: string;
  direction: 1 | -1;
  pyramid_level: 1 | 2;
  leg_role: string;
  side: 'BUY' | 'SELL';
  tradingsymbol: string;
  strike: number;
  option_type: 'CE' | 'PE';
  qty: number;
  entry_price: number | null;
  entry_time: string | null;
  exit_price: number | null;
  exit_time: string | null;
  exit_reason: string | null;
  status: 'PENDING' | 'OPEN' | 'CLOSED' | 'REJECTED';
  pnl_inr: number | null;
  order_id: string | null;
  paper_mode: 0 | 1;
  expiry_date: string;
  t_minus_1_date: string;
  created_at?: string;
}

export interface MSTEvent {
  id: number;
  event_type: string;
  direction: number | null;
  bar_dt: string;
  price: number | null;
  flip_high: number | null;
  flip_low: number | null;
  pyramid_level: number | null;
  notes: string | null;
  created_at: string;
}

export interface MSTConfig {
  enabled: boolean;
  paper_trading_mode: boolean;
  live_trading_enabled: boolean;
  underlying: string;
  lot_size: number;
  lots_per_leg: number;
  atr_period: number;
  multiplier: number;
  stoch_k: number;
  stoch_d: number;
  stoch_smooth: number;
  stoch_ob: number;
  stoch_os: number;
  spread_width: number;
  reset_width: number;
  min_credit_per_lot: number;
  min_dte_at_entry: number;
  pyramid_max_level: number;
  pyramid_d_lookback: number;
  pyramid_d_threshold: number;
  pyramid_safety_wing_pct: number;
  t_minus_1_close_hour: number;
  t_minus_1_close_minute: number;
}

export interface MSTState {
  state_machine: MSTStateMachine;
  mst_direction: 1 | -1 | 0;
  armed_high: number | null;
  armed_low: number | null;
  activated_at_bar: string | null;
  activated_at_atm: number | null;
  pyramid_atm: number | null;
  pyramid_level: 1 | 2;
  current_expiry_dt: string | null;
  current_t_minus_1: string | null;
  last_cst_bar: string | null;
  last_cst_high: number | null;
  last_cst_low: number | null;
  stoch_left_zone_since_cst: boolean;
  enabled: boolean;
  paper_mode: boolean;
  buffer_size: number;

  last_bar: MSTBar | null;
  open_legs: MSTPosition[];
  closed_today: MSTPosition[];
  today_pnl: number;
  config: MSTConfig;
  live_trading: boolean;
}

export interface MSTBarsResponse { bars: MSTBar[]; }
export interface MSTEventsResponse { events: MSTEvent[]; }
export interface MSTPositionsResponse { positions: MSTPosition[]; }

export interface MSTToggleResponse {
  mode?: MSTMode;
  enabled: boolean;
  paper_trading_mode: boolean;
  live_trading_enabled: boolean;
  error?: string;
}
