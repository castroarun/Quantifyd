"""
ORB -- Opening Range Breakout -- Live Trading Engine
=====================================================
Intraday cash equity system: MIS orders on Zerodha Kite.
Runs on Contabo VPS with APScheduler cron triggers.

LIVE TRADING -- REAL MONEY. No paper mode.

Universe: 7 high-beta F&O stocks on NSE.
Strategy: 15-min opening range breakout with VWAP, RSI, CPR filters.
Exits: OR opposite (SL), 1.5R target, or 15:20 EOD squareoff.
"""

import json
import logging
import threading
import time
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta

from services.orb_db import get_orb_db
from services.notifications import get_notification_service

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------

def _build_stocks_from_config(config):
    """Build stocks dict from config universe list."""
    universe = config.get('universe', [])
    return {sym: {'lot': 1, 'token': None} for sym in universe}


class ORBLiveEngine:
    """
    Live ORB trading engine.

    Lifecycle (called by APScheduler jobs in app.py):
        09:14  initialize_day()
        09:15-09:30  update_or()  [every minute or on 5-min close]
        09:30  update_or() finalizes OR
        09:35-14:00  evaluate_signals()  [every 5-min candle close]
        continuous   monitor_positions()  [every 30 seconds]
        15:20  eod_squareoff()
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.db = get_orb_db()
        self.stocks = _build_stocks_from_config(config)
        self._lock = threading.Lock()
        self._or_state = {}      # {sym: {'high': float, 'low': float, 'finalized': bool}}
        self._vwap_state = {}    # {sym: {'cum_pv': float, 'cum_vol': float}}
        self._positions = {}     # {sym: position_dict}  -- open position cache
        self._tokens_resolved = False
        self._last_margin = None  # cached margin info from Kite

    # ===================================================================
    # Kite helpers
    # ===================================================================

    def _get_kite(self):
        """Get authenticated Kite instance. Import deferred to avoid circular."""
        from services.kite_service import get_kite
        return get_kite()

    @property
    def allocation_per_trade(self):
        """Derived: capital / max_concurrent_trades."""
        capital = self.cfg.get('capital', 100000)
        max_trades = self.cfg.get('max_concurrent_trades', 3)
        return capital / max_trades

    @property
    def min_margin_for_trade(self):
        """Derived: allocation_per_trade * margin_buffer_multiplier."""
        return self.allocation_per_trade * self.cfg.get('margin_buffer_multiplier', 1.2)

    def _get_available_margin(self):
        """Fetch available CASH from Kite. For MIS equity, cash is what gets blocked."""
        try:
            kite = self._get_kite()
            margins = kite.margins()
            eq = margins.get('equity', {})
            cash = eq.get('available', {}).get('cash', 0)
            self._last_margin = {
                'cash': round(cash, 2),
                'live_balance': round(eq.get('available', {}).get('live_balance', 0), 2),
                'used': round(eq.get('utilised', {}).get('debits', 0), 2),
                'updated_at': datetime.now().isoformat(),
            }
            return cash
        except Exception as e:
            logger.warning(f"[ORB] Margin check failed: {e}")
            return None

    def _check_fund_alert(self):
        """Check if available margin is below 1.2x allocation threshold."""
        if not self._last_margin:
            return None
        alloc = self.allocation_per_trade
        min_required = self.min_margin_for_trade
        available = self._last_margin.get('available', 0)
        if available < min_required:
            return {
                'type': 'warning',
                'message': f'Low funds: Rs {available:,.0f} available, need Rs {min_required:,.0f} (1.2x per-stock alloc) for next trade',
                'available': available,
                'required': min_required,
            }
        return None

    def _resolve_tokens(self):
        """Resolve NSE instrument tokens for all stocks. Call once per day."""
        if self._tokens_resolved:
            return
        try:
            kite = self._get_kite()
            instruments = kite.instruments('NSE')
            token_map = {
                i['tradingsymbol']: i['instrument_token']
                for i in instruments
                if i['tradingsymbol'] in self.stocks
            }
            for sym in self.stocks:
                tok = token_map.get(sym)
                if tok:
                    self.stocks[sym]['token'] = tok
                else:
                    logger.warning(f"[ORB] No NSE token for {sym}")
            self._tokens_resolved = True
            logger.info(f"[ORB] Tokens resolved: {len(token_map)}/{len(self.stocks)} stocks")
        except Exception as e:
            logger.error(f"[ORB] Token resolution failed: {e}")

    def get_live_ltp(self, instruments):
        """
        Get LTP for multiple instruments.
        Args: instruments -- list of symbols (e.g. ['ADANIENT', 'TATASTEEL'])
        Returns: {symbol: ltp} dict
        """
        try:
            kite = self._get_kite()
            keys = [f'NSE:{sym}' for sym in instruments]
            quotes = kite.ltp(keys)
            result = {}
            for sym in instruments:
                key = f'NSE:{sym}'
                data = quotes.get(key)
                if data:
                    result[sym] = data['last_price']
            return result
        except Exception as e:
            logger.error(f"[ORB] LTP fetch failed: {e}")
            return {}

    def fetch_5min_candles(self, instrument, from_date, to_date):
        """
        Fetch 5-min historical candles from Kite API.
        Returns list of dicts with keys: date, open, high, low, close, volume.
        """
        token = self.stocks.get(instrument, {}).get('token')
        if not token:
            logger.error(f"[ORB] No token for {instrument}, cannot fetch candles")
            return []
        try:
            kite = self._get_kite()
            candles = kite.historical_data(
                instrument_token=token,
                from_date=from_date,
                to_date=to_date,
                interval='5minute',
                oi=False
            )
            return [
                {
                    'date': c['date'],
                    'open': c['open'],
                    'high': c['high'],
                    'low': c['low'],
                    'close': c['close'],
                    'volume': c.get('volume', 0),
                }
                for c in candles
            ]
        except Exception as e:
            logger.error(f"[ORB] 5-min candle fetch failed for {instrument}: {e}")
            return []

    def _fetch_daily_candles(self, instrument, lookback_days=10):
        """Fetch daily candles for prev-day HLC. Returns list of dicts."""
        token = self.stocks.get(instrument, {}).get('token')
        if not token:
            return []
        try:
            kite = self._get_kite()
            to_dt = datetime.now()
            from_dt = to_dt - timedelta(days=lookback_days)
            candles = kite.historical_data(
                instrument_token=token,
                from_date=from_dt,
                to_date=to_dt,
                interval='day',
                oi=False
            )
            return [
                {
                    'date': c['date'],
                    'open': c['open'],
                    'high': c['high'],
                    'low': c['low'],
                    'close': c['close'],
                    'volume': c.get('volume', 0),
                }
                for c in candles
            ]
        except Exception as e:
            logger.error(f"[ORB] Daily candle fetch failed for {instrument}: {e}")
            return []

    # ===================================================================
    # Indicators
    # ===================================================================

    def compute_vwap(self, candles_today):
        """
        Compute intraday VWAP from 5-min candles since 09:15.
        Returns float VWAP value, or None if no data.
        """
        if not candles_today:
            return None
        cum_pv = 0.0
        cum_vol = 0.0
        for c in candles_today:
            typical = (c['high'] + c['low'] + c['close']) / 3.0
            vol = c.get('volume', 0) or 0
            cum_pv += typical * vol
            cum_vol += vol
        if cum_vol <= 0:
            return None
        return round(cum_pv / cum_vol, 2)

    def compute_rsi_15m(self, instrument):
        """
        Compute RSI(14) on 15-min bars resampled from 5-min candles.
        Seeds from last 30 calendar days of 5-min data for accurate Wilder smoothing.
        Returns float RSI value (0-100), or None on failure.
        """
        try:
            now = datetime.now()
            from_dt = now - timedelta(days=30)
            candles = self.fetch_5min_candles(instrument, from_dt, now)
            if not candles or len(candles) < 20:
                return None

            df = pd.DataFrame(candles)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

            # Resample to 15-min OHLCV
            df_15m = df.resample('15min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna(subset=['close'])

            if len(df_15m) < 15:
                return None

            # Wilder's RSI(14)
            delta = df_15m['close'].diff()
            gain = delta.clip(lower=0)
            loss = (-delta).clip(lower=0)
            period = 14
            avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            rsi = 100.0 - (100.0 / (1.0 + rs))

            latest_rsi = rsi.iloc[-1]
            if pd.isna(latest_rsi):
                return None
            return round(float(latest_rsi), 2)
        except Exception as e:
            logger.error(f"[ORB] RSI computation failed for {instrument}: {e}")
            return None

    # ===================================================================
    # Day initialization (call at 09:14 AM)
    # ===================================================================

    def initialize_day(self):
        """
        Called at 09:14 AM. For each stock:
        1. Resolve instrument tokens
        2. Fetch prev trading day HLC (safe 7-day lookback)
        3. Compute CPR (pivot, tc, bc, width)
        4. Get today's open price
        5. Compute gap_pct
        6. Create/update orb_daily_state row
        7. Compute qty per stock
        """
        logger.info("[ORB] === Initializing day ===")
        self._resolve_tokens()

        today_str = date.today().isoformat()
        cfg = self.cfg

        # Reset in-memory state
        with self._lock:
            self._or_state = {}
            self._vwap_state = {}
            self._positions = {}

        # Reload any open positions from DB (server restart recovery)
        open_pos = self.db.get_open_positions()
        with self._lock:
            for p in open_pos:
                self._positions[p['instrument']] = p

        for sym in self.stocks:
            try:
                # 1. Daily state row
                state = self.db.get_or_create_daily_state(sym, today_str)

                # 2. Prev day HLC (fetch last 7 days, take most recent completed day)
                daily_candles = self._fetch_daily_candles(sym, lookback_days=10)
                if not daily_candles:
                    logger.warning(f"[ORB] No daily data for {sym}, skipping")
                    continue

                # Filter out today's candle if present
                today_dt = date.today()
                prev_candles = [
                    c for c in daily_candles
                    if (c['date'].date() if hasattr(c['date'], 'date') else
                        datetime.strptime(str(c['date'])[:10], '%Y-%m-%d').date()) < today_dt
                ]
                if not prev_candles:
                    logger.warning(f"[ORB] No previous day data for {sym}")
                    continue

                prev = prev_candles[-1]  # most recent previous trading day
                prev_high = prev['high']
                prev_low = prev['low']
                prev_close = prev['close']
                prev_date_raw = prev['date']
                if hasattr(prev_date_raw, 'date'):
                    prev_date_str = prev_date_raw.date().isoformat()
                else:
                    prev_date_str = str(prev_date_raw)[:10]

                # 3. CPR
                pivot = (prev_high + prev_low + prev_close) / 3.0
                bc = (prev_high + prev_low) / 2.0
                tc = (pivot - bc) + pivot
                cpr_width_pct = abs(tc - bc) / pivot * 100.0 if pivot > 0 else 0
                is_wide = 1 if cpr_width_pct > cfg.get('cpr_width_threshold_pct', 0.5) else 0

                # 4. Today's open (from LTP at this time, or first candle later)
                ltps = self.get_live_ltp([sym])
                today_open = ltps.get(sym)

                # 5. Gap
                gap_pct = None
                if today_open and prev_close and prev_close > 0:
                    gap_pct = round((today_open - prev_close) / prev_close * 100.0, 4)

                # 6. Qty per stock
                allocation = self.allocation_per_trade
                if today_open and today_open > 0:
                    qty = int(allocation // today_open)
                else:
                    qty = 0
                self.stocks[sym]['qty'] = max(qty, 1)

                # 7. Update DB
                self.db.update_daily_state(sym, today_str,
                    prev_day_high=round(prev_high, 2),
                    prev_day_low=round(prev_low, 2),
                    prev_day_close=round(prev_close, 2),
                    prev_day_date=prev_date_str,
                    cpr_pivot=round(pivot, 2),
                    cpr_tc=round(tc, 2),
                    cpr_bc=round(bc, 2),
                    cpr_width_pct=round(cpr_width_pct, 4),
                    is_wide_cpr_day=is_wide,
                    today_open=round(today_open, 2) if today_open else None,
                    gap_pct=gap_pct,
                )

                # Init OR tracking
                with self._lock:
                    self._or_state[sym] = {
                        'high': None,
                        'low': None,
                        'finalized': False,
                    }
                    self._vwap_state[sym] = {
                        'cum_pv': 0.0,
                        'cum_vol': 0.0,
                    }

                logger.info(
                    f"[ORB] {sym}: prev_close={prev_close:.1f} open={today_open} "
                    f"gap={gap_pct:.2f}% CPR_w={cpr_width_pct:.3f}% "
                    f"{'WIDE' if is_wide else 'narrow'} qty={self.stocks[sym].get('qty', 0)}"
                )

            except Exception as e:
                logger.error(f"[ORB] Init failed for {sym}: {e}", exc_info=True)

        # --- Startup notification ---
        if self.cfg.get('notify_on_system', True):
            try:
                ns = get_notification_service(self.cfg)
                wide_cpr = [s for s in self.stocks
                            if self.db.get_or_create_daily_state(s, today_str).get('is_wide_cpr_day')]
                ns.send_alert('login_success', 'ORB Day Initialized',
                    f'{len(self.stocks)} stocks ready | Wide CPR (skipped): {len(wide_cpr)}')
            except Exception as e:
                logger.warning(f"[ORB] Startup notification failed: {e}")

        logger.info("[ORB] Day initialization complete")

    # ===================================================================
    # Opening Range tracking (09:15 - 09:30)
    # ===================================================================

    def update_or(self):
        """
        Called every minute (or on 5-min candle close) during 09:15-09:30.
        For each stock: fetch latest 5-min candles, update OR high/low.
        At 09:30 (after or_minutes elapsed): mark OR finalized.
        """
        now = datetime.now()
        today_str = date.today().isoformat()
        or_minutes = self.cfg.get('or_minutes', 15)
        session_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        or_end = session_start + timedelta(minutes=or_minutes)

        for sym in self.stocks:
            try:
                with self._lock:
                    or_s = self._or_state.get(sym)
                    if not or_s or or_s.get('finalized'):
                        continue

                # Fetch 5-min candles for today's session so far
                candles = self.fetch_5min_candles(sym, session_start, now)
                if not candles:
                    continue

                # Filter only OR-window candles (09:15 to OR end)
                or_candles = []
                for c in candles:
                    c_time = c['date']
                    if hasattr(c_time, 'hour'):
                        c_dt = c_time
                    else:
                        c_dt = pd.Timestamp(c_time)
                    # Strip timezone for comparison (Kite returns tz-aware)
                    if hasattr(c_dt, 'tzinfo') and c_dt.tzinfo is not None:
                        c_dt = c_dt.replace(tzinfo=None)
                    # Include candles that opened within the OR window
                    if c_dt < or_end:
                        or_candles.append(c)

                if not or_candles:
                    continue

                or_high = max(c['high'] for c in or_candles)
                or_low = min(c['low'] for c in or_candles)
                or_range = or_high - or_low

                with self._lock:
                    self._or_state[sym]['high'] = or_high
                    self._or_state[sym]['low'] = or_low

                # Check if OR window is complete
                finalized = now >= or_end
                if finalized:
                    with self._lock:
                        self._or_state[sym]['finalized'] = True

                    self.db.update_daily_state(sym, today_str,
                        or_high=round(or_high, 2),
                        or_low=round(or_low, 2),
                        or_range=round(or_range, 2),
                        or_finalized=1,
                    )
                    logger.info(
                        f"[ORB] {sym} OR finalized: H={or_high:.2f} L={or_low:.2f} "
                        f"R={or_range:.2f}"
                    )
                else:
                    # Partial update
                    self.db.update_daily_state(sym, today_str,
                        or_high=round(or_high, 2),
                        or_low=round(or_low, 2),
                        or_range=round(or_range, 2),
                    )

            except Exception as e:
                logger.error(f"[ORB] OR update failed for {sym}: {e}", exc_info=True)

    # ===================================================================
    # Signal evaluation (after OR finalized, every 5-min candle close)
    # ===================================================================

    def evaluate_signals(self):
        """
        Called at each 5-min candle close after OR is finalized.
        For each stock (no open position, trades_taken < max):
        1. Fetch latest 5-min candle
        2. Check breakout: close > OR_high (long) or close < OR_low (short)
        3. Compute VWAP from session candles
        4. Get 15-min RSI
        5. Check all filters
        6. If valid signal: place MIS market order
        7. Record position in DB
        """
        now = datetime.now()
        today_str = date.today().isoformat()
        cfg = self.cfg

        # Time window check
        last_entry = cfg.get('last_entry_time', '14:00')
        current_time = now.strftime('%H:%M')
        if current_time > last_entry:
            return

        session_start = now.replace(hour=9, minute=15, second=0, microsecond=0)

        for sym in self.stocks:
            try:
                # Skip if OR not finalized
                with self._lock:
                    or_s = self._or_state.get(sym)
                if not or_s or not or_s.get('finalized'):
                    continue

                or_high = or_s['high']
                or_low = or_s['low']
                if or_high is None or or_low is None:
                    continue

                # Skip if already in a position
                with self._lock:
                    if sym in self._positions:
                        continue

                # Check trades_taken for today
                daily_state = self.db.get_or_create_daily_state(sym, today_str)
                max_trades = cfg.get('max_trades_per_day', 1)
                if (daily_state.get('trades_taken') or 0) >= max_trades:
                    continue

                # Skip wide CPR days entirely
                if daily_state.get('is_wide_cpr_day'):
                    continue

                # Fetch today's 5-min candles
                candles = self.fetch_5min_candles(sym, session_start, now)
                if not candles or len(candles) < 4:
                    continue

                latest = candles[-1]
                prev_candle = candles[-2]
                close_price = latest['close']
                prev_close = prev_candle['close']

                # --- Breakout detection (close-based, with confirmation) ---
                direction = None
                if close_price > or_high and prev_close <= or_high:
                    direction = 'LONG'
                elif close_price < or_low and prev_close >= or_low:
                    direction = 'SHORT'

                if direction is None:
                    continue

                # --- Compute indicators ---
                vwap = self.compute_vwap(candles)
                rsi = self.compute_rsi_15m(sym)

                # Update daily state with latest indicators
                self.db.update_daily_state(sym, today_str,
                    vwap=vwap,
                    rsi_15m=rsi,
                )

                # --- Filter checks ---
                filters = {}
                all_passed = True

                # Direction filter (long/short allowed?)
                if direction == 'LONG' and not cfg.get('allow_longs', True):
                    filters['allow_longs'] = False
                    all_passed = False
                elif direction == 'SHORT' and not cfg.get('allow_shorts', True):
                    filters['allow_shorts'] = False
                    all_passed = False
                else:
                    filters['direction_allowed'] = True

                # VWAP filter
                if cfg.get('use_vwap_filter', True) and vwap is not None:
                    if direction == 'LONG' and close_price <= vwap:
                        filters['vwap'] = False
                        all_passed = False
                    elif direction == 'SHORT' and close_price >= vwap:
                        filters['vwap'] = False
                        all_passed = False
                    else:
                        filters['vwap'] = True
                else:
                    filters['vwap'] = 'disabled'

                # RSI filter (15-min)
                if cfg.get('use_rsi_filter', True) and rsi is not None:
                    rsi_long = cfg.get('rsi_long_threshold', 60)
                    rsi_short = cfg.get('rsi_short_threshold', 40)
                    if direction == 'LONG' and rsi < rsi_long:
                        filters['rsi'] = False
                        all_passed = False
                    elif direction == 'SHORT' and rsi > rsi_short:
                        filters['rsi'] = False
                        all_passed = False
                    else:
                        filters['rsi'] = True
                else:
                    filters['rsi'] = 'disabled'

                # CPR direction filter
                cpr_tc = daily_state.get('cpr_tc')
                cpr_bc = daily_state.get('cpr_bc')
                if cfg.get('use_cpr_dir_filter', True) and cpr_tc and cpr_bc:
                    if direction == 'LONG' and close_price <= cpr_tc:
                        filters['cpr_dir'] = False
                        all_passed = False
                    elif direction == 'SHORT' and close_price >= cpr_bc:
                        filters['cpr_dir'] = False
                        all_passed = False
                    else:
                        filters['cpr_dir'] = True
                else:
                    filters['cpr_dir'] = 'disabled'

                # Gap direction filter
                gap_pct = daily_state.get('gap_pct')
                gap_block_pct = cfg.get('gap_long_block_pct', 0.3)
                if cfg.get('use_gap_filter', True) and gap_pct is not None:
                    if direction == 'LONG' and gap_pct > gap_block_pct:
                        filters['gap'] = False
                        all_passed = False
                    else:
                        filters['gap'] = True
                else:
                    filters['gap'] = 'disabled'

                # --- Compute SL and target ---
                entry_price = close_price
                if direction == 'LONG':
                    sl_price = or_low
                    risk = entry_price - sl_price
                else:
                    sl_price = or_high
                    risk = sl_price - entry_price

                if risk <= 0:
                    logger.warning(f"[ORB] {sym} {direction}: zero/negative risk, skipping")
                    continue

                r_multiple = cfg.get('r_multiple', 1.5)
                if direction == 'LONG':
                    target_price = entry_price + (r_multiple * risk)
                else:
                    target_price = entry_price - (r_multiple * risk)

                # --- Log signal ---
                action_taken = 'ENTERED' if all_passed else self._get_block_reason(filters)
                self.db.log_signal(
                    instrument=sym,
                    signal_time=now.isoformat(),
                    direction=direction,
                    entry_price=round(entry_price, 2),
                    sl_price=round(sl_price, 2),
                    target_price=round(target_price, 2),
                    or_high=round(or_high, 2),
                    or_low=round(or_low, 2),
                    gap_pct=gap_pct,
                    vwap=vwap,
                    rsi_15m=rsi,
                    cpr_pivot=daily_state.get('cpr_pivot'),
                    cpr_tc=cpr_tc,
                    cpr_bc=cpr_bc,
                    cpr_width_pct=daily_state.get('cpr_width_pct'),
                    filters_passed=json.dumps(filters),
                    action_taken=action_taken,
                )

                if not all_passed:
                    logger.info(f"[ORB] {sym} {direction} signal BLOCKED: {action_taken}")
                    continue

                # --- Check available funds before placing ---
                qty = self.stocks[sym].get('qty', 1)
                if qty <= 0:
                    logger.warning(f"[ORB] {sym}: qty=0, skipping entry")
                    continue

                alloc = self.allocation_per_trade
                min_balance_required = self.min_margin_for_trade
                required_capital = qty * entry_price
                available = self._get_available_margin()

                if available is not None and available < min_balance_required:
                    logger.warning(
                        f"[ORB] {sym}: LOW FUNDS — available Rs {available:.0f} < "
                        f"required Rs {min_balance_required:.0f} (1.2x of {alloc}). SKIPPING TRADE."
                    )
                    self.db.log_signal(
                        instrument=sym, signal_time=now.isoformat(),
                        direction=direction, entry_price=entry_price,
                        sl_price=sl_price, target_price=target_price,
                        or_high=or_high, or_low=or_low,
                        action_taken=f'BLOCKED_LOW_FUNDS (available Rs {available:.0f} < 1.2x Rs {min_balance_required:.0f})',
                    )

                    # --- Low funds notification ---
                    if self.cfg.get('notify_on_risk', True):
                        try:
                            ns = get_notification_service(self.cfg)
                            ns.send_alert('margin_alert', f'Low funds — {sym} trade skipped',
                                f'Available: Rs {available:.0f} | Required: Rs {min_balance_required:.0f}',
                                data={'symbol': sym, 'available_margin': available, 'required_margin': min_balance_required},
                                priority='high')
                        except Exception as e:
                            logger.warning(f"[ORB] Low funds notification failed: {e}")

                    continue

                if available is not None and required_capital > available:
                    old_qty = qty
                    qty = int(available // entry_price)
                    if qty <= 0:
                        logger.warning(f"[ORB] {sym}: insufficient funds for even 1 share. Skipping.")
                        self.db.log_signal(
                            instrument=sym, signal_time=now.isoformat(),
                            direction=direction, entry_price=entry_price,
                            sl_price=sl_price, target_price=target_price,
                            or_high=or_high, or_low=or_low,
                            action_taken=f'BLOCKED_NO_FUNDS (need {required_capital:.0f}, have {available:.0f})',
                        )
                        continue
                    logger.info(f"[ORB] {sym}: reduced qty {old_qty}->{qty} (available Rs {available:.0f})")
                    self.stocks[sym]['qty'] = qty

                kite_order_id = self.place_entry_order(sym, direction, qty, entry_price)
                if kite_order_id is None:
                    continue

                # --- Record position ---
                position_id = self.db.add_position(
                    instrument=sym,
                    trade_date=today_str,
                    direction=direction,
                    qty=qty,
                    entry_price=round(entry_price, 2),
                    entry_time=now.isoformat(),
                    sl_price=round(sl_price, 2),
                    target_price=round(target_price, 2),
                    or_high=round(or_high, 2),
                    or_low=round(or_low, 2),
                    kite_entry_order_id=kite_order_id,
                    status='OPEN',
                    gap_pct=gap_pct,
                    vwap_at_entry=vwap,
                    rsi_at_entry=rsi,
                    cpr_tc=cpr_tc,
                    cpr_bc=cpr_bc,
                    cpr_width_pct=daily_state.get('cpr_width_pct'),
                )

                # Update trades_taken
                self.db.update_daily_state(sym, today_str,
                    trades_taken=(daily_state.get('trades_taken') or 0) + 1,
                )

                # Cache in memory
                pos = self.db.get_open_positions(instrument=sym)
                with self._lock:
                    if pos:
                        self._positions[sym] = pos[0]

                logger.info(
                    f"[ORB] ENTRY {sym} {direction} qty={qty} @{entry_price:.2f} "
                    f"SL={sl_price:.2f} TGT={target_price:.2f} "
                    f"order={kite_order_id} pos_id={position_id}"
                )

                # --- Entry notification ---
                if self.cfg.get('notify_on_entry', True):
                    try:
                        ns = get_notification_service(self.cfg)
                        ns.send_alert('trade_entry', f'{direction} {sym} @ {entry_price:.2f}',
                            f'Qty: {qty} | SL: {sl_price:.2f} | TGT: {target_price:.2f} | OR: {or_low:.2f}-{or_high:.2f}',
                            data={'symbol': sym, 'direction': direction, 'entry_price': entry_price,
                                  'sl_price': sl_price, 'target_price': target_price, 'qty': qty})
                    except Exception as e:
                        logger.warning(f"[ORB] Entry notification failed: {e}")

            except Exception as e:
                logger.error(f"[ORB] Signal eval failed for {sym}: {e}", exc_info=True)

    def _get_block_reason(self, filters):
        """Build a human-readable block reason from filter dict."""
        failed = [k for k, v in filters.items() if v is False]
        if not failed:
            return 'UNKNOWN'
        return 'BLOCKED_' + '_'.join(failed).upper()

    # ===================================================================
    # Position monitoring (every 30 seconds)
    # ===================================================================

    def monitor_positions(self):
        """
        Called every 30 seconds.
        For each open position:
        1. Get current LTP
        2. Check SL hit
        3. Check target hit
        4. If hit: place exit order, close position in DB
        """
        with self._lock:
            open_syms = list(self._positions.keys())

        if not open_syms:
            return

        # Batch LTP fetch
        ltps = self.get_live_ltp(open_syms)
        now = datetime.now()

        for sym in open_syms:
            try:
                ltp = ltps.get(sym)
                if ltp is None:
                    continue

                with self._lock:
                    pos = self._positions.get(sym)
                if not pos:
                    continue

                direction = pos['direction']
                sl_price = pos['sl_price']
                target_price = pos['target_price']
                entry_price = pos['entry_price']
                qty = pos['qty']

                exit_reason = None

                # Check SL (conservative: SL checked first)
                if direction == 'LONG' and ltp <= sl_price:
                    exit_reason = 'SL_HIT'
                elif direction == 'SHORT' and ltp >= sl_price:
                    exit_reason = 'SL_HIT'
                # Check target
                elif direction == 'LONG' and ltp >= target_price:
                    exit_reason = 'TARGET_HIT'
                elif direction == 'SHORT' and ltp <= target_price:
                    exit_reason = 'TARGET_HIT'

                if exit_reason:
                    self.place_exit_order(pos, ltp, exit_reason)

            except Exception as e:
                logger.error(f"[ORB] Monitor failed for {sym}: {e}", exc_info=True)

    # ===================================================================
    # EOD squareoff (call at 15:20)
    # ===================================================================

    def eod_squareoff(self):
        """Close ALL open positions at market. Called at 15:20 sharp."""
        logger.info("[ORB] === EOD Squareoff ===")
        open_positions = self.db.get_open_positions()
        if not open_positions:
            logger.info("[ORB] No open positions to squareoff")
            return []

        # Batch LTP
        syms = list(set(p['instrument'] for p in open_positions))
        ltps = self.get_live_ltp(syms)

        results = []
        for pos in open_positions:
            try:
                ltp = ltps.get(pos['instrument'])
                if ltp is None:
                    # Last resort: use entry price (should not happen)
                    ltp = pos['entry_price']
                    logger.warning(f"[ORB] No LTP for {pos['instrument']} at EOD, using entry price")

                self.place_exit_order(pos, ltp, 'EOD_SQUAREOFF')
                results.append({
                    'instrument': pos['instrument'],
                    'direction': pos['direction'],
                    'exit_price': ltp,
                    'exit_reason': 'EOD_SQUAREOFF',
                })
            except Exception as e:
                logger.error(f"[ORB] EOD exit failed for {pos['instrument']}: {e}", exc_info=True)

        logger.info(f"[ORB] EOD squareoff complete: {len(results)} positions closed")
        return results

    # ===================================================================
    # Order placement
    # ===================================================================

    def place_entry_order(self, instrument, direction, qty, price):
        """
        Place MIS market order on Kite for entry.
        BUY for LONG, SELL for SHORT.
        Returns kite_order_id string, or None on failure.
        """
        transaction_type = 'BUY' if direction == 'LONG' else 'SELL'

        try:
            kite = self._get_kite()
            order_id = kite.place_order(
                variety='regular',
                exchange='NSE',
                tradingsymbol=instrument,
                transaction_type=transaction_type,
                quantity=qty,
                product='MIS',
                order_type='MARKET',
            )
            order_id_str = str(order_id)

            # Log order
            self.db.log_order(
                position_id=None,  # will be updated after position is created
                instrument=instrument,
                tradingsymbol=instrument,
                transaction_type=transaction_type,
                qty=qty,
                order_type='MARKET',
                price=price,
                kite_order_id=order_id_str,
                status='PLACED',
                exchange='NSE',
                product='MIS',
                notes=f'ENTRY {direction}',
            )

            # Verify order status
            self._verify_order(kite, order_id_str, instrument, 'entry')

            logger.info(f"[ORB] Entry order placed: {transaction_type} {qty} {instrument} "
                        f"@ MARKET, order_id={order_id_str}")
            return order_id_str

        except Exception as e:
            logger.error(f"[ORB] Entry order FAILED for {instrument}: {e}", exc_info=True)
            self.db.log_order(
                instrument=instrument,
                tradingsymbol=instrument,
                transaction_type=transaction_type,
                qty=qty,
                order_type='MARKET',
                price=price,
                kite_order_id=None,
                status='REJECTED',
                exchange='NSE',
                product='MIS',
                notes=f'ENTRY {direction} FAILED: {str(e)[:200]}',
            )
            return None

    def place_exit_order(self, position, exit_price, exit_reason):
        """
        Close position: SELL for LONG, BUY for SHORT. MIS market order.
        Updates DB position and clears in-memory cache.
        """
        direction = position['direction']
        transaction_type = 'SELL' if direction == 'LONG' else 'BUY'
        instrument = position['instrument']
        qty = position['qty']
        entry_price = position['entry_price']
        now = datetime.now()

        # Compute P&L
        if direction == 'LONG':
            pnl_pts = exit_price - entry_price
        else:
            pnl_pts = entry_price - exit_price
        pnl_inr = round(pnl_pts * qty, 2)
        pnl_pts = round(pnl_pts, 2)

        kite_exit_order_id = None
        try:
            kite = self._get_kite()
            order_id = kite.place_order(
                variety='regular',
                exchange='NSE',
                tradingsymbol=instrument,
                transaction_type=transaction_type,
                quantity=qty,
                product='MIS',
                order_type='MARKET',
            )
            kite_exit_order_id = str(order_id)

            self._verify_order(kite, kite_exit_order_id, instrument, 'exit')

            logger.info(
                f"[ORB] Exit order placed: {transaction_type} {qty} {instrument} "
                f"@ MARKET ({exit_reason}), order_id={kite_exit_order_id}"
            )
        except Exception as e:
            logger.error(
                f"[ORB] Exit order FAILED for {instrument}: {e}. "
                f"Position may remain open on Kite!",
                exc_info=True
            )

        # Log order
        self.db.log_order(
            position_id=position['id'],
            instrument=instrument,
            tradingsymbol=instrument,
            transaction_type=transaction_type,
            qty=qty,
            order_type='MARKET',
            price=round(exit_price, 2),
            kite_order_id=kite_exit_order_id,
            status='PLACED' if kite_exit_order_id else 'FAILED',
            exchange='NSE',
            product='MIS',
            notes=f'EXIT {exit_reason}',
        )

        # Close position in DB
        self.db.close_position(
            position_id=position['id'],
            exit_price=round(exit_price, 2),
            exit_time=now.isoformat(),
            exit_reason=exit_reason,
            pnl_pts=pnl_pts,
            pnl_inr=pnl_inr,
            kite_exit_order_id=kite_exit_order_id,
        )

        # Clear in-memory cache
        with self._lock:
            self._positions.pop(instrument, None)

        pnl_label = f"+Rs {pnl_inr:.0f}" if pnl_inr >= 0 else f"-Rs {abs(pnl_inr):.0f}"
        logger.info(
            f"[ORB] CLOSED {instrument} {direction} @{exit_price:.2f} "
            f"({exit_reason}) P&L={pnl_label}"
        )

        # --- Exit notification ---
        if self.cfg.get('notify_on_exit', True):
            try:
                ns = get_notification_service(self.cfg)
                # Map exit_reason to alert type
                reason_upper = (exit_reason or '').upper()
                if 'SL' in reason_upper:
                    alert_type = 'sl_hit'
                elif 'TARGET' in reason_upper:
                    alert_type = 'target_hit'
                elif 'EOD' in reason_upper or 'SQUAREOFF' in reason_upper:
                    alert_type = 'eod_close'
                else:
                    alert_type = 'trade_exit'
                ns.send_alert(alert_type, f'{exit_reason} — {instrument} {direction}',
                    f'Exit @ {exit_price:.2f} | Entry @ {entry_price:.2f} | P&L: {pnl_label}',
                    data={'symbol': instrument, 'direction': direction,
                          'entry_price': entry_price, 'exit_price': exit_price,
                          'exit_reason': exit_reason, 'pnl_pts': pnl_pts,
                          'pnl_inr': pnl_inr, 'qty': qty},
                    priority='high' if 'SL' in reason_upper else 'normal')
            except Exception as e:
                logger.warning(f"[ORB] Exit notification failed: {e}")

    def _verify_order(self, kite, order_id, instrument, order_type_label):
        """
        Check order status after placement. Log warning if rejected/cancelled.
        Non-blocking -- does not retry.
        """
        try:
            time.sleep(0.5)  # brief wait for order to process
            history = kite.order_history(order_id)
            if history:
                final = history[-1]
                status = final.get('status', '')
                if status == 'REJECTED':
                    reason = final.get('status_message', 'unknown')
                    logger.error(
                        f"[ORB] Order REJECTED: {instrument} {order_type_label} "
                        f"order_id={order_id} reason={reason}"
                    )
                elif status == 'CANCELLED':
                    logger.warning(
                        f"[ORB] Order CANCELLED: {instrument} {order_type_label} "
                        f"order_id={order_id}"
                    )
                elif status == 'COMPLETE':
                    avg_price = final.get('average_price', 0)
                    logger.info(
                        f"[ORB] Order FILLED: {instrument} {order_type_label} "
                        f"order_id={order_id} avg_price={avg_price}"
                    )
        except Exception as e:
            logger.warning(f"[ORB] Order verify failed for {order_id}: {e}")

    # ===================================================================
    # Dashboard state
    # ===================================================================

    def get_full_state(self):
        """
        Dashboard API: return full state for all stocks.
        Same pattern as NasExecutor.get_full_state().
        """
        today_str = date.today().isoformat()
        stats = self.db.get_stats()
        open_positions = self.db.get_open_positions()
        closed_today = self.db.get_today_closed(instrument=None)
        recent_signals = self.db.get_recent_signals(limit=30)
        recent_trades = self.db.get_recent_trades(limit=50)
        equity_curve = self.db.get_equity_curve()

        # Per-stock state
        stocks_state = {}
        for sym in self.stocks:
            daily = self.db.get_or_create_daily_state(sym, today_str)
            sym_open = [p for p in open_positions if p['instrument'] == sym]
            sym_closed = [p for p in closed_today if p['instrument'] == sym]

            with self._lock:
                or_s = self._or_state.get(sym, {})

            qty = self.stocks[sym].get('qty', 0)
            today_open = daily.get('today_open') or 0
            capital_per_trade = round(qty * today_open, 0) if today_open > 0 else 0
            sl_risk = 0
            if daily.get('or_high') and daily.get('or_low') and today_open > 0:
                or_range = (daily['or_high'] or 0) - (daily['or_low'] or 0)
                sl_risk = round(or_range * qty, 0)

            stocks_state[sym] = {
                'daily_state': daily,
                'or_finalized': or_s.get('finalized', daily.get('or_finalized', 0)),
                'or_high': or_s.get('high', daily.get('or_high')),
                'or_low': or_s.get('low', daily.get('or_low')),
                'position': sym_open[0] if sym_open else None,
                'closed_today': sym_closed,
                'token': self.stocks[sym].get('token'),
                'qty': qty,
                'capital_per_trade': capital_per_trade,
                'sl_risk_inr': sl_risk,
                'price': today_open,
            }

        return {
            'stocks': stocks_state,
            'stats': stats,
            'config': {
                'enabled': self.cfg.get('enabled', True),
                'capital': self.cfg.get('capital', 100000),
                'allocation_per_trade': round(self.allocation_per_trade),
                'max_concurrent_trades': self.cfg.get('max_concurrent_trades', 3),
                'min_margin_for_trade': round(self.min_margin_for_trade),
                'margin_buffer': self.cfg.get('margin_buffer_multiplier', 1.2),
                'or_minutes': self.cfg.get('or_minutes', 15),
                'last_entry_time': self.cfg.get('last_entry_time', '14:00'),
                'eod_exit_time': self.cfg.get('eod_exit_time', '15:20'),
                'max_trades_per_day': self.cfg.get('max_trades_per_day', 1),
                'r_multiple': self.cfg.get('r_multiple', 1.5),
                'use_vwap_filter': self.cfg.get('use_vwap_filter', True),
                'use_rsi_filter': self.cfg.get('use_rsi_filter', True),
                'use_cpr_dir_filter': self.cfg.get('use_cpr_dir_filter', True),
                'use_cpr_width_filter': self.cfg.get('use_cpr_width_filter', True),
                'use_gap_filter': self.cfg.get('use_gap_filter', True),
                'allow_longs': self.cfg.get('allow_longs', True),
                'allow_shorts': self.cfg.get('allow_shorts', True),
            },
            'positions': {
                'open': open_positions,
                'closed_today': closed_today,
                'total_open': len(open_positions),
                'total_closed_today': len(closed_today),
            },
            'recent_signals': recent_signals,
            'recent_trades': recent_trades,
            'equity_curve': equity_curve,
            'margin': self._last_margin,
            'fund_alert': self._check_fund_alert(),
        }

    # ===================================================================
    # EOD report
    # ===================================================================

    def generate_eod_report(self):
        """Generate and send EOD summary report."""
        ns = get_notification_service(self.cfg)
        today_str = date.today().isoformat()

        closed = self.db.get_today_closed()
        stats = self.db.get_stats()

        # Collect blocked signals
        signals = self.db.get_recent_signals(limit=100)
        today_signals = [s for s in signals if (s.get('signal_time') or '')[:10] == today_str]
        blocked = [s for s in today_signals if s.get('action_taken', '').startswith('BLOCKED')]

        # Margin info
        margin = self._last_margin or {}

        report_data = {
            'trades': closed,
            'cumulative_pnl': stats.get('total_pnl', 0),
            'blocked_signals': [{'instrument': s.get('instrument'), 'reason': s.get('action_taken')} for s in blocked],
            'errors': [],
            'margin': margin,
            'capital': self.cfg.get('capital', 100000),
        }

        ns.send_eod_report(report_data)

    # ===================================================================
    # Emergency
    # ===================================================================

    def emergency_exit_all(self):
        """Kill switch -- close all open positions immediately at market."""
        logger.warning("[ORB] EMERGENCY EXIT triggered")
        return self.eod_squareoff()
