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
        """Derived notional allocation per trade.
        For MIS intraday Zerodha gives ~5x leverage on Nifty 500 cash.
        per-trade notional = capital × mis_leverage / max_concurrent_trades
        (mis_leverage=1 disables leverage — pure deposit-based sizing)."""
        capital = self.cfg.get('capital', 100000)
        lev = self.cfg.get('mis_leverage', 1)
        max_trades = self.cfg.get('max_concurrent_trades', 3)
        return (capital * lev) / max_trades

    @property
    def daily_loss_limit_inr(self) -> float:
        """Resolved daily loss cap in Rs.
        Falls back to pct x capital if explicit Rs override is not set."""
        override = self.cfg.get('daily_loss_limit')
        if override is not None and override != 0:
            return float(override)
        pct = self.cfg.get('daily_loss_limit_pct', 0.03)
        capital = self.cfg.get('capital', 100000)
        return round(float(capital) * float(pct), 2)

    def compute_day_pnl(self, include_open: bool = True) -> tuple[float, float]:
        """Return (realized, unrealized) P&L for today.
        Realized = closed trades today · Unrealized = open positions MTM."""
        realized = 0.0
        for t in (self.db.get_today_closed() or []):
            realized += float(t.get('pnl_inr') or 0)
        unrealized = 0.0
        if include_open:
            open_pos = self.db.get_open_positions() or []
            if open_pos:
                try:
                    ltps = self.get_live_ltp([p['instrument'] for p in open_pos])
                except Exception:
                    ltps = {}
                for p in open_pos:
                    ltp = ltps.get(p['instrument'])
                    if ltp is None:
                        continue
                    entry = p['entry_price']
                    qty = p['qty']
                    pts = (ltp - entry) if p['direction'] == 'LONG' else (entry - ltp)
                    unrealized += pts * qty
        return round(realized, 2), round(unrealized, 2)

    def daily_loss_gate(self) -> dict:
        """Evaluate the two-tier loss gate.
        - block_new_entries: realized loss ≥ daily_loss_limit
        - force_close_all:  realized + unrealized ≥ daily_loss_limit × panic_multiplier

        Returns dict with booleans + numbers. Safe to call frequently.
        If enforce_daily_loss_cap=False, booleans always False (still computes
        numbers so the UI can show the running loss total)."""
        realized, unrealized = self.compute_day_pnl(include_open=True)
        cap = self.daily_loss_limit_inr
        panic_mult = float(self.cfg.get('daily_loss_panic_multiplier', 1.5))
        enforce = bool(self.cfg.get('enforce_daily_loss_cap', True))
        realized_loss = max(0.0, -realized)
        total_loss = max(0.0, -(realized + unrealized))
        return {
            'realized': realized,
            'unrealized': unrealized,
            'realized_loss': realized_loss,
            'total_loss': total_loss,
            'cap': cap,
            'panic_cap': round(cap * panic_mult, 2),
            'enforce': enforce,
            'block_new_entries': enforce and realized_loss >= cap,
            'force_close_all': enforce and total_loss >= cap * panic_mult,
        }

    def compute_risk_based_qty(self, entry_price: float, sl_price: float) -> dict:
        """Risk-first sizing: risk exactly risk_per_trade_pct × capital on this
        trade (down to SL). Qty is trimmed if notional would blow past
        max_notional_per_trade. Falls back to the legacy notional-based rule
        if use_risk_based_sizing is False.

        Returns dict with qty + diagnostics so the caller can log/notify."""
        if entry_price is None or entry_price <= 0:
            return {'qty': 0, 'reason': 'invalid_entry_price'}
        R = abs(float(entry_price) - float(sl_price))
        if R <= 0:
            return {'qty': 0, 'reason': 'zero_risk_per_share'}

        if not self.cfg.get('use_risk_based_sizing', True):
            # Legacy: fixed-notional allocation
            qty = int(self.allocation_per_trade // entry_price)
            return {
                'qty': max(qty, 0),
                'mode': 'notional',
                'R_per_share': R,
                'risk_rs_target': round(qty * R, 2),
                'notional': round(qty * entry_price, 2),
            }

        capital = self.cfg.get('capital', 100000)
        risk_pct = float(self.cfg.get('risk_per_trade_pct', 0.008))
        risk_rs_target = capital * risk_pct
        raw_qty = int(risk_rs_target // R)

        max_notional = self.cfg.get('max_notional_per_trade', capital)
        cap_qty = int(max_notional // entry_price)
        capped = raw_qty > cap_qty
        final_qty = min(raw_qty, cap_qty)

        return {
            'qty': max(final_qty, 0),
            'mode': 'risk',
            'R_per_share': round(R, 4),
            'risk_rs_target': round(risk_rs_target, 2),
            'risk_rs_actual': round(final_qty * R, 2),
            'notional': round(final_qty * entry_price, 2),
            'notional_cap_hit': capped,
            'raw_qty_uncapped': raw_qty,
        }

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

                # 4. Today's open — use first 5-min candle open (9:15), NOT current LTP
                existing_state = self.db.get_or_create_daily_state(sym, today_str)
                today_open = existing_state.get('today_open')
                if not today_open:
                    # Fetch first 5-min candle for today
                    session_start = datetime.now().replace(hour=9, minute=15, second=0, microsecond=0)
                    first_candles = self.fetch_5min_candles(sym, session_start, session_start + timedelta(minutes=5))
                    if first_candles:
                        today_open = first_candles[0]['open']
                    else:
                        # Fallback to LTP if market hasn't opened yet (pre-9:15 init)
                        ltps = self.get_live_ltp([sym])
                        today_open = ltps.get(sym)

                # 5. Gap (from actual opening price, not current LTP)
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

        # Daily loss gate: once realized loss hits the cap, block new entries.
        gate = self.daily_loss_gate()
        if gate['block_new_entries']:
            if not getattr(self, '_loss_gate_notified', False):
                self._loss_gate_notified = True
                logger.warning(
                    f"[ORB] Daily loss cap breached (realized loss Rs {gate['realized_loss']:,.0f} "
                    f"≥ cap Rs {gate['cap']:,.0f}) — blocking new entries"
                )
                try:
                    ns = get_notification_service(self.cfg)
                    ns.send_alert(
                        'risk_alert',
                        f'Daily loss cap breached — new ORB entries blocked',
                        f"Realized loss Rs {gate['realized_loss']:,.0f} ≥ cap Rs {gate['cap']:,.0f} "
                        f"({self.cfg.get('daily_loss_limit_pct', 0.03)*100:.1f}% of capital). "
                        f"Existing open positions continue with their SLs.",
                        data=gate, priority='critical',
                    )
                except Exception:
                    pass
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
                close_price = latest['close']

                # --- Compute indicators (always, for dashboard display) ---
                vwap = self.compute_vwap(candles)
                rsi = self.compute_rsi_15m(sym)

                # Persist to DB so dashboard can show them
                self.db.update_daily_state(sym, today_str,
                    vwap=vwap,
                    rsi_15m=rsi,
                )

                # --- Breakout detection ---
                # Walk ALL candles in order, find every OR boundary crossing.
                # Start the walk from the first post-OR candle so a transition
                # that lands on the 9:30 candle itself (common — move begins
                # right at OR end) is compared against the last pre-OR candle.
                # We take the LATEST valid transition that is still consistent
                # with the current 5-min close (so a SHORT that reversed into
                # a LONG later this morning catches the LONG, not the stale
                # SHORT). Stops the "already-past-OR" blind spot that missed
                # TRENT SHORT on 2026-04-23 (close 4380.10 vs OR_low 4380.20).
                def _ctime(c):
                    t = c['date']
                    if hasattr(t, 'tzinfo') and t.tzinfo is not None:
                        t = t.replace(tzinfo=None)
                    return t
                or_end = session_start + timedelta(minutes=cfg.get('or_minutes', 15))
                or_end_idx = next(
                    (i for i, c in enumerate(candles) if _ctime(c) >= or_end),
                    len(candles)
                )
                direction = None
                for i in range(or_end_idx, len(candles)):
                    if i == 0:
                        continue
                    prev_c = candles[i - 1]
                    cur = candles[i]
                    if cur['close'] > or_high and prev_c['close'] <= or_high:
                        if close_price > or_high:
                            direction = 'LONG'
                    elif cur['close'] < or_low and prev_c['close'] >= or_low:
                        if close_price < or_low:
                            direction = 'SHORT'

                if direction is None:
                    continue

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

                # --- Risk-based position sizing (uses the OR-opposite SL) ---
                sizing = self.compute_risk_based_qty(entry_price, sl_price)
                qty = sizing['qty']
                if qty <= 0:
                    logger.warning(f"[ORB] {sym}: qty=0 ({sizing.get('reason', 'n/a')}), skipping entry")
                    continue
                self.stocks[sym]['qty'] = qty  # cache so dashboard reflects post-sizing qty
                logger.info(
                    f"[ORB] {sym} sizing: mode={sizing.get('mode')} "
                    f"R={sizing.get('R_per_share')} qty={qty} "
                    f"risk=Rs {sizing.get('risk_rs_actual', 0):.0f} "
                    f"notional=Rs {sizing.get('notional', 0):.0f}"
                    f"{' (capped at ' + str(self.cfg.get('max_notional_per_trade')) + ')' if sizing.get('notional_cap_hit') else ''}"
                )

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

                # Compute conviction (same rubric as dashboard candidates).
                import json as _json_conv
                past_pct = ((entry_price - or_high) / or_high * 100) if direction == 'LONG' \
                    else ((or_low - entry_price) / or_low * 100)
                or_width_pct = ((or_high - or_low) / entry_price * 100) if entry_price else None
                conv = self._compute_conviction(
                    side=direction, past_pct=past_pct,
                    cpr_pct=daily_state.get('cpr_width_pct'),
                    or_width_pct=or_width_pct, rsi=rsi,
                )

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
                    conviction_grade=conv['conviction_grade'],
                    conviction_score=conv['conviction_score'],
                    conviction_stars=_json_conv.dumps(conv['conviction_stars']),
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

                # Place on-exchange SL-M so the position is protected even if
                # the service crashes between monitor polls. Refresh the
                # in-memory cache afterwards so kite_sl_order_id is populated.
                if pos and self.cfg.get('use_exchange_sl_m', True):
                    self.place_sl_m_order(pos[0])
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
        1. Check if on-exchange SL order has completed (fast path — no soft re-exit)
        2. Get current LTP
        3. Check soft SL hit (fallback if exchange SL missing/failed)
        4. Check target hit
        5. If hit: place exit order, close position in DB
        """
        with self._lock:
            open_syms = list(self._positions.keys())

        if not open_syms:
            return

        # --- Step 0: panic tier — force-close all if MTM loss blew past 1.5x cap ---
        gate = self.daily_loss_gate()
        if gate['force_close_all'] and not getattr(self, '_loss_panic_fired', False):
            self._loss_panic_fired = True
            logger.critical(
                f"[ORB] PANIC: MTM loss Rs {gate['total_loss']:,.0f} ≥ panic cap "
                f"Rs {gate['panic_cap']:,.0f} — force-closing all open positions"
            )
            try:
                ns = get_notification_service(self.cfg)
                ns.send_alert(
                    'risk_alert',
                    f"PANIC: Force-closing ALL ORB positions",
                    f"MTM loss Rs {gate['total_loss']:,.0f} has breached "
                    f"{self.cfg.get('daily_loss_panic_multiplier', 1.5):.1f}× the daily cap "
                    f"(Rs {gate['panic_cap']:,.0f}). Closing {len(open_syms)} open positions now.",
                    data=gate, priority='critical',
                )
            except Exception:
                pass
            try:
                panic_ltps = self.get_live_ltp(open_syms)
            except Exception:
                panic_ltps = {}
            for sym_p in list(open_syms):
                with self._lock:
                    pos_p = self._positions.get(sym_p)
                if not pos_p:
                    continue
                px = panic_ltps.get(sym_p) or pos_p.get('entry_price')
                try:
                    self.place_exit_order(pos_p, px, 'DAILY_LOSS_PANIC')
                except Exception as e:
                    logger.error(f"[ORB] Panic exit failed for {sym_p}: {e}")
            return

        # --- Step 1a: general Kite-DB reconciliation -------------------
        # If the Kite net qty is already flat for any DB-open position
        # (manual close, SL fill, any external intervention), mark it
        # closed in DB so we never place a phantom exit at EOD/target.
        # Fetch positions once per cycle to keep API usage bounded.
        kite_positions = self.fetch_kite_mis_positions()
        if kite_positions:
            for sym in list(open_syms):
                try:
                    with self._lock:
                        pos = self._positions.get(sym)
                    if not pos:
                        continue
                    kite_state = kite_positions.get(sym, {
                        'qty': 0, 'buy_price': 0, 'sell_price': 0,
                        'buy_qty': 0, 'sell_qty': 0, 'source': 'kite',
                    })
                    if self.reconcile_position_with_kite(pos, kite_state=kite_state):
                        if sym in open_syms:
                            open_syms.remove(sym)
                except Exception as e:
                    logger.error(f"[ORB] Kite reconcile failed for {sym}: {e}")

        if not open_syms:
            return

        # --- Step 1b: exchange-SL order reconciliation -----------------
        # If the SL order specifically fired on Kite, close the DB position
        # with the precise SL fill price (more accurate than the generic
        # Step 1a path which uses avg buy/sell across all fills).
        for sym in list(open_syms):
            try:
                with self._lock:
                    pos = self._positions.get(sym)
                if not pos:
                    continue
                sl_order_id = pos.get('kite_sl_order_id')
                if not sl_order_id:
                    continue
                kite = self._get_kite()
                hist = kite.order_history(sl_order_id) or []
                if not hist:
                    continue
                last = hist[-1]
                if (last.get('status') or '').upper() != 'COMPLETE':
                    continue
                # Filled at exchange. Close the DB position with the actual fill price.
                fill_price = last.get('average_price') or pos['sl_price']
                direction = pos['direction']
                entry_price = pos['entry_price']
                qty = pos['qty']
                pnl_pts = fill_price - entry_price if direction == 'LONG' \
                    else entry_price - fill_price
                pnl_inr = round(pnl_pts * qty, 2)
                self.db.close_position(
                    position_id=pos['id'],
                    exit_price=round(float(fill_price), 2),
                    exit_time=datetime.now().isoformat(),
                    exit_reason='SL_HIT_EXCHANGE',
                    pnl_pts=round(pnl_pts, 2),
                    pnl_inr=pnl_inr,
                    kite_exit_order_id=sl_order_id,
                )
                with self._lock:
                    self._positions.pop(sym, None)
                open_syms.remove(sym)
                pnl_label = f"+Rs {pnl_inr:.0f}" if pnl_inr >= 0 else f"-Rs {abs(pnl_inr):.0f}"
                logger.info(
                    f"[ORB] SL-exchange CLOSED {sym} {direction} @{fill_price:.2f} "
                    f"P&L={pnl_label} (order={sl_order_id})"
                )
                # Exit notification
                try:
                    ns = get_notification_service(self.cfg)
                    ns.send_alert(
                        'sl_hit',
                        f'SL-EXCHANGE hit — {sym} {direction} @ {float(fill_price):.2f}',
                        f'P&L {pnl_label} · filled via Kite SL order {sl_order_id}',
                        data={'symbol': sym, 'direction': direction,
                              'entry_price': entry_price, 'exit_price': float(fill_price),
                              'qty': qty, 'pnl_inr': pnl_inr,
                              'exit_reason': 'SL_HIT_EXCHANGE'},
                    )
                except Exception as e2:
                    logger.warning(f"[ORB] SL-exchange notify failed: {e2}")
            except Exception as e:
                logger.error(f"[ORB] SL reconciliation failed for {sym}: {e}")

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
    # 14:30 trail activation — V9t_lock50 (lock half the profit)
    # ===================================================================

    def activate_trail_lock50(self):
        """At 14:30, for each open position lock 50% of the current profit
        by moving SL up (for longs) / down (for shorts) to entry + 0.5 * (LTP - entry).
        Only moves SL tighter, never looser. Loss-making positions keep their
        original OR-opposite SL. Final hard-EOD at 15:18 (separate cron).

        Backtest (250 days): V9t_lock50 -> Calmar 676 vs baseline V9 force-close 281
        (MaxDD 1.0% vs 2.1%, +70% P&L). See docs/ORB-VARIANTS-FINDINGS.md."""
        logger.info("[ORB] === 14:30 V9t_lock50: activating trail ===")
        open_positions = self.db.get_open_positions()
        if not open_positions:
            logger.info("[ORB] No open positions — trail activation no-op")
            return []

        syms = list({p['instrument'] for p in open_positions})
        ltps = self.get_live_ltp(syms)
        now = datetime.now()
        adjusted = []

        for pos in open_positions:
            try:
                sym = pos['instrument']
                direction = pos['direction']
                entry_price = pos['entry_price']
                current_sl = pos['sl_price']
                ltp = ltps.get(sym)
                if ltp is None:
                    logger.warning(f"[ORB] {sym} no LTP, skipping trail")
                    continue

                # V9t_lock50 STRICT (matches the 60-day backtest: Calmar 3,152 vs
                # 466 for lenient on 2026-04-23 sweep):
                #   profitable → SL = entry ± 0.5 × gain  (lock half profit)
                #   losing     → SL = entry              (breakeven — cuts afternoon
                #                                         drawdowns that the wide
                #                                         OR-opposite SL would miss)
                # The `tightened` check still prevents any SL loosening.
                if direction == 'LONG':
                    gain = ltp - entry_price
                    new_sl = (entry_price + 0.5 * gain) if gain > 0 else entry_price
                    tightened = new_sl > current_sl
                else:  # SHORT
                    gain = entry_price - ltp
                    new_sl = (entry_price - 0.5 * gain) if gain > 0 else entry_price
                    tightened = new_sl < current_sl

                if tightened:
                    # Kite rejects SL orders whose trigger is already on the
                    # wrong side of LTP (would fire immediately):
                    #   LONG  SL (SELL): trigger must be BELOW LTP
                    #   SHORT SL (BUY):  trigger must be ABOVE LTP
                    # For losing positions at 14:30, the strict rule sets
                    # new_sl = entry, and 'losing' means LTP has crossed
                    # entry in the unfavourable direction — so the new SL
                    # would be on the wrong side, Kite rejects, leaving
                    # the position unprotected. The backtest models this
                    # as 'SL fires at entry → instant breakeven stop'; the
                    # live equivalent is a MARKET EXIT now at LTP.
                    would_fire_immediately = (
                        (direction == 'LONG' and new_sl >= ltp) or
                        (direction == 'SHORT' and new_sl <= ltp)
                    )
                    if would_fire_immediately:
                        logger.info(
                            f"[ORB] {sym} {direction} trail: new_sl {new_sl:.2f} on wrong "
                            f"side of LTP {ltp:.2f} — market-exit now instead of SL-modify"
                        )
                        fresh_pos = self.db.get_open_positions(instrument=sym)
                        if fresh_pos:
                            self.place_exit_order(fresh_pos[0], ltp, 'V9T_LOCK50_BE')
                        adjusted.append({
                            'symbol': sym, 'direction': direction,
                            'old_sl': current_sl, 'new_sl': round(new_sl, 2),
                            'ltp': ltp, 'locked_pnl': 0,
                            'action': 'MARKET_EXIT_BE',
                        })
                        continue

                    self.db.update_position(pos['id'], sl_price=round(new_sl, 2))
                    with self._lock:
                        if sym in self._positions:
                            self._positions[sym]['sl_price'] = round(new_sl, 2)
                    # Sync the on-exchange SL-M trigger to the new (tighter) SL
                    if self.cfg.get('use_exchange_sl_m', True):
                        fresh = self.db.get_open_positions(instrument=sym)
                        if fresh:
                            self.modify_sl_m_order(fresh[0], new_sl)
                    adjusted.append({
                        'symbol': sym, 'direction': direction,
                        'old_sl': current_sl, 'new_sl': round(new_sl, 2),
                        'ltp': ltp, 'locked_pnl': round(0.5 * gain * pos['qty'], 2),
                    })
                    logger.info(
                        f"[ORB] {sym} {direction} trail: SL {current_sl:.2f} -> {new_sl:.2f} "
                        f"(locks 50% of +{gain:.2f} gain per share, "
                        f"~Rs {0.5 * gain * pos['qty']:.0f} total)"
                    )
                else:
                    logger.info(f"[ORB] {sym} {direction} not in profit (or new_sl doesn't tighten), keeping SL")
            except Exception as e:
                logger.error(f"[ORB] Trail activation failed for {pos.get('instrument')}: {e}", exc_info=True)

        logger.info(f"[ORB] Trail activation complete: {len(adjusted)} positions trailed")
        # Notification
        if adjusted and self.cfg.get('notify_on_system', True):
            try:
                ns = get_notification_service(self.cfg)
                lines = [f"{a['symbol']} {a['direction']}: SL {a['old_sl']} -> {a['new_sl']} (lock Rs {a['locked_pnl']:.0f})"
                         for a in adjusted]
                ns.send_alert(
                    'system_alert',
                    f'ORB 14:30 trail activated on {len(adjusted)} position(s)',
                    '\n'.join(lines),
                    data={'adjusted': adjusted},
                    priority='normal',
                )
            except Exception as e:
                logger.warning(f"[ORB] Trail notification failed: {e}")
        return adjusted

    # ===================================================================
    # Hard EOD squareoff (call at 15:18)
    # ===================================================================

    def eod_squareoff(self):
        """Close ALL open positions at market. Called at 15:18 sharp."""
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
                order_type='LIMIT',
                price=round(price * (1.002 if transaction_type == 'BUY' else 0.998), 1),  # 0.2% buffer for fill
            )
            order_id_str = str(order_id)

            # Log order
            self.db.log_order(
                position_id=None,  # will be updated after position is created
                instrument=instrument,
                tradingsymbol=instrument,
                transaction_type=transaction_type,
                qty=qty,
                order_type='LIMIT',
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
                order_type='LIMIT',
                price=price,
                kite_order_id=None,
                status='REJECTED',
                exchange='NSE',
                product='MIS',
                notes=f'ENTRY {direction} FAILED: {str(e)[:200]}',
            )
            return None

    # ===================================================================
    # Exchange-side SL-M orders (hard stop)
    # ===================================================================

    def place_sl_m_order(self, position):
        """Place an on-exchange SL-M order for an open position.
        Protects the position if the service crashes between monitor polls.

        For LONG entry: SELL SL-M with trigger_price = sl_price (fires when LTP drops).
        For SHORT entry: BUY SL-M with trigger_price = sl_price (fires when LTP rises).

        Returns kite_order_id string or None on failure. Stores the id on the
        position row via self.db.update_position(kite_sl_order_id=...).
        """
        direction = position['direction']
        instrument = position['instrument']
        qty = position['qty']
        sl_price = float(position['sl_price'])
        position_id = position['id']

        transaction_type = 'SELL' if direction == 'LONG' else 'BUY'
        # Round to this instrument's tick size (0.05 for most, 0.10 for some).
        tick = self._get_tick_size(instrument)
        def _tick(v):
            return round(round(v / tick) * tick, 2)
        trigger_price = _tick(sl_price)
        if transaction_type == 'SELL':  # LONG being stopped out -> sell
            limit_price = _tick(sl_price * 0.995)
        else:  # SHORT being stopped out -> buy to cover
            limit_price = _tick(sl_price * 1.005)

        try:
            kite = self._get_kite()
            order_id = kite.place_order(
                variety='regular',
                exchange='NSE',
                tradingsymbol=instrument,
                transaction_type=transaction_type,
                quantity=qty,
                product='MIS',
                order_type='SL',
                trigger_price=trigger_price,
                price=limit_price,
                validity='DAY',
            )
            order_id_str = str(order_id)
            self.db.update_position(position_id, kite_sl_order_id=order_id_str)
            self.db.log_order(
                position_id=position_id,
                instrument=instrument, tradingsymbol=instrument,
                transaction_type=transaction_type, qty=qty,
                order_type='SL', price=limit_price,
                kite_order_id=order_id_str, status='PLACED',
                exchange='NSE', product='MIS',
                notes=f'SL trigger={trigger_price} limit={limit_price}',
            )
            logger.info(
                f"[ORB] SL placed: {transaction_type} {qty} {instrument} "
                f"trigger={trigger_price} limit={limit_price} order_id={order_id_str}"
            )
            return order_id_str
        except Exception as e:
            logger.error(
                f"[ORB] SL-M FAILED for {instrument}: {e}. "
                f"Position relies on soft monitor only.",
                exc_info=True
            )
            try:
                ns = get_notification_service(self.cfg)
                ns.send_alert(
                    'risk_alert',
                    f'SL-M placement failed — {instrument} on soft SL only',
                    f'Could not place exchange SL-M for {direction} {instrument} '
                    f'(trigger {trigger_price}): {str(e)[:150]}. '
                    f'Position now depends on 30s LTP monitor.',
                    data={'symbol': instrument, 'direction': direction,
                          'trigger_price': trigger_price, 'qty': qty,
                          'error': str(e)[:200]},
                    priority='high',
                )
            except Exception:
                pass
            return None

    def cancel_sl_m_order(self, position):
        """Cancel the SL-M order for a position if it exists and is still open.
        Returns True if cancelled or already inactive, False on error.
        Safe to call even when no SL-M was placed."""
        sl_order_id = position.get('kite_sl_order_id')
        if not sl_order_id:
            return True
        try:
            kite = self._get_kite()
            # Check current status — skip cancel if already complete/cancelled/rejected
            try:
                history = kite.order_history(sl_order_id) or []
                last_status = (history[-1].get('status') if history else '').upper()
                if last_status in ('COMPLETE', 'CANCELLED', 'REJECTED'):
                    logger.info(f"[ORB] SL-M {sl_order_id} already {last_status} — skip cancel")
                    return True
            except Exception:
                pass
            kite.cancel_order(variety='regular', order_id=sl_order_id)
            logger.info(f"[ORB] SL-M cancelled: {sl_order_id} ({position['instrument']})")
            return True
        except Exception as e:
            logger.error(f"[ORB] SL-M cancel FAILED for {position['instrument']}: {e}")
            return False

    def _get_tick_size(self, instrument: str) -> float:
        """Fetch instrument tick size (cached). Defaults to 0.05 if unknown."""
        cache = getattr(self, '_tick_size_cache', None)
        if cache is None:
            cache = {}
            self._tick_size_cache = cache
        if instrument in cache:
            return cache[instrument]
        try:
            kite = self._get_kite()
            q = kite.quote([f'NSE:{instrument}']) or {}
            info = q.get(f'NSE:{instrument}', {})
            # Kite quote() doesn't always return tick_size — fall back to instruments()
            ts = info.get('tick_size')
            if not ts:
                insts = kite.instruments('NSE') or []
                for r in insts:
                    if r.get('tradingsymbol') == instrument and r.get('segment') == 'NSE':
                        ts = r.get('tick_size')
                        break
            cache[instrument] = float(ts) if ts else 0.05
        except Exception as e:
            logger.warning(f"[ORB] tick_size lookup failed for {instrument}: {e}")
            cache[instrument] = 0.05
        return cache[instrument]

    def modify_sl_m_order(self, position, new_trigger_price):
        """Update the SL trigger + limit (used by V9t_lock50 trail). Re-places
        if the order was already consumed or doesn't exist."""
        sl_order_id = position.get('kite_sl_order_id')
        tick = self._get_tick_size(position['instrument'])
        def _tick(v):
            return round(round(float(v) / tick) * tick, 2)
        trigger_price = _tick(new_trigger_price)
        direction = position['direction']
        if direction == 'LONG':
            limit_price = _tick(float(new_trigger_price) * 0.995)
        else:
            limit_price = _tick(float(new_trigger_price) * 1.005)
        if not sl_order_id:
            return self.place_sl_m_order({**position, 'sl_price': new_trigger_price})
        try:
            kite = self._get_kite()
            kite.modify_order(
                variety='regular', order_id=sl_order_id,
                trigger_price=trigger_price,
                price=limit_price,
            )
            logger.info(
                f"[ORB] SL modified: {position['instrument']} "
                f"new trigger={trigger_price} limit={limit_price} order={sl_order_id}"
            )
            return sl_order_id
        except Exception as e:
            logger.warning(
                f"[ORB] SL modify failed ({e}); re-placing with new trigger"
            )
            self.cancel_sl_m_order(position)
            return self.place_sl_m_order({**position, 'sl_price': new_trigger_price})

    @staticmethod
    def _compute_conviction(side, past_pct, cpr_pct, or_width_pct, rsi):
        """4-star rubric: CPR<0.3% · Risk<0.8% · RSI conviction ·
        Past% clean (0.2%-0.7%). Returns {conviction_score, conviction_grade,
        conviction_stars}."""
        stars = []
        stars.append({'key': 'cpr_narrow',
                      'hit': cpr_pct is not None and cpr_pct < 0.3,
                      'desc': 'CPR < 0.3% (clean setup)'})
        stars.append({'key': 'tight_risk',
                      'hit': or_width_pct is not None and or_width_pct < 0.8,
                      'desc': 'Risk % < 0.8 (tight SL)'})
        if side == 'LONG':
            rsi_ok = rsi is not None and rsi >= 65
        else:
            rsi_ok = rsi is not None and rsi <= 35
        stars.append({'key': 'rsi_conviction', 'hit': bool(rsi_ok),
                      'desc': 'RSI ≥65 (long) / ≤35 (short)'})
        stars.append({'key': 'clean_past',
                      'hit': past_pct is not None and 0.2 <= past_pct <= 0.7,
                      'desc': 'Past % in 0.2–0.7 (not chasing)'})
        score = sum(1 for s in stars if s['hit'])
        grade = 'A+' if score == 4 else 'A' if score == 3 else 'B' if score == 2 else 'C'
        return {'conviction_score': score,
                'conviction_grade': grade,
                'conviction_stars': stars}

    def fetch_kite_mis_positions(self) -> dict:
        """One-shot fetch of all Kite MIS NSE net positions keyed by
        tradingsymbol. Returns empty dict on failure so callers treat it as
        'no Kite data — proceed with normal flow'."""
        result: dict = {}
        try:
            kite = self._get_kite()
            pos_resp = kite.positions() or {}
            for p in pos_resp.get('net', []) or []:
                if p.get('product') == 'MIS' and p.get('exchange') == 'NSE':
                    result[p.get('tradingsymbol')] = {
                        'qty': int(p.get('quantity') or 0),
                        'buy_price': float(p.get('buy_price') or 0),
                        'sell_price': float(p.get('sell_price') or 0),
                        'buy_qty': int(p.get('buy_quantity') or 0),
                        'sell_qty': int(p.get('sell_quantity') or 0),
                    }
        except Exception as e:
            logger.warning(f"[ORB] Kite positions fetch failed: {e}")
        return result

    def get_kite_net_qty(self, instrument: str) -> dict:
        """Per-instrument wrapper (still useful for one-off callers).
        Returns {'qty': None, 'source': 'error'} if fetch fails so caller
        falls back to normal flow."""
        all_pos = self.fetch_kite_mis_positions()
        if instrument in all_pos:
            return {**all_pos[instrument], 'source': 'kite'}
        # Explicit fetch failed vs instrument simply not held — without
        # richer signal, assume qty 0 only if the fetch succeeded (empty
        # dict here could mean both). Safer to return None on empty.
        if not all_pos:
            return {'qty': None, 'source': 'error'}
        return {'qty': 0, 'source': 'kite', 'buy_price': 0, 'sell_price': 0,
                'buy_qty': 0, 'sell_qty': 0}

    def reconcile_position_with_kite(self, position: dict, kite_state: dict | None = None) -> bool:
        """Close the DB position only when Kite is genuinely 'gone':
            kite_qty == 0                    → flat (manual close / SL fill)
            sign(kite_qty) != expected_sign  → flipped (position reversed)

        Same-direction qty LARGER than DB expected (user manually ADDED to the
        position) is NOT a reason to close. Previous implementation closed on
        any mismatch, which wrongly marked IRCTC CLOSED on 2026-04-23 after
        the user manually added 360 more shorts to a 360 SHORT position
        (kite_qty -720 vs DB expected -360). Now the engine leaves DB alone
        in that case — user keeps managing the extra qty manually.

        Returns True if DB was closed (caller should return early without
        placing new orders). Returns False if Kite still matches (or we
        couldn't reach Kite) — caller proceeds with normal exit flow."""
        direction = position['direction']
        qty = position['qty']
        instrument = position['instrument']
        if kite_state is None:
            kite_state = self.get_kite_net_qty(instrument)
        kite_qty = kite_state.get('qty')
        if kite_qty is None:
            return False  # couldn't reach Kite — let caller proceed

        expected_sign = 1 if direction == 'LONG' else -1
        expected_abs = int(qty)
        # Match: exact expected qty (normal) OR same direction with MORE qty
        # (user added — leave alone, engine manages its share only).
        if kite_qty == expected_sign * expected_abs:
            return False
        if expected_sign > 0 and kite_qty >= expected_abs:
            return False  # LONG and Kite has >= expected → user added more longs
        if expected_sign < 0 and kite_qty <= -expected_abs:
            return False  # SHORT and Kite has <= -expected → user added more shorts

        # Genuine divergence: flat, opposite side, or partial close detected.
        logger.warning(
            f"[ORB] RECONCILE {instrument} {direction}: DB says open qty={qty} "
            f"but Kite net qty={kite_qty}. Closing DB without a new order to "
            f"avoid reopening the position."
        )
        # Derive exit price from Kite fills. For LONG the closing leg is a
        # SELL so use avg sell_price. For SHORT the closing leg is a BUY so
        # use avg buy_price.
        if direction == 'LONG':
            exit_price = kite_state.get('sell_price') or position['entry_price']
        else:
            exit_price = kite_state.get('buy_price') or position['entry_price']
        entry_price = position['entry_price']
        pnl_pts = (exit_price - entry_price) if direction == 'LONG' else (entry_price - exit_price)
        pnl_inr = round(pnl_pts * qty, 2)
        self.db.close_position(
            position_id=position['id'],
            exit_price=round(float(exit_price), 2),
            exit_time=datetime.now().isoformat(),
            exit_reason='RECONCILED_KITE_FLAT',
            pnl_pts=round(pnl_pts, 2),
            pnl_inr=pnl_inr,
        )
        with self._lock:
            self._positions.pop(instrument, None)
        # Also cancel any lingering SL order
        self.cancel_sl_m_order(position)
        try:
            ns = get_notification_service(self.cfg)
            ns.send_alert(
                'system_alert',
                f'ORB reconcile — {instrument} DB closed (Kite already flat)',
                f"DB thought {direction} {qty} @ {entry_price:.2f} was still open, "
                f"but Kite net qty was {kite_qty}. Closed DB at exit {exit_price:.2f} "
                f"(P&L Rs {pnl_inr:.0f}) without placing a new order.",
                data={'symbol': instrument, 'direction': direction,
                      'db_qty': qty, 'kite_qty': kite_qty,
                      'exit_price': exit_price, 'pnl_inr': pnl_inr},
                priority='normal',
            )
        except Exception:
            pass
        return True

    def ensure_sl_orders_placed(self):
        """Scan all OPEN positions and place SL-M on any that lack one.
        Safe to call repeatedly. Returns a summary dict."""
        if not self.cfg.get('use_exchange_sl_m', True):
            return {'placed': 0, 'skipped': 0, 'reason': 'use_exchange_sl_m=False'}

        placed, skipped = [], []
        for pos in self.db.get_open_positions() or []:
            if pos.get('kite_sl_order_id'):
                # Verify still active on Kite — if cancelled/rejected, re-place.
                sid = pos['kite_sl_order_id']
                try:
                    kite = self._get_kite()
                    hist = kite.order_history(sid) or []
                    last_status = (hist[-1].get('status') if hist else '').upper()
                    if last_status in ('TRIGGER PENDING', 'OPEN'):
                        skipped.append({'sym': pos['instrument'], 'reason': f'active ({last_status})'})
                        continue
                    if last_status == 'COMPLETE':
                        skipped.append({'sym': pos['instrument'], 'reason': 'already_triggered'})
                        continue
                    logger.info(
                        f"[ORB] SL-M {sid} for {pos['instrument']} is {last_status} — re-placing"
                    )
                except Exception as e:
                    logger.warning(f"[ORB] SL-M history check failed for {pos['instrument']}: {e}")

            new_id = self.place_sl_m_order(pos)
            if new_id:
                placed.append({'sym': pos['instrument'], 'order_id': new_id,
                               'trigger': pos['sl_price']})
            else:
                skipped.append({'sym': pos['instrument'], 'reason': 'place_failed'})

        logger.info(f"[ORB] ensure_sl_orders_placed: placed={len(placed)} skipped={len(skipped)}")
        return {'placed': placed, 'skipped': skipped}

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

        # Pre-flight: reconcile with Kite. If the position is already flat
        # there (manual close, SL fill we haven't seen yet), close DB and
        # return — placing a new exit would reopen a reverse position.
        if self.reconcile_position_with_kite(position):
            return

        # Cancel the on-exchange SL-M first — otherwise both could execute
        # and we'd oversell. Safe no-op if no SL-M was placed.
        self.cancel_sl_m_order(position)

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
                order_type='LIMIT',
                price=round(exit_price * (1.002 if transaction_type == 'BUY' else 0.998), 1),  # 0.2% buffer for fill
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
                f"Position remains OPEN on Kite — DB will NOT mark closed.",
                exc_info=True
            )
            # Log the failed order attempt so we can see what went wrong
            try:
                self.db.log_order(
                    position_id=position['id'],
                    instrument=instrument, tradingsymbol=instrument,
                    transaction_type=transaction_type, qty=qty,
                    order_type='LIMIT', price=round(exit_price, 2),
                    kite_order_id=None, status='FAILED',
                    exchange='NSE', product='MIS',
                    notes=f'EXIT {exit_reason} FAILED: {str(e)[:200]}',
                )
            except Exception as e2:
                logger.error(f"[ORB] Failed to log failed exit order: {e2}")
            # Critical alert — operator must intervene before MIS auto-squareoff at 15:20
            try:
                ns = get_notification_service(self.cfg)
                ns.send_alert(
                    'risk_alert',
                    f'EXIT FAILED — {instrument} {direction} still open on Kite',
                    f"Exit order failed ({exit_reason}): {str(e)[:150]}. "
                    f"Position qty={qty} entry={entry_price:.2f}. "
                    f"Place manual BUY/SELL before 15:20 MIS auto-squareoff.",
                    data={'symbol': instrument, 'direction': direction, 'qty': qty,
                          'entry_price': entry_price, 'exit_price': exit_price,
                          'exit_reason': exit_reason, 'error': str(e)[:200]},
                    priority='critical',
                )
            except Exception as e2:
                logger.error(f"[ORB] Exit-fail alert dispatch error: {e2}")
            # CRITICAL: do NOT mark position closed in DB. Return early.
            # Position stays OPEN in DB so next monitor/eod tick can retry.
            return

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
    # Candidates snapshot
    # ===================================================================

    def get_candidates(self) -> dict:
        """Return a structured snapshot of all stocks showing who's eligible,
        who already broke out (awaiting signal eval), who's watching inside OR,
        and who's excluded by wide-CPR. Also flags stocks already in a position
        or that have traded today.

        Returns:
          {
            'broken_out': [ { sym, ltp, or_high, or_low, side, past_pct, ... } ],
            'watching':   [ { sym, ltp, or_high, or_low, dist_up_pct, dist_dn_pct, ... } ],
            'excluded':   [ { sym, ltp, reason } ],
            'in_position':[ syms ],
            'traded_today':[ syms ],
            'as_of': iso_timestamp,
          }
        """
        today_str = date.today().isoformat()
        cpr_th = self.cfg.get('cpr_width_threshold_pct', 0.5)
        gap_th = self.cfg.get('gap_threshold_pct', 1.0)
        rsi_long = self.cfg.get('rsi_long_threshold', 60)
        rsi_short = self.cfg.get('rsi_short_threshold', 40)

        # Positions + closed trades today
        open_positions = self.db.get_open_positions() or []
        in_pos = {p['instrument'] for p in open_positions}
        closed = self.db.get_today_closed() or []
        traded_today = {t['instrument'] for t in closed}

        # Current LTPs for the universe
        symbols = list(self.stocks.keys())
        ltps = self.get_live_ltp(symbols)

        broken_out = []
        watching = []
        excluded = []

        for sym in symbols:
            state = self.db.get_or_create_daily_state(sym, today_str) or {}
            or_h = state.get('or_high')
            or_l = state.get('or_low')
            cpr = state.get('cpr_width_pct') or 0
            gap = state.get('gap_pct') or 0
            rsi = state.get('rsi_15m') or state.get('rsi')
            ltp = ltps.get(sym) or 0
            is_wide = cpr > cpr_th
            has_or = or_h is not None and or_l is not None

            if sym in in_pos:
                excluded.append({'sym': sym, 'ltp': round(ltp, 2), 'reason': 'in_position'})
                continue
            if sym in traded_today:
                excluded.append({'sym': sym, 'ltp': round(ltp, 2), 'reason': 'traded_today'})
                continue
            if is_wide:
                excluded.append({'sym': sym, 'ltp': round(ltp, 2), 'cpr': round(cpr, 3),
                                 'reason': f'wide_cpr_{cpr:.2f}pct'})
                continue
            if not has_or:
                excluded.append({'sym': sym, 'ltp': round(ltp, 2), 'reason': 'no_or_yet'})
                continue

            dist_up_pct = ((or_h - ltp) / ltp * 100) if ltp else None
            dist_dn_pct = ((ltp - or_l) / ltp * 100) if ltp else None
            long_gap_ok = gap <= gap_th
            long_rsi_ok = rsi is None or rsi >= rsi_long
            short_rsi_ok = rsi is None or rsi <= rsi_short

            # OR width as % of LTP — proxy for the risk size if the trade
            # fires (SL sits at OR-opposite, so risk ≈ OR range).
            or_width_abs = (or_h - or_l) if (or_h and or_l) else 0
            or_width_pct = (or_width_abs / ltp * 100) if ltp else 0
            row = {
                'sym': sym,
                'ltp': round(ltp, 2),
                'or_high': or_h, 'or_low': or_l,
                'or_width_pct': round(or_width_pct, 2),
                'cpr_width_pct': round(cpr, 3),
                'cpr_is_wide': bool(is_wide),
                'gap_pct': round(gap, 2),
                'rsi_15m': round(rsi, 1) if rsi is not None else None,
                'dist_up_pct': round(dist_up_pct, 2) if dist_up_pct is not None else None,
                'dist_dn_pct': round(dist_dn_pct, 2) if dist_dn_pct is not None else None,
                'long_gap_ok': long_gap_ok,
                'long_rsi_ok': long_rsi_ok,
                'short_rsi_ok': short_rsi_ok,
            }

            def _conviction(side: str, past_pct: float) -> dict:
                return self._compute_conviction(
                    side=side, past_pct=past_pct, cpr_pct=cpr,
                    or_width_pct=or_width_pct, rsi=rsi,
                )

            if ltp > or_h and long_gap_ok and long_rsi_ok:
                past_pct = (ltp - or_h) / or_h * 100
                broken_out.append({**row, 'side': 'LONG',
                                   'past_pct': round(past_pct, 2),
                                   **_conviction('LONG', past_pct)})
            elif ltp < or_l and short_rsi_ok:
                past_pct = (or_l - ltp) / or_l * 100
                broken_out.append({**row, 'side': 'SHORT',
                                   'past_pct': round(past_pct, 2),
                                   **_conviction('SHORT', past_pct)})
            else:
                side_hint = None
                if long_gap_ok and long_rsi_ok and short_rsi_ok:
                    side_hint = 'both'
                elif long_gap_ok and long_rsi_ok:
                    side_hint = 'long'
                elif short_rsi_ok:
                    side_hint = 'short'
                else:
                    side_hint = 'blocked'
                watching.append({**row, 'side_hint': side_hint})

        # Sort: broken_out by how far past (most decisive first),
        # watching by closest-to-breakout (smallest min distance)
        broken_out.sort(key=lambda r: -abs(r.get('past_pct') or 0))
        watching.sort(key=lambda r: min(
            abs(r.get('dist_up_pct') or 99), abs(r.get('dist_dn_pct') or 99)))

        return {
            'broken_out': broken_out,
            'watching': watching,
            'excluded': excluded,
            'in_position': sorted(in_pos),
            'traded_today': sorted(traded_today),
            'as_of': datetime.now().isoformat(timespec='seconds'),
        }

    # ===================================================================
    # Catch-up: take trades for breakouts we missed earlier today
    # ===================================================================

    def catchup_missed_breakouts(self) -> dict:
        """Recovery helper — if the engine was offline/broken when breakouts
        fired earlier, walk today's 5-min candles and take entries for stocks
        whose first post-OR candle closed decisively past OR and are STILL
        beyond OR right now. Uses current LTP as entry price.

        Respects last_entry_time and all standard filters (VWAP, RSI, CPR
        direction, gap, funds, qty). Returns a dict of what it did.
        """
        import json as _json
        now = datetime.now()
        today_str = date.today().isoformat()
        cfg = self.cfg
        result = {'entered': [], 'skipped': [], 'as_of': now.isoformat(timespec='seconds')}

        last_entry = cfg.get('last_entry_time', '14:00')
        if now.strftime('%H:%M') > last_entry:
            result['error'] = f'past last_entry_time {last_entry}'
            return result

        gate = self.daily_loss_gate()
        if gate['block_new_entries']:
            result['error'] = (
                f"daily_loss_cap_breached "
                f"(realized loss Rs {gate['realized_loss']:,.0f} ≥ cap Rs {gate['cap']:,.0f})"
            )
            result['gate'] = gate
            return result

        session_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
        ltps = self.get_live_ltp(list(self.stocks.keys()))

        for sym in self.stocks:
            try:
                with self._lock:
                    or_s = self._or_state.get(sym) or {}
                or_high = or_s.get('high')
                or_low = or_s.get('low')
                if not or_s.get('finalized') or or_high is None or or_low is None:
                    result['skipped'].append({'sym': sym, 'reason': 'or_not_finalized'})
                    continue
                with self._lock:
                    if sym in self._positions:
                        result['skipped'].append({'sym': sym, 'reason': 'in_position'})
                        continue
                daily_state = self.db.get_or_create_daily_state(sym, today_str) or {}
                if (daily_state.get('trades_taken') or 0) >= cfg.get('max_trades_per_day', 1):
                    result['skipped'].append({'sym': sym, 'reason': 'traded_today'})
                    continue
                if daily_state.get('is_wide_cpr_day'):
                    result['skipped'].append({'sym': sym, 'reason': 'wide_cpr'})
                    continue

                candles = self.fetch_5min_candles(sym, session_start, now)
                if not candles or len(candles) < 4:
                    result['skipped'].append({'sym': sym, 'reason': 'insufficient_candles'})
                    continue

                # Walk candles from the first post-OR candle onward. Start
                # comparisons at or_end_idx but use candles[i-1] (which may
                # be the last PRE-OR candle) as the prev — otherwise a
                # transition that lands on the very first post-OR candle
                # (TRENT 2026-04-23: 9:30 close 4380.10 vs OR_low 4380.20)
                # has no prior post-OR candle to compare against and is
                # lost. Pick the LATEST transition still consistent with
                # current LTP (so a SHORT that reversed to LONG catches
                # the LONG, not the stale SHORT).
                or_end = session_start + timedelta(minutes=cfg.get('or_minutes', 15))
                def _ctime(c):
                    t = c['date']
                    if hasattr(t, 'tzinfo') and t.tzinfo is not None:
                        t = t.replace(tzinfo=None)
                    return t
                or_end_idx = next(
                    (i for i, c in enumerate(candles) if _ctime(c) >= or_end),
                    len(candles)
                )
                ltp = ltps.get(sym) or 0
                direction = None
                first_breakout = None  # name kept for schema compat — actually the LATEST valid breakout
                all_transitions = []   # for debug logging only
                for i in range(or_end_idx, len(candles)):
                    if i == 0:
                        continue
                    prev_c = candles[i - 1]
                    cur = candles[i]
                    if cur['close'] > or_high and prev_c['close'] <= or_high:
                        all_transitions.append(('LONG', cur))
                        if ltp > or_high:
                            direction = 'LONG'
                            first_breakout = cur
                    elif cur['close'] < or_low and prev_c['close'] >= or_low:
                        all_transitions.append(('SHORT', cur))
                        if ltp < or_low:
                            direction = 'SHORT'
                            first_breakout = cur
                if direction is None or first_breakout is None:
                    if all_transitions:
                        last_dir, last_c = all_transitions[-1]
                        result['skipped'].append({
                            'sym': sym,
                            'reason': f'ltp_back_inside_OR (latest transition {last_dir} '
                                      f'at {_ctime(last_c).strftime("%H:%M")} reversed)',
                            'ltp': ltp, 'or_high': or_high, 'or_low': or_low,
                            'transitions': len(all_transitions),
                        })
                    else:
                        result['skipped'].append({'sym': sym, 'reason': 'no_breakout_found'})
                    continue

                # Slippage cap: reject if LTP has run too far past the breakout close.
                # Measured in R-multiples (R = initial risk from the breakout candle).
                # If the move has already eaten ~X R, a late entry has bad R:R.
                br_close = first_breakout['close']
                if direction == 'LONG':
                    r0 = br_close - or_low
                    slippage = ltp - br_close
                else:
                    r0 = or_high - br_close
                    slippage = br_close - ltp
                slip_cap_r = cfg.get('catchup_max_slippage_r', 0.5)
                if r0 > 0 and slippage / r0 > slip_cap_r:
                    result['skipped'].append({
                        'sym': sym,
                        'reason': f'slippage_too_far ({slippage/r0:.2f}R > {slip_cap_r}R cap)',
                        'direction': direction,
                        'first_breakout_close': br_close,
                        'ltp': ltp,
                        'slippage_r': round(slippage / r0, 2),
                    })
                    continue

                # Reuse filter checks — need VWAP + RSI now, CPR and gap from daily_state
                vwap = self.compute_vwap(candles)
                rsi = self.compute_rsi_15m(sym)
                self.db.update_daily_state(sym, today_str, vwap=vwap, rsi_15m=rsi)

                filters = {}
                all_passed = True
                if direction == 'LONG' and not cfg.get('allow_longs', True):
                    filters['allow_longs'] = False; all_passed = False
                if direction == 'SHORT' and not cfg.get('allow_shorts', True):
                    filters['allow_shorts'] = False; all_passed = False
                if cfg.get('use_vwap_filter', True) and vwap is not None:
                    if direction == 'LONG' and ltp <= vwap:
                        filters['vwap'] = False; all_passed = False
                    elif direction == 'SHORT' and ltp >= vwap:
                        filters['vwap'] = False; all_passed = False
                    else:
                        filters['vwap'] = True
                if cfg.get('use_rsi_filter', True) and rsi is not None:
                    rsi_long = cfg.get('rsi_long_threshold', 60)
                    rsi_short = cfg.get('rsi_short_threshold', 40)
                    if direction == 'LONG' and rsi < rsi_long:
                        filters['rsi'] = False; all_passed = False
                    elif direction == 'SHORT' and rsi > rsi_short:
                        filters['rsi'] = False; all_passed = False
                    else:
                        filters['rsi'] = True
                cpr_tc = daily_state.get('cpr_tc')
                cpr_bc = daily_state.get('cpr_bc')
                if cfg.get('use_cpr_dir_filter', True) and cpr_tc and cpr_bc:
                    if direction == 'LONG' and ltp <= cpr_tc:
                        filters['cpr_dir'] = False; all_passed = False
                    elif direction == 'SHORT' and ltp >= cpr_bc:
                        filters['cpr_dir'] = False; all_passed = False
                    else:
                        filters['cpr_dir'] = True
                gap_pct = daily_state.get('gap_pct')
                gap_block_pct = cfg.get('gap_long_block_pct', 0.3)
                if cfg.get('use_gap_filter', True) and gap_pct is not None:
                    if direction == 'LONG' and gap_pct > gap_block_pct:
                        filters['gap'] = False; all_passed = False
                    else:
                        filters['gap'] = True

                if not all_passed:
                    result['skipped'].append({
                        'sym': sym, 'reason': 'filters_blocked',
                        'direction': direction, 'filters': filters,
                    })
                    continue

                # Compute SL/target from entry = current LTP
                entry_price = ltp
                if direction == 'LONG':
                    sl_price = or_low
                    risk = entry_price - sl_price
                else:
                    sl_price = or_high
                    risk = sl_price - entry_price
                if risk <= 0:
                    result['skipped'].append({'sym': sym, 'reason': 'zero_risk'})
                    continue
                r_multiple = cfg.get('r_multiple', 1.5)
                target_price = entry_price + (r_multiple * risk) if direction == 'LONG' \
                    else entry_price - (r_multiple * risk)

                # Risk-based sizing (same helper as live eval) — uses the
                # OR-opposite SL we just computed.
                sizing = self.compute_risk_based_qty(entry_price, sl_price)
                qty = sizing['qty']
                if qty <= 0:
                    result['skipped'].append({
                        'sym': sym, 'reason': sizing.get('reason', 'qty_zero'),
                    })
                    continue
                self.stocks[sym]['qty'] = qty

                available = self._get_available_margin()
                min_balance_required = self.min_margin_for_trade
                if available is not None and available < min_balance_required:
                    result['skipped'].append({
                        'sym': sym, 'reason': 'low_funds',
                        'available': available, 'required': min_balance_required,
                    })
                    continue
                required_capital = qty * entry_price
                if available is not None and required_capital > available:
                    qty = int(available // entry_price)
                    if qty <= 0:
                        result['skipped'].append({'sym': sym, 'reason': 'insufficient_funds_zero_qty'})
                        continue

                # Log the "catchup" signal
                self.db.log_signal(
                    instrument=sym, signal_time=now.isoformat(),
                    direction=direction, entry_price=round(entry_price, 2),
                    sl_price=round(sl_price, 2), target_price=round(target_price, 2),
                    or_high=round(or_high, 2), or_low=round(or_low, 2),
                    gap_pct=gap_pct, vwap=vwap, rsi_15m=rsi,
                    cpr_pivot=daily_state.get('cpr_pivot'),
                    cpr_tc=cpr_tc, cpr_bc=cpr_bc,
                    cpr_width_pct=daily_state.get('cpr_width_pct'),
                    filters_passed=_json.dumps(filters),
                    action_taken='CATCHUP_ENTRY',
                )

                kite_order_id = self.place_entry_order(sym, direction, qty, entry_price)
                if kite_order_id is None:
                    result['skipped'].append({'sym': sym, 'reason': 'order_placement_failed'})
                    continue

                # Compute conviction (same rubric as dashboard candidates).
                past_pct_conv = ((entry_price - or_high) / or_high * 100) if direction == 'LONG' \
                    else ((or_low - entry_price) / or_low * 100)
                or_width_pct_conv = ((or_high - or_low) / entry_price * 100) if entry_price else None
                conv = self._compute_conviction(
                    side=direction, past_pct=past_pct_conv,
                    cpr_pct=daily_state.get('cpr_width_pct'),
                    or_width_pct=or_width_pct_conv, rsi=rsi,
                )

                # Persist position (without this, monitor_positions will never
                # see it and SL/target will never fire). Matches the normal
                # signal flow in evaluate_signals().
                position_id = self.db.add_position(
                    instrument=sym, trade_date=today_str, direction=direction,
                    qty=qty, entry_price=round(entry_price, 2),
                    entry_time=now.isoformat(),
                    sl_price=round(sl_price, 2), target_price=round(target_price, 2),
                    or_high=round(or_high, 2), or_low=round(or_low, 2),
                    kite_entry_order_id=kite_order_id, status='OPEN',
                    gap_pct=gap_pct, vwap_at_entry=vwap, rsi_at_entry=rsi,
                    cpr_tc=cpr_tc, cpr_bc=cpr_bc,
                    cpr_width_pct=daily_state.get('cpr_width_pct'),
                    conviction_grade=conv['conviction_grade'],
                    conviction_score=conv['conviction_score'],
                    conviction_stars=_json.dumps(conv['conviction_stars']),
                    notes='CATCHUP_ENTRY',
                )
                self.db.update_daily_state(sym, today_str,
                    trades_taken=(daily_state.get('trades_taken') or 0) + 1,
                )
                pos = self.db.get_open_positions(instrument=sym)
                with self._lock:
                    if pos:
                        self._positions[sym] = pos[0]

                if pos and self.cfg.get('use_exchange_sl_m', True):
                    self.place_sl_m_order(pos[0])
                    pos = self.db.get_open_positions(instrument=sym)
                    with self._lock:
                        if pos:
                            self._positions[sym] = pos[0]

                first_bt = _ctime(first_breakout)
                logger.info(
                    f'[ORB-CATCHUP] {sym} {direction} qty={qty} @{entry_price:.2f} '
                    f'SL={sl_price:.2f} TGT={target_price:.2f} order={kite_order_id} '
                    f'pos_id={position_id} (missed breakout at {first_bt.strftime("%H:%M")} '
                    f'close {first_breakout["close"]:.2f})'
                )

                # Entry notification — route through NotificationService
                if self.cfg.get('notify_on_entry', True):
                    try:
                        ns = get_notification_service(self.cfg)
                        ns.send_alert(
                            'trade_entry',
                            f'CATCHUP {direction} {sym} @ {entry_price:.2f}',
                            f'Qty: {qty} | SL: {sl_price:.2f} | TGT: {target_price:.2f} | '
                            f'Missed breakout {first_bt.strftime("%H:%M")}',
                            data={'symbol': sym, 'direction': direction,
                                  'entry_price': entry_price, 'sl_price': sl_price,
                                  'target_price': target_price, 'qty': qty,
                                  'catchup': True},
                        )
                    except Exception as e:
                        logger.warning(f'[ORB-CATCHUP] Entry notification failed: {e}')

                result['entered'].append({
                    'sym': sym, 'direction': direction,
                    'first_breakout_time': first_bt.strftime('%H:%M'),
                    'first_breakout_close': first_breakout['close'],
                    'entry_price': entry_price,
                    'sl_price': sl_price, 'target_price': target_price,
                    'qty': qty, 'kite_order_id': kite_order_id,
                    'position_id': position_id,
                })
            except Exception as e:
                logger.error(f'[ORB-CATCHUP] {sym} error: {e}', exc_info=True)
                result['skipped'].append({'sym': sym, 'reason': f'error: {e}'})
        return result

    # ===================================================================
    # Mid-morning status (10:30)
    # ===================================================================

    def send_midmorning_status(self):
        """Send a snapshot of the day's progress at 10:30 IST — email + WhatsApp.
        Includes candidate breakdown (broken out, watching, excluded)."""
        if not self.cfg.get('notify_midmorning_status', True):
            return
        from services.notifications import get_notification_service
        ns = get_notification_service(self.cfg)
        try:
            open_positions = self.db.get_open_positions() or []
            closed = self.db.get_today_closed() or []
            day_pnl = sum((t.get('pnl_inr') or 0) for t in closed)
            day_pnl += sum((p.get('pnl_inr') or 0) for p in open_positions)
            margin = self._last_margin or {}

            # Pull candidates snapshot
            cand = self.get_candidates()
            broken = cand.get('broken_out', [])
            watching = cand.get('watching', [])
            excluded = cand.get('excluded', [])

            lines = [
                f"Open: {len(open_positions)} · Closed today: {len(closed)} · "
                f"Day P&L: {'+'if day_pnl>=0 else ''}Rs {day_pnl:,.0f}"
            ]
            if margin.get('cash') is not None:
                lines[0] += f" · Margin: Rs {margin['cash']:,.0f}"

            if broken:
                lines.append('')
                lines.append(f"Broken out ({len(broken)}) — awaiting signal eval:")
                for r in broken[:10]:
                    arrow = '▲' if r['side'] == 'LONG' else '▼'
                    lines.append(
                        f"  {arrow} {r['sym']} {r['side']} @ {r['ltp']} "
                        f"(OR {r['or_low']}-{r['or_high']}, {r['past_pct']:+.2f}% past)"
                    )
            if watching:
                lines.append('')
                lines.append(f"Watching ({len(watching)}) — inside OR:")
                for r in watching[:10]:
                    nearest = min((r.get('dist_up_pct') or 99), (r.get('dist_dn_pct') or 99))
                    lines.append(
                        f"  · {r['sym']} {r['ltp']} "
                        f"(OR {r['or_low']}-{r['or_high']}, nearest {nearest:.2f}%)"
                    )
            if excluded:
                wide = [e for e in excluded if str(e.get('reason','')).startswith('wide_cpr')]
                if wide:
                    lines.append('')
                    lines.append(
                        f"Excluded ({len(wide)}) — wide CPR: "
                        + ', '.join(e['sym'] for e in wide)
                    )

            msg = '\n'.join(lines)
            ns.send_alert(
                'system_alert',
                'ORB mid-morning status (10:30)',
                msg,
                data={
                    'open_positions': len(open_positions),
                    'closed_today': len(closed),
                    'day_pnl': day_pnl,
                    'available_margin': margin.get('cash', 0),
                    'broken_out_count': len(broken),
                    'watching_count': len(watching),
                    'excluded_wide_cpr': [e['sym'] for e in excluded if str(e.get('reason','')).startswith('wide_cpr')],
                },
                priority='low',
            )
        except Exception as e:
            logger.error(f"[ORB] midmorning status error: {e}", exc_info=True)

    # ===================================================================
    # Emergency
    # ===================================================================

    def emergency_exit_all(self):
        """Kill switch -- close all open positions immediately at market."""
        logger.warning("[ORB] EMERGENCY EXIT triggered")
        return self.eod_squareoff()
