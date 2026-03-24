"""
Maruthi WebSocket Candle Aggregator
=====================================

Streams live ticks via KiteTicker WebSocket and aggregates them into
30-minute OHLCV candles. Fires strategy logic the instant a candle closes.

This replaces 14 scheduled cron jobs with a single persistent connection.

Architecture:
    KiteTicker (ticks) → CandleAggregator (30-min OHLCV) → on_candle_close callback
                                                              ↓
                                                      MaruthiExecutor.run_candle_check()

Also subscribes to the MARUTI futures token so we get accurate futures price
for trigger order fill detection.
"""

import logging
import threading
import pandas as pd
from datetime import datetime, time as dtime, timedelta
from typing import Callable, Dict, List, Optional
from collections import deque

from kiteconnect import KiteTicker
from services.kite_service import get_access_token, KITE_API_KEY

logger = logging.getLogger(__name__)

# Market hours
MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)

# 30-minute candle boundaries (IST)
CANDLE_MINUTES = 30
CANDLE_BOUNDARIES = []
t = datetime(2000, 1, 1, 9, 15)
end = datetime(2000, 1, 1, 15, 30)
while t < end:
    CANDLE_BOUNDARIES.append(t.time())
    t += timedelta(minutes=CANDLE_MINUTES)
CANDLE_BOUNDARIES.append(MARKET_CLOSE)  # 15:30 is the last boundary


class CandleAggregator:
    """
    Aggregates ticks into 30-minute OHLCV candles.

    Maintains a rolling window of completed candles (for SuperTrend computation)
    and fires a callback when each candle closes.
    """

    def __init__(self, max_candles: int = 200):
        self.max_candles = max_candles
        self.completed_candles: deque = deque(maxlen=max_candles)
        self.current_candle: Optional[dict] = None
        self.current_candle_end: Optional[dtime] = None
        self.on_candle_close: Optional[Callable] = None
        self._lock = threading.Lock()

    def _get_candle_boundary(self, now: datetime) -> tuple:
        """
        Get the start and end time of the current 30-min candle.

        Returns (candle_start: datetime, candle_end: time)
        """
        current_time = now.time()

        for i in range(len(CANDLE_BOUNDARIES) - 1):
            if current_time >= CANDLE_BOUNDARIES[i] and current_time < CANDLE_BOUNDARIES[i + 1]:
                candle_start = now.replace(
                    hour=CANDLE_BOUNDARIES[i].hour,
                    minute=CANDLE_BOUNDARIES[i].minute,
                    second=0, microsecond=0
                )
                return candle_start, CANDLE_BOUNDARIES[i + 1]

        # Outside market hours
        return None, None

    def process_tick(self, ltp: float, volume: int = 0, timestamp: datetime = None):
        """
        Process a single tick and update the current candle.

        Args:
            ltp: Last traded price
            volume: Cumulative volume
            timestamp: Tick timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Skip ticks outside market hours
        if timestamp.time() < MARKET_OPEN or timestamp.time() >= MARKET_CLOSE:
            return

        with self._lock:
            candle_start, candle_end = self._get_candle_boundary(timestamp)
            if candle_start is None:
                return

            # Check if we've crossed into a new candle period
            if self.current_candle_end is not None and candle_end != self.current_candle_end:
                # Current candle just closed — finalize it
                if self.current_candle:
                    self._close_current_candle()

            # Start new candle or update existing
            if self.current_candle is None or candle_end != self.current_candle_end:
                self.current_candle = {
                    'date': candle_start,
                    'open': ltp,
                    'high': ltp,
                    'low': ltp,
                    'close': ltp,
                    'volume': volume,
                }
                self.current_candle_end = candle_end
            else:
                # Update OHLC
                self.current_candle['high'] = max(self.current_candle['high'], ltp)
                self.current_candle['low'] = min(self.current_candle['low'], ltp)
                self.current_candle['close'] = ltp
                self.current_candle['volume'] = volume  # Kite sends cumulative volume

    def _close_current_candle(self):
        """Finalize the current candle and fire callback."""
        candle = self.current_candle.copy()
        self.completed_candles.append(candle)
        logger.info(
            f"Candle closed: {candle['date']} O={candle['open']:.1f} "
            f"H={candle['high']:.1f} L={candle['low']:.1f} C={candle['close']:.1f}"
        )

        # Fire callback in a separate thread to not block tick processing
        if self.on_candle_close:
            threading.Thread(
                target=self._fire_callback,
                args=(candle,),
                daemon=True
            ).start()

        self.current_candle = None
        self.current_candle_end = None

    def _fire_callback(self, candle: dict):
        """Fire the candle close callback safely."""
        try:
            self.on_candle_close(candle)
        except Exception as e:
            logger.error(f"Candle close callback error: {e}", exc_info=True)

    def force_close(self):
        """Force-close current candle (called at market close 15:30)."""
        with self._lock:
            if self.current_candle:
                self._close_current_candle()

    def get_dataframe(self) -> pd.DataFrame:
        """
        Get completed candles as a DataFrame (for SuperTrend computation).

        Returns DataFrame with columns: date, open, high, low, close, volume
        """
        with self._lock:
            candles = list(self.completed_candles)
            # Include current in-progress candle for the latest ST value
            if self.current_candle:
                candles.append(self.current_candle)

        if not candles:
            return pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])

        return pd.DataFrame(candles)

    def load_historical(self, candles: List[dict]):
        """
        Preload historical candles (e.g., from Kite API on startup).

        This seeds the aggregator so SuperTrend has enough history
        from the very first live candle.
        """
        with self._lock:
            for c in candles:
                self.completed_candles.append(c)
        logger.info(f"Loaded {len(candles)} historical candles")


class MaruthiTicker:
    """
    Manages the KiteTicker WebSocket connection for the Maruthi strategy.

    Subscribes to MARUTI equity ticks, aggregates into 30-min candles,
    and triggers strategy execution on each candle close.
    """

    def __init__(self, config: dict = None):
        from config import MARUTHI_DEFAULTS
        self.config = config or MARUTHI_DEFAULTS.copy()
        self.symbol = self.config.get('symbol', 'MARUTI')
        self.kws: Optional[KiteTicker] = None
        self.aggregator = CandleAggregator(max_candles=200)
        self.is_connected = False
        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self._instrument_token: Optional[int] = None
        self._last_ltp: float = 0.0
        self._last_tick_time: Optional[datetime] = None
        self._lock = threading.Lock()

        # Set candle close callback
        self.aggregator.on_candle_close = self._on_candle_close

    def _get_instrument_token(self) -> Optional[int]:
        """Get MARUTI NSE equity instrument token."""
        if self._instrument_token:
            return self._instrument_token

        try:
            from services.nifty500_universe import get_instrument_token
            token = get_instrument_token(self.symbol)
            if token:
                self._instrument_token = token
                return token
        except Exception:
            pass

        # Fallback: load from Kite
        try:
            from services.kite_service import get_kite
            kite = get_kite()
            instruments = kite.instruments("NSE")
            for inst in instruments:
                if inst['tradingsymbol'] == self.symbol and inst['instrument_type'] == 'EQ':
                    self._instrument_token = inst['instrument_token']
                    return self._instrument_token
        except Exception as e:
            logger.error(f"Failed to get instrument token for {self.symbol}: {e}")

        return None

    def _load_historical_candles(self):
        """Load recent 30-min candles from Kite to seed SuperTrend."""
        try:
            from services.kite_service import get_kite
            kite = get_kite()
            token = self._get_instrument_token()
            if not token:
                logger.warning("No instrument token — cannot load history")
                return

            to_date = datetime.now()
            from_date = to_date - timedelta(days=30)  # ~200 candles of 30-min data
            data = kite.historical_data(
                instrument_token=token,
                from_date=from_date,
                to_date=to_date,
                interval='30minute'
            )

            candles = [{
                'date': d['date'],
                'open': d['open'],
                'high': d['high'],
                'low': d['low'],
                'close': d['close'],
                'volume': d.get('volume', 0),
            } for d in data]

            self.aggregator.load_historical(candles)
            if candles:
                self._last_ltp = candles[-1]['close']

        except Exception as e:
            logger.error(f"Failed to load historical candles: {e}", exc_info=True)

    def _catchup_signal_check(self):
        """
        Run signal check on historical candles to catch flips that
        happened while the ticker was offline (e.g., gap open, overnight).

        Scans all candles since last processed candle for master/child flips.
        Processes each flip in chronological order using the DF truncated to
        that flip's candle (so detect_signals sees it as the latest row).
        """
        try:
            df = self.aggregator.get_dataframe()
            if len(df) < 20:
                logger.info("[MaruthiTicker] Not enough candles for catch-up check")
                return

            from services.maruthi_executor import get_maruthi_executor
            from services.maruthi_strategy import compute_dual_supertrend
            from services.maruthi_db import get_maruthi_db

            db = get_maruthi_db()
            regime = db.get_regime()
            db_last_candle = str(regime.get('last_candle_time', ''))

            # Compute current indicators
            cfg = self.config
            df_st = compute_dual_supertrend(
                df,
                master_period=cfg.get('master_atr_period', 7),
                master_mult=cfg.get('master_multiplier', 5.0),
                child_period=cfg.get('child_atr_period', 7),
                child_mult=cfg.get('child_multiplier', 2.0),
            )

            # Find all flip candles since last processed
            flips = []
            for i in range(1, len(df_st)):
                row = df_st.iloc[i]
                prev = df_st.iloc[i - 1]
                candle_date = str(row.get('date', ''))

                # Skip candles already processed
                if db_last_candle and candle_date <= db_last_candle:
                    continue

                has_flip = False
                if int(row['master_dir']) != int(prev['master_dir']):
                    has_flip = True
                    logger.info(f"[MaruthiTicker] Catch-up: master flip at {candle_date}")
                if int(row['child_dir']) != int(prev['child_dir']):
                    has_flip = True
                    logger.info(f"[MaruthiTicker] Catch-up: child flip at {candle_date}")

                if has_flip:
                    flips.append(i)

            executor = get_maruthi_executor(self.config)

            if flips:
                # Process each flip in order
                for idx in flips:
                    flip_df = df_st.iloc[:idx + 1].copy()
                    results = executor.run_candle_check(flip_df)
                    if results:
                        logger.info(f"[MaruthiTicker] Catch-up actions at row {idx}: {results}")

                # Also run on full DF to update ATR/SL to latest
                executor.run_candle_check(df_st)
                logger.info(f"[MaruthiTicker] Catch-up complete: processed {len(flips)} flips")
            else:
                # No missed flips — just update ATR/SL
                executor.run_candle_check(df_st)
                logger.info("[MaruthiTicker] Catch-up: no missed flips, ATR/SL updated")
        except Exception as e:
            logger.error(f"[MaruthiTicker] Catch-up signal check error: {e}", exc_info=True)

    def _on_ticks(self, ws, ticks):
        """Handle incoming ticks from KiteTicker."""
        for tick in ticks:
            token = tick.get('instrument_token')
            ltp = tick.get('last_price', 0)

            if token == self._instrument_token and ltp > 0:
                volume = tick.get('volume_traded', 0)
                timestamp = tick.get('exchange_timestamp') or datetime.now()

                self._last_ltp = ltp
                self._last_tick_time = datetime.now()
                self.aggregator.process_tick(ltp, volume, timestamp)

                # Real-time hard SL check on every tick
                self._check_hard_sl_tick(ltp)

                # Check if any pending trigger orders have filled
                self._check_pending_triggers(ltp)

                # Broadcast to dashboard via SocketIO (~2/sec throttle)
                self._emit_tick(ltp)

            # Forward ALL ticks to NAS ticker (if active)
            if ltp > 0:
                self._forward_to_nas(tick)

            # Forward BANKNIFTY ticks to BNF for live spot
            if tick.get('instrument_token') == 260105 and ltp > 0:
                self._forward_to_bnf(tick)

            # Forward position-subscribed ticks to positions page via SocketIO
            if ltp > 0 and hasattr(self, '_pos_token_map') and token in self._pos_token_map:
                self._emit_pos_tick(token, ltp)

    _pending_cache = None
    _pending_cache_time = 0

    def _check_pending_triggers(self, ltp: float):
        """
        Check if pending SL-L trigger orders have been filled.

        Paper mode: Simulate fill when LTP crosses trigger (tick-level).
        Live mode: Poll kite.orders() every 30 seconds to detect fills.
                   On fill → activate position, place protective option, notify.

        Uses a 5-second DB cache to avoid hitting SQLite on every tick.
        """
        import time
        now = time.time()

        # Refresh DB cache every 5 seconds
        if MaruthiTicker._pending_cache is None or now - MaruthiTicker._pending_cache_time > 5:
            try:
                from services.maruthi_db import get_maruthi_db
                db = get_maruthi_db()
                MaruthiTicker._pending_cache = db.get_pending_positions()
                MaruthiTicker._pending_cache_time = now
            except Exception:
                return

        pending = MaruthiTicker._pending_cache
        if not pending:
            return

        is_paper = self.config.get('paper_trading_mode', True)

        if is_paper:
            # Paper mode: simulate trigger fill at tick level
            self._check_paper_triggers(pending, ltp)
        # Live mode: no polling needed — on_order_update handles fills instantly

    def _check_paper_triggers(self, pending: list, ltp: float):
        """Paper mode: simulate trigger fill when LTP crosses trigger price."""
        try:
            from services.maruthi_db import get_maruthi_db
            from services.maruthi_executor import get_maruthi_executor
            db = get_maruthi_db()

            for pos in pending:
                trigger = pos.get('trigger_price', 0)
                direction = pos.get('transaction_type', '')
                if not trigger:
                    continue

                triggered = False
                if direction == 'BUY' and ltp >= trigger:
                    triggered = True
                elif direction == 'SELL' and ltp <= trigger:
                    triggered = True

                if triggered:
                    db.activate_position(pos['id'], entry_price=trigger)
                    MaruthiTicker._pending_cache = None  # Force refresh
                    logger.info(
                        f"[MaruthiTicker] Paper trigger filled: {pos.get('tradingsymbol')} "
                        f"{direction} @ {trigger} (LTP={ltp})"
                    )
                    # Run post-fill actions (protective options, etc.)
                    try:
                        executor = get_maruthi_executor(self.config)
                        executor.on_trigger_fill(pos, fill_price=trigger)
                    except Exception as e:
                        logger.error(f"[MaruthiTicker] Post-fill action error: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"[MaruthiTicker] Paper trigger check error: {e}", exc_info=True)


    def _check_hard_sl_tick(self, ltp: float):
        """
        Instant tick-level hard SL check.

        Fires immediately when price breaches the trailing hard SL level,
        without waiting for 30-min candle close. Closes ALL positions.
        """
        try:
            from services.maruthi_db import get_maruthi_db
            db = get_maruthi_db()
            regime = db.get_regime()

            current_regime = regime.get('regime', 'FLAT')
            hard_sl = regime.get('hard_sl_price', 0)

            if current_regime == 'FLAT' or not hard_sl:
                return

            sl_hit = False
            if current_regime == 'BULL' and ltp <= hard_sl:
                sl_hit = True
            elif current_regime == 'BEAR' and ltp >= hard_sl:
                sl_hit = True

            if not sl_hit:
                return

            # Prevent duplicate triggers (check if already went FLAT)
            regime = db.get_regime()  # Re-read to avoid race
            if regime.get('regime') == 'FLAT':
                return

            logger.warning(f"[MaruthiTicker] HARD SL HIT! LTP={ltp} SL={hard_sl} Regime={current_regime}")
            logger.warning(f"[MaruthiTicker] Closing ALL positions immediately — going FLAT")

            from services.maruthi_executor import get_maruthi_executor
            executor = get_maruthi_executor(self.config)

            # Close ALL positions — no exceptions on hard SL
            closed = executor.close_all_positions(
                exit_reason='HARD_SL',
                keep_last_option=False,  # Hard SL = close EVERYTHING
            )

            # Set regime to FLAT
            db.update_regime(
                regime='FLAT',
                last_signal_time=datetime.now().isoformat(),
            )

            # Log signal
            db.log_signal(
                signal_type='HARD_SL',
                regime='FLAT',
                master_direction=regime.get('master_direction', 0),
                child_direction=regime.get('child_direction', 0),
                master_st_value=regime.get('master_st_value', 0),
                child_st_value=0,
                candle_time=datetime.now().isoformat(),
                candle_high=ltp,
                candle_low=ltp,
                candle_close=ltp,
                action_taken=f'TICK HARD SL: closed {closed} positions at LTP={ltp}',
            )

            logger.warning(f"[MaruthiTicker] HARD SL complete: closed {closed} positions, now FLAT")

        except Exception as e:
            logger.error(f"[MaruthiTicker] Hard SL tick check error: {e}", exc_info=True)

    _last_emit_time = 0

    def _emit_tick(self, ltp: float):
        """Broadcast tick to dashboard via SocketIO. Throttled to ~2/sec."""
        import time
        now = time.time()
        if now - MaruthiTicker._last_emit_time < 0.5:
            return
        MaruthiTicker._last_emit_time = now

        try:
            from app import socketio
            from services.maruthi_db import get_maruthi_db

            db = get_maruthi_db()
            regime = db.get_regime()

            candle = self.aggregator.current_candle or {}

            socketio.emit('maruthi_tick', {
                'ltp': round(ltp, 1),
                'ts': now,
                'master_st': regime.get('master_st_value', 0),
                'child_st': regime.get('child_st_value', 0),
                'hard_sl': regime.get('hard_sl_price', 0),
                'regime': regime.get('regime', 'FLAT'),
                'master_dir': regime.get('master_direction', 0),
                'child_dir': regime.get('child_direction', 0),
                'candle': {
                    'o': candle.get('open', 0),
                    'h': candle.get('high', 0),
                    'l': candle.get('low', 0),
                    'c': candle.get('close', 0),
                } if candle else None,
            })
        except Exception:
            pass  # Silently fail — dashboard is optional

    # Throttle for position ticks: per-token last emit time
    _pos_emit_times: Dict[int, float] = {}

    def _emit_pos_tick(self, token: int, ltp: float):
        """Broadcast a position tick via SocketIO. Throttled to ~2/sec per token."""
        import time
        now = time.time()
        last = MaruthiTicker._pos_emit_times.get(token, 0)
        if now - last < 0.5:
            return
        MaruthiTicker._pos_emit_times[token] = now

        try:
            from app import socketio
            tsym = self._pos_token_map.get(token, '')
            socketio.emit('pos_tick', {
                'token': token,
                'tradingsymbol': tsym,
                'ltp': round(ltp, 2),
            })
        except Exception:
            pass

    def _forward_to_nas(self, tick):
        """Forward ticks to NAS ticker for NIFTY candle aggregation."""
        try:
            from services.nas_ticker import get_nas_ticker
            nas = get_nas_ticker()
            if nas and nas.is_running:
                nas._on_ticks(None, [tick])
        except Exception:
            pass

    def _forward_to_bnf(self, tick):
        """Forward BANKNIFTY ticks for live spot display on BNF dashboard."""
        try:
            from services.bnf_executor import get_bnf_executor
            bnf = get_bnf_executor()
            if bnf:
                bnf._last_live_spot = tick.get('last_price', 0)
        except Exception:
            pass

    def _on_connect(self, ws, response):
        """Handle WebSocket connect."""
        logger.info(f"[MaruthiTicker] Connected to KiteTicker")
        self.is_connected = True

        tokens = []
        token = self._get_instrument_token()
        if token:
            tokens.append(token)
            logger.info(f"[MaruthiTicker] Subscribed to {self.symbol} (token={token})")
        else:
            logger.error(f"[MaruthiTicker] No token for {self.symbol} — cannot subscribe")

        # Also subscribe NIFTY 50 for NAS strategy
        NIFTY_TOKEN = 256265
        tokens.append(NIFTY_TOKEN)
        logger.info(f"[MaruthiTicker] Also subscribing NIFTY 50 (token={NIFTY_TOKEN}) for NAS")

        # Also subscribe BANKNIFTY for BNF strategy (live spot)
        BANKNIFTY_TOKEN = 260105
        tokens.append(BANKNIFTY_TOKEN)
        logger.info(f"[MaruthiTicker] Also subscribing BANKNIFTY (token={BANKNIFTY_TOKEN}) for BNF")

        if tokens:
            ws.subscribe(tokens)
            ws.set_mode(ws.MODE_FULL, tokens)

    def _on_close(self, ws, code, reason):
        """Handle WebSocket close."""
        logger.warning(f"[MaruthiTicker] Disconnected: {code} - {reason}")
        self.is_connected = False

    def _on_error(self, ws, code, reason):
        """Handle WebSocket error."""
        logger.error(f"[MaruthiTicker] Error: {code} - {reason}")

    def _on_reconnect(self, ws, attempts):
        """Handle reconnect attempts."""
        logger.info(f"[MaruthiTicker] Reconnecting... attempt {attempts}")

    def _on_order_update(self, ws, data):
        """
        Handle real-time order update pushed by Zerodha via WebSocket.

        KiteTicker sends order status changes as text messages with type='order'.
        This fires instantly when an order is placed, modified, completed, cancelled, or rejected.
        No polling needed — this is the authoritative source for fill detection.

        data dict keys: order_id, status, tradingsymbol, transaction_type,
                        average_price, filled_quantity, status_message, etc.
        """
        try:
            order_id = str(data.get('order_id', ''))
            status = data.get('status', '')
            tradingsymbol = data.get('tradingsymbol', '')
            txn_type = data.get('transaction_type', '')

            logger.info(
                f"[MaruthiTicker] Order update: {txn_type} {tradingsymbol} "
                f"status={status} order_id={order_id}"
            )

            # Only act on terminal states
            if status not in ('COMPLETE', 'CANCELLED', 'REJECTED'):
                return

            # Match against our pending positions
            from services.maruthi_db import get_maruthi_db
            db = get_maruthi_db()
            pending = db.get_pending_positions()

            for pos in pending:
                if str(pos.get('kite_order_id', '')) != order_id:
                    continue

                if status == 'COMPLETE':
                    fill_price = data.get('average_price', pos.get('trigger_price', 0))
                    db.activate_position(pos['id'], fill_price)
                    MaruthiTicker._pending_cache = None  # Invalidate cache

                    logger.info(
                        f"[MaruthiTicker] ORDER FILLED: {txn_type} {tradingsymbol} "
                        f"@ {fill_price} (order_id={order_id})"
                    )

                    # Post-fill actions: protective option, hard SL update
                    from services.maruthi_executor import get_maruthi_executor
                    executor = get_maruthi_executor(self.config)
                    executor.on_trigger_fill(pos, fill_price=fill_price)

                elif status in ('CANCELLED', 'REJECTED'):
                    db.cancel_position(pos['id'])
                    MaruthiTicker._pending_cache = None

                    logger.warning(
                        f"[MaruthiTicker] ORDER {status}: {tradingsymbol} "
                        f"(order_id={order_id}, reason={data.get('status_message', '')})"
                    )
                break

        except Exception as e:
            logger.error(f"[MaruthiTicker] Order update handler error: {e}", exc_info=True)

    def _on_candle_close(self, candle: dict):
        """
        Fired when a 30-min candle closes.
        Runs the full strategy check.
        """
        logger.info(f"[MaruthiTicker] 30-min candle closed: {candle['date']} close={candle['close']:.1f}")

        try:
            from services.maruthi_executor import get_maruthi_executor

            executor = get_maruthi_executor(self.config)

            # Get DataFrame with all completed candles
            df = self.aggregator.get_dataframe()
            if len(df) < 20:
                logger.warning(f"[MaruthiTicker] Only {len(df)} candles — need ~20 for SuperTrend")
                return

            # Run strategy
            results = executor.run_candle_check(df)
            if results:
                logger.info(f"[MaruthiTicker] Actions: {results}")

            # Check pending trigger fills
            fills = executor.run_verify_triggers(self._last_ltp)
            if fills:
                logger.info(f"[MaruthiTicker] Trigger fills: {fills}")

        except Exception as e:
            logger.error(f"[MaruthiTicker] Strategy execution error: {e}", exc_info=True)

    def start(self):
        """Start the WebSocket ticker (background thread)."""
        if self.is_running:
            logger.warning("[MaruthiTicker] Already running")
            return

        access_token = get_access_token()
        if not access_token:
            logger.error("[MaruthiTicker] No access token — run TOTP login first")
            return

        if not KITE_API_KEY:
            logger.error("[MaruthiTicker] KITE_API_KEY not set")
            return

        # Load historical candles for SuperTrend warmup
        self._load_historical_candles()

        # Run catch-up signal check on historical data to detect flips
        # that happened while ticker was offline (e.g., overnight gap open)
        self._catchup_signal_check()

        # Initialize KiteTicker
        self.kws = KiteTicker(KITE_API_KEY, access_token)
        self.kws.on_ticks = self._on_ticks
        self.kws.on_connect = self._on_connect
        self.kws.on_close = self._on_close
        self.kws.on_error = self._on_error
        self.kws.on_reconnect = self._on_reconnect
        self.kws.on_order_update = self._on_order_update

        self.is_running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("[MaruthiTicker] Started in background thread")

    def _run(self):
        """Internal WebSocket run loop."""
        try:
            self.kws.connect(threaded=False)
        except Exception as e:
            logger.error(f"[MaruthiTicker] Connection error: {e}")
        finally:
            self.is_running = False
            self.is_connected = False

    def stop(self):
        """Stop the WebSocket ticker."""
        self.is_running = False
        if self.kws:
            try:
                self.kws.close()
            except Exception:
                pass
        self.is_connected = False
        logger.info("[MaruthiTicker] Stopped")

    def restart(self):
        """Restart the ticker (e.g., after daily TOTP re-login)."""
        self.stop()
        import time
        time.sleep(2)
        self.start()

    def get_status(self) -> dict:
        """Get ticker status for dashboard."""
        return {
            'is_running': self.is_running,
            'is_connected': self.is_connected,
            'symbol': self.symbol,
            'last_ltp': self._last_ltp,
            'last_tick_time': self._last_tick_time.isoformat() if self._last_tick_time else None,
            'completed_candles': len(self.aggregator.completed_candles),
            'current_candle': self.aggregator.current_candle,
            'instrument_token': self._instrument_token,
        }


# ---- Singleton ----
_maruthi_ticker: Optional[MaruthiTicker] = None
_ticker_lock = threading.Lock()


def get_maruthi_ticker(config: dict = None) -> MaruthiTicker:
    """Get or create the global MaruthiTicker instance."""
    global _maruthi_ticker
    with _ticker_lock:
        if _maruthi_ticker is None:
            _maruthi_ticker = MaruthiTicker(config)
        return _maruthi_ticker


def stop_maruthi_ticker():
    """Stop and clear the global ticker."""
    global _maruthi_ticker
    with _ticker_lock:
        if _maruthi_ticker:
            _maruthi_ticker.stop()
            _maruthi_ticker = None
