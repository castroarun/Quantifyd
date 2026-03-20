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

    def _on_ticks(self, ws, ticks):
        """Handle incoming ticks from KiteTicker."""
        for tick in ticks:
            if tick.get('instrument_token') == self._instrument_token:
                ltp = tick.get('last_price', 0)
                volume = tick.get('volume_traded', 0)
                timestamp = tick.get('exchange_timestamp') or datetime.now()

                if ltp > 0:
                    self._last_ltp = ltp
                    self.aggregator.process_tick(ltp, volume, timestamp)

                    # Real-time hard SL check on every tick
                    self._check_hard_sl_tick(ltp)

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

            from services.maruthi_executor import MaruthiExecutor
            executor = MaruthiExecutor(config=self.config)

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

    def _on_connect(self, ws, response):
        """Handle WebSocket connect."""
        logger.info(f"[MaruthiTicker] Connected to KiteTicker")
        self.is_connected = True

        token = self._get_instrument_token()
        if token:
            ws.subscribe([token])
            ws.set_mode(ws.MODE_FULL, [token])
            logger.info(f"[MaruthiTicker] Subscribed to {self.symbol} (token={token})")
        else:
            logger.error(f"[MaruthiTicker] No token for {self.symbol} — cannot subscribe")

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

    def _on_candle_close(self, candle: dict):
        """
        Fired when a 30-min candle closes.
        Runs the full strategy check.
        """
        logger.info(f"[MaruthiTicker] 30-min candle closed: {candle['date']} close={candle['close']:.1f}")

        try:
            from services.maruthi_executor import MaruthiExecutor

            executor = MaruthiExecutor(config=self.config)

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

        # Initialize KiteTicker
        self.kws = KiteTicker(KITE_API_KEY, access_token)
        self.kws.on_ticks = self._on_ticks
        self.kws.on_connect = self._on_connect
        self.kws.on_close = self._on_close
        self.kws.on_error = self._on_error
        self.kws.on_reconnect = self._on_reconnect

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
