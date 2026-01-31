"""
Kite Ticker WebSocket Service
=============================

Real-time price streaming using Zerodha Kite Ticker WebSocket.
Broadcasts live prices to connected clients via Flask-SocketIO.
"""

import os
import json
import logging
import threading
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from pathlib import Path

from kiteconnect import KiteTicker
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
KITE_API_KEY = os.getenv("KITE_API_KEY", "")

# Instrument token cache file
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "backtest_data"
INSTRUMENT_CACHE = DATA_DIR / "instrument_tokens.json"


class KiteTickerService:
    """
    Manages Kite Ticker WebSocket connection for real-time price streaming.

    Usage:
        ticker_service = KiteTickerService(access_token)
        ticker_service.set_callback(on_tick_callback)
        ticker_service.subscribe(['RELIANCE', 'TCS', 'INFY'])
        ticker_service.start()
    """

    def __init__(self, access_token: str):
        """
        Initialize the ticker service.

        Args:
            access_token: Valid Kite access token
        """
        self.access_token = access_token
        self.kws: Optional[KiteTicker] = None
        self.is_connected = False
        self.subscribed_tokens: List[int] = []
        self.symbol_to_token: Dict[str, int] = {}
        self.token_to_symbol: Dict[int, str] = {}
        self.on_tick_callback: Optional[Callable] = None
        self.on_connect_callback: Optional[Callable] = None
        self.on_disconnect_callback: Optional[Callable] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Load instrument cache
        self._load_instrument_cache()

    def _load_instrument_cache(self):
        """Load instrument token mappings from cache file."""
        try:
            if INSTRUMENT_CACHE.exists():
                data = json.loads(INSTRUMENT_CACHE.read_text())
                self.symbol_to_token = {k: int(v) for k, v in data.get('symbol_to_token', {}).items()}
                self.token_to_symbol = {int(k): v for k, v in data.get('token_to_symbol', {}).items()}
                logger.info(f"Loaded {len(self.symbol_to_token)} instrument tokens from cache")
        except Exception as e:
            logger.warning(f"Failed to load instrument cache: {e}")

    def _save_instrument_cache(self):
        """Save instrument token mappings to cache file."""
        try:
            DATA_DIR.mkdir(exist_ok=True)
            data = {
                'symbol_to_token': self.symbol_to_token,
                'token_to_symbol': {str(k): v for k, v in self.token_to_symbol.items()},
                'updated_at': datetime.now().isoformat()
            }
            INSTRUMENT_CACHE.write_text(json.dumps(data, indent=2))
            logger.info("Instrument cache saved")
        except Exception as e:
            logger.error(f"Failed to save instrument cache: {e}")

    def update_instruments(self, kite) -> bool:
        """
        Fetch and cache instrument tokens from Kite API.

        Args:
            kite: KiteConnect instance with valid access token

        Returns:
            True if successful, False otherwise
        """
        try:
            # Fetch NSE instruments
            instruments = kite.instruments("NSE")

            # Build symbol to token mapping for equity
            for inst in instruments:
                if inst['segment'] == 'NSE' and inst['instrument_type'] == 'EQ':
                    symbol = inst['tradingsymbol']
                    token = inst['instrument_token']
                    self.symbol_to_token[symbol] = token
                    self.token_to_symbol[token] = symbol

            self._save_instrument_cache()
            logger.info(f"Updated {len(self.symbol_to_token)} NSE equity instruments")
            return True

        except Exception as e:
            logger.error(f"Failed to update instruments: {e}")
            return False

    def get_token(self, symbol: str) -> Optional[int]:
        """Get instrument token for a symbol."""
        return self.symbol_to_token.get(symbol)

    def get_symbol(self, token: int) -> Optional[str]:
        """Get symbol for an instrument token."""
        return self.token_to_symbol.get(token)

    def set_callback(self, on_tick: Callable[[Dict[str, Any]], None]):
        """
        Set callback for tick data.

        Args:
            on_tick: Callback function that receives tick data dict
                     {symbol, ltp, change, change_pct, volume, timestamp}
        """
        self.on_tick_callback = on_tick

    def set_connect_callback(self, on_connect: Callable[[], None]):
        """Set callback for connection established."""
        self.on_connect_callback = on_connect

    def set_disconnect_callback(self, on_disconnect: Callable[[int, str], None]):
        """Set callback for disconnection."""
        self.on_disconnect_callback = on_disconnect

    def _on_ticks(self, ws, ticks: List[Dict]):
        """Internal tick handler - transforms and broadcasts."""
        try:
            for tick in ticks:
                token = tick.get('instrument_token')
                symbol = self.get_symbol(token)

                if not symbol:
                    continue

                # Transform tick data
                tick_data = {
                    'symbol': symbol,
                    'ltp': tick.get('last_price', 0),
                    'open': tick.get('ohlc', {}).get('open', 0),
                    'high': tick.get('ohlc', {}).get('high', 0),
                    'low': tick.get('ohlc', {}).get('low', 0),
                    'close': tick.get('ohlc', {}).get('close', 0),
                    'change': tick.get('change', 0),
                    'change_pct': tick.get('change', 0) / tick.get('ohlc', {}).get('close', 1) * 100 if tick.get('ohlc', {}).get('close') else 0,
                    'volume': tick.get('volume_traded', 0),
                    'buy_quantity': tick.get('total_buy_quantity', 0),
                    'sell_quantity': tick.get('total_sell_quantity', 0),
                    'last_trade_time': tick.get('last_trade_time').isoformat() if tick.get('last_trade_time') else None,
                    'timestamp': datetime.now().isoformat()
                }

                # Call user callback
                if self.on_tick_callback:
                    self.on_tick_callback(tick_data)

        except Exception as e:
            logger.error(f"Error processing ticks: {e}")

    def _on_connect(self, ws, response):
        """Internal connect handler."""
        logger.info("Kite Ticker connected")
        self.is_connected = True

        # Subscribe to tokens if any pending
        if self.subscribed_tokens:
            ws.subscribe(self.subscribed_tokens)
            ws.set_mode(ws.MODE_FULL, self.subscribed_tokens)
            logger.info(f"Subscribed to {len(self.subscribed_tokens)} tokens")

        if self.on_connect_callback:
            self.on_connect_callback()

    def _on_close(self, ws, code, reason):
        """Internal close handler."""
        logger.warning(f"Kite Ticker closed: {code} - {reason}")
        self.is_connected = False

        if self.on_disconnect_callback:
            self.on_disconnect_callback(code, reason)

    def _on_error(self, ws, code, reason):
        """Internal error handler."""
        logger.error(f"Kite Ticker error: {code} - {reason}")

    def _on_reconnect(self, ws, attempts_count):
        """Internal reconnect handler."""
        logger.info(f"Kite Ticker reconnecting... attempt {attempts_count}")

    def _on_noreconnect(self, ws):
        """Internal no-reconnect handler."""
        logger.error("Kite Ticker: max reconnect attempts exhausted")
        self.is_connected = False

    def subscribe(self, symbols: List[str]) -> List[str]:
        """
        Subscribe to live prices for given symbols.

        Args:
            symbols: List of NSE trading symbols

        Returns:
            List of successfully subscribed symbols
        """
        subscribed = []
        new_tokens = []

        for symbol in symbols:
            token = self.get_token(symbol)
            if token:
                if token not in self.subscribed_tokens:
                    self.subscribed_tokens.append(token)
                    new_tokens.append(token)
                subscribed.append(symbol)
            else:
                logger.warning(f"No instrument token found for {symbol}")

        # If already connected, subscribe immediately
        if self.is_connected and self.kws and new_tokens:
            self.kws.subscribe(new_tokens)
            self.kws.set_mode(self.kws.MODE_FULL, new_tokens)
            logger.info(f"Subscribed to {len(new_tokens)} new tokens")

        return subscribed

    def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from live prices for given symbols."""
        tokens_to_remove = []

        for symbol in symbols:
            token = self.get_token(symbol)
            if token and token in self.subscribed_tokens:
                self.subscribed_tokens.remove(token)
                tokens_to_remove.append(token)

        if self.is_connected and self.kws and tokens_to_remove:
            self.kws.unsubscribe(tokens_to_remove)
            logger.info(f"Unsubscribed from {len(tokens_to_remove)} tokens")

    def start(self, threaded: bool = True):
        """
        Start the WebSocket connection.

        Args:
            threaded: Run in background thread (default True)
        """
        if not KITE_API_KEY:
            raise RuntimeError("KITE_API_KEY not configured")

        if not self.access_token:
            raise RuntimeError("No access token provided")

        # Initialize KiteTicker
        self.kws = KiteTicker(KITE_API_KEY, self.access_token)

        # Assign callbacks
        self.kws.on_ticks = self._on_ticks
        self.kws.on_connect = self._on_connect
        self.kws.on_close = self._on_close
        self.kws.on_error = self._on_error
        self.kws.on_reconnect = self._on_reconnect
        self.kws.on_noreconnect = self._on_noreconnect

        self._stop_event.clear()

        if threaded:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            logger.info("Kite Ticker started in background thread")
        else:
            self._run()

    def _run(self):
        """Internal run loop."""
        try:
            self.kws.connect(threaded=False)
        except Exception as e:
            logger.error(f"Kite Ticker connection error: {e}")
            self.is_connected = False

    def stop(self):
        """Stop the WebSocket connection."""
        self._stop_event.set()

        if self.kws:
            try:
                self.kws.close()
            except Exception as e:
                logger.warning(f"Error closing Kite Ticker: {e}")

        self.is_connected = False
        logger.info("Kite Ticker stopped")

    def is_running(self) -> bool:
        """Check if ticker is running and connected."""
        return self.is_connected


# Global ticker instance (singleton)
_ticker_instance: Optional[KiteTickerService] = None
_ticker_lock = threading.Lock()


def get_ticker_service(access_token: str = None) -> Optional[KiteTickerService]:
    """
    Get or create the global ticker service instance.

    Args:
        access_token: Access token (required on first call)

    Returns:
        KiteTickerService instance or None if not initialized
    """
    global _ticker_instance

    with _ticker_lock:
        if _ticker_instance is None and access_token:
            _ticker_instance = KiteTickerService(access_token)
        return _ticker_instance


def stop_ticker_service():
    """Stop and clear the global ticker service."""
    global _ticker_instance

    with _ticker_lock:
        if _ticker_instance:
            _ticker_instance.stop()
            _ticker_instance = None