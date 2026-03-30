"""
NAS — Nifty ATR Strangle — WebSocket Ticker
=============================================

Receives NIFTY 50 ticks forwarded from Maruthi's KiteTicker (avoids
Twisted reactor conflict — only one KiteTicker per process) and aggregates
them into 5-minute OHLCV candles. On each candle close:
  1. Computes ATR squeeze indicators
  2. Checks exits on active positions (combined SL, profit target)
  3. Checks adjustments (premium 2x / 0.5x)
  4. Fires entry signal if squeeze detected

Also monitors option leg premiums in real-time for instant SL triggers
(premium doubles = combined SL breach detected between candles).

Architecture:
    MaruthiTicker._on_ticks() → forward NIFTY ticks → NasTicker._on_ticks()
                                     ↓
                              CandleAggregator (5-min OHLCV)
                                     ↓ on candle close
                              NasScanner.scan() → NasExecutor.run_scan()

    MaruthiTicker._on_ticks() → forward option ticks → _check_premium_tick()
                                     ↓ cross-leg imbalance
                              NasExecutor.execute_adjustment(ROLL_OUT / ROLL_IN)
"""

import logging
import threading
import pandas as pd
from datetime import datetime, date, time as dtime, timedelta
from typing import Callable, Dict, List, Optional
from collections import deque

from services.kite_service import get_access_token, KITE_API_KEY

logger = logging.getLogger(__name__)

# ─── Constants ──────────────────────────────────────────────

MARKET_OPEN = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)
CANDLE_MINUTES = 5

# NIFTY 50 index instrument token (NSE)
NIFTY_INSTRUMENT_TOKEN = 256265

# Build 5-min candle boundaries: 09:15, 09:20, 09:25, ... 15:25, 15:30
CANDLE_BOUNDARIES = []
t = datetime(2000, 1, 1, 9, 15)
end = datetime(2000, 1, 1, 15, 30)
while t < end:
    CANDLE_BOUNDARIES.append(t.time())
    t += timedelta(minutes=CANDLE_MINUTES)
CANDLE_BOUNDARIES.append(MARKET_CLOSE)


# ─── 5-Min Candle Aggregator ────────────────────────────────

class NiftyCandleAggregator:
    """
    Aggregates NIFTY 50 ticks into 5-minute OHLCV candles.
    Fires callback on each candle close for ATR squeeze computation.
    """

    def __init__(self, max_candles: int = 500):
        self.max_candles = max_candles
        self.completed_candles: deque = deque(maxlen=max_candles)
        self.current_candle: Optional[dict] = None
        self.current_candle_end: Optional[dtime] = None
        self.on_candle_close: Optional[Callable] = None
        self._lock = threading.Lock()

    def _get_candle_boundary(self, now: datetime) -> tuple:
        """Get the start and end time of the current 5-min candle."""
        current_time = now.time()

        for i in range(len(CANDLE_BOUNDARIES) - 1):
            if current_time >= CANDLE_BOUNDARIES[i] and current_time < CANDLE_BOUNDARIES[i + 1]:
                candle_start = now.replace(
                    hour=CANDLE_BOUNDARIES[i].hour,
                    minute=CANDLE_BOUNDARIES[i].minute,
                    second=0, microsecond=0
                )
                return candle_start, CANDLE_BOUNDARIES[i + 1]

        return None, None

    def process_tick(self, ltp: float, volume: int = 0, timestamp: datetime = None):
        """Process a single NIFTY tick and update the current 5-min candle."""
        if timestamp is None:
            timestamp = datetime.now()

        if timestamp.time() < MARKET_OPEN or timestamp.time() >= MARKET_CLOSE:
            return

        with self._lock:
            candle_start, candle_end = self._get_candle_boundary(timestamp)
            if candle_start is None:
                return

            # Check if we've crossed into a new candle period
            if self.current_candle_end is not None and candle_end != self.current_candle_end:
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
                self.current_candle['high'] = max(self.current_candle['high'], ltp)
                self.current_candle['low'] = min(self.current_candle['low'], ltp)
                self.current_candle['close'] = ltp
                self.current_candle['volume'] = volume

    def _close_current_candle(self):
        """Finalize the current candle and fire callback."""
        candle = self.current_candle.copy()
        self.completed_candles.append(candle)
        logger.info(
            f"[NAS] 5-min candle closed: {candle['date']} "
            f"O={candle['open']:.1f} H={candle['high']:.1f} "
            f"L={candle['low']:.1f} C={candle['close']:.1f}"
        )

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
            logger.error(f"[NAS] Candle close callback error: {e}", exc_info=True)

    def force_close(self):
        """Force-close current candle (called at market close 15:30)."""
        with self._lock:
            if self.current_candle:
                self._close_current_candle()

    def get_dataframe(self) -> pd.DataFrame:
        """Get all candles as a DataFrame (for ATR computation)."""
        with self._lock:
            candles = list(self.completed_candles)
            if self.current_candle:
                candles.append(self.current_candle)

        if not candles:
            return pd.DataFrame(columns=['date', 'open', 'high', 'low', 'close', 'volume'])

        return pd.DataFrame(candles)

    def load_historical(self, candles: List[dict]):
        """Preload historical candles (seeds ATR with enough history)."""
        with self._lock:
            for c in candles:
                self.completed_candles.append(c)
        logger.info(f"[NAS] Loaded {len(candles)} historical 5-min candles")


# ─── NAS Ticker ─────────────────────────────────────────────

class NasTicker:
    """
    Manages KiteTicker WebSocket for the NAS strategy.

    Subscribes to:
      1. NIFTY 50 index (token 256265) — builds 5-min candles, computes ATR
      2. Active option leg tokens — monitors premiums for instant SL/adjustment

    On each 5-min candle close:
      - Runs full NAS scan (ATR squeeze → entry/exit/adjustment)

    On every option tick:
      - Checks if losing leg premium has doubled (combined SL)
    """

    def __init__(self, config: dict = None):
        from config import NAS_DEFAULTS
        self.config = config or NAS_DEFAULTS.copy()
        self.aggregator = NiftyCandleAggregator(max_candles=500)
        self.is_connected = False
        self.is_running = False
        self._last_ltp: float = 0.0
        self._lock = threading.Lock()

        # Option leg monitoring (NAS OTM)
        self._option_tokens: Dict[int, dict] = {}  # token → {tradingsymbol, entry_premium, leg}
        self._option_ltps: Dict[int, float] = {}   # token → last price
        self._sl_triggered = False  # prevent duplicate SL fires
        self._adj_triggered = False  # prevent duplicate adjustments
        self._adj_confirm: Dict[int, int] = {}  # token → consecutive trigger tick count
        self._adj_next_direction = 'OUT'  # alternates OUT/IN, resets on new strangle

        # NAS ATM option leg monitoring (separate SL tracking)
        self._atm_option_tokens: Dict[int, dict] = {}  # token → {tradingsymbol, sl_price, position_id, ...}
        self._atm_option_ltps: Dict[int, float] = {}   # token → last price
        self._atm_sl_processing = False  # prevent concurrent SL handling

        # NAS ATM2 option leg monitoring (separate SL tracking)
        self._atm2_option_tokens: Dict[int, dict] = {}
        self._atm2_option_ltps: Dict[int, float] = {}
        self._atm2_sl_processing = False

        # Set candle close callback
        self.aggregator.on_candle_close = self._on_candle_close

    # ─── Historical Seeding ──────────────────────────────────

    def _load_historical_candles(self):
        """Load recent 5-min NIFTY candles from Kite to seed ATR computation."""
        try:
            from services.kite_service import get_kite
            kite = get_kite()

            to_date = datetime.now()
            from_date = to_date - timedelta(days=5)  # ~375 candles (5 trading days)

            all_candles = []
            chunk_start = from_date
            while chunk_start < to_date:
                chunk_end = min(chunk_start + timedelta(days=7), to_date)
                data = kite.historical_data(
                    instrument_token=NIFTY_INSTRUMENT_TOKEN,
                    from_date=chunk_start,
                    to_date=chunk_end,
                    interval='5minute'
                )
                for d in data:
                    all_candles.append({
                        'date': d['date'],
                        'open': d['open'],
                        'high': d['high'],
                        'low': d['low'],
                        'close': d['close'],
                        'volume': d.get('volume', 0),
                    })
                chunk_start = chunk_end
                import time
                time.sleep(0.35)

            self.aggregator.load_historical(all_candles)
            if all_candles:
                self._last_ltp = all_candles[-1]['close']

        except Exception as e:
            logger.error(f"[NAS] Failed to load historical candles: {e}", exc_info=True)

    def _load_daily_candles(self):
        """Load daily NIFTY candles for daily ATR (strike computation)."""
        try:
            from services.kite_service import get_kite
            kite = get_kite()

            to_date = datetime.now()
            from_date = to_date - timedelta(days=30)
            data = kite.historical_data(
                instrument_token=NIFTY_INSTRUMENT_TOKEN,
                from_date=from_date,
                to_date=to_date,
                interval='day'
            )
            self._daily_candles = pd.DataFrame([{
                'date': d['date'],
                'open': d['open'],
                'high': d['high'],
                'low': d['low'],
                'close': d['close'],
                'volume': d.get('volume', 0),
            } for d in data])
            logger.info(f"[NAS] Loaded {len(self._daily_candles)} daily candles for strike computation")
        except Exception as e:
            logger.error(f"[NAS] Failed to load daily candles: {e}")
            self._daily_candles = pd.DataFrame()

    # ─── Option Leg Monitoring ───────────────────────────────

    def _get_maruthi_kws(self):
        """Get Maruthi ticker's KiteTicker WebSocket for subscribing option tokens."""
        try:
            from services.maruthi_ticker import get_maruthi_ticker
            mt = get_maruthi_ticker()
            if mt and mt.kws and mt.is_connected:
                return mt.kws
        except Exception:
            pass
        return None

    def subscribe_option_legs(self, positions: List[dict]):
        """
        Subscribe to option leg tokens for real-time premium monitoring.

        Uses Maruthi ticker's WebSocket to subscribe (avoids Twisted conflict).
        Called after entry or adjustment — updates the subscription.

        Args:
            positions: list of active position dicts from NasDB
                       (must have 'tradingsymbol', 'entry_price', 'leg', 'instrument_type')
        """
        kws = self._get_maruthi_kws()
        if not kws:
            logger.warning("[NAS] Cannot subscribe option legs — Maruthi ticker not connected")
            return

        with self._lock:
            # Detect new strangle → reset alternation direction
            old_strangle_ids = set(
                info.get('strangle_id') for info in self._option_tokens.values()
            )
            new_strangle_ids = set(
                pos.get('strangle_id') for pos in positions
            ) if positions else set()
            if new_strangle_ids and new_strangle_ids != old_strangle_ids:
                # New strangle entered — reset alternation to OUT
                self._adj_next_direction = 'OUT'
                logger.info(f"[NAS] New strangle detected — reset adj direction to OUT")

            # Clear old option tokens
            old_tokens = set(self._option_tokens.keys())
            self._option_tokens.clear()
            self._option_ltps.clear()
            self._sl_triggered = False

            if not positions:
                # Unsubscribe old tokens
                if old_tokens:
                    kws.unsubscribe(list(old_tokens))
                    logger.info(f"[NAS] Unsubscribed {len(old_tokens)} option tokens")
                return

            # Resolve instrument tokens for active option legs
            new_tokens = []
            for pos in positions:
                tsym = pos.get('tradingsymbol', '')
                if not tsym:
                    continue

                token = self._resolve_option_token(tsym)
                if token:
                    self._option_tokens[token] = {
                        'tradingsymbol': tsym,
                        'entry_premium': pos.get('entry_price', 0),
                        'leg': pos.get('leg', ''),
                        'instrument_type': pos.get('instrument_type', ''),
                        'position_id': pos.get('id'),
                        'strangle_id': pos.get('strangle_id'),
                    }
                    new_tokens.append(token)

            # Unsubscribe tokens no longer needed
            tokens_to_remove = old_tokens - set(new_tokens)
            if tokens_to_remove:
                kws.unsubscribe(list(tokens_to_remove))

            # Subscribe new tokens via Maruthi's WebSocket
            tokens_to_add = set(new_tokens) - old_tokens
            if tokens_to_add:
                kws.subscribe(list(tokens_to_add))
                kws.set_mode(kws.MODE_LTP, list(tokens_to_add))

            logger.info(f"[NAS] Monitoring {len(new_tokens)} option legs: "
                        f"{[t['tradingsymbol'] for t in self._option_tokens.values()]}")

    def _resolve_option_token(self, tradingsymbol: str) -> Optional[int]:
        """Resolve an NFO tradingsymbol to instrument token."""
        try:
            from services.kite_service import get_kite
            kite = get_kite()
            key = f'NFO:{tradingsymbol}'
            quote = kite.quote([key])
            data = quote.get(key, {})
            return data.get('instrument_token')
        except Exception as e:
            logger.warning(f"[NAS] Could not resolve token for {tradingsymbol}: {e}")
            return None

    # ─── Tick Handlers ───────────────────────────────────────

    def _on_ticks(self, ws, ticks):
        """Handle incoming ticks from KiteTicker."""
        for tick in ticks:
            token = tick.get('instrument_token')
            ltp = tick.get('last_price', 0)

            if ltp <= 0:
                continue

            if token == NIFTY_INSTRUMENT_TOKEN:
                # NIFTY spot tick → build 5-min candle
                self._last_ltp = ltp
                volume = tick.get('volume_traded', 0)
                timestamp = tick.get('exchange_timestamp') or datetime.now()
                self.aggregator.process_tick(ltp, volume, timestamp)

            elif token in self._option_tokens:
                # Option leg tick → check premium SL (NAS OTM)
                self._option_ltps[token] = ltp
                self._check_premium_tick(token, ltp)

            if token in self._atm_option_tokens:
                # NAS ATM option leg tick → check per-leg SL
                self._atm_option_ltps[token] = ltp
                self._check_atm_premium_tick(token, ltp)

            if token in self._atm2_option_tokens:
                # NAS ATM2 option leg tick → check per-leg SL
                self._atm2_option_ltps[token] = ltp
                self._check_atm2_premium_tick(token, ltp)

    def _check_premium_tick(self, token: int, ltp: float):
        """
        Instant tick-level premium check for option legs.

        1. Combined SL: if total buyback > 2x total entry → emergency exit
        2. Cross-leg adjustment: if one leg's premium >= 2x the other leg's
           premium for 2 consecutive ticks → alternating roll OUT/IN with
           boundary guards (target premium must be 4-24)
        """
        if self._sl_triggered:
            return

        info = self._option_tokens.get(token)
        if not info:
            return

        # Need at least 2 legs to compare
        if len(self._option_tokens) < 2:
            return

        cfg = self.config
        imbalance_trigger = cfg.get('premium_double_trigger', 2.0)
        adj_min_prem = cfg.get('adj_min_premium', 4.0)
        adj_max_prem = cfg.get('adj_max_premium', 24.0)

        # Gather current premiums for all legs
        leg_premiums = {}  # token → (ltp, info)
        for t, leg_info in self._option_tokens.items():
            leg_ltp = self._option_ltps.get(t, leg_info['entry_premium'])
            leg_premiums[t] = (leg_ltp, leg_info)

        # Cross-leg adjustment: compare legs against each other
        if self._adj_triggered:
            return  # Already processing an adjustment

        # Find the most expensive and cheapest legs
        sorted_legs = sorted(leg_premiums.items(), key=lambda x: x[1][0])
        cheap_token, (cheap_ltp, cheap_info) = sorted_legs[0]
        expensive_token, (expensive_ltp, expensive_info) = sorted_legs[-1]

        # Check if expensive leg >= imbalance_trigger × cheap leg
        if cheap_ltp <= 0:
            return

        cross_ratio = expensive_ltp / cheap_ltp
        if cross_ratio >= imbalance_trigger:
            # Use a single confirmation key for cross-leg imbalance
            confirm_key = 'cross_leg'
            count = self._adj_confirm.get(confirm_key, 0) + 1
            self._adj_confirm[confirm_key] = count
            if count >= 2:
                # Confirmed — determine direction with alternation + boundary checks
                self._adj_triggered = True
                self._adj_confirm[confirm_key] = 0

                # Determine adjustment: direction + boundary logic
                adj_leg_info, adj_ltp, action, target_prem, close_both = \
                    self._resolve_adjustment_direction(
                        expensive_info, expensive_ltp,
                        cheap_info, cheap_ltp,
                        adj_min_prem, adj_max_prem,
                    )

                if close_both:
                    # Both directions fail boundary — exit strangle
                    logger.warning(
                        f"[NAS] ADJ BOUNDARY FAIL — both directions out of "
                        f"[{adj_min_prem}-{adj_max_prem}] range. "
                        f"Expensive={expensive_ltp:.1f}, Cheap={cheap_ltp:.1f}. "
                        f"Closing both legs."
                    )
                    threading.Thread(
                        target=self._fire_emergency_exit,
                        args=('adj_boundary_exit',),
                        daemon=True
                    ).start()
                else:
                    logger.info(
                        f"[NAS] TICK ADJ! {action} (next_dir was {self._adj_next_direction}): "
                        f"{adj_leg_info['tradingsymbol']} @{adj_ltp:.1f} → "
                        f"target={target_prem:.1f} | "
                        f"Expensive={expensive_info['tradingsymbol']}@{expensive_ltp:.1f} "
                        f"Cheap={cheap_info['tradingsymbol']}@{cheap_ltp:.1f}"
                    )
                    threading.Thread(
                        target=self._fire_tick_adjustment,
                        args=(adj_leg_info, adj_ltp, action, target_prem),
                        daemon=True
                    ).start()
        else:
            # Reset confirmation counter if ratio is back to normal
            self._adj_confirm['cross_leg'] = 0

    def _resolve_adjustment_direction(
        self,
        expensive_info: dict, expensive_ltp: float,
        cheap_info: dict, cheap_ltp: float,
        min_prem: float, max_prem: float,
    ):
        """
        Determine which leg to adjust and target premium, applying:
        - Alternation: OUT → IN → OUT → IN ...
        - Boundary guards: target must be [min_prem, max_prem]
        - Flip on boundary fail, close both if both fail
        - Alternation advances only on non-overridden adjustments

        Returns: (leg_info, leg_ltp, action, target_prem, close_both)
        """
        direction = self._adj_next_direction  # 'OUT' or 'IN'

        # OUT: adjust expensive leg → target = cheap leg's premium
        # IN:  adjust cheap leg    → target = expensive leg's premium
        def _params_for(d):
            if d == 'OUT':
                return expensive_info, expensive_ltp, 'ROLL_OUT', cheap_ltp
            else:
                return cheap_info, cheap_ltp, 'ROLL_IN', expensive_ltp

        leg_info, leg_ltp, action, target = _params_for(direction)

        # Boundary check on target premium
        if min_prem <= target <= max_prem:
            # Good — advance alternation
            self._adj_next_direction = 'IN' if direction == 'OUT' else 'OUT'
            return leg_info, leg_ltp, action, target, False

        # Flip direction
        flipped = 'IN' if direction == 'OUT' else 'OUT'
        leg_info, leg_ltp, action, target = _params_for(flipped)

        if min_prem <= target <= max_prem:
            # Flipped works — do NOT advance alternation (override doesn't count)
            return leg_info, leg_ltp, action, target, False

        # Both fail — close both
        return None, 0, None, 0, True

    def _fire_emergency_exit(self, reason: str):
        """Execute emergency exit from tick-level SL trigger."""
        try:
            from services.nas_executor import NasExecutor
            executor = NasExecutor(config=self.config)
            # Pass live option LTPs so exit uses real premiums, not BS
            scan_result = {'spot': self._last_ltp, 'iv': 0.15}
            exits = executor.exit_all_positions(reason, scan_result)
            logger.warning(f"[NAS] Tick SL exit complete: {len(exits)} positions closed")

            # Unsubscribe option legs
            self.subscribe_option_legs([])

        except Exception as e:
            logger.error(f"[NAS] Tick SL exit error: {e}", exc_info=True)

    def _fire_tick_adjustment(self, info: dict, current_prem: float,
                               action: str, target_prem: float):
        """Execute adjustment triggered by tick-level premium breach."""
        try:
            from services.nas_executor import NasExecutor
            from services.nas_db import get_nas_db
            from services.nas_scanner import get_current_week_expiry

            db = get_nas_db()
            executor = NasExecutor(config=self.config)

            # Find the position by tradingsymbol
            active = db.get_active_positions()
            pos = next((p for p in active if p['tradingsymbol'] == info['tradingsymbol']), None)
            if not pos:
                logger.warning(f"[NAS] Tick adj: position {info['tradingsymbol']} not found")
                return

            # Build adjustment dict matching what executor.execute_adjustment expects
            adj = {
                'position_id': pos['id'],
                'action': action,
                'leg': pos['leg'],
                'instrument_type': pos['instrument_type'],
                'strike': pos['strike'],
                'current_premium': round(current_prem, 2),
                'entry_premium': round(pos['entry_price'], 2),
                'ratio': round(current_prem / pos['entry_price'], 2) if pos['entry_price'] > 0 else 0,
                'target_premium': round(target_prem, 2),
                'other_leg_premium': round(target_prem, 2),
            }

            # Build a minimal scan_result for the executor
            spot = self._last_ltp
            expiry = get_current_week_expiry()
            scan_result = {
                'spot': spot,
                'iv': 0.15,
                'daily_atr': getattr(self, '_daily_atr', 100),
                'dte': 1,
                'expiry': str(expiry),
            }

            adj_id, msg = executor.execute_adjustment(adj, scan_result)
            if adj_id:
                logger.info(f"[NAS] Tick adjustment done: {action} on {pos['instrument_type']} "
                            f"→ new pos #{adj_id}")
                # Re-subscribe with updated positions
                updated = db.get_active_positions()
                self.subscribe_option_legs(updated)
            else:
                logger.warning(f"[NAS] Tick adjustment failed: {msg}")

        except Exception as e:
            logger.error(f"[NAS] Tick adjustment error: {e}", exc_info=True)
        finally:
            self._adj_triggered = False

    # ─── NAS ATM Option Leg Monitoring ────────────────────────

    def subscribe_atm_option_legs(self, positions: List[dict]):
        """
        Subscribe to NAS ATM option leg tokens for per-leg SL monitoring.
        Each position must have: tradingsymbol, sl_price, id, strangle_id, entry_price, leg.
        """
        kws = self._get_maruthi_kws()
        if not kws:
            logger.warning("[NAS-ATM] Cannot subscribe option legs — Maruthi ticker not connected")
            return

        with self._lock:
            old_tokens = set(self._atm_option_tokens.keys())
            self._atm_option_tokens.clear()
            self._atm_option_ltps.clear()

            if not positions:
                if old_tokens:
                    # Only unsubscribe tokens not used by OTM
                    tokens_to_unsub = old_tokens - set(self._option_tokens.keys())
                    if tokens_to_unsub:
                        kws.unsubscribe(list(tokens_to_unsub))
                    logger.info(f"[NAS-ATM] Unsubscribed {len(old_tokens)} option tokens")
                return

            new_tokens = []
            for pos in positions:
                tsym = pos.get('tradingsymbol', '')
                if not tsym:
                    continue

                token = self._resolve_option_token(tsym)
                if token:
                    self._atm_option_tokens[token] = {
                        'tradingsymbol': tsym,
                        'sl_price': pos.get('sl_price', 0),
                        'entry_price': pos.get('entry_price', 0),
                        'position_id': pos.get('id'),
                        'strangle_id': pos.get('strangle_id'),
                        'leg': pos.get('leg', ''),
                        'instrument_type': pos.get('instrument_type', ''),
                    }
                    new_tokens.append(token)

            # Subscribe new tokens (avoid duplicating OTM subscriptions)
            all_existing = set(self._option_tokens.keys()) | old_tokens
            tokens_to_add = set(new_tokens) - all_existing
            if tokens_to_add:
                kws.subscribe(list(tokens_to_add))
                kws.set_mode(kws.MODE_LTP, list(tokens_to_add))

            # Unsubscribe old ATM tokens no longer needed (and not used by OTM)
            tokens_to_remove = old_tokens - set(new_tokens) - set(self._option_tokens.keys())
            if tokens_to_remove:
                kws.unsubscribe(list(tokens_to_remove))

            logger.info(f"[NAS-ATM] Monitoring {len(new_tokens)} ATM option legs: "
                        f"{[t['tradingsymbol'] for t in self._atm_option_tokens.values()]}")

    def _check_atm_premium_tick(self, token: int, ltp: float):
        """
        Per-leg SL check for NAS ATM positions.
        If ltp >= sl_price, fire SL handling via NasAtmExecutor.
        """
        if self._atm_sl_processing:
            return

        info = self._atm_option_tokens.get(token)
        if not info:
            return

        sl_price = info.get('sl_price', 0)
        if sl_price <= 0:
            return

        if ltp >= sl_price:
            self._atm_sl_processing = True
            logger.info(
                f"[NAS-ATM] SL TICK! {info['tradingsymbol']} ltp={ltp:.2f} >= SL={sl_price:.2f}"
            )
            threading.Thread(
                target=self._fire_atm_sl_handler,
                args=(info, ltp),
                daemon=True
            ).start()

    def _fire_atm_sl_handler(self, info: dict, ltp: float):
        """Handle ATM leg SL hit — delegates to NasAtmExecutor.check_and_handle_sl()."""
        try:
            from services.nas_atm_executor import NasAtmExecutor
            from services.nas_atm_db import get_nas_atm_db
            from config import NAS_ATM_DEFAULTS

            db = get_nas_atm_db()
            executor = NasAtmExecutor(config=NAS_ATM_DEFAULTS)

            # Build live LTPs from our tracking
            live_ltps = {}
            for t, atm_info in self._atm_option_tokens.items():
                tsym = atm_info['tradingsymbol']
                if t in self._atm_option_ltps:
                    live_ltps[tsym] = self._atm_option_ltps[t]

            positions = db.get_active_positions()
            actions = executor.check_and_handle_sl(positions=positions, live_ltps=live_ltps)

            if actions:
                logger.info(f"[NAS-ATM] SL handler: {len(actions)} actions taken")
                # Re-subscribe with updated positions
                updated = db.get_active_positions()
                self.subscribe_atm_option_legs(updated)
            else:
                logger.info("[NAS-ATM] SL handler: no actions taken (threshold not met in executor)")

        except Exception as e:
            logger.error(f"[NAS-ATM] SL handler error: {e}", exc_info=True)
        finally:
            self._atm_sl_processing = False

    # ─── NAS ATM2 Option Leg Monitoring ───────────────────────

    def subscribe_atm2_option_legs(self, positions: List[dict]):
        """
        Subscribe to NAS ATM2 option leg tokens for per-leg SL monitoring.
        Each position must have: tradingsymbol, sl_price, id, strangle_id, entry_price, leg.
        """
        kws = self._get_maruthi_kws()
        if not kws:
            logger.warning("[NAS-ATM2] Cannot subscribe option legs — Maruthi ticker not connected")
            return

        with self._lock:
            old_tokens = set(self._atm2_option_tokens.keys())
            self._atm2_option_tokens.clear()
            self._atm2_option_ltps.clear()

            if not positions:
                if old_tokens:
                    # Only unsubscribe tokens not used by OTM or ATM
                    tokens_to_unsub = old_tokens - set(self._option_tokens.keys()) - set(self._atm_option_tokens.keys())
                    if tokens_to_unsub:
                        kws.unsubscribe(list(tokens_to_unsub))
                    logger.info(f"[NAS-ATM2] Unsubscribed {len(old_tokens)} option tokens")
                return

            new_tokens = []
            for pos in positions:
                tsym = pos.get('tradingsymbol', '')
                if not tsym:
                    continue

                token = self._resolve_option_token(tsym)
                if token:
                    self._atm2_option_tokens[token] = {
                        'tradingsymbol': tsym,
                        'sl_price': pos.get('sl_price', 0),
                        'entry_price': pos.get('entry_price', 0),
                        'position_id': pos.get('id'),
                        'strangle_id': pos.get('strangle_id'),
                        'leg': pos.get('leg', ''),
                        'instrument_type': pos.get('instrument_type', ''),
                    }
                    new_tokens.append(token)

            # Subscribe new tokens (avoid duplicating OTM/ATM subscriptions)
            all_existing = set(self._option_tokens.keys()) | set(self._atm_option_tokens.keys()) | old_tokens
            tokens_to_add = set(new_tokens) - all_existing
            if tokens_to_add:
                kws.subscribe(list(tokens_to_add))
                kws.set_mode(kws.MODE_LTP, list(tokens_to_add))

            # Unsubscribe old ATM2 tokens no longer needed (and not used by OTM/ATM)
            tokens_to_remove = old_tokens - set(new_tokens) - set(self._option_tokens.keys()) - set(self._atm_option_tokens.keys())
            if tokens_to_remove:
                kws.unsubscribe(list(tokens_to_remove))

            logger.info(f"[NAS-ATM2] Monitoring {len(new_tokens)} ATM2 option legs: "
                        f"{[t['tradingsymbol'] for t in self._atm2_option_tokens.values()]}")

    def _check_atm2_premium_tick(self, token: int, ltp: float):
        """
        Per-leg SL check for NAS ATM2 positions.
        If ltp >= sl_price, fire SL handling via NasAtm2Executor.
        """
        if self._atm2_sl_processing:
            return

        info = self._atm2_option_tokens.get(token)
        if not info:
            return

        sl_price = info.get('sl_price', 0)
        if sl_price <= 0:
            return

        if ltp >= sl_price:
            self._atm2_sl_processing = True
            logger.info(
                f"[NAS-ATM2] SL TICK! {info['tradingsymbol']} ltp={ltp:.2f} >= SL={sl_price:.2f}"
            )
            threading.Thread(
                target=self._fire_atm2_sl_handler,
                args=(info, ltp),
                daemon=True
            ).start()

    def _fire_atm2_sl_handler(self, info: dict, ltp: float):
        """Handle ATM2 leg SL hit — delegates to NasAtm2Executor.check_and_handle_sl()."""
        try:
            from services.nas_atm2_executor import NasAtm2Executor
            from services.nas_atm2_db import get_nas_atm2_db
            from config import NAS_ATM2_DEFAULTS

            db = get_nas_atm2_db()
            executor = NasAtm2Executor(config=NAS_ATM2_DEFAULTS)

            # Build live LTPs from our tracking
            live_ltps = {}
            for t, atm2_info in self._atm2_option_tokens.items():
                tsym = atm2_info['tradingsymbol']
                if t in self._atm2_option_ltps:
                    live_ltps[tsym] = self._atm2_option_ltps[t]

            positions = db.get_active_positions()
            actions = executor.check_and_handle_sl(positions=positions, live_ltps=live_ltps)

            if actions:
                logger.info(f"[NAS-ATM2] SL handler: {len(actions)} actions taken")
                # Re-subscribe with updated positions
                updated = db.get_active_positions()
                self.subscribe_atm2_option_legs(updated)
            else:
                logger.info("[NAS-ATM2] SL handler: no actions taken (threshold not met in executor)")

        except Exception as e:
            logger.error(f"[NAS-ATM2] SL handler error: {e}", exc_info=True)
        finally:
            self._atm2_sl_processing = False

    # ─── Candle Close Handler ────────────────────────────────

    def _on_candle_close(self, candle: dict):
        """
        Fired when a 5-min candle closes.
        Runs the full NAS scan pipeline.
        """
        logger.info(f"[NAS] 5-min candle closed: {candle['date']} close={candle['close']:.1f}")

        try:
            from services.nas_executor import NasExecutor
            from services.nas_scanner import NasScanner

            scanner = NasScanner(self.config)
            executor = NasExecutor(config=self.config)

            # Build DataFrame from aggregated candles (or let scanner fetch its own)
            df_5min = self.aggregator.get_dataframe()
            if len(df_5min) < 60:
                logger.info(f"[NAS] Only {len(df_5min)} ticker candles — scanner will fetch from Kite/DB")
                df_5min = None  # scanner.scan() will call load_5min_bars()

            # Use pre-loaded daily candles for strike computation
            daily_df = getattr(self, '_daily_candles', None)

            # Inject live option premiums into active positions
            from services.nas_db import get_nas_db
            db = get_nas_db()
            active = db.get_active_positions()
            if active:
                for pos in active:
                    tsym = pos.get('tradingsymbol', '')
                    # Find matching token and use live LTP
                    for token, info in self._option_tokens.items():
                        if info['tradingsymbol'] == tsym and token in self._option_ltps:
                            pos['_live_premium'] = self._option_ltps[token]
                            break

            # Run full scan with live data
            scan = scanner.scan(df_5min=df_5min, daily_df=daily_df)
            if scan.get('error'):
                logger.warning(f"[NAS] Scan error: {scan['error']}")
                return

            spot = scan['spot']
            iv = scan.get('iv', 0.15)

            # 1. Check exits on active positions
            if active:
                strangle_ids = set(p.get('strangle_id') for p in active)
                for sid in strangle_ids:
                    sid_positions = [p for p in active if p.get('strangle_id') == sid]
                    total_entry_prem = sum(p['entry_price'] for p in sid_positions)
                    exit_checks = scanner.check_exits(sid_positions, spot, total_entry_prem, iv)
                    if exit_checks:
                        exit_reason = exit_checks[0][0]
                        exits = executor.exit_all_positions(exit_reason, scan)
                        logger.info(f"[NAS] Exit: {exit_reason}, {len(exits)} positions closed")
                        self.subscribe_option_legs([])
                        executor._update_state(scan)
                        return

            # 2. Adjustments are handled tick-by-tick in _check_premium_tick()
            #    (2-tick confirmation, then fire immediately — no candle wait)

            # 3. Entry signal (only if no active positions)
            active_after = db.get_active_positions()
            if not active_after:
                for signal in scan.get('signals', []):
                    if signal['type'] == 'SQUEEZE_ENTRY':
                        sid, msg = executor.execute_strangle_entry(signal, scan)
                        if sid:
                            logger.info(
                                f"[NAS] Entry: strangle #{sid} "
                                f"CE={signal['call_strike']} PE={signal['put_strike']} "
                                f"premium={signal['total_premium']:.1f}"
                            )
                            # Subscribe to new option legs
                            new_active = db.get_active_positions()
                            self.subscribe_option_legs(new_active)

            # 4. Update state
            executor._update_state(scan)

        except Exception as e:
            logger.error(f"[NAS] Candle close handler error: {e}", exc_info=True)

        # ── NAS ATM: check entry on same squeeze signal ──
        self._on_candle_close_atm(candle)

        # ── NAS ATM2: check entry on same squeeze signal ──
        self._on_candle_close_atm2(candle)

    def _on_candle_close_atm2(self, candle: dict):
        """
        ATM2 entry check on candle close.
        Shares the same ATR squeeze detection as NAS OTM.
        """
        try:
            from config import NAS_ATM2_DEFAULTS
            if not NAS_ATM2_DEFAULTS.get('enabled', True):
                return

            from services.nas_atm2_executor import NasAtm2Executor
            from services.nas_atm2_db import get_nas_atm2_db
            from services.nas_scanner import NasScanner

            atm2_executor = NasAtm2Executor(config=NAS_ATM2_DEFAULTS)
            atm2_db = get_nas_atm2_db()

            # Build DataFrame from aggregated candles (or let scanner fetch its own)
            df_5min = self.aggregator.get_dataframe()
            if len(df_5min) < 60:
                df_5min = None  # scanner.scan() will call load_5min_bars()

            # Run scan to check squeeze state
            scanner = NasScanner(NAS_ATM2_DEFAULTS)
            daily_df = getattr(self, '_daily_candles', None)
            scan = scanner.scan(df_5min=df_5min, daily_df=daily_df)
            if scan.get('error'):
                return

            spot = scan['spot']

            # Update ATM2 state
            atm2_db.update_state(
                atr_value=scan.get('atr'),
                atr_ma=scan.get('atr_ma'),
                is_squeezing=1 if scan.get('is_squeezing') else 0,
                squeeze_count=scan.get('squeeze_count', 0),
                spot_price=spot,
                last_scan_time=datetime.now().isoformat(),
            )

            # Check for squeeze entry signal
            for signal in scan.get('signals', []):
                if signal['type'] == 'SQUEEZE_ENTRY':
                    sid, msg = atm2_executor.execute_strangle_entry(spot=spot)
                    if sid:
                        logger.info(f"[NAS-ATM2] Entry: strangle #{sid} at spot {spot:.1f}")
                        # Subscribe ATM2 option legs for SL monitoring
                        new_active = atm2_db.get_active_positions()
                        self.subscribe_atm2_option_legs(new_active)
                    else:
                        logger.info(f"[NAS-ATM2] Entry blocked: {msg}")
                    break  # Only one entry per candle

        except Exception as e:
            logger.error(f"[NAS-ATM2] Candle close handler error: {e}", exc_info=True)

    def _on_candle_close_atm(self, candle: dict):
        """
        ATM entry check on candle close.
        Shares the same ATR squeeze detection as NAS OTM.
        """
        try:
            from config import NAS_ATM_DEFAULTS
            if not NAS_ATM_DEFAULTS.get('enabled', True):
                return

            from services.nas_atm_executor import NasAtmExecutor
            from services.nas_atm_db import get_nas_atm_db
            from services.nas_scanner import NasScanner

            atm_executor = NasAtmExecutor(config=NAS_ATM_DEFAULTS)
            atm_db = get_nas_atm_db()

            # Build DataFrame from aggregated candles (or let scanner fetch its own)
            df_5min = self.aggregator.get_dataframe()
            if len(df_5min) < 60:
                df_5min = None  # scanner.scan() will call load_5min_bars()

            # Run scan to check squeeze state
            scanner = NasScanner(NAS_ATM_DEFAULTS)
            daily_df = getattr(self, '_daily_candles', None)
            scan = scanner.scan(df_5min=df_5min, daily_df=daily_df)
            if scan.get('error'):
                return

            spot = scan['spot']

            # Update ATM state
            atm_db.update_state(
                atr_value=scan.get('atr'),
                atr_ma=scan.get('atr_ma'),
                is_squeezing=1 if scan.get('is_squeezing') else 0,
                squeeze_count=scan.get('squeeze_count', 0),
                spot_price=spot,
                last_scan_time=datetime.now().isoformat(),
            )

            # Check for squeeze entry signal
            for signal in scan.get('signals', []):
                if signal['type'] == 'SQUEEZE_ENTRY':
                    sid, msg = atm_executor.execute_strangle_entry(spot=spot)
                    if sid:
                        logger.info(f"[NAS-ATM] Entry: strangle #{sid} at spot {spot:.1f}")
                        # Subscribe ATM option legs for SL monitoring
                        new_active = atm_db.get_active_positions()
                        self.subscribe_atm_option_legs(new_active)
                    else:
                        logger.info(f"[NAS-ATM] Entry blocked: {msg}")
                    break  # Only one entry per candle

        except Exception as e:
            logger.error(f"[NAS-ATM] Candle close handler error: {e}", exc_info=True)

    # ─── Lifecycle (piggybacks on Maruthi's KiteTicker) ──────

    def start(self):
        """
        Start NAS ticker — loads historical data and marks running.
        Does NOT create its own KiteTicker WebSocket. NIFTY ticks are
        forwarded from Maruthi's ticker via _on_ticks().
        """
        if self.is_running:
            logger.warning("[NAS] Ticker already running")
            return

        access_token = get_access_token()
        if not access_token:
            logger.error("[NAS] No access token — run Kite login first")
            return

        # Load historical candles for ATR warmup
        self._load_historical_candles()
        self._load_daily_candles()

        self.is_running = True
        self.is_connected = True  # Connected via Maruthi's WebSocket
        logger.info("[NAS] Ticker started — receiving ticks via Maruthi ticker")

        # Subscribe to active option legs (via Maruthi's kws)
        threading.Thread(target=self._subscribe_active_legs, daemon=True).start()

    def _subscribe_active_legs(self):
        """Subscribe to active option leg tokens via Maruthi's WebSocket."""
        try:
            from services.nas_db import get_nas_db
            db = get_nas_db()
            active = db.get_active_positions()
            if active:
                self.subscribe_option_legs(active)
        except Exception as e:
            logger.warning(f"[NAS] Could not subscribe active legs: {e}")

        # Also subscribe NAS ATM active legs
        try:
            from services.nas_atm_db import get_nas_atm_db
            atm_db = get_nas_atm_db()
            atm_active = atm_db.get_active_positions()
            if atm_active:
                self.subscribe_atm_option_legs(atm_active)
        except Exception as e:
            logger.warning(f"[NAS-ATM] Could not subscribe active legs: {e}")

        # Also subscribe NAS ATM2 active legs
        try:
            from services.nas_atm2_db import get_nas_atm2_db
            atm2_db = get_nas_atm2_db()
            atm2_active = atm2_db.get_active_positions()
            if atm2_active:
                self.subscribe_atm2_option_legs(atm2_active)
        except Exception as e:
            logger.warning(f"[NAS-ATM2] Could not subscribe active legs: {e}")

    def stop(self):
        """Stop NAS ticker — stop processing ticks."""
        self.is_running = False
        self.is_connected = False
        logger.info("[NAS] Ticker stopped")

    def restart(self):
        """Restart the ticker (e.g., after daily re-login)."""
        self.stop()
        import time
        time.sleep(1)
        self.start()

    def get_status(self) -> dict:
        """Get ticker status for dashboard."""
        # Check if Maruthi ticker is actually connected
        maruthi_connected = False
        try:
            from services.maruthi_ticker import get_maruthi_ticker
            mt = get_maruthi_ticker()
            maruthi_connected = mt is not None and mt.is_connected
        except Exception:
            pass

        return {
            'is_running': self.is_running,
            'is_connected': self.is_running and maruthi_connected,
            'connection_via': 'maruthi_ticker',
            'last_ltp': self._last_ltp,
            'completed_candles': len(self.aggregator.completed_candles),
            'current_candle': self.aggregator.current_candle,
            'option_legs_monitored': len(self._option_tokens),
            'option_legs': [
                {
                    'tradingsymbol': info['tradingsymbol'],
                    'entry_premium': info['entry_premium'],
                    'current_premium': self._option_ltps.get(token),
                    'leg': info['leg'],
                }
                for token, info in self._option_tokens.items()
            ],
            'sl_triggered': self._sl_triggered,
            'adj_triggered': self._adj_triggered,
            'adj_next_direction': self._adj_next_direction,
            # NAS ATM leg info
            'atm_option_legs_monitored': len(self._atm_option_tokens),
            'atm_option_legs': [
                {
                    'tradingsymbol': info['tradingsymbol'],
                    'entry_price': info['entry_price'],
                    'sl_price': info['sl_price'],
                    'current_premium': self._atm_option_ltps.get(token),
                    'leg': info['leg'],
                }
                for token, info in self._atm_option_tokens.items()
            ],
            # NAS ATM2 leg info
            'atm2_option_legs_monitored': len(self._atm2_option_tokens),
            'atm2_option_legs': [
                {
                    'tradingsymbol': info['tradingsymbol'],
                    'entry_price': info['entry_price'],
                    'sl_price': info['sl_price'],
                    'current_premium': self._atm2_option_ltps.get(token),
                    'leg': info['leg'],
                }
                for token, info in self._atm2_option_tokens.items()
            ],
        }


# ─── Singleton ──────────────────────────────────────────────

_nas_ticker: Optional[NasTicker] = None
_ticker_lock = threading.Lock()


def get_nas_ticker(config: dict = None) -> NasTicker:
    """Get or create the global NasTicker instance."""
    global _nas_ticker
    with _ticker_lock:
        if _nas_ticker is None:
            _nas_ticker = NasTicker(config)
        return _nas_ticker


def stop_nas_ticker():
    """Stop and clear the global ticker."""
    global _nas_ticker
    with _ticker_lock:
        if _nas_ticker:
            _nas_ticker.is_running = False
            _nas_ticker.is_connected = False
            _nas_ticker = None
            logger.info("[NAS] Ticker stopped and cleared")
