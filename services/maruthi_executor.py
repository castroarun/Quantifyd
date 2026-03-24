"""
Maruthi Always-On Strategy - Order Executor
==============================================

Handles all order placement, position management, and the core
state machine for the Maruthi dual-SuperTrend strategy.

Key responsibilities:
- Place SL-L trigger orders for futures entries
- Place option short/buy orders
- Execute regime changes (close positions, set SLs)
- Handle hard SL exits
- Manage contract rolling
- Paper + live trading modes
- Safety guardrails

Order types used:
- SL-L (trigger+limit) for futures entry — ensures we only buy above candle high
- LIMIT for option shorts (or MARKET if liquid)
- MARKET for closing positions
"""

import logging
import pandas as pd
from datetime import datetime, date, time as dtime, timedelta
from typing import Dict, List, Optional

from config import MARUTHI_DEFAULTS
from services.maruthi_db import get_maruthi_db, MaruthiDB
from services.maruthi_strategy import (
    MaruthiSignal, determine_actions, compute_dual_supertrend,
    detect_signals, compute_hard_sl, resolve_sl_buffer,
    compute_trigger_price, compute_limit_price
)
from services.maruthi_contract_manager import MaruthiContractManager
from services.kite_service import get_kite, is_authenticated

logger = logging.getLogger(__name__)


class MaruthiExecutor:
    """Order executor for Maruthi always-on strategy."""

    def __init__(self, config: dict = None):
        self.config = config or MARUTHI_DEFAULTS.copy()
        self.db = get_maruthi_db()
        self.contract_mgr = None  # Set when kite is available
        self._current_master_atr = 100.0  # Updated each candle check

    def _get_kite(self):
        """Get authenticated Kite instance."""
        try:
            return get_kite()
        except Exception as e:
            logger.error(f"Failed to get Kite: {e}")
            return None

    def _ensure_contract_mgr(self):
        """Initialize contract manager with Kite instance."""
        if self.contract_mgr is None:
            kite = self._get_kite() if not self.config.get('paper_trading_mode') else None
            self.contract_mgr = MaruthiContractManager(
                kite=kite,
                symbol=self.config.get('symbol', 'MARUTI'),
                exchange=self.config.get('exchange_fo', 'NFO'),
                strike_interval=self.config.get('strike_interval', 100),
            )

    def _get_futures_candle(self, candle_time: str = None) -> Optional[dict]:
        """
        Fetch the matching 30-min futures candle from Kite historical API.

        Trigger prices must be based on the futures candle (not equity)
        because the order is placed on the futures instrument.

        Returns dict with open, high, low, close or None.
        """
        try:
            kite = self._get_kite()
            if not kite:
                return None

            self._ensure_contract_mgr()
            fut = self.contract_mgr.get_futures_symbol()
            if not fut:
                logger.warning("No futures contract — cannot fetch futures candle")
                return None

            fut_token = fut.get('instrument_token')
            if not fut_token:
                # Look up token from instruments
                instruments = kite.instruments('NFO')
                for inst in instruments:
                    if inst['tradingsymbol'] == fut['tradingsymbol']:
                        fut_token = inst['instrument_token']
                        break
            if not fut_token:
                logger.warning(f"No instrument token for {fut['tradingsymbol']}")
                return None

            from_date = datetime.now() - timedelta(days=2)
            to_date = datetime.now()
            data = kite.historical_data(fut_token, from_date, to_date, '30minute')

            if not data:
                return None

            # Find matching candle by time, or use latest
            if candle_time:
                candle_str = str(candle_time).replace('+05:30', '')
                for d in reversed(data):
                    if candle_str in str(d['date']):
                        return {
                            'open': d['open'], 'high': d['high'],
                            'low': d['low'], 'close': d['close'],
                            'date': str(d['date']),
                        }

            # Default: return latest completed candle
            return {
                'open': data[-1]['open'], 'high': data[-1]['high'],
                'low': data[-1]['low'], 'close': data[-1]['close'],
                'date': str(data[-1]['date']),
            }
        except Exception as e:
            logger.error(f"Failed to fetch futures candle: {e}")
            return None

    # =========================================================================
    # Live Position Counts (Kite as source of truth)
    # =========================================================================

    def get_live_position_counts(self) -> dict:
        """
        Get position counts from Kite live positions (source of truth).

        Returns dict with:
            futures_lots: number of active futures lots (abs, regardless of direction)
            futures_direction: 'BUY' or 'SELL' (net direction)
            short_options_lots: number of short option lots
            short_options_by_symbol: {tradingsymbol: lots} breakdown
            protective_options_lots: number of long (bought) option lots
            source: 'KITE' or 'DB'

        Falls back to DB counts only if Kite is unavailable.
        """
        cfg = self.config
        symbol = cfg.get('symbol', 'MARUTI')
        lot_size = cfg.get('lot_size', 50)

        # Always try Kite first if authenticated — even in paper mode,
        # positions may exist from manual trades or previous live sessions.
        # DB is fallback only when Kite is truly unavailable.
        try:
            kite = self._get_kite()
            if not kite or not is_authenticated():
                raise Exception("Kite not available")

            positions = kite.positions().get('net', [])

            fut_qty = 0
            opt_shorts = {}
            opt_longs = {}

            for p in positions:
                ts = p.get('tradingsymbol', '')
                qty = p.get('quantity', 0)
                if qty == 0 or symbol not in ts:
                    continue

                if 'FUT' in ts:
                    fut_qty += qty
                elif 'PE' in ts or 'CE' in ts:
                    if qty < 0:
                        opt_shorts[ts] = opt_shorts.get(ts, 0) + abs(qty)
                    else:
                        opt_longs[ts] = opt_longs.get(ts, 0) + qty

            total_short_lots = sum(v // lot_size for v in opt_shorts.values())
            total_long_lots = sum(v // lot_size for v in opt_longs.values())

            result = {
                'futures_lots': abs(fut_qty) // lot_size,
                'futures_direction': 'SELL' if fut_qty < 0 else 'BUY' if fut_qty > 0 else 'NONE',
                'short_options_lots': total_short_lots,
                'short_options_by_symbol': {k: v // lot_size for k, v in opt_shorts.items()},
                'protective_options_lots': total_long_lots,
                'protective_options_by_symbol': {k: v // lot_size for k, v in opt_longs.items()},
                'source': 'KITE',
            }
            logger.debug(
                f"[LiveCounts] Kite: {result['futures_lots']} fut ({result['futures_direction']}), "
                f"{result['short_options_lots']} short opts, {result['protective_options_lots']} protective opts"
            )
            return result

        except Exception as e:
            logger.warning(f"[LiveCounts] Kite unavailable ({e}), falling back to DB")
            return {
                'futures_lots': self.db.get_active_futures_count(),
                'futures_direction': 'UNKNOWN',
                'short_options_lots': self.db.get_active_short_options_count(),
                'short_options_by_symbol': {},
                'protective_options_lots': len(self.db.get_active_positions('PROTECTIVE_OPTION')),
                'protective_options_by_symbol': {},
                'source': 'DB',
            }

    # =========================================================================
    # Guardrails
    # =========================================================================

    def _check_guardrails(self, is_entry: bool = True, is_option: bool = False) -> tuple:
        """Pre-order safety checks. Returns (passed, reason)."""
        cfg = self.config

        # Mode check
        if not cfg.get('live_trading_enabled') and not cfg.get('paper_trading_mode'):
            return False, "Neither live nor paper mode enabled"

        # Kite auth (live only)
        if cfg.get('live_trading_enabled') and not cfg.get('paper_trading_mode'):
            if not is_authenticated():
                return False, "Kite API not authenticated"

        # Market hours
        now = datetime.now()
        if now.weekday() >= 5:
            return False, f"Market closed (weekend: {now.strftime('%A')})"
        if now.time() < dtime(9, 15) or now.time() > dtime(15, 30):
            return False, f"Market closed ({now.time()})"

        # Daily order limit
        max_orders = cfg.get('max_daily_orders', 20)
        today_orders = self.db.get_today_order_count()
        if today_orders >= max_orders:
            return False, f"Daily order limit ({today_orders}/{max_orders})"

        # Live position counts from Kite (source of truth)
        counts = self.get_live_position_counts()
        active_fut = counts['futures_lots']
        active_opts = counts['short_options_lots']

        # Max futures check (entry only)
        if is_entry:
            max_fut = cfg.get('max_futures_lots', 5)
            if active_fut >= max_fut:
                return False, f"Max futures lots reached ({active_fut}/{max_fut}) [src={counts['source']}]"

        # Option count guard: short options must not exceed active futures count
        # Rule: 1 short option per 1 futures lot — never more
        if is_option:
            if active_opts >= active_fut:
                return False, f"Short options ({active_opts}) already match futures ({active_fut}) — no more allowed [src={counts['source']}]"

        return True, "OK"

    # =========================================================================
    # Order Placement
    # =========================================================================

    def _place_order_kite(self, params: dict) -> Optional[str]:
        """
        Place order via Kite API.

        Returns kite_order_id on success, None on failure.
        """
        kite = self._get_kite()
        if not kite:
            return None

        try:
            order_id = kite.place_order(
                variety=params.get('variety', 'regular'),
                exchange=params.get('exchange', 'NFO'),
                tradingsymbol=params['tradingsymbol'],
                transaction_type=params['transaction_type'],
                quantity=params['quantity'],
                product=params.get('product', 'NRML'),
                order_type=params.get('order_type', 'LIMIT'),
                price=params.get('price'),
                trigger_price=params.get('trigger_price'),
            )
            logger.info(f"Order placed: {order_id} | {params['tradingsymbol']} {params['transaction_type']}")
            return str(order_id)
        except Exception as e:
            logger.error(f"Order failed: {e} | params={params}")
            return None

    def _get_option_limit_price(self, tradingsymbol: str, transaction_type: str) -> float:
        """Get current bid/ask for option LIMIT order pricing."""
        kite = self._get_kite()
        if not kite:
            return 0
        try:
            q = kite.quote([f'NFO:{tradingsymbol}'])
            data = q.get(f'NFO:{tradingsymbol}', {})
            depth = data.get('depth', {})
            if transaction_type == 'SELL':
                # Selling: use best bid
                bids = depth.get('buy', [{}])
                return bids[0].get('price', data.get('last_price', 0))
            else:
                # Buying: use best ask
                asks = depth.get('sell', [{}])
                return asks[0].get('price', data.get('last_price', 0))
        except Exception as e:
            logger.error(f"Failed to get quote for {tradingsymbol}: {e}")
            return 0

    def _place_option_with_retry(self, params: dict, max_retries: int = 3,
                                  retry_delay: float = 2.0) -> Optional[str]:
        """
        Place option order as LIMIT with fill-check-revise loop.

        Stock options on Zerodha reject MARKET orders (illiquidity protection).
        Places LIMIT at current bid/ask, checks fill, revises price if unfilled.

        Returns kite_order_id on fill, None on failure.
        """
        import time
        kite = self._get_kite()
        if not kite:
            return None

        symbol = params['tradingsymbol']
        txn = params['transaction_type']

        # Force LIMIT order type for stock options
        params['order_type'] = 'LIMIT'
        params['price'] = self._get_option_limit_price(symbol, txn)
        if not params['price']:
            logger.error(f"Cannot get price for {symbol}, aborting")
            return None

        # Place initial order
        order_id = self._place_order_kite(params)
        if not order_id:
            return None

        # Check fill with retries
        for attempt in range(max_retries):
            time.sleep(retry_delay)
            try:
                order_history = kite.order_history(order_id)
                latest = order_history[-1] if order_history else {}
                status = latest.get('status', '')

                if status == 'COMPLETE':
                    logger.info(f"Option order filled: {symbol} @ {latest.get('average_price')}")
                    return order_id

                if status in ('CANCELLED', 'REJECTED'):
                    logger.error(f"Option order {status}: {symbol} — {latest.get('status_message', '')}")
                    return None

                # Still OPEN — revise price to chase fill
                new_price = self._get_option_limit_price(symbol, txn)
                if new_price and new_price != params['price']:
                    logger.info(f"Revising {symbol} order: {params['price']} -> {new_price} (attempt {attempt+1})")
                    kite.modify_order(
                        variety='regular',
                        order_id=order_id,
                        price=new_price,
                    )
                    params['price'] = new_price

            except Exception as e:
                logger.error(f"Fill check failed for {symbol}: {e}")

        # Final check
        try:
            order_history = kite.order_history(order_id)
            latest = order_history[-1] if order_history else {}
            if latest.get('status') == 'COMPLETE':
                return order_id
            # Cancel unfilled order
            logger.warning(f"Option order not filled after {max_retries} retries, cancelling: {symbol}")
            kite.cancel_order('regular', order_id)
        except Exception as e:
            logger.error(f"Final fill check/cancel failed: {e}")

        return None

    def place_futures_entry(self, direction: str, trigger_price: float,
                            limit_price: float, hard_sl: float,
                            signal_type: str, regime: str) -> Optional[int]:
        """
        Place SL-L trigger order for futures entry.

        Args:
            direction: 'BUY' or 'SELL'
            trigger_price: Price that activates the order
            limit_price: Max/min fill price
            hard_sl: Hard stop loss price
            signal_type: What signal triggered this
            regime: Current regime

        Returns: position_id or None
        """
        passed, reason = self._check_guardrails(is_entry=True)
        if not passed:
            logger.warning(f"Guardrail blocked futures entry: {reason}")
            return None

        self._ensure_contract_mgr()
        cfg = self.config
        lot_size = cfg.get('lot_size', 200)

        # Get futures contract
        if cfg.get('paper_trading_mode'):
            tradingsymbol = f"MARUTI_FUT_PAPER"
            expiry_date = None
        else:
            fut = self.contract_mgr.get_futures_symbol()
            if not fut:
                logger.error("No futures contract found")
                return None
            tradingsymbol = fut['tradingsymbol']
            expiry_date = fut['expiry']
            lot_size = fut.get('lot_size', lot_size)

        now = datetime.now().isoformat()

        # Log order
        order_id = self.db.log_order(
            tradingsymbol=tradingsymbol,
            exchange=cfg.get('exchange_fo', 'NFO'),
            transaction_type=direction,
            qty=lot_size,
            price=limit_price,
            trigger_price=trigger_price,
            order_type='SL',  # SL-L in Kite
            product='NRML',
            signal_type=signal_type,
            status='PAPER' if cfg.get('paper_trading_mode') else 'PENDING',
        )

        # Place on Kite (live mode)
        kite_order_id = None
        if cfg.get('live_trading_enabled') and not cfg.get('paper_trading_mode'):
            kite_order_id = self._place_order_kite({
                'exchange': cfg.get('exchange_fo', 'NFO'),
                'tradingsymbol': tradingsymbol,
                'transaction_type': direction,
                'quantity': lot_size,
                'product': 'NRML',
                'order_type': 'SL',
                'price': limit_price,
                'trigger_price': trigger_price,
            })
            if kite_order_id:
                self.db.update_order(order_id, kite_order_id=kite_order_id, status='PLACED')
            else:
                self.db.update_order(order_id, status='FAILED', error_message='Kite order failed')
                return None

        # Create position (PENDING until trigger fires)
        # In paper mode, we'll assume fill on the next candle check
        position_id = self.db.add_position(
            position_type='FUTURES',
            tradingsymbol=tradingsymbol,
            exchange=cfg.get('exchange_fo', 'NFO'),
            transaction_type=direction,
            qty=lot_size,
            entry_price=trigger_price,  # Will be updated on fill
            entry_time=now,
            trigger_price=trigger_price,
            sl_price=hard_sl,
            expiry_date=str(expiry_date) if expiry_date else None,
            kite_order_id=kite_order_id,
            regime=regime,
            signal_type=signal_type,
            status='PENDING' if not cfg.get('paper_trading_mode') else 'ACTIVE',
        )

        logger.info(f"Futures entry: {direction} {tradingsymbol} trigger={trigger_price} SL={hard_sl}")
        return position_id

    def place_option_short(self, option_type: str, strike: float,
                           signal_type: str, regime: str) -> Optional[int]:
        """
        Place short option order (SELL CE or PE).

        Args:
            option_type: 'CE' or 'PE'
            strike: Strike price
            signal_type: What triggered this
            regime: Current regime
        """
        passed, reason = self._check_guardrails(is_entry=False, is_option=True)
        if not passed:
            logger.warning(f"Guardrail blocked option short: {reason}")
            return None

        self._ensure_contract_mgr()
        cfg = self.config
        lot_size = cfg.get('lot_size', 200)
        min_expiry = cfg.get('min_expiry_days_new', 6)

        if cfg.get('paper_trading_mode'):
            tradingsymbol = f"MARUTI_{strike}{option_type}_PAPER"
            expiry_date = None
            entry_price = 0  # Would be filled from market
        else:
            opt = self.contract_mgr.get_otm_option(
                spot_price=strike,  # Approximate — contract_mgr handles rounding
                option_type=option_type,
                otm_strikes=1,
                min_expiry_days=min_expiry,
            )
            if not opt:
                # Try with the exact strike
                current_expiry = self.contract_mgr.get_current_expiry(option_type)
                dte = self.contract_mgr.days_to_expiry(current_expiry)
                expiry = current_expiry if dte > min_expiry else self.contract_mgr.get_next_expiry(option_type)
                opt = self.contract_mgr.get_option_symbol(strike, option_type, expiry)

            if not opt:
                logger.error(f"No option contract found: {option_type} {strike}")
                return None

            tradingsymbol = opt['tradingsymbol']
            expiry_date = opt['expiry']
            lot_size = opt.get('lot_size', lot_size)

            # Get current premium (for SL reference on master reversal)
            try:
                kite = self._get_kite()
                quote = kite.quote([f"NFO:{tradingsymbol}"])
                q = quote.get(f"NFO:{tradingsymbol}", {})
                entry_price = q.get('last_price', 0)
            except Exception:
                entry_price = 0

        now = datetime.now().isoformat()

        # Log order
        order_id = self.db.log_order(
            tradingsymbol=tradingsymbol,
            exchange='NFO',
            transaction_type='SELL',
            qty=lot_size,
            price=entry_price,
            order_type='LIMIT',  # Stock options MUST use LIMIT (Zerodha rejects MARKET)
            product='NRML',
            signal_type=signal_type,
            status='PAPER' if cfg.get('paper_trading_mode') else 'PENDING',
        )

        # Place on Kite — stock options require LIMIT with fill-check-revise
        kite_order_id = None
        if cfg.get('live_trading_enabled') and not cfg.get('paper_trading_mode'):
            kite_order_id = self._place_option_with_retry({
                'exchange': 'NFO',
                'tradingsymbol': tradingsymbol,
                'transaction_type': 'SELL',
                'quantity': lot_size,
                'product': 'NRML',
            })
            if kite_order_id:
                # Get actual fill price
                try:
                    kite = self._get_kite()
                    oh = kite.order_history(kite_order_id)
                    fill_price = oh[-1].get('average_price', entry_price) if oh else entry_price
                    entry_price = fill_price
                except Exception:
                    pass
                self.db.update_order(order_id, kite_order_id=kite_order_id, status='COMPLETE')
            else:
                self.db.update_order(order_id, status='FAILED')
                return None

        # Create position
        position_id = self.db.add_position(
            position_type='SHORT_OPTION',
            tradingsymbol=tradingsymbol,
            exchange='NFO',
            transaction_type='SELL',
            qty=lot_size,
            entry_price=entry_price,
            entry_time=now,
            instrument_type=option_type,
            strike=strike,
            expiry_date=str(expiry_date) if expiry_date else None,
            kite_order_id=kite_order_id,
            regime=regime,
            signal_type=signal_type,
            status='ACTIVE',
        )

        logger.info(f"Option short: SELL {tradingsymbol} @ {entry_price}")
        return position_id

    def place_protective_option(self, option_type: str, spot_price: float,
                                regime: str) -> Optional[int]:
        """
        Buy far OTM protective option (5% OTM).

        Args:
            option_type: 'PE' for bull regime, 'CE' for bear regime
            spot_price: Current spot price
            regime: Current regime
        """
        # Guard: protectives must not exceed futures count (from Kite)
        counts = self.get_live_position_counts()
        if counts['protective_options_lots'] >= counts['futures_lots']:
            logger.info(
                f"[Protective] Already have {counts['protective_options_lots']} protectives "
                f"for {counts['futures_lots']} futures — skipping [{counts['source']}]"
            )
            return None

        self._ensure_contract_mgr()
        cfg = self.config
        lot_size = cfg.get('lot_size', 200)
        strike_interval = cfg.get('strike_interval', 100)
        otm_strikes = cfg.get('protective_otm_strikes', 5)

        # Calculate strike: N strikes OTM from spot
        # PE (bull protection): N strikes below spot
        # CE (bear protection): N strikes above spot
        if option_type == 'PE':
            base = int(spot_price // strike_interval) * strike_interval  # ATM or 1 below
            strike = float(base - otm_strikes * strike_interval)
        else:
            base = int(-(-spot_price // strike_interval)) * strike_interval  # ATM or 1 above
            strike = float(base + otm_strikes * strike_interval)

        logger.info(f"[Protective] spot={spot_price:.0f} → {option_type} strike={strike:.0f} ({otm_strikes} strikes OTM)")

        if cfg.get('paper_trading_mode'):
            tradingsymbol = f"MARUTI_{strike}{option_type}_PROT_PAPER"
            expiry_date = None
            entry_price = 0
        else:
            opt = self.contract_mgr.get_option_symbol(
                strike, option_type,
                self.contract_mgr.get_current_expiry(option_type)
            )
            if not opt:
                logger.error(f"No protective option found: {option_type} {strike}")
                return None

            tradingsymbol = opt['tradingsymbol']
            expiry_date = opt['expiry']
            lot_size = opt.get('lot_size', lot_size)

            try:
                kite = self._get_kite()
                quote = kite.quote([f"NFO:{tradingsymbol}"])
                entry_price = quote.get(f"NFO:{tradingsymbol}", {}).get('last_price', 0)
            except Exception:
                entry_price = 0

        now = datetime.now().isoformat()

        order_id = self.db.log_order(
            tradingsymbol=tradingsymbol,
            exchange='NFO',
            transaction_type='BUY',
            qty=lot_size,
            price=entry_price,
            order_type='LIMIT',  # Stock options MUST use LIMIT
            product='NRML',
            signal_type='PROTECTIVE',
            status='PAPER' if cfg.get('paper_trading_mode') else 'PENDING',
        )

        kite_order_id = None
        if cfg.get('live_trading_enabled') and not cfg.get('paper_trading_mode'):
            kite_order_id = self._place_option_with_retry({
                'exchange': 'NFO',
                'tradingsymbol': tradingsymbol,
                'transaction_type': 'BUY',
                'quantity': lot_size,
                'product': 'NRML',
            })
            if kite_order_id:
                try:
                    kite = self._get_kite()
                    oh = kite.order_history(kite_order_id)
                    fill_price = oh[-1].get('average_price', entry_price) if oh else entry_price
                    entry_price = fill_price
                except Exception:
                    pass
                self.db.update_order(order_id, kite_order_id=kite_order_id, status='COMPLETE')
            else:
                self.db.update_order(order_id, status='FAILED')
                return None

        position_id = self.db.add_position(
            position_type='PROTECTIVE_OPTION',
            tradingsymbol=tradingsymbol,
            exchange='NFO',
            transaction_type='BUY',
            qty=lot_size,
            entry_price=entry_price,
            entry_time=now,
            instrument_type=option_type,
            strike=strike,
            expiry_date=str(expiry_date) if expiry_date else None,
            kite_order_id=kite_order_id,
            regime=regime,
            signal_type='PROTECTIVE',
            status='ACTIVE',
        )

        logger.info(f"Protective option: BUY {tradingsymbol} @ {entry_price}")
        return position_id

    # =========================================================================
    # Position Closing
    # =========================================================================

    def _close_position(self, pos: dict, exit_reason: str, exit_price: float = None) -> bool:
        """
        Close a single position. Uses MARKET for futures, LIMIT with retry for options.

        Verifies position exists on Kite before placing close order.
        Logs close order to DB for audit trail.
        """
        cfg = self.config

        if exit_price is None:
            exit_price = pos.get('entry_price', 0)  # Fallback

        # Determine close direction
        if pos['transaction_type'] == 'BUY':
            close_direction = 'SELL'
        else:
            close_direction = 'BUY'

        if not cfg.get('paper_trading_mode'):
            ts = pos['tradingsymbol']
            qty = pos['qty']

            # Verify this position actually exists on Kite before closing
            try:
                kite = self._get_kite()
                if kite and is_authenticated():
                    kite_positions = kite.positions().get('net', [])
                    kite_qty = 0
                    for kp in kite_positions:
                        if kp.get('tradingsymbol') == ts:
                            kite_qty = kp.get('quantity', 0)
                            break
                    # Verify: position must exist and be in the expected direction
                    if kite_qty == 0:
                        logger.warning(
                            f"[ClosePos] #{pos['id']} {ts} NOT on Kite (qty=0) — marking closed in DB only"
                        )
                        self.db.close_position(pos['id'], exit_price, f'{exit_reason}_NOT_ON_KITE')
                        return True
                    # Don't close more than what's on Kite
                    if abs(kite_qty) < qty:
                        logger.warning(
                            f"[ClosePos] #{pos['id']} {ts} Kite has {kite_qty} but DB says {qty} — capping to Kite qty"
                        )
                        qty = abs(kite_qty)
            except Exception as e:
                logger.error(f"[ClosePos] Kite check failed: {e} — proceeding with DB qty")

            is_option = pos.get('position_type') in ('SHORT_OPTION', 'PROTECTIVE_OPTION')

            # Log close order to DB for audit trail
            order_id = self.db.log_order(
                tradingsymbol=ts,
                exchange=pos.get('exchange', 'NFO'),
                transaction_type=close_direction,
                qty=qty,
                price=exit_price,
                order_type='LIMIT' if is_option else 'MARKET',
                product='NRML',
                signal_type=f'CLOSE_{exit_reason}',
                status='PENDING',
            )

            if is_option:
                kite_order_id = self._place_option_with_retry({
                    'exchange': pos.get('exchange', 'NFO'),
                    'tradingsymbol': ts,
                    'transaction_type': close_direction,
                    'quantity': qty,
                    'product': 'NRML',
                })
            else:
                kite_order_id = self._place_order_kite({
                    'exchange': pos.get('exchange', 'NFO'),
                    'tradingsymbol': ts,
                    'transaction_type': close_direction,
                    'quantity': qty,
                    'product': 'NRML',
                    'order_type': 'MARKET',
                })

            if kite_order_id:
                self.db.update_order(order_id, kite_order_id=kite_order_id, status='PLACED')
            else:
                self.db.update_order(order_id, status='FAILED', error_message='Kite order failed')
                logger.error(f"Failed to close position {pos['id']}: {ts}")
                return False

        self.db.close_position(pos['id'], exit_price, exit_reason)
        logger.info(f"Closed position {pos['id']}: {pos['tradingsymbol']} reason={exit_reason}")
        return True

    def close_all_positions(self, exit_reason: str, keep_last_option: bool = False,
                            exit_price: float = None) -> int:
        """
        Close all active positions.

        Args:
            exit_reason: Why we're closing
            keep_last_option: If True, keep the most recent SHORT_OPTION with SL at entry
            exit_price: Override exit price (for paper mode)

        Returns: Number of positions closed
        """
        all_positions = self.db.get_active_positions()
        closed = 0

        # Find the last short option (to potentially keep)
        last_option = None
        if keep_last_option:
            last_option = self.db.get_last_short_option()

        # Close in order: futures first, then options (margin-safe)
        # Sort: FUTURES first, then SHORT_OPTION, then PROTECTIVE_OPTION
        order = {'FUTURES': 0, 'SHORT_OPTION': 1, 'PROTECTIVE_OPTION': 2}
        sorted_positions = sorted(all_positions, key=lambda p: order.get(p['position_type'], 9))

        for pos in sorted_positions:
            # Skip the last short option if we're keeping it
            if keep_last_option and last_option and pos['id'] == last_option['id']:
                # Set SL at entry premium for this option
                self.db.update_position(pos['id'], sl_price=pos['entry_price'],
                                        notes='Carried over on regime change — SL at entry premium')
                logger.info(f"Keeping last option {pos['id']}: {pos['tradingsymbol']} SL={pos['entry_price']}")
                continue

            self._close_position(pos, exit_reason, exit_price)
            closed += 1

        # Also cancel any pending trigger orders
        pending = self.db.get_pending_positions()
        for pos in pending:
            if cfg := self.config:
                if cfg.get('live_trading_enabled') and pos.get('kite_order_id'):
                    try:
                        kite = self._get_kite()
                        kite.cancel_order('regular', pos['kite_order_id'])
                    except Exception as e:
                        logger.error(f"Failed to cancel trigger order: {e}")
            self.db.cancel_position(pos['id'])
            closed += 1

        return closed

    # =========================================================================
    # Main Execution Pipeline
    # =========================================================================

    def execute_actions(self, actions: List[dict], signal: MaruthiSignal) -> List[str]:
        """
        Execute a list of actions from determine_actions().

        Returns list of result messages.
        """
        results = []
        regime = self.db.get_regime()
        current_regime = regime.get('regime', 'FLAT')

        # Cancel pending triggers invalidated by child flip
        # e.g., CHILD_BULL in BEAR regime → any pending SELL trigger is stale
        # (child crossed above its ST, bearish entry signal is no longer valid)
        if signal.signal_type in (MaruthiSignal.CHILD_BULL, MaruthiSignal.CHILD_BEAR):
            invalidate_dir = 'SELL' if signal.signal_type == MaruthiSignal.CHILD_BULL else 'BUY'
            pending = self.db.get_pending_positions()
            for pend in pending:
                if pend.get('transaction_type') == invalidate_dir and pend.get('position_type') == 'FUTURES':
                    if self.config.get('live_trading_enabled') and pend.get('kite_order_id'):
                        try:
                            kite = self._get_kite()
                            kite.cancel_order('regular', pend['kite_order_id'])
                            logger.info(f"Cancelled invalidated Kite order {pend['kite_order_id']}")
                        except Exception as e:
                            logger.error(f"Failed to cancel invalidated order: {e}")
                    self.db.cancel_position(pend['id'])
                    results.append(
                        f"Cancelled stale {invalidate_dir} trigger #{pend['id']} "
                        f"(child flipped {signal.signal_type})"
                    )
                    logger.info(
                        f"SIGNAL INVALIDATED: #{pend['id']} {invalidate_dir} trigger cancelled — "
                        f"child flipped to {'BULL' if signal.signal_type == MaruthiSignal.CHILD_BULL else 'BEAR'}"
                    )

        for action in actions:
            act = action['action']

            if act == 'REGIME_CHANGE':
                new_regime = action['new_regime']
                logger.info(f"REGIME CHANGE: {current_regime} → {new_regime}")

                # Close all positions (keep last option on master reversal)
                closed = self.close_all_positions(
                    exit_reason=f'REGIME_{new_regime}',
                    keep_last_option=True,
                )
                results.append(f"Closed {closed} positions for regime change to {new_regime}")

                # Update regime in DB — fresh SL (no trailing on new regime)
                sl_buf = resolve_sl_buffer(self.config, self._current_master_atr)
                self.db.update_regime(
                    regime=new_regime,
                    master_st_value=signal.master_st,
                    child_st_value=signal.child_st,
                    master_direction=signal.master_direction,
                    child_direction=signal.child_direction,
                    hard_sl_price=compute_hard_sl(
                        signal.master_st, new_regime,
                        sl_buf,
                        prev_hard_sl=0,  # Fresh start — no trailing on regime change
                    ),
                    regime_start_time=datetime.now().isoformat(),
                    last_signal_time=datetime.now().isoformat(),
                )
                logger.info(f"Regime {new_regime}: SL buffer={sl_buf:.1f} (ATR={self._current_master_atr:.1f})")
                current_regime = new_regime

            elif act in ('BUY_FUTURES', 'SHORT_FUTURES'):
                direction = 'BUY' if act == 'BUY_FUTURES' else 'SELL'
                opposite = 'SELL' if direction == 'BUY' else 'BUY'
                trigger = action['trigger_price']
                limit = action['limit_price']

                # 1. Cancel any pending triggers in the OPPOSITE direction
                #    (signal reversed — stale orders are invalid)
                pending = self.db.get_pending_positions()
                for pend in pending:
                    if pend.get('transaction_type') == opposite:
                        # Cancel on Kite if live
                        if self.config.get('live_trading_enabled') and pend.get('kite_order_id'):
                            try:
                                kite = self._get_kite()
                                kite.cancel_order('regular', pend['kite_order_id'])
                                logger.info(f"Cancelled stale Kite order {pend['kite_order_id']}")
                            except Exception as e:
                                logger.error(f"Failed to cancel stale order: {e}")
                        self.db.cancel_position(pend['id'])
                        results.append(f"Cancelled stale {opposite} trigger #{pend['id']}")

                # 2. Compute trigger from futures candle (not equity)
                if not self.config.get('paper_trading_mode'):
                    candle_time = signal.candle.get('date')
                    fut_candle = self._get_futures_candle(candle_time)
                    if fut_candle:
                        trigger = compute_trigger_price(fut_candle, direction)
                        limit = compute_limit_price(trigger, direction)
                        logger.info(
                            f"Futures candle {fut_candle['date']}: "
                            f"H={fut_candle['high']} L={fut_candle['low']} "
                            f"→ trigger={trigger} limit={limit}"
                        )

                # 3. If same-direction pending exists, REFRESH its trigger price
                #    (new signal candle = better trigger, don't skip it)
                pending_refresh = self.db.get_pending_positions()
                existing_pending = [p for p in pending_refresh if p.get('transaction_type') == direction]
                if existing_pending:
                    old = existing_pending[0]
                    old_trigger = old.get('trigger_price', 0)
                    if abs(old_trigger - trigger) > 0.5:
                        # Update trigger in DB
                        old_limit = old.get('entry_price', limit)  # entry_price stores limit for pending
                        self.db.update_position(
                            old['id'],
                            trigger_price=trigger,
                            entry_price=trigger,  # Will be updated on fill
                            sl_price=action['hard_sl'],
                            signal_type=signal.signal_type,
                        )
                        logger.info(
                            f"SIGNAL REFRESH: #{old['id']} {direction} trigger "
                            f"{old_trigger:.1f} → {trigger:.1f} (new signal candle)"
                        )

                        # Modify Kite order if live
                        if self.config.get('live_trading_enabled') and old.get('kite_order_id'):
                            try:
                                kite = self._get_kite()
                                kite.modify_order(
                                    variety='regular',
                                    order_id=old['kite_order_id'],
                                    trigger_price=trigger,
                                    price=limit,
                                )
                                logger.info(
                                    f"Modified Kite order {old['kite_order_id']}: "
                                    f"trigger {old_trigger:.1f} → {trigger:.1f}"
                                )
                                results.append(
                                    f"Refreshed {direction} trigger: {old_trigger:.1f} → {trigger:.1f} "
                                    f"(Kite order updated)"
                                )
                            except Exception as e:
                                logger.error(f"Failed to modify Kite order: {e}")
                                # Cancel old and place new
                                try:
                                    kite.cancel_order('regular', old['kite_order_id'])
                                except Exception:
                                    pass
                                self.db.cancel_position(old['id'])
                                # Fall through to place new order below
                                existing_pending = []
                        else:
                            results.append(
                                f"Refreshed {direction} trigger: {old_trigger:.1f} → {trigger:.1f}"
                            )
                    else:
                        logger.info(f"Pending {direction} trigger unchanged ({old_trigger:.1f}) — no refresh needed")
                        results.append(f"Pending {direction} trigger @ {old_trigger:.1f} still valid")

                if not existing_pending:
                    # No pending — place new entry
                    pos_id = self.place_futures_entry(
                        direction=direction,
                        trigger_price=trigger,
                        limit_price=limit,
                        hard_sl=action['hard_sl'],
                        signal_type=signal.signal_type,
                        regime=current_regime,
                    )
                    if pos_id:
                        results.append(f"{act.replace('_', ' ').title()} trigger @ {trigger}")

            elif act == 'SHORT_CALL':
                pos_id = self.place_option_short(
                    option_type='CE',
                    strike=action['strike'],
                    signal_type=signal.signal_type,
                    regime=current_regime,
                )
                if pos_id:
                    results.append(f"Short CE {action['strike']}")

            elif act == 'SHORT_PUT':
                pos_id = self.place_option_short(
                    option_type='PE',
                    strike=action['strike'],
                    signal_type=signal.signal_type,
                    regime=current_regime,
                )
                if pos_id:
                    results.append(f"Short PE {action['strike']}")

            elif act == 'HARD_SL_EXIT':
                # Hard SL = close EVERYTHING, no exceptions
                closed = self.close_all_positions(
                    exit_reason='HARD_SL',
                    keep_last_option=False,
                )
                self.db.update_regime(regime='FLAT', last_signal_time=datetime.now().isoformat())
                results.append(f"HARD SL HIT — closed ALL {closed} positions, now FLAT")

        # Log signal
        self.db.log_signal(
            signal_type=signal.signal_type,
            regime=current_regime,
            master_direction=signal.master_direction,
            child_direction=signal.child_direction,
            master_st_value=signal.master_st,
            child_st_value=signal.child_st,
            candle_time=signal.candle.get('time'),
            candle_high=signal.candle['high'],
            candle_low=signal.candle['low'],
            candle_close=signal.candle['close'],
            action_taken='; '.join(results),
        )

        # Update regime timestamps
        self.db.update_regime(
            child_direction=signal.child_direction,
            child_st_value=signal.child_st,
            last_signal_time=datetime.now().isoformat(),
            last_candle_time=signal.candle.get('time'),
        )

        return results

    # =========================================================================
    # Signal Candle Refresh
    # =========================================================================

    def _refresh_pending_triggers(self, latest: pd.Series, current_regime: str,
                                   sl_buffer: float) -> None:
        """
        Refresh unfilled pending triggers with the current candle.

        When a child signal fires but its trigger (candle low - buffer for SELL,
        candle high + buffer for BUY) is never breached, the pending position
        sits with a stale trigger. If a subsequent candle still confirms the
        same direction (close below child ST in BEAR, above in BULL), we
        update the trigger to use the new candle — it's a fresher, more
        relevant entry level.

        Also handles the case where price briefly crosses child ST intraday
        (no close-based flip detected) but the candle still closes in the
        right direction.
        """
        pending = self.db.get_pending_positions()
        if not pending:
            return

        cfg = self.config
        child_st = float(latest['child_st'])
        child_dir = int(latest['child_dir'])
        close_price = float(latest['close'])

        for pos in pending:
            if pos.get('position_type') != 'FUTURES':
                continue

            direction = pos.get('transaction_type', '')
            old_trigger = pos.get('trigger_price', 0)

            # Validate: pending direction must match regime
            if direction == 'SELL' and current_regime != 'BEAR':
                continue
            if direction == 'BUY' and current_regime != 'BULL':
                continue

            # Check if current candle confirms the direction:
            # SELL (BEAR): close should be below child ST and child_dir == -1
            # BUY (BULL): close should be above child ST and child_dir == 1
            confirms = False
            if direction == 'SELL' and child_dir == -1 and close_price < child_st:
                confirms = True
            elif direction == 'BUY' and child_dir == 1 and close_price > child_st:
                confirms = True

            if not confirms:
                continue

            # Build candle dict for trigger computation
            candle = {
                'high': float(latest['high']),
                'low': float(latest['low']),
                'close': close_price,
            }

            # Use futures candle if available (live mode)
            if not cfg.get('paper_trading_mode'):
                candle_time = str(latest.get('date', latest.name))
                fut_candle = self._get_futures_candle(candle_time)
                if fut_candle:
                    candle = fut_candle

            new_trigger = compute_trigger_price(candle, direction)
            new_limit = compute_limit_price(new_trigger, direction)

            # Only refresh if trigger actually changed meaningfully
            if abs(old_trigger - new_trigger) < 1.0:
                continue

            # Update in DB
            hard_sl = compute_hard_sl(
                float(latest['master_st']), current_regime,
                sl_buffer, prev_hard_sl=pos.get('sl_price', 0) or 0,
            )
            self.db.update_position(
                pos['id'],
                trigger_price=new_trigger,
                entry_price=new_trigger,
                sl_price=hard_sl,
            )
            logger.info(
                f"SIGNAL REFRESH (no-flip): #{pos['id']} {direction} trigger "
                f"{old_trigger:.1f} → {new_trigger:.1f} (candle close={close_price:.1f})"
            )

            # Modify Kite order if live
            if cfg.get('live_trading_enabled') and pos.get('kite_order_id'):
                try:
                    kite = self._get_kite()
                    kite.modify_order(
                        variety='regular',
                        order_id=pos['kite_order_id'],
                        trigger_price=new_trigger,
                        price=new_limit,
                    )
                    logger.info(
                        f"Modified Kite order {pos['kite_order_id']}: "
                        f"trigger {old_trigger:.1f} → {new_trigger:.1f}"
                    )
                except Exception as e:
                    logger.error(f"Failed to modify Kite order for refresh: {e}")
                    # Cancel old and re-place
                    try:
                        kite.cancel_order('regular', pos['kite_order_id'])
                        new_kite_id = self._place_order_kite({
                            'exchange': cfg.get('exchange_fo', 'NFO'),
                            'tradingsymbol': pos['tradingsymbol'],
                            'transaction_type': direction,
                            'quantity': pos.get('qty', cfg.get('lot_size', 50)),
                            'product': 'NRML',
                            'order_type': 'SL',
                            'price': new_limit,
                            'trigger_price': new_trigger,
                        })
                        if new_kite_id:
                            self.db.update_position(pos['id'], kite_order_id=new_kite_id)
                            logger.info(f"Re-placed order: {new_kite_id} trigger={new_trigger:.1f}")
                    except Exception as e2:
                        logger.error(f"Failed to re-place order after cancel: {e2}")

    # =========================================================================
    # Scheduled Tasks
    # =========================================================================

    def run_candle_check(self, df: pd.DataFrame) -> List[str]:
        """
        Main 30-min candle check. Called by scheduler.

        Args:
            df: 30-min OHLCV DataFrame for MARUTI (with enough history for SuperTrend)

        Returns: List of action messages
        """
        cfg = self.config

        # Sync DB with Kite before any logic — ensures position counts are accurate
        try:
            sync_msgs = self.sync_positions_with_kite()
            if sync_msgs:
                logger.info(f"[CandleCheck] Position sync: {sync_msgs}")
        except Exception as e:
            logger.error(f"[CandleCheck] Position sync failed: {e}")

        # Compute SuperTrend
        df = compute_dual_supertrend(
            df,
            master_period=cfg.get('master_atr_period', 7),
            master_mult=cfg.get('master_multiplier', 5.0),
            child_period=cfg.get('child_atr_period', 7),
            child_mult=cfg.get('child_multiplier', 2.0),
        )

        # Get current regime
        regime = self.db.get_regime()
        current_regime = regime.get('regime', 'FLAT')

        # Get current master ATR for SL buffer calculation
        latest = df.iloc[-1]
        master_atr = float(latest.get('master_atr', 100))
        self._current_master_atr = master_atr
        sl_buffer = resolve_sl_buffer(cfg, master_atr)

        # Reconcile regime with computed directions
        # If app restarted and missed a master flip, fix the regime from live data
        computed_master_dir = int(latest['master_dir'])
        if current_regime == 'BULL' and computed_master_dir == -1:
            logger.warning(f"REGIME MISMATCH: DB says BULL but computed master is BEAR — correcting to BEAR")
            current_regime = 'BEAR'
            self.db.update_regime(
                regime='BEAR',
                master_st_value=float(latest['master_st']),
                child_st_value=float(latest['child_st']),
                master_direction=computed_master_dir,
                child_direction=int(latest['child_dir']),
                hard_sl_price=compute_hard_sl(float(latest['master_st']), 'BEAR', sl_buffer, prev_hard_sl=0),
                regime_start_time=datetime.now().isoformat(),
            )
        elif current_regime == 'BEAR' and computed_master_dir == 1:
            logger.warning(f"REGIME MISMATCH: DB says BEAR but computed master is BULL — correcting to BULL")
            current_regime = 'BULL'
            self.db.update_regime(
                regime='BULL',
                master_st_value=float(latest['master_st']),
                child_st_value=float(latest['child_st']),
                master_direction=computed_master_dir,
                child_direction=int(latest['child_dir']),
                hard_sl_price=compute_hard_sl(float(latest['master_st']), 'BULL', sl_buffer, prev_hard_sl=0),
                regime_start_time=datetime.now().isoformat(),
            )
        elif current_regime == 'FLAT' and computed_master_dir != 0:
            # FLAT but master has a direction — adopt it
            new_regime = 'BULL' if computed_master_dir == 1 else 'BEAR'
            logger.warning(f"REGIME MISMATCH: DB says FLAT but computed master is {new_regime} — correcting")
            current_regime = new_regime
            self.db.update_regime(
                regime=new_regime,
                master_st_value=float(latest['master_st']),
                child_st_value=float(latest['child_st']),
                master_direction=computed_master_dir,
                child_direction=int(latest['child_dir']),
                hard_sl_price=compute_hard_sl(float(latest['master_st']), new_regime, sl_buffer, prev_hard_sl=0),
                regime_start_time=datetime.now().isoformat(),
            )

        # Detect signals
        signals = detect_signals(
            df, current_regime,
            hard_sl_buffer=cfg.get('hard_sl_buffer', 0),
            hard_sl_atr_mult=cfg.get('hard_sl_atr_mult', 1.0),
        )

        if not signals:
            # Trail hard SL with master SuperTrend even without signals
            prev_hard_sl = regime.get('hard_sl_price', 0) or 0
            new_hard_sl = compute_hard_sl(
                float(latest['master_st']), current_regime,
                sl_buffer,
                prev_hard_sl=prev_hard_sl,
            )
            self.db.update_regime(
                master_st_value=float(latest['master_st']),
                child_st_value=float(latest['child_st']),
                master_direction=int(latest['master_dir']),
                child_direction=int(latest['child_dir']),
                last_candle_time=str(latest.get('date', latest.name)),
                hard_sl_price=new_hard_sl,
            )
            if new_hard_sl != prev_hard_sl and prev_hard_sl > 0:
                logger.info(f"Trailing hard SL: {prev_hard_sl} -> {new_hard_sl} (buffer={sl_buffer:.1f})")

            # --- Signal candle refresh for unfilled pending triggers ---
            # No new flip, but if we have a PENDING futures position whose trigger
            # was never hit, refresh it with the current candle's trigger level.
            # This handles: original signal candle's low/high was never breached,
            # but a later candle in the same direction provides a better trigger.
            self._refresh_pending_triggers(latest, current_regime, sl_buffer)

            return []

        # Execute each signal
        all_results = []
        for signal in signals:
            counts = self.get_live_position_counts()
            active_fut = counts['futures_lots']
            active_opts = counts['short_options_lots']
            actions = determine_actions(
                signal, current_regime, active_fut,
                max_futures=cfg.get('max_futures_lots', 5),
                config=cfg,
                master_atr=master_atr,
                active_short_options_count=active_opts,
            )
            results = self.execute_actions(actions, signal)
            all_results.extend(results)

        return all_results

    def run_eod_protection(self, spot_price: float) -> List[str]:
        """
        End-of-day: Buy protective options for unprotected futures.
        Called around 3:00 PM.
        """
        regime = self.db.get_regime()
        current_regime = regime.get('regime', 'FLAT')
        if current_regime == 'FLAT':
            return []

        results = []

        # Use live Kite counts (source of truth)
        counts = self.get_live_position_counts()
        fut_count = counts['futures_lots']
        prot_count = counts['protective_options_lots']

        logger.info(
            f"[EODProtection] futures={fut_count}, protectives={prot_count} "
            f"[src={counts['source']}]"
        )

        # Need 1 protective per futures lot
        needed = fut_count - prot_count

        if needed <= 0:
            return ["All futures have protective options"]

        # Buy protective options
        if current_regime == 'BULL':
            opt_type = 'PE'  # Protective puts for long futures
        else:
            opt_type = 'CE'  # Protective calls for short futures

        for _ in range(needed):
            pos_id = self.place_protective_option(opt_type, spot_price, current_regime)
            if pos_id:
                results.append(f"Bought protective {opt_type} for futures hedge")

        return results

    def run_roll_check(self) -> List[str]:
        """
        Check if any positions need to be rolled to next expiry.
        Called daily around 3:15 PM.

        Uses Kite live positions to verify what actually exists before rolling.
        Only rolls positions that exist on BOTH DB and Kite.
        """
        self._ensure_contract_mgr()

        # Verify against Kite first — only roll what actually exists
        counts = self.get_live_position_counts()
        if counts['source'] != 'KITE':
            logger.warning("[RollCheck] Kite unavailable — skipping roll check")
            return ['Kite unavailable — roll check skipped']

        kite_fut_lots = counts['futures_lots']
        kite_short_opts = counts['short_options_by_symbol']
        kite_prot_opts = counts['protective_options_by_symbol']

        # Get DB positions for metadata (IDs, strikes, etc.)
        all_active = self.db.get_active_positions()
        rolls = self.contract_mgr.check_rolls_needed(all_active)

        results = []
        futures_rolled = 0
        shorts_rolled = 0
        protectives_rolled = 0

        for roll in rolls:
            pos_id = roll['position_id']
            pos = next((p for p in all_active if p['id'] == pos_id), None)
            if not pos:
                continue

            ts = pos.get('tradingsymbol', '')

            # Verify this position actually exists on Kite before rolling
            if pos['position_type'] == 'FUTURES':
                if futures_rolled >= kite_fut_lots:
                    logger.info(f"[RollCheck] Skipping futures #{pos_id} — already rolled {futures_rolled}/{kite_fut_lots}")
                    continue
                futures_rolled += 1
            elif pos['position_type'] == 'SHORT_OPTION':
                kite_lots_for_ts = kite_short_opts.get(ts, 0)
                if shorts_rolled >= kite_lots_for_ts:
                    logger.info(f"[RollCheck] Skipping short opt #{pos_id} {ts} — not on Kite")
                    continue
                shorts_rolled += 1
            elif pos['position_type'] == 'PROTECTIVE_OPTION':
                kite_lots_for_ts = kite_prot_opts.get(ts, 0)
                if protectives_rolled >= kite_lots_for_ts:
                    logger.info(f"[RollCheck] Skipping protective #{pos_id} {ts} — not on Kite")
                    continue
                protectives_rolled += 1

            logger.info(f"Rolling position {pos_id}: {roll['current_symbol']} → {roll['new_symbol']}")

            # Step 1: Close current (margin-safe: close first, open second)
            self._close_position(pos, exit_reason='ROLL')

            # Step 2: Open new position with next month contract
            if pos['position_type'] == 'FUTURES':
                regime = self.db.get_regime()
                current_hard_sl = regime.get('hard_sl_price', pos.get('sl_price', 0))
                self.place_futures_entry(
                    direction=pos['transaction_type'],
                    trigger_price=pos.get('entry_price', 0),
                    limit_price=pos.get('entry_price', 0),
                    hard_sl=current_hard_sl,
                    signal_type='ROLL',
                    regime=pos.get('regime', 'FLAT'),
                )
            elif pos['position_type'] == 'SHORT_OPTION':
                self.place_option_short(
                    option_type=pos.get('instrument_type', 'CE'),
                    strike=pos.get('strike', 0),
                    signal_type='ROLL',
                    regime=pos.get('regime', 'FLAT'),
                )
            elif pos['position_type'] == 'PROTECTIVE_OPTION':
                self.place_protective_option(
                    option_type=pos.get('instrument_type', 'PE'),
                    spot_price=pos.get('strike', 0) * 1.05,
                    regime=pos.get('regime', 'FLAT'),
                )

            results.append(f"Rolled {roll['current_symbol']} → {roll['new_symbol']} ({roll['reason']})")

        return results

    def on_trigger_fill(self, position: dict, fill_price: float) -> List[str]:
        """
        Post-fill actions when a pending trigger order gets executed.

        Called by ticker immediately on fill detection (paper or live).
        Actions:
          1. Place protective option (PE for BULL, CE for BEAR)
          2. Update hard SL for the new position
          3. Log the fill event

        Args:
            position: The position dict that just filled
            fill_price: The actual fill price
        """
        results = []
        cfg = self.config
        regime = self.db.get_regime()
        current_regime = regime.get('regime', 'FLAT')

        direction = position.get('transaction_type', '')
        tradingsymbol = position.get('tradingsymbol', '')

        logger.info(
            f"[PostFill] {direction} {tradingsymbol} filled @ {fill_price} | "
            f"Regime={current_regime}"
        )
        results.append(f"Trigger filled: {direction} {tradingsymbol} @ {fill_price}")

        # 1. Place protective option — only if protectives < futures (live count)
        try:
            counts = self.get_live_position_counts()
            if counts['protective_options_lots'] < counts['futures_lots']:
                if current_regime == 'BULL':
                    opt_type = 'PE'
                elif current_regime == 'BEAR':
                    opt_type = 'CE'
                else:
                    opt_type = None

                if opt_type:
                    pos_id = self.place_protective_option(opt_type, fill_price, current_regime)
                    if pos_id:
                        results.append(f"Protective {opt_type} placed for {tradingsymbol}")
                    else:
                        results.append(f"WARNING: Failed to place protective {opt_type}")
            else:
                results.append(
                    f"Protectives ({counts['protective_options_lots']}) already cover "
                    f"futures ({counts['futures_lots']}) — skipping [src={counts['source']}]"
                )
        except Exception as e:
            logger.error(f"[PostFill] Protective option error: {e}", exc_info=True)
            results.append(f"ERROR placing protective option: {e}")

        # 2. Update hard SL (recalculate from current master ST + ATR)
        try:
            master_st = regime.get('master_st', 0)
            atr = getattr(self, '_current_master_atr', cfg.get('hard_sl_buffer', 100))
            atr_mult = cfg.get('hard_sl_atr_mult', 1.0)
            buffer = atr_mult * atr if not cfg.get('hard_sl_buffer', 0) else cfg['hard_sl_buffer']

            if current_regime == 'BULL':
                hard_sl = master_st - buffer
            elif current_regime == 'BEAR':
                hard_sl = master_st + buffer
            else:
                hard_sl = 0

            if hard_sl > 0:
                self.db.update_regime(hard_sl_price=hard_sl)
                results.append(f"Hard SL set: {hard_sl:.1f}")
        except Exception as e:
            logger.error(f"[PostFill] Hard SL update error: {e}", exc_info=True)

        for r in results:
            logger.info(f"[PostFill] {r}")

        return results

    def run_verify_triggers(self, current_price: float) -> List[str]:
        """
        Check if any pending trigger orders have been filled.
        For paper mode: check if price crossed the trigger level.
        For live mode: query Kite order status.
        """
        pending = self.db.get_pending_positions()
        results = []

        for pos in pending:
            if self.config.get('paper_trading_mode'):
                # Paper mode: simulate fill if price crossed trigger
                trigger = pos.get('trigger_price', 0)
                filled = False
                if pos['transaction_type'] == 'BUY' and current_price >= trigger:
                    self.db.activate_position(pos['id'], trigger)
                    results.append(f"Paper fill: BUY {pos['tradingsymbol']} @ {trigger}")
                    filled = True
                elif pos['transaction_type'] == 'SELL' and current_price <= trigger:
                    self.db.activate_position(pos['id'], trigger)
                    results.append(f"Paper fill: SELL {pos['tradingsymbol']} @ {trigger}")
                    filled = True
                if filled:
                    self.on_trigger_fill(pos, fill_price=trigger)
            else:
                # Live mode: check Kite order status (fallback — ticker polls every 30s)
                if pos.get('kite_order_id'):
                    try:
                        kite = self._get_kite()
                        orders = kite.orders()
                        for order in orders:
                            if str(order['order_id']) == pos['kite_order_id']:
                                if order['status'] == 'COMPLETE':
                                    fill_price = order.get('average_price', pos.get('trigger_price', 0))
                                    self.db.activate_position(pos['id'], fill_price)
                                    results.append(f"Filled: {pos['tradingsymbol']} @ {fill_price}")
                                    self.on_trigger_fill(pos, fill_price=fill_price)
                                elif order['status'] in ('CANCELLED', 'REJECTED'):
                                    self.db.cancel_position(pos['id'])
                                    results.append(f"Cancelled/Rejected: {pos['tradingsymbol']}")
                                break
                    except Exception as e:
                        logger.error(f"Failed to verify order {pos['kite_order_id']}: {e}")

        return results

    # =========================================================================
    # Cross-Day Order Persistence
    # =========================================================================

    def re_place_pending_orders(self) -> List[str]:
        """
        Re-place pending trigger orders on Kite at market open.

        Zerodha cancels all unfilled SL-L orders at 3:30 PM EOD.
        If we have PENDING positions in DB from yesterday, we need to
        re-place them on Kite this morning with the same trigger/limit.

        Gap protection: If the market opens past the trigger price (gap through),
        we DON'T re-place the order. Instead, mark as GAP_PENDING and let
        handle_gap_entry() deal with it after 5 minutes.

        Called by scheduled job at 9:16 AM (just after market open).
        """
        results = []
        pending = self.db.get_pending_positions()
        if not pending:
            return results

        cfg = self.config
        is_paper = cfg.get('paper_trading_mode', True)

        # Get current LTP for gap detection
        current_ltp = 0
        try:
            from services.maruthi_ticker import get_maruthi_ticker
            ticker = get_maruthi_ticker()
            current_ltp = ticker._last_ltp
        except Exception:
            pass

        # Fallback: fetch LTP from Kite quote
        if current_ltp <= 0 and not is_paper:
            try:
                kite = self._get_kite()
                if kite:
                    quote = kite.quote([f"NSE:{cfg.get('symbol', 'MARUTI')}"])
                    current_ltp = quote.get(f"NSE:{cfg.get('symbol', 'MARUTI')}", {}).get('last_price', 0)
            except Exception:
                pass

        if not is_paper:
            kite = self._get_kite()
            if not kite:
                return ['Kite not authenticated — cannot re-place orders']

        for pos in pending:
            try:
                trigger = pos.get('trigger_price', 0)
                direction = pos.get('transaction_type', '')

                # --- Gap detection ---
                # SELL trigger: price should be ABOVE trigger (waiting to drop to it)
                #   Gap = LTP already BELOW trigger → market gapped past our level
                # BUY trigger: price should be BELOW trigger (waiting to rise to it)
                #   Gap = LTP already ABOVE trigger → market gapped past our level
                gap_detected = False
                if current_ltp > 0 and trigger > 0:
                    if direction == 'SELL' and current_ltp < trigger:
                        gap_detected = True
                    elif direction == 'BUY' and current_ltp > trigger:
                        gap_detected = True

                if gap_detected:
                    gap_pct = abs(current_ltp - trigger) / trigger * 100
                    self.db.update_position(pos['id'], status='GAP_PENDING')
                    results.append(
                        f"GAP DETECTED: {direction} trigger {trigger:.0f} but LTP={current_ltp:.0f} "
                        f"({gap_pct:.1f}% gap) — deferring to 9:21 AM gap handler"
                    )
                    logger.warning(
                        f"[GapProtect] #{pos['id']} {direction} trigger={trigger} "
                        f"LTP={current_ltp} gap={gap_pct:.1f}% — deferred to gap handler"
                    )
                    continue

                # --- Normal re-placement (no gap) ---
                if is_paper:
                    results.append(f"Paper mode — pending {direction} @ {trigger} remains in DB")
                    continue

                # Get current futures contract (may have changed month)
                self._ensure_contract_mgr()
                fut = self.contract_mgr.get_futures_symbol()
                if not fut:
                    results.append(f"No futures contract — skipping {pos['tradingsymbol']}")
                    continue

                tradingsymbol = fut['tradingsymbol']
                lot_size = fut.get('lot_size', cfg.get('lot_size', 50))

                # Update tradingsymbol if expiry rolled
                if tradingsymbol != pos.get('tradingsymbol'):
                    self.db.update_position(pos['id'],
                                            tradingsymbol=tradingsymbol,
                                            qty=lot_size)
                    logger.info(f"Updated pending #{pos['id']}: {pos['tradingsymbol']} → {tradingsymbol}")

                # Place fresh SL-L order on Kite
                kite_order_id = self._place_order_kite({
                    'exchange': cfg.get('exchange_fo', 'NFO'),
                    'tradingsymbol': tradingsymbol,
                    'transaction_type': direction,
                    'quantity': lot_size,
                    'product': 'NRML',
                    'order_type': 'SL',
                    'price': trigger - 5 if direction == 'SELL' else trigger + 5,
                    'trigger_price': trigger,
                })

                if kite_order_id:
                    self.db.update_position(pos['id'], kite_order_id=kite_order_id)
                    results.append(f"Re-placed {direction} trigger @ {trigger} → {kite_order_id}")
                    logger.info(f"Re-placed pending #{pos['id']} on Kite: {kite_order_id}")
                else:
                    results.append(f"Failed to re-place {pos['tradingsymbol']}")
            except Exception as e:
                logger.error(f"Failed to re-place pending #{pos['id']}: {e}")
                results.append(f"Error re-placing #{pos['id']}: {e}")

        return results

    def handle_gap_entry(self) -> List[str]:
        """
        Handle GAP_PENDING positions after the first 5 minutes of market open.

        Called at 9:21 AM (5 mins after market open). For each GAP_PENDING position:
        1. Check if the gap is still unfilled (price hasn't retraced back past trigger)
        2. Check if the signal direction is still valid (regime hasn't changed)
        3. If both: enter at MARKET + place full hedges immediately
           (protective option + short option — same as EOD + child signal combo)
        4. If gap filled or signal reversed: cancel the position

        This prevents entering at a terrible gap price that might retrace,
        while still entering if the move is genuine after 5 mins of confirmation.
        """
        results = []
        cfg = self.config

        # Get GAP_PENDING positions
        try:
            conn = self.db._get_conn()
            rows = conn.execute(
                "SELECT * FROM maruthi_positions WHERE status = 'GAP_PENDING' ORDER BY id"
            ).fetchall()
            gap_positions = [dict(r) for r in rows]
            conn.close()
        except Exception as e:
            logger.error(f"[GapHandler] DB error: {e}")
            return [f"DB error: {e}"]

        if not gap_positions:
            return ['No GAP_PENDING positions']

        # Get current state
        regime = self.db.get_regime()
        current_regime = regime.get('regime', 'FLAT')

        # Get current LTP
        current_ltp = 0
        try:
            from services.maruthi_ticker import get_maruthi_ticker
            ticker = get_maruthi_ticker()
            current_ltp = ticker._last_ltp
        except Exception:
            pass

        if current_ltp <= 0:
            try:
                kite = self._get_kite()
                if kite:
                    quote = kite.quote([f"NSE:{cfg.get('symbol', 'MARUTI')}"])
                    current_ltp = quote.get(f"NSE:{cfg.get('symbol', 'MARUTI')}", {}).get('last_price', 0)
            except Exception:
                pass

        if current_ltp <= 0:
            return ['Cannot get LTP — gap handler deferred']

        for pos in gap_positions:
            trigger = pos.get('trigger_price', 0)
            direction = pos.get('transaction_type', '')
            pos_id = pos['id']

            # Check 1: Is the gap still unfilled?
            # SELL: gap was LTP < trigger. Gap unfilled = LTP still below trigger
            # BUY: gap was LTP > trigger. Gap unfilled = LTP still above trigger
            gap_still_open = False
            if direction == 'SELL' and current_ltp < trigger:
                gap_still_open = True
            elif direction == 'BUY' and current_ltp > trigger:
                gap_still_open = True

            if not gap_still_open:
                # Gap filled — price retraced back. Cancel the position.
                self.db.cancel_position(pos_id)
                results.append(
                    f"GAP FILLED: {direction} trigger={trigger:.0f} LTP={current_ltp:.0f} "
                    f"— price retraced, position cancelled"
                )
                logger.info(f"[GapHandler] #{pos_id} gap filled, cancelled")
                continue

            # Check 2: Is the signal direction still valid?
            # SELL pending = BEAR regime should still be active
            # BUY pending = BULL regime should still be active
            signal_valid = False
            if direction == 'SELL' and current_regime == 'BEAR':
                signal_valid = True
            elif direction == 'BUY' and current_regime == 'BULL':
                signal_valid = True

            if not signal_valid:
                self.db.cancel_position(pos_id)
                results.append(
                    f"SIGNAL INVALID: {direction} but regime={current_regime} "
                    f"— position cancelled"
                )
                logger.info(f"[GapHandler] #{pos_id} signal invalid (regime={current_regime}), cancelled")
                continue

            # Both checks passed — enter at MARKET with full hedges
            logger.info(
                f"[GapHandler] #{pos_id} gap confirmed after 5 min: "
                f"{direction} LTP={current_ltp:.0f} (trigger was {trigger:.0f})"
            )

            # Activate the position at current LTP (MARKET entry)
            self.db.activate_position(pos_id, entry_price=current_ltp)
            results.append(f"GAP ENTRY: {direction} @ {current_ltp:.0f} (trigger was {trigger:.0f})")

            # Place MARKET order on Kite (live mode)
            if cfg.get('live_trading_enabled') and not cfg.get('paper_trading_mode'):
                self._ensure_contract_mgr()
                fut = self.contract_mgr.get_futures_symbol()
                if fut:
                    kite_order_id = self._place_order_kite({
                        'exchange': cfg.get('exchange_fo', 'NFO'),
                        'tradingsymbol': fut['tradingsymbol'],
                        'transaction_type': direction,
                        'quantity': fut.get('lot_size', cfg.get('lot_size', 50)),
                        'product': 'NRML',
                        'order_type': 'MARKET',
                    })
                    if kite_order_id:
                        self.db.update_position(pos_id, kite_order_id=kite_order_id)
                        results.append(f"MARKET order placed: {kite_order_id}")

            # Place hedges only if position counts allow it
            # Rule: 1 short option per 1 futures lot — guardrails enforce this
            counts = self.get_live_position_counts()
            active_fut = counts['futures_lots']
            active_opts = counts['short_options_lots']

            if current_regime == 'BEAR':
                short_type = 'PE'  # Short put to collect premium
            else:
                short_type = 'CE'  # Short call to collect premium

            # Short option only if we have more futures than short options
            if active_opts < active_fut:
                strike_interval = cfg.get('strike_interval', 100)
                if short_type == 'PE':
                    short_strike = float(int(current_ltp // strike_interval) * strike_interval - strike_interval)
                else:
                    short_strike = float(int(-(-current_ltp // strike_interval)) * strike_interval + strike_interval)

                short_id = self.place_option_short(short_type, short_strike,
                                                    signal_type='GAP_ENTRY', regime=current_regime)
                if short_id:
                    results.append(f"Short {short_type} {short_strike} placed")
            else:
                results.append(f"Short options ({active_opts}) already match futures ({active_fut}) — skipping")

            # Update hard SL for the new entry
            master_st = regime.get('master_st_value', 0)
            atr = getattr(self, '_current_master_atr', 100)
            sl_buffer = resolve_sl_buffer(cfg, atr)
            hard_sl = compute_hard_sl(master_st, current_regime, sl_buffer, prev_hard_sl=0)
            if hard_sl > 0:
                self.db.update_regime(hard_sl_price=hard_sl)
                results.append(f"Hard SL set: {hard_sl:.1f}")

        for r in results:
            logger.info(f"[GapHandler] {r}")

        return results

    # =========================================================================
    # Position Reconciliation
    # =========================================================================

    def sync_positions_with_kite(self) -> List[str]:
        """
        Reconcile DB positions with actual Kite positions.

        Compares MARUTI positions on Kite (net qty) against DB active positions.
        Adds missing positions, marks closed ones. This ensures the DB always
        reflects reality, even if orders were placed manually or DB writes failed.

        Called at the start of each candle check.
        """
        results = []
        cfg = self.config

        if cfg.get('paper_trading_mode'):
            return results

        try:
            kite = self._get_kite()
            if not kite:
                return ['Kite not authenticated — skipping position sync']
        except Exception:
            return ['Kite not available — skipping position sync']

        try:
            kite_positions = kite.positions().get('net', [])
        except Exception as e:
            logger.error(f"[PositionSync] Failed to fetch Kite positions: {e}")
            return [f"Position sync failed: {e}"]

        symbol = cfg.get('symbol', 'MARUTI')
        lot_size = cfg.get('lot_size', 50)

        # Parse Kite positions for MARUTI instruments
        kite_fut_qty = 0  # Negative = short
        kite_options = {}  # tradingsymbol → qty (negative = short)

        for p in kite_positions:
            ts = p.get('tradingsymbol', '')
            qty = p.get('quantity', 0)
            if qty == 0:
                continue
            if symbol not in ts:
                continue

            if 'FUT' in ts:
                kite_fut_qty += qty
            elif 'PE' in ts or 'CE' in ts:
                kite_options[ts] = kite_options.get(ts, 0) + qty

        # Count DB positions
        db_futures = self.db.get_active_positions('FUTURES')
        db_short_opts = self.db.get_active_positions('SHORT_OPTION')

        db_fut_lots = len(db_futures)  # Each row = 1 lot
        db_fut_qty = sum(
            p.get('qty', lot_size) * (-1 if p.get('transaction_type') == 'SELL' else 1)
            for p in db_futures
        )

        # --- Futures reconciliation ---
        kite_fut_lots = abs(kite_fut_qty) // lot_size
        db_fut_lots_count = len(db_futures)

        if kite_fut_lots != db_fut_lots_count:
            diff = kite_fut_lots - db_fut_lots_count
            direction = 'SELL' if kite_fut_qty < 0 else 'BUY'
            regime = self.db.get_regime()
            current_regime = regime.get('regime', 'FLAT')

            if diff > 0:
                # Kite has MORE futures than DB — add missing
                for i in range(diff):
                    avg_price = abs(kite_fut_qty) and abs(
                        next((p['average_price'] for p in kite_positions
                              if 'FUT' in p.get('tradingsymbol', '') and symbol in p.get('tradingsymbol', '')),
                             0)
                    )
                    # Find the futures tradingsymbol from Kite
                    fut_ts = next(
                        (p['tradingsymbol'] for p in kite_positions
                         if 'FUT' in p.get('tradingsymbol', '') and symbol in p.get('tradingsymbol', '')),
                        f'{symbol}FUT'
                    )
                    pos_id = self.db.add_position(
                        position_type='FUTURES',
                        tradingsymbol=fut_ts,
                        exchange=cfg.get('exchange_fo', 'NFO'),
                        transaction_type=direction,
                        qty=lot_size,
                        entry_price=avg_price,
                        entry_time=datetime.now().isoformat(),
                        regime=current_regime,
                        signal_type='SYNC',
                        status='ACTIVE',
                    )
                    msg = f"[PositionSync] Added missing futures #{pos_id}: {direction} {fut_ts} (Kite has {kite_fut_lots} lots, DB had {db_fut_lots_count})"
                    logger.warning(msg)
                    results.append(msg)

            elif diff < 0:
                # DB has MORE futures than Kite — mark excess as closed
                excess = abs(diff)
                for pos in reversed(db_futures[:excess]):
                    self.db.close_position(pos['id'], exit_price=0, exit_reason='SYNC_REMOVED')
                    msg = f"[PositionSync] Closed stale futures #{pos['id']} (not on Kite)"
                    logger.warning(msg)
                    results.append(msg)

        # --- Short options reconciliation ---
        # Group DB options by tradingsymbol
        db_opt_by_ts = {}
        for p in db_short_opts:
            ts = p.get('tradingsymbol', '')
            db_opt_by_ts[ts] = db_opt_by_ts.get(ts, 0) + 1

        # Compare with Kite
        for ts, kite_qty in kite_options.items():
            if kite_qty >= 0:  # Not a short position
                continue
            kite_lots = abs(kite_qty) // lot_size
            db_lots = db_opt_by_ts.get(ts, 0)

            if kite_lots > db_lots:
                diff = kite_lots - db_lots
                regime = self.db.get_regime()
                opt_type = 'PE' if 'PE' in ts else 'CE'
                for i in range(diff):
                    avg_price = abs(
                        next((p['average_price'] for p in kite_positions
                              if p.get('tradingsymbol') == ts), 0)
                    )
                    pos_id = self.db.add_position(
                        position_type='SHORT_OPTION',
                        tradingsymbol=ts,
                        exchange=cfg.get('exchange_fo', 'NFO'),
                        transaction_type='SELL',
                        qty=lot_size,
                        entry_price=avg_price,
                        entry_time=datetime.now().isoformat(),
                        regime=regime.get('regime', 'FLAT'),
                        signal_type='SYNC',
                        status='ACTIVE',
                    )
                    msg = f"[PositionSync] Added missing {opt_type} #{pos_id}: {ts} (Kite has {kite_lots}, DB had {db_lots})"
                    logger.warning(msg)
                    results.append(msg)

        # Check for DB options not on Kite
        for ts, db_count in db_opt_by_ts.items():
            kite_lots = abs(kite_options.get(ts, 0)) // lot_size
            if kite_lots < db_count:
                excess = db_count - kite_lots
                matching = [p for p in db_short_opts if p.get('tradingsymbol') == ts]
                for pos in reversed(matching[:excess]):
                    self.db.close_position(pos['id'], exit_price=0, exit_reason='SYNC_REMOVED')
                    msg = f"[PositionSync] Closed stale option #{pos['id']} {ts} (not on Kite)"
                    logger.warning(msg)
                    results.append(msg)

        if results:
            for r in results:
                logger.info(r)
        else:
            logger.debug("[PositionSync] DB matches Kite — no changes needed")

        return results

    # =========================================================================
    # Emergency
    # =========================================================================

    def emergency_exit_all(self) -> int:
        """Kill switch — close everything at market."""
        return self.close_all_positions(exit_reason='EMERGENCY', keep_last_option=False)

    # =========================================================================
    # State Query
    # =========================================================================

    def get_state(self) -> dict:
        """Get full strategy state for dashboard. Uses Kite for live position data."""
        regime = self.db.get_regime()

        # Live position counts from Kite (source of truth for numbers)
        live_counts = self.get_live_position_counts()

        # DB positions for display details (tradingsymbol, entry price, etc.)
        futures = self.db.get_active_positions('FUTURES')
        short_opts = self.db.get_active_positions('SHORT_OPTION')
        protectives = self.db.get_active_positions('PROTECTIVE_OPTION')
        pending = self.db.get_pending_positions()
        stats = self.db.get_stats()
        recent_signals = self.db.get_recent_signals(20)
        recent_trades = self.db.get_recent_trades(20)

        # Get ticker status if available
        ticker_status = {}
        try:
            from services.maruthi_ticker import get_maruthi_ticker
            ticker = get_maruthi_ticker()
            ticker_status = ticker.get_status()
        except Exception:
            pass

        # Use live ATR if available, otherwise instance value
        master_atr = self._current_master_atr
        sl_buffer = resolve_sl_buffer(self.config, master_atr)

        return {
            'regime': regime,
            'futures': futures,
            'short_options': short_opts,
            'protective_options': protectives,
            'pending_orders': pending,
            'stats': stats,
            'recent_signals': recent_signals,
            'recent_trades': recent_trades,
            'ticker': ticker_status,
            'config': {
                'paper_mode': self.config.get('paper_trading_mode', True),
                'paper_trading_mode': self.config.get('paper_trading_mode', True),
                'live_enabled': self.config.get('live_trading_enabled', False),
                'enabled': self.config.get('enabled', True),
                'max_futures': self.config.get('max_futures_lots', 5),
                'hard_sl_atr_mult': self.config.get('hard_sl_atr_mult', 1.0),
                'hard_sl_buffer_pts': round(sl_buffer, 1),
                'master_atr': round(master_atr, 1),
            },
            'live_counts': live_counts,
        }


# Singleton
_executor_instance = None


def get_maruthi_executor(config: dict = None) -> MaruthiExecutor:
    """Get or create the singleton MaruthiExecutor."""
    global _executor_instance
    if _executor_instance is None:
        _executor_instance = MaruthiExecutor(config)
    elif config:
        _executor_instance.config = config
    return _executor_instance
