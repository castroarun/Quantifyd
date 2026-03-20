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
    detect_signals, compute_hard_sl, resolve_sl_buffer
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

    # =========================================================================
    # Guardrails
    # =========================================================================

    def _check_guardrails(self, is_entry: bool = True) -> tuple:
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

        # Max futures check (entry only)
        if is_entry:
            max_fut = cfg.get('max_futures_lots', 5)
            active_fut = self.db.get_active_futures_count()
            if active_fut >= max_fut:
                return False, f"Max futures lots reached ({active_fut}/{max_fut})"

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
        passed, reason = self._check_guardrails(is_entry=False)
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
        self._ensure_contract_mgr()
        cfg = self.config
        lot_size = cfg.get('lot_size', 200)
        otm_pct = cfg.get('protective_otm_pct', 0.05)
        strike_interval = cfg.get('strike_interval', 100)

        # Calculate strike
        if option_type == 'PE':
            target = spot_price * (1 - otm_pct)
            strike = float(int(target // strike_interval) * strike_interval)
        else:
            target = spot_price * (1 + otm_pct)
            strike = float(int(-(-target // strike_interval)) * strike_interval)

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
        """Close a single position. Uses MARKET for futures, LIMIT with retry for options."""
        cfg = self.config

        if exit_price is None:
            exit_price = pos.get('entry_price', 0)  # Fallback

        # Determine close direction
        if pos['transaction_type'] == 'BUY':
            close_direction = 'SELL'
        else:
            close_direction = 'BUY'

        if cfg.get('live_trading_enabled') and not cfg.get('paper_trading_mode'):
            is_option = pos.get('position_type') in ('SHORT_OPTION', 'PROTECTIVE_OPTION')

            if is_option:
                # Stock options: LIMIT with fill-check-revise
                kite_order_id = self._place_option_with_retry({
                    'exchange': pos.get('exchange', 'NFO'),
                    'tradingsymbol': pos['tradingsymbol'],
                    'transaction_type': close_direction,
                    'quantity': pos['qty'],
                    'product': 'NRML',
                })
            else:
                # Futures: MARKET is fine
                kite_order_id = self._place_order_kite({
                    'exchange': pos.get('exchange', 'NFO'),
                    'tradingsymbol': pos['tradingsymbol'],
                    'transaction_type': close_direction,
                    'quantity': pos['qty'],
                    'product': 'NRML',
                    'order_type': 'MARKET',
                })
            if not kite_order_id:
                logger.error(f"Failed to close position {pos['id']}: {pos['tradingsymbol']}")
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

            elif act == 'BUY_FUTURES':
                pos_id = self.place_futures_entry(
                    direction='BUY',
                    trigger_price=action['trigger_price'],
                    limit_price=action['limit_price'],
                    hard_sl=action['hard_sl'],
                    signal_type=signal.signal_type,
                    regime=current_regime,
                )
                if pos_id:
                    results.append(f"Buy futures trigger @ {action['trigger_price']}")

            elif act == 'SHORT_FUTURES':
                pos_id = self.place_futures_entry(
                    direction='SELL',
                    trigger_price=action['trigger_price'],
                    limit_price=action['limit_price'],
                    hard_sl=action['hard_sl'],
                    signal_type=signal.signal_type,
                    regime=current_regime,
                )
                if pos_id:
                    results.append(f"Short futures trigger @ {action['trigger_price']}")

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
            last_signal_time=datetime.now().isoformat(),
            last_candle_time=signal.candle.get('time'),
        )

        return results

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
                master_direction=int(latest['master_dir']),
                child_direction=int(latest['child_dir']),
                last_candle_time=str(latest.get('date', latest.name)),
                hard_sl_price=new_hard_sl,
            )
            if new_hard_sl != prev_hard_sl and prev_hard_sl > 0:
                logger.info(f"Trailing hard SL: {prev_hard_sl} -> {new_hard_sl} (buffer={sl_buffer:.1f})")
            return []

        # Execute each signal
        all_results = []
        for signal in signals:
            active_fut = self.db.get_active_futures_count()
            actions = determine_actions(
                signal, current_regime, active_fut,
                max_futures=cfg.get('max_futures_lots', 5),
                config=cfg,
                master_atr=master_atr,
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
        futures = self.db.get_active_positions('FUTURES')
        protectives = self.db.get_active_positions('PROTECTIVE_OPTION')

        # Count existing protections
        prot_count = len(protectives)
        fut_count = len(futures)

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
        """
        self._ensure_contract_mgr()
        all_active = self.db.get_active_positions()
        rolls = self.contract_mgr.check_rolls_needed(all_active)

        results = []
        for roll in rolls:
            pos_id = roll['position_id']
            pos = next((p for p in all_active if p['id'] == pos_id), None)
            if not pos:
                continue

            logger.info(f"Rolling position {pos_id}: {roll['current_symbol']} → {roll['new_symbol']}")

            # Step 1: Close current (margin-safe: close first, open second)
            self._close_position(pos, exit_reason='ROLL')

            # Step 2: Open new position with next month contract
            # Re-enter same direction with new symbol, carry forward trailing SL
            if pos['position_type'] == 'FUTURES':
                # Use current regime's trailing SL (not the old position's SL)
                regime = self.db.get_regime()
                current_hard_sl = regime.get('hard_sl_price', pos.get('sl_price', 0))
                self.place_futures_entry(
                    direction=pos['transaction_type'],
                    trigger_price=pos.get('entry_price', 0),  # Enter at market
                    limit_price=pos.get('entry_price', 0),
                    hard_sl=current_hard_sl,
                    signal_type='ROLL',
                    regime=pos.get('regime', 'FLAT'),
                )
            elif pos['position_type'] in ('SHORT_OPTION', 'PROTECTIVE_OPTION'):
                if pos['position_type'] == 'SHORT_OPTION':
                    self.place_option_short(
                        option_type=pos.get('instrument_type', 'CE'),
                        strike=pos.get('strike', 0),
                        signal_type='ROLL',
                        regime=pos.get('regime', 'FLAT'),
                    )
                else:
                    self.place_protective_option(
                        option_type=pos.get('instrument_type', 'PE'),
                        spot_price=pos.get('strike', 0) * 1.05,  # Approximate spot
                        regime=pos.get('regime', 'FLAT'),
                    )

            results.append(f"Rolled {roll['current_symbol']} → {roll['new_symbol']} ({roll['reason']})")

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
                if pos['transaction_type'] == 'BUY' and current_price >= trigger:
                    self.db.activate_position(pos['id'], trigger)
                    results.append(f"Paper fill: BUY {pos['tradingsymbol']} @ {trigger}")
                elif pos['transaction_type'] == 'SELL' and current_price <= trigger:
                    self.db.activate_position(pos['id'], trigger)
                    results.append(f"Paper fill: SELL {pos['tradingsymbol']} @ {trigger}")
            else:
                # Live mode: check Kite order status
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
                                elif order['status'] in ('CANCELLED', 'REJECTED'):
                                    self.db.cancel_position(pos['id'])
                                    results.append(f"Cancelled/Rejected: {pos['tradingsymbol']}")
                                break
                    except Exception as e:
                        logger.error(f"Failed to verify order {pos['kite_order_id']}: {e}")

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
        """Get full strategy state for dashboard."""
        regime = self.db.get_regime()
        futures = self.db.get_active_positions('FUTURES')
        short_opts = self.db.get_active_positions('SHORT_OPTION')
        protectives = self.db.get_active_positions('PROTECTIVE_OPTION')
        pending = self.db.get_pending_positions()
        stats = self.db.get_stats()
        recent_signals = self.db.get_recent_signals(20)
        recent_trades = self.db.get_recent_trades(20)

        return {
            'regime': regime,
            'futures': futures,
            'short_options': short_opts,
            'protective_options': protectives,
            'pending_orders': pending,
            'stats': stats,
            'recent_signals': recent_signals,
            'recent_trades': recent_trades,
            'config': {
                'paper_mode': self.config.get('paper_trading_mode', True),
                'live_enabled': self.config.get('live_trading_enabled', False),
                'max_futures': self.config.get('max_futures_lots', 5),
                'hard_sl_atr_mult': self.config.get('hard_sl_atr_mult', 1.0),
                'hard_sl_buffer_pts': round(resolve_sl_buffer(self.config, self._current_master_atr), 1),
                'master_atr': round(self._current_master_atr, 1),
            },
        }
