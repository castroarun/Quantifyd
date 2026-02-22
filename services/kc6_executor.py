"""
KC6 Order Executor
====================

Handles order placement for KC6 mean reversion strategy.
All safety guardrails live here - 10-point pre-order checks.

Supports paper trading mode (logs orders without Kite API calls)
and live trading mode (places real CNC orders via Kite).
"""

import logging
from datetime import datetime, time as dtime
from typing import Dict, List, Optional

from config import KC6_DEFAULTS
from services.kc6_db import get_kc6_db
from services.kite_service import get_kite, is_authenticated

logger = logging.getLogger(__name__)


class KC6Executor:
    """Order executor with safety guardrails for KC6 strategy."""

    def __init__(self, config: dict = None, capital: float = 1_000_000):
        self.config = config or KC6_DEFAULTS.copy()
        self.capital = capital
        self.db = get_kc6_db()

    # =========================================================================
    # Guardrail Checks
    # =========================================================================

    def _check_guardrails(self, is_entry: bool = True,
                          universe_atr_ratio: float = None,
                          symbol: str = None) -> tuple:
        """
        Run all pre-order safety checks.
        Returns (passed: bool, reason: str).
        """
        cfg = self.config

        # 1. Live trading must be enabled
        if not cfg.get('live_trading_enabled', False):
            if not cfg.get('paper_trading_mode', True):
                return False, "Neither live_trading_enabled nor paper_trading_mode is True"

        # 2. Paper mode check (not a failure - just routes differently)
        # This is handled in place_entry_order / place_exit_order

        # 3. Kite API authenticated (only matters for live mode)
        if cfg.get('live_trading_enabled') and not cfg.get('paper_trading_mode'):
            if not is_authenticated():
                return False, "Kite API not authenticated"

        # 4. Market hours (9:15 AM - 3:30 PM IST, weekdays)
        now = datetime.now()
        if now.weekday() >= 5:  # Saturday/Sunday
            return False, f"Market closed (weekend: {now.strftime('%A')})"
        market_open = dtime(9, 15)
        market_close = dtime(15, 30)
        current_time = now.time()
        if current_time < market_open or current_time > market_close:
            return False, f"Market closed (current time: {current_time})"

        # 5. Daily order count limit
        max_orders = cfg.get('max_daily_orders', 5)
        today_orders = self.db.get_today_order_count()
        if today_orders >= max_orders:
            return False, f"Daily order limit reached ({today_orders}/{max_orders})"

        # 6. Daily loss limit
        max_loss_pct = cfg.get('max_daily_loss_pct', 3.0)
        today_loss = self.db.get_today_loss_pct(self.capital)
        if today_loss >= max_loss_pct:
            return False, f"Daily loss limit reached ({today_loss:.1f}% >= {max_loss_pct}%)"

        # Entry-specific checks
        if is_entry:
            # 7. Max positions check
            max_pos = cfg.get('max_positions', 5)
            active_count = self.db.get_active_positions_count()
            if active_count >= max_pos:
                return False, f"Max positions reached ({active_count}/{max_pos})"

            # 8. Crash filter check
            if universe_atr_ratio is not None:
                threshold = cfg.get('atr_ratio_threshold', 1.3)
                if universe_atr_ratio >= threshold:
                    return False, f"Crash filter active (ATR ratio {universe_atr_ratio:.3f} >= {threshold})"

            # 9. Symbol in Nifty 500 universe
            if symbol:
                try:
                    from services.nifty500_universe import get_nifty500
                    universe = get_nifty500()
                    if symbol not in universe.symbols:
                        return False, f"{symbol} not in Nifty 500 universe"
                except Exception:
                    pass  # Skip universe check if can't load

            # 10. No duplicate position
            if symbol:
                existing = self.db.get_position_by_symbol(symbol)
                if existing:
                    return False, f"Already holding {symbol} (position #{existing['id']})"

        return True, "All checks passed"

    # =========================================================================
    # Position Sizing
    # =========================================================================

    def calculate_qty(self, price: float) -> int:
        """Calculate number of shares to buy based on position size config."""
        position_pct = self.config.get('position_size_pct', 0.10)
        allocation = self.capital * position_pct
        qty = int(allocation / price)
        return max(qty, 1)

    # =========================================================================
    # Order Placement
    # =========================================================================

    def place_entry_order(self, symbol: str, limit_price: float,
                          universe_atr_ratio: float = None,
                          kc6_mid: float = None,
                          sma200: float = None) -> Dict:
        """
        Place a BUY CNC limit order for entry.

        In paper mode: logs to DB with status='PAPER', creates position.
        In live mode: calls kite.place_order(), logs to DB, creates position on fill.
        """
        # Run guardrails
        passed, reason = self._check_guardrails(
            is_entry=True,
            universe_atr_ratio=universe_atr_ratio,
            symbol=symbol
        )

        if not passed:
            logger.warning(f"Entry BLOCKED for {symbol}: {reason}")
            return {'status': 'BLOCKED', 'reason': reason, 'symbol': symbol}

        qty = self.calculate_qty(limit_price)
        sl_pct = self.config.get('sl_pct', 5.0)
        tp_pct = self.config.get('tp_pct', 15.0)
        sl_price = round(limit_price * (1 - sl_pct / 100), 2)
        tp_price = round(limit_price * (1 + tp_pct / 100), 2)
        today = datetime.now().strftime('%Y-%m-%d')

        is_paper = self.config.get('paper_trading_mode', True)

        if is_paper:
            # Paper trading: log order and create position immediately
            order_id = self.db.log_order(
                symbol=symbol, side='BUY', qty=qty, price=limit_price,
                status='PAPER', exit_reason=None
            )
            position_id = self.db.add_position(
                symbol=symbol, entry_price=limit_price, entry_date=today,
                qty=qty, sl_price=sl_price, tp_price=tp_price,
                kc6_mid=kc6_mid, sma200=sma200
            )
            logger.info(f"[PAPER] BUY {qty} {symbol} @ {limit_price} | SL={sl_price} TP={tp_price}")
            return {
                'status': 'PAPER',
                'symbol': symbol,
                'qty': qty,
                'price': limit_price,
                'sl_price': sl_price,
                'tp_price': tp_price,
                'order_id': order_id,
                'position_id': position_id,
            }

        # Live trading: place order via Kite
        try:
            kite = get_kite()
            kite_order_id = kite.place_order(
                variety='regular',
                exchange='NSE',
                tradingsymbol=symbol,
                transaction_type='BUY',
                quantity=qty,
                product='CNC',
                order_type='LIMIT',
                price=limit_price,
            )

            order_id = self.db.log_order(
                symbol=symbol, side='BUY', qty=qty, price=limit_price,
                status='PLACED', kite_order_id=str(kite_order_id)
            )

            # Create position (will verify fill in verify_orders)
            position_id = self.db.add_position(
                symbol=symbol, entry_price=limit_price, entry_date=today,
                qty=qty, sl_price=sl_price, tp_price=tp_price,
                kc6_mid=kc6_mid, sma200=sma200
            )

            logger.info(f"[LIVE] BUY {qty} {symbol} @ {limit_price} | Kite #{kite_order_id}")
            return {
                'status': 'PLACED',
                'symbol': symbol,
                'qty': qty,
                'price': limit_price,
                'sl_price': sl_price,
                'tp_price': tp_price,
                'kite_order_id': str(kite_order_id),
                'order_id': order_id,
                'position_id': position_id,
            }

        except Exception as e:
            error_msg = str(e)
            logger.error(f"[LIVE] BUY FAILED for {symbol}: {error_msg}")
            self.db.log_order(
                symbol=symbol, side='BUY', qty=qty, price=limit_price,
                status='FAILED', error_message=error_msg
            )
            return {'status': 'FAILED', 'symbol': symbol, 'error': error_msg}

    def place_exit_order(self, position: Dict, exit_price: float,
                         exit_reason: str) -> Dict:
        """
        Place a SELL CNC limit order for exit.

        In paper mode: logs order, closes position immediately.
        In live mode: calls kite.place_order(), logs order.
        """
        passed, reason = self._check_guardrails(is_entry=False)

        if not passed:
            logger.warning(f"Exit BLOCKED for {position['symbol']}: {reason}")
            return {'status': 'BLOCKED', 'reason': reason, 'symbol': position['symbol']}

        symbol = position['symbol']
        qty = position['qty']
        position_id = position['id']
        today = datetime.now().strftime('%Y-%m-%d')

        is_paper = self.config.get('paper_trading_mode', True)

        if is_paper:
            order_id = self.db.log_order(
                symbol=symbol, side='SELL', qty=qty, price=exit_price,
                status='PAPER', position_id=position_id, exit_reason=exit_reason
            )
            trade = self.db.close_position(position_id, exit_price, today, exit_reason)
            logger.info(
                f"[PAPER] SELL {qty} {symbol} @ {exit_price} | "
                f"Reason={exit_reason} | P/L={trade['pnl_pct']:+.2f}%"
            )
            return {
                'status': 'PAPER',
                'symbol': symbol,
                'qty': qty,
                'price': exit_price,
                'exit_reason': exit_reason,
                'pnl_pct': trade['pnl_pct'],
                'pnl_abs': trade['pnl_abs'],
                'order_id': order_id,
            }

        # Live trading
        try:
            kite = get_kite()

            # For SL and Max Hold, use MARKET order for guaranteed fill
            order_type = 'MARKET' if exit_reason in ('STOP_LOSS', 'MAX_HOLD', 'EMERGENCY') else 'LIMIT'

            order_params = dict(
                variety='regular',
                exchange='NSE',
                tradingsymbol=symbol,
                transaction_type='SELL',
                quantity=qty,
                product='CNC',
                order_type=order_type,
            )
            if order_type == 'LIMIT':
                order_params['price'] = exit_price

            kite_order_id = kite.place_order(**order_params)

            order_id = self.db.log_order(
                symbol=symbol, side='SELL', qty=qty, price=exit_price,
                order_type=order_type, status='PLACED',
                kite_order_id=str(kite_order_id),
                position_id=position_id, exit_reason=exit_reason
            )

            # Close position in DB
            trade = self.db.close_position(position_id, exit_price, today, exit_reason)

            logger.info(
                f"[LIVE] SELL {qty} {symbol} @ {exit_price} ({order_type}) | "
                f"Reason={exit_reason} | Kite #{kite_order_id}"
            )
            return {
                'status': 'PLACED',
                'symbol': symbol,
                'qty': qty,
                'price': exit_price,
                'exit_reason': exit_reason,
                'pnl_pct': trade['pnl_pct'],
                'kite_order_id': str(kite_order_id),
                'order_id': order_id,
            }

        except Exception as e:
            error_msg = str(e)
            logger.error(f"[LIVE] SELL FAILED for {symbol}: {error_msg}")
            self.db.log_order(
                symbol=symbol, side='SELL', qty=qty, price=exit_price,
                status='FAILED', position_id=position_id,
                exit_reason=exit_reason, error_message=error_msg
            )
            return {'status': 'FAILED', 'symbol': symbol, 'error': error_msg}

    # =========================================================================
    # Position Sync & Verification
    # =========================================================================

    def sync_positions_with_kite(self) -> Dict:
        """
        Compare DB positions with Kite holdings.
        Flags mismatches for manual review.
        Run daily at 9:20 AM.
        """
        if self.config.get('paper_trading_mode', True):
            return {'status': 'PAPER_MODE', 'message': 'Skipped sync in paper mode'}

        try:
            kite = get_kite()
            holdings = kite.holdings()
        except Exception as e:
            logger.error(f"Position sync failed - can't fetch holdings: {e}")
            return {'status': 'ERROR', 'error': str(e)}

        kite_holdings = {}
        for h in holdings:
            sym = h.get('tradingsymbol', '')
            qty = h.get('quantity', 0)
            if qty > 0:
                kite_holdings[sym] = qty

        db_positions = self.db.get_active_positions()
        db_symbols = {p['symbol']: p['qty'] for p in db_positions}

        mismatches = []

        # Check DB positions exist in Kite
        for sym, db_qty in db_symbols.items():
            kite_qty = kite_holdings.get(sym, 0)
            if kite_qty != db_qty:
                mismatches.append({
                    'symbol': sym,
                    'db_qty': db_qty,
                    'kite_qty': kite_qty,
                    'type': 'QTY_MISMATCH' if kite_qty > 0 else 'MISSING_IN_KITE',
                })

        if mismatches:
            logger.warning(f"Position sync: {len(mismatches)} mismatches found!")
            for m in mismatches:
                logger.warning(f"  {m['symbol']}: DB={m['db_qty']} Kite={m['kite_qty']} ({m['type']})")

        return {
            'status': 'OK' if not mismatches else 'MISMATCH',
            'db_positions': len(db_positions),
            'kite_holdings': len(kite_holdings),
            'mismatches': mismatches,
        }

    def verify_orders(self) -> List[Dict]:
        """
        Check pending orders status via Kite.
        Update DB with fills/rejections.
        Run at 3:25 PM.
        """
        if self.config.get('paper_trading_mode', True):
            return []

        pending = self.db.get_pending_orders()
        if not pending:
            return []

        try:
            kite = get_kite()
            kite_orders = kite.orders()
        except Exception as e:
            logger.error(f"Order verification failed: {e}")
            return []

        kite_order_map = {str(o['order_id']): o for o in kite_orders}
        results = []

        for order in pending:
            kite_id = order.get('kite_order_id')
            if not kite_id or kite_id not in kite_order_map:
                continue

            kite_order = kite_order_map[kite_id]
            kite_status = kite_order.get('status', '').upper()

            if kite_status == 'COMPLETE':
                self.db.update_order_status(order['id'], 'FILLED', kite_id)
                results.append({'order_id': order['id'], 'symbol': order['symbol'], 'status': 'FILLED'})
            elif kite_status in ('CANCELLED', 'REJECTED'):
                self.db.update_order_status(
                    order['id'], kite_status, kite_id,
                    error_message=kite_order.get('status_message', '')
                )
                results.append({'order_id': order['id'], 'symbol': order['symbol'], 'status': kite_status})

        if results:
            logger.info(f"Order verification: {len(results)} orders updated")

        return results

    # =========================================================================
    # Emergency Exit
    # =========================================================================

    def emergency_exit_all(self) -> List[Dict]:
        """
        Market sell ALL active KC6 positions.
        Kill switch for crash protection.
        """
        positions = self.db.get_active_positions()
        if not positions:
            logger.info("Emergency exit: No active positions")
            return []

        results = []
        for pos in positions:
            try:
                # Use current market price (approximate with last close)
                # In live mode, MARKET order will get best available
                result = self.place_exit_order(
                    position=pos,
                    exit_price=pos['entry_price'],  # Placeholder for paper mode
                    exit_reason='EMERGENCY'
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Emergency exit failed for {pos['symbol']}: {e}")
                results.append({'status': 'FAILED', 'symbol': pos['symbol'], 'error': str(e)})

        logger.warning(f"EMERGENCY EXIT: {len(results)} positions closed")
        return results

    # =========================================================================
    # Daily Target Limit Orders (KC6 Mid)
    # =========================================================================

    def cancel_stale_target_orders(self) -> List[Dict]:
        """
        Cancel yesterday's unfilled target orders.
        Regular Kite orders auto-expire EOD, but we clean DB state.
        For live mode, explicitly cancel any open orders.
        """
        positions = self.db.get_positions_with_target_orders()
        if not positions:
            return []

        is_paper = self.config.get('paper_trading_mode', True)
        results = []

        for pos in positions:
            old_order_id = pos.get('target_order_id')

            # In live mode, try to cancel the Kite order
            if not is_paper and old_order_id and old_order_id != 'PAPER':
                try:
                    kite = get_kite()
                    kite.cancel_order(variety='regular', order_id=old_order_id)
                    logger.info(f"Cancelled stale target order {old_order_id} for {pos['symbol']}")
                except Exception as e:
                    # Order may already be expired/cancelled - that's fine
                    logger.debug(f"Cancel target order {old_order_id}: {e}")

            # Clear target order in DB
            self.db.clear_target_order(pos['id'])
            results.append({'symbol': pos['symbol'], 'cancelled_order': old_order_id})

        logger.info(f"Cleared {len(results)} stale target orders")
        return results

    def place_target_orders(self, symbol_data: Dict = None) -> List[Dict]:
        """
        Place SELL LIMIT orders at today's KC6 mid for all active positions.
        Called every morning after market open.

        Flow:
        1. Cancel any stale target orders from yesterday
        2. Load fresh data and compute today's KC6 mid
        3. Place SELL LIMIT at KC6 mid for each position
        """
        from services.kc6_scanner import (
            compute_target_prices, load_daily_data_from_kite,
            load_daily_data_from_db
        )

        # Step 1: Cancel stale orders
        self.cancel_stale_target_orders()

        positions = self.db.get_active_positions()
        if not positions:
            logger.info("No active positions - skipping target order placement")
            return []

        # Step 2: Load data for position symbols
        pos_symbols = [p['symbol'] for p in positions]

        if symbol_data is None:
            is_paper = self.config.get('paper_trading_mode', True)
            if not is_paper and is_authenticated():
                kite = get_kite()
                symbol_data = load_daily_data_from_kite(kite, pos_symbols, days=300)
            else:
                from config import MARKET_DATA_DB
                symbol_data = load_daily_data_from_db(pos_symbols, str(MARKET_DATA_DB))

        # Step 3: Compute target prices
        targets = compute_target_prices(positions, symbol_data)

        is_paper = self.config.get('paper_trading_mode', True)
        results = []

        for t in targets:
            pos_id = t['position_id']
            symbol = t['symbol']
            kc6_mid = t['kc6_mid_today']
            qty = next(p['qty'] for p in positions if p['id'] == pos_id)

            if is_paper:
                # Paper mode: just log the target order
                order_id = self.db.log_order(
                    symbol=symbol, side='SELL', qty=qty, price=kc6_mid,
                    status='PAPER_TARGET', position_id=pos_id,
                    exit_reason='SIGNAL_KC6_MID'
                )
                self.db.update_target_order(pos_id, 'PAPER', kc6_mid, kc6_mid)
                logger.info(f"[PAPER] Target SELL {symbol} @ {kc6_mid} (KC6 mid)")
                results.append({
                    'status': 'PAPER_TARGET',
                    'symbol': symbol,
                    'target_price': kc6_mid,
                    'order_id': order_id,
                })
            else:
                # Live mode: place actual SELL LIMIT via Kite
                try:
                    kite = get_kite()
                    kite_order_id = kite.place_order(
                        variety='regular',
                        exchange='NSE',
                        tradingsymbol=symbol,
                        transaction_type='SELL',
                        quantity=qty,
                        product='CNC',
                        order_type='LIMIT',
                        price=kc6_mid,
                    )

                    order_id = self.db.log_order(
                        symbol=symbol, side='SELL', qty=qty, price=kc6_mid,
                        status='TARGET_PLACED', kite_order_id=str(kite_order_id),
                        position_id=pos_id, exit_reason='SIGNAL_KC6_MID'
                    )
                    self.db.update_target_order(pos_id, str(kite_order_id), kc6_mid, kc6_mid)

                    logger.info(f"[LIVE] Target SELL {symbol} @ {kc6_mid} | Kite #{kite_order_id}")
                    results.append({
                        'status': 'TARGET_PLACED',
                        'symbol': symbol,
                        'target_price': kc6_mid,
                        'kite_order_id': str(kite_order_id),
                    })

                except Exception as e:
                    logger.error(f"Target order failed for {symbol}: {e}")
                    results.append({
                        'status': 'FAILED',
                        'symbol': symbol,
                        'error': str(e),
                    })

        logger.info(f"Target orders: {len(results)} placed for {len(positions)} positions")
        return results

    def check_target_fills(self) -> List[Dict]:
        """
        Check if any target limit orders have been filled.
        If filled, close the position in DB.
        Called at 3:25 PM and can be called during the day.
        """
        positions = self.db.get_positions_with_target_orders()
        if not positions:
            return []

        is_paper = self.config.get('paper_trading_mode', True)

        if is_paper:
            # In paper mode, we check if today's high > target price
            # to simulate the fill
            from services.kc6_scanner import load_daily_data_from_db
            from config import MARKET_DATA_DB

            pos_symbols = [p['symbol'] for p in positions]
            symbol_data = load_daily_data_from_db(pos_symbols, str(MARKET_DATA_DB), min_bars=10)

            results = []
            today = datetime.now().strftime('%Y-%m-%d')

            for pos in positions:
                sym = pos['symbol']
                target_price = pos.get('target_order_price')
                if not target_price or sym not in symbol_data:
                    continue

                df = symbol_data[sym]
                row = df.iloc[-1]
                if float(row['high']) > target_price:
                    # Simulated fill
                    trade = self.db.close_position(pos['id'], target_price, today, 'SIGNAL_KC6_MID')
                    self.db.clear_target_order(pos['id'])
                    logger.info(
                        f"[PAPER] Target FILLED {sym} @ {target_price} | "
                        f"P/L={trade['pnl_pct']:+.2f}%"
                    )
                    results.append({
                        'status': 'FILLED',
                        'symbol': sym,
                        'price': target_price,
                        'pnl_pct': trade['pnl_pct'],
                    })

            return results

        # Live mode: check Kite order status
        try:
            kite = get_kite()
            kite_orders = kite.orders()
        except Exception as e:
            logger.error(f"Target fill check failed: {e}")
            return []

        kite_order_map = {str(o['order_id']): o for o in kite_orders}
        results = []
        today = datetime.now().strftime('%Y-%m-%d')

        for pos in positions:
            kite_id = pos.get('target_order_id')
            if not kite_id or kite_id not in kite_order_map:
                continue

            kite_order = kite_order_map[kite_id]
            kite_status = kite_order.get('status', '').upper()

            if kite_status == 'COMPLETE':
                fill_price = float(kite_order.get('average_price', pos['target_order_price']))
                trade = self.db.close_position(pos['id'], fill_price, today, 'SIGNAL_KC6_MID')
                self.db.clear_target_order(pos['id'])

                # Update the order log
                order_logs = self.db.get_recent_orders(limit=100)
                for ol in order_logs:
                    if ol.get('kite_order_id') == kite_id:
                        self.db.update_order_status(ol['id'], 'FILLED', kite_id)
                        break

                logger.info(f"[LIVE] Target FILLED {pos['symbol']} @ {fill_price} | P/L={trade['pnl_pct']:+.2f}%")
                results.append({
                    'status': 'FILLED',
                    'symbol': pos['symbol'],
                    'price': fill_price,
                    'pnl_pct': trade['pnl_pct'],
                })

        return results


# =============================================================================
# Pipeline Functions (called by scheduler)
# =============================================================================

def run_place_targets(config: dict = None, capital: float = 1_000_000) -> List[Dict]:
    """
    Morning job (9:20 AM): Place SELL LIMIT orders at today's KC6 mid
    for all active positions. Cancels yesterday's stale orders first.
    """
    if config is None:
        config = KC6_DEFAULTS

    executor = KC6Executor(config=config, capital=capital)
    return executor.place_target_orders()


def run_exit_check(config: dict = None, capital: float = 1_000_000) -> List[Dict]:
    """
    3:15 PM exit check. Steps:
    1. Check if any target limit orders filled (KC6 mid exits)
    2. Cancel remaining unfilled target orders
    3. Check SL/TP/MaxHold for positions that didn't exit via target
    """
    from services.kc6_scanner import run_full_scan
    from services.kite_service import is_authenticated

    if config is None:
        config = KC6_DEFAULTS

    executor = KC6Executor(config=config, capital=capital)
    all_results = []

    # Step 1: Check target fills first
    fills = executor.check_target_fills()
    all_results.extend(fills)
    filled_symbols = {r['symbol'] for r in fills if r.get('status') == 'FILLED'}

    # Step 2: Cancel unfilled target orders (so we can place SL/MaxHold exits)
    executor.cancel_stale_target_orders()

    # Step 3: Check SL/TP/MaxHold for remaining positions
    db = get_kc6_db()
    positions = db.get_active_positions()

    if not positions:
        logger.info("No remaining positions after target fill check")
        return all_results

    kite = None
    if is_authenticated() and not config.get('paper_trading_mode', True):
        kite = get_kite()

    scan = run_full_scan(kite=kite, config=config)

    # Only process exits for non-SIGNAL exits (SL, TP, MAX_HOLD)
    # SIGNAL_KC6_MID exits are handled by the target limit orders above
    exit_signals = [s for s in scan.get('exits', [])
                    if s['exit_reason'] != 'SIGNAL_KC6_MID'
                    and s['symbol'] not in filled_symbols]

    for sig in exit_signals:
        pos = db.get_position_by_symbol(sig['symbol'])
        if not pos:
            continue
        result = executor.place_exit_order(pos, sig['exit_price'], sig['exit_reason'])
        all_results.append(result)

    return all_results


def run_entry_scan(config: dict = None, capital: float = 1_000_000) -> List[Dict]:
    """Full scan + execute entries. Called at 3:20 PM."""
    from services.kc6_scanner import run_full_scan
    from services.kite_service import is_authenticated

    if config is None:
        config = KC6_DEFAULTS

    kite = None
    if is_authenticated() and not config.get('paper_trading_mode', True):
        kite = get_kite()

    scan = run_full_scan(kite=kite, config=config)

    entry_signals = scan.get('entries', [])
    if not entry_signals:
        logger.info("No entry signals")
        return []

    executor = KC6Executor(config=config, capital=capital)
    max_entries = config.get('max_positions', 5) - get_kc6_db().get_active_positions_count()

    results = []
    for sig in entry_signals[:max_entries]:
        result = executor.place_entry_order(
            symbol=sig['symbol'],
            limit_price=sig['entry_price'],
            universe_atr_ratio=scan.get('universe_atr_ratio'),
            kc6_mid=sig.get('kc6_mid'),
            sma200=sig.get('sma200'),
        )
        results.append(result)

    return results
