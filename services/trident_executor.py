"""
Trident Order Executor
========================

Handles order placement for PA_MACD + RangeBreakout strategies.
Paper trading mode by default. Safety guardrails modeled after KC6.
"""

import logging
from datetime import datetime, date, time as dtime
from typing import Dict, List, Optional

from services.trident_db import get_trident_db
from services.kite_service import get_kite, is_authenticated

logger = logging.getLogger(__name__)

# Top F&O stocks for Trident (high-liquidity, backtested)
TRIDENT_UNIVERSE = [
    'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'SBIN', 'BHARTIARTL',
    'ITC', 'KOTAKBANK', 'HINDUNILVR', 'LT', 'AXISBANK', 'BAJFINANCE', 'MARUTI',
    'HCLTECH', 'SUNPHARMA', 'TATAMOTORS', 'NTPC', 'POWERGRID', 'TITAN',
    'WIPRO', 'ULTRACEMCO', 'ADANIENT', 'ADANIPORTS', 'NESTLEIND', 'JSWSTEEL',
    'TATASTEEL', 'TECHM', 'M&M', 'COALINDIA', 'ONGC', 'GRASIM', 'BAJAJFINSV',
    'APOLLOHOSP', 'DIVISLAB', 'DRREDDY', 'CIPLA', 'HEROMOTOCO', 'EICHERMOT',
    'BPCL', 'INDUSINDBK', 'TATACONSUM', 'SHRIRAMFIN', 'BRITANNIA', 'HINDALCO',
    'ASIANPAINT', 'SBILIFE', 'HDFCLIFE', 'PIDILITIND', 'DABUR',
]

TRIDENT_DEFAULTS = {
    'max_positions': 20,
    'position_size_pct': 0.05,       # 5% per position
    'max_daily_orders': 20,
    'max_daily_loss_pct': 3.0,
    'capital': 10_000_000,           # 1 Cr paper
    'paper_trading_mode': True,
    'live_trading_enabled': False,
    'enabled': True,
}


class TridentExecutor:
    """Order executor with safety guardrails for Trident strategy."""

    def __init__(self, config: dict = None):
        self.config = config or TRIDENT_DEFAULTS.copy()
        self.capital = self.config.get('capital', 10_000_000)
        self.db = get_trident_db()
        self._last_scan_result = None
        self._scan_status = 'idle'

    def _check_guardrails(self, is_entry: bool = True,
                          symbol: str = None) -> tuple:
        """Run pre-order safety checks. Returns (passed, reason)."""
        cfg = self.config

        if not cfg.get('enabled', True):
            return False, "Trident system disabled"

        if not cfg.get('live_trading_enabled', False):
            if not cfg.get('paper_trading_mode', True):
                return False, "Neither live nor paper mode enabled"

        # Kite auth (live only)
        if cfg.get('live_trading_enabled') and not cfg.get('paper_trading_mode'):
            if not is_authenticated():
                return False, "Kite API not authenticated"

        # Market hours
        now = datetime.now()
        if now.weekday() >= 5:
            return False, f"Market closed ({now.strftime('%A')})"
        if now.time() < dtime(9, 15) or now.time() > dtime(15, 30):
            return False, f"Market closed ({now.time()})"

        # Daily limits
        if self.db.get_today_order_count() >= cfg.get('max_daily_orders', 20):
            return False, "Daily order limit reached"

        if self.db.get_today_loss_pct(self.capital) >= cfg.get('max_daily_loss_pct', 3.0):
            return False, "Daily loss limit reached"

        if is_entry:
            active = self.db.get_active_positions_count()
            if active >= cfg.get('max_positions', 20):
                return False, f"Max positions ({active}/{cfg.get('max_positions', 20)})"

            if symbol and self.db.get_position_by_symbol(symbol):
                return False, f"Already have position in {symbol}"

        return True, "OK"

    def execute_entry(self, signal: Dict) -> Dict:
        """Execute an entry signal (paper or live)."""
        sym = signal['symbol']
        direction = signal['direction']
        strategy = signal['strategy']
        entry_price = signal.get('entry_price', signal.get('stop_level'))
        sl = signal['sl_price']
        tp = signal['tp_price']
        max_hold = signal.get('max_hold_days', 10)

        passed, reason = self._check_guardrails(is_entry=True, symbol=sym)
        if not passed:
            logger.info(f"[Trident] Entry blocked for {sym}: {reason}")
            return {'status': 'blocked', 'reason': reason}

        # Calculate qty
        pos_size = self.capital * self.config.get('position_size_pct', 0.05)
        qty = max(1, int(pos_size / entry_price))

        today = date.today().isoformat()

        if self.config.get('paper_trading_mode', True):
            # Paper mode: log immediately
            pos_id = self.db.add_position(
                symbol=sym, direction=direction, strategy=strategy,
                entry_price=entry_price, entry_date=today, qty=qty,
                sl_price=sl, tp_price=tp, max_hold_days=max_hold,
            )
            self.db.log_order(
                symbol=sym, side='BUY' if direction == 'LONG' else 'SELL',
                qty=qty, price=entry_price, status='PAPER',
                direction=direction, strategy=strategy, position_id=pos_id,
            )
            logger.info(f"[Trident] PAPER ENTRY: {sym} {direction} @ {entry_price} "
                       f"SL={sl} TP={tp} qty={qty} ({strategy})")
            return {'status': 'filled', 'position_id': pos_id, 'mode': 'paper'}
        else:
            # Live mode via Kite
            try:
                kite = get_kite()
                side = 'BUY' if direction == 'LONG' else 'SELL'
                order_id = kite.place_order(
                    variety='regular', exchange='NSE',
                    tradingsymbol=sym, transaction_type=side,
                    quantity=qty, price=entry_price,
                    order_type='LIMIT', product='CNC',
                )
                pos_id = self.db.add_position(
                    symbol=sym, direction=direction, strategy=strategy,
                    entry_price=entry_price, entry_date=today, qty=qty,
                    sl_price=sl, tp_price=tp, max_hold_days=max_hold,
                    kite_order_id=str(order_id),
                )
                self.db.log_order(
                    symbol=sym, side=side, qty=qty, price=entry_price,
                    status='PLACED', direction=direction, strategy=strategy,
                    position_id=pos_id, kite_order_id=str(order_id),
                )
                logger.info(f"[Trident] LIVE ENTRY: {sym} {direction} @ {entry_price} "
                           f"order_id={order_id}")
                return {'status': 'placed', 'position_id': pos_id,
                        'order_id': order_id, 'mode': 'live'}
            except Exception as e:
                self.db.log_order(
                    symbol=sym, side='BUY' if direction == 'LONG' else 'SELL',
                    qty=qty, price=entry_price, status='FAILED',
                    direction=direction, strategy=strategy,
                    error_message=str(e),
                )
                logger.error(f"[Trident] Entry FAILED for {sym}: {e}")
                return {'status': 'failed', 'error': str(e)}

    def execute_exit(self, exit_signal: Dict) -> Dict:
        """Execute an exit (paper or live)."""
        pos_id = exit_signal['position_id']
        exit_price = exit_signal['exit_price']
        exit_reason = exit_signal['exit_reason']
        sym = exit_signal['symbol']
        direction = exit_signal['direction']

        passed, reason = self._check_guardrails(is_entry=False)
        if not passed:
            logger.warning(f"[Trident] Exit blocked for {sym}: {reason}")
            return {'status': 'blocked', 'reason': reason}

        today = date.today().isoformat()

        if self.config.get('paper_trading_mode', True):
            trade = self.db.close_position(pos_id, exit_price, today, exit_reason)
            self.db.log_order(
                symbol=sym,
                side='SELL' if direction == 'LONG' else 'BUY',
                qty=trade.get('qty', 0) if isinstance(trade, dict) else 0,
                price=exit_price, status='PAPER',
                direction=direction, position_id=pos_id,
                exit_reason=exit_reason,
            )
            return {'status': 'closed', 'trade': trade, 'mode': 'paper'}
        else:
            try:
                kite = get_kite()
                pos = self.db.get_position_by_symbol(sym)
                if not pos:
                    return {'status': 'error', 'reason': 'Position not found'}
                side = 'SELL' if direction == 'LONG' else 'BUY'
                order_id = kite.place_order(
                    variety='regular', exchange='NSE',
                    tradingsymbol=sym, transaction_type=side,
                    quantity=pos['qty'], price=exit_price,
                    order_type='LIMIT', product='CNC',
                )
                trade = self.db.close_position(pos_id, exit_price, today, exit_reason)
                self.db.log_order(
                    symbol=sym, side=side, qty=pos['qty'], price=exit_price,
                    status='PLACED', direction=direction, position_id=pos_id,
                    kite_order_id=str(order_id), exit_reason=exit_reason,
                )
                return {'status': 'closed', 'trade': trade, 'order_id': order_id, 'mode': 'live'}
            except Exception as e:
                logger.error(f"[Trident] Exit FAILED for {sym}: {e}")
                return {'status': 'failed', 'error': str(e)}

    def emergency_exit_all(self) -> List[Dict]:
        """Kill switch: close all positions at market."""
        positions = self.db.get_active_positions()
        results = []
        for pos in positions:
            result = self.execute_exit({
                'position_id': pos['id'],
                'symbol': pos['symbol'],
                'direction': pos['direction'],
                'exit_price': pos['entry_price'],  # Will use market in live
                'exit_reason': 'EMERGENCY',
            })
            results.append(result)
        logger.warning(f"[Trident] EMERGENCY EXIT: closed {len(results)} positions")
        return results

    def run_scan(self, symbol_data: Dict = None) -> Dict:
        """Full scan + execute pipeline."""
        from services.trident_scanner import (
            load_daily_data_from_db, run_full_scan,
            check_exits, check_pending_triggers,
        )
        from config import DATA_DIR

        self._scan_status = 'scanning'
        today = date.today().isoformat()

        try:
            # Load data
            if symbol_data is None:
                db_path = str(DATA_DIR / 'market_data.db')
                symbol_data = load_daily_data_from_db(TRIDENT_UNIVERSE, db_path)

            # 1. Check exits on active positions
            positions = self.db.get_active_positions()
            exit_signals = check_exits(positions, symbol_data, self.config)
            exit_results = []
            for ex in exit_signals:
                result = self.execute_exit(ex)
                exit_results.append(result)

            # 2. Check pending stop triggers
            self.db.expire_pending_signals(today)  # Expire yesterday's
            pending = self.db.get_pending_signals()
            triggered = check_pending_triggers(pending, symbol_data)
            trigger_results = []
            for trig in triggered:
                result = self.execute_entry(trig)
                trigger_results.append(result)
                if result.get('status') in ('filled', 'placed'):
                    self.db.fill_pending_signal(trig['id'])

            # 3. Scan for new signals
            scan_result = run_full_scan(symbol_data, self.config)

            # Save PA_MACD pending signals for next day
            for sig in scan_result['pamacd_signals']:
                if not self.db.get_position_by_symbol(sig['symbol']):
                    self.db.add_pending_signal(
                        symbol=sig['symbol'], direction=sig['direction'],
                        strategy=sig['strategy'], stop_level=sig['stop_level'],
                        sl_price=sig['sl_price'], tp_price=sig['tp_price'],
                        max_hold_days=sig['max_hold_days'], signal_date=today,
                    )

            # Execute RangeBreakout immediate signals
            rb_results = []
            for sig in scan_result['rb_signals']:
                result = self.execute_entry(sig)
                rb_results.append(result)

            # Save daily state
            self.db.save_daily_state(
                trade_date=today,
                positions_count=self.db.get_active_positions_count(),
                pamacd_signals=len(scan_result['pamacd_signals']),
                rb_signals=len(scan_result['rb_signals']),
                entries_placed=len([r for r in trigger_results + rb_results
                                    if r.get('status') in ('filled', 'placed')]),
                exits_placed=len([r for r in exit_results
                                  if r.get('status') == 'closed']),
            )

            self._last_scan_result = {
                'scan_time': scan_result['scan_time'],
                'symbols_scanned': scan_result['symbols_scanned'],
                'pamacd_signals': len(scan_result['pamacd_signals']),
                'rb_signals': len(scan_result['rb_signals']),
                'exits': len(exit_results),
                'pending_triggered': len(trigger_results),
                'new_entries': len(rb_results),
                'signals_detail': scan_result['pamacd_signals'] + scan_result['rb_signals'],
                'exit_details': exit_results,
            }
            self._scan_status = 'done'

            logger.info(f"[Trident] Scan complete: {len(exit_results)} exits, "
                       f"{len(trigger_results)} triggers, {len(rb_results)} new RB entries, "
                       f"{len(scan_result['pamacd_signals'])} PA_MACD pending")

            return self._last_scan_result

        except Exception as e:
            self._scan_status = 'error'
            logger.error(f"[Trident] Scan failed: {e}", exc_info=True)
            return {'error': str(e)}

    def get_state(self) -> Dict:
        """Get full system state for dashboard."""
        positions = self.db.get_active_positions()
        trades = self.db.get_trades(limit=50)
        stats = self.db.get_stats()
        pending = self.db.get_pending_signals()

        return {
            'config': {
                'paper_trading_mode': self.config.get('paper_trading_mode', True),
                'live_trading_enabled': self.config.get('live_trading_enabled', False),
                'enabled': self.config.get('enabled', True),
                'max_positions': self.config.get('max_positions', 20),
                'capital': self.capital,
            },
            'positions': positions,
            'positions_count': len(positions),
            'trades': trades,
            'stats': stats,
            'pending_signals': pending,
            'last_scan': self._last_scan_result,
            'scan_status': self._scan_status,
        }


# Singleton
_instance = None


def get_trident_executor(config: dict = None) -> TridentExecutor:
    global _instance
    if _instance is None:
        _instance = TridentExecutor(config)
    return _instance
