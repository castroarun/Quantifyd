"""
BNF Squeeze & Fire — Executor
================================
Handles trade execution for BankNifty options.
Two modes:
  - SQUEEZE: Sell strangles (CE + PE) when BB squeezing
  - FIRE: Sell naked options directionally when BB fires

Paper mode: simulates fills immediately.
Live mode: places orders via Kite API.
"""

import logging
from datetime import datetime, date

from services.bnf_db import get_bnf_db
from services.bnf_scanner import (
    BnfScanner, LOT_SIZE, STRIKE_STEP,
    bs_call, bs_put, estimate_iv, get_monthly_expiry, round_strike,
)

logger = logging.getLogger(__name__)


class BnfExecutor:
    """
    State machine for BNF Squeeze & Fire execution.
    Runs once daily (after market close or at 3:20 PM).
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.db = get_bnf_db()
        self.scanner = BnfScanner(config)

    # ─── Guardrails ──────────────────────────────────────────

    def _check_guardrails(self, mode='FIRE'):
        """Pre-order safety checks."""
        cfg = self.cfg

        if not cfg.get('enabled', True):
            return False, 'System disabled'

        # Check max positions per mode
        active = self.db.get_active_positions(mode=mode)
        max_pos = cfg.get('max_squeeze_positions', 2) if mode == 'SQUEEZE' else cfg.get('max_fire_positions', 1)
        if len(active) >= max_pos:
            return False, f'Max {mode} positions ({max_pos}) reached'

        # Check daily order limit
        today_orders = self.db.get_today_order_count()
        if today_orders >= cfg.get('max_daily_orders', 10):
            return False, f'Daily order limit ({cfg["max_daily_orders"]}) reached'

        return True, 'OK'

    # ─── Order Placement ─────────────────────────────────────

    def _place_order(self, tradingsymbol, transaction_type, qty, price,
                     mode, instrument_type, strike, expiry_date,
                     signal_type, sl_price=None, max_loss=None,
                     hold_bars=None):
        """
        Place an order. Paper mode: fill immediately. Live: place on Kite.
        Returns position_id or None.
        """
        cfg = self.cfg
        now = datetime.now().isoformat()

        # Create position record
        position_id = self.db.add_position(
            mode=mode,
            tradingsymbol=tradingsymbol,
            exchange='NFO',
            transaction_type=transaction_type,
            instrument_type=instrument_type,
            qty=qty,
            strike=strike,
            expiry_date=str(expiry_date),
            entry_price=price,
            entry_time=now,
            sl_price=sl_price,
            sl_reason=f'{cfg.get("fire_sl_mult", 3.0)}x premium' if mode == 'FIRE' else 'max_loss_cap',
            signal_type=signal_type,
            hold_bars_remaining=hold_bars,
            max_loss_rupees=max_loss,
            status='ACTIVE' if cfg.get('paper_trading_mode', True) else 'PENDING',
        )

        # Log order
        order_status = 'PAPER' if cfg.get('paper_trading_mode', True) else 'PLACED'
        self.db.log_order(
            tradingsymbol=tradingsymbol,
            transaction_type=transaction_type,
            qty=qty,
            price=price,
            order_type='MARKET',
            status=order_status,
            position_id=position_id,
            signal_type=signal_type,
        )

        # Live mode: place on Kite
        if not cfg.get('paper_trading_mode', True):
            try:
                from services.kite_service import get_kite
                kite = get_kite()
                order_id = kite.place_order(
                    variety='regular',
                    exchange='NFO',
                    tradingsymbol=tradingsymbol,
                    transaction_type=transaction_type,
                    quantity=qty,
                    product='NRML',
                    order_type='MARKET',
                )
                self.db.update_position(position_id, kite_order_id=str(order_id))
                logger.info(f"Kite order placed: {order_id} for {tradingsymbol}")
            except Exception as e:
                logger.error(f"Kite order failed: {e}")
                self.db.update_position(position_id, status='FAILED', notes=str(e))
                return None

        logger.info(f"[{order_status}] {transaction_type} {qty} {tradingsymbol} @ {price:.2f} "
                     f"({mode}/{signal_type})")
        return position_id

    def _build_tradingsymbol(self, instrument_type, strike, expiry):
        """Build NFO tradingsymbol like BANKNIFTY2530151400CE."""
        # Format: BANKNIFTY + YYMDD + Strike + CE/PE
        # Example: BANKNIFTY25MAR51400CE
        month_map = {1: 'JAN', 2: 'FEB', 3: 'MAR', 4: 'APR', 5: 'MAY', 6: 'JUN',
                     7: 'JUL', 8: 'AUG', 9: 'SEP', 10: 'OCT', 11: 'NOV', 12: 'DEC'}
        if isinstance(expiry, str):
            expiry = date.fromisoformat(expiry[:10])
        yy = expiry.year % 100
        mon = month_map[expiry.month]
        dd = expiry.day
        return f"BANKNIFTY{yy}{mon}{dd}{int(strike)}{instrument_type}"

    # ─── Signal Execution ────────────────────────────────────

    def execute_fire_signal(self, signal, scan_result):
        """Execute a fire mode signal (sell naked option)."""
        ok, reason = self._check_guardrails('FIRE')
        if not ok:
            logger.info(f"Fire guardrail: {reason}")
            return None, reason

        cfg = self.cfg
        lots = cfg.get('fire_lots', 5)
        qty = lots * LOT_SIZE
        strike = signal['strike']
        premium = signal['premium']
        sl_price = premium * cfg.get('fire_sl_mult', 3.0)
        max_loss = signal.get('max_loss', cfg.get('fire_max_loss_rupees', 20000))
        hold_bars = signal.get('hold_bars', cfg.get('fire_hold_bars', 7))

        dte, expiry = get_monthly_expiry(date.today(), min_dte=15)

        if signal['type'] == 'FIRE_LONG':
            instrument_type = 'PE'
        else:
            instrument_type = 'CE'

        tradingsymbol = self._build_tradingsymbol(instrument_type, strike, expiry)

        position_id = self._place_order(
            tradingsymbol=tradingsymbol,
            transaction_type='SELL',
            qty=qty,
            price=premium,
            mode='FIRE',
            instrument_type=instrument_type,
            strike=strike,
            expiry_date=expiry,
            signal_type=signal['type'],
            sl_price=sl_price,
            max_loss=max_loss,
            hold_bars=hold_bars,
        )

        # Log signal
        self.db.log_signal(
            signal_type=signal['type'],
            mode='FIRE',
            bb_state=scan_result.get('bb_state'),
            squeeze_count=scan_result.get('squeeze_count', 0),
            direction=scan_result.get('direction'),
            trend_strength=scan_result.get('trend_strength'),
            spot_price=scan_result.get('spot'),
            atr_value=scan_result.get('atr'),
            action_taken=f"SELL {instrument_type} {strike} @ {premium:.2f} x{lots}L",
        )

        return position_id, 'OK'

    def execute_squeeze_signal(self, signal, scan_result):
        """Execute a squeeze mode signal (sell strangle — CE + PE)."""
        ok, reason = self._check_guardrails('SQUEEZE')
        if not ok:
            logger.info(f"Squeeze guardrail: {reason}")
            return None, reason

        cfg = self.cfg
        lots = cfg.get('squeeze_lots', 5)
        qty = lots * LOT_SIZE
        max_loss = signal.get('max_loss', cfg.get('squeeze_max_loss_rupees', 30000))
        hold_bars = signal.get('hold_bars', cfg.get('squeeze_hold_bars', 10))

        dte, expiry = get_monthly_expiry(date.today(), min_dte=15)

        results = []

        # Sell CALL
        call_strike = signal['call_strike']
        call_prem = signal['call_premium']
        call_symbol = self._build_tradingsymbol('CE', call_strike, expiry)

        pos_ce = self._place_order(
            tradingsymbol=call_symbol,
            transaction_type='SELL',
            qty=qty,
            price=call_prem,
            mode='SQUEEZE',
            instrument_type='CE',
            strike=call_strike,
            expiry_date=expiry,
            signal_type='SQUEEZE_SELL_CE',
            max_loss=max_loss,
            hold_bars=hold_bars,
        )
        results.append(('CE', pos_ce))

        # Sell PUT
        put_strike = signal['put_strike']
        put_prem = signal['put_premium']
        put_symbol = self._build_tradingsymbol('PE', put_strike, expiry)

        pos_pe = self._place_order(
            tradingsymbol=put_symbol,
            transaction_type='SELL',
            qty=qty,
            price=put_prem,
            mode='SQUEEZE',
            instrument_type='PE',
            strike=put_strike,
            expiry_date=expiry,
            signal_type='SQUEEZE_SELL_PE',
            max_loss=max_loss,
            hold_bars=hold_bars,
        )
        results.append(('PE', pos_pe))

        # Log signal
        self.db.log_signal(
            signal_type='SQUEEZE_ENTRY',
            mode='SQUEEZE',
            bb_state=scan_result.get('bb_state'),
            squeeze_count=scan_result.get('squeeze_count', 0),
            direction=scan_result.get('direction'),
            trend_strength=scan_result.get('trend_strength'),
            spot_price=scan_result.get('spot'),
            atr_value=scan_result.get('atr'),
            action_taken=f"SELL {call_strike}CE @ {call_prem:.2f} + {put_strike}PE @ {put_prem:.2f} x{lots}L",
        )

        return results, 'OK'

    # ─── Exit Management ─────────────────────────────────────

    def check_and_exit(self, scan_result=None):
        """Check all active positions for exit conditions. Returns list of exit actions."""
        active = self.db.get_active_positions()
        if not active:
            return []

        spot = scan_result['spot'] if scan_result else 0
        iv = scan_result.get('iv') if scan_result else None

        # Store ATR for SL checks
        if scan_result:
            self.cfg['_last_atr'] = scan_result.get('atr', 500)

        exits = self.scanner.check_exits(active, spot, iv)
        results = []

        for pos_id, exit_reason, est_exit_price in exits:
            pos = next((p for p in active if p['id'] == pos_id), None)
            if not pos:
                continue

            # Determine exit price
            if est_exit_price:
                exit_price = est_exit_price
            elif pos['instrument_type'] == 'PE':
                exit_price = bs_put(spot, pos['strike'], 1 / 365.0, 0.065, iv or 0.15)
            elif pos['instrument_type'] == 'CE':
                exit_price = bs_call(spot, pos['strike'], 1 / 365.0, 0.065, iv or 0.15)
            else:
                exit_price = 0

            # Close position
            self.db.close_position(pos_id, round(exit_price, 2), exit_reason)

            # In live mode, place buy-back order
            if not self.cfg.get('paper_trading_mode', True):
                try:
                    from services.kite_service import get_kite
                    kite = get_kite()
                    kite.place_order(
                        variety='regular',
                        exchange='NFO',
                        tradingsymbol=pos['tradingsymbol'],
                        transaction_type='BUY',
                        quantity=pos['qty'],
                        product='NRML',
                        order_type='MARKET',
                    )
                except Exception as e:
                    logger.error(f"Exit order failed for {pos['tradingsymbol']}: {e}")

            results.append({
                'position_id': pos_id,
                'tradingsymbol': pos['tradingsymbol'],
                'mode': pos['mode'],
                'exit_reason': exit_reason,
                'exit_price': round(exit_price, 2),
            })

            logger.info(f"[EXIT] {pos['tradingsymbol']} ({pos['mode']}) — {exit_reason} @ {exit_price:.2f}")

        return results

    def decrement_hold_bars(self):
        """Called daily to decrement hold_bars_remaining for all active positions."""
        active = self.db.get_active_positions()
        for pos in active:
            if pos.get('hold_bars_remaining') is not None and pos['hold_bars_remaining'] > 0:
                self.db.update_position(pos['id'],
                                         hold_bars_remaining=pos['hold_bars_remaining'] - 1)

    # ─── Daily Scan Pipeline ─────────────────────────────────

    def run_daily_scan(self):
        """
        Full daily pipeline:
        1. Load bars + compute indicators
        2. Decrement hold bars
        3. Check exits
        4. Check new signals
        5. Execute signals
        6. Update state
        7. Save daily snapshot

        Returns dict with scan results and actions taken.
        """
        logger.info("BNF daily scan starting...")
        results = {
            'scan': None,
            'exits': [],
            'entries': [],
            'errors': [],
        }

        try:
            # 1. Scan
            scan = self.scanner.scan()
            results['scan'] = scan

            if scan.get('error'):
                results['errors'].append(scan['error'])
                return results

            # 2. Decrement hold bars
            self.decrement_hold_bars()

            # 3. Check exits
            exits = self.check_and_exit(scan)
            results['exits'] = exits

            # 4. Execute new signals
            for signal in scan.get('signals', []):
                try:
                    if signal['type'] in ('FIRE_LONG', 'FIRE_SHORT'):
                        pos_id, msg = self.execute_fire_signal(signal, scan)
                        if pos_id:
                            results['entries'].append({
                                'type': signal['type'],
                                'strike': signal['strike'],
                                'premium': signal['premium'],
                                'position_id': pos_id,
                            })
                        else:
                            results['errors'].append(f"Fire skip: {msg}")

                    elif signal['type'] == 'SQUEEZE_ENTRY':
                        pos_ids, msg = self.execute_squeeze_signal(signal, scan)
                        if pos_ids:
                            results['entries'].append({
                                'type': 'SQUEEZE_ENTRY',
                                'call_strike': signal['call_strike'],
                                'put_strike': signal['put_strike'],
                                'total_premium': signal['total_premium'],
                            })
                        else:
                            results['errors'].append(f"Squeeze skip: {msg}")

                except Exception as e:
                    logger.error(f"Signal execution error: {e}")
                    results['errors'].append(str(e))

            # 5. Update system state
            self.db.update_state(
                bb_state=scan['bb_state'],
                squeeze_count=scan['squeeze_count'],
                bb_width=scan['bb_width'],
                bb_width_ma=scan['bb_width_ma'],
                sma_value=scan['sma'],
                atr_value=scan['atr'],
                direction=scan['direction'],
                trend_strength=scan['trend_strength'],
                last_close=scan['spot'],
                last_scan_time=datetime.now().isoformat(),
            )

            if scan['signals']:
                self.db.update_state(last_signal_time=datetime.now().isoformat())

            # 6. Save daily snapshot
            stats = self.db.get_stats()
            sq_active = len(self.db.get_active_positions(mode='SQUEEZE'))
            fr_active = len(self.db.get_active_positions(mode='FIRE'))

            self.db.save_daily_state(
                trade_date=date.today().isoformat(),
                bb_state=scan['bb_state'],
                squeeze_positions=sq_active,
                fire_positions=fr_active,
                daily_pnl=sum(e.get('pnl_abs', 0) for e in exits if isinstance(e, dict)),
                cumulative_pnl=stats['total_pnl'],
            )

            logger.info(f"BNF scan complete: {scan['bb_state']}, "
                         f"{len(scan['signals'])} signals, {len(exits)} exits, "
                         f"{len(results['entries'])} entries")

        except Exception as e:
            logger.error(f"BNF daily scan error: {e}", exc_info=True)
            results['errors'].append(str(e))

        return results

    # ─── Emergency ───────────────────────────────────────────

    def emergency_exit_all(self):
        """Kill switch — close all active positions immediately."""
        active = self.db.get_active_positions()
        closed = 0

        for pos in active:
            try:
                # Use intrinsic value as exit price estimate
                self.db.close_position(pos['id'], exit_price=0, exit_reason='EMERGENCY')

                if not self.cfg.get('paper_trading_mode', True):
                    from services.kite_service import get_kite
                    kite = get_kite()
                    kite.place_order(
                        variety='regular',
                        exchange='NFO',
                        tradingsymbol=pos['tradingsymbol'],
                        transaction_type='BUY',
                        quantity=pos['qty'],
                        product='NRML',
                        order_type='MARKET',
                    )
                closed += 1
            except Exception as e:
                logger.error(f"Emergency exit failed for {pos['tradingsymbol']}: {e}")

        logger.warning(f"EMERGENCY EXIT: closed {closed}/{len(active)} positions")
        return closed

    # ─── State for Dashboard ─────────────────────────────────

    def get_full_state(self):
        """Get complete state for dashboard API."""
        state = self.db.get_state()
        stats = self.db.get_stats()
        active = self.db.get_active_positions()
        signals = self.db.get_recent_signals(limit=20)
        trades = self.db.get_recent_trades(limit=30)

        sq_active = [p for p in active if p['mode'] == 'SQUEEZE']
        fr_active = [p for p in active if p['mode'] == 'FIRE']

        return {
            'state': state,
            'stats': stats,
            'config': {
                'paper_trading_mode': self.cfg.get('paper_trading_mode', True),
                'enabled': self.cfg.get('enabled', True),
                'fire_lots': self.cfg.get('fire_lots', 5),
                'squeeze_lots': self.cfg.get('squeeze_lots', 5),
                'fire_hold_bars': self.cfg.get('fire_hold_bars', 7),
                'squeeze_hold_bars': self.cfg.get('squeeze_hold_bars', 10),
                'fire_sl_mult': self.cfg.get('fire_sl_mult', 3.0),
            },
            'positions': {
                'squeeze': sq_active,
                'fire': fr_active,
                'total_active': len(active),
            },
            'recent_signals': signals,
            'recent_trades': trades,
        }
