"""
NAS — Nifty ATR Strangle — Executor
======================================
Handles trade execution for intraday Nifty options strangles.

Modes:
  - Paper: simulates fills immediately (default)
  - Live: places orders via Kite API

Key features:
  - ATR squeeze entry detection
  - Dynamic leg adjustments (2x loss / 0.5x profit triggers)
  - EOD mandatory squareoff
  - Guardrails and safety checks
"""

import logging
from datetime import datetime, date, timedelta

from services.nas_db import get_nas_db
from services.nas_scanner import (
    NasScanner, LOT_SIZE, STRIKE_STEP,
    bs_call, bs_put, estimate_iv, round_strike,
    get_current_week_expiry, is_expiry_day,
)

logger = logging.getLogger(__name__)


class NasExecutor:
    """
    State machine for NAS intraday strangle execution.
    Runs every 5 minutes during market hours.
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.db = get_nas_db()
        self.scanner = NasScanner(config)

    # ─── Guardrails ──────────────────────────────────────────

    def _check_guardrails(self, is_entry=True):
        """Pre-order safety checks. Returns (passed, reason)."""
        cfg = self.cfg

        if not cfg.get('enabled', True):
            return False, 'System disabled'

        # Daily order limit
        today_orders = self.db.get_today_order_count()
        if today_orders >= cfg.get('max_daily_orders', 20):
            return False, f'Daily order limit ({cfg["max_daily_orders"]}) reached'

        if is_entry:
            # Max concurrent strangles
            active = self.db.get_active_positions()
            strangle_ids = set(p.get('strangle_id') for p in active if p.get('strangle_id'))
            max_strangles = cfg.get('max_strangles', 1)
            if len(strangle_ids) >= max_strangles:
                return False, f'Max strangles ({max_strangles}) reached'

            # Daily P&L circuit breaker
            state = self.db.get_state()
            daily_pnl = state.get('daily_pnl', 0) or 0
            max_loss = cfg.get('max_daily_loss', 15000)
            if daily_pnl < -max_loss:
                return False, f'Daily loss Rs {abs(daily_pnl):.0f} exceeds limit Rs {max_loss}'

            # Total adjustments limit
            total_adj = state.get('total_adjustments', 0) or 0
            max_adj = cfg.get('max_adjustments_total', 4)
            if total_adj >= max_adj:
                return False, f'Max daily adjustments ({max_adj}) reached'

        return True, 'OK'

    # ─── Tradingsymbol Builder ─────────────────────────────────

    def _build_tradingsymbol(self, instrument_type, strike, expiry):
        """Build NFO tradingsymbol — delegates to scanner for correct format."""
        return self.scanner._build_tradingsymbol(instrument_type, strike, expiry)

    # ─── Order Placement ─────────────────────────────────────

    def _place_order(self, tradingsymbol, transaction_type, qty, price,
                     leg, instrument_type, strike, expiry_date,
                     signal_type, strangle_id, sl_price=None):
        """
        Place an order. Paper mode: fill immediately. Live: place on Kite.
        Returns position_id or None.
        """
        cfg = self.cfg
        now = datetime.now().isoformat()

        position_id = self.db.add_position(
            strangle_id=strangle_id,
            leg=leg,
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
            signal_type=signal_type,
            status='ACTIVE' if cfg.get('paper_trading_mode', True) else 'PENDING',
        )

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
                    product='MIS',  # Intraday product
                    order_type='MARKET',
                )
                self.db.update_position(position_id, kite_order_id=str(order_id))
                logger.info(f"Kite order placed: {order_id} for {tradingsymbol}")
            except Exception as e:
                logger.error(f"Kite order failed: {e}")
                self.db.update_position(position_id, status='FAILED', notes=str(e))
                return None

        logger.info(f"[{order_status}] {transaction_type} {qty} {tradingsymbol} @ {price:.2f} ({signal_type})")
        return position_id

    def _close_leg(self, pos, exit_price, exit_reason):
        """Close a single position leg."""
        cfg = self.cfg
        self.db.close_position(pos['id'], round(exit_price, 2), exit_reason)

        # Live mode: buy back
        if not cfg.get('paper_trading_mode', True):
            try:
                from services.kite_service import get_kite
                kite = get_kite()
                kite.place_order(
                    variety='regular',
                    exchange='NFO',
                    tradingsymbol=pos['tradingsymbol'],
                    transaction_type='BUY',
                    quantity=pos['qty'],
                    product='MIS',
                    order_type='MARKET',
                )
            except Exception as e:
                logger.error(f"Exit order failed for {pos['tradingsymbol']}: {e}")

        self.db.log_order(
            tradingsymbol=pos['tradingsymbol'],
            transaction_type='BUY',
            qty=pos['qty'],
            price=exit_price,
            order_type='MARKET',
            status='PAPER' if cfg.get('paper_trading_mode', True) else 'PLACED',
            position_id=pos['id'],
            signal_type=exit_reason,
        )

        logger.info(f"[EXIT] {pos['tradingsymbol']} — {exit_reason} @ {exit_price:.2f}")

    # ─── Entry Execution ──────────────────────────────────────

    def execute_strangle_entry(self, signal, scan_result):
        """Execute a strangle entry (sell CE + PE)."""
        ok, reason = self._check_guardrails(is_entry=True)
        if not ok:
            logger.info(f"Entry guardrail: {reason}")
            return None, reason

        cfg = self.cfg
        lots = cfg.get('lots_per_leg', 2)
        qty = lots * LOT_SIZE
        expiry = signal['expiry']
        strangle_id = self.db.get_next_strangle_id()

        # Sell CALL
        call_strike = signal['call_strike']
        call_prem = signal['call_premium']
        call_sl = call_prem * cfg.get('premium_double_trigger', 2.0)
        call_symbol = self._build_tradingsymbol('CE', call_strike, expiry)

        pos_ce = self._place_order(
            tradingsymbol=call_symbol,
            transaction_type='SELL',
            qty=qty,
            price=call_prem,
            leg='CE',
            instrument_type='CE',
            strike=call_strike,
            expiry_date=expiry,
            signal_type='SQUEEZE_SELL_CE',
            strangle_id=strangle_id,
            sl_price=call_sl,
        )

        # Sell PUT
        put_strike = signal['put_strike']
        put_prem = signal['put_premium']
        put_sl = put_prem * cfg.get('premium_double_trigger', 2.0)
        put_symbol = self._build_tradingsymbol('PE', put_strike, expiry)

        pos_pe = self._place_order(
            tradingsymbol=put_symbol,
            transaction_type='SELL',
            qty=qty,
            price=put_prem,
            leg='PE',
            instrument_type='PE',
            strike=put_strike,
            expiry_date=expiry,
            signal_type='SQUEEZE_SELL_PE',
            strangle_id=strangle_id,
            sl_price=put_sl,
        )

        # Log signal
        self.db.log_signal(
            signal_type='SQUEEZE_ENTRY',
            atr_value=scan_result.get('atr'),
            atr_ma=scan_result.get('atr_ma'),
            squeeze_count=scan_result.get('squeeze_count'),
            spot_price=scan_result.get('spot'),
            call_strike=call_strike,
            put_strike=put_strike,
            call_premium=call_prem,
            put_premium=put_prem,
            action_taken=f"SELL {call_strike}CE @{call_prem:.1f} + {put_strike}PE @{put_prem:.1f} x{lots}L",
        )

        logger.info(f"Strangle #{strangle_id}: SELL {call_strike}CE @{call_prem:.1f} + "
                     f"{put_strike}PE @{put_prem:.1f} | {lots} lots")

        return strangle_id, 'OK'

    # ─── Adjustment Execution ─────────────────────────────────

    def execute_adjustment(self, adj, scan_result):
        """
        Execute a leg adjustment (close old leg, open new one).

        Premium-matching logic: new leg targets the OTHER leg's current premium.
        Uses find_strike_by_premium() to scan Kite option chain for the best match.
        """
        ok, reason = self._check_guardrails(is_entry=False)
        if not ok:
            logger.info(f"Adjustment guardrail: {reason}")
            return None, reason

        cfg = self.cfg
        pos_id = adj['position_id']
        active_positions = self.db.get_active_positions()
        pos = next((p for p in active_positions if p['id'] == pos_id), None)
        if not pos:
            return None, f'Position {pos_id} not found'

        # Close the old leg
        exit_price = adj['current_premium']
        action = adj['action']
        self._close_leg(pos, exit_price, action)

        # Find new strike by premium-matching (target = other leg's current premium)
        spot = scan_result.get('spot', 0)
        expiry = pos.get('expiry_date', str(get_current_week_expiry()))
        target_prem = adj.get('target_premium', 20.0)
        min_prem_guard = cfg.get('min_leg_premium', 5.0)
        min_otm = cfg.get('min_otm_distance', 100)
        # No max premium cap on adjustments — match whatever the other leg's premium is
        adj_max_prem = target_prem * 1.5  # Allow some tolerance above target

        new_strike, new_prem = self.scanner.find_strike_by_premium(
            spot, pos['instrument_type'], target_prem, expiry,
            min_prem_guard, adj_max_prem, min_otm)

        if new_strike is None:
            # Fallback to ATR-based
            daily_atr = scan_result.get('daily_atr') or 100
            distance = cfg.get('strike_distance_atr', 1.5)
            if pos['instrument_type'] == 'CE':
                new_strike = round_strike(spot + distance * daily_atr, 'up')
            else:
                new_strike = round_strike(spot - distance * daily_atr, 'down')
            iv = scan_result.get('iv', 0.15)
            dte = scan_result.get('dte', 3)
            T = max(dte, 0.5) / 365.0
            if pos['instrument_type'] == 'CE':
                new_prem = bs_call(spot, new_strike, T, 0.065, iv)
            else:
                new_prem = bs_put(spot, new_strike, T, 0.065, iv)
            logger.warning(f"[NAS] Premium-based strike failed, ATR fallback: "
                           f"{new_strike}{pos['instrument_type']} @{new_prem:.1f}")

        # Place new leg
        new_symbol = self._build_tradingsymbol(pos['instrument_type'], new_strike, expiry)

        new_pos_id = self._place_order(
            tradingsymbol=new_symbol,
            transaction_type='SELL',
            qty=pos['qty'],
            price=round(new_prem, 2),
            leg=pos['leg'],
            instrument_type=pos['instrument_type'],
            strike=new_strike,
            expiry_date=expiry,
            signal_type=f'ADJ_{action}',
            strangle_id=pos['strangle_id'],
            sl_price=round(new_prem * cfg.get('premium_double_trigger', 2.0), 2),
        )

        # Update adjustment count
        if new_pos_id:
            self.db.update_position(new_pos_id,
                                     adjustment_count=(pos.get('adjustment_count', 0) or 0) + 1)

        # Update total adjustments in state
        state = self.db.get_state()
        total_adj = (state.get('total_adjustments', 0) or 0) + 1
        self.db.update_state(total_adjustments=total_adj)

        self.db.log_signal(
            signal_type=action,
            spot_price=spot,
            call_strike=new_strike if pos['instrument_type'] == 'CE' else None,
            put_strike=new_strike if pos['instrument_type'] == 'PE' else None,
            action_taken=f"{action}: {pos['strike']}{pos['instrument_type']} @{exit_price:.1f} → "
                         f"{new_strike}{pos['instrument_type']} @{new_prem:.1f} "
                         f"(target={target_prem:.1f}, other_leg={adj.get('other_leg_premium', 0):.1f})",
        )

        logger.info(f"Adjustment: {pos['strike']}{pos['instrument_type']} → "
                     f"{new_strike}{pos['instrument_type']} @{new_prem:.1f} "
                     f"(matched other leg @ {adj.get('other_leg_premium', 0):.1f})")

        return new_pos_id, 'OK'

    # ─── Exit All ─────────────────────────────────────────────

    def exit_all_positions(self, exit_reason, scan_result=None):
        """Close all active positions."""
        active = self.db.get_active_positions()
        if not active:
            return []

        spot = scan_result.get('spot', 0) if scan_result else 0
        iv = scan_result.get('iv', 0.15) if scan_result else 0.15
        today = date.today()
        results = []

        # Group by strangle for trade recording
        strangle_groups = {}
        for pos in active:
            sid = pos.get('strangle_id', 0)
            if sid not in strangle_groups:
                strangle_groups[sid] = []
            strangle_groups[sid].append(pos)

        for pos in active:
            # Get exit price — always try live quote first, BS fallback
            exit_price = None
            if pos.get('tradingsymbol'):
                try:
                    exit_price = self.scanner.get_live_option_premium(pos['tradingsymbol'])
                except Exception:
                    pass

            if exit_price is None:
                try:
                    exp_str = str(pos.get('expiry_date', ''))[:10]
                    exp_date = date.fromisoformat(exp_str) if exp_str else today + timedelta(days=3)
                    rem_dte = max((exp_date - today).days, 0.5)
                    T = rem_dte / 365.0
                    if pos['instrument_type'] == 'CE':
                        exit_price = bs_call(spot, pos['strike'], T, 0.065, iv)
                    else:
                        exit_price = bs_put(spot, pos['strike'], T, 0.065, iv)
                except Exception:
                    exit_price = 0

            self._close_leg(pos, exit_price, exit_reason)
            results.append({
                'position_id': pos['id'],
                'tradingsymbol': pos['tradingsymbol'],
                'exit_reason': exit_reason,
                'exit_price': round(exit_price, 2),
            })

        # Record trades per strangle
        for sid, positions in strangle_groups.items():
            self._record_trade(sid, positions, exit_reason, spot)

        return results

    def _record_trade(self, strangle_id, positions, exit_reason, spot_at_exit):
        """Record a completed strangle trade."""
        ce_pos = [p for p in positions if p['instrument_type'] == 'CE']
        pe_pos = [p for p in positions if p['instrument_type'] == 'PE']

        call_entry = ce_pos[0]['entry_price'] if ce_pos else 0
        put_entry = pe_pos[0]['entry_price'] if pe_pos else 0
        call_exit = ce_pos[0].get('exit_price', 0) or 0 if ce_pos else 0
        put_exit = pe_pos[0].get('exit_price', 0) or 0 if pe_pos else 0
        call_strike = ce_pos[0]['strike'] if ce_pos else 0
        put_strike = pe_pos[0]['strike'] if pe_pos else 0

        total_collected = call_entry + put_entry
        total_paid = call_exit + put_exit
        lots = (ce_pos[0]['qty'] // LOT_SIZE) if ce_pos else 0
        gross_pnl = (total_collected - total_paid) * LOT_SIZE * lots
        net_pnl = gross_pnl - (80 * 2)  # brokerage: Rs 40/order × 4 legs (2 entry + 2 exit)

        total_adj = sum(p.get('adjustment_count', 0) or 0 for p in positions)
        entry_time = min(p['entry_time'] for p in positions) if positions else None
        spot_at_entry = None  # Would need from signal log

        self.db.add_trade(
            strangle_id=strangle_id,
            trade_date=date.today().isoformat(),
            entry_time=entry_time,
            exit_time=datetime.now().isoformat(),
            spot_at_entry=spot_at_entry,
            spot_at_exit=spot_at_exit,
            call_strike=call_strike,
            put_strike=put_strike,
            call_entry_premium=call_entry,
            put_entry_premium=put_entry,
            call_exit_premium=call_exit,
            put_exit_premium=put_exit,
            total_premium_collected=round(total_collected, 2),
            total_premium_paid=round(total_paid, 2),
            gross_pnl=round(gross_pnl, 2),
            net_pnl=round(net_pnl, 2),
            lots=lots,
            adjustments=total_adj,
            exit_reason=exit_reason,
            expiry_date=positions[0].get('expiry_date') if positions else None,
        )

    # ─── Scan Pipeline ────────────────────────────────────────

    def run_scan(self):
        """
        Full scan pipeline (runs every 5 min):
        1. Scan for ATR squeeze signals
        2. Check exits on active positions
        3. Check adjustments on active positions
        4. Execute entries / adjustments / exits
        5. Update state

        Returns dict with scan results and actions taken.
        """
        logger.info("NAS scan starting...")
        results = {
            'scan': None,
            'exits': [],
            'adjustments': [],
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

            spot = scan['spot']
            iv = scan.get('iv', 0.15)
            active = self.db.get_active_positions()

            # 2. Check exits first
            if active:
                # Get total entry premium for the active strangle
                strangle_ids = set(p.get('strangle_id') for p in active)
                for sid in strangle_ids:
                    sid_positions = [p for p in active if p.get('strangle_id') == sid]
                    total_entry_prem = sum(p['entry_price'] for p in sid_positions)

                    exit_checks = self.scanner.check_exits(
                        sid_positions, spot, total_entry_prem, iv)

                    if exit_checks:
                        exit_reason = exit_checks[0][0]
                        exit_results = self.exit_all_positions(exit_reason, scan)
                        results['exits'] = exit_results
                        # After exit, don't try to enter on same bar
                        self._update_state(scan)
                        return results

            # 3. Check adjustments (use live premiums in live mode)
            if active:
                if not self.cfg.get('paper_trading_mode', True):
                    for pos in active:
                        if pos.get('tradingsymbol'):
                            live_prem = self.scanner.get_live_option_premium(pos['tradingsymbol'])
                            if live_prem is not None:
                                pos['_live_premium'] = live_prem
                adj_checks = self.scanner.check_adjustments(active, spot, iv)
                for adj in adj_checks:
                    adj_id, msg = self.execute_adjustment(adj, scan)
                    if adj_id:
                        results['adjustments'].append(adj)
                    else:
                        results['errors'].append(f"Adjustment skip: {msg}")

            # 4. Check for new entry (only if no active positions)
            active_after = self.db.get_active_positions()
            if not active_after:
                for signal in scan.get('signals', []):
                    if signal['type'] == 'SQUEEZE_ENTRY':
                        sid, msg = self.execute_strangle_entry(signal, scan)
                        if sid:
                            results['entries'].append({
                                'strangle_id': sid,
                                'call_strike': signal['call_strike'],
                                'put_strike': signal['put_strike'],
                                'total_premium': signal['total_premium'],
                            })
                        else:
                            results['errors'].append(f"Entry skip: {msg}")

            # 5. Update state
            self._update_state(scan)

            logger.info(f"NAS scan complete: squeeze={scan['is_squeezing']}, "
                         f"{len(scan.get('signals', []))} signals, "
                         f"{len(results['exits'])} exits, "
                         f"{len(results['adjustments'])} adjustments, "
                         f"{len(results['entries'])} entries")

        except Exception as e:
            logger.error(f"NAS scan error: {e}", exc_info=True)
            results['errors'].append(str(e))

        return results

    def _update_state(self, scan):
        """Update system state from scan results."""
        self.db.update_state(
            atr_value=scan.get('atr'),
            atr_ma=scan.get('atr_ma'),
            is_squeezing=1 if scan.get('is_squeezing') else 0,
            squeeze_count=scan.get('squeeze_count', 0),
            spot_price=scan.get('spot'),
            daily_atr=scan.get('daily_atr'),
            last_scan_time=datetime.now().isoformat(),
        )

        if scan.get('signals'):
            self.db.update_state(last_signal_time=datetime.now().isoformat())

    # ─── EOD Squareoff ────────────────────────────────────────

    def eod_squareoff(self):
        """Mandatory end-of-day squareoff + daily summary."""
        scan = self.scanner.scan()
        exits = self.exit_all_positions('eod_squareoff', scan)

        # Save daily state
        stats = self.db.get_stats()
        today_trades = [t for t in self.db.get_recent_trades(limit=50)
                        if t.get('trade_date') == date.today().isoformat()]

        daily_pnl = sum(t.get('net_pnl', 0) or 0 for t in today_trades)
        wins = sum(1 for t in today_trades if (t.get('net_pnl', 0) or 0) > 0)
        losses = sum(1 for t in today_trades if (t.get('net_pnl', 0) or 0) <= 0)

        self.db.save_daily_state(
            trade_date=date.today().isoformat(),
            trades_taken=len(today_trades),
            adjustments=sum(t.get('adjustments', 0) or 0 for t in today_trades),
            daily_pnl=round(daily_pnl, 2),
            cumulative_pnl=round(stats.get('total_pnl', 0), 2),
            win_count=wins,
            loss_count=losses,
        )

        # Reset daily counters
        self.db.update_state(daily_pnl=0, total_adjustments=0)

        logger.info(f"NAS EOD: {len(exits)} positions closed, daily P&L: Rs {daily_pnl:.0f}")
        return exits

    # ─── Emergency ───────────────────────────────────────────

    def emergency_exit_all(self):
        """Kill switch — close all active positions immediately."""
        scan = self.scanner.scan()
        exits = self.exit_all_positions('EMERGENCY', scan)
        logger.warning(f"NAS EMERGENCY EXIT: closed {len(exits)} positions")
        return len(exits)

    # ─── State for Dashboard ─────────────────────────────────

    def get_full_state(self):
        """Get complete state for dashboard API."""
        state = self.db.get_state()
        stats = self.db.get_stats()
        active = self.db.get_active_positions()
        signals = self.db.get_recent_signals(limit=20)
        trades = self.db.get_recent_trades(limit=30)

        # Group active by leg
        ce_positions = [p for p in active if p.get('leg') == 'CE']
        pe_positions = [p for p in active if p.get('leg') == 'PE']

        # Today's closed positions
        closed_today = self.db.get_today_closed_positions()

        return {
            'state': state,
            'stats': stats,
            'config': {
                'paper_trading_mode': self.cfg.get('paper_trading_mode', True),
                'enabled': self.cfg.get('enabled', True),
                'lots_per_leg': self.cfg.get('lots_per_leg', 2),
                'strike_distance_atr': self.cfg.get('strike_distance_atr', 1.5),
                'premium_double_trigger': self.cfg.get('premium_double_trigger', 2.0),
                'premium_half_trigger': self.cfg.get('premium_half_trigger', 0.5),
                'max_daily_loss': self.cfg.get('max_daily_loss', 15000),
                'eod_squareoff_time': self.cfg.get('eod_squareoff_time', '15:15'),
                'min_squeeze_bars': self.cfg.get('min_squeeze_bars', 1),
                'entry_start_time': self.cfg.get('entry_start_time', '09:30'),
                'entry_end_time': self.cfg.get('entry_end_time', '14:30'),
                'time_exit': self.cfg.get('time_exit', '14:45'),
                'target_entry_premium': self.cfg.get('target_entry_premium', 20.0),
            },
            'positions': {
                'ce': ce_positions,
                'pe': pe_positions,
                'total_active': len(active),
                'closed_today': closed_today,
            },
            'recent_signals': signals,
            'recent_trades': trades,
        }
