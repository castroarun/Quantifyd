"""
NAS 916 — Executor subclasses for all 4 x 9:16 AM Entry variants
=================================================================
Same strategy logic as their parents, but enter at 9:16 AM regardless
of ATR squeeze state. Each uses its own DB for isolated tracking.

OTM 916: overrides run_scan() to skip squeeze requirement
ATM/ATM2/ATM4 916: just swap DB (ATM entry logic has no squeeze check)
"""

import logging
from datetime import datetime

from services.nas_executor import NasExecutor
from services.nas_atm_executor import NasAtmExecutor
from services.nas_atm2_executor import NasAtm2Executor
from services.nas_atm4_executor import NasAtm4Executor
from services.nas_scanner import NasScanner
from services.nas_atm_scanner import NasAtmScanner
from services.nas_916_db import (
    get_nas_916_otm_db, get_nas_916_atm_db,
    get_nas_916_atm2_db, get_nas_916_atm4_db,
)
from config import (
    NAS_916_OTM_DEFAULTS, NAS_916_ATM_DEFAULTS,
    NAS_916_ATM2_DEFAULTS, NAS_916_ATM4_DEFAULTS,
)

logger = logging.getLogger(__name__)


# --- OTM 916 ---

class Nas916OtmExecutor(NasExecutor):
    """
    OTM 916: same OTM strangle logic but enters at 9:16 AM
    without waiting for ATR squeeze.
    """

    def __init__(self, config: dict = None):
        self.cfg = config or dict(NAS_916_OTM_DEFAULTS)
        self.db = get_nas_916_otm_db()
        self.scanner = NasScanner(self.cfg)

    def run_scan(self):
        """
        Override: build entry signal from scan data even when no squeeze.
        Everything else (exits, adjustments, state) runs identically.
        """
        logger.info("[NAS-916-OTM] scan starting...")
        results = {
            'scan': None, 'exits': [], 'adjustments': [],
            'entries': [], 'errors': [],
        }

        try:
            scan = self.scanner.scan()
            results['scan'] = scan

            if scan.get('error'):
                results['errors'].append(scan['error'])
                return results

            spot = scan['spot']
            iv = scan.get('iv', 0.15)
            active = self.db.get_active_positions()

            # 1. Exits
            if active:
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
                        self._update_state(scan)
                        return results

            # 2. Adjustments
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

            # 3. Entry — skip squeeze, use scan data directly
            active_after = self.db.get_active_positions()
            if not active_after:
                # Check if scan already has a SQUEEZE_ENTRY signal (squeeze happened to be active)
                existing_signals = [s for s in scan.get('signals', []) if s['type'] == 'SQUEEZE_ENTRY']

                if existing_signals:
                    signal = existing_signals[0]
                else:
                    # Build signal without squeeze — use the computed strikes/premiums
                    call_strike = scan.get('call_strike')
                    put_strike = scan.get('put_strike')
                    call_prem = scan.get('call_premium', 0)
                    put_prem = scan.get('put_premium', 0)

                    if call_strike and put_strike and call_prem > 0 and put_prem > 0:
                        # Check filter blocks (time window, expiry day, etc.) but NOT squeeze
                        block = scan.get('filters', {}).get('block_reason')
                        if block:
                            results['errors'].append(f"Entry blocked: {block}")
                            signal = None
                        else:
                            signal = {
                                'type': 'SQUEEZE_ENTRY',
                                'action': 'SELL STRANGLE',
                                'call_strike': call_strike,
                                'put_strike': put_strike,
                                'call_premium': call_prem,
                                'put_premium': put_prem,
                                'total_premium': round(call_prem + put_prem, 2),
                                'expiry': scan.get('expiry'),
                                'dte': scan.get('dte'),
                            }
                    else:
                        signal = None
                        results['errors'].append('No valid strikes/premiums for 916 entry')

                if signal:
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

            # 4. Update state
            self._update_state(scan)

            logger.info(f"[NAS-916-OTM] scan complete: "
                        f"{len(results['exits'])} exits, "
                        f"{len(results['adjustments'])} adjustments, "
                        f"{len(results['entries'])} entries")

        except Exception as e:
            logger.error(f"[NAS-916-OTM] scan error: {e}", exc_info=True)
            results['errors'].append(str(e))

        return results

    def get_full_state(self):
        result = super().get_full_state()
        result['config']['entry_mode'] = '916'
        result['config']['skip_squeeze'] = True
        return result


# --- ATM 916 ---

class Nas916AtmExecutor(NasAtmExecutor):
    """ATM 916: same ATM logic, own DB, enter at 9:16."""

    def __init__(self, config: dict = None):
        self.cfg = config or dict(NAS_916_ATM_DEFAULTS)
        self.db = get_nas_916_atm_db()
        self.scanner = NasAtmScanner(self.cfg)

    def get_full_state(self):
        result = super().get_full_state()
        result['config']['entry_mode'] = '916'
        return result


# --- ATM2 916 ---

class Nas916Atm2Executor(NasAtm2Executor):
    """ATM2 916: same ATM2 (exit-both-on-SL) logic, own DB, enter at 9:16."""

    def __init__(self, config: dict = None):
        self.cfg = config or dict(NAS_916_ATM2_DEFAULTS)
        self.db = get_nas_916_atm2_db()
        self.scanner = NasAtmScanner(self.cfg)

    def get_full_state(self):
        result = super().get_full_state()
        result['config']['entry_mode'] = '916'
        result['config']['exit_both_on_sl'] = True
        result['config']['re_enter_on_sl'] = False
        result['config']['trail_to_cost_on_sl'] = False
        return result


# --- ATM4 916 ---

class Nas916Atm4Executor(NasAtm4Executor):
    """ATM4 916: same ATM4 (roll-to-match) logic, own DB, enter at 9:16."""

    def __init__(self, config: dict = None):
        self.cfg = config or dict(NAS_916_ATM4_DEFAULTS)
        self.db = get_nas_916_atm4_db()
        self.scanner = NasAtmScanner(self.cfg)

    def get_full_state(self):
        result = super().get_full_state()
        result['config']['entry_mode'] = '916'
        result['config']['max_rolls'] = self.cfg.get('max_rolls', 1)
        result['config']['roll_to_match'] = True
        result['config']['trail_to_cost_on_sl'] = False
        result['config']['re_enter_on_sl'] = False
        return result
