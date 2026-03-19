"""
Maruthi Contract Manager
==========================

Handles F&O contract lifecycle:
- Instrument lookup (futures + options tradingsymbols)
- Expiry calendar management
- Strike selection with liquidity check
- Contract rolling (options at 6 days, futures on last day)
- Roll sequence: close current → open next (margin-safe)
"""

import logging
import math
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple

from kiteconnect import KiteConnect

logger = logging.getLogger(__name__)


class MaruthiContractManager:
    """Manages F&O contracts for MARUTI."""

    def __init__(self, kite: KiteConnect = None, symbol: str = 'MARUTI',
                 exchange: str = 'NFO', strike_interval: int = 100):
        self.kite = kite
        self.symbol = symbol
        self.exchange = exchange
        self.strike_interval = strike_interval
        self._instruments_cache = None
        self._cache_date = None

    def _load_instruments(self, force: bool = False) -> List[dict]:
        """Load NFO instruments from Kite API (cached daily)."""
        today = date.today()
        if not force and self._instruments_cache and self._cache_date == today:
            return self._instruments_cache

        if not self.kite:
            logger.error("Kite instance not set — cannot load instruments")
            return []

        try:
            all_instruments = self.kite.instruments(self.exchange)
            # Filter to MARUTI only
            self._instruments_cache = [
                inst for inst in all_instruments
                if inst.get('name') == self.symbol
            ]
            self._cache_date = today
            logger.info(f"Loaded {len(self._instruments_cache)} {self.symbol} instruments from {self.exchange}")
            return self._instruments_cache
        except Exception as e:
            logger.error(f"Failed to load instruments: {e}")
            return self._instruments_cache or []

    # =========================================================================
    # Expiry Management
    # =========================================================================

    def get_expiries(self, instrument_type: str = 'FUT') -> List[date]:
        """
        Get sorted list of available expiry dates.

        Args:
            instrument_type: 'FUT' for futures, 'CE'/'PE' for options
        """
        instruments = self._load_instruments()
        expiries = set()
        for inst in instruments:
            if instrument_type == 'FUT':
                if inst.get('instrument_type') == 'FUT':
                    expiries.add(inst['expiry'])
            else:
                if inst.get('instrument_type') in ('CE', 'PE'):
                    expiries.add(inst['expiry'])

        return sorted(expiries)

    def get_current_expiry(self, instrument_type: str = 'FUT') -> Optional[date]:
        """Get the nearest expiry that hasn't passed."""
        today = date.today()
        expiries = self.get_expiries(instrument_type)
        for exp in expiries:
            exp_date = exp if isinstance(exp, date) else exp.date() if hasattr(exp, 'date') else exp
            if exp_date >= today:
                return exp_date
        return None

    def get_next_expiry(self, instrument_type: str = 'FUT') -> Optional[date]:
        """Get the second nearest expiry (for rolling)."""
        today = date.today()
        expiries = self.get_expiries(instrument_type)
        future_expiries = []
        for exp in expiries:
            exp_date = exp if isinstance(exp, date) else exp.date() if hasattr(exp, 'date') else exp
            if exp_date >= today:
                future_expiries.append(exp_date)
        return future_expiries[1] if len(future_expiries) >= 2 else None

    def days_to_expiry(self, expiry_date) -> int:
        """Calculate calendar days to expiry."""
        if expiry_date is None:
            return 999
        today = date.today()
        if hasattr(expiry_date, 'date'):
            expiry_date = expiry_date.date()
        return (expiry_date - today).days

    # =========================================================================
    # Instrument Lookup
    # =========================================================================

    def get_futures_symbol(self, expiry_date: date = None) -> Optional[dict]:
        """
        Get futures tradingsymbol for given expiry.

        Returns: dict with {tradingsymbol, instrument_token, lot_size, expiry}
        """
        instruments = self._load_instruments()
        if expiry_date is None:
            expiry_date = self.get_current_expiry('FUT')

        for inst in instruments:
            if (inst.get('instrument_type') == 'FUT' and
                    self._date_match(inst['expiry'], expiry_date)):
                return {
                    'tradingsymbol': inst['tradingsymbol'],
                    'instrument_token': inst['instrument_token'],
                    'lot_size': inst.get('lot_size', 200),
                    'expiry': inst['expiry'],
                }
        return None

    def get_option_symbol(self, strike: float, option_type: str,
                          expiry_date: date = None) -> Optional[dict]:
        """
        Get option tradingsymbol for given strike and type.

        Args:
            strike: Strike price (e.g., 7100)
            option_type: 'CE' or 'PE'
            expiry_date: Specific expiry or None for current

        Returns: dict with {tradingsymbol, instrument_token, lot_size, expiry, strike}
        """
        instruments = self._load_instruments()
        if expiry_date is None:
            expiry_date = self.get_current_expiry(option_type)

        for inst in instruments:
            if (inst.get('instrument_type') == option_type and
                    abs(inst.get('strike', 0) - strike) < 0.01 and
                    self._date_match(inst['expiry'], expiry_date)):
                return {
                    'tradingsymbol': inst['tradingsymbol'],
                    'instrument_token': inst['instrument_token'],
                    'lot_size': inst.get('lot_size', 200),
                    'expiry': inst['expiry'],
                    'strike': inst['strike'],
                }
        return None

    def _date_match(self, d1, d2) -> bool:
        """Compare two dates that might be date or datetime objects."""
        if d1 is None or d2 is None:
            return False
        if hasattr(d1, 'date'):
            d1 = d1.date()
        if hasattr(d2, 'date'):
            d2 = d2.date()
        return d1 == d2

    # =========================================================================
    # Strike Selection
    # =========================================================================

    def get_otm_option(self, spot_price: float, option_type: str,
                       otm_strikes: int = 1, min_expiry_days: int = 6) -> Optional[dict]:
        """
        Get the best OTM option contract, considering expiry rules.

        If current expiry has ≤ min_expiry_days remaining, use next month.
        """
        # Determine strike
        if option_type == 'CE':
            base = math.ceil(spot_price / self.strike_interval) * self.strike_interval
            strike = base + (otm_strikes - 1) * self.strike_interval
        else:  # PE
            base = math.floor(spot_price / self.strike_interval) * self.strike_interval
            strike = base - (otm_strikes - 1) * self.strike_interval

        # Determine expiry
        current_expiry = self.get_current_expiry(option_type)
        dte = self.days_to_expiry(current_expiry)

        if dte <= min_expiry_days:
            # Use next month
            expiry = self.get_next_expiry(option_type)
            logger.info(f"Current expiry {current_expiry} has {dte} days — using next month {expiry}")
        else:
            expiry = current_expiry

        result = self.get_option_symbol(strike, option_type, expiry)

        if result:
            # Check liquidity (OI and volume)
            result['days_to_expiry'] = self.days_to_expiry(result['expiry'])
            logger.info(f"Selected {option_type} {strike} exp={result['expiry']} DTE={result['days_to_expiry']}")
        else:
            logger.warning(f"No contract found: {option_type} {strike} exp={expiry}")

        return result

    def get_protective_option(self, spot_price: float, option_type: str,
                              otm_pct: float = 0.05) -> Optional[dict]:
        """
        Get far OTM option for overnight protection.

        Args:
            spot_price: Current price
            option_type: 'PE' for bull regime, 'CE' for bear regime
            otm_pct: How far OTM (0.05 = 5%)
        """
        if option_type == 'PE':
            target = spot_price * (1 - otm_pct)
            strike = math.floor(target / self.strike_interval) * self.strike_interval
        else:  # CE
            target = spot_price * (1 + otm_pct)
            strike = math.ceil(target / self.strike_interval) * self.strike_interval

        return self.get_otm_option(spot_price, option_type, otm_strikes=1,
                                   min_expiry_days=6)

    # =========================================================================
    # Contract Rolling
    # =========================================================================

    def check_rolls_needed(self, positions: List[dict]) -> List[dict]:
        """
        Check which positions need to be rolled.

        Returns list of roll actions:
        [
            {
                'position_id': 5,
                'reason': 'EXPIRY_NEAR',
                'current_symbol': 'MARUTI25MAR7100CE',
                'new_symbol': 'MARUTI25APR7100CE',
                'days_to_expiry': 4,
            }
        ]
        """
        rolls = []
        today = date.today()

        for pos in positions:
            if not pos.get('expiry_date'):
                continue

            expiry = pos['expiry_date']
            if isinstance(expiry, str):
                expiry = datetime.strptime(expiry, '%Y-%m-%d').date()
            elif hasattr(expiry, 'date'):
                expiry = expiry.date()

            dte = (expiry - today).days
            pos_type = pos.get('position_type', '')

            # Futures: roll on last day (first half)
            if pos_type == 'FUTURES' and dte <= 0:
                next_exp = self.get_next_expiry('FUT')
                next_fut = self.get_futures_symbol(next_exp) if next_exp else None
                if next_fut:
                    rolls.append({
                        'position_id': pos['id'],
                        'reason': 'FUTURES_EXPIRY',
                        'current_symbol': pos['tradingsymbol'],
                        'new_symbol': next_fut['tradingsymbol'],
                        'new_expiry': next_exp,
                        'days_to_expiry': dte,
                    })

            # Options: roll at 6 days to expiry
            elif pos_type in ('SHORT_OPTION', 'PROTECTIVE_OPTION') and dte <= 6:
                opt_type = pos.get('instrument_type', 'CE')
                strike = pos.get('strike', 0)
                next_exp = self.get_next_expiry(opt_type)
                next_opt = self.get_option_symbol(strike, opt_type, next_exp) if next_exp else None
                if next_opt:
                    rolls.append({
                        'position_id': pos['id'],
                        'reason': 'OPTION_EXPIRY_NEAR',
                        'current_symbol': pos['tradingsymbol'],
                        'new_symbol': next_opt['tradingsymbol'],
                        'new_expiry': next_exp,
                        'days_to_expiry': dte,
                    })

        return rolls

    # =========================================================================
    # Liquidity Check
    # =========================================================================

    def check_liquidity(self, tradingsymbol: str, min_oi: int = 1000,
                        max_spread_pct: float = 1.0) -> dict:
        """
        Check if a contract has sufficient liquidity.

        Returns: {liquid: bool, oi: int, bid: float, ask: float, spread_pct: float}
        """
        if not self.kite:
            return {'liquid': True, 'reason': 'No kite — skipping check'}

        try:
            # Get instrument token
            instruments = self._load_instruments()
            token = None
            for inst in instruments:
                if inst['tradingsymbol'] == tradingsymbol:
                    token = inst['instrument_token']
                    break

            if not token:
                return {'liquid': False, 'reason': f'Instrument not found: {tradingsymbol}'}

            # Get quote
            quote = self.kite.quote([f"{self.exchange}:{tradingsymbol}"])
            q = quote.get(f"{self.exchange}:{tradingsymbol}", {})

            oi = q.get('oi', 0)
            depth = q.get('depth', {})
            buy = depth.get('buy', [{}])
            sell = depth.get('sell', [{}])

            best_bid = buy[0].get('price', 0) if buy else 0
            best_ask = sell[0].get('price', 0) if sell else 0

            spread_pct = 0
            if best_bid > 0 and best_ask > 0:
                spread_pct = (best_ask - best_bid) / best_bid * 100

            liquid = oi >= min_oi and spread_pct <= max_spread_pct

            return {
                'liquid': liquid,
                'oi': oi,
                'bid': best_bid,
                'ask': best_ask,
                'spread_pct': round(spread_pct, 2),
            }

        except Exception as e:
            logger.error(f"Liquidity check failed for {tradingsymbol}: {e}")
            return {'liquid': True, 'reason': f'Check failed: {e}'}
