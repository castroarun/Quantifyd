"""
NAS ATM — Nifty ATM Strangle — Scanner
=========================================
Minimal scanner for ATM strangle system.
Reuses expiry logic and instruments cache from NAS OTM scanner.
ATM strike selection is simple: round spot to nearest strike interval.
"""

import logging
from datetime import date

from services.nas_scanner import (
    NasScanner, LOT_SIZE, STRIKE_STEP,
    get_current_week_expiry, is_expiry_day,
)

logger = logging.getLogger(__name__)


class NasAtmScanner:
    """
    Lightweight scanner for NAS ATM strangle system.

    Delegates heavy lifting (5-min bars, ATR squeeze, instruments cache,
    tradingsymbol building, live quotes) to a shared NasScanner instance.
    Adds ATM-specific strike selection.
    """

    def __init__(self, config: dict):
        self.cfg = config
        # Reuse NasScanner for data loading, indicators, instruments cache,
        # tradingsymbol building, and live quote fetching
        self._scanner = NasScanner(config)

    def get_atm_strike(self, spot):
        """Round spot to nearest strike interval."""
        interval = self.cfg.get('strike_interval', 50)
        return round(spot / interval) * interval

    def build_tradingsymbol(self, instrument_type, strike, expiry):
        """Build NFO tradingsymbol — delegates to NasScanner."""
        return self._scanner._build_tradingsymbol(instrument_type, strike, expiry)

    def get_live_option_premium(self, tradingsymbol):
        """Get live option premium from Kite for a specific contract."""
        return self._scanner.get_live_option_premium(tradingsymbol)

    def get_live_spot(self):
        """Get current NIFTY spot price from Kite API."""
        return self._scanner.get_live_spot()

    def get_current_week_expiry(self, ref_date=None):
        """Get nearest NIFTY weekly expiry date."""
        return get_current_week_expiry(ref_date)

    def scan(self, df_5min=None, daily_df=None):
        """
        Run ATR squeeze scan via the underlying NasScanner.
        Returns the full scan result dict (atr, squeeze state, spot, etc.).
        The executor uses this for squeeze detection; strike selection is ATM.
        """
        return self._scanner.scan(df_5min, daily_df)
