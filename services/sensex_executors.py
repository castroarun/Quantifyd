"""SENSEX 9:16 executors — the SAME rules as the NAS NIFTY systems, on BSE/BFO.

WHY SENSEX: NIFTY expires Tuesday, SENSEX Thursday. The NAS edge lives on 0/1-DTE (research/79),
so NIFTY covers Mon/Tue and SENSEX covers Wed(DTE1)/Thu(DTE0) — the days NIFTY has nothing to
harvest and where real money is now switched off.

These subclass the NIFTY executors and override ONLY the venue (exchange, spot, lot, strike step,
expiry, symbol lookup). Every rule is inherited, unforked:
    ATM   per-leg 30% SL, naked-survivor trailing stop
    ATM2  +/-0.4% underlying move-stop, re-center (per-leg SL deliberately disabled)
    ATM4  roll-to-match on first SL, max_rolls=1
plus the safety machinery: partial-fill rollback (no naked leg), fill confirmation, broker-truth
exit guard, day-matrix gate, kill switch and manual freeze.

SAFETY: ships PAPER. The live SENSEX evidence is 1 day; the 59-day chain backfill has not run yet.
Flip to live only from the day matrix, and only once that evidence exists.
"""
import logging

from services.nas_atm_executor import NasAtmExecutor
from services.nas_atm2_executor import NasAtm2Executor
from services.nas_atm4_executor import NasAtm4Executor
from services.sensex_scanner import SensexScanner, EXCHANGE, SPOT_KEY, LOT_SIZE
from services.sensex_db import (get_sensex_atm_db, get_sensex_atm2_db, get_sensex_atm4_db)

logger = logging.getLogger(__name__)


class _SensexVenueMixin:
    """Everything venue-specific, in one place. Rules come from the NIFTY base classes."""
    EXCHANGE = EXCHANGE            # 'BFO'
    SPOT_KEY = SPOT_KEY            # 'BSE:SENSEX'
    LOT_SIZE = LOT_SIZE            # 20 (verified live 2026-07-20; config's 10 was stale)

    def _init_venue(self):
        self.scanner = SensexScanner()
        # strike step 100 (SENSEX) — the base reads cfg['strike_interval'] for ATM rounding
        self.cfg.setdefault('strike_interval', 100)

    def _build_tradingsymbol(self, instrument_type, strike, expiry):
        """Look the contract up rather than construct it: SENSEX weeklies and monthlies use
        different formats (SENSEX2672377700CE vs SENSEX26JUL77700CE) and a hand-built string
        silently produces a non-existent symbol."""
        return self.scanner.build_tradingsymbol(instrument_type, strike, expiry)


class SensexAtmExecutor(_SensexVenueMixin, NasAtmExecutor):
    """SENSEX ATM straddle — per-leg 30% SL + naked-survivor trail (the 916-ATM rules)."""

    def __init__(self, config=None):
        from config import SENSEX_ATM_DEFAULTS
        super().__init__(config=config or SENSEX_ATM_DEFAULTS)
        self.db = get_sensex_atm_db()
        self._init_venue()


class SensexAtm2Executor(_SensexVenueMixin, NasAtm2Executor):
    """SENSEX ATM 2.0 — +/-0.4% underlying move-stop with re-center (the 916-ATM2 rules)."""

    def __init__(self, config=None):
        from config import SENSEX_ATM2_DEFAULTS
        super().__init__(config=config or SENSEX_ATM2_DEFAULTS)
        self.db = get_sensex_atm2_db()
        self._init_venue()


class SensexAtm4Executor(_SensexVenueMixin, NasAtm4Executor):
    """SENSEX ATM V4 — roll-to-match on the first SL, one roll (the 916-ATM4 rules)."""

    def __init__(self, config=None):
        from config import SENSEX_ATM4_DEFAULTS
        super().__init__(config=config or SENSEX_ATM4_DEFAULTS)
        self.db = get_sensex_atm4_db()
        self._init_venue()
