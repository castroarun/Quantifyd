"""SENSEX scanner — the same 5-method interface the NAS ATM executors expect, for BSE/BFO.

Drop-in for NasAtmScanner so the SENSEX executors can inherit ALL the NAS rules (per-leg SL,
0.4% move-stop, roll-to-match, partial-fill rollback, broker-truth exit guard) without
reimplementing any of them.

KEY DESIGN CHOICE — symbols are LOOKED UP, never constructed.
SENSEX weeklies and monthlies use different formats:
    weekly   SENSEX2672377000CE   (YY + M + DD + strike + type; month is 1-9 then O/N/D)
    monthly  SENSEX26JUL77000PE   (YY + MON + strike + type)  <- the last expiry of the month
Building those strings by hand means encoding the month-code table AND the "is this the monthly"
rule, and getting either wrong produces an order for a contract that does not exist. Looking the
symbol up in Kite's own instrument dump cannot drift from reality. The dump is cached per day.

SENSEX facts (verified live 2026-07-20): exchange BFO, spot 'BSE:SENSEX', lot 20, strike step 100,
weekly expiry THURSDAY.
"""
import logging
from datetime import date, datetime

logger = logging.getLogger(__name__)

EXCHANGE = "BFO"
SPOT_KEY = "BSE:SENSEX"
LOT_SIZE = 20
STRIKE_STEP = 100
NAME = "SENSEX"

_CACHE = {"day": None, "by_key": {}, "expiries": []}


def _load_instruments(force=False):
    """Cache Kite's BFO SENSEX option list for the day: (expiry, strike, type) -> tradingsymbol."""
    today = date.today()
    if not force and _CACHE["day"] == today and _CACHE["by_key"]:
        return _CACHE["by_key"], _CACHE["expiries"]
    from services.kite_service import get_kite
    kite = get_kite()
    by_key, exps = {}, set()
    for i in kite.instruments(EXCHANGE):
        if i.get("name") != NAME or i.get("instrument_type") not in ("CE", "PE"):
            continue
        exp = i.get("expiry")
        if not exp:
            continue
        if isinstance(exp, datetime):
            exp = exp.date()
        by_key[(exp, int(i["strike"]), i["instrument_type"])] = i["tradingsymbol"]
        if exp >= today:
            exps.add(exp)
    _CACHE.update({"day": today, "by_key": by_key, "expiries": sorted(exps)})
    logger.info("[SENSEX] instruments cached: %d contracts, next expiries %s",
                len(by_key), _CACHE["expiries"][:3])
    return by_key, _CACHE["expiries"]


class SensexScanner:
    """Mirrors NasAtmScanner's interface: get_atm_strike / build_tradingsymbol /
    get_live_option_premium / get_live_spot / get_current_week_expiry."""

    EXCHANGE = EXCHANGE
    SPOT_KEY = SPOT_KEY
    LOT_SIZE = LOT_SIZE
    STRIKE_STEP = STRIKE_STEP

    def get_atm_strike(self, spot):
        if not spot:
            return None
        return int(round(spot / STRIKE_STEP) * STRIKE_STEP)

    def get_current_week_expiry(self, ref_date=None):
        """Nearest SENSEX weekly expiry (Thursday) at/after ref_date — from Kite, not a weekday rule,
        so holiday shifts are handled automatically."""
        ref = ref_date or date.today()
        _, exps = _load_instruments()
        for e in exps:
            if e >= ref:
                return e
        return None

    def build_tradingsymbol(self, instrument_type, strike, expiry):
        """LOOK UP the contract. Returns None if it does not exist (caller must handle)."""
        if isinstance(expiry, str):
            # positions store expiry_date as a 'YYYY-MM-DD' string; the cache is
            # keyed by date objects, so a raw string silently misses every strike
            # (2026-07-29: broke the ATM4 roll — every candidate came back
            # "no contract" and the strangle boundary-exited instead of rolling).
            expiry = datetime.strptime(expiry[:10], '%Y-%m-%d').date()
        elif isinstance(expiry, datetime):
            expiry = expiry.date()
        by_key, _ = _load_instruments()
        sym = by_key.get((expiry, int(strike), instrument_type))
        if not sym:
            # one forced refresh — the dump may be stale on a new listing
            by_key, _ = _load_instruments(force=True)
            sym = by_key.get((expiry, int(strike), instrument_type))
        if not sym:
            logger.warning("[SENSEX] no contract for %s %s %s", expiry, strike, instrument_type)
        return sym

    def get_live_spot(self):
        try:
            from services.kite_service import get_kite
            q = get_kite().ltp([SPOT_KEY])
            return q[SPOT_KEY]["last_price"]
        except Exception as e:
            logger.warning("[SENSEX] spot fetch failed: %s", e)
            return None

    def get_live_option_premium(self, tradingsymbol):
        if not tradingsymbol:
            return None
        try:
            from services.kite_service import get_kite
            key = f"{EXCHANGE}:{tradingsymbol}"
            q = get_kite().ltp([key])
            v = q.get(key)
            return v["last_price"] if v else None
        except Exception as e:
            logger.warning("[SENSEX] premium fetch failed for %s: %s", tradingsymbol, e)
            return None

    def scan(self, df_5min=None, daily_df=None):
        """The 9:16 systems enter on time, not on a squeeze signal, so no scan is needed.
        Present only to satisfy the interface."""
        return {}
