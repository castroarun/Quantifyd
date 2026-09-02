"""Dad's (secondary) Zerodha account — Kite client, fully isolated from the primary
kite_service so it can never touch the live trading account's session or tokens.

Separate api_key/secret + a separate token file (backtest_data/dad_access_token.json),
refreshed daily by scripts/dad_auto_login.py (TOTP).

Two clients:
  - get_dad_kite()          READ-ONLY — place/modify/cancel raise. Use everywhere by default.
  - get_dad_kite_trading()  ORDER-CAPABLE — used ONLY by the explicit order endpoints
                            (/api/holdings/order, /api/holdings/exit) that Arun drives by hand.

.env (VPS, never committed):
  DAD_KITE_API_KEY, DAD_KITE_API_SECRET, DAD_KITE_USER_ID, DAD_KITE_PASSWORD, DAD_KITE_TOTP_SECRET
"""
import os
import json
import logging
from pathlib import Path

from kiteconnect import KiteConnect

logger = logging.getLogger(__name__)

DAD_KITE_API_KEY = os.getenv("DAD_KITE_API_KEY", "")
DAD_KITE_API_SECRET = os.getenv("DAD_KITE_API_SECRET", "")

_DATA_DIR = Path(__file__).resolve().parent.parent / "backtest_data"
DAD_TOKEN_FILE = _DATA_DIR / "dad_access_token.json"


def is_configured() -> bool:
    return bool(DAD_KITE_API_KEY)


def get_dad_access_token() -> str:
    try:
        if DAD_TOKEN_FILE.exists():
            return json.loads(DAD_TOKEN_FILE.read_text()).get("access_token", "")
    except Exception:
        pass
    return ""


def save_dad_access_token(token: str, request_token: str = "") -> None:
    try:
        data = {"access_token": token}
        if request_token:
            data["request_token"] = request_token
        DAD_TOKEN_FILE.write_text(json.dumps(data))
        logger.info("[DAD] access token saved")
    except Exception as e:
        logger.error(f"[DAD] failed to save access token: {e}")


def _block_orders(kite: KiteConnect) -> KiteConnect:
    """Hard-disable any order mutation on Dad's account (read-only guarantee)."""
    def _blocked(*_a, **_k):
        raise RuntimeError("Dad's account is read-only — order placement is disabled.")
    kite.place_order = _blocked      # type: ignore[assignment]
    kite.modify_order = _blocked     # type: ignore[assignment]
    kite.cancel_order = _blocked     # type: ignore[assignment]
    kite.place_gtt = _blocked        # type: ignore[assignment]
    return kite


def _raw_dad_kite() -> KiteConnect:
    if not DAD_KITE_API_KEY:
        raise RuntimeError("DAD_KITE_API_KEY not set — Dad's account is not configured.")
    kite = KiteConnect(api_key=DAD_KITE_API_KEY)
    tok = get_dad_access_token()
    if tok:
        kite.set_access_token(tok)
    return kite


def get_dad_kite() -> KiteConnect:
    """READ-ONLY KiteConnect for Dad's account (orders blocked). Default everywhere."""
    return _block_orders(_raw_dad_kite())


def get_dad_kite_trading() -> KiteConnect:
    """ORDER-CAPABLE KiteConnect for Dad's account.

    Deliberately NOT order-blocked — reserved for the manual buy/exit endpoints that Arun
    drives from the Holdings page. Never import this into any automated/background path.
    """
    return _raw_dad_kite()
