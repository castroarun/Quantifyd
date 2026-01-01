"""
Zerodha Kite Connect Service
============================

Handles OAuth authentication, token management, and API access for Zerodha Kite.
Adapted from quantflow/kiteconnect01 for Covered Calls Backtester.
"""

import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from kiteconnect import KiteConnect

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment
KITE_API_KEY = os.getenv("KITE_API_KEY", "")
KITE_API_SECRET = os.getenv("KITE_API_SECRET", "")
KITE_REDIRECT_URL = os.getenv("KITE_REDIRECT_URL", "http://127.0.0.1:5000/zerodha/callback")

# Token file location - in backtest_data folder
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "backtest_data"
DATA_DIR.mkdir(exist_ok=True)
TOKEN_FILE = DATA_DIR / "access_token.json"


def get_access_token() -> str:
    """Get the current access token from file"""
    try:
        if TOKEN_FILE.exists():
            data = json.loads(TOKEN_FILE.read_text())
            return data.get("access_token", "")
    except Exception:
        pass
    return ""


def get_request_token() -> str:
    """Get the request token from file (used for refresh)"""
    try:
        if TOKEN_FILE.exists():
            data = json.loads(TOKEN_FILE.read_text())
            return data.get("request_token", "")
    except Exception:
        pass
    return ""


def save_access_token(token: str, request_token: str = "") -> None:
    """Save access token (and optionally request token) to file"""
    try:
        data = {"access_token": token}
        if request_token:
            data["request_token"] = request_token
        TOKEN_FILE.write_text(json.dumps(data))
        logger.info("Access token saved successfully")
    except Exception as e:
        logger.error(f"Failed to save access token: {e}")


def invalidate_token() -> bool:
    """
    Invalidate the current access token
    Returns True if successful, False otherwise
    """
    try:
        access_token = get_access_token()
        if not access_token:
            logger.warning("No access token to invalidate")
            return False

        kite = KiteConnect(api_key=KITE_API_KEY)
        kite.set_access_token(access_token)

        # Call invalidate API
        result = kite.invalidate_access_token(access_token)
        logger.info(f"Access token invalidated successfully: {result}")

        # Clear the token file
        save_access_token("")

        return True
    except Exception as e:
        logger.error(f"Failed to invalidate token: {e}")
        # Clear token anyway
        save_access_token("")
        return False


def refresh_token(request_token: str = None) -> str:
    """
    Refresh the access token by generating a new session

    Args:
        request_token: The request token from OAuth flow (optional if stored)

    Returns:
        New access token or empty string if failed
    """
    try:
        # Use provided request_token or get from storage
        if not request_token:
            request_token = get_request_token()

        if not request_token:
            logger.error("No request token available for refresh")
            return ""

        if not KITE_API_SECRET:
            logger.error("KITE_API_SECRET not configured")
            return ""

        # Create new Kite instance
        kite = KiteConnect(api_key=KITE_API_KEY)

        # Generate new session
        logger.info("Generating new session with request token...")
        data = kite.generate_session(request_token, api_secret=KITE_API_SECRET)

        new_access_token = data.get("access_token")
        if not new_access_token:
            logger.error("No access token in response")
            return ""

        # Save new token
        save_access_token(new_access_token, request_token)
        logger.info("New access token generated and saved")

        return new_access_token

    except Exception as e:
        logger.error(f"Failed to refresh token: {e}")
        return ""


def invalidate_and_refresh(request_token: str = None) -> str:
    """
    Convenience function: Invalidate current token and generate new one

    Args:
        request_token: The request token from OAuth flow (optional if stored)

    Returns:
        New access token or empty string if failed
    """
    logger.info("Invalidating current token and generating new session...")

    # Step 1: Invalidate current token
    invalidate_token()

    # Step 2: Generate new session
    new_token = refresh_token(request_token)

    if new_token:
        logger.info("Token refresh completed successfully")
    else:
        logger.error("Token refresh failed")

    return new_token


def get_kite() -> KiteConnect:
    """Get KiteConnect instance with current access token"""
    if not KITE_API_KEY:
        raise RuntimeError("KITE_API_KEY missing. Set it in .env")
    kite = KiteConnect(api_key=KITE_API_KEY)
    access_token = get_access_token()
    if access_token:
        kite.set_access_token(access_token)
    return kite


def get_kite_with_refresh() -> KiteConnect:
    """
    Get KiteConnect instance with automatic token refresh on failure

    This is a smart wrapper that will:
    1. Try with current token
    2. If it fails, invalidate and refresh
    3. Return new Kite instance
    """
    kite = get_kite()

    # Test if token is valid
    try:
        kite.profile()
        logger.info("Current token is valid")
        return kite
    except Exception as e:
        error_msg = str(e).lower()

        # Check if it's a token/permission error
        if "token" in error_msg or "permission" in error_msg or "session" in error_msg:
            logger.warning(f"Token appears invalid: {e}")
            logger.info("Attempting to refresh token...")

            # Try to refresh
            new_token = invalidate_and_refresh()

            if new_token:
                # Create new Kite instance with fresh token
                new_kite = KiteConnect(api_key=KITE_API_KEY)
                new_kite.set_access_token(new_token)

                # Verify it works
                try:
                    new_kite.profile()
                    logger.info("Token refresh successful")
                    return new_kite
                except Exception as verify_error:
                    logger.error(f"Refreshed token still invalid: {verify_error}")
                    raise RuntimeError("Token refresh failed - please login again via /login")
            else:
                raise RuntimeError("Could not refresh token - please login again via /login")
        else:
            # Not a token error, re-raise
            raise


def get_login_url() -> str:
    """Get the Zerodha login URL for OAuth"""
    if not KITE_API_KEY:
        raise RuntimeError("KITE_API_KEY missing. Set it in .env")
    kite = KiteConnect(api_key=KITE_API_KEY)
    return kite.login_url()


def is_authenticated() -> bool:
    """Check if we have a valid access token"""
    access_token = get_access_token()
    if not access_token:
        return False

    try:
        kite = get_kite()
        kite.profile()
        return True
    except Exception:
        return False
