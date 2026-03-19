"""
Kite TOTP Auto-Login Service
==============================

Automated daily token refresh for Zerodha Kite Connect using TOTP.
No manual browser login required.

Flow:
1. POST to Kite login with user_id + password
2. Submit TOTP (generated from secret key via pyotp)
3. Get request_token from redirect URL
4. Exchange request_token for access_token via Kite API
5. Save access_token for the day

Requirements:
- pip install pyotp requests
- KITE_USER_ID, KITE_PASSWORD, KITE_TOTP_SECRET env vars (or in config)
"""

import os
import re
import time
import logging
import requests
from urllib.parse import urlparse, parse_qs

try:
    import pyotp
except ImportError:
    pyotp = None

from kiteconnect import KiteConnect
from services.kite_service import (
    KITE_API_KEY, KITE_API_SECRET,
    save_access_token, get_access_token
)

logger = logging.getLogger(__name__)


def get_totp_config() -> dict:
    """Get TOTP credentials from env vars."""
    return {
        'user_id': os.getenv('KITE_USER_ID', ''),
        'password': os.getenv('KITE_PASSWORD', ''),
        'totp_secret': os.getenv('KITE_TOTP_SECRET', ''),
    }


def generate_totp(secret: str) -> str:
    """Generate current TOTP code from secret key."""
    if not pyotp:
        raise ImportError("pyotp not installed. Run: pip install pyotp")
    if not secret:
        raise ValueError("TOTP secret not configured. Set KITE_TOTP_SECRET env var.")
    totp = pyotp.TOTP(secret)
    return totp.now()


def auto_login() -> str:
    """
    Fully automated Kite login using TOTP.

    Returns:
        access_token (str) on success, empty string on failure.
    """
    creds = get_totp_config()
    if not all([creds['user_id'], creds['password'], creds['totp_secret']]):
        logger.error("Missing TOTP credentials. Set KITE_USER_ID, KITE_PASSWORD, KITE_TOTP_SECRET")
        return ""

    if not KITE_API_KEY or not KITE_API_SECRET:
        logger.error("Missing KITE_API_KEY or KITE_API_SECRET")
        return ""

    try:
        session = requests.Session()

        # Step 1: POST login credentials
        logger.info(f"TOTP auto-login: Posting credentials for {creds['user_id']}...")
        login_url = "https://kite.zerodha.com/api/login"
        resp = session.post(login_url, data={
            'user_id': creds['user_id'],
            'password': creds['password'],
        })

        if resp.status_code != 200:
            logger.error(f"Login POST failed: {resp.status_code} {resp.text[:200]}")
            return ""

        login_data = resp.json()
        if login_data.get('status') != 'success':
            logger.error(f"Login failed: {login_data}")
            return ""

        request_id = login_data['data']['request_id']
        logger.info(f"Login step 1 OK. request_id={request_id}")

        # Step 2: Submit TOTP
        totp_code = generate_totp(creds['totp_secret'])
        logger.info(f"Generated TOTP code: {totp_code[:2]}****")

        twofa_url = "https://kite.zerodha.com/api/twofa"
        resp2 = session.post(twofa_url, data={
            'user_id': creds['user_id'],
            'request_id': request_id,
            'twofa_value': totp_code,
            'twofa_type': 'totp',
        })

        if resp2.status_code != 200:
            logger.error(f"TOTP submission failed: {resp2.status_code} {resp2.text[:200]}")
            return ""

        twofa_data = resp2.json()
        if twofa_data.get('status') != 'success':
            logger.error(f"TOTP failed: {twofa_data}")
            return ""

        logger.info("TOTP step 2 OK. Session authenticated.")

        # Step 3: Hit the Kite Connect login URL to get request_token
        kite = KiteConnect(api_key=KITE_API_KEY)
        login_redirect = kite.login_url()

        # Follow the redirect — Kite redirects to our callback URL with request_token
        resp3 = session.get(login_redirect, allow_redirects=False)

        # The redirect Location header contains the request_token
        redirect_url = resp3.headers.get('Location', '')

        if not redirect_url:
            # Sometimes the response itself is a redirect page
            # Try following one more redirect
            if resp3.status_code in (301, 302, 303, 307, 308):
                redirect_url = resp3.headers.get('Location', '')
            else:
                # Check if we got HTML with the redirect
                text = resp3.text
                match = re.search(r'request_token=([a-zA-Z0-9]+)', text)
                if match:
                    request_token = match.group(1)
                else:
                    logger.error(f"No redirect URL found. Status: {resp3.status_code}")
                    logger.debug(f"Response headers: {dict(resp3.headers)}")
                    logger.debug(f"Response body: {text[:500]}")
                    return ""

        if redirect_url:
            # Parse request_token from URL
            parsed = urlparse(redirect_url)
            params = parse_qs(parsed.query)
            request_token = params.get('request_token', [None])[0]

            if not request_token:
                logger.error(f"request_token not found in redirect URL: {redirect_url}")
                return ""

        logger.info(f"Got request_token: {request_token[:8]}...")

        # Step 4: Exchange request_token for access_token
        data = kite.generate_session(request_token, api_secret=KITE_API_SECRET)
        access_token = data.get('access_token')

        if not access_token:
            logger.error("No access_token in session response")
            return ""

        # Save for the day
        save_access_token(access_token, request_token)
        logger.info(f"TOTP auto-login SUCCESS. Token saved for user {creds['user_id']}")

        return access_token

    except Exception as e:
        logger.error(f"TOTP auto-login failed: {e}", exc_info=True)
        return ""


def ensure_authenticated() -> bool:
    """
    Ensure we have a valid Kite session. Try existing token first,
    fall back to TOTP auto-login.

    Returns True if authenticated, False otherwise.
    """
    # Check existing token
    token = get_access_token()
    if token:
        try:
            kite = KiteConnect(api_key=KITE_API_KEY)
            kite.set_access_token(token)
            kite.profile()
            logger.info("Existing token is valid")
            return True
        except Exception:
            logger.info("Existing token expired, attempting TOTP auto-login...")

    # Auto-login
    new_token = auto_login()
    if new_token:
        return True

    logger.error("Authentication failed — no valid token and TOTP login failed")
    return False
