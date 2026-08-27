#!/usr/bin/env python3
"""Dad's Kite TOTP auto-login — self-contained (does NOT touch the primary kite_auth).

Replicates the primary TOTP flow using DAD_* env vars and Dad's api_key/secret, saving to
backtest_data/dad_access_token.json. Run daily via cron (morning) + on demand.
Env: DAD_KITE_API_KEY, DAD_KITE_API_SECRET, DAD_KITE_USER_ID, DAD_KITE_PASSWORD, DAD_KITE_TOTP_SECRET
"""
import os
import re
import sys
import logging
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, "/home/arun/quantifyd")
os.chdir("/home/arun/quantifyd")

import requests
import pyotp
from kiteconnect import KiteConnect
from services.dad_kite import DAD_KITE_API_KEY, DAD_KITE_API_SECRET, save_dad_access_token

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("dad_auto_login")


def auto_login() -> str:
    uid = os.getenv("DAD_KITE_USER_ID", "")
    pwd = os.getenv("DAD_KITE_PASSWORD", "")
    totp_secret = os.getenv("DAD_KITE_TOTP_SECRET", "")
    if not all([uid, pwd, totp_secret]):
        log.error("[DAD] Missing DAD_KITE_USER_ID / PASSWORD / TOTP_SECRET")
        return ""
    if not DAD_KITE_API_KEY or not DAD_KITE_API_SECRET:
        log.error("[DAD] Missing DAD_KITE_API_KEY / SECRET")
        return ""
    try:
        s = requests.Session()
        r = s.post("https://kite.zerodha.com/api/login", data={"user_id": uid, "password": pwd})
        if r.status_code != 200 or r.json().get("status") != "success":
            log.error(f"[DAD] login step1 failed: {r.status_code} {r.text[:200]}")
            return ""
        request_id = r.json()["data"]["request_id"]

        code = pyotp.TOTP(totp_secret).now()
        r2 = s.post("https://kite.zerodha.com/api/twofa", data={
            "user_id": uid, "request_id": request_id, "twofa_value": code, "twofa_type": "totp"})
        if r2.status_code != 200 or r2.json().get("status") != "success":
            log.error(f"[DAD] twofa failed: {r2.status_code} {r2.text[:200]}")
            return ""

        kite = KiteConnect(api_key=DAD_KITE_API_KEY)
        url = kite.login_url()
        request_token = None
        for _ in range(5):
            r3 = s.get(url, allow_redirects=False)
            loc = r3.headers.get("Location", "")
            for cu in (loc, url):
                if "request_token=" in cu:
                    request_token = parse_qs(urlparse(cu).query).get("request_token", [None])[0]
                    if request_token:
                        break
            if request_token:
                break
            if not loc:
                m = re.search(r"request_token=([a-zA-Z0-9]+)", r3.text)
                if m:
                    request_token = m.group(1)
                break
            url = loc
        if not request_token:
            log.error("[DAD] request_token not found after redirects")
            return ""

        data = kite.generate_session(request_token, api_secret=DAD_KITE_API_SECRET)
        access_token = data.get("access_token")
        if not access_token:
            log.error("[DAD] no access_token in session response")
            return ""
        save_dad_access_token(access_token, request_token)
        log.info(f"[DAD] auto-login SUCCESS for {uid}")
        return access_token
    except Exception as e:  # noqa: BLE001
        log.error(f"[DAD] auto-login failed: {e}", exc_info=True)
        return ""


if __name__ == "__main__":
    tok = auto_login()
    sys.exit(0 if tok else 1)
