#!/usr/bin/env python3
"""READ-ONLY: confirm Kite token works, resolve India VIX instrument token,
and verify Kite serves VIX 5min + day historical candles. No DB writes."""
import sys, json
sys.path.insert(0, "/home/arun/quantifyd")
from datetime import datetime, timedelta
from kiteconnect import KiteConnect
from config import KITE_API_KEY

tok = json.load(open("/home/arun/quantifyd/backtest_data/access_token.json"))["access_token"]
kite = KiteConnect(api_key=KITE_API_KEY); kite.set_access_token(tok)

# 1) token + live value (also proves the session is valid)
q = kite.ltp(["NSE:INDIA VIX"])
itok = q["NSE:INDIA VIX"]["instrument_token"]
print("INDIA VIX instrument_token:", itok, "| last:", q["NSE:INDIA VIX"]["last_price"])

# 2) sample historical (read-only) for both timeframes
to = datetime(2026, 3, 25)
for tf, frm in (("5minute", to - timedelta(days=5)), ("day", to - timedelta(days=30))):
    c = kite.historical_data(itok, frm, to, tf)
    print(f"\n{tf}: {len(c)} candles  {c[0]['date'] if c else None} .. {c[-1]['date'] if c else None}")
    for row in c[:2] + c[-1:]:
        print("   ", row["date"], "O",row["open"],"H",row["high"],"L",row["low"],"C",row["close"])
