#!/usr/bin/env python3
"""Overlay TODAY's forming candle onto static/holdings_ohlc.json from Kite quotes.
Runs intraday (every few minutes, 09:15-15:30 IST). Fixes the 'no price action today' gap:
gen_holdings_ohlc.py uses yfinance which lags today's daily bar for NSE stocks during market
hours (and its never-downgrade logic then silently keeps yesterday). Kite's quote() returns
today's live open/high/low + last_price, so we stamp an accurate forming candle. Fast + idempotent."""
import json
import os
import sys
import datetime

sys.path.insert(0, "/home/arun/quantifyd")
os.chdir("/home/arun/quantifyd")
OUT = "/home/arun/quantifyd/static/holdings_ohlc.json"
from services.kite_service import get_kite  # noqa: E402


def main():
    if not os.path.exists(OUT):
        print("no holdings_ohlc.json yet — run gen_holdings_ohlc.py first")
        return
    d = json.load(open(OUT))
    syms = d.get("symbols", {})
    if not syms:
        print("empty symbols")
        return
    kite = get_kite()
    keys, ts_by_key = [], {}
    for h in (kite.holdings() or []):
        ts = h.get("tradingsymbol")
        if ts and ts in syms:
            k = "%s:%s" % (h.get("exchange") or "NSE", ts)
            keys.append(k)
            ts_by_key[k] = ts
    today = datetime.date.today().strftime("%Y-%m-%d")
    q = kite.quote(keys) if keys else {}
    n = 0
    for k, data in q.items():
        ts = ts_by_key.get(k)
        ser = syms.get(ts)
        if not ser:
            continue
        oh = data.get("ohlc") or {}
        o, c = oh.get("open"), data.get("last_price")
        if not o or not c:
            continue
        cand = {
            "t": today, "o": round(float(o), 2),
            "h": round(float(oh.get("high") or c), 2),
            "l": round(float(oh.get("low") or c), 2),
            "c": round(float(c), 2), "v": int(data.get("volume") or 0),
        }
        if ser and ser[-1].get("t") == today:
            ser[-1] = cand          # refresh today's forming candle
        else:
            ser.append(cand)        # first stamp of the day
        n += 1
    d["updated"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, separators=(",", ":"))
    os.replace(tmp, OUT)
    print("overlay: today candle for %d/%d symbols @ %s" % (n, len(syms), d["updated"]))


if __name__ == "__main__":
    main()
