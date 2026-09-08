#!/usr/bin/env python3
"""Generate static/dad_holdings_ohlc.json — daily OHLC for Dad's holdings, so his Charts
tab gets the same candlesticks. Same resilient yfinance approach as gen_holdings_ohlc.py,
but the symbol list comes from Dad's (read-only) Kite holdings. Cron daily + on demand.
"""
import json
import os
import sys
import time
import datetime

sys.path.insert(0, "/home/arun/quantifyd")
os.chdir("/home/arun/quantifyd")

import yfinance as yf  # noqa: E402
from services.holdings_service import get_yahoo_symbol  # noqa: E402
from services.dad_kite import get_dad_kite, is_configured  # noqa: E402

OUT = "/home/arun/quantifyd/static/dad_holdings_ohlc.json"


def fetch_bars(symbol, attempts=3):
    best = []
    for i in range(attempts):
        try:
            hist = yf.Ticker(get_yahoo_symbol(symbol)).history(period="1y")
        except Exception as e:  # noqa: BLE001
            print("  err", symbol, e); hist = None
        if hist is not None and not hist.empty:
            rows = []
            for idx, row in hist.iterrows():
                o, h, l, c = row["Open"], row["High"], row["Low"], row["Close"]
                if any(v != v for v in (o, h, l, c)):
                    continue
                rows.append({"t": idx.strftime("%Y-%m-%d"),
                             "o": round(float(o), 2), "h": round(float(h), 2),
                             "l": round(float(l), 2), "c": round(float(c), 2),
                             "v": int(row["Volume"]) if row["Volume"] == row["Volume"] else 0})
            if len(rows) > len(best):
                best = rows
            if len(best) >= 20:
                break
        if i < attempts - 1:
            time.sleep(1.2)
    return best


def backfill_kite(kite, symbol, token, rows):
    """Append recent daily bars from Kite that yfinance is missing (freshness for
    lagging / recently-listed symbols like DYCL). Kite is the broker's own data."""
    if not (kite and token and rows):
        return rows
    frm = datetime.datetime.strptime(rows[-1]["t"], "%Y-%m-%d")
    to = datetime.datetime.now()
    if (to.date() - frm.date()).days < 1:
        return rows
    try:
        candles = kite.historical_data(token, frm, to, "day")
    except Exception as e:  # noqa: BLE001
        print("  kite backfill err", symbol, e)
        return rows
    have = {r["t"] for r in rows}
    added = 0
    for c in candles or []:
        d = c["date"].strftime("%Y-%m-%d") if hasattr(c["date"], "strftime") else str(c["date"])[:10]
        if d in have:
            continue
        rows.append({"t": d, "o": round(float(c["open"]), 2), "h": round(float(c["high"]), 2),
                     "l": round(float(c["low"]), 2), "c": round(float(c["close"]), 2),
                     "v": int(c.get("volume") or 0)})
        added += 1
    if added:
        rows.sort(key=lambda r: r["t"])
        print(f"  +{added} Kite bars {symbol}")
    return rows


def main():
    if not is_configured():
        print("DAD not configured — skipping"); sys.exit(0)
    try:
        raw = get_dad_kite().holdings() or []
    except Exception as e:  # noqa: BLE001
        print("FATAL dad holdings fetch failed:", e); sys.exit(1)
    syms = [h["tradingsymbol"] for h in raw
            if ((h.get("quantity") or 0) + (h.get("t1_quantity") or 0)
                + (h.get("collateral_quantity") or 0) + ((h.get("mtf") or {}).get("quantity") or 0)) > 0]
    print(f"{len(syms)} dad symbols")
    tokmap = {h["tradingsymbol"]: h.get("instrument_token") for h in raw}
    # historical-data API isn't enabled on Stanly's Kite app, so backfill via the MAIN
    # account (historical data is instrument-level, not account-specific)
    try:
        from services.kite_service import get_kite
        kite = get_kite()
    except Exception as e:  # noqa: BLE001
        print("  main kite unavailable, no backfill:", e); kite = None

    prev = {}
    if os.path.exists(OUT):
        try:
            prev = (json.load(open(OUT)) or {}).get("symbols", {})
        except Exception:  # noqa: BLE001
            prev = {}

    out = {}
    for s in syms:
        rows = fetch_bars(s)
        rows = backfill_kite(kite, s, tokmap.get(s), rows)
        merged = {r["t"]: r for r in (prev.get(s, []) or [])}
        merged.update({r["t"]: r for r in rows})
        keep = [merged[t] for t in sorted(merged)]
        if keep:
            out[s] = keep
            print("ok" if len(keep) >= 20 else "THIN", s, len(keep), "last", keep[-1]["t"])
        else:
            print("NO-DATA", s)
        time.sleep(0.3)

    payload = {"updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "symbols": out}
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    os.replace(tmp, OUT)
    print("WROTE", OUT, "symbols", len(out), "bytes", os.path.getsize(OUT))


if __name__ == "__main__":
    main()
