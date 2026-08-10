"""Generate static/momentum_ohlc.json — 1 year of daily OHLC for the momentum book's holdings.

Same shape as holdings_ohlc.json so the existing HoldingsCharts component can render it unchanged:
    {"updated": "...", "symbols": {"SYM": [{"t","o","h","l","c","v"}, ...]}}

Unlike the holdings generator this reads market_data.db directly instead of yfinance — we already own
the data, it needs no network, and it cannot go stale differently from what the book actually trades.
Includes the target basket too, so the charts still work when the gate is risk-off and nothing is held.
"""
import json, sqlite3, sys
from datetime import datetime, timedelta

sys.path.insert(0, "/home/arun/quantifyd")
OUT = "/home/arun/quantifyd/static/momentum_ohlc.json"
DB = "/home/arun/quantifyd/backtest_data/market_data.db"
DAYS = 400


def main():
    from services import momentum_paper as mp
    mp.init_db()
    st = mp.get_state()
    syms = [h["symbol"] for h in st.get("holdings", []) if not h.get("is_cash")]
    syms += [t["symbol"] for t in (st.get("target_basket") or [])]
    syms = sorted(set(syms))
    if not syms:
        print("no symbols to chart"); return
    since = (datetime.now() - timedelta(days=DAYS)).date().isoformat()
    con = sqlite3.connect(DB)
    out = {}
    for s in syms:
        rows = con.execute(
            "SELECT date,open,high,low,close,volume FROM market_data_unified "
            "WHERE symbol=? AND timeframe='day' AND date>=? AND close>0 ORDER BY date",
            (s, since)).fetchall()
        if not rows:
            print(f"  no data: {s}"); continue
        n = mp.CFG["donchian"]
        bars = []
        lows = [float(l or c) for d, o, h, l, c, v in rows]
        for i, (d, o, h, l, c, v) in enumerate(rows):
            # the live rule: exit when the close breaks the PRIOR n-day low
            stop = min(lows[max(0, i - n):i]) if i >= n else None
            bars.append({"t": d, "o": float(o or c), "h": float(h or c), "l": float(l or c),
                         "c": float(c), "v": int(v or 0),
                         "stop": round(stop, 2) if stop is not None else None})
        out[s] = bars
    con.close()
    # the book's own marks, so the chart can show WHY a position is held or about to be exited
    pos = {h["symbol"]: {"entry": h.get("entry_price"), "entry_date": h.get("entry_date"),
                         "stop": h.get("stop"), "qty": h.get("qty"),
                         "pnl_pct": h.get("pnl_pct"), "days": h.get("days")}
           for h in st.get("holdings", []) if not h.get("is_cash")}
    json.dump({"updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               "donchian": mp.CFG["donchian"], "positions": pos, "symbols": out},
              open(OUT, "w"))
    print(f"wrote {OUT}: {len(out)} symbols, "
          f"{sum(len(v) for v in out.values())} bars, "
          f"range {min(v[0]['t'] for v in out.values())} .. {max(v[-1]['t'] for v in out.values())}")


if __name__ == "__main__":
    main()
