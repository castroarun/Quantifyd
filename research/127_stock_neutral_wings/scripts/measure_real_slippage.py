"""Measure REAL slippage from live NAS/SENSEX order fills vs the 1-min options
recorder (bid/ask/LTP at the same minute). Answers: is 0.5%/side realistic?"""
import sqlite3, glob, os
import pandas as pd

BD = "/home/arun/quantifyd/backtest_data"
OPT = os.path.join(BD, "options_data.db")

oc = sqlite3.connect("file:%s?mode=ro" % OPT, uri=True)
cols = [d[1] for d in oc.execute("PRAGMA table_info(option_chain)")]
print("option_chain cols:", cols[:20])

orders = []
for f in glob.glob(os.path.join(BD, "*trading.db")):
    con = sqlite3.connect("file:%s?mode=ro" % f, uri=True)
    for t in ("nas_atm_orders", "nas_orders"):
        try:
            d = pd.read_sql_query(
                "SELECT tradingsymbol, transaction_type, qty, price, status, mode, created_at "
                "FROM %s" % t, con)
            d["book"] = os.path.basename(f).replace("_trading.db", "")
            orders.append(d)
        except Exception:
            pass
    con.close()
o = pd.concat(orders, ignore_index=True)
print("\nmodes:", o["mode"].value_counts().to_dict())
print("statuses:", o["status"].value_counts().head(6).to_dict())

live = o[(o["mode"].str.lower().isin(["live", "real"])) &
         (o["status"].str.upper().str.contains("COMPLETE", na=False)) &
         (o["price"] > 0)].copy()
live["ts"] = pd.to_datetime(live["created_at"])
print("\nlive filled orders:", len(live), "| span:", live["ts"].min(), "->", live["ts"].max())

# reference: recorder snapshot same tradingsymbol within +/-90s
has_tsym = "tradingsymbol" in cols
res = []
for _, r in live.iterrows():
    ts = r["ts"]
    q = ("SELECT ltp, bid, ask, snapshot_time FROM option_chain "
         "WHERE tradingsymbol=? AND snapshot_time BETWEEN ? AND ? "
         "ORDER BY ABS(julianday(snapshot_time)-julianday(?)) LIMIT 1")
    row = oc.execute(q, (r["tradingsymbol"],
                         (ts - pd.Timedelta(seconds=90)).isoformat(),
                         (ts + pd.Timedelta(seconds=90)).isoformat(),
                         ts.isoformat())).fetchone() if has_tsym else None
    if not row:
        continue
    ltp, bid, ask, st = row
    if not ltp or ltp <= 0:
        continue
    mid = (bid + ask) / 2 if bid and ask and bid > 0 and ask > 0 else ltp
    sign = 1 if r["transaction_type"].upper() == "BUY" else -1
    slip = sign * (r["price"] - mid) / mid          # + = paid worse than mid
    spread = (ask - bid) / mid if bid and ask and bid > 0 and ask > 0 else None
    res.append(dict(book=r["book"], side=r["transaction_type"], fill=r["price"],
                    mid=mid, slip=slip, spread=spread, ts=str(ts)))

d = pd.DataFrame(res)
print("\nmatched to recorder:", len(d))
if len(d):
    d = d[d["slip"].abs() < 0.25]                   # drop mismatched contracts/glitches
    print("after sanity filter:", len(d))
    print("\nREAL SLIPPAGE vs MID (positive = cost):")
    print("  mean  %+.3f%%   median %+.3f%%   p75 %+.3f%%   p95 %+.3f%%"
          % (100*d["slip"].mean(), 100*d["slip"].median(),
             100*d["slip"].quantile(0.75), 100*d["slip"].quantile(0.95)))
    print("  by side:", {s: round(100*g["slip"].mean(), 3) for s, g in d.groupby("side")})
    print("  quoted half-spread mean: %.3f%%" % (100*(d["spread"].dropna()/2).mean()))
    print("\n  by book:")
    for b, g in d.groupby("book"):
        print("   %-16s n=%4d  mean %+0.3f%%  median %+0.3f%%"
              % (b, len(g), 100*g["slip"].mean(), 100*g["slip"].median()))
