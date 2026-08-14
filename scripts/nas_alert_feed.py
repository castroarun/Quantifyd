"""NAS suite -> desktop-alert bridge. Scans the 6 NAS order books for NEW orders and
publishes them as events in static/app/nas_alerts.json, which the Windows watcher
(scripts/csl_alert_watcher.pyw) polls alongside the CSL feed - so EVERY book's
entries/exits pop on the desktop, live suite included. Cron: every minute 9-15h Mon-Fri.
State (last seen order id per book): backtest_data/nas_alert_feed_state.json"""
import json, os, sqlite3
from datetime import datetime
from pathlib import Path

Q = Path("/home/arun/quantifyd")
OUT = Q / "static/app/nas_alerts.json"
STATE = Q / "backtest_data/nas_alert_feed_state.json"
BOOKS = [("nas_916_atm", "NAS ATM"), ("nas_916_atm2", "NAS ATM2"), ("nas_916_atm4", "NAS ATM4"),
         ("sensex_atm", "SENSEX ATM"), ("sensex_atm2", "SENSEX ATM2"), ("sensex_atm4", "SENSEX ATM4")]

def load(p, d):
    try: return json.load(open(p))
    except Exception: return d

st = load(STATE, {})
feed = load(OUT, {"events": []})
new_ev = []
for db, lbl in BOOKS:
    p = Q / "backtest_data" / (db + "_trading.db")
    if not p.exists(): continue
    try:
        c = sqlite3.connect(str(p)); c.row_factory = sqlite3.Row
        cols = [x[1] for x in c.execute("PRAGMA table_info(nas_atm_orders)")]
        tcol = next((x for x in ("tradingsymbol", "symbol", "instrument") if x in cols), None)
        scol = next((x for x in ("transaction_type", "side", "action") if x in cols), None)
        qcol = next((x for x in ("quantity", "qty") if x in cols), None)
        pcol = next((x for x in ("average_price", "price", "fill_price") if x in cols), None)
        last = st.get(db, 0)
        sel = ["id", "created_at", "mode", "status"] + [c2 for c2 in (tcol, scol, qcol, pcol) if c2]
        for r in c.execute("SELECT %s FROM nas_atm_orders WHERE id > ? ORDER BY id" % ",".join(sel), (last,)):
            st[db] = max(st.get(db, 0), r["id"])
            mode = (r["mode"] or "paper").lower()
            bits = []
            if scol: bits.append(str(r[scol]))
            if tcol: bits.append(str(r[tcol]))
            if qcol: bits.append("x%s" % r[qcol])
            if pcol and r[pcol]: bits.append("@ %s" % r[pcol])
            bits.append("(%s)" % (r["status"] or "?"))
            new_ev.append({"ts": str(r["created_at"])[:19].replace("T", " "), "book": lbl,
                           "type": "ORDER", "source": "REAL" if mode == "live" else "PAPER",
                           "msg": " ".join(bits) + (" [REAL MONEY]" if mode == "live" else "")})
        c.close()
    except Exception as e:
        print(db, e)

if new_ev or not OUT.exists():
    feed["events"] = (feed.get("events", []) + new_ev)[-200:]
    feed["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    json.dump(feed, open(OUT, "w"))
    json.dump(st, open(STATE, "w"))
    print("published %d new events" % len(new_ev))
else:
    json.dump(st, open(STATE, "w"))
    print("no new orders")
