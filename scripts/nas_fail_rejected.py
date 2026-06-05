"""One-shot + reusable: mark NAS DB legs FAILED when their Kite order was REJECTED.
Identifies phantom ACTIVE/PENDING legs (order placed then rejected, e.g. insufficient funds) and
flips them to FAILED so the app stops showing/managing them. Touches ONLY rejected-order legs."""
import os, json, sqlite3
from kiteconnect import KiteConnect

ROOT = "/home/arun/quantifyd"
tok = json.load(open(ROOT + "/backtest_data/access_token.json"))
k = KiteConnect(api_key=os.getenv("KITE_API_KEY", "")); k.set_access_token(tok["access_token"])
rejected = {str(o["order_id"]): o.get("status_message", "") for o in k.orders() if o["status"] == "REJECTED"}
print("REJECTED order ids today:", len(rejected))

DBS = [("nas_trading.db", "nas_positions"), ("nas_atm_trading.db", "nas_atm_positions"),
       ("nas_atm2_trading.db", "nas_atm_positions"), ("nas_atm4_trading.db", "nas_atm_positions"),
       ("nas_916_otm_trading.db", "nas_positions"), ("nas_916_atm_trading.db", "nas_atm_positions"),
       ("nas_916_atm2_trading.db", "nas_atm_positions"), ("nas_916_atm4_trading.db", "nas_atm_positions")]
total = 0
for db, t in DBS:
    p = os.path.join(ROOT, "backtest_data", db)
    if not os.path.exists(p):
        continue
    c = sqlite3.connect(p); cols = [r[1] for r in c.execute("PRAGMA table_info(" + t + ")")]
    has_st = "st_monitoring" in cols
    for r in c.execute("SELECT id,tradingsymbol,kite_order_id,status FROM " + t +
                       " WHERE status IN ('ACTIVE','PENDING') AND kite_order_id IS NOT NULL"):
        oid = str(r[2])
        if oid in rejected:
            msg = rejected[oid][:60]
            c.execute("UPDATE " + t + " SET status='FAILED', exit_reason=?, notes=? WHERE id=?",
                      ("order-rejected", "REJECTED: " + msg, r[0]))
            if has_st:
                c.execute("UPDATE " + t + " SET st_monitoring=0 WHERE id=?", (r[0],))
            print("  %s id=%s %s oid=%s -> FAILED (%s)" % (db.replace("_trading.db", ""), r[0], r[1], oid, msg))
            total += 1
    c.commit(); c.close()
print("TOTAL legs marked FAILED:", total)
