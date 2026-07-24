import sqlite3, datetime as dt
from services.kite_service import get_kite
k = get_kite()
spot = k.ltp(["NSE:NIFTY 50"])["NSE:NIFTY 50"]["last_price"]


def ltp(s):
    try:
        return k.ltp(["NFO:" + s])["NFO:" + s]["last_price"]
    except Exception:
        return None


print("=== NAS LIVE 5-min report  %s  |  NIFTY %.1f ===" % (dt.datetime.now().strftime("%H:%M:%S"), spot))
grand = 0
for db, lab in [("nas_916_atm", "916-ATM  LIVE"), ("nas_916_atm2", "916-ATM2 LIVE"), ("nas_916_atm4", "916-ATM4")]:
    c = sqlite3.connect("backtest_data/%s_trading.db" % db); c.row_factory = sqlite3.Row
    rows = c.execute("SELECT tradingsymbol,qty,entry_price,exit_price,exit_reason,instrument_type,status "
                     "FROM nas_atm_positions WHERE date(entry_time)=date('now','localtime') ORDER BY id").fetchall()
    if not rows:
        print("  %-14s (no legs today)" % lab)
        continue
    real = 0.0; unre = 0.0; act = []; closed = []
    for r in rows:
        if r["status"] == "ACTIVE":
            cur = ltp(r["tradingsymbol"]); pnl = (r["entry_price"] - cur) * r["qty"] if cur else 0; unre += pnl
            act.append("%s %.1f→%.1f=%+.0f" % (r["instrument_type"], r["entry_price"], cur or 0, pnl))
        elif r["exit_price"] is not None:
            pnl = (r["entry_price"] - r["exit_price"]) * r["qty"]; real += pnl
            closed.append("%s %.1f→%.1f=%+.0f[%s]" % (r["instrument_type"], r["entry_price"], r["exit_price"], pnl, (r["exit_reason"] or "")[:8]))
    tot = real + unre
    print("  %-14s open[%s]%s | realized %+.0f | open %+.0f | TOTAL %+.0f" % (
        lab, "  ".join(act) if act else "flat",
        ("  closed[%s]" % "  ".join(closed)) if closed else "", real, unre, tot))
    grand += tot
print("  ====== LIVE 916 book (realized + open) = Rs %+.0f ======" % grand)
try:
    kt = sum(p["pnl"] for p in k.positions()["net"] if p["product"] == "MIS" and "NIFTY" in p["tradingsymbol"])
    print("  (Kite MIS NIFTY ground-truth: Rs %+.0f)" % kt)
except Exception:
    pass
