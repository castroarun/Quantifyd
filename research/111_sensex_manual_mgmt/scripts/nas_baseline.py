"""NAS live-book day P&L baseline -> static/app/nas_baseline.json
Per-day net P&L for the 3 NIFTY nas_916 books (LIVE REAL) + SENSEX ATM2 (LIVE PAPER),
with lots carried per-row where the DB has them (else month map). Consumed by the
/app/straddles curve explorer as the comparison strip. Runs in the 15:40 regen."""
import json, sqlite3
from datetime import datetime
from pathlib import Path

BT = Path("/home/arun/quantifyd/backtest_data")
LOTS_BY_MONTH = {"2026-03": 1, "2026-04": 5, "2026-05": 1, "2026-06": 1, "2026-07": 2, "2026-08": 3}
days = {}

def put(day, book, pnl, lots, source):
    days.setdefault(day, [])
    for r in days[day]:
        if r["book"] == book:
            r["pnl"] = round(r["pnl"] + pnl); return
    days[day].append({"book": book, "pnl": round(pnl), "lots": lots, "source": source})

for name, lbl in (("nas_916_atm", "NAS ATM"), ("nas_916_atm2", "NAS ATM2"), ("nas_916_atm4", "NAS ATM4")):
    try:
        c = sqlite3.connect(str(BT / (name + "_trading.db"))); c.row_factory = sqlite3.Row
        cols = [x[1] for x in c.execute("PRAGMA table_info(nas_atm_trades)")]
        pcol = "pnl" if "pnl" in cols else "net_pnl"
        dcol = "entry_time" if "entry_time" in cols else "created_at"
        lcol = "lots" if "lots" in cols else None
        # per-day mode from the orders table: a day is REAL only if a live-mode order fired
        live_days = {str(r[0])[:10] for r in c.execute(
            "SELECT DISTINCT substr(created_at,1,10) FROM nas_atm_orders WHERE mode='live'")}
        for r in c.execute("SELECT %s d,%s p%s FROM nas_atm_trades WHERE %s IS NOT NULL" % (
                dcol, pcol, (",%s l" % lcol) if lcol else "", pcol)):
            dd = str(r["d"])[:10]
            lots = (r["l"] if lcol and r["l"] else None) or LOTS_BY_MONTH.get(dd[:7])
            put(dd, lbl, r["p"], lots, "REAL" if dd in live_days else "PAPER")
        c.close()
    except Exception as e:
        print(name, e)
try:
    c = sqlite3.connect(str(BT / "sensex_atm2_trading.db")); c.row_factory = sqlite3.Row
    cols = [x[1] for x in c.execute("PRAGMA table_info(nas_atm_trades)")]
    lcol = "lots" if "lots" in cols else None
    for r in c.execute("SELECT trade_date d, net_pnl p%s FROM nas_atm_trades WHERE net_pnl IS NOT NULL" % (
            (",%s l" % lcol) if lcol else "")):
        dd = str(r["d"])[:10]
        put(dd, "SENSEX ATM2", r["p"], (r["l"] if lcol and r["l"] else None), "PAPER")
    c.close()
except Exception as e:
    print("sensex", e)

out = {"generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"), "days": days}
for p in ("/home/arun/quantifyd/static/app/nas_baseline.json",
          "/home/arun/quantifyd/frontend/public/nas_baseline.json"):
    json.dump(out, open(p, "w"))
print("nas_baseline: %d days" % len(days))
