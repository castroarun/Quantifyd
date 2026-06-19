"""Download BANKNIFTY + NIFTY 50 DAILY history via Kite (VPS-only), back to 2015, into market_data_unified.
Extends the NIFTY-vs-BNF beta study into the 2018/2020/2022 high-vol regimes (current DB BNF starts 2023-03)."""
import sqlite3
from datetime import datetime, date, timedelta
from services.kite_service import get_kite
kite = get_kite()
instr = kite.instruments("NSE")
def find_tok(names):
    for i in instr:
        if i.get("tradingsymbol") in names and i.get("segment") == "INDICES":
            return i["instrument_token"], i["tradingsymbol"]
    return None, None
targets = [("BANKNIFTY", ["NIFTY BANK", "BANKNIFTY"]), ("NIFTY50", ["NIFTY 50"])]
DB = "backtest_data/market_data.db"
con = sqlite3.connect(DB)
for store_sym, names in targets:
    tok, kname = find_tok(names)
    if not tok:
        print(f"{store_sym}: token NOT found (names {names})"); continue
    rows_all = []
    # daily: chunk in ~5y windows (Kite daily ~2000-candle cap)
    for frm, to in [(date(2015,1,1), date(2019,12,31)), (date(2020,1,1), date(2024,12,31)), (date(2025,1,1), date.today())]:
        try:
            c = kite.historical_data(tok, frm, to, "day")
            rows_all += c or []
        except Exception as e:
            print(f"{store_sym} {frm}->{to} fetch failed: {e}")
    # dedupe + insert
    seen = {}
    for r in rows_all:
        d = r["date"].strftime("%Y-%m-%d")
        seen[d] = r
    n = 0
    for d, r in seen.items():
        con.execute("INSERT OR REPLACE INTO market_data_unified (symbol,timeframe,date,open,high,low,close,volume,created_at) "
                    "VALUES (?,?,?,?,?,?,?,?,datetime('now'))",
                    (store_sym, "day", d, r["open"], r["high"], r["low"], r["close"], r.get("volume", 0)))
        n += 1
    con.commit()
    rng = con.execute("SELECT MIN(date),MAX(date),COUNT(*) FROM market_data_unified WHERE symbol=? AND timeframe='day'", (store_sym,)).fetchone()
    print(f"{store_sym} (kite '{kname}', tok {tok}): inserted/updated {n} daily rows -> DB now {rng[2]} rows {rng[0]}..{rng[1]}")
con.close()
print("DONE")
