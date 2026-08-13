# -*- coding: utf-8 -*-
"""Download BSE F&O UDiFF bhavcopy (SENSEX + BANKEX index options) -> bse_options_bhav in market_data.db.
Covers 2024-01-01 -> today (UDiFF scheme). Resumable (skips trade_dates already loaded). Run on VPS."""
import sqlite3, csv, io, urllib.request, datetime as dt, time, sys
DB = "/home/arun/quantifyd/backtest_data/market_data.db"
LOG = "/home/arun/quantifyd/research/100_sensex_dte3_straddle/results/download.log"
SYMS = {"SENSEX", "BANKEX"}
c = sqlite3.connect(DB); c.execute("PRAGMA busy_timeout=60000")
c.execute("""CREATE TABLE IF NOT EXISTS bse_options_bhav(
  symbol TEXT, trade_date TEXT, expiry_date TEXT, strike REAL, option_type TEXT,
  open REAL, high REAL, low REAL, close REAL, underlying REAL, settle REAL,
  open_interest INTEGER, volume INTEGER, lot INTEGER)""")
c.execute("CREATE INDEX IF NOT EXISTS idx_bse_sym_date ON bse_options_bhav(symbol, trade_date)")
c.execute("CREATE INDEX IF NOT EXISTS idx_bse_expiry ON bse_options_bhav(symbol, expiry_date)")
c.commit()
done = {r[0] for r in c.execute("SELECT DISTINCT trade_date FROM bse_options_bhav")}

def log(m):
    print(m, flush=True)
    try:
        open(LOG, "a").write(m + "\n")
    except Exception: pass

def fnum(x):
    try: return float(x)
    except Exception: return None

start = dt.date(2024, 1, 1); end = dt.date.today()
d = start; ok = skip = holi = 0; total_rows = 0
log(f"START {dt.datetime.now():%Y-%m-%d %H:%M} range {start}..{end}, already have {len(done)} days")
while d <= end:
    ds = d.isoformat()
    if d.weekday() >= 5 or ds in done:
        d += dt.timedelta(days=1); continue
    url = f"https://www.bseindia.com/download/Bhavcopy/Derivative/BhavCopy_BSE_FO_0_0_0_{d:%Y%m%d}_F_0000.CSV"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=45).read().decode("utf-8", "ignore")
    except Exception as e:
        log(f"{ds}: ERR {e}"); d += dt.timedelta(days=1); time.sleep(0.5); continue
    if not raw.startswith("TradDt"):
        holi += 1; d += dt.timedelta(days=1); continue          # holiday -> HTML page
    rows = []
    for r in csv.reader(io.StringIO(raw)):
        if len(r) < 29 or r[0] == "TradDt": continue
        if r[7] not in SYMS or r[12] not in ("CE", "PE"): continue
        rows.append((r[7], r[0], r[9], fnum(r[11]), r[12], fnum(r[14]), fnum(r[15]),
                     fnum(r[16]), fnum(r[17]), fnum(r[20]), fnum(r[21]),
                     int(fnum(r[22]) or 0), int(fnum(r[24]) or 0), int(fnum(r[28]) or 0)))
    c.executemany("INSERT INTO bse_options_bhav VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    c.commit(); ok += 1; total_rows += len(rows)
    if ok % 20 == 0 or len(rows) == 0:
        log(f"{ds}: {len(rows)} rows | days={ok} totrows={total_rows}")
    time.sleep(0.3); d += dt.timedelta(days=1)
log(f"DONE days_loaded={ok} holidays={holi} total_rows={total_rows} at {dt.datetime.now():%H:%M}")
n = c.execute("SELECT symbol, COUNT(*), MIN(trade_date), MAX(trade_date) FROM bse_options_bhav GROUP BY symbol").fetchall()
log("FINAL: " + str(n))
c.close()
