# -*- coding: utf-8 -*-
"""Extend nse_options_bhav BACK to 2011: download old NSE F&O bhavcopy (2011-2015) for NIFTY+BANKNIFTY
index options → same table. Lets us build MONTHLY straddles over ~15 years / several crashes (2011 EU,
2013 taper, 2015-16 China) instead of the 2019+ weekly window. Resumable."""
import sqlite3, csv, io, zipfile, urllib.request, datetime as dt, time, sys
DB = "/home/arun/quantifyd/backtest_data/market_data.db"
LOG = "/home/arun/quantifyd/research/101_trend_sleeve/results/nse_fo_download.log"
c = sqlite3.connect(DB); c.execute("PRAGMA busy_timeout=60000")
done = {r[0] for r in c.execute("SELECT DISTINCT trade_date FROM nse_options_bhav WHERE symbol='NIFTY' AND trade_date<'2016-01-01'")}
def log(m):
    print(m, flush=True)
    try: open(LOG, "a").write(m + "\n")
    except Exception: pass
def pdate(x):
    x = x.strip().title()
    for fmt in ("%d-%b-%Y", "%d-%b-%y"):        # some 2012 files use 2-digit year
        try: return dt.datetime.strptime(x, fmt).date().isoformat()
        except ValueError: continue
    raise ValueError(x)
def fnum(x):
    try: return float(x)
    except Exception: return None
start = dt.date(2011, 1, 1); end = dt.date(2015, 12, 31)
d = start; ok = holi = rowsN = 0
log(f"START {dt.datetime.now():%Y-%m-%d %H:%M} 2011-2015, have {len(done)} pre-2016 days")
while d <= end:
    if d.weekday() >= 5 or d.isoformat() in done:
        d += dt.timedelta(days=1); continue
    MON = d.strftime("%b").upper(); url = f"https://archives.nseindia.com/content/historical/DERIVATIVES/{d.year}/{MON}/fo{d.strftime('%d')}{MON}{d.year}bhav.csv.zip"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.nseindia.com"})
        raw = urllib.request.urlopen(req, timeout=45).read()
        z = zipfile.ZipFile(io.BytesIO(raw)); txt = z.read(z.namelist()[0]).decode("utf-8", "ignore")
    except Exception:
        holi += 1; d += dt.timedelta(days=1); time.sleep(0.2); continue
    rows = []
    for r in csv.reader(io.StringIO(txt)):
        if len(r) < 15 or r[0] != "OPTIDX" or r[1] not in ("NIFTY", "BANKNIFTY") or r[4] not in ("CE", "PE"): continue
        try:
            rows.append((pdate(r[14]), r[1], pdate(r[2]), fnum(r[3]), r[4], fnum(r[5]), fnum(r[6]), fnum(r[7]),
                         fnum(r[8]), fnum(r[9]), int(fnum(r[10]) or 0), fnum(r[11]), int(fnum(r[12]) or 0), int(fnum(r[13]) or 0)))
        except Exception:
            continue
    c.executemany("INSERT INTO nse_options_bhav(trade_date,symbol,expiry_date,strike,option_type,open,high,low,close,settle_price,contracts,value_in_lakhs,open_interest,change_in_oi) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    c.commit(); ok += 1; rowsN += len(rows)
    if ok % 25 == 0: log(f"{d.isoformat()}: {len(rows)} rows | days={ok} totrows={rowsN}")
    time.sleep(0.25); d += dt.timedelta(days=1)
log(f"DONE days={ok} holidays={holi} rows={rowsN} at {dt.datetime.now():%H:%M}")
n = c.execute("SELECT COUNT(*),MIN(trade_date),MAX(trade_date) FROM nse_options_bhav WHERE symbol='NIFTY'").fetchone()
log(f"NIFTY options now: {n}")
c.close()
