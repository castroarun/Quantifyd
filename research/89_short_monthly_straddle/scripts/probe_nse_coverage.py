#!/usr/bin/env python3
"""Probe: what does NSE actually serve from the VPS across years, and do STOCK
options (OPTSTK/STO) parse? Fetches one mid-month trading day per year 2016-2026,
reports HTTP status, row counts, and whether NIFTY + a sample stock (RELIANCE) appear.
No DB writes."""
import io, sys, zipfile, time
from datetime import date
import requests

SWITCH = date(2024, 7, 8)
MON = {1:'JAN',2:'FEB',3:'MAR',4:'APR',5:'MAY',6:'JUN',7:'JUL',8:'AUG',9:'SEP',10:'OCT',11:'NOV',12:'DEC'}
H = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
     'Accept':'*/*','Accept-Language':'en-US,en;q=0.9','Referer':'https://www.nseindia.com/','Connection':'keep-alive'}

def url(d):
    if d < SWITCH:
        return f"https://nsearchives.nseindia.com/content/historical/DERIVATIVES/{d.year}/{MON[d.month]}/fo{d.day:02d}{MON[d.month]}{d.year}bhav.csv.zip"
    return f"https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{d.strftime('%Y%m%d')}_F_0000.csv.zip"

s = requests.Session(); s.headers.update(H)
try: s.get('https://www.nseindia.com/', timeout=10)
except Exception as e: print("warmup:", e)
time.sleep(1)

probe = [date(2016,6,15), date(2018,6,15), date(2019,6,14), date(2020,6,15), date(2021,6,15),
         date(2022,6,15), date(2023,6,15), date(2024,3,15), date(2024,9,16), date(2025,6,16), date(2026,6,15)]
for d in probe:
    u = url(d)
    try:
        r = s.get(u, timeout=30)
        st = r.status_code
        if st != 200:
            print(f"{d}  HTTP {st}  (old={d<SWITCH})"); time.sleep(2); continue
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        name = [f for f in zf.namelist() if f.lower().endswith('.csv')][0]
        txt = zf.read(name).decode('utf-8', errors='replace')
        head = txt.split('\n')[0][:80]
        lines = txt.count('\n')
        # crude counts
        nifty = txt.count(',NIFTY,') + txt.count('NIFTY,')
        has_optstk = ('OPTSTK' in txt) or (',STO,' in txt)
        has_rel = 'RELIANCE' in txt
        fmt = 'OLD' if ('INSTRUMENT' in head.upper()) else ('UDiFF' if ('TckrSymb' in head or 'FinInstrmTp' in head) else '?')
        print(f"{d}  HTTP200  {fmt:5s} lines={lines:6d}  stockOpt={has_optstk} RELIANCE={has_rel}  hdr='{head}'")
    except Exception as e:
        print(f"{d}  EXC {str(e)[:60]}")
    time.sleep(2)
print("DONE")
