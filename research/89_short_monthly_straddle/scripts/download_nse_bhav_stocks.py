#!/usr/bin/env python3
"""
research/89 — Extend nse_options_bhav to F&O STOCKS (+ refresh indices), 2016->now.

Reuses the production downloader's URL/format/session logic but:
  - targets the 86 F&O stocks (read live from services/data_manager.py) + NIFTY/BANKNIFTY,
  - parses OPTSTK/STO (stock options) in addition to OPTIDX/IDO,
  - keeps only a near-ATM band (|strike/spot-1| <= ATM_BAND) and expiries <= MAX_DTE days
    so the table stays tractable (spot = UndrlygPric [UDiFF] / daily close / max-OI proxy),
  - resumes by STOCK-dates (so existing index-only dates get stock rows added too),
  - inserts incrementally per date into the SAME nse_options_bhav table (INSERT OR IGNORE).

Writes to market_data.db on the VPS (canonical host). Long-running -> launch via nohup.
Progress + heartbeat go to results/NSE_BHAV_STOCK_DOWNLOAD.log (tail it to monitor).
"""
import os, sys, io, csv, re, time, zipfile, sqlite3, argparse
from datetime import datetime, timedelta, date
from pathlib import Path
import requests

# ---------------- config ----------------
def find_db():
    here = Path(__file__).resolve()
    for c in [here.parents[3] / "backtest_data" / "market_data.db",
              Path("/home/arun/quantifyd/backtest_data/market_data.db")]:
        if c.exists():
            return str(c)
    raise FileNotFoundError("market_data.db not found")

DB_PATH = find_db()
DM_PATH = "/home/arun/quantifyd/services/data_manager.py"
SWITCH = date(2024, 7, 8)
ATM_BAND = 0.25          # keep strikes within +/-25% of spot
MAX_DTE = 75             # keep expiries within 75 days of trade date
RATE_LIMIT = 2.0
REQUEST_TIMEOUT = 30
INDEX_SYMS = {"NIFTY", "BANKNIFTY"}
UNDERLYING_MAP = {"NIFTY": "NIFTY50", "BANKNIFTY": "BANKNIFTY"}
MON = {1:'JAN',2:'FEB',3:'MAR',4:'APR',5:'MAY',6:'JUN',7:'JUL',8:'AUG',9:'SEP',10:'OCT',11:'NOV',12:'DEC'}
HEADERS = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
           'Accept':'*/*','Accept-Language':'en-US,en;q=0.9','Referer':'https://www.nseindia.com/','Connection':'keep-alive'}
LOG = Path(__file__).resolve().parent.parent / "results" / "NSE_BHAV_STOCK_DOWNLOAD.log"
LOG.parent.mkdir(exist_ok=True)


def log(m):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def load_fno_symbols():
    try:
        txt = open(DM_PATH, encoding="utf-8").read()
        block = txt.split("FNO_LOT_SIZES", 1)[1]
        block = block.split("}", 1)[0]
        keys = re.findall(r"'([A-Z0-9&\-]+)'\s*:", block)
        return set(keys)
    except Exception as e:
        log(f"WARN could not read FNO list ({e}); falling back to research/89 basket")
        return {"RELIANCE","HDFCBANK","ICICIBANK","INFY","TCS","SBIN","ITC","HINDUNILVR",
                "KOTAKBANK","BHARTIARTL","LT","AXISBANK","MARUTI","BAJFINANCE","HCLTECH",
                "ASIANPAINT","TITAN","SUNPHARMA","ULTRACEMCO","TATAMOTORS","TATASTEEL",
                "M&M","NTPC","POWERGRID","WIPRO","NESTLEIND"}


def preload_spot(conn, symbols):
    """dict[(underlying_symbol, 'YYYY-MM-DD')] -> daily close, for ATM filtering."""
    want = {UNDERLYING_MAP.get(s, s) for s in symbols}
    spot = {}
    cur = conn.cursor()
    qmarks = ",".join("?" * len(want))
    try:
        cur.execute(f"SELECT symbol, substr(date,1,10), close FROM market_data_unified "
                    f"WHERE timeframe='day' AND symbol IN ({qmarks}) AND date>='2015-12-01'",
                    tuple(want))
        for sym, d, c in cur.fetchall():
            if c and c > 0:
                spot[(sym, d)] = c
    except Exception as e:
        log(f"WARN preload_spot: {e}")
    log(f"preloaded {len(spot)} underlying daily closes for {len(want)} names")
    return spot


# ---------------- DB ----------------
def init_db(conn):
    conn.execute('''CREATE TABLE IF NOT EXISTS nse_options_bhav (
        id INTEGER PRIMARY KEY AUTOINCREMENT, trade_date TEXT, symbol TEXT, expiry_date TEXT,
        strike REAL, option_type TEXT, open REAL, high REAL, low REAL, close REAL,
        settle_price REAL, contracts INTEGER, value_in_lakhs REAL, open_interest INTEGER,
        change_in_oi INTEGER, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(trade_date, symbol, expiry_date, strike, option_type))''')
    conn.execute("CREATE INDEX IF NOT EXISTS idx_bhav_symbol_date ON nse_options_bhav(symbol, trade_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_bhav_expiry ON nse_options_bhav(symbol, expiry_date, strike, option_type)")
    conn.commit()


def existing_stock_dates(conn):
    try:
        cur = conn.execute("SELECT DISTINCT trade_date FROM nse_options_bhav "
                           "WHERE symbol NOT IN ('NIFTY','BANKNIFTY')")
        return {r[0] for r in cur.fetchall()}
    except sqlite3.OperationalError:
        return set()


def insert_rows(conn, rows):
    if not rows:
        return 0
    conn.executemany('''INSERT OR IGNORE INTO nse_options_bhav
        (trade_date, symbol, expiry_date, strike, option_type, open, high, low, close,
         settle_price, contracts, value_in_lakhs, open_interest, change_in_oi)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', rows)
    conn.commit()
    return conn.total_changes


# ---------------- fetch / parse ----------------
def url_for(d):
    if d < SWITCH:
        return f"https://nsearchives.nseindia.com/content/historical/DERIVATIVES/{d.year}/{MON[d.month]}/fo{d.day:02d}{MON[d.month]}{d.year}bhav.csv.zip"
    return f"https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{d.strftime('%Y%m%d')}_F_0000.csv.zip"

def pdate(s):
    s = (s or "").strip()
    for fmt in ('%d-%b-%Y','%d-%B-%Y','%Y-%m-%d','%d/%m/%Y','%Y%m%d','%Y-%m-%dT%H:%M:%S'):
        try: return datetime.strptime(s, fmt).strftime('%Y-%m-%d')
        except ValueError: continue
    return s[:10]

def ff(v, d=0.0):
    try: return float(str(v).strip()) if v not in (None,"") else d
    except: return d
def fi(v, d=0):
    try: return int(float(str(v).strip())) if v not in (None,"") else d
    except: return d


def parse_day(csv_text, tds, targets, spot_lookup):
    """Return list of db rows filtered to target symbols, options, ATM band, DTE<=MAX_DTE."""
    head = csv_text.split('\n', 1)[0]
    old = ('INSTRUMENT' in head.upper() and 'SYMBOL' in head.upper())
    reader = csv.DictReader(io.StringIO(csv_text))
    if reader.fieldnames:
        reader.fieldnames = [f.strip() for f in reader.fieldnames]
    # collect candidates grouped by symbol
    cand = {}   # symbol -> list of dicts
    undrlyg = {}  # symbol -> list of underlying prices seen
    for row in reader:
        row = {k.strip(): v for k, v in row.items()}
        if old:
            inst = row.get('INSTRUMENT','').strip()
            if inst not in ('OPTSTK','OPTIDX'): continue
            sym = row.get('SYMBOL','').strip()
            if sym not in targets: continue
            ot = row.get('OPTION_TYP','').strip()
            if ot not in ('CE','PE'): continue
            exp = pdate(row.get('EXPIRY_DT',''))
            rec = dict(sym=sym, exp=exp, strike=ff(row.get('STRIKE_PR')), ot=ot,
                       o=ff(row.get('OPEN')), h=ff(row.get('HIGH')), l=ff(row.get('LOW')),
                       c=ff(row.get('CLOSE')), st=ff(row.get('SETTLE_PR')),
                       con=fi(row.get('CONTRACTS')), val=ff(row.get('VAL_INLAKH')),
                       oi=fi(row.get('OPEN_INT')), coi=fi(row.get('CHG_IN_OI')), und=0.0)
        else:
            ft = row.get('FinInstrmTp','').strip()
            if ft not in ('STO','IDO'): continue
            sym = row.get('TckrSymb','').strip()
            if sym not in targets: continue
            ot = row.get('OptnTp','').strip()
            if ot not in ('CE','PE'): continue
            exp = pdate(row.get('XpryDt','') or row.get('FininstrmActlXpryDt',''))
            und = ff(row.get('UndrlygPric'))
            rec = dict(sym=sym, exp=exp, strike=ff(row.get('StrkPric')), ot=ot,
                       o=ff(row.get('OpnPric')), h=ff(row.get('HghPric')), l=ff(row.get('LwPric')),
                       c=ff(row.get('ClsPric')) or ff(row.get('LastPric')), st=ff(row.get('SttlmPric')),
                       con=fi(row.get('TtlTradgVol')), val=ff(row.get('TtlTrfVal'))/1e5,
                       oi=fi(row.get('OpnIntrst')), coi=fi(row.get('ChngInOpnIntrst')), und=und)
            if und > 0:
                undrlyg.setdefault(sym, []).append(und)
        # DTE filter
        try:
            dte = (datetime.strptime(rec['exp'],'%Y-%m-%d').date()
                   - datetime.strptime(tds,'%Y-%m-%d').date()).days
        except Exception:
            continue
        if dte < 0 or dte > MAX_DTE or rec['strike'] <= 0:
            continue
        cand.setdefault(sym, []).append(rec)

    out = []
    for sym, recs in cand.items():
        # determine spot for this symbol/day
        spot = None
        if undrlyg.get(sym):
            u = sorted(undrlyg[sym]); spot = u[len(u)//2]  # median
        if not spot:
            spot = spot_lookup.get((UNDERLYING_MAP.get(sym, sym), tds))
        if not spot:
            # fallback: max-OI strike as ATM proxy
            spot = max(recs, key=lambda r: r['oi'])['strike'] if recs else None
        for r in recs:
            if spot and abs(r['strike']/spot - 1.0) > ATM_BAND:
                continue
            out.append((tds, r['sym'], r['exp'], r['strike'], r['ot'], r['o'], r['h'],
                        r['l'], r['c'], r['st'], r['con'], r['val'], r['oi'], r['coi']))
    return out


def make_session():
    s = requests.Session(); s.headers.update(HEADERS)
    try: s.get('https://www.nseindia.com/', timeout=10)
    except Exception as e: log(f"session warmup: {e}")
    return s


def fetch(session, d):
    try:
        r = session.get(url_for(d), timeout=REQUEST_TIMEOUT)
        if r.status_code == 404: return None, 'holiday'
        if r.status_code == 403: return None, 'blocked'
        if r.status_code != 200: return None, f'http_{r.status_code}'
        ct = r.headers.get('Content-Type','')
        if 'text/html' in ct and len(r.content) < 5000: return None, 'blocked'
        zf = zipfile.ZipFile(io.BytesIO(r.content))
        cs = [f for f in zf.namelist() if f.lower().endswith('.csv')]
        if not cs: return None, 'no_csv'
        return zf.read(cs[0]).decode('utf-8', errors='replace'), 'ok'
    except zipfile.BadZipFile:
        return None, 'bad_zip'
    except requests.exceptions.Timeout:
        return None, 'timeout'
    except Exception as e:
        return None, f'err:{str(e)[:40]}'


def trading_days(a, b):
    cur = a
    while cur <= b:
        if cur.weekday() < 5:
            yield cur
        cur += timedelta(days=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2016-01-01')
    ap.add_argument('--end', default=(date.today()-timedelta(days=1)).strftime('%Y-%m-%d'))
    ap.add_argument('--test', action='store_true')
    args = ap.parse_args()

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    fno = load_fno_symbols()
    targets = fno | INDEX_SYMS
    log(f"DB={DB_PATH}  targets={len(targets)} symbols (FNO {len(fno)} + indices)")
    spot_lookup = preload_spot(conn, targets)

    start = datetime.strptime(args.start,'%Y-%m-%d').date()
    end = datetime.strptime(args.end,'%Y-%m-%d').date()
    if args.test:
        start = end - timedelta(days=12)
    done = existing_stock_dates(conn)
    days = [d for d in trading_days(start, end) if d.strftime('%Y-%m-%d') not in done]
    log(f"range {start}..{end}  todo={len(days)} days  (skipping {len(done)} with stock rows)")

    session = make_session(); time.sleep(1)
    ok = hol = blk = err = 0; consec = 0; total = 0
    for i, d in enumerate(days, 1):
        tds = d.strftime('%Y-%m-%d')
        txt, st = fetch(session, d)
        if st == 'ok' and txt:
            rows = parse_day(txt, tds, targets, spot_lookup)
            n = insert_rows(conn, rows)
            total += len(rows); ok += 1; consec = 0
            nsym = len({r[1] for r in rows})
            if i % 10 == 0 or i <= 5:
                log(f"[{i}/{len(days)}] {tds}  {len(rows):5d} rows / {nsym} syms  (cum {total:,})")
        elif st == 'holiday':
            hol += 1; consec = 0
        elif st == 'blocked':
            blk += 1; consec += 1; log(f"[{i}/{len(days)}] {tds} BLOCKED")
        else:
            err += 1; consec += 1
            if i % 20 == 0: log(f"[{i}/{len(days)}] {tds} {st}")
        if blk and blk % 3 == 0:
            log("refreshing session..."); session = make_session(); time.sleep(3)
        if consec >= 12:
            log(f"ABORT: {consec} consecutive errors — NSE may be blocking. Resume later (script is idempotent).")
            break
        time.sleep(RATE_LIMIT)
    log(f"DONE ok={ok} holiday={hol} blocked={blk} err={err} rows_added~{total:,}")
    # summary
    cur = conn.execute("SELECT COUNT(*), COUNT(DISTINCT symbol), MIN(trade_date), MAX(trade_date) FROM nse_options_bhav")
    log("TABLE now: rows=%d symbols=%d range=%s..%s" % cur.fetchone())
    conn.close()


if __name__ == '__main__':
    main()
