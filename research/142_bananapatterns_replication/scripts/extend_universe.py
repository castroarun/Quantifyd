"""
Phase-2 universe build (VPS-only, background job).

1. Repair ALL scale-broken daily symbols DB-wide (close ratio <= 0.62 step):
   backup rows to market_data_unified_bak142 -> delete -> refetch full adjusted
   history. Re-fetching from Kite is lossless for genuine demergers (same data
   comes back), and fixes true unadjusted splits.
2. Download full daily history (2005 -> today) for every NSE EQ instrument not
   yet in the DB (plain symbols + -BE/-B series). Kite lists only current
   instruments -> survivorship caveat recorded in STATUS.

Idempotent/resumable: symbols with >=100 rows and no scale break are skipped.
Log: stdout (redirect to /tmp/universe_ext.log). Progress line per symbol.
"""
import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DB = ROOT / 'backtest_data' / 'market_data.db'
sys.path.insert(0, str(ROOT))

RATIO_FLOOR = 0.62
FROM_DATE = datetime(2005, 1, 1)


def scale_broken(cur, symbol):
    rows = cur.execute(
        "select close from market_data_unified where symbol=? and timeframe='day' "
        "order by date", (symbol,)).fetchall()
    for (c1,), (c2,) in zip(rows, rows[1:]):
        if c1 and c2 and c2 / c1 <= RATIO_FLOOR:
            return True, len(rows)
    return False, len(rows)


def main():
    from kiteconnect import KiteConnect
    from config import KITE_API_KEY
    from services.data_manager import get_data_manager

    token = json.load(open(ROOT / 'backtest_data' / 'access_token.json'))['access_token']
    kite = KiteConnect(api_key=KITE_API_KEY)
    kite.set_access_token(token)
    kite.profile()
    print('token OK', flush=True)

    conn = sqlite3.connect(str(DB))
    cur = conn.cursor()

    instruments = kite.instruments('NSE')
    eq = []
    for inst in instruments:
        ts = inst['tradingsymbol']
        if inst['instrument_type'] != 'EQ' or inst.get('segment') != 'NSE':
            continue
        if '-' in ts and not (ts.endswith('-BE') or ts.endswith('-B')):
            continue
        eq.append(ts)
    eq = sorted(set(eq))
    print(f'NSE EQ instruments considered: {len(eq)}', flush=True)

    have = {r[0] for r in cur.execute(
        "select distinct symbol from market_data_unified where timeframe='day'")}

    repair, fresh = [], []
    for s in eq:
        if s in have:
            broken, n = scale_broken(cur, s)
            if broken or n < 100:
                repair.append(s)
        else:
            fresh.append(s)
    print(f'repair (broken/thin): {len(repair)} | fresh downloads: {len(fresh)}', flush=True)
    print('repair list:', repair, flush=True)

    cur.execute("create table if not exists market_data_unified_bak142 as "
                "select * from market_data_unified where 0")
    for s in repair:
        cur.execute("insert into market_data_unified_bak142 "
                    "select * from market_data_unified where symbol=? and timeframe='day'", (s,))
        cur.execute("delete from market_data_unified where symbol=? and timeframe='day'", (s,))
    conn.commit()
    conn.close()
    print('broken symbols backed up + deleted', flush=True)

    dm = get_data_manager(kite=kite)
    todo = repair + fresh
    t0 = time.time()

    def cb(i, total, sym, status):
        if status in ('completed',) or status.startswith('failed'):
            el = time.time() - t0
            print(f'[{i}/{total}] {sym} {status} ({el/60:.1f} min elapsed)', flush=True)

    okc, fail, errs = dm.download_data(todo, timeframe='day',
                                       from_date=FROM_DATE, to_date=datetime.now(),
                                       progress_callback=cb)
    print(f'DONE: {okc} ok, {fail} failed, {time.time()-t0:.0f}s', flush=True)
    for e in errs:
        print('ERR', e, flush=True)


if __name__ == '__main__':
    main()
