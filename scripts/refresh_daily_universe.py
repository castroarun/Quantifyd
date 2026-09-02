"""Nightly incremental refresh of ALL active daily symbols in market_data.db (VPS-only).

For every symbol with >=260 rows whose last bar is older than today, downloads from
(last bar - 5d) to today via CentralizedDataManager (which inserts only missing dates).
Keeps the broad universe current for the BlueSky paper book and future studies —
fixes the 2026-09-02 finding that ~1,064 names had frozen at their last manual pull.

Cron: 17:45 IST Mon-Fri (before the 18:40 paper-book run). Log: /tmp/universe_refresh.log
"""
import json
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    # Guard: never store intraday snapshots as daily bars. The downloader skips
    # existing dates, so a partial candle written during market hours would be
    # frozen forever (this bit us on 2026-09-02: 14:15 prices became "closes").
    ist = datetime.utcnow() + timedelta(hours=5, minutes=30)
    if '--intraday-ok' not in sys.argv and ist.weekday() < 5 and not (ist.hour, ist.minute) >= (15, 35):
        print(f'{ist} — market hours; aborting (use --intraday-ok to override)')
        return
    from kiteconnect import KiteConnect
    from config import KITE_API_KEY
    from services.data_manager import get_data_manager

    tok = json.load(open(ROOT / 'backtest_data' / 'access_token.json'))['access_token']
    kite = KiteConnect(api_key=KITE_API_KEY)
    kite.set_access_token(tok)
    kite.profile()

    conn = sqlite3.connect(str(ROOT / 'backtest_data' / 'market_data.db'))
    today = datetime.now().strftime('%Y-%m-%d')
    rows = conn.execute(
        "select symbol, mx from (select symbol, max(date) mx, count(*) n from "
        "market_data_unified where timeframe='day' group by symbol) "
        "where n >= 260 and mx >= date(?, '-30 day')", (today,)).fetchall()
    conn.close()
    print(f'{datetime.now()} refresh: {len(rows)} symbols behind {today}', flush=True)
    if not rows:
        return

    dm = get_data_manager(kite=kite)
    t0 = time.time()
    ok = fail = 0
    wconn = sqlite3.connect(str(ROOT / 'backtest_data' / 'market_data.db'))
    for i, (s, mx) in enumerate(rows, 1):
        frm = datetime.strptime(mx[:10], '%Y-%m-%d') - timedelta(days=5)
        try:
            # delete the overlap window so refreshed (final) candles can replace
            # any partial ones — the store layer inserts only missing dates
            wconn.execute("delete from market_data_unified where symbol=? and "
                          "timeframe='day' and date >= ?",
                          (s, frm.strftime('%Y-%m-%d')))
            wconn.commit()
            n_ok, n_fail, errs = dm.download_data([s], timeframe='day',
                                                  from_date=frm, to_date=datetime.now())
            ok += n_ok
            fail += n_fail
        except Exception as e:
            fail += 1
            print(f'  ERR {s}: {e}', flush=True)
        if i % 200 == 0:
            print(f'  [{i}/{len(rows)}] ok {ok} fail {fail} ({(time.time()-t0)/60:.1f} min)', flush=True)
    print(f'DONE: {ok} ok, {fail} fail, {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
