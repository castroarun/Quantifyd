"""
Phase-2 data repair for research/142 (VPS-only; refuses laptop via data_manager guard).

1. Classify every symbol in both ground-truth CSVs: OK / MISSING / SCALE_BROKEN.
   Scale check: any day-over-day close ratio <= 0.62 (unadjusted split/bonus step;
   NSE circuits make genuine -38% overnight moves near-impossible in these names).
2. Backup broken symbols' daily rows to market_data_unified_bak142, DELETE them,
   re-download FULL adjusted history (2005-01-01 -> today) from Kite.
3. Fresh-download missing symbols.
4. Post-check: coverage + scale re-scan.

Also (read-only): full-DB scale-break scan to size the defect beyond this study.
Run: venv/bin/python research/142_bananapatterns_replication/scripts/repair_data.py [--apply]
Without --apply it only reports (dry run).
"""
import csv
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STUDY = Path(__file__).resolve().parents[1]
DB = ROOT / 'backtest_data' / 'market_data.db'
APPLY = '--apply' in sys.argv

sys.path.insert(0, str(ROOT))

RATIO_FLOOR = 0.62
FROM_DATE = datetime(2005, 1, 1)


def gt_symbols():
    syms = set()
    for name in ('trades_groundtruth.csv', 'trades_groundtruth_bluesky.csv'):
        for row in csv.DictReader(open(STUDY / 'data' / name)):
            syms.add(row['symbol'])
    return sorted(syms)


def scale_broken_dates(cur, symbol):
    rows = cur.execute(
        "select date, close from market_data_unified where symbol=? and timeframe='day' "
        "order by date", (symbol,)).fetchall()
    breaks = []
    for (d1, c1), (d2, c2) in zip(rows, rows[1:]):
        if c1 and c2 and c2 / c1 <= RATIO_FLOOR:
            breaks.append((d1, d2, round(c2 / c1, 3)))
    return breaks, len(rows)


def main():
    conn = sqlite3.connect(str(DB))
    cur = conn.cursor()
    syms = gt_symbols()

    missing, broken, ok = [], [], []
    for s in syms:
        breaks, n = scale_broken_dates(cur, s)
        if n == 0:
            missing.append(s)
        elif breaks:
            broken.append(s)
            print(f'BROKEN  {s}: {breaks[:3]}{" +more" if len(breaks) > 3 else ""}')
        else:
            ok.append(s)
    print(f'\nStudy symbols: {len(syms)} | ok {len(ok)} | broken {len(broken)} | missing {len(missing)}')
    print('missing:', missing)
    print('broken :', broken)

    # full-DB defect sizing (read-only)
    all_syms = [r[0] for r in cur.execute(
        "select distinct symbol from market_data_unified where timeframe='day'")]
    db_broken = 0
    for s in all_syms:
        b, _ = scale_broken_dates(cur, s)
        if b:
            db_broken += 1
    print(f'\nFULL-DB SCAN: {db_broken}/{len(all_syms)} daily symbols have >=1 suspected '
          f'unadjusted split/bonus step (close ratio <= {RATIO_FLOOR})')

    if not APPLY:
        print('\nDRY RUN — rerun with --apply to repair.')
        return

    # ---- apply: backup, delete, re-download ----
    from kiteconnect import KiteConnect
    from config import KITE_API_KEY
    from services.data_manager import get_data_manager

    token = json.load(open(ROOT / 'backtest_data' / 'access_token.json'))['access_token']
    kite = KiteConnect(api_key=KITE_API_KEY)
    kite.set_access_token(token)
    kite.profile()  # raises if token invalid
    print('\nKite token valid; starting repair...')

    cur.execute("create table if not exists market_data_unified_bak142 as "
                "select * from market_data_unified where 0")
    for s in broken:
        cur.execute("insert into market_data_unified_bak142 "
                    "select * from market_data_unified where symbol=? and timeframe='day'", (s,))
        cur.execute("delete from market_data_unified where symbol=? and timeframe='day'", (s,))
        print(f'backed up + deleted daily rows for {s}')
    conn.commit()
    conn.close()

    dm = get_data_manager(kite=kite)
    todo = broken + missing
    okc, fail, errs = dm.download_data(todo, timeframe='day',
                                       from_date=FROM_DATE, to_date=datetime.now())
    print(f'\nDownload: {okc} ok, {fail} failed')
    for e in errs:
        print('  ERR', e)

    # post-check
    conn = sqlite3.connect(str(DB))
    cur = conn.cursor()
    print('\nPost-repair check:')
    for s in todo:
        b, n = scale_broken_dates(cur, s)
        r = cur.execute("select min(date), max(date) from market_data_unified "
                        "where symbol=? and timeframe='day'", (s,)).fetchone()
        print(f'{s:12s} rows={n} {r[0]}..{r[1]} residual_breaks={len(b)} {b[:2]}')


if __name__ == '__main__':
    main()
