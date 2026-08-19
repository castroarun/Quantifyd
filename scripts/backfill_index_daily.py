#!/usr/bin/env python3
"""Backfill + daily top-up of NSE index daily bars into market_data.db.

Broad indices (Nifty50/500, Midcap, Smallcap, VIX) largely exist already;
this adds Next 50 and the sector indices, and tops everything up to today.
Rerunnable: fetches only what is missing after the stored max(date).
Also usable as the 16:20 cron top-up.
"""
import sqlite3
import sys
import time
from datetime import date, datetime, timedelta

sys.path.insert(0, '/home/arun/quantifyd')

from services.kite_service import get_kite_with_refresh  # noqa: E402

DB = '/home/arun/quantifyd/backtest_data/market_data.db'
START = date(2015, 1, 1)

# storage symbol -> Kite index name (NSE INDICES segment)
INDICES = {
    'NIFTY50':        'NIFTY 50',
    'NIFTYNEXT50':    'NIFTY NEXT 50',
    'NIFTYMIDCAP150': 'NIFTY MIDCAP 150',
    'NIFTYSMLCAP250': 'NIFTY SMLCAP 250',
    'NIFTY500':       'NIFTY 500',
    'BANKNIFTY':      'NIFTY BANK',
    'NIFTYIT':        'NIFTY IT',
    'NIFTYPHARMA':    'NIFTY PHARMA',
    'NIFTYREALTY':    'NIFTY REALTY',
    'NIFTYAUTO':      'NIFTY AUTO',
    'NIFTYMETAL':     'NIFTY METAL',
    'NIFTYFMCG':      'NIFTY FMCG',
    'NIFTYENERGY':    'NIFTY ENERGY',
    'NIFTYPSUBANK':   'NIFTY PSU BANK',
    'NIFTYFINSRV':    'NIFTY FIN SERVICE',
    'INDIAVIX':       'INDIA VIX',
}


def main():
    kite = get_kite_with_refresh()
    # resolve tokens once from the instruments dump
    toks = {}
    for inst in kite.instruments('NSE'):
        if inst.get('segment') == 'INDICES' and inst['tradingsymbol'] in INDICES.values():
            toks[inst['tradingsymbol']] = inst['instrument_token']
    print(f'resolved {len(toks)}/{len(INDICES)} index tokens')

    con = sqlite3.connect(DB)
    total = 0
    for store_sym, kite_name in INDICES.items():
        tok = toks.get(kite_name)
        if not tok:
            print(f'  {store_sym}: no token for "{kite_name}" — skipped')
            continue
        row = con.execute(
            "SELECT MAX(date) FROM market_data_unified "
            "WHERE symbol=? AND timeframe='day'", (store_sym,)).fetchone()
        last = row[0][:10] if row and row[0] else None
        frm = (datetime.fromisoformat(last) + timedelta(days=1)).date() if last \
            else START
        if frm > date.today():
            print(f'  {store_sym}: up to date ({last})')
            continue
        n_sym = 0
        cur = frm
        while cur <= date.today():
            chunk_end = min(cur + timedelta(days=1900), date.today())
            try:
                data = kite.historical_data(tok, cur, chunk_end, 'day')
            except Exception as e:
                print(f'  {store_sym}: fetch {cur}..{chunk_end} failed: {e}')
                break
            for b in data:
                d = b['date'].strftime('%Y-%m-%d')
                con.execute(
                    "INSERT OR IGNORE INTO market_data_unified "
                    "(symbol, timeframe, date, open, high, low, close, volume) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (store_sym, 'day', d, b['open'], b['high'], b['low'],
                     b['close'], b.get('volume', 0)))
                n_sym += 1
            cur = chunk_end + timedelta(days=1)
            time.sleep(0.4)
        con.commit()
        total += n_sym
        print(f'  {store_sym}: +{n_sym} bars (from {frm})')
    con.close()
    print(f'DONE — {total} bars added')


if __name__ == '__main__':
    main()
