"""Onboard newly listed NSE equities into market_data.db (VPS-only).

Nothing in this repo ever added a *new* NSE symbol to `market_data_unified`. Symbols
arrived when Arun ran something by hand, which is why the coverage table shows manual
bulk waves (45 symbols all starting 2026-08-17, 41 on 2026-04-20). For a book that
trades stocks listed within the last six months, "we notice a listing when someone
remembers to look" is not a data pipeline — so this runs nightly.

WHY FULL HISTORY MATTERS (research/153 §2). The IPO-Base screen keys off a vetted
listing date derived from each symbol's FIRST ROW in the DB. Fetch a fixed window and
that first row is the window edge, not a listing: onboarding ALKALI and BIRLACABLE with
a 1900-day window gave both 1,292 bars starting 2021-06-23, which would have vetted two
decades-old companies as IPOs that listed on the same day.

`ipo_listing_table.py` only rejects a shared first-row date when >=8 symbols share it,
so small batches slip fake listings UNDER the guard instead of tripping it. Batching is
therefore no defence at all. `_fetch_full_history()` instead walks backwards until Kite
returns an empty window, so every symbol lands with its true first bar and the
bulk-wave heuristic never has to arbitrate.

Cron: 17:30 IST Mon-Fri, before the 17:45 universe refresh (so a name onboarded tonight
is kept current from tomorrow). Log: /tmp/onboard_new_listings.log
State: backtest_data/onboard_queue.json  (symbols seen but not yet fetched)
"""
import json
import re
import sqlite3
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DB = ROOT / 'backtest_data' / 'market_data.db'
QUEUE = ROOT / 'backtest_data' / 'onboard_queue.json'
MAX_PER_RUN = 25         # nightly there are 0-2 new names; the cap only paces the backlog
LOOKBACK_DAYS = 1900     # < Kite's 2000-day daily cap; a new listing has less anyway

# Kite's NSE "EQ" instrument_type is NOT just equities: it also carries thousands of
# listed bonds and debentures (0ABCL31-N0, 1003IIFL29-NC, ...). Filtering on
# instrument_type alone reports ~7,100 "missing symbols", almost all debt.
#
# The reliable discriminator is `name`: real equities carry one ("RELIANCE INDUSTRIES"),
# debt lines have name == ''. The regexes below are belt-and-braces on top of that.
SKIP_SUFFIX = re.compile(r'-(BE|BZ|RE|PP|RR|PC|PD|GS|PU|NP|PS|IV|PT|P1|N1|PM)$')
DEBT_SUFFIX = re.compile(r'-[A-Z0-9]{2}$')          # -N0 -NC -YW -Z4; BAJAJ-AUTO is safe
# INAV lines are an ETF's indicative NAV feed, not a tradeable stock (AB10BKINAV,
# ABSLLQINAV, ...). They pass every other test, so they are excluded by name.
ETF_RE = re.compile(r'(BEES|ETF|IETF|GOLD|SILVER|LIQUID|GSEC|SDL|NIFTY|SENSEX|BOND|INAV$)')


def is_equity(i):
    s = i.get('tradingsymbol') or ''
    return bool(
        i.get('instrument_type') == 'EQ' and i.get('segment') == 'NSE'
        and (i.get('name') or '').strip()          # debt lines have an empty name
        and s and not s[0].isdigit()
        and not SKIP_SUFFIX.search(s)
        and not DEBT_SUFFIX.search(s)
        and not ETF_RE.search(s))


def _fetch_full_history(dm, symbol, max_windows=14):
    """Walk backwards in <2000-day windows until Kite returns nothing.

    THIS IS THE WHOLE POINT OF THE SCRIPT, so it is worth being explicit. A single
    fixed-window fetch gives a *truncated* first row: onboarding ALKALI and BIRLACABLE
    with a 1900-day window produced 1,292 bars both starting 2021-06-23 — the window
    edge, not a listing date. `ipo_listing_table.py` derives listing dates from first
    rows, so those two decades-old companies would have been vetted as brand-new IPOs
    that listed on the same day.

    Worse, the per-run cap was actively harmful there: the listing table only rejects a
    shared first-row date when >=8 symbols share it, so onboarding in batches of five
    slipped fake listings in UNDER the guard rather than tripping it.

    Walking back until a window comes back empty gives the true first bar, so a genuine
    new listing gets its real listing date and an old company gets its real history —
    and the bulk-wave heuristic never has to arbitrate. New listings cost two calls;
    only the one-off backlog pays for the deep walk.
    """
    end = datetime.now()
    total = 0
    for _ in range(max_windows):
        start = end - timedelta(days=LOOKBACK_DAYS)
        try:
            n_ok, _n_fail, _errs = dm.download_data([symbol], timeframe='day',
                                                    from_date=start, to_date=end)
        except Exception as e:
            print(f'      window {start:%Y-%m-%d}..{end:%Y-%m-%d} failed: {e}', flush=True)
            break
        c = sqlite3.connect(str(DB))
        got = c.execute("select count(*) from market_data_unified where symbol=? and "
                        "timeframe='day' and date >= ? and date <= ?",
                        (symbol, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))).fetchone()[0]
        c.close()
        total += got
        if got == 0:                 # nothing in this window: we are past the listing
            break
        end = start - timedelta(days=1)
        time.sleep(0.4)
    return total


def _known():
    c = sqlite3.connect(str(DB))
    got = {r[0] for r in c.execute(
        "select distinct symbol from market_data_unified where timeframe='day'")}
    c.close()
    return got


def _load_queue():
    try:
        return json.load(open(QUEUE))
    except Exception:
        return dict(pending=[], onboarded=[], last_run=None)


def _save_queue(q):
    tmp = QUEUE.with_suffix('.json.tmp')
    json.dump(q, open(tmp, 'w'), indent=1)
    tmp.replace(QUEUE)


def main():
    ist = datetime.utcnow() + timedelta(hours=5, minutes=30)
    if '--intraday-ok' not in sys.argv and ist.weekday() < 5 and (ist.hour, ist.minute) < (15, 35):
        print(f'{ist} — market hours; aborting (use --intraday-ok to override)')
        return

    from kiteconnect import KiteConnect
    from config import KITE_API_KEY
    from services.data_manager import get_data_manager

    tok = json.load(open(ROOT / 'backtest_data' / 'access_token.json'))['access_token']
    kite = KiteConnect(api_key=KITE_API_KEY)
    kite.set_access_token(tok)
    kite.profile()

    known = _known()
    q = _load_queue()
    pending = list(q.get('pending', []))

    inst = kite.instruments('NSE')
    fresh = []
    for i in inst:
        s = i.get('tradingsymbol', '')
        if not is_equity(i) or s in known or s in pending:
            continue
        fresh.append(s)
    fresh.sort()
    pending.extend(fresh)
    print(f'{datetime.now()} NSE EQ instruments {len(inst)}; known {len(known)}; '
          f'newly seen {len(fresh)}; queue now {len(pending)}', flush=True)

    batch, pending = pending[:MAX_PER_RUN], pending[MAX_PER_RUN:]
    if not batch:
        q.update(pending=pending, last_run=str(datetime.now()))
        _save_queue(q)
        print('nothing to onboard', flush=True)
        return

    dm = get_data_manager(kite=kite)
    done = []
    for s in batch:
        try:
            _fetch_full_history(dm, s)
            c = sqlite3.connect(str(DB))
            row = c.execute("select count(*), min(date), max(date) from market_data_unified "
                            "where symbol=? and timeframe='day'", (s,)).fetchone()
            c.close()
            if row and row[0]:
                print(f'  + {s}: {row[0]} bars, {row[1][:10]} -> {row[2][:10]}', flush=True)
                done.append(dict(symbol=s, bars=row[0], first=row[1][:10],
                                 ts=str(datetime.now())))
            else:
                print(f'  ! {s}: no data returned', flush=True)
        except Exception as e:
            print(f'  ERR {s}: {e}', flush=True)

    q['pending'] = pending
    q.setdefault('onboarded', []).extend(done)
    q['last_run'] = str(datetime.now())
    _save_queue(q)
    print(f'DONE: onboarded {len(done)}, {len(pending)} still queued', flush=True)
    if done:
        print('NOTE: rerun scripts/ipo_listing_table.py to re-vet listing dates.', flush=True)


if __name__ == '__main__':
    main()
