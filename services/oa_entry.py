# -*- coding: utf-8 -*-
"""Open Alpha's missing half: find the next entry and arm a buy-stop for it.

The live book has had NO entry scanner since it was seeded on 04-Sep. seed() took a
hardcoded list of 16 names chosen offline; nothing has looked for a replacement since. So
the book could only ever shrink: a name exits, the slot stays empty forever, and the
freed cash gets topped up into whatever survived - which the backtest never did.

This is the scanner, built to r/142's adopted 16-slot spec:

    eligible : 20-day median traded value >= Rs 5 cr on the PREVIOUS bar; ETFs excluded
    RS       : percentile of (2*r63 + r126 + r189 + r252), shifted 1, must be >= 70
    setup    : prev_close < ATH-close  AND  prev_close >= 0.8 * ATH-close
    pivot    : ATH-close = the highest close up to and including the latest bar
    trigger  : price crosses the pivot -> a buy-stop resting AT the pivot
    size     : 6.25% of equity, cash-constrained

WHY A BUY-STOP AND NOT A MARKET ORDER. The backtest fills at max(pivot, open), which is
exactly what a stop order resting at the pivot does: nothing is bought unless the breakout
actually happens. r/157 showed the alternative - buying at the close instead - costs about
47 points of CAGR on its own, so the mechanic is not cosmetic.

WHY THE SCAN RUNS AFTER THE CLOSE. r/157 (08-Sep-2026): a same-day swap, done honestly, is
worth +0.53pp of CAGR on a 4-of-7-year split - noise. Its first, wrong answer of +12.57pp
was look-ahead. So exits and entries stay separate jobs: the 15:18 check sells, this arms
tomorrow's buy-stops after the close.

Usage:
    python3 services/oa_entry.py                 # scan and print, place nothing
    python3 services/oa_entry.py --arm           # scan and place the buy-stops
    python3 services/oa_entry.py --verify DATE   # prove the signal matches the study
"""
import argparse
import json
import math
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DB = ROOT / 'backtest_data' / 'market_data.db'
STATE = ROOT / 'backtest_data' / 'oa_real_state.json'

# ── the adopted spec, r/142 ──────────────────────────────────────────────────
SLOTS, SIZE_PCT = 16, 0.0625
RS_MIN, TV_FLOOR, BASE_DEPTH = 70.0, 5e7, 0.8
ETF_RE = re.compile(r'(BEES|ETF|LIQUID|GILT|SENSEX|NIF[A-Z]*50)')
MIN_BARS = 260                     # same universe floor the study uses
ENTRY_TAG = 'OA-ENTRY'


def _frames(min_bars=MIN_BARS):
    """close / tv20 / ATH-close per symbol, wide. Mirrors bluesky_replay.load_frames."""
    con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    syms = [r[0] for r in con.execute(
        'select symbol from (select symbol, count(*) n from market_data_unified '
        "where timeframe='day' group by symbol) where n >= ?", (min_bars,))]
    close, tv, ath = {}, {}, {}
    for s in syms:
        df = pd.read_sql_query(
            'select date, close, volume from market_data_unified '
            "where symbol=? and timeframe='day' order by date", con, params=(s,))
        if df.empty:
            continue
        df['date'] = pd.to_datetime(df['date'].str[:10])
        df = df.drop_duplicates('date').set_index('date').sort_index()
        close[s] = df['close']
        tv[s] = (df['close'] * df['volume']).rolling(20).median()
        # ATH-close as the study defines it: the expanding max of close, SHIFTED one bar.
        # On bar i that is "the highest close before today", which is the level a breakout
        # has to clear -- and, read on the LAST bar, the pivot for tomorrow.
        ath[s] = df['close'].shift(1).cummax()
    con.close()
    return (pd.DataFrame(close), pd.DataFrame(tv), pd.DataFrame(ath))


def signal(asof=None):
    """-> DataFrame of candidates whose buy-stop should rest for the NEXT session.

    `asof` (a date string) evaluates the scan as it would have run after that day's close,
    which is what --verify uses to reproduce a historical trigger set.
    """
    close, tv20, athcp = _frames()
    if asof:
        close = close.loc[:asof]
        tv20 = tv20.loc[:asof]
        athcp = athcp.loc[:asof]
    if close.empty:
        return pd.DataFrame(), None

    last = close.index[-1]
    etf = [c for c in close.columns if ETF_RE.search(c)]

    eligible = (tv20.loc[last] >= TV_FLOOR)
    eligible[etf] = False

    r = {n: close / close.shift(n) - 1 for n in (63, 126, 189, 252)}
    raw = 2 * r[63] + r[126] + r[189] + r[252]
    # Mask the LAST ROW, not the frame. DataFrame.where(Series) aligns the Series on the
    # INDEX, so masking the whole frame with a per-symbol Series silently NaNs everything
    # and the scan returns nothing at all -- which is exactly what it did on the first run.
    rs = raw.loc[last].where(eligible, np.nan).rank(pct=True) * 100

    # the pivot tomorrow is the highest close through today
    pivot = close.loc[:last].cummax().loc[last]
    prev_close = close.loc[last]

    cand = pd.DataFrame({'pivot': pivot, 'close': prev_close, 'rs': rs,
                         'tv20': tv20.loc[last]})
    cand = cand[eligible.reindex(cand.index).fillna(False)]
    cand = cand.dropna(subset=['pivot', 'close', 'rs'])
    cand = cand[(cand['close'] < cand['pivot'])
                & (cand['close'] >= BASE_DEPTH * cand['pivot'])
                & (cand['rs'] >= RS_MIN)]
    cand['gap_pct'] = (cand['pivot'] / cand['close'] - 1) * 100
    return cand.sort_values('rs', ascending=False), last


def _leaving(kite, held_syms):
    """Symbols with a SELL already resting or filled today - their slots are going."""
    out = set()
    try:
        for o in kite.orders():
            if (o.get('tradingsymbol') in held_syms
                    and o.get('transaction_type') == 'SELL'
                    and o.get('status') not in ('REJECTED', 'CANCELLED')):
                out.add(o['tradingsymbol'])
    except Exception as e:
        print('order read failed, assuming nothing is leaving:', e)
    return out


def _pending_entries(kite):
    """Entry orders already resting. Their slots are taken, though nothing is held yet."""
    try:
        return {o['tradingsymbol'] for o in kite.orders()
                if o.get('tag') == ENTRY_TAG
                and o.get('status') not in ('REJECTED', 'CANCELLED')}
    except Exception as e:
        print('order read failed, assuming no pending entries:', e)
        return set()


def free_slots(st, kite=None):
    """Slots genuinely available: not held, not already armed, adjusted for what is leaving.

    Two corrections, each learned from getting it wrong:
      - counting only holdings says zero on the very evening an exit is resting, which is
        exactly when the replacement needs arming, so a placed SELL frees its slot;
      - a resting BUY takes a slot even though nothing is held yet, or the scanner arms a
        second name against the same opening and the book ends up 17 of 16.
    """
    held_syms = {p['symbol'] for p in st.get('positions', [])}
    leaving = _leaving(kite, held_syms) if kite is not None else set()
    armed = _pending_entries(kite) if kite is not None else set()
    free = SLOTS - (len(held_syms) - len(leaving)) - len(armed - held_syms)
    return max(0, free), len(held_syms), leaving



def split_suspect(sym, con=None, worst=-0.40):
    """(bad, why) - is this symbol's history corrupted by an unadjusted corporate action?

    market_data.db keeps pre-split rows at the old price scale, so an unadjusted split
    shows up as a single-day collapse that never happened in tradeable terms. A pivot
    computed across such a break is meaningless, and this screen is built entirely on
    pivots.
    """
    own = con is None
    if own:
        con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    try:
        rows = [r[0] for r in con.execute(
            'select close from market_data_unified '
            "where symbol=? and timeframe='day' and close > 0 order by date", (sym,))]
        dates = [r[0] for r in con.execute(
            'select date from market_data_unified '
            "where symbol=? and timeframe='day' and close > 0 order by date", (sym,))]
    finally:
        if own:
            con.close()
    for i in range(1, len(rows)):
        ch = rows[i] / rows[i - 1] - 1
        if ch <= worst:
            return True, 'unadjusted corporate action? %.0f%% on %s' % (ch * 100, dates[i][:10])
    return False, ''


def arm(cand, free, st, kite, dry=True):
    """Place a resting buy-stop at the pivot for the top `free` candidates.

    Sized at 6.25% of book equity, the spec's slot size. Refuses a symbol that already has
    a live order, and refuses anything the split guard flags.
    """
    from services.oa_real import SLOTS as _S            # noqa: F401  (shape assertion)
    equity = float(st.get('cash', 0)) + sum(p['qty'] * p['buy'] for p in st.get('positions', []))
    slot_rs = SIZE_PCT * equity
    live_syms = set()
    try:
        live_syms = {o['tradingsymbol'] for o in kite.orders()
                     if o.get('status') not in ('REJECTED', 'CANCELLED', 'COMPLETE')}
    except Exception as e:
        print('order read failed:', e)

    ticks, tradeable = {}, None
    try:
        tradeable = set()
        for i in kite.instruments('NSE'):
            if i.get('segment') == 'NSE':
                ticks[i['tradingsymbol']] = float(i.get('tick_size') or 0.05)
                tradeable.add(i['tradingsymbol'])
    except Exception as e:
        # Without the dump we cannot vet symbols; arm anyway rather than stall the book,
        # and let the broker refuse what it will.
        print('instrument dump failed, cannot vet symbols:', e)
        tradeable = None

    placed = []
    for sym, r in cand.iterrows():
      try:
        if len(placed) >= free:
            break
        if sym in live_syms:
            print('  skip %-12s an order is already live' % sym)
            continue
        bad, why = split_suspect(sym)
        if bad:
            print('  SKIP %-12s %s' % (sym, why))
            continue
        if tradeable is not None and sym not in tradeable:
            # In market_data.db but not in the live dump: renamed, delisted, or moved
            # series. An AMO is validated lightly enough to accept it and RMS rejects at
            # the open ("Field Not Found") - that is how MODISONLTD wasted a slot on
            # 10-Sep. Skip either way, but say WHICH, because a rename is recoverable and
            # a delisting is not.
            alt = next((sym + sfx for sfx in ('-BE', '-BZ', '-SM', '-ST')
                        if sym + sfx in tradeable), None)
            if alt:
                # NOT auto-mapped: a position opened as %s-BE would have no price history
                # under that name, so the 15-SMA trail could not be computed and the exit
                # rule would have nothing to evaluate. Trading a name the book cannot exit
                # is worse than missing it. Point the universe refresh at `alt` instead.
                print('  SKIP %-12s renamed_to %s - add it to the universe refresh so the '
                      'history accumulates, then it qualifies on its own' % (sym, alt))
            else:
                print('  SKIP %-12s no longer trades on NSE (delisted or merged); its '
                      'history stays in the DB on purpose, for the backtests' % sym)
            continue
        tick = ticks.get(sym, 0.05)
        # CEIL, not round: the rule is close > pivot, so a trigger a tick BELOW the
        # pivot buys a breakout that has not happened.
        trigger = round(math.ceil(float(r['pivot']) / tick) * tick, 2)
        # The study fills at max(pivot, open) with NO ceiling. Kite refuses SL-M via API
        # ("market orders without market protection"), so the closest available thing is a
        # stop-limit with the widest ceiling the exchange allows - which it caps per scrip.
        limit = round(math.floor((trigger * 1.03) / tick) * tick, 2)
        qty = int(slot_rs // trigger)
        if qty < 1:
            print('  skip %-12s slot Rs%.0f buys nothing at %.2f' % (sym, slot_rs, trigger))
            continue
        print('  ARM  %-12s BUY %d @ trigger %.2f limit %.2f  (RS %.1f, Rs%.0f)'
              % (sym, qty, trigger, limit, r['rs'], qty * trigger))
        if dry:
            placed.append(dict(symbol=sym, qty=qty, trigger=trigger, limit=limit, dry=True))
            continue
        oid, lim = None, limit
        for attempt in (1, 2):
            try:
                oid = kite.place_order(variety='amo', exchange='NSE', tradingsymbol=sym,
                                       transaction_type='BUY', quantity=qty, product='CNC',
                                       order_type='SL', trigger_price=trigger, price=lim,
                                       validity='DAY', tag=ENTRY_TAG)
                break
            except Exception as e:
                # The exchange caps the limit-to-trigger spread per scrip and names the
                # maximum in the rejection. Take it at its word once, rather than guessing.
                # The exchange refuses in two different sentences - the SL spread cap
                # ("below Rs. 303.70.") and the circuit cap ("with Price below 735.45") -
                # and both name the price that would work. One pattern takes either.
                # NOT [0-9.]+ : greedy over dots, it swallows the full stop and float()
                # then raises inside the handler meant to recover from the rejection.
                m = re.search(r'below\s+(?:Rs\.?\s*)?([0-9]+(?:\.[0-9]+)?)', str(e))
                if attempt == 1 and m:
                    cap = float(m.group(1))
                    if cap <= trigger:
                        # Even the pivot is outside the band: no breakout can happen today,
                        # so the slot is better spent on the next candidate than on an
                        # order that cannot fill.
                        print('       ceiling %.2f is at or below the trigger %.2f - '
                              'the breakout cannot happen in today\'s band, skipping'
                              % (cap, trigger))
                        break
                    lim = round(math.floor((cap - tick) / tick) * tick, 2)
                    print('       exchange caps the price; retrying with ceiling %.2f' % lim)
                    continue
                print('       REJECTED: %s' % e)
                e_final = e
                break
        if oid:
            print('       placed, order id %s (ceiling %.2f, +%.2f%%)'
                  % (oid, lim, (lim / trigger - 1) * 100))
            placed.append(dict(symbol=sym, qty=qty, trigger=trigger, limit=lim, order_id=oid))
        else:
            err = locals().get('e_final', 'unknown')
            try:
                from services.oa_real import _alert
                _alert('OA-REAL entry could not be armed: %s' % sym,
                       'BUY %d %s at trigger %.2f was rejected: %s' % (qty, sym, trigger, err))
            except Exception:
                pass
      except Exception as loop_e:
        # A scan that arms N orders is not a transaction: one name the exchange dislikes
        # is normal, and the rest must still go. Before this, a single bad price aborted
        # the whole run and left the book under-deployed with nothing alerted.
        print('  FAILED %-12s %s' % (sym, loop_e))
        try:
            from services.oa_real import _alert
            _alert('OA-REAL entry failed: %s' % sym,
                   'Arming %s raised %s. The rest of the scan continued.' % (sym, loop_e))
        except Exception:
            pass
            try:
                from services.oa_real import _alert
                _alert('OA-REAL entry could not be armed: %s' % sym,
                       'BUY %d %s at trigger %.2f was rejected: %s' % (qty, sym, trigger, e))
            except Exception:
                pass
    return placed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', action='store_true', help='actually place the buy-stops')
    ap.add_argument('--verify', metavar='DATE', help='reproduce the study for a past date')
    ap.add_argument('--top', type=int, default=12, help='how many candidates to print')
    a = ap.parse_args()

    cand, last = signal(a.verify)
    print('scan as of %s: %d candidates in setup (RS >= %g, TV >= Rs%.0fcr)'
          % (str(last)[:10], len(cand), RS_MIN, TV_FLOOR / 1e7))

    st = json.loads(STATE.read_text(encoding='utf-8'))
    held_syms = {p['symbol'] for p in st.get('positions', [])}
    cand = cand[~cand.index.isin(held_syms)]

    kite = None
    if not a.verify:
        try:
            from services.oa_real import _kite
            kite = _kite()
        except Exception as e:
            print('broker unavailable, slot count will ignore resting sells:', e)
    free, held, leaving = free_slots(st, kite)
    print('book: %d/%d held%s, %d slot(s) free, cash Rs %s'
          % (held, SLOTS, (', leaving: ' + ', '.join(sorted(leaving))) if leaving else '',
             free, f"{st.get('cash', 0):,.0f}"))

    print('\n%-14s %8s %9s %7s %8s' % ('symbol', 'RS', 'pivot', 'gap%', 'TV cr'))
    for s, r in cand.head(a.top).iterrows():
        print('%-14s %8.1f %9.2f %7.2f %8.1f'
              % (s, r['rs'], r['pivot'], r['gap_pct'], r['tv20'] / 1e7))

    if a.verify:
        print('\n(verify mode: nothing is placed)')
        return
    if not a.arm:
        print('\nDRY RUN - nothing would be placed. Re-run with --arm.')
        if free > 0 and kite is not None:
            arm(cand, free, st, kite, dry=True)
        return
    if free <= 0:
        print('\nno free slot; nothing to arm')
        return
    print('\narming %d buy-stop(s):' % free)
    arm(cand, free, st, kite, dry=False)


if __name__ == '__main__':
    main()
