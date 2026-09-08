"""IPO-Base book — the research/153 adopted spec, run forward on real prices.

WHAT THIS IS. research/153 found that a breakout from a base built by a recently listed
stock is the first genuine complement to True North and Open Alpha: 31.03% CAGR /
-20.88% drawdown / Calmar 1.50 over 2006-2026 after tax, at correlation 0.16 to OA and
0.18 to TN. It has never traded. This runs it forward so its trades are visible before
any money is committed.

PAPER UNTIL ARMED (Arun, 06-Sep-2026). The book runs on a notional Rs 10,00,000 until a
real deposit is routed to it through the Capital Desk, which flips `ipo_status` to
'live' in backtest_data/allocation_targets.json. From then on the same signals are
real-money instructions. Execution is manual-assisted either way: like Open Alpha this
book has no executor, so it alerts with the exact order and Arun places it.

THE SPEC (results/ipo_adopted_spec.json, unchanged):
  universe    NSE equities with a VETTED listing date, ETFs excluded, all rows before
              the listing date masked
  age band    listed <= 6 months ago AND >= 25 bars of history
  liquidity   20-day median traded value >= Rs 5 cr at t-1
  base        last 25 bars; pivot = highest CLOSE, shifted 1; depth (pivot to lowest
              low) <= 30%; and close[t-1] < pivot, so it is not already extended
  RS          OFF. r/153 section 3: a strict RS >= 70 yields ZERO signals in this age
              band, because a 252-day relative-strength score does not exist for a
              stock that has traded for four months
  trigger     close[t] > pivot
  fill        next day, buy-stop AT the pivot, filled max(pivot, open)
  exits       priority order: stop (close <= fill x 0.92) -> target (close >= fill x
              1.25) -> trail (close < SMA-20, never on the entry bar)
  book        8 slots at 18.75% of equity, no market gate, 25 bps per side

TWO RULES THAT ARE NOT IN THE BACKTEST, pre-registered here before the book wrote a
single row, because a live book has to answer questions a backtest never faced:

  1. TIEBREAK. When more than 8 candidates trigger, the backtest picked among them with
     rng.permutation and published the median of 30 seeds — a 28.82-33.44% spread that
     is pure selection luck. A live book cannot draw lots. Open Alpha breaks ties by
     highest relative strength, which is unavailable here (see RS above), so this book
     takes the HIGHEST 20-DAY MEDIAN TRADED VALUE first. It is deterministic, and it
     leans toward the capacity-friendly end of the candidate set. It is NOT what was
     backtested, and the soak must report realised selection against the seed band.

  2. CORPORATE-ACTIONS GUARD. market_data.db is not retroactively split-adjusted, and
     IPO-age names split and issue bonuses often. A 1:10 split drops the close ~90%
     overnight, which would fire the -8% stop and book a fake -90% trade. Any single-day
     close move below -40% is treated as a DATA EVENT: the position is held, and an
     alert is raised for a human to check. A real -40% day would also be held, which is
     the safer error of the two.

Modes:
  main (default)  nightly cycle: exits -> fills from yesterday's pending -> scan ->
                  nav point -> write UI
  --dry           compute and print, write nothing
  --ui-only       rebake the UI JSON from frozen state (safe any time, no Kite)

State: backtest_data/ipo_paper_state.json   UI: static/app/ipo_paper.json
"""
import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / 'backtest_data' / 'market_data.db'
STATE = ROOT / 'backtest_data' / 'ipo_paper_state.json'
LOCK = ROOT / 'backtest_data' / 'ipo_paper_state.lock'
UI_JSON = ROOT / 'static' / 'app' / 'ipo_paper.json'
ALLOC = ROOT / 'backtest_data' / 'allocation_targets.json'
LISTINGS = ROOT / 'research' / '153_ipo_base' / 'results' / 'listing_dates.csv'
FEED = Path('/tmp/nas_alert_feed.log')

CAPITAL = 1_000_000          # notional while on paper
SLOTS = 8
SIZE_PCT = 0.1875
STOP = 0.08                  # close <= fill * 0.92
TARGET = 0.25                # close >= fill * 1.25
TRAIL_SMA = 20               # close < SMA-20, entry bar exempt
BASE_L = 25                  # base window, in bars
MAX_AGE_M = 6

# MIN_BARS: the spec says 25. The BACKTEST NEVER TESTED 25.
#
# research/153's adopted spec records `min_bars: 25`, and ipo_replay.build_trigger()
# honours it — but the panel that harness scans is built by Ctx with
#   "... group by symbol) where n >= 60"
# so a stock is invisible to the study until it has SIXTY daily bars, about three
# months of trading. The 25-bar floor is therefore never the binding constraint in the
# published result: the 31.03% CAGR was earned on stocks aged roughly 3-6 months, not
# 25 days to 6 months.
#
# Found on 06-Sep-2026 by reconciling this engine against ipo_replay over the last 34
# trading days. At min_bars=25 the two agreed on only 75% of signals, and every single
# disagreement was this engine seeing a genuine recent IPO the study could not:
# INDOMIM (27 bars), LASERPOWER (37), CORDELIA (48), TURTLEMINT (50), VAML and
# VEDPOWER (59). Notably the study never produced a signal this engine missed.
#
# A forward book must run the strategy that was VALIDATED, not a more permissive
# reading of its written spec, so this matches the harness at 60. The wider band may
# well be better — those are real breakouts on real IPOs — but that is an untested
# hypothesis and belongs in a study, not in a live book. Registered for review.
MIN_BARS = 60
MAX_DEPTH = 0.30
TV_FLOOR = 5e7               # Rs 5 cr, 20-day median traded value
COST = 0.0025
DATA_EVENT_DROP = -0.40      # single-day close move below this is a split, not a loss

ETF_PAT = ('BEES', 'ETF', 'IETF', 'GOLD', 'SILVER', 'LIQUID', 'GSEC', 'SDL', 'INAV')


def ist_now():
    return datetime.utcnow() + timedelta(hours=5, minutes=30)


def _alert(title, body, urgency='critical'):
    try:
        with open(FEED, 'a') as f:
            f.write(json.dumps(dict(ts=str(datetime.now()), book='IPO-BASE',
                                    urgency=urgency, title=title, body=body)) + '\n')
    except Exception:
        pass


# ───────────────────────── state ─────────────────────────
IPO_TAG = 'IPO-ENTRY'
SEEN_ORDERS = ROOT / 'backtest_data' / 'ipo_applied_orders.json'


def own_fills():
    """{symbol: (qty, avg)} from THIS BOOK'S completed orders only, never holdings.

    Matching a book against broker holdings by symbol is unsafe in a shared account:
    on 08-Sep that approach handed Open Alpha 481 shares of a stock it owned 9 of,
    because the account also carries personal holdings and other books. This reads the
    order book, keeps only completed CNC buys carrying this book's tag, and skips any
    order already applied on an earlier run.
    """
    try:
        from kiteconnect import KiteConnect
        api_key = [l.split('=', 1)[1].strip() for l in open(ROOT / '.env')
                   if l.startswith('KITE_API_KEY')][0]
        tok = json.load(open(ROOT / 'backtest_data' / 'access_token.json'))
        k = KiteConnect(api_key=api_key)
        k.set_access_token(tok.get('access_token') or tok.get('token'))
        orders = k.orders()
    except Exception as e:
        print('order book unreachable:', e)
        return None, None
    seen = json.load(open(SEEN_ORDERS)) if SEEN_ORDERS.exists() else {}
    out = {}
    for o in orders:
        oid = str(o.get('order_id'))
        if (o.get('status') != 'COMPLETE' or o.get('transaction_type') != 'BUY'
                or o.get('product') != 'CNC' or not o.get('filled_quantity')):
            continue
        if (o.get('tag') or '') != IPO_TAG or oid in seen:
            continue
        s = o['tradingsymbol']
        q, px = int(o['filled_quantity']), float(o['average_price'])
        pq, pv = out.get(s, (0, 0.0))
        out[s] = (pq + q, pv + q * px)
        seen[oid] = dict(ts=str(datetime.now()), symbol=s, qty=q, price=px)
    return {s: (q, v / q) for s, (q, v) in out.items()}, seen


def book_mode():
    """('paper'|'live', capital). Live capital is whatever the Capital Desk has funded."""
    try:
        a = json.load(open(ALLOC))
        if a.get('ipo_status') == 'live':
            return 'live', float(a.get('ipo_funded', 0.0) or 0.0)
    except Exception:
        pass
    return 'paper', float(CAPITAL)


def load_state():
    if STATE.exists():
        return json.load(open(STATE))
    mode, cap = book_mode()
    return dict(book='IPO-BASE', mode=mode, capital=cap, cash=cap, positions=[],
                pending=[], nav=[], trades=[], missed=[], data_events=[],
                started=str(date.today()), last_run=None)


def save_state(st):
    tmp = STATE.with_suffix('.json.tmp')
    json.dump(st, open(tmp, 'w'), indent=1, default=str)
    os.replace(tmp, STATE)


def acquire_lock(tries=30, wait=2.0):
    for _ in range(tries):
        try:
            fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            time.sleep(wait)
    return False


def release_lock():
    try:
        LOCK.unlink()
    except FileNotFoundError:
        pass


# ───────────────────────── universe ─────────────────────────
def load_listings():
    """The VETTED listing table. Never the naive 'first row in the DB' proxy.

    r/153 measured that proxy at 70% accuracy: bulk data-onboarding waves masquerade as
    IPOs (451 symbols 'listed' on 2005-01-03, ABB among them) and pre-listing junk rows
    sit on reused tickers (DELHIVERY carries 8 rows at Rs 5-11 before its real Rs 536
    listing — a 93x jump INSIDE what a base window would measure).
    """
    df = pd.read_csv(LISTINGS)
    df = df[df['accepted'].astype(str).str.lower().isin(('true', '1'))]
    df = df[df['list_date'].notna()]
    return {r.symbol: pd.Timestamp(str(r.list_date)[:10]) for r in df.itertuples()}


def load_wide(asof=None):
    """Wide panels for names inside the age band, with pre-listing rows masked."""
    import sqlite3
    listing = load_listings()
    conn = sqlite3.connect(str(DB))
    asof = pd.Timestamp(asof or date.today())
    lo = asof - pd.Timedelta(days=int(MAX_AGE_M * 30.44))
    cand = [s for s, d in listing.items() if lo <= d < asof]
    closes, opens, lows, tv = {}, {}, {}, {}
    for s in cand:
        if any(p in s for p in ETF_PAT):
            continue
        df = pd.read_sql_query(
            "select date, open, high, low, close, volume from market_data_unified "
            "where symbol=? and timeframe='day' order by date", conn, params=(s,))
        if df.empty:
            continue
        df['date'] = pd.to_datetime(df['date'].str[:10])
        df = df.drop_duplicates('date').set_index('date').sort_index()
        df = df[df.index >= listing[s]]           # mask pre-listing junk rows
        if len(df) < MIN_BARS:
            continue
        closes[s] = df['close']
        opens[s] = df['open']
        lows[s] = df['low']
        tv[s] = (df['close'] * df['volume']).rolling(20).median()
    conn.close()
    if not closes:
        return None
    return dict(close=pd.DataFrame(closes).sort_index(),
                open=pd.DataFrame(opens).sort_index(),
                low=pd.DataFrame(lows).sort_index(),
                tv=pd.DataFrame(tv).sort_index()), listing


def scan(wide, listing, asof):
    """Candidates whose close TODAY breaks the base pivot. Returns rows sorted by the
    pre-registered tiebreak (highest 20-day median traded value first)."""
    close, low, tvp = wide['close'], wide['low'], wide['tv']
    if asof not in close.index:
        return []
    i = close.index.get_loc(asof)
    if i < BASE_L:
        return []
    win = close.iloc[i - BASE_L:i]                 # last 25 bars, EXCLUDING today
    pivot = win.max()
    baselow = low.iloc[i - BASE_L:i].min()
    prev = close.iloc[i - 1]
    today = close.iloc[i]
    out = []
    for s in close.columns:
        pv, bl, pc, tc = pivot.get(s), baselow.get(s), prev.get(s), today.get(s)
        if not np.isfinite([pv, bl, pc, tc]).all() or pv <= 0:
            continue
        bars = int(close[s].iloc[:i + 1].notna().sum())
        if bars < MIN_BARS:
            continue
        depth = (pv - bl) / pv
        if depth > MAX_DEPTH:
            continue
        if pc >= pv:                               # already extended
            continue
        if tc <= pv:                               # no trigger
            continue
        liq = tvp[s].iloc[i - 1] if i >= 1 else np.nan
        if not np.isfinite(liq) or liq < TV_FLOOR:
            continue
        out.append(dict(symbol=s, pivot=round(float(pv), 2), close=round(float(tc), 2),
                        depth_pct=round(float(depth) * 100, 1), tv=float(liq),
                        listed=str(listing[s].date()),
                        age_days=int((asof - listing[s]).days)))
    # PRE-REGISTERED TIEBREAK — deterministic, capacity-friendly. See the module header.
    out.sort(key=lambda r: -r['tv'])
    return out


def sma20(close, sym, upto):
    s = close[sym].loc[:upto].dropna()
    return float(s.iloc[-TRAIL_SMA:].mean()) if len(s) >= TRAIL_SMA else None


# ───────────────────────── UI ─────────────────────────
def write_ui(st, wide, asof, log, dry=False):
    close = wide['close'] if wide else None
    rows = []
    tot_val = tot_pnl = 0.0
    for p in st['positions']:
        lp = float(close[p['symbol']].loc[:asof].dropna().iloc[-1]) \
            if close is not None and p['symbol'] in close.columns else p['buy']
        val = p['qty'] * lp
        pnl = p['qty'] * (lp - p['buy'])
        tot_val += val
        tot_pnl += pnl
        tr = sma20(close, p['symbol'], asof) if close is not None and p['symbol'] in close.columns else None
        rows.append(dict(**p, ltp=round(lp, 2), value=round(val), pnl=round(pnl),
                         pnl_pct=round((lp / p['buy'] - 1) * 100, 2),
                         trail=round(tr, 2) if tr else None,
                         target=round(p['buy'] * (1 + TARGET), 2),
                         to_stop_pct=round((lp / p['stop'] - 1) * 100, 1),
                         to_trail_pct=round((lp / tr - 1) * 100, 1) if tr else None,
                         days=(pd.Timestamp(asof) - pd.Timestamp(p['entry_date'])).days))
    cash = float(st['cash'])
    nav = tot_val + cash
    for r in rows:
        r['weight'] = round(100 * r['value'] / nav, 1) if nav else 0
    realized = sum(t.get('net_pnl', 0) for t in st.get('trades', []))
    cap = float(st['capital'])
    ui = dict(updated=str(datetime.now()), asof=str(asof)[:10], mode=st.get('mode', 'paper'),
              positions=rows, capital=round(cap), cash=round(cash), value=round(tot_val),
              nav=round(nav), pnl=round(tot_pnl), realized=round(realized),
              gain=round(nav + realized - cap),
              return_pct=round(100 * (nav + realized - cap) / cap, 2) if cap else 0,
              invested_pct=round(100 * tot_val / nav, 1) if nav else 0,
              slots=SLOTS, slots_used=len(rows),
              pending=st.get('pending', []), navcurve=st.get('nav', []),
              trades=st.get('trades', [])[-100:], data_events=st.get('data_events', [])[-20:],
              started=st.get('started'), log=log)
    if dry:
        print(json.dumps({k: ui[k] for k in ('asof', 'mode', 'nav', 'cash', 'slots_used')}, indent=1))
        return ui
    tmp = UI_JSON.with_suffix('.json.tmp')
    json.dump(ui, open(tmp, 'w'), indent=1, default=str)
    os.replace(tmp, UI_JSON)
    return ui


# ───────────────────────── engine ─────────────────────────
def arm():
    """Take the book live NOW, without running a scan.

    The full cycle refuses to run before 15:35 on a weekday, because scanning on partial
    candles would arm tomorrow buy-stops off a half-formed bar. But ARMING is not a scan:
    it is a state transition, and it has to be possible between sessions — otherwise a
    deposit made at 07:36 cannot trade until the following day.

    Paper positions are discarded (they were bought with notional money and are not in
    the account). The PENDING BUY-STOP IS KEPT: the signal came from a completed close
    and is just as valid for real money; only its size changes, and the executor computes
    that from live equity when it places the order.
    """
    mode, cap = book_mode()
    if mode != 'live':
        print('allocation says the sleeve is still on paper - nothing to arm')
        return
    if not acquire_lock():
        print('book busy')
        return
    try:
        st = load_state()
        if st.get('mode') == 'live':
            print('already live; capital Rs %s cash Rs %s'
                  % (format(round(st['capital']), ','), format(round(st['cash']), ',')))
            return
        ghosts = [dict(x) for x in st.get('positions', [])]
        kept = list(st.get('pending', []))
        st['positions'] = []
        st.setdefault('discarded_on_arming', []).extend(ghosts)
        st['mode'] = 'live'
        st['capital'] = cap
        st['cash'] = cap
        st['pending'] = kept
        st['nav'] = []
        st['started'] = str(date.today())
        save_state(st)
    finally:
        release_lock()
    if ghosts:
        names = '; '.join('%s x%d @%.2f' % (g['symbol'], g['qty'], g['buy']) for g in ghosts)
        print('discarded paper positions: %s' % names)
        _alert('IPO is LIVE - paper positions discarded',
               'You do NOT own these, they were paper fills: %s. The book restarts flat on '
               'Rs %s of real capital.' % (names, format(round(cap), ',')))
    print('ARMED live on Rs %s; buy-stops carried: %s'
          % (format(round(cap), ','), [(c['symbol'], c['pivot']) for c in kept] or 'none'))


def main():
    dry = '--dry' in sys.argv
    ui_only = '--ui-only' in sys.argv
    now = ist_now()
    if not (dry or ui_only) and now.weekday() < 5 and (now.hour, now.minute) < (15, 35):
        print(f'{now} — market hours; refusing to run the cycle on partial candles')
        return
    if not acquire_lock():
        print('locked — another run in progress')
        return
    try:
        st = load_state()
        mode, cap = book_mode()
        if mode != st.get('mode'):
            was = st.get('mode')
            if was == 'paper' and mode == 'live':
                # GOING LIVE RESETS THE BOOK. Paper positions were bought with notional
                # money and are not in the account; carrying them across would put
                # holdings nobody owns onto a real-money book. Start from the broker.
                ghosts = [dict(p) for p in st.get('positions', [])]
                st['positions'] = []
                st['pending'] = []
                st.setdefault('discarded_on_arming', []).extend(ghosts)
                st['cash'] = cap
                st['capital'] = cap
                st['started'] = str(date.today())
                st['nav'] = []
                if ghosts:
                    names = ', '.join('%s x%d @%.2f' % (g['symbol'], g['qty'], g['buy'])
                                      for g in ghosts)
                    _alert('IPO went LIVE — paper positions discarded',
                           f'You do NOT own these; they were paper fills: {names}. '
                           f'The book restarts flat on Rs {cap:,.0f} of real capital.')
                    print(f'ARMED: discarded {len(ghosts)} paper positions ({names})')
                else:
                    _alert('IPO went LIVE', f'Flat, on Rs {cap:,.0f} of real capital.', 'low')
            else:
                delta = cap - float(st['capital'])
                st['capital'] = cap
                st['cash'] = float(st['cash']) + delta
                _alert('IPO book mode change',
                       f'now {mode.upper()} with capital Rs {cap:,.0f}', 'low')
            st['mode'] = mode
        loaded = load_wide()
        if loaded is None:
            print('no symbols inside the age band today')
            write_ui(st, None, date.today(), ['no candidates in the age band'], dry)
            return
        wide, listing = loaded
        close = wide['close']
        asof = close.index[-1]
        log = [f'panel {close.shape[1]} names in the age band, asof {str(asof)[:10]}']

        if ui_only:
            write_ui(st, wide, asof, log + ['ui-only rebake'], dry=False)
            print('ui-only done')
            return

        # ---- 1. exits, on today's close ----
        keep = []
        for p in st['positions']:
            s = p['symbol']
            if s not in close.columns:
                keep.append(p)
                continue
            ser = close[s].loc[:asof].dropna()
            if ser.empty:
                keep.append(p)
                continue
            px = float(ser.iloc[-1])
            prev = float(ser.iloc[-2]) if len(ser) > 1 else px
            # corporate-actions guard, BEFORE any exit test
            if prev > 0 and (px / prev - 1) <= DATA_EVENT_DROP:
                ev = dict(d=str(asof)[:10], symbol=s, prev=prev, px=px,
                          note='close moved <= -40% in one day: treated as a split/bonus, '
                               'not a loss. Position HELD; verify the price series.')
                st.setdefault('data_events', []).append(ev)
                _alert(f'IPO data event: {s}',
                       f'{s} close {prev:.2f} -> {px:.2f} in one day. Held, not stopped out. '
                       f'Check for a split or bonus and refresh the series.')
                log.append(f'DATA EVENT {s} {prev:.2f}->{px:.2f} held')
                keep.append(p)
                continue
            tr = sma20(close, s, asof)
            why = None
            if px <= p['stop']:
                why = 'STOP'
            elif px >= p['buy'] * (1 + TARGET):
                why = 'TARGET'
            elif tr and px < tr and str(asof)[:10] != p['entry_date']:
                why = 'TRAIL'
            if not why:
                keep.append(p)
                continue
            gross = p['qty'] * (px - p['buy'])
            costs = COST * p['qty'] * (px + p['buy'])
            st['cash'] += p['qty'] * px - COST * p['qty'] * px
            st.setdefault('trades', []).append(dict(
                symbol=s, qty=p['qty'], buy=p['buy'], sell=round(px, 2),
                entry_date=p['entry_date'], exit_date=str(asof)[:10], reason=why,
                net_pnl=round(gross - costs), pnl_pct=round((px / p['buy'] - 1) * 100, 2)))
            log.append(f'EXIT {why} {s} @{px:.2f} ({(px/p["buy"]-1)*100:+.1f}%)')
            _alert(f'IPO EXIT DUE: {s}',
                   f'SELL {s} x{p["qty"]} — {why} at {px:.2f} (entry {p["buy"]}). '
                   f'{"Place it" if st.get("mode") == "live" else "Paper book: no order needed"}.')
        st['positions'] = keep

        # ---- 2. fills from YESTERDAY's pending buy-stops ----
        # LIVE: reconcile against the account. The book records a position only when the
        # broker actually holds it, at the broker's own quantity and average price. It
        # never books a fill it merely hoped for.
        still = []
        if st.get('mode') == 'live':
            held, seen = own_fills()
            if held is None:
                log.append('broker unreachable — pending buy-stops carried, nothing booked')
                _alert('IPO reconcile failed',
                       'Could not read holdings; no fills booked. Pending orders carried.', 'low')
                still = list(st.get('pending', []))
            else:
                owned = {p['symbol'] for p in st['positions']}
                for cand in st.get('pending', []):
                    s = cand['symbol']
                    if s in owned:
                        continue
                    if s not in held:
                        # not in the account: either it never triggered, or it was not
                        # placed. Either way there is nothing to book.
                        st.setdefault('missed', []).append(
                            dict(**cand, why='not held at the broker', d=str(asof)[:10]))
                        log.append(f'NO FILL {s} (not in the account)')
                        continue
                    qty, avg = held[s]
                    cost = qty * avg
                    if cost > float(st['cash']) + 1:
                        log.append(f'{s} held x{qty} @{avg:.2f} costs more than book cash '
                                   f'({st["cash"]:,.0f}) — booking at cash, CHECK THIS')
                        _alert(f'IPO reconcile mismatch: {s}',
                               f'Broker shows x{qty} @{avg:.2f} = Rs {cost:,.0f} but the book '
                               f'only had Rs {st["cash"]:,.0f}. Verify the position is this '
                               f'book\'s and not another.')
                    st['cash'] = max(0.0, float(st['cash']) - cost)
                    st['positions'].append(dict(
                        symbol=s, qty=int(qty), buy=round(float(avg), 2),
                        entry_date=str(asof)[:10], stop=round(float(avg) * (1 - STOP), 2),
                        pivot=cand['pivot'], listed=cand.get('listed'), src='broker'))
                    log.append(f'CONFIRMED {s} x{qty} @{avg:.2f} (from an order this book placed)')
                if seen is not None:
                    json.dump(seen, open(SEEN_ORDERS, 'w'), indent=1, default=str)
            st['pending'] = still
        else:
          for cand in st.get('pending', []):
              s = cand['symbol']
              if s not in close.columns or asof not in wide['open'].index:
                  st.setdefault('missed', []).append(dict(**cand, why='no bar'))
                  continue
              op = wide['open'][s].loc[asof]
              px_today = close[s].loc[asof]
              if not np.isfinite(op) or not np.isfinite(px_today):
                  st.setdefault('missed', []).append(dict(**cand, why='no price'))
                  continue
              if len(st['positions']) >= SLOTS:
                  st.setdefault('missed', []).append(dict(**cand, why='no slot'))
                  continue
              fill = max(float(cand['pivot']), float(op))     # buy-stop AT the pivot
              nav_now = st['cash'] + sum(p['qty'] * p['buy'] for p in st['positions'])
              size = min(SIZE_PCT, 0.30) * nav_now
              qty = int(size / fill)
              if qty < 1 or qty * fill * (1 + COST) > st['cash']:
                  st.setdefault('missed', []).append(dict(**cand, why='cash short'))
                  continue
              st['cash'] -= qty * fill * (1 + COST)
              st['positions'].append(dict(symbol=s, qty=qty, buy=round(fill, 2),
                                          entry_date=str(asof)[:10],
                                          stop=round(fill * (1 - STOP), 2),
                                          pivot=cand['pivot'], listed=cand.get('listed')))
              log.append(f'FILL {s} x{qty} @{fill:.2f} (pivot {cand["pivot"]})')
              _alert(f'IPO ENTRY: {s}',
                     f'BUY {s} x{qty} at {fill:.2f} (buy-stop at pivot {cand["pivot"]}). '
                     f'{"Place it" if st.get("mode") == "live" else "Paper book: no order needed"}.',
                     'low')
        st['pending'] = still

        # ---- 3. scan today for TOMORROW's buy-stops ----
        held = {p['symbol'] for p in st['positions']}
        cands = [c for c in scan(wide, listing, asof) if c['symbol'] not in held]
        free = max(0, SLOTS - len(st['positions']))
        st['pending'] = cands[:free]
        log.append(f'{len(cands)} candidates, {free} slots free, '
                   f'{len(st["pending"])} buy-stops armed for tomorrow')
        if st['pending']:
            _alert('IPO candidates for tomorrow',
                   '; '.join(f'{c["symbol"]} buy-stop {c["pivot"]}' for c in st['pending']), 'low')

        # ---- 4. nav point ----
        tot = sum(p['qty'] * float(close[p['symbol']].loc[:asof].dropna().iloc[-1])
                  for p in st['positions'] if p['symbol'] in close.columns)
        nav = tot + st['cash']
        nc = st.setdefault('nav', [])
        d = str(asof)[:10]
        nc[:] = [x for x in nc if x['d'] != d]
        nc.append(dict(d=d, nav=round(nav)))
        st['last_run'] = str(datetime.now())

        if dry:
            print('\n'.join(log))
            write_ui(st, wide, asof, log, dry=True)
            return
        save_state(st)
        write_ui(st, wide, asof, log, dry=False)
        print('\n'.join(log))
    finally:
        release_lock()


if __name__ == '__main__':
    if '--arm-now' in sys.argv:
        arm()
    else:
        main()
