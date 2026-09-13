"""IPO-Base book — the research/167 re-fitted spec, run forward on real prices.

WHAT THIS IS. A breakout from a base built by a recently listed stock, held on a moving-
average trail. It is the least correlated of the three momentum books (weekly 0.28 to
Open Alpha · Base Age, 0.26 to True North).

THE SPEC CHANGED ON 12-SEP-2026, AND THE REASON MATTERS MORE THAN THE NUMBERS.

research/153 published 31.03% CAGR for this book. That figure was earned on an entry no
order can place: the trigger is a close above the pivot and the fill was booked at the
SAME day's open, which is already in the past by the time the close exists. research/158
and research/159 found that defect across three books in this project; research/167
re-measured this one on the entry the book actually uses — next day, buy-stop at the
pivot — and the adopted rules returned 14.90% after tax, −38.6% drawdown.

Worse than weak: at those parameters the rules were INDISTINGUISHABLE FROM CHANCE. Against
a date-matched random-entry control — same days, same number of entries, names drawn at
random from the same young-and-liquid universe, the same fill convention on both arms —
the real rules won 14 of 30 paired runs. There was no edge in picking the breakout.

One dial fixed it, and only one: THE TRAIL. Real minus control, in points of CAGR, at
trail 10 / 15 / 20 / 30 / 40 / 50 / 60 / 75 / 100 bars:

    +0.04  -0.71  -0.10  +1.26  +3.08  +4.78  +2.05  +0.43  +0.31

Zero or negative across the whole region the old spec sat in (<= 20), unanimous 30-of-30
across the entire 30-75 band, the same shape at a second independent stop value. A 25-bar
base breakout in a young stock is a real edge only if it is given room to run. The old
20-bar trail sold every winner back into the noise that produced it.

CORRECTED 13-SEP-2026 (research/169). That random-control comparison ran on a panel whose
rolling windows had gaps, which quietly handicapped the random arm. On a gap-free panel the
edge at trail 50 is +2.25pp over the whole period: it holds for 2006-2015 and ties random
young names in 2016-2026. The return is real and the blend value stands, because
research/168's curves reproduce exactly; what is weaker than first stated is the claim that
the breakout itself picks better than drawing young names at random.

So three dials moved: trail 20 -> 50, stop 8% -> 10%, and a NEW market gate. Everything
else research/153 found was re-confirmed on the honest entry and is unchanged: the base
geometry tops a 128-cell grid, and 8 slots at 18.75% beats 5x20%, 10x10% and 16x6.25%.

After: 21.80% CAGR after tax, -26.6% median drawdown, Calmar 0.819, and it beats the
random control on 30 of 30 runs. Read the headline down to 19-22%: ~350 cells were scored
to find it, and the worst of the 30 runs drew -32.9%.

THE GATE IS INSURANCE WITH NO PREMIUM. No new entries while NIFTYBEES closes below its
150-day average; positions already held are untouched and keep their own exits. It is a
coin flip on return (14 of 30 runs) that removes 17.8 points of drawdown on 30 of 30.
Bounded at both ends: the 200-day average agrees, the 100-day average fails, which is what
a real effect looks like rather than a fitted one.

WHAT THIS BOOK MUST NOT BE SIZED PAST. At Rs 10L the MEDIAN position is 1.56% of the name's
own 20-day median traded value and the 90th percentile is 9.05% (research/167 labelled the
median as the 90th percentile; research/169 caught it). At Rs 1cr the 90th percentile is
~90% of a day's volume. This
sleeve cannot exceed roughly Rs 20-25L, ever, and that is a property of the universe
rather than of the rules.

THE ONE HONEST BLACK MARK. In 2008 the re-fit loses 10.3% where the old spec made +0.4%.
The fast 20-bar trail that costs seven points a year in normal times is exactly what
sidestepped that crash, and the gate recovers only part of it. If this book is ever relied
on for a crash cushion, it is the wrong instrument.

PAPER UNTIL ARMED (Arun, 06-Sep-2026). The book runs on a notional Rs 10,00,000 until a
real deposit is routed to it through the Capital Desk, which flips `ipo_status` to
'live' in backtest_data/allocation_targets.json. From then on the same signals are
real-money instructions. Execution is manual-assisted either way: like Open Alpha this
book has no executor, so it alerts with the exact order and Arun places it.

THE SPEC (research/167 "Spec A"; what changed on 12-Sep-2026 is marked):
  universe    NSE equities with a VETTED listing date, ETFs excluded, all rows before
              the listing date masked
  age band    listed <= 6 months ago AND >= 25 bars at the signal date. The book ran 60
              from 6 to 13-Sep-2026 on a misreading of the harness; see MIN_BARS
  liquidity   20-day median traded value >= Rs 5 cr at t-1
  base        last 25 bars; pivot = highest CLOSE, shifted 1; depth (pivot to lowest
              low) <= 30%; and close[t-1] < pivot, so it is not already extended
  RS          OFF. r/153 section 3: a strict RS >= 70 yields ZERO signals in this age
              band, because a 252-day relative-strength score does not exist for a
              stock that has traded for four months
  gate        CHANGED: no NEW entries while NIFTYBEES closes below its 150-day average.
              Held positions are untouched and keep their own exits. The gate is read
              from the close that has already happened, never from a forming bar, so it
              is evaluated when buy-stops are armed for the next session — which is the
              same day's data the study's gate used
  trigger     close[t] > pivot
  fill        next day, buy-stop AT the pivot, filled max(pivot, open), AND ONLY IF the
              day's HIGH reached the pivot. That last clause was missing before 12-Sep-
              2026 and is the buy-stop's whole meaning: an order resting above the market
              that the market never reaches does not fill
  exits       priority order: stop (close <= fill x 0.90, CHANGED from 0.92) ->
              target (close >= fill x 1.25, unchanged) -> trail (close < SMA-50, CHANGED
              from SMA-20, never on the entry bar)
  book        8 slots at 18.75% of equity, 25 bps per side (both re-confirmed on the
              honest entry: 8 slots beat 5, 10 and 16)

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
  --migrate       bring stored state onto the current spec without scanning. Needed when
                  a spec ships outside a trading day, so held positions are not left
                  running the previous stop until the next session
  --gate          print the gate and exit. Read-only, safe any time

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
# Run as a script, Python puts services/ on the path, not the repo root -
# so `import services.x` fails and every alert delivery died silently in a
# try/except (found 09-Sep-2026 in the KTKBANK exit log).
sys.path.insert(0, str(ROOT))
from services import cash_park  # idle cash parked in CASHIETF (13-Sep-2026)
DB = ROOT / 'backtest_data' / 'market_data.db'
STATE = ROOT / 'backtest_data' / 'ipo_paper_state.json'
LOCK = ROOT / 'backtest_data' / 'ipo_paper_state.lock'
UI_JSON = ROOT / 'static' / 'app' / 'ipo_paper.json'
ALLOC = ROOT / 'backtest_data' / 'allocation_targets.json'
LISTINGS = ROOT / 'research' / '153_ipo_base' / 'results' / 'listing_dates.csv'
# A real feed the pages can read. The old path was another job's cron log.
FEED = ROOT / 'backtest_data' / 'book_alerts.jsonl'

# RESTORED 12-Sep-2026. Both of these were deleted by commit 3829ad71 ("IPO Base marks
# intraday"), which rewrote this constants block and dropped two lines. They are only ever
# read inside the `mode == 'live'` branch, so nothing failed while the book was on paper —
# and then the sleeve was funded on 08-Sep and EVERY nightly cycle and every reconcile
# crashed on a NameError from that moment. Four sessions of a real-money book with no exit
# evaluation and no buy-stops armed. Recovered verbatim from 3bc0b4f4:services/ipo_paper.py.
#
# IPO_TAG is how this book claims its own orders out of a shared account: matching by
# symbol handed Open Alpha 481 shares of a stock it held 9 of on 08-Sep. The tag must keep
# this exact value or the already-applied KISSHT order stops being recognised.
IPO_TAG = 'IPO-ENTRY'
# Exits are placed by this book since 13-Sep-2026 (before: an alert saying "Place it").
IPO_EXIT_TAG = 'IPO-EXIT'
SEEN_ORDERS = ROOT / 'backtest_data' / 'ipo_applied_orders.json'

CAPITAL = 1_000_000          # notional while on paper
SLOTS = 8
SIZE_PCT = 0.1875

# research/167 Spec A. The two numbers that moved, and by how much it mattered:
#   STOP       0.08 -> 0.10   worth about +0.8pp of CAGR; 10% is the centre of a
#                             plateau (6 < 8 < 10 ~ 15 > none), 8% was one notch tight
#   TRAIL_SMA  20   -> 50     worth about +7pp of CAGR, and it is the ONLY dial that
#                             separates this book from random stock selection. At trail
#                             <= 20 the rules beat a date-matched random control on 14 of
#                             30 paired runs; at 50, on 30 of 30. A hump, not a ramp: it
#                             peaks at 50 and rolls back off by 100-150, so do not read
#                             "longer is better" from this and lengthen it again.
# SPEC_VERSION exists so state written under the old dials can be migrated exactly once.
STOP = 0.10                  # close <= fill * 0.90
TARGET = 0.25                # close >= fill * 1.25
TRAIL_SMA = 50               # close < SMA-50, entry bar exempt
SPEC_VERSION = 'r167-A-mb25'
SPEC_NOTES = {
    'r167-A': ('research/167 Spec A: trail SMA-20 -> SMA-50, stop 8%% -> 10%%, new '
               'NIFTYBEES < SMA-150 entry gate, and the buy-stop now requires the day high '
               'to have reached the pivot. Stops re-based on %d open position(s).'),
    'r167-A-mb25': ('research/169: MIN_BARS 60 -> 25, the floor Spec A was validated at. '
                    'The 60 rested on reading a row-count-today filter as a bars-at-signal '
                    'rule. No exit rule changed; %d stop(s) re-based.'),
}

# The market gate. NEW on 12-Sep-2026, and it gates ENTRIES ONLY — nothing about a held
# position changes when it comes on. NIFTYBEES is the NIFTY 50 ETF and stands in for the
# index here because it is the series this project already has clean daily history for.
GATE_SYMBOL = 'NIFTYBEES'
GATE_SMA = 150
BASE_L = 25                  # base window, in bars
MAX_AGE_M = 6

# MIN_BARS = 25, the floor Spec A was VALIDATED at. Corrected 13-Sep-2026 (research/169).
#
# From 6-Sep to 13-Sep-2026 this book ran 60, on the reasoning that research/153's harness
# admitted only symbols with 60+ bars. That was a misreading. The loader's
#   "... group by symbol) where n >= 60"
# counts a symbol's rows over the WHOLE database TODAY, so a name with 2,000 rows today is
# scanned from its 25th bar at the signal date. The bar test that decides a signal is
# `ctx.BARS >= min_bars`, evaluated on the day, and research/167 ran it at 25. The 6-Sep
# reconciliation saw 75% agreement because it only looked at the last ~60 sessions, which is
# exactly the window the row-count filter hides.
#
# What the misreading cost, on research/167's own engine and panel, 30 seeds, after tax:
#   min bars 25   21.80% CAGR, -26.6% drawdown, beats its random control on 30 of 30
#   min bars 40   17.87%,      -32.4%,          loses to it on 27 of 30
#   min bars 60   11.57%,      -39.0%
# So the return is concentrated in a stock's first 25-40 sessions after listing. That is
# also where the book buys its thinnest names: at today's sleeve size a position is well
# under 1% of those names' daily traded value, but re-check it as the sleeve grows.
#
# One interaction to know: the SMA-50 trail needs 50 closes, so a position entered at bar
# 25 carries no trail exit until bar 50 - only its stop and target. research/167's engine
# behaves identically (a 50-bar rolling mean is undefined until then), so this is the
# validated behaviour, not a gap.
MIN_BARS = 25
MAX_DEPTH = 0.30
TV_FLOOR = 5e7               # Rs 5 cr, 20-day median traded value
COST = 0.0025
DATA_EVENT_DROP = -0.40      # single-day close move below this is a split, not a loss

ETF_PAT = ('BEES', 'ETF', 'IETF', 'GOLD', 'SILVER', 'LIQUID', 'GSEC', 'SDL', 'INAV')


def ist_now():
    return datetime.utcnow() + timedelta(hours=5, minutes=30)


def _alert(title, body, urgency='critical'):
    """Record the alert, and for a critical one actually send it.

    The feed is a real file the pages read. `/tmp/nas_alert_feed.log`, which this used to
    write to, is the cron output log of a different job and is read by nobody -- an exit
    alert written there reached no one at all (found 08-Sep-2026, SPORTKING).
    """
    line = dict(ts=str(datetime.now()), book='IPO-BASE', urgency=urgency,
                title=title, body=body)
    try:
        with open(FEED, 'a') as f:
            f.write(json.dumps(line) + '\n')
    except Exception as e:
        print('alert feed write failed:', e)
    if urgency != 'critical':
        return
    try:
        from services.dividend_notify import send_email, send_push
        print('  email:', send_email(title, '<pre>%s</pre>' % body))
        print('  push:', send_push(title, body))
    except Exception as e:
        # never let a notification failure break the run that produced the signal
        print('alert delivery failed:', e)

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


def migrate_spec(st, log=None):
    """Bring positions opened under the old dials onto the new ones. Idempotent.

    A stop is stored on the position row, at entry, as a price. So a position bought
    before 12-Sep-2026 carries a stop computed at fill x 0.92 and would keep running the
    OLD rule for its whole life unless it is re-based. The trail needs no migration — it
    is recomputed from TRAIL_SMA on every pass — but the stop does.

    This LOOSENS the stop on anything already held (0.92 -> 0.90 of the fill). That is the
    intended direction: research/167 found 8% one notch too tight, with 10% at the centre
    of the plateau, and a book half on one rule and half on the other is neither.

    The prior stop is kept on the row as `stop_prev` so the change is auditable rather than
    silently overwritten.
    """
    if st.get('spec_version') == SPEC_VERSION:
        return False
    moved = []
    for p in st.get('positions', []):
        want = round(float(p['buy']) * (1 - STOP), 2)
        have = round(float(p.get('stop', 0) or 0), 2)
        if abs(want - have) >= 0.01:
            p['stop_prev'] = have
            p['stop'] = want
            p['stop_rebased_on'] = str(date.today())
            moved.append('%s %.2f -> %.2f' % (p['symbol'], have, want))
    st['spec_version'] = SPEC_VERSION
    st.setdefault('spec_history', []).append(dict(
        on=str(date.today()), to=SPEC_VERSION,
        note=SPEC_NOTES[SPEC_VERSION] % len(moved),
        rebased=moved))
    msg = ('SPEC MIGRATED to %s; stops re-based: %s'
           % (SPEC_VERSION, '; '.join(moved) if moved else 'none needed'))
    if log is not None:
        log.append(msg)
    else:
        print(msg)
    return True


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
    closes, opens, highs, lows, tv = {}, {}, {}, {}, {}
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
        # highs: needed to answer whether a resting buy-stop was ever actually touched.
        # The column was always read from the DB and then thrown away, which is how the
        # never-triggered fills got booked (see the fill branch).
        highs[s] = df['high']
        lows[s] = df['low']
        tv[s] = (df['close'] * df['volume']).rolling(20).median()
    conn.close()
    if not closes:
        return None
    return dict(close=pd.DataFrame(closes).sort_index(),
                open=pd.DataFrame(opens).sort_index(),
                high=pd.DataFrame(highs).sort_index(),
                low=pd.DataFrame(lows).sort_index(),
                tv=pd.DataFrame(tv).sort_index()), listing


def market_gate(asof=None):
    """Is the market weak enough to stop taking NEW positions?

    NIFTYBEES closing below its 150-day average. Returns (blocked, detail) where detail
    carries the two numbers so the page and the log can show WHY, not just that it fired.

    Read from closes that have already happened. This is called when buy-stops are armed
    for the next session, so the value used is the previous close relative to the fill —
    exactly the one-day shift the study's gate series used. Nothing here ever looks at a
    forming bar.

    Fails OPEN, deliberately: if the index series is missing or too short the book keeps
    trading. A gate that silently halts a real-money book on a data outage is a worse
    failure than one that misses a signal, because the first is invisible.
    """
    import sqlite3
    con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    try:
        rows = [(r[0], r[1]) for r in con.execute(
            "select date, close from market_data_unified where symbol=? and "
            "timeframe='day' and close > 0 order by date desc limit ?",
            (GATE_SYMBOL, GATE_SMA + 10))]
    finally:
        con.close()
    if asof is not None:
        cut = str(pd.Timestamp(asof))[:10]
        rows = [r for r in rows if str(r[0])[:10] <= cut]
    if len(rows) < GATE_SMA:
        return False, dict(ok=False, why='%s has only %d of the %d closes the gate needs '
                                         '- gate OPEN' % (GATE_SYMBOL, len(rows), GATE_SMA))
    closes = [r[1] for r in rows[:GATE_SMA]]
    sma = sum(closes) / GATE_SMA
    last = closes[0]
    blocked = last < sma
    return blocked, dict(ok=True, symbol=GATE_SYMBOL, n=GATE_SMA,
                         asof=str(rows[0][0])[:10], close=round(last, 2),
                         sma=round(sma, 2), above_pct=round(100 * (last / sma - 1), 2),
                         blocked=bool(blocked))


# near_pct is DISPLAY ONLY. It widens the watchlist the page shows; it can never add
# a row the book arms, because arming reads `triggered` and nothing else. 15% was
# chosen because at 10% the watchlist was one name.
def scan(wide, listing, asof, include_near=False, near_pct=15.0):
    """Names whose close TODAY breaks the base pivot — and optionally the ones that nearly
    did.

    Every row satisfies the whole spec except, for a watchlist row, the trigger itself:
    listed inside the age band, at least MIN_BARS of history, base depth <= 30%, not
    already extended at yesterday's close, and 20-day median traded value >= TV_FLOOR. So
    a watchlist name needs exactly one thing to become an order, a close above its pivot,
    and nothing else about it has to be re-checked.

    `triggered=True` rows are the ones the book arms. `include_near` adds rows whose close
    is within `near_pct` of the pivot, for the page — they are NOT candidates for an order
    and must never be armed.
    """
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
        liq = tvp[s].iloc[i - 1] if i >= 1 else np.nan
        if not np.isfinite(liq) or liq < TV_FLOOR:
            continue
        triggered = bool(tc > pv)
        if not triggered and not include_near:
            continue
        # gap_pct: how far today's close sits BELOW the pivot. 0 or negative means it has
        # already broken out. This is the only number that says how close a watchlist name
        # actually is, so it is what the watchlist sorts on.
        gap = (pv / tc - 1) * 100 if tc > 0 else None
        if not triggered and (gap is None or gap > near_pct):
            continue
        out.append(dict(symbol=s, pivot=round(float(pv), 2), close=round(float(tc), 2),
                        depth_pct=round(float(depth) * 100, 1), tv=float(liq),
                        listed=str(listing[s].date()),
                        age_days=int((asof - listing[s]).days),
                        triggered=triggered,
                        gap_pct=round(float(gap), 2) if gap is not None else None))
    # PRE-REGISTERED TIEBREAK — deterministic, capacity-friendly. See the module header.
    # Triggered names first, then by traded value; within the untriggered watchlist, the
    # nearest to its pivot first, because that is the one that could fire tomorrow.
    out.sort(key=lambda r: (not r['triggered'],
                            r['gap_pct'] if not r['triggered'] else -r['tv']))
    return out


def sma_trail(close, sym, upto):
    """Where the trail sits: the mean of the last TRAIL_SMA closes up to and including
    `upto`. Returns None rather than a short-window average — a 50-bar trail computed off
    30 bars is a different, tighter rule, and a young listing often has only 30."""
    s = close[sym].loc[:upto].dropna()
    return float(s.iloc[-TRAIL_SMA:].mean()) if len(s) >= TRAIL_SMA else None


# ───────────────────────── UI ─────────────────────────
def spec_block():
    """The rules, in the page's own words. Generated from the constants rather than typed,
    so the page can never describe a spec the engine is not running."""
    return dict(
        version=SPEC_VERSION,
        study='research/167_ipo_base_honest_reopt',
        changed='2026-09-12',
        entry='Close above the highest close of the last %d bars, in a stock listed '
              'within %d months with at least %d bars of history, base no deeper than '
              '%d%%, and 20-day median traded value of at least Rs %.0f cr. Next session, '
              'a buy-stop at that pivot.' % (BASE_L, MAX_AGE_M, MIN_BARS,
                                             int(MAX_DEPTH * 100), TV_FLOOR / 1e7),
        gate='No new entries while %s closes below its %d-day average. Positions already '
             'held are untouched.' % (GATE_SYMBOL, GATE_SMA),
        exits='Stop at %d%% below the fill, target %d%% above it, otherwise trail out on a '
              'close below the %d-day average. Evaluated on closes, in that order.'
              % (int(STOP * 100), int(TARGET * 100), TRAIL_SMA),
        book='%d slots at %.2f%% of equity, %d bps a side.'
             % (SLOTS, SIZE_PCT * 100, int(COST * 10000)),
        what_changed='Trail %s, stop %s, and the gate is new. The old dials were fitted '
                     'against a fill no order could place; measured honestly they beat a '
                     'random-entry control on only 14 of 30 runs, and the re-fit beats it '
                     'on 30 of 30.' % ('20 -> 50 bars', '8% -> 10%'),
        capacity='Do not size this sleeve past about Rs 20-25L: at Rs 1cr a typical '
                 'position would be most of a day volume in these names.')


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
        tr = sma_trail(close, p['symbol'], asof) if close is not None and p['symbol'] in close.columns else None
        rows.append(dict(**p, ltp=round(lp, 2), value=round(val), pnl=round(pnl),
                         pnl_pct=round((lp / p['buy'] - 1) * 100, 2),
                         trail=round(tr, 2) if tr else None,
                         target=round(p['buy'] * (1 + TARGET), 2),
                         to_stop_pct=round((lp / p['stop'] - 1) * 100, 1),
                         to_trail_pct=round((lp / tr - 1) * 100, 1) if tr else None,
                         days=(pd.Timestamp(asof) - pd.Timestamp(p['entry_date'])).days))
    cash = float(st['cash'])
    # cash includes money parked in CASHIETF at cost; the gain on it is the only new term
    nav = tot_val + cash + cash_park.gain(st)
    for r in rows:
        r['weight'] = round(100 * r['value'] / nav, 1) if nav else 0
    realized = sum(t.get('net_pnl', 0) for t in st.get('trades', []))
    cap = float(st['capital'])
    ui = dict(updated=str(datetime.now()), asof=str(asof)[:10], mode=st.get('mode', 'paper'),
              positions=rows, capital=round(cap), cash=round(cash), value=round(tot_val),
              nav=round(nav), pnl=round(tot_pnl), realized=round(realized),
              # nav already contains realised P&L (see oa_real.py, 09-Sep-2026)
              gain=round(nav - cap),
              # nav already contains realised P&L - the same double-count fixed in
              # `gain` on 09-Sep-2026, one line further down than I looked
              return_pct=round(100 * (nav - cap) / cap, 2) if cap else 0,
              invested_pct=round(100 * tot_val / nav, 1) if nav else 0,
              slots=SLOTS, slots_used=len(rows),
              pending=st.get('pending', []), navcurve=st.get('nav', []),
              trades=st.get('trades', [])[-100:], data_events=st.get('data_events', [])[-20:],
              started=st.get('started'), log=log,
              failed_orders=st.get('failed_orders', []),
              # the re-fitted spec, carried onto the page so the rules it is judged
              # against are visible beside the numbers rather than only in this file
              spec=spec_block(), gate=st.get('gate'),
              candidates=st.get('candidates', []), watchlist=st.get('watchlist', []),
              missed=st.get('missed', [])[-40:],
              park=cash_park.ui_block(st))
    if dry:
        print(json.dumps({k: ui[k] for k in ('asof', 'mode', 'nav', 'cash', 'slots_used')}, indent=1))
        return ui
    tmp = UI_JSON.with_suffix('.json.tmp')
    json.dump(ui, open(tmp, 'w'), indent=1, default=str)
    os.replace(tmp, UI_JSON)
    return ui


# ───────────────────────── engine ─────────────────────────
def own_sell_fills(seen):
    """{symbol: (qty, avg)} for THIS BOOK'S completed exit sells not yet applied.

    Same discipline as own_fills(): the order book, this book's tag only, never holdings, and
    `seen` so an order is applied once. Returns None when the broker is unreachable."""
    try:
        orders = _kite().orders()
    except Exception as e:
        print('order book unreachable for exit fills:', e)
        return None
    out = {}
    for o in orders:
        oid = str(o.get('order_id'))
        if (o.get('status') != 'COMPLETE' or o.get('transaction_type') != 'SELL'
                or o.get('product') != 'CNC' or not o.get('filled_quantity')):
            continue
        if (o.get('tag') or '') != IPO_EXIT_TAG or oid in seen:
            continue
        s = o['tradingsymbol']
        q, px = int(o['filled_quantity']), float(o['average_price'])
        pq, pv = out.get(s, (0, 0.0))
        out[s] = (pq + q, pv + q * px)
        seen[oid] = dict(ts=str(datetime.now()), symbol=s, qty=q, price=px, side='SELL')
    return {s: (q, v / q) for s, (q, v) in out.items()}


def apply_sell_fills(st, sells, today):
    """Book broker-confirmed exit sells onto state. -> list of human-readable lines.

    The sale is recorded at the BROKER'S price, with the signal close it was decided on and the
    slippage against it, because the study sells at that close and the soak has to see the gap."""
    booked = []
    for s, (q, avg) in (sells or {}).items():
        p = next((x for x in st['positions'] if x['symbol'] == s), None)
        if p is None:
            _alert('IPO exit fill for a name the book does not hold: %s' % s,
                   'A tagged IPO-EXIT SELL for %s x%d filled at %.2f, but the book has no such '
                   'position. Nothing applied - check whether it was already booked by hand.'
                   % (s, q, avg))
            continue
        q = min(int(q), int(p['qty']))
        ed = p.get('exit_due') or {}
        proceeds = q * avg
        st['cash'] = round(float(st['cash']) + proceeds - COST * proceeds, 2)
        gross = q * (avg - p['buy'])
        costs = COST * q * (avg + p['buy'])
        sc = ed.get('signal_close')
        st.setdefault('trades', []).append(dict(
            symbol=s, qty=q, buy=p['buy'], sell=round(avg, 2), entry_date=p['entry_date'],
            exit_date=str(today)[:10], reason=ed.get('reason', 'EXIT'), signal_date=ed.get('date'),
            signal_close=sc, slip_vs_close_pct=round((avg / sc - 1) * 100, 2) if sc else None,
            net_pnl=round(gross - costs), pnl_pct=round((avg / p['buy'] - 1) * 100, 2)))
        if q >= int(p['qty']):
            st['positions'] = [x for x in st['positions'] if x['symbol'] != s]
        else:
            p['qty'] = int(p['qty']) - q           # partial: the rest stays exiting
        booked.append('%s x%d @%.2f (%s, signal close %s)' % (s, q, avg, ed.get('reason', 'EXIT'), sc))
    return booked


def record_exit(st, p, why, px, asof, log):
    """A stop, target or trail fired on the close.

    LIVE: mark it due and KEEP the position until the broker fills the sale -> True.
    PAPER: book the sale at the signal close, as the backtest does -> False."""
    s = p['symbol']
    if st.get('mode') == 'live':
        p['exit_due'] = dict(reason=why, signal_close=round(float(px), 2),
                             date=str(asof)[:10], orders=[])
        log.append(f'EXIT DUE {why} {s} @{px:.2f} ({(px/p["buy"]-1)*100:+.1f}%) - sell for the next open')
        return True
    gross = p['qty'] * (px - p['buy'])
    costs = COST * p['qty'] * (px + p['buy'])
    st['cash'] += p['qty'] * px - COST * p['qty'] * px
    st.setdefault('trades', []).append(dict(
        symbol=s, qty=p['qty'], buy=p['buy'], sell=round(px, 2),
        entry_date=p['entry_date'], exit_date=str(asof)[:10], reason=why,
        net_pnl=round(gross - costs), pnl_pct=round((px / p['buy'] - 1) * 100, 2)))
    log.append(f'EXIT {why} {s} @{px:.2f} ({(px/p["buy"]-1)*100:+.1f}%)')
    _alert(f'IPO EXIT (paper): {s}',
           f'SELL {s} x{p["qty"]} - {why} at {px:.2f} (entry {p["buy"]}). Paper book: no order needed.',
           'low')
    return False


def place_exit_orders(st, log, kite=None, placer=None):
    """Place one next-open SELL per exiting position that has none placed today. -> placed lines.

    Uses Open Alpha's place_exit_amo: MARKET AMO first, LIMIT 2% under the signal close if the
    broker refuses MARKET, and never a second SELL while one is live. A refusal is CRITICAL and is
    retried at the next evening run - an exit that silently does not go is the one failure a
    book cannot have."""
    due = [p for p in st.get('positions', []) if p.get('exit_due')]
    if not due:
        return []
    today = str(date.today())
    if placer is None:
        from services.oa_real import place_exit_amo as placer
    if kite is None:
        try:
            kite = _kite()
        except Exception as e:
            _alert('IPO EXIT NOT PLACED - broker unreachable',
                   'Exits due: %s. Sell by hand, or they are retried at the next evening run. (%s)'
                   % (', '.join('%s x%d' % (p['symbol'], p['qty']) for p in due), e))
            return []
    ticks = {}
    try:
        from services import equity_executor as ex
        ex.load_ticks(kite)
        ticks = ex._TICKS
    except Exception as e:
        print('tick sizes unavailable (%s) - 0.05 fallback' % e)
    placed = []
    for p in due:
        s, ed = p['symbol'], p['exit_due']
        if any(o.get('d') == today for o in ed.get('orders', [])):
            continue
        oid, kind, err = placer(kite, s, int(p['qty']), float(ed['signal_close']),
                                tick=ticks.get(s, 0.05), tag=IPO_EXIT_TAG)
        if oid:
            ed.setdefault('orders', []).append(dict(d=today, ts=str(datetime.now())[:19],
                                                    order_id=str(oid), kind=kind))
            line = '%s x%d %s (%s on the close %.2f)' % (s, p['qty'], kind, ed['reason'], ed['signal_close'])
            placed.append(line)
            log.append('EXIT PLACED ' + line)
            _alert('IPO EXIT PLACED: %s' % s,
                   'SELL %s x%d at the next open (%s) - %s fired on the close %.2f. Nothing to do.'
                   % (s, p['qty'], kind, ed['reason'], ed['signal_close']), 'low')
        elif err and 'already live' in err:
            log.append('EXIT %s: a sell is already resting' % s)
        else:
            log.append('EXIT FAILED TO PLACE %s: %s' % (s, err))
            _alert('IPO EXIT FAILED TO PLACE: %s' % s,
                   'SELL %s x%d (%s on the close %.2f) was refused: %s. Sell by hand, or it is '
                   'retried at the next evening run.' % (s, p['qty'], ed['reason'], ed['signal_close'], err))
    return placed


def free_slots(st):
    """Slots open for arming. An exiting position counts as FREE: the backtest frees the slot at
    the exit close, so the next open's entry may use it."""
    return max(0, SLOTS - len([p for p in st.get('positions', []) if not p.get('exit_due')]))


def reconcile_now():
    """Apply this book's completed orders to state and rebake the page.

    The nightly cycle refuses to run before 15:35 so it never scans a half-formed bar,
    and arming deliberately does not scan either. That left a gap: after the sleeve went
    live and its first order filled, the page still showed the previous evening's bake -
    PAPER, a notional ten lakh, and KISSHT sitting as an unfilled buy-stop it had in fact
    already bought. Reconciling is not scanning, so it can run at any hour.
    """
    fills, seen = own_fills()
    if fills is None:
        print('order book unreachable; nothing applied')
        return
    sells = own_sell_fills(seen) or {}
    if not acquire_lock():
        print('book busy')
        return
    try:
        st = load_state()
        booked = []
        by_sym = {p['symbol']: p for p in st['positions']}
        for s, (qty, avg) in fills.items():
            cost = qty * avg
            if cost > float(st['cash']) + 1:
                _alert('IPO reconcile refused: %s' % s,
                       'Broker shows x%d at %.2f costing Rs %s, but the book holds only '
                       'Rs %s. Nothing applied for this name.'
                       % (qty, avg, format(round(cost), ','), format(round(st['cash']), ',')))
                print('REFUSED %s: costs more than the book holds' % s)
                continue
            if s in by_sym:
                p0 = by_sym[s]
                nq = p0['qty'] + qty
                p0['buy'] = round((p0['qty'] * p0['buy'] + cost) / nq, 2)
                p0['qty'] = nq
                p0['stop'] = round(p0['buy'] * (1 - STOP), 2)
            else:
                st['positions'].append(dict(
                    symbol=s, qty=int(qty), buy=round(avg, 2),
                    entry_date=str(date.today()), stop=round(avg * (1 - STOP), 2),
                    pivot=next((c['pivot'] for c in st.get('pending', [])
                                if c['symbol'] == s), None), src='executor'))
            st['cash'] = round(max(0.0, float(st['cash']) - cost), 2)
            st['pending'] = [c for c in st.get('pending', []) if c['symbol'] != s]
            booked.append('%s x%d @%.2f' % (s, qty, avg))
        booked += ['EXIT ' + x for x in apply_sell_fills(st, sells, date.today())]
        if booked:
            save_state(st)
            json.dump(seen, open(SEEN_ORDERS, 'w'), indent=1, default=str)
    finally:
        release_lock()
    print('booked: ' + ('; '.join(booked) if booked else 'nothing new'))
    loaded = load_wide()
    if loaded:
        wide, _l = loaded
        write_ui(load_state(), wide, wide['close'].index[-1],
                 ['reconciled ' + str(datetime.now())[:19]], dry=False)
        print('page refreshed')


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

        # ---- FUNDING SYNC (13-Sep-2026) ----
        # Money moved through the Capital Desk while the book is ALREADY live. Before this,
        # capital and cash were copied from `ipo_funded` only on a paper->live switch, so a
        # later deposit never reached the cash that sizes buys.
        if mode == 'live' and abs(cap - float(st['capital'])) >= 1.0:
            delta = cap - float(st['capital'])
            # FREE cash: money parked in CASHIETF cannot be paid out until it is sold
            if cash_park.free_cash(st) + delta < -1.0:
                _alert('IPO funding NOT applied',
                       'The Capital Desk shows Rs %s funded against the book capital Rs %s. '
                       'Applying Rs %s would take cash to Rs %s, below zero, so nothing was '
                       'changed. A withdrawal is never funded by selling positions.'
                       % (format(round(cap), ','), format(round(st['capital']), ','),
                          format(round(delta), ','),
                          format(round(float(st['cash']) + delta), ',')))
                print('funding sync refused: cash would go negative')
            else:
                st['capital'] = cap
                st['cash'] = float(st['cash']) + delta
                st.setdefault('fund_flows', []).append(dict(
                    ts=str(datetime.now())[:19], kind='deposit' if delta > 0 else 'withdraw',
                    amount=round(abs(delta), 2), via='capital desk',
                    capital_after=round(cap, 2), cash_after=round(float(st['cash']), 2)))
                _alert('IPO funding applied',
                       '%s Rs %s. Capital now Rs %s, cash Rs %s; new buys size off it.'
                       % ('Deposit' if delta > 0 else 'Withdrawal',
                          format(round(abs(delta)), ','), format(round(cap), ','),
                          format(round(float(st['cash'])), ',')), 'low')
                print('funding sync: %+.0f applied, capital %.0f cash %.0f'
                      % (delta, cap, float(st['cash'])))

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

        # ---- 0. spec migration, BEFORE any exit is evaluated ----
        # Belt and braces: the migration is run once at deploy, but state can be restored
        # from a backup written under the old dials, and an exit must never be tested
        # against a stop the book no longer runs.
        migrate_spec(st, log)

        # ---- 1. exits, on today's close ----
        keep = []
        for p in st['positions']:
            s = p['symbol']
            if p.get('exit_due'):                     # already exiting: its sell is handled below
                keep.append(p)
                continue
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
            tr = sma_trail(close, s, asof)
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
            if record_exit(st, p, why, px, asof, log):
                keep.append(p)
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
                for line in apply_sell_fills(st, own_sell_fills(seen) or {}, asof):
                    log.append('EXIT FILLED ' + line)
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
              hi = wide['high'][s].loc[asof] if s in wide['high'].columns else np.nan
              if not np.isfinite(op) or not np.isfinite(px_today):
                  st.setdefault('missed', []).append(dict(**cand, why='no price'))
                  continue
              # DID THE BUY-STOP ACTUALLY TRIGGER? Before 12-Sep-2026 this branch booked
              # a fill at max(pivot, open) whether or not the market ever reached the
              # pivot, so orders that never executed entered the record as positions.
              # research/167 measured it at about 1.5% of signals, every one of them
              # flattering. A buy-stop resting above the market fills only if the market
              # trades there, and the day's high is the evidence that it did.
              if not np.isfinite(hi) or float(hi) < float(cand['pivot']):
                  st.setdefault('missed', []).append(dict(
                      **cand, why='never reached the pivot',
                      day_high=round(float(hi), 2) if np.isfinite(hi) else None))
                  log.append('NO FILL %s - high %s never reached the buy-stop at %s'
                             % (s, round(float(hi), 2) if np.isfinite(hi) else 'n/a',
                                cand['pivot']))
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

        # ---- 2b. place tomorrow-open SELLs for every exit that is due (LIVE) ----
        if st.get('mode') == 'live':
            if dry:
                for p in st['positions']:
                    if p.get('exit_due'):
                        log.append('DRY: would place SELL %s x%d for the next open' % (p['symbol'], p['qty']))
            else:
                place_exit_orders(st, log)

        # ---- 3. scan today for TOMORROW's buy-stops ----
        held = {p['symbol'] for p in st['positions']}
        rows = [c for c in scan(wide, listing, asof, include_near=True)
                if c['symbol'] not in held]
        cands = [c for c in rows if c['triggered']]
        watch = [c for c in rows if not c['triggered']]
        free = free_slots(st)                        # exiting positions free their slot

        # THE GATE, applied here and nowhere else: it blocks NEW entries only, and the
        # place a new entry is created is the arming of a buy-stop. Held positions run
        # their own stop, target and trail untouched — that is the rule as tested, and it
        # is also why the gate costs nothing in return while removing drawdown.
        blocked, gate = market_gate(asof)
        st['gate'] = gate
        if blocked:
            st['pending'] = []
            log.append('GATE ON - %s %s is below its %d-day average %s, so NO buy-stops '
                       'are armed (%d would have qualified). Held positions unaffected.'
                       % (GATE_SYMBOL, gate.get('close'), GATE_SMA, gate.get('sma'),
                          len(cands)))
            if cands:
                _alert('IPO gate ON - %d candidates NOT armed' % len(cands),
                       '%s closed %s against its %d-day average %s, so no new entries. '
                       'Skipped: %s' % (GATE_SYMBOL, gate.get('close'), GATE_SMA,
                                        gate.get('sma'),
                                        ', '.join(c['symbol'] for c in cands[:8])), 'low')
        else:
            st['pending'] = cands[:free]
            log.append('%d candidates, %d slots free, %d buy-stops armed for tomorrow '
                       '(gate OFF: %s %s vs %d-day average %s)'
                       % (len(cands), free, len(st['pending']), GATE_SYMBOL,
                          gate.get('close'), GATE_SMA, gate.get('sma')))
            if st['pending']:
                _alert('IPO candidates for tomorrow',
                       '; '.join(f'{c["symbol"]} buy-stop {c["pivot"]}'
                                 for c in st['pending']), 'low')

        # Everything that qualified today, armed or not, plus the near-misses. Stored on
        # state so the page can show WHY a name was passed over rather than leaving the
        # reader to guess between "no slot", "gated" and "not a candidate".
        armed = {c['symbol'] for c in st['pending']}
        st['candidates'] = [dict(c, armed=c['symbol'] in armed,
                                 passed_over=('gate' if blocked else
                                              (None if c['symbol'] in armed else 'no slot')))
                            for c in cands]
        st['watchlist'] = watch[:25]

        # ---- 4. nav point ----
        tot = sum(p['qty'] * float(close[p['symbol']].loc[:asof].dropna().iloc[-1])
                  for p in st['positions'] if p['symbol'] in close.columns)
        nav = tot + st['cash'] + cash_park.gain(st, cash_park.db_close(asof))
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




def _kite():
    """The broker client. Raises if the token is missing - callers decide what that means."""
    from kiteconnect import KiteConnect
    api_key = [l.split('=', 1)[1].strip() for l in open(ROOT / '.env')
               if l.startswith('KITE_API_KEY')][0]
    tok = json.load(open(ROOT / 'backtest_data' / 'access_token.json'))
    k = KiteConnect(api_key=api_key)
    k.set_access_token(tok.get('access_token') or tok.get('token'))
    return k


def _live_px(kite, syms):
    """-> (ltp, previous close) per held symbol. Missing quotes fall back to the book.

    The previous close comes from the quote's own OHLC block, which is what makes a
    today's-P&L figure possible at all: value minus what the same shares were worth at
    last night's close.
    """
    px, prev = {}, {}
    for i in range(0, len(syms), 200):
        try:
            q = kite.quote(['NSE:' + s for s in syms[i:i + 200]])
            for k, v in q.items():
                s = k.split(':', 1)[1]
                lp = float(v.get('last_price') or 0)
                if lp > 0:
                    px[s] = lp
                pc = float((v.get('ohlc') or {}).get('close') or 0)
                if pc > 0:
                    prev[s] = pc
        except Exception as e:
            print('quote batch failed:', e)
    return px, prev


def _trail_sma_proxy(syms, live):
    """Where the trail would sit if today closed at the current price: the last
    TRAIL_SMA-1 stored closes plus the live price, averaged.

    The same close-proxy Open Alpha uses for its own trail. An intraday trail computed any
    other way would either lag a day or invent a bar that has not closed.

    The window follows TRAIL_SMA, so it became 50 bars on 12-Sep-2026 with the rest of the
    spec. A name with fewer stored closes than that gets no trail shown rather than a
    short-window average, which would be a tighter rule than the book runs.
    """
    import sqlite3
    n = TRAIL_SMA - 1
    con = sqlite3.connect(f'file:{DB}?mode=ro', uri=True)
    out = {}
    try:
        for s in syms:
            if s not in live:
                continue
            rows = [r[0] for r in con.execute(
                "select close from market_data_unified where symbol=? and timeframe='day' "
                'and close > 0 order by date desc limit ?', (s, n))]
            if len(rows) == n:
                out[s] = (sum(rows) + live[s]) / float(TRAIL_SMA)
    finally:
        con.close()
    return out


def mark():
    """Re-price the book from live quotes and rewrite the UI. Never touches state."""
    st = load_state()
    syms = [p['symbol'] for p in st.get('positions', [])]
    if not syms:
        print('mark: nothing held')
        return
    kite = _kite()
    live, prev = _live_px(kite, syms)
    smas = _trail_sma_proxy(syms, live)

    rows, tot_val, tot_pnl = [], 0.0, 0.0
    today = date.today()
    for p in st['positions']:
        lp = live.get(p['symbol'], p['buy'])
        val = p['qty'] * lp
        pnl = p['qty'] * (lp - p['buy'])
        tot_val += val
        tot_pnl += pnl
        tr = smas.get(p['symbol'])
        pc = prev.get(p['symbol'])
        rows.append(dict(**p, ltp=round(lp, 2), value=round(val), pnl=round(pnl),
                         pnl_pct=round((lp / p['buy'] - 1) * 100, 2),
                         prev_close=round(pc, 2) if pc else None,
                         day_move_pct=round((lp / pc - 1) * 100, 2) if pc else None,
                         trail=round(tr, 2) if tr else None,
                         target=round(p['buy'] * (1 + TARGET), 2),
                         to_stop_pct=round((lp / p['stop'] - 1) * 100, 1),
                         to_trail_pct=round((lp / tr - 1) * 100, 1) if tr else None,
                         days=(today - date.fromisoformat(p['entry_date'])).days))
    cash = float(st['cash'])
    # cash includes money parked in CASHIETF at cost; the gain on it is the only new term
    nav = tot_val + cash + cash_park.gain(st)
    for r in rows:
        r['weight'] = round(100 * r['value'] / nav, 1) if nav else 0
    realized = sum(t.get('net_pnl', 0) for t in st.get('trades', []))
    cap = float(st['capital'])
    ui = dict(updated=str(datetime.now()), asof=str(today), mode=st.get('mode', 'paper'),
              positions=rows, capital=round(cap), cash=round(cash), value=round(tot_val),
              nav=round(nav), pnl=round(tot_pnl), realized=round(realized),
              gain=round(nav - cap),
              return_pct=round(100 * (nav - cap) / cap, 2) if cap else 0,
              invested_pct=round(100 * tot_val / nav, 1) if nav else 0,
              slots=SLOTS, slots_used=len(rows),
              pending=st.get('pending', []), navcurve=st.get('nav', []),
              trades=st.get('trades', [])[-100:], data_events=st.get('data_events', [])[-20:],
              failed_orders=st.get('failed_orders', []), started=st.get('started'),
              spec=spec_block(), gate=st.get('gate'),
              candidates=st.get('candidates', []), watchlist=st.get('watchlist', []),
              missed=st.get('missed', [])[-40:],
              park=cash_park.ui_block(st),
              log=['intraday mark - prices live, exits still decided by the 18:45 run'])
    tmp = UI_JSON.with_suffix('.json.tmp')
    json.dump(ui, open(tmp, 'w'), indent=1, default=str)
    os.replace(tmp, UI_JSON)
    print('%s marked %d positions: value Rs %s nav Rs %s'
          % (datetime.now().strftime('%H:%M:%S'), len(rows),
             format(round(tot_val), ','), format(round(nav), ',')))


def migrate_now(dry=False):
    """Apply the spec migration to stored state on its own, without a scan.

    Needed because the nightly cycle refuses to run outside market days, so deploying a
    spec change on a Saturday would otherwise leave held positions running the old stop
    until the next session. Takes the lock like everything else that writes state.
    """
    if not acquire_lock():
        print('book busy')
        return
    try:
        st = load_state()
        before = [(p['symbol'], p.get('stop')) for p in st.get('positions', [])]
        changed = migrate_spec(st)
        print('spec_version now %s' % st.get('spec_version'))
        for (s, old), p in zip(before, st.get('positions', [])):
            print('  %-12s stop %s -> %s   target %s   trail now the %d-day average'
                  % (s, old, p.get('stop'), round(p['buy'] * (1 + TARGET), 2), TRAIL_SMA))
        blocked, gate = market_gate()
        print('gate: %s' % ('ON - no new entries' if blocked else 'OFF - entries allowed'))
        print('  %s' % json.dumps(gate))
        if dry:
            print('DRY - nothing written')
            return
        if changed:
            save_state(st)
            print('state written')
        else:
            print('already on %s - nothing to do' % SPEC_VERSION)
    finally:
        release_lock()


if __name__ == '__main__':
    if '--migrate' in sys.argv:
        migrate_now(dry='--dry' in sys.argv)
    elif '--gate' in sys.argv:
        b, g = market_gate()
        print('BLOCKED' if b else 'OPEN', json.dumps(g, indent=1))
    elif '--mark' in sys.argv:
        mark()
    elif '--arm-now' in sys.argv:
        arm()
    elif '--reconcile' in sys.argv:
        reconcile_now()
    else:
        main()
