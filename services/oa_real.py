"""Open Alpha REAL book — state, marks, capital ledger, and the EOD-faithful exit checker.

Seeded 04-Sep-2026 (Arun's explicit go, ahead of the Dec-5 soak gate — logged
override). Rules mirror the paper spec: -8% hard stop on CLOSE, 15-SMA trail on
CLOSE (entry-day trail exempt). Real execution is manual-assisted for now:

  mode `mark`  : refresh static/app/oa_real.json from live quotes (page display)
  mode `check` : 15:18 IST close-proxy check — if price is below the stop or the
                 15-SMA trail, raise a desktop alert with the exact sell order.
                 ALERT-ONLY: this script never places orders.
  mode `seed`  : build state from today's executed CNC orders (one-off)

State: backtest_data/oa_real_state.json   Feed: /tmp/nas_alert_feed.log (popups)

CAPITAL LEDGER (added 05-Sep-2026, defect D1).
Before this, the real book had no `capital`, no `cash` and no `fund_flows`. NAV was
`tot_val + 0` by construction and returns were `pnl / invested`, so the numbers would
have gone silently wrong the moment money moved. Worse, every deposit and withdrawal
routed through /api/sleeves/openalpha/* was mutating `bluesky_paper_state.json` — the
RETIRED paper book — so real capital was untracked and unwithdrawable by any path.

Now:
  capital     total external money contributed (deposits - withdrawals), the
              denominator for returns and the base for the allocation targets
  cash        contributed money not yet in positions
  fund_flows  append-only ledger of every deposit/withdrawal

NAV = positions value + cash. Return = (NAV - capital) / capital, which is flow-neutral:
a deposit moves NAV and capital by the same amount, so it moves the return line by zero.

Flows are ALERT-AND-LEDGER only, exactly like exits: this book has no automated
executor, so `deposit()` records the money and tells Arun what to buy, and `withdraw()`
frees cash and names the weakest positions to sell. It never places an order and never
force-sells. No entry, exit, stop, trail, sizing or gate rule is touched by this change.
"""
import json
import os
import sys
import time
from datetime import date, datetime
from pathlib import Path

# ─────────────────────────── THE RULESET SWITCH ───────────────────────────
# 'legacy'  : the book as seeded on 04-Sep-2026 — buy-stop-at-the-pivot entries (paused),
#             -8% hard stop on CLOSE, 15-SMA close trail, exits placed at 15:18.
# 'baseage' : Open Alpha · Base Age, adopted in research/161 and re-fitted in research/164 —
#             new-ATH-close entry with a >= 60-bar-old prior high and a >= 20% base depth,
#             filled at the NEXT day's open; SuperTrend(14,4) close trail as the ONLY exit,
#             no hard stop, no time stop; 16 slots at 6.25% of NAV.
#
# THIS IS THE ONLY THING THAT HAS TO CHANGE TO CONVERT THE BOOK, and it is deliberately a
# constant in the source rather than an environment variable or a state-file field: a
# real-money ruleset should be visible in `git log`, reviewable in a diff, and impossible to
# flip by accident from a shell. `services/oa_entry.py` and `services/oa_baseage_entry.py`
# both read it from here, so the entry and the exit can never disagree about which book is
# running. Staged 12-Sep-2026; see
# research/165_oa_baseage_live_conversion/OA_BASE_AGE_LIVE_CONVERSION_DEPLOY_STATUS.md.
OA_RULESET = 'legacy'          # 'legacy' | 'baseage'

ROOT = Path(__file__).resolve().parents[1]
# Run as a script, Python puts services/ on the path, not the repo root -
# so `import services.x` fails and every alert delivery died silently in a
# try/except (found 09-Sep-2026 in the KTKBANK exit log).
sys.path.insert(0, str(ROOT))
STATE = ROOT / 'backtest_data' / 'oa_real_state.json'
LOCK = ROOT / 'backtest_data' / 'oa_real_state.lock'
UI = ROOT / 'static' / 'app' / 'oa_real.json'
# A real feed the pages can read. The old path was another job's cron log.
FEED = ROOT / 'backtest_data' / 'book_alerts.jsonl'
STOP_PCT = 0.08
COST_PCT = 0.0025        # 25 bps per side, the study's assumption
# 16 equal slots at 6.25% of NAV each - the book's shape, stated once so the pages do
# not have to know it.
SLOTS = 16

OA_TAG = 'OA-TOPUP'          # top-ups into existing holdings
# Every tag this book answers to. An order without one of these is somebody else's, in an
# account that holds 54 names across personal holdings and other books.
BOOK_TAGS = ('OA-TOPUP', 'OA-ENTRY', 'OA-EXIT')
SEEN_ORDERS = ROOT / 'backtest_data' / 'oa_applied_orders.json'
TRAIL_N = 15
MAX_FLOW = 10_000_000
SYMS16 = ['INDSWFTLAB', 'SETL', 'WELCORP', 'SHILPAMED', 'KMEW', 'SBCL', 'IOLCP',
          'SPORTKING', 'IRISDOREME', 'INOXINDIA', 'MANINDS', 'SSWL', 'ENTERO',
          'NITINSPIN', 'TMB', 'KTKBANK']


# ─────────────────────── state: lock + atomic save ───────────────────────
# `mark` runs every minute in market hours; before this the state was written
# unlocked and in place (json.dump straight over the file), which is the same
# shape as the 2026-08-05 race that corrupted the NWV paper book. Same fix as
# services/bluesky_paper.py: O_EXCL lockfile + .tmp + os.replace.

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


def load_state():
    st = json.load(open(STATE))
    return _migrate(st)


def save_state(st):
    tmp = STATE.with_suffix('.json.tmp')
    json.dump(st, open(tmp, 'w'), indent=1, default=str)
    os.replace(tmp, STATE)


def _migrate(st):
    """Backfill the capital ledger onto a pre-D1 state file, in place and idempotently.

    The seeded book put every rupee into stock, so at seed time capital == cost of the
    positions and cash == 0. That is the only honest starting point: `invested` was the
    sole capital proxy the old schema had.
    """
    if 'capital' not in st:
        cost = sum(p['qty'] * p['buy'] for p in st.get('positions', []))
        # `invested` was stored rounded to the rupee, so trusting it alone yields a
        # capital below the true position cost and a negative cash balance. Take the
        # larger of the two: every seeded rupee went into stock, so cash starts at 0.
        st['capital'] = round(max(float(st.get('invested') or 0.0), cost), 2)
        st['cash'] = round(float(st['capital']) - cost, 2)
        st['fund_flows'] = [dict(ts=st.get('seeded') or str(datetime.now()),
                                 kind='deposit', amount=st['capital'],
                                 via='seed 04-Sep-2026 (backfilled by the D1 migration)',
                                 positions_touched=True)]
    st.setdefault('cash', 0.0)
    st.setdefault('fund_flows', [])
    return st


def _cost(st):
    return sum(p['qty'] * p['buy'] for p in st.get('positions', []))


def _kite():
    from kiteconnect import KiteConnect
    api_key = [l.split('=', 1)[1].strip() for l in open(ROOT / '.env')
               if l.startswith('KITE_API_KEY')][0]
    tok = json.load(open(ROOT / 'backtest_data' / 'access_token.json'))
    k = KiteConnect(api_key=api_key)
    k.set_access_token(tok.get('access_token') or tok.get('token'))
    return k


def _alert(title, body, urgency='critical'):
    """Record the alert, and for a critical one actually send it.

    The feed is a real file the pages read. `/tmp/nas_alert_feed.log`, which this used to
    write to, is the cron output log of a different job and is read by nobody -- an exit
    alert written there reached no one at all (found 08-Sep-2026, SPORTKING).
    """
    line = dict(ts=str(datetime.now()), book='OA-REAL', urgency=urgency,
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


def _sma15(kite, syms, live):
    """SMA15 close-proxy per symbol: last 14 DB closes + today's live price."""
    import sqlite3
    con = sqlite3.connect(str(ROOT / 'backtest_data' / 'market_data.db'))
    out = {}
    for s in syms:
        rows = [r[0] for r in con.execute(
            "SELECT close FROM market_data_unified WHERE symbol=? AND timeframe='day' "
            "ORDER BY date DESC LIMIT 14", (s,))]
        if len(rows) == 14 and live.get(s):
            out[s] = (sum(rows) + live[s]) / 15.0
    con.close()
    return out


def _live(kite, syms):
    q = {}
    for i in range(0, len(syms), 25):
        q.update(kite.quote(['NSE:' + s for s in syms[i:i + 25]]))
    return q


# ───────────────────── the Base Age exit: SuperTrend(14,4) ─────────────────────
# The SAME function research/161 measured the book with, imported rather than re-typed
# (`services/oa_baseage.supertrend_dir` is bt_core's byte-for-byte). An exit that is
# "basically the same SuperTrend" is a different book, and the 11.85 percentage points this
# trail is worth over the 15-SMA-plus-8%-stop is the largest single number in that study.

def _st14(syms, live=None, quotes=None, asof=None):
    """{symbol: dict(dir, line, close, src)} under SuperTrend(14,4).

    With `live`/`quotes` the still-forming bar is appended as a CLOSE PROXY — today's open,
    running high, running low and the last traded price — which is how the 15:18 check reads
    the rule before the official close exists. That reading is advisory ONLY: it alerts, it
    never sells. Without them the answer is the official one from `market_data.db`, which is
    what `confirm()` acts on.
    """
    from services import oa_baseage as spec
    out = {}
    con = spec.connect()
    try:
        for s in syms:
            proxy = None
            if live and live.get(s):
                oh = (quotes or {}).get('NSE:' + s, {}).get('ohlc', {}) or {}
                lp = float(live[s])
                proxy = (float(oh.get('open') or lp),
                         max(float(oh.get('high') or lp), lp),
                         min(float(oh.get('low') or lp), lp), lp)
            d, line, c, last = spec.st_state(s, asof=asof, proxy=proxy, con=con)
            if d is None:
                continue
            out[s] = dict(dir=d, line=round(line, 2) if line == line else None,
                          close=round(c, 2), src='proxy' if proxy else last)
    finally:
        con.close()
    return out


def _db_bar(sym, asof=None):
    """(date, close, prev_close) of the most recent OFFICIAL daily bar in the DB."""
    from services import oa_baseage as spec
    con = spec.connect()
    try:
        d, _ = spec.load_bars(con, sym, asof)
    finally:
        con.close()
    if d is None or len(d) < 2:
        return None, None, None
    return (str(d['date'].iloc[-1])[:10], float(d['close'].iloc[-1]),
            float(d['close'].iloc[-2]))


# ───────────────────────── money in and out ─────────────────────────

def deposit(amount, dry_run=True):
    """Credit external money to the book. Ledger + plan only — never places an order.

    The book holds 16 equal slots at 6.25% of NAV. A deposit is reported as the
    per-slot top-up it implies so Arun can execute it by hand; the money sits in
    `cash` until he does, and the page shows it as undeployed.
    """
    amt = round(float(amount), 2)
    if amt <= 0 or amt > MAX_FLOW:
        return dict(ok=False, error=f'amount must be between 0 and {MAX_FLOW:,.0f}')
    st = load_state()
    n = len(st.get('positions', []))
    per = amt / n if n else 0.0
    plan = [f'credit Rs {amt:,.0f} to the book (capital Rs {st["capital"]:,.0f} '
            f'-> Rs {st["capital"] + amt:,.0f})']
    if n:
        plan.append(f'implies Rs {per:,.0f} per slot across {n} holdings to stay equal-weight')
    plan.append('MANUAL: no executor on this book — cash sits undeployed until you buy')
    out = dict(ok=True, book='open-alpha', kind='deposit', amount=amt, dry_run=dry_run,
               plan=plan, capital_after=round(st['capital'] + amt, 2),
               cash_after=round(st['cash'] + amt, 2))
    if dry_run:
        return out
    if not acquire_lock():
        return dict(ok=False, error='book is busy (a mark or check is running) — try again')
    try:
        st = load_state()
        st['capital'] = round(st['capital'] + amt, 2)
        st['cash'] = round(st['cash'] + amt, 2)
        st['fund_flows'].append(dict(ts=str(datetime.now()), kind='deposit', amount=amt,
                                     via='capital desk', positions_touched=False))
        save_state(st)
    finally:
        release_lock()
    _alert('OA-REAL deposit recorded',
           f'Rs {amt:,.0f} added. Cash now Rs {st["cash"]:,.0f} — deploy manually '
           f'(~Rs {per:,.0f} per slot).', 'low')
    return out


def withdraw(amount, dry_run=True):
    """Take money out. Frees cash first; never force-sells — names what to sell instead."""
    amt = round(float(amount), 2)
    if amt <= 0 or amt > MAX_FLOW:
        return dict(ok=False, error=f'amount must be between 0 and {MAX_FLOW:,.0f}')
    st = load_state()
    cash = float(st['cash'])
    feasible = amt <= cash + 1
    plan = []
    if feasible:
        plan.append(f'pay out Rs {amt:,.0f} from idle cash (Rs {cash:,.0f} available)')
    else:
        short = amt - cash
        plan.append(f'only Rs {cash:,.0f} is free cash — Rs {short:,.0f} short')
        plan.append('positions are never force-sold: withdraw less, or sell manually first')
        weak = sorted(st.get('positions', []), key=lambda p: p.get('buy', 0) * p.get('qty', 0))
        for p in weak[:3]:
            plan.append(f'  candidate to raise cash: SELL {p["symbol"]} x{p["qty"]}')
    out = dict(ok=True, book='open-alpha', kind='withdraw', amount=amt, dry_run=dry_run,
               feasible=feasible, plan=plan,
               capital_after=round(st['capital'] - amt, 2) if feasible else st['capital'],
               cash_after=round(cash - amt, 2) if feasible else cash)
    if dry_run or not feasible:
        return out
    if not acquire_lock():
        return dict(ok=False, error='book is busy (a mark or check is running) — try again')
    try:
        st = load_state()
        st['capital'] = round(st['capital'] - amt, 2)
        st['cash'] = round(st['cash'] - amt, 2)
        st['fund_flows'].append(dict(ts=str(datetime.now()), kind='withdraw', amount=amt,
                                     via='capital desk', positions_touched=False))
        save_state(st)
    finally:
        release_lock()
    _alert('OA-REAL withdrawal recorded', f'Rs {amt:,.0f} paid out.', 'low')
    return out


def status():
    """Read-only snapshot for the Capital Desk."""
    st = load_state()
    ui = json.load(open(UI)) if UI.exists() else {}
    return dict(book='open-alpha', capital=st['capital'], cash=st['cash'],
                positions=len(st.get('positions', [])),
                value=ui.get('value'), nav=ui.get('nav'), updated=ui.get('updated'),
                flows=st.get('fund_flows', [])[-20:])


# ───────────────────────── seed / mark / check ─────────────────────────

def seed():
    kite = _kite()
    fills = {}
    for o in kite.orders():
        if (o['status'] == 'COMPLETE' and o['transaction_type'] == 'BUY'
                and o['product'] == 'CNC' and o['tradingsymbol'] in SYMS16):
            f = fills.setdefault(o['tradingsymbol'], dict(qty=0, value=0.0))
            f['qty'] += o['filled_quantity']
            f['value'] += o['filled_quantity'] * o['average_price']
    positions = []
    invested = 0.0
    for s in SYMS16:
        f = fills.get(s)
        if not f or f['qty'] == 0:
            print(f'WARNING: no fill for {s}')
            continue
        buy = f['value'] / f['qty']
        invested += f['value']
        positions.append(dict(symbol=s, qty=f['qty'], buy=round(buy, 2),
                              entry_date=str(date.today()),
                              stop=round(buy * (1 - STOP_PCT), 2), src='real'))
    st = dict(book='OA-REAL', seeded=str(datetime.now()), positions=positions,
              invested=round(invested, 0),
              capital=round(invested, 2), cash=0.0,
              fund_flows=[dict(ts=str(datetime.now()), kind='deposit',
                               amount=round(invested, 2), via='seed',
                               positions_touched=True)],
              note='Seeded 04-Sep-2026 from Arun-executed CNC fills (top-16 by RS of the '
                   'day\'s 21 triggered candidates). LIQUIDCASE 1757u sold to fund. '
                   'Deliberate override of the Dec-5 soak gate. Exits manual-assisted: '
                   '15:18 checker alerts; no automated selling yet.',
              trades=[])
    save_state(st)
    print(f'seeded {len(positions)} positions, invested Rs {invested:,.0f}')
    mark()


def mark():
    kite = _kite()
    st = load_state()
    syms = [p['symbol'] for p in st['positions']]
    q = _live(kite, syms)
    live = {s: q.get('NSE:' + s, {}).get('last_price') for s in syms}
    # The page shows whichever trail the ACTIVE ruleset would exit on, under the same
    # `trail` / `to_trail_pct` field names — so the dashboard needs no change to tell the
    # truth, and it cannot show a 15-SMA the book no longer obeys.
    if OA_RULESET == 'baseage':
        stx = _st14(syms, live, q)
        smas = {s: v['line'] for s, v in stx.items() if v['line']}
    else:
        stx = {}
        smas = _sma15(kite, syms, live)
    rows, tot_val, tot_pnl = [], 0.0, 0.0
    for p in st['positions']:
        lp = live.get(p['symbol'])
        oh = q.get('NSE:' + p['symbol'], {}).get('ohlc', {})
        prev = oh.get('close')
        val = p['qty'] * lp if lp else p['qty'] * p['buy']
        pnl = p['qty'] * (lp - p['buy']) if lp else 0.0
        tot_val += val
        tot_pnl += pnl
        sma = smas.get(p['symbol'])
        days_held = (date.today() - date.fromisoformat(p['entry_date'])).days
        rows.append(dict(**p, ltp=lp, days=days_held,
                         day_move_pct=round((lp / prev - 1) * 100, 2) if lp and prev else None,
                         value=round(val), pnl=round(pnl),
                         pnl_pct=round((lp / p['buy'] - 1) * 100, 2) if lp else None,
                         trail=round(sma, 2) if sma else None,
                         trail_rule='ST(14,4)' if OA_RULESET == 'baseage' else '15-SMA',
                         st_dir=stx.get(p['symbol'], {}).get('dir'),
                         # the -8% stop is not a rule under 'baseage'; the field stays for
                         # the rollback path but the page must not read it as live
                         stop_active=(OA_RULESET != 'baseage'),
                         to_stop_pct=round((lp / p['stop'] - 1) * 100, 1) if lp else None,
                         to_trail_pct=round((lp / sma - 1) * 100, 1) if lp and sma else None))
    cash = float(st.get('cash', 0.0))
    capital = float(st.get('capital', 0.0))
    cost = _cost(st)
    nav = tot_val + cash
    for r in rows:
        r['weight'] = round(100 * r['value'] / nav, 1) if nav else 0
    # append the daily nav point on the post-close mark (>= 16:00 IST)
    if datetime.now().hour >= 16:
        if not acquire_lock():
            print('mark: could not take the lock, skipping the nav append')
        else:
            try:
                st = load_state()
                nc = st.setdefault('navcurve', [])
                today_s = str(date.today())
                nc[:] = [x for x in nc if x['d'] != today_s]
                nc.append(dict(d=today_s, nav=round(nav), capital=round(capital)))
                save_state(st)
            finally:
                release_lock()
    realized = sum(t.get('net_pnl', 0) for t in st.get('trades', []))
    # nav already contains realised P&L: a sale moved the money into cash at the
    # exit price. Adding `realized` here would subtract every closed trade twice.
    gain = nav - capital
    ui = dict(updated=str(datetime.now()), positions=rows, invested=round(cost),
              capital=round(capital), value=round(tot_val), cash=round(cash),
              nav=round(nav), pnl=round(tot_pnl), realized=round(realized),
              gain=round(gain),
              pnl_pct=round(100 * tot_pnl / cost, 2) if cost else 0,
              return_pct=round(100 * gain / capital, 2) if capital else 0,
              slots=SLOTS, slots_used=len(rows), ruleset=OA_RULESET,
              inception='04-Sep-2026', navcurve=st.get('navcurve', []),
              flows=st.get('fund_flows', [])[-20:],
              note=st['note'], trades=st.get('trades', []),
              failed_orders=st.get('failed_orders', []))
    tmp = UI.with_suffix('.json.tmp')
    json.dump(ui, open(tmp, 'w'), indent=1, default=str)
    os.replace(tmp, UI)
    print(f"marked {len(rows)} positions: value Rs {tot_val:,.0f} P&L {tot_pnl:+,.0f} "
          f"cash Rs {cash:,.0f} nav Rs {nav:,.0f}")


def ui_only():
    """Rebuild static/app/oa_real.json from state WITHOUT calling Kite.

    `mark()` needs live quotes, and the Kite token is only refreshed by the weekday
    auto-login cron — so between a Friday close and a Monday morning there is no way to
    regenerate the feed at all. That matters whenever the state schema changes: the page
    would keep serving the old shape until the next trading day, which is precisely when
    someone is looking at it after a deploy.

    This reuses the last known prices from the existing UI file (falling back to entry
    price), recomputes every derived field from CURRENT state, and marks the payload
    `stale` with the timestamp those prices came from, so the page can say so rather than
    quietly presenting Friday's marks as today's.
    """
    st = load_state()
    prev = {}
    prev_updated = None
    if UI.exists():
        try:
            old = json.load(open(UI))
            prev = {r['symbol']: r for r in old.get('positions', [])}
            prev_updated = old.get('updated')
        except Exception:
            pass
    rows, tot_val, tot_pnl = [], 0.0, 0.0
    for pos in st['positions']:
        o = prev.get(pos['symbol'], {})
        lp = o.get('ltp') or pos['buy']
        val = pos['qty'] * lp
        pnl = pos['qty'] * (lp - pos['buy'])
        tot_val += val
        tot_pnl += pnl
        days_held = (date.today() - date.fromisoformat(pos['entry_date'])).days
        rows.append(dict(**pos, ltp=lp, days=days_held,
                         day_move_pct=o.get('day_move_pct'),
                         value=round(val), pnl=round(pnl),
                         pnl_pct=round((lp / pos['buy'] - 1) * 100, 2),
                         trail=o.get('trail'),
                         to_stop_pct=round((lp / pos['stop'] - 1) * 100, 1),
                         to_trail_pct=o.get('to_trail_pct')))
    cash = float(st.get('cash', 0.0))
    capital = float(st.get('capital', 0.0))
    cost = _cost(st)
    nav = tot_val + cash
    for r in rows:
        r['weight'] = round(100 * r['value'] / nav, 1) if nav else 0
    realized = sum(t.get('net_pnl', 0) for t in st.get('trades', []))
    # nav already contains realised P&L: a sale moved the money into cash at the
    # exit price. Adding `realized` here would subtract every closed trade twice.
    gain = nav - capital
    ui = dict(updated=prev_updated or str(datetime.now()), stale=True,
              positions=rows, invested=round(cost), capital=round(capital),
              value=round(tot_val), cash=round(cash), nav=round(nav),
              pnl=round(tot_pnl), realized=round(realized), gain=round(gain),
              pnl_pct=round(100 * tot_pnl / cost, 2) if cost else 0,
              return_pct=round(100 * gain / capital, 2) if capital else 0,
              slots=SLOTS, slots_used=len(rows), ruleset=OA_RULESET,
              inception='04-Sep-2026', navcurve=st.get('navcurve', []),
              flows=st.get('fund_flows', [])[-20:],
              note=st['note'], trades=st.get('trades', []),
              failed_orders=st.get('failed_orders', []))
    tmp = UI.with_suffix('.json.tmp')
    json.dump(ui, open(tmp, 'w'), indent=1, default=str)
    os.replace(tmp, UI)
    print(f'ui-only rebuild: {len(rows)} positions, nav Rs {nav:,.0f}, '
          f'prices as of {prev_updated or "entry"}')


def reconcile(dry=True):
    """Apply THIS BOOK'S OWN completed orders to the book. Never absolute holdings.

    THE INCIDENT THIS EXISTS TO PREVENT (08-Sep-2026). The first version of this matched
    the book's positions against broker holdings BY SYMBOL and overwrote quantity and
    average price from them. The account is shared: it carries 54 equity names worth
    about Rs 1.89 crore across personal holdings and other books. So the book was handed
    KMEW 481 @ 1427.92 in place of its own 9 @ 3037.64, and INOXINDIA 132 in place of 12.
    Book value leapt to Rs 21.1 lakh with Rs 7.6 lakh of invented profit. State was
    restored from the last commit.

    A book in a shared account may only ever count what IT bought. So this reads the
    order book, keeps only COMPLETE CNC buys carrying this book's own tag, and applies
    them as INCREMENTS. An untagged order, or one placed by hand, is ignored by design:
    the cost of missing a fill is a stale book that a human notices, while the cost of
    claiming someone else's shares is a book that lies about how much money exists.
    """
    kite = _kite()
    seen = json.load(open(SEEN_ORDERS)) if SEEN_ORDERS.exists() else {}
    adds, sells = {}, {}
    for o in kite.orders():
        oid = str(o.get('order_id'))
        if (o.get('status') != 'COMPLETE' or o.get('product') != 'CNC'
                or not o.get('filled_quantity')):
            continue
        if (o.get('tag') or '') not in BOOK_TAGS:   # not this book's order
            continue
        if oid in seen:                             # already applied on an earlier run
            continue
        s = o['tradingsymbol']
        q, px = int(o['filled_quantity']), float(o['average_price'])
        if o.get('transaction_type') == 'BUY':
            pq, pv = adds.get(s, (0, 0.0))
            adds[s] = (pq + q, pv + q * px)
        else:
            pq, pv = sells.get(s, (0, 0.0))
            sells[s] = (pq + q, pv + q * px)
        seen[oid] = dict(ts=str(datetime.now()), symbol=s, side=o.get('transaction_type'),
                         qty=q, price=px)

    if not adds and not sells:
        print('reconcile: no new tagged fills')
        if not dry:
            mark()
        return

    lines, spend, raised = [], 0.0, 0.0
    for s, (q, val) in adds.items():
        lines.append('%s +%d @%.2f = Rs %s' % (s, q, val / q, format(round(val), ',')))
        spend += val
    for s, (q, val) in sells.items():
        lines.append('%s -%d @%.2f = Rs %s' % (s, q, val / q, format(round(val), ',')))
        raised += val
    print('reconcile would apply:' if dry else 'reconcile applying:')
    for l in lines:
        print('   ', l)
    print('    bought Rs %s, sold Rs %s'
          % (format(round(spend), ','), format(round(raised), ',')))
    if dry:
        print('  (dry run - pass dry=False to write)')
        return

    if not acquire_lock():
        print('reconcile: book busy, skipping')
        return
    try:
        st = load_state()
        if spend - raised > float(st['cash']) + 1:
            _alert('OA reconcile refused',
                   'Tagged fills cost Rs %s but the book only holds Rs %s in cash. '
                   'Nothing applied - check for a missed deposit.'
                   % (format(round(spend), ','), format(round(st['cash']), ',')))
            print('REFUSED: fills exceed book cash')
            return
        by_sym = {p['symbol']: p for p in st['positions']}

        # ---- exits first: a sale funds the buys, and frees the slot ----
        for s, (q, val) in sells.items():
            pos = by_sym.get(s)
            if not pos:
                _alert('OA reconcile: sold something the book does not hold',
                       'A tagged SELL filled for %s x%d but the book has no such position. '
                       'Check whether it was already applied by hand.' % (s, q))
                continue
            px = val / q
            gross = q * (px - pos['buy'])
            st.setdefault('trades', []).append(dict(
                symbol=s, qty=q, buy=pos['buy'], sell=round(px, 2),
                entry_date=pos.get('entry_date'), exit_date=str(date.today()),
                reason='rule_exit', net_pnl=round(gross - COST_PCT * q * (px + pos['buy'])),
                pnl_pct=round((px / pos['buy'] - 1) * 100, 2)))
            if q >= pos['qty']:
                st['positions'] = [x for x in st['positions'] if x['symbol'] != s]
            else:
                pos['qty'] -= q                    # partial fill: keep the remainder
            # net of charges, at the same rate the trade record was costed at, or the
            # ledger disagrees with the P&L by exactly the fees
            st['cash'] = round(float(st['cash']) + val - COST_PCT * val, 2)
        by_sym = {p['symbol']: p for p in st['positions']}
        for s, (q, val) in adds.items():
            avg = val / q
            if s in by_sym:
                p = by_sym[s]
                nq = p['qty'] + q
                p['buy'] = round((p['qty'] * p['buy'] + val) / nq, 2)
                p['qty'] = nq
                p['stop'] = round(p['buy'] * (1 - STOP_PCT), 2)
            else:
                st['positions'].append(dict(symbol=s, qty=q, buy=round(avg, 2),
                                            entry_date=str(date.today()),
                                            stop=round(avg * (1 - STOP_PCT), 2), src='executor'))
        st['cash'] = round(max(0.0, float(st['cash']) - spend * (1 + COST_PCT)), 2)
        save_state(st)
        json.dump(seen, open(SEEN_ORDERS, 'w'), indent=1, default=str)
    finally:
        release_lock()
    _alert('Open Alpha: fills applied', '; '.join(lines), 'low')
    print('applied; cash now Rs %s' % format(round(st['cash']), ','))
    mark()


EXIT_TAG = 'OA-EXIT'
EXIT_FLOOR = 0.02          # a sell limit 2% under the last price


def _resting(kite, symbol, side):
    """Is one of this book's orders already live for that symbol and side?"""
    try:
        for o in kite.orders():
            if (o.get('tradingsymbol') == symbol and o.get('transaction_type') == side
                    and o.get('status') not in ('REJECTED', 'CANCELLED')):
                return True
    except Exception as e:
        print('order read failed, refusing to place blind:', e)
        return True                       # unknown state: do NOT risk a duplicate
    return False


def place_exit(kite, symbol, qty, ltp, tick=0.05):
    """Send one exit. Returns (order_id, error).

    A LIMIT with a floor rather than a market order: Kite refuses bare market orders via
    API, and a floor costs nothing in a normal session - a limit sell fills at the best
    price at or above it - while refusing to dump into a collapse.
    """
    import math
    if _resting(kite, symbol, 'SELL'):
        return None, 'a SELL is already live'
    floor = round(math.floor((ltp * (1 - EXIT_FLOOR)) / tick) * tick, 2)
    try:
        oid = kite.place_order(variety='regular', exchange='NSE', tradingsymbol=symbol,
                               transaction_type='SELL', quantity=int(qty), product='CNC',
                               order_type='LIMIT', price=floor, validity='DAY',
                               tag=EXIT_TAG)
        return oid, None
    except Exception as e:
        return None, str(e)


def place_exit_amo(kite, symbol, qty, ref_close, tick=0.05):
    """One AMO sell for TOMORROW's open. Returns (order_id, order_type_used, error).

    MARKET FIRST, LIMIT AS THE FALLBACK, and the choice is logged. The study exits at the
    next open with no floor, which only a market order reproduces; Zerodha's RMS has
    historically refused after-market MARKET orders on some segments, and `place_exit` above
    already carries the scar of a bare market order being refused through the API. So the
    refusal is expected and answered with a LIMIT 2% UNDER the last close rather than being
    allowed to skip the exit. An exit that silently does not go is the one failure this book
    cannot have.

    Not test-fired: which branch this account takes is settled by the first real evening run,
    and both are safe. A LIMIT floored 2% under the close fills at the open in any ordinary
    session and refuses only to dump into a collapse.
    """
    if _resting(kite, symbol, 'SELL'):
        return None, None, 'a SELL is already live'
    err = None
    try:
        oid = kite.place_order(variety='amo', exchange='NSE', tradingsymbol=symbol,
                               transaction_type='SELL', quantity=int(qty), product='CNC',
                               order_type='MARKET', validity='DAY', tag=EXIT_TAG)
        return oid, 'MARKET', None
    except Exception as e:
        err = str(e)
        print('       AMO MARKET refused (%s); falling back to LIMIT' % err[:90])
    import math
    floor = round(math.floor((ref_close * (1 - EXIT_FLOOR)) / tick) * tick, 2)
    try:
        oid = kite.place_order(variety='amo', exchange='NSE', tradingsymbol=symbol,
                               transaction_type='SELL', quantity=int(qty), product='CNC',
                               order_type='LIMIT', price=floor, validity='DAY', tag=EXIT_TAG)
        return oid, 'LIMIT %.2f' % floor, err
    except Exception as e2:
        return None, None, '%s | LIMIT also refused: %s' % (err, e2)


def _check_baseage(arm=False):
    """15:18 close-proxy check under Base Age. QUEUES an exit; never sells.

    WHY THIS DOES NOT PLACE AN ORDER, even with --arm. The Base Age exit is a CLOSE signal:
    SuperTrend(14,4) flips on the official close and the study fills at the NEXT open. A
    15:18 proxy is a forecast of that close, and forecasts reverse — an intraday poke through
    the band that the last twelve minutes take back would, under the legacy 15:18 behaviour,
    have sold a position the rule never told us to sell. So the proxy's whole job here is to
    give Arun warning, and the decision is made tonight on the real close by `confirm()`.
    """
    kite = _kite()
    st = load_state()
    syms = [p['symbol'] for p in st['positions']]
    if not syms:
        print('no positions')
        return
    q = _live(kite, syms)
    live = {s: q.get('NSE:' + s, {}).get('last_price') for s in syms}
    stx = _st14(syms, live, q)
    queued = []
    for p in st['positions']:
        v = stx.get(p['symbol'])
        lp = live.get(p['symbol'])
        if not v or not lp:
            continue
        if v['dir'] == -1:
            queued.append((p, lp, v))
    if not queued:
        _alert('OA-REAL 15:18 check (Base Age): all clear',
               '%d positions, SuperTrend(14,4) still long on every one' % len(syms), 'low')
        print('all clear (%d positions, ST(14,4) long)' % len(syms))
        return
    body = []
    for p, lp, v in queued:
        body.append('%s x%d at %.2f, ST line %.2f (entry %.2f, %+.1f%%)'
                    % (p['symbol'], p['qty'], lp, v['line'] or 0, p['buy'],
                       (lp / p['buy'] - 1) * 100))
        print('QUEUED (not sold):', body[-1])
    _alert('OA-REAL: %d Base Age exit(s) QUEUED for tonight' % len(queued),
           'SuperTrend(14,4) would flip down on this close-proxy. NOTHING IS SOLD NOW. '
           'The flip is confirmed tonight on the official close and an after-market sell '
           'is placed for tomorrow\'s open.\n  ' + '\n  '.join(body), 'low')


def confirm(arm=False, asof=None):
    """The Base Age exit decision, on the OFFICIAL close. This is the one that sells.

    Runs in the evening, AFTER the 17:45 nightly universe refresh has written the day's
    daily bars. Three guards, each of which has a live incident behind it somewhere in this
    repo:

      1. NEVER ON A PARTIAL CANDLE. It refuses to run before 17:50 IST on a weekday, and it
         refuses any symbol whose latest DB bar is not the latest session. A SuperTrend read
         off a half-formed bar is not the rule.
      2. NEVER SELL INTO A SPLIT. A single-day close move of -40% or worse is treated as a
         corporate action, the position is HELD and an alert is raised, exactly as
         `services/ipo_paper.py` does. `market_data.db` is not retroactively split-adjusted.
      3. NEVER TWICE. `place_exit_amo` refuses a symbol that already has a live SELL.

    Under 'legacy' this is a no-op, so the mode is safe to wire into cron before any flip.
    """
    if OA_RULESET != 'baseage':
        print('OA_RULESET is %r - Base Age exit confirmation does not apply; nothing done.'
              % OA_RULESET)
        return
    now = datetime.now()
    if arm and now.weekday() < 5 and (now.hour, now.minute) < (17, 50):
        print('%s - the daily bars are not in yet (universe refresh runs 17:45); '
              'refusing to confirm an exit on a partial candle' % now)
        return
    from services import oa_baseage as spec
    st = load_state()
    syms = [p['symbol'] for p in st['positions']]
    if not syms:
        print('no positions')
        return
    session = asof or spec.last_session()
    stx = _st14(syms, asof=asof)
    kite = _kite() if arm else None
    due, stale, held_split = [], [], []
    for p in st['positions']:
        s = p['symbol']
        v = stx.get(s)
        if not v:
            stale.append((s, 'no usable price history'))
            continue
        d, c, prev = _db_bar(s, asof)
        if d != session:
            # The name did not trade today, or the refresh missed it. Either way the rule
            # has not been evaluated on today's close, so the position is simply not acted
            # on - loudly.
            stale.append((s, 'last DB bar %s, latest session %s' % (d, session)))
            continue
        if prev and prev > 0 and (c / prev - 1) <= spec.DATA_EVENT_DROP:
            held_split.append((s, prev, c))
            continue
        if v['dir'] == -1:
            due.append((p, v, c))
    for s, why in stale:
        print('  STALE %-14s %s' % (s, why))
    if stale:
        _alert('OA-REAL Base Age confirm: %d position(s) not evaluated' % len(stale),
               'No exit decision was made for: '
               + '; '.join('%s (%s)' % (s, w) for s, w in stale)
               + '. Check the 17:45 universe refresh.')
    for s, prev, c in held_split:
        print('  DATA EVENT %-14s %.2f -> %.2f  HELD, not sold' % (s, prev, c))
        _alert('OA-REAL data event: %s' % s,
               '%s close %.2f -> %.2f in one day. Treated as a split or bonus: the position '
               'is HELD, not exited. Verify the price series before the next session.'
               % (s, prev, c))
    if not due:
        print('confirm %s: no Base Age exit due (%d positions)' % (session, len(syms)))
        _alert('OA-REAL Base Age confirm: no exit due',
               '%d positions, SuperTrend(14,4) long on every one as of the %s close'
               % (len(syms) - len(stale) - len(held_split), session), 'low')
        return
    for p, v, c in due:
        head = ('SELL %s x%d CNC at tomorrow\'s open - SuperTrend(14,4) flipped down on the '
                '%s close %.2f (trail %.2f). Entry %.2f, %+.1f%%.'
                % (p['symbol'], p['qty'], session, c, v['line'] or 0, p['buy'],
                   (c / p['buy'] - 1) * 100))
        if not arm:
            print('EXIT DUE (not armed):', head)
            continue
        oid, kind, err = place_exit_amo(kite, p['symbol'], p['qty'], c)
        if oid:
            print('EXIT PLACED:', p['symbol'], oid, kind)
            _alert('OA-REAL EXIT PLACED (Base Age): %s' % p['symbol'],
                   head + ' After-market order %s is in as %s.' % (oid, kind))
        else:
            _alert('OA-REAL EXIT FAILED TO PLACE: %s' % p['symbol'],
                   head + ' The order was NOT sent: %s. Place it by hand before 09:15.' % err)
            print('EXIT FAILED:', p['symbol'], err)


def check(arm=False):
    """15:18 close-proxy rule check.

    With `arm`, it PLACES the exits it finds instead of only naming them. Until 08-Sep-2026
    it was alert-only, and the alert went to a log nobody read - so SPORTKING sat a full
    day below its trail with the right order sitting in a file. Arun approved automatic
    exits the same evening.

    Under 'baseage' the whole meaning of 15:18 changes - see `_check_baseage`.
    """
    if OA_RULESET == 'baseage':
        return _check_baseage(arm)
    kite = _kite()
    st = load_state()
    syms = [p['symbol'] for p in st['positions']]
    q = _live(kite, syms)
    live = {s: q.get('NSE:' + s, {}).get('last_price') for s in syms}
    smas = _sma15(kite, syms, live)
    today = str(date.today())
    hits = []
    for p in st['positions']:
        lp = live.get(p['symbol'])
        if not lp:
            continue
        if lp <= p['stop']:
            hits.append((p, lp, f"below -8% stop {p['stop']}"))
        elif p['entry_date'] != today and smas.get(p['symbol']) and lp < smas[p['symbol']]:
            hits.append((p, lp, f"below 15-SMA trail {smas[p['symbol']]:.2f}"))
    if not hits:
        _alert('OA-REAL 15:18 check: all clear', f'{len(syms)} positions, no exits due', 'low')
        print('all clear')
    for p, lp, why in hits:
        head = (f"SELL {p['symbol']} x{p['qty']} CNC — {why}. "
                f"Entry {p['buy']}, now {lp} ({(lp/p['buy']-1)*100:+.1f}%).")
        if not arm:
            _alert(f"OA-REAL EXIT DUE: {p['symbol']}",
                   head + f" Place before 15:30 (limit ~{lp:.2f}).")
            print('EXIT DUE:', head)
            continue
        oid, err = place_exit(kite, p['symbol'], p['qty'], lp)
        if oid:
            _alert(f"OA-REAL EXIT PLACED: {p['symbol']}",
                   head + f" Order {oid} is in, limit floored 2% under {lp:.2f}.")
            print('EXIT PLACED:', p['symbol'], oid)
        else:
            # An exit that could not be sent is the one thing that must never be quiet.
            _alert(f"OA-REAL EXIT FAILED TO PLACE: {p['symbol']}",
                   head + f" The order was NOT sent: {err}. Place it by hand.")
            print('EXIT FAILED:', p['symbol'], err)


if __name__ == '__main__':
    def _asof():
        return sys.argv[sys.argv.index('--asof') + 1] if '--asof' in sys.argv else None

    _modes = {'seed': seed, 'mark': mark, 'ui-only': ui_only,
              'check': lambda: check(arm='--arm' in sys.argv),
              # Base Age only; a no-op under 'legacy', so it is safe to leave in cron
              'confirm': lambda: confirm(arm='--arm' in sys.argv, asof=_asof()),
              'ruleset': lambda: print(OA_RULESET),
              'reconcile': lambda: reconcile(dry='--arm' not in sys.argv)}
    _modes[sys.argv[1] if len(sys.argv) > 1 else 'mark']()
