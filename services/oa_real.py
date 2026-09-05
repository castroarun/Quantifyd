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

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'backtest_data' / 'oa_real_state.json'
LOCK = ROOT / 'backtest_data' / 'oa_real_state.lock'
UI = ROOT / 'static' / 'app' / 'oa_real.json'
FEED = Path('/tmp/nas_alert_feed.log')
STOP_PCT = 0.08
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
    with open(FEED, 'a') as f:
        f.write(json.dumps(dict(ts=str(datetime.now()), book='OA-REAL', urgency=urgency,
                                title=title, body=body)) + '\n')


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
    gain = nav + realized - capital
    ui = dict(updated=str(datetime.now()), positions=rows, invested=round(cost),
              capital=round(capital), value=round(tot_val), cash=round(cash),
              nav=round(nav), pnl=round(tot_pnl), realized=round(realized),
              gain=round(gain),
              pnl_pct=round(100 * tot_pnl / cost, 2) if cost else 0,
              return_pct=round(100 * gain / capital, 2) if capital else 0,
              inception='04-Sep-2026', navcurve=st.get('navcurve', []),
              flows=st.get('fund_flows', [])[-20:],
              note=st['note'], trades=st.get('trades', []))
    tmp = UI.with_suffix('.json.tmp')
    json.dump(ui, open(tmp, 'w'), indent=1, default=str)
    os.replace(tmp, UI)
    print(f"marked {len(rows)} positions: value Rs {tot_val:,.0f} P&L {tot_pnl:+,.0f} "
          f"cash Rs {cash:,.0f} nav Rs {nav:,.0f}")


def check():
    """15:18 close-proxy rule check. Alert-only."""
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
        msg = (f"SELL {p['symbol']} x{p['qty']} CNC (limit ~{lp:.2f}) — {why}. "
               f"Entry {p['buy']}, now {lp} ({(lp/p['buy']-1)*100:+.1f}%). Place before 15:30.")
        _alert(f"OA-REAL EXIT DUE: {p['symbol']}", msg)
        print('EXIT DUE:', msg)


if __name__ == '__main__':
    {'seed': seed, 'mark': mark, 'check': check}[sys.argv[1] if len(sys.argv) > 1 else 'mark']()
