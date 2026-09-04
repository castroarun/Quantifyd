"""
Sleeves money-management API — Flask blueprint, momentum-framework contract.

Open Alpha (paper) endpoints follow True North's battle-tested deposit/withdraw
shape: POST {"amount": N, "dry_run": true} returns a PLAN (no mutation);
dry_run false executes. Withdrawals draw from cash + CASHIETF sweep only —
positions are never force-sold. Every executed flow is ledgered in the state.

The unified portal on /app/sleeves dispatches: its True North leg calls the
existing /api/momentum-paper/deposit|withdraw (live, hardened, confirms real
orders itself); its Open Alpha leg calls these endpoints. This module touches
no trading logic and edits the paper ledger under the engine's own lockfile.

  GET  /api/sleeves/status
  POST /api/sleeves/openalpha/deposit   {"amount": N, "dry_run": bool}
  POST /api/sleeves/openalpha/withdraw  {"amount": N, "dry_run": bool}
  POST /api/sleeves/truenorth/deposit   {"amount": N, "dry_run": bool}
  POST /api/sleeves/truenorth/withdraw  {"amount": N, "dry_run": bool}
  GET  /api/sleeves/dividends           (policy state + ledger, both books)
  POST /api/sleeves/dividends/preview   (dry-run declaration, both books)

True North flows edit only the cash/capital ledger in momentum_paper.db
(mp_state) — the momentum engine then sizes off the changed cash at its own
next scheduled step. No order is placed here and no engine code is touched.
"""
import json
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path

from flask import Blueprint, jsonify, request

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'backtest_data' / 'bluesky_paper_state.json'
LOCK = ROOT / 'backtest_data' / 'bluesky_paper_state.lock'
UI_JSON = ROOT / 'static' / 'app' / 'bluesky_paper.json'

sleeves_bp = Blueprint('sleeves', __name__)

MAX_FLOW = 10_000_000  # sanity cap per operation (Rs 1 Cr)


def _locked():
    for _ in range(10):
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, b'sleeves_api')
            os.close(fd)
            return True
        except FileExistsError:
            time.sleep(1)
    return False


def _unlock():
    if LOCK.exists():
        LOCK.unlink()


def _load():
    return json.load(open(STATE)) if STATE.exists() else None


def _liquid(st):
    sw = st.get('sweep') or {}
    return float(st.get('cash', 0.0)) + float(sw.get('cost', 0.0))


def _params():
    body = request.get_json(silent=True) or {}
    try:
        amt = round(float(body.get('amount', 0)), 0)
    except (TypeError, ValueError):
        return None, None
    if not (0 < amt <= MAX_FLOW):
        return None, None
    return amt, bool(body.get('dry_run', False))


def _dep_plan(st, amt):
    return dict(book='open-alpha', kind='deposit', amount=amt,
                cash_now=round(st['cash'], 0), liquid_now=round(_liquid(st), 0),
                plan=[f'Rs {amt:,.0f} lands in cash immediately',
                      'sweeps into CASHIETF at tonight\'s run',
                      'funds new pivot buy-stops from the next signal (gate permitting)'],
                capital_after=round(st.get('capital', 0) + amt, 0))


def _wd_plan(st, amt):
    liq = _liquid(st)
    ok = amt <= liq + 1
    take_cash = min(st['cash'], amt)
    from_sweep = max(0.0, amt - take_cash)
    return dict(book='open-alpha', kind='withdraw', amount=amt, feasible=ok,
                liquid_now=round(liq, 0),
                plan=([f'Rs {take_cash:,.0f} from free cash',
                       f'Rs {from_sweep:,.0f} by redeeming CASHIETF units',
                       'open positions untouched'] if ok else
                      [f'only Rs {liq:,.0f} is liquid (cash + sweep)',
                       'positions are never force-sold — withdraw less or wait for exits']),
                capital_after=round(st.get('capital', 0) - amt, 0) if ok else None)


def _execute(kind, amt):
    if not _locked():
        return None, 'book is busy (nightly run in progress) — try again in a minute'
    try:
        st = _load()
        if st is None:
            return None, 'paper book state not found'
        if kind == 'withdraw':
            if amt > _liquid(st) + 1:
                return None, _wd_plan(st, amt)['plan'][0]
            take_cash = min(st['cash'], amt)
            st['cash'] -= take_cash
            rem = amt - take_cash
            if rem > 0:
                sw = st['sweep']
                frac = rem / sw['cost'] if sw['cost'] else 0
                sw['units'] = round(sw['units'] * (1 - frac), 3)
                sw['cost'] = round(sw['cost'] - rem, 2)
            st['capital'] = round(st.get('capital', 0) - amt, 0)
        else:
            st['cash'] += amt
            st['capital'] = round(st.get('capital', 0) + amt, 0)
        st.setdefault('fund_flows', []).append(dict(
            ts=str(datetime.now()), kind=kind, amount=amt,
            via='sleeves portal', positions_touched=False))
        tmp = STATE.with_suffix('.json.tmp')
        json.dump(st, open(tmp, 'w'), indent=1, default=str)
        os.replace(tmp, STATE)
        return st, None
    finally:
        _unlock()


@sleeves_bp.route('/api/sleeves/status')
def sleeves_status():
    try:
        ui = json.load(open(UI_JSON))
    except Exception:
        ui = {}
    st = _load() or {}
    return jsonify(dict(
        open_alpha=dict(nav=ui.get('nav'), cash=st.get('cash'), sweep=st.get('sweep'),
                        liquid=round(_liquid(st), 0) if st else 0,
                        capital=st.get('capital'),
                        flows=(st.get('fund_flows') or [])[-10:]),
        note='True North legs dispatch to /api/momentum-paper/deposit|withdraw '
             '(its own live, hardened flow with real-order confirms).'))


def _flow_route(kind):
    amt, dry = _params()
    if amt is None:
        return jsonify(error='amount must be a number between 1 and 1,00,00,000'), 400
    st = _load()
    if st is None:
        return jsonify(error='paper book state not found'), 500
    plan = _dep_plan(st, amt) if kind == 'deposit' else _wd_plan(st, amt)
    if dry:
        return jsonify(plan)
    if kind == 'withdraw' and not plan['feasible']:
        return jsonify(error=plan['plan'][0]), 409
    st2, err = _execute(kind, amt)
    if err:
        return jsonify(error=err), 409
    return jsonify(dict(ok=True, executed=plan, cash=st2['cash'], capital=st2['capital']))


@sleeves_bp.route('/api/sleeves/openalpha/deposit', methods=['POST'])
def sleeves_deposit():
    return _flow_route('deposit')


@sleeves_bp.route('/api/sleeves/openalpha/withdraw', methods=['POST'])
def sleeves_withdraw():
    return _flow_route('withdraw')


# ───────────────────── True North (momentum book) flows ─────────────────────
MP_DB = ROOT / 'backtest_data' / 'momentum_paper.db'


def _mp_conn():
    c = sqlite3.connect(str(MP_DB)); c.row_factory = sqlite3.Row
    return c


def _mp_get(key, default=None):
    c = _mp_conn()
    r = c.execute('SELECT val FROM mp_state WHERE key=?', (key,)).fetchone()
    c.close()
    return json.loads(r['val']) if r else default


def _tn_flow(kind):
    amt, dry = _params()
    if amt is None:
        return jsonify(error='amount must be a number between 1 and 1,00,00,000'), 400
    if not MP_DB.exists():
        return jsonify(error='momentum book DB not found'), 500
    cash = float(_mp_get('cash', 0.0))
    cap = float(_mp_get('capital', 0.0))
    if kind == 'withdraw' and amt > cash + 1:
        plan = dict(book='true-north', kind=kind, amount=amt, feasible=False,
                    plan=[f'only Rs {cash:,.0f} is free cash',
                          'holdings are never force-sold — withdraw less or wait for '
                          'the next Donchian exit / rebalance to free cash'])
        return (jsonify(plan) if dry else (jsonify(error=plan['plan'][0]), 409))
    plan = dict(book='true-north', kind=kind, amount=amt, feasible=True,
                cash_now=round(cash), capital_now=round(cap),
                plan=([f'Rs {amt:,.0f} lands in book cash (earns liquid yield while idle)',
                       'deployed at the next monthly rebalance / gate redeploy']
                      if kind == 'deposit' else
                      [f'Rs {amt:,.0f} from free cash', 'holdings untouched']),
                capital_after=round(cap + (amt if kind == 'deposit' else -amt)))
    if dry:
        return jsonify(plan)
    c = _mp_conn()
    delta = amt if kind == 'deposit' else -amt
    c.execute('INSERT OR REPLACE INTO mp_state(key,val) VALUES(?,?)',
              ('cash', json.dumps(cash + delta)))
    c.execute('INSERT OR REPLACE INTO mp_state(key,val) VALUES(?,?)',
              ('capital', json.dumps(cap + delta)))
    flows = _mp_get('fund_flows', []) or []
    flows.append(dict(ts=str(datetime.now()), kind=kind, amount=amt,
                      via='sleeves portal', positions_touched=False))
    c.execute('INSERT OR REPLACE INTO mp_state(key,val) VALUES(?,?)',
              ('fund_flows', json.dumps(flows)))
    c.commit(); c.close()
    return jsonify(dict(ok=True, executed=plan, cash=round(cash + delta),
                        capital=round(cap + delta)))


@sleeves_bp.route('/api/sleeves/truenorth/deposit', methods=['POST'])
def tn_deposit():
    return _tn_flow('deposit')


@sleeves_bp.route('/api/sleeves/truenorth/withdraw', methods=['POST'])
def tn_withdraw():
    return _tn_flow('withdraw')


# ───────────────────── Open Alpha: initiate a cycle from the UI ─────────────────────
@sleeves_bp.route('/api/sleeves/openalpha/run', methods=['POST'])
def oa_run():
    """UI-initiated engine run. Before ~17:50 IST the day's official closes are not
    in the DB yet, so a FULL cycle would trade on stale data — we run a display
    refresh instead. After 17:50 (or on weekends) the full nightly cycle runs:
    pending buy-stop fills, exits, fresh scan, CASHIETF sweep."""
    import subprocess
    now = datetime.now()
    weekday = now.weekday() < 5
    market_stale = weekday and (now.hour < 17 or (now.hour == 17 and now.minute < 50))
    args = ['--ui-only'] if market_stale else []
    py = str(ROOT / 'venv' / 'bin' / 'python')
    subprocess.Popen([py, str(ROOT / 'services' / 'bluesky_paper.py'), *args],
                     cwd=str(ROOT), stdout=open('/tmp/bluesky_ui_run.log', 'a'),
                     stderr=subprocess.STDOUT)
    return jsonify(ok=True,
                   mode=('display refresh only — full cycle unlocks after 17:50 IST '
                         'once the day\'s official closes are in' if market_stale
                         else 'full nightly cycle initiated (fills, exits, scan, sweep)'),
                   log='/tmp/bluesky_ui_run.log')


# ───────────────────── dividends (quarterly HWM policy) ─────────────────────
@sleeves_bp.route('/api/sleeves/dividends')
def sleeves_dividends():
    from services.dividend_engine import POLICY, status
    return jsonify(dict(policy=POLICY,
                        truenorth=status('truenorth'),
                        openalpha=status('openalpha')))


@sleeves_bp.route('/api/sleeves/dividends/preview', methods=['POST'])
def sleeves_div_preview():
    from services.dividend_engine import declare
    return jsonify(dict(truenorth=declare('truenorth', dry_run=True),
                        openalpha=declare('openalpha', dry_run=True)))
