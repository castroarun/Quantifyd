"""
Sleeves money-management API (paper-mode) — Flask blueprint.

Manages FUND FLOWS for the Open Alpha PAPER book only (deposit / withdraw against
its cash + CASHIETF sweep; never force-sells positions). True North is read-only
here — its real-money funding moves through its own existing flows. This blueprint
touches no trading logic: it edits the paper book's cash ledger under the same
lockfile the nightly engine uses, and logs every flow.

Endpoints:
  GET  /api/sleeves/status              -> both sleeves' NAV + split + flows tail
  POST /api/sleeves/openalpha/deposit   {"amount": 100000}
  POST /api/sleeves/openalpha/withdraw  {"amount": 50000}
"""
import json
import os
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
    """Acquire the paper book's lockfile (short wait); returns True on success."""
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


def _liquid(st):
    """Withdrawable liquidity = free cash + sweep at cost (conservative)."""
    sw = st.get('sweep') or {}
    return float(st.get('cash', 0.0)) + float(sw.get('cost', 0.0))


def _apply_flow(kind, amount):
    if not _locked():
        return None, 'book is busy (nightly run in progress) — try again in a minute'
    try:
        st = json.load(open(STATE))
        liq = _liquid(st)
        if kind == 'withdraw' and amount > liq + 1:
            return None, (f'only Rs {liq:,.0f} is liquid (cash + sweep); positions are never '
                          f'force-sold — withdraw less or wait for exits')
        sign = 1 if kind == 'deposit' else -1
        # draw from cash first, then break sweep units proportionally
        if kind == 'withdraw':
            take_cash = min(st['cash'], amount)
            st['cash'] -= take_cash
            rem = amount - take_cash
            if rem > 0:
                sw = st['sweep']
                frac = rem / sw['cost'] if sw['cost'] else 0
                sw['units'] = round(sw['units'] * (1 - frac), 3)
                sw['cost'] = round(sw['cost'] - rem, 2)
        else:
            st['cash'] += amount
        st['capital'] = round(st.get('capital', 0.0) + sign * amount, 0)
        st.setdefault('fund_flows', []).append(dict(
            ts=str(datetime.now()), kind=kind, amount=round(amount, 0),
            nav_note='applied to cash/sweep; positions untouched'))
        tmp = STATE.with_suffix('.json.tmp')
        json.dump(st, open(tmp, 'w'), indent=1, default=str)
        os.replace(tmp, STATE)
        return st, None
    finally:
        _unlock()


def _amount():
    try:
        amt = float((request.get_json(silent=True) or {}).get('amount', 0))
    except (TypeError, ValueError):
        return None
    if not (0 < amt <= MAX_FLOW):
        return None
    return round(amt, 0)


@sleeves_bp.route('/api/sleeves/status')
def sleeves_status():
    try:
        ui = json.load(open(UI_JSON))
    except Exception:
        ui = {}
    st = json.load(open(STATE)) if STATE.exists() else {}
    return jsonify(dict(
        open_alpha=dict(nav=ui.get('nav'), cash=st.get('cash'),
                        sweep=st.get('sweep'), liquid=_liquid(st) if st else 0,
                        capital=st.get('capital'),
                        flows=(st.get('fund_flows') or [])[-10:]),
        note='True North funding is real money and moves through its own page/flows; '
             'this panel manages the Open Alpha paper sleeve only.'))


@sleeves_bp.route('/api/sleeves/openalpha/deposit', methods=['POST'])
def sleeves_deposit():
    amt = _amount()
    if amt is None:
        return jsonify(error='amount must be a number between 1 and 1,00,00,000'), 400
    st, err = _apply_flow('deposit', amt)
    if err:
        return jsonify(error=err), 409
    return jsonify(ok=True, cash=st['cash'], capital=st['capital'],
                   note='cash added; it sweeps to CASHIETF and funds new signals from the next nightly run')


@sleeves_bp.route('/api/sleeves/openalpha/withdraw', methods=['POST'])
def sleeves_withdraw():
    amt = _amount()
    if amt is None:
        return jsonify(error='amount must be a number between 1 and 1,00,00,000'), 400
    st, err = _apply_flow('withdraw', amt)
    if err:
        return jsonify(error=err), 409
    return jsonify(ok=True, cash=st['cash'], capital=st['capital'],
                   note='withdrawn from cash + sweep; open positions untouched')
