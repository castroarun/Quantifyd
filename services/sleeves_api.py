"""
Capital Desk API — Flask blueprint. Every rupee in or out of every book goes here.

Contract, uniform across books: POST {"amount": N, "dry_run": true} returns a PLAN and
mutates nothing; dry_run false executes. Positions are never force-sold — a withdrawal
that cannot be funded from cash (plus the CASHIETF sweep, on True North) is refused with
409 and a plan showing what would have to be sold. Every executed flow is ledgered.

Each leg dispatches to the BOOK'S OWN hardened implementation rather than reimplementing
it here:

  truenorth  -> momentum_paper.cash_deposit / cash_withdraw
                (deploy plan, CASHIETF unsweep before selling, weakest-momentum-first,
                 mp_fills audit rows, capital fence, fund-flow ledger)
  openalpha  -> oa_real.deposit / withdraw
                (capital ledger + lockfile; alert-and-ledger only, since the real book
                 has no automated executor — it records the money and tells Arun what to
                 buy, it never places an order)

Rewritten 05-Sep-2026. What this file used to claim, and what it actually did:

  - the docstring said the True North leg "calls the existing /api/momentum-paper/
    deposit|withdraw"; it did not — it did raw INSERT OR REPLACE on mp_state, skipping
    the deploy plan, the unsweep, the weakest-first sell and the audit trail (D3), which
    is why the same Rs 1L withdrawal 409'd here and executed on the True North page (D4)
  - the Open Alpha endpoints edited bluesky_paper_state.json, the RETIRED paper book,
    while the real money book had no capital concept and no route at all (D1)

  GET  /api/sleeves/status                 (all books + allocation drift)
  POST /api/sleeves/{truenorth,openalpha}/{deposit,withdraw}  {"amount": N, "dry_run": bool}
  GET  /api/sleeves/allocation             (targets, current weights, drift, gaps)
  POST /api/sleeves/allocation/route       {"amount": N} -> where a deposit should go
  POST /api/sleeves/allocation/targets     {"targets": {...}, "note": str}
  GET  /api/sleeves/dividends              (policy state + ledger, both books)
  POST /api/sleeves/dividends/preview      (dry-run declaration, both books)

No trading logic is touched here: no entry, exit, stop, trail, sizing, gate or selection
rule. This module moves money and reports; the engines decide what to do with it.
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


def _cashietf_px():
    """Latest CASHIETF close from the market DB (liquid fund — moves ~0.02%/day,
    so the last close is a safe realtime sweep price)."""
    try:
        import sqlite3
        con = sqlite3.connect(str(ROOT / 'backtest_data' / 'market_data.db'))
        r = con.execute("SELECT close FROM market_data_unified WHERE symbol='CASHIETF' "
                        "AND timeframe='day' ORDER BY date DESC LIMIT 1").fetchone()
        con.close()
        return float(r[0]) if r else None
    except Exception:
        return None


def _dep_plan(st, amt):
    return dict(book='open-alpha', kind='deposit', amount=amt,
                cash_now=round(st['cash'], 0), liquid_now=round(_liquid(st), 0),
                plan=[f'Rs {amt:,.0f} lands in cash and sweeps into CASHIETF immediately '
                      f'(at the latest close; the nightly cycle trues it up)',
                      'funds new pivot buy-stops from the next signal'],
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
            # realtime sweep (Arun 2026-09-04): park everything above the Rs 5k
            # float into CASHIETF now at the latest close, instead of waiting for
            # the nightly cycle. The cycle's redeem-and-resweep trues up interest.
            px = _cashietf_px()
            RESERVE = 5000.0
            if px and st['cash'] > RESERVE:
                sweep_amt = st['cash'] - RESERVE
                sw = st.setdefault('sweep', {'symbol': 'CASHIETF', 'units': 0.0, 'cost': 0.0})
                sw['units'] = round(sw.get('units', 0.0) + sweep_amt / px, 3)
                sw['cost'] = round(sw.get('cost', 0.0) + sweep_amt, 2)
                st['cash'] = RESERVE
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
    """One status payload covering every book the Capital Desk can move money into.

    Previously this returned Open Alpha only — and the Open Alpha it returned was the
    retired PAPER book, so the page showed paper balances for a real-money sleeve and
    showed nothing at all for True North (defect D1).
    """
    out = dict(books={}, note='Every leg runs the book\'s own hardened flow: True North '
                              'through cash_deposit/cash_withdraw, Open Alpha through '
                              'the real book\'s capital ledger.')
    try:
        mp = _tn_book()
        out['books']['truenorth'] = dict(
            name='True North', kind='live', capital=float(mp._get('capital', 0.0) or 0.0),
            cash=round(float(mp._get('cash', 0.0) or 0.0)),
            liquid=round(float(mp._get('cash', 0.0) or 0.0) + mp._sweep_value()),
            flows=(mp._get('fund_flows', []) or [])[-10:])
    except Exception as e:
        out['books']['truenorth'] = dict(name='True North', error=str(e))
    try:
        out['books']['openalpha'] = dict(name='Open Alpha', kind='live',
                                         **_oa_book().status())
    except Exception as e:
        out['books']['openalpha'] = dict(name='Open Alpha', error=str(e))
    try:                                    # the retired paper model, display only
        st = _load() or {}
        ui = json.load(open(UI_JSON))
        out['books']['openalpha_model'] = dict(
            name='Open Alpha (reference model)', kind='paper', nav=ui.get('nav'),
            cash=st.get('cash'), capital=st.get('capital'),
            note='Study-spec notional book. Takes no deposits.')
    except Exception:
        pass
    out['allocation'] = _drift()
    return jsonify(out)


# ═════════════════ money flows — one hardened path per book ═════════════════
# Rewritten 05-Sep-2026 (defects D1, D3, D4).
#
#   D1  Every /api/sleeves/openalpha/* call used to mutate bluesky_paper_state.json —
#       the RETIRED paper book — while the real money book (oa_real_state.json,
#       Rs 4.46L) had no capital concept and no route at all. Real capital was
#       untracked and unwithdrawable. OA now dispatches to services.oa_real.
#   D3  The True North leg used to do raw INSERT OR REPLACE on mp_state, bypassing
#       cash_deposit()/cash_withdraw() — so no deploy plan, no CASHIETF unsweep, no
#       weakest-first sell, no mp_fills audit row, no capital fence. Two docstrings
#       and the UI both claimed the opposite. It now calls the real functions.
#   D4  Consequence of D3: the same Rs 1L withdrawal 409'd here and executed on the
#       True North page, because this leg capped at ledger cash and never unswept the
#       Rs 3.3L sitting in CASHIETF. Fixed by D3.
#
# The paper Open Alpha book stays running as the reference model; it simply no longer
# takes money. Its notional capital is fixed by the study spec.
MP_DB = ROOT / 'backtest_data' / 'momentum_paper.db'


def _tn_book():
    from services import momentum_paper as mp
    return mp


def _oa_book():
    from services import oa_real
    return oa_real


def _flow(book, kind):
    """Dispatch a deposit/withdraw to the named book's own hardened implementation."""
    amt, dry = _params()
    if amt is None:
        return jsonify(error='amount must be a number between 1 and 1,00,00,000'), 400
    try:
        if book == 'truenorth':
            mp = _tn_book()
            res = (mp.cash_deposit(amt, mode='immediate', dry_run=dry) if kind == 'deposit'
                   else mp.cash_withdraw(amt, dry_run=dry))
            res.setdefault('book', 'true-north')
        else:
            oa = _oa_book()
            res = (oa.deposit(amt, dry_run=dry) if kind == 'deposit'
                   else oa.withdraw(amt, dry_run=dry))
    except Exception as e:
        return jsonify(error=f'{book} {kind} failed: {e}'), 500
    if not res.get('ok', True):
        return jsonify(res), 400
    # a withdrawal that cannot be funded is a refusal, not a silent partial
    if kind == 'withdraw' and not dry:
        if res.get('feasible') is False or res.get('shortfall', 0) > 1:
            return jsonify(res), 409
    return jsonify(res)


@sleeves_bp.route('/api/sleeves/openalpha/deposit', methods=['POST'])
def sleeves_deposit():
    return _flow('openalpha', 'deposit')


@sleeves_bp.route('/api/sleeves/openalpha/withdraw', methods=['POST'])
def sleeves_withdraw():
    return _flow('openalpha', 'withdraw')


@sleeves_bp.route('/api/sleeves/truenorth/deposit', methods=['POST'])
def tn_deposit():
    return _flow('truenorth', 'deposit')


@sleeves_bp.route('/api/sleeves/truenorth/withdraw', methods=['POST'])
def tn_withdraw():
    return _flow('truenorth', 'withdraw')


# ═════════════════ target allocation and the deposit router ═════════════════
# Defect D6: no cross-book allocation concept existed anywhere — the only ratio in the
# repo was Math.round(n/2) in the UI. Arun's target is TN 40 / OA 40 / IPO 20, funded
# over time, with True North as the BASE: nothing is ever sold to rebalance, arriving
# cash is steered to whichever book is furthest below its share.
ALLOC = ROOT / 'backtest_data' / 'allocation_targets.json'
DEFAULT_ALLOC = dict(targets={'truenorth': 0.40, 'openalpha': 0.40, 'ipo': 0.20},
                     base='truenorth',
                     ipo_status='paper',
                     changelog=[dict(date='2026-09-05',
                                     text='Adopted TN 40 / OA 40 / IPO 20. IPO is on paper, '
                                          'so its share is held in the liquid ETF until the '
                                          'sleeve graduates.')])


def _alloc():
    try:
        return json.load(open(ALLOC))
    except Exception:
        return dict(DEFAULT_ALLOC)


def _book_values():
    """Current deployed value per book, from each book's own source of truth."""
    out = {}
    try:
        mp = _tn_book()
        out['truenorth'] = float(mp._get('capital', 0.0) or 0.0)
    except Exception:
        out['truenorth'] = 0.0
    try:
        out['openalpha'] = float(_oa_book().load_state().get('capital', 0.0) or 0.0)
    except Exception:
        out['openalpha'] = 0.0
    # IPO is a paper book: its live money is whatever has been earmarked, not notional NAV
    out['ipo'] = float(_alloc().get('ipo_funded', 0.0) or 0.0)
    return out


def _drift():
    a = _alloc()
    tg = a['targets']
    vals = _book_values()
    total = sum(vals.values())
    rows = []
    for k, w in tg.items():
        cur = vals.get(k, 0.0)
        want = total * w
        rows.append(dict(book=k, value=round(cur), target_pct=round(w * 100, 1),
                         current_pct=round(100 * cur / total, 1) if total else 0.0,
                         target_value=round(want), gap=round(want - cur)))
    return dict(total=round(total), base=a.get('base', 'truenorth'),
                ipo_status=a.get('ipo_status', 'paper'), rows=rows,
                changelog=a.get('changelog', []))


def _route(amount):
    """Plan where an arriving deposit should go.

    Rule (Arun, 05-Sep-2026): True North is the BASE — never sold down. Compute each
    book's shortfall against its target share of the post-deposit total, fill the
    largest shortfalls first, then split any remainder at the target weights so the
    book lands ON target rather than overshooting.
    """
    a = _alloc()
    tg = a['targets']
    vals = _book_values()
    total_after = sum(vals.values()) + amount

    # Shortfalls are measured against each book's share of the POST-deposit total, which
    # is what makes the deposit land ON target rather than overshooting. A consequence
    # worth stating: those shortfalls always sum to at least the deposit. Without the
    # max(0,...) clip they sum to exactly it (total_after - total == amount), and the clip
    # only removes negatives, so it can only push the sum up. There is therefore never an
    # "excess" left over to split at the target weights — when no book is overweight the
    # proportional fill IS the 40/40/20 split, and it lands exactly on target.
    short = {k: max(0.0, total_after * w - vals.get(k, 0.0)) for k, w in tg.items()}
    total_short = sum(short.values())
    over = [k for k, w in tg.items() if vals.get(k, 0.0) > total_after * w + 1]

    alloc = {}
    if total_short > 0:
        for k, s in short.items():
            alloc[k] = amount * s / total_short
    else:                                   # degenerate: already exactly on target
        for k, w in tg.items():
            alloc[k] = amount * w

    legs = [dict(book=k, amount=round(v, 2)) for k, v in alloc.items() if round(v, 2) > 0]
    legs.sort(key=lambda r: -r['amount'])

    notes = []
    if a.get('ipo_status') == 'paper':
        notes.append('IPO is a paper book — its share is parked in the liquid ETF and '
                     'earmarked, not sent to a live sleeve.')
    if over:
        pretty = ', '.join(over)
        notes.append(f'{pretty} is above its target share, so it receives nothing from '
                     f'this deposit. Nothing is ever sold to rebalance — the base is held '
                     f'and the others are funded up to it.')
    if total_short > amount + 1:
        notes.append(f'This deposit closes Rs {amount:,.0f} of a Rs {total_short:,.0f} gap '
                     f'— it moves the book toward target without reaching it.')
    else:
        notes.append('This deposit lands the book exactly on its target weights.')
    return dict(amount=round(amount, 2), legs=legs, notes=notes, drift_before=_drift())


@sleeves_bp.route('/api/sleeves/allocation')
def sleeves_allocation():
    return jsonify(_drift())


@sleeves_bp.route('/api/sleeves/allocation/route', methods=['POST'])
def sleeves_route():
    amt, _dry = _params()
    if amt is None:
        return jsonify(error='amount must be a number between 1 and 1,00,00,000'), 400
    return jsonify(_route(amt))


@sleeves_bp.route('/api/sleeves/allocation/targets', methods=['POST'])
def sleeves_set_targets():
    d = request.get_json(silent=True) or {}
    tg = d.get('targets')
    if not isinstance(tg, dict) or not tg:
        return jsonify(error='targets must be an object of book -> weight'), 400
    try:
        tg = {k: float(v) for k, v in tg.items()}
    except Exception:
        return jsonify(error='weights must be numbers'), 400
    if abs(sum(tg.values()) - 1.0) > 1e-6:
        return jsonify(error=f'weights must sum to 1.0 (got {sum(tg.values()):.4f})'), 400
    a = _alloc()
    a['targets'] = tg
    a.setdefault('changelog', []).append(dict(
        date=str(datetime.now())[:19],
        text=d.get('note') or ('Targets set to ' +
                               ', '.join(f'{k} {v*100:.0f}' for k, v in tg.items()))))
    tmp = ALLOC.with_suffix('.json.tmp')
    json.dump(a, open(tmp, 'w'), indent=1)
    os.replace(tmp, ALLOC)
    return jsonify(_drift())


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
