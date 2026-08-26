#!/usr/bin/env python3
"""Live-vs-app reconciliation — does the broker hold what the app thinks it holds?

Born from 2026-08-20: NAS_COMB20 sent 325 qty to the exchange (a deliberate
Thursday 5-lot consolidation) while the page rendered 130, so the displayed size
and P&L were 2.5x light. Nothing was wrong with the trade — but nothing in the
app could tell us that, which is the actual problem.

Four checks, all READ-ONLY (positions/orders reads; no orders, no writes to any
book's state):

  1. ORPHAN     — a broker leg no app record claims
  2. SIZE       — broker qty != the qty the app records for that symbol
  3. GHOST      — the app records a leg the broker does not hold
  4. NAKED      — a short option with no protective order resting at the exchange

Writes static/app/live_recon.json (page-readable), appends a dated line to
docs/LIVE_RECON_LOG.md, and sends the report over email + WhatsApp.

Usage:  venv/bin/python3 scripts/live_vs_app_recon.py [--quiet]
        --quiet   only notify when something is flagged
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
BD = ROOT / 'backtest_data'
OUT_JSON = ROOT / 'static' / 'app' / 'live_recon.json'
LOG_MD = ROOT / 'docs' / 'LIVE_RECON_LOG.md'

IST = datetime.now()


# ───────────────────────── broker truth ─────────────────────────
def broker_state():
    from services.kite_service import get_kite
    kite = get_kite()
    legs, orders = {}, []
    for p in kite.positions()['net']:
        if p['quantity'] == 0:
            continue
        legs[p['tradingsymbol']] = {
            'qty': int(p['quantity']),
            'avg': float(p['average_price']),
            'ltp': float(p.get('last_price') or 0),
            'pnl': round(float(p.get('pnl') or 0), 1),
            'product': p.get('product'),
            'exchange': p.get('exchange'),
        }
    today = IST.strftime('%Y-%m-%d')
    for o in kite.orders():
        ts = str(o.get('order_timestamp') or '')
        if today not in ts:
            continue
        orders.append({
            'symbol': o['tradingsymbol'], 'side': o['transaction_type'],
            'qty': int(o.get('filled_quantity') or 0),
            'status': o.get('status'), 'tag': o.get('tag'),
            'type': o.get('order_type'), 'time': ts[11:19],
        })
    return legs, orders


# ─────────────────────── what the app claims ───────────────────────
# Source: the app's own state endpoints — the same payloads the pages render.
# Anything the UI cannot see is, by definition, unreconciled.

NAS_ENDPOINTS = [
    'nas', 'nas-atm', 'nas-atm2', 'nas-atm4',
    'nas-916-otm', 'nas-916-atm', 'nas-916-atm2', 'nas-916-atm4',
]
LOCAL = 'http://127.0.0.1:5000'

# Books running PAPER today — filled by app_legs(); their legs never reach the
# broker, so a "missing" broker position is correct, not a break.
PAPER_BOOKS = set()


def _matrix_live():
    """{system_key: bool} — the per-day live/paper grid that gates real orders."""
    try:
        m = json.loads((BD / 'nas_day_matrix.json').read_text())
        return {k: bool(v.get('live')) for k, v in (m.get('systems') or {}).items()}
    except Exception:
        return {}


MATRIX_LIVE = _matrix_live()


def _get(path, default=None):
    import urllib.request
    try:
        with urllib.request.urlopen(f'{LOCAL}{path}', timeout=20) as r:
            return json.load(r)
    except Exception:
        return default


def app_legs():
    """{tradingsymbol: {'qty': signed, 'books': [...]}} as the APP reports it."""
    claimed = defaultdict(lambda: {'qty': 0, 'books': []})

    def add(sym, qty, book):
        if not sym or not qty:
            return
        claimed[sym]['qty'] += int(qty)
        claimed[sym]['books'].append(book)

    # NAS + SENSEX suites — each open leg states its OWN mode ('live'/'paper').
    # That beats both config.paper_trading_mode (can read live while the day/gap
    # matrix runs paper-shadow) and the matrix itself (a per-DTE gate can still
    # force paper on a matrix-live book, which is exactly what happened to the
    # 916 arms on 2026-08-20). Only live legs are expected at the broker.
    today_str = IST.strftime('%Y-%m-%d')
    for db in sorted(BD.glob('nas_*_trading.db')) + sorted(BD.glob('sensex_*_trading.db')):
        book = db.stem.replace('_trading', '')
        try:
            c = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
            for tbl in ('nas_atm_positions', 'nas_positions'):
                cols = [d[1] for d in c.execute(f'PRAGMA table_info({tbl})')]
                if 'tradingsymbol' not in cols:
                    continue
                modecol = 'mode' if 'mode' in cols else "'live'"
                for sym, qty, mode in c.execute(
                        f'select tradingsymbol, qty, {modecol} from {tbl} '
                        f'where exit_time IS NULL and date(entry_time)=?', (today_str,)):
                    if not sym or not qty:
                        continue
                    if str(mode or 'live').lower().startswith('paper'):
                        PAPER_BOOKS.add(book)
                        continue                      # paper leg: no broker position expected
                    add(sym, -abs(int(qty)), book)    # suites are short premium
            c.close()
        except Exception as e:
            print(f'  [warn] {db.name}: {e}')

    # CSL sleeves — the page feed, with the frozen per-DTE lots/qty applied
    csl = _get('/app/csl_paper_live.json') or {}
    try:
        cfg = json.loads((BD / 'csl_paper_config.json').read_text()).get('books', {})
    except Exception:
        cfg = {}
    for name, b in (csl.get('books') or {}).items():
        if not isinstance(b, dict) or b.get('state') != 'OPEN' or not b.get('live'):
            continue                                   # paper sleeves place no real orders
        qty = b.get('qty')
        if not qty:
            dte = str(b.get('dte', ''))
            qty = ((cfg.get(name) or {}).get(dte) or {}).get('qty')
        for key in ('ce_sym', 'pe_sym'):
            add(b.get(key), -abs(int(qty)) if qty else 0, name)

    return claimed


# Cash holdings that belong to Arun directly, not to a book. Flagged as MANUAL
# (informational) rather than ORPHAN, so the alert count means something.
def manual_symbols():
    syms = set()
    for db, table, col in (('momentum_paper.db', 'mp_positions', 'symbol'),
                           ('breakout_paper.db', 'bp_positions', 'symbol')):
        path = BD / db
        if not path.exists():
            continue
        try:
            c = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
            syms |= {r[0] for r in c.execute(f'select {col} from {table}')}
            c.close()
        except Exception:
            pass
    return syms


# ───────────────────────────── checks ─────────────────────────────
def _tagged_net(orders):
    """{symbol: {tag: signed_qty}} from today's COMPLETE fills."""
    out = {}
    for o in orders:
        if o['status'] != 'COMPLETE' or not o['qty']:
            continue
        sign = -1 if o['side'] == 'SELL' else 1
        out.setdefault(o['symbol'], {})
        tag = o['tag'] or 'untagged'
        out[o['symbol']][tag] = out[o['symbol']].get(tag, 0) + sign * o['qty']
    return out


def reconcile():
    legs, orders = broker_state()
    claimed = app_legs()
    book_cash = manual_symbols()
    tagged = _tagged_net(orders)
    findings = []

    for sym, b in legs.items():
        c = claimed.get(sym)
        if not c or c['qty'] == 0:
            tags = sorted({o['tag'] for o in orders if o['symbol'] == sym and o['tag']})
            is_cash = b['exchange'] == 'NSE'
            # A cash leg with no book and no order tag is a manual holding.
            manual = is_cash and not tags and sym not in book_cash
            tg = tagged.get(sym, {})
            explained = tg and sum(tg.values()) == b['qty'] and 'untagged' not in tg
            if explained:
                named = ', '.join(f'{k} {v:+d}' for k, v in sorted(tg.items(), key=lambda kv: kv[1]))
                findings.append({
                    'level': 'INFO', 'kind': 'MULTI-BOOK', 'symbol': sym,
                    'broker_qty': b['qty'], 'app_qty': 0,
                    'detail': f"broker {b['qty']} = {named} — placed by our books, "
                              f"not visible in the page feed",
                })
                continue
            findings.append({
                'level': 'INFO' if manual else 'ALERT',
                'kind': 'MANUAL' if manual else 'ORPHAN', 'symbol': sym,
                'broker_qty': b['qty'], 'app_qty': 0,
                'detail': (f"manual holding, {b['qty']} — no book tracks it" if manual else
                           f"broker holds {b['qty']} that no book claims"
                           + (f" (order tag: {', '.join(tags)})" if tags else ' (no order tag)')),
            })
        elif c['qty'] != b['qty']:
            tags = tagged.get(sym, {})
            tag_net = sum(tags.values())
            named = ', '.join(f'{k} {v:+d}' for k, v in sorted(tags.items(), key=lambda kv: kv[1]))
            explained = tag_net == b['qty'] and 'untagged' not in tags
            findings.append({
                'level': 'INFO' if explained else 'ALERT',
                'kind': 'MULTI-BOOK' if explained else 'SIZE', 'symbol': sym,
                'broker_qty': b['qty'], 'app_qty': c['qty'],
                'detail': (f"broker {b['qty']} = {named} — several books share this strike; "
                           f"the page feed only shows {c['qty']}" if explained else
                           f"broker {b['qty']} vs app {c['qty']} "
                           f"({'/'.join(sorted(set(c['books'])))})"
                           + (f" · order tags: {named}" if named else ' · no order tags')),
            })

    for sym, c in claimed.items():
        if c['qty'] and sym not in legs:
            books = sorted(set(c['books']))
            all_paper = books and all(b in PAPER_BOOKS for b in books)
            findings.append({
                'level': 'INFO' if all_paper else 'ALERT',
                'kind': 'PAPER-SHADOW' if all_paper else 'GHOST',
                'symbol': sym, 'broker_qty': 0, 'app_qty': c['qty'],
                'detail': (f"{'/'.join(books)} records {c['qty']} on paper today "
                           f"(no broker leg expected)" if all_paper else
                           f"{'/'.join(books)} records {c['qty']}, broker holds none"),
            })

    # naked shorts: an option short with no protective order resting
    protective = {o['symbol'] for o in orders
                  if o['status'] in ('TRIGGER PENDING', 'OPEN')
                  and (o['type'] or '').upper().startswith('SL')}
    for sym, b in legs.items():
        if b['qty'] < 0 and b['exchange'] in ('NFO', 'BFO') and sym not in protective:
            findings.append({
                'level': 'WARN', 'kind': 'NAKED', 'symbol': sym,
                'broker_qty': b['qty'], 'app_qty': (claimed.get(sym) or {}).get('qty', 0),
                'detail': 'short option with no SL resting at the exchange '
                          '(software-side stop only)',
            })

    by_tag = defaultdict(int)
    for o in orders:
        if o['status'] == 'COMPLETE' and o['qty']:
            by_tag[o['tag'] or 'untagged'] += o['qty']

    return {
        'generated_at': IST.isoformat(timespec='seconds'),
        'broker_legs': legs,
        'app_claimed': {k: v for k, v in claimed.items() if v['qty']},
        'findings': findings,
        'filled_by_tag': dict(sorted(by_tag.items(), key=lambda kv: -kv[1])),
        'alerts': sum(1 for f in findings if f['level'] == 'ALERT'),
        'warns': sum(1 for f in findings if f['level'] == 'WARN'),
        'infos': sum(1 for f in findings if f['level'] == 'INFO'),
    }


def render(rep) -> str:
    lines = [f"Live vs app reconciliation · {rep['generated_at'][:16].replace('T', ' ')} IST",
             f"{len(rep['broker_legs'])} broker legs · {rep['alerts']} alerts · {rep['warns']} warnings", '']
    if not rep['findings']:
        lines.append('CLEAN — every broker leg matches what the app records.')
    for f in rep['findings']:
        lines.append(f"[{f['level']}] {f['kind']} {f['symbol']}: {f['detail']}")
    if rep['filled_by_tag']:
        lines += ['', 'Filled today by tag:']
        lines += [f"  {t or 'untagged':24s} {q:6d}" for t, q in rep['filled_by_tag'].items()]
    return '\n'.join(lines)


def main():
    quiet = '--quiet' in sys.argv
    rep = reconcile()
    body = render(rep)
    print(body)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rep, indent=1))

    LOG_MD.parent.mkdir(parents=True, exist_ok=True)
    if not LOG_MD.exists():
        LOG_MD.write_text('# Live vs app reconciliation — run log\n\n'
                          'Twice daily (11:00, 14:00 IST). Source: scripts/live_vs_app_recon.py\n\n')
    with LOG_MD.open('a') as fh:
        fh.write(f"\n## {rep['generated_at'][:16].replace('T', ' ')} IST — "
                 f"{rep['alerts']} alerts, {rep['warns']} warnings\n\n```\n{body}\n```\n")

    if quiet and not (rep['alerts'] or rep['warns']):
        print('(quiet: nothing to notify)')
        return 0
    try:
        from services.notifications import get_notification_service
        svc = get_notification_service()
        title = (f"Recon: {rep['alerts']} alert(s), {rep['warns']} warning(s)"
                 if rep['findings'] else 'Recon CLEAN — broker matches the app')
        svc.send_alert('live_recon', title, body,
                       data={'alerts': rep['alerts'], 'warns': rep['warns']},
                       priority='high' if rep['alerts'] else 'normal')
        print('notified: email + whatsapp')
    except Exception as e:
        print(f'[warn] notification failed: {e}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
