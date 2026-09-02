#!/usr/bin/env python3
"""Patch app.py: account-aware buy + new exit + funds routes for the Holdings page."""
import io

APP = "/home/arun/quantifyd/app.py"
src = io.open(APP, encoding="utf-8").read()

OLD = """@app.route('/api/holdings/order', methods=['POST'])
@login_required
def api_holdings_topup():
    \"\"\"Smart-limit CNC top-up of an existing holding (manual buy from the chart wall).\"\"\"
    try:
        from services.holdings_order import start_topup
        data = request.get_json(silent=True) or {}
        symbol = (data.get('symbol') or '').strip().upper()
        amount = data.get('amount')
        paper = bool(data.get('paper', False))
        try:
            from services.kite_service import get_kite
            _k = get_kite()
            syms = {h.get('tradingsymbol') for h in (_k.holdings() or [])}
        except Exception as he:
            logger.error(f"[topup] holdings fetch failed: {he}")
            syms = set()
        res = start_topup(symbol, amount, syms, paper=paper)
        return jsonify(res), (400 if res.get('error') else 200)
    except Exception as e:
        logger.exception("[topup] route error")
        return jsonify({'error': str(e)}), 500


@app.route('/api/holdings/order/<exec_id>')
@login_required
def api_holdings_topup_status(exec_id):
    from services.holdings_order import get_status
    s = get_status(exec_id)
    if not s:
        return jsonify({'error': 'not found'}), 404
    return jsonify(s)"""

NEW = """def _holdings_read_kite(account):
    \"\"\"Read-only Kite client for the requested account ('me' | 'dad').\"\"\"
    if account == 'dad':
        from services.dad_kite import get_dad_kite
        return get_dad_kite()
    from services.kite_service import get_kite
    return get_kite()


def _account_holdings(account):
    \"\"\"{symbol: qty} for the requested account. Empty on any failure.\"\"\"
    try:
        k = _holdings_read_kite(account)
        return {h.get('tradingsymbol'): int(h.get('quantity') or 0)
                for h in (k.holdings() or []) if (h.get('quantity') or 0) > 0}
    except Exception as he:
        logger.error(f"[order] {account} holdings fetch failed: {he}")
        return {}


@app.route('/api/holdings/order', methods=['POST'])
@login_required
def api_holdings_topup():
    \"\"\"Smart-limit CNC top-up of an existing holding (manual buy from the chart wall).\"\"\"
    try:
        from services.holdings_order import start_topup
        data = request.get_json(silent=True) or {}
        account = 'dad' if data.get('account') == 'dad' else 'me'
        symbol = (data.get('symbol') or '').strip().upper()
        amount = data.get('amount')
        paper = bool(data.get('paper', False))
        syms = set(_account_holdings(account).keys())
        res = start_topup(symbol, amount, syms, paper=paper, account=account)
        return jsonify(res), (400 if res.get('error') else 200)
    except Exception as e:
        logger.exception("[topup] route error")
        return jsonify({'error': str(e)}), 500


@app.route('/api/holdings/exit', methods=['POST'])
@login_required
def api_holdings_exit():
    \"\"\"Market SELL of an existing holding (full or partial exit) — user-driven.\"\"\"
    try:
        from services.holdings_order import start_exit
        data = request.get_json(silent=True) or {}
        account = 'dad' if data.get('account') == 'dad' else 'me'
        symbol = (data.get('symbol') or '').strip().upper()
        qty = data.get('qty')
        paper = bool(data.get('paper', False))
        held = _account_holdings(account).get(symbol, 0)
        res = start_exit(symbol, qty, held, account=account, paper=paper)
        return jsonify(res), (400 if res.get('error') else 200)
    except Exception as e:
        logger.exception("[exit] route error")
        return jsonify({'error': str(e)}), 500


@app.route('/api/holdings/order/<exec_id>')
@login_required
def api_holdings_topup_status(exec_id):
    from services.holdings_order import get_status
    s = get_status(exec_id)
    if not s:
        return jsonify({'error': 'not found'}), 404
    return jsonify(s)


@app.route('/api/holdings/funds')
@login_required
def api_holdings_funds():
    \"\"\"Available funds for the requested account. 'available' matches Kite's
    'Available margin' (equity.net); 'cash'/'live_balance' are detail.\"\"\"
    account = 'dad' if request.args.get('account') == 'dad' else 'me'
    try:
        kite = _holdings_read_kite(account)
        eq = (kite.margins() or {}).get('equity', {})
        avail = eq.get('available', {})
        return jsonify({
            'account': account,
            'available': round(eq.get('net', 0) or 0, 2),
            'cash': round(avail.get('cash', 0) or 0, 2),
            'live_balance': round(avail.get('live_balance', 0) or 0, 2),
            'used': round(eq.get('utilised', {}).get('debits', 0) or 0, 2),
        })
    except Exception as e:  # noqa: BLE001
        logger.error(f"[funds] {account} fetch failed: {e}")
        return jsonify({'account': account, 'error': str(e)}), 200"""

n = src.count(OLD)
if n != 1:
    raise SystemExit(f"ABORT: anchor found {n} times, expected 1")
src = src.replace(OLD, NEW)
io.open(APP, "w", encoding="utf-8").write(src)

import ast
ast.parse(src)
print("PATCH OK — routes added (exit, funds), buy is account-aware")
