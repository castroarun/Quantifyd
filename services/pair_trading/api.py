"""Pair-Trading (Config D) — Flask blueprint at /api/pair_trading/*.

Mount via: app.register_blueprint(pair_trading_bp)

All routes return JSON. Auth is intentionally NOT enforced here (mirrors the
public read-only journal pattern); the user-facing pages on /app/pair-trading
will run inside the existing /app login-protected SPA.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Dict

from flask import Blueprint, jsonify, request

from services.pair_trading.cohort import load_cohort
from services.pair_trading.db import get_pair_trading_db
from services.pair_trading.pair_engine import get_pair_engine

logger = logging.getLogger(__name__)

pair_trading_bp = Blueprint('pair_trading', __name__, url_prefix='/api/pair_trading')


def _err(msg: str, code: int = 400):
    return jsonify({'error': msg}), code


# ---------------------------------------------------------------------------
# State / pairs / positions / trades
# ---------------------------------------------------------------------------

@pair_trading_bp.route('/state', methods=['GET'])
def api_state():
    """Current mode + capital + P&L + open pair count."""
    try:
        engine = get_pair_engine()
        mode = engine.current_mode()
        pnl = engine.compute_daily_pnl()
        opens = engine.db.get_open_positions()
        from config import PAIR_TRADING_DEFAULTS
        return jsonify({
            'mode': mode,
            'enabled': bool(PAIR_TRADING_DEFAULTS.get('enabled', False)),
            'paper_trading_mode': bool(PAIR_TRADING_DEFAULTS.get('paper_trading_mode', True)),
            'live_trading_enabled': bool(PAIR_TRADING_DEFAULTS.get('live_trading_enabled', False)),
            'capital': float(PAIR_TRADING_DEFAULTS.get('capital', 0)),
            'risk_per_pair_rs': float(PAIR_TRADING_DEFAULTS.get('risk_per_pair_rs', 0)),
            'max_concurrent': int(PAIR_TRADING_DEFAULTS.get('max_concurrent', 0)),
            'realized_pnl': pnl['realized'],
            'unrealized_pnl': pnl['unrealized'],
            'open_pairs': len(opens),
            'cohort_size': len(engine.cohort.pairs),
        })
    except Exception as e:
        logger.exception(f"pair_trading state error: {e}")
        return _err(str(e), 500)


@pair_trading_bp.route('/pairs', methods=['GET'])
def api_pairs():
    """All 6 pairs + their config + current open status. Optionally
    includes today's z-score if a scan has run."""
    try:
        engine = get_pair_engine()
        opens = engine.db.get_open_positions()
        opens_by_pair = {p['pair_name']: p for p in opens}
        today = date.today().isoformat()
        sigs = engine.db.list_signals(trade_date=today, limit=200)
        # Latest signal per pair
        sig_by_pair: Dict[str, Dict[str, Any]] = {}
        for s in sigs:
            pn = s['pair_name']
            if pn not in sig_by_pair:
                sig_by_pair[pn] = s
        out = []
        for p in engine.cohort.to_dicts():
            opn = opens_by_pair.get(p['name'])
            sig = sig_by_pair.get(p['name'])
            out.append({
                **p,
                'is_open': opn is not None,
                'open_position_id': opn['id'] if opn else None,
                'open_direction': int(opn['direction']) if opn else None,
                'open_entry_z': float(opn['entry_z']) if opn else None,
                'today_z': float(sig['z']) if sig and sig.get('z') is not None else None,
                'today_action': sig['action'] if sig else None,
            })
        return jsonify({'pairs': out, 'count': len(out)})
    except Exception as e:
        logger.exception(f"pair_trading pairs error: {e}")
        return _err(str(e), 500)


@pair_trading_bp.route('/positions', methods=['GET'])
def api_positions():
    """Open + recent closed pair-positions."""
    try:
        engine = get_pair_engine()
        include_closed = (request.args.get('include_closed') or '').lower() in ('1', 'true', 'yes')
        opens = engine.db.get_open_positions()
        result = {'open': opens}
        if include_closed:
            limit = int(request.args.get('limit') or 50)
            result['closed'] = engine.db.get_recent_closed_positions(limit=limit)
        return jsonify(result)
    except Exception as e:
        logger.exception(f"pair_trading positions error: {e}")
        return _err(str(e), 500)


@pair_trading_bp.route('/trades', methods=['GET'])
def api_trades():
    """List closed pair-trades."""
    try:
        engine = get_pair_engine()
        args = request.args
        trades = engine.db.list_trades(
            date_from=args.get('from') or None,
            date_to=args.get('to') or None,
            pair_name=args.get('pair') or None,
            limit=int(args.get('limit') or 200),
        )
        return jsonify({'trades': trades, 'count': len(trades)})
    except Exception as e:
        logger.exception(f"pair_trading trades error: {e}")
        return _err(str(e), 500)


@pair_trading_bp.route('/signals', methods=['GET'])
def api_signals():
    """Per-pair z-score + action history."""
    try:
        engine = get_pair_engine()
        args = request.args
        signals = engine.db.list_signals(
            trade_date=args.get('date') or None,
            pair_name=args.get('pair') or None,
            limit=int(args.get('limit') or 500),
        )
        return jsonify({'signals': signals, 'count': len(signals)})
    except Exception as e:
        logger.exception(f"pair_trading signals error: {e}")
        return _err(str(e), 500)


@pair_trading_bp.route('/equity-curve', methods=['GET'])
def api_equity_curve():
    """Daily NAV equity curve."""
    try:
        engine = get_pair_engine()
        args = request.args
        curve = engine.db.get_equity_curve(
            date_from=args.get('from') or None,
            date_to=args.get('to') or None,
        )
        return jsonify({'curve': curve, 'count': len(curve)})
    except Exception as e:
        logger.exception(f"pair_trading equity-curve error: {e}")
        return _err(str(e), 500)


@pair_trading_bp.route('/orders', methods=['GET'])
def api_orders():
    """Per-leg order audit."""
    try:
        engine = get_pair_engine()
        args = request.args
        position_id = args.get('position_id')
        position_id = int(position_id) if position_id is not None else None
        orders = engine.db.list_orders(
            position_id=position_id,
            limit=int(args.get('limit') or 200),
        )
        return jsonify({'orders': orders, 'count': len(orders)})
    except Exception as e:
        logger.exception(f"pair_trading orders error: {e}")
        return _err(str(e), 500)


# ---------------------------------------------------------------------------
# Mode toggle / kill-switch / manual scan
# ---------------------------------------------------------------------------

@pair_trading_bp.route('/toggle-mode', methods=['POST'])
def api_toggle_mode():
    """Flip the 3-state mode. Body: {'mode': 'off' | 'paper' | 'live'}.
    Persists by mutating the in-process PAIR_TRADING_DEFAULTS dict.
    Note: changes are NOT written back to config.py — they're per-process.
    """
    try:
        body = request.get_json(silent=True) or {}
        mode = (body.get('mode') or 'paper').lower()
        if mode not in ('off', 'paper', 'live'):
            return _err(f"invalid mode: {mode!r}; expected off|paper|live")

        from config import PAIR_TRADING_DEFAULTS
        if mode == 'off':
            PAIR_TRADING_DEFAULTS['enabled'] = False
            PAIR_TRADING_DEFAULTS['paper_trading_mode'] = True
            PAIR_TRADING_DEFAULTS['live_trading_enabled'] = False
            logger.info("PairTrading: mode -> OFF")
        elif mode == 'paper':
            PAIR_TRADING_DEFAULTS['enabled'] = True
            PAIR_TRADING_DEFAULTS['paper_trading_mode'] = True
            PAIR_TRADING_DEFAULTS['live_trading_enabled'] = False
            logger.info("PairTrading: mode -> PAPER")
        else:
            PAIR_TRADING_DEFAULTS['enabled'] = True
            PAIR_TRADING_DEFAULTS['paper_trading_mode'] = False
            PAIR_TRADING_DEFAULTS['live_trading_enabled'] = True
            logger.warning("PairTrading: mode -> LIVE")
        return jsonify({
            'mode': mode,
            'enabled': PAIR_TRADING_DEFAULTS['enabled'],
            'paper_trading_mode': PAIR_TRADING_DEFAULTS['paper_trading_mode'],
            'live_trading_enabled': PAIR_TRADING_DEFAULTS['live_trading_enabled'],
        })
    except Exception as e:
        logger.exception(f"pair_trading toggle-mode error: {e}")
        return _err(str(e), 500)


@pair_trading_bp.route('/kill-switch', methods=['POST'])
def api_kill_switch():
    """Square off all open pairs + disable for the week."""
    try:
        engine = get_pair_engine()
        result = engine.kill_switch()
        return jsonify(result)
    except Exception as e:
        logger.exception(f"pair_trading kill-switch error: {e}")
        return _err(str(e), 500)


@pair_trading_bp.route('/scan', methods=['POST'])
def api_scan():
    """Manual trigger of the daily scan (for debugging)."""
    try:
        engine = get_pair_engine()
        summary = engine.daily_scan()
        return jsonify(summary)
    except Exception as e:
        logger.exception(f"pair_trading scan error: {e}")
        return _err(str(e), 500)
