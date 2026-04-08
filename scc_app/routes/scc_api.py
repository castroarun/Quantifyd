"""SCC API routes — aggregated data for the dashboard and proxy to parent app."""
import requests
import logging
from flask import Blueprint, jsonify, request
from ..config import STRATEGY_META, PARENT_API_URL

bp = Blueprint('scc_api', __name__, url_prefix='/api/scc')
logger = logging.getLogger(__name__)


def _proxy_parent(path):
    """Proxy a GET request to the parent Flask app."""
    try:
        resp = requests.get(f'{PARENT_API_URL}{path}', timeout=5)
        return resp.json()
    except Exception as e:
        logger.warning(f"Parent API proxy failed for {path}: {e}")
        return None


@bp.route('/dashboard')
def dashboard():
    """Aggregated dashboard data for all strategies."""
    result = {
        'strategies': {},
        'total_pnl_today': 0,
        'total_pnl': 0,
        'deployed_capital': 0,
        'free_cash': 0,
        'active_strategies': 0,
        'open_positions': 0,
    }

    for sid, meta in STRATEGY_META.items():
        state = _proxy_parent(f'{meta["api_base"]}/state')
        if state:
            result['strategies'][sid] = {
                'name': meta['name'],
                'type': meta['type'],
                'status': state.get('status', 'unknown'),
                'pnl_today': state.get('pnl_today', 0),
                'pnl_total': state.get('pnl_total', 0),
            }

    return jsonify(result)


@bp.route('/positions')
def positions():
    """All open positions across strategies."""
    all_positions = []

    # Try Kite positions directly
    try:
        from services.kite_service import get_kite, is_authenticated
        if is_authenticated():
            kite = get_kite()
            kite_positions = kite.positions().get('net', [])
            for p in kite_positions:
                if p.get('quantity', 0) != 0:
                    all_positions.append({
                        'symbol': p['tradingsymbol'],
                        'exchange': p['exchange'],
                        'qty': p['quantity'],
                        'avg_price': p['average_price'],
                        'ltp': p['last_price'],
                        'pnl': p['pnl'],
                        'product': p.get('product', ''),
                    })
    except Exception as e:
        logger.warning(f"Kite positions fetch failed: {e}")

    return jsonify({'positions': all_positions})


@bp.route('/trades')
def trades():
    """Closed trade history."""
    # Placeholder — will aggregate from strategy DBs
    return jsonify({'trades': []})


@bp.route('/kill-all', methods=['POST'])
def kill_all():
    """Emergency kill switch — close all positions."""
    return jsonify({'status': 'not_implemented', 'message': 'Kill-all requires confirmation flow'})


@bp.route('/proxy/<path:api_path>')
def proxy(api_path):
    """Generic proxy to parent app API."""
    data = _proxy_parent(f'/{api_path}')
    if data is None:
        return jsonify({'error': 'Parent API unavailable'}), 502
    return jsonify(data)
