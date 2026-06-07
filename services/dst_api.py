"""DST (NIFTY dual-Supertrend paper) dashboard API.

Thin read-only blueprint over the research/56 paper logger DB
(research/56_nifty_dual_supertrend/results/paper_dst.db). The logger
(scripts/nifty_dst_paper.py, cron) computes a rich `snapshot` each run; this
just serves it. Register in app.py:
    from services.dst_api import dst_bp
    app.register_blueprint(dst_bp)
"""
from flask import Blueprint, jsonify
from pathlib import Path
import sqlite3, json

dst_bp = Blueprint('dst', __name__)
DB = (Path(__file__).parent.parent / 'research' /
      '56_nifty_dual_supertrend' / 'results' / 'paper_dst.db')


def _con():
    c = sqlite3.connect(str(DB))
    c.row_factory = sqlite3.Row
    return c


@dst_bp.route('/api/dst/state')
def dst_state():
    """Latest snapshot: regime, status, open book (with MTM), realized summary."""
    try:
        if not DB.exists():
            return jsonify({'snapshot': None,
                            'message': 'Paper logger has not started yet.'})
        c = _con()
        r = c.execute("SELECT v FROM state WHERE k='snapshot'").fetchone()
        c.close()
        return jsonify({'snapshot': json.loads(r['v']) if r else None})
    except Exception as e:  # noqa: BLE001
        return jsonify({'error': str(e)}), 500


@dst_bp.route('/api/dst/trades')
def dst_trades():
    """Closed structures (P&L) + open-events activity log."""
    try:
        if not DB.exists():
            return jsonify({'closed': [], 'opens': []})
        c = _con()
        closed = [dict(x) for x in c.execute(
            "SELECT * FROM closed ORDER BY id DESC LIMIT 300")]
        opens = [dict(x) for x in c.execute(
            "SELECT id,ts,action,role,dir,expiry,cashflow,fees,note "
            "FROM trades ORDER BY id DESC LIMIT 300")]
        c.close()
        return jsonify({'closed': closed, 'opens': opens})
    except Exception as e:  # noqa: BLE001
        return jsonify({'error': str(e)}), 500


@dst_bp.route('/api/dst/equity')
def dst_equity():
    """Cumulative realized-net curve by close time."""
    try:
        if not DB.exists():
            return jsonify([])
        c = _con()
        rows = c.execute(
            "SELECT close_ts, pnl, fees FROM closed ORDER BY id").fetchall()
        c.close()
        cum = 0.0
        out = []
        for r in rows:
            cum += (r['pnl'] or 0) - (r['fees'] or 0)
            out.append({'ts': r['close_ts'], 'net': round(cum)})
        return jsonify(out)
    except Exception as e:  # noqa: BLE001
        return jsonify({'error': str(e)}), 500
