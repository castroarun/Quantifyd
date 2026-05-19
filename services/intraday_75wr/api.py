"""Intraday 75WR — Flask API blueprint.

Mounts at /api/intraday75wr/*. All endpoints work in paper or live mode;
the blueprint NEVER bypasses the engine's paper/live gating.

Endpoints:
  GET  /state                       — current mode + P&L per config
  GET  /positions                   — open + recent closed
  GET  /trades                      — closed trades
  GET  /signals                     — signal log (incl. blocked)
  GET  /equity-curve                — daily P&L per config + combined
  POST /toggle-mode                 — flip 3-state for one config
  POST /kill-switch                 — square off + halt
  POST /scan                        — manual scan trigger (debug)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime
from pathlib import Path

from flask import Blueprint, jsonify, request

logger = logging.getLogger(__name__)

intraday75wr_bp = Blueprint(
    'intraday75wr', __name__, url_prefix='/api/intraday75wr',
)

# Mode-override JSON file. We don't mutate config.py at runtime (risky).
# Instead, the toggle-mode endpoint writes here, and the engine reads
# this on each scan via _resolve_runtime_cfg() in this module.
_MODE_OVERRIDE_PATH = Path(__file__).parent.parent.parent / \
    'backtest_data' / 'intraday_75wr_mode_overrides.json'


def _read_overrides() -> dict:
    if not _MODE_OVERRIDE_PATH.exists():
        return {}
    try:
        with open(_MODE_OVERRIDE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f) or {}
    except Exception as e:
        logger.warning(f'[I75 api] override read err: {e}')
        return {}


def _write_overrides(d: dict) -> None:
    _MODE_OVERRIDE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_MODE_OVERRIDE_PATH, 'w', encoding='utf-8') as f:
        json.dump(d, f, indent=2)


def _apply_overrides_to_engine(engine):
    """Read JSON overrides + apply them to the engine.cfg dict in-place.
    Safe to call at any tick — reads are O(1) on a small JSON file.
    """
    over = _read_overrides()
    cid = engine.config_id
    cfg_over = over.get(cid)
    if not cfg_over:
        return
    for k in ('enabled', 'paper_trading_mode', 'live_trading_enabled'):
        if k in cfg_over:
            engine.cfg[k] = bool(cfg_over[k])


# ---------------------------------------------------------------------------
# State / introspection
# ---------------------------------------------------------------------------

def _summarise_engine(engine) -> dict:
    _apply_overrides_to_engine(engine)
    cfg = engine.cfg
    pnl = engine.compute_daily_pnl()
    conc = engine.concurrency_check()
    return {
        'config_id': engine.config_id,
        'config_name': cfg.get('config_name'),
        'mode': _resolve_mode_label(engine),
        'enabled': bool(cfg.get('enabled', False)),
        'paper_trading_mode': bool(cfg.get('paper_trading_mode', True)),
        'live_trading_enabled': bool(cfg.get('live_trading_enabled', False)),
        'is_live_authorized': engine.is_live(),
        'capital': cfg.get('capital'),
        'risk_per_trade_rs': cfg.get('risk_per_trade_rs'),
        'tp_pct': cfg.get('tp_pct'),
        'sl_pct': cfg.get('sl_pct'),
        'daily_loss_limit_rs': cfg.get('daily_loss_limit_rs'),
        'pnl': pnl,
        'concurrency': conc,
        'killed_today': engine.is_killed_today(),
    }


def _resolve_mode_label(engine) -> str:
    """Reduce the 3 flags to 'off' / 'paper' / 'live'."""
    if not engine.is_enabled():
        return 'off'
    if engine.is_live():
        return 'live'
    return 'paper'


@intraday75wr_bp.route('/state', methods=['GET'])
def state():
    try:
        from services.intraday_75wr import get_all_engines
        engines = get_all_engines()
        out = {cid: _summarise_engine(eng) for cid, eng in engines.items()}
        # Combined snapshot
        try:
            any_eng = next(iter(engines.values()))
            combined_open = any_eng.total_open_across_configs()
        except Exception:
            combined_open = 0
        try:
            from config import INTRADAY_75WR_COMBINED_MAX_CONCURRENT as cap
        except Exception:
            cap = 5
        return jsonify({
            'configs': out,
            'combined': {
                'open_count': combined_open,
                'cap': cap,
                'allowed': combined_open < cap,
            },
            'as_of': datetime.now().isoformat(),
        })
    except Exception as e:
        logger.error(f'[I75 api] /state err: {e}', exc_info=True)
        return jsonify({'error': str(e)}), 500


@intraday75wr_bp.route('/positions', methods=['GET'])
def positions():
    cid = (request.args.get('config') or '').upper().strip()
    include_closed = request.args.get('include_closed', '0') in ('1', 'true', 'True')
    try:
        from services.intraday_75wr.db import get_i75_db
        db = get_i75_db()
        prefixes = [cid] if cid else ['A', 'B', 'C']
        out = {'open': [], 'closed': []}
        # Open
        with db.db_lock:
            conn = db._get_conn()
            try:
                like_clauses = ' OR '.join(['system_id LIKE ?' for _ in prefixes])
                params = [f'{p}%' for p in prefixes]
                rows = conn.execute(
                    f'SELECT * FROM i75_positions WHERE status="OPEN" '
                    f'AND ({like_clauses}) ORDER BY entry_time DESC',
                    params,
                ).fetchall()
                out['open'] = [dict(r) for r in rows]
                if include_closed:
                    today = date.today().isoformat()
                    rows = conn.execute(
                        f'SELECT * FROM i75_positions WHERE status="CLOSED" '
                        f'AND trade_date=? AND ({like_clauses}) '
                        f'ORDER BY exit_time DESC',
                        [today, *params],
                    ).fetchall()
                    out['closed'] = [dict(r) for r in rows]
            finally:
                conn.close()
        return jsonify(out)
    except Exception as e:
        logger.error(f'[I75 api] /positions err: {e}', exc_info=True)
        return jsonify({'error': str(e)}), 500


@intraday75wr_bp.route('/trades', methods=['GET'])
def trades():
    cid = (request.args.get('config') or '').upper().strip()
    date_from = request.args.get('from')
    date_to = request.args.get('to')
    day = request.args.get('date')
    try:
        from services.intraday_75wr.db import get_i75_db
        db = get_i75_db()
        prefixes = [cid] if cid else ['A', 'B', 'C']
        like_clauses = ' OR '.join(['system_id LIKE ?' for _ in prefixes])
        params = [f'{p}%' for p in prefixes]
        sql = (
            f'SELECT * FROM i75_positions WHERE status="CLOSED" '
            f'AND ({like_clauses})'
        )
        if day:
            sql += ' AND trade_date=?'
            params.append(day)
        else:
            if date_from:
                sql += ' AND trade_date >= ?'
                params.append(date_from)
            if date_to:
                sql += ' AND trade_date <= ?'
                params.append(date_to)
        sql += ' ORDER BY exit_time DESC LIMIT 500'
        with db.db_lock:
            conn = db._get_conn()
            try:
                rows = conn.execute(sql, params).fetchall()
                return jsonify({'trades': [dict(r) for r in rows]})
            finally:
                conn.close()
    except Exception as e:
        logger.error(f'[I75 api] /trades err: {e}', exc_info=True)
        return jsonify({'error': str(e)}), 500


@intraday75wr_bp.route('/signals', methods=['GET'])
def signals():
    cid = (request.args.get('config') or '').upper().strip()
    day = request.args.get('date') or date.today().isoformat()
    try:
        from services.intraday_75wr.db import get_i75_db
        db = get_i75_db()
        prefixes = [cid] if cid else ['A', 'B', 'C']
        like_clauses = ' OR '.join(['system_id LIKE ?' for _ in prefixes])
        params = [f'{p}%' for p in prefixes]
        sql = (
            f'SELECT * FROM i75_signals WHERE DATE(signal_time)=? '
            f'AND ({like_clauses}) ORDER BY signal_time DESC LIMIT 500'
        )
        with db.db_lock:
            conn = db._get_conn()
            try:
                rows = conn.execute(sql, [day, *params]).fetchall()
                return jsonify({'signals': [dict(r) for r in rows]})
            finally:
                conn.close()
    except Exception as e:
        logger.error(f'[I75 api] /signals err: {e}', exc_info=True)
        return jsonify({'error': str(e)}), 500


@intraday75wr_bp.route('/equity-curve', methods=['GET'])
def equity_curve():
    cid = (request.args.get('config') or '').upper().strip() or None
    date_from = request.args.get('from')
    date_to = request.args.get('to')
    try:
        from services.intraday_75wr.db import get_i75_db
        db = get_i75_db()
        prefixes = [cid] if cid else ['A', 'B', 'C']
        out = {}
        for p in prefixes:
            curve = db.get_equity_curve(system_id=None, start=date_from, end=date_to)
            out[p] = curve
        # Combined curve = sum of A+B+C per day
        combined = {}
        for p in prefixes:
            for row in out.get(p, []) or []:
                d = row['trade_date']
                combined.setdefault(d, 0.0)
                combined[d] += float(row.get('daily_pnl') or 0)
        combined_list = [
            {'trade_date': d, 'daily_pnl': round(v, 2)}
            for d, v in sorted(combined.items())
        ]
        cum = 0.0
        for r in combined_list:
            cum += r['daily_pnl']
            r['cumulative_pnl'] = round(cum, 2)
        return jsonify({'per_config': out, 'combined': combined_list})
    except Exception as e:
        logger.error(f'[I75 api] /equity-curve err: {e}', exc_info=True)
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# Mode toggling
# ---------------------------------------------------------------------------

@intraday75wr_bp.route('/toggle-mode', methods=['POST'])
def toggle_mode():
    """Body: {'config': 'A'|'B'|'C', 'mode': 'off'|'paper'|'live'}.

    Persists to the JSON override file. The engine's next scan tick reads
    the new mode automatically. No backend restart required.
    """
    try:
        body = request.get_json(force=True, silent=True) or {}
        cid = (body.get('config') or '').upper().strip()
        mode = (body.get('mode') or '').lower().strip()
        if cid not in ('A', 'B', 'C'):
            return jsonify({'error': 'config must be A|B|C'}), 400
        if mode not in ('off', 'paper', 'live'):
            return jsonify({'error': 'mode must be off|paper|live'}), 400

        # Compose the 3 flags
        if mode == 'off':
            new = {'enabled': False, 'paper_trading_mode': True,
                   'live_trading_enabled': False}
        elif mode == 'paper':
            new = {'enabled': True, 'paper_trading_mode': True,
                   'live_trading_enabled': False}
        else:  # live — REQUIRES BOTH FLAGS
            new = {'enabled': True, 'paper_trading_mode': False,
                   'live_trading_enabled': True}

        over = _read_overrides()
        over[cid] = new
        _write_overrides(over)

        # Push to live engine
        from services.intraday_75wr import get_engine
        eng = get_engine(cid)
        for k, v in new.items():
            eng.cfg[k] = v

        logger.warning(
            f'[I75 api] Config {cid} mode toggled to {mode!r} (enabled={new["enabled"]}, '
            f'paper={new["paper_trading_mode"]}, live={new["live_trading_enabled"]})'
        )
        return jsonify({'ok': True, 'config': cid, 'mode': mode, 'flags': new})
    except Exception as e:
        logger.error(f'[I75 api] /toggle-mode err: {e}', exc_info=True)
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------------

@intraday75wr_bp.route('/kill-switch', methods=['POST'])
def kill_switch():
    try:
        body = request.get_json(force=True, silent=True) or {}
        cid = (body.get('config') or '').upper().strip()
        if cid not in ('A', 'B', 'C'):
            return jsonify({'error': 'config must be A|B|C'}), 400
        from services.intraday_75wr import get_engine
        eng = get_engine(cid)
        out = eng.kill_switch()
        return jsonify({'ok': True, 'result': out})
    except Exception as e:
        logger.error(f'[I75 api] /kill-switch err: {e}', exc_info=True)
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# Manual scan (debug)
# ---------------------------------------------------------------------------

@intraday75wr_bp.route('/scan', methods=['POST'])
def manual_scan():
    """Body: {'config': 'A'|'B'|'C', 'sub_signal': 'A1'|'A2'|'A3' (optional, A/B only)}."""
    try:
        body = request.get_json(force=True, silent=True) or {}
        cid = (body.get('config') or '').upper().strip()
        sub = (body.get('sub_signal') or '').upper().strip()
        if cid not in ('A', 'B', 'C'):
            return jsonify({'error': 'config must be A|B|C'}), 400

        from services.intraday_75wr import get_engine
        eng = get_engine(cid)
        _apply_overrides_to_engine(eng)

        if cid in ('A', 'B'):
            results: dict = {}
            if sub in ('', f'{cid}1'):
                results['a1'] = eng.scan_a1()
            if sub in ('', f'{cid}2'):
                results['a2'] = eng.scan_a2()
            if sub in ('', f'{cid}3'):
                results['a3'] = eng.scan_a3()
            return jsonify({'ok': True, 'config': cid, 'results': results})
        else:
            results = eng.scan()
            return jsonify({'ok': True, 'config': cid, 'results': results})
    except Exception as e:
        logger.error(f'[I75 api] /scan err: {e}', exc_info=True)
        return jsonify({'error': str(e)}), 500
