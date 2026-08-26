"""Desk feed — what is exposed, whether the app is telling the truth, and what
needs attention today. READ-ONLY: reads JSON the jobs already write plus a few
config files. No engine imports, no writes, no orders.

Serves /api/overview/desk for the Overview page:
  exposure   open broker legs, live vs paper, naked shorts
  recon      last live-vs-app run: alerts, warnings, age
  gates      breakout regime gate; NAS day/gap matrix live-vs-paper today
  ops        reviews due or overdue
  health     watchdog + the morning token chain
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

from flask import Blueprint, jsonify

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
BD = ROOT / 'backtest_data'
SA = ROOT / 'static' / 'app'

overview_bp = Blueprint('overview_desk', __name__, url_prefix='/api/overview')


def _json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default if default is not None else {}


def _age_mins(iso: str) -> int | None:
    try:
        return int((datetime.now() - datetime.fromisoformat(iso)).total_seconds() // 60)
    except Exception:
        return None


def exposure() -> Dict[str, Any]:
    rec = _json(SA / 'live_recon.json')
    legs = rec.get('broker_legs') or {}
    opts = {k: v for k, v in legs.items() if v.get('exchange') in ('NFO', 'BFO')}
    cash = {k: v for k, v in legs.items() if v.get('exchange') == 'NSE'}
    naked = [f['symbol'] for f in (rec.get('findings') or []) if f.get('kind') == 'NAKED']
    return {
        'as_of': rec.get('generated_at'),
        'age_mins': _age_mins(rec.get('generated_at', '')),
        'option_legs': len(opts),
        'cash_legs': len(cash),
        'short_option_legs': sum(1 for v in opts.values() if v.get('qty', 0) < 0),
        'naked': naked,
        'naked_count': len(naked),
        'open_pnl': round(sum(float(v.get('pnl') or 0) for v in opts.values()), 0),
        'legs': [{'symbol': k, **v} for k, v in sorted(opts.items())],
    }


def recon() -> Dict[str, Any]:
    rec = _json(SA / 'live_recon.json')
    findings = rec.get('findings') or []
    return {
        'as_of': rec.get('generated_at'),
        'age_mins': _age_mins(rec.get('generated_at', '')),
        'alerts': rec.get('alerts', 0),
        'warns': rec.get('warns', 0),
        'clean': not rec.get('alerts'),
        'items': [f for f in findings if f.get('level') in ('ALERT', 'WARN')][:8],
        'filled_by_tag': rec.get('filled_by_tag') or {},
    }


def gates() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    bp = _json(BD / 'breakout_paper.db.json', {})          # not used; kept for shape stability
    try:
        import sqlite3
        con = sqlite3.connect(f"file:{BD / 'breakout_paper.db'}?mode=ro", uri=True)
        row = con.execute("select value from bp_state where key='gate'").fetchone()
        con.close()
        if row:
            g = json.loads(row[0]) if row[0].startswith('{') else {'gate': row[0]}
            out['breakout'] = g
    except Exception as e:
        logger.debug('[desk] breakout gate: %s', e)

    matrix = _json(BD / 'nas_day_matrix.json')
    systems = matrix.get('systems') or {}
    out['nas_matrix'] = {
        'live': sorted(k for k, v in systems.items() if v.get('live')),
        'paper': sorted(k for k, v in systems.items() if not v.get('live')),
    }
    master = _json(BD / 'nas_master_mode.json')
    out['nas_master_mode'] = master.get('mode')
    freeze = (BD / 'nas_manual_freeze.flag').exists()
    out['freeze_flag'] = freeze
    return out


def ops() -> Dict[str, Any]:
    oc = _json(SA / 'straddles' / 'ops_center.json')
    reviews = oc.get('reviews') or []
    today = datetime.now().date()
    soon, overdue = [], []
    for r in reviews:
        due = r.get('due')
        try:
            d = datetime.fromisoformat(str(due)[:10]).date()
        except Exception:
            continue
        if r.get('status') in ('DONE', 'CLOSED'):
            continue
        delta = (d - today).days
        item = {'title': r.get('title'), 'due': str(due)[:10],
                'in_days': delta, 'status': r.get('status')}
        if delta < 0:
            overdue.append(item)
        elif delta <= 7:
            soon.append(item)
    overdue.sort(key=lambda x: x['in_days'])
    soon.sort(key=lambda x: x['in_days'])
    return {'as_of': oc.get('generated_at'), 'overdue': overdue[:6],
            'due_soon': soon[:6], 'tracked': len(reviews)}


def health() -> Dict[str, Any]:
    wd = _json(SA / 'watchdog.json')
    out = {'as_of': wd.get('polled_at'), 'summary': wd.get('summary')}
    # the morning token chain — the guards that keep the live book alive
    logs = ROOT / 'logs'
    chain = {}
    for name, f in (('auto_login', ROOT / 'auto_login.log'),
                    ('token_heal', logs / 'token_heal.log'),
                    ('preopen_restart', logs / 'preopen_restart.log')):
        try:
            ts = datetime.fromtimestamp(f.stat().st_mtime)
            chain[name] = {'last': ts.isoformat(timespec='minutes'),
                           'today': ts.date() == datetime.now().date()}
        except Exception:
            chain[name] = {'last': None, 'today': False}
    out['token_chain'] = chain
    tok = BD / 'access_token.json'
    try:
        out['token_file_age_mins'] = _age_mins(
            datetime.fromtimestamp(tok.stat().st_mtime).isoformat())
    except Exception:
        out['token_file_age_mins'] = None
    return out


@overview_bp.route('/desk', methods=['GET'])
def api_overview_desk():
    try:
        return jsonify({
            'generated_at': datetime.now().isoformat(timespec='seconds'),
            'exposure': exposure(),
            'recon': recon(),
            'gates': gates(),
            'ops': ops(),
            'health': health(),
        })
    except Exception as e:
        logger.error('[desk] failed: %s', e, exc_info=True)
        return jsonify({'error': str(e)}), 500
