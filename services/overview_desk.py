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

# ---------------------------------------------------------------------------
# Equity: weekly candles of cumulative P&L + index overlays
# ---------------------------------------------------------------------------
OVERLAYS = [
    ('NIFTY50', 'Nifty 50'),
    ('NIFTYMIDCAP150', 'Midcap 150'),
    ('NIFTYSMLCAP250', 'Smallcap 250'),
    ('NIFTY500', 'Nifty 500'),
]


def _journal_daily():
    """[(date, net)] per trading day, oldest first."""
    import sqlite3
    db = BD / 'journal.db'
    if not db.exists():
        return []
    con = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
    try:
        rows = con.execute(
            "SELECT date(entry_time) d, COALESCE(SUM(pnl_net), 0) n "
            "FROM journal_trades WHERE entry_time IS NOT NULL "
            "GROUP BY 1 ORDER BY 1"
        ).fetchall()
    finally:
        con.close()
    return [(r[0], float(r[1] or 0)) for r in rows if r[0]]


def _index_series(sym, start):
    import sqlite3
    db = BD / 'market_data.db'
    if not db.exists():
        return []
    con = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
    try:
        rows = con.execute(
            "SELECT date, close FROM market_data_unified "
            "WHERE symbol=? AND timeframe='day' AND date >= ? ORDER BY date",
            (sym, start)).fetchall()
    finally:
        con.close()
    return [(r[0][:10], float(r[1])) for r in rows if r[1]]


def equity(weeks: int = 52):
    """Weekly candles on the cumulative curve + index overlays, same weeks."""
    from datetime import date as _date
    daily = _journal_daily()
    if not daily:
        return {'candles': [], 'overlays': {}, 'labels': dict(OVERLAYS)}

    # cumulative, then bucket by ISO week
    cum, curve = 0.0, []
    for d, n in daily:
        cum += n
        curve.append((d, cum))

    buckets = {}
    for d, c in curve:
        y, w, _ = _date.fromisoformat(d).isocalendar()
        buckets.setdefault((y, w), []).append((d, c))

    keys = sorted(buckets)[-weeks:]
    candles, prev_close = [], None
    for k in keys:
        pts = buckets[k]
        vals = [c for _, c in pts]
        o = prev_close if prev_close is not None else vals[0]
        candles.append({
            'd': pts[0][0], 'end': pts[-1][0],
            'o': round(o, 1), 'h': round(max(vals + [o]), 1),
            'l': round(min(vals + [o]), 1), 'c': round(vals[-1], 1),
            'n': len(pts),
        })
        prev_close = vals[-1]

    if not candles:
        return {'candles': [], 'overlays': {}, 'labels': dict(OVERLAYS)}

    start = candles[0]['d']
    overlays = {}
    for sym, _label in OVERLAYS:
        s = _index_series(sym, start)
        if len(s) < 2:
            continue
        base = s[0][1]
        weekly = {}
        for d, v in s:
            y, w, _ = _date.fromisoformat(d).isocalendar()
            weekly[(y, w)] = (d, v)
        pts = []
        for k in keys:
            if k in weekly:
                d, v = weekly[k]
                pts.append({'d': d, 'p': round((v / base - 1) * 100, 2)})
        if len(pts) > 1:
            overlays[sym] = pts

    return {'candles': candles, 'overlays': overlays, 'labels': dict(OVERLAYS),
            'weeks': len(candles)}


@overview_bp.route('/equity', methods=['GET'])
def api_overview_equity():
    from flask import request
    try:
        weeks = max(8, min(260, int(request.args.get('weeks', 52))))
    except Exception:
        weeks = 52
    try:
        return jsonify(equity(weeks))
    except Exception as e:
        logger.error('[desk] equity failed: %s', e, exc_info=True)
        return jsonify({'error': str(e), 'candles': [], 'overlays': {}}), 500
