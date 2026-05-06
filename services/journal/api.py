"""
Flask blueprint for the trading journal.

Mount via: app.register_blueprint(journal_bp)
All routes under /api/journal/*.
"""

from __future__ import annotations
from flask import Blueprint, jsonify, request
from datetime import datetime, date
from typing import Any, Dict
import logging

from .journal_db import get_journal_db
from .sync import sync_all

logger = logging.getLogger(__name__)

journal_bp = Blueprint('journal', __name__, url_prefix='/api/journal')


def _err(msg: str, code: int = 400):
    return jsonify({'error': msg}), code


@journal_bp.route('/sync', methods=['POST'])
def api_sync():
    """Force a re-sync from all source DBs."""
    counts = sync_all()
    return jsonify({'ok': True, 'synced': counts})


@journal_bp.route('/trades', methods=['GET'])
def api_list_trades():
    db = get_journal_db()
    args = request.args
    trades = db.list_trades(
        date_from=args.get('from') or None,
        date_to=args.get('to') or None,
        strategy=args.get('strategy') or None,
        instrument=args.get('instrument') or None,
        mode=args.get('mode') or None,
        limit=int(args.get('limit') or 200),
        offset=int(args.get('offset') or 0),
    )
    return jsonify({'trades': trades, 'count': len(trades)})


@journal_bp.route('/trades/<int:trade_id>', methods=['GET'])
def api_get_trade(trade_id: int):
    db = get_journal_db()
    trade = db.get_trade(trade_id)
    if not trade:
        return _err('Trade not found', 404)
    return jsonify(trade)


@journal_bp.route('/trades/<int:trade_id>', methods=['PATCH'])
def api_update_trade(trade_id: int):
    body = request.get_json(silent=True) or {}
    db = get_journal_db()
    if not db.get_trade(trade_id):
        return _err('Trade not found', 404)

    # Top-level fields go to journal_trades
    field_payload: Dict[str, Any] = {}
    for k in ('grade', 'mistake_flag', 'mode', 'exit_price', 'exit_reason'):
        if k in body:
            field_payload[k] = body[k]
    if field_payload:
        db.update_trade(trade_id, field_payload)

    # Notes
    if 'notes' in body and body['notes'] is not None:
        db.save_note(trade_id, str(body['notes']))

    # Tags (replace-all semantics if provided)
    if 'tag_ids' in body and isinstance(body['tag_ids'], list):
        # Reset by detaching all existing tags then attaching the new set
        existing = [t['id'] for t in (db.get_trade(trade_id) or {}).get('tags', [])]
        for t in existing:
            db.detach_tag(trade_id, t)
        db.attach_tags(trade_id, [int(t) for t in body['tag_ids']])

    return jsonify(db.get_trade(trade_id))


@journal_bp.route('/trades/<int:trade_id>/notes', methods=['POST'])
def api_save_note(trade_id: int):
    body = request.get_json(silent=True) or {}
    md = body.get('body_md', '')
    db = get_journal_db()
    if not db.get_trade(trade_id):
        return _err('Trade not found', 404)
    db.save_note(trade_id, str(md))
    return jsonify({'ok': True})


@journal_bp.route('/trades/<int:trade_id>/tags', methods=['POST'])
def api_attach_tags(trade_id: int):
    body = request.get_json(silent=True) or {}
    ids = body.get('tag_ids') or []
    db = get_journal_db()
    if not db.get_trade(trade_id):
        return _err('Trade not found', 404)
    db.attach_tags(trade_id, [int(t) for t in ids])
    return jsonify({'ok': True})


@journal_bp.route('/trades/<int:trade_id>/tags/<int:tag_id>', methods=['DELETE'])
def api_detach_tag(trade_id: int, tag_id: int):
    db = get_journal_db()
    db.detach_tag(trade_id, tag_id)
    return jsonify({'ok': True})


@journal_bp.route('/tags', methods=['GET'])
def api_list_tags():
    db = get_journal_db()
    return jsonify({'tags': db.list_tags()})


@journal_bp.route('/tags', methods=['POST'])
def api_create_tag():
    body = request.get_json(silent=True) or {}
    name = (body.get('name') or '').strip()
    category = (body.get('category') or 'CUSTOM').upper()
    color = body.get('color_hex')
    if not name:
        return _err('Tag name required')
    db = get_journal_db()
    tag_id = db.create_tag(name, category, color)
    return jsonify({'ok': True, 'id': tag_id})


@journal_bp.route('/summary', methods=['GET'])
def api_summary():
    """Calendar-grid bundle: daily summary + month metrics.

    Query: from=YYYY-MM-DD, to=YYYY-MM-DD, strategy=...
    """
    args = request.args
    db = get_journal_db()
    df = args.get('from') or _today_minus_days(60)
    dt = args.get('to') or _today()
    strategy = args.get('strategy') or None
    days = db.daily_summary(df, dt, strategy)

    # Month-level metrics
    total_trades = sum(d['trades'] for d in days)
    total_wins = sum(d['wins'] or 0 for d in days)
    total_losses = sum(d['losses'] or 0 for d in days)
    pnl_net = round(sum(d['pnl_net'] or 0 for d in days), 2)
    pnl_gross = round(sum(d['pnl_gross'] or 0 for d in days), 2)
    best_day = max(days, key=lambda d: d['pnl_net'] or 0, default=None)
    worst_day = min(days, key=lambda d: d['pnl_net'] or 0, default=None)
    sum_wins = 0.0
    sum_losses = 0.0
    avg_win = 0.0
    avg_loss = 0.0
    # Approximate win/loss avg from per-day summaries — not as accurate as
    # per-trade pass, but adequate for the calendar header strip.
    for d in days:
        if (d['pnl_net'] or 0) > 0:
            sum_wins += d['pnl_net']
        elif (d['pnl_net'] or 0) < 0:
            sum_losses += d['pnl_net']
    pf = round(sum_wins / abs(sum_losses), 2) if sum_losses < 0 else None
    win_rate = round(total_wins / total_trades * 100, 2) if total_trades else 0
    if total_wins:
        avg_win = round(sum_wins / total_wins, 2)
    if total_losses:
        avg_loss = round(abs(sum_losses) / total_losses, 2)

    return jsonify({
        'from': df,
        'to': dt,
        'days': days,
        'metrics': {
            'pnl_net': pnl_net,
            'pnl_gross': pnl_gross,
            'trades': total_trades,
            'wins': total_wins,
            'losses': total_losses,
            'win_rate': win_rate,
            'profit_factor': pf,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'best_day': best_day,
            'worst_day': worst_day,
        },
    })


@journal_bp.route('/day/<date_str>', methods=['GET'])
def api_day(date_str: str):
    """Bundle for the day page."""
    db = get_journal_db()
    trades = db.list_trades(date_from=date_str, date_to=date_str, limit=500)
    review = db.get_daily_review(date_str)
    pnl_net = round(sum(t['pnl_net'] or 0 for t in trades), 2)
    pnl_gross = round(sum(t['pnl_gross'] or 0 for t in trades), 2)
    wins = sum(1 for t in trades if (t['pnl_net'] or 0) > 0)
    losses = sum(1 for t in trades if (t['pnl_net'] or 0) < 0)
    return jsonify({
        'date': date_str,
        'trades': trades,
        'review': review,
        'metrics': {
            'pnl_net': pnl_net,
            'pnl_gross': pnl_gross,
            'trades_count': len(trades),
            'wins': wins,
            'losses': losses,
        },
    })


@journal_bp.route('/day/<date_str>/review', methods=['POST'])
def api_save_day_review(date_str: str):
    body = request.get_json(silent=True) or {}
    db = get_journal_db()
    db.save_daily_review(date_str, body)
    return jsonify({'ok': True, 'review': db.get_daily_review(date_str)})


@journal_bp.route('/insights', methods=['GET'])
def api_insights():
    args = request.args
    db = get_journal_db()
    df = args.get('from') or _today_minus_days(180)
    dt = args.get('to') or _today()
    strategy = args.get('strategy') or None
    eq = db.equity_curve(df, dt, strategy)
    attribution = db.per_strategy_attribution(df, dt)
    rdist = db.r_distribution(df, dt, strategy)
    wr_by_tag = db.win_rate_by_tag(df, dt)
    drawdowns = db.drawdown_windows(df, dt)
    # Headline metrics
    total_trades = sum(a['trades'] for a in attribution)
    pnl_net = round(sum(a['pnl_net'] or 0 for a in attribution), 2)
    wins = sum(a['wins'] or 0 for a in attribution)
    losses = sum(a['losses'] or 0 for a in attribution)
    win_rate = round(wins / total_trades * 100, 2) if total_trades else 0
    cum = [r['cum_net'] for r in eq]
    max_dd = 0.0
    if cum:
        peak = cum[0]
        for v in cum:
            if v > peak:
                peak = v
            dd = peak - v
            if dd > max_dd:
                max_dd = dd
    sw = sum(max(0, a['pnl_net'] or 0) for a in attribution)
    sl = sum(min(0, a['pnl_net'] or 0) for a in attribution)
    pf = round(sw / abs(sl), 2) if sl < 0 else None
    expectancy_r = None
    if rdist:
        expectancy_r = round(sum(x['r'] for x in rdist) / len(rdist), 3)

    return jsonify({
        'from': df,
        'to': dt,
        'metrics': {
            'trades': total_trades,
            'pnl_net': pnl_net,
            'win_rate': win_rate,
            'profit_factor': pf,
            'expectancy_r': expectancy_r,
            'max_drawdown': round(max_dd, 2),
        },
        'equity_curve': eq,
        'drawdowns': drawdowns,
        'per_strategy': attribution,
        'r_distribution': rdist,
        'win_rate_by_tag': wr_by_tag,
    })


def _today() -> str:
    return date.today().isoformat()


def _today_minus_days(n: int) -> str:
    from datetime import timedelta
    return (date.today() - timedelta(days=n)).isoformat()
