"""Blueprints tab — strategy rules and backtest metrics."""
from flask import Blueprint, render_template
from ..config import BLUEPRINTS

bp = Blueprint('blueprints', __name__)


@bp.route('/blueprints')
@bp.route('/blueprints/<strategy_id>')
def index(strategy_id=None):
    active_bp = None
    if strategy_id:
        active_bp = next((b for b in BLUEPRINTS if b['id'] == strategy_id), None)
    if not active_bp:
        active_bp = BLUEPRINTS[0]
    return render_template('blueprints.html', blueprints=BLUEPRINTS,
                           active_bp=active_bp, active_tab='blueprints')
