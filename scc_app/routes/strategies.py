"""Individual strategy pages — proxied from parent app."""
from flask import Blueprint, render_template
from ..config import STRATEGY_META

bp = Blueprint('strategies', __name__)


@bp.route('/strategy/<strategy_id>')
def detail(strategy_id):
    meta = STRATEGY_META.get(strategy_id)
    if not meta:
        return render_template('error.html', message=f'Strategy {strategy_id} not found'), 404
    return render_template(f'strategies/{strategy_id}.html',
                           strategy=meta, strategy_id=strategy_id,
                           active_tab='strategies', active_strategy=strategy_id)
