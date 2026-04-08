"""Day Deep Dive tab — historical day-level snapshots."""
from flask import Blueprint, render_template

bp = Blueprint('deep_dive', __name__)


@bp.route('/deep-dive')
@bp.route('/deep-dive/<date>')
def index(date=None):
    return render_template('deep_dive.html', selected_date=date, active_tab='deep_dive')
