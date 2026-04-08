"""Dashboard tab — the 5-second glance view."""
from flask import Blueprint, render_template
from ..config import STRATEGY_META

bp = Blueprint('dashboard', __name__)


@bp.route('/')
def index():
    return render_template('dashboard.html', strategies=STRATEGY_META, active_tab='dashboard')
