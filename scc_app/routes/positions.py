"""Positions & Trades tab."""
from flask import Blueprint, render_template

bp = Blueprint('positions', __name__)


@bp.route('/positions')
def index():
    return render_template('positions.html', active_tab='positions')
