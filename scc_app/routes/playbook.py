"""Options Playbook — CC Scanner, Strangle Scanner, Positions, Alerts."""
from flask import Blueprint, render_template

bp = Blueprint('playbook', __name__)


@bp.route('/playbook')
def index():
    return render_template('playbook/index.html', active_tab='playbook')


@bp.route('/playbook/cc-scanner')
def cc_scanner():
    return render_template('playbook/cc_scanner.html', active_tab='playbook', active_section='cc')


@bp.route('/playbook/strangle-scanner')
def strangle_scanner():
    return render_template('playbook/strangle_scanner.html', active_tab='playbook', active_section='strangle')


@bp.route('/playbook/positions')
def positions():
    return render_template('playbook/active_positions.html', active_tab='playbook', active_section='positions')


@bp.route('/playbook/alerts')
def alerts():
    return render_template('playbook/alerts.html', active_tab='playbook', active_section='alerts')
