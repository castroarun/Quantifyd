"""
Breakout Scanner — Flask routes + 15-min poll/email job.

Wired into app.py with a single line:
    from services.breakout_scanner_api import register as register_breakout_scanner
    register_breakout_scanner(app, scheduler)

Routes (React SPA at /app/breakout-scanner, JSON at /api/breakout-scanner/*):
  GET  /api/breakout-scanner/state      -> cached scan (settings + matches)
  POST /api/breakout-scanner/scan       -> run now with posted filters (transient)
  GET  /api/breakout-scanner/settings   -> saved settings
  POST /api/breakout-scanner/settings   -> patch + save settings (incl. email toggle)
  POST /api/breakout-scanner/test-email -> send a one-off test email

Poll job: every 15 min Mon-Fri 09:15-15:30 IST -> scan + email NEW breakouts
(deduped per symbol+rule+day) when email alerts are enabled. Self-guards off-hours.
"""
from __future__ import annotations

import logging
import time as _t

from flask import jsonify, request

logger = logging.getLogger(__name__)

_cache = {'snapshot': None, 'ts': 0.0}


def register(app, scheduler):
    from services.breakout_scanner_service import (
        get_breakout_scanner_service, load_settings, save_settings,
        UNIVERSE_KEYS, DEFAULT_SETTINGS,
    )

    def _scan(settings=None):
        return get_breakout_scanner_service().scan(settings)

    @app.route('/api/breakout-scanner/state')
    def api_breakout_state():
        now = _t.time()
        cached = _cache.get('snapshot')
        if cached is not None and (now - _cache.get('ts', 0)) < 25:
            return jsonify(cached)
        try:
            snap = _scan()
            _cache['snapshot'] = snap
            _cache['ts'] = now
            return jsonify(snap)
        except Exception as e:
            logger.error(f"[BreakoutScanner] state error: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/breakout-scanner/scan', methods=['POST'])
    def api_breakout_scan():
        """Run a scan NOW with the posted filters (does not persist them)."""
        try:
            body = request.get_json(silent=True) or {}
            base = load_settings()
            base.update({k: v for k, v in body.items() if k in DEFAULT_SETTINGS})
            snap = _scan(base)
            _cache['snapshot'] = snap
            _cache['ts'] = _t.time()
            return jsonify(snap)
        except Exception as e:
            logger.error(f"[BreakoutScanner] scan error: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/breakout-scanner/settings', methods=['GET', 'POST'])
    def api_breakout_settings():
        if request.method == 'GET':
            return jsonify({'settings': load_settings(),
                            'universe_keys': UNIVERSE_KEYS,
                            'defaults': DEFAULT_SETTINGS})
        try:
            patch = request.get_json(silent=True) or {}
            s = save_settings(patch)
            return jsonify({'status': 'saved', 'settings': s})
        except Exception as e:
            logger.error(f"[BreakoutScanner] settings error: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/breakout-scanner/test-email', methods=['POST'])
    def api_breakout_test_email():
        try:
            from services.notifications import get_notification_service
            s = load_settings()
            get_notification_service().send_alert(
                'breakout', '✅ Breakout Scanner test',
                'This is a test email from the Quantifyd Breakout Scanner. '
                'Alerts are wired correctly.\n\nView: '
                'http://94.136.185.54:5000/app/breakout-scanner',
                priority='normal')
            return jsonify({'status': 'sent', 'to': s.get('email_to')})
        except Exception as e:
            logger.error(f"[BreakoutScanner] test-email error: {e}")
            return jsonify({'error': str(e)}), 500

    def _poll():
        try:
            r = get_breakout_scanner_service().poll_and_alert()
            if r.get('emailed'):
                logger.info("[BreakoutScanner] poll emailed %d new breakout(s)", r['emailed'])
        except Exception as e:
            logger.warning(f"[BreakoutScanner] poll failed: {e}")

    try:
        scheduler.add_job(
            _poll, 'cron', day_of_week='mon-fri', hour='9-15', minute='*/15',
            id='breakout_scanner_poll', replace_existing=True,
        )
        logger.info("Breakout Scanner poll job registered: every 15 min, 09:15-15:30 IST")
    except Exception as e:
        logger.warning(f"Could not register Breakout Scanner poll job: {e}")
