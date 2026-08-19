#!/usr/bin/env python3
"""Register /api/index-pulse/* endpoints + the 16:20 index top-up cron."""
from pathlib import Path

P = Path('/home/arun/quantifyd/app.py')
s = P.read_text()

if "'/api/index-pulse/strip'" in s:
    print('ALREADY PATCHED')
    raise SystemExit(0)

anchor = """try:
    from services.journal.api import journal_bp"""
assert anchor in s

block = '''# ---- Index Pulse: index strip / compare / drilldown ----
@app.route('/api/index-pulse/strip')
def api_indexpulse_strip():
    try:
        from services.index_pulse import strip
        return jsonify(strip())
    except Exception as e:
        logger.error('[index-pulse] strip failed: %s', e, exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/index-pulse/compare')
def api_indexpulse_compare():
    from flask import request as _rq
    try:
        from services.index_pulse import compare
        syms = _rq.args.get('symbols')
        return jsonify(compare(
            window=_rq.args.get('window', '3m'),
            mode=_rq.args.get('mode', 'abs'),
            symbols=syms.split(',') if syms else None))
    except Exception as e:
        logger.error('[index-pulse] compare failed: %s', e, exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/index-pulse/drilldown')
def api_indexpulse_drill():
    from flask import request as _rq
    try:
        from services.index_pulse import drilldown
        return jsonify(drilldown(
            _rq.args.get('index', 'NIFTY50'),
            window=_rq.args.get('window', '1m')))
    except Exception as e:
        logger.error('[index-pulse] drilldown failed: %s', e, exc_info=True)
        return jsonify({'error': str(e)}), 500


def _index_daily_topup_job():
    """16:20 Mon-Fri: pull today's daily bar for every tracked index."""
    try:
        import subprocess
        subprocess.run(
            ['/home/arun/quantifyd/venv/bin/python3',
             '/home/arun/quantifyd/scripts/backfill_index_daily.py'],
            timeout=600, capture_output=True)
        logger.info('[index-pulse] daily index top-up done')
    except Exception as e:
        logger.warning('[index-pulse] top-up failed: %s', e)


try:
    scheduler.add_job(
        _index_daily_topup_job, 'cron', day_of_week='mon-fri',
        hour=16, minute=20, id='index_daily_topup', replace_existing=True,
        misfire_grace_time=3600)
    logger.info('[index-pulse] endpoints registered + 16:20 top-up scheduled')
except Exception as _e:
    logger.warning('[index-pulse] could not schedule top-up: %s', _e)


'''

P.write_text(s.replace(anchor, block + anchor, 1))
print('APP PATCHED')
