#!/usr/bin/env python3
"""Register /api/breakout-scan (ATH + consolidation scanner) in app.py."""
from pathlib import Path

P = Path('/home/arun/quantifyd/app.py')
s = P.read_text()

if 'breakout-scan' in s:
    print('ALREADY PATCHED')
    raise SystemExit(0)

anchor = """try:
    from services.journal.api import journal_bp"""
assert anchor in s, 'journal blueprint anchor not found'

block = '''# ---- Breakout scanner: new ATHs, ATH+volume, consolidation breakouts ----
_bscan_cache = {}


@app.route('/api/breakout-scan')
def api_breakout_scan():
    """Scan for new all-time highs and consolidation breakouts.

    Query: asof=live|eod|t1|t2|t3, universe=nifty500|fno|smallcap|all,
           vol_mult, vol_window (10|20), consol_days, consol_max_range_pct,
           min_turnover_cr
    """
    import time as _t
    from flask import request as _rq
    try:
        from services.ath_scanner import scan
        args = {
            'asof': _rq.args.get('asof', 'eod'),
            'universe': _rq.args.get('universe', 'nifty500'),
            'vol_mult': _rq.args.get('vol_mult', type=float),
            'vol_window': _rq.args.get('vol_window', type=int),
            'consol_days': _rq.args.get('consol_days', type=int),
            'consol_max_range_pct': _rq.args.get('consol_max_range_pct', type=float),
            'min_turnover_cr': _rq.args.get('min_turnover_cr', type=float),
        }
        key = repr(sorted(args.items()))
        ttl = 45 if args['asof'] == 'live' else 600
        hit = _bscan_cache.get(key)
        if hit and (_t.time() - hit[0]) < ttl:
            out = dict(hit[1])
            out['cached'] = True
            return jsonify(out)
        res = scan(**args)
        _bscan_cache[key] = (_t.time(), res)
        res = dict(res)
        res['cached'] = False
        return jsonify(res)
    except Exception as e:
        logger.error('[breakout-scan] failed: %s', e, exc_info=True)
        return jsonify({'error': str(e)}), 500


'''

P.write_text(s.replace(anchor, block + anchor, 1))
print('APP PATCHED — /api/breakout-scan registered')
