"""NAS Day & Gap entry matrix.

Single source of truth for WHICH NAS system enters on WHICH day (by trading-DTE
and/or gap), and whether that entry is real-money (LIVE) or paper. Editable from
the React page /app/nas-config via GET/POST /api/nas/day-matrix.

DTE is TRADING-day DTE against the current-week (Tuesday) expiry:
    Tue=0, Mon=1, Fri=2, Thu=3, Wed=4.

Entry rule per system on a given morning:
    enter  =  dte_checked[today_dte]
              OR (gap_up_day   AND system.gap_up)
              OR (gap_down_day AND system.gap_down)
Mode (real vs paper):
    live   =  master_mode == 'live'  AND  system.live
    else paper.  Master 'paper'/'off' always forces paper (safety).

Gap = (today official open - prev close) / prev close, via Kite quote ohlc
(the recorder's first snapshot lags the real open by a few seconds, so it is NOT
used for the gap).
"""
import json
from datetime import date, timedelta
from pathlib import Path

_BASE = Path(__file__).resolve().parents[1] / 'backtest_data'
MATRIX_PATH = _BASE / 'nas_day_matrix.json'
_MASTER_PATH = _BASE / 'nas_master_mode.json'

# (key, display label) — key matches the per-variant DB / API naming
SYSTEMS = [
    ('nas',          'Squeeze Base/OTM'),
    ('nas_atm',      'Squeeze ATM'),
    ('nas_atm2',     'Squeeze ATM2'),
    ('nas_atm4',     'Squeeze ATM4'),
    ('nas_916_otm',  '9:16 OTM'),
    ('nas_916_atm',  '9:16 ATM'),
    ('nas_916_atm2', '9:16 ATM2'),
    ('nas_916_atm4', '9:16 ATM4'),
]
DTES = (4, 3, 2, 1, 0)


def _row(dtes, gap_up, gap_down, live):
    return {'dte': {str(d): (d in dtes) for d in DTES},
            'gap_up': bool(gap_up), 'gap_down': bool(gap_down), 'live': bool(live)}


# Default = current live posture (3 live 916 ATMs on 2/1/0, all else paper) +
# gap-on for 916 ATM2/ATM4 so a gap day like Wed/Thu fires them.
DEFAULT = {
    'gap_up_pct': 0.45,
    'gap_down_pct': 0.45,
    'systems': {
        'nas':          _row({2, 1, 0}, False, False, False),
        'nas_atm':      _row({2, 1, 0}, False, False, False),
        'nas_atm2':     _row({2, 1, 0}, False, False, False),
        'nas_atm4':     _row({2, 1, 0}, False, False, False),
        'nas_916_otm':  _row({2, 1, 0}, False, False, False),
        'nas_916_atm':  _row({2, 1, 0}, False, False, True),
        'nas_916_atm2': _row({2, 1, 0}, True,  True,  True),
        'nas_916_atm4': _row({2, 1, 0}, True,  True,  True),
    },
}


def load():
    """Load the matrix, merging in any missing systems/keys from DEFAULT."""
    m = None
    try:
        if MATRIX_PATH.exists():
            m = json.loads(MATRIX_PATH.read_text())
    except Exception:
        m = None
    if not isinstance(m, dict):
        m = json.loads(json.dumps(DEFAULT))
    m.setdefault('gap_up_pct', DEFAULT['gap_up_pct'])
    m.setdefault('gap_down_pct', DEFAULT['gap_down_pct'])
    m.setdefault('systems', {})
    for k, _ in SYSTEMS:
        d = DEFAULT['systems'][k]
        row = m['systems'].setdefault(k, json.loads(json.dumps(d)))
        row.setdefault('dte', dict(d['dte']))
        for col in DTES:
            row['dte'].setdefault(str(col), False)
        for f in ('gap_up', 'gap_down', 'live'):
            row.setdefault(f, False)
    return m


def save(m):
    MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    MATRIX_PATH.write_text(json.dumps(m, indent=2))
    return True


def trading_dte(today=None):
    """Trading-DTE by weekday, for the standard Tuesday weekly expiry:
        Tue=0, Mon=1, Fri=2, Thu=3, Wed=4.
    The matrix columns are weekday buckets labelled by their normal DTE, so this
    fixed map (not a holiday-aware count) is what matches the user's model and
    stays stable across holiday weeks. Returns None on Sat/Sun."""
    today = today or date.today()
    return {0: 1, 1: 0, 2: 4, 3: 3, 4: 2}.get(today.weekday())


_GAP_CACHE = {'date': None, 'gap_pct': None}


def compute_gap(matrix=None):
    """(gap_pct, is_up, is_down) from Kite official open vs prev close.
    The raw gap_pct is cached per calendar day (open + prev close are fixed once
    the market opens) so the per-candle squeeze path does not hammer Kite.
    gap_pct is None if Kite is unavailable (then neither up nor down)."""
    m = matrix or load()
    up_t = float(m.get('gap_up_pct', 0.45))
    dn_t = float(m.get('gap_down_pct', 0.45))
    today = date.today()
    gap = _GAP_CACHE['gap_pct'] if _GAP_CACHE['date'] == today else None
    if gap is None:
        try:
            from services.kite_service import get_kite
            ohlc = get_kite().quote(['NSE:NIFTY 50'])['NSE:NIFTY 50']['ohlc']
            op = float(ohlc['open'])
            pc = float(ohlc['close'])
            if pc > 0 and op > 0:
                gap = (op - pc) / pc * 100.0
                _GAP_CACHE['date'] = today
                _GAP_CACHE['gap_pct'] = gap
        except Exception:
            gap = None
    if gap is None:
        return (None, False, False)
    return (gap, gap >= up_t, gap <= -dn_t)


def _master_mode():
    try:
        return (json.loads(_MASTER_PATH.read_text()) or {}).get('mode', 'off')
    except Exception:
        return 'off'


def gate(system_key, today=None, master_mode=None, matrix=None, gap=None):
    """The single entry authority. Returns:
        {allow: bool, mode: 'live'|'paper', dte: int|None, gap_pct: float|None, reason: str}
    """
    m = matrix or load()
    row = m['systems'].get(system_key)
    if not row:
        return {'allow': False, 'mode': 'paper', 'dte': None, 'gap_pct': None,
                'reason': 'no matrix row'}
    dte = trading_dte(today)
    if gap is None:
        gap = compute_gap(m)
    gap_pct, is_up, is_down = gap
    enter, reasons = False, []
    if dte is not None and row['dte'].get(str(dte)):
        enter = True
        reasons.append('DTE%s on' % dte)
    if is_up and row.get('gap_up'):
        enter = True
        reasons.append('gap-up %+.2f%%' % gap_pct)
    if is_down and row.get('gap_down'):
        enter = True
        reasons.append('gap-down %+.2f%%' % gap_pct)
    mm = master_mode if master_mode is not None else _master_mode()
    # live only when gated-on (enter) AND master live AND this is a live row; else paper
    mode = 'live' if (mm == 'live' and row.get('live') and enter) else 'paper'
    # paper-shadow (user 2026-06-23): these systems ALSO enter in PAPER on EVERY day for
    # the daily P&L curve, regardless of the live gating / master mode.
    allow = enter or bool(row.get('paper_shadow'))
    if not enter and row.get('paper_shadow'):
        reasons.append('paper-shadow')
    return {'allow': allow, 'mode': mode, 'dte': dte, 'gap_pct': gap_pct,
            'reason': '; '.join(reasons) or ('DTE%s off, no gap' % dte)}
