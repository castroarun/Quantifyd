"""SENSEX naked-leg ST(7,3) trail — REST-based, because nas_ticker resolves NFO tokens only and
cannot ST-monitor SENSEX (BFO). Builds 5-minute premium candles from the 10s SL-monitor polls and
computes SuperTrend, returning a trailing stop for the naked SHORT survivor.

SAFETY: the caller clamps the returned stop to <= breakeven, so the ST trail can only ever make the
stop TIGHTER (lock more of the survivor's profit), never looser than the BE_PROTECT fallback. Returns
None during warm-up (< ATR_PERIOD+2 candles) or on any error -> caller falls back to breakeven.
State is per position_id, in-memory (rebuilds after a restart; BE_PROTECT covers the rebuild window)."""
import datetime

ATR_PERIOD = 7
MULT = 3.0
_state = {}   # position_id -> {'candles': [ {open,high,low,close} ], 'cur': {...}, 'bkt': datetime}

try:
    import pandas as pd
    from services.technical_indicators import calc_supertrend
    _OK = True
except Exception:
    _OK = False


def _bucket(now):
    return now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0)


def trail_stop(pos_id, ltp, entry, now=None):
    """Return the ST trailing stop for a naked short leg, or None to use breakeven."""
    if not _OK or not ltp or ltp <= 0 or not entry:
        return None
    now = now or datetime.datetime.now()
    s = _state.setdefault(pos_id, {"candles": [], "cur": None, "bkt": None})
    bkt = _bucket(now)
    if s["bkt"] != bkt:
        if s["cur"]:
            s["candles"].append(s["cur"])
            s["candles"] = s["candles"][-80:]
        s["cur"] = {"open": ltp, "high": ltp, "low": ltp, "close": ltp}
        s["bkt"] = bkt
    else:
        c = s["cur"]
        c["high"] = max(c["high"], ltp); c["low"] = min(c["low"], ltp); c["close"] = ltp
    bars = s["candles"] + ([s["cur"]] if s["cur"] else [])
    if len(bars) < ATR_PERIOD + 2:
        return None                       # warm-up -> breakeven
    try:
        df = pd.DataFrame(bars)
        st, _direction = calc_supertrend(df, ATR_PERIOD, MULT)
        st_line = float(st.iloc[-1])
    except Exception:
        return None
    if st_line <= 0:
        return None
    # short leg: the supertrend line during the premium's downtrend is the trailing stop.
    # (Caller clamps to <= entry, so this only ever tightens vs breakeven.)
    return round(st_line, 1)


def reset(pos_id):
    _state.pop(pos_id, None)
