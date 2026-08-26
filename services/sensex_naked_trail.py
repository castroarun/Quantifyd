"""SENSEX naked-leg trailing stop — REST-based, because nas_ticker resolves NFO tokens only and
cannot ST-monitor SENSEX (BFO). Builds 5-minute premium candles from the 10s SL-monitor polls and
returns a trailing CEILING for the naked SHORT survivor.

2026-08-26 REWRITE (research/128, Arun sign-off). Two defects fixed:

  1. It used `calc_supertrend`, which returns the LOWER band while direction == 1 — and a decaying
     premium never flips direction. The value therefore landed BELOW the live premium on 62% of
     episodes (research/128, n=86). This is the SAME bug that was found and fixed on the NIFTY side
     on 2026-07-14; `NasAtm4Executor.compute_short_trailing_stop` is that fix, and its docstring
     documents it. The SENSEX path simply never got it. We now call that same function: a ratcheting
     upper band that sits ABOVE the premium by construction (100% of episodes) and only tightens.

  2. The caller wrote the value into `sl_price`, where the generic `check_and_handle_sl` fires on
     `live >= sl_price` — so a value below the market self-triggered the instant it was written
     (live example 2026-08-26 11:00:02: ceiling 90.4 written while the premium was 134.0, exited 3s
     later). We now return a CEILING plus an explicit exit decision and never touch `sl_price`,
     mirroring nas_ticker's `_atm_naked_st_val`. `sl_price` stays at breakeven as the hard floor.

Exit requires the premium to hold above the ceiling for CONFIRM_POLLS consecutive polls (~60s at the
10s cadence) — research/128 measured 30s-3min as a plateau and >=5min as measurably worse.

The ceiling is computed from COMPLETED 5-min bars only (recomputed on each bar close) and compared
to the live premium every poll — exactly the NIFTY mechanism. Returns None during warm-up
(< ATR_PERIOD+1 completed bars) so the caller falls back to breakeven. State is per position_id and
in-memory; it rebuilds after a restart, and BE_PROTECT covers the rebuild window.
"""
import datetime

ATR_PERIOD = 7
MULT = 3.0
CONFIRM_POLLS = 6          # ~60s at the 10s SL-monitor cadence

_state = {}   # pos_id -> {'candles': [...], 'cur': {...}, 'bkt': dt, 'ceil': float|None, 'breach': int}

try:
    from services.nas_atm4_executor import NasAtm4Executor
    _OK = True
except Exception:
    _OK = False


def _blank():
    return {"candles": [], "cur": None, "bkt": None, "ceil": None, "breach": 0}


def _bucket(now):
    return now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0)


def _recompute(s):
    """Ceiling from COMPLETED bars only (mirrors nas_ticker, which recomputes on bar close)."""
    bars = s["candles"]
    if len(bars) < ATR_PERIOD + 1:
        return None
    try:
        stop, _breached = NasAtm4Executor.compute_short_trailing_stop(bars, ATR_PERIOD, MULT)
    except Exception:
        return None
    if not stop or stop <= 0:
        return None
    return round(float(stop), 1)


def seed(pos_id, bars):
    """Prime the candle history from the leg's own 09:16->now 5-min premium candles, so the trail
    is armed immediately instead of after a median 39-minute warm-up (research/128). Best-effort:
    the caller ignores failures and we simply cold-start."""
    if not bars:
        return False
    s = _state.setdefault(pos_id, _blank())
    if s["candles"]:
        return False                     # already seeded / running
    clean = []
    for b in bars:
        try:
            clean.append({"open": float(b["open"]), "high": float(b["high"]),
                          "low": float(b["low"]), "close": float(b["close"])})
        except Exception:
            continue
    if not clean:
        return False
    s["candles"] = clean[-200:]
    s["ceil"] = _recompute(s)
    return True


def trail_ceiling(pos_id, ltp, entry, now=None):
    """Return (ceiling, exit_now) for a naked SHORT leg.

    ceiling  — trailing stop ABOVE the premium, clamped to <= entry, or None during warm-up.
    exit_now — True only after CONFIRM_POLLS consecutive polls with the premium above the ceiling.

    NEVER write the ceiling into sl_price: it is not a stop-loss level for the generic
    `live >= sl_price` check, it is a ceiling the premium must break upward through.
    """
    if not _OK or not ltp or ltp <= 0 or not entry:
        return None, False
    now = now or datetime.datetime.now()
    s = _state.setdefault(pos_id, _blank())

    bkt = _bucket(now)
    if s["bkt"] != bkt:
        if s["cur"]:
            s["candles"].append(s["cur"])
            s["candles"] = s["candles"][-200:]
            s["ceil"] = _recompute(s)          # recompute on bar close, like nas_ticker
        s["cur"] = {"open": ltp, "high": ltp, "low": ltp, "close": ltp}
        s["bkt"] = bkt
    else:
        c = s["cur"]
        c["high"] = max(c["high"], ltp)
        c["low"] = min(c["low"], ltp)
        c["close"] = ltp

    if s["ceil"] is None:
        s["ceil"] = _recompute(s)
    if s["ceil"] is None:
        s["breach"] = 0
        return None, False                     # warm-up -> caller uses breakeven

    # Clamp to breakeven. research/128: never binding once armed, so this is free insurance.
    ceiling = min(s["ceil"], round(float(entry), 1))
    if ltp > ceiling:
        s["breach"] += 1
        return ceiling, s["breach"] >= CONFIRM_POLLS
    s["breach"] = 0
    return ceiling, False


def reset(pos_id):
    _state.pop(pos_id, None)
